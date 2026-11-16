"""The one-source run (engine.run), the command line, the config loader and the run state, plus
the engine pieces whose tests lived with the runner: the heavy whisper slots, the score without a
chat signal, the default encoder and the niche passed to the render."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from clipengine import cli, config, engine, runstate, transcript


# ---------------------------------------------------------------- heavy whisper slots

def test_heavy_whisper_jobs_never_exceed_the_slots(monkeypatch, tmp_path):
    monkeypatch.setenv("HEAVY_WHISPER_LOCK_DIR", str(tmp_path / "slots"))
    monkeypatch.setenv("HEAVY_WHISPER_SLOTS", "2")
    held, release = [], threading.Event()

    def hold():
        with transcript.heavy_slot("large-v3-turbo", None):
            held.append(1)
            release.wait(5)
    ts = [threading.Thread(target=hold) for _ in range(2)]
    for t in ts:
        t.start()
    deadline = time.monotonic() + 5
    while len(held) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            with transcript.heavy_slot("large-v3-turbo", 0.4):
                pass
        with transcript.heavy_slot("small.en", 7.0) as left:      # not heavy: straight through
            assert left == 7.0
        calls = []
        monkeypatch.setattr(transcript, "_worker_call", lambda argv, t=None: calls.append(t))
        with pytest.raises(subprocess.TimeoutExpired):
            transcript._worker_run(["a.wav", "--model", "large-v3-turbo"], timeout=0.4)
        transcript._worker_run(["a.wav", "--model", "base.en"], timeout=9.0)
        assert calls == [9.0]
    finally:
        release.set()
        for t in ts:
            t.join()
    with transcript.heavy_slot("large-v3-turbo", 30.0) as left:    # free again: time left
        assert 1.0 <= left <= 30.0
    monkeypatch.delenv("HEAVY_WHISPER_SLOTS")
    monkeypatch.setenv("WINDOW_CONCURRENCY", "3")
    assert transcript.heavy_slots() == 3


# ---------------------------------------------------------------- the score, the render argv

def test_score_is_the_llm_and_energy_blend_minus_the_unclean_end():
    """w_llm*llm/100 + w_energy*sigmoid(mean_z), 0.1 off when the sentence does not end cleanly;
    no LLM score: sigmoid(mean_z) alone."""
    z = np.zeros(200, dtype=np.float64)
    z[50:60] = 1.5
    vc = {"w_llm": 0.8, "w_energy": 0.2}
    c = {"start": 50.0, "end": 60.0, "boundary_type": "sentence", "last": "so it is.",
         "llm_score": 75.0}
    want = 0.8 * 0.75 + 0.2 * engine.sigmoid(z[50:60].mean())
    assert engine._score(dict(c), z, vc)["score"] == pytest.approx(want, abs=1e-9)
    c2 = {**c, "last": "so it is and"}
    assert engine._score(dict(c2), z, vc)["score"] == pytest.approx(want - 0.1, abs=1e-9)
    c3 = {**c, "llm_score": None}
    assert engine._score(dict(c3), z, vc)["score"] == pytest.approx(
        engine.sigmoid(z[50:60].mean()), abs=1e-9)


LIBX264 = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high",
           "-pix_fmt", "yuv420p", "-g", "60"]


def test_the_render_encoder_is_libx264_unless_render_encoder_names_another():
    def codec(enc):
        e = {k: v for k, v in os.environ.items() if k != "RENDER_ENCODER"}
        if enc is not None:
            e["RENDER_ENCODER"] = enc
        return subprocess.run([sys.executable, "-c", "import json; from clipengine.render import "
                               "recompose; print(json.dumps(recompose.VIDEO_CODEC))"], env=e,
                              capture_output=True, text=True, timeout=120)
    r = codec(None)
    assert json.loads(r.stdout) == LIBX264, r.stderr
    vt = json.loads(codec("h264_videotoolbox").stdout)
    assert vt[:2] == ["-c:v", "h264_videotoolbox"] and vt[vt.index("-b:v") + 1] == "8M"
    r = codec("bogus")
    assert r.returncode == 1 and "unknown RENDER_ENCODER 'bogus'" in r.stderr


def test_the_render_command_passes_the_niche_and_the_titles_content_kind():
    runstate.add_source("Abc_w02", title="Live trading the open", niche="finance")
    clip = {"source_id": "Abc_w02", "start_s": 1.0, "end_s": 20.0, "layout": "auto"}
    cmd = engine.recompose_cmd(clip, {"source_id": "Abc_w02", "vertical": "general"}, "/m.mp4",
                               "/o.mp4")
    assert cmd[:3] == engine.RECOMPOSE and cmd[3:7] == ["/m.mp4", "1.0", "19.0", "/o.mp4"]
    assert cmd[cmd.index("--niche") + 1] == "finance"
    assert cmd[cmd.index("--content-kind") + 1] == "screen"
    assert cmd[cmd.index("--vertical") + 1] == "general"
    for gone in ("--broadcast", "--clean-first", "--levels"):
        assert gone not in cmd
    runstate.add_source("plain", title="Live trading the open")         # no niche
    cmd = engine.recompose_cmd(dict(clip, source_id="plain"), {"source_id": "plain"}, "/m.mp4",
                               "/o.mp4")
    assert "--niche" not in cmd and "--content-kind" not in cmd


def test_source_ids_and_niche_verticals():
    assert engine.source_id_for("https://www.youtube.com/watch?v=abcdefghijk") == "abcdefghijk"
    assert engine.source_id_for("https://youtu.be/abcdefghijk?t=3") == "abcdefghijk"
    assert engine.source_id_for("https://www.twitch.tv/videos/123456") == "tw_123456"
    assert engine.source_id_for("/clips/a.mp4").startswith("loc_")
    assert engine.source_id_for("/clips/a.mp4") == engine.source_id_for("/clips/a.mp4")
    assert [engine.vertical_for_niche(n) for n in ("podcasts", "streamers", "gaming", "poker",
                                                   "tech", None)] == \
        ["podcast", "stream", "stream", "poker", "general", "general"]


# ---------------------------------------------------------------- the run

def _fake_steps(monkeypatch, tmp_path, workers):
    def harvest(src):
        runstate.update_source(src["source_id"], duration_s=600.0, state="harvested")
        return True

    def select(src):
        sid = src["source_id"]
        runstate.add_clip(sid, 10.0, 40.0, 0.9, "auto", hook="First title",
                          features_json={"refined": True}, boundary_type="sentence")
        runstate.add_clip(sid, 100.0, 130.0, 0.7, "auto", hook="Second title",
                          features_json={}, boundary_type="sentence")
        runstate.update_source(sid, state="selected")
        return True

    def render(clip):
        cid = clip["clip_id"]
        if cid == 1:
            out = tmp_path / "work.mp4"
            out.write_bytes(b"mp4")
            url = engine._store_clip(out, clip)
            feats = engine._features_merge(clip, gate_report={"verdict": "pass", "drop_reason": ""})
            runstate.update_clip(cid, features_json=feats)
            runstate.update_clip(cid, out_path=url, state="rendered",
                                 qc={"width": 1080, "height": 1920, "layout": "track",
                                     "integrated_lufs": -14.1, "passed": True})
            return True
        report = {"verdict": "fail", "drop_reason": "ending: ends mid-sentence",
                  "ending": {"passed": False, "errors": ["ends mid-sentence"]}}
        engine._drop_clip(clip, report["drop_reason"], "gate", report)
        return False
    monkeypatch.setattr(engine, "harvest", harvest)
    monkeypatch.setattr(engine, "select", select)
    monkeypatch.setattr(engine, "render", render)
    monkeypatch.setattr(transcript, "close_worker", lambda: workers.append(transcript.WORKER))


def test_run_writes_the_clips_their_sidecars_and_run_json(tmp_path, monkeypatch):
    src = tmp_path / "talk.mp4"
    src.write_bytes(b"\0")
    workers = []
    _fake_steps(monkeypatch, tmp_path, workers)
    monkeypatch.delenv("TRANSCRIBE_WORKER", raising=False)
    rec = engine.run(str(src), niche="podcasts", out_dir=str(tmp_path / "out"), title="A talk")
    sid = engine.source_id_for(str(src))
    d = tmp_path / "out" / sid
    assert workers == [True]                             # the warm worker ran and was closed
    assert rec["made"] == 1 and rec["selected"] == 2 and rec["vertical"] == "podcast"
    assert (d / "clip_01.mp4").read_bytes() == b"mp4" and not (d / "clip_02.mp4").exists()
    one = json.loads((d / "clip_01.json").read_text())
    assert one["file"] == "clip_01.mp4" and one["gate"]["verdict"] == "pass"
    assert (one["start"], one["end"], one["duration"]) == (10.0, 40.0, 30.0)
    assert one["title"] == "First title" and one["layout"] == "auto"
    assert one["rendered_layout"] == "track" and one["qc"]["width"] == 1080
    two = json.loads((d / "clip_02.json").read_text())
    assert two["state"] == "failed" and two["gate"]["verdict"] == "fail" and two["file"] is None
    assert set(two["reason_codes"]) == {"cut_end"}
    run = json.loads((d / "run.json").read_text())
    assert run["source"]["niche"] == "podcasts" and run["source"]["title"] == "A talk"
    assert [c["clip"] for c in run["clips"]] == [1, 2]
    assert run["steps"] == {"harvest": True, "select": True}
    assert any(e["stage"] == "run" for e in run["events"])
    assert src.exists()                                  # a local source is never deleted


def test_the_worker_switch(tmp_path, monkeypatch):
    src = tmp_path / "talk.mp4"
    src.write_bytes(b"\0")
    workers = []
    _fake_steps(monkeypatch, tmp_path, workers)
    monkeypatch.setenv("TRANSCRIBE_WORKER", "0")
    engine.run(str(src), niche=None, out_dir=str(tmp_path / "out"))
    assert workers == [False]
    assert engine.worker_enabled({}) is True and engine.worker_enabled({"TRANSCRIBE_WORKER": "0"}) is False


def test_a_failed_harvest_selects_and_renders_nothing(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(engine, "harvest", lambda src: False)
    monkeypatch.setattr(engine, "select", lambda src: seen.append("select"))
    monkeypatch.setattr(engine, "render", lambda c: seen.append("render"))
    monkeypatch.setattr(transcript, "close_worker", lambda: None)
    rec = engine.run("https://example.com/v.mp4", out_dir=str(tmp_path / "out"))
    assert seen == [] and rec["made"] == 0 and rec["steps"]["harvest"] is False


# ---------------------------------------------------------------- the command line

def test_cli_refuses_bad_targets_niches_and_cookies(tmp_path, capsys):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"\0")
    assert cli.main(["run", str(tmp_path / "missing.mp4")]) == 2
    assert cli.main(["run", "ftp://example.com/a.mp4"]) == 2
    assert cli.main(["run", "file:///etc/passwd"]) == 2
    assert cli.main(["run", str(tmp_path)]) == 2                       # a folder
    assert cli.main(["run", str(f), "--niche", "../config"]) == 2
    assert cli.main(["run", str(f), "--cookies", str(tmp_path / "none.txt")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]")
    assert cli.main(["run", str(f), "--config", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "neither an existing file nor an http(s) URL" in err and "must hold a JSON object" in err
    assert cli._check_target("https://example.com/v") == "https://example.com/v"
    with pytest.raises(ValueError):
        cli._check_target("-o/tmp/x")


def test_cli_help_lists_the_packs(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "podcasts" in out and "--cookies" in out


# ---------------------------------------------------------------- the config

def test_config_merges_a_user_file_and_env_over_the_defaults(tmp_path, monkeypatch):
    base = config._read(config.DEFAULTS_PATH, required=True)
    user = tmp_path / "user.json"
    user.write_text(json.dumps({"select": {"top_n": 2, "verticals": {"podcast": {"max_dur_s": 50}}},
                                "scratch_dir": "rel/cache"}))
    monkeypatch.setenv(config.ENV_FILE, str(user))
    monkeypatch.delenv(config.ENV_SCRATCH, raising=False)
    monkeypatch.setenv(config.ENV_FONT_DIR, str(tmp_path / "fonts"))
    monkeypatch.chdir(tmp_path)
    cfg = config.load()
    assert cfg["select"]["top_n"] == 2
    assert cfg["select"]["verticals"]["podcast"] == {**base["select"]["verticals"]["podcast"],
                                                     "max_dur_s": 50}
    assert cfg["select"]["refine_model"] == base["select"]["refine_model"]   # untouched keys stay
    assert cfg["scratch_dir"] == str(tmp_path / "rel" / "cache")
    assert cfg["render"]["font_dir"] == str(tmp_path / "fonts")
    assert config.merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}, "d": [1]}) == \
        {"a": {"b": 3, "c": 2}, "d": [1]}


def test_the_shipped_defaults_hold_no_private_paths():
    text = config.DEFAULTS_PATH.read_text()
    assert '"/' not in text and '"~' not in text            # no absolute or home-relative path
    assert "claude_config_dir" not in text
    cfg = json.loads(text)
    assert sorted(k for k in cfg if not k.startswith("_")) == \
        ["layout_crops", "quality_gate", "render", "scratch_dir", "select"]
    assert cfg["scratch_dir"] == ".cache" and cfg["render"]["font_dir"] is None


# ---------------------------------------------------------------- the run state

def test_runstate_keeps_one_clip_per_source_moment():
    runstate.add_source("s1")
    a = runstate.add_clip("s1", 10.0, 20.0)
    assert a and runstate.add_clip("s1", 10.0, 25.0) == 0             # same start: refused
    b = runstate.add_clip("s1", 30.0, 40.0)
    with pytest.raises(ValueError):
        runstate.update_clip(b, start_s=10.0)
    runstate.update_clip(b, start_s=31.0, features_json={"x": np.float64(1.5)})
    assert json.loads(runstate.clip(b)["features_json"]) == {"x": 1.5}
    assert [c["clip_id"] for c in runstate.clips_for_source("s1")] == [a, b]
    row = runstate.clip(a)
    row["state"] = "mutated"                                            # rows are copies
    assert runstate.clip(a)["state"] == "selected"
