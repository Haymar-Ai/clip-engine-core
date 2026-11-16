"""clip-quality 10-21 stage 3: the burned caption words come from the strong model (large-v3-turbo,
the repair lane's recaption model) through the window's warm whisper worker, within the window
cap; the fallback re-hears only the phrases select flagged as crosstalk. Offline: the strong pass
is stubbed. Phrases are modelled on 10-21 clip 332 ('flyway champion' burned for the spoken
'flyweight champion', base.en words after a failed precision pass)."""
from __future__ import annotations

import importlib
import json
import time

import pytest

from clipengine import runstate


def W(*triples, **extra):
    return [{"w": w, "start": s, "end": e, **extra} for w, s, e in triples]


GARBLED = W(("Who's", 20.0, 20.2), ("the", 20.2, 20.3), ("flyway", 20.3, 20.7),
            ("champion", 20.7, 21.2), ("now?", 21.2, 21.5), ("The", 22.0, 22.2),
            ("level", 22.2, 22.5), ("is", 22.5, 22.6), ("at", 22.6, 22.7), ("an", 22.7, 22.8),
            ("all-time", 22.8, 23.3), ("high.", 23.3, 23.7))
STRONG = W(("Who's", 20.0, 20.2), ("the", 20.2, 20.3), ("flyweight", 20.3, 20.7),
           ("champion", 20.7, 21.2), ("now?", 21.2, 21.5), ("The", 22.0, 22.2),
           ("level", 22.2, 22.5), ("is", 22.5, 22.6), ("at", 22.6, 22.7), ("an", 22.7, 22.8),
           ("all-time", 22.8, 23.3), ("high.", 23.3, 23.7), p=0.9)


@pytest.fixture
def db(tmp_path, monkeypatch):
    runstate.reset()
    from clipengine import engine
    monkeypatch.setattr(engine, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(engine, "CAPTION_STRONG_MODEL", "large-v3-turbo")
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", None)
    runstate.init()
    return tmp_path


def _clip(tmp_path, side, start=19.8, end=24.0, refined=True):
    runstate.add_source("S_w00", local_path=str(tmp_path / "w.mp4"))
    wp = tmp_path / "side.json"
    wp.write_text(json.dumps(side))
    cid = runstate.add_clip("S_w00", start, end, 0.5, "auto", hook="h", words_path=str(wp),
                         features_json={"refined": refined})
    return runstate.clip(cid)


def test_full_pass_replaces_the_garbled_words(db, tmp_path):
    from clipengine import engine
    clip = _clip(tmp_path, GARBLED + W(("after", 24.5, 24.8)))
    heard = []

    def strong(media, lo, hi, timeout):
        heard.append((round(lo, 2), round(hi, 2)))
        return STRONG
    side, info = engine.strong_captions(clip, "w.mp4", words_fn=strong)
    texts = [w["w"] for w in side]
    assert "flyweight" in texts and "flyway" not in texts
    assert texts[-1] == "after"                                # context outside the cut kept
    # the pass hears 1.5 s before the cut and 2 s after it (render.caption_context)
    assert heard == [(18.3, 26.0)] and info["mode"] == "full"
    assert info["model"] == "large-v3-turbo" and info["captioned"] == len(STRONG)


def test_a_crosstalk_phrase_select_never_heard_stays_unburned(db, tmp_path):
    from clipengine import engine
    clip = _clip(tmp_path, GARBLED)
    other = W(("buy", 22.0, 22.2), ("my", 22.2, 22.4), ("course", 22.4, 22.7),
              ("today", 22.7, 23.0), ("folks", 23.0, 23.3), p=0.9)
    strong = STRONG[:5] + other
    side, info = engine.strong_captions(clip, "w.mp4", words_fn=lambda *a: strong)
    assert all(w.get("cap") is False for w in side if w["w"] in ("buy", "course"))
    assert info["dropped"] and info["captioned"] == 5


def test_a_base_sidecar_does_not_veto_what_the_strong_model_heard(db, tmp_path):
    """261: base.en (precision pass failed) never heard "What's the end?"; large-v3-turbo did,
    with confidence. Only the word probabilities judge it then."""
    from clipengine import engine
    base = GARBLED[:5]                                           # misses the second sentence
    clip = _clip(tmp_path, base, refined=False)
    side, info = engine.strong_captions(clip, "w.mp4", words_fn=lambda *a: STRONG)
    assert info["judged_by"] == ["prob"] and info["captioned"] == len(STRONG)
    unsure = [dict(w, p=0.2) for w in STRONG[5:]]
    side, info = engine.strong_captions(clip, "w.mp4", words_fn=lambda *a: STRONG[:5] + unsure)
    assert info["captioned"] == 5                                # a mumble stays unburned


def test_short_on_time_only_the_flagged_phrases_are_reheard(db, tmp_path, monkeypatch):
    from clipengine import engine
    side0 = GARBLED[:5] + [dict(w, cap=False) for w in GARBLED[5:]]
    clip = _clip(tmp_path, side0)
    # ~1 s per audio second, warm: the 5.2 s full pass does not fit 4.5 s, the 2.7 s phrase does
    monkeypatch.setattr(engine, "_strong_cost", lambda s: float(s))
    monkeypatch.setattr(engine, "WINDOW_DEADLINE",
                        time.monotonic() + engine.CAPTION_RESERVE_S + 4.5)
    heard = []

    def strong(media, lo, hi, timeout):
        heard.append((lo, hi))
        return STRONG
    side, info = engine.strong_captions(clip, "w.mp4", words_fn=strong)
    assert info["mode"] == "phrases" and len(heard) == 1
    assert heard[0][0] >= 21.4 and heard[0][1] <= 24.3
    texts = [w["w"] for w in side]
    assert texts.count("flyway") == 1                          # outside the phrase: unchanged
    assert all(w.get("cap") is not False for w in side if w["w"] in ("level", "high."))


def test_no_time_and_nothing_flagged_keeps_selects_words(db, tmp_path, monkeypatch):
    from clipengine import engine
    clip = _clip(tmp_path, GARBLED)
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", time.monotonic() + 10.0)
    side, info = engine.strong_captions(clip, "w.mp4", words_fn=lambda *a: pytest.fail("ran"))
    assert side is None and info["mode"] == "skipped"


def test_the_window_clips_still_to_render_are_reserved(db, tmp_path, monkeypatch):
    from clipengine import engine
    clip = _clip(tmp_path, GARBLED)
    runstate.add_clip("S_w00", 40.0, 60.0, 0.5, "auto")              # still 'selected'
    monkeypatch.setattr(engine, "_strong_cost", lambda s: 10.0)
    monkeypatch.setattr(engine, "WINDOW_DEADLINE",
                        time.monotonic() + engine.CAPTION_RESERVE_S + 30.0)
    side, info = engine.strong_captions(clip, "w.mp4", words_fn=lambda *a: STRONG)
    assert side is None and info["mode"] == "skipped"            # 2 x reserve > what is left


def test_caption_step_puts_the_new_sidecar_on_the_clip_and_survives_errors(db, tmp_path,
                                                                         monkeypatch):
    from clipengine import engine
    clip = _clip(tmp_path, GARBLED)
    monkeypatch.setattr(engine, "_strong_pass", lambda *a: STRONG)
    monkeypatch.setattr(engine.transcript, "WORKER", None)
    new = engine._caption_step(clip, "w.mp4")
    row = runstate.clip(clip["clip_id"])
    assert row["words_path"].endswith("_cap.json") and new["words_path"] == row["words_path"]
    assert json.loads(row["features_json"])["captions"]["mode"] == "full"
    monkeypatch.setattr(engine, "_strong_pass", lambda *a: (_ for _ in ()).throw(
        RuntimeError("whisper died")))
    again = engine._caption_step(runstate.clip(clip["clip_id"]), "w.mp4")
    assert again["words_path"] == row["words_path"]               # the words on the row stay
    assert "whisper died" in json.loads(again["features_json"])["captions"]["error"]


def test_the_strong_model_is_a_caption_model_so_the_judge_stays_independent():
    from clipengine import config, second_asr
    cfg = config.load()
    assert cfg["render"]["caption_strong_model"] in second_asr.caption_models(cfg)
    assert second_asr.independent(cfg["render"]["caption_strong_model"], cfg)
    assert second_asr.independent(second_asr.settings()["model"], cfg) is None


def test_the_strong_pass_runs_in_the_warm_worker(monkeypatch, tmp_path):
    """the model loads once per window (worker), not per clip (CLI)."""
    import pathlib
    import subprocess
    from clipengine import strong_asr
    from clipengine import transcript
    seen = {}

    def fake_ffmpeg(cmd, **kw):
        pathlib.Path(cmd[-1]).write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0)

    def fake_worker(argv, timeout=None):
        seen["argv"] = argv
        out = pathlib.Path(argv[argv.index("--output_dir") + 1]) / "span.json"
        out.write_text(json.dumps({"segments": [{"words": [
            {"word": " flyweight", "start": 0.1, "end": 0.5, "probability": 0.9}]}]}))
    monkeypatch.setattr(subprocess, "run", fake_ffmpeg)
    monkeypatch.setattr(transcript, "WORKER", True)
    monkeypatch.setattr(transcript, "_worker_run", fake_worker)
    got = strong_asr.transcribe_strong("w.mp4", 10.0, 12.0, model="large-v3-turbo")
    assert got == [{"w": "flyweight", "start": 10.1, "end": 10.5, "p": 0.9}]
    assert seen["argv"][seen["argv"].index("--model") + 1] == "large-v3-turbo"
    assert "whisper" not in seen["argv"][0]


def test_a_number_whisper_split_in_two_is_burned_as_one_word():
    """299: whisper wrote "$60" + ",000," and the captions burned "60 ,000"."""
    import sys
    from pathlib import Path
    from clipengine.render import captions
    got = captions.clean_words([{"w": "about", "start": 0.0, "end": 0.3},
                                {"w": "$60", "start": 0.3, "end": 0.6},
                                {"w": ",000,", "start": 0.6, "end": 0.9},
                                {"w": "a", "start": 1.0, "end": 1.1},
                                {"w": "year,", "start": 1.1, "end": 1.4},
                                {"w": ",so", "start": 1.5, "end": 1.7}])
    assert [w["w"] for w in got] == ["about", "$60,000,", "a", "year,", ",so"]
    assert got[1]["start"] == 0.3 and got[1]["end"] == 0.9
