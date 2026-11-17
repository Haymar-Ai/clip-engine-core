"""The crop probe reads a source the way a window runner does: one harvested window at a time.

A window runner harvests windows of at most 72 s (60 s spans with 6 s of context each side) and
probes each window file on its own (cropcheck.probe(window, 0.0, None)); an uncroppable window is
skipped before select. A long source probed as a whole reads a cam card or a logo that shows on
most of it as a plate on every frame and is refused although each window of it crops clean. So:
  - a source no longer than one window gets exactly the window runner's one probe call;
  - a longer one is probed as consecutive windows of at most PROBE_WINDOW_S, each with its own
    verdict, its frame verdicts in source seconds;
  - the source is skipped before transcription only when every window is uncroppable, and a
    candidate with most of its span in an uncroppable window leaves before its render;
  - probing a window of the source reads the same frames as probing that window cut to a file
    (real media: the verdicts are identical).
"""
import subprocess

import numpy as np
import pytest

from clipengine import engine
from clipengine import runstate
from clipengine.render import cropcheck
from clipengine.render import reframe


def _rep(verdict, verdicts=(), why=None, plates=0, frames=None):
    vs = [[float(t), ok, how] for t, ok, how in verdicts]
    share, n = cropcheck.share_ok([tuple(v) for v in vs])
    return {"verdict": verdict, "why": why or verdict, "verdicts": vs, "ok_share": share,
            "verdict_frames": n, "plates": plates, "frames": frames or len(vs), "elapsed_s": 1.0}


@pytest.fixture
def probes(monkeypatch):
    """cropcheck.probe replaced by a recorder; `answer(start)` gives each window's report."""
    calls = []

    def install(dur, answer):
        monkeypatch.setattr(engine, "_media_duration", lambda m: dur)

        def probe(media, start=0.0, dur=None, niche=None, vertical=None, stack=False,
                  variants=False):
            calls.append({"media": media, "start": start, "dur": dur, "niche": niche,
                          "stack": stack})
            return answer(start)
        monkeypatch.setattr(cropcheck, "probe", probe)
        return calls
    return install


# ---------------------------------------------------------------- the windows

def test_probe_windows_are_harvested_window_lengths():
    assert engine.PROBE_WINDOW_S == 72.0
    for dur in (None, 0, 30.0, 71.9, 72.0):
        assert engine.probe_windows(dur) == [(0.0, None)]          # one window: as harvested
    ws = engine.probe_windows(1500.0)
    assert len(ws) == 21 and ws[0][0] == 0.0 and ws[-1][1] == 1500.0
    assert all(b - a <= 72.0 + 1e-6 for a, b in ws)
    assert all(abs((b - a) - 1500.0 / 21) < 0.01 for a, b in ws)    # equal, no stub at the end
    assert all(ws[k][1] == ws[k + 1][0] for k in range(len(ws) - 1))   # consecutive, no gap
    assert engine.probe_windows(73.0) == [(0.0, 36.5), (36.5, 73.0)]


def test_a_source_of_one_window_gets_the_window_runners_own_probe(probes):
    """The call a window runner makes for a harvested window file, and its report unchanged."""
    rep = _rep("ok", [(t, True, "full_bleed") for t in range(60)])
    calls = probes(65.0, lambda start: rep)
    assert engine.crop_probe("w.mp4", "s_w00", "gaming") is rep
    assert calls == [{"media": "w.mp4", "start": 0.0, "dur": None, "niche": "gaming",
                      "stack": True}]
    calls.clear()
    assert engine.crop_probe("w.mp4", "s_w00", "podcasts") is rep
    assert calls[0]["stack"] is False and calls[0]["dur"] is None


def test_no_niche_or_probe_off_reads_nothing(probes, monkeypatch):
    calls = probes(1500.0, lambda start: _rep("ok"))
    assert engine.crop_probe("s.mp4", "s", None) is None
    monkeypatch.setattr(engine, "CROP_PROBE", False)
    assert engine.crop_probe("s.mp4", "s", "gaming") is None
    assert calls == []


def test_a_long_source_is_probed_window_by_window(probes):
    """Each window its own probe call and verdict; frame verdicts move to source seconds."""
    def answer(start):
        if 60.0 <= start < 120.0:              # one window shows a sliced cover graphic
            return _rep("uncroppable", [(t, False, "sliced") for t in range(60)],
                        why="no clean crop for the speaker on 60 of 60 sampled frames")
        return _rep("ok", [(t, True, "full_bleed") for t in range(60)])
    calls = probes(300.0, answer)
    probe = engine.crop_probe("s.mp4", "s", "streamers")
    assert [(c["start"], c["dur"]) for c in calls] == [(0.0, 60.0), (60.0, 60.0), (120.0, 60.0),
                                                       (180.0, 60.0), (240.0, 60.0)]
    assert all(c["niche"] == "streamers" and c["stack"] is True for c in calls)
    assert [w["verdict"] for w in probe["windows"]] == ["ok", "uncroppable", "ok", "ok", "ok"]
    assert probe["verdict"] == "ok" and probe["why"] == "clean crop in 4 of 5 probe windows " \
                                                        "(1 uncroppable)"
    ts = [v[0] for v in probe["verdicts"]]
    assert ts[:2] == [0.0, 1.0] and ts[60:62] == [60.0, 61.0] and ts[-1] == 299.0
    assert {v[2] for v in probe["verdicts"] if 60.0 <= v[0] < 120.0} == {"sliced"}
    assert probe["frames"] == 300 and probe["elapsed_s"] == 5.0
    msgs = [e["message"] for e in runstate.events()]
    assert any("crop probe in 5 windows of <= 72s" in m and "60-120s" in m for m in msgs)


def test_the_source_is_refused_only_when_every_window_is():
    bad = _rep("uncroppable", [(t, False, "sliced") for t in range(10)], why="w0 why")
    unk = _rep("unknown", [(t, None, "no face") for t in range(10)])
    ok = _rep("ok", [(t, True, "full_bleed") for t in range(10)])
    every = engine.merge_window_probes([(0.0, 60.0, bad), (60.0, 120.0, bad)])
    assert every["verdict"] == engine.UNCROPPABLE
    assert every["why"] == "no clean crop for the speaker in any of 2 probe windows (first: w0 why)"
    some = engine.merge_window_probes([(0.0, 60.0, bad), (60.0, 120.0, unk)])
    assert some["verdict"] == "unknown"        # a window runner would select the unread window
    assert engine.merge_window_probes([(0.0, 60.0, unk), (60.0, 120.0, ok)])["verdict"] == "ok"


def test_a_window_the_probe_cannot_read_has_no_verdict(probes):
    def answer(start):
        if start == 0.0:
            raise RuntimeError("unreadable")
        return _rep("uncroppable", [(t, False, "sliced") for t in range(10)])
    probes(144.0, answer)
    probe = engine.crop_probe("s.mp4", "s", "gaming")
    assert [w["verdict"] for w in probe["windows"]] == ["unknown", "uncroppable"]
    assert probe["verdict"] == "unknown"       # not every window refused: select goes on


# ---------------------------------------------------------------- select

@pytest.fixture
def src(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "SCRATCH", tmp_path / "scratch")
    runstate.reset()
    media = tmp_path / "long.mp4"
    media.write_bytes(b"\0")
    runstate.add_source("long", local_path=str(media), vertical="stream", niche="gaming")
    runstate.update_source("long", duration_s=1500.0, state="harvested")
    return runstate.source("long")


def test_select_skips_a_source_every_window_refuses_before_transcribing(src, probes,
                                                                        monkeypatch):
    calls = probes(1500.0, lambda start: _rep("uncroppable", [(t, False, "sliced")
                                                              for t in range(70)],
                                              why="no clean crop for the speaker on 70 of 70 "
                                                  "sampled frames (1 graphic plate(s))"))
    monkeypatch.setattr(engine.transcript, "get_words",
                        lambda *a, **k: pytest.fail("a refused source is never transcribed"))
    assert engine.select(src) is False
    assert len(calls) == 21
    row = runstate.source("long")
    assert row["state"] == "failed"
    assert row["error"].startswith("uncroppable: no clean crop for the speaker in any of 21 "
                                   "probe windows")


def test_select_goes_on_when_one_window_crops_clean(src, probes, monkeypatch):
    probes(1500.0, lambda start: _rep("ok" if start == 0.0 else "uncroppable",
                                      [(t, start == 0.0, "x") for t in range(70)]))
    seen = []

    def words(*a, **k):
        seen.append("transcribe")
        raise RuntimeError("stop here")
    monkeypatch.setattr(engine.transcript, "get_words", words)
    monkeypatch.setattr(engine, "energy_envelope", lambda media: (_ for _ in ()).throw(
        RuntimeError("no audio")))
    engine.select(src)
    assert seen == ["transcribe"]
    assert not str(runstate.source("long").get("error") or "").startswith("uncroppable")


def test_a_candidate_in_a_refused_window_leaves_before_its_render():
    reps = [(0.0, 60.0, _rep("ok", [(t, True, "full_bleed") for t in range(60)])),
            (60.0, 120.0, _rep("uncroppable", [(t, t >= 30, "x") for t in range(60)])),
            (120.0, 180.0, _rep("ok", [(t, True, "full_bleed") for t in range(60)]))]
    probe = engine.merge_window_probes(reps)
    cands = [{"start": 10.0, "end": 40.0},      # in a clean window
             {"start": 95.0, "end": 115.0},     # all in the refused window (its frames read ok)
             {"start": 50.0, "end": 75.0},      # 15 of 25 s in the refused window
             {"start": 110.0, "end": 140.0}]    # 10 of 30 s in it
    kept = engine._drop_uncroppable_picks(cands, probe, "s")
    assert kept == [cands[0], cands[3]]
    assert cands[0]["crop_probe"] == {"ok_share": 1.0, "frames": 31}
    msgs = [e["message"] for e in runstate.events()]
    assert any("95.0-115.0s before render: 20.0s of it in probe windows with no clean crop" in m
               for m in msgs)


def test_one_window_probe_reports_drop_picks_as_before():
    """No "windows" key (a source of one window): only the span rule, unchanged."""
    probe = {"verdicts": [[float(t), t >= 10, "x"] for t in range(40)]}
    cands = [{"start": 0.0, "end": 12.0}, {"start": 15.0, "end": 35.0}]
    assert engine._drop_uncroppable_picks(cands, probe, "w") == [cands[1]]
    assert engine._drop_uncroppable_picks(cands, {"verdicts": []}, "w") == cands


# ---------------------------------------------------------------- real media

W, H, FPS = 1280, 720, 30


def _red_face(frame):
    """A fake face detector: the red marker block (BGR) is the face, normalized like
    reframe.detect_faces."""
    fh, fw = frame.shape[:2]
    m = (frame[..., 2] > 200) & (frame[..., 1] < 60) & (frame[..., 0] < 60)
    ys, xs = np.nonzero(m)
    if len(xs) < 20:
        return []
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return [{"cx": round((x0 + x1) / 2 / fw, 4), "cy": round((y0 + y1) / 2 / fh, 4),
             "w": round((x1 - x0) / fw, 4), "h": round((y1 - y0) / fh, 4), "score": 0.9}]


def _source(path):
    """A 30 s 1280x720 source: a slowly moving gradient, a red 'face' that walks across the frame
    and is on screen 4 s of every 7 (each window has its own pattern of face / no-face frames), and
    from 12 s on a light plate with dark bars (a banner) in its lower third."""
    graph = ("[0:v]format=yuv420p[g];[g][1:v]overlay=x='200+28*t':y=180:shortest=1:"
             "enable='lt(mod(t,7),4)',"
             "drawbox=x=380:y=470:w=520:h=120:color=0xf0f0f0:t=fill:enable='gte(t,12)',"
             "drawbox=x=400:y=490:w=480:h=14:color=0x101010:t=fill:enable='gte(t,12)',"
             "drawbox=x=400:y=520:w=430:h=14:color=0x101010:t=fill:enable='gte(t,12)',"
             "drawbox=x=400:y=550:w=460:h=14:color=0x101010:t=fill:enable='gte(t,12)'[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"gradients=s={W}x{H}:r={FPS}:speed=0.01:seed=3:d=30", "-f", "lavfi", "-i",
                    f"color=c=red:s=150x260:r={FPS}:d=30", "-filter_complex", graph,
                    "-map", "[v]", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-an",
                    str(path)], check=True)
    return path


def _cut(src, start, dur, path):
    """The harvested window: [start, start + dur] of the source cut to its own file, losslessly
    (every decoded frame is the source's)."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(dur), "-i",
                    str(src), "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-an",
                    str(path)], check=True)
    return path


def test_each_probe_window_reads_as_its_harvested_window_file(tmp_path, monkeypatch):
    """The engine's windows of a source against the same windows cut to files and probed the
    way a window runner probes a harvested window: the same frames, plates, verdicts and why."""
    monkeypatch.setattr(reframe, "detect_faces", _red_face)
    src = _source(tmp_path / "src.mp4")
    monkeypatch.setattr(engine, "PROBE_WINDOW_S", 12.5)
    merged = engine.crop_probe(str(src), "src", "gaming")
    wins = merged["windows"]
    assert [(w["start"], w["end"]) for w in wins] == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    for w in wins:
        a, b = w["start"], w["end"]
        window = _cut(src, a, b - a, tmp_path / f"w{int(a)}.mp4")
        as_file = cropcheck.probe(str(window), 0.0, None, niche="gaming", stack=True)
        assert as_file["verdict_frames"] >= cropcheck.MIN_VERDICT_FRAMES
        for k in ("verdict", "why", "ok_share", "verdict_frames", "plates"):
            assert w[k] == as_file[k], (a, k, w[k], as_file[k])
        mine = [v for v in merged["verdicts"] if a <= v[0] < b]
        assert mine == [[round(float(t) + a, 2), ok, how] for t, ok, how in as_file["verdicts"]]
    assert merged["verdict_frames"] == sum(w["verdict_frames"] for w in wins)
