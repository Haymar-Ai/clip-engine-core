"""Tests for the stream-vertical lean gate.

Covers each measure in `engine._run_stream_lean_gate`:
  - duration_bounds (happy + both error paths)
  - audio_lufs (happy + too quiet + too loud + missing)
  - caption_presence (happy + empty)
  - black_frozen (via a stubbed ffmpeg runner)
  - the top-level verdict collapse: any `passed=False` -> `fail`, else `pass`.

Also covers the entry point: `_run_quality_gate` for a `vertical=stream` source routes to
the lean gate and does NOT short-circuit to pass without reading pixels.
"""
import pathlib
import subprocess
import types

import pytest

from clipengine import engine


# ---- unit measures ---------------------------------------------------------

def test_duration_bounds_pass():
    r = engine._measure_duration_bounds(10.0, 42.0)
    assert r["passed"] is True
    assert r["dur_s"] == 32.0


def test_duration_bounds_too_short():
    r = engine._measure_duration_bounds(10.0, 13.0)
    assert r["passed"] is False
    assert "too short" in r["errors"][0]


def test_duration_bounds_too_long():
    r = engine._measure_duration_bounds(0.0, 120.0)
    assert r["passed"] is False
    assert "too long" in r["errors"][0]


def test_duration_bounds_missing_inputs():
    r = engine._measure_duration_bounds(None, None)
    assert r["passed"] is None


def test_lufs_bounds_pass():
    r = engine._measure_lufs_bounds(-14.0)
    assert r["passed"] is True


def test_lufs_bounds_too_quiet():
    r = engine._measure_lufs_bounds(-20.0)
    assert r["passed"] is False
    assert "too quiet" in r["errors"][0]


def test_lufs_bounds_too_loud():
    r = engine._measure_lufs_bounds(-6.0)
    assert r["passed"] is False
    assert "too loud" in r["errors"][0]


def test_lufs_bounds_missing():
    r = engine._measure_lufs_bounds(None)
    assert r["passed"] is None


def test_caption_presence_pass():
    r = engine._measure_caption_presence([
        {"w": "hello", "start": 0.0, "end": 0.5},
        {"w": "world", "start": 0.6, "end": 1.4},
    ])
    assert r["passed"] is True


def test_caption_presence_empty_fails():
    r = engine._measure_caption_presence([])
    assert r["passed"] is False
    assert "no transcribed words" in r["errors"][0]


def test_caption_presence_too_little_speech_fails():
    r = engine._measure_caption_presence([{"w": "a", "start": 0.0, "end": 0.1}])
    assert r["passed"] is False
    assert "< 1 s" in r["errors"][0]


# ---- black/frozen (stubbed ffmpeg) ----------------------------------------

def test_black_frozen_pass(monkeypatch, tmp_path):
    """ffmpeg stderr contains no black/freeze matches -> passed=True."""
    mp4 = tmp_path / "x.mp4"
    mp4.write_bytes(b"\x00")   # file just needs to exist for the real codepath

    def _fake_run(cmd, *a, **kw):
        if cmd[0] == "ffprobe":
            return types.SimpleNamespace(returncode=0, stdout="42.0\n", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="no issues")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    r = engine._measure_black_frozen(mp4)
    assert r["passed"] is True
    assert r["black_total_s"] == 0.0
    assert r["frozen_total_s"] == 0.0


def test_black_frozen_fail_on_too_much_black(monkeypatch, tmp_path):
    mp4 = tmp_path / "x.mp4"
    mp4.write_bytes(b"\x00")

    def _fake_run(cmd, *a, **kw):
        if cmd[0] == "ffprobe":
            return types.SimpleNamespace(returncode=0, stdout="30.0\n", stderr="")
        stderr = ("[blackdetect @ 0x1] black_start:0 black_end:5 black_duration:5.00\n"
                   "[blackdetect @ 0x2] black_start:10 black_end:11 black_duration:1.00\n")
        return types.SimpleNamespace(returncode=0, stdout="", stderr=stderr)
    monkeypatch.setattr(subprocess, "run", _fake_run)
    r = engine._measure_black_frozen(mp4)
    # 6 s black / 30 s dur = 20 % -> fails the 2 % cap
    assert r["passed"] is False
    assert "black_fraction" in r["errors"][0]


# ---- top-level entry -------------------------------------------------------

def test_run_quality_gate_routes_stream_vertical_to_lean(monkeypatch, tmp_path):
    """`vertical=stream` or layout in {gamestack, broadcast, ...} must route to the lean
    gate — no silent short-circuit to pass. This is the exact review regression that
    produced `note='gate skipped: stream/auto'` in READY2."""
    # Fake runstate.source to return a stream vertical row.
    monkeypatch.setattr(engine.runstate, "source",
                        lambda sid: {"source_id": sid, "vertical": "stream"})
    # Stub out the heavy measures so the test is deterministic + offline.
    monkeypatch.setattr(engine, "_measure_black_frozen",
                        lambda *a, **kw: {"passed": True, "black_total_s": 0.0,
                                           "frozen_total_s": 0.0, "dur_s": 30.0})
    clip = {"source_id": "s1", "start_s": 2.0, "end_s": 32.0, "layout": "auto",
             "words_path": None}
    qc = {"integrated_lufs": -14.1, "layout": "track"}
    rep = engine._run_quality_gate(tmp_path / "x.mp4", clip, qc=qc)
    # It is NOT a short-circuit: the report has the lean-gate measures.
    assert "duration_bounds" in rep
    assert "audio_lufs" in rep
    assert "black_frozen" in rep
    # No words -> caption_presence fails -> verdict=fail
    assert rep["verdict"] == "fail"
    assert "caption_presence" in rep["drop_reason"]


def test_run_quality_gate_pass_when_measures_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(engine.runstate, "source",
                        lambda sid: {"source_id": sid, "vertical": "stream"})
    monkeypatch.setattr(engine, "_measure_black_frozen",
                        lambda *a, **kw: {"passed": True, "black_total_s": 0.0,
                                           "frozen_total_s": 0.0, "dur_s": 30.0})
    monkeypatch.setattr(engine, "_load_words_sidecar", lambda p: [
        {"w": "hello", "start": 0.0, "end": 0.5},
        {"w": "world", "start": 0.6, "end": 1.5},
        {"w": "clip", "start": 2.0, "end": 2.5},
    ])
    monkeypatch.setattr(engine, "_hook_judge", lambda: None)     # no LLM call in a unit test
    # the lean gate needs a hook now (10-20 clip 113 shipped with hook=None)
    clip = {"source_id": "s1", "start_s": 2.0, "end_s": 32.0, "layout": "auto",
             "words_path": "/tmp/w.json", "hook": "The clip"}   # "clip" is said in the cut
    qc = {"integrated_lufs": -14.1, "layout": "track"}
    rep = engine._run_quality_gate(tmp_path / "x.mp4", clip, qc=qc)
    assert rep["verdict"] == "pass", rep
    assert rep["note"] == "stream lean gate"


# ----: the five new checks run on the lean gate too -------------------------

def _lean(monkeypatch, tmp_path, clip_extra=None, hay=None):
    monkeypatch.setattr(engine.runstate, "source",
                        lambda sid: {"source_id": sid, "vertical": "stream",
                                     "local_path": "/src/window.mp4"})
    monkeypatch.setattr(engine, "_measure_black_frozen",
                        lambda *a, **kw: {"passed": True, "black_total_s": 0.0,
                                           "frozen_total_s": 0.0, "dur_s": 30.0})
    monkeypatch.setattr(engine, "_load_words_sidecar", lambda p: [
        {"w": "hello", "start": 2.0, "end": 2.5}, {"w": "world.", "start": 2.6, "end": 3.5}])
    monkeypatch.setattr(engine.quality_gate, "measure_visual_quality",
                        lambda *a, **k: {"passed": True})
    monkeypatch.setattr(engine, "_hook_judge", lambda: "JUDGE")
    seen = {}

    def fake(mp4, **kw):
        seen.update(kw)
        return hay or {k: {"passed": True} for k in engine.quality_gate.CLIP_CHECK_KEYS}
    monkeypatch.setattr(engine.quality_gate, "clip_checks", fake)
    clip = {"source_id": "s1", "start_s": 2.0, "end_s": 32.0, "layout": "auto",
            "words_path": "/tmp/w.json", "hook": "Hello world", **(clip_extra or {})}
    qc = {"integrated_lufs": -14.1, "layout": "track",
          "crop_views": [{"t0": 0, "t1": 30, "x": 0, "y": 0, "w": 10, "h": 10}]}
    return engine._run_quality_gate(tmp_path / "x.mp4", clip, qc=qc), seen


def test_lean_gate_passes_source_hook_views_and_judge_to_the_clip_checks(monkeypatch,
                                                                           tmp_path):
    rep, seen = _lean(monkeypatch, tmp_path)
    assert rep["verdict"] == "pass" and rep["note"] == "stream lean gate"
    assert seen["src_media"] == "/src/window.mp4" and seen["hook"] == "Hello world"
    assert seen["crop_views"][0]["w"] == 10 and seen["hook_judge"] == "JUDGE"
    assert (seen["start_s"], seen["end_s"]) == (2.0, 32.0)
    assert set(engine.quality_gate.CLIP_CHECK_KEYS) <= set(rep)


@pytest.mark.parametrize("key", ["edge_bands", "crop_slices", "hook_grounding", "ending"])
def test_lean_gate_fails_on_any_clip_check(monkeypatch, tmp_path, key):
    hay = {k: {"passed": True} for k in engine.quality_gate.CLIP_CHECK_KEYS}
    hay[key] = {"passed": False, "errors": ["10-20 eye-check reject"]}
    rep, _ = _lean(monkeypatch, tmp_path, hay=hay)
    assert rep["verdict"] == "fail" and f"{key}: 10-20 eye-check reject" in rep["drop_reason"]


def test_lean_gate_reads_a_failed_caption_check_as_a_warning(monkeypatch, tmp_path):
    """yield 10-22 Amendment 1: captions never reject a clip."""
    hay = {k: {"passed": True} for k in engine.quality_gate.CLIP_CHECK_KEYS}
    hay["caption_asr"] = {"passed": False, "errors": ["10-20 eye-check reject"]}
    rep, _ = _lean(monkeypatch, tmp_path, hay=hay)
    assert rep["verdict"] == "pass" and rep["caption_warning"] == "10-20 eye-check reject"


def test_lean_gate_drops_a_clip_without_a_hook_end_to_end(monkeypatch, tmp_path):
    """10-20 clip 113 (streamers, hook=None) passed the lean gate. The real checks now run:
    the mp4 / source are missing here, so only the hook rule can decide - and it fails."""
    monkeypatch.setattr(engine.runstate, "source",
                        lambda sid: {"source_id": sid, "vertical": "stream"})
    monkeypatch.setattr(engine, "_measure_black_frozen",
                        lambda *a, **kw: {"passed": True, "black_total_s": 0.0,
                                           "frozen_total_s": 0.0, "dur_s": 30.0})
    monkeypatch.setattr(engine, "_load_words_sidecar", lambda p: [
        {"w": "all", "start": 46.6, "end": 47.0}, {"w": "right,", "start": 47.0, "end": 47.4},
        {"w": "man.", "start": 47.5, "end": 47.6}])
    monkeypatch.setattr(engine, "_hook_judge", lambda: None)
    clip = {"source_id": "tw_2904552549_w02", "start_s": 23.19, "end_s": 47.99,
            "layout": "auto", "words_path": "/tmp/w.json", "hook": None}
    rep = engine._run_quality_gate(tmp_path / "x.mp4", clip,
                                   qc={"integrated_lufs": -14.0, "layout": "gamestack"})
    assert rep["verdict"] == "fail" and "hook_grounding: no hook" in rep["drop_reason"]
