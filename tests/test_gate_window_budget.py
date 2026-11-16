"""The gate's second ASR and hook judge live inside the window's
cap. They use the window's warm whisper worker and stop GATE_TAIL_S before the cap, so a slow
pass turns checks 3-5 informational instead of the window being killed with its clip."""
from __future__ import annotations

import subprocess
import time

import pytest

from clipengine import second_asr
from clipengine import transcript


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(second_asr, "DEADLINE", None)
    monkeypatch.setattr(transcript, "WORKER", None)


def test_the_pass_is_capped_by_the_window_deadline(monkeypatch):
    assert second_asr._timeout(30.0) == second_asr.TIMEOUT_BASE_S + 30.0 * second_asr.TIMEOUT_PER_AUDIO_S
    monkeypatch.setattr(second_asr, "DEADLINE", time.monotonic() + 100.0)
    assert 95.0 < second_asr._timeout(30.0) <= 100.0
    monkeypatch.setattr(second_asr, "DEADLINE", time.monotonic() + 5.0)
    with pytest.raises(TimeoutError, match="window cap near"):
        second_asr._timeout(30.0)


def test_no_time_left_means_no_pass_and_informational_checks(monkeypatch, tmp_path):
    from clipengine import quality_gate
    monkeypatch.setattr(second_asr, "DEADLINE", time.monotonic() + 1.0)
    calls = []
    monkeypatch.setattr(second_asr, "extract_wav", lambda *a, **k: calls.append(a))
    media = tmp_path / "src.mp4"
    media.write_bytes(b"x")
    with pytest.raises(TimeoutError):
        second_asr.transcribe(str(media), 0.0, 20.0, media_dur=60.0,
                              cache_dir=str(tmp_path / "c"))
    assert calls == []                                    # not even the wav extract
    monkeypatch.setattr(quality_gate, "measure_edge_bands", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(quality_gate, "measure_crop_slices", lambda *a, **k: {"passed": True})
    rep = quality_gate.clip_checks(str(media), words=None, start_s=0.0, end_s=20.0,
                                     hook="A hook", src_media=str(media),
                                     hook_judge=lambda h, t: {"supported": True})
    assert "TimeoutError" in rep["second_asr"]["error"]
    assert rep["caption_asr"]["passed"] is None and rep["ending"]["passed"] is None


def test_a_daily_window_runs_the_pass_in_the_warm_worker(monkeypatch, tmp_path):
    seen = {}

    def fake_worker(argv, timeout=None):
        seen.update(argv=argv, timeout=timeout)
        out = argv[argv.index("--output_dir") + 1]
        (tmp_path / "w.json").write_text("")
        import json, pathlib
        pathlib.Path(out, "clip.json").write_text(json.dumps(
            {"segments": [{"words": [{"word": " Hello", "start": 0.1, "end": 0.4}]}]}))

    def no_cli(*a, **k):
        raise AssertionError("the CLI must not start when the worker is on")
    monkeypatch.setattr(transcript, "WORKER", True)
    monkeypatch.setattr(transcript, "_worker_run", fake_worker)
    monkeypatch.setattr(subprocess, "run", no_cli)
    words = second_asr.whisper_cli_words(tmp_path / "clip.wav", "small", threads=4, timeout=77.0)
    assert words == [{"w": "Hello", "start": 0.1, "end": 0.4}]
    assert seen["timeout"] == 77.0 and seen["argv"][seen["argv"].index("--model") + 1] == "small"


def test_the_hook_judge_is_not_asked_without_time_left(monkeypatch):
    from clipengine import quality_gate
    from clipengine import select_semantic as ss
    asked = []
    monkeypatch.setattr(ss, "_claude_cli", lambda prompt, mdl, t, opts: asked.append(t) or
                        '{"supported": true, "quote": "q", "reason": "r"}')
    judge = quality_gate.make_hook_ground_judge({}, timeout_s=120)
    monkeypatch.setattr(second_asr, "DEADLINE", time.monotonic() + 60.0)
    assert judge("h", "t")["supported"] is True and 50.0 < asked[-1] <= 55.0
    monkeypatch.setattr(second_asr, "DEADLINE", time.monotonic() + 8.0)
    assert "window cap near" in judge("h", "t")["error"] and len(asked) == 1
