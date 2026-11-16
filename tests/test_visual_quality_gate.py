"""the quality gate reads brightness, contrast and sharpness off the
rendered pixels and fails a canvas the blurred fill dominates (streamers tw_2899188846 clip_2
passed every gate as a dark game frame letterboxed between two blur fills: median luma 11.9,
luma std 26.9, 68 % blur fill). Both the full gate and the stream lean gate run it.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from clipengine import quality_gate as qg
from clipengine.render import recompose
from clipengine import engine


def _stats(luma=100.0, contrast=60.0, sharpness=3000.0, n=10):
    return [{"luma": luma, "contrast": contrast, "sharpness": sharpness}] * n


def test_verdict_rules():
    assert qg.visual_quality_verdict(_stats())["passed"] is True
    dark = qg.visual_quality_verdict(_stats(luma=11.9, contrast=26.9), blur_fill_frac=0.684)
    assert dark["passed"] is False and len(dark["errors"]) == 3      # the clip_2 numbers
    assert any("too dark" in e for e in dark["errors"])
    assert any("low contrast" in e for e in dark["errors"])
    assert any("blurred fill covers 68%" in e for e in dark["errors"])
    assert qg.visual_quality_verdict(_stats(sharpness=150))["errors"][0].startswith("blurry")
    assert qg.visual_quality_verdict(_stats(), blur_fill_frac=0.0)["passed"] is True
    assert qg.visual_quality_verdict([])["passed"] is None


def test_frame_stats_read_the_pixels():
    import cv2
    rng = np.random.default_rng(3)
    sharp = np.kron(rng.integers(0, 255, (240, 135)), np.ones((8, 8))).astype(np.uint8)
    s = qg.frame_visual_stats(sharp)
    assert s["luma"] > 100 and s["contrast"] > 60 and s["sharpness"] > qg.VQ_MIN_SHARPNESS
    blurred = cv2.GaussianBlur(sharp, (0, 0), 25)
    assert qg.frame_visual_stats(blurred)["sharpness"] < qg.VQ_MIN_SHARPNESS
    assert qg.frame_visual_stats((sharp // 12).astype(np.uint8))["luma"] < qg.VQ_MIN_LUMA


def test_decide_fails_on_visual_quality():
    verdict, reason = qg.decide({"visual_quality": {"passed": False,
                                                     "errors": ["too dark: median luma 11.9"]}})
    assert verdict == "fail" and reason.startswith("visual_quality: too dark")


def test_stream_lean_gate_fails_a_dark_clip(monkeypatch, tmp_path):
    monkeypatch.setattr(engine.runstate, "source",
                        lambda sid: {"source_id": sid, "vertical": "stream"})
    monkeypatch.setattr(engine, "_measure_black_frozen",
                        lambda *a, **kw: {"passed": True, "black_total_s": 0.0,
                                          "frozen_total_s": 0.0, "dur_s": 22.6})
    monkeypatch.setattr(engine, "_load_words_sidecar", lambda p: [
        {"w": "open", "start": 0.0, "end": 0.6}, {"w": "his", "start": 0.7, "end": 1.6}])
    seen = {}

    def fake_vq(path, blur_fill_frac=None):
        seen["fill"] = blur_fill_frac
        return qg.visual_quality_verdict(_stats(luma=11.9, contrast=26.9), blur_fill_frac)
    monkeypatch.setattr(qg, "measure_visual_quality", fake_vq)
    clip = {"source_id": "tw_2899188846_w00", "start_s": 41.355, "end_s": 64.016,
            "layout": "auto", "words_path": "/tmp/w.json"}
    qc = {"integrated_lufs": -14.0, "layout": "general", "blur_fill_frac": 0.684}
    rep = engine._run_quality_gate(tmp_path / "x.mp4", clip, qc=qc)
    assert seen["fill"] == 0.684
    assert rep["verdict"] == "fail" and "visual_quality" in rep["drop_reason"]
    assert "too dark" in rep["drop_reason"]


def _encode(tmp_path, name, vf, src="testsrc2=size=1280x720:rate=30:duration=2"):
    out = tmp_path / name
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", src,
                    "-filter_complex", vf, "-map", "[v0]", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def test_real_letterbox_of_a_dark_frame_fails_and_full_bleed_passes(tmp_path):
    dark_letterbox = _encode(tmp_path, "dark.mp4",
                             "[0:v]eq=brightness=-0.42:contrast=0.5[d];"
                             + recompose.general_vf().replace("[0:v]", "[d]"))
    a = recompose.parse_args(["s.mp4", "0", "2", "o.mp4"])
    fill = recompose.blur_fill_frac({"layout": "general", "src_w": 1280, "src_h": 720}, a, {})
    bad = qg.measure_visual_quality(dark_letterbox, blur_fill_frac=fill)
    assert bad["passed"] is False
    assert any("too dark" in e for e in bad["errors"])
    assert any("blurred fill" in e for e in bad["errors"])
    full = _encode(tmp_path, "full.mp4", "[0:v]crop=404:720:438:0,scale=1080:1920[v0]")
    good = qg.measure_visual_quality(full, blur_fill_frac=0.0)
    assert good["passed"] is True, good
