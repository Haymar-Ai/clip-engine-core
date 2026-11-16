""" render fix: a source's own dead band never reaches the canvas.

10-20 clip 108 (ka4yFVr8tQY_w01) was a 360x640 vertical stream whose layout ends in 72 black
rows. gamestack found no cam inset, the track fallback's 9:16 crop of a 9:16 source is the whole
frame, so the bottom 216 px (11.2 %) of every frame was solid black. The banner scan now finds
dead bands (render/edgebands.py) and the track crop leaves them out.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from clipengine.render import banners
from clipengine.render import edgebands
from clipengine.render import recompose
from clipengine import quality_gate as qg

VW, VH = 360, 640            # clip 108's source
BAND = 72                    # its dead rows


def _face(cx, cy=0.25, w=0.3, h=0.15):
    return {"cx": cx, "cy": cy, "w": w, "h": h, "score": 0.9}


def _sc(dead, graphics=(), faces=None):
    return {"src_w": VW, "src_h": VH, "frames": 10, "graphics": list(graphics), "panels": [],
            "faces": faces if faces is not None else [(t, [_face(0.5)]) for t in range(10)],
            "dead_bands": dead, "hgrad": None, "scale": VW / 640}


def _plan():
    return {"layout": "track", "src_w": VW, "src_h": VH,
            "segments": [{"t0": 0.0, "t1": 29.0, "x": 0, "cut": True}]}


# ---------------------------------------------------------------- pure band measure

def test_flat_edge_depths_find_a_solid_band_and_ignore_texture():
    rng = np.random.default_rng(1)
    g = rng.integers(40, 200, (VH, VW)).astype(np.uint8)
    g[-BAND:] = 0
    d = edgebands.flat_edge_depths(g)
    assert d["bottom"] == BAND and d["top"] == 0 and d["left"] == 0 and d["right"] == 0


def test_band_must_hold_one_depth_in_most_frames():
    fixed = [{"top": 0, "bottom": 107, "left": 0, "right": 0}] * 10
    assert edgebands.persistent_bands(fixed)["bottom"] == 107
    # a dark backdrop's flat region wanders with the speaker: no single depth in 60 % of frames
    wander = [{"top": v, "bottom": 0, "left": 0, "right": 0}
              for v in (93, 97, 101, 104, 108, 112, 95, 99, 110, 103)]
    assert edgebands.persistent_bands(wander)["top"] == 0
    # a band on only part of the clip is not "most frames"
    part = fixed[:5] + [{"top": 0, "bottom": 0, "left": 0, "right": 0}] * 5
    assert edgebands.persistent_bands(part)["bottom"] == 0


def test_scan_reports_the_dead_band_in_source_px():
    rng = np.random.default_rng(2)
    frames = []
    for i in range(8):
        f = rng.integers(30, 220, (1138, 640, 3)).astype(np.uint8)   # 640-wide scan of 360x640
        f[-128:] = 0                                                 # 72 src rows x 640/360
        frames.append((float(i), f))
    sc = banners.scan(None, 0, 8, frames=frames, src_size=(VW, VH))
    assert abs(sc["dead_bands"]["bottom"] - BAND) <= 1
    assert sc["dead_bands"]["top"] == 0


# ---------------------------------------------------------------- the crop leaves the band out

def test_full_frame_crop_of_a_vertical_source_drops_its_dead_rows():
    plan = _plan()
    rep = banners.trim_dead_bands(plan, _sc({"top": 0, "bottom": BAND, "left": 0, "right": 0}))
    assert rep["action"] == "trim"
    box = plan["crop_box"]
    assert box["y"] == 0 and box["y"] + box["h"] <= VH - BAND          # no dead row in the crop
    assert box["w"] == round(box["h"] * 9 / 16) // 2 * 2               # still 9:16: fills canvas
    x = plan["segments"][0]["x"]
    assert 0 <= x and x + box["w"] <= VW
    assert x + 0.01 * VW <= (0.5 - 0.15) * VW and (0.5 + 0.15) * VW <= x + box["w"] - 0.01 * VW


def test_no_band_or_a_thin_one_leaves_the_plan_alone():
    plan = _plan()
    assert banners.trim_dead_bands(plan, _sc({"top": 0, "bottom": 0}))["action"] == "none"
    assert banners.trim_dead_bands(plan, _sc({"bottom": 6}))["action"] == "none"   # < 2 %
    assert "crop_box" not in plan


def test_a_banner_the_narrower_window_would_slice_boxes_the_rows_instead():
    strip = {"box": [0, 520, 360, 560], "frac": 1.0, "banner": True}   # full-width HUD strip
    plan = _plan()
    rep = banners.trim_dead_bands(plan, _sc({"top": 0, "bottom": BAND}, graphics=[strip]))
    assert rep["action"] == "box"
    assert plan["crop_box"] == {"w": VW, "h": VH - BAND - banners.DEAD_BAND_MARGIN, "y": 0,
                                "fit": "box"}
    a = recompose.parse_args(["s.mp4", "0", "5", "o.mp4"])
    assert recompose.blur_fill_frac(plan, a, {}) < 0.5                # gate's blur-fill cap


def test_guard_banners_applies_the_dead_band_trim(monkeypatch):
    monkeypatch.setattr(banners, "scan",
                        lambda *a, **k: _sc({"top": 0, "bottom": BAND, "left": 0, "right": 0}))
    plan = _plan()
    a = recompose.parse_args(["s.mp4", "0", "29", "o.mp4", "--niche", "gaming"])
    rep = recompose.guard_banners(plan, a, "s.mp4")
    assert rep["action"] == "dead_band_trim" and rep["dead_band"]["action"] == "trim"
    assert plan["crop_box"]["h"] <= VH - BAND
    assert f"crop={plan['crop_box']['w']}:{plan['crop_box']['h']}:" in recompose.track_vf(plan)


# ---------------------------------------------------------------- real encode: band luma

@pytest.fixture(scope="module")
def banded_source(tmp_path_factory):
    """A 360x640 'vertical stream' whose bottom 72 rows are black, like clip 108's source."""
    p = tmp_path_factory.mktemp("deadband") / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc2=size={VW}x{VH - BAND}:rate=30:duration=3",
                    "-vf", f"pad={VW}:{VH}:0:0:black", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(p)], check=True)
    return p


def _render(src, plan, out):
    vf = recompose.track_vf(plan)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", "2",
                    "-filter_complex", vf, "-map", "[v0]", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", str(out)], check=True)
    return out


def test_rendered_canvas_has_no_black_band_after_the_fix(banded_source, tmp_path):
    sc = banners.scan(str(banded_source), 0, 3, src_size=(VW, VH))
    assert abs(sc["dead_bands"]["bottom"] - BAND) <= 2
    before = _render(banded_source, _plan(), tmp_path / "before.mp4")
    m_before = qg.measure_edge_bands(before)
    assert m_before["passed"] is False and m_before["band_frac"]["bottom"] >= 0.1   # 216/1920
    plan = _plan()
    sc["faces"] = [(t, [_face(0.5)]) for t in range(3)]
    assert banners.trim_dead_bands(plan, sc)["action"] == "trim"
    after = _render(banded_source, plan, tmp_path / "after.mp4")
    m_after = qg.measure_edge_bands(after)
    assert m_after["passed"] is True and m_after["band_frac"]["bottom"] == 0.0
    frames = [g for _t, g in qg._iter_gray_frames(after, 2.0, 540)]
    assert min(float(g[-108:].mean()) for g in frames) > 30           # bottom 216 px not black
