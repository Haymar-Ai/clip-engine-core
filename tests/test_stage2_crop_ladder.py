"""clip-quality 10-21 stage 2: a speaker crop that slices a graphic is edited, not rejected. The
planned crop is measured as the gate will measure the render; the ladder then tries (a) the clean
9:16 recrop, (b) the speaker's own tile on a boxed broadcast, (c) a slice-free box (or the whole
frame) fit to width over a blurred fill. Offline: synthetic scans, no OCR, no frames."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from clipengine.render import cleancrop
from clipengine.render import recompose

W, H = 1280, 720


def _face(cx, cy, h=0.2, w=0.12, t=1.0):
    return (t, [{"cx": cx, "cy": cy, "w": w, "h": h, "score": 0.9}])


def _plan(xs=(438,), w=W, h=H):
    return {"layout": "track", "src_w": w, "src_h": h,
            "segments": [{"t0": i * 5.0, "t1": (i + 1) * 5.0, "x": x} for i, x in enumerate(xs)]}


def _scan(graphics=(), faces=None, w=W, h=H):
    return {"src_w": w, "src_h": h, "frames": 10,
            "graphics": [{"box": list(b), "frac": 0.9, "banner": ban} for b, ban in graphics],
            "panels": [], "dead_bands": {},
            "faces": faces if faces is not None else [_face(0.5, 0.35)]}


class A:
    start, dur = 0.0, 10.0


@pytest.fixture
def ladder(monkeypatch):
    """slice_ladder with the scan, OCR and window scan replaced by `sc` / `wins`."""
    from clipengine import quality_gate

    def run(plan, sc, wins=None, guard=None, rejected=None):
        recompose._SCANS.clear()
        monkeypatch.setattr(recompose, "_scan", lambda src, a, size: sc)
        monkeypatch.setattr(quality_gate, "ocr_source_text", lambda *a, **k: None)
        monkeypatch.setattr(cleancrop, "window_boxes", lambda *a, **k: wins)
        return recompose.slice_ladder(plan, A(), "src.mp4", guard, rejected)
    return run


def _slices(plan, sc):
    from clipengine import quality_gate
    v = quality_gate.crop_slice_verdict(quality_gate.crop_slices(
        sc, recompose.crop_views(plan, A.dur), A.dur, words=None))
    return v["crop_share"]


# ---------------------------------------------------------------- cleancrop.clean_box

def test_clean_box_holds_a_graphic_whole_beside_the_speaker():
    """472: the POT / board graphic beside the player; no 9:16 window keeps both, a box does."""
    sc = _scan(graphics=[((700, 400, 1000, 440), False)])        # too wide to sit beside the
    ex, ns = cleancrop.regions_from_scan(sc, strict=False)        # face in a 9:16 window
    with pytest.raises(cleancrop.CleanCropError):
        cleancrop.clean_crop(_plan(), sc, ex, ns)
    box = cleancrop.clean_box(_plan(), sc, ex, ns)
    x, w = box["x"][0], box["w"]
    assert box["fit"] == "box" and x <= 700 and x + w >= 1000  # whole, beside the face
    assert w <= cleancrop.BOX_MAX_ASPECT * box["h"] + 2 and box["blur_fill"] <= 0.5


def test_clean_box_refuses_when_nothing_fits():
    sc = _scan(graphics=[((0, 200, 1280, 230), False)], faces=[_face(0.5, 0.45, h=0.3)])
    with pytest.raises(cleancrop.CleanCropError, match="no box"):
        cleancrop.clean_box(_plan(), sc, *cleancrop.regions_from_scan(sc, strict=False))


# ---------------------------------------------------------------- recompose.slice_ladder

def test_a_clean_crop_is_left_alone(ladder):
    plan, sc = _plan(), _scan()
    guard = {"action": "none"}
    assert ladder(plan, sc, guard=guard) is guard and "crop_box" not in plan


def test_a_slicing_crop_is_recropped_clean(ladder):
    """266: a logo bug cut at the crop edge; a crop that leaves it whole or out exists."""
    plan = _plan(xs=(500,))                                      # x 500..905 slices 880..1000
    sc = _scan(graphics=[((880, 60, 1000, 100), False)], faces=[_face(0.55, 0.4)])
    assert _slices(plan, sc) > 0.4
    rep = ladder(plan, sc, guard={"action": "none"})
    assert rep["action"] == "ladder_a_recrop" and _slices(plan, sc) == 0.0
    assert "fit" not in plan["crop_box"]                         # full-bleed 9:16


def test_a_boxed_speaker_gets_its_own_tile(ladder):
    """331 / 490: the crop holds the speaker's tile border and a sliver of the next tile."""
    plan = _plan(xs=(300,))
    sc = _scan(faces=[_face(0.30, 0.45, h=0.18, w=0.1)])
    tile = [10, 40, 630, 700]                                    # the speaker's window
    rep = ladder(plan, sc, wins=[tile], guard={"action": "none"})
    assert rep["action"].startswith("ladder_b_tile")
    x, cb = plan["segments"][0]["x"], plan["crop_box"]
    assert x >= tile[0] and x + cb["w"] <= tile[2]
    assert cb["y"] >= tile[1] and cb["y"] + cb["h"] <= tile[3]


def test_a_rejected_crop_falls_back_to_a_fit_to_width_box(ladder):
    """the guard could not fix it and no 9:16 window slices nothing: (c) a box fit to width."""
    plan = _plan()
    sc = _scan(graphics=[((700, 400, 1000, 440), False)])
    rep = ladder(plan, sc, rejected="crop would slice a source banner: segment 0: ...")
    assert rep["action"] == "ladder_c_box" and plan["crop_box"]["fit"] == "box"
    assert recompose.track_boxed(plan) and _slices(plan, sc) == 0.0
    assert recompose.blur_fill_frac(plan, None, {}) <= 0.5      # the gate's cap still holds


def test_a_small_slice_is_kept_when_nothing_better_fits(ladder):
    """a slice under LADDER_BOX_SHARE and no clean recrop: no box, no reject, the crop ships."""
    plan = _plan()
    sc = _scan(graphics=[((700, 400, 1000, 440), False)])
    sc["graphics"][0]["frac"] = 0.1
    guard = {"action": "none"}
    rep = ladder(plan, sc, guard=guard)
    assert rep is guard and "crop_box" not in plan and rep["ladder_tried"]


def test_a_slice_the_gate_passes_but_the_eye_check_rejects_is_boxed(ladder):
    """472 / 491: a board graphic sliced for 0.3 of the clip passed the gate's 0.4 cap and failed
    the eye-check; with no clean recrop the ladder boxes it."""
    plan = _plan()
    sc = _scan(graphics=[((700, 400, 1000, 440), False)])
    sc["graphics"][0]["frac"] = 0.3
    rep = ladder(plan, sc, guard={"action": "none"})
    assert rep["action"] == "ladder_c_box" and _slices(plan, sc) == 0.0


def test_nothing_fits_on_a_wide_frame_is_still_a_reject(ladder):
    """16:9 whole frame fit to width = 68 % blurred fill > the gate's 50 %: not rendered."""
    sc = _scan(graphics=[((0, 200, 1280, 230), False)], faces=[_face(0.5, 0.45, h=0.3)])
    with pytest.raises(recompose.RenderError, match="crop would slice a source banner"):
        ladder(_plan(), sc, rejected="crop would slice a source banner: x")


def test_a_square_frame_falls_back_to_the_whole_frame_fit_to_width(ladder):
    """a frame no wider than the box cap: the box is the whole frame (44 % blurred fill)."""
    sc = _scan(graphics=[((0, 200, 720, 230), False)], faces=[_face(0.5, 0.45, h=0.3)],
               w=720, h=720)
    plan = _plan(xs=(158,), w=720, h=720)
    rep = ladder(plan, sc, rejected="crop would slice a source banner: x")
    assert rep["action"] == "ladder_c_box"
    assert (plan["crop_box"]["w"], plan["crop_box"]["h"]) == (720, 720)
    assert recompose.blur_fill_frac(plan, None, {}) == pytest.approx(0.438, abs=0.01)


def test_non_speaker_layouts_keep_the_old_behaviour(ladder):
    plan = {"layout": "gamestack", "src_w": W, "src_h": H}
    with pytest.raises(recompose.RenderError, match="split crop"):
        ladder(plan, _scan(), rejected="split crop slices banner(s)")
    assert ladder(plan, _scan(), guard={"action": "none"}) == {"action": "none"}



# ---------------------------------------------------------------- flat crop bands

def _gray_frames(band_rows=0, n=10, fw=640, fh=360, band_cols=None):
    """Textured gray frames (scan size) whose top `band_rows` rows are flat (in `band_cols`)."""
    import numpy as np
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        g = rng.integers(40, 220, size=(fh, fw)).astype(np.uint8)
        if band_rows:
            c0, c1 = band_cols or (0, fw)
            g[:band_rows, c0:c1] = 30
        out.append((float(i), g))
    return out


def test_crop_bands_reads_a_flat_band_like_the_gate():
    """315 / 385: a flat desk band under the host's webcam, in every frame of the crop."""
    plan = _plan()
    v, bands = recompose.crop_bands(plan, _gray_frames(band_rows=30))      # 60 src px of 720
    assert v["passed"] is False and bands[0]["top"] == pytest.approx(60, abs=2)
    ex = recompose.band_regions(plan, bands, v["min_frac"])
    keep = int(recompose.BAND_KEEP_FRAC * 720)                          # under the gate's 4 %
    assert ex and ex[0][1] == 0 and ex[0][3] == pytest.approx(60 - keep, abs=2)
    v2, _ = recompose.crop_bands(plan, _gray_frames())
    assert v2["passed"] is True


def test_the_ladder_moves_a_full_bleed_crop_off_its_flat_band(ladder, monkeypatch):
    """311: the unlit headliner above the speaker; the recrop starts below it."""
    import cv2
    frames = _gray_frames(band_rows=40)                                     # 80 src px
    monkeypatch.setattr(recompose.reframe, "iter_frames", lambda *a, **k: [
        (t, cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)) for t, g in frames])
    plan, sc = _plan(), _scan(faces=[_face(0.5, 0.5, h=0.25)])
    rep = ladder(plan, sc, guard={"action": "none"})
    assert rep["action"] == "ladder_a_recrop" and "flat crop band" in rep["why"]
    assert plan["crop_box"]["y"] >= 80 - recompose.BAND_KEEP_FRAC * 720
    v, _ = recompose.crop_bands(plan, frames)
    assert v["passed"] is True


def test_a_dark_shirt_is_not_a_band():
    """465 (PASS): the bottom of the crop is the speaker's black hoodie: flat inside the crop, not
    across the scene, so no band and the crop stays."""
    import numpy as np
    rng = np.random.default_rng(1)
    frames = _gray_frames(band_rows=0)
    for _t, g in frames:                         # the hoodie: dark, a little texture (sd ~4,
        g[-60:, 200:440] = rng.integers(13, 28, size=(60, 240))   # the gate's 3.0 misses it)
    plan = _plan(xs=(438,))                      # crop 438..842 src px = 219..421 scan px
    v, _ = recompose.crop_bands(plan, frames)
    assert v["passed"] is True
    for _t, g in frames:                         # the same texture as a desk band across the scene
        g[-60:, :] = rng.integers(13, 28, size=(60, g.shape[1]))
    v, _ = recompose.crop_bands(plan, frames)
    assert v["passed"] is False
    for _t, g in frames:                         # 315: a decoration breaks it on one side only
        g[-60:, 100:200] = rng.integers(40, 220, size=(60, 100))
    v, _ = recompose.crop_bands(plan, frames)
    assert v["passed"] is False


def test_a_box_is_band_checked_on_the_slice_its_blurred_fill_shows():
    """311: the box's top rows hold windows at the sides (not flat) but an unlit headliner in
    the middle; the fill stretches the middle over the canvas top, a flat band there."""
    plan = _plan()
    plan["crop_box"] = {"w": 704, "h": 640, "y": 0, "fit": "box"}
    plan["segments"][0]["x"] = 288                     # box 288..992 src = 144..496 scan px;
    frames = _gray_frames(band_rows=40, band_cols=(220, 420))   # its 9:16 middle 230..410 flat
    v, bands = recompose.crop_bands(plan, frames)
    assert v["passed"] is False and bands[0]["top"] >= 60


def test_a_crop_edge_stays_clear_of_a_text_line_by_the_gate_tolerance(ladder):
    """315: a recrop starting 2 px above a text line's bottom row still shows that row; the
    ladder keeps LADDER_TEXT_PAD px between an edge and any text item."""
    plan = _plan()
    sc = _scan(graphics=[((32, 20, 446, 60), False), ((700, 400, 1000, 440), False)],
               faces=[_face(0.5, 0.45)])
    rep = ladder(plan, sc, rejected="crop would slice a source banner: x")
    cb = plan["crop_box"]
    x = plan["segments"][0]["x"]
    top_text = (32, 20, 446, 60)
    shows = x < top_text[2] and x + cb["w"] > top_text[0] and cb["y"] < top_text[3]
    assert not shows or cb["y"] <= top_text[1]       # out of the crop, or whole in it
    assert _slices(plan, sc) == 0.0


def test_a_graded_band_the_gate_passes_earns_a_recrop_not_a_box(ladder, monkeypatch):
    """306: a dark graded top band (night-time car) the gate's edge_bands passes; no clean
    recrop exists, so the planned crop stays rather than a box showing more of the frame."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(2)
    frames = _gray_frames(band_rows=0)
    for _t, g in frames:
        g[:60, :] = rng.integers(13, 28, size=(60, g.shape[1]))      # graded, sd ~4
    monkeypatch.setattr(recompose.reframe, "iter_frames", lambda *a, **k: [
        (t, cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)) for t, g in frames])
    plan = _plan()
    sc = _scan(graphics=[((700, 400, 1000, 440), False)], faces=[_face(0.5, 0.25, h=0.3)])
    sc["graphics"][0]["frac"] = 0.0                     # no slice: only the band speaks
    guard = {"action": "none"}
    rep = ladder(plan, sc, guard=guard)
    assert "fit" not in (plan.get("crop_box") or {})   # never boxed for a graded band
    assert rep is guard or rep["action"] == "ladder_a_recrop"


# ---------------------------------------------------------------- tolerant step

def _tolerant(seg1_s, monkeypatch):
    """A 10 s clip: the speaker centred for 10 - seg1_s s, then at the right for seg1_s s, where
    every 9:16 crop that frames her slices a 450 px text graphic (OCR-confirmed, persistent)."""
    from clipengine import quality_gate
    t1 = 10.0 - seg1_s
    plan = {"layout": "track", "src_w": W, "src_h": H,
            "segments": [{"t0": 0.0, "t1": t1, "x": 438}, {"t0": t1, "t1": 10.0, "x": 820}]}
    sc = _scan(graphics=[((800, 300, 1250, 330), False)],
               faces=[_face(0.5, 0.35, t=1.0), _face(0.8, 0.35, t=t1 + seg1_s / 2)])
    words = [(810, 302, 1240, 328, "BREAKING NEWS TICKER", fi) for fi in range(3)]
    monkeypatch.setattr(quality_gate, "ocr_source_text", lambda *a, **k: None)
    import copy
    planned = copy.deepcopy(plan)
    a = A()
    tried = []
    rep = recompose._tolerant_full_bleed(plan, planned, a, sc, words, [],
                                         [("a_recrop", cleancrop.clean_crop, None)], [],
                                         ["slices burned-in text"], tried, None)
    return rep, plan, sc, words, tried


def test_the_last_step_takes_a_crop_whose_slice_the_gate_weighs_light(monkeypatch):
    rep, plan, sc, words, tried = _tolerant(1.0, monkeypatch)
    assert rep["action"] == "ladder_a_recrop_tolerant" and rep["crop_share"] <= 0.2
    assert rep["left_out"] == [[800, 300, 1250, 330]] and "fit" not in plan["crop_box"]
    from clipengine import quality_gate
    v = quality_gate.crop_slice_verdict(quality_gate.crop_slices(
        sc, recompose.crop_views(plan, A.dur), A.dur, words=words))
    assert v["passed"] and v["crop_share"] == rep["crop_share"]


def test_the_last_step_refuses_a_slice_on_screen_for_long(monkeypatch):
    rep, plan, _sc, _w, tried = _tolerant(5.0, monkeypatch)
    assert rep is None and "crop_box" not in plan
    assert any("tolerant" in t and "share 0.45" in t for t in tried)
