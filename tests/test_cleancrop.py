"""the clean speaker crop, the caption band moved off
on-screen text, the slur mute and the forced levels. Offline: synthetic scans and frames."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from clipengine.render import captions
from clipengine.render import cleancrop
from clipengine.render import recompose
from clipengine.render import textzones

W, H = 1280, 720


def _face(cx, cy, h=0.2, w=0.12, t=1.0):
    return (t, [{"cx": cx, "cy": cy, "w": w, "h": h, "score": 0.9}])


def _plan(xs=(438,)):
    n = len(xs)
    return {"layout": "track", "src_w": W, "src_h": H,
            "segments": [{"t0": i * 5.0, "t1": (i + 1) * 5.0, "x": x} for i, x in enumerate(xs)]}


def _scan(graphics=(), panels=(), dead=None, faces=None):
    return {"src_w": W, "src_h": H, "frames": 10,
            "graphics": [{"box": list(b), "frac": 0.9, "banner": ban} for b, ban in graphics],
            "panels": [list(p) for p in panels], "dead_bands": dead or {},
            "faces": faces if faces is not None else [_face(0.5, 0.35)]}


def test_min_crop_height_is_two_thirds_of_the_source():
    assert cleancrop.min_crop_h(720) == 480 and cleancrop.min_crop_h(640) == 427
    assert cleancrop.canvas_width(720, 720) == 1080.0 and cleancrop.canvas_width(480, 720) == 720.0


def test_full_height_crop_when_nothing_is_in_the_way():
    sc = _scan()
    crop = cleancrop.clean_crop(_plan(), sc, *cleancrop.regions_from_scan(sc))
    assert crop["h"] == H and crop["canvas_w"] == 1080.0
    x = crop["x"][0]
    assert x <= 0.44 * W and x + crop["w"] >= 0.56 * W            # the face is inside


def test_a_lower_third_is_cut_out_of_the_rows_when_strict():
    sc = _scan(graphics=[((200, 620, 1100, 660), True)])          # banner across the bottom
    ex, ns = cleancrop.regions_from_scan(sc, strict=True)
    crop = cleancrop.clean_crop(_plan(), sc, ex, ns)
    assert crop["y"] + crop["h"] <= 620 - 4                      # the banner plate is outside
    assert crop["canvas_w"] >= 720
    soft_ex, soft_ns = cleancrop.regions_from_scan(sc, strict=False)
    assert not soft_ex and soft_ns                                # soft: only not sliced


def test_no_clean_crop_when_the_ui_leaves_too_few_rows():
    # tickers above and below the face leave 300 rows: < 480 (720 px of canvas) -> not fixable
    sc = _scan(graphics=[((0, 150, 1280, 190), True), ((0, 480, 1280, 520), True)],
               faces=[_face(0.5, 0.45, h=0.15)])
    with pytest.raises(cleancrop.CleanCropError, match="720 px canvas width"):
        cleancrop.clean_crop(_plan(), sc, *cleancrop.regions_from_scan(sc, strict=True))


def test_a_graphic_beside_the_speaker_is_whole_or_out_never_sliced():
    sc = _scan(graphics=[((800, 300, 1100, 330), False)])        # a text line right of the face
    ex, ns = cleancrop.regions_from_scan(sc, strict=True)
    crop = cleancrop.clean_crop(_plan(), sc, ex, ns)
    x, w = crop["x"][0], crop["w"]
    inside = x <= 800 and x + w >= 1100
    outside = x + w <= 800 + cleancrop.SLICE_TOL or x >= 1100 - cleancrop.SLICE_TOL
    assert inside or outside
    assert crop["h"] == H                     # leaving it out costs no rows
    sc = _scan(graphics=[((700, 300, 1000, 330), False)])        # touches the face's columns:
    with pytest.raises(cleancrop.CleanCropError):                # neither whole nor out fits
        cleancrop.clean_crop(_plan(), sc, *cleancrop.regions_from_scan(sc, strict=True))


def test_the_crop_stays_inside_the_speakers_boxed_window():
    sc = _scan(faces=[_face(0.3, 0.4, h=0.18)])
    win = [[60, 20, 700, 600]]
    crop = cleancrop.clean_crop(_plan((180,)), sc, *cleancrop.regions_from_scan(sc), windows=win)
    assert crop["y"] >= 20 and crop["y"] + crop["h"] <= 600
    assert crop["x"][0] >= 60 and crop["x"][0] + crop["w"] <= 700
    plan = cleancrop.apply(_plan((180,)), crop)
    assert plan["crop_box"] == {"w": crop["w"], "h": crop["h"], "y": crop["y"]}
    assert recompose.track_crop_window(plan) == (crop["w"], crop["h"], crop["y"])
    assert not recompose.track_boxed(plan)                        # full bleed, not boxed


def test_dead_bands_are_excluded():
    sc = _scan(dead={"bottom": 72})
    ex, _ = cleancrop.regions_from_scan(sc, strict=False)
    assert [0, H - 72, W, H] in ex
    crop = cleancrop.clean_crop(_plan(), sc, *cleancrop.regions_from_scan(sc))
    assert crop["y"] + crop["h"] <= H - 72


def _persistence_with_box(texture=True):
    """A 360x640 persistence map: a box border (thin lines) around a face, and a static patterned
    shirt (a thick band of edges) under the face, like 10-20 clip 135."""
    P = np.zeros((360, 640), np.float32)
    P[10, 10:320] = P[295, 10:320] = 1.0                          # top / bottom border
    P[10:296, 10] = P[10:296, 319] = 1.0                          # left / right border
    if texture:
        P[205:225, 12:318] = 0.95                                 # the shirt
    return P


def test_find_window_takes_the_thin_border_not_the_patterned_shirt():
    box, real = cleancrop.find_window(_persistence_with_box(), (120, 40, 200, 160))
    assert all(abs(a - b) <= 1 for a, b in zip(box, [10, 10, 319, 295]))   # not the shirt (205)
    assert set(real) == {"left", "right", "top", "bottom"}


def test_find_window_needs_two_closed_borders():
    P = np.zeros((360, 640), np.float32)
    P[30:330, 400] = 1.0                                          # a door frame beside the face
    assert cleancrop.find_window(P, (120, 40, 200, 160)) is None


def test_clean_crop_plan_refuses_a_plan_without_a_speaker_track(tmp_path):
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--clean-crop", "strict"])
    with pytest.raises(recompose.RenderError, match="clean crop: the planner gave 'general'"):
        recompose.clean_crop_plan({"layout": "general"}, a, Path("s.mp4"))


def test_clean_crop_plan_turns_no_clean_crop_into_a_render_error(monkeypatch):
    sc = _scan(graphics=[((0, 150, 1280, 190), True), ((0, 480, 1280, 520), True)],
               faces=[_face(0.5, 0.45, h=0.15)])
    monkeypatch.setattr(recompose.banners, "scan", lambda *a, **k: sc)
    a = recompose.parse_args(["s.mp4", "0", "10", "o.mp4", "--clean-crop", "strict"])
    with pytest.raises(recompose.RenderError, match="^clean crop: no clean 9:16 crop"):
        recompose.clean_crop_plan(_plan(), a, Path("s.mp4"))
    monkeypatch.setattr(recompose.banners, "scan", lambda *a, **k: _scan())
    plan = _plan()
    rep = recompose.clean_crop_plan(plan, a, Path("s.mp4"))
    assert rep["action"] == "clean_crop" and plan["crop_box"]["h"] == H


def test_free_caption_y_moves_off_a_text_band():
    assert captions.free_caption_y([], 1270) == 1270
    y = captions.free_caption_y([(1180, 1360)], 1270)              # a lower third under captions
    assert captions.band_overlap(y, captions.CAPTION_BLOCK_H, [(1180, 1360)],
                                 captions.CAPTION_TEXT_PAD) == 0
    assert captions.CAPTION_Y_MIN <= y <= captions.CAPTION_Y_MAX
    full = [(0, 1920)]
    assert captions.free_caption_y(full, 1270) == 1270             # nowhere free: unchanged
    ev = captions.caption_events([{"w": "hi", "start": 0.0, "end": 0.5}], cap_y=y)
    assert f"\\pos(540,{y})" in ev[0]


def test_clip_text_bands_need_text_in_a_fifth_of_the_clip(monkeypatch):
    monkeypatch.setattr(textzones, "ocr_words", lambda g: [])
    busy = np.zeros((1920, 1080), np.uint8)
    busy[1300:1340, 100:1000:6] = 255                              # an edge-dense ticker row
    calm = np.zeros((1920, 1080), np.uint8)
    got = textzones.clip_text_bands("s", 0, 10, "[0:v]null[v0]", frames=[busy] * 3 + [calm] * 7)
    assert got["frames"] == 10 and got["bands"] and 1290 <= got["bands"][0][0] <= 1310
    got = textzones.clip_text_bands("s", 0, 10, "[0:v]null[v0]", frames=[busy] + [calm] * 9)
    assert got["bands"] == []


def test_mute_spans_and_the_audio_chain_bleep():
    spans = recompose.mute_spans("1.200-1.650,7.000-7.400")
    assert spans == [(1.2, 1.65), (7.0, 7.4)] and recompose.mute_spans(None) == []
    chain = recompose.audio_chain(10.0, mute=spans)
    assert chain.startswith("volume=0:enable='between(t,1.200,1.650)+between(t,7.000,7.400)'")
    assert "volume=0" not in recompose.audio_chain(10.0)


def test_levels_flag_measures_any_vertical(monkeypatch):
    monkeypatch.setattr(recompose.reframe, "iter_frames", lambda *a, **k: iter(
        [(0.0, np.full((90, 160, 3), 20, np.uint8)), (1.0, np.full((90, 160, 3), 70, np.uint8))]))
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--vertical", "podcast"])
    assert recompose.measure_levels(a, "s.mp4") is None
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--vertical", "podcast", "--levels"])
    assert recompose.measure_levels(a, "s.mp4") is not None


def test_ocr_text_the_gate_reads_is_kept_whole_or_out(monkeypatch):
    """10-20 clip 135: a headline on a box border the line detector missed; the gate's OCR read
    it and failed the crop for slicing it. The clean crop now sees the same text items."""
    ex, ns = cleancrop.regions_from_text([{"box": [198, 590, 1083, 631]},
                                          {"box": [700, 300, 760, 330]}], W, H, strict=True)
    assert ex == [[198, 590, 1083, 631]] and ns == [[700, 300, 760, 330]]
    assert cleancrop.regions_from_text([{"box": [198, 590, 1083, 631]}], W, H, strict=False) == \
        ([], [[198, 590, 1083, 631]])
    monkeypatch.setattr(recompose.banners, "scan", lambda *a, **k: _scan())
    monkeypatch.setattr(recompose, "_gate_text_items",
                        lambda src, a, sc: [{"box": [198, 590, 1083, 631], "src": "ocr"}])
    a = recompose.parse_args(["s.mp4", "0", "10", "o.mp4", "--clean-crop", "strict"])
    plan = _plan()
    recompose.clean_crop_plan(plan, a, Path("s.mp4"))
    assert plan["crop_box"]["y"] + plan["crop_box"]["h"] <= 590 - cleancrop.MARGIN
