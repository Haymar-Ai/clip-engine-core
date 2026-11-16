"""Gamestack v3 (render/gamelayout.py): the game fills the screen. The canvas numbers
the brief sets (band >= 60 % of the canvas at a 0.50 share, cam card <= 14 %, blurred fill <= 25 %,
the bottom blur <= 22 %, a <= 200 px hook header, captions on the band's lower rows, game only at
0.42 with no card), the hook on the header for the whole clip, and a real ffmpeg render of the
graph: the band under the header, no flat edge band for the gate."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from clipengine.render import captions
from clipengine.render import gamelayout as gl
from clipengine.render import recompose
from clipengine import config

CANVAS = gl.OUT_W * gl.OUT_H
SIZES = [(1280, 720), (1920, 1080), (2560, 1440)]


def _canvas(sw, sh, share=gl.GAME_SHARE, area=gl.INSET_MAX_AREA, side="left"):
    g = gl.band_geometry(sw, sh, share)
    card = gl.card_size(gl.card_room(g["gh"]), max_area=area)
    lay = gl.canvas_layout(g["gh"], card, side)
    return g, lay, gl.canvas_shares(g["gh"], lay["card"], gl.HEADER_H)


@pytest.mark.parametrize("size", SIZES)
def test_band_is_at_least_60_percent_of_the_canvas_at_share_050(size):
    g, lay, sh = _canvas(*size, share=0.50)
    assert g["share"] == pytest.approx(0.50, abs=0.002)
    assert g["gh"] >= 1210 and sh["band"] >= 0.60          # v2's 68 % band was 894 px (46 %)
    assert lay["band"] == [200, 200 + g["gh"]]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("share", [0.48, 0.49, 0.50, 0.56, 0.68])
def test_cam_card_is_at_most_12_percent_and_never_above_14(size, share):
    g, lay, _sh = _canvas(*size, share=share)
    x, y, w, h = lay["card"]
    assert w * h <= 0.12 * CANVAS                           # the default cam_max_area
    _g, big, _ = _canvas(*size, share=share, area=0.40)
    assert big["card"][2] * big["card"][3] <= 0.14 * CANVAS  # the hard cap
    assert y >= lay["band"][1]                              # never over the band (the action)
    assert y + h == gl.OUT_H - gl.INSET_MARGIN and x == gl.INSET_MARGIN


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("share", [0.48, 0.49, 0.50])
def test_blur_is_at_most_25_percent_and_the_bottom_blur_22(size, share):
    g, lay, sh = _canvas(*size, share=share)
    assert sh["blur"] <= 0.25
    assert sh["bottom_blur"] <= 0.22
    assert sh["header"] <= 200 / 1920 + 1e-3
    assert lay["card"][1] == lay["band"][1] or lay["card"][1] - lay["band"][1] <= 2


def test_the_056_alternate_trades_band_for_blur():
    """S=0.56 (the 2620 alternate): a 1084 px band (56 %), a gap above the max-height card."""
    g, lay, sh = _canvas(1920, 1080, share=0.56)
    assert 1080 <= g["gh"] <= 1090 and 0.56 <= sh["band"] <= 0.57
    assert lay["card"][3] == gl.INSET_MAX_H and lay["card"][1] > lay["band"][1]
    assert 0.25 < sh["blur"] < 0.35 and sh["bottom_blur"] <= 0.22


@pytest.mark.parametrize("size", SIZES)
def test_game_only_takes_042_and_the_middle_with_no_card(size):
    sw, sh = size
    g = gl.band_geometry(sw, sh, gl.GAME_ONLY_SHARE, min_share=gl.GAME_ONLY_MIN_SHARE,
                         max_h=gl.GAME_ONLY_BAND_MAX_H)
    lay = gl.canvas_layout(g["gh"], None, band_y=gl.HEADER_H)
    shares = gl.canvas_shares(g["gh"], None, gl.HEADER_H)
    assert g["share"] == pytest.approx(0.42, abs=0.002) and lay["card"] is None
    assert lay["band"][0] == 200 and 1640 <= lay["band"][1] <= 1650
    assert shares["blur"] <= 0.25 and shares["band"] >= 0.75


@pytest.mark.parametrize("share", [0.48, 0.50, 0.56])
def test_captions_ride_on_the_band_lower_rows(share):
    _g, lay, _sh = _canvas(1920, 1080, share=share)
    top, bot = lay["cap_y"] - gl.CAP_HALF, lay["cap_y"] + gl.CAP_HALF
    assert lay["band"][0] < top and bot <= lay["band"][1]
    assert bot >= lay["band"][1] - 2 * gl.CAP_HALF          # its lower rows, not mid-band
    if share == 0.50:
        assert 1240 <= top and bot <= 1410                  # brief: y ~1250-1400


def test_canvas_shares_count_a_card_over_the_band_once():
    sh = gl.canvas_shares(1214, [40, 1300, 500, 580], 200)  # 114 rows of it on the band
    assert sh["card"] == round(500 * 466 / CANVAS, 3)
    assert sh["blur"] == round(1 - (1080 * 1214 + 500 * 466) / CANVAS, 3)


def test_knobs_come_from_config_and_the_header_is_capped():
    gs = config.load()["render"]["gamestack"]
    assert gs["game_share"] == 0.50 and gs["cam_max_area"] == 0.12
    assert gs["game_only_share"] == 0.42 and gs["header_h"] == 200
    k = recompose._v2_cfg(gs)
    assert (k["share"], k["area"], k["game_only_share"], k["header_h"]) == (0.50, 0.12, 0.42, 200)
    k = recompose._v2_cfg({"game_share": 0.56, "cam_max_area": 0.3, "header_h": 320})
    assert k["share"] == 0.56 and k["area"] == gl.INSET_HARD_CAP and k["header_h"] == 200
    assert recompose._v2_cfg({})["share"] == gl.GAME_SHARE == 0.50


@pytest.mark.parametrize("hook", ["He thinks this machine will make millions",
                                  "This game looks exactly like Little Nightmares",
                                  "Secret lab hidden under city sewers"])
def test_hook_sits_on_the_header_for_the_whole_clip(hook, tmp_path):
    out = captions.build_ass([], tmp_path / "c.ass", hook=hook, layout="gamestack", dur=21.5,
                             hook_box=(0, 200))
    line = [l for l in out.read_text().splitlines() if ",Hook," in l][0]
    assert ",0:00:00.00,0:00:21.50,Hook," in line          # on screen for the whole clip
    assert "\\pos(540,100)" in line and "\\fad(120,0)" in line
    _line, size = captions.hook_event(hook, "gamestack", 21.5, box=(0, 200))
    n = line.count("\\N") + 1
    assert n <= captions.HOOK_BOX_LINES and n * size * captions.HOOK_LINE_H <= 200 - 20


def test_hook_without_a_header_is_unchanged():
    line, _size = captions.hook_event("Secret lab hidden under city sewers", "track", 21.5)
    assert ",0:00:00.00,0:00:03.40,Hook," in line and "\\fad(120,400)" in line
    y = int(line.split("\\pos(540,")[1].split(")")[0])
    assert y >= captions.HOOK_TOP


def _v3_plan(layout="gamestack"):
    g = gl.band_geometry(1280, 720) if layout == "gamestack" else gl.band_geometry(
        1280, 720, gl.GAME_ONLY_SHARE, min_share=gl.GAME_ONLY_MIN_SHARE,
        max_h=gl.GAME_ONLY_BAND_MAX_H)
    card = gl.card_size(gl.card_room(g["gh"])) if layout == "gamestack" else None
    lay = gl.canvas_layout(g["gh"], card, "left")
    plan = {"layout": layout, "v2": True, "src_w": 1280, "src_h": 720, "top_h": g["gh"],
            "game": [g["gw"], 320, False],
            "band": {"w": g["gw"], "h": g["ch"], "out_h": g["gh"], "share": g["share"], "y": 0,
                     "lo": 300, "hi": 600},
            "pan": [[0.0, 300], [1.0, 300], [2.0, 600], [4.0, 600]], "pan_y": None,
            "band_y": gl.HEADER_H, "header": lay["header"], "card": lay["card"],
            "cap_y": lay["cap_y"], "fg_h": lay["cap_y"] - 50,
            "action": {"inside_share": 1.0, "peak": None}}
    if card:
        plan["cam"] = list(gl.inset_crop([0, 0, 288, 246], (150, 120, 100), *card)) + [False]
    return plan


def test_game_only_graph_has_no_card():
    vf = recompose.gamestack_v2_vf(_v3_plan("game_only"))
    assert "[card]" not in vf and "overlay=0:200" in vf and vf.endswith("[v0]")
    assert recompose.blur_fill_frac(_v3_plan("game_only"), None, {}) <= 0.25


@pytest.mark.parametrize("layout", ["gamestack", "game_only"])
def test_v3_render_band_under_the_header_and_no_flat_edge_band(layout, tmp_path):
    """A real ffmpeg render of the v3 graph on a busy 1280x720 source: the header rows are the
    band's blur (soft), the band rows the sharp game, the card's rows sharp in its corner; the
    gate's own edge-band check passes (no flat bar on any canvas edge)."""
    import cv2
    from clipengine import quality_gate
    src = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=1280x720:rate=30", "-t", "4", "-pix_fmt", "yuv420p",
                    str(src)], check=True)
    plan = _v3_plan(layout)
    out = tmp_path / "o.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-filter_complex",
                    recompose.gamestack_v2_vf(plan), "-map", "[v0]", "-t", "4",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", "0.5", "-i", str(out), "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True)
    g = np.frombuffer(r.stdout, np.uint8).reshape(1920, 1080).astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    y0, y1 = plan["band_y"], plan["band_y"] + plan["band"]["out_h"]
    header = lap[10:y0 - 10].mean()
    band = lap[y0 + 10:y1 - 10].mean()
    assert band > 3 * header                                  # sharp game under a soft header
    if plan["card"]:
        kx, ky, kw, kh = plan["card"]
        assert lap[ky + 10:ky + kh - 10, kx + 10:kx + kw - 10].mean() > 3 * header
    v = quality_gate.measure_edge_bands(str(out))
    assert v["passed"] is True, v


def _band_with_banner(monkeypatch, frac, widen_max=0.0):
    """_v2_band on a 1280x720 source whose only graphic is an 810 px banner row (2620's wall sign
    shape), on screen in `frac` of the scan frames; the gate's crop-slice verdict passes."""
    import types
    from clipengine import quality_gate
    sc = {"graphics": [{"box": [508, 224, 1246, 318], "banner": True, "frac": frac}],
          "panels": [], "frames": 13, "src_w": 1280, "src_h": 720}
    monkeypatch.setattr(recompose, "_scan", lambda src, a, size: sc)
    monkeypatch.setattr(recompose, "_overlays", lambda src, a, size: [])
    monkeypatch.setattr(recompose, "_ocr", lambda src, a, size: [])
    monkeypatch.setattr(recompose, "keep_plates", lambda *a, **k: [])
    monkeypatch.setattr(quality_gate, "crop_slices", lambda *a, **k: [])
    monkeypatch.setattr(quality_gate, "crop_slice_verdict",
                        lambda s: {"passed": True, "crop_share": 0.0, "sliced_share": 0.0})

    def frames(src, start, dur, fps=3, width=640, src_size=None):
        rng = np.random.default_rng(0)
        for i in range(10):
            f = np.zeros((180, 320, 3), np.uint8)
            f[60:120, 120 + 8 * i:160 + 8 * i] = rng.integers(100, 255, (60, 40, 3))
            yield i / 5.0, f
    monkeypatch.setattr(recompose.reframe, "iter_frames", frames)
    a = types.SimpleNamespace(start="0", dur=2.0)
    return recompose._v2_band("s.mp4", a, 1280, 720, 0.50, lambda gw: [(300, 1280 - gw)],
                              widen_max=widen_max)


def test_scene_text_no_50_percent_band_holds_is_left_to_the_gate(monkeypatch):
    b = _band_with_banner(monkeypatch, 0.231)
    assert not b["band"]["sliced"] and b["band"]["share"] <= 0.50
    assert b["band"]["scene_text"] == [[470, 208, 1280, 334]]


def test_a_persistent_banner_still_rejects_the_band(monkeypatch):
    b = _band_with_banner(monkeypatch, 0.9)               # an overlay ticker: never cut
    assert b["band"]["sliced"] and b["band"]["scene_text"] == []


def test_a_band_no_share_holds_widens_to_the_narrowest_clean_share(monkeypatch):
    """2670: a persistent 700 px row no 50 % band passes; a gamestack band widens (up to v2's 0.68)
    to the narrowest share that holds it whole, rather than the clip being rejected."""
    b = _band_with_banner(monkeypatch, 0.9, widen_max=recompose.V2_WIDEN_MAX)
    assert not b["band"]["sliced"] and b["band"]["widened_from"] == 0.50
    assert 0.61 <= b["band"]["share"] <= 0.66      # the 810 px plate, banners.sliced's 10 px slack
    tol = 0.008 * 1280
    assert b["band"]["lo"] <= 470 + tol and b["band"]["lo"] + b["band"]["w"] >= 1280 - tol


def test_captions_move_off_a_subtitle_row_on_the_band():
    """2670 (widened to 59 %: band rows 200..1230, card top 1400): the game's subtitle plate
    [312, 598, 1084, 686] shows on canvas rows ~1055..1182, under the default caption line;
    the captions move to the nearest clear line (under it, onto the band's edge)."""
    band, card = [200, 1230], [40, 1400, 518, 480]
    rows = gl.band_text_rows([[312, 598, 1084, 686]], 200, 1030, 720, x_range=(320, 1076))
    assert rows and 1050 <= rows[0][0] <= 1060 and 1178 <= rows[0][1] <= 1186
    default = gl.canvas_layout(1030, (518, 480), "left")["cap_y"]
    y = gl.caption_y_clear(default, rows, band, card)
    assert y != default and y - gl.CAP_HALF - gl.CAP_TEXT_PAD >= rows[0][1]
    assert y + gl.CAP_HALF <= card[1]                       # never on the card
    # a row outside the band's columns, or none at all: the default line stays
    assert gl.band_text_rows([[0, 598, 300, 686]], 200, 1030, 720, x_range=(320, 1076)) == []
    assert gl.caption_y_clear(default, [], band, card) == default
