"""Gamestack v2 / v3 (/ 11210, render/gamelayout.py): the layout maths (band share,
cam card, canvas rows), the action tracker (motion samples, overlay rect) and the path smoothing
(EMA + dead zone + max pan speed, knots, the crop x expression), plus the recompose wiring.
The v3 canvas numbers the brief sets are in test_gamestack_v3.py."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from clipengine.render import gamelayout as gl
from clipengine.render import recompose
from clipengine import config

CANVAS = gl.OUT_W * gl.OUT_H


# ---------------------------------------------------------------- layout maths

@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080), (854, 480), (2560, 1080)])
def test_band_shows_at_least_the_floor_share_of_the_width_at_full_height(size):
    sw, sh = size
    g = gl.band_geometry(sw, sh)
    assert g["gw"] / sw >= gl.MIN_SHARE and g["gw"] % 2 == 0
    assert g["ch"] == sh                                  # 16:9 and wider: no height crop
    assert abs(g["gh"] - 1080 * sh / g["gw"]) <= 2        # fit to the canvas width, no stretch
    assert g["gh"] % 2 == 0


def test_band_share_is_never_below_the_floor():
    assert gl.band_geometry(1280, 720, share=0.45)["gw"] / 1280 >= gl.MIN_SHARE
    assert gl.band_geometry(1280, 720, share=0.68)["gh"] == 894      # the v2 default band
    assert gl.band_geometry(1280, 720)["gh"] == 1214                 # the v3 default band (0.50)
    g = gl.band_geometry(1920, 1080, share=0.42, min_share=gl.GAME_ONLY_MIN_SHARE,
                         max_h=gl.GAME_ONLY_BAND_MAX_H)
    assert g["gh"] == 1446 and g["share"] < gl.MIN_SHARE              # game only's own floor


def test_narrow_source_band_is_capped_and_cropped_in_height():
    g = gl.band_geometry(960, 720)                        # 4:3
    assert g["gh"] == gl.BAND_MAX_H and g["ch"] < 720
    assert abs(g["gw"] / g["ch"] - 1080 / g["gh"]) < 0.01


@pytest.mark.parametrize("room", [120, 300, 416, 466, 596, 900])
@pytest.mark.parametrize("area", [0.10, 0.12, 0.14, 0.18, 0.30])
def test_cam_card_never_exceeds_its_area_cap(room, area):
    w, h = gl.card_size(room, max_area=area)
    assert w * h <= min(area, gl.INSET_HARD_CAP) * CANVAS
    assert w * h <= 0.14 * CANVAS                         # hard cap 14 % of the frame
    assert h <= min(room, gl.INSET_MAX_H) and w <= gl.INSET_MAX_W
    assert gl.INSET_AR_MIN - 0.02 <= w / h <= gl.INSET_AR_MAX + 0.02


def test_cam_card_crop_stays_in_the_box_and_centres_the_face():
    box = [0, 0, 288, 246]
    face = (165.0, 132.0, 137.0)
    w, h = gl.card_size(466)
    cw, ch, cx, cy = gl.inset_crop(box, face, w, h)
    assert box[0] <= cx and cx + cw <= box[2] and box[1] <= cy and cy + ch <= box[3]
    assert abs(cw / ch - w / h) < 0.03
    # a wide box: the crop is narrower than the box and its centre sits on the face
    cw, ch, cx, cy = gl.inset_crop([0, 0, 600, 200], (300.0, 100.0, 60.0), 400, 300)
    assert abs(cx + cw / 2 - 300) <= 2


def test_cam_card_crop_is_at_most_a_few_face_heights_tall():
    cw, ch, cx, cy = gl.inset_crop([0, 0, 800, 800], (400.0, 300.0, 50.0), 480, 480)
    assert ch <= gl.INSET_MAX_FACE_H * 50 + 2
    assert cy <= 300 - 0.3 * ch and 300 <= cy + ch     # the face inside, in the upper half


def test_canvas_layout_puts_captions_on_the_band_and_the_card_under_it():
    gh = 1214
    w, h = gl.card_size(gl.card_room(gh))
    lay = gl.canvas_layout(gh, (w, h), "left")
    assert lay["header"] == [0, gl.HEADER_H] and lay["band"] == [gl.HEADER_H, gl.HEADER_H + gh]
    x, y, cw, chh = lay["card"]
    assert (x, y + chh) == (gl.INSET_MARGIN, gl.OUT_H - gl.INSET_MARGIN)
    assert y == lay["band"][1]                            # right under the band, never over it
    top, bot = lay["cap_y"] - gl.CAP_HALF, lay["cap_y"] + gl.CAP_HALF
    assert lay["band"][0] < top and bot <= lay["band"][1]  # on the band's lower rows
    right = gl.canvas_layout(gh, (w, h), "right")["card"]
    assert right[0] + right[2] == gl.OUT_W - gl.INSET_MARGIN


def test_fill_share_stays_under_the_gate_cap():
    for sw, sh in ((1280, 720), (1920, 1080)):
        for share in (gl.MIN_SHARE, gl.GAME_SHARE, 0.56, 0.68):
            g = gl.band_geometry(sw, sh, share)
            lay = gl.canvas_layout(g["gh"], gl.card_size(gl.card_room(g["gh"])))
            fill = gl.fill_frac(g["gh"], lay["card"], gl.HEADER_H)
            assert fill < 0.5                                  # quality_gate VQ_MAX_BLUR_FILL
    # at the default share a band alone (no card) leaves 37 % blur: the card sits under it
    assert gl.fill_frac(gl.band_geometry(1280, 720)["gh"], None, gl.HEADER_H) > 0.25


def test_cam_free_ranges_keep_the_cam_columns_out():
    gw = 870
    assert gl.cam_free_ranges(1280, gw, [0, 0, 300, 250]) == [(300, 410)]
    assert gl.cam_free_ranges(1280, gw, [990, 0, 1280, 250]) == [(0, 120)]
    assert gl.cam_free_ranges(1280, gw, [500, 0, 780, 250]) == []      # a cam mid-frame
    for lo, hi in gl.cam_free_ranges(1280, 600, [600, 0, 680, 250]):
        assert lo % 2 == 0 and hi % 2 == 0
        for x in (lo, hi):
            assert x + 600 <= 600 or x >= 680


def test_inset_needs_a_frame_edge():
    assert gl.is_inset([0, 0, 288, 246], 1280, 720)
    assert gl.is_inset([1000, 500, 1270, 715], 1280, 720)
    assert not gl.is_inset([322, 122, 564, 364], 1280, 720)   # 3032: a face in an IRL scene


def test_legal_runs_are_maximal():
    bad = set(range(20, 31))
    assert gl.legal_runs(0, 50, lambda x: x not in bad) == [(0, 18), (32, 50)]
    assert gl.legal_runs(0, 10, lambda x: False) == []


# ---------------------------------------------------------------- path smoothing

DT = 1.0 / gl.TRACK_FPS


def test_dead_zone_holds_a_jittering_target():
    targets = [300 + (8 if i % 2 else -8) for i in range(50)]
    xs = gl.smooth_path(targets, 0, 600, dt=DT, dead_zone=30, max_speed=200)
    assert max(xs) - min(xs) < 1.0


def test_max_pan_speed_is_never_exceeded():
    targets = [0] * 20 + [600] * 40 + [0] * 20
    vmax = 150.0
    xs = gl.smooth_path(targets, 0, 600, dt=DT, alpha=0.9, dead_zone=0, max_speed=vmax)
    steps = [abs(b - a) for a, b in zip(xs, xs[1:])]
    assert max(steps) <= vmax * DT + 1e-6


def test_forward_backward_average_has_no_lag():
    targets = [100.0] * 40 + [500.0] * 40
    xs = gl.smooth_path(targets, 0, 600, dt=DT, alpha=0.35, dead_zone=0, max_speed=1e9)
    mid = next(i for i, x in enumerate(xs) if x >= 300)
    assert abs(mid - 40) <= 1                 # halfway exactly at the step, not after it


def test_path_clamps_to_the_legal_range_and_holds_without_action():
    xs = gl.smooth_path([None, 900, None, None, -50, None], 100, 400, dt=DT)
    assert all(100 <= x <= 400 for x in xs)
    assert gl.smooth_path([None] * 5, 100, 300) == [200.0] * 5


def test_knots_follow_the_path_within_tolerance_and_stay_few():
    ts = [i * DT for i in range(300)]
    xs = [300 + 120 * np.sin(t / 3.0) for t in ts]
    ks = gl.knots(ts, xs, tol=8, max_knots=40)
    assert 2 <= len(ks) <= 40
    assert max(abs(gl.x_at(ks, t) - x) for t, x in zip(ts, xs)) <= 10
    assert gl.knots(ts, [250] * 300, tol=8) == [[0.0, 250]]


def _eval_expr(expr, t):
    """The pan expression evaluated in Python (if / lt as ffmpeg defines them)."""
    py = expr.replace("if(", "_if(").replace("lt(", "_lt(")
    return eval(py, {"_if": lambda c, a, b: a if c else b, "_lt": lambda a, b: a < b, "t": t})


def test_pan_expression_matches_the_knot_path():
    ks = [[0.0, 300], [2.0, 300], [4.0, 400], [6.5, 350], [9.0, 350]]
    expr = gl.pan_expr(ks)
    assert expr.count("if(") <= 2 * len(ks)
    for t in np.arange(0, 11, 0.13):
        assert abs(_eval_expr(expr, t) - gl.x_at(ks, t)) < 1e-6
    assert gl.pan_expr([[0.0, 410]]) == "410"


def test_plan_path_puts_the_action_mid_band():
    samples = [{"t": (i + 1) * DT, "cx": 900.0, "cy": 360.0, "e": 1.0} for i in range(50)]
    ks, _xs = gl.plan_path(samples, 600, 0, 680, 1280, 10.0)
    assert abs(gl.x_at(ks, 5.0) - 600) <= 4          # 900 - 600 / 2
    assert gl.inside_share(samples, ks, 600) == 1.0


def test_plan_path_holds_one_x_when_the_range_is_too_narrow():
    samples = [{"t": (i + 1) * DT, "cx": 400.0 + 300 * (i % 2), "cy": 360.0, "e": 1.0}
               for i in range(40)]
    ks, xs = gl.plan_path(samples, 870, 400, 406, 1280, 8.0)
    assert len(ks) == 1 and 400 <= ks[0][1] <= 406 and len(set(xs)) == 1


def test_plan_path_pins_the_peak():
    samples = [{"t": (i + 1) * DT, "cx": 200.0, "cy": 360.0, "e": 1.0} for i in range(60)]
    samples[30] = {"t": samples[30]["t"], "cx": 1150.0, "cy": 360.0, "e": 9.0}   # a one-sample peak
    peak = gl.peak_sample(samples)
    ks, _ = gl.plan_path(samples, 600, 0, 680, 1280, 12.0, pin=(peak["t"], peak["cx"]))
    x = gl.x_at(ks, peak["t"])
    assert x <= peak["cx"] < x + 600


# ---------------------------------------------------------------- action

def _frames(n, w=320, h=180, box=None, distractor=None, cut_at=None):
    """Gray frames: a bright 16 px square moving right 6 px a frame from x=40 (optionally a
    flickering distractor patch at `distractor`, a full-frame change at `cut_at`)."""
    rng = np.random.default_rng(0)
    base = rng.integers(40, 60, size=(h, w)).astype(np.uint8)
    out = []
    for i in range(n):
        f = base.copy()
        x = 40 + 6 * i
        f[80:96, x:x + 16] = 230
        if distractor is not None:
            x0, y0, x1, y1 = distractor
            f[y0:y1, x0:x1] = 255 if i % 2 else 0
        if cut_at is not None and i >= cut_at:
            f = 255 - f
        out.append((round(i * DT, 3), f))
    return out


def test_motion_centroid_tracks_the_moving_subject():
    s = gl.motion_samples(_frames(20), 1280, 720)
    assert len(s) == 19 and all(x["cx"] is not None for x in s)
    for i, x in enumerate(s, start=1):
        assert abs(x["cx"] / 4 - (40 + 6 * i + 5)) <= 8      # scan px of the square's middle
        assert abs(x["cy"] / 4 - 88) <= 6


def test_motion_mask_leaves_the_cam_and_hud_out():
    frames = _frames(20, distractor=(260, 10, 310, 60))      # a flashing HUD / cam corner
    raw = gl.motion_samples(frames, 1280, 720)
    masked = gl.motion_samples(frames, 1280, 720, exclude=[(1040, 40, 1240, 240)])
    off = sum(abs(a["cx"] / 4 - (40 + 6 * i + 5)) for i, a in enumerate(raw, 1)) / len(raw)
    on = sum(abs(a["cx"] / 4 - (40 + 6 * i + 5)) for i, a in enumerate(masked, 1)) / len(masked)
    assert on < 8 < off


def test_scene_cut_and_still_frames_hold():
    s = gl.motion_samples(_frames(12, cut_at=6), 1280, 720)
    cut = [x for x in s if x["cut"]]
    assert len(cut) == 1 and cut[0]["cx"] is None
    still = [(i * DT, np.full((180, 320), 50, np.uint8)) for i in range(5)]
    assert all(x["cx"] is None for x in gl.motion_samples(still, 1280, 720))


def _cam_frames(n=8):
    """BGR 640x360 frames: a textured, changing game with a warm static cam overlay at the
    top-left (x 0..150, y 0..120) holding a 'face' blob."""
    import cv2
    rng = np.random.default_rng(1)
    out = []
    for _i in range(n):
        small = rng.integers(0, 90, size=(18, 32, 3)).astype(np.uint8)       # blocky game noise
        f = cv2.resize(small, (640, 360), interpolation=cv2.INTER_LINEAR)
        f[:, :, 0] = np.clip(f[:, :, 0].astype(int) + 80, 0, 255)            # blue game
        f[0:120, 0:150] = (40, 90, 160)                                       # warm cam room
        f[30:80, 50:95] = (120, 160, 220)                                     # the face
        out.append(f)
    return out


def test_overlay_rect_finds_the_cam_borders():
    r = gl.overlay_rect(_cam_frames(), (72.0, 55.0, 50.0))
    assert r is not None
    x0, y0, x1, y1 = r
    assert x0 == 0 and y0 == 0
    assert 140 <= x1 <= 150 and 110 <= y1 <= 120


def test_snap_to_rect_moves_only_the_sides_toward_the_frame_centre():
    box = [0, 0, 370, 406]                   # 2500's head-and-shoulders box, past the cam
    rect = [12, 20, 288, 246]                # a rect with interior 'borders' on the edge sides
    assert gl.snap_to_rect(box, rect, (165, 132, 137), 1280, 720) == [0, 0, 288, 246]
    assert gl.snap_to_rect([900, 400, 1280, 720], [990, 470, 1270, 715], (1100, 600, 80),
                           1280, 720) == [990, 470, 1280, 720]


# ---------------------------------------------------------------- recompose wiring

def _v2_plan(layout="gamestack"):
    g = gl.band_geometry(1280, 720)
    w, h = gl.card_size(gl.card_room(g["gh"]))
    lay = gl.canvas_layout(g["gh"], (w, h) if layout == "gamestack" else None, "left")
    plan = {"layout": layout, "v2": True, "src_w": 1280, "src_h": 720, "top_h": g["gh"],
            "game": [g["gw"], 320, False],
            "band": {"w": g["gw"], "h": g["ch"], "out_h": g["gh"], "share": g["share"], "y": 0,
                     "lo": 300, "hi": 410},
            "pan": [[0.0, 300], [2.0, 300], [3.0, 410], [8.0, 410]], "pan_y": None,
            "band_y": gl.HEADER_H, "header": lay["header"],
            "card": lay["card"], "cap_y": lay["cap_y"], "fg_h": lay["cap_y"] - 50,
            "action": {"inside_share": 1.0, "peak": None}}
    if layout == "gamestack":
        plan["cam"] = list(gl.inset_crop([0, 0, 288, 246], (165, 132, 137), w, h)) + [False]
    return plan


def test_v2_graph_and_views():
    plan = _v2_plan()
    vf = recompose.gamestack_vf(plan)
    assert vf.endswith("[v0]") and "vstack" not in vf
    assert f"crop={plan['band']['w']}:720:'{gl.pan_expr(plan['pan'])}':'0'" in vf
    kx, ky, kw, kh = plan["card"]
    assert f"scale={kw}:{kh}" in vf and f"overlay={kx}:{ky}" in vf
    assert f"[fill][band]overlay=0:{gl.HEADER_H}" in vf
    views = recompose.crop_views(plan, 8.0)
    band = [v for v in views if v["w"] == plan["band"]["w"]]
    assert band[0]["x"] == 300 and band[-1]["x"] == 410
    assert min(v["t0"] for v in band) == 0.0 and max(v["t1"] for v in band) == 8.0
    cw, ch, cx, cy, _ = plan["cam"]
    assert {"x": cx, "y": cy, "w": cw, "h": ch} in [
        {k: v[k] for k in ("x", "y", "w", "h")} for v in views]
    assert recompose.hook_avoid(plan) == [(gl.HEADER_H, 1920)]     # the hook on the header
    fill = recompose.blur_fill_frac(plan, None, {})
    assert fill == gl.fill_frac(plan["band"]["out_h"], plan["card"], gl.HEADER_H) and fill <= 0.25
    v2_shaped = {k: v for k, v in plan.items() if k not in ("band_y", "header")}
    assert recompose.hook_avoid(v2_shaped) == [(0, plan["band"]["out_h"]), (plan["card"][1], 1920)]


def test_v1_plans_keep_their_graph():
    """v2 off (or a streamer split / v1 plan): the v1 stack graph, unchanged."""
    plan = {"layout": "gamestack", "src_w": 1920, "src_h": 1080, "top_h": 1220,
            "game": [956, 900, False], "cam": [640, 414, 200, 300, False]}
    vf = recompose.gamestack_vf(plan)
    assert "vstack=inputs=2" in vf and "crop=956:ih:900:0" in vf


@pytest.mark.parametrize("layout", ["gamestack", "game_only"])
def test_v2_graph_renders_a_1080x1920_canvas(layout, tmp_path):
    """The graph runs in ffmpeg (expression crop, quarter-scale fill, card overlay)."""
    src = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=1280x720:rate=30", "-t", "4", "-pix_fmt", "yuv420p",
                    str(src)], check=True)
    out = tmp_path / "o.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-filter_complex",
                    recompose.gamestack_v2_vf(_v2_plan(layout)), "-map", "[v0]", "-t", "4",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height", "-of", "csv=p=0", str(out)],
                       capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "1080,1920"


def test_v2_views_cut_ramps_into_short_pieces():
    plan = _v2_plan()
    views = recompose.v2_views(plan, 8.0)
    assert views[0][:2] == (0.0, 2.0) and views[-1][1] == 8.0
    ramp = [v for v in views if 2.0 <= v[0] and v[1] <= 3.0]
    assert len(ramp) == 2 and ramp[0][2] < ramp[1][2]


def test_config_turns_v2_on_with_a_share_and_card_cap_inside_the_brief():
    import json
    gs = config.load()["render"]["gamestack"]
    assert gs["v2"] is True
    assert gs["game_share"] >= gl.MIN_SHARE and gs["cam_max_area"] <= gl.INSET_HARD_CAP


def test_v2_full_render_on_a_corner_cam(tmp_path, monkeypatch, capsys):
    """End to end (recompose.main): a 1280x720 game with a cam inset in its top-left corner renders
    the v3 canvas: the band >= 48 % of the width under the 200 px header, the cam card <= 14 % of
    the canvas, blur <= 25 %. The cam box detection is mocked (as in test_facecam's v1 render
    test)."""
    import json
    from clipengine.render import facecam
    src = tmp_path / "g.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=1280x720:rate=30", "-f", "lavfi", "-i", "sine=frequency=440",
                    "-t", "3", "-filter_complex",
                    "[0:v]drawbox=x=0:y=0:w=260:h=200:color=0x8040a0:t=fill[v]",
                    "-map", "[v]", "-map", "1:a", "-pix_fmt", "yuv420p", "-shortest", str(src)],
                   check=True)
    box = [0.0, 0.0, 260 / 1280, 200 / 720]
    monkeypatch.setattr(facecam, "get_or_detect",
                        lambda channel, s, start, dur, source_duration=None, cache_dir=None,
                        log=None: (box, box, True, 1.0))
    monkeypatch.setattr(recompose, "_gamestack_cfg",
                        lambda: {"v2": True, "game_share": 0.50, "cam_max_area": 0.12})
    out = tmp_path / "o.mp4"
    plan_json = tmp_path / "p.json"
    rc = recompose.main([str(src), "0", "3", str(out), "--layout", "gamestack", "--channel", "s1",
                         "--emit-plan", str(plan_json)])
    assert rc == 0
    rep = json.loads(capsys.readouterr().out.strip().splitlines()[-1][3:])
    assert rep["layout"] == "gamestack" and rep["passed"]
    v2 = rep["gamestack_v2"]
    assert v2["band_share"] >= 0.48 and v2["card_share"] <= 0.14 and rep["blur_fill_frac"] <= 0.25
    assert v2["band_y"] == 200 and v2["band_canvas"] >= 0.60
    plan = json.loads(plan_json.read_text())
    assert plan["v2"] and plan["band"]["lo"] >= 260          # the band leaves the cam's columns out
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height", "-of", "csv=p=0", str(out)],
                       capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "1080,1920"
