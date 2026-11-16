"""HUD niches 10-21 (gaming / poker / streamers): the dual view. The full source frame fit to the
canvas width on top (no crop: no burned-in graphic sliced), a zoom of the same frame below (the
facecam, the speaking player, the action), captions on the seam. The gate reads crop_slices and
edge_bands on the top panel and not on the zoom; nothing changes for any other layout. Offline:
synthetic scans, frames and samples."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from clipengine.render import captions
from clipengine.render import cleancrop
from clipengine import quality_gate as qg
from clipengine.render import recompose

W, H = 1280, 720


class A:
    start, dur = 0.0, 10.0
    niche, vertical = "poker", "poker"    # the HUD rule's niche; the dual view itself is gaming /
                                          # streamers only (G below)


class G(A):
    niche, vertical = "gaming", "stream"


class C(A):
    niche, vertical = None, "stream"      # a campaign / manual render (no --niche): render yield
                                          # 11-09 gives daily gaming / streamers clips the streamer
                                          # split instead (tests/test_render_yield_1109.py)


def _dv(segs=None, zw=790, zh=600):
    hh, ph, seam = recompose.dual_view_geometry(W, H)
    return {"layout": "dual_view", "src_w": W, "src_h": H, "header_h": hh, "panel_h": ph,
            "seam_y": seam, "bottom": "zoom", "zoom": {"w": zw, "h": zh},
            "zoom_segments": segs or [{"t0": 0.0, "t1": 10.0, "x": 400, "y": 60, "cut": True}]}


def _scan(graphics=(), faces=None):
    return {"src_w": W, "src_h": H, "frames": 10,
            "graphics": [{"box": list(b), "frac": 0.9, "banner": ban} for b, ban in graphics],
            "panels": [], "dead_bands": {},
            "faces": faces if faces is not None else [(1.0, [{"cx": 0.5, "cy": 0.35, "w": 0.12,
                                                              "h": 0.2, "score": 0.9}])]}


# ---------------------------------------------------------------- canvas geometry

def test_a_16_9_frame_sits_whole_under_the_hook_header_with_the_zoom_below():
    hh, ph, seam = recompose.dual_view_geometry(1280, 720)
    assert (hh, ph, seam) == (492, 608, 1100)
    assert hh >= captions.HOOK_TOP and recompose.OUT_H - seam == 820
    hh, ph, seam = recompose.dual_view_geometry(1440, 1080)       # 4:3: the header keeps its floor
    assert ph == 810 and hh == recompose.DV_HEADER_MIN and recompose.OUT_H - seam >= 560


def test_a_vertical_or_square_source_gets_no_dual_view():
    assert recompose.dual_view_geometry(1080, 1920) is None
    assert recompose.dual_view_geometry(1080, 1080) is None


def test_the_graph_stacks_header_full_frame_and_zoom_to_1920_rows():
    p = _dv([{"t0": 0.0, "t1": 5.0, "x": 100, "y": 50, "cut": True},
             {"t0": 5.0, "t1": 10.0, "x": 300, "y": 80, "cut": True}])
    g = recompose.dual_view_vf(p)
    assert "[dvt]scale=1080:608:" in g                  # the whole frame, no crop before it
    assert "crop=790:600:'if(lt(t,5),100,300)':'if(lt(t,5),50,80)'" in g
    assert "scale=1080:492:" in g and "scale=1080:820:" in g
    assert 492 + 608 + 820 == recompose.OUT_H and g.endswith("vstack=inputs=3,setsar=1[v0]")
    assert recompose.video_graph(p, A()) == g


def test_the_blurred_fill_is_the_header_alone_under_the_gate_cap():
    f = recompose.blur_fill_frac(_dv(), A(), {})
    assert f == round(492 / 1920, 3) and f <= qg.VQ_MAX_BLUR_FILL


def test_captions_sit_on_the_seam_never_over_the_full_frame():
    y = captions.caption_y("dual_view", 1100)
    assert y - captions.CAPTION_BLOCK_H / 2 >= 1100 and y <= captions.CAPTION_Y_MAX


def test_the_hook_fits_in_the_header_when_the_frame_and_zoom_are_off_limits():
    line, _size = captions.hook_event("He bluffed the whole table with seven deuce off",
                                      "dual_view", 10.0, 1100, avoid=[],
                                      text_avoid=[(492, 1920)])
    y = int(line.split("\\pos(540,")[1].split(")")[0])
    assert captions.HOOK_TOP <= y <= 492


# ---------------------------------------------------------------- what the gate reads

def test_the_zoom_windows_are_tagged_and_only_a_dual_view_drops_them():
    views = recompose.crop_views(_dv(), 10.0)
    assert views and all(v["panel"] == "zoom" for v in views)
    assert qg.slice_views(views, "dual_view") == []
    assert qg.slice_views(views, "track") == views              # no other layout changes
    assert qg.slice_views(None, "dual_view") is None


def test_a_graphic_the_zoom_cuts_is_not_a_slice_in_dual_view_but_is_in_any_other_layout():
    sc = _scan(graphics=[((1100, 600, 1270, 640), False)])     # the POT box, the zoom cuts it
    views = recompose.crop_views(_dv(), 10.0)                   # zoom x 400..1190
    assert qg.crop_slice_verdict(qg.crop_slices(sc, views, 10.0))["crop_share"] > 0.4
    dual = qg.crop_slices(sc, qg.slice_views(views, "dual_view"), 10.0)
    assert qg.crop_slice_verdict(dual)["passed"] and not [s for s in dual if s["how"] == "crop"]


def _canvas(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, (960, 540)).astype(np.uint8)   # the gate's 540-wide scan


PANELS = {"header": [0, 492], "top": [492, 1100], "zoom": [1100, 1920]}


def _bands(frames):
    pd, tops = zip(*[qg.dual_view_edge_depths(g, PANELS) for g in frames])
    k = 960 / 1920
    return qg.dual_view_edge_verdict(list(pd), list(tops), 540,
                                     int(round(1100 * k)) - int(round(492 * k)), 960, PANELS)


def test_a_clean_dual_view_passes_the_edge_bands():
    assert _bands([_canvas(i) for i in range(6)])["passed"] is True


def test_a_letterbox_bar_in_the_full_frame_fails_on_the_top_panel():
    frames = []
    for i in range(6):
        g = _canvas(i)
        g[550 - 20:550, :] = 0                               # the source's bottom 20 scan rows
        frames.append(g)
    v = _bands(frames)
    assert v["passed"] is False and v["errors"][0].startswith("top panel: flat bottom band")


def test_a_flat_header_top_fails_on_the_canvas():
    frames = []
    for i in range(6):
        g = _canvas(i)
        g[:60, :] = 40
        frames.append(g)
    v = _bands(frames)
    assert v["passed"] is False and v["errors"][0].startswith("canvas: flat top band")


def test_the_zoom_panel_edges_are_not_read_in_dual_view():
    frames = []
    for i in range(6):
        g = _canvas(i)
        g[-60:, :] = 0                                       # a dark band at the zoom's bottom
        frames.append(g)
    assert _bands(frames)["passed"] is True
    whole = qg.edge_band_verdict([qg.flat_edge_depths(g) for g in frames], 540, 960)
    assert whole["passed"] is False                          # the canvas-wide rule is unchanged


def test_clip_checks_read_the_top_panel_for_a_dual_view_only(monkeypatch):
    seen = {}

    def saw(key, value):
        seen[key] = value
        return {"passed": True}
    monkeypatch.setattr(qg, "measure_edge_bands", lambda p: saw("eb", "canvas"))
    monkeypatch.setattr(qg, "measure_dual_view_edge_bands", lambda p, panels: saw("eb", panels))
    monkeypatch.setattr(qg, "measure_crop_slices",
                        lambda m, s, e, views, layout=None: saw("views",
                                                                qg.slice_views(views, layout)))
    views = recompose.crop_views(_dv(), 10.0)
    qg.clip_checks("x.mp4", crop_views=views, layout="dual_view", panels=PANELS,
                     asr=({"words": []}, 0.0))
    assert seen == {"eb": PANELS, "views": []}
    seen.clear()
    qg.clip_checks("x.mp4", crop_views=views, layout="track", panels=None,
                     asr=({"words": []}, 0.0))
    assert seen == {"eb": "canvas", "views": views}
    seen.clear()                                             # dual view without its panels:
    qg.clip_checks("x.mp4", crop_views=views, layout="dual_view", panels=None,  # whole canvas
                     asr=({"words": []}, 0.0))
    assert seen["eb"] == "canvas"


# ---------------------------------------------------------------- the zoom panel

def _sample(t, faces=()):
    return {"t": t, "faces": [{"cx": cx, "cy": cy, "w": h * 0.56, "h": h, "score": 0.9}
                              for cx, cy, h in faces]}


def _grays(n, moving=None, shape=(90, 160), seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, shape).astype(np.float32)
    out = []
    for i in range(n):
        g = base.copy()
        if moving:
            y0, y1, x0, x1 = moving
            g[y0:y1, x0:x1] = rng.integers(0, 255, (y1 - y0, x1 - x0))
        out.append(g)
    return out


def test_a_face_shot_zooms_on_the_face_at_the_panel_share():
    samples = [_sample(t / 2, [(0.7, 0.4, 0.15)]) for t in range(20)]
    z = recompose.dual_view_zoom_plan(samples, _grays(20), [], 10.0, W, H, 820)
    zw, zh = z["zoom"]["w"], z["zoom"]["h"]
    assert abs(zh - 0.15 * H / recompose.DV_FACE_FRAC) <= 2 and abs(zw / zh - 1080 / 820) < 0.01
    (s,) = z["zoom_segments"]
    assert s["how"] == "face" and s["x"] <= 0.7 * W <= s["x"] + zw
    assert abs((0.4 * H - s["y"]) / zh - recompose.DV_FACE_Y) < 0.02


def test_the_speaking_player_is_zoomed_not_the_biggest_face():
    """poker: the speaker crop the planner chose follows the left player; a bigger face sits right"""
    samples = [_sample(t / 2, [(0.8, 0.4, 0.3), (0.25, 0.45, 0.15)]) for t in range(20)]
    speaker = [(0.0, 0.1, 0.42)]
    z = recompose.dual_view_zoom_plan(samples, _grays(20), [], 10.0, W, H, 820, speaker=speaker)
    s = z["zoom_segments"][0]
    assert s["x"] <= 0.25 * W <= s["x"] + z["zoom"]["w"] < 0.8 * W


def test_a_shot_without_a_face_zooms_on_the_action_and_ignores_scrolling_chat():
    moving = (40, 70, 20, 60)                         # the action, lower left
    chat = (0, 90, 130, 160)                          # chat scrolls on the right edge
    grays = _grays(20, moving)
    rng = np.random.default_rng(3)
    for g in grays:
        g[chat[0]:chat[1], chat[2]:chat[3]] = rng.integers(0, 255, (90, 30))
    mask = np.zeros((90, 160), np.float32)
    mask[:, 128:] = 1.0
    z = recompose.dual_view_zoom_plan([_sample(t / 2) for t in range(20)], grays, [], 10.0, W, H,
                                      820, mask=mask)
    zw = z["zoom"]["w"]
    (s,) = z["zoom_segments"]
    k = 160 / W
    assert s["how"] == "motion" and abs(zw - recompose.DV_MOTION_FRAC * H * 1080 / 820) < 4
    assert s["x"] * k <= 20 and (s["x"] + zw) * k >= 60 and (s["x"] + zw) * k <= 130


def test_each_camera_shot_gets_its_own_zoom_window():
    samples = [_sample(t / 2, [(0.3, 0.4, 0.2)] if t < 10 else []) for t in range(20)]
    grays = _grays(10) + _grays(10, (10, 40, 110, 150), seed=1)
    z = recompose.dual_view_zoom_plan(samples, grays, [5.0], 10.0, W, H, 820)
    a, b = z["zoom_segments"]
    assert (a["how"], b["how"]) == ("face", "motion") and b["t0"] == 5.0 and b["cut"]
    assert a["t0"] == 0.0 and b["t1"] == 10.0 and a["x"] < b["x"]


def test_a_tiny_face_zoom_is_capped_at_the_upscale_limit():
    samples = [_sample(t / 2, [(0.5, 0.5, 0.05)]) for t in range(20)]
    z = recompose.dual_view_zoom_plan(samples, _grays(20), [], 10.0, W, H, 820)
    assert z["zoom"]["h"] >= 820 / recompose.DV_MAX_UPSCALE - 2


def test_a_cam_is_cover_cropped_to_the_panel_and_stays_in_its_box():
    ar = 1080 / 820
    w, h, x, y = recompose.dual_view_cam_crop([960, 280, 1270, 450], [0.85, 0.5, 0.15], ar, W, H)
    assert h == 170 and abs(w / h - ar) < 0.02 and 960 <= x and x + w <= 1270 and y == 280
    w, h, x, y = recompose.dual_view_cam_crop([1000, 100, 1240, 600], [0.86, 0.3, 0.1], ar, W, H)
    assert w == 240 and abs(w / h - ar) < 0.03 and 100 <= y and y + h <= 600    # portrait: covered


# ---------------------------------------------------------------- the layout chooser

@pytest.fixture
def hud(monkeypatch):
    """hud_layout with the scan, OCR, window scan and the dual view planner replaced."""
    def run(plan, sc, a=None, guard=None, rejected=None, dv=True, wins=None):
        recompose._SCANS.clear()
        monkeypatch.setattr(recompose, "_scan", lambda src, a, size: sc)
        monkeypatch.setattr(qg, "ocr_source_text", lambda *a, **k: None)
        monkeypatch.setattr(cleancrop, "window_boxes", lambda *a, **k: wins)
        monkeypatch.setattr(recompose, "plan_dual_view",
                            lambda src, a, prior=None: _dv() if dv else None)
        monkeypatch.setattr(recompose.reframe, "probe_size", lambda src: (W, H))
        return recompose.hud_layout(plan, a or G(), "src.mp4", guard, rejected)
    return run


def _track(xs=(438,)):
    return {"layout": "track", "src_w": W, "src_h": H,
            "segments": [{"t0": i * 5.0, "t1": (i + 1) * 5.0, "x": x} for i, x in enumerate(xs)]}


def test_a_clean_full_bleed_crop_is_unchanged(hud):
    plan, guard = _track(), {"action": "none"}
    before = copy.deepcopy(plan)
    assert hud(plan, _scan(), guard=guard) is guard
    assert plan == before and plan["layout"] == "track" and "crop_box" not in plan


def test_a_slicing_crop_keeps_full_bleed_when_a_clean_recrop_exists(hud):
    plan = _track(xs=(500,))
    sc = _scan(graphics=[((880, 60, 1000, 100), False)],
               faces=[(1.0, [{"cx": 0.55, "cy": 0.4, "w": 0.12, "h": 0.2, "score": 0.9}])])
    rep = hud(plan, sc, guard={"action": "none"})
    assert rep["action"] == "ladder_a_recrop" and plan["layout"] == "track"


def test_a_slicing_crop_with_no_clean_recrop_becomes_the_dual_view_not_a_box(hud):
    """472: the board graphic beside the player; the ladder alone would box it (c_box). A campaign
    render: a niche clip gets no dual view since 11-09."""
    plan = _track()
    sc = _scan(graphics=[((700, 400, 1000, 440), False)])
    sc["graphics"][0]["frac"] = 0.3
    rep = hud(plan, sc, a=C(), guard={"action": "none"})
    assert rep["action"] == "dual_view" and rep["replaced"] == "track"
    assert plan["layout"] == "dual_view" and plan["banner_guard"] is rep
    assert "a_recrop" in "; ".join(rep["tried"])


def test_the_same_slice_outside_the_hud_niches_still_takes_the_ladder(monkeypatch):
    plan = _track()
    sc = _scan(graphics=[((700, 400, 1000, 440), False)])
    sc["graphics"][0]["frac"] = 0.3
    recompose._SCANS.clear()
    monkeypatch.setattr(recompose, "_scan", lambda src, a, size: sc)
    monkeypatch.setattr(qg, "ocr_source_text", lambda *a, **k: None)
    monkeypatch.setattr(cleancrop, "window_boxes", lambda *a, **k: None)
    pod = type("P", (), {"start": 0.0, "dur": 10.0, "niche": "podcasts", "vertical": "podcast"})()
    rep = recompose.slice_ladder(plan, pod, "src.mp4", {"action": "none"})
    assert rep["action"] == "ladder_c_box"


def test_poker_gets_no_box_and_no_dual_view_when_no_full_bleed_crop_is_clean(monkeypatch):
    """poker is full-bleed (a HUD niche that may not stack). The same slice with no clean
    recrop is rejected before the encode: no fit-to-width box, no dual view."""
    plan = _track()
    sc = _scan(graphics=[((700, 400, 1000, 440), False)])
    sc["graphics"][0]["frac"] = 0.3
    recompose._SCANS.clear()
    monkeypatch.setattr(recompose, "_scan", lambda src, a, size: sc)
    monkeypatch.setattr(qg, "ocr_source_text", lambda *a, **k: None)
    monkeypatch.setattr(cleancrop, "window_boxes", lambda *a, **k: None)
    monkeypatch.setattr(recompose, "plan_dual_view", lambda *a, **k: pytest.fail("no dual view"))
    assert not recompose.dual_view_allowed(A()) and recompose.hud_niche(A())
    with pytest.raises(recompose.RenderError, match="a_recrop"):
        recompose.slice_ladder(plan, A(), "src.mp4", {"action": "none"})


def test_a_stacked_layout_that_slices_text_becomes_the_dual_view(hud):
    gs = {"layout": "gamestack", "src_w": W, "src_h": H, "top_h": 1344,
          "game": [578, 300, False], "cam": [400, 300, 860, 20, False],
          "cam_box": [860, 20, 1260, 320], "face_median": [0.83, 0.2, 0.15]}
    sc = _scan(graphics=[((250, 650, 700, 690), True)])         # a socials bar the game crop cuts
    rep = hud(gs, sc, a=C(), guard={"action": "none"})           # a campaign render (11-09)
    assert rep["action"] == "dual_view" and rep["replaced"] == "gamestack"


def test_a_stacked_layout_that_slices_nothing_is_unchanged(hud):
    gs = {"layout": "gamestack", "src_w": W, "src_h": H, "top_h": 1344,
          "game": [578, 300, False], "cam": [400, 300, 860, 20, False]}
    guard = {"action": "none"}
    assert hud(gs, _scan(), guard=guard) is guard and gs["layout"] == "gamestack"


def test_a_full_frame_letterbox_over_the_blur_cap_becomes_the_dual_view(hud):
    plan = {"layout": "general"}
    rep = hud(plan, _scan(), a=C())                               # a campaign render (11-09)
    assert rep["action"] == "dual_view" and "blurred fill" in rep["why"]


def test_a_rejected_crop_with_no_room_for_a_zoom_is_rejected_as_before(hud):
    gs = {"layout": "gamestack", "src_w": W, "src_h": H, "top_h": 1344,
          "game": [578, 300, False], "cam": [400, 300, 860, 20, False]}
    with pytest.raises(recompose.RenderError, match="game crop"):
        hud(gs, _scan(), rejected="crop would slice a source banner: game crop ...", dv=False)


# a poker table under a broadcast's graphics: discover calls the source a screen and no face is
# found, so the screen fit's box applies to it too (poker is no stacking niche)
POKER_SCREEN = [(380, 110, 760, 150), (1150, 160, 1240, 600), (330, 620, 900, 660)]


class PokerScreen(A):
    content_kind = "screen"


@pytest.fixture
def no_recrop(monkeypatch):
    """No clean full-bleed recrop (the chooser's first step): the dual view and the box are next."""
    def fail(*a, **k):
        raise cleancrop.CleanCropError("no clean 9:16 window")
    monkeypatch.setattr(cleancrop, "clean_crop", fail)


def test_a_poker_screen_gets_the_screen_box_never_the_dual_view(no_recrop, monkeypatch):
    """poker may not stack, so a poker screen the speaker crop would slice is fit to
    width (the screen box, poker being a fit-to-width niche), and main never asks hud_layout."""
    monkeypatch.setattr(recompose, "content_blocks", lambda *a, **k: [])
    monkeypatch.setattr(recompose, "plan_dual_view", lambda *a, **k: pytest.fail("no dual view"))
    plan = _track()
    sc = _scan(graphics=[(g, False) for g in POKER_SCREEN], faces=[])
    recompose._SCANS.clear()
    monkeypatch.setattr(recompose, "_scan", lambda src, a, size: sc)
    monkeypatch.setattr(qg, "ocr_source_text", lambda *a, **k: None)
    monkeypatch.setattr(cleancrop, "window_boxes", lambda *a, **k: None)
    assert recompose.fit_width_allowed(PokerScreen()) and recompose.screen_content(PokerScreen(), sc)
    assert not recompose.dual_view_allowed(PokerScreen())
    rep = recompose.slice_ladder(plan, PokerScreen(), "src.mp4", {"action": "none"},
                                 rejected="crop would slice a source banner: board")
    assert rep["action"] == "ladder_d_screen_fit"
    assert plan["layout"] == "general" and plan["content_box"]


def test_only_gaming_and_streamers_get_the_dual_view():
    """the dual view is a stack; only the stacking niches render it. Poker keeps the HUD
    rule (whole plates) without the stack."""
    mk = lambda n, v=None: type("X", (), {"niche": n, "vertical": v})()   # noqa: E731
    assert all(recompose.dual_view_allowed(mk(n)) for n in ("gaming", "streamers"))
    assert not any(recompose.dual_view_allowed(mk(n, "stream"))
                   for n in ("poker", "podcasts", "tech", "finance", "lifestyle"))
    assert recompose.dual_view_allowed(mk(None, "stream"))
    assert not recompose.dual_view_allowed(mk(None, "poker"))
    assert not recompose.dual_view_allowed(mk(None, "podcast"))
    assert recompose.hud_niche(mk("poker")) and recompose.hud_niche(mk(None, "poker"))
    assert not recompose.hud_niche(mk("podcasts", "stream"))


def test_tech_and_finance_never_render_the_dual_view():
    a = type("X", (), {"niche": "tech"})()
    with pytest.raises(recompose.RenderError, match="single view"):
        recompose.enforce_single_view(_dv(), a)


def test_hud_panels_keep_a_board_graphic_whole():
    """472: OCR reads '$24,500' and two cards of the board, not the panel around them."""
    p = recompose.hud_panels([[1124, 560, 1223, 587], [1061, 590, 1159, 629],
                              [1164, 597, 1230, 619], [44, 569, 165, 627]], W, H)
    assert len(p) == 2
    board = max(p, key=lambda b: b[0])
    assert board[0] <= 1061 - 30 and board[2] >= 1230 + 30 and board[1] < 560 and board[3] > 629


# ---------------------------------------------------------------- the facecam below

def test_a_cam_that_holds_a_face_in_every_shot_is_the_bottom():
    box = [0.75, 0.4, 0.99, 0.62]
    samples = [_sample(t / 2, [(0.85, 0.5, 0.05)]) for t in range(20)]
    assert recompose.cam_holds(samples, [5.0], 10.0, box)


def test_a_cam_lost_when_the_stream_cuts_to_its_full_camera_is_not():
    """569: the inset holds for 5 s, then the stream is the streamer's full camera."""
    box = [0.0, 0.35, 0.15, 0.71]
    samples = ([_sample(t / 2, [(0.07, 0.5, 0.12)]) for t in range(10)]
               + [_sample(5 + t / 2, [(0.48, 0.46, 0.35)]) for t in range(10)])
    assert not recompose.cam_holds(samples, [5.0], 10.0, box)
    z = recompose.dual_view_zoom_plan(samples, _grays(20), [5.0], 10.0, W, H, 820, cam=box)
    a, b = z["zoom_segments"]
    assert a["x"] <= 0.07 * W <= a["x"] + z["zoom"]["w"]        # the cam face where it shows
    assert b["x"] <= 0.48 * W <= b["x"] + z["zoom"]["w"]        # the streamer after the cut


def test_a_one_sample_flash_shot_does_not_drop_the_cam():
    box = [0.75, 0.4, 0.99, 0.62]
    samples = [_sample(t / 2, [] if t == 4 else [(0.85, 0.5, 0.05)]) for t in range(20)]
    assert recompose.cam_holds(samples, [2.0, 2.5], 10.0, box)


def test_a_small_cam_is_shown_whole_inside_its_panel_within_the_blur_cap():
    """570: a 290x164 inset; covering 820 rows would be 5x, fit inside it is 3.7x."""
    z = recompose.dual_view_cam_zoom([974, 284, 1264, 448], [0.84, 0.55, 0.05], 820, 492, W, H)
    assert z == {"w": 290, "h": 164, "x": 974, "y": 284, "fit": "inside"}
    p = dict(_dv(), bottom="cam", zoom={"w": 290, "h": 164, "fit": "inside"},
             zoom_segments=[{"t0": 0.0, "t1": 10.0, "x": 974, "y": 284, "cut": True}])
    assert recompose.blur_fill_frac(p, A(), {}) <= qg.VQ_MAX_BLUR_FILL
    g = recompose.dual_view_vf(p)
    assert "crop=290:164:'974':'284',split=2[zbg][zfg]" in g and "gblur" in g.split("[zm]")[0]


def test_a_cam_big_enough_is_covered():
    z = recompose.dual_view_cam_zoom([600, 100, 1240, 600], [0.7, 0.4, 0.2], 820, 492, W, H)
    assert "fit" not in z and 820 / z["h"] <= recompose.DV_MAX_UPSCALE


def test_a_split_of_one_picture_twice_is_a_defect_two_people_side_by_side_are_not():
    dup = {"layout": "split", "split": {"left": {"x": 244, "y": 0, "w": 798, "h": 710},
                                        "right": {"x": 482, "y": 10, "w": 798, "h": 710}}}
    two = {"layout": "split", "split": {"left": {"x": 0, "y": 0, "w": 600, "h": 720},
                                        "right": {"x": 680, "y": 0, "w": 600, "h": 720}}}
    assert recompose.split_overlap(dup) >= recompose.DV_SPLIT_OVERLAP
    assert recompose.split_overlap(two) == 0.0 and recompose.split_overlap(_track()) == 0.0


def test_a_split_of_one_camera_twice_becomes_the_dual_view(hud):
    """309: a streamer's camera stacked twice (a pumpkin as the second 'speaker'); its logo and
    socials bar are cut where the OCR reads nothing."""
    sp = {"layout": "split", "src_w": W, "src_h": H,
          "split": {"left": {"x": 244, "y": 0, "w": 798, "h": 710},
                    "right": {"x": 482, "y": 10, "w": 798, "h": 710}}}
    rep = hud(sp, _scan(), a=C(), guard={"action": "none"})      # a campaign render (11-09)
    assert rep["action"] == "dual_view" and "same picture twice" in rep["why"]


def test_a_big_face_cam_is_no_inset():
    """309: the 'cam' small_cam_inset found on a streamer's full camera was a pumpkin (0.17)."""
    assert recompose.small_cam_face({"face_median": [0.84, 0.55, 0.05]})
    assert not recompose.small_cam_face({"face_median": [0.94, 0.6, 0.17]})
    assert not recompose.small_cam_face({"face_median": None})


def test_a_big_face_still_gets_a_real_zoom_not_the_frame_again():
    """569 / 561: a face 0.35 of the frame asked for a window taller than the frame."""
    samples = [_sample(t / 2, [(0.5, 0.45, 0.35)]) for t in range(20)]
    z = recompose.dual_view_zoom_plan(samples, _grays(20), [], 10.0, W, H, 820)
    assert z["zoom"]["h"] <= recompose.DV_MOTION_FRAC * H + 2
    s = z["zoom_segments"][0]
    assert s["y"] <= (0.45 - 0.35 / 2) * H                     # the whole face is in the window
    assert s["y"] + z["zoom"]["h"] >= (0.45 + 0.35 / 2) * H


def test_flash_cuts_do_not_make_the_zoom_jump():
    """570: muzzle flashes read as 17 cuts in 15.7 s; shots under DV_MIN_SHOT_S join a neighbour."""
    assert recompose.zoom_shots([0.5, 1.0, 4.5, 5.0, 9.0], 10.0) == [(0.0, 5.0), (5.0, 10.0)]
    shots = recompose.zoom_shots([0.5, 1.0, 1.5, 2.0, 6.0], 10.0)
    assert shots[0][0] == 0.0 and shots[-1][1] == 10.0
    assert all(b - a >= recompose.DV_MIN_SHOT_S for a, b in shots)


def test_in_dual_view_a_graphic_is_cut_at_the_frame_edge_only_where_its_text_reaches_it():
    """377: the FIELD box and the board end 10-35 px inside the frame; the line detector's boxes
    reach its edge. Another layout keeps reading the detector box."""
    sc = _scan(graphics=[((1148, 6, 1280, 94), False), ((826, 600, 1280, 720), False)])
    words = [(1160, 10, 1240, 40, "FIELD", 0), (1170, 50, 1236, 80, "7/479", 0),
             (830, 650, 1262, 680, "POT 940,000", 0)]
    whole = qg.crop_slices(sc, [], 10.0, words=words)
    assert qg.crop_slice_verdict(whole)["passed"] is False      # general: unchanged
    dual = qg.crop_slices(sc, [], 10.0, words=words, text_edges=True)
    assert dual == []
    cut = [(1160, 10, 1279, 40, "FIELD SIZE", 0)]                 # text that runs off the frame
    sc2 = _scan(graphics=[((1148, 6, 1280, 94), False)])
    assert qg.crop_slices(sc2, [], 10.0, words=cut, text_edges=True)[0]["how"] == "frame"
