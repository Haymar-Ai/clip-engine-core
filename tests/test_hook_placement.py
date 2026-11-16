"""Hook placement and crop guards against on-screen source text, one rule each:

  1. hook placement knows about on-screen source text/UI (a finance screen share whose ticker
     rows the hook would cover): textzones finds the bands, captions puts the hook in a clear
     band or raises HookPlacementError (the render is rejected);
  2. a 9:16 crop never slices a persistent source banner (a lower third cut to "...23 in OT"):
     banners shifts or trims the crop, or the render is rejected; the gamestack game crop also
     stays off a stream's chat column;
  3. stream sources render as gamestack, not the general blur letterbox, and recompose reports
     the blur-fill share it drew.

Offline: synthetic frames / plans; the two encode tests render tiny testsrc2 media.
"""
import json
import subprocess

import numpy as np
import pytest

from clipengine.render import banners
from clipengine.render import captions
from clipengine.render import recompose
from clipengine.render import textzones

PW, PH = captions.PLAY_W, captions.PLAY_H


@pytest.fixture(autouse=True)
def _fresh_scans():
    """recompose keeps each window's banner scan / OCR for the rest of the process
    (recompose._SCANS, keyed on source and window). The tests here reuse one fake source and
    window with different fake scans, so every test starts without a kept scan."""
    recompose._SCANS.clear()
    yield
    recompose._SCANS.clear()


def _hook(hook, **kw):
    line, size = captions.hook_event(hook, dur=9, **kw)
    y = int(line.split("\\pos(540,")[1].split(")")[0])
    n_lines = line.count("\\N") + 1
    half = n_lines * size * captions.HOOK_LINE_H / 2
    return y - half, y + half


# ---------------------------------------------------------------- 1. hook vs source text

def test_hook_moves_below_a_ticker_band():
    top, bottom = _hook("Why I bet against the crowd", text_avoid=[(150, 440)])
    assert top >= 440 + captions.TEXT_PAD
    assert bottom <= captions.CAPTION_Y - captions.CAPTION_CLEAR


def test_source_text_outranks_faces():
    faces = [(600, 1180)]
    text = [(230, 590)]
    top, bottom = _hook("Why I bet against the crowd", avoid=faces, text_avoid=text)
    assert captions.band_overlap((top + bottom) / 2, bottom - top, text, captions.TEXT_PAD) == 0
    assert top >= 590                                  # on the face rather than on the text


def test_no_clear_band_rejects():
    with pytest.raises(captions.HookPlacementError, match="clears the on-screen text"):
        captions.hook_event("Why I bet against the crowd", dur=9,
                            text_avoid=[(200, 700), (700, 1300)])
    with pytest.raises(captions.HookPlacementError):     # general: hook sits in the top band
        captions.hook_event("Big hook", layout="general", dur=9, text_avoid=[(250, 400)])
    # no text bands -> the old behaviour, unchanged
    assert captions.hook_event("Big hook", dur=9, text_avoid=[])[0]


def test_tesseract_tsv_keeps_confident_words_only():
    tsv = "\n".join([
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
        "5\t1\t1\t1\t1\t1\t32\t227\t293\t52\t90.9\t$7,811.050",
        "5\t1\t1\t1\t1\t2\t240\t154\t81\t24\t94.0\tIndex",
        "5\t1\t1\t1\t1\t3\t507\t139\t153\t54\t30.6\tHGEC",        # low confidence
        "5\t1\t1\t1\t1\t4\t931\t98\t73\t93\t93.2\tN",            # single letter
        "5\t1\t1\t1\t1\t5\t0\t1373\t1075\t492\t95.0\tab",       # texture-sized "word"
    ])
    assert textzones.parse_tesseract_tsv(tsv) == [(32, 227, 325, 279, "$7,811.050"),
                                                  (240, 154, 321, 178, "Index")]


def test_bands_need_two_frames_and_merge_close_rows():
    counts = np.zeros(PH, int)
    counts[100:140] = 3
    counts[150:170] = 2           # 10 px gap -> merged
    counts[400:420] = 1           # one frame only (a cursor, a flash) -> ignored
    assert textzones.bands_from_counts(counts) == [(100, 170)]


def _ticker_canvas():
    """A finance-like canvas: dense ticker rows at the top, dark empty chart below, a soft
    face-cam blob at the bottom."""
    import cv2
    g = np.full((PH, PW), 12, np.uint8)
    for i, y in enumerate((170, 250, 330, 410)):
        cv2.putText(g, "S&P 500 INDEX SPX $7,811.350 +3.05 (0.04%) OPEN BET VOLUME"[i * 3:],
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.25, 235, 3)
    g[1450:1920] = cv2.GaussianBlur(np.random.default_rng(0).integers(60, 200, (470, PW))
                                    .astype(np.uint8), (0, 0), 9)
    return g


def test_ticker_rows_become_bands_and_the_hook_clears_them(monkeypatch):
    monkeypatch.setattr(textzones, "_ocr_available", lambda: False)   # edge density alone
    frames = [_ticker_canvas()] * 4
    tz = textzones.hook_zone_text(None, 0, 3.4, None, frames=frames)
    assert tz["frames"] == 4 and tz["bands"]
    lo = min(b[0] for b in tz["bands"])
    hi = max(b[1] for b in tz["bands"] if b[0] < 1000)
    assert lo <= 150 and 400 <= hi <= 460                  # the four ticker rows, nothing else
    top, _ = _hook("Why I bet against the crowd", text_avoid=tz["bands"])
    assert top >= hi + captions.TEXT_PAD


def test_flat_or_blurred_canvas_has_no_bands(monkeypatch):
    import cv2
    monkeypatch.setattr(textzones, "_ocr_available", lambda: False)
    blur = cv2.GaussianBlur(np.random.default_rng(1).integers(0, 255, (PH, PW)).astype(np.uint8),
                            (0, 0), 22)
    assert textzones.hook_zone_text(None, 0, 3, None, frames=[blur] * 3)["bands"] == []
    assert not textzones.dense_rows(np.full((PH, PW), 90, np.uint8)).any()


# ---------------------------------------------------------------- 2. crops vs source banners

SW, SH = 1280, 720
CW = 404                                     # 9:16 full-height crop of 1280x720


def _face(cx, cy=0.4, w=0.1, h=0.2):
    return {"cx": cx, "cy": cy, "w": w, "h": h, "score": 0.9}


def _sc(graphics, faces, panels=()):
    return {"src_w": SW, "src_h": SH, "frames": 20, "graphics": graphics,
            "panels": [list(p) for p in panels], "faces": faces, "hgrad": None, "scale": 2.0}


LOWER_THIRD = {"box": [244, 658, 1036, 688], "frac": 1.0, "banner": True}   # "Ravens beat ..."


def test_sliced_means_partly_in_partly_out():
    b = banners.plate(LOWER_THIRD["box"], SW, SH)
    assert banners.sliced(b, 855, CW, SW)                  # clip_5's crop: "...23 in OT"
    assert not banners.sliced([900, 600, 1100, 630], 855, CW, SW)   # whole inside
    assert not banners.sliced([0, 600, 300, 630], 855, CW, SW)      # whole outside


def test_lower_third_wider_than_the_crop_is_trimmed_out():
    faces = [(t, [_face(0.82)]) for t in range(0, 20)] + \
            [(t, [_face(0.2)]) for t in range(20, 27)]
    plan = {"layout": "track", "src_w": SW, "src_h": SH,
            "segments": [{"t0": 0.85, "t1": 20.85, "x": 855, "cut": True},
                         {"t0": 20.85, "t1": 27.05, "x": 72, "cut": True}]}
    rep = banners.fix_track_plan(plan, _sc([LOWER_THIRD], faces))
    assert rep["action"] == "trim"
    box = plan["crop_box"]
    plate = banners.plate(LOWER_THIRD["box"], SW, SH)
    assert box["y"] == 0 and box["y"] + box["h"] <= plate[1]       # banner rows cut away
    assert box["h"] >= banners.MIN_KEEP_H_FRAC * SH
    assert box["w"] == round(box["h"] * 9 / 16) // 2 * 2
    for seg, cx in zip(plan["segments"], (0.82, 0.2)):              # faces still framed
        assert seg["x"] + 0.01 * SW <= (cx - 0.05) * SW
        assert (cx + 0.05) * SW <= seg["x"] + box["w"] - 0.01 * SW


def test_small_banner_is_shifted_fully_into_the_crop():
    g = {"box": [700, 600, 900, 630], "frac": 1.0, "banner": True}
    plan = {"layout": "track", "src_w": SW, "src_h": SH,
            "segments": [{"t0": 0.0, "t1": 10.0, "x": 800, "cut": True}]}
    rep = banners.fix_track_plan(plan, _sc([g], [(t, [_face(0.75)]) for t in range(10)]))
    x = plan["segments"][0]["x"]
    assert rep["action"] == "shift" and "crop_box" not in plan
    assert not banners.sliced(banners.plate(g["box"], SW, SH), x, CW, SW)


def test_mid_frame_ticker_that_cannot_be_trimmed_rejects():
    ticker = {"box": [0, 330, 1280, 360], "frac": 1.0, "banner": True}
    plan = {"layout": "track", "src_w": SW, "src_h": SH,
            "segments": [{"t0": 0.0, "t1": 10.0, "x": 438, "cut": True}]}
    rep = banners.fix_track_plan(plan, _sc([ticker], [(t, [_face(0.5)]) for t in range(10)]))
    assert rep["action"] == "reject" and "leaves" in rep["reason"]


def test_trim_keeps_screen_share_labels_whole():
    """Finance: the crop sat flush on the frame's left edge, where the UI labels start."""
    top_banner = {"box": [350, 8, 922, 92], "frac": 0.8, "banner": True}   # NOT FINANCIAL ADVICE
    plan = {"layout": "track", "src_w": SW, "src_h": SH,
            "segments": [{"t0": 0.0, "t1": 23.7, "x": 4, "cut": True}]}
    words = [(14, 480, 28, 497), (33, 482, 89, 494)]                       # "My Positions"
    rep = banners.fix_track_plan(plan, _sc([top_banner], [(t, [_face(0.15, 0.85, 0.08, 0.15)])
                                                          for t in range(20)]),
                                 words_fn=lambda: words)
    assert rep["action"] == "trim" and plan["crop_box"]["y"] >= 92
    x, w = plan["segments"][0]["x"], plan["crop_box"]["w"]
    assert banners._cuts_words(words, x, w, plan["crop_box"]["y"], SH) == 0
    assert x == 4


def test_gamestack_game_crop_moves_off_the_chat_column():
    plan = {"layout": "gamestack", "src_w": SW, "src_h": SH, "top_h": 1344,
            "game": [578, 468, False], "cam": [180, 96, 28, 66, False],
            "raw_cam_box": [3, 18, 235, 250]}
    subtitle = {"box": [388, 548, 910, 636], "frac": 1.0, "banner": True}
    rep = banners.fix_gamestack_plan(plan, _sc([subtitle], [], panels=[[1000, 0, 1280, 280]]))
    gx = plan["game"][1]
    assert rep["action"] == "shift" and rep["unresolved_panels"] == []
    assert gx >= 235 and gx + 578 <= 1000 + banners.SLICE_TOL_FRAC * SW   # cam + chat out
    assert not banners.sliced(subtitle["box"], gx, 578, SW)               # subtitle whole


def test_gamestack_rejects_when_a_banner_cannot_be_kept_whole():
    plan = {"layout": "gamestack", "src_w": SW, "src_h": SH, "top_h": 1344,
            "game": [578, 468, False], "cam": [180, 96, 28, 66, False],
            "raw_cam_box": [3, 18, 235, 250]}
    wide = {"box": [300, 600, 1200, 640], "frac": 1.0, "banner": True}
    assert banners.fix_gamestack_plan(plan, _sc([wide], []))["action"] == "reject"


def test_banner_scan_finds_a_persistent_lower_third():
    import cv2
    frames = []
    for i in range(10):
        f = np.full((360, 640, 3), 120, np.uint8)
        f[:300] = cv2.GaussianBlur(np.random.default_rng(i).integers(0, 255, (300, 640, 3))
                                   .astype(np.uint8), (0, 0), 6)
        cv2.rectangle(f, (110, 322), (530, 348), (30, 40, 90), -1)
        cv2.putText(f, "Ravens beat Falcons 30-23 in OT", (122, 342),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        frames.append((float(i), f))
    sc = banners.scan(None, 0, 10, frames=frames, src_size=(SW, SH))
    lt = [g for g in sc["graphics"] if g["banner"]]
    assert len(lt) == 1 and lt[0]["frac"] == 1.0
    x0, y0, x1, y1 = lt[0]["box"]
    assert x0 < 300 and x1 > 700 and 620 < y0 < y1 < 720       # the text line, in src px


def test_guard_rejects_the_render(monkeypatch):
    ticker = {"box": [0, 330, 1280, 360], "frac": 1.0, "banner": True}
    monkeypatch.setattr(banners, "scan", lambda *a, **k: _sc([ticker], [(0.0, [_face(0.5)])]))
    plan = {"layout": "track", "src_w": SW, "src_h": SH,
            "segments": [{"t0": 0.0, "t1": 5.0, "x": 438, "cut": True}]}
    a = recompose.parse_args(["s.mp4", "0", "5", "o.mp4"])
    with pytest.raises(recompose.RenderError, match="slice a source banner"):
        recompose.guard_banners(plan, a, "s.mp4")
    assert recompose.guard_banners({"layout": "general"}, a, "s.mp4") is None


def test_unverifiable_crop_is_rejected(monkeypatch):
    def broken(*a, **k):
        raise OSError("ffmpeg missing")
    monkeypatch.setattr(banners, "scan", broken)
    plan = {"layout": "track", "src_w": SW, "src_h": SH,
            "segments": [{"t0": 0.0, "t1": 5.0, "x": 438, "cut": True}]}
    a = recompose.parse_args(["s.mp4", "0", "5", "o.mp4"])
    with pytest.raises(recompose.RenderError, match="banner scan failed"):
        recompose.guard_banners(plan, a, "s.mp4")


def test_trimmed_crop_reaches_the_graph_and_the_hook_views():
    plan = {"layout": "track", "src_w": SW, "src_h": SH,
            "crop_box": {"w": 356, "h": 634, "y": 0},
            "segments": [{"t0": 0.85, "t1": 20.0, "x": 883}, {"t0": 20.0, "t1": 27.0, "x": 96}]}
    assert "crop=356:634:'if(lt(t,20),883,96)':0,scale=1080:1920" in recompose.track_vf(plan)
    # before the first segment the crop is the first segment's (crop_x_expr), not the last's
    assert recompose._hook_views(plan, 0.2) == [(883, 356, 0, 634, 0, 1920)]
    assert recompose._hook_views(plan, 21.0)[0][0] == 96
    full = {k: v for k, v in plan.items() if k != "crop_box"}
    assert "crop=404:ih:" in recompose.track_vf(full)


# ---------------------------------------------------------------- 2b. box + screen stack

TOP_STRIP = {"box": [0, 0, 264, 20], "frac": 0.75, "banner": True}   # 10-19 clip 84


def test_banner_rows_box_the_crop_when_the_trim_window_loses_the_face():
    """10-19 clip 84: a 20 px strip at the top-left; trimming it narrows the 9:16 window to 384
    px and a close-up head no longer fits. The box keeps the 404 px width and drops the rows."""
    faces = [(t, [_face(0.24, 0.42, 0.21, 0.55)]) for t in range(0, 6)] + \
            [(t, [_face(0.537, 0.42, 0.285, 0.55)]) for t in range(6, 12)]   # 365 px wide
    plan = {"layout": "track", "src_w": SW, "src_h": SH,
            "segments": [{"t0": 0.0, "t1": 6.0, "x": 85, "cut": True},
                         {"t0": 6.0, "t1": 12.0, "x": 486, "cut": True}]}
    rep = banners.fix_track_plan(plan, _sc([TOP_STRIP], faces))
    assert rep["action"] == "box", rep
    plate = banners.plate(TOP_STRIP["box"], SW, SH)
    assert plan["crop_box"] == {"w": CW, "h": SH - plate[3] - banners.CUT_MARGIN,
                                "y": plate[3] + banners.CUT_MARGIN, "fit": "box"}
    assert [s["x"] for s in plan["segments"]] == [85, 486]          # faces were already framed


def test_box_needs_enough_rows():
    ticker = {"box": [0, 300, 1280, 330], "frac": 1.0, "banner": True}
    plan = {"layout": "track", "src_w": SW, "src_h": SH,
            "segments": [{"t0": 0.0, "t1": 10.0, "x": 438, "cut": True}]}
    rep = banners.fix_track_plan(plan, _sc([ticker], [(t, [_face(0.5, 0.75)]) for t in range(10)]))
    assert rep["action"] == "reject" and "box:" in rep["reason"] and "crop_box" not in plan


def test_boxed_crop_graph_blur_fill_and_hook_views():
    plan = {"layout": "track", "src_w": SW, "src_h": SH,
            "crop_box": {"w": 404, "h": 682, "y": 38, "fit": "box"},
            "segments": [{"t0": 0.0, "t1": 6.0, "x": 85}, {"t0": 6.0, "t1": 12.0, "x": 486}]}
    vf = recompose.track_vf(plan)
    assert "crop=404:682:'if(lt(t,6),85,486)':38,split=2" in vf and "gblur" in vf
    assert "overlay=(W-w)/2:(H-h)/2" in vf
    a = recompose.parse_args(["s.mp4", "0", "5", "o.mp4"])
    out_h = round(1080 * 682 / 404)
    assert recompose.blur_fill_frac(plan, a, {}) == pytest.approx(1 - out_h / 1920, abs=0.002)
    assert recompose._hook_views(plan, 7.0) == [(486, 404, 38, 682, (1920 - out_h) // 2, out_h)]


def _screen_share_sc():
    rows = [{"box": [0, y, 1100, y + 30], "frac": 1.0, "banner": True} for y in range(60, 660, 80)]
    status = {"box": [34, 706, 1280, 720], "frac": 1.0, "banner": True}
    return _sc(rows + [status], [(t, [_face(0.96, 0.82, 0.08, 0.13)]) for t in range(10)])


def _cam(box, face):
    return {"src_w": SW, "src_h": SH, "top_h": 960, "cam_box": box, "guard_box": box,
            "box_norm": [v / d for v, d in zip(box, (SW, SH, SW, SH))],
            "raw_box_norm": [v / d for v, d in zip(box, (SW, SH, SW, SH))], "face_frac": 1.0,
            "from_profile": False, "face_median": face, "channel": None, "profile_key": None}


def test_screen_share_whose_crop_slices_its_ui_is_reframed_as_a_screen_stack(monkeypatch):
    """10-19 clips 73/74/76/78/80: a trading platform / editor with a corner cam. The 9:16
    crop following the cam face slices every UI row; the whole screen goes on top instead."""
    monkeypatch.setattr(banners, "scan", lambda *a, **k: _screen_share_sc())
    monkeypatch.setattr(recompose, "_resolve_cam",
                        lambda *a, **k: _cam([1110, 500, 1280, 720], [0.977, 0.826, 0.133]))
    plan = {"layout": "track", "src_w": SW, "src_h": SH,
            "segments": [{"t0": 0.0, "t1": 10.0, "x": 876, "cut": True}]}
    a = recompose.parse_args(["s.mp4", "0", "10", "o.mp4", "--niche", "streamers"])
    rep = recompose.guard_banners(plan, a, "s.mp4")
    assert rep["action"] == "reframe" and "rejected_crop" in rep
    assert plan["layout"] == "reaction" and plan["screen_stack"] is True
    assert plan["content"] == [SW, SH, 0, 0] and plan["content_fit"] == "fit"
    assert plan["top_h"] == recompose.SCREEN_STACK_TOP_H
    cw, ch, cx, cy, fit = plan["cam"]
    assert cy + ch <= 706 - banners.CUT_MARGIN and fit is True     # status bar left out
    vf = recompose.reaction_vf(plan)
    assert "[tfg]crop=1280:720:0:0" in vf and "overlay=(W-w)/2:H-h[top]" in vf    # hook room on top
    assert plan["banner_guard"]["action"] == "reframe"
    assert recompose.blur_fill_frac(plan, a, {}) < 0.5


def test_talking_head_whose_banner_sits_on_the_speaker_still_rejects(monkeypatch):
    """10-19 clips 82/85: the 'cam' found is the speaker filling half the frame, not an inset."""
    monkeypatch.setattr(banners, "scan", lambda *a, **k: _sc(
        [{"box": [0, 330, 1280, 360], "frac": 1.0, "banner": True}], [(0.0, [_face(0.5)])]))
    monkeypatch.setattr(recompose, "_resolve_cam",
                        lambda *a, **k: _cam([270, 24, 944, 700], [0.47, 0.35, 0.31]))
    plan = {"layout": "track", "src_w": SW, "src_h": SH,
            "segments": [{"t0": 0.0, "t1": 5.0, "x": 438, "cut": True}]}
    a = recompose.parse_args(["s.mp4", "0", "5", "o.mp4"])
    with pytest.raises(recompose.RenderError, match="slice a source banner"):
        recompose.guard_banners(plan, a, "s.mp4")
    assert plan["layout"] == "track"


def test_cam_crop_is_trimmed_off_text_but_never_through_the_face():
    face = [0.977, 0.826, 0.133]
    got = recompose._cam_clear_of_text([170, 220, 1110, 500, False], [[34, 706, 1280, 720]],
                                       face, SW, SH)
    assert got is not None and got[3] + got[1] <= 698 and got[4] is True
    assert recompose._cam_clear_of_text([170, 220, 1110, 500, False], [[0, 580, 1280, 610]],
                                        face, SW, SH) is None            # across the face
    assert recompose._cam_clear_of_text([170, 220, 1110, 500, True], [[0, 650, 1100, 670]],
                                        face, SW, SH) == [170, 220, 1110, 500, True]   # outside


# ---------------------------------------------------------------- 3. stream layout + blur fill

def test_stream_vertical_renders_gamestack(monkeypatch):
    assert recompose._vertical_layouts()["stream"] == "gamestack"
    seen = {}
    monkeypatch.setattr(recompose, "_plan_gamestack",
                        lambda src, a: seen.setdefault("plan", {"layout": "gamestack"}))
    a = recompose.parse_args(["s.mp4", "0", "5", "o.mp4", "--vertical", "stream"])
    assert recompose.plan_layout(a, "s.mp4")["layout"] == "gamestack" and seen


def test_blur_fill_share_by_layout():
    a = recompose.parse_args(["s.mp4", "0", "5", "o.mp4"])
    letterbox = recompose.blur_fill_frac({"layout": "general", "src_w": SW, "src_h": SH}, a, {})
    assert letterbox == pytest.approx(1 - 608 / 1920, abs=0.002)        # 68 % blur
    track = {"layout": "track", "src_w": SW, "src_h": SH, "segments": [{"t0": 0, "t1": 1, "x": 0}]}
    assert recompose.blur_fill_frac(track, a, {}) == 0.0
    gs = {"layout": "gamestack", "src_w": SW, "src_h": SH, "top_h": 1344,
          "game": [578, 380, False], "cam": [180, 96, 28, 66, False]}
    assert recompose.blur_fill_frac(gs, a, {}) == 0.0
    gs_fit = dict(gs, game=[700, 300, True])
    assert 0 < recompose.blur_fill_frac(gs_fit, a, {}) < 0.5


# ---------------------------------------------------------------- encode wiring

@pytest.fixture(scope="module")
def media(tmp_path_factory):
    p = tmp_path_factory.mktemp("hook") / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=640x360:rate=30:duration=3", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=3", "-shortest", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(p)], check=True)
    return p


def test_hook_on_text_rejects_the_render(media, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(textzones, "hook_zone_text",
                        lambda *a, **k: {"bands": [(0, PH)], "frames": 7, "ocr": True,
                                         "words": ["$7,811.350"]})
    out = tmp_path / "clip.mp4"
    rc = recompose.main([str(media), "0", "2", str(out), "--layout", "general",
                         "--hook", "Why I bet against the crowd"])
    assert rc == 1 and not out.exists()
    assert "hook placement" in capsys.readouterr().err


def test_qc_reports_blur_fill_and_hook_bands(media, tmp_path, capsys):
    out = tmp_path / "clip.mp4"
    rc = recompose.main([str(media), "0", "2", str(out), "--layout", "general",
                         "--hook", "Big hook"])
    assert rc == 0
    rep = json.loads(capsys.readouterr().out.strip().splitlines()[-1][3:])
    assert rep["blur_fill_frac"] == pytest.approx(0.684, abs=0.002)
    assert "hook_text_bands" in rep


# ---------------------------------------------------------------- dark stream levels

def test_auto_levels_lifts_a_dark_window_mildly_and_leaves_a_bright_one():
    rng = np.random.default_rng(3)
    dark = [np.clip(rng.normal(30, 18, (90, 160)), 0, 140).astype(np.uint8) for _ in range(8)]
    lv = recompose.auto_levels(dark)
    assert lv and lv["src_luma"] < recompose.DARK_LUMA
    assert lv["white"] >= 255 / recompose.LEVELS_MAX_GAIN - 0.01        # gain capped at 1.5
    assert lv["black"] <= recompose.LEVELS_MAX_BLACK
    assert lv["filter"].startswith("colorlevels=rimin=") and "eq=gamma=1.15" in lv["filter"]
    bright = [np.clip(rng.normal(120, 40, (90, 160)), 0, 255).astype(np.uint8) for _ in range(8)]
    assert recompose.auto_levels(bright) is None
    assert recompose.auto_levels([np.zeros((90, 160), np.uint8)] * 4) is None   # black: nothing
    assert recompose.auto_levels([]) is None
    lifted = [np.clip(rng.normal(30, 15, (90, 160)), 16, 200).astype(np.uint8) for _ in range(8)]
    lv2 = recompose.auto_levels(lifted)                    # black point 16, little highlight
    assert 255 / (lv2["white"] - lv2["black"]) <= recompose.LEVELS_MAX_GAIN + 1e-6


def test_levels_feed_every_use_of_the_source_in_the_graph():
    one = recompose.with_levels("[0:v]crop=404:ih:0:0,scale=1080:1920[v0]", "colorlevels=rimax=0.6")
    assert one == "[0:v]colorlevels=rimax=0.6[lv0];[lv0]crop=404:ih:0:0,scale=1080:1920[v0]"
    two = recompose.with_levels(recompose.stack_vf((10, 10, 0, 0), (10, 10, 0, 0), 960), "eq=gamma=1.1")
    assert two.startswith("[0:v]eq=gamma=1.1,split=2[lv0][lv1];") and "[0:v]" not in two.split(";", 1)[1]
    assert recompose.with_levels("[0:v]null[v0]", None) == "[0:v]null[v0]"


def test_levels_only_for_stream_footage(monkeypatch):
    dark = np.full((90, 160, 3), 25, np.uint8)
    dark[:10] = 200
    monkeypatch.setattr(recompose.reframe, "iter_frames", lambda *a, **k: iter([(0.0, dark)] * 4))
    a = recompose.parse_args(["s.mp4", "0", "5", "o.mp4", "--vertical", "stream"])
    assert recompose.measure_levels(a, "s.mp4")["src_luma"] < recompose.DARK_LUMA
    a2 = recompose.parse_args(["s.mp4", "0", "5", "o.mp4", "--vertical", "podcast"])
    assert recompose.measure_levels(a2, "s.mp4") is None
