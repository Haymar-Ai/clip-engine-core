"""Render yield 11-09 (spec 2026-11-09-render-yield-reject-classes): the four eye-check reject
classes the 11-09 01:55 prep kept seeing, fixed in the rendered pixels.

  1 black_band  a daily gaming / streamers clip whose planned layout slices: the streamer split
                (the game window over the facecam's own box, both filling the frame) instead of
                the dual view's blurred header; no clean split: rejected, never a band
  2 edge_slice  a static corner overlay (a network's logo bug) is whole in a crop or out of it;
                a cut no crop avoids is painted out (the part the crop shows), never left partial
  3 edge_slice  burned-in source text per shot: a window that holds it whole or leaves it out,
                else the frame fit to width for that shot, else cropcheck.TEXT_SLICE
  4 wrong_crop  a tech screen with a streamer's facecam: the split, the face whole in its panel

Offline: synthetic frames, scans and samples; one ffmpeg pass over a synthetic noise source.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

import cv2                  # noqa: E402
from clipengine.render import cleancrop
from clipengine.render import cropcheck
from clipengine.render import overlays
from clipengine import quality_gate as qg
from clipengine.render import recompose
from clipengine.render import textzones

W, H = 1280, 720


def _args(niche, vertical="general", dur=20.0, **kw):
    attrs = dict(start=0.0, dur=dur, niche=niche, vertical=vertical, content_kind=None,
                 clean_first=False)
    attrs.update(kw)
    return type("A", (), attrs)()


def _face(cx, cy, h=0.2, w=0.12, t=1.0):
    return (t, [{"cx": cx, "cy": cy, "w": w, "h": h, "score": 0.9}])


def _scan(graphics=(), faces=None, panels=()):
    return {"src_w": W, "src_h": H, "frames": 10,
            "graphics": [{"box": list(b), "frac": 0.9, "banner": ban} for b, ban in graphics],
            "panels": [list(p) for p in panels], "dead_bands": {},
            "faces": faces if faces is not None else [_face(0.5, 0.35)]}


def _track(xs=(437,), dur=20.0):
    n = len(xs)
    return {"layout": "track", "src_w": W, "src_h": H,
            "segments": [{"t0": i * dur / n, "t1": (i + 1) * dur / n, "x": x}
                         for i, x in enumerate(xs)]}


def _win(v):
    return [v["x"], v["y"], v["x"] + v["w"], v["y"] + v["h"]]


@pytest.fixture
def offline(monkeypatch):
    """recompose's scan, OCR, window scan and overlay scan replaced by the test's own."""
    def setup(sc, ovs=(), wins=None):
        recompose._SCANS.clear()
        recompose._CAMS.clear()
        monkeypatch.setattr(recompose, "_scan", lambda src, a, size: sc)
        monkeypatch.setattr(qg, "ocr_source_text", lambda *a, **k: None)
        monkeypatch.setattr(cleancrop, "window_boxes", lambda *a, **k: wins)
        monkeypatch.setattr(overlays, "scan", lambda *a, **k: [{"box": list(b)} for b in ovs])
        monkeypatch.setattr(recompose.reframe, "iter_frames", lambda *a, **k: iter(()))
        monkeypatch.setattr(recompose.reframe, "probe_size", lambda src: (W, H))
    return setup


# ---------------------------------------------------------------- class 2: logo bugs

def _logo_frames(n=12, seed=4):
    """640x360 gray frames: moving noise everywhere, a still 'PLAY' bug at the top right and a
    still word in the middle of the frame."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        g = cv2.GaussianBlur(rng.integers(0, 255, (360, 640)).astype(np.uint8), (0, 0), 3)
        cv2.putText(g, "PLAY", (520, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(g, "MID", (290, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        out.append(g)
    return out


def test_a_still_corner_logo_is_found_and_still_text_in_the_middle_is_not():
    found = overlays.find(_logo_frames(), W, H)
    assert len(found) == 1 and found[0]["corner"] == "top_right"
    x0, y0, x1, y1 = found[0]["box"]
    (tw, th), base = cv2.getTextSize("PLAY", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    bug = [2 * 520, 2 * (24 - th), 2 * (520 + tw), 2 * (24 + base)]       # src px
    assert x0 <= bug[0] + 4 and y0 <= bug[1] + 4 and x1 >= bug[2] - 4 and y1 >= bug[3] - 8
    assert x0 > W / 2 and y1 < 0.2 * H


def test_moving_content_holds_no_overlay():
    rng = np.random.default_rng(1)
    frames = [cv2.GaussianBlur(rng.integers(0, 255, (360, 640)).astype(np.uint8), (0, 0), 3)
              for _ in range(10)]
    assert overlays.find(frames, W, H) == []
    assert overlays.find(_logo_frames()[:3], W, H) == []          # too few frames: no verdict


LOGO = [1000, 22, 1262, 76]       # the Playmaker bug's box at the guest tile's top right (2889)


def test_the_speaker_crop_never_cuts_a_corner_logo(offline):
    """The planned crop (x 850) runs through the bug; the recrop holds it whole."""
    plan = _track(xs=(850,))
    sc = _scan(faces=[_face(0.82, 0.45, h=0.2)])
    offline(sc, ovs=[LOGO])
    a = _args("podcasts", "podcast")
    assert overlays.cut(recompose.crop_views(plan, a.dur), [LOGO])
    recompose.slice_ladder(plan, a, "src.mp4", {"action": "none"})
    recompose.cover_overlays(plan, a, "src.mp4")
    assert overlays.cut(recompose.crop_views(plan, a.dur), [LOGO]) == []
    assert "cover" not in plan


def test_a_logo_wider_than_the_speakers_tile_is_left_out_above_the_crop(offline):
    """The bug reaches over the tile's border: no crop in the tile holds it, one under it does."""
    plan = _track(xs=(850,))
    sc = _scan(faces=[_face(0.82, 0.45, h=0.2)])
    offline(sc, ovs=[LOGO], wins=[[870, 20, 1240, 650]])
    a = _args("podcasts", "podcast")
    recompose.slice_ladder(plan, a, "src.mp4", {"action": "none"})
    views = recompose.crop_views(plan, a.dur)
    assert overlays.cut(views, [LOGO]) == [] and all(v["y"] >= LOGO[3] - 3 for v in views)
    assert all(870 <= v["x"] and v["x"] + v["w"] <= 1240 for v in views)


def test_a_logo_no_crop_can_avoid_is_painted_out_where_the_crop_shows_it(offline):
    """A bug wider than any crop beside a speaker whose head sits at its height: no window holds
    it whole or leaves it out, so the crop keeps the speaker and the part of the bug it shows is
    covered (delogo), never a partial logo on the canvas."""
    bug = [700, 22, 1020, 70]
    plan = _track(xs=(850,))
    sc = _scan(faces=[_face(0.86, 0.15, h=0.2, w=0.1)])          # head level with the bug
    offline(sc, ovs=[bug])
    a = _args("podcasts", "podcast")
    recompose.slice_ladder(plan, a, "src.mp4", {"action": "none"})
    views = recompose.crop_views(plan, a.dur)
    assert overlays.cut(views, [bug])                            # the crop still cuts it ...
    cov = recompose.cover_overlays(plan, a, "src.mp4")
    assert cov and plan["cover"] == cov                          # ... so its shown part is covered
    for b in cov:
        assert any(b[0] >= v["x"] and b[2] <= v["x"] + v["w"] and b[1] >= v["y"]
                   and b[3] <= v["y"] + v["h"] for v in views)
        assert b[2] == bug[2] and b[1] == bug[1] and b[3] == bug[3]
        assert b[0] == max(bug[0], min(v["x"] for v in views))   # from the crop's edge on
    g = recompose.video_graph(plan, a)
    assert g.startswith("[0:v]delogo=") and g.count("[0:v]") == 1


def test_a_cover_never_paints_over_a_face(offline):
    plan = _track(xs=(850,))
    sc = _scan(faces=[_face(0.9, 0.06, h=0.1, w=0.1)])           # a face inside the bug's box
    offline(sc, ovs=[LOGO])
    plan["crop_box"] = {"w": 405, "h": 680, "y": 40}
    assert recompose.cover_overlays(plan, _args("podcasts"), "src.mp4") == []
    assert "cover" not in plan


def test_cover_boxes_stay_one_px_inside_the_frame_as_delogo_needs():
    f = overlays.cover_filter([[0, 0, 100, 40], [1200, 690, 1280, 720], [10, 10, 12, 11]], W, H)
    parts = f.split(",")
    assert parts[0] == "delogo=x=1:y=1:w=99:h=39"
    assert parts[1] == "delogo=x=1200:y=690:w=79:h=29"
    assert parts[2] == "delogo=x=10:y=10:w=4:h=4"
    g = overlays.cover_graph("[0:v]split=2[a][b];[a]null[c];[b]null[d];[c][d]vstack[v0]",
                             [[10, 10, 50, 30]], W, H)
    assert g.startswith("[0:v]delogo=x=10:y=10:w=40:h=20[cv0];[cv0]split=2[a][b]")


# ---------------------------------------------------------------- class 3: burned-in text per shot

def _text_frames(on=(4, 10), n=20):
    """640x360 gray frames at 2 fps: a wide two-line caption on frames [on[0], on[1])."""
    rng = np.random.default_rng(2)
    out = []
    for i in range(n):
        g = cv2.GaussianBlur(rng.integers(60, 120, (360, 640)).astype(np.uint8), (0, 0), 4)
        if on[0] <= i < on[1]:
            cv2.putText(g, "The work is off your plate", (60, 260), cv2.FONT_HERSHEY_DUPLEX,
                        1.1, 255, 2)
        out.append((i / 2.0, g))
    return out


def test_a_shot_of_burned_in_text_becomes_one_event_in_source_px():
    frames = _text_frames()
    tracks = textzones.track_lines(frames)
    words = lambda g, box: [(0, 0, 10, 10, "work"), (12, 0, 30, 10, "plate")]   # noqa: E731
    at = dict(frames)
    ev = textzones.text_events(tracks, lambda t: at[t], W, H, 640, 10.0, ocr=words)
    assert len(ev) == 1
    e = ev[0]
    assert e["t0"] == pytest.approx(2.0 - textzones.SRC_EVENT_PAD_S)
    assert e["t1"] == pytest.approx(4.5 + textzones.SRC_EVENT_PAD_S)
    (tw, _th), _b = cv2.getTextSize("The work is off your plate", cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
    assert e["box"][0] <= 2 * 60 + 16 and e["box"][2] >= 2 * (60 + tw) - 16
    assert e["box"][2] - e["box"][0] > 405                       # wider than any 9:16 crop


def test_a_line_on_screen_all_window_is_no_shot_and_unread_text_is_none():
    frames = _text_frames(on=(0, 20))
    at = dict(frames)
    words = lambda g, box: [(0, 0, 10, 10, "ticker")]   # noqa: E731
    assert textzones.text_events(textzones.track_lines(frames), lambda t: at[t], W, H, 640,
                                 10.0, ocr=words) == []
    frames = _text_frames()
    assert textzones.text_events(textzones.track_lines(frames), lambda t: dict(frames)[t], W, H,
                                 640, 10.0, ocr=lambda g, b: []) == []


def test_a_sign_in_the_scene_the_line_detector_reads_now_and_then_is_no_caption():
    """2884: the neon 'NIGHT CAP' behind the host, read as a line on 2 of 34 samples ('WD XL'):
    its box looks the same before and after, so it never came and went; a caption does."""
    sign = _text_frames(on=(0, 20))                              # on screen the whole window
    caption = _text_frames()                                     # on frames 4-9 only
    words = lambda g, box: [(0, 0, 10, 10, "NIGHT"), (12, 0, 30, 10, "CAP")]   # noqa: E731
    for frames, n in ((sign, 0), (caption, 1)):
        at = dict(frames)
        tracks = [max(textzones.track_lines(frames), key=lambda tr: len(tr["frames"]))]
        if frames is sign:                                        # the detector saw it briefly
            tracks[0].update(t0=3.0, t1=3.5,
                             frames=[f for f in tracks[0]["frames"] if 3.0 <= f[0] <= 3.5])
        ev = textzones.text_events(tracks, lambda t: at[t], W, H, 640, 10.0, ocr=words,
                                   times=[t for t, _g in frames])
        assert len(ev) == n


WIDE = [200, 480, 1080, 660]      # 2977: 'The work is off your plate / but the decisions aren't'


def _text_fix(monkeypatch, events, faces, dur=20.0):
    monkeypatch.setattr(textzones, "source_text_spans", lambda *a, **k: events)
    monkeypatch.setattr(recompose, "_scan", lambda src, a, size: _scan(faces=faces))
    plan = _track(dur=dur)
    a = _args("lifestyle", dur=dur)
    return plan, a, recompose.shot_text_fix(plan, a, "src.mp4")


def test_text_wider_than_the_crop_is_never_sliced_a_punch_in_leaves_it_out(monkeypatch):
    plan, a, shots = _text_fix(monkeypatch, [{"t0": 2.0, "t1": 5.0, "box": WIDE, "text": "w"}],
                               [_face(0.5, 0.3, t=float(t)) for t in range(20)])
    assert shots and shots[0]["mode"] == "crop"
    s = shots[0]
    assert s["h"] < H and s["h"] >= cleancrop.min_crop_h(H) and s["y"] + s["h"] <= WIDE[1] + 3
    assert abs(s["w"] / s["h"] - 9 / 16) < 0.01                  # full-bleed 9:16, no band
    views = recompose.crop_views(plan, a.dur)
    assert not any(cropcheck.cuts(_win(v), WIDE) for v in views if v["t1"] > 2.0 and v["t0"] < 5.0)
    assert any(v["t0"] == 0.0 and v["h"] == H for v in views)    # the planned crop elsewhere
    g = recompose.video_graph(plan, a)
    assert "enable='between(t,2.000,5.000)'" in g and "gblur" not in g
    assert recompose.blur_fill_frac(plan, a, {"width": W, "height": H}) == 0.0


def test_text_over_the_speaker_is_fit_to_width_for_its_shot(monkeypatch):
    """No 9:16 window keeps the face whole and the text whole or out: the frame fit to width
    for the shot, nothing sliced, its blurred fill counted for its span only."""
    faces = [_face(0.5, 0.78, h=0.25, t=float(t)) for t in range(20)]   # the face in the text rows
    plan, a, shots = _text_fix(monkeypatch, [{"t0": 2.0, "t1": 5.0, "box": WIDE, "text": "w"}],
                               faces)
    assert [s["mode"] for s in shots] == ["fit"]
    views = recompose.crop_views(plan, a.dur)
    assert not any(v["t1"] > 2.0 and v["t0"] < 5.0 for v in views)   # the whole frame shows
    f = recompose.blur_fill_frac(plan, a, {"width": W, "height": H})
    assert 0 < f == pytest.approx(0.684 * 3.0 / 20.0, abs=0.01)
    assert "split=2[tf0b][tf0f]" in recompose.video_graph(plan, a)


def test_text_shots_too_long_to_fit_fail_cropcheck_with_its_code(monkeypatch):
    faces = [_face(0.5, 0.78, h=0.25, t=float(t)) for t in range(20)]
    ev = [{"t0": 0.0, "t1": 9.9, "box": WIDE, "text": "a"},
          {"t0": 10.0, "t1": 19.9, "box": WIDE, "text": "b"}]
    with pytest.raises(recompose.RenderError, match=cropcheck.TEXT_SLICE):
        _text_fix(monkeypatch, ev, faces)
    assert stages_reads_it_as_a_slice(cropcheck.text_slice_error("x"))


def stages_reads_it_as_a_slice(msg):
    import re
    return bool(re.search(r"slic|blurred fill", msg, re.I))      # = engine._SLICE_RENDER


def test_text_the_crop_does_not_cut_changes_nothing(monkeypatch):
    plan, a, shots = _text_fix(monkeypatch, [{"t0": 2.0, "t1": 5.0, "box": [20, 40, 180, 80],
                                              "text": "a"}], [_face(0.5, 0.3)])
    assert shots is None and "text_shots" not in plan


# ---------------------------------------------------------------- class 1: the streamer split

CAM = [2, 118, 258, 314]                      # 2992's facecam box (its tile)
CAM_FACE = [0.1, 0.28, 0.09]
STREAM_PLATES = [[900, 554, 1280, 720],       # the speedometer
                 [0, 326, 358, 488],          # the chat under the cam
                 [1068, 190, 1280, 250]]      # the map's location plate


@pytest.fixture
def split(monkeypatch):
    """plan_streamer_split with the cam, samples, plates and activity replaced."""
    def run(a, cam=CAM, face=CAM_FACE, plates=STREAM_PLATES, tile=True, holds=True):
        recompose._SCANS.clear()
        recompose._CAMS.clear()
        monkeypatch.setattr(recompose, "_cam_candidate",
                            lambda src, a, prior, bh: {"cam_box": list(cam), "face_median": face})
        ov = {"face": {"cx": face[0], "cy": face[1], "h": face[2]},
              "tile": list(cam) if tile else None}
        monkeypatch.setattr(recompose, "cam_overlay", lambda src, a, size: ov)
        samples = [{"t": t / 2.0, "faces": [{"cx": face[0], "cy": face[1], "w": 0.05,
                                             "h": face[2]}] if holds else []} for t in range(40)]
        monkeypatch.setattr(recompose, "_dual_view_samples",
                            lambda src, a, sw, sh: (samples, [], [], None))
        monkeypatch.setattr(recompose, "_scan", lambda src, a, size: _scan(faces=[]))
        monkeypatch.setattr(qg, "ocr_source_text", lambda *a, **k: None)
        monkeypatch.setattr(recompose, "keep_plates",
                            lambda sc, words, sw, sh, pad=0, min_frac=0.0:
                            [{"box": list(p), "frac": 1.0} for p in plates])
        monkeypatch.setattr(overlays, "scan", lambda *a, **k: [])
        monkeypatch.setattr(recompose.facecam, "activity_centroid", lambda *a, **k: 640.0)
        return recompose.plan_streamer_split("src.mp4", a, prior={"src_w": W, "src_h": H})
    return run


def test_the_split_fills_both_panels_and_slices_nothing(split):
    p = split(_args("streamers", "stream"))
    assert p["layout"] == "gamestack" and p["streamer_split"]
    gw, gh, gx, gy = p["game_box"]
    game = [gx, gy, gx + gw, gy + gh]
    assert abs(gw / gh - recompose.OUT_W / p["top_h"]) < 0.01      # fills its panel: no fill
    assert not any(cropcheck.cuts(game, b) for b in STREAM_PLATES)
    assert game[0] >= CAM[2] or game[2] <= CAM[0]                  # the cam stays out of it
    assert gx <= W / 2 <= gx + gw                                   # the frame's centre (the car)
    cw, ch, cx, cy, fit = p["cam"]
    assert not fit and CAM[0] <= cx and cx + cw <= CAM[2] and CAM[1] <= cy and cy + ch <= CAM[3]
    assert abs(cw / ch - recompose.OUT_W / (recompose.OUT_H - p["top_h"])) < 0.02
    assert recompose.blur_fill_frac(p, _args("streamers"), {}) == 0.0
    g = recompose.video_graph(p, _args("streamers"))
    assert "gblur" not in g and f"crop={gw}:{gh}:{gx}:{gy}" in g


def test_no_split_without_a_facecam_that_holds_every_shot(split):
    assert split(_args("streamers", "stream"), holds=False) is None
    assert split(_args("streamers", "stream"), face=[0.1, 0.28, 0.3]) is None   # a camera shot


def _canvas_rows_std(graph, n=3):
    """Per-row luma std of `graph`'s canvas over a 1280x720 noise source: (n, 1920) array."""
    rng = np.random.default_rng(7)
    src = rng.integers(0, 255, (n, H, W, 3)).astype(np.uint8).tobytes()
    cmd = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", "10", "-i", "-", "-filter_complex", graph + ";[v0]format=gray[o]", "-map", "[o]",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    out = subprocess.run(cmd, input=src, capture_output=True, check=True).stdout
    k = recompose.OUT_W * recompose.OUT_H
    frames = np.frombuffer(out[:len(out) // k * k], np.uint8).reshape(-1, recompose.OUT_H,
                                                                      recompose.OUT_W)
    return frames.astype(np.float32).std(axis=2)


def test_the_split_canvas_has_no_band_rows_the_dual_view_does(split):
    """Row-variance check on the rendered canvas: every row of the split shows picture (noise
    keeps its spread); the dual view's blurred header rows are flat."""
    split_rows = _canvas_rows_std(recompose.gamestack_vf(split(_args("streamers", "stream"))))
    assert split_rows.min() > 20.0
    hh, ph, seam = recompose.dual_view_geometry(W, H)
    dv = {"layout": "dual_view", "src_w": W, "src_h": H, "header_h": hh, "panel_h": ph,
          "seam_y": seam, "bottom": "cam", "zoom": {"w": 250, "h": 190, "fit": "inside"},
          "zoom_segments": [{"t0": 0.0, "t1": 1.0, "x": 4, "y": 120, "cut": True}]}
    dv_rows = _canvas_rows_std(recompose.dual_view_vf(dv))
    assert (dv_rows[:, 20:hh - 20] < 5.0).all()                 # the band the eye-check reason_codes


def test_a_daily_streamer_clip_never_gets_the_dual_view(monkeypatch, offline):
    """A gamestack that slices text: the streamer split takes its place; with no split the clip is
    rejected before the encode (the planned slice does not ship, the dual view is not made)."""
    gs = {"layout": "gamestack", "src_w": W, "src_h": H, "top_h": 1344,
          "game": [578, 300, False], "cam": [246, 130, 0, 154, False], "cam_box": CAM}
    offline(_scan(graphics=[((250, 650, 700, 690), True)]))
    monkeypatch.setattr(recompose, "plan_dual_view", lambda *a, **k: pytest.fail("no dual view"))
    monkeypatch.setattr(recompose, "plan_streamer_split", lambda src, a, prior=None: None)
    with pytest.raises(recompose.RenderError, match="no clean streamer split"):
        recompose.hud_layout(dict(gs), _args("streamers", "stream"), "src.mp4", None, None)
    sp = dict(gs, streamer_split=True, game_box=[700, 720, 580, 0])
    monkeypatch.setattr(recompose, "plan_streamer_split", lambda src, a, prior=None: dict(sp))
    plan = dict(gs)
    rep = recompose.hud_layout(plan, _args("streamers", "stream"), "src.mp4", None, None)
    assert rep["action"] == "streamer_split" and plan["streamer_split"]


def test_a_daily_gaming_crop_is_never_boxed_over_a_blur(monkeypatch, offline):
    """472's board graphic, no clean recrop, no facecam: rejected, no c_box, no dual view."""
    sc = _scan(graphics=[((700, 400, 1000, 440), False)])
    sc["graphics"][0]["frac"] = 0.3
    offline(sc)
    monkeypatch.setattr(recompose, "plan_dual_view", lambda *a, **k: pytest.fail("no dual view"))
    monkeypatch.setattr(recompose, "plan_streamer_split", lambda src, a, prior=None: None)
    with pytest.raises(recompose.RenderError, match="a_recrop"):
        recompose.hud_layout(_track(), _args("gaming", "stream"), "src.mp4", {"action": "none"},
                             None)


def test_gameplay_without_a_cam_gets_a_full_bleed_window_over_the_action(monkeypatch, offline):
    """No speaker and no facecam (the planner's general layout: 68 % blurred fill): a full-bleed
    9:16 window near the action, every HUD plate whole or out, instead of the dual view."""
    plates = [[0, 600, 330, 720], [1060, 20, 1270, 180]]          # a health bar, a minimap
    offline(_scan(faces=[]))
    monkeypatch.setattr(recompose, "keep_plates", lambda sc, words, sw, sh, pad=0, min_frac=0.0:
                        [{"box": list(p), "frac": 1.0} for p in plates])
    monkeypatch.setattr(recompose.facecam, "activity_centroid", lambda *a, **k: 1000.0)
    monkeypatch.setattr(recompose, "plan_dual_view", lambda *a, **k: pytest.fail("no dual view"))
    monkeypatch.setattr(recompose, "plan_streamer_split", lambda src, a, prior=None: None)
    plan = {"layout": "general"}
    rep = recompose.hud_layout(plan, _args("gaming", "stream"), "src.mp4", None, None)
    assert rep["action"] == "full_bleed_action" and plan["layout"] == "track"
    views = recompose.crop_views(plan, 20.0)
    assert views and not any(cropcheck.cuts(_win(v), p) for v in views for p in plates)
    assert recompose.blur_fill_frac(plan, _args("gaming"), {"width": W, "height": H}) == 0.0


# ---------------------------------------------------------------- class 4: a screen with a facecam

TECH_CAM = [1058, 468, 1270, 682]             # 2876: the streamer keyed over the client's corner
TECH_FACE = [0.908, 0.75, 0.094]
TECH_PLATES = [[738, 622, 1076, 720],          # the client's toolbar
               [0, 552, 544, 720],             # the chat box
               [0, 0, 247, 347]]               # the XP trackers


def test_a_tech_screen_with_a_facecam_keeps_the_face_whole_in_its_panel(split, monkeypatch):
    a = _args("tech", content_kind="screen")
    monkeypatch.setattr(recompose, "_scan", lambda src, a, size: _scan(
        faces=[_face(TECH_FACE[0], TECH_FACE[1], h=TECH_FACE[2], w=0.06, t=float(t))
               for t in range(20)]))
    p = split(a, cam=TECH_CAM, face=TECH_FACE, plates=TECH_PLATES, tile=False)
    assert p and p["streamer_split"]
    cw, ch, cx, cy, _fit = p["cam"]
    keep = recompose._face_keep(TECH_FACE, W, H)
    assert cx <= keep[0] and cx + cw >= keep[2] and cy <= keep[1] and cy + ch >= keep[3]
    assert not any(cropcheck.cuts([cx, cy, cx + cw, cy + ch], b) for b in TECH_PLATES)
    gw, gh, gx, gy = p["game_box"]
    assert not any(cropcheck.cuts([gx, gy, gx + gw, gy + gh], b) for b in TECH_PLATES)
    recompose.enforce_single_view(p, a)                          # the split ships for tech
    views = recompose.crop_views(p, a.dur)
    fx, fy = TECH_FACE[0] * W, TECH_FACE[1] * H
    assert any(v["x"] <= fx <= v["x"] + v["w"] and v["y"] <= fy <= v["y"] + v["h"] for v in views)


def test_the_screen_split_replaces_the_tech_single_view_only_with_a_facecam(monkeypatch, offline):
    sc = _scan(faces=[_face(0.9, 0.75, h=0.09, w=0.06, t=float(t)) for t in range(20)])
    offline(sc)
    plan = _track(xs=(876,))
    a = _args("tech", content_kind="screen")
    monkeypatch.setattr(recompose, "cam_overlay", lambda src, a, size: None)
    assert recompose.screen_cam_split(plan, a, "src.mp4") is None and plan["layout"] == "track"
    monkeypatch.setattr(recompose, "cam_overlay", lambda src, a, size: {
        "face": {"cx": 0.9, "cy": 0.75, "h": 0.09}, "tile": None})
    sp = {"layout": "gamestack", "streamer_split": True, "src_w": W, "src_h": H, "top_h": 1056,
          "game": [542, 298, False], "game_box": [542, 530, 298, 22],
          "cam": [188, 150, 1074, 480, False]}
    monkeypatch.setattr(recompose, "plan_streamer_split", lambda src, a, prior=None: dict(sp))
    rep = recompose.screen_cam_split(plan, a, "src.mp4")
    assert rep["action"] == "streamer_split" and plan["layout"] == "gamestack"
    finance = _args("finance", content_kind="screen")
    assert recompose.screen_cam_split(_track(), finance, "src.mp4") is None   # charts fit to width
    with pytest.raises(recompose.RenderError, match="single view"):
        recompose.enforce_single_view(dict(sp), finance)
