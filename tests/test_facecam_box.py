"""a stream's facecam is shown below the game only inside its own box. 10-26 02:07
run: 7 of 16 gaming eye-check reason_codes were edge_slice, every one a stack cam panel or face zoom
that reached past the cam into the game (1396 'CAMERA' cut to 'ERA', 1390 / 1403 dialogue
choices, 1551 a stats bar and a countdown). The facecam is the small face that recurs at one
place (cropcheck.cam_face); its box is read as a boxed tile (persistent borders). The probe counts
a frame that shows the cam with its box read as clean; the planner bounds the cam panel to that
box and drops, before the encode, a plan whose window shows the cam with the game around it.
An unread box is no proof of a facecam (posted IRL streamers 1017 / 1019 / 1293, posted gaming
recrops of a corner cam 1400 / 1405): those plan as before. Offline: synthetic faces and tiles."""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

from clipengine.render import cropcheck
from clipengine.render import recompose

W, H = 1280, 720
CAM = [3, 3, 257, 193]                 # 10-26 1390 / 1403: the cam in the top-left corner


def _f(cx, cy, h, w=None):
    return {"cx": cx, "cy": cy, "w": w if w is not None else h * 0.5, "h": h, "score": 0.9}


def _cam_frames(n=10, every=1, other=None):
    """n sampled frames, the cam face top left on every `every`-th one, `other` faces too."""
    out = []
    for i in range(n):
        fs = [_f(0.1 + 0.002 * (i % 3), 0.2, 0.11)] if i % every == 0 else []
        if other:
            fs.append(other(i))
        out.append((float(i), fs))
    return out


def _args(niche, *extra):
    return recompose.parse_args(["src.mp4", "0", "10", "o.mp4", "--niche", niche, *extra])


def _ov(tile=CAM):
    return {"face": {"cx": 0.1, "cy": 0.2, "h": 0.11, "share": 1.0}, "tile": tile,
            "read": 1.0 if tile else 0.0}


# ---------------------------------------------------------------- cropcheck: the facecam

def test_the_facecam_is_the_small_face_that_stays_put_not_a_game_character():
    walking = lambda i: _f(0.3 + 0.05 * i, 0.5, 0.15)          # noqa: E731 - a character moves
    cf = cropcheck.cam_face(_cam_frames(other=walking))
    assert cf and abs(cf["cx"] - 0.1) < 0.01 and cf["share"] == 1.0
    assert cropcheck.cam_face([(float(i), [walking(i)]) for i in range(10)]) is None
    assert cropcheck.cam_face(_cam_frames(every=3)) is None     # on 4 of 10 frames: no cam
    big = [(float(i), [_f(0.5, 0.45, 0.4)]) for i in range(10)]
    assert cropcheck.cam_face(big) is None                      # a talking head is no facecam


def test_the_cam_box_is_the_tile_read_around_its_face_and_never_a_scene_sized_one():
    frames = _cam_frames()
    ov = cropcheck.cam_overlay(frames, lambda box: [0, 0, 260, 196], W, H)
    assert ov["tile"] == [3, 3, 257, 193] and ov["read"] == 1.0
    tall = cropcheck.cam_overlay(frames, lambda box: [0, 14, 340, 720], W, H)  # 1396's grown box
    assert tall["tile"] is None and tall["face"]
    assert cropcheck.cam_overlay(frames, None, W, H)["tile"] is None   # too few frames: unread
    once = iter([[0, 0, 260, 196]] + [None] * 9)
    assert cropcheck.cam_overlay(frames, lambda box: next(once), W, H)["tile"] is None
    assert cropcheck.cam_overlay([(0.0, [_f(0.5, 0.45, 0.4)])], None, W, H) is None


def test_a_frame_that_shows_the_cam_is_clean_when_its_box_is_read_else_no_verdict():
    frames = _cam_frames(n=3)
    pls = [{"box": [0, 0, W, H], "frac": 1.0}]                 # nothing else could be clean
    read = cropcheck.frame_verdicts(frames, pls, W, H, cam=_ov())
    assert [(ok, how) for _t, ok, how in read] == [(True, "cam")] * 3
    unread = cropcheck.frame_verdicts(frames, pls, W, H, cam=_ov(None))
    assert [ok for _t, ok, _h in unread] == [None] * 3          # the render decides, as before
    assert [ok for _t, ok, _h in cropcheck.frame_verdicts(frames, pls, W, H)] == [None] * 3


# ---------------------------------------------------------------- the planner

@pytest.fixture
def cam_is(monkeypatch):
    def set_(ov):
        monkeypatch.setattr(recompose, "cam_overlay", lambda src, a, size: ov)
    return set_


def test_a_cam_panel_that_reaches_past_the_facecam_gets_the_cams_own_box(cam_is):
    """1390: the gamestack's busy-pixel box [36, 14, 410, 720] ran down a calm game."""
    cam_is(_ov())
    box, why = recompose.bound_cam([36, 14, 410, 720], _args("gaming"), "src.mp4", (W, H))
    assert box == [4, 4, 256, 192] and "reaches past" in why
    assert recompose.bound_cam([6, 6, 250, 190], _args("gaming"), "src.mp4", (W, H)) == \
        ([6, 6, 250, 190], None)                                # inside the cam's box: as planned
    other = [700, 40, 1270, 360]                                 # another face's box (1032)
    assert recompose.bound_cam(other, _args("gaming"), "src.mp4", (W, H)) == (other, None)
    cam_is(_ov(None))                                            # 1396: no box read, 7 faces tall
    box, why = recompose.bound_cam([0, 14, 340, 720], _args("gaming"), "src.mp4", (W, H))
    assert box is None and "face heights tall" in why
    assert recompose.bound_cam([0, 14, 340, 300], _args("gaming"), "src.mp4", (W, H)) == \
        ([0, 14, 340, 300], None)                               # plausible: as planned
    assert recompose.bound_cam([0, 14, 340, 720], _args("poker"), "src.mp4", (W, H)) == \
        ([0, 14, 340, 720], None)                               # no stack niche: as planned


def test_a_crop_that_shows_the_cam_with_the_game_around_it_is_dropped_before_the_encode(cam_is):
    """1408: a full-bleed recrop of the cam corner (the game's 'CAMERA' label cut to 'ERA')."""
    from clipengine import engine
    cam_is(_ov())
    track = {"layout": "track", "src_w": W, "src_h": H,
             "segments": [{"t0": 0.0, "t1": 10.0, "x": 0}], "crop_box": {"w": 404, "h": 720, "y": 0}}
    with pytest.raises(recompose.RenderError) as e:
        recompose.facecam_view_check(track, _args("gaming"), "src.mp4")
    assert "shows the cam with the game around it" in str(e.value)
    assert engine._SLICE_RENDER.search(str(e.value))           # the window is replaced
    cut = dict(track, segments=[{"t0": 0.0, "t1": 10.0, "x": 150}])
    with pytest.raises(recompose.RenderError, match="cuts the cam"):
        recompose.facecam_view_check(cut, _args("gaming"), "src.mp4")
    clear = dict(track, segments=[{"t0": 0.0, "t1": 10.0, "x": 600}])
    assert recompose.facecam_view_check(clear, _args("gaming"), "src.mp4") is None
    cam_is(_ov(None))                                            # unread: as planned (1400)
    assert recompose.facecam_view_check(track, _args("gaming"), "src.mp4") is None
    cam_is(_ov())
    assert recompose.facecam_view_check(track, _args("podcasts"), "src.mp4") is None
    manual = recompose.parse_args(["src.mp4", "0", "10", "o.mp4"])
    assert recompose.facecam_view_check(track, manual, "src.mp4") is None


def _dv_rig(monkeypatch, cam_is, ov, cam=None):
    cam_is(ov)
    monkeypatch.setattr(recompose, "_cam_candidate", lambda src, a, prior, bh: cam)
    samples = [{"t": t / 2.0, "faces": [_f(0.1, 0.2, 0.11)]} for t in range(20)]
    monkeypatch.setattr(recompose, "_dual_view_samples",
                        lambda src, a, sw, sh: (samples, [], [], None))
    monkeypatch.setattr(recompose, "_zoom_plates", lambda src, a, sw, sh: ([], []))
    monkeypatch.setattr(recompose.reframe, "probe_size", lambda src: (W, H))


def test_the_dual_view_shows_the_facecam_inside_its_own_box(monkeypatch, cam_is):
    """1403: no cam was planned, the zoom followed a game character's face; 1551: the face zoom
    took in the stats bar and the countdown around the cam."""
    _dv_rig(monkeypatch, cam_is, _ov())
    plan = recompose.plan_dual_view("src.mp4", _args("gaming"))
    assert plan["bottom"] == "cam" and plan["cam_box"] == [4, 4, 256, 192]
    z, sg = plan["zoom"], plan["zoom_segments"][0]
    assert CAM[0] <= sg["x"] and sg["x"] + z["w"] <= CAM[2] + 1
    assert CAM[1] <= sg["y"] and sg["y"] + z["h"] <= CAM[3] + 1


def test_a_cam_box_grown_down_a_calm_game_is_cut_to_the_cams_box(monkeypatch, cam_is):
    _dv_rig(monkeypatch, cam_is, _ov(), cam={"cam_box": [0, 14, 340, 720],
                                            "face_median": [0.1, 0.2, 0.11]})
    plan = recompose.plan_dual_view("src.mp4", _args("gaming"))
    assert plan["bottom"] == "cam" and plan["cam_box"] == [4, 4, 256, 192]


def test_a_facecam_that_leaves_some_shots_gets_no_face_zoom_out_of_its_box(monkeypatch, cam_is):
    """1384: the stream cut to the full game for a shot; the face zoom on the cam took in the
    game around it."""
    _dv_rig(monkeypatch, cam_is, _ov())
    samples = [{"t": t / 2.0, "faces": [_f(0.1, 0.2, 0.11)] if t < 10 else []} for t in range(20)]
    monkeypatch.setattr(recompose, "_dual_view_samples",
                        lambda src, a, sw, sh: (samples, [], [5.0], None))
    assert recompose.plan_dual_view("src.mp4", _args("gaming")) is None


def test_an_unread_box_plans_the_dual_view_as_before(monkeypatch, cam_is):
    """1019 / 1293 (posted streamers): a wide IRL shot's small faces recur; no box is read."""
    _dv_rig(monkeypatch, cam_is, _ov(None))
    seen = {}

    def zoom(*a, **k):
        seen["cam"] = k.get("cam")
        return {"zoom": {"w": 568, "h": 432}, "zoom_segments": [
            {"t0": 0.0, "t1": 10.0, "x": 0, "y": 0, "cut": True, "how": "face",
             "faces": [[0.1, 0.2, 0.05, 0.11]]}]}
    monkeypatch.setattr(recompose, "dual_view_zoom_plan", zoom)
    plan = recompose.plan_dual_view("src.mp4", _args("streamers"))
    assert plan["bottom"] == "zoom" and seen["cam"] is None


def test_a_zoom_on_the_action_keeps_every_plate_whole_or_there_is_no_dual_view(monkeypatch,
                                                                               cam_is):
    """891: the motion zoom kept the gate's text items whole and cut the article's lines."""
    _dv_rig(monkeypatch, cam_is, None)
    monkeypatch.setattr(recompose, "_dual_view_samples",
                        lambda src, a, sw, sh: ([{"t": 0.0, "faces": []}], [], [], None))
    lines = [{"box": [20, 40 * i, 1260, 40 * i + 40], "frac": 1.0} for i in range(18)]  # a page
    monkeypatch.setattr(recompose, "_zoom_plates", lambda src, a, sw, sh: (lines, []))
    for how, dv in (("motion", False), ("face", True)):
        monkeypatch.setattr(recompose, "dual_view_zoom_plan", lambda *a, how=how, **k: {
            "zoom": {"w": 396, "h": 302}, "zoom_segments": [
                {"t0": 0.0, "t1": 10.0, "x": 852, "y": 28, "cut": True, "how": how}]})
        assert (recompose.plan_dual_view("src.mp4", _args("gaming")) is not None) is dv
