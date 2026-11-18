"""A stream / gaming game band no longer shows a strip of an in-game phone along its bottom edge.

The phone is a still UI panel docked at the frame's bottom right (src ~[954, 563, 1252, 720], its
lower half under the frame). overlays.find reads only its call card's rows nearest the bottom edge
([964, 623, 1066, 681]): the status bar sits above a gap wider than ROW_GAP, the panel's outline is
taken out as a border line. So the crops kept that box whole or out and still showed the phone:
the card split ended its game window 3 px over the status bar's OCR plate (the phone's bezel on
the band's edge), and a 640 px band held the call card "whole" with the rest of the phone.
overlays.docked_panels gives the panel (its box past the frame, so no window holds it whole), the
card split leaves it out first, and a band that shows it takes the split that leaves it out (else
the band as planned). Synthetic 640x360 scan frames: moving noise, a still phone docked at the
bottom right; the cam, its samples, the banner scan and the OCR are faked."""
import cv2
import numpy as np
import pytest

from clipengine.render import cropcheck
from clipengine.render import overlays
from clipengine.render import recompose

W, H = 1280, 720
PHONE = [954, 564, 1252, 720]        # the phone's body on screen (src px), under the frame below
STATUS = [974, 574, 1060, 602]       # the OCR plate of its status bar's clock
CAM = (4, 120, 254, 310)             # the streamer's room cam, the left edge
STREAM = ["--niche", "streamers", "--content-kind", "talk", "--vertical", "stream"]


def _phone_frames(n=12, seed=3, docked=True):
    """640x360 gray scan frames: moving noise, and at the bottom right a still dark phone (its
    lower edge under the frame unless not `docked`) with a status bar, a call card whose timer
    ticks, a call count and a hang-up button."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        g = cv2.GaussianBlur(rng.integers(0, 255, (360, 640)).astype(np.uint8), (0, 0), 1.2)
        g[282:360 if docked else 340, 477:626] = 30
        cv2.putText(g, "1:20 PM", (490, 296), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 230, 1)
        g[290:296, 590:612] = 230
        cv2.putText(g, "alex", (500, 324), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 240, 1)
        cv2.putText(g, f"02:{20 + i:02d}", (500, 338), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 200, 1)
        cv2.putText(g, "1200 Calls", (490, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 200, 1)
        cv2.circle(g, (605, 320), 7, 200, -1)
        out.append(g)
    return out


def _shows(win, box, tol=cropcheck.SLICE_TOL):
    """Whether window [x0, y0, x1, y1] shows any of `box` (more than tol px both ways)."""
    return min(win[2], box[2]) - max(win[0], box[0]) > tol and \
        min(win[3], box[3]) - max(win[1], box[1]) > tol


# ---------------------------------------------------------------- the panel

def test_find_reads_only_the_call_card_and_the_panel_runs_off_the_frame():
    frames = _phone_frames()
    found = overlays.find(frames, W, H)
    assert len(found) == 1 and found[0]["corner"] == "bottom_right"
    card = found[0]["box"]
    assert card[1] > STATUS[3] and card[1] > PHONE[1] + 40     # the root cause: no status bar
    panels = overlays.docked_panels(frames, found, W, H)
    assert len(panels) == 1
    x0, y0, x1, y1 = panels[0]
    assert x0 <= PHONE[0] and y0 <= PHONE[1] - 3 and x1 >= PHONE[2]
    assert x0 >= PHONE[0] - 12 and y0 >= PHONE[1] - 14 and x1 <= PHONE[2] + 12
    assert y1 == H + overlays.OFF_FRAME                         # it runs on under the frame
    # so a window holding the whole screen still cuts it: whole is not on offer
    assert cropcheck.cuts([0, 0, W, H], panels[0])
    assert not cropcheck.cuts([0, 0, W, H], card)


def test_no_docked_panel_for_a_floating_panel_a_bare_logo_or_a_still_scene():
    floating = _phone_frames(docked=False)
    assert overlays.docked_panels(floating, overlays.find(floating, W, H), W, H) == []
    rng = np.random.default_rng(5)
    logo = []
    for _ in range(12):
        g = cv2.GaussianBlur(rng.integers(0, 255, (360, 640)).astype(np.uint8), (0, 0), 1.2)
        cv2.putText(g, "PLAY", (540, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        logo.append(g)
    found = overlays.find(logo, W, H)
    assert found and overlays.docked_panels(logo, found, W, H) == []   # glyphs, no plate
    still = [_phone_frames(n=1)[0]] * 12                      # nothing moves: no panel to read
    assert overlays.docked_panels(still, [{"box": [970, 625, 1076, 711]}], W, H) == []


# ---------------------------------------------------------------- the crops

@pytest.fixture
def rig(monkeypatch):
    """The stream window: the real overlay scan (find, docked_panels) on the phone frames; the
    cam, its samples, the banner scan, the OCR plates (the status bar's clock) and the activity
    faked."""
    recompose._SCANS.clear()
    recompose._CAMS.clear()

    def frames(src, start, dur, fps=1.0, width=640, src_size=None):
        for i, g in enumerate(_phone_frames()):
            yield float(i), cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    monkeypatch.setattr(recompose.reframe, "iter_frames", frames)
    face = (0.1, 0.30, 0.06)
    monkeypatch.setattr(recompose, "_cam_candidate",
                        lambda src, a, prior, bh: {"cam_box": list(CAM), "face_median": face})
    monkeypatch.setattr(recompose, "cam_overlay", lambda src, a, size: {
        "face": {"cx": face[0], "cy": face[1], "h": face[2]}, "tile": list(CAM)})
    monkeypatch.setattr(recompose, "_dual_view_samples", lambda src, a, sw, sh: ([], [], [], None))
    monkeypatch.setattr(recompose, "cam_holds", lambda samples, cuts, dur, box: True)
    sc = {"graphics": [], "panels": [], "frames": 12, "src_w": W, "src_h": H, "faces": []}
    monkeypatch.setattr(recompose, "_scan", lambda src, a, size: sc)
    monkeypatch.setattr(recompose, "_ocr", lambda src, a, size: [])
    monkeypatch.setattr(recompose, "keep_plates",
                        lambda sc, words, sw, sh, pad=0, min_frac=0.0:
                        [{"box": list(STATUS), "frac": 1.0}])
    monkeypatch.setattr(recompose.facecam, "activity_centroid", lambda *a, **k: 700.0)
    monkeypatch.setattr(recompose, "_gamestack_cfg",
                        lambda: {"v2": True, "game_share": 0.50, "cam_max_area": 0.12,
                                 "game_only_share": 0.42, "header_h": 200})
    return monkeypatch


def _window(plan):
    gw, gh, gx, gy = plan["game_box"]
    return [gx, gy, gx + gw, gy + gh]


def test_the_card_split_leaves_the_phone_out(rig):
    """The split's game window kept the phone's text plates whole or out and showed the phone
    itself (its bezel under the status bar plate, or its body cut by the window's side). It now
    leaves the panel out, and still is a clean card split."""
    a = recompose.parse_args(["src.mp4", "0", "28", "o.mp4"] + STREAM)
    p = recompose.plan_streamer_split("src.mp4", a, prior={"src_w": W, "src_h": H})
    assert p and p["streamer_split"] and p["v2"] and p["card"]
    win = _window(p)
    assert not _shows(win, PHONE)
    assert win[0] >= CAM[2] and not cropcheck.cuts(win, STATUS)
    # the phone's own text boxes alone leave the window showing it (the defect)
    old = recompose._split_card_plan(
        "src.mp4", a, W, H, list(CAM), [0.1, 0.30, 0.06], p["raw_cam_box"],
        [list(STATUS)] + recompose._overlays("src.mp4", a, (W, H)), 700.0,
        (W / 2.0, recompose.SPLIT_CENTRE, recompose.SPLIT_CENTRE_TOP))
    assert _shows(_window(old), PHONE)


def _band(x):
    """A planned v3 band 640x720 at x, the card under it."""
    return {"layout": "gamestack", "v2": True, "src_w": W, "src_h": H, "top_h": 1214,
            "game": [640, x, False], "band": {"w": 640, "h": 720, "out_h": 1214, "share": 0.5,
                                              "y": 0, "lo": x, "hi": x, "runs": [[x, x]],
                                              "sliced": False, "scene_text": [],
                                              "widened_from": None},
            "pan": [[0.0, x]], "pan_y": None, "band_y": 200, "header": [0, 200],
            "cam": [210, 184, 50, 130, False], "card": [40, 1414, 532, 466], "card_side": "left",
            "cap_y": 1325, "fg_h": 1275, "cam_box": list(CAM), "raw_cam_box": [0, 94, 280, 336],
            "face_median": [0.1, 0.30, 0.06]}


def test_a_band_that_shows_the_phone_takes_the_split_that_leaves_it_out(rig):
    """The band at x 630 held the call card whole, the rest of the phone with it, and passed every
    slice check. It takes the card split whose window leaves the phone out."""
    rig.setattr(recompose, "shown_defect", lambda plan, a, src: None)
    a = recompose.parse_args(["src.mp4", "0", "18", "o.mp4"] + STREAM)
    plan = _band(630)
    rep = recompose.hud_layout(plan, a, "src.mp4", {"action": "none"}, None)
    assert rep["action"] == "streamer_split" and "UI panel" in rep["why"]
    assert plan["streamer_split"] and plan["card"] and not _shows(_window(plan), PHONE)


def test_the_band_stands_when_no_split_leaves_the_phone_out_or_it_shows_none(rig):
    rig.setattr(recompose, "shown_defect", lambda plan, a, src: None)
    a = recompose.parse_args(["src.mp4", "0", "18", "o.mp4"] + STREAM)
    calls = []

    def split(src, a, prior=None):
        calls.append(prior["band"]["lo"])
        return dict(_band(600), streamer_split=True, game_box=[640, 720, 600, 0])
    rig.setattr(recompose, "plan_streamer_split", split)
    plan = _band(630)
    assert recompose.hud_layout(plan, a, "src.mp4", {"action": "none"}, None) == {"action": "none"}
    assert plan == _band(630) and calls == [630]            # the split showed it too: as planned
    plan = _band(280)                                       # clear of the phone: no split tried
    assert recompose.hud_layout(plan, a, "src.mp4", {"action": "none"}, None) == {"action": "none"}
    assert plan == _band(280) and calls == [630]
    # outside the stream / gaming rule (no vertical): the band as before
    plain = recompose.parse_args(["src.mp4", "0", "18", "o.mp4", "--niche", "streamers"])
    plan = _band(630)
    assert recompose.hud_layout(plan, plain, "src.mp4", {"action": "none"}, None) == \
        {"action": "none"}
    assert plan == _band(630) and calls == [630]


def test_a_window_with_no_phone_has_no_panel_and_the_split_is_as_before(rig):
    """No docked panel read (a still scene, a source with no phone): _docked_panels is [] and the
    card split takes the plates alone, as before."""
    rig.setattr(recompose.overlays, "docked_panels", lambda grays, found, w, h: [])
    a = recompose.parse_args(["src.mp4", "0", "28", "o.mp4"] + STREAM)
    p = recompose.plan_streamer_split("src.mp4", a, prior={"src_w": W, "src_h": H})
    old = recompose._split_card_plan(
        "src.mp4", a, W, H, list(CAM), [0.1, 0.30, 0.06], p["raw_cam_box"],
        [list(STATUS)] + recompose._overlays("src.mp4", a, (W, H)), 700.0,
        (W / 2.0, recompose.SPLIT_CENTRE, recompose.SPLIT_CENTRE_TOP))
    assert recompose._docked_panels("src.mp4", a, (W, H)) == []
    assert p["game_box"] == old["game_box"]
