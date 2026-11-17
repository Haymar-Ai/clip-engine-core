"""A stream / gaming clip whose source has a streamer cam always shows it as an inset in a bottom
corner of the 1080x1920 frame.

  - the inset: the v3 card under the band when it fits, else overlaid on the frame's bottom corner
    (the cam's side, flipped off the tracked action); <= 18 % of the frame, >= 30 % of its width,
    the captions above it;
  - path (a): a mid-frame cam game only takes keeps it as the inset (the band clear of it / with it
    painted out);
  - path (b): no room for the card under the band: the inset over the frame's bottom corner;
  - path (c): gameplay with no cam card (the facecam's box unbound, no cam read in the window):
    the game band plus the cam (the window's, else the source's anywhere in the file);
  - path (d): the streamer split keeps a cam that loses the face in some shots, as the card;
  - a source with no cam still renders game only and the gate passes it;
  - the gate fails a stream / gaming render with no inset from a source with a cam: cam_missing.
Synthetic sources (lavfi: a moving game, a magenta cam overlay with a face blob); the face
detector, the banner scan and the OCR are faked; ffmpeg renders the graphs."""
import json
import subprocess

import numpy as np
import pytest

from clipengine import engine
from clipengine import quality_gate
from clipengine import reason_codes as rejects
from clipengine.render import cropcheck
from clipengine.render import facecam
from clipengine.render import gamelayout as gl
from clipengine.render import recompose

W, H = 1280, 720
GAME = ("gradients=s=1280x720:r=30:speed=0.03:seed=7:c0=0x203060:c1=0x406090:c2=0x305070:"
        "c3=0x20406a")
MAGENTA = "0xff00ff"
CORNER = [0, 176, 248, 316]          # a room cam at the left edge, mid-height
PART_LEFT = [580, 240, 820, 480]     # a mid-frame cam no 0.48 band avoids
CENTRED = [520, 240, 760, 480]


def _source(path, cam=None, face=None, dur=3):
    """A `dur` s 1280x720 synthetic stream: a smooth moving game, a white square (the action)
    moving across its lower rows, and at `cam` a magenta overlay with a skin-coloured face blob."""
    chain = []
    if cam:
        x0, y0, x1, y1 = cam
        chain.append(f"drawbox=x={x0}:y={y0}:w={x1 - x0}:h={y1 - y0}:color={MAGENTA}:t=fill")
    if face:
        cx, cy, fh = face
        chain.append(f"drawbox=x={int(cx - 22)}:y={int(cy - fh / 2)}:w=44:h={int(fh)}:"
                     f"color=0xdca078:t=fill")
    graph = "[0:v][1:v]overlay=x='60+300*t':y=560:shortest=1" + "".join("," + c for c in chain)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", GAME, "-f", "lavfi", "-i",
                    "color=c=white:s=60x60:r=30", "-f", "lavfi", "-i", "sine=frequency=440",
                    "-t", str(dur), "-filter_complex", graph + "[v]", "-map", "[v]", "-map", "2:a",
                    "-pix_fmt", "yuv420p", "-shortest", str(path)], check=True)
    return path


def _face_of(cam):
    x0, y0, x1, y1 = cam
    return ((x0 + x1) / 2.0, y0 + 0.55 * (y1 - y0), 40.0)


def _norm(box):
    return [box[0] / W, box[1] / H, box[2] / W, box[3] / H]


def _det(box, face, frac=1.0):
    """facecam.detect's dict for a cam `box` (src px) with `face` (cx, cy, h) in it."""
    return {"box": _norm(box), "raw_box": _norm(box), "face_frac": frac, "mode": "busy",
            "face_median": [face[0] / W, face[1] / H, face[2] / H], "span": [0.0, 3.0]}


def _rig(monkeypatch, box=None, face=None, window=True, source=None, cfg=None):
    """The detection the clip gets: the profile / busy rule's box (`box`, None: no cam read in the
    window when `window` is False), facecam.detect_source's cam (`source`: (box, face)), an empty
    banner scan / OCR (the gate's slice verdict passes), no facecam box read (bound_cam leaves
    the planned box)."""
    recompose._SCANS.clear()
    recompose._CAMS.clear()
    monkeypatch.setattr(facecam, "get_or_detect",
                        lambda channel, s, start, dur, source_duration=None, cache_dir=None,
                        log=None: (_norm(box), _norm(box), True, 1.0) if box and window
                        else (None, None, False, 0.0))
    monkeypatch.setattr(facecam, "detect", lambda *a, **k: _det(box, face) if box and window
                        else None)
    calls = []

    def detect_source(src, **k):
        calls.append(src)
        return _det(*source) if source else None
    monkeypatch.setattr(facecam, "detect_source", detect_source)
    faces = [face] * 15 if face and window else []
    monkeypatch.setattr(facecam, "_collect_faces", lambda frames, max_face_h=None: (faces, (W, H)))
    sc = {"graphics": [], "panels": [], "frames": 13, "src_w": W, "src_h": H, "faces": []}
    monkeypatch.setattr(recompose, "_scan", lambda src, a, size: sc)
    monkeypatch.setattr(recompose, "_overlays", lambda src, a, size: [])
    monkeypatch.setattr(recompose, "_ocr", lambda src, a, size: [])
    monkeypatch.setattr(recompose, "keep_plates", lambda *a, **k: [])
    monkeypatch.setattr(recompose, "cam_overlay", lambda src, a, size: None)
    monkeypatch.setattr(quality_gate, "crop_slices", lambda *a, **k: [])
    monkeypatch.setattr(quality_gate, "crop_slice_verdict",
                        lambda s: {"passed": True, "crop_share": 0.0, "sliced_share": 0.0})
    gs = {"v2": True, "game_share": 0.50, "cam_max_area": 0.12, "game_only_share": 0.42,
          "header_h": 200}
    gs.update(cfg or {})
    monkeypatch.setattr(recompose, "_gamestack_cfg", lambda: dict(gs))
    monkeypatch.setattr(recompose, "_config", lambda: {"render": {"gamestack": dict(gs)}})
    return calls


GAMEPLAY = ["--layout", "auto", "--vertical", "stream", "--niche", "gaming",
            "--content-kind", "screen"]


def _args(src, extra=GAMEPLAY, dur="3"):
    return recompose.parse_args([str(src), "0", dur, "o.mp4"] + list(extra))


def _no_face_crop(monkeypatch):
    def analyze(*a, **k):
        raise AssertionError("gameplay must not plan a face-follow crop")
    monkeypatch.setattr(recompose.reframe, "analyze", analyze)


def _band_windows(plan, dur=3.0):
    b = plan["band"]
    return [[x, y, x + b["w"], y + b["h"]] for _t0, _t1, x, y in recompose.v2_views(plan, dur)]


def _assert_inset(plan, cam):
    """The plan shows `cam` as an inset in a bottom corner within the inset bounds."""
    assert plan["layout"] == "gamestack" and plan["v2"]
    box = plan["card"]
    assert gl.inset_ok(box), box
    assert gl.inset_corner(box) in ("left", "right")
    assert gl.inset_area(box) <= gl.CAM_INSET_MAX_AREA
    assert box[2] >= gl.CAM_INSET_MIN_W * gl.OUT_W
    cw, ch, cx, cy, _ = plan["cam"]                       # the card is cut from the cam
    x0, y0, x1, y1 = cam
    assert x0 <= cx and cx + cw <= x1 and y0 <= cy and cy + ch <= y1
    assert not gl.caption_overlap(plan["cap_y"], box)      # the captions sit above it


def _render(vf, src, out, dur=3):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-filter_complex", vf,
                    "-map", "[v0]", "-t", str(dur), "-pix_fmt", "yuv420p", str(out)], check=True)
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", "1.5", "-i", str(out), "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True,
                       check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(1920, 1080, 3)


def _magenta(img):
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (np.abs(hsv[..., 0].astype(int) - 150) <= 8) & (hsv[..., 1] > 150) & (hsv[..., 2] > 150)


# ---------------------------------------------------------------- the inset's bounds (pure)

def test_the_v3_card_meets_the_inset_rule():
    g = gl.band_geometry(W, H)
    card = gl.card_size(gl.card_room(g["gh"]))
    for side in ("left", "right"):
        box = gl.corner_box(*card, side)
        assert gl.inset_ok(box) and gl.inset_corner(box) == side
    lay = gl.canvas_layout(g["gh"], card, "left")
    assert gl.inset_ok(lay["card"])


@pytest.mark.parametrize("ar", [0.4, 0.8, 1.0, 1.33, 1.78, 2.4, 4.0])
def test_an_overlaid_inset_keeps_the_bounds_at_any_cam_aspect(ar):
    w, h = gl.overlay_size(ar)
    assert w >= gl.CAM_INSET_MIN_W * gl.OUT_W and w * h <= gl.CAM_INSET_MAX_AREA * 1080 * 1920
    assert gl.INSET_AR_MIN - 0.02 <= w / h <= gl.INSET_AR_MAX + 0.02
    for w_frac in (gl.OVERLAY_MIN_W, gl.OVERLAY_W, 0.9):
        w, h = gl.overlay_size(ar, w_frac)
        box = gl.corner_box(w, h, "right")
        assert gl.inset_ok(box), (ar, w_frac, box)
        assert box[1] + box[3] == gl.OUT_H - gl.INSET_MARGIN


def test_inset_ok_refuses_a_big_a_narrow_or_a_floating_inset():
    assert not gl.inset_ok([40, 1000, 1000, 880])                 # 42 % of the frame
    assert not gl.inset_ok([40, 1700, 300, 180])                  # 28 % of the width
    assert not gl.inset_ok([380, 1400, 400, 225])                 # not in a corner
    assert not gl.inset_ok([40, 800, 400, 225])                   # not at the bottom
    assert not gl.inset_ok(None)
    assert gl.inset_ok([40, 1655, 432, 225]) and gl.inset_ok([608, 1655, 432, 225])


def test_the_inset_moves_off_the_action_and_the_captions_above_it():
    w, h = gl.overlay_size(16 / 9)
    left = gl.corner_box(w, h, "left")
    under = [(t / 5.0, left[0] + 100, left[1] + 60) for t in range(10)]
    box, side, hits = gl.place_inset(w, h, "left", under)
    assert side == "right" and hits == 0 and gl.inset_ok(box)
    both = under + [(t / 5.0, 900, left[1] + 60) for t in range(3)]
    box, side, hits = gl.place_inset(w, h, "left", both)
    assert side == "right" and hits == 3                            # the fewest points under it
    assert gl.place_inset(w, h, "left", [])[1] == "left"            # the cam's own side first
    for cap_y in (1270, 1325, 1500, 1640):
        y = gl.caption_clear_of(cap_y, box)
        assert not gl.caption_overlap(y, box) and y <= cap_y
    assert gl.caption_clear_of(1270, [40, 1700, 400, 180]) == 1270   # already clear: unchanged


# ---------------------------------------------------------------- path (a): a mid-frame cam

def test_path_a_mid_frame_cam_game_only_keeps_the_cam_as_the_inset(tmp_path, monkeypatch):
    """The mid-frame cam no 0.48 band takes (and none may paint out): game only's 0.42 band
    clear of the cam, now with the cam as the inset (was: no card). A real ffmpeg render: the cam
    (magenta) is on the inset in a bottom corner and not on the band."""
    face = _face_of(PART_LEFT)
    src = _source(tmp_path / "s.mp4", PART_LEFT, face)
    _rig(monkeypatch, PART_LEFT, face)
    _no_face_crop(monkeypatch)
    monkeypatch.setattr(recompose, "MID_CAM_MASK_MAX", 0.0)
    plan = recompose.plan_layout(_args(src), src)
    _assert_inset(plan, PART_LEFT)
    assert plan["cam_mid"] and plan["cam_mask"] is None
    assert plan["band"]["share"] < gl.MIN_SHARE                    # game only's band ...
    assert all(w[2] <= PART_LEFT[0] for w in _band_windows(plan))  # ... clear of the cam
    assert "mid-frame cam" in plan["cam_kept"]["why"]
    img = _render(recompose.gamestack_v2_vf(plan), src, tmp_path / "o.mp4")
    mag = _magenta(img)
    kx, ky, kw, kh = plan["card"]
    assert mag[ky + 8:ky + kh - 8, kx + 8:kx + kw - 8].mean() > 0.3
    y0, y1 = plan["band_y"], plan["band_y"] + plan["band"]["out_h"]
    assert mag[y0:min(y1, ky)].mean() < 0.002


def test_path_a_outside_the_rule_game_only_has_no_card_as_before(tmp_path, monkeypatch):
    """A --layout gamestack request with no stream / gaming vertical keeps game only (no card)."""
    face = _face_of(PART_LEFT)
    src = _source(tmp_path / "s.mp4", PART_LEFT, face)
    _rig(monkeypatch, PART_LEFT, face)
    _no_face_crop(monkeypatch)
    monkeypatch.setattr(recompose, "MID_CAM_MASK_MAX", 0.0)
    plan = recompose.plan_layout(_args(src, ["--layout", "gamestack"]), src)
    assert plan["layout"] == "game_only" and plan["card"] is None


# ---------------------------------------------------------------- path (b): no room under the band

def test_path_b_no_room_for_the_card_overlays_the_inset_on_the_bottom_corner(tmp_path,
                                                                            monkeypatch):
    """A card under the band narrower than 30 % of the frame (a 2 % cam_max_area) used to drop the
    cam ("no room for the cam card under the band; falling back"): the inset now goes over the
    frame's bottom corner, the captions above it."""
    face = _face_of(CORNER)
    src = _source(tmp_path / "s.mp4", CORNER, face)
    _rig(monkeypatch, CORNER, face, cfg={"cam_max_area": 0.02})
    _no_face_crop(monkeypatch)
    plan = recompose.plan_layout(_args(src), src)
    _assert_inset(plan, CORNER)
    assert plan["card_over"] is True and plan["card_side"] in ("left", "right")
    assert plan["cam_kept"]["why"].startswith("no room for the cam card")
    assert all(w[0] >= CORNER[2] for w in _band_windows(plan))   # the band clear of the cam
    img = _render(recompose.gamestack_v2_vf(plan), src, tmp_path / "o.mp4")
    kx, ky, kw, kh = plan["card"]
    assert _magenta(img)[ky + 8:ky + kh - 8, kx + 8:kx + kw - 8].mean() > 0.3


def test_path_b_the_overlay_flips_to_the_other_corner_off_the_action(monkeypatch):
    """The overlaid inset sits on the cam's side unless the tracked action is under it there."""
    g = gl.band_geometry(W, H, 0.42, max_h=gl.GAME_ONLY_BAND_MAX_H, min_share=0.40)
    plan = {"layout": "game_only", "v2": True, "src_w": W, "src_h": H,
            "band": {"w": g["gw"], "h": g["ch"], "out_h": 1680, "share": g["share"], "y": 0},
            "pan": [[0.0, 400]], "pan_y": None, "band_y": 200, "header": [0, 200],
            "card": None, "cap_y": 1500, "action": {"samples": []}}
    cam = {"box": CORNER, "guard": CORNER, "face": _face_of(CORNER), "how": "window",
           "face_frac": 1.0}
    k = g["gw"] / 1080.0
    # the action at the band's bottom-left rows (canvas ~ (150, 1800))
    plan["action"]["samples"] = [[t / 5.0, 400 + 150 * k, (1800 - 200) * g["ch"] / 1680.0, 1.0]
                                 for t in range(15)]
    recompose._attach_cam(plan, cam, (W, H), recompose._v2_cfg({"cam_max_area": 0.12}),
                          "a test")
    assert plan["card_over"] and plan["card_side"] == "right" and gl.inset_ok(plan["card"])
    assert gl.covered(plan["card"], gl.band_points(plan)) == 0
    assert not gl.caption_overlap(plan["cap_y"], plan["card"])


# ---------------------------------------------------------------- path (c): gameplay, no cam card

def test_path_c_the_cam_read_in_the_source_not_the_window_is_kept(tmp_path, monkeypatch):
    """No cam read in the clip's window (the busy and calm rules miss a small face): the game band
    alone used to ship (game_only). The source's cam (facecam.detect_source, anywhere in the file)
    is kept as the inset."""
    face = _face_of(CORNER)
    src = _source(tmp_path / "s.mp4", CORNER, face)
    calls = _rig(monkeypatch, CORNER, face, window=False, source=(CORNER, face))
    _no_face_crop(monkeypatch)
    plan = recompose.plan_layout(_args(src), src)
    _assert_inset(plan, CORNER)
    assert plan["cam_how"] == "source" and calls == [str(src)]
    assert all(w[0] >= CORNER[2] for w in _band_windows(plan))
    assert plan["cam_kept"]["why"] == "gameplay with no cam card"


def test_path_c_a_cam_box_bound_cam_refuses_is_kept_from_the_detection(tmp_path, monkeypatch):
    """A snapped box '4.1 face heights tall' that bound_cam refused used to ship the clip game
    only. The detected box is kept as the inset."""
    face = _face_of(CORNER)
    src = _source(tmp_path / "s.mp4", CORNER, face)
    _rig(monkeypatch, CORNER, face)
    _no_face_crop(monkeypatch)
    monkeypatch.setattr(recompose, "bound_cam", lambda box, a, src, size: (None, "too tall"))
    plan = recompose.plan_layout(_args(src), src)
    _assert_inset(plan, CORNER)
    assert "could not be bound" in plan["cam_kept"]["why"] and plan["cam_box"] == CORNER


def test_path_c_a_source_with_no_cam_still_renders_game_only(tmp_path, monkeypatch):
    src = _source(tmp_path / "s.mp4")
    calls = _rig(monkeypatch, None, None, window=False, source=None)
    _no_face_crop(monkeypatch)
    a = _args(src)
    plan = recompose.plan_layout(a, src)
    assert plan["layout"] == "game_only" and plan["card"] is None and calls == [str(src)]
    rep = recompose.keep_cam(plan, a, src)                        # the last check: no cam, no inset
    assert rep["source_cam"] is None and rep["inset"] is None and "no streamer cam" in rep["why"]
    assert plan["layout"] == "game_only"
    assert quality_gate.measure_cam_inset("stream", rep)["passed"] is True


def test_the_source_scan_is_for_gameplay_only(tmp_path, monkeypatch):
    """A talk stream (an IRL vlog) never takes a cam from elsewhere in the file."""
    src = _source(tmp_path / "s.mp4")
    calls = _rig(monkeypatch, None, None, window=False, source=(CORNER, _face_of(CORNER)))
    talk = _args(src, ["--layout", "auto", "--vertical", "stream", "--niche", "streamers",
                       "--content-kind", "talk"])
    assert recompose.stream_cam(src, talk, (W, H)) is None and calls == []
    assert recompose.stream_cam(src, _args(src), (W, H))["how"] == "source"


# ---------------------------------------------------------------- path (d): the streamer split

@pytest.fixture
def split(monkeypatch):
    """plan_streamer_split with the cam, its samples (cam_holds), plates and activity replaced."""
    def run(a, holds=True, cam=(2, 118, 258, 314), face=(0.1, 0.30, 0.06)):
        recompose._SCANS.clear()
        recompose._CAMS.clear()
        monkeypatch.setattr(recompose, "_cam_candidate",
                            lambda src, a, prior, bh: {"cam_box": list(cam), "face_median": face})
        monkeypatch.setattr(recompose, "cam_overlay", lambda src, a, size: {
            "face": {"cx": face[0], "cy": face[1], "h": face[2]}, "tile": list(cam)})
        monkeypatch.setattr(recompose, "_dual_view_samples", lambda src, a, sw, sh: ([], [], [], None))
        monkeypatch.setattr(recompose, "cam_holds", lambda samples, cuts, dur, box: holds)
        sc = {"graphics": [], "panels": [], "frames": 13, "src_w": W, "src_h": H, "faces": []}
        monkeypatch.setattr(recompose, "_scan", lambda src, a, size: sc)
        monkeypatch.setattr(recompose, "_ocr", lambda src, a, size: [])
        monkeypatch.setattr(recompose, "keep_plates",
                            lambda sc, words, sw, sh, pad=0, min_frac=0.0:
                            [{"box": [900, 554, 1280, 720], "frac": 1.0}])
        monkeypatch.setattr(recompose, "_overlays", lambda src, a, size: [])
        monkeypatch.setattr(recompose.facecam, "activity_centroid", lambda *a, **k: 700.0)
        monkeypatch.setattr(recompose, "_gamestack_cfg",
                            lambda: {"v2": True, "game_share": 0.50, "cam_max_area": 0.12,
                                     "game_only_share": 0.42, "header_h": 200})
        return recompose.plan_streamer_split("src.mp4", a, prior={"src_w": W, "src_h": H})
    return run


def test_path_d_a_cam_that_loses_the_face_in_a_shot_is_kept_as_the_card(split):
    """'The cam does not hold a face in every shot' used to refuse the split and the clip. For a
    stream / gaming clip the split keeps the cam (the cam region is the cam, face or
    not) and shows it as the card in a bottom corner, not a full-width panel."""
    a = _args("src.mp4", dur="20")
    p = split(a, holds=False)
    assert p and p["streamer_split"] and p["v2"] and p["layout"] == "gamestack"
    _assert_inset(p, [2, 118, 258, 314])
    gw, gh, gx, gy = p["game_box"]
    assert gx >= 258 and not cropcheck.cuts([gx, gy, gx + gw, gy + gh], [900, 554, 1280, 720])
    assert abs(p["band"]["out_h"] - 1080 * gh / gw) <= 2
    assert p["card"][1] >= p["band_y"] + p["band"]["out_h"] - 2    # the card under the band
    vf = recompose.video_graph(p, a)
    assert f"crop={gw}:{gh}:'{gx}':'{gy}'" in vf and "[card]" in vf


def test_path_d_outside_the_rule_the_split_is_refused_as_before(split):
    """A render with no stream / gaming vertical keeps the old rule (no face in a shot: no split)
    and the panel split's shape."""
    a = recompose.parse_args(["src.mp4", "0", "20", "o.mp4", "--niche", "streamers"])
    assert split(a, holds=False) is None
    p = split(a, holds=True)
    assert p and p["streamer_split"] and not p.get("v2") and "card" not in p
    assert recompose.SPLIT_CAM_MIN_H <= 1920 - p["top_h"] <= recompose.SPLIT_CAM_MAX_H


# ---------------------------------------------------------------- the last check: keep_cam

def test_keep_cam_overlays_the_inset_on_a_full_canvas_action_crop(tmp_path, monkeypatch):
    """A full-bleed action crop (or any full-canvas plan) of a source with a cam gets the inset
    overlaid on the frame's bottom corner; the graph draws it, crop_views lists its cut of the cam,
    a real render shows the cam there."""
    face = _face_of(CORNER)
    src = _source(tmp_path / "s.mp4", CORNER, face)
    _rig(monkeypatch, CORNER, face)
    a = _args(src)
    plan = {"layout": "track", "src_w": W, "src_h": H, "segments": [{"t0": 0.0, "t1": 3.0,
                                                                      "x": 600}],
            "banner_guard": {"action": "full_bleed_action"}}
    rep = recompose.keep_cam(plan, a, src)
    assert rep["inset"] and rep["ok"] and rep["over"] and rep["source_cam"] == CORNER
    ov = plan["cam_overlay"]
    assert gl.inset_ok(ov["box"])
    cw, ch, cx, cy = ov["crop"]
    assert CORNER[0] <= cx and cx + cw <= CORNER[2] and CORNER[1] <= cy and cy + ch <= CORNER[3]
    assert any(v["x"] == cx and v["w"] == cw for v in recompose.crop_views(plan, 3.0))
    vf = recompose.video_graph(plan, a)
    assert vf.endswith("[v0]") and vf.count("[v0]") == 1 and "[cpip]" in vf
    img = _render(vf, src, tmp_path / "o.mp4")
    x, y, w, h = ov["box"]
    assert _magenta(img)[y + 8:y + h - 8, x + 8:x + w - 8].mean() > 0.3


def test_keep_cam_records_the_card_and_leaves_speaker_crops_and_podcasts(tmp_path, monkeypatch):
    src = _source(tmp_path / "s.mp4", CORNER, _face_of(CORNER))
    _rig(monkeypatch, CORNER, _face_of(CORNER))
    a = _args(src)
    card = {"layout": "gamestack", "v2": True, "src_w": W, "src_h": H, "card": [40, 1414, 532, 466],
            "cam_box": CORNER, "cam_how": "busy"}
    rep = recompose.keep_cam(card, a, src)
    assert rep["inset"] == [40, 1414, 532, 466] and rep["corner"] == "left" and rep["ok"]
    talk = _args(src, ["--layout", "auto", "--vertical", "stream", "--niche", "streamers",
                       "--content-kind", "talk"])
    speaker = {"layout": "track", "src_w": W, "src_h": H,
               "segments": [{"t0": 0.0, "t1": 3.0, "x": 440}]}
    rep = recompose.keep_cam(speaker, talk, src)
    assert rep["inset"] is None and rep["source_cam"] is None and "cam_overlay" not in speaker
    assert quality_gate.measure_cam_inset("stream", rep)["passed"] is True
    podcast = _args(src, ["--layout", "auto", "--vertical", "podcast", "--niche", "podcasts"])
    plain = {"layout": "track", "src_w": W, "src_h": H, "segments": [{"t0": 0.0, "t1": 3.0,
                                                                      "x": 440}]}
    assert recompose.keep_cam(plain, podcast, src) is None and "cam_inset" not in plain


def test_keep_cam_puts_the_card_on_a_game_only_band_and_paints_the_cam_out_of_it(tmp_path,
                                                                                monkeypatch):
    face = _face_of(CORNER)
    src = _source(tmp_path / "s.mp4", CORNER, face)
    _rig(monkeypatch, CORNER, face)
    a = _args(src)
    g = gl.band_geometry(W, H, 0.42, max_h=gl.GAME_ONLY_BAND_MAX_H, min_share=0.40)
    plan = {"layout": "game_only", "v2": True, "src_w": W, "src_h": H,
            "band": {"w": g["gw"], "h": g["ch"], "out_h": g["gh"], "share": g["share"], "y": 0},
            "pan": [[0.0, 0]], "pan_y": None, "band_y": 200, "header": [0, 200], "card": None,
            "cap_y": 1325, "action": {"samples": []}}
    rep = recompose.keep_cam(plan, a, src)
    assert plan["layout"] == "gamestack" and rep["inset"] == plan["card"] and rep["ok"]
    assert plan["cam_mask"] and plan["cam_mask_share"] > 0           # the band showed the cam
    _assert_inset(plan, CORNER)


def test_main_moves_the_captions_above_an_overlaid_inset(tmp_path, monkeypatch):
    """A full render (main) of a plan with an overlaid inset that the default caption row would
    run into: the captions go above it, the QC line says so."""
    face = _face_of(CORNER)
    src = _source(tmp_path / "s.mp4", CORNER, face)
    words = tmp_path / "w.json"
    words.write_text(json.dumps([{"w": "hello", "start": 0.2, "end": 0.8},
                                 {"w": "world.", "start": 0.9, "end": 1.6}]))
    plan = {"layout": "track", "src_w": W, "src_h": H,
            "segments": [{"t0": 0.0, "t1": 3.0, "x": 600}],
            "banner_guard": {"action": "full_bleed_action"},
            "cam_overlay": {"box": [608, 1340, 432, 540], "crop": [100, 124, 60, 186],
                            "side": "right", "action_hits": 0},
            "cam_inset": {"rule": True, "layout": "track", "source_cam": CORNER,
                          "inset": [608, 1340, 432, 540], "over": True}}
    monkeypatch.setattr(recompose, "choose_layout", lambda a, s, info: (dict(plan), None))
    monkeypatch.setattr(recompose, "measure_levels", lambda *a, **k: None)
    out = tmp_path / "o.mp4"
    lines = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: lines.append(" ".join(map(str, a))))
    rc = recompose.main([str(src), "0", "3", str(out), "--captions", "--words", str(words),
                         "--vertical", "stream", "--niche", "gaming"])
    assert rc == 0, lines
    qc = json.loads(next(ln for ln in lines if ln.startswith("QC "))[3:])
    ci = qc["cam_inset"]
    assert ci["cap_clear"] is True and ci["cap_y"] <= 1340 - gl.CAP_HALF - gl.CAP_CARD_PAD
    assert ci["cap_y"] < 1270                                       # moved up from the default


# ---------------------------------------------------------------- the snap keeps the cam's own edge

def test_a_room_edge_inside_the_cam_is_no_border_the_box_side_on_the_cam_edge_stays():
    """overlay_rect read a wall's edge inside a room cam as the cam's border (x 140 for a cam to
    x 246) and the snap cut the card to half the cam. border_sides
    finds the box's side already on the cam's real edge (the game moving outside it) and the snap
    keeps it there."""
    frames = []
    room = np.full((H, W, 3), 60, np.uint8)
    room[:, 140:] = 120                                  # a static room edge at x 140
    xs = np.arange(W, dtype=np.float32)
    for k in range(12):
        game = 128 + 100 * np.sin(xs / 37.0 + 0.9 * k)    # the game: smooth, moving each frame
        f = np.repeat(np.repeat(game[None, :, None], H, 0), 3, 2).astype(np.uint8)
        f[176:316, 0:246] = room[176:316, 0:246]          # the cam: a static room
        frames.append(f)
    face = (121.0, 262.0, 35.0)
    box = [0, 176, 246, 316]
    on = gl.border_sides(frames, box, face)
    assert 240 <= on["x1"] <= 248
    rect = [66, 180, 144, 314]                           # the room edge read as the border
    snapped = gl.snap_to_rect(box, rect, face, W, H, keep=on)
    assert snapped[2] >= 240
    assert gl.snap_to_rect(box, rect, face, W, H)[2] == 144   # without it: half the cam


# ---------------------------------------------------------------- the gate: cam_missing

def test_measure_cam_inset():
    m = quality_gate.measure_cam_inset
    assert m("podcast", {"rule": True, "source_cam": CORNER})["passed"] is None
    assert m("stream", None)["passed"] is None                         # an older render
    assert m("stream", {"rule": True, "error": "boom"})["passed"] is None
    miss = m("stream", {"rule": True, "layout": "game_only", "source_cam": CORNER,
                        "inset": None, "why": "dropped"})
    assert miss["passed"] is False and miss["errors"][0].startswith("cam_missing:")
    assert m("gaming", {"rule": True, "source_cam": CORNER, "inset": None})["passed"] is False
    ok = m("stream", {"rule": True, "source_cam": CORNER, "inset": [40, 1414, 532, 466],
                      "corner": "left", "area": 0.12})
    assert ok["passed"] is True and ok["corner"] == "left"
    assert m("stream", {"rule": True, "source_cam": None, "inset": None})["passed"] is True


def _lean(monkeypatch, tmp_path, qc):
    monkeypatch.setattr(engine.runstate, "source",
                        lambda sid: {"source_id": sid, "vertical": "stream",
                                     "local_path": "/src/window.mp4"})
    monkeypatch.setattr(engine, "_niche_of", lambda sid: "gaming")
    monkeypatch.setattr(engine, "_content_kind_of", lambda sid: "screen")
    monkeypatch.setattr(engine, "_measure_black_frozen",
                        lambda *a, **kw: {"passed": True, "black_total_s": 0.0,
                                          "frozen_total_s": 0.0, "dur_s": 30.0})
    monkeypatch.setattr(engine, "_load_words_sidecar", lambda p: [
        {"w": "hello", "start": 2.0, "end": 2.5}, {"w": "world.", "start": 2.6, "end": 3.5}])
    monkeypatch.setattr(engine.quality_gate, "measure_visual_quality",
                        lambda *a, **k: {"passed": True})
    monkeypatch.setattr(engine, "_hook_judge", lambda: None)
    monkeypatch.setattr(engine.quality_gate, "clip_checks",
                        lambda mp4, **kw: {k: {"passed": True}
                                           for k in engine.quality_gate.CLIP_CHECK_KEYS})
    clip = {"clip_id": 1, "source_id": "s1_w00", "start_s": 2.0, "end_s": 32.0,
            "layout": "auto", "words_path": "/tmp/w.json", "hook": "Hello world"}
    return engine._run_quality_gate(tmp_path / "x.mp4", clip,
                                    qc=dict({"integrated_lufs": -14.1}, **qc))


def test_the_gate_rejects_a_cam_less_render_from_a_cam_source(monkeypatch, tmp_path):
    rep = _lean(monkeypatch, tmp_path, {"layout": "game_only", "cam_inset": {
        "rule": True, "layout": "game_only", "source_cam": CORNER, "inset": None,
        "why": "the cam shown full width, not as an inset"}})
    assert rep["verdict"] == "fail" and rep["cam_inset"]["passed"] is False
    assert rep["drop_reason"].startswith("cam_inset: cam_missing:")
    codes = rejects.codes_from_gate_report(rep)
    assert set(codes) == {"cam_missing"}
    assert not rejects.fixable("cam_missing")                     # never queued for repair
    assert rejects.classify(f"gate: {rep['drop_reason']}") == ["cam_missing"]
    assert set(rejects.codes_from_error(f"gate: {rep['drop_reason']}")) == {"cam_missing"}


@pytest.mark.parametrize("qc", [
    {"layout": "gamestack", "cam_inset": {"rule": True, "layout": "gamestack",
                                          "source_cam": CORNER, "inset": [40, 1414, 532, 466]}},
    {"layout": "game_only", "cam_inset": {"rule": True, "layout": "game_only",
                                          "source_cam": None, "inset": None}},
    {"layout": "gamestack"}])                                       # a render before the rule
def test_the_gate_passes_a_cam_inset_or_a_source_with_no_cam(monkeypatch, tmp_path, qc):
    rep = _lean(monkeypatch, tmp_path, qc)
    assert rep["verdict"] == "pass", rep["drop_reason"]
    assert rep["cam_inset"]["passed"] is not False


def test_the_new_code_is_in_the_taxonomy():
    fix, why = rejects.CODES["cam_missing"]
    assert fix is None and "cam" in why
    assert "cam_missing" not in rejects.classify("the streamer's camera angle is off")
