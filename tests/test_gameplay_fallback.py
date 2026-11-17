"""Gameplay never falls back to a face-follow crop.

A stream whose cam is burned in mid-frame used to fail gamestack's corner-inset rule and fall back
to the track crop: centred on the cam, the game cropped tight. Now, for gameplay:
  - a cam overlay anywhere in the frame is cut as the card where it is; the game band (gamelayout
    v3) takes an x clear of it, else shows it painted out (cam_mask), else the clip is game only;
  - no cam card (no cam, or a face in the game with no overlay border round it): game only;
  - no clean game band at all: RenderError (dropped), never track;
  - the stream lean gate fails a gameplay clip rendered as a face-follow crop
    (gameplay_face_follow).
Corner cams, game only with no face, and the stream vertical's talking heads / IRL vlogs are as
before. Synthetic sources (lavfi: a smooth moving game, a magenta cam overlay with a face blob);
the face detector, the banner scan and the OCR are faked; ffmpeg renders the masked graph."""
import subprocess

import numpy as np
import pytest

from clipengine import engine
from clipengine import quality_gate
from clipengine import reason_codes as rejects
from clipengine.render import facecam
from clipengine.render import gamelayout as gl
from clipengine.render import recompose

W, H = 1280, 720
GAME = ("gradients=s=1280x720:r=30:speed=0.03:seed=7:c0=0x203060:c1=0x406090:c2=0x305070:"
        "c3=0x20406a")
MAGENTA = "0xff00ff"
CENTRED = [520, 240, 760, 480]      # no 48 % band (614 px) fits either side of it
OFF_CENTRE = [860, 240, 1100, 480]  # mid-frame (no edge within 6 %), 834 px of game left of it
PART_LEFT = [580, 240, 820, 480]    # 554 px left of it: a 0.42 game only band fits, a 0.48 does not
CORNER = [0, 0, 260, 200]


def _source(path, cam=None, face=None):
    """A 3 s 1280x720 synthetic stream: a smooth moving game, a white square (the action) moving
    across its lower rows, and at `cam` a magenta overlay with a skin-coloured face blob (`face`
    alone: the blob in the game, no overlay)."""
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
                    "-t", "3", "-filter_complex", graph + "[v]", "-map", "[v]", "-map", "2:a",
                    "-pix_fmt", "yuv420p", "-shortest", str(path)], check=True)
    return path


def _face_of(cam):
    x0, y0, x1, y1 = cam
    return ((x0 + x1) / 2.0, y0 + 96.0, 60.0)


def _norm(box):
    return [box[0] / W, box[1] / H, box[2] / W, box[3] / H]


def _rig(monkeypatch, box=None, face=None):
    """The detection the clip would get: facecam's box (`box`, None: no cam by either rule) with
    the face in it, an empty banner scan / OCR (the gate's slice verdict passes), no facecam box
    read (bound_cam leaves the planned box)."""
    monkeypatch.setattr(facecam, "get_or_detect",
                        lambda channel, s, start, dur, source_duration=None, cache_dir=None,
                        log=None: (_norm(box), _norm(box), True, 1.0) if box
                        else (None, None, False, 0.0))
    monkeypatch.setattr(facecam, "detect", lambda *a, **k: None)
    faces = [face] * 15 if face else []
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
    monkeypatch.setattr(recompose, "_gamestack_cfg",
                        lambda: {"v2": True, "game_share": 0.50, "cam_max_area": 0.12,
                                 "game_only_share": 0.42, "header_h": 200})


GAMEPLAY = ["--layout", "auto", "--vertical", "stream", "--niche", "gaming",
            "--content-kind", "screen"]


def _args(src, extra=GAMEPLAY):
    return recompose.parse_args([str(src), "0", "3", "o.mp4"] + list(extra))


def _no_face_crop(monkeypatch):
    """reframe.analyze is the face-follow planner the old fallback ran: gameplay never calls it."""
    def analyze(*a, **k):
        raise AssertionError("gameplay must not plan a face-follow crop")
    monkeypatch.setattr(recompose.reframe, "analyze", analyze)


def _band_windows(plan, dur=3.0):
    b = plan["band"]
    return [[x, y, x + b["w"], y + b["h"]] for _t0, _t1, x, y in recompose.v2_views(plan, dur)]


# ---------------------------------------------------------------- a mid-frame cam

def test_mid_frame_cam_off_centre_band_clear_of_it_and_the_cam_card(tmp_path, monkeypatch):
    src = _source(tmp_path / "s.mp4", OFF_CENTRE, _face_of(OFF_CENTRE))
    _rig(monkeypatch, OFF_CENTRE, _face_of(OFF_CENTRE))
    _no_face_crop(monkeypatch)
    plan = recompose.plan_layout(_args(src), src)
    assert plan["layout"] == "gamestack" and plan["v2"] and plan["card"]
    assert plan["cam_mid"] and plan["cam_mask"] is None           # the band is clear of it
    guard_x0 = plan["raw_cam_box"][0]
    assert all(w[2] <= guard_x0 for w in _band_windows(plan))     # every band window left of it
    assert plan["band"]["share"] >= gl.MIN_SHARE
    cw, ch, cx, cy, _ = plan["cam"]                               # the card: cut where the cam is
    x0, y0, x1, y1 = OFF_CENTRE
    assert x0 <= cx and cx + cw <= x1 and y0 <= cy and cy + ch <= y1
    kx, ky, kw, kh = plan["card"]
    assert kw * kh <= 0.18 * gl.OUT_W * gl.OUT_H                  # a small card
    assert ky >= plan["band_y"] + plan["band"]["out_h"]          # never over the band


def test_mid_frame_cam_centred_is_painted_out_of_the_band_and_kept_on_the_card(tmp_path,
                                                                              monkeypatch):
    """No 48 % band fits either side of a centred cam: the band goes where the action is and the
    cam is painted out of it (delogo on the band's branch only); the card still shows it. A real
    ffmpeg render: no magenta on the band, magenta on the card."""
    import cv2
    src = _source(tmp_path / "s.mp4", CENTRED, _face_of(CENTRED))
    _rig(monkeypatch, CENTRED, _face_of(CENTRED))
    _no_face_crop(monkeypatch)
    plan = recompose.plan_layout(_args(src), src)
    assert plan["layout"] == "gamestack" and plan["card"] and plan["cam_mid"]
    mask = plan["cam_mask"]
    assert mask and gl.cover_share(mask, _band_windows(plan)) <= recompose.MID_CAM_MASK_MAX
    assert mask[0] <= CENTRED[0] or mask[0] >= min(w[0] for w in _band_windows(plan))
    vf = recompose.gamestack_v2_vf(plan)
    band_chain = vf.split(";")[1]
    assert band_chain.startswith("[gsrc]delogo=") and "delogo" not in vf.split(";")[0]
    assert vf.count("delogo") == 1 and "[csrc]crop=" in vf        # the card reads the raw source
    assert plan["cam_mask_share"] > 0
    assert recompose.blur_fill_frac(plan, None, {}) > gl.fill_frac(
        plan["band"]["out_h"], plan["card"], plan["band_y"])      # the painted cam counts as fill
    out = tmp_path / "o.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-filter_complex", vf,
                    "-map", "[v0]", "-t", "3", "-pix_fmt", "yuv420p", str(out)], check=True)
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", "1.5", "-i", str(out), "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True,
                       check=True)
    img = np.frombuffer(r.stdout, np.uint8).reshape(1920, 1080, 3)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    magenta = (np.abs(hsv[..., 0].astype(int) - 150) <= 8) & (hsv[..., 1] > 150) & \
        (hsv[..., 2] > 150)
    y0, y1 = plan["band_y"], plan["band_y"] + plan["band"]["out_h"]
    assert magenta[y0:y1].mean() < 0.002                          # the cam is not on the band
    kx, ky, kw, kh = plan["card"]
    assert magenta[ky:ky + kh, kx:kx + kw].mean() > 0.3           # it is on the card


def test_mid_frame_cam_no_gamestack_band_takes_is_game_only_clear_of_it(tmp_path, monkeypatch):
    """A cam no 48 % band avoids and (cap 0) none may paint out: game only (no card), its 0.42
    band clear of the cam."""
    src = _source(tmp_path / "s.mp4", PART_LEFT, _face_of(PART_LEFT))
    _rig(monkeypatch, PART_LEFT, _face_of(PART_LEFT))
    _no_face_crop(monkeypatch)
    monkeypatch.setattr(recompose, "MID_CAM_MASK_MAX", 0.0)
    plan = recompose.plan_layout(_args(src), src)
    assert plan["layout"] == "game_only" and plan["card"] is None and plan["v2"]
    assert plan["cam_mid"] and plan["cam_mask"] is None
    assert all(w[2] <= PART_LEFT[0] for w in _band_windows(plan))


def test_mid_frame_cam_with_no_clean_game_band_is_dropped_never_track(tmp_path, monkeypatch):
    src = _source(tmp_path / "s.mp4", CENTRED, _face_of(CENTRED))
    _rig(monkeypatch, CENTRED, _face_of(CENTRED))
    _no_face_crop(monkeypatch)
    monkeypatch.setattr(recompose, "MID_CAM_MASK_MAX", 0.0)
    with pytest.raises(recompose.RenderError, match="no clean game band round the mid-frame cam"):
        recompose.plan_layout(_args(src), src)


def test_a_face_in_the_game_with_no_overlay_border_is_game_only(tmp_path, monkeypatch):
    """facecam's box round a face the game shows (no overlay borders round it): no cam card, and
    gameplay never takes the face crop: game only."""
    face = (640.0, 336.0, 60.0)
    src = _source(tmp_path / "s.mp4", None, face)
    _rig(monkeypatch, [560, 240, 720, 432], face)
    _no_face_crop(monkeypatch)
    plan = recompose.plan_layout(_args(src), src)
    assert plan["layout"] == "game_only" and plan["card"] is None
    assert "cam_mid" not in plan


def test_gamestack_request_with_a_face_and_no_cam_is_game_only(tmp_path, monkeypatch):
    src = _source(tmp_path / "s.mp4", None, (640.0, 336.0, 60.0))
    _rig(monkeypatch, None, (640.0, 336.0, 60.0))
    _no_face_crop(monkeypatch)
    plan = recompose.plan_layout(_args(src, ["--layout", "gamestack"]), src)
    assert plan["layout"] == "game_only"


def test_talking_head_stream_with_a_face_in_the_scene_keeps_the_speaker_crop(tmp_path,
                                                                            monkeypatch):
    """The stream vertical's talking heads / IRL vlogs (content kind talk) are not gameplay: a box
    inside the scene is no cam and the fallback is the speaker crop, as before."""
    face = (640.0, 336.0, 60.0)
    src = _source(tmp_path / "s.mp4", None, face)
    _rig(monkeypatch, [560, 240, 720, 432], face)
    monkeypatch.setattr(recompose.reframe, "analyze", lambda *a, **k: {
        "layout": "track", "src_w": W, "src_h": H, "shots": [],
        "segments": [{"t0": 0.0, "t1": 3.0, "x": 440, "cut": True}]})
    a = _args(src, ["--layout", "auto", "--vertical", "stream", "--niche", "streamers",
                    "--content-kind", "talk"])
    assert recompose.plan_layout(a, src)["layout"] == "track"


def test_the_facecam_check_accepts_a_band_that_shows_the_cam_painted_out(monkeypatch):
    """The facecam check drops a window holding the facecam with the game round it; a band whose
    cam is painted out (cam_mask over the cam's read box) shows no cam, so it ships."""
    g = gl.band_geometry(W, H)
    plan = {"layout": "gamestack", "v2": True, "src_w": W, "src_h": H,
            "band": {"w": g["gw"], "h": g["ch"], "out_h": g["gh"], "y": 0},
            "pan": [[0.0, 300], [3.0, 400]], "pan_y": None,
            "cam": [234, 204, 522, 244, False], "cam_mask": [510, 230, 768, 488]}
    monkeypatch.setattr(recompose, "cam_overlay", lambda src, a, size: {
        "face": {"cx": 640 / W, "cy": 336 / H, "h": 60 / H}, "tile": [522, 242, 756, 476],
        "read": 1.0})
    a = _args("s.mp4")
    assert recompose.facecam_view_check(plan, a, "s.mp4") is None
    with pytest.raises(recompose.RenderError, match="shows the cam with the game around it"):
        recompose.facecam_view_check({k: v for k, v in plan.items() if k != "cam_mask"}, a,
                                     "s.mp4")
    short = dict(plan, cam_mask=[600, 230, 768, 488])            # the mask misses part of it
    with pytest.raises(recompose.RenderError):
        recompose.facecam_view_check(short, a, "s.mp4")


def test_a_mid_frame_cam_game_only_with_a_defect_gets_no_action_crop(monkeypatch):
    """The 9:16 action crop (hud_layout's last step for game only) could centre on a mid-frame
    cam: such a plan is dropped instead; a game only plan with no cam keeps the step as before."""
    calls = []
    monkeypatch.setattr(recompose, "shown_defect", lambda plan, a, src: "slices burned-in text")
    monkeypatch.setattr(recompose, "stack_fallback", lambda *a, **k: None)
    monkeypatch.setattr(recompose, "full_bleed_action",
                        lambda plan, a, src, why: calls.append(why) or {"action": ACTION})
    a = _args("s.mp4")
    mid = {"layout": "game_only", "v2": True, "src_w": W, "src_h": H, "cam_mid": [1, 2, 3, 4]}
    with pytest.raises(recompose.RenderError, match="no clean streamer split or full-bleed"):
        recompose.hud_layout(mid, a, "s.mp4", None, None)
    assert calls == []
    plain = {"layout": "game_only", "v2": True, "src_w": W, "src_h": H}
    assert recompose.hud_layout(plain, a, "s.mp4", None, None) == {"action": ACTION}
    assert calls == ["slices burned-in text"]


ACTION = "full_bleed_action"


# ---------------------------------------------------------------- unchanged paths

def test_corner_cam_takes_the_corner_path_unchanged(tmp_path, monkeypatch):
    face = (130.0, 90.0, 60.0)
    src = _source(tmp_path / "s.mp4", CORNER, face)
    _rig(monkeypatch, CORNER, face)
    _no_face_crop(monkeypatch)

    def mid(*a, **k):
        raise AssertionError("a corner cam is no mid-frame cam")
    monkeypatch.setattr(recompose, "_mid_cam_band", mid)
    plan = recompose.plan_layout(_args(src), src)
    assert plan["layout"] == "gamestack" and plan["card"]
    assert not {"cam_mid", "cam_mask", "cam_mask_share"} & set(plan)
    assert all(w[0] >= plan["raw_cam_box"][2] for w in _band_windows(plan))
    assert "delogo" not in recompose.gamestack_v2_vf(plan)


def test_no_face_is_game_only_as_before(tmp_path, monkeypatch):
    src = _source(tmp_path / "s.mp4")
    _rig(monkeypatch, None, None)
    _no_face_crop(monkeypatch)
    a = _args(src)
    plan = recompose.plan_layout(a, src)
    assert plan["layout"] == "game_only" and plan["card"] is None and "cam_mid" not in plan
    assert plan == recompose._plan_game_only(src, a)              # the no-face planner itself


# ---------------------------------------------------------------- the last check before the encode

def _track_plan(**extra):
    return {"layout": "track", "src_w": W, "src_h": H,
            "segments": [{"t0": 0.0, "t1": 3.0, "x": 440}], **extra}


@pytest.mark.parametrize("extra", [GAMEPLAY, ["--layout", "track", "--vertical", "stream",
                                              "--niche", "gaming"],
                                   ["--layout", "gamestack"]])
def test_a_face_crop_for_gameplay_is_refused_before_the_encode(extra):
    a = _args("s.mp4", extra)
    with pytest.raises(recompose.RenderError, match="face-follow crop") as e:
        recompose.enforce_game_band(_track_plan(), a)
    assert rejects.codes_from_error(f"recompose exit 1: RENDER FAILED: {e.value}") == \
        {"gameplay_face_follow": f"recompose exit 1: RENDER FAILED: {e.value}"[:300]}
    with pytest.raises(recompose.RenderError):
        recompose.enforce_game_band({**_track_plan(), "layout": "split"}, a)


def test_the_face_crop_check_leaves_talk_poker_and_the_action_crop_alone():
    talk = _args("s.mp4", ["--vertical", "stream", "--niche", "streamers", "--content-kind",
                           "talk"])
    poker = _args("s.mp4", ["--vertical", "poker", "--niche", "poker"])
    podcast = _args("s.mp4", ["--vertical", "podcast", "--niche", "podcasts"])
    for a in (talk, poker, podcast):
        assert recompose.enforce_game_band(_track_plan(), a) is None
    action = _track_plan(banner_guard={"action": "full_bleed_action"})   # no face: as before
    assert recompose.enforce_game_band(action, _args("s.mp4")) is None
    assert recompose.enforce_game_band({"layout": "game_only", "v2": True}, _args("s.mp4")) is None


def test_choose_layout_drops_a_gameplay_track_request(monkeypatch):
    """A --layout track request: for gameplay the plan never ships."""
    a = _args("s.mp4", ["--layout", "track", "--vertical", "stream", "--niche", "gaming",
                        "--content-kind", "screen"])
    monkeypatch.setattr(recompose, "plan_layout", lambda a, src: _track_plan())
    monkeypatch.setattr(recompose, "guard_banners", lambda plan, a, src: None)
    monkeypatch.setattr(recompose, "hud_layout", lambda plan, a, src, guard, rej: guard)
    monkeypatch.setattr(recompose, "screen_fit_general", lambda *a, **k: None)
    with pytest.raises(recompose.RenderError, match="gameplay planned as 'track'"):
        recompose.choose_layout(a, "s.mp4", {"width": W, "height": H})


# ---------------------------------------------------------------- the gate

@pytest.mark.parametrize("vertical,niche,kind,layout,want", [
    ("stream", "gaming", "screen", "auto", True),
    ("stream", "gaming", "unknown", "auto", True),      # a gaming creator with no layout hint
    ("stream", "gaming", None, "auto", True),
    ("stream", "streamers", "screen", "auto", True),    # facecam gameplay
    ("stream", "gaming", "talk", "auto", False),        # talking heads
    ("stream", "streamers", "talk", "auto", False),     # an IRL vlog
    ("stream", "streamers", "unknown", "auto", False),  # a streamer with no kind
    ("stream", "streamers", None, "auto", False),
    ("stream", None, None, "auto", False),              # a manual render
    ("gaming", None, None, "auto", True),
    ("poker", "poker", None, "auto", False),
    ("podcast", "podcasts", "talk", "track", False),
    (None, None, None, "gamestack", True),              # a gamestack request
    ("stream", "gaming", "screen", "track", True),      # a --layout track request
])
def test_gameplay_clip(vertical, niche, kind, layout, want):
    assert quality_gate.gameplay_clip(vertical, niche, kind, layout) is want


def _lean(monkeypatch, tmp_path, layout, niche="gaming", kind="screen", guard=None):
    monkeypatch.setattr(engine.runstate, "source",
                        lambda sid: {"source_id": sid, "vertical": "stream",
                                     "local_path": "/src/window.mp4"})
    monkeypatch.setattr(engine, "_niche_of", lambda sid: niche)
    monkeypatch.setattr(engine, "_content_kind_of", lambda sid: kind)
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
    qc = {"integrated_lufs": -14.1, "layout": layout}
    if guard:
        qc["banner_guard"] = {"action": guard}
    return engine._run_quality_gate(tmp_path / "x.mp4", clip, qc=qc)


@pytest.mark.parametrize("layout", ["track", "split", "full_bleed_speaker"])
def test_the_lean_gate_fails_gameplay_rendered_as_a_face_crop(monkeypatch, tmp_path, layout):
    rep = _lean(monkeypatch, tmp_path, layout)
    assert rep["verdict"] == "fail" and rep["note"] == "stream lean gate"
    assert rep["gameplay_layout"]["passed"] is False
    assert rep["drop_reason"].startswith("gameplay_layout: gameplay rendered as")
    codes = rejects.codes_from_gate_report(rep)
    assert set(codes) == {"gameplay_face_follow"}
    assert not rejects.fixable("gameplay_face_follow")            # no repair step fixes it
    assert rejects.classify(f"gate: {rep['drop_reason']}") == ["gameplay_face_follow"]


@pytest.mark.parametrize("layout,niche,kind,guard", [
    ("gamestack", "gaming", "screen", None), ("game_only", "gaming", "screen", None),
    ("dual_view", "gaming", "screen", None),
    ("track", "gaming", "screen", "full_bleed_action"),           # no face: the action crop
    ("track", "streamers", "talk", None),                          # an IRL vlog's speaker crop
    ("track", "gaming", "talk", None)])
def test_the_lean_gate_passes_the_game_band_and_talking_heads(monkeypatch, tmp_path, layout,
                                                              niche, kind, guard):
    rep = _lean(monkeypatch, tmp_path, layout, niche, kind, guard)
    assert rep["verdict"] == "pass", rep["drop_reason"]
    assert rep["gameplay_layout"]["passed"] is not False


def test_the_new_code_is_in_the_taxonomy():
    fix, why = rejects.CODES["gameplay_face_follow"]
    assert fix is None and "face-follow" in why
    # a reviewer's prose about an IRL clip's crop is no gameplay reject
    reason = ("For roughly the first 60% of the clip, the face-follow crop shows the ceiling, an "
              "arm, an ear and the back of a seat, and no speaker's face is visible.")
    assert "gameplay_face_follow" not in rejects.classify(reason)
