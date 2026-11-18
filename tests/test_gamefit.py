"""A gaming / stream window whose every crop slices its text, HUD or banners renders `gamefit`
instead of being refused: the whole game frame fit to the canvas width (gameplay 100 % of the
width, nothing sliced), the hook header above it, the captions under it, the streamer cam as the
inset in a bottom corner (the cam rule's bounds), over the frame blurred and darkened.

  - geometry (gamelayout.fit_geometry / fit_card / fit_shares);
  - selection: the refusal (hud_layout: "crop would slice a source graphic or show a blurred
    fill ... no clean streamer split or full-bleed crop") becomes gamefit, the refusal recorded;
    a window the layouts render ships as before; a talking head, poker, a non-slicing refusal and a
    cam gamefit cannot place keep the refusal;
  - the graph (ffmpeg): the frame at its rows, the cam card in its corner, a darkened fill;
  - the gate: crop_slices passes the uncropped frame by construction (the card's crop still
    counts), the blur-fill cap is gamefit's own, the lean gate passes a gamefit render with its
    inset, main's QC line and the engine keep the layout and the refusal it replaced.
Synthetic sources (lavfi); the face detector, banner scan and OCR are faked."""
import json
import re
import subprocess

import numpy as np
import pytest

from clipengine import engine
from clipengine import quality_gate
from clipengine.render import gamelayout as gl
from clipengine.render import recompose

W, H = 1280, 720
GAME = ("gradients=s=1280x720:r=30:speed=0.03:seed=7:c0=0x203060:c1=0x406090:c2=0x305070:"
        "c3=0x20406a")
MAGENTA = "0xff00ff"
CORNER = [0, 176, 248, 316]          # an edge cam (a streamer's room cam at the left edge)
MID = [116, 302, 340, 476]           # a cam box inside the scene (the streamer split refused it)
REFUSAL = ("crop would slice a source graphic or show a blurred fill: crop would slice a source "
           "banner: game band 640px slices banner(s) [[0, 27, 714, 79], [190, 623, 1084, 667]] on "
           "its path; no clean streamer split or full-bleed crop")
GAMEPLAY = ["--layout", "auto", "--vertical", "stream", "--niche", "gaming",
            "--content-kind", "screen"]


def _source(path, cam=None, dur=3):
    chain = ""
    if cam:
        x0, y0, x1, y1 = cam
        chain = f",drawbox=x={x0}:y={y0}:w={x1 - x0}:h={y1 - y0}:color={MAGENTA}:t=fill"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", GAME, "-f", "lavfi", "-i",
                    "sine=frequency=440", "-t", str(dur), "-vf", "null" + chain, "-pix_fmt",
                    "yuv420p", "-shortest", str(path)], check=True)
    return path


def _args(src="s.mp4", extra=GAMEPLAY, dur="3"):
    return recompose.parse_args([str(src), "0", dur, "o.mp4"] + list(extra))


def _gamestack(cam=CORNER):
    """The v2 gamestack plan the planner made of the window (its cam card)."""
    x0, y0, x1, y1 = cam
    return {"layout": "gamestack", "v2": True, "src_w": W, "src_h": H, "cam_box": list(cam),
            "face_median": [(x0 + x1) / 2.0 / W, (y0 + 0.55 * (y1 - y0)) / H, 40.0 / H],
            "cam_how": "busy", "card": [40, 1414, 532, 466]}


@pytest.fixture
def rig(monkeypatch):
    """No source bars, no stream cam unless set, an empty banner scan / OCR (the gate's slice
    verdict passes), the v3 knobs."""
    recompose._SCANS.clear()
    recompose._CAMS.clear()
    state = {"stream_cam": None}
    sc = {"graphics": [], "panels": [], "frames": 13, "src_w": W, "src_h": H, "faces": []}
    monkeypatch.setattr(recompose, "_scan", lambda src, a, size: sc)
    monkeypatch.setattr(recompose, "_ocr", lambda src, a, size: [])
    monkeypatch.setattr(recompose, "source_bars", lambda src, a, sw, sh: None)
    monkeypatch.setattr(recompose, "stream_cam", lambda src, a, size: state["stream_cam"])
    # the face read of the card (tests/test_gamefit_cam.py): unread here, the card as cut (its
    # cam region still covered on the frame)
    monkeypatch.setattr(recompose, "_cam_samples", lambda src, a, size, face: None)
    gs = {"v2": True, "game_share": 0.50, "cam_max_area": 0.12, "game_only_share": 0.42,
          "header_h": 200}
    monkeypatch.setattr(recompose, "_gamestack_cfg", lambda: dict(gs))
    return state


def _refusing(monkeypatch, prior, err=None):
    """choose_layout's layouts plan `prior` and refuse it with `err` (the refusal)."""
    def choose(a, src, info, seen):
        seen["plan"] = dict(prior)
        raise err or recompose.RenderError(REFUSAL)
    monkeypatch.setattr(recompose, "_choose_layout", choose)


def _assert_card(plan, cam):
    card = plan["card"]
    assert gl.inset_ok(card) and gl.inset_corner(card) in ("left", "right")
    assert gl.inset_area(card) <= gl.CAM_INSET_MAX_AREA and card[2] >= gl.CAM_INSET_MIN_W * gl.OUT_W
    cw, ch, cx, cy = plan["cam"]                          # the card is cut from the cam box
    x0, y0, x1, y1 = cam
    assert x0 <= cx and cx + cw <= x1 and y0 <= cy and cy + ch <= y1
    assert card[1] >= plan["frame_y"] + plan["frame_h"]  # never over the game frame
    assert not gl.caption_overlap(plan["cap_y"], card)   # the captions above it


# ---------------------------------------------------------------- geometry (pure)

def test_a_16_9_frame_fits_the_width_under_the_header_with_captions_and_card_under_it():
    g = gl.fit_geometry(W, H)
    assert g["fh"] == 608 and g["frame"] == [492, 1100]          # gameplay 1080 / 1080 wide
    assert g["header"] == [gl.FIT_HOOK_TOP, 492] and g["header"][1] - g["header"][0] >= 250
    assert g["cap_y"] == 1100 + gl.FIT_CAP_GAP and g["cap_y"] <= gl.CAPTION_Y_MAX
    for side in ("left", "right"):
        card = gl.fit_card(g, side, max_area=0.12)
        assert gl.inset_ok(card) and gl.inset_corner(card) == side
        assert card[1] >= g["cap_y"] + gl.CAP_HALF and card[1] + card[3] <= gl.OUT_H - gl.INSET_MARGIN
    sh = gl.fit_shares(g["fh"], gl.fit_card(g, "left", 0.12))
    assert sh["frame"] == pytest.approx(0.317, abs=0.002) and sh["card"] <= 0.12
    assert sh["fill"] == pytest.approx(1 - sh["frame"] - sh["card"], abs=0.002)
    assert gl.fit_shares(608)["fill"] == pytest.approx(0.683, abs=0.002)


def test_the_fill_is_darkened_less_in_a_dark_window():
    """A near-black scene: a darkened fill of a dark frame reads as a black band."""
    assert gl.fit_fill_dark(None) == gl.FIT_FILL_DARK
    assert gl.fit_fill_dark(120.0) == gl.fit_fill_dark(gl.FIT_DARK_LUMA) == gl.FIT_FILL_DARK
    assert gl.fit_fill_dark(4.2) == gl.fit_fill_dark(gl.FIT_DARK_LUMA / 2) == 1.0
    mid = gl.fit_fill_dark(gl.FIT_DARK_LUMA * 0.75)
    assert gl.FIT_FILL_DARK < mid < 1.0
    plan = {"layout": "gamefit", "src_w": W, "src_h": H, "frame_y": 492, "frame_h": 608,
            "card": None, "fill_dark": 0.85}
    assert "colorchannelmixer=rr=0.85:gg=0.85:bb=0.85" in recompose.gamefit_vf(plan)
    assert f"rr={gl.FIT_FILL_DARK}:" in recompose.gamefit_vf(dict(plan, fill_dark=gl.FIT_FILL_DARK))


def test_a_4_3_frame_keeps_the_header_and_a_portrait_frame_has_no_gamefit():
    g = gl.fit_geometry(1440, 1080)
    assert g["fh"] == 810 and g["frame"][0] == gl.FIT_HEADER_MIN
    assert gl.fit_card(g, "right", 0.12) is not None
    assert gl.fit_geometry(720, 1280) is None                    # a vertical stream
    assert gl.fit_geometry(1000, 1000) is None                   # under FIT_MIN_AR
    assert gl.fit_geometry(0, 720) is None


# ---------------------------------------------------------------- selection: gamefit replaces a refusal

def test_the_refusal_becomes_gamefit_with_the_cam_as_the_inset(rig, monkeypatch):
    _refusing(monkeypatch, _gamestack(CORNER))
    plan, guard = recompose.choose_layout(_args(), "s.mp4", {"width": W, "height": H})
    assert plan["layout"] == recompose.GAMEFIT
    assert plan["frame_y"] == 492 and plan["frame_h"] == 608
    assert guard["action"] == "gamefit" and guard["why"] == REFUSAL
    assert plan["gamefit"] == {"replaced": REFUSAL, "replaced_layout": "gamestack"}
    _assert_card(plan, CORNER)
    assert plan["card_side"] == "left"                           # the cam's side in the source
    ci = plan["cam_inset"]                                       # the gate's cam_missing check
    assert ci["source_cam"] == CORNER and ci["inset"] == plan["card"] and ci["ok"]
    assert quality_gate.measure_cam_inset("stream", ci)["passed"] is True
    assert quality_gate.measure_gameplay_layout(True, plan["layout"], guard["action"])["passed"]


def test_the_real_hud_layout_refusal_path_ends_in_gamefit(rig, monkeypatch):
    """The refusal flow itself: the planner's v2 band, the banner guard rejecting it, no streamer
    split: hud_layout raises its refusal and choose_layout renders gamefit instead."""
    monkeypatch.setattr(recompose, "plan_layout", lambda a, src: _gamestack(MID))
    monkeypatch.setattr(recompose, "guard_banners", lambda plan, a, src: (_ for _ in ()).throw(
        recompose.RenderError("crop would slice a source banner: game band 616px slices "
                              "banner(s) [[952, 560, 1280, 634]] on its path")))
    monkeypatch.setattr(recompose, "to_streamer_split", lambda *a, **k: None)
    seen = []
    real = recompose.hud_layout

    def hud(*a, **k):
        try:
            return real(*a, **k)
        except recompose.RenderError as e:
            seen.append(str(e))
            raise
    monkeypatch.setattr(recompose, "hud_layout", hud)
    plan, guard = recompose.choose_layout(_args(), "s.mp4", {"width": W, "height": H})
    assert seen and "no clean streamer split or full-bleed crop" in seen[0]
    assert plan["layout"] == "gamefit" and guard["why"] == seen[0]
    _assert_card(plan, MID)                     # a mid-frame cam: the plan's box, as the inset


def test_a_window_the_layouts_render_ships_unchanged(rig, monkeypatch):
    done = (_gamestack(), {"action": "none"})
    monkeypatch.setattr(recompose, "_choose_layout", lambda a, src, info, seen: done)
    assert recompose.choose_layout(_args(), "s.mp4", {"width": W, "height": H}) is done


@pytest.mark.parametrize("extra, prior", [
    (["--layout", "auto", "--vertical", "stream", "--niche", "streamers", "--content-kind", "talk"],
     {"layout": "track", "src_w": W, "src_h": H, "segments": [{"t0": 0, "t1": 3, "x": 600}]}),
    (["--layout", "auto", "--vertical", "poker", "--niche", "poker"], _gamestack()),
    (["--layout", "auto", "--niche", "tech"], _gamestack()),
])
def test_a_talking_head_poker_or_tech_window_keeps_its_refusal(rig, monkeypatch, extra, prior):
    _refusing(monkeypatch, prior)
    with pytest.raises(recompose.RenderError, match="no clean streamer split") as e:
        recompose.choose_layout(_args(extra=extra), "s.mp4", {"width": W, "height": H})
    assert not str(e.value).startswith("gamefit")


@pytest.mark.parametrize("extra", [
    ["--layout", "track", "--vertical", "stream", "--niche", "gaming", "--content-kind", "screen"],
    GAMEPLAY + ["--clean-crop", "soft"]])
def test_a_repair_recrop_keeps_its_refusal(rig, monkeypatch, extra):
    """The repair lane's recrop (--layout track, --clean-crop) of a rejected clip is no first render:
    its refusal stands (a gamefit there would re-render what the eye-check rejected)."""
    _refusing(monkeypatch, _gamestack(), recompose.RenderError(
        "gameplay planned as 'track', a face-follow crop: gameplay renders the game band"))
    with pytest.raises(recompose.RenderError, match="^gameplay planned as 'track'"):
        recompose.choose_layout(_args(extra=extra), "s.mp4", {"width": W, "height": H})


def test_a_refusal_that_is_no_slicing_stands(rig, monkeypatch):
    _refusing(monkeypatch, _gamestack(), recompose.RenderError("banner scan failed: OSError: x"))
    with pytest.raises(recompose.RenderError, match="^banner scan failed"):
        recompose.choose_layout(_args(), "s.mp4", {"width": W, "height": H})


def test_a_cam_gamefit_cannot_place_keeps_the_refusal_with_both_reasons(rig, monkeypatch):
    """Never a fit without the cam: no inset placement -> the refusal stands; the refusal stays
    at the end of the message, where the engine reads it (uncroppable)."""
    _refusing(monkeypatch, _gamestack(CORNER))
    monkeypatch.setattr(gl, "fit_card", lambda geo, side, max_area=0.12: None)
    with pytest.raises(recompose.RenderError) as e:
        recompose.choose_layout(_args(), "s.mp4", {"width": W, "height": H})
    msg = str(e.value)
    assert msg.startswith("gamefit: the cam [0, 176, 248, 316] cannot be placed as the inset")
    assert msg.endswith(REFUSAL)
    assert engine._SLICE_RENDER.search(f"recompose exit 1: {msg}"[-300:])


def test_a_portrait_frame_keeps_the_refusal(rig, monkeypatch):
    _refusing(monkeypatch, dict(_gamestack(), src_w=720, src_h=1280))
    with pytest.raises(recompose.RenderError, match="^gamefit: a 720x1280 frame fit to width"):
        recompose.choose_layout(_args(), "s.mp4", {"width": 720, "height": 1280})


def test_a_mid_frame_cam_no_band_took_rides_on_the_refusal(rig, monkeypatch):
    """_mid_cam_game_only's refusal (every band x slices a banner round the mid-frame cam)
    carries the cam; gamefit shows it as the inset."""
    face = ((MID[0] + MID[2]) / 2.0, 400.0, 44.0)
    monkeypatch.setattr(recompose, "_plan_game_only_v2", lambda *a, **k: None)
    with pytest.raises(recompose.RenderError) as e:
        recompose._mid_cam_game_only("s.mp4", _args(), {}, (W, H), MID, MID, "every 50% band x "
                                     "slices a banner", {"box": list(MID), "face": face,
                                                         "how": "busy"})
    assert e.value.gamefit_cam["box"] == MID
    _refusing(monkeypatch, {}, e.value)
    plan, guard = recompose.choose_layout(_args(), "s.mp4", {"width": W, "height": H})
    assert plan["layout"] == "gamefit" and guard["why"].startswith("gameplay: no clean game band")
    _assert_card(plan, MID)


def test_a_source_with_no_cam_fits_with_no_card_and_the_gate_reads_no_cam(rig, monkeypatch):
    _refusing(monkeypatch, {"layout": "game_only", "v2": True, "src_w": W, "src_h": H})
    plan, _g = recompose.choose_layout(_args(), "s.mp4", {"width": W, "height": H})
    assert plan["layout"] == "gamefit" and plan["card"] is None and plan["cam"] is None
    assert plan["cam_inset"]["source_cam"] is None
    assert quality_gate.measure_cam_inset("stream", plan["cam_inset"])["passed"] is True
    rig["stream_cam"] = {"box": list(CORNER), "face": None, "how": "source"}   # the source's cam
    plan, _g = recompose.choose_layout(_args(), "s.mp4", {"width": W, "height": H})
    _assert_card(plan, CORNER)


# ---------------------------------------------------------------- the graph (ffmpeg)

def test_gamefit_renders_the_whole_frame_the_card_and_a_darkened_fill(tmp_path, rig, monkeypatch):
    src = _source(tmp_path / "s.mp4", CORNER)
    plan = recompose.plan_gamefit(src, _args(src), {"width": W, "height": H},
                                  prior=_gamestack(CORNER))
    out = tmp_path / "o.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-filter_complex",
                    recompose.video_graph(plan, _args(src)), "-map", "[v0]", "-t", "1",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    img = np.frombuffer(subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", "0.5", "-i", str(out), "-frames:v", "1", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout, np.uint8)
    img = img.reshape(1920, 1080, 3).astype(int)
    fr = np.frombuffer(subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", "0.5", "-i", str(src), "-frames:v", "1", "-vf",
         "scale=1080:608", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True,
        check=True).stdout, np.uint8).reshape(608, 1080, 3).astype(int)
    y0 = plan["frame_y"]
    vx, vy, vw, vh = plan["cam_cover_canvas"]                  # the cam covered
    shown = np.ones((600, 1080), bool)
    shown[max(0, vy - y0 - 4 - 2):vy - y0 - 4 + vh + 2, :vx + vw + 2] = False
    assert np.abs(img[y0 + 4:y0 + 604] - fr[4:604])[shown].mean() < 6   # the whole frame, uncut
    magenta = (img[..., 2] > 200) & (img[..., 1] < 80) & (img[..., 0] > 200)
    kx, ky, kw, kh = plan["card"]
    assert magenta[ky + 20:ky + kh - 20, kx + 20:kx + kw - 20].mean() > 0.9   # the cam card
    assert not magenta[y0:y0 + 608].any()                      # ... and only there
    fill = img[gl.FIT_HOOK_TOP - 100:gl.FIT_HOOK_TOP, 300:800]       # header rows: the fill
    assert fill.mean() < fr.mean() * 0.85 and fill.mean() > 5      # darkened, never black
    assert recompose.crop_views(plan, 3.0) == [{"t0": 0.0, "t1": 3.0, "x": float(plan["cam"][2]),
                                                "y": float(plan["cam"][3]),
                                                "w": float(plan["cam"][0]),
                                                "h": float(plan["cam"][1])}]
    assert recompose.blur_fill_frac(plan, _args(src), {}) == \
        gl.fit_shares(608, plan["card"], plan["cam_cover_canvas"])["fill"]


# ---------------------------------------------------------------- the gate

def _item_at_the_edge():
    """A HUD line running into the frame's right edge (OCR-confirmed), and one inside the frame."""
    sc = {"src_w": W, "src_h": H, "graphics": [
        {"box": [1100, 20, 1280, 60], "frac": 1.0, "banner": True},
        {"box": [400, 600, 700, 640], "frac": 1.0, "banner": True}]}
    words = [(1110, 25, 1270, 55, "KILLFEED", 0), (410, 605, 690, 635, "SCOREBOARD", 0)]
    return sc, words


def test_crop_slices_passes_the_uncropped_frame_by_construction():
    sc, words = _item_at_the_edge()
    whole = quality_gate.crop_slices(sc, [], 3.0, words=words)
    assert [s["how"] for s in whole] == ["frame"]                  # any other layout: a frame cut
    assert quality_gate.crop_slices(sc, [], 3.0, words=words, frame_edges=False) == []
    assert not quality_gate.frame_edges_cut("gamefit") and quality_gate.frame_edges_cut("general")
    card = [{"t0": 0.0, "t1": 3.0, "x": 500, "y": 560, "w": 300, "h": 160}]   # a card through it
    cut = quality_gate.crop_slices(sc, card, 3.0, words=words, frame_edges=False)
    assert [s["how"] for s in cut] == ["crop"]                     # the card's crop still counts
    v = quality_gate.crop_slice_verdict(cut)
    assert not v["passed"]


def test_the_blur_fill_cap_is_gamefits_own():
    stats = [{"luma": 90.0, "contrast": 50.0, "sharpness": 1500.0}] * 4
    v = quality_gate.visual_quality_verdict
    assert v(stats, 0.683, layout="gamefit")["passed"]             # a 16:9 frame, no card
    assert v(stats, 0.563, layout="gamefit")["passed"]
    assert not v(stats, 0.76, layout="gamefit")["passed"]          # a 21:9 strip
    assert not v(stats, 0.683, layout="general")["passed"]         # every other layout as before
    assert not v(stats, 0.683)["passed"]
    dark = [{"luma": 20.0, "contrast": 50.0, "sharpness": 1500.0}] * 4
    assert not v(dark, 0.6, layout="gamefit")["passed"]            # luma still applies


def test_the_lean_gate_passes_a_gamefit_render_with_its_inset(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(engine.runstate, "source",
                        lambda sid: {"source_id": sid, "vertical": "stream",
                                     "local_path": "/src/window.mp4"})
    monkeypatch.setattr(engine, "_niche_of", lambda sid: "gaming")
    monkeypatch.setattr(engine, "_content_kind_of", lambda sid: "screen")
    monkeypatch.setattr(engine, "_measure_black_frozen", lambda *a, **kw: {"passed": True})
    monkeypatch.setattr(engine, "_load_words_sidecar", lambda p: [
        {"w": "hello", "start": 2.0, "end": 2.5}, {"w": "world.", "start": 2.6, "end": 3.5}])

    def vq(mp4, blur_fill_frac=None, layout=None, **k):
        seen["vq"] = (blur_fill_frac, layout)
        return quality_gate.visual_quality_verdict(
            [{"luma": 80.0, "contrast": 45.0, "sharpness": 900.0}], blur_fill_frac, layout=layout)
    monkeypatch.setattr(engine.quality_gate, "measure_visual_quality", vq)
    monkeypatch.setattr(engine, "_hook_judge", lambda: None)

    def checks(mp4, **kw):
        seen["checks"] = kw["layout"]
        return {k: {"passed": True} for k in engine.quality_gate.CLIP_CHECK_KEYS}
    monkeypatch.setattr(engine.quality_gate, "clip_checks", checks)
    clip = {"clip_id": 1, "source_id": "s1_w00", "start_s": 2.0, "end_s": 32.0,
            "layout": "auto", "words_path": "/tmp/w.json", "hook": "Hello world"}
    qc = {"integrated_lufs": -14.1, "layout": "gamefit", "blur_fill_frac": 0.563,
          "banner_guard": {"action": "gamefit"},
          "cam_inset": {"rule": True, "layout": "gamefit", "source_cam": CORNER,
                        "inset": [40, 1400, 518, 480], "ok": True}}
    rep = engine._run_quality_gate(tmp_path / "x.mp4", clip, qc=qc)
    assert rep["verdict"] == "pass", rep["drop_reason"]
    assert seen == {"vq": (0.563, "gamefit"), "checks": "gamefit"}
    assert rep["gameplay_layout"]["passed"] and rep["cam_inset"]["passed"]


def test_main_records_gamefit_and_what_it_replaced_in_the_qc_line(tmp_path, rig, monkeypatch):
    src = _source(tmp_path / "s.mp4", CORNER)
    words = tmp_path / "w.json"
    words.write_text(json.dumps([{"w": "hello", "start": 0.2, "end": 0.8},
                                 {"w": "world.", "start": 0.9, "end": 1.6}]))
    _refusing(monkeypatch, _gamestack(CORNER))
    monkeypatch.setattr(recompose, "measure_levels", lambda *a, **k: None)
    monkeypatch.setattr(recompose.textzones, "hook_zone_text",
                        lambda *a, **k: {"frames": 3, "bands": []})
    out = tmp_path / "o.mp4"
    lines = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: lines.append(" ".join(map(str, a))))
    rc = recompose.main([str(src), "0", "3", str(out), "--captions", "--words", str(words),
                         "--hook", "He fit the whole frame"] + GAMEPLAY[2:])
    assert rc == 0, lines
    assert any(ln.startswith("DONE") and ln.endswith("layout=gamefit") for ln in lines)
    qc = json.loads(next(ln for ln in lines if ln.startswith("QC "))[3:])
    assert qc["layout"] == "gamefit" and qc["passed"]
    assert qc["gamefit"]["replaced"] == REFUSAL and qc["gamefit"]["replaced_layout"] == "gamestack"
    assert qc["gamefit"]["frame"] == [0, 492, 1080, 608]
    assert qc["banner_guard"]["action"] == "gamefit"
    assert qc["blur_fill_frac"] == qc["gamefit"]["shares"]["fill"] < 0.70
    ci = qc["cam_inset"]
    assert ci["inset"] == qc["gamefit"]["card"] and ci["cap_clear"] is True
    assert 1100 < ci["cap_y"] < ci["inset"][1]                     # under the frame, above the card
    assert quality_gate.measure_cam_inset("stream", ci)["passed"] is True


def test_the_engine_keeps_the_layout_and_the_refusal_in_the_render_metadata():
    qc = {"layout": "gamefit", "gamefit": {"replaced": REFUSAL, "replaced_layout": "gamestack",
                                           "card": [40, 1400, 518, 480], "cam_box": CORNER,
                                           "shares": {"fill": 0.563}, "frame": [0, 492, 1080, 608]}}
    meta = engine.render_meta(qc)
    assert meta["layout"] == "gamefit" and meta["gamefit"]["replaced"] == REFUSAL
    assert meta["gamefit"]["replaced_layout"] == "gamestack" and "frame" not in meta["gamefit"]
    note = engine.gamefit_note(qc)
    assert note.startswith("[gamestack] crop would slice") and len(note) <= 254
    assert engine.render_meta({"layout": "gamestack"}) == {"layout": "gamestack"}
    assert re.search("slic", engine.gamefit_note(qc))
