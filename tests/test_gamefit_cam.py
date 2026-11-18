"""The gaming rule is the game zoomed out and the streamer cam as ONE small bottom-corner inset.
gamefit showed the streamer twice (the source's own cam in the fit frame and the card), and on an
edited source whose cam jumps between a corner cam and a zoomed one the static card cut or missed
the face.

  - (a) one cam: the cam region the card shows is covered on the frame with the frame's own
    blurred, darkened fill (never a black box), grown to the overlay's border but never over a
    text item next to it; the render shows the cam once, in the card;
  - (b) a jumping cam: a card that would hold no whole face in more than 20 % of the sampled
    frames is no card: the cam stays in the frame where
    the source has it, uncovered, and the gate's cam check reads it there.
Synthetic sources (lavfi): the cam is a magenta box, and the face detector "sees" a face in the
upper middle of the magenta it finds in a frame; the banner scan and OCR are faked."""
import json
import subprocess

import numpy as np
import pytest

from clipengine import engine
from clipengine import quality_gate
from clipengine.render import gamelayout as gl
from clipengine.render import recompose
from clipengine.render import reframe

W, H = 1280, 720
GAME = ("gradients=s=1280x720:r=30:speed=0.03:seed=7:c0=0x203060:c1=0x406090:c2=0x305070:"
        "c3=0x20406a")
CORNER = [0, 176, 248, 316]          # a corner cam at the frame's left edge
ZOOMED = [0, 300, 560, 720]          # the edited source's zoomed cam (some of its shots)
REFUSAL = ("crop would slice a source graphic or show a blurred fill: crop would slice a source "
           "banner: game band 640px slices banner(s) [[0, 27, 714, 79]] on its path; no clean "
           "streamer split or full-bleed crop")
GAMEPLAY = ["--layout", "auto", "--vertical", "stream", "--niche", "gaming",
            "--content-kind", "screen"]


def _source(path, shots, dur=3):
    """A game gradient with a magenta cam box per shot: `shots` [(t0, t1, box or None)]."""
    boxes = [f"drawbox=x={b[0]}:y={b[1]}:w={b[2] - b[0]}:h={b[3] - b[1]}:color=0xff00ff:t=fill:"
             f"enable='between(t,{t0},{t1 - 0.001})'" for t0, t1, b in shots if b]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", GAME, "-f", "lavfi", "-i",
                    "sine=frequency=440", "-t", str(dur), "-vf", ",".join(boxes) or "null",
                    "-pix_fmt", "yuv420p", "-shortest", str(path)], check=True)
    return path


def _magenta(img):
    return (img[..., 2] > 200) & (img[..., 1] < 80) & (img[..., 0] > 200)


def _fake_faces(frame):
    """The face detector on a synthetic frame: a face in the upper middle of its magenta cam."""
    ys, xs = np.nonzero(_magenta(frame.astype(int)))
    if len(xs) < 50:
        return []
    fh, fw = frame.shape[:2]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    h = 0.4 * (y1 - y0)
    return [{"cx": (x0 + x1) / 2.0 / fw, "cy": (y0 + 0.45 * (y1 - y0)) / fh, "w": 0.8 * h / fw,
             "h": h / fh, "score": 0.9}]


def _args(src="s.mp4", dur="3"):
    return recompose.parse_args([str(src), "0", dur, "o.mp4"] + GAMEPLAY)


def _gamestack(cam=CORNER):
    """The refused v2 gamestack plan: its cam box and the median face in it."""
    x0, y0, x1, y1 = cam
    h = 0.4 * (y1 - y0)
    return {"layout": "gamestack", "v2": True, "src_w": W, "src_h": H, "cam_box": list(cam),
            "face_median": [(x0 + x1) / 2.0 / W, (y0 + 0.45 * (y1 - y0)) / H, h / H],
            "cam_how": "busy", "card": [40, 1414, 532, 466]}


@pytest.fixture
def rig(monkeypatch):
    """The faked face detector, no source bars, the scan / OCR given (an empty one by default),
    the v3 knobs; the refused plan is the CORNER gamestack."""
    recompose._SCANS.clear()
    recompose._CAMS.clear()
    state = {"sc": {"graphics": [], "panels": [], "frames": 13, "src_w": W, "src_h": H,
                    "faces": []}, "words": []}
    monkeypatch.setattr(reframe, "_detector", lambda: ("yunet", None))
    monkeypatch.setattr(reframe, "detect_faces", _fake_faces)
    monkeypatch.setattr(recompose, "_scan", lambda src, a, size: state["sc"])
    monkeypatch.setattr(recompose, "_ocr", lambda src, a, size: state["words"])
    monkeypatch.setattr(recompose, "source_bars", lambda src, a, sw, sh: None)
    monkeypatch.setattr(recompose, "stream_cam", lambda src, a, size: None)
    monkeypatch.setattr(recompose, "_window_luma", lambda src, a, size: 90.0)
    gs = {"v2": True, "game_share": 0.50, "cam_max_area": 0.12, "game_only_share": 0.42,
          "header_h": 200}
    monkeypatch.setattr(recompose, "_gamestack_cfg", lambda: dict(gs))

    def choose(a, src, info, seen):
        seen["plan"] = _gamestack(CORNER)
        raise recompose.RenderError(REFUSAL)
    monkeypatch.setattr(recompose, "_choose_layout", choose)
    return state


def _render(plan, src, out, t):
    """The plan's graph on `src` (ffmpeg), the canvas at clip time `t` as BGR ints."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-filter_complex",
                    recompose.video_graph(plan, _args(src)), "-map", "[v0]", "-pix_fmt", "yuv420p",
                    str(out)], check=True)
    return _frame(out, t, 1080, 1920)


def _frame(path, t, w, h):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True,
                         check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3).astype(int)


# ---------------------------------------------------------------- pure helpers

def test_a_card_holds_a_face_only_when_the_face_is_whole_in_its_cut():
    crop = [100, 100, 300, 300]
    assert gl.face_in([150, 150, 250, 250], crop)
    assert gl.face_in([95, 150, 195, 250], crop)              # 5 px past: the detector's jitter
    assert not gl.face_in([60, 150, 160, 250], crop)          # 40 % of it cut by the card's edge
    assert not gl.face_in([500, 150, 600, 250], crop)         # elsewhere in the frame
    samples = [[[150, 150, 250, 250]]] * 8 + [[[60, 150, 160, 250]], []]
    assert gl.card_face_share(samples, crop) == 0.8
    assert gl.card_face_share([], crop) is None and gl.card_face_share(None, crop) is None


def test_the_cover_is_the_cam_grown_to_its_border_never_over_text_and_cut_to_the_frame():
    frame = [0, 0, W, H]
    cam = [66, 184, 384, 376]                                  # the overlay's border at 14
    rect = [14, 170, 384, 376]
    assert gl.fit_cover(cam, rect, [], frame) == [0, 164, 390, 382]      # ... and the frame edge
    assert gl.fit_cover([166, 184, 484, 376], rect, [], frame)[0] == 160   # far from both: the pad
    assert gl.fit_cover(cam, [14, 170, 384, 700], [], frame)[3] == 382   # a far 'border': no
    hud = [[56, 112, 252, 168]]                                # a HUD line just above the cam
    assert gl.fit_cover(cam, rect, hud, frame) == [0, 170, 390, 382]     # the top stops at it
    chat = [[0, 380, 300, 480]]                                # chat right under the cam
    assert gl.fit_cover(cam, rect, chat, frame)[3] == 376
    assert gl.fit_cover(cam, rect, [[0, 150, 10, 400]], frame)[0] == 14  # text at the frame edge
    assert gl.fit_cover(cam, rect, None, frame) == cam         # text unread: the cam box alone
    docked = [38, 112, 260, 316]                               # docked 38 px off the left edge
    assert gl.fit_cover(docked, [52, 116, 260, 314], [[0, 0, 300, 20], [0, 313, 330, 490]],
                        frame) == [0, 106, 266, 316]           # the chat under it stays whole
    assert gl.fit_cover([0, 176, 248, 316], None, [], frame)[0] == 0     # cut to the frame
    assert gl.fit_cover([0, 0, 120, 720], None, [], [160, 0, 1120, 720]) is None   # in a bar


def test_the_cover_maps_onto_the_fit_frame_and_counts_as_fill():
    box = gl.fit_canvas_box([0, 176, 248, 316], [0, 0, W, H], 492, 608)
    assert box[0] == 0 and box[1] == 492 + 148 and all(v % 2 == 0 for v in box)
    assert box[2] >= 248 * 1080 / 1280.0 and box[3] >= 140 * 608 / 720.0
    card = [40, 1400, 518, 480]
    plain, covered = gl.fit_shares(608, card), gl.fit_shares(608, card, box)
    assert covered["cover"] == pytest.approx(box[2] * box[3] / (1080 * 1920.0), abs=0.001)
    assert covered["fill"] == pytest.approx(plain["fill"] + covered["cover"], abs=0.002)
    assert covered["frame"] == pytest.approx(plain["frame"] - covered["cover"], abs=0.002)


# ---------------------------------------------------------------- (a) one cam on screen

def test_a_gamefit_plan_with_an_in_band_cam_covers_it_and_the_card_shows_it_once(tmp_path, rig):
    src = _source(tmp_path / "s.mp4", [(0, 3, CORNER)])
    plan, guard = recompose.choose_layout(_args(src), str(src), {"width": W, "height": H})
    assert plan["layout"] == "gamefit" and guard["action"] == "gamefit"
    assert plan["card_faces"]["share"] == 1.0 and plan["card"]
    cv = plan["cam_cover"]                                     # the card's cam region, covered
    assert cv[0] <= CORNER[0] and cv[1] <= CORNER[1] and cv[2] >= CORNER[2] and \
        cv[3] >= CORNER[3]
    assert plan["cam_cover_canvas"] == gl.fit_canvas_box(cv, [0, 0, W, H], 492, 608)
    assert plan.get("cam_in_band") is None
    ci = plan["cam_inset"]                                     # the gate reads one inset: the card
    assert ci["source_cam"] == CORNER and ci["inset"] == plan["card"] and ci["ok"]
    assert quality_gate.measure_cam_inset("stream", ci)["passed"] is True
    assert recompose.blur_fill_frac(plan, _args(src), {}) == \
        gl.fit_shares(608, plan["card"], plan["cam_cover_canvas"])["fill"] < 0.70
    img = _render(plan, src, tmp_path / "o.mp4", 1.5)
    mag = _magenta(img)
    y0, y1 = plan["frame_y"], plan["frame_y"] + plan["frame_h"]
    kx, ky, kw, kh = plan["card"]
    assert mag[ky + 10:ky + kh - 10, kx + 10:kx + kw - 10].mean() > 0.9   # the cam: in the card
    assert mag[y0:y1].sum() == 0                               # ... and nowhere on the frame
    assert mag.sum() < 1.05 * kw * kh                          # once on screen
    vx, vy, vw, vh = plan["cam_cover_canvas"]
    patch = img[vy + 2:vy + vh - 2, vx + 2:vx + vw - 2]
    k = plan["fill_dark"]                                      # the fill alone, as gamefit_vf draws it
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1.5", "-i", str(src), "-frames:v", "1",
                    "-vf", f"scale=270:480,gblur=sigma={gl.FILL_SIGMA},"
                    f"colorchannelmixer=rr={k}:gg={k}:bb={k},scale=1080:1920",
                    str(tmp_path / "fill.png")], check=True)
    fill = _frame(tmp_path / "fill.png", 0, 1080, 1920)[vy + 2:vy + vh - 2, vx + 2:vx + vw - 2]
    assert patch.mean() > 5 and np.abs(patch - fill).mean() < 8   # the frame's fill, not a box
    src_px = _frame(src, 1.5, W, H)                            # the rest of the frame as it was
    fr = np.array([[src_px[int(y * H / 608.0), int(x * W / 1080.0)] for x in range(0, 1080, 8)]
                   for y in range(0, 608, 8)])
    shown = img[y0:y1:8, 0:1080:8]
    keep = np.ones(shown.shape[:2], bool)
    keep[max(0, (vy - y0) // 8 - 1):(vy - y0 + vh) // 8 + 2, :(vx + vw) // 8 + 2] = False
    assert np.abs(shown[keep] - fr[keep]).mean() < 8


def test_main_records_the_cover_in_the_qc_line(tmp_path, rig, monkeypatch):
    src = _source(tmp_path / "s.mp4", [(0, 3, CORNER)])
    words = tmp_path / "w.json"
    words.write_text(json.dumps([{"w": "hello", "start": 0.2, "end": 0.8},
                                 {"w": "world.", "start": 0.9, "end": 1.6}]))
    monkeypatch.setattr(recompose, "measure_levels", lambda *a, **k: None)
    monkeypatch.setattr(recompose.textzones, "hook_zone_text",
                        lambda *a, **k: {"frames": 3, "bands": []})
    lines = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: lines.append(" ".join(map(str, a))))
    assert recompose.main([str(src), "0", "3", str(tmp_path / "o.mp4"), "--captions", "--words",
                           str(words), "--hook", "One cam on screen"] + GAMEPLAY[2:]) == 0, lines
    qc = json.loads(next(ln for ln in lines if ln.startswith("QC "))[3:])
    gf = qc["gamefit"]
    assert qc["layout"] == "gamefit" and gf["cam_cover"] and gf["cam_in_band"] is None
    assert gf["card_faces"]["share"] == 1.0 and gf["shares"]["cover"] > 0
    assert qc["blur_fill_frac"] == gf["shares"]["fill"]
    assert qc["cam_inset"]["inset"] == gf["card"]
    meta = engine.render_meta(qc)["gamefit"]
    assert meta["cam_cover"] == gf["cam_cover"] and meta["card_faces"]["share"] == 1.0


def test_the_cover_leaves_a_hud_line_next_to_the_cam_whole(tmp_path, rig):
    """Do not hide HUD, chat or inventory: the cover grows past the cam box (its border line) only
    over rows / columns with no text item."""
    rig["sc"]["graphics"] = [{"box": [0, 318, 300, 340], "frac": 1.0, "banner": False}]
    rig["words"] = [(10, 320, 290, 338, "CHATLINE", 0)]
    src = _source(tmp_path / "s.mp4", [(0, 3, CORNER)])
    plan, _g = recompose.choose_layout(_args(src), str(src), {"width": W, "height": H})
    assert plan["cam_cover"][3] == CORNER[3]                   # the chat line under it stays
    assert plan["cam_cover"][2] > CORNER[2]                    # the free side takes the border pad


def test_unread_faces_keep_the_card_and_still_cover_the_cam(rig, monkeypatch):
    monkeypatch.setattr(reframe, "_detector", lambda: ("none", None))
    plan, _g = recompose.choose_layout(_args(), "s.mp4", {"width": W, "height": H})
    assert plan["card"] and plan["card_faces"]["share"] is None and plan["cam_cover"]


# ---------------------------------------------------------------- (b) a jumping cam

JUMPS = [(0, 1, CORNER), (1, 2, ZOOMED), (2, 3, None)]   # corner cam, zoomed cam, a game-only shot


def test_a_jumping_cam_gets_no_card_and_keeps_its_cam_in_the_frame(tmp_path, rig):
    src = _source(tmp_path / "s.mp4", JUMPS)
    plan, guard = recompose.choose_layout(_args(src), str(src), {"width": W, "height": H})
    assert plan["layout"] == "gamefit" and guard["action"] == "gamefit"
    held = plan["card_faces"]
    assert held["share"] < gl.FIT_CARD_FACE_MIN and held["samples"] == recompose.GAMEFIT_FACE_SAMPLES
    assert plan["card"] is None and plan["cam"] is None        # no inset: no card frame without
    assert plan.get("cam_cover") is None                       # a face; the cam not covered
    assert plan["cam_in_band"] == gl.fit_canvas_box(CORNER, [0, 0, W, H], 492, 608)
    ci = plan["cam_inset"]                                     # the gate reads the cam in the frame
    assert ci["source_cam"] == CORNER and ci["inset"] == plan["cam_in_band"] and ci["in_band"]
    assert ci["over"] and not ci["ok"]                         # it is no bottom-corner card
    assert quality_gate.measure_cam_inset("stream", ci)["passed"] is True
    assert recompose.blur_fill_frac(plan, _args(src), {}) == gl.fit_shares(608)["fill"] < 0.70
    assert recompose.crop_views(plan, 3.0) == []               # nothing of ours cuts the frame
    out = tmp_path / "o.mp4"
    _render(plan, src, out, 0.5)
    y0, y1 = plan["frame_y"], plan["frame_y"] + plan["frame_h"]
    for t in (0.5, 1.5, 2.5):                                  # every shot: the source as it is
        mag = _magenta(_frame(out, t, 1080, 1920))
        assert mag[:y0].sum() == 0 and mag[y1:].sum() == 0     # no card anywhere
        assert (mag[y0:y1].sum() > 0) == (t < 2.0)             # the in-band cam, uncovered


def test_a_static_cam_in_the_same_source_keeps_its_card(tmp_path, rig):
    """The rule reads the card's faces, not the source kind: the same corner cam held for the
    whole window keeps the card (and its cover)."""
    src = _source(tmp_path / "s.mp4", [(0, 3, CORNER)])
    plan, _g = recompose.choose_layout(_args(src), str(src), {"width": W, "height": H})
    assert plan["card"] and plan["card_faces"]["share"] >= gl.FIT_CARD_FACE_MIN
    assert plan["cam_cover"] and not plan.get("cam_in_band")


def test_a_jumping_cam_outside_the_shown_frame_keeps_the_refusal(tmp_path, rig, monkeypatch):
    """No card and no cam in the frame would drop the cam: never; the refusal stands."""
    monkeypatch.setattr(recompose, "source_bars", lambda src, a, sw, sh: {
        "x": 260, "y": 0, "w": 1020, "h": 720, "bars": {"left": 260}})
    src = _source(tmp_path / "s.mp4", JUMPS)
    with pytest.raises(recompose.RenderError) as e:
        recompose.choose_layout(_args(src), str(src), {"width": W, "height": H})
    assert str(e.value).startswith("gamefit: the card's cut of the cam") and \
        str(e.value).endswith(REFUSAL)
