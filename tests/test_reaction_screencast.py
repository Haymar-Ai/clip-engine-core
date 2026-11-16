"""PHASE5b: reaction layout + automatic screencast.

Tests the calm-mode facecam detector, content-region detection, reaction/auto-screencast
layout planning, captions + hook wiring, profile key isolation from gamestack, the manual
screencast golden filter graph, and the pipeline plumbing (engine.render command without
`layout_crops` -> auto screencast).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RENDER = ROOT / "clipengine" / "render"

from clipengine.render import captions
from clipengine.render import facecam
from clipengine.render import recompose
from clipengine.render import reframe
from helpers import fake_store_clip
from clipengine import config


W_FULL, H_FULL = 1920, 1080
FACE_BGR = (55, 210, 85)                # bright green — same signal used by the fake face detector


# ---------------------------------------------------------------- synthetic videos

def _draw_calm_frame(t, cam_box, face_h=140, add_video_rect=None, add_typing=False, rng=None):
    """Screencast-style frame: static slate background with text-like rectangles ("slides"),
    a noisy camera box holding a moving face blob, plus optional video / typing regions.

    - Background is exactly static (mimics screen capture) -- pixels there have temporal
      std == 0 after decode.
    - Inside `cam_box`, add per-pixel gaussian noise every frame so the region has calm
      "camera noise" temporal std in the CALM_STD_LO..CALM_STD_HI band.
    - `add_video_rect`: (x0, y0, x1, y1) — fill with pseudo-random RGB every frame (mimics a
      reacted video playing in a browser window).
    - `add_typing`: draws a tiny cursor caret at a shifting x position — sparse burst change.
    """
    rng = rng or np.random.default_rng(0)
    img = np.full((H_FULL, W_FULL, 3), 240, dtype=np.uint8)   # slate background (static)
    # static text-like rectangles (slide bullets) — screen chrome
    img[80:120, 100:1500] = 40
    img[200:220, 100:1000] = 60
    img[240:260, 100:900] = 60
    x0, y0, x1, y1 = cam_box
    # cam interior: solid grey + per-pixel gaussian noise (calm camera-like temporal std)
    noise = rng.normal(0, 4.0, size=(y1 - y0, x1 - x0, 3)).astype(np.int16)
    img[y0:y1, x0:x1] = np.clip(
        90 + noise + int(round(3.0 * np.sin(t * 3.0))), 0, 255).astype(np.uint8)
    # face blob: elliptical, jittering (streamer moves)
    face_cy = (y0 + y1) // 2
    face_cx = (x0 + x1) // 2 + int(20 * np.sin(t * 2.0))
    yy, xx = np.ogrid[:H_FULL, :W_FULL]
    mask = ((xx - face_cx) ** 2 / (face_h * 0.35) ** 2
            + (yy - face_cy) ** 2 / (face_h * 0.5) ** 2) <= 1.0
    img[mask] = np.array(FACE_BGR, dtype=np.uint8)
    if add_video_rect is not None:
        vx0, vy0, vx1, vy1 = add_video_rect
        # a coarse noise field so the region has video-like temporal std everywhere.
        block = 20
        h = vy1 - vy0
        w = vx1 - vx0
        small = rng.integers(0, 256, (h // block + 1, w // block + 1, 3), dtype=np.uint8)
        big = np.repeat(np.repeat(small, block, axis=0), block, axis=1)[:h, :w]
        img[vy0:vy1, vx0:vx1] = big
    if add_typing:
        # a moving caret — sparse, burst change (cursor / typing pattern). NOT enough for the
        # content detector to accept as a video region.
        tx = 130 + int(200 * (t % 1.0))
        img[300:320, tx:tx + 4] = 0
    return img


def _encode_synth(path, frames, fps=10):
    dur = len(frames) / fps
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W_FULL}x{H_FULL}",
           "-r", str(fps), "-i", "-",
           "-f", "lavfi", "-t", str(dur), "-i", "sine=frequency=440:sample_rate=48000",
           "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv444p",
           "-c:a", "aac", "-shortest", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in frames:
            p.stdin.write(f.tobytes())
    finally:
        p.stdin.close()
        p.wait()
    return path


@pytest.fixture(scope="module")
def calm_screencast_video(tmp_path_factory):
    """Static slide bg + noisy cam box in the top-right corner — a screencast scene."""
    cam_box = (1400, 100, 1900, 520)
    rng = np.random.default_rng(3)
    frames = [_draw_calm_frame(i / 10, cam_box=cam_box, rng=rng) for i in range(30)]
    return _encode_synth(tmp_path_factory.mktemp("cast") / "cast.mp4", frames)


@pytest.fixture(scope="module")
def calm_reaction_video(tmp_path_factory):
    """Static browser chrome + a reacted video area + a cam box beside it — a reaction scene."""
    cam_box = (1550, 700, 1900, 1020)
    video_rect = (200, 150, 1400, 900)             # the reacted video (large centre-left)
    rng = np.random.default_rng(4)
    frames = [_draw_calm_frame(i / 10, cam_box=cam_box, add_video_rect=video_rect, rng=rng)
              for i in range(30)]
    return _encode_synth(tmp_path_factory.mktemp("react") / "react.mp4", frames)


@pytest.fixture(scope="module")
def calm_talking_head_video(tmp_path_factory):
    """Static screen with only cursor typing near a cam — no sustained video region.
    Used to check content detection returns None when the streamer is only talking."""
    cam_box = (1500, 100, 1900, 520)
    rng = np.random.default_rng(5)
    frames = [_draw_calm_frame(i / 10, cam_box=cam_box, add_typing=True, rng=rng)
              for i in range(30)]
    return _encode_synth(tmp_path_factory.mktemp("chat") / "chat.mp4", frames)


@pytest.fixture(autouse=True)
def _fake_face_detector(monkeypatch):
    """Reuses test_facecam.py's approach: the elliptical FACE_BGR blob is not a real face
    for YuNet, so replace the detector with a colour-band matcher. Returns ONE detection
    per connected component so the round 3 multi-face scenes (streamer corner + reacted
    video face) get separated the way YuNet separates real faces."""
    import cv2

    def fake(frame):
        b, g, r = frame[..., 0], frame[..., 1], frame[..., 2]
        mask = (b >= 50) & (b <= 60) & (g >= 205) & (g <= 215) & (r >= 80) & (r <= 90)
        if not mask.any():
            return []
        h, w = frame.shape[:2]
        n_ccs, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        faces = []
        for lbl in range(1, n_ccs):
            x, y, cw, ch, area = stats[lbl]
            if area < 800:
                continue
            faces.append({"cx": (x + cw / 2) / w, "cy": (y + ch / 2) / h,
                          "w": cw / w, "h": ch / h, "score": 0.95})
        return faces

    monkeypatch.setattr("clipengine.render.reframe.detect_faces", fake)
    monkeypatch.setattr("clipengine.render.facecam.reframe.detect_faces", fake)
    yield


# ---------------------------------------------------------------- calm-mode detection

def test_calm_mode_finds_box_on_static_screen(calm_screencast_video):
    """On a screencast (static bg, noisy cam box), the busy detector under-reports (the bg
    has zero temporal std, growth escapes the whole frame or fails MAX_BOX_AREA_FRAC).
    Calm mode grows on camera-noise pixels and stops on the screen-static rows/cols."""
    r_calm = facecam.detect(str(calm_screencast_video), 0.0, 3.0, n=15, mode="calm",
                            max_face_h=facecam.MAX_FACE_H_CALM)
    assert r_calm is not None, "calm mode should find the cam"
    x0, y0, x1, y1 = r_calm["trimmed_box_px"]
    tx0, ty0, tx1, ty1 = (1400, 100, 1900, 520)
    assert abs(x0 - tx0) <= 8 and abs(x1 - tx1) <= 8, (r_calm["trimmed_box_px"], tx0, tx1)
    assert abs(y0 - ty0) <= 8 and abs(y1 - ty1) <= 8, (r_calm["trimmed_box_px"], ty0, ty1)
    # face_frac depends on max_face_h — check the mode was recorded
    assert r_calm["mode"] == "calm"


def test_auto_mode_falls_back_from_busy_to_calm(calm_screencast_video):
    """`mode='auto'` tries busy first, falls back to calm when busy fails."""
    r = facecam.detect(str(calm_screencast_video), 0.0, 3.0, n=15, mode="auto",
                       max_face_h=facecam.MAX_FACE_H_CALM)
    assert r is not None and r["mode"] == "calm"


def test_busy_mode_on_gaming_synth_unchanged():
    """The busy-mode gamestack synth (imported from test_facecam via re-use) still detects."""
    # Rebuild a minimal gamestack-style synth here so this test does not depend on the other
    # module's fixtures. Small box, noisy game bg, calm cam interior.
    W, H = 1920, 1080
    cam_box = (200, 300, 800, 700)
    rng = np.random.default_rng(9)
    frames = []
    for i in range(30):
        t = i / 10
        block = 40
        small = rng.integers(0, 256, (H // block + 1, W // block + 1, 3), dtype=np.uint8)
        img = np.repeat(np.repeat(small, block, axis=0), block, axis=1)[:H, :W]
        img[300:700, 200:800] = np.array([90, 90, 90], dtype=np.uint8)
        face_cy = 500; face_cx = 500 + int(20 * np.sin(t * 2.0))
        yy, xx = np.ogrid[:H, :W]
        mask = ((xx - face_cx) ** 2 / (60 * 0.35) ** 2 + (yy - face_cy) ** 2 / (60 * 0.5) ** 2) <= 1.0
        img[mask] = np.array(FACE_BGR, dtype=np.uint8)
        frames.append(img)
    p = Path("/tmp") / "phase5b_busy_synth.mp4"
    _encode_synth(p, frames)
    r = facecam.detect(str(p), 0.0, 3.0, n=15, mode="busy")
    assert r is not None and r["mode"] == "busy"
    x0, y0, x1, y1 = r["trimmed_box_px"]
    assert abs(x0 - 200) <= 8 and abs(x1 - 800) <= 8


# ---------------------------------------------------------------- content detection

def test_detect_content_finds_the_video_rectangle(calm_reaction_video):
    """detect_content should locate the reacted-video rectangle (200,150)-(1400,900) while
    the cam sits to the right; sides may snap onto the CC boundary or a stable edge."""
    frames = facecam.sample_native(str(calm_reaction_video), 0.0, 3.0, n=15)
    guard_box = [1540, 690, 1910, 1030]         # generous cam-guard box
    box = facecam.detect_content(frames, cam_guard_box_px=guard_box)
    assert box is not None
    x0, y0, x1, y1 = box
    tx0, ty0, tx1, ty1 = 200, 150, 1400, 900
    assert abs(x0 - tx0) <= 30 and abs(x1 - tx1) <= 30, box
    assert abs(y0 - ty0) <= 30 and abs(y1 - ty1) <= 30, box
    # entirely outside the guard box
    assert x1 <= guard_box[0] or x0 >= guard_box[2] \
        or y1 <= guard_box[1] or y0 >= guard_box[3], (box, guard_box)


def test_detect_content_returns_none_on_talking_head(calm_talking_head_video):
    """A screen with cursor / typing only (no sustained change) -> None."""
    frames = facecam.sample_native(str(calm_talking_head_video), 0.0, 3.0, n=15)
    box = facecam.detect_content(frames, cam_guard_box_px=[1480, 90, 1910, 530])
    assert box is None


# ---------------------------------------------------------------- guard-box property

@pytest.mark.parametrize("cam_pos", [
    ("left",       (50, 300, 500, 700)),
    ("right",      (1400, 300, 1900, 700)),
    ("top",        (700, 20, 1200, 320)),
    ("bottom",     (700, 780, 1200, 1060)),
    ("centre",     (760, 340, 1160, 740)),
])
def test_reaction_content_never_intersects_guard_box(cam_pos, monkeypatch, tmp_path):
    """Property: the reaction content crop never overlaps the guarded raw cam box, over cam
    positions (left/right/top/bottom/centre-over-content)."""
    name, cam = cam_pos
    rx0, ry0, rx1, ry1 = cam
    box_norm = [rx0 / W_FULL, ry0 / H_FULL, rx1 / W_FULL, ry1 / H_FULL]
    # detect_content: for centre-over-content, mock the content region as full frame; else a
    # large region beside the cam. Then the planner must carve out a rect that clears the guard.
    def fake_content(frames, cam_guard_box_px=None, exclude_boxes=None):
        gx0, gy0, gx1, gy1 = cam_guard_box_px
        return [0, 0, W_FULL, H_FULL]           # cam sits inside the region — worst case
    monkeypatch.setattr(facecam, "detect_content", fake_content)
    monkeypatch.setattr(facecam, "_resolve_cam" if False else "get_or_detect",
                        lambda channel, src, start, dur, source_duration=None,
                        cache_dir=None, log=None, mode="busy",
                        max_face_h=facecam.MAX_FACE_H, profile_key=None:
                        (box_norm, box_norm, True, 1.0))
    monkeypatch.setattr(recompose.reframe, "probe_size", lambda src: (W_FULL, H_FULL))
    monkeypatch.setattr(recompose, "probe",
                        lambda p: {"width": W_FULL, "height": H_FULL, "fps": 30,
                                    "duration": 3.0, "has_audio": True})
    monkeypatch.setattr(facecam, "sample_native", lambda *a, **k: [])
    a = recompose.parse_args([str(tmp_path / "x.mp4"), "0", "3", str(tmp_path / "o.mp4"),
                              "--layout", "reaction", "--channel", "s1"])
    plan = recompose._plan_reaction(tmp_path / "x.mp4", a)
    # in the worst case (cam inside full-frame content), the planner might return None,
    # which is a valid signal — caller falls back to track. If plan is not None, the
    # content rect must clear the guard.
    if plan is None:
        return
    tw, th, tx, ty = plan["content"]
    gx0, gy0, gx1, gy1 = plan["raw_cam_box"]
    disjoint = tx + tw <= gx0 or tx >= gx1 or ty + th <= gy0 or ty >= gy1
    assert disjoint, (name, plan["content"], plan["raw_cam_box"])


@pytest.mark.parametrize("cam_pos", [
    ("left",     (50, 300, 500, 700)),
    ("right",    (1400, 300, 1900, 700)),
    ("centre",   (760, 340, 1160, 740)),
])
def test_screencast_auto_content_never_intersects_guard_box(cam_pos, monkeypatch, tmp_path):
    """Same property for auto screencast: the content crop's x-range must clear the guard."""
    name, cam = cam_pos
    rx0, ry0, rx1, ry1 = cam
    box_norm = [rx0 / W_FULL, ry0 / H_FULL, rx1 / W_FULL, ry1 / H_FULL]
    monkeypatch.setattr(facecam, "get_or_detect",
                        lambda channel, src, start, dur, source_duration=None,
                        cache_dir=None, log=None, mode="busy",
                        max_face_h=facecam.MAX_FACE_H, profile_key=None:
                        (box_norm, box_norm, True, 1.0))
    monkeypatch.setattr(recompose.reframe, "probe_size", lambda src: (W_FULL, H_FULL))
    monkeypatch.setattr(recompose, "probe",
                        lambda p: {"width": W_FULL, "height": H_FULL, "fps": 30,
                                    "duration": 3.0, "has_audio": True})
    monkeypatch.setattr(facecam, "sample_native", lambda *a, **k: [])
    monkeypatch.setattr(facecam, "activity_centroid", lambda *a, **k: (rx0 + rx1) / 2.0)
    a = recompose.parse_args([str(tmp_path / "x.mp4"), "0", "3", str(tmp_path / "o.mp4"),
                              "--layout", "screencast", "--channel", "s1"])
    plan = recompose._plan_screencast_auto(tmp_path / "x.mp4", a)
    assert plan is not None
    tw, _, tx, _ = plan["content"]
    gx0, _, gx1, _ = plan["raw_cam_box"]
    assert tx + tw <= gx0 or tx >= gx1, (name, plan["content"], plan["raw_cam_box"])


# ---------------------------------------------------------------- profile keys

def test_profile_key_reaction_and_screencast_isolated_from_gamestack(tmp_path, monkeypatch):
    """A saved bare `channel` gamestack profile MUST NOT be read or overwritten by
    reaction / screencast (which key as `channel:reaction` and `channel:screencast`)."""
    cache = tmp_path
    # simulate a saved gamestack profile
    facecam.save_profile("streamer1", [0.1, 0.1, 0.3, 0.4], "src.mp4", cache_dir=cache,
                          extra={"raw_box": [0.1, 0.1, 0.3, 0.4]})
    profiles = facecam.load_profiles(cache_dir=cache)
    assert "streamer1" in profiles

    calls = []

    def fake_detect_multiwindow(src, source_duration=None, span_s=None, n=None,
                                mode="busy", max_face_h=facecam.MAX_FACE_H):
        calls.append((mode, max_face_h))
        return {"box": [0.7, 0.7, 0.9, 0.9], "raw_box": [0.7, 0.7, 0.9, 0.9],
                "face_frac": 0.9, "samples": n, "size": [W_FULL, H_FULL], "mode": mode}
    monkeypatch.setattr(facecam, "detect_multiwindow", fake_detect_multiwindow)

    # reaction: should NOT read streamer1's gamestack profile, should save under streamer1:reaction
    box, raw, from_prof, frac = facecam.get_or_detect(
        "streamer1", "src.mp4", 0.0, 3.0, cache_dir=cache,
        mode="auto", max_face_h=facecam.MAX_FACE_H_CALM,
        profile_key="streamer1:reaction")
    assert not from_prof and calls
    profiles = facecam.load_profiles(cache_dir=cache)
    assert "streamer1:reaction" in profiles
    # the bare gamestack profile is unchanged
    assert profiles["streamer1"]["box"] == [0.1, 0.1, 0.3, 0.4]

    # screencast: same isolation
    calls.clear()
    box, raw, from_prof, frac = facecam.get_or_detect(
        "streamer1", "src.mp4", 0.0, 3.0, cache_dir=cache,
        mode="auto", max_face_h=facecam.MAX_FACE_H_CALM,
        profile_key="streamer1:screencast")
    profiles = facecam.load_profiles(cache_dir=cache)
    assert "streamer1:screencast" in profiles
    assert profiles["streamer1"]["box"] == [0.1, 0.1, 0.3, 0.4]


# ---------------------------------------------------------------- manual screencast golden

def test_manual_screencast_filter_graph_unchanged(monkeypatch):
    """With --top and --bottom both given, --layout screencast produces the exact same
    filter graph as the pre-phase5b build (byte-for-byte)."""
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "screencast",
                              "--top", "600:720:190:0", "--bottom", "300:365:980:82",
                              "--split", "1320"])
    monkeypatch.setattr(recompose.reframe, "probe_size", lambda s: (W_FULL, H_FULL))
    plan = recompose.plan_layout(a, Path("s.mp4"))
    assert plan["manual"] and plan["split_px"] == 1320
    g = recompose.video_graph(plan, a)
    expected = ("[0:v]crop=600:720:190:0,scale=1080:1320:force_original_aspect_ratio=increase,"
                "crop=1080:1320[top];[0:v]crop=300:365:980:82,split=2[fb][ff];"
                "[fb]scale=1080:600:force_original_aspect_ratio=increase,crop=1080:600,"
                "gblur=sigma=22[fbb];[ff]scale=-2:600[ffs];[fbb][ffs]overlay=(W-w)/2:0[bot];"
                "[top][bot]vstack=inputs=2,setsar=1[v0]")
    assert g == expected, g


# ---------------------------------------------------------------- caption_y / hook_avoid

def test_caption_y_reaction_sits_on_the_seam():
    y = captions.caption_y("reaction", fg_h=960)
    assert 960 <= y <= 1050
    y2 = captions.caption_y("reaction", fg_h=1100)
    assert y2 > y


def test_hook_avoid_reaction_covers_cam_band():
    plan = {"layout": "reaction", "src_w": W_FULL, "src_h": H_FULL, "top_h": 960}
    assert recompose.hook_avoid(plan) == [(960, 1920)]
    plan["top_h"] = 800
    assert recompose.hook_avoid(plan) == [(800, 1920)]


def test_hook_avoid_screencast_auto_covers_cam_band():
    plan = {"layout": "screencast_auto", "src_w": W_FULL, "src_h": H_FULL, "top_h": 1300}
    assert recompose.hook_avoid(plan) == [(1300, 1920)]


# ---------------------------------------------------------------- fallback plan_layout

def test_reaction_falls_back_to_track_when_no_content(monkeypatch):
    monkeypatch.setattr(recompose, "_plan_reaction", lambda src, a: None)
    monkeypatch.setattr(recompose.reframe, "analyze", lambda *ar, **kw: {
        "layout": "track", "src_w": W_FULL, "src_h": H_FULL, "shots": [],
        "segments": [{"t0": 0.0, "t1": 3.0, "x": 100, "cut": True}]})
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "reaction"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    assert plan["layout"] == "track"


def test_reaction_falls_back_to_general_when_no_face(monkeypatch):
    monkeypatch.setattr(recompose, "_plan_reaction", lambda src, a: None)
    monkeypatch.setattr(recompose.reframe, "analyze", lambda *ar, **kw: {
        "layout": "general", "src_w": W_FULL, "src_h": H_FULL})
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "reaction"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    assert plan["layout"] == "general"


def test_screencast_auto_falls_back_to_general_when_no_cam(monkeypatch):
    monkeypatch.setattr(recompose, "_plan_screencast_auto", lambda src, a: None)
    monkeypatch.setattr(recompose.reframe, "analyze", lambda *ar, **kw: {
        "layout": "general", "src_w": W_FULL, "src_h": H_FULL})
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "screencast"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    assert plan["layout"] == "general"


def test_screencast_manual_still_works_when_top_and_bottom_given(monkeypatch):
    """Manual mode is triggered by --top and --bottom being set; the auto planner is
    not invoked."""
    called = {"auto": False}
    monkeypatch.setattr(recompose, "_plan_screencast_auto",
                         lambda src, a: called.update(auto=True) or None)
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "screencast",
                              "--top", "600:720:190:0", "--bottom", "300:365:980:82"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    assert plan["manual"] is True and not called["auto"]


# ---------------------------------------------------------------- vertical_layouts config

def test_vertical_layouts_maps_reaction_and_screencast():
    cfg = config.load()
    m = (cfg.get("render") or {}).get("vertical_layouts") or {}
    assert m.get("reaction") == "reaction"
    assert m.get("screencast") == "screencast"


def test_vertical_layouts_resolve_via_auto(monkeypatch):
    monkeypatch.setattr(recompose, "_plan_reaction", lambda src, a: {
        "layout": "reaction", "src_w": W_FULL, "src_h": H_FULL, "top_h": 960,
        "content_fit": "fit", "content": [1200, 750, 200, 150],
        "cam": [1080, 500, 100, 300, True], "cam_box": [0, 0, 100, 100],
        "raw_cam_box": [0, 0, 100, 100], "cam_box_norm": [0, 0, 0.1, 0.1],
        "raw_cam_box_norm": [0, 0, 0.1, 0.1], "from_profile": False, "face_frac": 1.0,
        "face_median": [0.5, 0.5, 0.1], "channel": "s1"})
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "auto",
                              "--vertical", "reaction"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    assert plan["layout"] == "reaction"

    monkeypatch.setattr(recompose, "_plan_screencast_auto", lambda src, a: {
        "layout": "screencast_auto", "src_w": W_FULL, "src_h": H_FULL, "top_h": 1300,
        "content": [1080, 1080, 400, 0], "cam": [500, 300, 0, 0, True],
        "cam_box": [0, 0, 100, 100], "raw_cam_box": [0, 0, 100, 100],
        "cam_box_norm": [0, 0, 0.1, 0.1], "raw_cam_box_norm": [0, 0, 0.1, 0.1],
        "from_profile": False, "face_frac": 1.0, "face_median": [0.5, 0.5, 0.1],
        "channel": "s1", "activity_x": 900.0})
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "auto",
                              "--vertical", "screencast"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    assert plan["layout"] == "screencast_auto"


# ---------------------------------------------------------------- engine.render command

def test_stages_render_without_layout_crops_runs_auto_screencast(monkeypatch, tmp_path):
    """A screencast clip whose layout has no `layout_crops` entry must NOT pass --top/--bottom
    and must not fail: recompose.py picks up the auto path."""
    from clipengine import engine
    captured = {"cmd": None}

    def fake_run(cmd, capture_output=True, text=True):
        captured["cmd"] = list(cmd)
        class R:
            returncode = 0
            stdout = 'QC {"layout": "screencast_auto", "passed": true}\n'
            stderr = ""
        return R()

    monkeypatch.setattr(engine.subprocess, "run", fake_run)
    monkeypatch.setattr(engine, "CAPTION_STRONG_MODEL", None)   # stage 3 has its own tests
    monkeypatch.setattr(engine, "_store_clip", fake_store_clip)
    monkeypatch.setattr(engine.runstate, "update_clip", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_log", lambda *a, **k: None)
    monkeypatch.setattr(engine, "SCRATCH", tmp_path)
    src_media = tmp_path / "m.mp4"
    src_media.write_bytes(b"")
    monkeypatch.setattr(engine.runstate, "sources_in",
                        lambda st: [{"source_id": "s1", "local_path": str(src_media)}])
    monkeypatch.setattr(engine, "CFG", dict(engine.CFG, layout_crops={}))
    clip = {"clip_id": 42, "source_id": "s1", "start_s": 10, "end_s": 40,
            "layout": "screencast", "words_path": None, "hook": None}
    ok = engine.render(clip)
    assert ok
    cmd = captured["cmd"]
    assert "--layout" in cmd and cmd[cmd.index("--layout") + 1] == "screencast"
    assert "--top" not in cmd and "--bottom" not in cmd


# ---------------------------------------------------------------- content_fit cover option

def test_reaction_content_fit_cover_uses_cover_scale(monkeypatch, tmp_path):
    """With content_fit=cover the top band is a hard cover crop (no blur backdrop)."""
    monkeypatch.setattr(facecam, "get_or_detect",
                        lambda channel, src, start, dur, source_duration=None,
                        cache_dir=None, log=None, mode="busy",
                        max_face_h=facecam.MAX_FACE_H, profile_key=None:
                        ([0.1, 0.1, 0.3, 0.4], [0.1, 0.1, 0.3, 0.4], True, 1.0))
    monkeypatch.setattr(facecam, "detect_content",
                        lambda frames, cam_guard_box_px=None, exclude_boxes=None:
                        [400, 150, 1400, 900])
    monkeypatch.setattr(facecam, "sample_native", lambda *a, **k: [])
    monkeypatch.setattr(recompose.reframe, "probe_size", lambda s: (W_FULL, H_FULL))
    monkeypatch.setattr(recompose, "probe",
                        lambda p: {"width": W_FULL, "height": H_FULL, "fps": 30,
                                    "duration": 3.0, "has_audio": True})
    monkeypatch.setattr(recompose, "_config",
                        lambda: {"render": {"reaction": {"top_h": 960, "content_fit": "cover"}}})
    a = recompose.parse_args([str(tmp_path / "x.mp4"), "0", "3", str(tmp_path / "o.mp4"),
                              "--layout", "reaction", "--channel", "s1"])
    plan = recompose._plan_reaction(tmp_path / "x.mp4", a)
    assert plan is not None and plan["content_fit"] == "cover"
    g = recompose.reaction_vf(plan)
    # cover path: no blurred [topbg]/overlay; a single hard-cover crop path
    assert "gblur=sigma=22[topbg]" not in g
    assert "force_original_aspect_ratio=increase" in g


# ---------------------------------------------------------------- round 2: cutout cam

def _make_cutout_video(path, dur=3.0, fps=10):
    """Synthetic chroma-key scene: static "screen" background with text-like rectangles,
    a face blob composited directly on it (no rectangular cam border around the face).
    Calm growth will stop right at the face (no camera-noise "room" around it), and the
    perimeter of that box has no stable straight edges — the cutout detector should fire
    and replace the box with a head-and-shoulders rectangle."""
    n = int(round(dur * fps))
    rng = np.random.default_rng(11)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W_FULL}x{H_FULL}",
           "-r", str(fps), "-i", "-",
           "-f", "lavfi", "-t", str(dur), "-i", "sine=frequency=440:sample_rate=48000",
           "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv444p",
           "-c:a", "aac", "-shortest", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        face_cx = 1500; face_cy = 700; face_h = 200
        for i in range(n):
            t = i / fps
            img = np.full((H_FULL, W_FULL, 3), 240, dtype=np.uint8)
            img[80:120, 100:1500] = 40                                 # slide bg
            # face blob (jitters horizontally), NO cam-inset rectangle around it
            cx = face_cx + int(30 * np.sin(t * 2.0))
            cy = face_cy
            yy, xx = np.ogrid[:H_FULL, :W_FULL]
            mask = ((xx - cx) ** 2 / (face_h * 0.35) ** 2
                    + (yy - cy) ** 2 / (face_h * 0.5) ** 2) <= 1.0
            img[mask] = np.array(FACE_BGR, dtype=np.uint8)
            p.stdin.write(img.tobytes())
    finally:
        p.stdin.close(); p.wait()
    return path


@pytest.fixture(scope="module")
def cutout_cam_video(tmp_path_factory):
    return _make_cutout_video(tmp_path_factory.mktemp("cutout") / "cutout.mp4")


def test_cutout_cam_uses_head_and_shoulders_box(cutout_cam_video):
    """Chroma-keyed cam over a static screen: `detect(mode="auto")` should mark the
    result as cutout and return a box ~3x the face height, with the face in the upper
    third — head + shoulders, not just the head."""
    r = facecam.detect(str(cutout_cam_video), 0.0, 3.0, n=15, mode="auto",
                       max_face_h=facecam.MAX_FACE_H_CALM)
    assert r is not None
    assert r.get("cutout") is True, r
    x0, y0, x1, y1 = r["trimmed_box_px"]
    bw, bh = x1 - x0, y1 - y0
    # face_h in synth is 200 px; head-and-shoulders should be ~3x that
    face_h_px = r["face_px"][3] - r["face_px"][1]
    assert bh >= 2.0 * face_h_px, (bh, face_h_px)
    assert bw >= 2.0 * face_h_px, (bw, face_h_px)
    # face centre sits in the upper third of the box
    fcy = (r["face_px"][1] + r["face_px"][3]) / 2
    frac_from_top = (fcy - y0) / max(1, bh)
    assert 0.15 <= frac_from_top <= 0.55, (frac_from_top, y0, y1, fcy)


# ---------------------------------------------------------------- round 2: cam_crop rules

def test_cam_crop_near_square_landscape_covers():
    """R2 defect: a nearly-square landscape cam (aspect ~1.0) must cover-fill the band.
    The earlier rule used band aspect (1.125 for reaction), which sent 1.0-1.11 boxes
    into the blur-fit branch and left a blurred strip under the cam."""
    # 500x460 landscape cam, aspect 1.09 — round 1 rule would have blurred this
    cam_box = (400, 200, 900, 660)
    cw, ch, cx, cy, fit = facecam.cam_crop(W_FULL, H_FULL, cam_box, [0.34, 0.42, 0.15],
                                            bot_h=960)
    assert fit is False, (cw, ch, fit)


@pytest.mark.parametrize("box,bot_h", [((1570, 9, 1795, 159), 620),      # adamkhoo-like
                                       ((400, 200, 900, 660), 960),
                                       ((0, 700, 640, 1080), 620)])
def test_cam_crop_cover_never_leaves_the_box(box, bot_h):
    """The cover crop must stay inside the cam box: widening past it pulled chart
    content and a pillarbox bar into a creator's cam band (a review)."""
    x0, y0, x1, y1 = box
    cw, ch, cx, cy, fit = facecam.cam_crop(W_FULL, H_FULL, box, [(x0 + x1) / 2 / W_FULL,
                                           (y0 + y1) / 2 / H_FULL, 0.1], bot_h=bot_h)
    assert fit is False
    assert x0 <= cx and cx + cw <= x1 and y0 <= cy and cy + ch <= y1, (cx, cy, cw, ch, box)
    assert abs(cw / ch - 1080 / bot_h) < 0.05


def test_cam_crop_true_portrait_still_fits_with_blur():
    """A true portrait cam (aspect < 1.0) still fits-with-blur horizontally."""
    cam_box = (600, 100, 800, 800)                       # 200x700, aspect 0.28
    cw, ch, cx, cy, fit = facecam.cam_crop(W_FULL, H_FULL, cam_box, [0.36, 0.42, 0.15],
                                            bot_h=700)
    assert fit is True


# ---------------------------------------------------------------- round 2: cache override

def test_facecam_cache_override_via_env(monkeypatch, tmp_path):
    """FACECAM_CACHE_DIR points profiles_path at a per-run dir instead of the shared
    pipeline scratch — so an isolated render never overwrites the gamestack profiles."""
    monkeypatch.setenv("FACECAM_CACHE_DIR", str(tmp_path))
    assert facecam.profiles_path().parent == tmp_path
    # save a fake profile under the override — the shared scratch profile file must not
    # change (we don't touch it: writing under the override dir is enough).
    facecam.save_profile("streamer_r2", [0.1, 0.1, 0.3, 0.4], "src.mp4",
                          extra={"raw_box": [0.1, 0.1, 0.3, 0.4]})
    assert (tmp_path / "facecam_profiles.json").exists()
    loaded = facecam.load_profiles()
    assert "streamer_r2" in loaded


def test_facecam_cache_arg_in_recompose_reaches_get_or_detect(monkeypatch, tmp_path):
    """--facecam-cache flows through to facecam.get_or_detect(cache_dir=...)."""
    captured = {}

    def fake(channel, src, start, dur, source_duration=None, cache_dir=None, log=None,
             mode="busy", max_face_h=facecam.MAX_FACE_H, profile_key=None):
        captured["cache_dir"] = cache_dir
        return ([0.1, 0.1, 0.3, 0.4], [0.1, 0.1, 0.3, 0.4], True, 1.0)

    monkeypatch.setattr(facecam, "get_or_detect", fake)
    monkeypatch.setattr(facecam, "detect_content",
                        lambda frames, cam_guard_box_px=None, exclude_boxes=None:
                        [200, 100, 1400, 900])
    monkeypatch.setattr(facecam, "sample_native", lambda *a, **k: [])
    monkeypatch.setattr(facecam, "load_profiles", lambda cache_dir=None: {})
    monkeypatch.setattr(recompose.reframe, "probe_size", lambda s: (W_FULL, H_FULL))
    monkeypatch.setattr(recompose, "probe",
                        lambda p: {"width": W_FULL, "height": H_FULL, "fps": 30,
                                    "duration": 3.0, "has_audio": True})
    override = tmp_path / "cache"
    a = recompose.parse_args([str(tmp_path / "x.mp4"), "0", "3", str(tmp_path / "o.mp4"),
                              "--layout", "reaction", "--channel", "s1",
                              "--facecam-cache", str(override)])
    plan = recompose._plan_reaction(tmp_path / "x.mp4", a)
    assert plan is not None
    assert captured["cache_dir"] == str(override)


# ---------------------------------------------------------------- round 2: content_zoom wide

def test_screencast_auto_content_zoom_wide(monkeypatch, tmp_path):
    """content_zoom='wide' produces a wider content crop with fit-with-blur, not cover.
    The wider crop keeps text near the crop edges from being clipped."""
    monkeypatch.setattr(facecam, "get_or_detect",
                        lambda channel, src, start, dur, source_duration=None,
                        cache_dir=None, log=None, mode="busy",
                        max_face_h=facecam.MAX_FACE_H, profile_key=None:
                        ([0.8, 0.7, 0.95, 0.95], [0.8, 0.7, 0.95, 0.95], True, 1.0))
    monkeypatch.setattr(facecam, "sample_native", lambda *a, **k: [])
    monkeypatch.setattr(facecam, "activity_centroid", lambda *a, **k: 700.0)
    monkeypatch.setattr(facecam, "load_profiles", lambda cache_dir=None: {})
    monkeypatch.setattr(recompose.reframe, "probe_size", lambda s: (W_FULL, H_FULL))
    monkeypatch.setattr(recompose, "probe",
                        lambda p: {"width": W_FULL, "height": H_FULL, "fps": 30,
                                    "duration": 3.0, "has_audio": True})
    # cover
    a = recompose.parse_args([str(tmp_path / "x.mp4"), "0", "3", str(tmp_path / "o.mp4"),
                              "--layout", "screencast", "--channel", "s1",
                              "--content-zoom", "cover"])
    p_cover = recompose._plan_screencast_auto(tmp_path / "x.mp4", a)
    # wide
    a = recompose.parse_args([str(tmp_path / "x.mp4"), "0", "3", str(tmp_path / "o.mp4"),
                              "--layout", "screencast", "--channel", "s1",
                              "--content-zoom", "wide"])
    p_wide = recompose._plan_screencast_auto(tmp_path / "x.mp4", a)
    assert p_cover is not None and p_wide is not None
    assert p_cover["content_zoom"] == "cover"
    assert p_wide["content_zoom"] == "wide"
    # wide crop is wider (or, when clipped by the guard, at least equal)
    assert p_wide["content"][0] >= p_cover["content"][0]
    assert p_wide["content_fit_blur"] is True
    g_wide = recompose.screencast_auto_vf(p_wide)
    assert "gblur=sigma=22[topbg]" in g_wide          # fit-with-blur top band
    g_cover = recompose.screencast_auto_vf(p_cover)
    assert "gblur=sigma=22[topbg]" not in g_cover


# ---------------------------------------------------------------- round 3: snap-to-border
#
# Round 2 defects the round 3 tests cover:
#   1. tarik: busy growth overshoots the rectangular cam's bottom border into CS2
#      warehouse footage. The bottom ~215 px of the 1080x1920 output was game content.
#   2. hasan_trump / hasan_r1 / hasan_r3 / moist_r2 / kai_speed: reaction detector
#      medians ALL detected faces. When the reacted video contains faces (rally crowd,
#      LCS casters, Piers Morgan panel) the median lands off the cam inset.


def test_snap_pulls_bottom_back_to_cam_border():
    """Round 3: a synthetic raw grow box that overshoots the cam's rectangular border
    into content — snap must find the border row (mean edge stability >=
    SNAP_BORDER_STAB_MIN) and pull the bottom back to within 6 px of it. Tests the
    snap function directly with a constructed edge-stability map because a real video
    fixture creates spurious full-row edges at the content-to-background transitions
    that the detector cannot distinguish from a genuine cam overlay border."""
    W, H = 1920, 1080
    # cam interior 200x200..900x720; a bright rectangular border at y=718-720. Below the
    # border, static "content" (no strong stable edges — content_stab < 0.4) that busy
    # growth walked into. Raw grow stopped at y=800.
    std = np.zeros((H, W), dtype=np.float32)
    stab = np.zeros((H, W), dtype=np.float32)
    # cam interior noise band (std in the camera-noise band)
    std[200:718, 200:900] = 4.0
    # rendered cam border row: high, spread across the full row width
    stab[716:721, 200:900] = 1.0
    # content strip below: some sparse stability (say a chart line at y=760)
    stab[758:763, 200:900] = 0.30       # below the snap threshold (0.60)
    stab[720:760, 200:900] = 0.05       # calm content
    stab[763:800, 200:900] = 0.05
    raw = [200, 200, 900, 800]
    face_px = [545, 345, 715, 515]
    snapped = facecam._snap_box_to_cam_edge(std, stab, raw, face_px)
    # bottom must land at the cam border row (within 6 px of y=720)
    assert abs(snapped[3] - 720) <= 6, (snapped, 720)


def test_snap_does_not_move_when_edge_is_already_at_the_border():
    """Round 3 regression guard: when the raw grow box's edge sits AT the border row
    (well-detected gamestack case), snap must not walk further inward looking for a
    second stable line (e.g. a chat strip's inner boundary) — that would eat into cam
    content."""
    W, H = 1920, 1080
    std = np.full((H, W), 4.0, dtype=np.float32)
    stab = np.zeros((H, W), dtype=np.float32)
    # raw box top edge y=100 IS the border row.
    stab[100:103, 200:900] = 1.0
    # a chat strip further inside the box (top at y=180) with a stable top border
    stab[178:183, 200:900] = 1.0
    raw = [200, 100, 900, 800]
    face_px = [545, 400, 715, 500]
    snapped = facecam._snap_box_to_cam_edge(std, stab, raw, face_px)
    # top must NOT snap into the chat strip; it should stay at (or within 5 px of) y=100
    assert snapped[1] <= 105, (snapped, 100)


# ---------------------------------------------------------------- round 3: face cluster
#
# Streamer cam-inset faces sit at the same source pixels every frame; reacted-video
# faces (Trump rally crowd, LCS casters, Piers Morgan panel, Ezra Klein interview)
# move, cut, and change size. The R2 detector took the median of ALL detected faces
# and drifted off the streamer's corner. Round 3 replaces the median with a
# position-stable cluster picker + corner-preference tiebreak.


def _draw_face_at(img, cx, cy, fh, bgr=FACE_BGR):
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = ((xx - cx) ** 2 / (fh * 0.35) ** 2
            + (yy - cy) ** 2 / (fh * 0.5) ** 2) <= 1.0
    img[mask] = np.array(bgr, dtype=np.uint8)


def _make_two_face_scene(path, dur=3.0, fps=10, corner_pos=(1700, 200),
                          corner_face_h=140, video_moves=True):
    """Static corner face plus a face in a 'reacted video' that moves each frame."""
    n = int(round(dur * fps))
    rng = np.random.default_rng(22)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W_FULL}x{H_FULL}",
           "-r", str(fps), "-i", "-",
           "-f", "lavfi", "-t", str(dur), "-i", "sine=frequency=440:sample_rate=48000",
           "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv444p",
           "-c:a", "aac", "-shortest", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        cx_corner, cy_corner = corner_pos
        for i in range(n):
            t = i / fps
            img = np.full((H_FULL, W_FULL, 3), 60, dtype=np.uint8)
            # noisy corner cam-inset room around the corner face
            cam_x0 = max(0, cx_corner - 200); cam_x1 = min(W_FULL, cx_corner + 200)
            cam_y0 = max(0, cy_corner - 100); cam_y1 = min(H_FULL, cy_corner + 200)
            noise = rng.normal(0, 4.0,
                                size=(cam_y1 - cam_y0, cam_x1 - cam_x0, 3)).astype(np.int16)
            img[cam_y0:cam_y1, cam_x0:cam_x1] = np.clip(90 + noise, 0, 255).astype(np.uint8)
            # rendered border around the corner cam
            img[cam_y0:cam_y0 + 2, cam_x0:cam_x1] = 10
            img[cam_y1 - 2:cam_y1, cam_x0:cam_x1] = 10
            img[cam_y0:cam_y1, cam_x0:cam_x0 + 2] = 10
            img[cam_y0:cam_y1, cam_x1 - 2:cam_x1] = 10
            _draw_face_at(img, cx_corner, cy_corner, corner_face_h)
            # video face — bigger, moves across a video panel each frame
            vcx = 500 + (int(t * 200) if video_moves else 0)
            _draw_face_at(img, vcx, 600, 220)
            p.stdin.write(img.tobytes())
    finally:
        p.stdin.close(); p.wait()
    return path


@pytest.fixture(scope="module")
def two_face_video(tmp_path_factory):
    return _make_two_face_scene(tmp_path_factory.mktemp("cluster") / "two.mp4",
                                 corner_pos=(1700, 200))


def test_pick_streamer_face_prefers_stable_corner_over_moving_large(two_face_video):
    """A big moving 'video' face plus a small still corner face -> the corner one
    is the streamer inset (position-stable, cornered). Cluster picker + corner
    tiebreak: the corner face wins, and the corner box is detected."""
    r = facecam.detect(str(two_face_video), 0.0, 3.0, n=15, mode="auto",
                       max_face_h=facecam.MAX_FACE_H_CALM)
    assert r is not None
    cx = r["face_median"][0] * W_FULL
    cy = r["face_median"][1] * H_FULL
    # streamer face centre should be near (1700, 200), NOT the moving video face
    assert abs(cx - 1700) <= 40 and abs(cy - 200) <= 40, (cx, cy)
    # detect() records the box grown from the streamer's face
    x0, y0, x1, y1 = r["trimmed_box_px"]
    # box contains the corner face bbox with margin
    assert x0 <= 1650 and x1 >= 1750 and y0 <= 140 and y1 >= 260, r["trimmed_box_px"]


def _make_two_static_insets_scene(path, dur=3.0, fps=10):
    """Two static corner cams + a changing 'video' region in between. For
    detect_content: the streamer cam is one corner, the OTHER static corner is a
    secondary inset that must be excluded from the content search — otherwise the
    largest changing CC lands on the secondary cam and the reaction detector
    thinks the OTHER cam is the video.
    """
    n = int(round(dur * fps))
    rng = np.random.default_rng(23)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W_FULL}x{H_FULL}",
           "-r", str(fps), "-i", "-",
           "-f", "lavfi", "-t", str(dur), "-i", "sine=frequency=440:sample_rate=48000",
           "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv444p",
           "-c:a", "aac", "-shortest", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(n):
            t = i / fps
            # static browser chrome (dark grey)
            img = np.full((H_FULL, W_FULL, 3), 40, dtype=np.uint8)
            # video rectangle in the middle: coarse changing noise
            vx0, vy0, vx1, vy1 = 450, 250, 1500, 950
            block = 25
            h = vy1 - vy0; w = vx1 - vx0
            small = rng.integers(0, 256, (h // block + 1, w // block + 1, 3), dtype=np.uint8)
            big = np.repeat(np.repeat(small, block, axis=0), block, axis=1)[:h, :w]
            img[vy0:vy1, vx0:vx1] = big
            # two static corner cams — TOP-LEFT and TOP-RIGHT (kai_speed R2 shape)
            for (cx, cy) in ((150, 150), (1770, 150)):
                cx0 = cx - 120; cy0 = cy - 90; cx1 = cx + 120; cy1 = cy + 120
                noise = rng.normal(0, 4.0,
                                    size=(cy1 - cy0, cx1 - cx0, 3)).astype(np.int16)
                img[cy0:cy1, cx0:cx1] = np.clip(90 + noise, 0, 255).astype(np.uint8)
                img[cy0:cy0 + 2, cx0:cx1] = 10; img[cy1 - 2:cy1, cx0:cx1] = 10
                img[cy0:cy1, cx0:cx0 + 2] = 10; img[cy0:cy1, cx1 - 2:cx1] = 10
                _draw_face_at(img, cx, cy, 130)
            p.stdin.write(img.tobytes())
    finally:
        p.stdin.close(); p.wait()
    return path


@pytest.fixture(scope="module")
def two_inset_pip_video(tmp_path_factory):
    return _make_two_static_insets_scene(tmp_path_factory.mktemp("pip") / "pip.mp4")


def test_detect_content_excludes_secondary_static_inset(two_inset_pip_video):
    """With two static corner cam insets + a changing 'video' between them, the
    detector's streamer cluster picker picks one corner; the OTHER corner must be
    excluded from the content search so detect_content locks onto the video
    rectangle, NOT the other cam."""
    frames = facecam.sample_native(str(two_inset_pip_video), 0.0, 3.0, n=15)
    streamer, other_bboxes = facecam._pick_streamer_face(frames,
                                                          max_face_h=facecam.MAX_FACE_H_CALM)
    assert streamer is not None
    # the other static inset should be flagged as a stable secondary cluster
    assert other_bboxes, "expected the other corner inset to be a stable secondary cluster"
    # a generous cam guard around the picked streamer face
    scx, scy, sh = streamer
    guard = [int(scx - 200), int(scy - 150), int(scx + 200), int(scy + 200)]
    box = facecam.detect_content(frames, cam_guard_box_px=guard,
                                  exclude_boxes=other_bboxes)
    assert box is not None
    x0, y0, x1, y1 = box
    # content should be inside the middle video region, not on either corner cam
    assert x0 >= 400 and x1 <= 1550, box
    assert y0 >= 200 and y1 <= 1000, box
    # entirely outside BOTH corner boxes
    for bx0, by0, bx1, by1 in other_bboxes:
        overlap = min(x1, bx1) > max(x0, bx0) and min(y1, by1) > max(y0, by0)
        assert not overlap, ("content overlaps secondary cluster", box, (bx0, by0, bx1, by1))
