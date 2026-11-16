"""Facecam detection, the per-channel cam profile, and the gamestack layout wiring.

Facecam detection needs pixel-realistic input, so tests synthesise short 1920x1080 videos:
a calm coloured box (the "cam room") sitting inside a heavily-varying noise field (the
"game"), with a moving face-shaped blob inside the box. Optional overlay strips can be
composed onto specific edges to exercise the trim rule.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from clipengine.render import captions
from clipengine.render import facecam
from clipengine.render import recompose
from clipengine.render import reframe
from clipengine import config


# ---------------------------------------------------------------- synthetic video

W_FULL, H_FULL = 1920, 1080
CAM_BOX = (200, 300, 800, 700)         # x0, y0, x1, y1 in source pixels
FACE_H = 120                            # face blob height (px) — comfortably < MAX_FACE_H*H (324)


FACE_BGR = (55, 210, 85)                # a distinctive bright green not present in the game bg
GAME_BLOCK = 40                         # block size for the game noise. Big blocks survive h264
                                         # compression, keeping high temporal std after decode.


def _draw_centered_icon(img, cam_box, y_top, icon_w, icon_h):
    """Draw a static centred narrow icon that straddles the bottom edge of the cam. The icon
    is a filled rectangle with concentric bright rings, ~`icon_w` wide and `icon_h` deep,
    centred horizontally in the cam. Used by the R2 icon-trim test: a whole-row edge fraction
    is < TRIM_DENSE_FRAC (only ~12% of the row is icon), so the row-scan misses it and only
    the sliding-window/blob detection catches it."""
    x0, y0, x1, y1 = cam_box
    icon_x = (x0 + x1) // 2 - icon_w // 2
    img[y_top:y_top + icon_h, icon_x:icon_x + icon_w] = np.array([200, 40, 40], dtype=np.uint8)
    for k in range(3, icon_h // 2, 4):
        img[y_top + k:y_top + k + 2, icon_x + k:icon_x + icon_w - k] = \
            np.array([255, 255, 255], dtype=np.uint8)


def _draw_frame(t, cam_box=CAM_BOX, face_h=FACE_H, overlay_top=0, overlay_bot=0,
                center_icon_w=0, center_icon_h=0, rng=None):
    """Build one 1920x1080 BGR frame.

    - Block-noise game background (blocks large enough to survive x264 so per-pixel temporal
      std stays above facecam's BUSY_STD threshold after decoding).
    - Inside `cam_box`, a slate-grey "room" (calm) with a moving face-shaped blob.
    - Optional `overlay_top`/`overlay_bot` strips: static text-like edges + a solid coloured band
      on the top / bottom rows of the box (exercises trim: chat line, sub-counter icon)."""
    rng = rng or np.random.default_rng(0)
    small = rng.integers(0, 256, (H_FULL // GAME_BLOCK + 1, W_FULL // GAME_BLOCK + 1, 3),
                          dtype=np.uint8)
    img = np.repeat(np.repeat(small, GAME_BLOCK, axis=0), GAME_BLOCK, axis=1)[:H_FULL, :W_FULL]
    x0, y0, x1, y1 = cam_box
    img[y0:y1, x0:x1] = np.array([90, 90, 90], dtype=np.uint8)      # slate-grey room
    # face blob: elliptical head with a subtle horizontal jitter over time (streamer moves)
    face_cy = (y0 + y1) // 2
    face_cx = (x0 + x1) // 2 + int(20 * np.sin(t * 2.0))
    yy, xx = np.ogrid[:H_FULL, :W_FULL]
    mask = ((xx - face_cx) ** 2 / (face_h * 0.35) ** 2
            + (yy - face_cy) ** 2 / (face_h * 0.5) ** 2) <= 1.0
    img[mask] = np.array(FACE_BGR, dtype=np.uint8)
    if overlay_top > 0:
        # bit-identical dark strip along the box's top; horizontal white "text" bars every 12 px
        img[y0:y0 + overlay_top, x0:x1] = np.array([20, 20, 20], dtype=np.uint8)
        for k in range(0, x1 - x0, 12):
            img[y0 + overlay_top // 2 - 3:y0 + overlay_top // 2 + 3, x0 + k:x0 + k + 6] = \
                np.array([220, 220, 220], dtype=np.uint8)
    if overlay_bot > 0:
        # a small square icon graphic near the bottom-right corner of the box
        icon_w = 60
        icon_x = x1 - icon_w - 4
        icon_y = y1 - overlay_bot
        img[icon_y:y1, icon_x:x1] = np.array([200, 40, 40], dtype=np.uint8)
        # inner rings so Canny finds a stack of edges
        for k in range(4, icon_w // 2, 6):
            img[icon_y + k:icon_y + k + 2, icon_x + k:icon_x + icon_w - k] = \
                np.array([255, 255, 255], dtype=np.uint8)
    if center_icon_w > 0 and center_icon_h > 0:
        # a small centered icon straddling the box's bottom edge — narrow enough that its
        # edges cover < TRIM_DENSE_FRAC of the row, so only the sliding-window / blob
        # detection catches it.
        _draw_centered_icon(img, cam_box, y1 - center_icon_h, center_icon_w, center_icon_h)
    return img


def _make_synth_video(path, dur=3.0, fps=10, **draw_kw):
    """Encode `dur*fps` `_draw_frame`s to `path` via ffmpeg rawvideo pipe.
    Uses libx264 -qp 0 (visually lossless) so temporal-noise blocks survive to the decoder,
    keeping per-pixel std above facecam.BUSY_STD when the game bg changes each frame."""
    n = int(round(dur * fps))
    rng = np.random.default_rng(1)
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
            p.stdin.write(_draw_frame(t, rng=rng, **draw_kw).tobytes())
    finally:
        p.stdin.close()
        p.wait()
    return path


@pytest.fixture(scope="module")
def synth_gamestack(tmp_path_factory):
    return _make_synth_video(tmp_path_factory.mktemp("cam") / "gs.mp4", dur=3.0, fps=10)


@pytest.fixture(scope="module")
def synth_gamestack_with_chat(tmp_path_factory):
    return _make_synth_video(tmp_path_factory.mktemp("cam") / "chat.mp4",
                              dur=3.0, fps=10, overlay_top=16)


@pytest.fixture(scope="module")
def synth_gamestack_with_icon(tmp_path_factory):
    return _make_synth_video(tmp_path_factory.mktemp("cam") / "icon.mp4",
                              dur=3.0, fps=10, overlay_bot=24)


@pytest.fixture(scope="module")
def synth_gamestack_center_icon(tmp_path_factory):
    """A small centred badge straddling the box's bottom edge, ~12% of box width and 30 px
    deep — the R2 case: a whole-row edge fraction is well under TRIM_DENSE_FRAC, so the
    row-scan alone misses it. Only the sliding-window / blob detection catches it."""
    # cam box is (200, 300, 800, 700): width 600. 12% -> 72 px icon.
    return _make_synth_video(tmp_path_factory.mktemp("cam") / "center_icon.mp4",
                              dur=3.0, fps=10, center_icon_w=72, center_icon_h=30)


@pytest.fixture(autouse=True)
def _fake_face_detector(monkeypatch):
    """YuNet is trained on real faces; the synthetic elliptical blob isn't face-like enough
    for it to fire. Replace `reframe.detect_faces` with a deterministic detector that reads
    the face-blob back out of the synthetic frame (the slate room fill is [90, 90, 90], the
    face fill is [180, 160, 150] BGR — we find its centre and bounding box)."""
    import cv2

    def fake(frame):
        # match FACE_BGR (55, 210, 85) tightly. lossless h264/yuv444p keeps them near-exact.
        b, g, r = frame[..., 0], frame[..., 1], frame[..., 2]
        mask = (b >= 50) & (b <= 60) & (g >= 205) & (g <= 215) & (r >= 80) & (r <= 90)
        n = int(mask.sum())
        if n < 1000:
            return []
        ys, xs = np.where(mask)
        h, w = frame.shape[:2]
        cx, cy = float(xs.mean()) / w, float(ys.mean()) / h
        fh = float(ys.max() - ys.min() + 1) / h
        fw = float(xs.max() - xs.min() + 1) / w
        return [{"cx": cx, "cy": cy, "w": fw, "h": fh, "score": 0.95}]

    monkeypatch.setattr("clipengine.render.reframe.detect_faces", fake)
    monkeypatch.setattr("clipengine.render.facecam.reframe.detect_faces", fake)
    yield


# ---------------------------------------------------------------- detection

def test_detect_finds_the_calm_box_within_8_px(synth_gamestack):
    r = facecam.detect(str(synth_gamestack), 0.0, 3.0, n=15)
    assert r is not None
    x0, y0, x1, y1 = r["trimmed_box_px"]
    tx0, ty0, tx1, ty1 = CAM_BOX
    assert abs(x0 - tx0) <= 8 and abs(x1 - tx1) <= 8
    assert abs(y0 - ty0) <= 8 and abs(y1 - ty1) <= 8
    assert r["face_frac"] >= 0.5
    assert r["samples"] == 15
    assert r["size"] == [W_FULL, H_FULL]


def test_detect_returns_none_when_face_is_full_frame(tmp_path):
    """A "just-chatting" cam fills the frame with a big face — no inset detected."""
    n_frames = 20
    fps = 10
    dur = n_frames / fps
    path = tmp_path / "chatting.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W_FULL}x{H_FULL}", "-r", str(fps),
           "-i", "-", "-f", "lavfi", "-t", str(dur), "-i", "sine=frequency=440",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
           "-shortest", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    rng = np.random.default_rng(7)
    for i in range(n_frames):
        img = rng.integers(60, 90, (H_FULL, W_FULL, 3), dtype=np.uint8)     # calm bg
        # a 500-px-tall face taking up ~46% of frame height — should be filtered by MAX_FACE_H
        cx, cy = W_FULL // 2, H_FULL // 2
        yy, xx = np.ogrid[:H_FULL, :W_FULL]
        big = ((xx - cx) ** 2 / 180 ** 2 + (yy - cy) ** 2 / 250 ** 2) <= 1.0
        img[big] = np.array([180, 160, 150], dtype=np.uint8)
        p.stdin.write(img.tobytes())
    p.stdin.close(); p.wait()
    assert facecam.detect(str(path), 0.0, dur, n=10) is None


def test_trim_removes_top_chat_strip(synth_gamestack_with_chat):
    r = facecam.detect(str(synth_gamestack_with_chat), 0.0, 3.0, n=15)
    assert r is not None
    # The chat strip is 16 px on the top of the box; trim should shave ~that many rows.
    trim = r["trim"]
    assert 8 <= trim["top"] <= 30
    # y0 of the trimmed box lies past the chat strip
    ty0 = r["trimmed_box_px"][1]
    assert ty0 >= CAM_BOX[1] + 8


def test_trim_removes_bottom_icon(synth_gamestack_with_icon):
    r = facecam.detect(str(synth_gamestack_with_icon), 0.0, 3.0, n=15)
    assert r is not None
    trim = r["trim"]
    assert trim["bot"] >= 8            # cut into the 24-px icon
    ty1 = r["trimmed_box_px"][3]
    assert ty1 <= CAM_BOX[3] - 8


def test_trim_removes_centered_bottom_icon(synth_gamestack_center_icon):
    """R2: a narrow (~12% of box width) centred badge that straddles the bottom edge 30 px
    deep must be trimmed out. Whole-row edge fraction is under TRIM_DENSE_FRAC; the trim
    picks the icon up via sliding-window peak-stability + the blob-CC pass."""
    r = facecam.detect(str(synth_gamestack_center_icon), 0.0, 3.0, n=15)
    assert r is not None
    trim = r["trim"]
    # icon is 30 px deep; trim must cut past the icon top.
    assert trim["bot"] >= 25, r["trim"]
    ty1 = r["trimmed_box_px"][3]
    assert ty1 <= CAM_BOX[3] - 25, r["trimmed_box_px"]
    # other sides essentially untouched by the icon (small boundary sub-px trims OK)
    assert trim["top"] <= 3 and trim["left"] <= 3 and trim["right"] <= 3, r["trim"]


def test_trim_capped_at_15_percent_per_side(synth_gamestack_with_chat):
    """A very tall overlay must not eat more than 15% of the box height."""
    path = Path(str(synth_gamestack_with_chat)).parent / "big_overlay.mp4"
    _make_synth_video(path, dur=3.0, fps=10, overlay_top=120)   # taller than the cap
    r = facecam.detect(str(path), 0.0, 3.0, n=15)
    box_h = CAM_BOX[3] - CAM_BOX[1]
    assert r["trim"]["top"] <= int(0.15 * box_h) + 1


def test_trim_never_touches_the_face_box(synth_gamestack):
    """Trim per side stays at least FACE_KEEP_PAD pixels away from the face box."""
    r = facecam.detect(str(synth_gamestack), 0.0, 3.0, n=15)
    fx0, fy0, fx1, fy1 = r["face_px"]
    tx0, ty0, tx1, ty1 = r["trimmed_box_px"]
    assert ty0 <= fy0 - facecam.FACE_KEEP_PAD or r["trim"]["top"] == 0
    assert ty1 >= fy1 + facecam.FACE_KEEP_PAD or r["trim"]["bot"] == 0
    assert tx0 <= fx0 - facecam.FACE_KEEP_PAD or r["trim"]["left"] == 0
    assert tx1 >= fx1 + facecam.FACE_KEEP_PAD or r["trim"]["right"] == 0


# ---------------------------------------------------------------- per-channel profile

def test_profile_save_reuse_and_redetect_when_face_gone(synth_gamestack, monkeypatch, tmp_path):
    """First get_or_detect: no profile -> detect + save. Second: same clip -> reuse.
    Third: a clip where the saved box no longer holds a face -> re-detect (no save)."""
    cache = tmp_path
    logs = []

    def log(m):
        logs.append(str(m))

    # first: no profile yet -> detect and save
    box, raw_box, from_profile, frac = facecam.get_or_detect(
        "streamer1", str(synth_gamestack), 0.0, 3.0, cache_dir=cache, log=log)
    assert box is not None and raw_box is not None and not from_profile and frac >= 0.5
    prof = facecam.load_profiles(cache_dir=cache)
    assert "streamer1" in prof and prof["streamer1"]["reviewed"] is False
    # raw_box must contain the trimmed box (trimmed is inside raw)
    assert raw_box[0] <= box[0] and raw_box[1] <= box[1]
    assert raw_box[2] >= box[2] and raw_box[3] >= box[3]

    # second: reuse (the saved box still holds a face for this clip)
    box2, raw_box2, from_profile2, _ = facecam.get_or_detect(
        "streamer1", str(synth_gamestack), 0.0, 3.0, cache_dir=cache, log=log)
    assert from_profile2 and box2 == prof["streamer1"]["box"]
    assert raw_box2 == prof["streamer1"]["raw_box"]

    # third: mock the saved profile to one that no longer sees a face for this clip
    saved = facecam.load_profiles(cache_dir=cache)
    saved["streamer1"]["box"] = [0.9, 0.9, 0.99, 0.99]      # far corner, no face
    facecam.profiles_path(cache).write_text(json.dumps(saved))
    box3, raw_box3, from_profile3, frac3 = facecam.get_or_detect(
        "streamer1", str(synth_gamestack), 0.0, 3.0, cache_dir=cache, log=log)
    assert not from_profile3
    assert any("re-detect" in m for m in logs)
    # the saved profile is NOT overwritten by a re-detect on a single clip
    assert facecam.load_profiles(cache_dir=cache)["streamer1"]["box"] == [0.9, 0.9, 0.99, 0.99]


def test_profiles_path_defaults_to_scratch_dir(monkeypatch, tmp_path):
    """With no FACECAM_CACHE_DIR the profiles live in the engine's scratch dir (config
    scratch_dir, env CLIPENGINE_SCRATCH_DIR), not inside the repository."""
    monkeypatch.delenv("FACECAM_CACHE_DIR", raising=False)
    monkeypatch.setenv("CLIPENGINE_SCRATCH_DIR", str(tmp_path / "scratch"))
    p = facecam.profiles_path()
    assert p == config.scratch_dir() / "facecam_profiles.json" == \
        tmp_path / "scratch" / "facecam_profiles.json"
    assert ROOT not in p.parents
    # a per-run FACECAM_CACHE_DIR still wins over the scratch dir
    monkeypatch.setenv("FACECAM_CACHE_DIR", str(tmp_path / "cams"))
    assert facecam.profiles_path() == tmp_path / "cams" / "facecam_profiles.json"


# ---------------------------------------------------------------- crop planners

@pytest.mark.parametrize("cam_x", [(50, 500), (700, 1200), (1400, 1900),
                                    (600, 1350)])   # left, centre, right, centred
def test_game_crop_never_overlaps_cam(cam_x):
    x0, x1 = cam_x
    cam_box = (x0, 300, x1, 700)
    for mode in ("tight", "wide"):
        gw, gx, fit = facecam.game_crop(W_FULL, H_FULL, cam_box, mode=mode)
        assert gw > 0 and 0 <= gx <= W_FULL - gw
        # crop and cam-x-range are disjoint
        assert gx + gw <= x0 or gx >= x1, (mode, cam_x, gx, gw)
        assert gw % 2 == 0 and gx % 2 == 0
    # tight preserves aspect on wide sources
    gw_tight, _, _ = facecam.game_crop(W_FULL, H_FULL, (50, 300, 500, 700), mode="tight")
    tight_ref = int(round(H_FULL * 1080 / 1220)) & ~1
    assert gw_tight == tight_ref


@pytest.mark.parametrize("mode", ["tight", "wide"])
@pytest.mark.parametrize("cam_x", [(50, 500), (700, 1200), (1400, 1900), (600, 1350)])
def test_gamestack_plan_game_crop_clears_raw_box_plus_margin(cam_x, mode, monkeypatch, tmp_path):
    """R3: the game crop must clear the RAW (untrimmed) cam grow box + a safety margin, not
    just the trimmed box. The cam's dark frame border sits just outside the raw grow box; a
    game crop that hugs the trimmed box picks up that border as a vertical sliver on the
    game band's cam-side edge (the R3 defect). Test both game genres: `tight`
    (cover crop) and `wide` (fit-with-blur), across a spread of cam positions."""
    rx0, rx1 = cam_x            # RAW grow box x-range
    ry0, ry1 = 300, 700
    # trimmed box is INSIDE the raw box on every side (typical: 5-40 px inward)
    tx0, tx1 = rx0 + 5, rx1 - 5
    ty0, ty1 = ry0 + 5, ry1 - 5
    raw_box_norm = [rx0 / W_FULL, ry0 / H_FULL, rx1 / W_FULL, ry1 / H_FULL]
    trim_box_norm = [tx0 / W_FULL, ty0 / H_FULL, tx1 / W_FULL, ty1 / H_FULL]
    monkeypatch.setattr(facecam, "get_or_detect",
                        lambda channel, src, start, dur, source_duration=None,
                        cache_dir=None, log=None: (trim_box_norm, raw_box_norm, True, 1.0))
    monkeypatch.setattr(recompose.reframe, "probe_size", lambda src: (W_FULL, H_FULL))
    monkeypatch.setattr(recompose, "probe",
                        lambda p: {"width": W_FULL, "height": H_FULL, "fps": 30,
                                    "duration": 3.0, "has_audio": True})
    # bypass the face-median sampling: no need to open a real video for this planning-only test
    monkeypatch.setattr(facecam, "sample_native", lambda *a, **k: [])
    # Force the game genre by pinning game_genre_mode: passing --game is enough since the
    # test cfg's `game_genres` maps the name we choose. Use a fixed name per mode.
    game_name = "league of legends" if mode == "wide" else "valorant"
    a = recompose.parse_args([str(tmp_path / "x.mp4"), "0", "3", str(tmp_path / "o.mp4"),
                              "--layout", "gamestack", "--channel", "s1",
                              "--game", game_name])
    plan = recompose._plan_gamestack(tmp_path / "x.mp4", a)
    assert plan is not None
    assert plan["game_mode"] == mode, (game_name, plan["game_mode"], mode)
    gw, gx, _ = plan["game"]
    margin = int(round(recompose.GAMESTACK_CAM_GUARD_FRAC * W_FULL))
    guard_x0 = max(0, rx0 - margin)
    guard_x1 = min(W_FULL, rx1 + margin)
    assert gx + gw <= guard_x0 or gx >= guard_x1, \
        (mode, cam_x, gx, gw, guard_x0, guard_x1)


def test_game_crop_wide_is_larger_than_tight_when_room_allows():
    cam_box = (0, 300, 400, 700)
    tight_gw, _, _ = facecam.game_crop(W_FULL, H_FULL, cam_box, mode="tight")
    wide_gw, _, wide_fit = facecam.game_crop(W_FULL, H_FULL, cam_box, mode="wide")
    assert wide_gw >= tight_gw and wide_fit is True


def test_game_crop_degrades_when_cam_centred_too_much():
    # cam so centred that tight width can't fit outside it → fall back to fit-with-blur
    cam_box = (700, 300, 1220, 700)
    gw, gx, fit = facecam.game_crop(W_FULL, H_FULL, cam_box, mode="tight")
    assert gx + gw <= 700 or gx >= 1220     # still disjoint
    assert fit is True or gw >= reframe.crop_width(W_FULL, H_FULL, 1080, 1220)


def test_cam_crop_landscape_cover_centred_on_face():
    cam_box = (100, 300, 900, 700)
    face_med = [0.5, 0.4, 0.1]              # face centre near 960 x, 432 y
    cw, ch, cx, cy, fit = facecam.cam_crop(W_FULL, H_FULL, cam_box, face_med,
                                           bot_h=700)
    assert fit is False
    # crop stays inside the cam box
    assert 100 <= cx <= 900 - cw
    # 1080:700 aspect ratio preserved
    assert abs(cw / max(1, ch) - 1080 / 700) < 0.05


def test_cam_crop_portrait_uses_blur_fit():
    # A tall cam (aspect < band aspect) -> fit-with-blur
    cam_box = (400, 100, 700, 900)          # 300 x 800 -> aspect 0.375 < 1080/700
    cw, ch, cx, cy, fit = facecam.cam_crop(W_FULL, H_FULL, cam_box, [0.55, 0.5, 0.2])
    assert fit is True
    assert (cw, cx, cy) == (300, 400, 100) and ch == 800


def test_game_genre_mode_maps_lowercase_names():
    cfg = {"render": {"game_genres": {"fortnite": "wide", "counter-strike": "tight"}}}
    assert facecam.game_genre_mode(cfg, "Fortnite") == "wide"
    assert facecam.game_genre_mode(cfg, "counter-strike") == "tight"
    assert facecam.game_genre_mode(cfg, "unknown") == "tight"
    assert facecam.game_genre_mode(cfg, None) == "tight"


# ---------------------------------------------------------------- recompose plumbing

def _fake_plan_gamestack(src_w=W_FULL, src_h=H_FULL):
    cam_box = [200, 300, 800, 700]
    gw, gx, gfit = facecam.game_crop(src_w, src_h, cam_box, top_h=1220)
    cw, ch, cx, cy, cfit = facecam.cam_crop(src_w, src_h, cam_box, [0.26, 0.46, 0.11],
                                             bot_h=700)
    return {"layout": "gamestack", "src_w": src_w, "src_h": src_h, "top_h": 1220,
            "game": [gw, gx, gfit], "cam": [cw, ch, cx, cy, cfit],
            "cam_box": cam_box, "cam_box_norm": [200/src_w, 300/src_h, 800/src_w, 700/src_h],
            "from_profile": False, "face_frac": 1.0, "game_mode": "tight",
            "face_median": [0.26, 0.46, 0.11], "channel": "streamer1"}


def test_gamestack_vf_stacks_game_over_cam_at_top_h():
    plan = _fake_plan_gamestack()
    vf = recompose.gamestack_vf(plan)
    gw, gx, gfit = plan["game"]
    cw, ch, cx, cy, cfit = plan["cam"]
    # game band uses full source height
    assert f"crop={gw}:ih:{gx}:0" in vf
    assert f"scale=1080:{plan['top_h']}:force_original_aspect_ratio=increase" in vf
    # cam band cover to 1080x(1920-top_h)
    assert f"crop={cw}:{ch}:{cx}:{cy}" in vf
    assert "scale=1080:700:force_original_aspect_ratio=increase" in vf
    assert vf.endswith("[v0]") and "vstack=inputs=2" in vf


def test_gamestack_layout_forced_via_flag(monkeypatch):
    captured = {}
    monkeypatch.setattr(recompose, "_plan_gamestack",
                        lambda src, a: (captured.setdefault("called", True), _fake_plan_gamestack())[1])
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "gamestack",
                              "--channel", "streamer1"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    assert captured.get("called") and plan["layout"] == "gamestack"


def test_gamestack_auto_for_gaming_vertical(monkeypatch):
    monkeypatch.setattr(recompose, "_plan_gamestack", lambda src, a: _fake_plan_gamestack())
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "auto",
                              "--vertical", "gaming"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    assert plan["layout"] == "gamestack"


def test_gamestack_falls_back_to_track_when_no_cam(monkeypatch):
    monkeypatch.setattr(recompose, "_plan_gamestack", lambda src, a: None)
    monkeypatch.setattr(recompose.reframe, "analyze", lambda *ar, **kw: {
        "layout": "track", "src_w": W_FULL, "src_h": H_FULL, "shots": [],
        "segments": [{"t0": 0.0, "t1": 3.0, "x": 100, "cut": True}]})
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "gamestack"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    assert plan["layout"] == "track"


def test_gamestack_falls_back_to_game_only_when_no_face(monkeypatch):
    """No cam AND no face at all — full-source 9:16 crop (game only)."""
    monkeypatch.setattr(recompose, "_plan_gamestack", lambda src, a: None)
    monkeypatch.setattr(recompose.reframe, "analyze", lambda *ar, **kw: {
        "layout": "general", "src_w": W_FULL, "src_h": H_FULL})
    monkeypatch.setattr(recompose.reframe, "probe_size", lambda src: (W_FULL, H_FULL))
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "gamestack"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    assert plan["layout"] == "game_only"


def test_gamestack_caption_y_sits_on_the_seam():
    y = captions.caption_y("gamestack", fg_h=1220)
    assert 1220 - 50 <= y <= 1220 + 100      # near the seam
    # a different top_h moves the seam
    y2 = captions.caption_y("gamestack", fg_h=1400)
    assert y2 > y


def test_gamestack_hook_avoid_returns_the_whole_cam_band():
    plan = _fake_plan_gamestack()
    bands = recompose.hook_avoid(plan)
    assert bands == [(1220, 1920)]
    # a different top_h shifts the band
    plan["top_h"] = 1000
    bands = recompose.hook_avoid(plan)
    assert bands == [(1000, 1920)]


def test_vertical_layouts_config_maps_gaming_to_gamestack():
    cfg = config.load()
    m = (cfg.get("render") or {}).get("vertical_layouts") or {}
    assert m.get("gaming") == "gamestack"
    assert (cfg.get("render") or {}).get("gamestack", {}).get("top_h") == recompose.GAMESTACK_TOP_H >= 0.65 * 1920


def test_full_render_produces_1080x1920_gamestack(synth_gamestack, tmp_path, monkeypatch, capsys):
    """End-to-end gamestack render on the synthetic clip. Mock get_or_detect to return the
    known cam box so the encode path is exercised without depending on real detection edge cases."""
    box_norm = [CAM_BOX[0] / W_FULL, CAM_BOX[1] / H_FULL,
                CAM_BOX[2] / W_FULL, CAM_BOX[3] / H_FULL]
    monkeypatch.setattr(facecam, "get_or_detect",
                        lambda channel, src, start, dur, source_duration=None, cache_dir=None, log=None:
                        (box_norm, box_norm, True, 1.0))
    out = tmp_path / "gs.mp4"
    rc = recompose.main([str(synth_gamestack), "0", "2", str(out), "--layout", "gamestack",
                         "--channel", "streamer1", "--loudnorm", "-14"])
    assert rc == 0
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(out)],
                       capture_output=True, text=True)
    s = json.loads(r.stdout)
    v = next(st for st in s["streams"] if st["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)
    rep = json.loads(capsys.readouterr().out.strip().splitlines()[-1][3:])
    assert rep["layout"] == "gamestack" and rep["passed"]


# ---------------------------------------------------------------- pipeline plumbing

def test_stages_skip_youtube_cookies_for_twitch_urls():
    from clipengine import engine  # local import: keeps the collection cheap
    assert not engine._is_youtube_url("https://www.twitch.tv/videos/12345")
    assert engine._is_youtube_url("https://www.youtube.com/watch?v=abc")
    assert engine._is_youtube_url("https://youtu.be/abc")


@pytest.fixture(autouse=True)
def _v1_gamestack(monkeypatch):
    """This module pins the v1 stack (render.gamestack.v2 off: the top_h game band over the
    full-width cam band); gamestack v2 is tests/test_gamelayout.py."""
    import copy
    real_config, real_gs = recompose._config, recompose._gamestack_cfg

    def config():
        cfg = copy.deepcopy(real_config())
        cfg.setdefault("render", {}).setdefault("gamestack", {})["v2"] = False
        return cfg
    monkeypatch.setattr(recompose, "_config", config)
    monkeypatch.setattr(recompose, "_gamestack_cfg", lambda: {**real_gs(), "v2": False})
