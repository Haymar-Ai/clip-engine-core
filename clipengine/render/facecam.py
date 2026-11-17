#!/usr/bin/env python3
"""Webcam-inset detection for the `gamestack` layout.

A gaming stream is one 16:9 frame holding two things: full-screen game plus a small webcam
inset the streamer composites in via OBS. Detection has three stages:

  1. Sample the window at native resolution, run YuNet face detection on each frame.
  2. Grow a box from the median face until each side hits busy game pixels or the frame edge
     (game footage varies frame-to-frame; a webcam room barely does). Same idea as
     `proto/detect_cam2.py` but ported properly.
  3. Trim each side inward while its outer band carries overlay content instead of camera
     content: chat text lines, sub-counter icons, alert banners. Cap at 15% per side and
     never trim into the face.

Per-channel profile: a streamer's inset is fixed across a stream and usually for months, so
detection is cached under `<scratch>/facecam_profiles.json` and reused. Each reuse re-verifies
that the box still holds a face for >= 40% of the clip window; if not the clip re-detects
(and does not overwrite the saved profile).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from clipengine import config

HERE = Path(__file__).resolve().parent
from clipengine.render import reframe  # noqa: E402


# --- growth phase ------------------------------------------------------------------
BUSY_STD = 30.0            # per-pixel temporal std above this = game (busy) pixel
STOP_FRAC = 0.5            # a side stops when this fraction of its 1-px line is busy
MAX_FACE_H = 0.30          # faces bigger than 30% of frame h are "just chatting", not an inset
MAX_FACE_H_CALM = 0.45     # reaction/screencast: streamers often run a bigger cam. The layout still
                           # wants an inset (not a full-screen face), just a taller one.
MIN_FACE_SAMPLES = 1 / 3   # at least this fraction of samples must find a face
MAX_BOX_AREA_FRAC = 0.40   # if the grown box covers more of the frame than this, growth ran into a
                           # source with too-static a "game" (menus, inventory) — no clean inset was
                           # found, fall back rather than crop half the frame as a "cam"

# --- calm-mode growth (screencast / reaction with a static screen around the cam) ---
# Screen-capture pixels are either exactly static (std ~ 0 after codec) or change in sharp,
# sparse bursts (cursor, typing, a chart tick, a scroll jump). Camera pixels carry sensor
# noise and micro-motion, so the median-band temporal std of a camera pixel is small but
# non-zero (typically 1.5 - 15 on 8-bit BGR). We grow while each 1-px line looks like camera
# noise (many pixels in the "calm" band AND not many exactly-static pixels), and stop when
# the line goes screen-like. Constants calibrated on real screencast + reaction sources.
CALM_STD_LO = 1.2          # a pixel's temporal std at/below this is "screen-static"
CALM_STD_HI = 15.0         # camera-noise upper band: pixels with std between LO and HI look
                           # like camera. Above HI is either game motion or streamer movement,
                           # both of which are still "inside the cam" for growth purposes.
CALM_CAM_FRAC_MIN = 0.55   # a line stays "inside the cam" while at least this fraction of its
                           # pixels are in the calm camera-noise band. Below this it's a screen
                           # side and growth stops.
CALM_STATIC_STOP_FRAC = 0.55  # a line stops if this fraction of its pixels are exactly static
                              # (std <= CALM_STD_LO). Complementary to the cam-frac check —
                              # cursor-only screen has a few noisy pixels but is mostly static.
CALM_BORDER_STAB = 0.90    # calm-mode: when a growing side's next line is at/above this mean
                           # edge stability, it's the cam overlay's rectangular border. Snap the
                           # side to that line and stop.

# --- trim phase --------------------------------------------------------------------
TRIM_CAP_FRAC = 0.15       # never trim inward more than this per side
TRIM_EDGE_MULT = 2.0       # edge density this many times the interior baseline (median frame Canny)
TRIM_STATIC_STD_FRAC = 0.6 # a pixel is "static" if temporal std < this * interior median std
TRIM_BLEED_STD_MULT = 4.0  # a pixel is "bleed" if temporal std > this * interior median std
TRIM_DENSE_FRAC = 0.10     # fraction of a row/col that must be static-and-edge-dense to be overlay
TRIM_DENSE_WEAK = 0.04     # weaker density that still counts when we're inside an active dense chain
TRIM_BLEED_FRAC = 0.45     # fraction of a row/col with bleed pixels needed to be "row-uniform bleed";
                           # higher than a streamer's body motion (< 0.40) so we don't chase motion
TRIM_BLEED_WEAK = 0.15     # weaker bleed used only to extend an active bleed chain (chat glyph fade
                           # rows next to the main chat strip). Streamer body motion has ~this much
                           # bleed too, so we only accept it in the shadow of a strong-bad row.
TRIM_ICON_WIN_FRAC = 0.15  # sliding-window width for icon detection, as fraction of box width. Whole-
                           # row edge fraction misses a narrow icon (a subs badge, an alert icon)
                           # that touches the box edge — the icon's edges only cover a small slice
                           # of the row. Windowed peak edge density picks it up.
TRIM_ICON_STRONG = 0.75    # peak windowed edge-STABILITY on a row/col: fraction of samples where a
                           # pixel is a Canny edge (after 3x3 dilation). Static overlay icons stay at
                           # the same pixels each frame — stability approaches 1.0. Streamer body
                           # motion or a picture on the wall shows edges too, but they don't line up
                           # every frame, so the stability peak stays well under this threshold.
TRIM_ICON_WEAK = 0.55      # weaker peak stability that extends an active icon chain (a badge's
                           # outer rows show partial-overlap edges after dilation).
TRIM_ICON_POS_TOL_FRAC = 0.10   # icon peak position tolerance across rows, as fraction of box
                                # dimension. A static icon has its peak at ~the same position on
                                # adjacent rows; interior wall art / typography peaks jump around.
                                # An icon hit only chains when its peak stays within this window.
TRIM_LOOKAHEAD = 8         # gap tolerance: trim keeps going while a strong bad row is within this
                           # many rows of the current position. Bridges the solid-coloured background
                           # bar between a sub-counter's title glyphs and the box edge.
TRIM_BLOB_STAB = 0.85      # a pixel's Canny-edge stability (fraction of samples that mark it an
                           # edge after 3x3 dilation) at or above this makes it an "overlay pixel"
                           # candidate. A rendered badge / logo re-lands at the same pixel every
                           # frame, so its outline sits near 1.0. Streamer body edges move with
                           # motion and stay well under this.
TRIM_BLOB_MIN_AREA = 30    # minimum blob area (px) after dilation to qualify as an overlay. Below
                           # this is Canny speckle from body motion or JPEG noise.
TRIM_BLOB_DILATE = 1       # 3x3 dilations applied to the stability mask before connected components,
                           # so a stylised icon whose outline has 1-2 px Canny breaks reads as one CC.
TRIM_BLOB_MAX_INWARD = 0.20   # a CC extending inward more than this fraction of the box dimension
                              # (height for top/bot scans, width for left/right) is not an edge
                              # overlay — it's a large embedded room feature (a wall sign, a bookcase)
                              # that shouldn't be trimmed away, e.g. diego's Eternity logo takes ~44%
                              # of the box's height.
TRIM_BLOB_MIN_INWARD = 5      # skip CCs that only extend 1-4 px inward: those are Canny hits on the
                              # box-to-game step (a sharp cam boundary reads as a persistent edge on
                              # every side) or compression speckle, not a real overlay.
CANNY_LO, CANNY_HI = 60, 180
FACE_KEEP_PAD = 4          # leave this many px of air between trim edge and the face box

# --- snap-to-cam-edge (post-grow) --------------------------------------------------
# Growth (busy or calm) can overshoot the real cam edge:
#   - busy: a row of static / calm-menu content past the cam has <50% busy pixels, so
#     growth steps through it (adamkhoo — TradingView chart candles under the cam;
#     tarik — CS warehouse under the cam) and stops on the next busy row.
#   - calm: on lossy Twitch VODs a row of content pixels can transiently read as calm
#     camera-noise.
# The snap pass searches each side of the raw grown box for the cam overlay's rectangular
# BORDER row/col — a straight line whose edge stability sits near 1.0 across the whole
# side because a rendered border draws the same pixels every frame. If a strong border
# is found within `SNAP_BORDER_CAP_FRAC` of the current side, snap to it. Otherwise leave
# the side alone (no change). Border stability is the only signal used: it is definitive
# and does not fire on synthetic uniform-slate cam interiors (test_facecam), on gaming
# HUD strips (moderate stability, well under threshold), or on chat overlays (whose
# stability is concentrated on the glyph pixels, not the full row).
# Cutout / chroma-keyed cams have no rectangular border; snap is skipped for them.
SNAP_BORDER_CAP_FRAC = 0.25       # snap can reach up to this fraction of the raw box side
                                   # inward looking for the border. Wider than the 15% trim
                                   # cap because busy growth can walk deep into calm-menu
                                   # content (tarik overshoot in R2 was ~5% of the frame
                                   # height, i.e. ~15% of the raw box height already).
SNAP_BORDER_STAB_MIN = 0.60       # a row/col with mean edge stability at/above this is the
                                   # cam overlay's rectangular border — snap to it. Real
                                   # OBS overlays sit between 0.6 and 1.0: tarik's dark
                                   # rectangular border reads 0.64 at its centre row (a mix
                                   # of anti-aliased top/side edges); adamkhoo's 0.98.
                                   # Interior cam rows peak at 0.25 and warehouse / chart
                                   # content stays under 0.4 — a 0.6 floor separates them
                                   # cleanly on the shipped sources.

# --- face-cluster picking (streamer's cam-inset face vs. faces in reacted video) ----
# On a reaction stream the reacted video often shows faces of its own (Trump rally
# crowd, LCS caster panel, Ezra Klein interview, Piers Morgan panel). Taking the median
# of ALL faces drags the "streamer" position off the cam inset and into the video.
# The streamer's face sits at the SAME source pixels every frame; video faces move,
# cut, and change size. Cluster face detections by position, pick the most-present +
# most-stable cluster, prefer a cluster near a frame edge / corner as a tiebreak.
FACE_CLUSTER_RADIUS_MULT = 0.8    # cluster radius as multiple of median face height:
                                   # a static inset's centre drifts less than a face; two
                                   # centres more than this apart are different clusters.
FACE_CLUSTER_CORNER_FRAC = 0.25   # a cluster is "cornered" when its centre sits within
                                   # this fraction of the *matching* frame dimension of a
                                   # corner on both axes (0.25*W on x, 0.25*H on y). Using
                                   # min(W,H) for both under-samples wide-anchored corners
                                   # on 16:9 sources (hasan at (1050, 555) in 1280x720 is
                                   # 220 px from the right + 165 px from the bottom).
FACE_CLUSTER_EDGE_FRAC = 0.25     # ...or within this fraction of any single frame edge
                                   # (streamer cams often anchor to the right / bottom edge).
FACE_CLUSTER_CORNER_BOOST = 0.6   # score multiplier for a cornered cluster (tiebreak).
FACE_CLUSTER_EDGE_BOOST = 0.25    # smaller score multiplier for an edge-only cluster.
FACE_CLUSTER_MIN_PRESENCE = 0.60  # fraction of sampled frames a cluster must appear in to
                                   # count as a secondary cam inset (excluded from
                                   # detect_content). A rally crowd face or an interview
                                   # panelist that comes and goes falls below.
FACE_CLUSTER_INSET_POS_STD = 4.0  # pos_std must be at/below this to look like a fixed
                                   # cam inset (kai_speed's second PIP cam: 0-2). A crowd
                                   # face jittering between shot cuts sits well above this.

# --- cutout / chroma-keyed cam ------------------------------------------------------
# A chroma-keyed streamer (green-screened over the screen) has NO rectangular cam border.
# Growth stops at the face because there's no static frame surrounding it — every pixel
# just outside the head is content, not "camera-noise inside the cam". Signal:
#   1. the grown box is barely bigger than the face bounding box (area ratio close to 1)
#   2. the perimeter of the box does NOT show a stable straight edge (no cam border).
# When both fire, replace the grown box with a face-anchored head-and-shoulders box
# (spec: ~3x face_h tall, face in the upper third, clipped to the frame).
CUTOUT_BORDER_STAB = 0.55     # if any edge of the grown box has mean edge stability >= this,
                              # a rectangular cam border WAS found — the cam is not a cutout.
CAM_CROP_PORTRAIT_AR = 0.85   # cam_crop cutoff below which we fit-with-blur instead of cover.
                              # Head-and-shoulders replacement produces boxes at aspect ~1.0,
                              # tarik and other rectangular cams sit at 0.9-1.4, so 0.85 is a
                              # safe floor that only catches genuinely portrait / 9:16 cams.
CUTOUT_HEAD_MULT_H = 3.0      # target head-and-shoulders box height, in face-heights
CUTOUT_HEAD_MULT_W = 3.0      # target head-and-shoulders box width, in face-heights
CUTOUT_FACE_UPPER_THIRD = 1.0 / 3.0  # face centre sits at this fraction from box top

# --- verify / re-detect ------------------------------------------------------------
FACE_IN_BOX_MIN = 0.40     # saved profile OK if face inside it in >= 40% of samples
FACE_ACCEPT = 0.50         # per-clip acceptance threshold for gamestack (spec: >= 50%)

# --- multiwindow (first clip of a channel) ----------------------------------------
MULTIWIN_SPAN_S = 120.0    # spread N=30 frames evenly across this many source seconds
MULTIWIN_N = 30            # samples for the profile detection (across the whole source span)


def _cache_dir():
    """The scratch dir under which facecam_profiles.json lives.

    Precedence: FACECAM_CACHE_DIR env var (per-run override, useful for isolated renders
    that must not touch the shared gamestack profiles) > the config's
    `scratch_dir` > /tmp/facecam_cache as a last resort.
    """
    import os
    env_dir = os.environ.get("FACECAM_CACHE_DIR")
    if env_dir:
        return Path(env_dir)
    try:
        cfg = config.load()
        d = cfg.get("scratch_dir")
        if d:
            return Path(d)
    except (OSError, ValueError):
        pass
    return Path("/tmp/facecam_cache")


def profiles_path(cache_dir=None):
    return Path(cache_dir or _cache_dir()) / "facecam_profiles.json"


def load_profiles(cache_dir=None):
    p = profiles_path(cache_dir)
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_profile(channel, box_norm, source, cache_dir=None, extra=None):
    """Persist a channel's detected box. `source` = a stable identifier (URL or file path)."""
    p = profiles_path(cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(cache_dir)
    entry = {"box": [float(v) for v in box_norm],
             "detected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "source": str(source), "reviewed": False}
    if extra:
        entry.update(extra)
    profiles[channel] = entry
    p.write_text(json.dumps(profiles, indent=2, sort_keys=True))
    return entry


# ---------------------------------------------------------------- sampling

def sample_native(src, start, dur, n=30):
    """Sample `n` full-resolution BGR frames evenly across the window (source seconds)."""
    import cv2
    src = str(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return []
    out = []
    try:
        for i in range(int(n)):
            t = start + (i + 0.5) * dur / max(1, n)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                out.append(frame)
    finally:
        cap.release()
    return out


def _probe_duration(src):
    try:
        s = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(src)], stderr=subprocess.STDOUT, text=True)
        return float((s or "").strip() or 0.0)
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


# ---------------------------------------------------------------- growth

def _grow_box(std_map, face_cx, face_cy, face_h, busy=BUSY_STD, stop=STOP_FRAC):
    """Grow a rectangle from the face outward while each side stays mostly non-busy.
    std_map: (H, W) per-pixel temporal std. Returns [x0, y0, x1, y1] (inclusive-exclusive).

    First pass: raw busy mask. If growth escapes (any side moves > face_h/4), that's the
    result. Second pass (only when the first pass produced no growth on ANY side): mask the
    face bounding box itself as not-busy — the face reads busy (streamer moves) and a small
    cam where the face nearly fills the initial box blocks every side on iteration 0. The
    face-mask only kicks in for the "trapped" case so diego / clix (where growth works cleanly)
    aren't shifted.
    """
    import numpy as np
    H, W = std_map.shape
    raw = (std_map > busy).astype(np.uint8)
    try:
        import cv2
        raw_mask = cv2.medianBlur(raw * 255, 5) > 0
    except Exception:
        raw_mask = raw > 0
    fh = int(face_h)
    ix0 = max(0, int(face_cx) - fh // 2)
    ix1 = min(W, int(face_cx) + fh // 2 + 1)
    iy0 = max(0, int(face_cy) - fh // 2)
    iy1 = min(H, int(face_cy) + fh // 2 + 1)

    def _grow(mask, x0, y0, x1, y1):
        while True:
            grew = False
            if x0 > 0 and mask[y0:y1, x0 - 1].mean() < stop:
                x0 -= 1; grew = True
            if x1 < W and mask[y0:y1, x1].mean() < stop:
                x1 += 1; grew = True
            if y0 > 0 and mask[y0 - 1, x0:x1].mean() < stop:
                y0 -= 1; grew = True
            if y1 < H and mask[y1, x0:x1].mean() < stop:
                y1 += 1; grew = True
            if not grew:
                break
        return x0, y0, x1, y1

    x0, y0, x1, y1 = _grow(raw_mask, ix0, iy0, ix1, iy1)
    trapped = (x0, y0, x1, y1) == (ix0, iy0, ix1, iy1)
    if trapped:
        # face-in-box workaround: mask a face-sized halo around the face bbox as non-busy.
        # The face itself + its immediate surroundings read as motion; the box's neighbours
        # sit inside that halo when the cam is tight around the head, so raw growth can't
        # leave. The halo is bounded by 3/4 * face_h so it doesn't spill deep into the game.
        halo = int(round(fh * 0.75))
        my0 = max(0, iy0 - halo); my1 = min(H, iy1 + halo)
        mx0 = max(0, ix0 - halo); mx1 = min(W, ix1 + halo)
        mask2 = raw_mask.copy()
        mask2[my0:my1, mx0:mx1] = False
        x0, y0, x1, y1 = _grow(mask2, ix0, iy0, ix1, iy1)
    return [int(x0), int(y0), int(x1), int(y1)]


def _grow_box_calm(std_map, edge_stability, face_cx, face_cy, face_h,
                   cam_lo=CALM_STD_LO, cam_hi=CALM_STD_HI,
                   cam_frac_min=CALM_CAM_FRAC_MIN,
                   static_stop=CALM_STATIC_STOP_FRAC,
                   border_stab=CALM_BORDER_STAB):
    """Grow a rectangle from the face while each side looks like camera noise (many pixels
    with small-but-nonzero temporal std) and stop when it goes screen-like (exactly-static
    dominates) or when it snaps to the cam overlay's rectangular border (a straight row/col
    of very high edge stability).

    std_map:        (H, W) per-pixel temporal std over the sample window.
    edge_stability: (H, W) fraction of samples where each pixel is a Canny edge (dilated 3x3).
    """
    import numpy as np
    H, W = std_map.shape
    cam_pixel = ((std_map > cam_lo) & (std_map < cam_hi))
    static_pixel = (std_map <= cam_lo)
    fh = int(face_h)
    ix0 = max(0, int(face_cx) - fh // 2)
    ix1 = min(W, int(face_cx) + fh // 2 + 1)
    iy0 = max(0, int(face_cy) - fh // 2)
    iy1 = min(H, int(face_cy) + fh // 2 + 1)

    def _line_is_cam(cam_line, stat_line):
        cf = float(cam_line.mean()) if len(cam_line) else 0.0
        sf = float(stat_line.mean()) if len(stat_line) else 0.0
        # cam-like when calm-band coverage passes the floor AND exactly-static coverage
        # doesn't dominate. Screen sides land as low-cam+high-static or high-cam+high-static
        # (a mostly-blank slide with a bit of noise on the letters); either kills growth.
        return cf >= cam_frac_min and sf < static_stop

    def _border_here(stab_line):
        # An OBS cam overlay draws a thin stable rectangular border (usually a dark 1-2 px
        # line + subtle drop shadow). Its pixels sit near stability 1.0 along the entire
        # side, while a random camera row's edge stability stays well below.
        return len(stab_line) and float(stab_line.mean()) >= border_stab

    x0, y0, x1, y1 = ix0, iy0, ix1, iy1
    grew = True
    while grew:
        grew = False
        if x0 > 0 and _line_is_cam(cam_pixel[y0:y1, x0 - 1], static_pixel[y0:y1, x0 - 1]) \
                and not _border_here(edge_stability[y0:y1, x0 - 1]):
            x0 -= 1; grew = True
        if x1 < W and _line_is_cam(cam_pixel[y0:y1, x1], static_pixel[y0:y1, x1]) \
                and not _border_here(edge_stability[y0:y1, x1]):
            x1 += 1; grew = True
        if y0 > 0 and _line_is_cam(cam_pixel[y0 - 1, x0:x1], static_pixel[y0 - 1, x0:x1]) \
                and not _border_here(edge_stability[y0 - 1, x0:x1]):
            y0 -= 1; grew = True
        if y1 < H and _line_is_cam(cam_pixel[y1, x0:x1], static_pixel[y1, x0:x1]) \
                and not _border_here(edge_stability[y1, x0:x1]):
            y1 += 1; grew = True
    return [int(x0), int(y0), int(x1), int(y1)]


# ---------------------------------------------------------------- trim

def _trim_box(frames, box_px, face_px, cap_frac=TRIM_CAP_FRAC):
    """Shrink each side inward while its outer band carries overlay content instead of camera
    content. Returns (trimmed_box, per-side trim px + diagnostics).

    A row (or col) is "bad" when any of:
      - dense overlay text/icon: static + edge-dense pixels cover >= TRIM_DENSE_FRAC of the row
        (chat glyph rows, sub-counter title, alert banner text).
      - row-uniform bleed: >= TRIM_BLEED_FRAC of the row is temporally very variable — the whole
        row is a scrolling ticker or partial game footage. Streamer body motion is spatially
        concentrated, doesn't span the row, so it stays below this floor.
      - icon peak: a sliding TRIM_ICON_WIN_FRAC-wide window over the median-frame Canny map
        has edge density >= TRIM_ICON_STRONG. Whole-row fraction misses a small icon
        (e.g. a subs badge ~12% of box width) sitting on the edge; the peak-in-a-15%-window
        catches it. Streamer body edges are position-variable and blur out in the median,
        so an interior body row stays well under this on the *median* edge map.

    Trim scans from the box edge inward with a TRIM_LOOKAHEAD-row walk-forward: bad rows
    within `lookahead` of the current position count, so a subscribers bar's edge-dense title
    row and the solid coloured background bar between it and the box edge come off together.
    Capped at cap_frac per side; never encroaches on the face box.

    Flat "solid-colour bar" detection was tried and dropped: streamer rooms often include
    calm, uniform surfaces (walls, ceilings) that read identically to a rendered overlay bar
    on std alone, so flat-only rows are treated as camera content unless dense/bleed also flags.
    """
    import cv2
    import numpy as np
    x0, y0, x1, y1 = box_px
    fx0, fy0, fx1, fy1 = face_px
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
    std = gray.std(axis=0)
    med = np.median(gray, axis=0).astype(np.uint8)
    edges = (cv2.Canny(med, CANNY_LO, CANNY_HI) > 0).astype(np.uint8)
    # Per-sample edge stability: for each pixel, fraction of samples where it's a Canny edge
    # (after a 3x3 dilation that swallows sub-pixel jitter between frames). Static overlays
    # like a subs badge land at the same pixels every frame — stability near 1.0. A streamer's
    # shirt has edges too, but they move with the streamer — a given pixel is only edge when
    # the shirt is in that position, so stability drops to 0.3-0.7.
    kern = np.ones((3, 3), np.uint8)
    per_sample = np.stack([
        cv2.dilate(cv2.Canny(gray[i].astype(np.uint8), CANNY_LO, CANNY_HI), kern, iterations=1) > 0
        for i in range(gray.shape[0])
    ])
    edge_stability = per_sample.mean(axis=0)
    ix0 = x0 + (x1 - x0) // 5
    ix1 = x1 - (x1 - x0) // 5
    iy0 = y0 + (y1 - y0) // 5
    iy1 = y1 - (y1 - y0) // 5
    if ix1 <= ix0 or iy1 <= iy0:
        return list(box_px), {"top": 0, "bot": 0, "left": 0, "right": 0}
    ref_std = max(float(np.median(std[iy0:iy1, ix0:ix1])), 3.0)
    ref_mean_std = max(float(std[iy0:iy1, ix0:ix1].mean()), 3.0)
    ref_edge = max(float(edges[iy0:iy1, ix0:ix1].mean()), 0.02)
    edge_hi = min(1.0, TRIM_EDGE_MULT * ref_edge)
    static_std_hi = TRIM_STATIC_STD_FRAC * ref_std
    bleed_std_hi = TRIM_BLEED_STD_MULT * ref_std

    bw = x1 - x0
    bh = y1 - y0
    win_row = max(4, int(TRIM_ICON_WIN_FRAC * bw))
    win_col = max(4, int(TRIM_ICON_WIN_FRAC * bh))
    pos_tol_row = max(4, int(TRIM_ICON_POS_TOL_FRAC * bw))
    pos_tol_col = max(4, int(TRIM_ICON_POS_TOL_FRAC * bh))

    def _peak_window(line, win):
        """(peak mean, peak start index) of `line` over any `win`-wide sliding window.
        `line` may be bool, uint8, or float; result is scaled 0-1 for bool/uint8, or the
        raw mean for float input (edge stability, [0,1])."""
        n = len(line)
        if n < win:
            return (float(line.mean()) if n else 0.0), 0
        cs = np.concatenate([[0.0], np.cumsum(line.astype(np.float32))])
        best = 0.0
        best_i = 0
        for i in range(n - win + 1):
            v = (cs[i + win] - cs[i]) / win
            if v > best:
                best = float(v)
                best_i = i
        return best, best_i

    def measure(edge_line, std_line, stab_line, icon_win):
        static_edge = ((edge_line > edge_hi) & (std_line < static_std_hi)).mean()
        bleed_frac = (std_line > bleed_std_hi).mean()
        icon_peak, icon_pos = _peak_window(stab_line, icon_win)
        return float(static_edge), float(bleed_frac), float(icon_peak), int(icon_pos)

    def _row_measure(y):
        return measure(edges[y, x0:x1], std[y, x0:x1], edge_stability[y, x0:x1], win_row)

    def _col_measure(x):
        return measure(edges[y0:y1, x], std[y0:y1, x], edge_stability[y0:y1, x], win_col)

    def _walk(measure_at, cap, boundary, pos_tol):
        """Trim depth for one side.

        Three families of "bad": edge-dense static (overlay glyphs — whole-row fraction),
        row-uniform bleed (scrolling ticker, translucent overlay), and icon peak (windowed
        edge density peak on the median frame — catches a narrow badge/icon touching the box
        edge without covering the whole row).

        The icon peak also fires on cam-interior wall art (framed pictures, wall typography),
        which is edge-dense too. Two guards keep the icon rule from following interior content:

        - Icon needs a dense/bleed chain from the box edge to open. A pure-icon strong hit
          on its own doesn't start a trim.
        - Icon extension has a stable-position gate: consecutive icon-strong rows must have
          their peak window within `pos_tol` of the last icon-anchor position. A static
          overlay icon holds its position across neighbour rows; wall art or typography
          peaks jump around from row to row, and the chain dies.
        """
        last_bad = -1
        last_strong = {"dense": -10 ** 9, "bleed": -10 ** 9, "icon": -10 ** 9}
        icon_anchor_pos = -10 ** 9        # peak position at the last accepted icon-strong row
        chain_open = False
        n = min(cap, boundary)
        for d in range(n):
            dense, bleed, icon, icon_pos = measure_at(d)
            is_bad = False
            if dense >= TRIM_DENSE_FRAC:
                last_strong["dense"] = d; is_bad = True; chain_open = True
            if bleed >= TRIM_BLEED_FRAC:
                last_strong["bleed"] = d; is_bad = True; chain_open = True
            if chain_open and icon >= TRIM_ICON_STRONG:
                # first icon-strong after chain opens anchors the position; subsequent hits
                # must stay close to it (a real icon holds its column position). Wall art
                # peaks jump around and the anchor gate stops them from chaining. Icons
                # that sit offset from the initial dense/bleed anchor (subs badge inside
                # the wider subs bar) are picked up by the blob-CC pass below instead.
                if icon_anchor_pos < 0 or abs(icon_pos - icon_anchor_pos) <= pos_tol:
                    last_strong["icon"] = d
                    icon_anchor_pos = icon_pos
                    is_bad = True
            if not is_bad:
                if dense >= TRIM_DENSE_WEAK and d - last_strong["dense"] <= TRIM_LOOKAHEAD:
                    is_bad = True
                elif bleed >= TRIM_BLEED_WEAK and d - last_strong["bleed"] <= TRIM_LOOKAHEAD:
                    is_bad = True
                elif chain_open and icon >= TRIM_ICON_WEAK \
                        and d - last_strong["icon"] <= TRIM_LOOKAHEAD \
                        and icon_anchor_pos >= 0 \
                        and abs(icon_pos - icon_anchor_pos) <= pos_tol:
                    is_bad = True
            if is_bad:
                last_bad = d
            elif d - last_bad > TRIM_LOOKAHEAD:
                break
        return last_bad + 1 if last_bad >= 0 else 0

    bh, bw = y1 - y0, x1 - x0
    cap_dy = int(cap_frac * bh)
    cap_dx = int(cap_frac * bw)
    top_bound = max(0, fy0 - FACE_KEEP_PAD - y0)
    bot_bound = max(0, y1 - 1 - fy1 - FACE_KEEP_PAD)
    left_bound = max(0, fx0 - FACE_KEEP_PAD - x0)
    right_bound = max(0, x1 - 1 - fx1 - FACE_KEEP_PAD)
    top = _walk(lambda d: _row_measure(y0 + d), cap_dy, top_bound, pos_tol_row)
    bot = _walk(lambda d: _row_measure(y1 - 1 - d), cap_dy, bot_bound, pos_tol_row)
    left = _walk(lambda d: _col_measure(x0 + d), cap_dx, left_bound, pos_tol_col)
    right = _walk(lambda d: _col_measure(x1 - 1 - d), cap_dx, right_bound, pos_tol_col)
    # Blob pass: a narrow, edge-stable badge (subs icon, watermark) touching the box edge
    # doesn't push the row-scan far because its stability peak is offset from the box border
    # and the position gate above pins the anchor at the border. Complement the row scan with
    # a connected-component sweep on the same edge-stability mask: any high-stability CC
    # that (a) touches an outer edge, (b) is small enough not to be a full-width overlay
    # (row-scan already handles those), and (c) extends inward less than the cap, contributes
    # its inward extent as a per-side trim. The per-side final trim = max(row-scan, blob-scan).
    stab_mask = (edge_stability[y0:y1, x0:x1] >= TRIM_BLOB_STAB).astype(np.uint8)
    # zero the 1-px border ring so the sharp cam-to-game edge (a persistent step at the
    # detected box boundary) doesn't produce phantom edge CCs on every side.
    if stab_mask.shape[0] > 2 and stab_mask.shape[1] > 2:
        stab_mask[0, :] = 0; stab_mask[-1, :] = 0
        stab_mask[:, 0] = 0; stab_mask[:, -1] = 0
    if TRIM_BLOB_DILATE > 0:
        stab_mask = cv2.dilate(stab_mask, np.ones((3, 3), np.uint8),
                                iterations=TRIM_BLOB_DILATE)
    n_ccs, labels, cc_stats, _ = cv2.connectedComponentsWithStats(stab_mask, connectivity=8)
    face_local = (max(0, fx0 - x0), max(0, fy0 - y0),
                  min(bw, fx1 - x0), min(bh, fy1 - y0))
    blob_top = blob_bot = blob_left = blob_right = 0
    for lbl in range(1, n_ccs):
        cx, cy, cw_, ch_, area = cc_stats[lbl]
        if area < TRIM_BLOB_MIN_AREA:
            continue
        # skip a CC that overlaps the face bbox — that's the face's own hair/features
        if cx < face_local[2] and cx + cw_ > face_local[0] \
                and cy < face_local[3] and cy + ch_ > face_local[1]:
            continue
        touches_top = cy == 0
        touches_bot = cy + ch_ >= bh
        touches_left = cx == 0
        touches_right = cx + cw_ >= bw
        n_edges = int(touches_top) + int(touches_bot) + int(touches_left) + int(touches_right)
        if n_edges == 0:
            continue
        # Corner CCs (touching two adjacent sides) are ambiguous: they're a chunk running
        # along both edges (a room corner, a shelf in the box corner) and using either side's
        # depth double-counts the CC. Skip them; if either side genuinely has an overlay it
        # also has its own non-corner CC on that side, which will apply.
        if n_edges >= 2:
            continue
        # A CC whose depth into the box (ch for top/bot, cw for left/right) is more than
        # TRIM_BLOB_MAX_INWARD of the box's matching dimension is too big to be an edge
        # overlay — it's a large wall feature (a bookcase, a signed poster, diego's
        # Eternity logo). Trimming through it would eat into cam content.
        # a CC with inward depth under TRIM_BLOB_MIN_INWARD is almost certainly a stray edge
        # at the box border (Canny fired on the step between cam and game) or a compression
        # artifact — not a real overlay depth worth trimming.
        max_top_bot = min(cap_dy, int(TRIM_BLOB_MAX_INWARD * bh))
        max_left_right = min(cap_dx, int(TRIM_BLOB_MAX_INWARD * bw))
        if touches_top and TRIM_BLOB_MIN_INWARD <= ch_ <= max_top_bot:
            blob_top = max(blob_top, min(top_bound, ch_))
        if touches_bot and TRIM_BLOB_MIN_INWARD <= ch_ <= max_top_bot:
            blob_bot = max(blob_bot, min(bot_bound, ch_))
        if touches_left and TRIM_BLOB_MIN_INWARD <= cw_ <= max_left_right:
            blob_left = max(blob_left, min(left_bound, cw_))
        if touches_right and TRIM_BLOB_MIN_INWARD <= cw_ <= max_left_right:
            blob_right = max(blob_right, min(right_bound, cw_))
    top = max(top, blob_top)
    bot = max(bot, blob_bot)
    left = max(left, blob_left)
    right = max(right, blob_right)
    trimmed = [x0 + left, y0 + top, x1 - right, y1 - bot]
    return trimmed, {"top": top, "bot": bot, "left": left, "right": right,
                     "blob": {"top": blob_top, "bot": blob_bot,
                              "left": blob_left, "right": blob_right},
                     "ref_std": round(ref_std, 2), "ref_mean_std": round(ref_mean_std, 2),
                     "ref_edge": round(ref_edge, 4)}


# ---------------------------------------------------------------- detect

def _collect_faces(frames, max_face_h=MAX_FACE_H):
    """[(cx, cy, h) in pixels] for every face smaller than `max_face_h` across all frames.
    Big-face frames are ignored as "just-chatting" full-screen cameras (not an inset)."""
    W = frames[0].shape[1]
    H = frames[0].shape[0]
    seen = []
    for f in frames:
        for face in reframe.detect_faces(f):
            if face["h"] < max_face_h:
                seen.append((face["cx"] * W, face["cy"] * H, face["h"] * H))
    return seen, (W, H)


def _collect_faces_by_frame(frames, max_face_h=MAX_FACE_H):
    """[(frame_idx, cx, cy, h) px] for every face < max_face_h — same as _collect_faces
    but keeps the source frame index so the cluster picker can measure per-frame presence.
    """
    if not frames:
        return [], (0, 0)
    W = frames[0].shape[1]
    H = frames[0].shape[0]
    seen = []
    for i, f in enumerate(frames):
        for face in reframe.detect_faces(f):
            if face["h"] < max_face_h:
                seen.append((i, face["cx"] * W, face["cy"] * H, face["h"] * H))
    return seen, (W, H)


def _cluster_faces_by_position(faces_by_frame, radius):
    """Greedy proximity clustering: (fi, cx, cy, h) -> [[member, ...], ...]. Two faces are
    the same cluster when their centres are within `radius` pixels of each other."""
    clusters = []
    for f in faces_by_frame:
        fi, cx, cy, h = f
        best = -1
        best_d = None
        for j, c in enumerate(clusters):
            mcx = c["cx_sum"] / c["n"]
            mcy = c["cy_sum"] / c["n"]
            d = ((cx - mcx) ** 2 + (cy - mcy) ** 2) ** 0.5
            if d <= radius and (best_d is None or d < best_d):
                best = j
                best_d = d
        if best >= 0:
            c = clusters[best]
            c["members"].append(f)
            c["cx_sum"] += cx; c["cy_sum"] += cy; c["n"] += 1
        else:
            clusters.append({"members": [f], "cx_sum": cx, "cy_sum": cy, "n": 1})
    return [c["members"] for c in clusters]


def _pick_streamer_face(frames, max_face_h=MAX_FACE_H):
    """Pick the streamer's cam-inset face from all detected faces in the sample window.

    Streamer cam-inset faces sit at the same source pixels every frame; reacted-video
    faces (a Trump rally crowd, LCS casters, Piers Morgan panel guests, an Ezra Klein
    interview shot) move, cut, resize, and drop out. Cluster face detections by
    position, score each cluster by presence * position-stability, and prefer a
    cornered / edge-anchored cluster on tiebreak.

    Returns (streamer_face, other_stable_bboxes):
      streamer_face: (cx, cy, h) in source px, or None when no face is present.
      other_stable_bboxes: [(x0, y0, x1, y1), ...] px for OTHER clusters with
        presence >= FACE_CLUSTER_MIN_PRESENCE AND near an edge — these are secondary
        cam insets (kai_speed has a second PIP cam in the top-left corner) and must
        be excluded from detect_content so the content detector doesn't lock onto them.
    """
    import numpy as np
    faces, (W, H) = _collect_faces_by_frame(frames, max_face_h=max_face_h)
    if len(faces) < max(1, int(MIN_FACE_SAMPLES * len(frames))):
        return None, []
    med_h = float(np.median([f[3] for f in faces]))
    radius = max(20.0, med_h * FACE_CLUSTER_RADIUS_MULT)
    clusters = _cluster_faces_by_position(faces, radius)
    if not clusters:
        return None, []
    n_frames = max(1, len(frames))
    min_dim = float(min(W, H))
    scored = []
    for c in clusters:
        frame_idx_set = {m[0] for m in c}
        presence = len(frame_idx_set) / n_frames
        cxs = np.array([m[1] for m in c], dtype=np.float64)
        cys = np.array([m[2] for m in c], dtype=np.float64)
        hs = np.array([m[3] for m in c], dtype=np.float64)
        pos_std = float(np.hypot(cxs.std(), cys.std()))
        mcx, mcy, mh = float(np.median(cxs)), float(np.median(cys)), float(np.median(hs))
        # stability: 1 for a perfectly still cluster, decays with pos spread / med face height
        stability = float(np.exp(-pos_std / max(1.0, med_h)))
        dx = min(mcx, W - mcx); dy = min(mcy, H - mcy)
        # per-axis edge-proximity check: an OBS corner cam sits within a small fraction of
        # frame WIDTH from a side edge AND within a small fraction of frame HEIGHT from the
        # matching top/bottom edge.
        near_corner = (dx < FACE_CLUSTER_CORNER_FRAC * W
                        and dy < FACE_CLUSTER_CORNER_FRAC * H)
        near_edge = (dx < FACE_CLUSTER_EDGE_FRAC * W
                      or dy < FACE_CLUSTER_EDGE_FRAC * H)
        boost = 1.0 + (FACE_CLUSTER_CORNER_BOOST if near_corner
                        else (FACE_CLUSTER_EDGE_BOOST if near_edge else 0.0))
        score = presence * stability * boost
        # cluster bbox (union of member face bboxes) — used to mask out secondary insets
        # in detect_content.
        pad = mh / 2.0
        bx0 = max(0, int(round(float(cxs.min()) - pad)))
        by0 = max(0, int(round(float(cys.min()) - pad)))
        bx1 = min(W, int(round(float(cxs.max()) + pad)))
        by1 = min(H, int(round(float(cys.max()) + pad)))
        scored.append({"cx": mcx, "cy": mcy, "h": mh, "score": score,
                       "presence": presence, "pos_std": pos_std,
                       "near_edge": near_edge, "near_corner": near_corner,
                       "bbox": (bx0, by0, bx1, by1)})
    scored.sort(key=lambda x: -x["score"])
    streamer = scored[0]
    # A "secondary cam inset" candidate is: cornered (both axes near a frame edge),
    # highly present, AND spatially very tight (pos_std well under the streamer's own
    # motion). Face clusters from a reacted video (rally crowd, panel talking heads)
    # miss on at least one of these — they either move too much, appear in only a
    # subset of frames, or sit in the middle rather than a corner. Applied strictly
    # so we don't eat the actual video-content region on frames like hasan_trump.
    other_stable = [s["bbox"] for s in scored[1:]
                     if s["presence"] >= FACE_CLUSTER_MIN_PRESENCE
                     and s["near_corner"]
                     and s["pos_std"] <= FACE_CLUSTER_INSET_POS_STD]
    return (streamer["cx"], streamer["cy"], streamer["h"]), other_stable


def _face_in_box_frac(frames, box_px, max_face_h=MAX_FACE_H):
    """Fraction of sampled frames with ANY small face whose centre falls inside `box_px`.
    On reaction streams a video face bigger than max_face_h can appear in the same frame
    as the streamer's cam-inset face — so a big face doesn't preclude the frame from
    counting: keep scanning that frame for a smaller in-box face."""
    if not frames:
        return 0.0
    x0, y0, x1, y1 = box_px
    W = frames[0].shape[1]; H = frames[0].shape[0]
    n_hit = 0
    for f in frames:
        for face in reframe.detect_faces(f):
            if face["h"] >= max_face_h:
                continue                  # big face = probably a video face, keep looking
            cx, cy = face["cx"] * W, face["cy"] * H
            if x0 <= cx < x1 and y0 <= cy < y1:
                n_hit += 1
                break                     # count each frame at most once
    return n_hit / len(frames)


def _head_shoulders_box(face_cx, face_cy, face_h, W, H,
                         mult_h=CUTOUT_HEAD_MULT_H, mult_w=CUTOUT_HEAD_MULT_W,
                         upper_third=CUTOUT_FACE_UPPER_THIRD):
    """Face-anchored head-and-shoulders bounding box for a chroma-keyed / cutout cam.

    Returns [x0, y0, x1, y1] in source pixels. The box is ~`mult_h` face-heights tall
    and ~`mult_w` face-heights wide, with the face centre sitting at `upper_third` of
    the box height from the top (so shoulders / mic sit in the lower two-thirds). All
    sides are clipped to the frame.
    """
    face_h = max(2.0, float(face_h))
    bh = int(round(mult_h * face_h))
    bw = int(round(mult_w * face_h))
    # face centre is `upper_third * bh` below the box top
    y0 = int(round(face_cy - upper_third * bh))
    x0 = int(round(face_cx - bw / 2.0))
    x1 = x0 + bw
    y1 = y0 + bh
    x0 = max(0, min(W - 2, x0))
    y0 = max(0, min(H - 2, y0))
    x1 = max(x0 + 2, min(W, x1))
    y1 = max(y0 + 2, min(H, y1))
    return [x0, y0, x1, y1]


def _border_edge_stability(edge_stability, box_px):
    """Max mean-edge-stability across the four one-pixel border rows/cols of the box.
    A rectangular cam overlay draws a straight stable border on all four sides — its
    max side stability sits near 1.0. A cutout / chroma-keyed cam has no such border;
    the four sides read as content (moving edges), stability well below 0.5.
    """
    x0, y0, x1, y1 = [int(v) for v in box_px]
    H, W = edge_stability.shape
    x0 = max(0, min(W - 1, x0)); x1 = max(x0 + 1, min(W, x1))
    y0 = max(0, min(H - 1, y0)); y1 = max(y0 + 1, min(H, y1))
    sides = []
    if y0 < H:
        sides.append(float(edge_stability[y0, x0:x1].mean()))
    if y1 - 1 >= 0 and y1 - 1 < H:
        sides.append(float(edge_stability[y1 - 1, x0:x1].mean()))
    if x0 < W:
        sides.append(float(edge_stability[y0:y1, x0].mean()))
    if x1 - 1 >= 0 and x1 - 1 < W:
        sides.append(float(edge_stability[y0:y1, x1 - 1].mean()))
    return max(sides) if sides else 0.0


def _snap_box_to_cam_edge(std_map, edge_stability, raw_box, face_px,
                           border_cap_frac=SNAP_BORDER_CAP_FRAC,
                           border_stab_min=SNAP_BORDER_STAB_MIN):
    """Snap each side of the raw grown box inward to the cam overlay's rectangular border
    when one is found within `border_cap_frac` of the current side. Fixes Round 3
    overshoot: busy growth on adamkhoo stepped past the cam's bottom edge into the
    TradingView chart; on tarik it stepped past into a strip of CS2 warehouse — both
    sources have visible rectangular cam borders that we can snap to.

    Signal: a row/col whose mean edge stability is at/above SNAP_BORDER_STAB_MIN is the
    cam border (rendered lines redraw the same pixels every frame, so their stability
    approaches 1.0 along the whole side; game HUD elements peak locally but average below
    0.5 across a row; camera rows sit near 0.1-0.3). Never encroaches on face + pad.
    Only side-snaps; the trim pass runs next on the snapped box to remove any chat / icon
    overlays that live INSIDE the snapped rectangle.
    """
    x0, y0, x1, y1 = [int(v) for v in raw_box]
    fx0, fy0, fx1, fy1 = [int(v) for v in face_px]
    H, W = std_map.shape
    bh = max(1, y1 - y0)
    bw = max(1, x1 - x0)
    border_cap_dy = int(border_cap_frac * bh)
    border_cap_dx = int(border_cap_frac * bw)

    def _snap_side(get_line, bound, border_cap):
        """Return the number of pixels to move the side inward (0 = no change).

        Scans up to `min(bound, border_cap)` inward looking for the first stable-border
        line — a full line with mean edge stability at/above SNAP_BORDER_STAB_MIN. When
        multiple candidates exist within the search band, snap to the FIRST one from the
        outside (the outermost border). Face + pad is the hard inner limit.
        """
        cap = max(0, min(border_cap, bound))
        for d in range(cap):
            std_line, stab_line = get_line(d)
            if len(stab_line) == 0:
                break
            if float(stab_line.mean()) >= border_stab_min:
                return d
        return 0

    top_bound = max(0, fy0 - FACE_KEEP_PAD - y0)
    dy_top = _snap_side(lambda d: (std_map[y0 + d, x0:x1],
                                     edge_stability[y0 + d, x0:x1]),
                          top_bound, border_cap_dy)
    bot_bound = max(0, y1 - 1 - fy1 - FACE_KEEP_PAD)
    dy_bot = _snap_side(lambda d: (std_map[y1 - 1 - d, x0:x1],
                                     edge_stability[y1 - 1 - d, x0:x1]),
                          bot_bound, border_cap_dy)
    new_y0 = y0 + dy_top
    new_y1 = y1 - dy_bot
    if new_y1 <= new_y0 + 1:
        new_y0, new_y1 = y0, y1
    left_bound = max(0, fx0 - FACE_KEEP_PAD - x0)
    dx_l = _snap_side(lambda d: (std_map[new_y0:new_y1, x0 + d],
                                    edge_stability[new_y0:new_y1, x0 + d]),
                         left_bound, border_cap_dx)
    right_bound = max(0, x1 - 1 - fx1 - FACE_KEEP_PAD)
    dx_r = _snap_side(lambda d: (std_map[new_y0:new_y1, x1 - 1 - d],
                                    edge_stability[new_y0:new_y1, x1 - 1 - d]),
                         right_bound, border_cap_dx)
    new_x0 = x0 + dx_l
    new_x1 = x1 - dx_r
    if new_x1 <= new_x0 + 1:
        new_x0, new_x1 = x0, x1
    return [int(new_x0), int(new_y0), int(new_x1), int(new_y1)]


def _edge_stability_map(frames):
    """Per-pixel Canny-edge stability over the sample window. See `_trim_box` for the same
    idea: pixels sit near 1.0 wherever an overlay draws the same edge every frame (a cam
    overlay's rectangular border, a badge outline)."""
    import cv2
    import numpy as np
    kern = np.ones((3, 3), np.uint8)
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.uint8)
    per_sample = np.stack([
        cv2.dilate(cv2.Canny(gray[i], CANNY_LO, CANNY_HI), kern, iterations=1) > 0
        for i in range(gray.shape[0])
    ])
    return per_sample.mean(axis=0)


def detect(src, start, dur, n=30, cache_dir=None, mode="busy",
           max_face_h=MAX_FACE_H):
    """Detect the webcam inset in one window. Returns
        {"box": [x0,y0,x1,y1] normalized 0-1, "face_frac": ..., "samples": n,
         "size": [W, H], "raw_box_px": [...], "trim": {...}, "mode": "busy"|"calm"}
    or None (no inset).

    `mode` selects the growth rule:
      "busy": legacy gamestack path — grow while sides are non-busy (game motion outside).
      "calm": grow while sides look like camera noise, stop on screen-static or a stable
              rectangular border (reaction/screencast, where the surrounding content is a
              slide, chart, or browser page and the busy rule underruns).
      "auto": try busy first; if the busy grow-box exceeds MAX_BOX_AREA_FRAC or trim
              collapses it, fall back to calm.

    `max_face_h`: faces bigger than this fraction of frame height are treated as
    "just-chatting" (no inset). Gamestack sticks to MAX_FACE_H = 0.30; reaction and
    screencast pass MAX_FACE_H_CALM = 0.45 (bigger presenter cams).
    """
    import numpy as np
    frames = sample_native(src, start, dur, n=n)
    if not frames:
        return None
    W = frames[0].shape[1]; H = frames[0].shape[0]
    # Pick the position-stable face cluster (streamer cam), not the median-of-all-faces.
    # Reacted-video faces move and drag a plain median off-corner (hasan_trump: rally
    # crowd; moist_r2: LCS casters; hasan_r1: Ezra Klein interview; hasan_r3: Piers
    # Morgan panel — every R2 fallback traced to this).
    streamer, other_face_bboxes = _pick_streamer_face(frames, max_face_h=max_face_h)
    if streamer is None:
        return None
    cx, cy, fh = streamer
    import cv2
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
    std = gray.std(axis=0)
    face_px = [int(cx - fh / 2), int(cy - fh / 2), int(cx + fh / 2), int(cy + fh / 2)]

    def _try_busy():
        raw = _grow_box(std, cx, cy, fh)
        area_frac = ((raw[2] - raw[0]) * (raw[3] - raw[1])) / max(1, W * H)
        if area_frac > MAX_BOX_AREA_FRAC:
            return None, area_frac
        return raw, area_frac

    def _try_calm():
        stab = _edge_stability_map(frames)
        raw = _grow_box_calm(std, stab, cx, cy, fh)
        area_frac = ((raw[2] - raw[0]) * (raw[3] - raw[1])) / max(1, W * H)
        if area_frac > MAX_BOX_AREA_FRAC:
            return None, area_frac
        return raw, area_frac

    if mode == "busy":
        raw, _ = _try_busy()
        used_mode = "busy"
    elif mode == "calm":
        raw, _ = _try_calm()
        used_mode = "calm"
    else:                                    # "auto"
        raw, _ = _try_busy()
        if raw is None:
            raw, _ = _try_calm()
            used_mode = "calm"
        else:
            used_mode = "busy"
    if raw is None:
        return None

    # Round 3: run snap-to-cam-edge FIRST on the raw grown box. If busy growth overshot
    # into calm content (tarik: CS warehouse under the cam; adamkhoo: chart under the
    # cam), snap pulls each side inward to the cam overlay's rectangular border. The
    # cutout classifier then measures border-stability on the tightened box, so a real
    # rectangular cam is not mis-diagnosed as a cutout just because growth overran it.
    stab = _edge_stability_map(frames)
    snapped_raw = _snap_box_to_cam_edge(std, stab, raw, face_px)
    snap_delta = {
        "top": snapped_raw[1] - raw[0 + 1], "bot": raw[3] - snapped_raw[3],
        "left": snapped_raw[0] - raw[0], "right": raw[2] - snapped_raw[2]}

    # Cutout / chroma-keyed cam check on the SNAPPED box. A rectangular OBS overlay
    # draws a stable straight border on all four sides (max side edge stability near
    # CUTOUT_BORDER_STAB). A chroma-keyed streamer has no such border, and the busy/
    # calm growth stops for a different reason: either the face is barely bigger than
    # the initial box (calm detected only the head — primeagen case) OR growth ran into
    # empty dark screen and produced a box shaped nothing like a rectangular cam
    # overlay (primetime case).
    # Diagnostic (post-snap):
    #   1. box aspect roughly square (0.65-1.5) — heads/shoulders shape rather than a
    #      rectangular OBS inset which almost always sits at 4:3 (1.33) or wider.
    #   2. no stable rectangular border around the four sides of the snapped box.
    # When both fire, replace the grown box with a face-anchored head-and-shoulders
    # rectangle (~3x face_h). Rectangular corner cams (e.g. tarik after snap) have
    # a strong border on all sides and stay untouched.
    bw_raw = max(1, snapped_raw[2] - snapped_raw[0])
    bh_raw = max(1, snapped_raw[3] - snapped_raw[1])
    raw_ar = bw_raw / bh_raw
    is_squarish = 0.65 <= raw_ar <= 1.5
    border_stab = _border_edge_stability(stab, snapped_raw)
    is_cutout = is_squarish and border_stab < CUTOUT_BORDER_STAB
    if is_cutout:
        raw = _head_shoulders_box(cx, cy, fh, W, H)
        used_mode = f"{used_mode}+cutout"
        # No rectangular overlay to trim inside — the box IS the head-and-shoulders crop.
        trimmed = list(raw)
        trim_diag = {"top": 0, "bot": 0, "left": 0, "right": 0, "cutout": True}
    else:
        raw = snapped_raw
        trimmed, trim_diag = _trim_box(frames, snapped_raw, face_px)
        trim_diag["snap"] = snap_delta
    tx0, ty0, tx1, ty1 = trimmed
    if tx1 - tx0 < 20 or ty1 - ty0 < 20:
        return None
    box_norm = [tx0 / W, ty0 / H, tx1 / W, ty1 / H]
    raw_box_norm = [raw[0] / W, raw[1] / H, raw[2] / W, raw[3] / H]
    face_frac = _face_in_box_frac(frames, trimmed, max_face_h=max_face_h)
    return {"box": box_norm, "raw_box": raw_box_norm,
            "face_frac": round(face_frac, 3), "samples": len(frames),
            "size": [W, H], "raw_box_px": raw, "trimmed_box_px": trimmed, "trim": trim_diag,
            "face_px": face_px, "face_median": [cx / W, cy / H, fh / H],
            "mode": used_mode, "cutout": bool(is_cutout),
            "other_face_bboxes": [list(b) for b in other_face_bboxes]}


def detect_multiwindow(src, source_duration=None, span_s=MULTIWIN_SPAN_S, n=MULTIWIN_N,
                       mode="busy", max_face_h=MAX_FACE_H):
    """Sample `n` frames evenly across up to `span_s` seconds of the source (or the whole
    source when it's shorter) and run one detection. Detecting on a wide window is more
    robust than three narrow 10-second windows medianed together, because chat text and
    scrolling overlays need to be sampled often enough to accumulate reliable temporal std."""
    dur = float(source_duration) if source_duration else _probe_duration(src)
    if dur <= 0:
        return None
    window = min(dur, float(span_s))
    start = max(0.0, (dur - window) / 2.0)
    r = detect(src, start, window, n=n, mode=mode, max_face_h=max_face_h)
    return r


SOURCE_SPANS = 4           # detect_source: spans beyond the middle one on a source this many
                           # times longer than a span


def source_spans(dur, span_s=MULTIWIN_SPAN_S, spans=SOURCE_SPANS):
    """[(start, length)] source seconds detect_source reads: the middle span (detect_multiwindow's),
    then `spans` more spread evenly over a source longer than 1.5 spans."""
    dur = float(dur)
    window = min(dur, float(span_s))
    out = [(max(0.0, (dur - window) / 2.0), window)]
    if dur > 1.5 * window:
        for i in range(int(spans)):
            s = min(max(0.0, (i + 0.5) * dur / spans - window / 2.0), dur - window)
            if all(abs(s - o[0]) >= window / 2.0 for o in out):
                out.append((s, window))
    return out


def detect_source(src, source_duration=None, span_s=MULTIWIN_SPAN_S, n=MULTIWIN_N, mode="auto",
                  max_face_h=MAX_FACE_H, accept=None):
    """The streamer cam anywhere in the source (detect() on source_spans): the first
    result whose face is in its box in >= `accept` (FACE_ACCEPT) of the span's samples, with
    "span" = (start, length); None when no span shows one (the source has no cam)."""
    dur = float(source_duration) if source_duration else _probe_duration(src)
    if dur <= 0:
        return None
    accept = FACE_ACCEPT if accept is None else accept
    for start, window in source_spans(dur, span_s):
        r = detect(src, start, window, n=n, mode=mode, max_face_h=max_face_h)
        if r is not None and r["face_frac"] >= accept:
            return dict(r, span=[round(start, 3), round(window, 3)])
    return None


# ---------------------------------------------------------------- profile IO

def verify_box(src, start, dur, box_norm, n=15, max_face_h=MAX_FACE_H):
    """Face-in-box fraction over the clip window at the saved profile box."""
    frames = sample_native(src, start, dur, n=n)
    if not frames:
        return 0.0, 0
    W, H = frames[0].shape[1], frames[0].shape[0]
    x0, y0, x1, y1 = box_norm
    box_px = [int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)]
    return _face_in_box_frac(frames, box_px, max_face_h=max_face_h), len(frames)


def get_or_detect(channel, src, start, dur, source_duration=None, cache_dir=None,
                  log=None, mode="busy", max_face_h=MAX_FACE_H, profile_key=None):
    """Return (box_norm, raw_box_norm, from_profile, face_frac). `raw_box_norm` is the
    untrimmed grow-box in normalized coords; callers use it to compute a safety margin for
    the game crop so the trimmed cam's overlay border doesn't bleed into the game band.

    Uses the saved profile when available and still valid on this clip (face inside it in
    >= FACE_IN_BOX_MIN of samples); else runs detect_multiwindow (first clip) or per-clip
    detect (later clip where the saved box no longer holds) — the per-clip re-detect does
    NOT overwrite the saved profile. If a saved profile predates the raw-box field the
    trimmed box is returned as raw too, which is a safe degradation (game crop just gets no
    added margin).

    `profile_key`: the key under which to store/read the profile. Defaults to `channel` for
    backwards compat (gamestack). Reaction/screencast pass `f"{channel}:reaction"` /
    `f"{channel}:screencast"` so a streamer who games and reacts doesn't share one box.
    """
    log = log or (lambda *a, **k: None)
    key = profile_key or channel
    profiles = load_profiles(cache_dir)
    saved = profiles.get(key) if key else None
    if saved and saved.get("box"):
        frac, _ = verify_box(src, start, dur, saved["box"], max_face_h=max_face_h)
        if frac >= FACE_IN_BOX_MIN:
            raw = saved.get("raw_box") or saved["box"]
            return saved["box"], raw, True, frac
        log(f"facecam: saved profile for {key!r} shows face in {frac:.0%} of samples "
            f"(< {FACE_IN_BOX_MIN:.0%}); re-detecting for this clip only")
        one = detect(src, start, dur, mode=mode, max_face_h=max_face_h)
        if one is None:
            return None, None, False, 0.0
        return one["box"], one["raw_box"], False, one["face_frac"]
    # first clip of this channel: multi-window detection, save
    result = detect_multiwindow(src, source_duration=source_duration, mode=mode,
                                max_face_h=max_face_h)
    if result is None:
        return None, None, False, 0.0
    if key:
        save_profile(key, result["box"], src, cache_dir=cache_dir,
                     extra={"face_frac": result["face_frac"], "size": result.get("size"),
                            "raw_box": result["raw_box"], "mode": result.get("mode"),
                            "cutout": bool(result.get("cutout", False)),
                            "other_face_bboxes": result.get("other_face_bboxes") or []})
    return result["box"], result["raw_box"], False, result["face_frac"]


# ---------------------------------------------------------------- crop planners (pure)

def _even(v):
    v = int(v)
    return v - (v % 2)


def game_crop(src_w, src_h, cam_box_px, mode="tight", top_h=1220, out_w=1080):
    """Source-pixel crop rectangle for the game band.

    - Returns (gw, gx, fit_blur): use gw wide at x=gx, full source height (cropped to the band
      by ffmpeg). fit_blur=False means the crop cover-scales to `out_w x top_h`; True means
      it fits to `out_w` wide over a blurred vertical fill.
    - Never overlaps `cam_box_px`'s x-range. Placed centred within the wider side (left or
      right of the cam). When `tight` doesn't fit outside the cam, silently degrades to
      fit-with-blur at the largest available width.
    """
    src_w = int(src_w); src_h = int(src_h)
    tight = _even(round(src_h * out_w / max(1, top_h)))
    tight = max(2, min(tight, _even(src_w)))
    cx0, _, cx1, _ = [int(v) for v in cam_box_px]
    left = _even(max(0, cx0))
    right = _even(max(0, src_w - cx1))
    room = max(left, right)
    if mode == "wide":
        wide = _even(round(1.8 * tight))
        gw = min(wide, room) if room > 0 else wide
        fit = True
    else:                                    # "tight" or unknown genre defaults to tight
        if tight <= room:
            gw = tight
            fit = False
        else:
            gw = max(2, room)                # fall back to fit at max avail
            fit = True
    gw = _even(max(2, gw))
    # centre within the wider side
    if left >= right:
        gx = _even(max(0, (left - gw) // 2))
    else:
        gx = _even(max(cx1, cx1 + (right - gw) // 2))
    gx = _even(max(0, min(src_w - gw, gx)))
    return gw, gx, fit


def cam_crop(src_w, src_h, cam_box_px, face_median_norm, bot_h=700, out_w=1080):
    """Source crop for the cam band. Landscape and near-square cams cover-fill the band;
    only truly portrait cams (aspect < `CAM_CROP_PORTRAIT_AR`) fit-with-blur.

    Rationale: the spec requires the cam band to be filled by the cam for anything but
    a portrait cam. The threshold was tightened from `box_ar < 1.0` to a lower value so
    that near-square heads-and-shoulders boxes (cutout-cam replacements come out at
    aspect ~1.0) cover-fill instead of falling into the fit-with-blur branch, which
    would produce a blurred strip below the cam. Real streamer cams sit at ≥ 0.9;
    the only cases we ship blur for are TikTok-style 9:16 cams and similar.

    When covering: if the box is not tall enough to produce a `band_ar`-wide crop
    (bh × band_ar > bw), expand the crop RECTANGLE (not the box) outward from the cam
    box's centre so the band aspect fits — the extra rows/columns come from the source
    around the cam, which is more visually complete than a blur strip.

    Returns (cw, ch, cx, cy, fit_blur).
    """
    src_w = int(src_w); src_h = int(src_h)
    x0, y0, x1, y1 = [int(v) for v in cam_box_px]
    bw, bh = max(2, x1 - x0), max(2, y1 - y0)
    band_ar = out_w / max(1, bot_h)
    box_ar = bw / bh
    # Portrait cam: box is markedly taller than wide. Fit-with-blur to keep the face.
    if box_ar < CAM_CROP_PORTRAIT_AR:
        return _even(bw), _even(bh), _even(x0), _even(y0), True
    # Landscape / square cam: cover-fill the band. Pick the largest cover rectangle that
    # matches the band aspect and covers the cam box (extending outside the box on the
    # side that isn't the limiting one). Centred on the face.
    fcx = int(face_median_norm[0] * src_w) if face_median_norm else (x0 + x1) // 2
    face_cy = int(face_median_norm[1] * src_h) if face_median_norm else (y0 + y1) // 2
    if box_ar >= band_ar:
        # box is wider than the band aspect: the box's height is the limiting axis. Take
        # the box's full height and crop a band-aspect slice from it (band-aspect width
        # < box width), centred horizontally on the face.
        ch = _even(bh)
        cw = _even(round(band_ar * ch))
        cw = max(2, min(cw, _even(bw)))
        cx = _even(max(x0, min(x1 - cw, fcx - cw // 2)))
        cy = _even(max(0, min(src_h - ch, y0)))
        return cw, ch, cx, cy, False
    # box_ar < band_ar: the box's WIDTH is the limiting axis. Take the full box width and
    # a band-aspect slice of its height, face at ~40% from the top. Never leave the box:
    # widening into the source around a cam inset pulls content (charts, game, page) into
    # the cam band (adamkhoo R2).
    cw = _even(bw)
    ch = max(2, min(_even(round(cw / band_ar)), _even(bh)))
    cy = _even(max(y0, min(y1 - ch, face_cy - int(0.4 * ch))))
    return cw, ch, _even(x0), cy, False


def game_genre_mode(cfg, game_name):
    """Look up a game name in `render.game_genres` (lowercase key), default 'tight'."""
    if not game_name:
        return "tight"
    m = ((cfg or {}).get("render") or {}).get("game_genres") or {}
    return str(m.get(str(game_name).strip().lower(), "tight"))


# ---------------------------------------------------------------- content region

# --- content region constants (reaction video / screencast content band) ---
CONTENT_MIN_FRAC = 0.15    # a candidate content region smaller than 15% of the frame is
                           # rejected: the streamer is on a talking-head shot or the reacted
                           # video isn't up. Caller falls back to `track` in that case.
CONTENT_CHANGE_STD = 3.0   # a pixel with temporal std at/above this over the sample window
                           # is "changing" (video plays here — a reacted YouTube video, a
                           # trading chart with moving prices, a coding session with typing).
                           # Round 2 calibration: 6.0 rejected compressed Twitch VODs where
                           # a Minecraft pan or a candlestick tick only moves a small subset
                           # of pixels per frame; 3.0 catches them without letting page
                           # chrome (chat text render antialiasing) qualify as video.
CONTENT_CHANGE_MIN_FRAC = 0.10   # a candidate content region must have this fraction of its
                                  # pixels changing across the window — otherwise it's static
                                  # (chrome, a poster, a slide) and doesn't qualify as video.
CONTENT_EDGE_STAB = 0.85   # rows/cols at or above this stability mean are player-window /
                           # slide borders — snap the content rectangle to the nearest one.
CONTENT_MIN_SIDE = 200     # source pixels — a candidate content region shorter than this on
                           # either side is either detector noise or a page sidebar; reject.


def _largest_rect_clearing_box(x0, y0, x1, y1, guard):
    """Given content region [x0,y0,x1,y1] and a guard box `guard` = [gx0,gy0,gx1,gy1] that
    intersects it, return the biggest-area of the four axis-aligned sub-rectangles that
    live entirely outside the guard: left / right / above / below the guard. `None` when
    no sub-rectangle clears the guard.
    """
    gx0, gy0, gx1, gy1 = guard
    cands = []
    left = (x0, y0, min(x1, gx0), y1)
    right = (max(x0, gx1), y0, x1, y1)
    above = (x0, y0, x1, min(y1, gy0))
    below = (x0, max(y0, gy1), x1, y1)
    for r in (left, right, above, below):
        rx0, ry0, rx1, ry1 = r
        if rx1 - rx0 >= CONTENT_MIN_SIDE and ry1 - ry0 >= CONTENT_MIN_SIDE:
            cands.append((r, (rx1 - rx0) * (ry1 - ry0)))
    if not cands:
        return None
    return max(cands, key=lambda x: x[1])[0]


def _snap_to_edge(stab_line, start, end, threshold=CONTENT_EDGE_STAB, look=30):
    """Slide `start` outward and `end` inward within `look` px looking for a stable border
    line (mean edge-stability > threshold). Returns the snapped (start, end)."""
    import numpy as np
    n = len(stab_line)
    if n == 0:
        return start, end
    # walk `start` outward (toward 0) while we can find a stronger stable line before it
    best_start = start
    for d in range(1, look + 1):
        p = start - d
        if p < 0:
            break
        if stab_line[p] >= threshold:
            best_start = p
            break
    best_end = end
    for d in range(1, look + 1):
        p = end + d
        if p >= n:
            break
        if stab_line[p] >= threshold:
            best_end = p + 1
            break
    return best_start, best_end


def detect_content(frames, cam_guard_box_px=None, exclude_boxes=None,
                   min_frac=CONTENT_MIN_FRAC, min_change_frac=CONTENT_CHANGE_MIN_FRAC):
    """Find the reacted-video / on-screen-content rectangle. Returns [x0,y0,x1,y1] px or
    None (no confident region — the caller falls back to `track`).

    Signals:
      - Per-pixel temporal std over the sample. Pixels where a video plays cluster
        above CONTENT_CHANGE_STD (video motion). Chrome, tabs, sidebars, chat and desktop
        stay well below.
      - Mask out the cam guard box before searching (the cam has plenty of change).
      - Also mask out `exclude_boxes` — other stable-position face clusters, i.e.
        secondary cam insets like the two-corner PIP layout on kai_speed. Without this
        the largest CC lands on the OTHER cam and the streamer's actual video content
        (Minecraft, in kai_speed) never registers.
      - Bounding box of the largest connected component of "changing" pixels IS the
        content region. Snap each side to a stable straight edge (player border, slide
        edge) or the frame edge.
      - Reject when the region is smaller than min_frac of the frame area, or when the
        density of changing pixels inside it is below min_change_frac (a sparse, cursor-
        only region — the streamer is only talking).

    When the region intersects the cam guard box (cam sits on top of a full-screen video),
    take the largest axis-aligned sub-rectangle of the region that clears the guard.
    """
    import cv2
    import numpy as np
    if not frames:
        return None
    H, W = frames[0].shape[:2]
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
    std = gray.std(axis=0)
    change = (std >= CONTENT_CHANGE_STD).astype(np.uint8)
    if cam_guard_box_px is not None:
        gx0, gy0, gx1, gy1 = [int(v) for v in cam_guard_box_px]
        change[max(0, gy0):min(H, gy1), max(0, gx0):min(W, gx1)] = 0
    # Round 3: exclude secondary-inset face bboxes (e.g. kai_speed's second PIP cam)
    # so the largest CC doesn't land on the other cam.
    exclude_pad = 40
    if exclude_boxes:
        for eb in exclude_boxes:
            ex0, ey0, ex1, ey1 = [int(v) for v in eb]
            ex0 -= exclude_pad; ey0 -= exclude_pad
            ex1 += exclude_pad; ey1 += exclude_pad
            change[max(0, ey0):min(H, ey1), max(0, ex0):min(W, ex1)] = 0
    # close small gaps so a mostly-uniform-but-briefly-static video region is one CC
    change = cv2.dilate(change, np.ones((5, 5), np.uint8), iterations=2)
    change = cv2.erode(change, np.ones((3, 3), np.uint8), iterations=1)
    n_ccs, labels, cc_stats, _ = cv2.connectedComponentsWithStats(change, connectivity=8)
    if n_ccs <= 1:
        return None
    # Round 3: pick the largest CC whose BOUNDING-BOX area passes the min_frac floor. A
    # small dense CC on a secondary cam inset (kai_speed R2: Speed's top-left PIP cam
    # takes ~10% of the frame; YuNet doesn't see its face so it isn't in exclude_boxes)
    # otherwise wins the "largest CC" pick even though the Minecraft video CC has a
    # bigger bbox. Excluding sub-15% bbox CCs skips those secondaries in one pass.
    frame_area = float(H * W)
    lbl = -1
    best_area = -1
    for i in range(1, n_ccs):
        bx, by, bw, bh, area = cc_stats[i]
        bbox_area = float(bw * bh)
        if bbox_area / frame_area < min_frac:
            continue
        if area > best_area:
            best_area = int(area)
            lbl = i
    if lbl < 0:
        return None
    cx, cy, cw, ch, _ = cc_stats[lbl]
    x0, y0, x1, y1 = int(cx), int(cy), int(cx + cw), int(cy + ch)
    # density check inside the raw bbox on the ORIGINAL changing mask (pre-morph)
    raw_change = (std >= CONTENT_CHANGE_STD).astype(np.uint8)
    if cam_guard_box_px is not None:
        gx0, gy0, gx1, gy1 = [int(v) for v in cam_guard_box_px]
        raw_change[max(0, gy0):min(H, gy1), max(0, gx0):min(W, gx1)] = 0
    if exclude_boxes:
        for eb in exclude_boxes:
            ex0, ey0, ex1, ey1 = [int(v) for v in eb]
            ex0 -= exclude_pad; ey0 -= exclude_pad
            ex1 += exclude_pad; ey1 += exclude_pad
            raw_change[max(0, ey0):min(H, ey1), max(0, ex0):min(W, ex1)] = 0
    inside = raw_change[y0:y1, x0:x1]
    if inside.size == 0 or float(inside.mean()) < min_change_frac:
        return None
    # snap each side to a stable edge if there is one within `look` px, else keep the CC edge.
    stab = _edge_stability_map(frames)
    xs, xe = _snap_to_edge(stab[max(0, (y0 + y1) // 2), :], x0, x1)
    ys, ye = _snap_to_edge(stab[:, max(0, (x0 + x1) // 2)], y0, y1)
    x0, x1, y0, y1 = xs, xe, ys, ye
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(W, x1); y1 = min(H, y1)
    if x1 - x0 < CONTENT_MIN_SIDE or y1 - y0 < CONTENT_MIN_SIDE:
        return None
    frame_area = W * H
    area = (x1 - x0) * (y1 - y0)
    if area / max(1, frame_area) < min_frac:
        return None
    # cam guard box: if it intersects the content rect, carve out the largest sub-rect
    if cam_guard_box_px is not None:
        gx0, gy0, gx1, gy1 = [int(v) for v in cam_guard_box_px]
        overlap_x = min(x1, gx1) > max(x0, gx0)
        overlap_y = min(y1, gy1) > max(y0, gy0)
        if overlap_x and overlap_y:
            sub = _largest_rect_clearing_box(x0, y0, x1, y1, [gx0, gy0, gx1, gy1])
            if sub is None:
                return None
            sub_area = (sub[2] - sub[0]) * (sub[3] - sub[1])
            if sub_area < 0.5 * area:
                # lost more than half the region — caller should switch to game_crop-style
                # tight crop of the region against the guard. Signal that with None so the
                # planner uses the guard-clearing tight crop instead.
                return None
            x0, y0, x1, y1 = sub
    if x1 - x0 < CONTENT_MIN_SIDE or y1 - y0 < CONTENT_MIN_SIDE:
        return None
    # final area check on possibly carved rect
    if (x1 - x0) * (y1 - y0) / max(1, frame_area) < min_frac:
        return None
    return [int(x0), int(y0), int(x1), int(y1)]


def verify_content(src, start, dur, content_box_norm, n=15,
                   min_change_frac=CONTENT_CHANGE_MIN_FRAC):
    """Sample the clip window and check the saved content box still holds sustained
    change (fraction of pixels with std >= CONTENT_CHANGE_STD >= min_change_frac). React
    streamers move / resize / hide the player window more often than gamers move cams,
    so this must be a real per-clip check, not a formality."""
    frames = sample_native(src, start, dur, n=n)
    if not frames:
        return 0.0, 0
    import cv2
    import numpy as np
    H, W = frames[0].shape[:2]
    x0 = max(0, int(round(content_box_norm[0] * W)))
    y0 = max(0, int(round(content_box_norm[1] * H)))
    x1 = min(W, int(round(content_box_norm[2] * W)))
    y1 = min(H, int(round(content_box_norm[3] * H)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0, len(frames)
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
    std = gray.std(axis=0)
    return float((std[y0:y1, x0:x1] >= CONTENT_CHANGE_STD).mean()), len(frames)


def activity_centroid(src, start, dur, guard_box_px=None, n=15):
    """Source-x coordinate of the median column-weighted activity centroid: where pixels
    change (cursor, typing, chart ticks, scrolling) is where the presenter is drawing
    attention. Falls back to frame centre when no measurable activity."""
    frames = sample_native(src, start, dur, n=n)
    if not frames:
        return None
    import cv2
    import numpy as np
    H, W = frames[0].shape[:2]
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
    std = gray.std(axis=0)
    change = (std >= CONTENT_CHANGE_STD).astype(np.float32)
    if guard_box_px is not None:
        gx0, gy0, gx1, gy1 = [int(v) for v in guard_box_px]
        change[max(0, gy0):min(H, gy1), max(0, gx0):min(W, gx1)] = 0.0
    col_energy = change.sum(axis=0)
    if float(col_energy.sum()) <= 0:
        return None
    xs = np.arange(W, dtype=np.float32)
    return float((xs * col_energy).sum() / col_energy.sum())


# ---------------------------------------------------------------- CLI (diagnostic)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Detect the streamer's webcam inset (diagnostic).")
    ap.add_argument("src")
    ap.add_argument("start", type=float, nargs="?", default=0.0)
    ap.add_argument("dur", type=float, nargs="?", default=10.0)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--channel", default=None)
    ap.add_argument("--cache-dir", default=None)
    a = ap.parse_args()
    if a.channel:
        box, raw_box, from_prof, frac = get_or_detect(a.channel, a.src, a.start, a.dur,
                                                       cache_dir=a.cache_dir)
        if box is None:
            print(json.dumps({"box": None, "reason": "no inset"}))
        else:
            print(json.dumps({"channel": a.channel, "box": box, "raw_box": raw_box,
                              "from_profile": from_prof, "face_frac": frac}))
    else:
        r = detect(a.src, a.start, a.dur, n=a.n)
        print(json.dumps(r if r else {"box": None}, indent=2))
