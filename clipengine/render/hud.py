#!/usr/bin/env python3
"""HUD profile handling for the poker `hudband` layout.

A show's HUD is a fixed template (player tiles + pot bar + outs strip in the same normalized
locations every episode). We measure those regions once per show and reuse them, rather than
re-detecting per clip. Profiles live in the config under
`render.hud_profiles.<name>`; each region is a normalized `[x0, y0, x1, y1]` rectangle. The
vertical -> profile map lives under `render.vertical_hud_profile`.

Per clip window we sample the source at low res every 0.5 s, build a per-region reference from
the temporal median of the region across samples, and mark a sample PRESENT when its MAD to the
reference is below a calibrated threshold. Presence intervals are merged (short gaps closed,
tiny blips dropped) and become ffmpeg `enable=` expressions for the band overlays -- so a
region's overlay only draws while the HUD is actually on screen, and the band shows plain
colour when it isn't.

Pure functions on numpy arrays for unit-testability; the ffmpeg sampler is a thin wrapper.
"""
import json
import subprocess
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- sampling defaults
SAMPLE_W = 480              # sample frames downscaled to this width (kept from 4f/4g so cache reuse works)
SAMPLE_HZ = 2.0             # spec: every 0.5 s
SAMPLE_PAD_S = 4.0          # a bit of pad each side helps stabilise the median

# ---------------------------------------------------------------- presence gating defaults
# Presence signal is per-region Sobel edge density (fraction of pixels above a magnitude
# threshold). Poker HUD tiles are dense text + card art on a dark backdrop: edge_density
# consistently reads 0.35-0.70 across every game state (2-tile / 3-tile / different POT
# amounts). Absent regions (face close-ups, GTO Wizard, table felt) read 0.02-0.08. The
# 0.15 threshold sits well above the noise floor and well below the loosest HUD frame.
# MAD-to-reference was tried first (see git history): it fails whenever the game state
# changes mid-clip (2-tile vs 3-tile stack), because the reference locks to one template
# and reads the other as absent. Edge density is game-state-agnostic.
PRESENT_EDGE_MIN = 0.15
PRESENT_MAD_MAX = 18.0      # secondary MAD check for the region_reference-based API (tests)
MERGE_GAP_S = 1.0           # close intervals separated by less than this
MIN_INTERVAL_S = 0.5        # drop present-blips shorter than this
FALLBACK_PLAYERS_MIN_FRAC = 0.20   # players present in < 20% of the window -> fall back to track

# ---------------------------------------------------------------- band packing defaults
BAND_MIN_H = 360
BAND_MAX_H = 640
BAND_MARGIN = 24            # margin around the band edges
BAND_SCALE_CAP = 3.0        # never upscale a source tile past 3x
OUT_W = 1080
HUDBAND_BG = "0x0b0b0f"


# ================================================================ profile lookup

def load_profile(cfg, name):
    """`cfg["render"]["hud_profiles"][name]` -> dict of region name -> [x0,y0,x1,y1] normalized.
    Raises KeyError with a clear message when the profile is missing/misspelled."""
    profiles = ((cfg or {}).get("render") or {}).get("hud_profiles") or {}
    if name not in profiles:
        raise KeyError(f"HUD profile {name!r} not in render.hud_profiles "
                       f"(have: {sorted(profiles)})")
    prof = profiles[name]
    regions = {}
    for k, v in prof.items():
        if k.startswith("_"):
            continue                                       # notes / metadata
        if not (isinstance(v, list) and len(v) == 4):
            continue
        x0, y0, x1, y1 = (float(x) for x in v)
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"HUD profile {name!r}: region {k!r} outside [0,1] or inverted: {v}")
        regions[k] = [x0, y0, x1, y1]
    return regions


def profile_for_vertical(cfg, vertical):
    """Vertical -> profile name from render.vertical_hud_profile. None when unmapped."""
    m = ((cfg or {}).get("render") or {}).get("vertical_hud_profile") or {}
    return m.get(vertical)


def profile_for_broadcast(cfg, broadcast):
    """Broadcast (the source's channel / discover creator) -> profile name from
    render.hud_profile_by_broadcast, matched case-insensitively. None when unmapped. One
    generic poker profile measured on one show sliced the graphics of three other broadcasts;
    each broadcast can have its own."""
    if not broadcast:
        return None
    m = ((cfg or {}).get("render") or {}).get("hud_profile_by_broadcast") or {}
    key = " ".join(str(broadcast).split()).lower()
    return next((v for k, v in m.items() if not k.startswith("_")
                 and " ".join(str(k).split()).lower() == key), None)


def region_pixels(region, src_w, src_h):
    """Normalized [x0,y0,x1,y1] -> (x, y, w, h) source pixels, even-sized and in-frame."""
    x0, y0, x1, y1 = region
    x = int(round(x0 * src_w))
    y = int(round(y0 * src_h))
    w = int(round((x1 - x0) * src_w))
    h = int(round((y1 - y0) * src_h))
    w -= w % 2
    h -= h % 2
    x = max(0, min(x, src_w - max(2, w)))
    y = max(0, min(y, src_h - max(2, h)))
    return x, y, max(2, w), max(2, h)


# ================================================================ profile vs the feed

OVERLAY_SAME_LUMA = 12.0     # a broadcast graphic reads the same (+-12 gray levels) in every shot
OVERLAY_EDGE_SHOTS = 0.7     # ... and is an edge in >= 70 % of the shots' mean frames
MISMATCH_EDGE_CUT = 0.15     # a band region whose side runs >= 15 % through graphics slices them


def overlay_mask(frames_bgr, cut_times):
    """Pixels of the feed's own graphics: the same gray level (+-OVERLAY_SAME_LUMA) and an edge in
    the mean frame of every camera shot. The scene changes from shot to shot, the HUD does not.
    `cut_times` splits `frames_bgr` (sampled at 2 fps) into shots, in sample-index / 2 seconds.
    None when the window has fewer than two shots (nothing to tell the HUD from the scene)."""
    import cv2
    bounds = [0] + [int(round(float(c) * SAMPLE_HZ)) for c in cut_times] + [len(frames_bgr)]
    shots = [frames_bgr[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]
    if len(shots) < 2:
        return None
    means = [np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in s]).astype(np.float32)
             .mean(axis=0) for s in shots]
    G = np.stack(means)
    same = (np.abs(G - np.median(G, axis=0)) < OVERLAY_SAME_LUMA).all(axis=0)
    k = np.ones((3, 3), np.uint8)
    edges = np.stack([cv2.dilate((cv2.Canny(m.astype(np.uint8), 60, 160) > 0).astype(np.uint8),
                                 k) > 0 for m in means])
    return same & (edges.mean(axis=0) >= OVERLAY_EDGE_SHOTS)


def region_edge_cuts(mask, region):
    """{side: share of the region's side that runs through overlay pixels} (region normalized)."""
    h, w = mask.shape
    x0, y0, x1, y1 = (int(round(v)) for v in (region[0] * w, region[1] * h,
                                              region[2] * w, region[3] * h))
    out = {}
    for side, (a, b, c, d) in {"left": (x0, y0, x0, y1), "right": (x1, y0, x1, y1),
                               "top": (x0, y0, x1, y0), "bottom": (x0, y1, x1, y1)}.items():
        if a == c:
            band = mask[max(0, b):min(h, d), max(0, a - 1):a + 2].any(axis=1) \
                if 0 < a < w - 1 else np.zeros(0, bool)
        else:
            band = mask[max(0, b - 1):b + 2, max(0, a):min(w, c)].any(axis=0) \
                if 0 < b < h - 1 else np.zeros(0, bool)
        out[side] = round(float(band.mean()), 3) if band.size else 0.0
    return out


def profile_mismatch(profile, mask, max_cut=MISMATCH_EDGE_CUT):
    """Why a HUD profile does not fit this feed, or None. A profile measured on one
    show cut another broadcast's pot box (its pot region's right side ran 22 % through
    graphics) and a third one's player tiles (players bottom 24 %); a band region whose
    side runs through the feed's own graphics (`overlay_mask`) slices them. `mask` None (one
    camera shot) cannot tell: no mismatch."""
    if mask is None:
        return None
    for name in ("players", "pot"):
        if name not in profile:
            continue
        cuts = region_edge_cuts(mask, profile[name])
        side, share = max(cuts.items(), key=lambda kv: kv[1])
        if share >= max_cut:
            return f"{name} region's {side} side runs {share:.0%} through the feed's graphics"
    return None

# ================================================================ frame sampling

def _probe_size(src):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "json", str(src)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    return int(s["width"]), int(s["height"])


def _probe_duration(src):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(src)], stdout=subprocess.PIPE, text=True, check=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def sample_frames(src, start, dur, hz=SAMPLE_HZ, width=SAMPLE_W, src_size=None,
                  total_dur=None, pad_s=SAMPLE_PAD_S):
    """List of (t_clip, frame_bgr) tuples covering [start, start+dur] plus `pad_s` each side
    (clamped to source). t is clip-relative seconds (matches the clip's t axis)."""
    sw, sh = src_size or _probe_size(src)
    height = max(2, int(round(sh * width / sw / 2)) * 2)
    total = float(total_dur) if total_dur is not None else _probe_duration(src)
    lo = max(0.0, float(start) - float(pad_s))
    hi = float(start) + float(dur) + float(pad_s)
    if total > 0:
        hi = min(hi, total)
    win = max(0.1, hi - lo)
    fps = max(0.5, float(hz))
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{lo:.3f}", "-t", f"{win:.3f}",
           "-i", str(src), "-an", "-vf", f"fps={fps},scale={width}:{height}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    size = width * height * 3
    out = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        i = 0
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            t = lo + i / fps - float(start)                # clip-relative
            out.append((round(t, 3), np.frombuffer(buf, np.uint8).reshape(height, width, 3).copy()))
            i += 1
    return out


# ================================================================ presence gating

def _sample_region_px(sample_w, sample_h, region, src_w, src_h):
    """Region as pixel coords inside a SAMPLE frame (which is `sample_w`x`sample_h`, source
    scaled uniformly). Returns (x, y, w, h). Widths are min 2."""
    x0, y0, x1, y1 = region
    x = max(0, int(round(x0 * sample_w)))
    y = max(0, int(round(y0 * sample_h)))
    w = max(2, min(sample_w - x, int(round((x1 - x0) * sample_w))))
    h = max(2, min(sample_h - y, int(round((y1 - y0) * sample_h))))
    return x, y, w, h


def region_crops(samples, region):
    """List of (t, region_crop_bgr) — region indexed at each sample. Returns [] on empty."""
    if not samples:
        return []
    _, f0 = samples[0]
    sh, sw = f0.shape[:2]
    x, y, w, h = _sample_region_px(sw, sh, region, sw, sh)
    return [(t, f[y:y + h, x:x + w]) for t, f in samples]


def _edge_density(crop):
    """Fraction of pixels above a Sobel threshold — HUD graphics (text, cards, tile borders)
    push this to 0.20+; a face / wall / table felt sits under ~0.06. Robust proxy for 'is HUD
    visible in this region right now?' Falls back to numpy central differences without cv2."""
    if crop is None or crop.size == 0:
        return 0.0
    if crop.ndim == 3:
        gray = crop.astype(np.float32).mean(axis=2)
    else:
        gray = crop.astype(np.float32)
    try:
        import cv2
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
    except ImportError:
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        mag = np.hypot(gx, gy)
    return float((mag >= 40.0).mean())


EDGE_MIN_ABS = 0.10          # regions under this in edge-density are never HUD (face / felt)
EDGE_TOP_FRAC = 0.30         # top 30% of qualifying frames define the HUD reference


def region_reference(crops):
    """Reference image the HUD looks like. Take frames whose region edge density is BOTH above
    EDGE_MIN_ABS AND in the top EDGE_TOP_FRAC of the sampled window; median their region crops.

    This works whether HUD is present in 20% or 90% of the window — the reference stays a HUD
    image rather than getting pulled toward close-up/graphics content. Falls back to the plain
    temporal median when no frames pass the edge floor (region has no HUD to speak of).
    """
    if not crops:
        return np.zeros((1, 1, 3), np.float32)
    edges = [(t, c, _edge_density(c)) for t, c in crops]
    qual = [ec for ec in edges if ec[2] >= EDGE_MIN_ABS]
    if not qual:
        stack = np.stack([c.astype(np.float32) for _, c in crops], axis=0)
        return np.median(stack, axis=0).astype(np.float32)
    qual.sort(key=lambda ec: ec[2], reverse=True)
    keep = max(1, int(round(len(qual) * EDGE_TOP_FRAC)))
    stack = np.stack([c.astype(np.float32) for _, c, _ in qual[:keep]], axis=0)
    return np.median(stack, axis=0).astype(np.float32)


def mad(sample_crop, reference):
    """Mean abs diff between a region sample and the region reference (both same shape).
    Returns float. Empty -> 0.0."""
    if sample_crop.size == 0 or reference.size == 0:
        return 0.0
    a = sample_crop.astype(np.float32)
    b = reference.astype(np.float32)
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).mean())


def presence_marks(crops, edge_min=PRESENT_EDGE_MIN):
    """[(t, present_bool)] over `crops`. Present = per-region Sobel edge density >= edge_min.

    Edge density is a game-state-agnostic HUD signal (see PRESENT_EDGE_MIN note): HUD tiles
    read 0.35-0.70 regardless of which players are in the hand; non-HUD (face, table felt,
    full-screen graphics) reads under 0.10. No reference frame needed — the region either
    has dense text/cards on it right now or it doesn't.
    """
    if not crops:
        return []
    return [(t, _edge_density(c) >= edge_min) for t, c in crops]


def presence_intervals(marks, merge_gap=MERGE_GAP_S, min_dur=MIN_INTERVAL_S):
    """[(t, present)] -> [(a, b)] merged present-intervals. `merge_gap` closes gaps this short;
    intervals shorter than `min_dur` after merging are dropped."""
    if not marks:
        return []
    step = 0.5
    if len(marks) > 1:
        step = max(0.05, marks[1][0] - marks[0][0])
    half = step / 2.0
    raw = []
    for t, present in marks:
        if present:
            a = t - half
            b = t + half
            if raw and a - raw[-1][1] <= 1e-6:
                raw[-1][1] = b
            else:
                raw.append([a, b])
    # merge across small gaps
    merged = []
    for a, b in raw:
        if merged and a - merged[-1][1] < merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    # drop tiny blips
    out = [(round(max(0.0, a), 3), round(b, 3)) for a, b in merged if b - a >= min_dur]
    return out


def enable_expr(intervals):
    """`between(t,a,b)+between(t,c,d)+...` for the overlay `enable=` parameter. Empty -> 0."""
    if not intervals:
        return "0"
    parts = [f"between(t,{a:.3f},{b:.3f})" for a, b in intervals]
    return "+".join(parts)


# ================================================================ band packing

def _row_scale(regions_px, gap, inner_w):
    """Largest uniform scale that fits a row of pixel widths inside inner_w with `gap` between."""
    if not regions_px:
        return float("inf")
    return (inner_w - gap * (len(regions_px) - 1)) / max(1.0, sum(w for _, _, w, _ in regions_px))


def plan_band(profile, src_w, src_h, present_regions=None,
              out_w=OUT_W, min_h=BAND_MIN_H, max_h=BAND_MAX_H,
              margin=BAND_MARGIN, scale_cap=BAND_SCALE_CAP, gap=12):
    """Pack a HUD profile into a 1080-wide dark band.

    Layout is ONE row: `players` on the left, `pot` on the right, at a uniform scale that fills
    the inner width. `outs_strip` is intentionally NOT shown (it's redundant with `pot`).

    Band height = packed content + top/bottom margins, clamped to [min_h, max_h] (scale drops
    when content would overflow max_h).

    Returns {"height": int, "items": [{"name","src":(x,y,w,h),"dst":(x,y,w,h)}]}.
    `present_regions` is optional; it filters the items to only those present at least once.
    Passing None keeps every region in the profile (useful for tests / static planning).
    """
    order = ("players", "pot")
    items_src = []
    for name in order:
        if name not in profile:
            continue
        if present_regions is not None and name not in present_regions:
            continue
        x, y, w, h = region_pixels(profile[name], src_w, src_h)
        items_src.append((name, x, y, w, h))
    if not items_src:
        return None

    inner_w = out_w - 2 * margin
    total_src_w = sum(w for _, _, _, w, _ in items_src)
    s_width = (inner_w - gap * (len(items_src) - 1)) / max(1.0, total_src_w)
    s = min(scale_cap, s_width)

    row_h = int(round(max(h for _, _, _, _, h in items_src) * s))
    content_h = row_h
    band_h = content_h + 2 * margin

    if band_h > max_h:
        target = max(1, max_h - 2 * margin)
        max_src_h = max(h for _, _, _, _, h in items_src)
        s = min(s, target / max(1.0, max_src_h))
        row_h = int(round(max_src_h * s))
        content_h = row_h
        band_h = content_h + 2 * margin

    band_h = max(min_h, min(max_h, band_h))
    band_h -= band_h % 2
    y_row = margin + max(0, (band_h - 2 * margin - content_h) // 2)

    items = []
    row_content_w = int(round(total_src_w * s)) + gap * (len(items_src) - 1)
    x_cursor = margin + max(0, (inner_w - row_content_w) // 2)
    for name, sx, sy, sw, sh in items_src:
        dst_w = max(2, int(round(sw * s)))
        dst_h = max(2, int(round(sh * s)))
        dst_w -= dst_w % 2
        dst_h -= dst_h % 2
        y_cell = int(round(y_row + (row_h - dst_h) / 2))
        items.append({
            "name": name,
            "src": (sx, sy, sw, sh),
            "dst": (x_cursor, y_cell, dst_w, dst_h),
        })
        x_cursor += dst_w + gap
    return {"height": band_h, "items": items}


# ================================================================ top crop safety

def _region_x_range(profile, src_w, name):
    """(x0, x1) source pixel range for a named region, or None when absent."""
    if name not in profile:
        return None
    x, _, w, _ = region_pixels(profile[name], src_w, 100)
    return x, x + w


def crop_slices_region(crop_x, cw, region_x_range):
    """True when [crop_x, crop_x+cw) partially overlaps [rx0, rx1) — i.e. slices it. Both fully
    outside and fully inside are considered non-slicing."""
    rx0, rx1 = region_x_range
    x0 = int(crop_x)
    x1 = x0 + int(cw)
    overlap = max(0, min(x1, rx1) - max(x0, rx0))
    if overlap == 0:
        return False                                       # fully outside
    if x0 <= rx0 and rx1 <= x1:
        return False                                       # region fully inside the crop
    return True


def safe_crop_x(desired_x, cw, src_w, region_x_ranges, subject_box=None,
                inside_min=0.70):
    """Pick a crop-x in [0, src_w-cw] where every region in `region_x_ranges` is either fully
    outside the crop or fully inside it (never sliced). Considered per region — the crop can
    exclude one region and contain another.

    Preference order:
      1. All-regions-outside positions. Score by subject-preserved fraction; require at least
         `inside_min` of the subject box to stay inside the crop.
      2. Non-slicing (mixed) positions where each region is individually outside OR inside.
         Same scoring.
      3. If neither fits, return the clamped `desired_x` (caller stays with its layout).

    `subject_box`: (x, w) of the tracked subject in source pixels. None -> not scored.
    """
    max_x = max(0, int(src_w) - int(cw))
    if max_x <= 0:
        return 0

    # Region-edge anchors. For each region, both fully-outside anchors (crop butts up against
    # each side) and fully-inside anchors (crop's left edge at rx0, right edge at rx1) are worth
    # checking. Plus desired_x itself.
    anchors = {0, max_x, int(max(0, min(max_x, desired_x)))}
    for rx0, rx1 in region_x_ranges:
        for c in (rx1, rx0 - cw, rx0, rx1 - cw):
            if 0 <= c <= max_x:
                anchors.add(int(c))

    def subject_inside_frac(x):
        if not subject_box:
            return 1.0
        sx, sw = subject_box
        overlap = max(0, min(x + cw, sx + sw) - max(x, sx))
        return overlap / max(1.0, sw)

    def per_region_kind(x, rx0, rx1):
        overlap = max(0, min(x + cw, rx1) - max(x, rx0))
        if overlap == 0:
            return "outside"
        if x <= rx0 and rx1 <= x + cw:
            return "inside"
        return "slices"

    all_outside_cands = []
    non_slice_cands = []
    for x in sorted(anchors):
        kinds = [per_region_kind(x, rx0, rx1) for rx0, rx1 in region_x_ranges]
        if "slices" in kinds:
            continue
        non_slice_cands.append(x)
        if all(k == "outside" for k in kinds):
            all_outside_cands.append(x)

    def pick(cands):
        best = None
        best_key = None
        for x in cands:
            frac = subject_inside_frac(x)
            key = (frac, -abs(x - desired_x))
            if best_key is None or key > best_key:
                best_key = key
                best = x
        return best

    outside = pick(all_outside_cands)
    if outside is not None and subject_inside_frac(outside) >= inside_min:
        return int(outside)
    non_slice = pick(non_slice_cands)
    if non_slice is not None:
        return int(non_slice)
    if outside is not None:
        return int(outside)
    return int(max(0, min(max_x, desired_x)))
