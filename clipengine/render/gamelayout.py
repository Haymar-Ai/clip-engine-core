#!/usr/bin/env python3
"""Gamestack v2 / v3: a zoomed-out game band that follows the
action, and a small facecam card.

v2: on the first gamestack Shorts (clips 2620, 2670, 2500) the event was off-centre, the
game too zoomed in, the streamer's cam too big (v1: ~45 % of the source width at one static x, a
576 px cam band, 30 % of the canvas). v2 showed 68 % of the width, followed the action and made the
cam a corner card. v3 on the v2 previews: "the action is only being shown on the top
3rd of the screen. Then a lot of dead space in the middle": the 68 % band was 1080x894 (46 % of
the canvas) over an ambient blur. At 1080 px wide a band showing S of a 16:9 source is
1080 / (S x 1.778) px tall, so the game fills the screen only at a smaller share.

  canvas   a hook header (rows 0..HEADER_H: the band's ambient blur, the hook on it for the whole
           clip), then the game band: GAME_SHARE (>= MIN_SHARE) of the source width at full source
           height, fit to the canvas width (0.50 -> 1080x1214, 63 % of the canvas), its x
           following the action; the captions on the band's lower rows; the cam card right under
           the band in a bottom corner (<= INSET_MAX_AREA of the canvas, the face centred), as
           tall as the rows under the band allow, so it never covers the band (the action) and
           leaves no flat strip on the bottom edge; the rest the band's ambient blur, <= 25 % of
           the canvas. A clip with no face (game only) takes GAME_ONLY_SHARE (1446 rows) and no
           card.
  action   motion energy: frame differences on downscaled gray at TRACK_FPS with the cam box and
           the HUD / overlay plates masked out; a scene cut (most pixels change) and a still
           sample hold; per sample the energy-weighted centroid (motion_samples).
  path     a forward and a backward EMA with a soft dead zone and a max pan speed, averaged (no
           lag either way), clamped to the band's legal x range (smooth_path), simplified to a
           few knots (knots) and rendered as a piecewise-linear crop x expression (pan_expr).

Pure maths except motion_samples' frames; recompose builds the plan and the graph.
"""
import math

OUT_W, OUT_H = 1080, 1920
MIN_SHARE = 0.48           # the band shows at least this share of the source width
GAME_SHARE = 0.50          # default share (render.gamestack.game_share): a 1080x1214 band
GAME_ONLY_SHARE = 0.42     # game only (render.gamestack.game_only_share): 1446 rows, no card
GAME_ONLY_MIN_SHARE = 0.40
HEADER_H = 200             # the hook header above the band (render.gamestack.header_h) ...
HEADER_MAX_H = 200         # ... never taller
BAND_MAX_H = 1270          # a narrow source (4:3, 9:16) is cropped in height past this band height
GAME_ONLY_BAND_MAX_H = 1520
INSET_MAX_AREA = 0.12      # the cam card's share of the canvas (render.gamestack.cam_max_area) ...
INSET_HARD_CAP = 0.14      # ... never above (v2's card was 13-16 %, v1's band 30 %)
INSET_MAX_H = 480
INSET_MAX_W = 864          # 80 % of the canvas width
INSET_MARGIN = 40          # card margin to the canvas edges (under the gate's 4 % edge-band floor)
INSET_AR_MIN, INSET_AR_MAX = 0.8, 16 / 9
INSET_FACE_Y = 0.45        # the face centre sits this far down the card's crop
INSET_MAX_FACE_H = 3.6     # the card's crop is at most this many face heights tall
INSET_BORDER = 4           # px, a light frame around the card
CAM_EDGE_TOL = 0.06        # a cam box this close (of the frame side) to a frame edge is an inset
CAP_HALF = 75              # half of captions.CAPTION_BLOCK_H
CAP_BAND_PAD = 14          # px between the caption block and the band's bottom edge
CAP_BAND_Y = 1325          # the captions' centre line at most (block 1250-1400: the band's foot)
CAP_TEXT_PAD = 20          # px between the caption block and a source text row on the band
CAP_CARD_PAD = 6           # px between the caption block and the card's top
CAPTION_Y_MAX = 1500       # = captions.CAPTION_Y_MAX (the platform's bottom UI below)
FILL_SCALE = 4             # the ambient fill is blurred at 1 / FILL_SCALE of the canvas
FILL_SIGMA = 7             # gblur sigma at that scale (~28 px on the canvas)

TRACK_FPS = 5.0            # motion samples per second
TRACK_W = 320              # scan width (px) of the motion frames
NOISE = 10.0               # gray levels: a smaller frame difference is codec noise
CUT_FRAC = 0.55            # this share of the unmasked pixels changing is a scene cut / whip pan
MIN_ACTIVE_FRAC = 0.002    # below this share changing, a sample has no action (the path holds)
ENERGY_SIGMA = 2.0         # scan px: the difference map's blur before the centroid
EMA_ALPHA = 0.35           # per sample at TRACK_FPS
DEAD_ZONE = 0.04           # of the source width: a target this close to the path does not pan it
MAX_SPEED = 0.20           # of the source width per second
KNOT_TOL = 0.006           # of the source width: the knot path's max distance from the smooth one
MAX_KNOTS = 40             # ffmpeg's expression parser takes ~200 nested if()s; stay well under


def even(v):
    v = int(round(float(v)))
    return v - (v % 2)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------------------------------------------------------------- layout maths

def band_geometry(src_w, src_h, share=GAME_SHARE, out_w=OUT_W, max_h=BAND_MAX_H,
                  min_share=MIN_SHARE):
    """{"gw", "ch", "gh", "share"}: the band's source crop (gw x ch; ch = src_h unless the band
    would be taller than max_h) and its canvas height gh (the crop fit to out_w wide). The share
    is never below min_share (MIN_SHARE; GAME_ONLY_MIN_SHARE for game only)."""
    src_w, src_h = int(src_w), int(src_h)
    share = clamp(float(share), float(min_share), 1.0)
    gw = min(even(src_w), even(share * src_w))
    while gw < min_share * src_w and gw + 2 <= src_w:
        gw += 2
    gh = even(out_w * src_h / float(gw))
    ch = src_h
    if gh > max_h:
        gh = even(max_h)
        ch = min(src_h, even(gw * gh / float(out_w)))
    return {"gw": gw, "ch": ch, "gh": gh, "share": round(gw / float(src_w), 4)}


def is_inset(box, src_w, src_h, tol=CAM_EDGE_TOL):
    """Whether a cam box [x0, y0, x1, y1] (src px) sits at a frame edge, as an overlay inset does
    (a face inside the scene is no inset: the band could not leave it out)."""
    x0, y0, x1, y1 = box
    return (x0 <= tol * src_w or x1 >= (1 - tol) * src_w or y0 <= tol * src_h
            or y1 >= (1 - tol) * src_h)


def cam_free_ranges(src_w, gw, guard):
    """[(lo, hi)] band x ranges whose columns leave the guard box [x0, y0, x1, y1] out: left of it
    and / or right of it; [] when neither side holds the band."""
    src_w, gw = int(src_w), int(gw)
    gx0, gx1 = int(guard[0]), int(guard[2])
    out = []
    if gx0 >= gw:
        out.append((0, (gx0 - gw) - (gx0 - gw) % 2))
    if src_w - gx1 >= gw:
        lo = gx1 + (gx1 % 2)
        if lo <= src_w - gw:
            out.append((lo, src_w - gw))
    return out


def legal_runs(lo, hi, ok, step=2):
    """[(a, b)] maximal runs of x in [lo, hi] (every `step` px) where ok(x) holds."""
    runs, start, prev = [], None, None
    x = int(lo)
    while x <= hi:
        if ok(x):
            start = x if start is None else start
            prev = x
        elif start is not None:
            runs.append((start, prev))
            start = None
        x += step
    if start is not None:
        runs.append((start, prev))
    return runs


def card_size(room_h, max_area=INSET_MAX_AREA, max_h=INSET_MAX_H, max_w=INSET_MAX_W):
    """(w, h) canvas px of the cam card for the `room_h` rows left under the band (the bottom
    margin out): as tall as that room (<= max_h), so it sits right under the band with no blur
    strip on the band's edge or on the canvas's, and as wide as max_area of the canvas allows
    (never above INSET_HARD_CAP) at an aspect inside INSET_AR_MIN..INSET_AR_MAX (inset_crop cuts
    the cam box to that aspect, the face centred). A room taller than the card's narrowest aspect
    allows leaves the card shorter (it keeps the bottom margin); a short room keeps the card
    under max_area at the widest aspect. None when the room holds no card."""
    area = min(float(max_area), INSET_HARD_CAP) * OUT_W * OUT_H
    h = min(float(max_h), float(room_h))
    if h < 2:
        return None
    w = area / h
    if w < INSET_AR_MIN * h:
        h = math.sqrt(area / INSET_AR_MIN)
        w = INSET_AR_MIN * h
    w, h = even(min(w, INSET_AR_MAX * h, float(max_w))), even(h)
    while w > 2 and w * h > area:
        w -= 2
    return w, h


def inset_crop(cam_box, face, card_w, card_h, src_w=None, src_h=None):
    """(cw, ch, cx, cy) src px: the card's crop of the cam box at the card's aspect, as big as the
    box allows (at most INSET_MAX_FACE_H face heights tall), the face centred across and
    INSET_FACE_Y down, kept inside the box. `face` = (cx, cy, h) src px or None (box centre)."""
    x0, y0, x1, y1 = [int(v) for v in cam_box]
    bw, bh = max(2, x1 - x0), max(2, y1 - y0)
    ar = card_w / float(card_h)
    if bw / float(bh) >= ar:
        ch = float(bh)
        cw = ch * ar
    else:
        cw = float(bw)
        ch = cw / ar
    fx, fy, fh = face if face else ((x0 + x1) / 2.0, (y0 + y1) / 2.0, None)
    if fh and ch > INSET_MAX_FACE_H * fh:
        ch = INSET_MAX_FACE_H * fh
        cw = ch * ar
    cw, ch = max(2, even(cw)), max(2, even(ch))
    cx = clamp(fx - cw / 2.0, x0, x1 - cw)
    cy = clamp(fy - INSET_FACE_Y * ch, y0, y1 - ch)
    return cw, ch, even(cx), even(cy)


def card_room(gh, band_y=HEADER_H):
    """Canvas rows under the band a card may take (the bottom margin out)."""
    return OUT_H - INSET_MARGIN - (int(band_y) + int(gh))


def canvas_layout(gh, card=None, side="left", band_y=HEADER_H):
    """Where everything sits on the 1080x1920 canvas: {"header": [0, band_y] (the hook's rows),
    "band": [y0, y1], "card": [x, y, w, h] or None, "cap_y": caption centre line}. The card sits in
    the bottom corner on `side` with its bottom margin, its top on or under the band's bottom
    edge (card_size gives it the room's height: never over the band, so never over the action);
    the captions ride on the band's lower rows (at most CAP_BAND_Y)."""
    gh, band_y = int(gh), int(band_y)
    band = [band_y, band_y + gh]
    box = None
    if card:
        w, h = int(card[0]), int(card[1])
        x = INSET_MARGIN if side == "left" else OUT_W - INSET_MARGIN - w
        box = [x, OUT_H - INSET_MARGIN - h, w, h]
    cap_y = min(band[1] - CAP_BAND_PAD - CAP_HALF, CAP_BAND_Y)
    return {"header": [0, band_y], "band": band, "card": box, "cap_y": int(cap_y)}


def band_text_rows(boxes, band_y, gh, ch, crop_y=0, x_range=None):
    """Canvas y-bands [(y0, y1)] of the source text rows `boxes` ([x0, y0, x1, y1] src px: the
    scan's banner plates, a game's subtitle line or a ticker) the band shows: its crop rows
    crop_y..crop_y + ch fit to gh canvas rows from band_y; only boxes whose columns meet x_range
    (the band's columns along its path)."""
    k = float(gh) / float(ch)
    out = []
    for x0, y0, x1, y1 in boxes:
        if x_range and (x1 <= x_range[0] or x0 >= x_range[1]):
            continue
        top, bot = max(float(y0), float(crop_y)), min(float(y1), float(crop_y + ch))
        if bot > top:
            out.append((int(band_y + (top - crop_y) * k),
                        int(math.ceil(band_y + (bot - crop_y) * k))))
    return out


def caption_y_clear(cap_y, rows, band, card=None):
    """The captions' centre line: cap_y unless its block meets a source text row on the band
    (2670: the game's own subtitle line under our captions); then the nearest line, from the band's
    middle down to the card's top (or CAPTION_Y_MAX), whose block clears every row; cap_y when
    none does."""
    lo = int(band[0] + (band[1] - band[0]) // 2)
    hi = int(card[1] - CAP_HALF - CAP_CARD_PAD) if card else CAPTION_Y_MAX

    def hit(y):
        a, b = y - CAP_HALF - CAP_TEXT_PAD, y + CAP_HALF + CAP_TEXT_PAD
        return any(min(b, r1) > max(a, r0) for r0, r1 in rows)
    if not hit(cap_y):
        return int(cap_y)
    ys = sorted(range(lo, hi + 1, 5), key=lambda y: (abs(y - cap_y), y))
    return next((y for y in ys if not hit(y)), int(cap_y))


def canvas_shares(gh, card=None, band_y=0):
    """{"band", "card", "blur", "header", "bottom", "bottom_blur"}: canvas shares of the band, the
    card's rows off the band, the ambient blur (all the rest), the rows above the band, the rows
    under it and the blur among those."""
    gh, band_y = int(gh), int(band_y)
    canvas = float(OUT_W * OUT_H)
    y1 = band_y + gh
    under = 0
    off_band = 0
    if card:
        cx, cy, cw, chh = (int(v) for v in card)
        rows_off = chh - max(0, min(cy + chh, y1) - max(cy, band_y))
        off_band = cw * rows_off
        under = cw * max(0, cy + chh - max(cy, y1))
    bottom = OUT_W * max(0, OUT_H - y1)
    return {"band": round(OUT_W * gh / canvas, 3), "card": round(off_band / canvas, 3),
            "blur": round(max(0.0, 1.0 - (OUT_W * gh + off_band) / canvas), 3),
            "header": round(OUT_W * max(0, band_y) / canvas, 3),
            "bottom": round(bottom / canvas, 3),
            "bottom_blur": round(max(0, bottom - under) / canvas, 3)}


def fill_frac(gh, card=None, band_y=0):
    """Share of the canvas the ambient blur shows (all but the band and the card)."""
    return canvas_shares(gh, card, band_y)["blur"]


RECT_SPAN = 0.6            # face heights each side of the face centre a border must run along
RECT_REACH = 4.5           # face heights from the face centre a side border is looked for ...
RECT_REACH_V = 3.5         # ... and a top / bottom border
RECT_MIN_STEP = 8.0        # a border's colour step (0-255, median along it, median over frames) ...
RECT_MIN_RATIO = 4.0       # ... and its ratio to the median step of the lines around it


def _step_profile(frames, axis, lo, hi, span):
    """{line: step}: per vertical (axis "x") or horizontal line in lo..hi, the colour step across
    it (max over channels, median along `span`, median over the frames)."""
    import numpy as np
    out = {}
    for p in range(max(1, int(lo)), int(hi)):
        vals = []
        for f in frames:
            if axis == "x":
                d = np.abs(f[span[0]:span[1], p - 1].astype(np.int16) - f[span[0]:span[1], p + 1])
            else:
                d = np.abs(f[p - 1, span[0]:span[1]].astype(np.int16) - f[p + 1, span[0]:span[1]])
            vals.append(float(np.median(d.max(axis=-1) if d.ndim > 1 else d)))
        out[p] = float(np.median(vals))
    return out


def _border(prof):
    if not prof:
        return None
    vals = sorted(prof.values())
    med = vals[len(vals) // 2]
    p = max(prof, key=prof.get)
    return p if prof[p] >= RECT_MIN_STEP and prof[p] >= RECT_MIN_RATIO * max(med, 0.5) else None


def overlay_rect(frames, face):
    """[x0, y0, x1, y1] src px of the facecam overlay around `face` (cx, cy, h src px), read on
    native frames ([BGR]) as the straight colour steps that run along the face's rows / columns
    in every frame: the cam's borders. The sides toward the frame centre must be found; a side
    toward a frame edge with no border runs to that edge (a corner cam) when the edge is within
    reach. None when a side is missing. A static room edge inside the cam can be taken for a
    border: the rect is then tighter than the cam, never wider (the card stays on the cam)."""
    if not frames:
        return None
    H, W = frames[0].shape[:2]
    cx, cy, fh = (float(v) for v in face)
    rows = (max(0, int(cy - RECT_SPAN * fh)), min(H, int(cy + RECT_SPAN * fh)))
    cols = (max(0, int(cx - RECT_SPAN * fh)), min(W, int(cx + RECT_SPAN * fh)))
    sides = {}
    for name, axis, lo, hi, span, edge, toward_edge in (
            ("x0", "x", cx - RECT_REACH * fh, cx - RECT_SPAN * fh, rows, 0, cx < W / 2.0),
            ("x1", "x", cx + RECT_SPAN * fh, cx + RECT_REACH * fh, rows, W, cx >= W / 2.0),
            ("y0", "y", cy - RECT_REACH_V * fh, cy - RECT_SPAN * fh, cols, 0, cy < H / 2.0),
            ("y1", "y", cy + RECT_SPAN * fh, cy + RECT_REACH_V * fh, cols, H, cy >= H / 2.0)):
        lim = W - 1 if axis == "x" else H - 1
        b = _border(_step_profile(frames, axis, max(1, lo), min(lim, hi), span))
        if b is None:
            reach = lo <= 1 if edge == 0 else hi >= lim
            if not (toward_edge and reach):
                return None
            b = edge
        sides[name] = int(b)
    x0, y0, x1, y1 = sides["x0"], sides["y0"], sides["x1"], sides["y1"]
    inset = 2                                   # off the border line itself
    x0, y0 = (x0 + inset if x0 else 0), (y0 + inset if y0 else 0)
    x1, y1 = (x1 - inset if x1 < W else W), (y1 - inset if y1 < H else H)
    return [x0 + x0 % 2, y0 + y0 % 2, x1 - x1 % 2, y1 - y1 % 2]


def snap_to_rect(box, rect, face, src_w, src_h):
    """The cam box with its sides toward the frame centre (right / bottom for a top-left cam)
    moved to the overlay rect's borders: there a grown or head-and-shoulders box runs into the
    game (2500's calm box reached 80 px past the cam) or stops short of the cam's edge. The sides
    toward the frame edges keep the box's (the rect can take a static room edge for a border
    there: 3172's would cut the streamer's hair)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    if face[0] < src_w / 2.0:
        x1 = int(rect[2])
    else:
        x0 = int(rect[0])
    if face[1] < src_h / 2.0:
        y1 = int(rect[3])
    else:
        y0 = int(rect[1])
    return [x0, y0, x1, y1] if x1 - x0 >= 20 and y1 - y0 >= 20 else [int(v) for v in box]


# ---------------------------------------------------------------- action

def _blur(a, sigma):
    try:
        import cv2
        return cv2.GaussianBlur(a, (0, 0), sigma)
    except ImportError:            # numpy-only box blur (tests, no cv2)
        import numpy as np
        r = max(1, int(round(sigma * 1.5)))
        k = np.ones(2 * r + 1, np.float32) / (2 * r + 1)
        a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
        return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a)


def motion_samples(frames, src_w, src_h, exclude=(), noise=NOISE, cut_frac=CUT_FRAC,
                   min_active=MIN_ACTIVE_FRAC, sigma=ENERGY_SIGMA):
    """[{"t", "cx", "cy", "e", "changed", "cut"}] for each consecutive pair of `frames` ([(t, gray
    or BGR)] at any width; t = the later frame's). cx / cy: the energy-weighted centroid (src px)
    of the frame difference with the `exclude` boxes (src px: the cam, HUD plates, overlays)
    masked out, None when the sample has no action (fewer than `min_active` of the pixels
    changed) or is a scene cut (`cut_frac` or more changed). e: mean difference per pixel."""
    import numpy as np
    out, prev, mask = [], None, None
    for t, g in frames:
        g = np.asarray(g, np.float32)
        if g.ndim == 3:
            g = g[..., 0] * 0.114 + g[..., 1] * 0.587 + g[..., 2] * 0.299
        if mask is None:
            h, w = g.shape
            kx, ky = w / float(src_w), h / float(src_h)
            mask = np.ones((h, w), np.float32)
            for x0, y0, x1, y1 in exclude or ():
                mask[max(0, int(y0 * ky)):max(0, min(h, int(math.ceil(y1 * ky)))),
                     max(0, int(x0 * kx)):max(0, min(w, int(math.ceil(x1 * kx))))] = 0.0
            area = max(1.0, float(mask.sum()))
            xs = (np.arange(w, dtype=np.float32) + 0.5) / kx
            ys = (np.arange(h, dtype=np.float32) + 0.5) / ky
        if prev is not None:
            d = np.abs(g - prev) * mask
            d[d < noise] = 0.0
            changed = float((d > 0).sum()) / area
            rec = {"t": round(float(t), 3), "cx": None, "cy": None, "e": 0.0,
                   "changed": round(changed, 4), "cut": changed >= cut_frac}
            if not rec["cut"] and changed >= min_active:
                e = _blur(d, sigma) * mask
                w2 = e * e
                s = float(w2.sum())
                if s > 0:
                    rec["cx"] = round(float((w2.sum(axis=0) * xs).sum() / s), 1)
                    rec["cy"] = round(float((w2.sum(axis=1) * ys).sum() / s), 1)
                    rec["e"] = round(float(d.sum()) / area, 3)
            out.append(rec)
        prev = g
    return out


def peak_sample(samples):
    """The sample with the most motion energy (the event's peak), or None."""
    live = [s for s in samples if s.get("cx") is not None]
    return max(live, key=lambda s: s["e"]) if live else None


# ---------------------------------------------------------------- path

def _ema_pass(targets, x0, lo, hi, alpha, dead, vstep):
    xs, x = [], x0
    for tgt in targets:
        if tgt is not None:
            err = tgt - x
            if abs(err) > dead:
                step = alpha * (err - math.copysign(dead, err))
                x = x + clamp(step, -vstep, vstep)
        x = clamp(x, lo, hi)
        xs.append(x)
    return xs


def smooth_path(targets, lo, hi, dt=1.0 / TRACK_FPS, alpha=EMA_ALPHA, dead_zone=0.0,
                max_speed=float("inf")):
    """The band's x at each sample from the per-sample `targets` (desired x, None = hold): an EMA
    with a soft dead zone (a target within `dead_zone` px moves nothing; past it the path moves
    toward the zone's edge) and a max pan speed (px / s), run forward and backward and averaged
    (no lag either way), clamped to [lo, hi]. Averaging two speed-limited paths keeps the limit."""
    lo, hi = float(lo), float(max(lo, hi))
    known = [t for t in targets if t is not None]
    if not targets:
        return []
    if not known:
        return [clamp((lo + hi) / 2.0, lo, hi)] * len(targets)
    vstep = max_speed * dt
    fwd = _ema_pass(targets, clamp(known[0], lo, hi), lo, hi, alpha, dead_zone, vstep)
    bwd = _ema_pass(targets[::-1], clamp(known[-1], lo, hi), lo, hi, alpha, dead_zone, vstep)[::-1]
    return [clamp((f + b) / 2.0, lo, hi) for f, b in zip(fwd, bwd)]


def _rdp(pts, tol):
    """Ramer-Douglas-Peucker on (t, x) with the x distance to the chord."""
    if len(pts) < 3:
        return list(pts)
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        (t0, x0), (t1, x1) = pts[i], pts[j]
        best, bk = -1.0, None
        for k in range(i + 1, j):
            t, x = pts[k]
            xi = x0 + (x1 - x0) * (t - t0) / (t1 - t0) if t1 > t0 else x0
            dist = abs(x - xi)
            if dist > best:
                best, bk = dist, k
        if bk is not None and best > tol:
            keep[bk] = True
            stack += [(i, bk), (bk, j)]
    return [p for p, k in zip(pts, keep) if k]


def knots(ts, xs, tol, max_knots=MAX_KNOTS):
    """[[t, x]] (x even) of a piecewise-linear path within `tol` px of (ts, xs), at most
    `max_knots` (the tolerance grows until it fits); one knot for a path that never moves."""
    pts = [(float(t), float(x)) for t, x in zip(ts, xs)]
    if not pts:
        return []
    if max(x for _t, x in pts) - min(x for _t, x in pts) <= max(1.0, tol):
        return [[round(pts[0][0], 3), even(sum(x for _t, x in pts) / len(pts))]]
    tol = max(0.5, float(tol))
    while True:
        ks = _rdp(pts, tol)
        if len(ks) <= max_knots:
            break
        tol *= 1.5
    out = []
    for t, x in ks:
        if out and out[-1][1] == even(x) and len(out) >= 2 and out[-2][1] == even(x):
            out[-1] = [round(t, 3), even(x)]       # a hold: one segment, not two
        else:
            out.append([round(t, 3), even(x)])
    return out


def x_at(ks, t):
    """The knot path's x at clip time t (held before the first knot and after the last)."""
    if not ks:
        return 0
    if t <= ks[0][0]:
        return ks[0][1]
    for (t0, x0), (t1, x1) in zip(ks, ks[1:]):
        if t < t1:
            return x0 + (x1 - x0) * (t - t0) / (t1 - t0) if t1 > t0 else x1
    return ks[-1][1]


def _num(v):
    s = f"{float(v):.3f}".rstrip("0").rstrip(".")
    return s or "0"


def pan_expr(ks):
    """ffmpeg expression (clip time t) of the knot path: linear between knots, held outside."""
    if not ks:
        return "0"
    expr = str(int(ks[-1][1]))
    for (t0, x0), (t1, x1) in reversed(list(zip(ks, ks[1:]))):
        x0, x1 = int(x0), int(x1)
        seg = str(x0) if x0 == x1 or t1 <= t0 else \
            f"({x0}+({x1 - x0})*(t-{_num(t0)})/{_num(t1 - t0)})"
        expr = f"if(lt(t,{_num(t1)}),{seg},{expr})"
    if len(ks) > 1:
        expr = f"if(lt(t,{_num(ks[0][0])}),{int(ks[0][1])},{expr})"
    return expr


PIN_S = 0.8                # s each side of the event's peak where the path is pinned to it


def plan_path(samples, gw, lo, hi, src_w, dur, axis="cx", ch=None, pin=None):
    """(knots, xs): the band's smoothed x path (or y with axis="cy" and the crop height `ch`) from
    the motion samples: each sample's target puts the action centre mid-band; samples with no
    action hold. `pin` = (t, centre): the samples within PIN_S of t target that centre instead
    (the event's peak, when the smooth path left it out). Clamped to [lo, hi]."""
    size = float(ch if axis == "cy" else gw)
    targets = [None if s.get(axis) is None else float(s[axis]) - size / 2.0 for s in samples]
    if hi - lo <= DEAD_ZONE * src_w:          # too little room to pan: hold one x
        known = sorted(t for t in targets if t is not None)
        x = even(clamp(known[len(known) // 2] if known else (lo + hi) / 2.0, lo, hi))
        x = int(clamp(x, lo, hi))
        return [[0.0, x]], [x] * len(samples)
    ts = [float(s["t"]) for s in samples]

    def path(tgts):
        xs = smooth_path(tgts, lo, hi, dt=1.0 / TRACK_FPS, alpha=EMA_ALPHA,
                         dead_zone=DEAD_ZONE * src_w, max_speed=MAX_SPEED * src_w)
        if not xs:
            return [[0.0, even((lo + hi) / 2.0)]], []
        ks = knots([0.0] + ts + [float(dur)], [xs[0]] + xs + [xs[-1]], KNOT_TOL * src_w)
        return [[t, int(clamp(x, lo, hi))] for t, x in ks], xs

    ks, xs = path(targets)
    if pin is None:
        return ks, xs
    # the pin holds the peak's centre for PIN_S each side, longer when the max pan speed needs
    # the time to get there (the event's subject is in the band at its peak)
    t0, c = float(pin[0]), float(pin[1])
    for win in (PIN_S, 1.6, 2.8, 4.5):
        pinned = [c - size / 2.0 if abs(t - t0) <= win else tg for t, tg in zip(ts, targets)]
        pks, pxs = path(pinned)
        x = x_at(pks, t0)
        if x <= c < x + size:
            return pks, pxs
    return ks, xs


def inside_share(samples, ks, gw, axis="cx"):
    """Share of the samples with action whose centroid lies inside the band [x(t), x(t) + gw)."""
    live = [s for s in samples if s.get(axis) is not None]
    if not live:
        return None
    hit = sum(1 for s in live if x_at(ks, s["t"]) <= s[axis] < x_at(ks, s["t"]) + gw)
    return round(hit / float(len(live)), 3)
