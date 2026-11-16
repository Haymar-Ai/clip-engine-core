#!/usr/bin/env python3
"""Clean speaker crop (repair lane): re-crop a rejected clip to the speaker so the foreign
UI, watermark and black band fall outside the frame.

The track crop (reframe) is a full-height 9:16 window that follows the face. On a broadcast
layout it shows whatever sits beside the speaker: traderTV's tickers and SUBSCRIBE panel, a
podcast's topic list, the neighbouring windows of a boxed multi-cam (10-20 clips 100, 125, 133,
135, 137), a poker HUD tile cut at the edge (127). `clean_crop` replaces it with the largest 9:16
window, at one size and one height for the whole clip, that for every segment of the plan

  - keeps the segment's framed faces whole, with head room;
  - stays inside the speaker's own window when the source boxes it (`window_boxes`: persistent
    straight borders that close around the face);
  - leaves every `exclude` region (banners, docked panels, dead bands, watermarks, HUD tiles)
    fully outside;
  - slices no `no_slice` region (any other persistent text line: whole or not at all).

It is fit full-bleed, so its height decides the zoom. A window under MIN_CANVAS_W of the 1080 px
canvas at the track's full-height scale (h < 2/3 of the source height on any source) is not a
clean crop: the clip is not fixable by a re-crop (`CleanCropError`).
"""
import math

OUT_W, OUT_H = 1080, 1920
MIN_CANVAS_W = 720           # the brief: no clean crop at least 720 px wide -> not fixable
FACE_PAD_X = 0.01            # of src_w, faces stay this far inside the crop
HEAD_UP, CHIN_DOWN = 0.2, 0.1    # face box (forehead-chin) grown this much (x face height) must
                                 # be inside the crop; more head room is preferred, not required
                                 # (a boxed speaker's head can sit right under the box's top)
SLICE_TOL = 3                # src px a region may poke over a crop edge without counting as sliced
MARGIN = 4                   # src px between a crop edge and an excluded region
H_STEP = 4
# window_boxes
WIN_FPS = 2.0
WIN_SCAN_W = 640
EDGE_PERSIST = 0.6           # a pixel is a border pixel when it is an edge in >= 60 % of frames
LINE_FRAC = 0.85             # a border runs along >= 85 % of the box side
THIN_OFFSETS = (4, 5, 6)     # ... and is a thin line: on one side, 4-6 scan px away, the
THIN_MAX = 0.55              # persistence drops to <= 55 % (a static patterned shirt is a thick
                             # band of edges: 10-20 clip 135's sweater read 0.85-0.93 over 7 rows)
MAX_LINE_PX = 16             # ... at most this many scan px thick: two boxes' borders side by
                             # side run 11 px (10-20 clip 135); the crisp edge of a textured
                             # block reads as a quiet side too, but its strong run goes on
MIN_BORDERS = 2              # a window has at least two real borders (the rest may be frame edges)


class CleanCropError(Exception):
    """No 9:16 window keeps the speaker, avoids the foreign regions and is big enough."""


def _even(v):
    v = int(v)
    return v - v % 2


def min_crop_h(src_h, min_canvas_w=MIN_CANVAS_W):
    """Smallest crop height whose 9:16 window spans `min_canvas_w` of the canvas at the full-height
    track's scale: a full-height crop fills 1080 px, so h / src_h * 1080 >= min_canvas_w."""
    return int(math.ceil(src_h * float(min_canvas_w) / OUT_W))


def canvas_width(h, src_h):
    """Canvas px the crop's width would span at the full-height track's scale."""
    return round(float(h) / float(src_h) * OUT_W, 1)


def regions_from_scan(sc, strict=True, extra=()):
    """(exclude, no_slice) boxes [x0, y0, x1, y1] in src px from a banners.scan result.
    strict (the clip was rejected for third-party UI / watermark / band): banners and docked
    panels must be outside the crop; otherwise they only must not be sliced. Dead bands and
    `extra` (watermark / HUD boxes) are always outside."""
    from clipengine.render import banners
    sw, sh = sc["src_w"], sc["src_h"]
    exclude, no_slice = [list(map(int, b)) for b in extra or ()], []
    for g in sc.get("graphics") or []:
        if g.get("banner"):
            (exclude if strict else no_slice).append(banners.plate(g["box"], sw, sh))
        else:
            no_slice.append(list(map(int, g["box"])))
    for p in sc.get("panels") or []:
        (exclude if strict else no_slice).append(list(map(int, p)))
    db = sc.get("dead_bands") or {}
    for edge, depth in db.items():
        d = int(depth or 0)
        if d <= 0:
            continue
        exclude.append({"top": [0, 0, sw, d], "bottom": [0, sh - d, sw, sh],
                        "left": [0, 0, d, sh], "right": [sw - d, 0, sw, sh]}[edge])
    return exclude, no_slice


def regions_from_text(items, src_w, src_h, strict=True):
    """(exclude, no_slice) from OCR'd text items (quality_gate.text_items: the burned-in text the
    gate's crop-slice check reads; 10-20 clip 135's headline sat on a box border where the line
    detector missed it, and the gate failed the crop for slicing it). Strict: a banner-wide line
    (>= banners.BANNER_MIN_W_FRAC of the width, >= 4x as wide as tall) stays outside; every other
    item is whole or out."""
    from clipengine.render import banners
    exclude, no_slice = [], []
    for it in items or []:
        x0, y0, x1, y1 = (int(v) for v in it["box"])
        w, h = x1 - x0, max(1, y1 - y0)
        wide = w >= banners.BANNER_MIN_W_FRAC * src_w and w >= banners.BANNER_MIN_ASPECT * h
        (exclude if strict and wide else no_slice).append([x0, y0, x1, y1])
    return exclude, no_slice


def segment_faces(segs, faces, src_w, src_h, cw):
    """Per segment, the face boxes (src px, grown to hair and chin) its track crop frames whole:
    faces sampled in its time span, inside its old full-height crop columns."""
    out = []
    for i, s in enumerate(segs):
        t0 = s["t0"] if i else float("-inf")
        t1 = s["t1"] if i < len(segs) - 1 else float("inf")
        boxes = []
        for t, fs in faces:
            if not t0 <= t < t1:
                continue
            for f in fs:
                fx0, fx1 = (f["cx"] - f["w"] / 2) * src_w, (f["cx"] + f["w"] / 2) * src_w
                if fx0 >= s["x"] and fx1 <= s["x"] + cw:
                    fh = f["h"] * src_h
                    boxes.append((fx0, max(0.0, f["cy"] * src_h - (0.5 + HEAD_UP) * fh),
                                  fx1, min(float(src_h), f["cy"] * src_h + (0.5 + CHIN_DOWN) * fh)))
        out.append(boxes)
    return out


def _valid_x(xs, w, y, h, faces, exclude, no_slice, win, src_w):
    """Boolean mask over candidate x (numpy array) for one segment at crop rows [y, y + h)."""
    import numpy as np
    ok = np.ones(len(xs), bool)
    bx0, by0, bx1, by1 = win
    if y < by0 - 0.5 or y + h > by1 + 0.5:
        return np.zeros(len(xs), bool)
    ok &= (xs >= bx0 - 0.5) & (xs + w <= bx1 + 0.5)
    pad = FACE_PAD_X * src_w
    for fx0, fy0, fx1, fy1 in faces:
        if fy0 < y - 0.5 or fy1 > y + h + 0.5:
            # a face that does not fit the rows cannot be framed at this y at all
            return np.zeros(len(xs), bool)
        ok &= (xs <= fx0 - pad) & (xs + w >= fx1 + pad)
    for x0, y0, x1, y1 in exclude:
        if y1 <= y - MARGIN or y0 >= y + h + MARGIN:
            continue                                   # rows MARGIN px clear of the crop
        ok &= (xs + w <= x0 - MARGIN) | (xs >= x1 + MARGIN)
    for x0, y0, x1, y1 in no_slice:
        rows_in = y0 >= y - SLICE_TOL and y1 <= y + h + SLICE_TOL
        if y1 <= y + SLICE_TOL or y0 >= y + h - SLICE_TOL:
            continue
        outside = (xs + w <= x0 + SLICE_TOL) | (xs >= x1 - SLICE_TOL)
        if rows_in:
            inside = (xs <= x0 + SLICE_TOL) & (xs + w >= x1 - SLICE_TOL)
            ok &= outside | inside
        else:                                          # cut by the top or bottom edge
            ok &= outside
    return ok


SPLIT_STEP_S = 3.0           # = reframe.DRIFT_STEP; a segment no single clean window holds is
                             # re-tried in pieces this long (a speaker leaning across one shot)
SPLIT_MIN_S = 4.5            # ... when it is at least this long


def split_segments(segs, step=SPLIT_STEP_S, min_s=SPLIT_MIN_S):
    """The plan's segments with each one of >= min_s cut into pieces of `step` s (the last piece
    takes the rest); the first piece keeps the segment's own keys (its cut flag)."""
    out = []
    for s in segs:
        t0, t1 = float(s["t0"]), float(s.get("t1", s["t0"]))
        if t1 - t0 < min_s:
            out.append(dict(s))
            continue
        a = t0
        while a < t1 - 1e-6:
            b = t1 if t1 - (a + step) < step / 2.0 else a + step
            piece = dict(s, t0=round(a, 3), t1=round(b, 3))
            if a > t0:
                piece.pop("cut", None)
            out.append(piece)
            a = b
    return out


def clean_crop(plan, sc, exclude, no_slice, windows=None, min_canvas_w=MIN_CANVAS_W, soft=None):
    """The clean window for a track-like plan (see module doc). `windows`: per segment
    [x0, y0, x1, y1] the crop must stay inside (None / missing = the whole frame). Returns
    {"w", "h", "y", "x": [per segment], "canvas_w"}; raises CleanCropError.

    when no window works on the plan's segments, the long ones are tried once more in
    SPLIT_STEP_S pieces (10-18 clip 1357: one 45 s segment while the speaker leaned 362 px
    across a 404 px crop, his shirt print cut on either side); the result then carries the new
    "segments" (pieces whose x match their neighbour's merged) that apply writes on the plan."""
    # `soft`: boxes kept whole or out where a segment's clean x allow it (an x
    # that leaves them whole is preferred), never required
    try:
        return _clean_crop(plan, plan.get("segments") or [], sc, exclude, no_slice, windows,
                           min_canvas_w, soft=soft)
    except CleanCropError:
        segs = plan.get("segments") or []
        if not segs:
            raise
        pieces, wins = [], []
        for i, sg in enumerate(segs):           # a boxed speaker's piece keeps its segment's box
            for piece in split_segments([sg]):
                pieces.append(piece)
                wins.append((windows or [None] * len(segs))[i] if windows else None)
        if len(pieces) == len(segs):
            raise
        try:
            crop = _clean_crop(plan, pieces, sc, exclude, no_slice, wins if windows else None,
                               min_canvas_w, sticky=True, soft=soft)
        except CleanCropError:
            pass
        else:
            merged = []
            for p, x in zip(pieces, crop["x"]):
                if merged and int(x) == merged[-1]["x"]:
                    merged[-1]["t1"] = p["t1"]
                    continue
                merged.append(dict(p, x=int(x)))
            crop["segments"] = merged
            crop["x"] = [m["x"] for m in merged]
            return crop
        raise


def _clean_crop(plan, segs, sc, exclude, no_slice, windows, min_canvas_w, sticky=False, soft=None):
    """clean_crop on `segs`; `sticky`: a segment keeps the one before it's x when that x is
    clean for it too (split pieces of one shot: no jump without a reason)."""
    import numpy as np
    from clipengine.render import reframe
    sw, sh = int(plan["src_w"]), int(plan["src_h"])
    if not segs:
        raise CleanCropError("plan has no track segments to re-crop")
    cw_full = reframe.crop_width(sw, sh)
    faces = segment_faces(segs, sc.get("faces") or [], sw, sh, cw_full)
    wins = [list((windows or [None] * len(segs))[i] or [0, 0, sw, sh]) for i in range(len(segs))]
    h_min = min_crop_h(sh, min_canvas_w)
    h_top = _even(min(sh, min(b[3] - b[1] for b in wins)))
    tried = []
    for h in range(h_top, h_min - 1, -H_STEP):
        h = _even(h)
        w = min(_even(round(h * 9 / 16)), sw - sw % 2)
        xs = np.arange(0, sw - w + 1, 2)
        ys = {0, sh - h}
        for b in wins:
            ys |= {b[1], b[3] - h}
        for x0, y0, x1, y1 in exclude:
            ys |= {y1 + MARGIN, y0 - MARGIN - h}
        for x0, y0, x1, y1 in no_slice:
            ys |= {y1 + SLICE_TOL, y0 - SLICE_TOL - h, y0 - SLICE_TOL}
        for fb in faces:
            for fx0, fy0, fx1, fy1 in fb:
                ys |= {fy0 - 0.15 * (fy1 - fy0), fy0}
        # face rows sit in the upper part of the crop; candidates tried closest to that first
        fy = [b[1] for fb in faces for b in fb]
        want = (min(fy) - 0.12 * h) if fy else (sh - h) / 2
        cands = sorted({_even(min(max(y, 0), sh - h)) for y in ys}, key=lambda y: abs(y - want))
        for y in cands:
            chosen = []
            for i, s in enumerate(segs):
                ok = _valid_x(xs, w, y, h, faces[i], exclude, no_slice, wins[i], sw)
                if not ok.any():
                    chosen = None
                    break
                if faces[i]:
                    cx = (min(b[0] for b in faces[i]) + max(b[2] for b in faces[i])) / 2
                else:
                    cx = s["x"] + cw_full / 2
                good = xs[ok]
                if soft:
                    whole = good[_valid_x(good, w, y, h, [], [], soft, wins[i], sw)]
                    good = whole if len(whole) else good
                if sticky and chosen and chosen[-1] in set(good.tolist()):
                    chosen.append(chosen[-1])
                    continue
                chosen.append(int(good[np.argmin(np.abs(good + w / 2 - cx))]))
            if chosen is not None:
                return {"w": w, "h": h, "y": y, "x": chosen,
                        "canvas_w": canvas_width(h, sh), "min_canvas_w": min_canvas_w}
        tried.append(h)
    raise CleanCropError(
        f"no clean 9:16 crop of at least {min_canvas_w} px canvas width (h >= {h_min} of "
        f"{sh} src px) keeps the speaker clear of {len(exclude)} foreign region(s) without "
        f"slicing {len(no_slice)} graphic(s)"
        + (f"; speaker window {min(b[3] - b[1] for b in wins)} px tall"
           if any(b != [0, 0, sw, sh] for b in wins) else ""))


BOX_MAX_ASPECT = 1.1         # clip-quality 10-21 stage 2 ladder (c): a fit-to-width box at most
                             # this wide for its height: 1080 px wide it is >= 982 px tall, so the
                             # blurred fill stays <= 49 % (quality_gate.VQ_MAX_BLUR_FILL is 50 %)
BOX_MIN_H_FRAC = 0.55        # = banners.BOX_MIN_H_FRAC: a box keeps at least this much height
BOX_MIN_W = 360              # ... and this many src px of width (fit to 1080 px: <= 3x upscale)


def clean_box(plan, sc, exclude, no_slice, windows=None, max_aspect=BOX_MAX_ASPECT,
              min_h_frac=BOX_MIN_H_FRAC):
    """Ladder step (c) of the stage-2 crop ladder: the largest box, at most `max_aspect` wide for
    its height, that for every segment keeps the framed faces whole, leaves `exclude` outside and
    slices no `no_slice` region (the same rules as clean_crop). It is fit to the canvas width
    over a blurred fill of the same frame (crop_box fit "box"), not full-bleed, so it can be
    wider than 9:16 and hold a whole graphic beside the speaker. Returns clean_crop's dict
    plus "fit": "box"; raises CleanCropError."""
    import numpy as np
    from clipengine.render import reframe
    sw, sh = int(plan["src_w"]), int(plan["src_h"])
    segs = plan.get("segments") or []
    if not segs:
        raise CleanCropError("plan has no track segments to box")
    faces = segment_faces(segs, sc.get("faces") or [], sw, sh, reframe.crop_width(sw, sh))
    wins = [list((windows or [None] * len(segs))[i] or [0, 0, sw, sh]) for i in range(len(segs))]
    h_top = _even(min(sh, min(b[3] - b[1] for b in wins)))
    h_min = int(math.ceil(min_h_frac * sh))
    for h in range(h_top, h_min - 1, -H_STEP):
        h = _even(h)
        # 4 px under the narrowest window: an even x must fit inside it (10-21 clip 331's
        # 386 px tile left none at its full width)
        w = min(_even(round(h * max_aspect)), sw - sw % 2,
                _even(min(b[2] - b[0] - (4 if b != [0, 0, sw, sh] else 0) for b in wins)))
        if w < max(round(h * 9 / 16), BOX_MIN_W):
            continue
        xs = np.arange(0, sw - w + 1, 2)
        ys = {0, sh - h}
        for b in wins:
            ys |= {b[1], b[3] - h}
        for x0, y0, x1, y1 in exclude:
            ys |= {y1 + MARGIN, y0 - MARGIN - h}
        for x0, y0, x1, y1 in no_slice:
            ys |= {y1 + SLICE_TOL, y0 - SLICE_TOL - h, y0 - SLICE_TOL}
        fy = [b[1] for fb in faces for b in fb]
        want = (min(fy) - 0.2 * h) if fy else (sh - h) / 2
        cands = sorted({_even(min(max(y, 0), sh - h)) for y in ys}, key=lambda y: abs(y - want))
        for y in cands:
            chosen = []
            for i, s in enumerate(segs):
                ok = _valid_x(xs, w, y, h, faces[i], exclude, no_slice, wins[i], sw)
                if not ok.any():
                    chosen = None
                    break
                cx = ((min(b[0] for b in faces[i]) + max(b[2] for b in faces[i])) / 2
                      if faces[i] else s["x"] + reframe.crop_width(sw, sh) / 2)
                good = xs[ok]
                chosen.append(int(good[np.argmin(np.abs(good + w / 2 - cx))]))
            if chosen is not None:
                return {"w": w, "h": h, "y": y, "x": chosen, "fit": "box",
                        "canvas_w": OUT_W, "min_canvas_w": None,
                        "blur_fill": round(1.0 - min(1.0, OUT_W * h / w / OUT_H), 3)}
    raise CleanCropError(
        f"no box (<= {max_aspect:g}:1, h >= {min_h_frac:.0%} of {sh} src px) keeps the speaker "
        f"clear of {len(exclude)} foreign region(s) without slicing {len(no_slice)} graphic(s)")


def apply(plan, crop):
    """Write a clean_crop result onto the plan: per-segment x (the result's own segments when it
    split the plan's) and a full-bleed crop_box."""
    if crop.get("segments"):
        plan["segments"] = [dict(s) for s in crop["segments"]]
    for s, x in zip(plan["segments"], crop["x"]):
        s["x"] = int(x)
    plan["crop_box"] = {"w": int(crop["w"]), "h": int(crop["h"]), "y": int(crop["y"])}
    if crop.get("fit"):
        plan["crop_box"]["fit"] = crop["fit"]
    plan["clean_crop"] = {k: crop[k] for k in ("w", "h", "y", "canvas_w", "min_canvas_w", "fit")
                          if k in crop}
    return plan


# ---------------------------------------------------------------- boxed windows

def edge_persistence(frames_gray):
    """Fraction of frames in which each pixel is an edge (Canny, dilated 1 px for jitter)."""
    import cv2
    import numpy as np
    acc = None
    k = np.ones((3, 3), np.uint8)
    for g in frames_gray:
        e = cv2.dilate((cv2.Canny(g, 60, 160) > 0).astype(np.uint8), k)
        acc = e.astype(np.float32) if acc is None else acc + e
    return None if acc is None else acc / len(frames_gray)


def _col_strength(P, c, r0, r1):
    c0, c1 = max(0, c - 1), min(P.shape[1], c + 2)
    r0, r1 = max(0, r0), min(P.shape[0], r1)
    if r1 <= r0 or c1 <= c0:
        return 0.0
    return float((P[r0:r1, c0:c1].max(axis=1) >= EDGE_PERSIST).mean())


def _row_strength(P, r, c0, c1):
    r0, r1 = max(0, r - 1), min(P.shape[0], r + 2)
    c0, c1 = max(0, c0), min(P.shape[1], c1)
    if r1 <= r0 or c1 <= c0:
        return 0.0
    return float((P[r0:r1, c0:c1].max(axis=0) >= EDGE_PERSIST).mean())


def _thin(strength, i):
    """True when line i is a thin line: the mean strength 4-6 px to one side is <= THIN_MAX and
    the run of strong lines through i is at most MAX_LINE_PX long."""
    sides = [sum(strength(i + sgn * d) for d in THIN_OFFSETS) / len(THIN_OFFSETS)
             for sgn in (-1, 1)]
    if min(sides) > THIN_MAX:
        return False
    run = 1
    for sgn in (-1, 1):
        k = i + sgn
        while run <= MAX_LINE_PX and strength(k) >= LINE_FRAC:
            run += 1
            k += sgn
    return run <= MAX_LINE_PX


def find_window(P, face, min_gap=4):
    """The boxed window around `face` (x0, y0, x1, y1 in P's px) from a persistence map:
    nearest persistent vertical lines left / right of the face and horizontal lines above / below
    it, each running along >= LINE_FRAC of the box side once the box is closed. None when fewer
    than MIN_BORDERS real borders close it (an open scene, a door frame beside the face)."""
    h, w = P.shape
    fx0, fy0, fx1, fy1 = (int(round(v)) for v in face)
    fw, fh = max(1, fx1 - fx0), max(1, fy1 - fy0)
    r0, r1 = fy0 - fh // 2, fy1 + fh // 2

    def col(c):
        return _col_strength(P, c, r0, r1)

    def is_col(c):
        return col(c) >= LINE_FRAC and _thin(col, c)

    left = next((c for c in range(fx0 - min_gap, 0, -1) if is_col(c)), None)
    right = next((c for c in range(fx1 + min_gap, w - 1) if is_col(c)), None)
    c0, c1 = (left or 0), (right or w - 1)

    def row(r):
        return _row_strength(P, r, c0, c1)

    def is_row(r):
        return row(r) >= LINE_FRAC and _thin(row, r)

    top = next((r for r in range(fy0 - min_gap, 0, -1) if is_row(r)), None)
    bot = next((r for r in range(fy1 + min_gap, h - 1) if is_row(r)), None)
    box = [left if left is not None else 0, top if top is not None else 0,
           right if right is not None else w, bot if bot is not None else h]
    real = [s for s, v in (("left", left), ("right", right), ("top", top), ("bottom", bot))
            if v is not None]
    if len(real) < MIN_BORDERS:
        return None
    # closure: every real border runs along the closed box's side
    for side in real:
        if side in ("left", "right"):
            c = box[0] if side == "left" else box[2]
            if _col_strength(P, c, box[1] + 2, box[3] - 2) < LINE_FRAC:
                return None
        else:
            r = box[1] if side == "top" else box[3]
            if _row_strength(P, r, box[0] + 2, box[2] - 2) < LINE_FRAC:
                return None
    return box, real


def window_boxes(src, start, dur, plan, faces, inset=3):
    """Per plan segment, the speaker's boxed window [x0, y0, x1, y1] (src px, `inset` px inside
    its border) or None when the segment's speaker is not boxed."""
    import cv2
    from clipengine.render import reframe
    sw, sh = int(plan["src_w"]), int(plan["src_h"])
    segs = plan.get("segments") or []
    frames = [(t, cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
              for t, fr in reframe.iter_frames(str(src), start, dur, fps=WIN_FPS, width=WIN_SCAN_W,
                                               src_size=(sw, sh))]
    if not frames:
        return [None] * len(segs)
    k = sw / frames[0][1].shape[1]
    fb = segment_faces(segs, faces, sw, sh, reframe.crop_width(sw, sh))
    out = []
    for i, s in enumerate(segs):
        t0 = s["t0"] if i else float("-inf")
        t1 = s["t1"] if i < len(segs) - 1 else float("inf")
        fr = [g for t, g in frames if t0 <= t < t1] or [g for _t, g in frames]
        if not fb[i] or len(fr) < 2:
            out.append(None)
            continue
        P = edge_persistence(fr)
        x0 = min(b[0] for b in fb[i]) / k
        y0 = min(b[1] for b in fb[i]) / k
        x1 = max(b[2] for b in fb[i]) / k
        y1 = max(b[3] for b in fb[i]) / k
        got = find_window(P, (x0, y0, x1, y1))
        if got is None:
            out.append(None)
            continue
        box, _real = got
        out.append([int(round(box[0] * k)) + inset, int(round(box[1] * k)) + inset,
                    int(round(box[2] * k)) - inset, int(round(box[3] * k)) - inset])
    return out
