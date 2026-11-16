#!/usr/bin/env python3
"""Source graphics vs the 9:16 crop.

A track crop is a 9:16 window (404 px of a 1280x720 source) that follows a face. A lower-third
banner ("Ravens beat Falcons 30-23 in OT") is far wider than that, so the crop showed
"...23 in OT" cut off at the left edge. The rule here: a crop either contains a persistent
source graphic whole, leaves it out entirely, or the clip is rejected.

`scan` samples the source window at 1 fps and finds persistent text-line boxes (horizontal
gradient + closing, the burned-in text signature) and faces. `fix_track_plan` then, for a
track-like plan, in order:

  1. shift: move a slicing segment's crop sideways so each banner is fully in or fully out,
     keeping the framed faces inside;
  2. trim: cut the crop's height so the banner rows are outside it (a narrower 9:16 window
     of the remaining rows inside the old one, its side cuts placed where the source is
     calmest), with the cut lines kept off any other text line and the faces inside, as
     long as >= MIN_KEEP_H_FRAC of the source height remains;
  3. box: the same rows cut, but the crop keeps its full 9:16 width and is fit to
     the canvas width over a blurred fill, so the subject keeps the track's scale. Used when
     the narrower trim window would lose a face (10-19 clip 84: a 20 px strip at the top made
     the 384 px trim window too narrow for a close-up head by under a pixel) or keep too few
     rows; needs >= BOX_MIN_H_FRAC of the source height (the gate fails > 50 % blur fill);
  4. reject: none works -> the caller tries a screen-stack reframe, then fails the render.
"""
from clipengine.render import edgebands
from clipengine.render import reframe

SCAN_W = 640
SAMPLE_FPS = 1.0
PERSIST_FRAC = 0.2          # a box in >= 20 % of sampled frames (and >= 2) is a persistent graphic
MATCH_IOU = 0.5             # same row: vertical overlap of the shorter box
SOBEL_T = 80                # |d/dx| (0-255) that counts as a glyph edge
LINE_MIN_W_FRAC = 0.04      # text line: at least this wide ...
LINE_MIN_ASPECT = 1.5       # ... and this elongated
LINE_MIN_H_PX = 6           # scan px
LINE_MAX_H_FRAC = 0.2
BANNER_MIN_W_FRAC = 0.20    # banner: a text line this wide (lower third, ticker, score strip)
BANNER_MIN_ASPECT = 4.0
BANNER_MAX_H_FRAC = 0.15    # a chart or a dense UI panel is taller than any lower third
SLICE_TOL_FRAC = 0.008      # a box must reach this far (of src_w) past a crop edge to be sliced
PAD_X_FRAC = 0.03           # banner plate around the detected text, horizontally ...
PAD_Y_TEXT = 0.5            # ... and this many text heights vertically (4-16 px)
CUT_MARGIN = 8              # src px between a trim line and the graphic it removes
MIN_KEEP_H_FRAC = 0.70      # a trim keeps at least this much of the source height
BOX_MIN_H_FRAC = 0.55       # a boxed crop keeps at least this much (16:9 -> fill <= ~43 %)
FACE_MARGIN_FRAC = 0.01     # framed faces stay this far (of src_w) inside the crop
PANEL_GRID = (32, 18)       # occupancy cells (x, y) for text-panel detection
PANEL_OCC = 0.2             # a cell holding text in >= 20 % of frames is part of a panel
PANEL_MIN_H_FRAC = 0.15     # panel: a text column at least this tall (chat, sidebar) ...
PANEL_MIN_W_FRAC = 0.06     # ... and this wide ...
PANEL_EDGE_FRAC = 0.06      # ... docked to the left or right frame edge (overlays are)


def line_boxes(frame_bgr):
    """Text-line boxes [(x0, y0, x1, y1)] (frame px) of one BGR frame."""
    import cv2
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mag = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    _, bw = cv2.threshold(mag, SOBEL_T, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 30), 3))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < LINE_MIN_W_FRAC * w or ch < LINE_MIN_H_PX or ch > LINE_MAX_H_FRAC * h:
            continue
        if cw < LINE_MIN_ASPECT * ch:
            continue
        out.append((x, y, x + cw, y + ch))
    return out


def _same_row(a, b):
    """Two text boxes on the same row: vertical overlap >= MATCH_IOU of the shorter box and
    overlapping columns. Subtitles and tickers change length frame to frame; IoU would split
    one changing line into many short-lived clusters."""
    oy = min(a[3], b[3]) - max(a[1], b[1])
    return oy >= MATCH_IOU * min(a[3] - a[1], b[3] - b[1]) and min(a[2], b[2]) > max(a[0], b[0])


def persistent_boxes(per_frame, n_frames, persist_frac=PERSIST_FRAC):
    """Cluster per-frame boxes by row; [(union_box, frac)] seen in enough frames."""
    clusters = []      # [union box, frames seen, last frame index]
    for fi, boxes in enumerate(per_frame):
        for b in boxes:
            best = next((c for c in clusters if _same_row(c[0], b)), None)
            if best is None:
                clusters.append([tuple(b), 1, fi])
                continue
            u = best[0]
            best[0] = (min(u[0], b[0]), min(u[1], b[1]), max(u[2], b[2]), max(u[3], b[3]))
            if best[2] != fi:
                best[1] += 1
                best[2] = fi
    need = max(2, persist_frac * n_frames)
    return [(c[0], c[1] / max(1, n_frames)) for c in clusters if c[1] >= need]


def text_panels(per_frame, frame_w, frame_h, n_frames):
    """Boxes (frame px) of edge-docked regions that keep holding text while the lines
    themselves change: a Twitch chat column, a sidebar, a stats stack. Built from per-frame
    line boxes on a PANEL_GRID occupancy map; tall-enough connected regions only."""
    import cv2
    import numpy as np
    gx, gy = PANEL_GRID
    cw, ch = frame_w / gx, frame_h / gy
    occ = np.zeros((gy, gx), float)
    for boxes in per_frame:
        hit = np.zeros((gy, gx), bool)
        for x0, y0, x1, y1 in boxes:
            hit[int(y0 // ch):int(np.ceil(y1 / ch)), int(x0 // cw):int(np.ceil(x1 / cw))] = True
        occ += hit
    mask = (occ / max(1, n_frames) >= PANEL_OCC).astype(np.uint8)
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(mask, connectivity=4)
    out = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i][:4])
        x0, x1 = x * cw, min(frame_w, (x + w) * cw)
        docked = x0 <= PANEL_EDGE_FRAC * frame_w or x1 >= (1 - PANEL_EDGE_FRAC) * frame_w
        if docked and h * ch >= PANEL_MIN_H_FRAC * frame_h and \
                w * cw >= PANEL_MIN_W_FRAC * frame_w:
            out.append((int(x0), int(y * ch), int(x1), int(min(frame_h, (y + h) * ch))))
    return out


def _iter_scan_frames(src, start, dur, src_w, src_h, fps=SAMPLE_FPS, width=SCAN_W):
    yield from reframe.iter_frames(src, start, dur, fps=fps, width=width,
                                   src_size=(src_w, src_h))


def scan(src, start, dur, frames=None, src_size=None):
    """{"src_w", "src_h", "frames", "graphics": [{"box": [x0,y0,x1,y1] src px, "frac",
    "banner"}], "panels": [[x0,y0,x1,y1] src px], "faces": [(t, [face...])]} for the clip
    window. `frames` ([(t, bgr)] at any width) skips the ffmpeg pass (tests)."""
    src_w, src_h = src_size or reframe.probe_size(src)
    if frames is None:
        frames = list(_iter_scan_frames(src, start, dur, src_w, src_h))
    import cv2
    import numpy as np
    per_frame, faces = [], []
    fw = frames[0][1].shape[1] if frames else SCAN_W
    k = src_w / fw
    hgrad = None
    for t, fr in frames:
        per_frame.append(line_boxes(fr))
        faces.append((t, reframe.detect_faces(fr)))
        gy = np.abs(cv2.Sobel(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 0, 1, ksize=3))
        hgrad = gy if hgrad is None else hgrad + gy
    graphics = []
    for (x0, y0, x1, y1), frac in persistent_boxes(per_frame, len(frames)):
        box = [int(round(x0 * k)), int(round(y0 * k)), int(round(x1 * k)), int(round(y1 * k))]
        w, h = box[2] - box[0], box[3] - box[1]
        banner = (w >= BANNER_MIN_W_FRAC * src_w and w >= BANNER_MIN_ASPECT * h
                  and h <= BANNER_MAX_H_FRAC * src_h)
        graphics.append({"box": box, "frac": round(frac, 3), "banner": banner})
    fh = frames[0][1].shape[0] if frames else int(round(src_h / k))
    panels = [[int(round(v * k)) for v in b]
              for b in text_panels(per_frame, fw, fh, len(frames))] if frames else []
    dead = edgebands.persistent_bands([edgebands.flat_edge_depths(edgebands.to_gray(fr))
                                       for _t, fr in frames]) if frames else {}
    return {"src_w": src_w, "src_h": src_h, "frames": len(frames), "graphics": graphics,
            "panels": panels, "faces": faces,
            "dead_bands": {e: int(round(v * k)) for e, v in dead.items()},
            "hgrad": None if hgrad is None else hgrad / len(frames), "scale": k}


def cut_cost(sc, x, ya, yb):
    """How much a crop edge at source column x cuts through: mean |d/dy| (horizontal edges:
    glyph strokes, logos, UI rows) down that column between rows ya..yb, worst of the three
    scan columns around it. A cut through a plain wall or along a vertical panel border
    scores ~0; through a text line or logo it does not."""
    g = sc.get("hgrad")
    if g is None:
        return 0.0
    k = sc.get("scale") or 1.0
    c = int(round(x / k))
    r0, r1 = max(0, int(ya / k)), max(1, int(yb / k))
    cols = [g[r0:r1, cc].mean() for cc in (c - 1, c, c + 1) if 0 <= cc < g.shape[1]]
    return float(max(cols)) if cols else 0.0


OCR_TIMES = (0.15, 0.5, 0.85)   # fractions of the clip OCR'd when a trim narrows the crop
OCR_UPSCALE = 2                 # small UI text reads far better at 2x


def ocr_source_words(src, start, dur, src_size):
    """Word boxes [(x0, y0, x1, y1)] (src px) OCR'd on a few source frames; [] without
    tesseract. Only the trim placement uses these (UI labels the line detector misses)."""
    import cv2
    from clipengine.render import textzones
    if not textzones._ocr_available():
        return []
    sw, sh = src_size
    out = []
    for f in OCR_TIMES:
        fr = next(iter(reframe.iter_frames(src, start + f * dur, 0.5, fps=1, width=sw,
                                           src_size=src_size)), None)
        if fr is None:
            continue
        g = cv2.cvtColor(fr[1], cv2.COLOR_BGR2GRAY)
        big = cv2.resize(g, (sw * OCR_UPSCALE, sh * OCR_UPSCALE), interpolation=cv2.INTER_CUBIC)
        out += [tuple(v // OCR_UPSCALE for v in w[:4]) for w in textzones.ocr_words(big)]
    return out


def _cuts_words(words, x, cw, ya, yb):
    """Words a crop [x, x + cw) x [ya, yb) shows only part of."""
    return sum(1 for x0, y0, x1, y1 in words
               if y1 > ya and y0 < yb and min(x1, x + cw) > max(x0, x)
               and (x0 < x or x1 > x + cw))


def plate(box, src_w, src_h):
    """A banner's detected text box grown to the plate drawn around it."""
    x0, y0, x1, y1 = box
    py = min(16, max(4, int(round(PAD_Y_TEXT * (y1 - y0)))))
    px = int(round(PAD_X_FRAC * src_w))
    return [max(0, x0 - px), max(0, y0 - py), min(src_w, x1 + px), min(src_h, y1 + py)]


def sliced(box, x, cw, src_w):
    """True when the crop columns [x, x + cw) show part, but not all, of `box`."""
    tol = SLICE_TOL_FRAC * src_w
    if min(box[2], x + cw) - max(box[0], x) <= tol:
        return False                         # (practically) not in the crop at all
    return box[0] < x - tol or box[2] > x + cw + tol


def _seg_faces(seg_i, segs, faces, x, cw, src_w, src_h):
    """Face boxes (src px) segment `seg_i` frames whole: sampled in its time span and fully
    inside its original crop columns. A fix must keep exactly these inside."""
    t0 = segs[seg_i]["t0"] if seg_i else float("-inf")
    t1 = segs[seg_i]["t1"] if seg_i < len(segs) - 1 else float("inf")
    m = FACE_MARGIN_FRAC * src_w
    out = []
    for t, fs in faces:
        if not t0 <= t < t1:
            continue
        for f in fs:
            fx0, fx1 = (f["cx"] - f["w"] / 2) * src_w, (f["cx"] + f["w"] / 2) * src_w
            if fx0 >= x + m and fx1 <= x + cw - m:
                out.append((fx0, (f["cy"] - f["h"] / 2) * src_h,
                            fx1, (f["cy"] + f["h"] / 2) * src_h))
    return out


def _faces_inside(face_boxes, x, cw, ya, yb, src_w):
    m = FACE_MARGIN_FRAC * src_w
    return all(fx0 >= x + m and fx1 <= x + cw - m and fy0 >= ya and fy1 <= yb
               for fx0, fy0, fx1, fy1 in face_boxes)


def _shift(x, cw, banners, face_boxes, ya, yb, src_w):
    """Nearest crop x (same width) that slices no banner and keeps the faces; None if none."""
    cands = {x}
    for b in banners:
        cands |= {b[2] + CUT_MARGIN, b[0] - CUT_MARGIN - cw,     # banner fully outside
                  b[0] - CUT_MARGIN, b[2] + CUT_MARGIN - cw}     # banner fully inside
    best = None
    for c in cands:
        c = int(round(min(max(c, 0), src_w - cw)))
        if any(sliced(b, c, cw, src_w) for b in banners):
            continue
        if not _faces_inside(face_boxes, c, cw, ya, yb, src_w):
            continue
        if best is None or abs(c - x) < abs(best - x):
            best = c
    return best


def _off_lines(y, boxes, down):
    """Move a cut row off every text line it would pass through."""
    moved = True
    while moved:
        moved = False
        for b in boxes:
            if b[1] < y < b[3]:
                y = b[3] + CUT_MARGIN if down else b[1] - CUT_MARGIN
                moved = True
    return y


def fix_track_plan(plan, sc, words_fn=None):
    """Make a track-like plan's crop slice no persistent banner. Mutates `plan` (segment x,
    `crop_box` {"w", "h", "y"}) and returns {"action": none|shift|trim|reject, ...}.
    `words_fn() -> [(x0, y0, x1, y1)]` (OCR'd source words) is called only when a trim has
    to place a narrower window; its side cuts then avoid slicing those words."""
    src_w, src_h = sc["src_w"], sc["src_h"]
    segs = plan.get("segments") or []
    cw = reframe.crop_width(src_w, src_h)
    banners = [plate(g["box"], src_w, src_h) for g in sc["graphics"] if g["banner"]]
    lines = [g["box"] for g in sc["graphics"]] + banners
    rep = {"banners": banners, "frames": sc["frames"]}
    hit = sorted({i for i, s in enumerate(segs) for b in banners
                  if sliced(b, s["x"], cw, src_w)})
    if not hit:
        return {**rep, "action": "none"}
    rep["sliced_segments"] = hit
    faces = sc.get("faces") or []
    seg_faces = [_seg_faces(i, segs, faces, s["x"], cw, src_w, src_h)
                 for i, s in enumerate(segs)]
    # 1. shift the slicing segments sideways
    shifted = {i: _shift(segs[i]["x"], cw, banners, seg_faces[i], 0, src_h, src_w) for i in hit}
    if all(v is not None for v in shifted.values()):
        for i, nx in shifted.items():
            segs[i]["x"] = nx
        return {**rep, "action": "shift", "x": {i: segs[i]["x"] for i in hit}}
    # 2. trim the slicing banners' rows out of a shorter 9:16 window
    cut = [b for b in banners if any(sliced(b, segs[i]["x"], cw, src_w) for i in hit)]
    tops = [b[3] + CUT_MARGIN for b in cut if (b[1] + b[3]) / 2 < src_h / 2]
    bots = [b[1] - CUT_MARGIN for b in cut if (b[1] + b[3]) / 2 >= src_h / 2]
    ya = _off_lines(max(tops, default=0), lines, down=True)
    yb = _off_lines(min(bots, default=src_h), lines, down=False)
    ya, yb = ya + ya % 2, yb - yb % 2
    h = yb - ya
    if h < MIN_KEEP_H_FRAC * src_h:
        return _box(plan, rep, segs, seg_faces, banners, cw, ya, yb, src_w, src_h,
                    f"removing the banner rows leaves {h}px of {src_h}px "
                    f"(< {MIN_KEEP_H_FRAC:.0%})")
    cw2 = min(src_w - src_w % 2, int(round(h * 9 / 16)) // 2 * 2)
    visible = [b for b in banners if b[3] > ya and b[1] < yb]
    words = list(words_fn() or []) if words_fn else []
    tol = SLICE_TOL_FRAC * src_w
    new_x = {}
    for i, s in enumerate(segs):
        # the narrower window's two side cuts slice the fewest OCR'd words (it may leave the
        # old window only to do that); then it stays inside the old window, flush with a
        # frame edge the old crop touched (a screen share's own UI labels start there); then
        # its cuts run where the source is calmest (logos the OCR can't read); then it stays
        # nearest the old centre
        fb = _seg_faces(i, segs, faces, s["x"], cw, src_w, src_h)
        lo = max(0, s["x"])
        hi = min(src_w - cw2, s["x"] + cw - cw2)
        centre = s["x"] + cw / 2 - cw2 / 2
        flush = lo if s["x"] <= tol else hi if s["x"] + cw >= src_w - tol else None
        ok = [c for c in sorted(set(range(0, src_w - cw2 + 1, 2)) | {lo, hi})
              if not any(sliced(b, c, cw2, src_w) for b in visible)
              and _faces_inside(fb, c, cw2, ya, yb, src_w)]
        if ok:
            x2 = min(ok, key=lambda c: (_cuts_words(words, c, cw2, ya, yb),
                                        not lo <= c <= hi,
                                        flush is not None and c != flush,
                                        round(cut_cost(sc, c, ya, yb)
                                              + cut_cost(sc, c + cw2, ya, yb), 1),
                                        abs(c - centre)))
        else:
            x2 = _shift(int(round(min(max(centre, 0), src_w - cw2))), cw2, visible, fb,
                        ya, yb, src_w)
        if x2 is None:
            return _box(plan, rep, segs, seg_faces, banners, cw, ya, yb, src_w, src_h,
                        f"segment {i}: no {cw2}x{h} crop keeps its faces without slicing "
                        f"a banner")
        new_x[i] = x2
    for i, x2 in new_x.items():
        segs[i]["x"] = x2
    plan["crop_box"] = {"w": cw2, "h": h, "y": ya}
    return {**rep, "action": "trim", "crop_box": plan["crop_box"]}


def _box(plan, rep, segs, seg_faces, banners, cw, ya, yb, src_w, src_h, why):
    """Step 3: rows ya..yb at the full crop width cw, each segment at the nearest x that keeps
    its faces whole and slices no banner left in those rows. Mutates `plan` only on success;
    otherwise the reject report carries `why` (the trim's reason) and this step's."""
    h = yb - ya
    if h < BOX_MIN_H_FRAC * src_h:
        return {**rep, "action": "reject",
                "reason": f"{why}; box: {h}px of {src_h}px (< {BOX_MIN_H_FRAC:.0%})"}
    visible = [b for b in banners if b[3] > ya and b[1] < yb]
    new_x = {}
    for i, s in enumerate(segs):
        x2 = _shift(s["x"], cw, visible, seg_faces[i], ya, yb, src_w)
        if x2 is None:
            return {**rep, "action": "reject",
                    "reason": f"{why}; box: segment {i} has no {cw}x{h} crop that keeps its "
                              f"faces without slicing a banner"}
        new_x[i] = x2
    for i, x2 in new_x.items():
        segs[i]["x"] = x2
    plan["crop_box"] = {"w": cw, "h": h, "y": ya, "fit": "box"}
    return {**rep, "action": "box", "crop_box": plan["crop_box"]}


DEAD_BAND_MIN_FRAC = 0.02   # a flat source band this deep (of the source height) is cropped out
DEAD_BAND_MARGIN = 2        # src px cut beyond a dead band's edge (encoder ringing on the seam)


def trim_dead_bands(plan, sc):
    """keep a track-like crop off the source's dead bands. 10-20 clip 108 was a
    9:16 vertical stream (360x640) whose own layout ends in 72 black rows; the track crop of a
    9:16 source is the whole frame, so the canvas carried them as a 216 px black band.

    `sc["dead_bands"]` (scan: rows/columns flat at one depth in most frames) gives the band
    depths in src px. When the crop window (crop_box, else the full-height 9:16 window) reaches
    into a top/bottom band >= DEAD_BAND_MIN_FRAC of the source height, its rows end at the band
    and, like `fix_track_plan`'s trim, a narrower 9:16 window of those rows fills the canvas: each
    segment keeps its centre where it can, inside its old columns, with its framed faces whole
    and no banner sliced. When no such window exists the rows are boxed instead (full crop
    width fit to the canvas width over the blurred fill), as long as BOX_MIN_H_FRAC of the
    source height remains. Mutates `plan`; returns {"action": none|trim|box|keep, ...}."""
    src_w, src_h = sc["src_w"], sc["src_h"]
    db = sc.get("dead_bands") or {}
    top = int(db.get("top", 0) or 0)
    bot = int(db.get("bottom", 0) or 0)
    top = top if top >= DEAD_BAND_MIN_FRAC * src_h else 0
    bot = bot if bot >= DEAD_BAND_MIN_FRAC * src_h else 0
    rep = {"dead_bands": {"top": top, "bottom": bot}}
    segs = plan.get("segments") or []
    if not (top or bot) or not segs:
        return {**rep, "action": "none"}
    box = plan.get("crop_box") or {}
    cw = int(box.get("w") or reframe.crop_width(src_w, src_h))
    y0 = int(box.get("y") or 0)
    y1 = y0 + int(box.get("h") or src_h)
    ya = max(y0, top + DEAD_BAND_MARGIN if top else y0)
    yb = min(y1, src_h - bot - DEAD_BAND_MARGIN if bot else y1)
    ya, yb = ya + ya % 2, yb - yb % 2
    if (ya, yb) == (y0, y1):
        return {**rep, "action": "none"}
    h = yb - ya
    if box.get("fit") == "box":
        if h < BOX_MIN_H_FRAC * src_h:
            return {**rep, "action": "keep", "reason": f"{h}px of {src_h}px left to box"}
        plan["crop_box"] = {"w": cw, "h": h, "y": ya, "fit": "box"}
        return {**rep, "action": "box", "crop_box": plan["crop_box"]}
    banners = [plate(g["box"], src_w, src_h) for g in sc.get("graphics") or [] if g["banner"]]
    visible = [b for b in banners if b[3] > ya and b[1] < yb]
    faces = sc.get("faces") or []
    cw2 = min(cw, int(round(h * 9 / 16)) // 2 * 2)
    new_x = {}
    for i, s in enumerate(segs):
        fb = _seg_faces(i, segs, faces, s["x"], cw, src_w, src_h)
        lo, hi = max(0, s["x"]), min(src_w - cw2, s["x"] + cw - cw2)
        centre = s["x"] + cw / 2 - cw2 / 2
        ok = [c for c in sorted(set(range(lo - lo % 2, hi + 1, 2)) | {lo, hi})
              if lo <= c <= hi and not any(sliced(b, c, cw2, src_w) for b in visible)
              and _faces_inside(fb, c, cw2, ya, yb, src_w)]
        if not ok:
            new_x = None
            break
        new_x[i] = min(ok, key=lambda c: abs(c - centre))
    if new_x is not None:
        for i, x2 in new_x.items():
            segs[i]["x"] = x2
        plan["crop_box"] = {"w": cw2, "h": h, "y": ya}
        return {**rep, "action": "trim", "crop_box": plan["crop_box"]}
    if h < BOX_MIN_H_FRAC * src_h:
        return {**rep, "action": "keep", "reason": f"{h}px of {src_h}px left to box"}
    plan["crop_box"] = {"w": cw, "h": h, "y": ya, "fit": "box"}
    return {**rep, "action": "box", "crop_box": plan["crop_box"]}


def fix_gamestack_plan(plan, sc):
    """Keep the gamestack game crop (full source height, columns [gx, gx + gw]) from slicing a
    banner (hard: reject when no shift works) or a text panel such as the chat column (best
    effort), never moving it over the cam guard box's columns. Mutates plan["game"][1]."""
    src_w = sc["src_w"]
    gw, gx, _fit = plan["game"]
    banners = [plate(g["box"], src_w, sc["src_h"]) for g in sc["graphics"] if g["banner"]]
    panels = [list(b) for b in sc.get("panels") or []]
    rep = {"banners": banners, "panels": panels, "frames": sc["frames"]}
    hard = [b for b in banners if sliced(b, gx, gw, src_w)]
    soft = [b for b in panels if sliced(b, gx, gw, src_w)]
    if not hard and not soft:
        return {**rep, "action": "none"}
    guard = plan.get("raw_cam_box") or plan.get("cam_box") or [0, 0, 0, 0]

    def legal(x):
        return 0 <= x <= src_w - gw and (x >= guard[2] or x + gw <= guard[0])

    def search(hard_boxes):
        def cost(x):
            return (sum(sliced(b, x, gw, src_w) for b in hard_boxes),
                    sum(sliced(b, x, gw, src_w) for b in panels), abs(x - gx))
        cands = {gx}
        for b in hard_boxes + panels:
            cands |= {b[2] + CUT_MARGIN, b[0] - CUT_MARGIN - gw,
                      b[0] - CUT_MARGIN, b[2] + CUT_MARGIN - gw}
        cands = [int(round(c)) // 2 * 2 for c in cands]
        best = min((c for c in cands if legal(c)), key=cost, default=gx)
        return best, cost(best)[0]

    best, n_hard = search(banners)
    if n_hard:
        # the plate margin is a guess; the burned-in text itself must still never be cut
        best, n_hard = search([g["box"] for g in sc["graphics"] if g["banner"]])
    if n_hard:
        return {**rep, "action": "reject",
                "reason": f"game crop {gw}px slices banner(s) at every legal x"}
    if best == gx:
        return {**rep, "action": "none", "unresolved_panels": soft}
    plan["game"] = [gw, best, plan["game"][2]]
    return {**rep, "action": "shift", "x": {"game": best}, "unresolved_panels":
            [b for b in panels if sliced(b, best, gw, src_w)]}
