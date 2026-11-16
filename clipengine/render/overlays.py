#!/usr/bin/env python3
"""Static corner overlays: a network's logo bug a crop keeps whole or out (render yield 11-09).

4 of 5 clips of one show were
rejected edge_slice for a logo bug at the top right of the remote guest's tile: the
speaker crop started a few px into the bug and ran out through its right side. Neither the banner
scan (a line detector + OCR of text lines) nor the gate's crop-slice check reads a script logo,
and the guests sit still, so the bug is one static pattern among the guests' own still edges.

A logo bug is burned in: its pixels do not change for the whole window, its edges are sharp and
dense, and it sits in a frame corner. Per pixel, over frames sampled at SAMPLE_FPS:
  - an edge (Canny, cleancrop.edge_persistence) in >= PERSIST of the frames, and
  - a luma std across the frames <= STILL_STD (a guest's head or a mic edge that moves a pixel
    reads a large std on its edge pixels: the logo stays put);
long straight runs (tile borders, the frame's edge) are taken out, the rest is averaged over a
DENSE_K window and kept where >= DENSE of it is such pixels (a logo's glyphs are dense; a window
frame's lines are not). A component whose centre lies within CORNER_X of the left / right edge and
CORNER_Y of the top / bottom edge, of bug size, is an overlay. A source whose camera moves (a
talking head shot handheld or zoomed) has no persistent still edges and gives none.

The crop rule (recompose): every overlay is whole in a crop window or out of it (the slice ladder
reads them as no_slice regions); one a final window still cuts is covered: the part of it the
window shows is painted out of the source before the crop (delogo: filled from the picture around
it, `cover_graph`), so no partial logo reaches the canvas.
"""
SAMPLE_FPS = 1.0
SCAN_W = 640
MIN_FRAMES = 6               # fewer sampled frames: no verdict (an overlay needs persistence)
PERSIST = 0.9                # an edge in >= this share of the frames ...
STILL_STD = 8.0              # ... whose luma std across the frames is at most this (0-255)
LINE_PX = 60                 # scan px: a straight run this long is a border, not a glyph
DENSE_K = (15, 7)            # scan px window the still-edge density is read on ...
DENSE = 0.18                 # ... a pixel is in an overlay when this share of the window is
CORNER_X = 0.3               # an overlay's centre is within this share of the width of a side ...
CORNER_Y = 0.15              # ... and of the height of the top or bottom ...
EDGE_Y = 0.08                # ... and it starts (ends) within this share of the height of the top
                             # (bottom) edge: a bug sits on the frame's margin; a still mic or a
                             # desk prop low in a guest's tile (2889) does not
ROW_FRAC = 0.25              # an overlay's rows: the run around its densest row of rows holding at
ROW_GAP = 6                  # least this share of that row's still edges (gaps of ROW_GAP rows
                             # bridged): a still window frame under the Playmaker bug (2940) is
                             # not the bug
MIN_AREA = 0.0008            # of the frame area
MAX_W = 0.4                  # of the frame width
MAX_H = 0.2                  # of the frame height
MIN_FILL = 0.08              # still-edge pixels / box area
PAD = 3                      # src px kept above / below an overlay's glyphs (cropcheck.SLICE_TOL):
                             # a bug sits a few px over a guest's cap (2889), every px counts ...
PAD_X = 10                   # ... and beside them: a glyph's outer stroke reads fainter than its
                             # edges (2889's NIGHT CAP: a crop at the box's edge cut the N)


def find(grays, src_w, src_h):
    """[{"box": [x0, y0, x1, y1] src px (padded PAD_X / PAD), "corner", "fill"}] of the static
    corner overlays in gray frames (one size, any scale) of one window; [] with < MIN_FRAMES
    frames."""
    import cv2
    import numpy as np
    from clipengine.render import cleancrop
    if len(grays) < MIN_FRAMES:
        return []
    P = cleancrop.edge_persistence(grays)
    S = np.std(np.stack([np.asarray(g, np.float32) for g in grays]), axis=0)
    M = ((P >= PERSIST) & (S <= STILL_STD)).astype(np.uint8)
    h, w = M.shape
    rect = cv2.getStructuringElement
    hl = cv2.morphologyEx(M, cv2.MORPH_OPEN, rect(cv2.MORPH_RECT, (LINE_PX, 1)))
    vl = cv2.morphologyEx(M, cv2.MORPH_OPEN, rect(cv2.MORPH_RECT, (1, LINE_PX)))
    M &= 1 - cv2.dilate(hl | vl, np.ones((3, 3), np.uint8))
    D = cv2.blur(M.astype(np.float32), DENSE_K)
    C = cv2.morphologyEx((D >= DENSE).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
    n, lab, st, _c = cv2.connectedComponentsWithStats(C)
    kx, ky = src_w / float(w), src_h / float(h)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in st[i])
        own = M[y:y + bh, x:x + bw] & (lab[y:y + bh, x:x + bw] == i)
        r0, r1 = _dense_rows(own.sum(axis=1))
        ys, xs = np.nonzero(own[r0:r1])
        if len(xs):                  # the box of the overlay's own still edges, not of the blur
            x, y, bw, bh = (x + int(xs.min()), y + r0 + int(ys.min()),
                            int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1)
        cx, cy = x + bw / 2.0, y + bh / 2.0
        side = "left" if cx < CORNER_X * w else "right" if cx > (1 - CORNER_X) * w else None
        end = "top" if cy < CORNER_Y * h and y <= EDGE_Y * h else \
            "bottom" if cy > (1 - CORNER_Y) * h and y + bh >= (1 - EDGE_Y) * h else None
        if not side or not end:
            continue
        if area < MIN_AREA * w * h or bw > MAX_W * w or bh > MAX_H * h:
            continue
        fill = float(M[y:y + bh, x:x + bw].mean())
        if fill < MIN_FILL:
            continue
        out.append({"box": [max(0, int(x * kx) - PAD_X), max(0, int(y * ky) - PAD),
                            min(int(src_w), int(round((x + bw) * kx)) + PAD_X),
                            min(int(src_h), int(round((y + bh) * ky)) + PAD)],
                    "corner": f"{end}_{side}", "fill": round(fill, 3)})
    return out


def _dense_rows(profile):
    """(r0, r1): the run of rows around the densest one whose still-edge count is at least
    ROW_FRAC of its, gaps of up to ROW_GAP rows bridged."""
    if not len(profile) or max(profile) <= 0:
        return 0, len(profile)
    peak = int(max(range(len(profile)), key=lambda r: profile[r]))
    on = [v >= ROW_FRAC * profile[peak] for v in profile]

    def walk(r, step):
        gap = 0
        while 0 <= r + step < len(on) and gap <= ROW_GAP:
            r += step
            gap = 0 if on[r] else gap + 1
        return r - step * gap
    return walk(peak, -1), walk(peak, 1) + 1


def scan(src, start, dur, src_w, src_h):
    """`find` on the clip window's frames (SAMPLE_FPS, SCAN_W wide)."""
    import cv2
    from clipengine.render import reframe
    grays = [cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) for _t, fr in reframe.iter_frames(
        str(src), start, dur, fps=SAMPLE_FPS, width=SCAN_W, src_size=(int(src_w), int(src_h)))]
    return find(grays, src_w, src_h)


def cut(views, boxes):
    """The overlay boxes a crop view (recompose.crop_views) shows part, but not all, of."""
    from clipengine.render import cropcheck
    out = []
    for b in boxes:
        for v in views or []:
            if cropcheck.cuts([v["x"], v["y"], v["x"] + v["w"], v["y"] + v["h"]], b):
                out.append(list(b))
                break
    return out


def visible_parts(views, boxes):
    """Per overlay box a crop view cuts, the part of it those views show: the box clipped to the
    union of the cutting windows (src px). Covering only that part keeps the cover's edges on the
    picture around the bug inside the crop (the Playmaker bug's box reaches over the guest tile's
    gold border, which a cover of the whole box smeared into the frame)."""
    from clipengine.render import cropcheck
    out = []
    for b in boxes:
        wins = [[v["x"], v["y"], v["x"] + v["w"], v["y"] + v["h"]] for v in views or []]
        wins = [w for w in wins if cropcheck.cuts(w, b)]
        if not wins:
            continue
        u = [min(w[0] for w in wins), min(w[1] for w in wins), max(w[2] for w in wins),
             max(w[3] for w in wins)]
        part = [max(b[0], u[0]), max(b[1], u[1]), min(b[2], u[2]), min(b[3], u[3])]
        if part[2] > part[0] and part[3] > part[1]:
            out.append([int(round(v)) for v in part])
    return out


def _overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def coverable(boxes, faces):
    """The boxes that overlap no face box (src px): a cover never paints over a face."""
    return [b for b in boxes if not any(_overlaps(b, f) for f in faces or [])]


DELOGO_MIN = 4               # px: a smaller cover box is grown to this


def cover_filter(boxes, src_w, src_h):
    """The ffmpeg filter chain that paints the `boxes` [x0, y0, x1, y1] out of the source:
    delogo (each box filled from the pixels around it), boxes kept 1 px inside the frame as
    delogo requires; "" without a box."""
    parts = []
    for x0, y0, x1, y1 in boxes or []:
        x0, y0 = max(1, int(x0)), max(1, int(y0))
        x1, y1 = min(int(src_w) - 1, int(x1)), min(int(src_h) - 1, int(y1))
        if x1 - x0 < DELOGO_MIN:
            x0 = max(1, min(x0, int(src_w) - 1 - DELOGO_MIN))
            x1 = x0 + DELOGO_MIN
        if y1 - y0 < DELOGO_MIN:
            y0 = max(1, min(y0, int(src_h) - 1 - DELOGO_MIN))
            y1 = y0 + DELOGO_MIN
        parts.append(f"delogo=x={x0}:y={y0}:w={x1 - x0}:h={y1 - y0}")
    return ",".join(parts)


def cover_graph(vgraph, boxes, src_w, src_h):
    """The layout graph reading a source whose `boxes` are painted out (cover_filter): every
    [0:v] use becomes an output of the covered source (recompose.with_levels' rule)."""
    n = vgraph.count("[0:v]")
    chain = cover_filter(boxes, src_w, src_h)
    if not chain or not n:
        return vgraph
    labels = [f"[cv{j}]" for j in range(n)]
    head = f"[0:v]{chain}" + (f",split={n}" if n > 1 else "") + "".join(labels) + ";"
    for lab in labels:
        vgraph = vgraph.replace("[0:v]", lab, 1)
    return head + vgraph
