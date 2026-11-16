#!/usr/bin/env python3
"""Boxed-broadcast detector (broadcast gate 10-21).

Three clips of a news show passed every gate and failed
eye-check: the source is a news show's row of 5 guest boxes over an animated network backdrop.
490 cropped the guest's box with the host's box cut at the crop edge, 534 fit the same span to
width over a blurred pad (the neighbouring boxes sliced at both canvas edges), and 535 opened on
6 frames of that boxed layout before the show cut to its speaker. The crop-slice check reads
burned-in TEXT, and a video box carries none, so nothing saw the boxes.

A video box shows up as long straight edges: its sides are perfectly vertical lines that run the
box's whole height and end where its top and bottom edges run across. Two boxes of one row share
those top and bottom edges, so the edge between them is a T-junction: the horizontal edge goes on
across it on both sides. A bookshelf, door or window edge in a studio is one line with no such
junctions (10-21 posted clips 265 / 357 / 407 have 0.2-0.5 frame-high ones).

Two uses:
  * a rendered 9:16 canvas (`canvas_boxes`): every tile border in it, and on which side a
    neighbouring box runs on to the canvas edge (a box the crop sliced). The quality gate fails a
    clip that keeps a sliced neighbour (edge_slice), counts those slivers as fill rather than
    content under a blurred pad, and labels each frame for the cover-frame check.
  * a source window (`detect_broadcast_panel`, the one entry point for other callers such as the
    coverage build): 3 or more tiled video boxes in a row, or a box (with a face in it) sitting on
    a backdrop graphic, with every face inside a box. The kinds step labels such a source
    `broadcast_panel`.

numpy + cv2 + ffmpeg only.
"""
import json
import subprocess
from pathlib import Path

EDGE_THR = 20                # luma step (0-255, 2 px apart, after a 3x3 blur) that is an edge
SEG_GAP_FRAC = 0.04          # a straight edge may be broken this long (a caption over it) ...
SEG_DENSITY = 0.8            # ... as long as this share of its length is edge
MERGE_PX_FRAC = 0.012        # parallel edges this close are one border (a border line's two sides)
JUNCTION_TOL_PX = 4          # rows a box's top / bottom edge may sit off a border's end
JUNCTION_DENSITY = 0.6       # share of the columns beside a border end that carry that edge
JUNCTION_REACH = 0.25        # ... looked at over this share of the width on each side
END_CLEAR_FRAC = 0.04        # past a box's end its side stops for this share of the height (a
                             # window grid's mullion goes on through the muntin: 10-21 clip 407)
END_CLEAR_DENSITY = 0.3      # ... "stops": at most this share of those rows is vertical edge
TEXT_STROKE_MAX = 0.4        # a junction row whose columns carry vertical strokes this densely is
                             # a line of text (the burned hook / captions: 0.63-0.81 measured;
                             # the box edges of 490 / 534 / 535: <= 0.26), not a box edge

CANVAS_SCAN_W = 540          # rendered 1080x1920 canvases are scanned at 540x960
BORDER_MIN_FRAC = 0.25       # a tile border spans at least this share of the canvas height
BORDER_X_MARGIN = 0.04       # ... away from the canvas sides (a box filling the width is clean)
BORDER_Y_MARGIN = 0.01       # ... with both ends inside the canvas
SLIVER_MIN_FRAC = 0.02       # a sliced neighbour shows at least this share of the width
FRAMED_SIDE_FRAC = 0.5       # a lone box's side (corners at both ends, no neighbour) this tall
                             # frames the picture: 10-20 clip 255 opened on 4 frames of a remote
                             # guest's outlined box on a dark backdrop

SOURCE_SCAN_W = 640          # source windows are scanned at 640 px wide
SOURCE_BORDER_MIN = 0.15     # a source box side spans at least this share of the frame height
TILE_MIN_W = 0.06            # a tile is at least this share of the frame width ...
ROW_TOL = 0.03               # ... and the sides of one row's tiles end within this share of H
PANEL_TILES = 3              # 3 or more boxes in a row = a tiled broadcast panel
BOX_MIN_AREA = 0.08          # a lone box on a backdrop covers at least this share of the frame
SOURCE_FRAMES = 8            # source frames sampled per window
PANEL_FRAME_FRAC = 0.4       # a window is a broadcast panel when this share of its frames are


# ---------------------------------------------------------------- straight edges

def _edge_maps(gray, thr=EDGE_THR):
    """(vertical-edge map, horizontal-edge map) of a gray frame: a pixel is on a vertical edge
    when the luma 2 px apart across it differs by >= thr; vertical edges are thinned to the local
    maximum across x (+- 1 px kept: an anti-aliased line wobbles by one)."""
    import cv2
    import numpy as np
    g = cv2.GaussianBlur(np.asarray(gray, np.uint8), (3, 3), 0).astype(np.int16)
    dx = np.zeros_like(g)
    dy = np.zeros_like(g)
    dx[:, 1:-1] = np.abs(g[:, 2:] - g[:, :-2])
    dy[1:-1, :] = np.abs(g[2:, :] - g[:-2, :])
    peak = (dx >= thr) & (dx >= np.roll(dx, 1, 1)) & (dx >= np.roll(dx, -1, 1))
    vert = peak | np.roll(peak, 1, 1) | np.roll(peak, -1, 1)
    return vert, dy >= thr


def _runs(on, gap):
    """[(a, b, n)] runs of True in a 1-D bool array, bridging gaps of <= gap; n = True count."""
    import numpy as np
    idx = np.flatnonzero(on)
    if not len(idx):
        return []
    out = []
    a = prev = int(idx[0])
    n = 1
    for i in idx[1:]:
        i = int(i)
        if i - prev - 1 <= gap:
            n += 1
        else:
            out.append((a, prev + 1, n))
            a, n = i, 1
        prev = i
    out.append((a, prev + 1, n))
    return out


def vertical_segments(vert, min_len, gap_frac=SEG_GAP_FRAC, density=SEG_DENSITY,
                      merge_frac=MERGE_PX_FRAC):
    """[(x, y0, y1)] straight vertical edges of >= min_len px in a vertical-edge map. Parallel
    edges a few px apart with the same ends (the two sides of one border line) are one; the
    longest is kept."""
    h, w = vert.shape
    gap = max(1, int(round(gap_frac * h)))
    found = []
    for x in range(2, w - 2):
        for a, b, n in _runs(vert[:, x], gap):
            if b - a >= min_len and n >= density * (b - a):
                found.append((x, a, b))
    merge = max(2, int(round(merge_frac * w)))
    tol = max(2, int(round(0.03 * h)))
    out = []
    for x, a, b in found:
        hit = next((i for i, (x2, a2, b2) in enumerate(out)
                    if abs(x - x2) <= merge and abs(a - a2) <= tol and abs(b - b2) <= tol), None)
        if hit is None:
            out.append((x, a, b))
        elif b - a > out[hit][2] - out[hit][1]:
            out[hit] = (x, a, b)
    return out


def edge_density(horiz, y, x0, x1, tol=JUNCTION_TOL_PX):
    """Share of the columns in [x0, x1) with a horizontal edge within +-tol rows of y."""
    import numpy as np
    h, w = horiz.shape
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    if x1 - x0 < 2:
        return 0.0
    band = horiz[max(0, int(y) - tol):min(h, int(y) + tol + 1), x0:x1]
    return float(np.any(band, axis=0).mean()) if band.size else 0.0


def box_edge(vert, horiz, y, x0, x1):
    """True when row y (+- JUNCTION_TOL_PX) holds a box's straight top / bottom edge over the
    columns [x0, x1): a horizontal edge on >= JUNCTION_DENSITY of them, and not a line of text
    (vertical strokes on < TEXT_STROKE_MAX of them)."""
    return (edge_density(horiz, y, x0, x1) >= JUNCTION_DENSITY
            and edge_density(vert, y, x0, x1) < TEXT_STROKE_MAX)


def side_stops(vert, x, y_end, step, clear_frac=END_CLEAR_FRAC, tol=JUNCTION_TOL_PX):
    """True when the vertical edge at column x stops past its end y_end (step -1: the top end,
    +1: the bottom end): over the next clear_frac of the height (inside the frame) at most
    END_CLEAR_DENSITY of the rows carry a vertical edge at x +- 1. A box's side ends at its top
    and bottom edges ("T"); a grid's line goes on through them ("+")."""
    h = vert.shape[0]
    n = max(2, int(round(clear_frac * h)))
    a = int(y_end) + step * (tol + 1)
    lo, hi = (a - n + 1, a + 1) if step < 0 else (a, a + n)
    if lo < 0 or hi > h:
        return False                               # no room to see it stop: not a box end
    col = vert[lo:hi, max(0, x - 1):x + 2].any(axis=1)
    return float(col.mean()) <= END_CLEAR_DENSITY


# ---------------------------------------------------------------- rendered canvas

def canvas_boxes(gray):
    """Tile borders on one rendered canvas frame (gray, any size; scanned at CANVAS_SCAN_W).

    A tile border is a straight vertical edge >= BORDER_MIN_FRAC of the canvas high, off the
    canvas sides, with both ends inside the canvas, and a T-junction at one of its ends (a
    horizontal edge going on across it on both sides: the top or bottom of a row of boxes, with
    the border stopping there: side_stops). The side between it and the nearer canvas edge is a
    sliced neighbour when that edge runs on all the way to the canvas edge (another box of the
    row, cut by the crop) over >= SLIVER_MIN_FRAC of the width. A side >= FRAMED_SIDE_FRAC of the
    canvas high with a corner at both ends (its box's top and bottom edges run off one side of it)
    is a framed box: one video box shown on a backdrop. Returns {"borders": [{"x", "y0", "y1",
    "sliced"}] (fractions of the canvas), "sliced": ["left" / "right"], "sliver_frac": canvas
    share of the sliced neighbours (the widest one per side), "framed": n framed-box sides,
    "label": "panel" (tile borders) | "framed" | "clean"}."""
    import cv2
    import numpy as np
    g = np.asarray(gray, np.uint8)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if g.shape[1] != CANVAS_SCAN_W:
        g = cv2.resize(g, (CANVAS_SCAN_W, max(2, int(round(g.shape[0] * CANVAS_SCAN_W
                                                                / g.shape[1])))),
                       interpolation=cv2.INTER_AREA)
    h, w = g.shape
    vert, horiz = _edge_maps(g)
    reach = int(round(JUNCTION_REACH * w))
    borders, sliver, framed = [], {}, 0
    for x, y0, y1 in vertical_segments(vert, int(round(BORDER_MIN_FRAC * h))):
        if not (BORDER_X_MARGIN * w <= x <= (1 - BORDER_X_MARGIN) * w):
            continue
        if y0 < BORDER_Y_MARGIN * h or y1 > (1 - BORDER_Y_MARGIN) * h:
            continue
        corner = {}
        for y, step in ((y0, -1), (y1 - 1, 1)):
            if side_stops(vert, x, y, step):
                corner[y] = (box_edge(vert, horiz, y, x - reach, x - 2),
                             box_edge(vert, horiz, y, x + 3, x + reach + 1))
        ends = [y for y, (left, right) in corner.items() if left and right]
        if not ends:
            if (y1 - y0) >= FRAMED_SIDE_FRAC * h and len(corner) == 2 \
                    and all(left or right for left, right in corner.values()):
                framed += 1
            continue
        side = "left" if x < w / 2 else "right"
        o0, o1 = (0, x - 2) if side == "left" else (x + 3, w)
        cut = (o1 - o0) >= SLIVER_MIN_FRAC * w and any(
            box_edge(vert, horiz, y, o0, o1) for y in ends)
        if cut:
            sliver[side] = max(sliver.get(side, 0.0), (o1 - o0) * (y1 - y0) / float(w * h))
        borders.append({"x": round(x / w, 3), "y0": round(y0 / h, 3), "y1": round(y1 / h, 3),
                        "sliced": side if cut else None})
    return {"borders": borders, "sliced": sorted(sliver),
            "sliver_frac": round(sum(sliver.values()), 4), "framed": framed,
            "label": "panel" if borders else "framed" if framed else "clean"}


# ---------------------------------------------------------------- source window

def source_boxes(gray, faces=None):
    """Video boxes on one source frame (gray; scanned at SOURCE_SCAN_W wide): rows of box sides
    (vertical edges >= SOURCE_BORDER_MIN of the height whose ends line up within ROW_TOL) and the
    boxes between neighbouring sides that carry a top or bottom edge. `faces` (normalised
    {cx, cy, h}) tell a video box from a picture on a wall: a box on a backdrop must hold a face,
    and a frame with a face outside every box is a studio shot, not a panel.

    Returns {"tiles": most boxes in one row, "boxes": [[x0, y0, x1, y1]] (fractions),
    "panel": bool, "why": str}."""
    import cv2
    import numpy as np
    g = np.asarray(gray, np.uint8)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if g.shape[1] != SOURCE_SCAN_W:
        g = cv2.resize(g, (SOURCE_SCAN_W, max(2, int(round(g.shape[0] * SOURCE_SCAN_W
                                                                / g.shape[1])))),
                       interpolation=cv2.INTER_AREA)
    h, w = g.shape
    vert, horiz = _edge_maps(g)
    sides = vertical_segments(vert, int(round(SOURCE_BORDER_MIN * h)))
    tol = ROW_TOL * h
    rows = []                                   # [[(x, y0, y1), ...]] sides of one row
    for s in sorted(sides):
        row = next((r for r in rows if abs(r[0][1] - s[1]) <= tol and abs(r[0][2] - s[2]) <= tol),
                   None)
        if row is None:
            rows.append([s])
        else:
            row.append(s)
    boxes, tiles = [], 0
    for row in rows:
        xs = sorted(x for x, _a, _b in row)
        y0 = int(np.median([a for _x, a, _b in row]))
        y1 = int(np.median([b for _x, _a, b in row]))
        n = 0
        for xa, xb in zip(xs, xs[1:]):
            if xb - xa < TILE_MIN_W * w:
                continue
            top = box_edge(vert, horiz, y0, xa + 2, xb - 1)
            bot = box_edge(vert, horiz, y1 - 1, xa + 2, xb - 1)
            if top or bot:
                n += 1
                boxes.append([round(xa / w, 3), round(y0 / h, 3), round(xb / w, 3),
                              round(y1 / h, 3), top and bot])
        tiles = max(tiles, n)
    faces = faces or []

    def inside(f, b):
        return b[0] <= f["cx"] <= b[2] and b[1] <= f["cy"] <= b[3]

    big = [f for f in faces if float(f.get("h") or 0) >= 0.04]
    outside = [f for f in big if not any(inside(f, b) for b in boxes)]
    if tiles >= PANEL_TILES and not outside:
        return {"tiles": tiles, "boxes": [b[:4] for b in boxes], "panel": True,
                "why": f"{tiles} tiled boxes in a row"}
    lone = [b for b in boxes
            if b[4] and (b[2] - b[0]) * (b[3] - b[1]) >= BOX_MIN_AREA
            and b[0] > 0.02 and b[2] < 0.98 and b[1] > 0.02 and b[3] < 0.98
            and any(inside(f, b) for f in big)]
    if lone and big and not outside and _backdrop_around(g, lone):
        return {"tiles": tiles, "boxes": [b[:4] for b in boxes], "panel": True,
                "why": f"{len(lone)} box(es) with a face on a backdrop graphic"}
    why = (f"face outside the boxes ({len(outside)})" if outside and boxes else
           f"{tiles} box(es) in a row" if boxes else "no video boxes")
    return {"tiles": tiles, "boxes": [b[:4] for b in boxes], "panel": False, "why": why}


def _backdrop_around(gray, boxes, ring=0.04):
    """True when the frame around the boxes is a graphic, not a pad: the ring just outside each
    box is not flat (a letterbox / pillarbox pad is a solid colour)."""
    import numpy as np
    h, w = gray.shape
    vals = []
    for x0, y0, x1, y1, *_ in boxes:
        X0, Y0 = int(max(0, (x0 - ring) * w)), int(max(0, (y0 - ring) * h))
        X1, Y1 = int(min(w, (x1 + ring) * w)), int(min(h, (y1 + ring) * h))
        m = np.zeros((h, w), bool)
        m[Y0:Y1, X0:X1] = True
        m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = False
        if m.any():
            vals.append(float(gray[m].std()))
    return bool(vals) and min(vals) > 6.0


def _probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True)
    try:
        d = json.loads(r.stdout or "{}")
        s = d["streams"][0]
        return int(s["width"]), int(s["height"]), float(d.get("format", {}).get("duration") or 0)
    except (ValueError, KeyError, IndexError):
        return 0, 0, 0.0


def _frame_at(path, t, width, size):
    import numpy as np
    sw, sh = size
    th = max(2, int(round(sh * width / sw / 2)) * 2)
    buf = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{float(t):.3f}", "-i",
                          str(path), "-frames:v", "1", "-vf", f"scale={width}:{th}", "-f",
                          "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, timeout=60).stdout
    if len(buf) < width * th * 3:
        return None
    return np.frombuffer(buf[:width * th * 3], np.uint8).reshape(th, width, 3)


def detect_broadcast_panel(media, start=0.0, dur=None, frames=SOURCE_FRAMES):
    """Is this source window a boxed broadcast panel? The one entry point for callers outside
    the gate (the kinds step, the coverage build's source fitness).

    Samples `frames` frames evenly over [start, start + dur] (the whole file by default) and runs
    `source_boxes` on each, with YuNet faces (render/reframe) to tell video boxes from wall
    pictures. A window is a panel when >= PANEL_FRAME_FRAC of the frames are. Never raises:
    {"panel": None, "reason": ...} when the media cannot be read.

    Returns {"panel": bool|None, "frac": share of panel frames, "frames": n, "tiles_max": k,
    "why": the reason of the first panel frame (or of the last frame)}."""
    try:
        import cv2
        from clipengine.render import reframe
        sw, sh, d = _probe(media)
        if not (sw and sh):
            return {"panel": None, "reason": f"unreadable media {media}"}
        span = float(dur) if dur else max(0.0, d - float(start))
        hits, n, tiles, why = 0, 0, 0, ""
        for i in range(int(frames)):
            t = float(start) + span * (i + 0.5) / int(frames)
            fr = _frame_at(media, t, min(sw, 1280), (sw, sh))
            if fr is None:
                continue
            n += 1
            v = source_boxes(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), reframe.detect_faces(fr))
            tiles = max(tiles, v["tiles"])
            if v["panel"]:
                hits += 1
                why = why or f"t={t - float(start):.1f}s: {v['why']}"
            elif not why:
                last = v["why"]
        if not n:
            return {"panel": None, "reason": "no frames read"}
        return {"panel": hits >= PANEL_FRAME_FRAC * n, "frac": round(hits / n, 3), "frames": n,
                "tiles_max": tiles, "why": why or last}
    except Exception as e:      # noqa: BLE001 - a label, never a crash
        return {"panel": None, "reason": f"{type(e).__name__}: {str(e)[:160]}"}


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(p, json.dumps(detect_broadcast_panel(p)))
