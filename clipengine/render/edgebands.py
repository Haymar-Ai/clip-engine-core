#!/usr/bin/env python3
"""Flat edge bands: solid rows / columns along a frame edge.

10-20 clip 108 was a 9:16 vertical stream (360x640) whose own layout ends in 72 black rows. The
track crop of a 9:16 source is the whole frame, so the render carried that dead band as the
bottom 216 px (11.2 %) of the canvas, and no gate looked for it. The same measure serves both
sides: `render.banners.scan` finds a source's dead bands so the crop can leave them out, and
`quality_gate.measure_edge_bands` fails a rendered clip that still has one.

Per frame, the band on an edge is the run of lines (rows for top/bottom, columns for left/right)
from that edge that are each flat (luma std across the line <= LINE_STD) at the luma of the
outermost line (+- DRIFT). A band counts when the same depth (+- STABLE_PX) is there in at least
FRAME_FRAC of the frames: a pad, letterbox or dead source band sits at a fixed line, while a
dark backdrop's flat region moves with the speaker and the noise. numpy only.
"""
LINE_STD = 3.0            # luma std (0-255) across one row/column: flat
DRIFT = 4.0               # a flat line this far from the edge line's luma ends the band
FRAME_FRAC = 0.6          # a band must be in this share of the frames ...
STABLE_PX = 2             # ... at one depth (+- this many px of the frame it was measured on)
EDGES = ("top", "bottom", "left", "right")


def flat_edge_depths(gray, line_std=LINE_STD, drift=DRIFT):
    """{edge: band depth in px} of one gray (h x w) frame."""
    import numpy as np
    g = np.asarray(gray, dtype=np.float32)
    out = {}
    for edge in EDGES:
        lines = g if edge in ("top", "bottom") else g.T
        if edge in ("bottom", "right"):
            lines = lines[::-1]
        std = lines.std(axis=1)
        mean = lines.mean(axis=1)
        ok = (std <= line_std) & (np.abs(mean - mean[0]) <= drift)
        out[edge] = int(len(ok) if ok.all() else np.argmin(ok))
    return out


def stable_depth(values, tol=STABLE_PX):
    """(depth, frames): the depth shared (within +-tol) by the most frames; the depth reported is
    the smallest member of that group (the band every one of those frames has)."""
    best = (0, 0)
    vals = sorted(int(v) for v in values)
    for v in set(vals):
        group = [x for x in vals if abs(x - v) <= tol]
        if len(group) > best[1] or (len(group) == best[1] and min(group) > best[0]):
            best = (min(group), len(group))
    return best


def persistent_bands(depths, frame_frac=FRAME_FRAC, tol=STABLE_PX):
    """{edge: depth px} from per-frame `flat_edge_depths` dicts: the stable depth when at least
    `frame_frac` of the frames share it, else 0."""
    n = len(depths)
    out = {}
    for edge in EDGES:
        d, k = stable_depth([x.get(edge, 0) for x in depths], tol)
        out[edge] = d if n and k >= frame_frac * n else 0
    return out


def to_gray(frame):
    """BGR or gray frame -> gray uint8 (cv2 when available, else a luma mix)."""
    import numpy as np
    f = np.asarray(frame)
    if f.ndim == 2:
        return f
    try:
        import cv2
        return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    except ImportError:
        return (0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]).astype(np.uint8)
