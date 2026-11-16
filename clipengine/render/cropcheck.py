#!/usr/bin/env python3
"""Crop feasibility: what a crop must keep whole, whether a window cuts it, and the
selection-time probe that tells, before a window costs a select and a render, whether its speaker
can be shown without slicing a graphic.

10-18..25: 426 of 1302 clips carried an edge_slice reject. 14 of the 20 studied were dual views
whose zoom panel cut HUD plates, chat and logos (the gate does not read the zoom panel). 8 of
those were dual views in poker / tech / finance, where the layout rule allows no stack at all.
They were rendered, gated and eye-checked, then thrown away (EDGESLICE-FINDINGS.md).

  plates      what no crop edge may cross: the burned-in text items the gate reads
              (quality_gate.text_items) plus every OCR'd word of >= PLATE_MIN_ALNUM letters
              or digits, each grown to the plate drawn around it and merged
              (recompose.hud_panels' rule). The OCR's 4-letter items miss a HUD's 'BB' box, a
              card's rank, a handwritten '$100'; their plates do not.
  cut         a window shows part, but not all, of a plate (SLICE_TOL px of play).
  find_window the window of a given size nearest a wanted position that keeps a box inside
              (the speaker's face) and cuts no plate; vectorized over x like
              cleancrop._valid_x.
  probe       one cheap pass over a window: banners.scan at 1 fps (faces per frame, graphics,
              panels) plus OCR of 3 frames, about 4 s at load 40 against select's 192 s median.
              Per sampled frame with a face it asks whether the niche's allowed layouts can
              show that face clean: a full-bleed 9:16 speaker crop, or (gaming / streamers
              only, the stacking niches) a dual view zoom window. Frames without a face give no
              verdict: the frame gate and the screen box decide those as before.
  cam overlay (gaming / streamers): the stream's facecam is the small face that
              recurs at one place (cam_face). Its stack panel is clean only inside the cam's
              own box, read as a boxed tile (persistent borders, tile_finder). 10-26 run: 7 of 16
              gaming eye-check rejects were edge_slice, and in all 7 the stack's cam panel or face
              zoom reached past the cam into the game ('CAMERA' cut to 'ERA', dialogue choices,
              a compass, a stats bar). The probe counts a frame that shows the cam with its box
              read as clean (the stack's cam panel); the render planner keeps every shown
              window inside that box or clear of it (recompose.facecam_view_check).
"""
from __future__ import annotations

import math

PLATE_MIN_ALNUM = 2          # an OCR word of >= this many letters / digits marks a plate ('BB',
                             # '7♠' reads '7', a pot '$235'): recompose's HUD-niche ladder rule
PLATE_PAD = 6                # src px kept between a crop edge and a plate (= LADDER_TEXT_PAD)
SLICE_TOL = 3                # src px a plate may poke over a window edge (= cleancrop.SLICE_TOL)
OCR_FRAMES = 3               # = len(quality_gate.SLICE_OCR_TIMES)
FACE_UP, FACE_DOWN = 0.2, 0.1    # face box grown by this x its height (= cleancrop HEAD_UP /
FACE_PAD_X = 0.01                # CHIN_DOWN, FACE_PAD_X of src_w): what a crop keeps whole
MIN_CANVAS_W = 720           # = cleancrop.MIN_CANVAS_W: a full-bleed crop at least 2/3 the height
FULL_BLEED_STEPS = 6         # crop heights tried from the full height down to that
PROBE_FPS = 1.0              # = banners.SAMPLE_FPS
MIN_VERDICT_FRAMES = 3       # fewer sampled frames with a face: no verdict on the window
WINDOW_MIN_OK = 0.5          # a window whose faces can be shown clean on less than this share of
                             # its face frames is dropped before select's transcribe
PROBE_MIN_FRAC = 0.6         # a text item the probe counts persists in >= 60 % of the sampled frames
SMALL_FACE_H = 0.25          # = recompose.SCREEN_FACE_MAX_H: a frame whose faces are all smaller is
                             # a screen with a cam (non-stacking niches: the screen box decides it)
                             # or a facecam over a game (gaming / streamers: the cam stack does)
SPAN_MIN_OK = 0.75           # a candidate span below this share is dropped before its render
                             # (the eye-check rejects a few sliced seconds: 1309's 4 s cover)
CAM_POS_TOL = 0.06           # small faces this close (of the frame, centre to
                             # centre) are one facecam's ...
CAM_MIN_SHARE = 0.5          # ... a recurring one when it shows on >= this share of the sampled frames
                             # (a game character's face moves; the streamer's cam stays put)
CAM_TILE_MAX_FACE = 5.0      # a cam box at most this many face heights tall (a taller 'box' closed
CAM_TILE_MAX_AREA = 0.3      # by a scene line is no cam's) and at most this share of the frame
CAM_TILE_MIN_SHARE = 0.5     # the cam's box is read on >= this share of the frames that show its face
CAM_INSET = 3                # src px kept inside the cam box's border (= cleancrop.window_boxes)
TEXT_SLICE = "burned_text_slice"   # render yield 11-09 class 3: the clip's burned-in source text
                             # (textzones.source_text_spans) has no crop that holds it whole or
                             # leaves it out, and fitting its shots to width would be more blurred
                             # fill than the gate takes: the render fails with this code (never a
                             # sliced line shipped)


def text_slice_error(detail):
    """The render's failure message for TEXT_SLICE (it reads 'slice', so the engine replaces the
    window as for any no-clean-crop render: stages._SLICE_RENDER)."""
    return f"crop would slice burned-in source text ({TEXT_SLICE}): {detail}"


def alnum(text):
    return sum(c.isalnum() for c in str(text))


def hud_panels(boxes, src_w, src_h):
    """The panels a crop must keep whole: each text box grown to the plate drawn around it
    (banners.plate) and overlapping plates merged. The OCR reads a HUD panel's words, not the
    panel: 10-21 clip 472's recrop kept clear of '$24,500' and cut the board's last card (a suit
    the OCR does not read) and the panel's border."""
    from clipengine.render import banners
    out = [banners.plate(list(map(int, b)), src_w, src_h) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                    out[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del out[j]
                    merged = True
                    break
            if merged:
                break
    return out


def _inside(pt, box):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def plates(items, words, src_w, src_h, min_alnum=PLATE_MIN_ALNUM, pad=PLATE_PAD,
           n_frames=OCR_FRAMES):
    """[{"box": [x0, y0, x1, y1], "frac"}] src px: the plates of the gate's text `items` and the
    OCR `words` [(x0, y0, x1, y1, text, frame_i)] of >= min_alnum letters / digits, padded.
    frac = the share of the OCR'd frames a plate's words show in, or its items' frac when that
    is larger (a persistent graphic the 3 OCR frames read once)."""
    boxes = [list(map(int, it["box"])) for it in items or []]
    ws = [w for w in words or [] if alnum(w[4]) >= min_alnum]
    boxes += [list(map(int, w[:4])) for w in ws]
    out = []
    for b in hud_panels(boxes, src_w, src_h):
        seen = {w[5] if len(w) > 5 else 0 for w in ws
                if _inside(((w[0] + w[2]) / 2.0, (w[1] + w[3]) / 2.0), b)}
        frac = len(seen) / float(max(1, n_frames))
        for it in items or []:
            c = ((it["box"][0] + it["box"][2]) / 2.0, (it["box"][1] + it["box"][3]) / 2.0)
            if _inside(c, b):
                frac = max(frac, float(it.get("frac") or 0.0))
        out.append({"box": [max(0, b[0] - pad), max(0, b[1] - pad), min(src_w, b[2] + pad),
                            min(src_h, b[3] + pad)], "frac": round(min(1.0, frac), 3)})
    return out


def cuts(win, box, tol=SLICE_TOL):
    """True when window [x0, y0, x1, y1] shows part, but not all, of plate `box`."""
    if min(win[2], box[2]) - max(win[0], box[0]) <= tol or \
            min(win[3], box[3]) - max(win[1], box[1]) <= tol:
        return False                                   # (practically) not in the window
    return box[0] < win[0] - tol or box[2] > win[2] + tol or \
        box[1] < win[1] - tol or box[3] > win[3] + tol


def _union_s(spans):
    covered, end = 0.0, float("-inf")
    for a, b in sorted(spans):
        if b > end:
            covered += b - max(a, end)
            end = b
    return covered


def view_cuts(views, pls, dur):
    """The plates the crop `views` (recompose.crop_views; any panel) cut:
    [{"box", "frac", "share", "panels"}], share = frac x the share of the clip it is cut, the
    quality gate's crop_slices units."""
    dur = max(1e-6, float(dur))
    out = []
    for p in pls:
        spans, panels = [], set()
        for v in views or []:
            win = [v["x"], v["y"], v["x"] + v["w"], v["y"] + v["h"]]
            if cuts(win, p["box"]):
                spans.append((max(0.0, float(v["t0"])), min(dur, float(v["t1"]))))
                panels.add(v.get("panel") or "main")
        if spans:
            out.append({"box": p["box"], "frac": p["frac"],
                        "share": round(p["frac"] * _union_s(spans) / dur, 3),
                        "panels": sorted(panels)})
    return sorted(out, key=lambda c: -c["share"])


def cut_share(views, pls, dur):
    return round(sum(c["share"] for c in view_cuts(views, pls, dur)), 3)


def valid_xs(xs, y, w, h, pls, contain=None, tol=SLICE_TOL):
    """Boolean mask over window lefts `xs` (numpy) for a w x h window at row y: keeps `contain`
    [x0, y0, x1, y1] inside and cuts no plate box in `pls`."""
    import numpy as np
    ok = np.ones(len(xs), bool)
    if contain is not None:
        if contain[1] < y - 0.5 or contain[3] > y + h + 0.5:
            return np.zeros(len(xs), bool)
        ok &= (xs <= contain[0]) & (xs + w >= contain[2])
    for b in pls:
        x0, y0, x1, y1 = b
        if y1 <= y + tol or y0 >= y + h - tol:
            continue                                   # rows clear of the window
        outside = (xs + w <= x0 + tol) | (xs >= x1 - tol)
        if y0 >= y - tol and y1 <= y + h + tol:
            ok &= outside | ((xs <= x0 + tol) & (xs + w >= x1 - tol))
        else:                                          # the top or bottom edge cuts it
            ok &= outside
    return ok


def find_window(w, h, src_w, src_h, pls, contain=None, prefer=None):
    """(x, y) of the w x h window (src px, even) nearest `prefer` (x, y) that keeps `contain`
    inside and cuts no plate of `pls`, or None."""
    import numpy as np
    w, h = int(w), int(h)
    if w > src_w or h > src_h or w < 2 or h < 2:
        return None
    xs = np.arange(0, src_w - w + 1, 2)
    px, py = prefer if prefer is not None else ((src_w - w) / 2.0, (src_h - h) / 2.0)
    ys = {0, src_h - h, int(py)}
    for b in pls:
        ys |= {b[3] + SLICE_TOL, b[1] - SLICE_TOL - h, b[1] - SLICE_TOL}
    if contain is not None:
        ys |= {int(contain[1]), int(contain[3]) - h, int(contain[1] - 0.15 * (contain[3] -
                                                                              contain[1]))}
    cand = sorted({int(min(max(y, 0), src_h - h)) // 2 * 2 for y in ys}, key=lambda y: abs(y - py))
    best = None
    for y in cand:
        ok = valid_xs(xs, y, w, h, pls, contain)
        if not ok.any():
            continue
        good = xs[ok]
        x = int(good[np.argmin(np.abs(good - px))])
        d = math.hypot(x - px, y - py)
        if best is None or d < best[0]:
            best = (d, x, y)
    return (best[1], best[2]) if best else None


def face_box(f, src_w, src_h):
    """A face (normalized cx, cy, w, h) as the box a crop keeps whole (src px)."""
    fh = f["h"] * src_h
    pad = FACE_PAD_X * src_w
    return [f["cx"] * src_w - f["w"] * src_w / 2 - pad, max(0.0, f["cy"] * src_h - (0.5 + FACE_UP) * fh),
            f["cx"] * src_w + f["w"] * src_w / 2 + pad,
            min(float(src_h), f["cy"] * src_h + (0.5 + FACE_DOWN) * fh)]


def full_bleed_ok(box, pls, src_w, src_h, tile=None):
    """Whether a full-bleed 9:16 crop (between the full height and MIN_CANVAS_W of canvas) keeps
    face `box` whole and cuts no plate; inside `tile` [x0, y0, x1, y1] when the speaker sits in a
    boxed window (a two-up's tile, a stream's webcam box: cleancrop's b_tile rule)."""
    h_min = int(math.ceil(src_h * float(MIN_CANVAS_W) / 1080.0))
    ox, oy, aw, ah = 0, 0, src_w, src_h                # the area the window must stay inside
    if tile is not None:
        tx0, ty0, tx1, ty1 = (int(v) for v in tile)
        if ty1 - ty0 < h_min:
            return False                               # the tile is shorter than any clean crop
        ox, oy, aw, ah = tx0, ty0, tx1 - tx0, ty1 - ty0
        pls = [[p[0] - ox, p[1] - oy, p[2] - ox, p[3] - oy] for p in pls
               if p[0] < tx1 and p[2] > tx0 and p[1] < ty1 and p[3] > ty0]
        box = [box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy]
    for k in range(FULL_BLEED_STEPS + 1):
        h = int(round(src_h - (src_h - h_min) * k / float(FULL_BLEED_STEPS)))
        h -= h % 2
        w = min(int(round(h * 9 / 16)) // 2 * 2, src_w - src_w % 2)
        if h > ah or w > aw:
            continue
        if find_window(w, h, aw, ah, pls, contain=box,
                       prefer=((box[0] + box[2]) / 2.0 - w / 2.0, 0)) is not None:
            return True
    return False


TILE_MIN_FRAMES = 6          # boxed-window borders are read on at least this many probe frames


def tile_finder(grays, src_w):
    """f(face_box src px) -> the speaker's boxed window (src px) or None, read the way
    cleancrop.window_boxes reads it (persistent thin borders closing around the face) on the
    probe's own gray frames; None when too few frames (no tile verdict)."""
    from clipengine.render import cleancrop
    if len(grays) < TILE_MIN_FRAMES:
        return None
    P = cleancrop.edge_persistence(grays)
    k = src_w / float(grays[0].shape[1])
    cache = {}

    def find(box):
        key = tuple(int(round(v / k / 8.0)) for v in box)
        if key not in cache:
            got = cleancrop.find_window(P, [v / k for v in box])
            cache[key] = None if got is None else [int(round(v * k)) for v in got[0]]
        return cache[key]
    return find


def zoom_ok(box, pls, src_w, src_h, ar, h_min, h_max):
    """Whether a dual view zoom window (aspect `ar` = w / h, height h_min..h_max) keeps face `box`
    whole and cuts no plate."""
    h_lo, h_hi = int(h_min), int(min(h_max, src_h, src_w / ar))
    if h_hi < h_lo:
        return False
    for k in range(5):
        h = int(round(h_lo + (h_hi - h_lo) * k / 4.0))
        h -= h % 2
        w = int(round(h * ar)) // 2 * 2
        if find_window(w, h, src_w, src_h, pls, contain=box,
                       prefer=((box[0] + box[2]) / 2.0 - w / 2.0, box[1])) is not None:
            return True
    return False


def probe_plates(sc, items, words, src_w, src_h, min_frac=PROBE_MIN_FRAC, pad=PLATE_PAD):
    """The probe's plates: the gate's text items that persist (frac >= min_frac: an overlay
    stays put, a one-frame OCR read of a shirt or a wall does not), each cut down to the OCR'd
    text inside it (quality_gate.text_extent: a line-detector box can span half the frame), plus
    the scan's docked text panels (a scrolling chat column: its lines never persist)."""
    from clipengine import quality_gate as qg
    out = []
    for it in items or []:
        if float(it.get("frac") or 0.0) < min_frac:
            continue
        b = qg.text_extent(it["box"], words) if it.get("src") == "graphic" and words else it["box"]
        out.append([int(v) for v in b])
    return [{"box": [max(0, b[0] - pad), max(0, b[1] - pad), min(src_w, b[2] + pad),
                     min(src_h, b[3] + pad)], "frac": 1.0} for b in out]


def panel_plates(sc, src_w, src_h, pad=PLATE_PAD):
    """The scan's docked text panels (a chat column, a sidebar) as plates: on every frame."""
    return [{"box": [max(0, int(b[0]) - pad), max(0, int(b[1]) - pad), min(src_w, int(b[2]) + pad),
                     min(src_h, int(b[3]) + pad)], "frac": 1.0} for b in sc.get("panels") or []]


def present(pls, lines, tol=PLATE_PAD):
    """The plates of `pls` a frame shows: one of its text-line boxes `lines` (src px) has its
    centre in the plate. A window-level graphic is not on every frame (10-18 clip 1282: a tweet
    shown for 30 s of a 72 s window, then the speaker alone)."""
    out = []
    for p in pls:
        b = p["box"]
        if any(b[0] - tol <= (l[0] + l[2]) / 2.0 <= b[2] + tol and
               b[1] - tol <= (l[1] + l[3]) / 2.0 <= b[3] + tol for l in lines):
            out.append(p)
    return out


def frame_verdicts(faces, pls, src_w, src_h, zoom=None, lines=None, always=(), tiles=None,
                   cam=None):
    """[(t, ok or None, how)] per sampled frame: ok None without a face, or when every face is
    small (SMALL_FACE_H); True when the frame's largest face, or any face at least half its
    height, can be shown clean full-bleed or (zoom = (ar, h_min, h_max), the stacking niches) in
    a dual view zoom. `lines` [per frame text-line boxes] limits `pls` to the plates each frame
    shows (present); `always` (docked panels) count on every frame. `cam` (cam_overlay, the
    stacking niches): a small-face frame that shows the recurring facecam is True when the cam's
    box was read ("cam": the stack's cam panel stays inside it), else no verdict as before (an
    unread box is no proof of a facecam: 10-24 posted streamers 1017 / 1019 / 1293 are wide IRL
    shots whose small faces recur; the render planner decides those)."""
    out = []
    for i, (t, fs) in enumerate(faces):
        if lines is not None:
            fp = present(pls, lines[i] if i < len(lines) else []) + list(always)
        else:
            fp = list(pls) + list(always)
        if not fs:
            out.append((t, None, "no face"))
            continue
        top = max(f["h"] for f in fs)
        if top < SMALL_FACE_H:
            if cam and cam.get("tile") and any(_at_cam(f, cam["face"]) for f in fs):
                out.append((t, True, "cam"))
            else:
                out.append((t, None, "small face"))
            continue
        ok, how = False, "sliced"
        for f in sorted(fs, key=lambda f: -f["h"]):
            if f["h"] < 0.5 * top:
                break
            box = face_box(f, src_w, src_h)
            tile = tiles(box) if tiles is not None else None
            if full_bleed_ok(box, [p["box"] for p in fp], src_w, src_h, tile=tile):
                ok, how = True, "full_bleed" if tile is None else "tile"
                break
            if zoom and zoom_ok(box, [p["box"] for p in fp], src_w, src_h, *zoom):
                ok, how = True, "zoom"
                break
        out.append((t, ok, how))
    return out


def share_ok(verdicts, t0=None, t1=None):
    """(ok share or None, frames with a verdict) over [t0, t1] (window seconds)."""
    vs = [ok for t, ok, _h in verdicts if ok is not None
          and (t0 is None or t >= t0 - 0.5) and (t1 is None or t <= t1 + 0.5)]
    if not vs:
        return None, 0
    return round(sum(vs) / float(len(vs)), 3), len(vs)


def _at_cam(f, cam_face, tol=CAM_POS_TOL):
    """Whether face `f` (normalized cx, cy, h) is the facecam's: a small face at its place."""
    return (f["h"] < SMALL_FACE_H and abs(f["cx"] - cam_face["cx"]) <= tol
            and abs(f["cy"] - cam_face["cy"]) <= tol)


def cam_face(faces, min_share=CAM_MIN_SHARE, tol=CAM_POS_TOL):
    """The stream's facecam face {"cx", "cy", "h", "share"} (normalized medians) from sampled
    `faces` [(t, [face])]: the small face (< SMALL_FACE_H) that recurs within `tol` of one place
    on >= min_share of the frames; None without one (an IRL / talking-head stream, a game whose
    only faces are its characters)."""
    import statistics
    small = [(i, f) for i, (_t, fs) in enumerate(faces) for f in fs if f["h"] < SMALL_FACE_H]
    if not faces or not small:
        return None
    best = None
    for _i, c in small:
        near = [(i, f) for i, f in small if abs(f["cx"] - c["cx"]) <= tol
                and abs(f["cy"] - c["cy"]) <= tol]
        n = len({i for i, _f in near})
        if best is None or n > best[0]:
            best = (n, near)
    n, near = best
    share = n / float(len(faces))
    if share < min_share:
        return None
    return {k: round(statistics.median(f[k] for _i, f in near), 4) for k in ("cx", "cy", "h")} \
        | {"share": round(share, 3)}


def cam_overlay(faces, tiles, src_w, src_h, min_share=CAM_MIN_SHARE):
    """{"face": cam_face, "tile": [x0, y0, x1, y1] src px (CAM_INSET inside its border) or None,
    "read": share of the cam's frames its box was read on} of a window's facecam, or None when it
    has no recurring small face. `tiles` = tile_finder's f(face_box) (None: no tile verdict, so
    no box). The box is the one read most often around the cam's face; a box over
    CAM_TILE_MAX_FACE face heights tall or CAM_TILE_MAX_AREA of the frame is a scene's, not the
    cam's."""
    import collections
    cf = cam_face(faces, min_share=min_share)
    if cf is None:
        return None
    seen, n = collections.Counter(), 0
    for _t, fs in faces:
        at = [f for f in fs if _at_cam(f, cf)]
        if not at:
            continue
        n += 1
        if tiles is None:
            continue
        got = tiles(face_box(max(at, key=lambda f: f["h"]), src_w, src_h))
        if got:
            seen[tuple(int(v) for v in got)] += 1
    out = {"face": cf, "tile": None, "read": 0.0}
    if not seen or not n:
        return out
    # boxes read a few px apart are one box: the most common one, voted for by its neighbours
    box, _c = seen.most_common(1)[0]
    votes = sum(c for b, c in seen.items() if all(abs(b[k] - box[k]) <= 8 for k in range(4)))
    out["read"] = round(votes / float(n), 3)
    x0, y0, x1, y1 = box
    fh = cf["h"] * src_h
    if votes / float(n) < CAM_TILE_MIN_SHARE or (y1 - y0) > CAM_TILE_MAX_FACE * fh \
            or (x1 - x0) * (y1 - y0) > CAM_TILE_MAX_AREA * src_w * src_h:
        return out
    out["tile"] = [min(x0 + CAM_INSET, x1), min(y0 + CAM_INSET, y1), max(x1 - CAM_INSET, x0),
                   max(y1 - CAM_INSET, y0)]
    return out


def probe(media, start=0.0, dur=None, niche=None, vertical=None, stack=False, variants=False):
    """The window's crop feasibility: {"frames", "verdicts": [(t, ok, how)], "ok_share",
    "verdict_frames", "plates", "elapsed_s", "verdict": "ok" | "uncroppable" | "unknown",
    "why"}. `stack`: the niche may stack (gaming / streamers): a dual view zoom counts."""
    import time
    from clipengine.render import banners
    from clipengine.render import reframe
    t0 = time.monotonic()
    sw, sh = reframe.probe_size(str(media))
    if not sw or not sh:
        return {"verdict": "unknown", "why": "no frame size", "elapsed_s": 0.0}
    if dur is None:
        dur = float(_duration(media) or 0.0) - float(start)
    frames = list(reframe.iter_frames(str(media), float(start), float(dur), fps=PROBE_FPS,
                                      width=banners.SCAN_W, src_size=(sw, sh)))
    sc = banners.scan(str(media), float(start), float(dur), frames=frames, src_size=(sw, sh))
    k = sw / float(frames[0][1].shape[1]) if frames else 1.0
    lines = [[[v * k for v in lb] for lb in banners.line_boxes(fr)] for _t, fr in frames]
    from clipengine import quality_gate as qg
    try:
        words = qg.ocr_source_text(str(media), float(start), float(dur), (sw, sh))
    except Exception:          # noqa: BLE001 - the line detector's graphics still count
        words = None
    items = qg.text_items(sc, words) if words is not None else []
    pls = probe_plates(sc, items, words, sw, sh)
    panels = panel_plates(sc, sw, sh)
    zoom = None
    if stack:
        from clipengine.render import recompose
        geo = recompose.dual_view_geometry(sw, sh)
        if geo is not None:
            bh = recompose.OUT_H - geo[2]
            zoom = (recompose.OUT_W / float(bh), bh / recompose.DV_MAX_UPSCALE,
                    recompose.DV_MOTION_FRAC * sh)
    faces = sc.get("faces") or []
    import cv2
    tiles = tile_finder([cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) for _t, fr in frames], sw)
    cam = cam_overlay(faces, tiles, sw, sh) if stack else None
    verdicts = frame_verdicts(faces, pls, sw, sh, zoom=zoom, lines=lines, always=panels,
                              tiles=tiles, cam=cam)
    extra = {}
    if variants:                       # calibration: the other plate rules on the same scan
        for name, vp, ln in (("plates2", plates(items, words or [], sw, sh), None),
                             ("nopresence", pls, None),
                             ("notile", pls, lines)):
            vv = frame_verdicts(faces, vp, sw, sh, zoom=zoom, lines=ln, always=panels,
                                tiles=None if name == "notile" else tiles)
            extra[name] = [[round(t, 2), ok, how] for t, ok, how in vv]
    share, n = share_ok(verdicts)
    rep = {"frames": sc.get("frames"), "verdicts": [[round(t, 2), ok, how] for t, ok, how in verdicts],
           "ok_share": share, "verdict_frames": n, "plates": len(pls) + len(panels), "src": [sw, sh],
           "elapsed_s": round(time.monotonic() - t0, 2)}
    if cam:
        rep["cam"] = cam
    if extra:
        rep["variants"] = extra
    if n < MIN_VERDICT_FRAMES:
        rep.update(verdict="unknown", why=f"{n} sampled frame(s) with a face")
    elif share < WINDOW_MIN_OK:
        rep.update(verdict="uncroppable",
                   why=(f"no clean crop for the speaker on {n - round(share * n)} of {n} sampled "
                        f"frames ({len(pls)} graphic plate(s); "
                        f"{'full-bleed or dual view zoom' if zoom else 'full-bleed'} tried)"))
    else:
        rep.update(verdict="ok", why=f"clean crop on {share:.0%} of {n} face frames")
    return rep


def _duration(media):
    import subprocess
    try:
        return float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", str(media)], timeout=20).decode().strip())
    except Exception:          # noqa: BLE001
        return None
