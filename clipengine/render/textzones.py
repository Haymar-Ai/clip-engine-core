#!/usr/bin/env python3
"""On-screen source text/UI in the hook zone.

The hook title is burned over the reframed canvas for its first ~3.4 s. Placement used to know
only about faces, so on a trading screen-share it landed on the ticker header ("S&P 500 Index",
"$7,811.350"). This module looks at the canvas the viewer will actually see during that window
(the layout's video graph, before captions) and returns the y-bands holding text or dense UI:

  - OCR (tesseract, when installed): confident words (conf >= OCR_MIN_CONF, >= 2 letters/digits).
  - Edge density: rows where >= EDGE_ROW_FRAC of the hook's horizontal span is strong edges
    (tickers, chart candles, UI grids). Natural footage and blurred fills stay well below it.

A row counts only when it is busy in >= MIN_FRAMES of the sampled frames, so a cursor or a
one-frame flash does not move the hook. `captions.hook_event` treats these bands as a hard
rule: the hook goes to a clear band or the render is rejected.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

PLAY_W, PLAY_H = 1080, 1920
HOOK_WINDOW_S = 3.4          # captions.HOOK_HOLD + fade-out
SAMPLE_FPS = 2.0             # ~7 frames over the hook window
MIN_FRAMES = 2               # a band must be busy in this many sampled frames
X_SPAN = (80, PLAY_W - 80)   # the hook's horizontal extent (captions.HOOK_MARGIN)
OCR_MIN_CONF = 60.0
OCR_MIN_ALNUM = 2
OCR_H_RANGE = (14, 160)      # canvas px; taller "words" are texture the OCR misread
EDGE_SCALE = 0.5             # edge density runs on a half-size canvas
EDGE_T = 200.0               # Sobel magnitude (0-255 gray, ksize 3) that counts as a strong edge
EDGE_ROW_FRAC = 0.25         # a row this full of strong edges is text/UI, not footage
EDGE_SMOOTH_ROWS = 9         # half-scale rows averaged before thresholding
MERGE_GAP = 24               # canvas px: busy rows closer than this merge into one band


def _ocr_available():
    return shutil.which("tesseract") is not None


def canvas_frames(src, start, dur, vgraph, fps=SAMPLE_FPS):
    """Gray canvas frames (PLAY_H x PLAY_W uint8) of `vgraph` (which must end in [v0]) over
    [start, start + dur] of `src`. Empty list when ffmpeg or numpy is unavailable."""
    try:
        import numpy as np
    except ImportError:
        return []
    graph = f"{vgraph};[v0]fps={fps},scale={PLAY_W}:{PLAY_H},format=gray[tz]"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{float(start):.3f}",
           "-t", f"{float(dur):.3f}", "-i", str(src), "-filter_complex", graph,
           "-map", "[tz]", "-an", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=180).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    size = PLAY_W * PLAY_H
    return [np.frombuffer(out[i * size:(i + 1) * size], np.uint8).reshape(PLAY_H, PLAY_W)
            for i in range(len(out) // size)]


def parse_tesseract_tsv(tsv, min_conf=OCR_MIN_CONF, min_alnum=OCR_MIN_ALNUM,
                        h_range=OCR_H_RANGE):
    """Word boxes [(x0, y0, x1, y1, text)] from `tesseract ... tsv` output, confident words
    only."""
    boxes = []
    for line in (tsv or "").splitlines()[1:]:
        p = line.split("\t")
        if len(p) < 12:
            continue
        text = p[11].strip()
        try:
            conf = float(p[10])
            x, y, w, h = (int(v) for v in p[6:10])
        except ValueError:
            continue
        if conf < min_conf or sum(c.isalnum() for c in text) < min_alnum:
            continue
        if not h_range[0] <= h <= h_range[1]:
            continue
        boxes.append((x, y, x + w, y + h, text))
    return boxes


def ocr_words(gray):
    """Confident OCR word boxes on one canvas frame; [] without tesseract."""
    if not _ocr_available():
        return []
    import cv2
    with tempfile.TemporaryDirectory(prefix="tz_") as d:
        img = Path(d) / "f.png"
        cv2.imwrite(str(img), gray)
        try:
            # one OpenMP thread: on the loaded box tesseract's default thread team spun one
            # 1080x1920 frame out to 257 s; single-threaded it is seconds
            r = subprocess.run(["tesseract", str(img), "stdout", "--psm", "11", "tsv"],
                               capture_output=True, text=True, timeout=60,
                               env={**os.environ, "OMP_THREAD_LIMIT": "1"})
        except (OSError, subprocess.TimeoutExpired):
            return []
    return parse_tesseract_tsv(r.stdout)


def dense_rows(gray, x_span=X_SPAN):
    """Boolean mask (len PLAY_H) of canvas rows dense with strong edges inside `x_span`."""
    import cv2
    import numpy as np
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (int(w * EDGE_SCALE), int(h * EDGE_SCALE)),
                       interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    x0, x1 = int(x_span[0] * EDGE_SCALE), int(x_span[1] * EDGE_SCALE)
    frac = (np.hypot(gx, gy)[:, x0:x1] > EDGE_T).mean(axis=1)
    k = np.ones(EDGE_SMOOTH_ROWS) / EDGE_SMOOTH_ROWS
    busy = np.convolve(frac, k, mode="same") >= EDGE_ROW_FRAC
    return np.repeat(busy, int(round(1 / EDGE_SCALE)))[:h]


def frame_mask(gray, words=None, x_span=X_SPAN):
    """Busy-row mask of one frame: OCR word rows inside x_span OR edge-dense rows."""
    import numpy as np
    mask = np.zeros(gray.shape[0], bool)
    for x0, y0, x1, y1, _t in (ocr_words(gray) if words is None else words):
        if x1 > x_span[0] and x0 < x_span[1]:
            mask[max(0, y0):min(gray.shape[0], y1)] = True
    return mask | dense_rows(gray, x_span)


def bands_from_counts(counts, min_frames=MIN_FRAMES, merge_gap=MERGE_GAP):
    """[(y0, y1)] runs of rows busy in >= min_frames frames, runs closer than merge_gap merged."""
    bands = []
    y = 0
    n = len(counts)
    while y < n:
        if counts[y] >= min_frames:
            y0 = y
            while y < n and counts[y] >= min_frames:
                y += 1
            if bands and y0 - bands[-1][1] < merge_gap:
                bands[-1] = (bands[-1][0], y)
            else:
                bands.append((y0, y))
        else:
            y += 1
    return bands


def hook_zone_text(src, start, dur, vgraph, frames=None):
    """{"bands": [(y0, y1)], "frames": n, "ocr": bool, "words": [...]} for the hook window.
    `frames` (gray canvas arrays) skips the ffmpeg pass (tests)."""
    import numpy as np
    if frames is None:
        frames = canvas_frames(src, start, min(float(dur), HOOK_WINDOW_S), vgraph)
    counts = np.zeros(PLAY_H, int)
    seen = []
    for g in frames:
        words = ocr_words(g)
        seen.extend(w[4] for w in words)
        counts += frame_mask(g, words)
    bands = bands_from_counts(counts) if frames else []
    return {"bands": bands, "frames": len(frames), "ocr": _ocr_available(),
            "words": sorted(set(seen))[:40]}


# ---------------------------------------------------------------- burned-in source text per shot

# render yield 11-09 class 3: one creator's uploads burn big kinetic captions
# into the lower frame for 1-4 s at a time ("The work is off your plate / but the decisions
# aren't", 840 src px wide), and the 405 px speaker crop sliced them at both sides. The banner
# scan keeps the lines that persist and the gate OCRs 3 frames, so neither saw them. This reads
# the SOURCE window at SRC_FPS: text-line candidates (horizontal Sobel edges closed into lines,
# long vertical runs such as a wall panel's edges taken out first) tracked frame to frame, each
# track confirmed by one OCR read of its box (>= SRC_MIN_ALNUM letters / digits), and the tracks on
# screen together merged into text events {"t0", "t1", "box"} (clip s, src px).
SRC_FPS = 2.0
SRC_SCAN_W = 640
SRC_GRAD_T = 90              # |Sobel x| (0-255) of a glyph's edge
SRC_VLINE_PX = 25            # scan px: a vertical edge run this long is a panel's or a frame's
SRC_MIN_W = 0.06             # a line candidate spans at least this share of the frame width ...
SRC_MIN_H_PX = 6             # ... at least this many scan px tall, at most SRC_MAX_H of the height
SRC_MAX_H = 0.3              # (a two-line caption fully on screen closes into one block: 2977's
                             # 'The work is off your plate / but the decisions aren't', 91 scan px),
SRC_MIN_ASPECT = 2.0         # ... at least this many times as wide as tall
SRC_MIN_FILL = 0.35          # ... and this full of edge pixels once closed
SRC_TRACK_IOU = 0.3          # a candidate continues a track it overlaps this much (IoU) ...
SRC_TRACK_GAP_S = 0.75       # ... seen at most this long before
SRC_OCR_UPSCALE = 3
SRC_MIN_ALNUM = 4            # = quality_gate.SLICE_MIN_ALNUM
SRC_EVENT_PAD_S = 0.75       # an event holds this long before its first / after its last sample
SRC_EVENT_GAP_S = 0.0        # confirmed lines whose held spans come this close are one event
SRC_BOX_PAD = 2              # src px around an event's box
SRC_TRACK_MAX_FRAC = 0.5     # a line on screen for more of the window is a persistent graphic (a
                             # ticker, a chat: the banner guard's), left out before tracks merge so
                             # a caption shown with it stays a shot of its own
SRC_CHANGE_GAP_S = 0.5       # a caption comes and goes: its box SRC_CHANGE_GAP_S before its first
SRC_CHANGE_MIN = 25.0        # or after its last sample differs from it by this mean luma (0-255)
                             # on one side at least; a neon sign behind a podcast host the line
                             # detector reads now and then does not (2884 'NIGH CA': 15; Dan
                             # Martell's captions: 33-85)


def text_line_candidates(gray):
    """[[x0, y0, x1, y1]] (px of `gray`) of text-line candidates on one gray frame."""
    import cv2
    import numpy as np
    h, w = gray.shape[:2]
    mag = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    _, bw = cv2.threshold(mag, SRC_GRAD_T, 255, cv2.THRESH_BINARY)
    vl = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (1, SRC_VLINE_PX)))
    bw = cv2.bitwise_and(bw, cv2.bitwise_not(cv2.dilate(vl, np.ones((3, 3), np.uint8))))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 40), 3))
    cl = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    cl = cv2.morphologyEx(cl, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n, _lab, st, _c = cv2.connectedComponentsWithStats(cl)
    out = []
    for i in range(1, n):
        x, y, bw_, bh, area = (int(v) for v in st[i])
        if bw_ < SRC_MIN_W * w or bh < SRC_MIN_H_PX or bh > SRC_MAX_H * h \
                or bw_ < SRC_MIN_ASPECT * bh or area < SRC_MIN_FILL * bw_ * bh:
            continue
        out.append([x, y, x + bw_, y + bh])
    return out


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / float(u) if u > 0 else 0.0


def track_lines(frames, candidates=text_line_candidates):
    """Candidate lines of gray `frames` [(t, gray)] tracked over time: [{"t0", "t1", "box",
    "frames": [(t, box)]}] (px of the frames)."""
    tracks = []
    for t, g in frames:
        for b in candidates(g):
            hit = next((tr for tr in tracks if t - tr["t1"] <= SRC_TRACK_GAP_S
                        and _iou(tr["last"], b) >= SRC_TRACK_IOU), None)
            if hit is None:
                tracks.append({"t0": t, "t1": t, "last": b, "box": list(b), "frames": [(t, b)]})
                continue
            u = hit["box"]
            hit.update(t1=t, last=b, box=[min(u[0], b[0]), min(u[1], b[1]), max(u[2], b[2]),
                                          max(u[3], b[3])])
            hit["frames"].append((t, b))
    return tracks


def _ocr_box(gray, box, pad=6):
    """Confident OCR words of `box` on `gray` (upscaled SRC_OCR_UPSCALE); [] without tesseract."""
    import cv2
    x0, y0, x1, y1 = box
    crop = gray[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    if crop.size == 0:
        return []
    big = cv2.resize(crop, None, fx=SRC_OCR_UPSCALE, fy=SRC_OCR_UPSCALE,
                     interpolation=cv2.INTER_CUBIC)
    return ocr_words(big)


def came_and_went(box, t0, t1, t_mid, gray_at, times):
    """Whether text box `box` (scan px) shown from t0 to t1 changes the picture around it
    (SRC_CHANGE_MIN, on the frame SRC_CHANGE_GAP_S before t0 or after t1, against the frame at
    t_mid): burned-in text that appears, not a sign in the scene. True when no frame is that far
    from it (no verdict)."""
    import numpy as np
    x0, y0, x1, y1 = (int(v) for v in box)
    mid = np.asarray(gray_at(t_mid), np.float32)[y0:y1, x0:x1]
    before = [u for u in times if u <= t0 - SRC_CHANGE_GAP_S]
    after = [u for u in times if u >= t1 + SRC_CHANGE_GAP_S]
    sides = ([before[-1]] if before else []) + ([after[0]] if after else [])
    if not sides or mid.size == 0:
        return True
    return max(float(np.abs(np.asarray(gray_at(u), np.float32)[y0:y1, x0:x1] - mid).mean())
               for u in sides) >= SRC_CHANGE_MIN


def text_events(tracks, gray_at, src_w, src_h, scan_w, dur, ocr=_ocr_box, times=None):
    """Text events [{"t0", "t1", "box" (src px), "text"}] of the tracks whose OCR read (one frame,
    the track's middle sample: `gray_at(t)`) holds >= SRC_MIN_ALNUM letters / digits and that are
    up for at most SRC_TRACK_MAX_FRAC of the window; tracks whose held spans (SRC_EVENT_PAD_S on
    either side) meet (SRC_EVENT_GAP_S) merged into one event, and (with the sampled `times`) an
    event that did not come and go (came_and_went: a sign in the scene) dropped."""
    k = float(src_w) / float(scan_w)
    hits = []
    for tr in tracks:
        if tr["t1"] - tr["t0"] > SRC_TRACK_MAX_FRAC * float(dur):
            continue
        t, _b = tr["frames"][len(tr["frames"]) // 2]
        words = ocr(gray_at(t), tr["box"])
        text = " ".join(str(w[4]) for w in words)
        if sum(c.isalnum() for c in text) >= SRC_MIN_ALNUM:
            hits.append((tr["t0"], tr["t1"], tr["box"], text, t))
    events = []
    for t0, t1, b, text, t in sorted(hits, key=lambda h: h[0]):
        if events and t0 - events[-1]["t1"] <= 2 * SRC_EVENT_PAD_S + SRC_EVENT_GAP_S:
            e = events[-1]
            e["t1"] = max(e["t1"], t1)
            e["box"] = [min(e["box"][0], b[0]), min(e["box"][1], b[1]), max(e["box"][2], b[2]),
                        max(e["box"][3], b[3])]
            e["text"] += " / " + text
            e["mids"].append(t)
        else:
            events.append({"t0": t0, "t1": t1, "box": list(b), "text": text, "mids": [t]})
    if times is not None:
        events = [e for e in events
                  if came_and_went(e["box"], e["t0"], e["t1"], e["mids"][len(e["mids"]) // 2],
                                   gray_at, times)]
    p = SRC_BOX_PAD
    return [{"t0": round(max(0.0, e["t0"] - SRC_EVENT_PAD_S), 3),
             "t1": round(min(float(dur), e["t1"] + SRC_EVENT_PAD_S), 3),
             "box": [max(0, int(e["box"][0] * k) - p), max(0, int(e["box"][1] * k) - p),
                     min(int(src_w), int(round(e["box"][2] * k)) + p),
                     min(int(src_h), int(round(e["box"][3] * k)) + p)],
             "text": e["text"][:120]} for e in events]


def source_text_spans(src, start, dur, src_w, src_h, frames=None):
    """The source window's burned-in text events (text_events). `frames` [(t, gray)] at any one
    width skips the decode (tests); [] when no frame is read."""
    if frames is None:
        import cv2
        from clipengine.render import reframe
        frames = [(t, cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)) for t, fr in reframe.iter_frames(
            str(src), start, dur, fps=SRC_FPS, width=SRC_SCAN_W, src_size=(int(src_w), int(src_h)))]
    if not frames:
        return []
    at = dict(frames)
    return text_events(track_lines(frames), lambda t: at[t], src_w, src_h,
                       frames[0][1].shape[1], dur, times=[t for t, _g in frames])


CLIP_SAMPLE_FPS = 0.5        # captions are up the whole clip: sample it all, sparser
CLIP_MIN_FRAC = 0.2          # ... a band counts when busy in >= 20 % of the samples


def clip_text_bands(src, start, dur, vgraph, frames=None, fps=CLIP_SAMPLE_FPS):
    """{"bands", "frames"} of canvas rows holding on-screen
    text / dense UI over the WHOLE clip (the caption band is up for all of it), by the same
    OCR + edge-density rule as the hook zone; a band must be busy in >= CLIP_MIN_FRAC of the
    sampled frames (and >= MIN_FRAMES)."""
    import numpy as np
    if frames is None:
        frames = canvas_frames(src, start, float(dur), vgraph, fps=fps)
    counts = np.zeros(PLAY_H, int)
    for g in frames:
        counts += frame_mask(g, ocr_words(g))
    need = max(MIN_FRAMES, int(np.ceil(CLIP_MIN_FRAC * len(frames))))
    return {"bands": bands_from_counts(counts, min_frames=need) if frames else [],
            "frames": len(frames)}
