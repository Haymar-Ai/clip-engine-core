#!/usr/bin/env python3
"""Subject-aware 9:16 reframe planning.

The I/O half samples the clip window at a low frame rate (one ffmpeg rawvideo pipe), detects
faces (vendored YuNet model, Haar cascade fallback) and finds shot cuts (HSV histogram
correlation). The decision half (decide_layout, track_crops, split_crops, crop_x_expr) is pure
so the framing rules are unit-testable with synthetic samples.

Conventions: face dicts are {"cx", "cy", "w", "h", "score"} normalized 0-1 (cx/cy = centre);
all times are clip-relative seconds; crop positions are source pixels.
"""
import json
import statistics
import subprocess
from pathlib import Path

from clipengine import config


MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
MIN_SCORE = 0.7          # detector confidence floor
MIN_FACE_H = 0.04        # faces smaller than 4% of frame height are crowd/noise
CUT_CORREL = 0.6         # HSV histogram correlation below this between samples = shot cut

MIN_FACE_RATIO = 0.5     # < 50% of samples with a face -> keep the full frame (general)
SPLIT_RATIO = 0.4        # two-speaker samples needed for split
SPLIT_MIN_H = 0.08       # each split speaker at least 8% of frame height
SPLIT_MIN_SEP = 0.30     # speaker centres at least 30% of frame width apart
SPLIT_STABLE = 0.7       # left/right assignment stable in >= 70% of two-speaker samples
SPLIT_POS_TOL = 0.15     # "stays put": within 15% of width of its median position

DRIFT_MIN_SHOT = 6.0     # only shots longer than this are checked for subject drift
DRIFT_MAX = 0.12         # primary face moving more than 12% of width = drift
DRIFT_STEP = 3.0         # drifting shots are re-locked every 3s
MERGE_TOL = 0.02         # adjacent segments closer than 2% of width are merged
TIE_TOL = 0.9            # faces within 10% of the largest height count as a size tie
CUT_MIN_GAP = 0.5        # a second "cut" within 0.5s of the last is a re-trigger (flash, zoom); dropped
CUT_SEARCH = (0.5, 0.35) # exact-frame cut search window: seconds before/after the sampled cut
PAN_S = 0.6              # in-shot re-locks ease over this many seconds (no snap)
SWITCH_MIN = 0.2         # an in-shot re-lock moving > 20% of width is a subject switch: hard, no whip pan
SPEAKER_DEAD_ZONE = 0.12 # speaker face must drift this far outside crop centre before re-locking

_DETECTOR = None         # (kind, detector) created lazily, once per process


# ---------------------------------------------------------------- I/O: frames, faces, cuts

HOOK_FACE_WINDOW = 4.0     # s; covers captions.HOOK_HOLD + fade-out


def probe_size(src):
    """(width, height) of the first video stream."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "json", str(src)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    st = json.loads(r.stdout)["streams"][0]
    return int(st["width"]), int(st["height"])


def probe_fps(src):
    """Native frame rate of the first video stream (24 when unknown)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", str(src)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    n, _, d = r.stdout.strip().partition("/")
    try:
        return float(n) / float(d or 1) or 24.0
    except (ValueError, ZeroDivisionError):
        return 24.0


def iter_frames(src, start, dur, fps=3, width=640, src_size=None):
    """Yield (t, frame_bgr) for the window, streamed from one ffmpeg rawvideo pipe."""
    import numpy as np
    sw, sh = src_size or probe_size(src)
    height = max(2, int(round(sh * width / sw / 2)) * 2)   # what scale=W:-2 would pick
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", str(start), "-t", str(dur), "-i", str(src),
           "-an", "-vf", f"fps={fps},scale={width}:{height}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    size = width * height * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        i = 0
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield round(i / fps, 3), np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            i += 1


def sample_frames(src, start, dur, fps=3, width=640):
    """[(t, frame_bgr)] sampled at `fps`, scaled to `width`; t is clip-relative."""
    return list(iter_frames(src, start, dur, fps=fps, width=width))


def _detector():
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR
    _DETECTOR = ("none", None)
    try:
        import cv2
    except ImportError:
        return _DETECTOR
    try:   # cv2 5.x warns about DNN targets on every create; not actionable
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    if MODEL.exists() and hasattr(cv2, "FaceDetectorYN"):
        try:
            _DETECTOR = ("yunet", cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320),
                                                              MIN_SCORE, 0.3, 5000))
            return _DETECTOR
        except Exception:
            pass
    cascades = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if cascades:
        path = Path(cascades) / "haarcascade_frontalface_default.xml"
        if path.exists():
            clf = cv2.CascadeClassifier(str(path))
            if not clf.empty():
                _DETECTOR = ("haar", clf)
    return _DETECTOR


def detect_faces(frame):
    """Faces in a BGR frame, normalized, largest first. [] when no detector is available."""
    kind, det = _detector()
    fh, fw = frame.shape[:2]
    boxes = []
    if kind == "yunet":
        det.setInputSize((fw, fh))
        _, found = det.detect(frame)
        for f in (found if found is not None else []):
            boxes.append((float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])))
    elif kind == "haar":
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m = max(1, int(MIN_FACE_H * fh))
        for x, y, w, h in det.detectMultiScale(gray, 1.1, 5, minSize=(m, m)):
            boxes.append((float(x), float(y), float(w), float(h), 1.0))
    faces = []
    for x, y, w, h, score in boxes:
        if score < MIN_SCORE or h / fh < MIN_FACE_H:
            continue
        faces.append({"cx": round(min(max((x + w / 2) / fw, 0.0), 1.0), 4),
                      "cy": round(min(max((y + h / 2) / fh, 0.0), 1.0), 4),
                      "w": round(w / fw, 4), "h": round(h / fh, 4), "score": round(score, 3)})
    return sorted(faces, key=lambda f: f["h"], reverse=True)


def _hist(frame):
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
    return h


def _is_cut(prev_hist, hist):
    import cv2
    return cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < CUT_CORREL


def _cut_times(hists):
    """[(t, hist)] -> sampled cut times, dropping re-triggers within CUT_MIN_GAP of the last."""
    cuts = []
    for (_, a), (t, b) in zip(hists, hists[1:]):
        if _is_cut(a, b) and (not cuts or t - cuts[-1] >= CUT_MIN_GAP):
            cuts.append(t)
    return cuts


def shot_boundaries(frames):
    """Cut times: samples whose HSV histogram correlates < 0.6 with the previous sample."""
    return _cut_times([(t, _hist(f)) for t, f in frames])


def refine_cut(src, start, t, fps, src_size):
    """Exact clip time of a camera cut sampled at `t`: decode the surrounding frames at the
    source rate and return the midpoint between the last pre-cut and first post-cut frame.

    Sampling at 3fps places a cut up to a third of a second off the real edit. A crop switch
    that lands on the edit is invisible; one that lands a few frames early or late moves the
    frame inside a continuous shot, which reads as a twitch. Falls back to `t` when the
    window holds no clear cut.
    """
    import cv2
    a = max(t - CUT_SEARCH[0], 0.0)
    hists = [(ft, _hist(f)) for ft, f in
             iter_frames(src, start + a, CUT_SEARCH[0] + CUT_SEARCH[1], fps=fps, src_size=src_size)]
    if len(hists) < 2:
        return t
    corr, ft = min((cv2.compareHist(h0, h1, cv2.HISTCMP_CORREL), t1)
                   for (_, h0), (t1, h1) in zip(hists, hists[1:]))
    return round(a + ft - 0.5 / fps, 3) if corr < CUT_CORREL else t


# ---------------------------------------------------------------- pure decisions

def _pair(faces):
    """(left, right) speakers if the two largest big faces are well separated, else None."""
    big = sorted((f for f in faces or [] if f["h"] >= SPLIT_MIN_H),
                 key=lambda f: f["h"], reverse=True)[:2]
    if len(big) == 2 and abs(big[0]["cx"] - big[1]["cx"]) >= SPLIT_MIN_SEP:
        return tuple(sorted(big, key=lambda f: f["cx"]))
    return None


def _stable_pair(pairs):
    left_big = sum(1 for l, r in pairs if l["h"] > r["h"])
    right_big = sum(1 for l, r in pairs if r["h"] > l["h"])
    if max(left_big, right_big) >= SPLIT_STABLE * len(pairs):
        return True
    # equal-size speakers: stable if both keep their seats (positions stay near their medians)
    lm = statistics.median(l["cx"] for l, _ in pairs)
    rm = statistics.median(r["cx"] for _, r in pairs)
    steady = sum(1 for l, r in pairs
                 if abs(l["cx"] - lm) <= SPLIT_POS_TOL and abs(r["cx"] - rm) <= SPLIT_POS_TOL)
    return steady >= SPLIT_STABLE * len(pairs)


def decide_layout(samples):
    """'general' | 'split' | 'track' from [{"t", "faces": [...]}]."""
    n = len(samples)
    if not n or sum(1 for s in samples if s.get("faces")) < MIN_FACE_RATIO * n:
        return "general"
    pairs = [p for p in (_pair(s.get("faces")) for s in samples) if p]
    if len(pairs) >= SPLIT_RATIO * n and _stable_pair(pairs):
        return "split"
    return "track"


def crop_width(src_w, src_h, out_w=9, out_h=16):
    """Crop width for a full-height crop at the given output aspect: round(src_h * out_w/out_h),
    even, within the frame. Defaults to 9:16."""
    cw = int(round(src_h * out_w / out_h))
    cw -= cw % 2
    return max(2, min(cw, src_w - src_w % 2))


# broadcast layout: top band 1080x1312 (subject track) + bottom band 1080x608 (full 16:9 frame).
BROADCAST_TOP_W = 1080
BROADCAST_TOP_H = 1312
BROADCAST_BOT_H = 1920 - BROADCAST_TOP_H          # 608 — output height minus top band


def broadcast_crop_width(src_w, src_h):
    """Top-band crop width for the broadcast layout: 1080x1312 aspect, even, within the frame."""
    return crop_width(src_w, src_h, BROADCAST_TOP_W, BROADCAST_TOP_H)


def hudband_crop_width(src_w, src_h, top_h, effective_src_h=None):
    """Top-band crop width for the hudband layout: 1080 wide x top_h high, even, within frame.
    When `effective_src_h` is passed (source rows actually sampled after strip removal), the crop
    width shrinks to keep the 1080:top_h aspect."""
    h = int(effective_src_h) if effective_src_h else int(src_h)
    return crop_width(src_w, h, 1080, max(1, int(top_h)))


def _primary(faces, ref_cx):
    """Largest face; near-equal sizes are resolved by continuity with ref_cx."""
    if not faces:
        return None
    top = max(f["h"] for f in faces)
    tied = [f for f in faces if f["h"] >= TIE_TOL * top]
    if ref_cx is None or len(tied) == 1:
        return max(tied, key=lambda f: f["h"])
    return min(tied, key=lambda f: abs(f["cx"] - ref_cx))


def _intervals(shots, dur):
    cuts = sorted({float(t) for t in shots if 0 < t < dur})
    edges = [0.0] + cuts + [float(dur)]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def track_crops(samples, shots, dur, src_w, src_h, cw=None):
    """Per-shot locked crops [{"t0", "t1", "x", "cut"}] covering [0, dur]; "cut" is true when the
    segment starts on a camera cut or an in-shot subject switch (switch hard), false for a drift
    re-lock (ease). Default width is 9:16 (`crop_width`); pass `cw` for other bands (broadcast top)."""
    if cw is None:
        cw = crop_width(src_w, src_h)
    samples = sorted(samples, key=lambda s: s["t"])
    spans = _intervals(shots, dur)
    segs, ref = [], None
    for k, (t0, t1) in enumerate(spans):
        last = k == len(spans) - 1
        pts = []
        for s in samples:
            if t0 <= s["t"] < t1 or (last and s["t"] >= t1):
                f = _primary(s.get("faces") or [], ref)
                if f:
                    pts.append((s["t"], f["cx"]))
        cxs = [cx for _, cx in pts]
        if t1 - t0 > DRIFT_MIN_SHOT and cxs and max(cxs) - min(cxs) > DRIFT_MAX:
            a = t0
            while a < t1:
                b = min(a + DRIFT_STEP, t1)
                sub = [cx for t, cx in pts if a <= t < b or (b == t1 and t >= t1)]
                segs.append([a, b, statistics.median(sub) if sub else None, a == t0])
                a = b
        else:
            segs.append([t0, t1, statistics.median(cxs) if cxs else None, True])
        known = [s[2] for s in segs if s[2] is not None]
        if known:
            ref = known[-1]
    # face-less segments inherit the previous segment's centre, else the next one's
    prev = None
    for s in segs:
        if s[2] is None:
            s[2] = prev
        prev = s[2]
    nxt = None
    for s in reversed(segs):
        if s[2] is None:
            s[2] = nxt
        nxt = s[2]
    out = []
    for t0, t1, cx, cut in segs:
        x = int(round(min(max((0.5 if cx is None else cx) * src_w - cw / 2, 0), src_w - cw)))
        if out and abs(x - out[-1]["x"]) < MERGE_TOL * src_w:
            out[-1]["t1"] = round(t1, 3)
        else:
            cut = cut or not out or abs(x - out[-1]["x"]) > SWITCH_MIN * src_w
            out.append({"t0": round(t0, 3), "t1": round(t1, 3), "x": x, "cut": cut})
    return out


def largest_face_track_crops(samples, shots, dur, src_w, src_h, cw=None):
    """Fallback: single-face-per-shot crop plan.

    For every shot span, pick the ONE face track that is both (a) most visible
    and (b) tallest on average, and lock the crop at that track's median cx.
    This is the rule for two-shot wide frames where splitting the difference
    would cut both faces by the frame edge — center ONE full face instead.

    Returns `[{"t0","t1","x","cut"}]` covering [0, dur]. Face-less spans inherit
    the neighbour crop. Shots whose chosen track has <2 samples fall back to the
    source-centre crop so the gate check won't see an off-screen x.
    """
    if cw is None:
        cw = crop_width(src_w, src_h)
    samples = sorted(samples, key=lambda s: s["t"])
    spans = _intervals(shots, dur)
    segs = []
    for k, (t0, t1) in enumerate(spans):
        last = k == len(spans) - 1
        shot_samples = [s for s in samples
                        if t0 <= s["t"] < t1 or (last and s["t"] >= t1)]
        # tally per-track (count, h_sum) within this shot
        per_tid = {}
        for s in shot_samples:
            for f in s.get("faces") or []:
                tid = f.get("track_id", id(f))
                d = per_tid.setdefault(tid, {"n": 0, "h_sum": 0.0, "cx_vals": []})
                d["n"] += 1
                d["h_sum"] += float(f.get("h", 0.0))
                d["cx_vals"].append(float(f["cx"]))
        if not per_tid:
            segs.append([t0, t1, None, True])
            continue
        # score: visibility weighted by average face height
        def score(d):
            return d["n"] * (d["h_sum"] / max(1, d["n"]))
        best_tid, best_d = max(per_tid.items(), key=lambda kv: score(kv[1]))
        if best_d["n"] < 2:
            segs.append([t0, t1, None, True])
            continue
        median_cx = statistics.median(best_d["cx_vals"])
        segs.append([t0, t1, median_cx, True])
    # inherit None cx values from neighbours
    prev = None
    for s in segs:
        if s[2] is None:
            s[2] = prev
        prev = s[2]
    nxt = None
    for s in reversed(segs):
        if s[2] is None:
            s[2] = nxt
        nxt = s[2]
    out = []
    for t0, t1, cx, cut in segs:
        cx = 0.5 if cx is None else cx
        x = int(round(min(max(cx * src_w - cw / 2, 0), src_w - cw)))
        if out and abs(x - out[-1]["x"]) < MERGE_TOL * src_w:
            out[-1]["t1"] = round(t1, 3)
        else:
            cut = cut or not out or abs(x - out[-1]["x"]) > SWITCH_MIN * src_w
            out.append({"t0": round(t0, 3), "t1": round(t1, 3), "x": x, "cut": cut})
    return out


def split_crops(samples, src_w, src_h):
    """Two 9:8 face crops {"left": {w,h,x,y}, "right": {...}} for a 1080x960 band each."""
    pairs = [p for p in (_pair(s.get("faces")) for s in samples) if p]
    if pairs:
        seats = [[f[i] for f in pairs] for i in (0, 1)]
    else:   # no clean two-speaker samples: fall back to the frame halves
        seats = [[{"cx": 0.25, "cy": 0.45, "h": 0.2}], [{"cx": 0.75, "cy": 0.45, "h": 0.2}]]
    face_h = max(statistics.median(f["h"] for f in seat) for seat in seats) * src_h
    ch = min(float(src_h), max(face_h * 3.5, 0.55 * src_h))
    cw = ch * 9 / 8
    if cw > src_w:
        cw, ch = float(src_w), src_w * 8 / 9
    cw, ch = int(cw) - int(cw) % 2, int(ch) - int(ch) % 2
    out = {}
    for name, seat in zip(("left", "right"), seats):
        cx = statistics.median(f["cx"] for f in seat) * src_w
        cy = statistics.median(f["cy"] for f in seat) * src_h
        x = int(round(min(max(cx - cw / 2, 0), src_w - cw)))
        y = int(round(min(max(cy - 0.42 * ch, 0), src_h - ch)))   # face a bit above centre
        out[name] = {"w": cw, "h": ch, "x": x, "y": y}
    return out


def _num(v):
    return f"{float(v):.3f}".rstrip("0").rstrip(".")


def crop_x_expr(segments):
    """ffmpeg crop x expression over clip time. A segment starting on a camera cut switches
    hard (the source edit hides it); an in-shot re-lock ("cut": false) eases linearly over
    PAN_S, clamped to the shorter neighbouring segment so ramps never overlap.
    """
    if not segments:
        return "0"
    expr = str(int(segments[-1]["x"]))
    for s, nxt in reversed(list(zip(segments, segments[1:]))):
        t1, x0, x1 = s["t1"], int(s["x"]), int(nxt["x"])
        ramp = 0 if nxt.get("cut", True) or x0 == x1 else \
            min(PAN_S, t1 - s["t0"], nxt["t1"] - t1)
        if ramp <= 0:
            expr = f"if(lt(t,{_num(t1)}),{x0},{expr})"
            continue
        t0r = t1 - ramp
        lerp = f"({x0}+({x1}-{x0})*(t-{_num(t0r)})/{_num(ramp)})"
        expr = f"if(lt(t,{_num(t0r)}),{x0},if(lt(t,{_num(t1)}),{lerp},{expr}))"
    return expr


# ---------------------------------------------------------------- entry point

def _speaker_cfg():
    """render.speaker block from the config; {} when config is missing / unreadable."""
    try:
        return (config.load().get("render") or {}).get("speaker") or {}
    except (OSError, ValueError):
        return {}


def _tied_shots(samples, shots, dur, tie_tol=TIE_TOL, min_faces=2):
    """Shot spans [(t0, t1)] where the MAJORITY of samples show >= `min_faces` faces at
    comparable size (each face within `tie_tol` of the biggest). These are the shots where
    _primary picks arbitrarily and the active-speaker path should take over."""
    spans = _intervals(shots, dur)
    out = []
    for (t0, t1) in spans:
        tied, total = 0, 0
        for s in samples:
            if not (t0 <= s["t"] < t1):
                continue
            total += 1
            faces = s.get("faces") or []
            if not faces:
                continue
            top = max(f["h"] for f in faces)
            tied += len([f for f in faces if f["h"] >= tie_tol * top]) >= min_faces
        if total and tied * 2 >= total:
            out.append((t0, t1))
    return out


def _best_in_frame_cx(speaker_segs, t):
    """Return the most recent speaker_seg['cx'] whose track_id was in frame BEFORE
    time t — the "last known position" for a hold. None when there is no prior
    speaker crop at all."""
    best = None
    for s in speaker_segs:
        if s["t0"] <= t:
            best = s["cx"]
    return best


def _speaker_dead_zone_relocks(seg_t0, seg_t1, cur_cx, track_id, samples, dead_zone):
    """within one speaker span [seg_t0, seg_t1] holding crop centre `cur_cx`,
    find times where the speaker's face drifts outside the dead zone (|face_cx - cur_cx|
    > dead_zone) and emit a smooth re-lock to the new median. Returns a list of
    sub-segments [{"t0","t1","cx","cut":False,...}] starting from `seg_t0`; the first
    always starts at `seg_t0` with `cur_cx` and `cut=True` (the caller's intent).

    Smooth re-lock means `cut=False` -> crop_x_expr eases the move over PAN_S."""
    # pick samples that belong to this speaker AND fall in the span
    pts = []
    for s in samples:
        if not (seg_t0 <= s["t"] <= seg_t1):
            continue
        for f in s.get("faces") or []:
            if f.get("track_id") == track_id:
                pts.append((s["t"], f["cx"]))
                break
    if not pts:
        return [{"t0": round(seg_t0, 3), "t1": round(seg_t1, 3), "cx": cur_cx,
                 "cut": True, "speaker": track_id}]
    # walk through time; whenever the face drifts outside the dead zone,
    # re-lock to the median of the next WINDOW seconds. Hold the new cx until a new
    # drift. Minimum re-lock gap PAN_S so successive relocks don't overlap.
    out = [{"t0": round(seg_t0, 3), "t1": round(seg_t1, 3), "cx": float(cur_cx),
            "cut": True, "speaker": track_id}]
    WINDOW = 2.0        # look-ahead for the median
    last_lock_t = seg_t0
    cur = float(cur_cx)
    for (t, cx) in pts:
        if t - last_lock_t < PAN_S + 0.4:
            continue
        if abs(cx - cur) <= dead_zone:
            continue
        # drift confirmed — target = median cx over the next WINDOW seconds, same speaker
        future = [c for (tt, c) in pts if t <= tt < t + WINDOW]
        if not future:
            continue
        new_cx = float(statistics.median(future))
        if abs(new_cx - cur) <= dead_zone:
            continue       # drift was transient
        # Shorten the previous sub-seg to end at t, start a new one at t with new_cx.
        out[-1]["t1"] = round(t, 3)
        out.append({"t0": round(t, 3), "t1": round(seg_t1, 3), "cx": new_cx,
                    "cut": False, "speaker": track_id, "relock": True})
        cur = new_cx
        last_lock_t = t
    return out


def _speaker_face_visible_in(samples, track_id, t0, t1):
    """True when the speaker's face is detected in >=1 sample of [t0, t1]."""
    if samples is None:
        return True
    for s in samples:
        if not (t0 <= s["t"] <= t1):
            continue
        for f in s.get("faces") or []:
            if f.get("track_id") == track_id:
                return True
    return False


def _speaker_off_frame_gaps(samples, track_id, t0, t1, min_gap=1.0):
    """[(gap_t0, gap_t1)] where the talker's face is NOT in frame for longer
    than `min_gap` seconds, inside [t0, t1]. Uses sample times as boundaries
    and extends gaps to the enclosing silence."""
    if samples is None:
        return []
    in_range = [s for s in samples if t0 <= s["t"] <= t1]
    if not in_range:
        return [(t0, t1)] if t1 - t0 >= min_gap else []
    gaps = []
    gap_start = None
    for s in in_range:
        has = any(f.get("track_id") == track_id for f in (s.get("faces") or []))
        if has:
            if gap_start is not None:
                if s["t"] - gap_start >= min_gap:
                    gaps.append((gap_start, s["t"]))
                gap_start = None
        else:
            if gap_start is None:
                gap_start = s["t"]
    if gap_start is not None:
        if t1 - gap_start >= min_gap:
            gaps.append((gap_start, t1))
    return gaps


def _dominant_face_cx_in(samples, t0, t1, exclude_track_id=None):
    """when the diarized talker's face is off-frame, pick the dominant
    OTHER face's median cx over this window — the reaction/listener face.
    Returns None when no face is visible in the window at all.

    Range is HALF-OPEN [t0, t1) so a face sample sitting on the gap's trailing
    boundary (the first post-gap sample where the talker reappears) is NOT
    counted — otherwise b-roll windows with no faces inside would still pick
    up the next sample's face and skip the hold path."""
    pos = []
    for s in samples or []:
        if not (t0 <= s["t"] < t1):
            continue
        for f in s.get("faces") or []:
            if exclude_track_id is not None and f.get("track_id") == exclude_track_id:
                continue
            pos.append((float(f["cx"]), float(f.get("h", 0.1))))
    if not pos:
        return None
    hs = [h for _, h in pos]
    tot = sum(hs) or 1.0
    return sum(cx * h for cx, h in pos) / tot


def _split_around_gaps(t0, t1, gaps):
    """[(t0, t1, is_gap)] walking [t0, t1] with `gaps` carved out as off-frame."""
    if not gaps:
        return [(t0, t1, False)]
    out = []
    cursor = t0
    for (gs, ge) in sorted(gaps):
        gs = max(gs, t0); ge = min(ge, t1)
        if ge <= cursor:
            continue
        if gs > cursor:
            out.append((cursor, gs, False))
        out.append((max(gs, cursor), ge, True))
        cursor = ge
    if cursor < t1:
        out.append((cursor, t1, False))
    return out


# ---------------------------------------------------------------- zero-phase tracked crop

SMOOTH_SAMPLE_FPS = 10.0       # resample / emit per 100 ms
SMOOTH_SIGMA_S = 0.35          # zero-phase gaussian smoothing window (~1 Hz cutoff)
SMOOTH_MAX_VEL_PCT = 0.012     # <=1.2% of src width per 100 ms step (well under gate's 1.5%)


def adjust_crop_for_overlay(x_px, cw, overlay_box, src_w):
    """adjust a crop position so a burned-in source overlay band is either fully inside the
    crop or fully outside. ``overlay_box`` is ``(x0, x1)`` in source pixels. Returns the new
    crop top-left x (clamped to [0, src_w - cw]).

    Rule: never leave the overlay half-cropped. If the overlay fits inside the crop width,
    shift the crop until the overlay is entirely contained; otherwise shift the crop so the
    overlay is entirely outside the crop. When both options are equally far from the subject
    centre, exclude the overlay (the common case: the overlay is a lower-third / corner band
    that the viewer does not need in frame)."""
    x_px = float(x_px); cw = int(cw)
    if overlay_box is None:
        return int(round(min(max(x_px, 0), max(0, src_w - cw))))
    ox0, ox1 = sorted((float(overlay_box[0]), float(overlay_box[1])))
    # Already clean?
    if ox1 <= x_px or ox0 >= x_px + cw:
        return int(round(min(max(x_px, 0), max(0, src_w - cw))))     # fully outside
    if ox0 >= x_px and ox1 <= x_px + cw:
        return int(round(min(max(x_px, 0), max(0, src_w - cw))))     # fully inside
    # Overlay straddles the crop boundary — compute the two candidate x values.
    ow = ox1 - ox0
    cands = []
    if ow <= cw:
        # Contain: crop can swallow the overlay whole
        contain_lo = max(0.0, ox1 - cw)
        contain_hi = min(float(src_w - cw), ox0)
        if contain_lo <= contain_hi:
            # pick the containment position closest to x_px
            cands.append(min(max(x_px, contain_lo), contain_hi))
    # Exclude: push crop to the left or to the right of the overlay
    if ox0 - cw >= 0:
        cands.append(float(ox0 - cw))                                # exclude on the right
    if ox1 <= src_w - cw:
        cands.append(float(ox1))                                     # exclude on the left
    if not cands:
        return int(round(min(max(x_px, 0), max(0, src_w - cw))))     # nothing fits; keep as-is
    best = min(cands, key=lambda v: abs(v - x_px))
    return int(round(min(max(best, 0), max(0, src_w - cw))))


def _zero_phase_gaussian(xs, sigma):
    """Symmetric gaussian smoothing — zero phase by construction (no lag).

    `sigma` is in samples. Pads with edge replication so endpoints are not pulled
    toward zero. Returns a float list the same length as `xs`."""
    import math
    xs = [float(x) for x in xs]
    if sigma <= 0 or len(xs) < 3:
        return xs
    radius = max(1, int(round(3.0 * sigma)))
    kernel = [math.exp(-(k * k) / (2.0 * sigma * sigma)) for k in range(-radius, radius + 1)]
    s = sum(kernel) or 1.0
    kernel = [k / s for k in kernel]
    n = len(xs)
    out = []
    for i in range(n):
        acc = 0.0
        for k, w in enumerate(kernel):
            j = i + (k - radius)
            if j < 0:
                j = 0
            elif j >= n:
                j = n - 1
            acc += w * xs[j]
        out.append(acc)
    return out


def _clamp_velocity_px(xs, max_step_px):
    """Clamp |xs[i] - xs[i-1]| <= max_step_px via a symmetric forward + backward pass.
    This bounds the max per-sample move without introducing phase lag (both passes keep
    the trajectory bounded). Returns a float list."""
    if max_step_px <= 0 or len(xs) < 2:
        return [float(x) for x in xs]
    out = [float(xs[0])]
    for v in xs[1:]:
        d = float(v) - out[-1]
        if d > max_step_px:
            out.append(out[-1] + max_step_px)
        elif d < -max_step_px:
            out.append(out[-1] - max_step_px)
        else:
            out.append(float(v))
    # backward pass smooths deceleration at the tail
    for i in range(len(out) - 2, -1, -1):
        d = out[i] - out[i + 1]
        if d > max_step_px:
            out[i] = out[i + 1] + max_step_px
        elif d < -max_step_px:
            out[i] = out[i + 1] - max_step_px
    return out


def _interp_face_track_cx(samples, track_id, grid, fallback_cx):
    """Linearly interpolate the face center cx of `track_id` at each time in `grid`.
    Returns a list of cx values (normalized [0,1]); gaps outside the detected range
    extend the nearest endpoint, and times beyond all detections use `fallback_cx`."""
    pts = []
    for s in samples or []:
        for f in s.get("faces") or []:
            if f.get("track_id") == track_id:
                pts.append((float(s["t"]), float(f["cx"])))
                break
    if not pts:
        return [float(fallback_cx)] * len(grid)
    pts.sort()
    ts = [p[0] for p in pts]
    cxs = [p[1] for p in pts]
    out = []
    j = 0
    for g in grid:
        if g <= ts[0]:
            out.append(cxs[0]); continue
        if g >= ts[-1]:
            out.append(cxs[-1]); continue
        while j + 1 < len(ts) and ts[j + 1] < g:
            j += 1
        t0, t1 = ts[j], ts[j + 1]
        if t1 == t0:
            out.append(cxs[j])
        else:
            a = (g - t0) / (t1 - t0)
            out.append(cxs[j] + a * (cxs[j + 1] - cxs[j]))
    return out


def _interp_dominant_other_cx(samples, exclude_track_id, grid, fallback_cx):
    """Per-grid-point: the dominant (height-weighted) cx of the biggest OTHER face.
    When no other face is visible at a grid point, extend the previous grid value
    (or fallback_cx when nothing yet). Used for reaction-cutaway tracking."""
    per_t = {}
    for s in samples or []:
        bucket = []
        for f in s.get("faces") or []:
            if f.get("track_id") == exclude_track_id:
                continue
            bucket.append((float(f["cx"]), float(f.get("h", 0.1))))
        if bucket:
            tot_h = sum(h for _, h in bucket) or 1.0
            cx = sum(cx * h for cx, h in bucket) / tot_h
            per_t[round(float(s["t"]), 3)] = cx
    if not per_t:
        return [float(fallback_cx)] * len(grid)
    ts = sorted(per_t)
    out = []
    last = None
    j = 0
    for g in grid:
        if g <= ts[0]:
            out.append(per_t[ts[0]])
            last = per_t[ts[0]]
            continue
        if g >= ts[-1]:
            out.append(per_t[ts[-1]])
            last = per_t[ts[-1]]
            continue
        while j + 1 < len(ts) and ts[j + 1] < g:
            j += 1
        t0, t1 = ts[j], ts[j + 1]
        a = (g - t0) / max(1e-9, (t1 - t0))
        v = per_t[t0] + a * (per_t[t1] - per_t[t0])
        out.append(v); last = v
    return out


def _smoothed_segments_from_cx(grid, cxs_norm, src_w, cw,
                                tag_speaker=None, is_cutaway=False, is_hold=False):
    """Build piecewise-linear segments from a `grid` of times and smoothed cx values.
    Emits len(grid)-1 segments with `cut=False` so crop_x_expr renders a continuous ramp.
    All x values are integer source pixels, clamped to [0, src_w - cw]."""
    if len(grid) < 2:
        return []
    xs_px = [min(max(cx * src_w - cw / 2.0, 0.0), float(src_w - cw)) for cx in cxs_norm]
    out = []
    for i in range(len(grid) - 1):
        rec = {"t0": round(float(grid[i]), 3), "t1": round(float(grid[i + 1]), 3),
               "x": int(round(xs_px[i])),
               "cut": False, "smoothed": True}
        if tag_speaker is not None:
            rec["speaker"] = tag_speaker
        if is_cutaway:
            rec["reaction_cutaway"] = True
        if is_hold:
            rec["off_frame_hold"] = True
        out.append(rec)
    # Append a tail anchor so the trajectory reaches xs_px[-1] at grid[-1] in crop_x_expr.
    # crop_x_expr emits a ramp into the FINAL segment's x value only via its previous
    # segment's nxt.x. A one-tick anchor at grid[-1] with the terminal x value gives us
    # that final ramp without changing semantics after the shot end.
    anchor_t1 = round(float(grid[-1]) + 0.01, 3)
    rec = {"t0": round(float(grid[-1]), 3), "t1": anchor_t1,
           "x": int(round(xs_px[-1])),
           "cut": False, "smoothed": True, "anchor": True}
    if tag_speaker is not None:
        rec["speaker"] = tag_speaker
    out.append(rec)
    return out


def _smooth_trajectory_px(cxs_norm, src_w, sigma, max_step_px):
    """Full smoothing pipeline on normalized cx values:
    zero-phase gaussian -> velocity clamp. Returns smoothed normalized cx list."""
    # work in source pixels for clamp, then convert back to normalized
    xs_px = [cx * float(src_w) for cx in cxs_norm]
    xs_px = _zero_phase_gaussian(xs_px, sigma)
    xs_px = _clamp_velocity_px(xs_px, max_step_px)
    return [x / float(src_w) for x in xs_px]


def _pick_shot_target_track(shot_samples, global_primary_tid, speaker_segs=None,
                             shot_t0=None, shot_t1=None):
    """Pick which face track to center on for this shot.
    Preference order:
      1. If `speaker_segs` cover this shot and the diarized talker's track_id is
         visible in the shot, pick that (= "the person who's actually talking").
      2. Else, pick `global_primary_tid` when visible in the shot.
      3. Else, pick the track with the highest shot-local visibility score.
    This fixes the handheld two-shot case where the taller / closer face is the
    host and the diarized talker is the smaller face at the frame edge."""
    tally = {}
    for s in shot_samples:
        for f in s.get("faces") or []:
            tid = f.get("track_id")
            if tid is None:
                continue
            tally[tid] = tally.get(tid, 0.0) + float(f.get("h", 0.0))
    if not tally:
        return None, "none"
    # 1. speaker-driven target
    if speaker_segs and shot_t0 is not None and shot_t1 is not None:
        voter = {}
        for sp in speaker_segs:
            lo = max(float(sp["t0"]), float(shot_t0))
            hi = min(float(sp["t1"]), float(shot_t1))
            if hi > lo and sp.get("track_id") in tally:
                voter[sp["track_id"]] = voter.get(sp["track_id"], 0.0) + (hi - lo)
        if voter:
            tid = max(voter.items(), key=lambda kv: kv[1])[0]
            return tid, "speaker"
    # 2. global primary
    if global_primary_tid is not None and tally.get(global_primary_tid, 0.0) > 0:
        return global_primary_tid, "global_primary"
    # 3. shot-local biggest
    return max(tally.items(), key=lambda kv: kv[1])[0], "shot_dominant"


def smooth_tracked_crops(samples, shots, dur, src_w, src_h, cw=None,
                          fps=SMOOTH_SAMPLE_FPS, sigma_s=SMOOTH_SIGMA_S,
                          max_vel_pct=SMOOTH_MAX_VEL_PCT,
                          speaker_segs=None):
    """zero-phase smoothed crop plan for the full window.

    Per shot: pick the dominant face at each sample (largest, with continuity bias),
    interpolate missing samples, apply a symmetric gaussian low-pass (zero-phase =
    no lag), clamp per-100ms velocity. Hard cuts live at shot boundaries only.
    `samples` must be the dense 10 fps samples from speaker.active_speaker's I/O
    path (with faces[].track_id already assigned)."""
    if cw is None:
        cw = crop_width(src_w, src_h)
    spans = _intervals(shots, dur)
    sigma = sigma_s * fps
    max_step_px = max_vel_pct * float(src_w)
    # Global primary subject: largest aggregate height across the clip. In handheld
    # vlog footage this reliably picks the main speaker; title/B-roll faces
    # are smaller and don't dominate the aggregate.
    global_tally = {}
    for s in samples:
        for f in s.get("faces") or []:
            tid = f.get("track_id")
            if tid is None:
                continue
            global_tally[tid] = global_tally.get(tid, 0.0) + float(f.get("h", 0.0))
    global_primary_tid = (max(global_tally.items(), key=lambda kv: kv[1])[0]
                            if global_tally else None)
    out = []
    for k, (t0, t1) in enumerate(spans):
        last = k == len(spans) - 1
        shot_samples = [s for s in samples
                        if t0 <= s["t"] < t1 or (last and s["t"] >= t0)]
        if not shot_samples:
            # fall back to source centre for this span
            x = int(round((src_w - cw) / 2.0))
            out.append({"t0": round(t0, 3), "t1": round(t1, 3),
                        "x": x, "cut": True, "smoothed": True,
                        "note": "no_samples_in_shot"})
            continue
        target_tid, target_reason = _pick_shot_target_track(
            shot_samples, global_primary_tid,
            speaker_segs=speaker_segs, shot_t0=t0, shot_t1=t1)
        # Build raw cx per sample: ONLY from the target track. Falls back to
        # _primary continuity when the target track is missing in a sample
        # (e.g. briefly occluded).
        raw = []
        ref = None
        for s in shot_samples:
            face = None
            for f in s.get("faces") or []:
                if f.get("track_id") == target_tid:
                    face = f; break
            if face is None:
                face = _primary(s.get("faces") or [], ref)
            if face is None:
                raw.append((s["t"], None))
            else:
                raw.append((s["t"], float(face["cx"])))
                ref = face["cx"]
        # Fill gaps: nearest-non-None interpolation
        times = [t for t, _ in raw]
        vals = [v for _, v in raw]
        known_idx = [i for i, v in enumerate(vals) if v is not None]
        if not known_idx:
            x = int(round((src_w - cw) / 2.0))
            out.append({"t0": round(t0, 3), "t1": round(t1, 3),
                        "x": x, "cut": True, "smoothed": True,
                        "note": "no_face_in_shot"})
            continue
        # fill leading / trailing
        for i in range(known_idx[0]):
            vals[i] = vals[known_idx[0]]
        for i in range(known_idx[-1] + 1, len(vals)):
            vals[i] = vals[known_idx[-1]]
        # linear interp for interior gaps
        i = 0
        while i < len(vals):
            if vals[i] is not None:
                i += 1; continue
            j = i
            while j < len(vals) and vals[j] is None:
                j += 1
            # interpolate [i, j)
            lo, hi = vals[i - 1], vals[j]
            tlo, thi = times[i - 1], times[j]
            for k2 in range(i, j):
                a = (times[k2] - tlo) / max(1e-9, (thi - tlo))
                vals[k2] = lo + a * (hi - lo)
            i = j
        # Build a uniform grid at `fps` across the shot so output samples align
        # with the gate (10 fps). Interpolate vals at the grid.
        span = max(0.0, float(t1) - float(t0))
        n = max(2, int(round(span * fps)) + 1)
        grid = [t0 + (t1 - t0) * i / (n - 1) for i in range(n)]
        g_vals = []
        j = 0
        for g in grid:
            if g <= times[0]:
                g_vals.append(vals[0]); continue
            if g >= times[-1]:
                g_vals.append(vals[-1]); continue
            while j + 1 < len(times) and times[j + 1] < g:
                j += 1
            a = (g - times[j]) / max(1e-9, (times[j + 1] - times[j]))
            g_vals.append(vals[j] + a * (vals[j + 1] - vals[j]))
        # Zero-phase smoothing in pixel space, then clamp velocity
        cx_px = [v * src_w for v in g_vals]
        cx_px = _zero_phase_gaussian(cx_px, sigma)
        # Convert cx (face centre px) -> crop x (top-left of crop band)
        x_px = [min(max(v - cw / 2.0, 0.0), float(src_w - cw)) for v in cx_px]
        x_px = _clamp_velocity_px(x_px, max_step_px)
        # Emit segments
        for i2 in range(n - 1):
            rec = {"t0": round(float(grid[i2]), 3),
                   "t1": round(float(grid[i2 + 1]), 3),
                   "x": int(round(x_px[i2])),
                   "cut": False, "smoothed": True,
                   "target_tid": target_tid, "target_reason": target_reason}
            if i2 == 0:
                rec["cut"] = True  # shot entry = hard cut
            out.append(rec)
        # Final anchor so the trajectory lands at x_px[-1] at grid[-1]
        tail_t1 = round(float(grid[-1]) + 0.01, 3)
        out.append({"t0": round(float(grid[-1]), 3), "t1": tail_t1,
                    "x": int(round(x_px[-1])),
                    "cut": False, "smoothed": True, "anchor": True,
                    "target_tid": target_tid, "target_reason": target_reason})
    return out


def _smoothed_track_for_speaker(samples, track_id, t0, t1, src_w, cw,
                                 fallback_cx=0.5,
                                 fps=SMOOTH_SAMPLE_FPS, sigma_s=SMOOTH_SIGMA_S,
                                 max_vel_pct=SMOOTH_MAX_VEL_PCT):
    """zero-phase smoothed per-100ms crop trajectory over [t0, t1] tracking
    `track_id`'s face. Returns piecewise-linear segments ready for crop_x_expr."""
    span = max(0.0, float(t1) - float(t0))
    if span <= 1.0 / fps:
        return [{"t0": round(t0, 3), "t1": round(t1, 3),
                 "x": int(round(min(max(fallback_cx * src_w - cw / 2.0, 0.0),
                                     float(src_w - cw)))),
                 "cut": True, "speaker": track_id, "smoothed": True}]
    n = max(2, int(round(span * fps)) + 1)
    grid = [t0 + (t1 - t0) * i / (n - 1) for i in range(n)]
    cxs = _interp_face_track_cx(samples, track_id, grid, fallback_cx=fallback_cx)
    sigma = sigma_s * fps
    max_step_px = max_vel_pct * float(src_w)
    cxs_s = _smooth_trajectory_px(cxs, src_w, sigma, max_step_px)
    segs = _smoothed_segments_from_cx(grid, cxs_s, src_w, cw, tag_speaker=track_id)
    # make the FIRST emitted segment a hard cut: this is a speaker-switch / shot entry
    if segs:
        segs[0]["cut"] = True
    return segs


def _smoothed_track_for_cutaway(samples, exclude_track_id, t0, t1, src_w, cw,
                                 fallback_cx=0.5, tag_speaker=None,
                                 fps=SMOOTH_SAMPLE_FPS, sigma_s=SMOOTH_SIGMA_S,
                                 max_vel_pct=SMOOTH_MAX_VEL_PCT):
    """zero-phase smoothed cutaway trajectory on the dominant OTHER face."""
    span = max(0.0, float(t1) - float(t0))
    if span <= 1.0 / fps:
        return [{"t0": round(t0, 3), "t1": round(t1, 3),
                 "x": int(round(min(max(fallback_cx * src_w - cw / 2.0, 0.0),
                                     float(src_w - cw)))),
                 "cut": False, "reaction_cutaway": True, "smoothed": True,
                 **({"speaker": tag_speaker} if tag_speaker is not None else {})}]
    n = max(2, int(round(span * fps)) + 1)
    grid = [t0 + (t1 - t0) * i / (n - 1) for i in range(n)]
    cxs = _interp_dominant_other_cx(samples, exclude_track_id, grid, fallback_cx=fallback_cx)
    sigma = sigma_s * fps
    max_step_px = max_vel_pct * float(src_w)
    cxs_s = _smooth_trajectory_px(cxs, src_w, sigma, max_step_px)
    segs = _smoothed_segments_from_cx(grid, cxs_s, src_w, cw,
                                       tag_speaker=tag_speaker, is_cutaway=True)
    if segs:
        segs[0]["cut"] = False
    return segs


def _clip_segments_by_shots(segs, shots):
    """Insert `cut=True` boundaries at any shot boundary that falls strictly inside
    one of our smoothed segments: the camera edit is a HARD cut (no smoothing across).
    Walks `segs` in order; when a shot time falls inside a segment, splits it there
    and marks the following piece as `cut=True`."""
    if not shots or not segs:
        return list(segs)
    cuts = sorted({round(float(t), 3) for t in shots})
    out = []
    for s in segs:
        a, b = float(s["t0"]), float(s["t1"])
        interior = [c for c in cuts if a < c < b]
        if not interior:
            out.append(s); continue
        cursor = a
        for c in interior:
            piece = dict(s); piece["t0"] = round(cursor, 3); piece["t1"] = round(c, 3)
            out.append(piece)
            cursor = c
        piece = dict(s); piece["t0"] = round(cursor, 3); piece["t1"] = round(b, 3)
        piece["cut"] = True
        out.append(piece)
    return out


def _speaker_replace(segments, tied_spans, speaker_segs, src_w, cw,
                     samples=None, dead_zone=SPEAKER_DEAD_ZONE, smooth_track=False,
                     shots=None):
    """Replace parts of `track_crops` segments that overlap any `tied_spans` with
    `speaker_segs`. Speaker switches are hard cuts (`cut=True`). `speaker_segs` =
    [{"t0","t1","cx","track_id"}] in clip-relative seconds. Consecutive same-speaker
    speaker_segs are merged (no new cut on the same person). Inside a span,
    the speaker's face can drift beyond the dead zone — the merged span is split
    into a re-lock sub-seg with `cut=False` so crop_x_expr eases, no snap.

    when the diarized talker's face is NOT visible during a sub-span
    (b-roll, cutaway), hold the previous crop x instead of jumping to the
    speaker's stale median — this fixes the "empty couch" framing in
    vi-city_0015_01_4 7-12 s.
    """
    def _speaker_x(cx):
        return int(round(min(max(cx * src_w - cw / 2, 0), src_w - cw)))
    # coalesce consecutive same-track_id speaker segments (defensive: pick_speakers
    # already does this, but a swallow-shaped time range inside a tied span may feed
    # non-contiguous speaker segments here).
    sp = []
    for s in sorted(speaker_segs, key=lambda s: s["t0"]):
        if sp and sp[-1].get("track_id") == s.get("track_id") and abs(s["t0"] - sp[-1]["t1"]) < 0.1:
            sp[-1]["t1"] = s["t1"]
            # re-median cx when samples are available later; keep the earliest cx for now
        else:
            sp.append(dict(s))
    speaker_segs = sp
    out = []
    for seg in segments:
        overlaps = [(max(seg["t0"], t0), min(seg["t1"], t1))
                    for (t0, t1) in tied_spans
                    if min(seg["t1"], t1) > max(seg["t0"], t0)]
        if not overlaps:
            out.append(seg)
            continue
        # merge adjacent/overlapping tied ranges inside this seg
        overlaps.sort()
        merged = [overlaps[0]]
        for a, b in overlaps[1:]:
            if a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        # Build the output by walking seg left to right, swapping tied ranges for
        # speaker sub-segments.
        cursor = seg["t0"]
        for (a, b) in merged:
            if a > cursor:
                out.append({**seg, "t0": round(cursor, 3), "t1": round(a, 3)})
            subs = [s for s in speaker_segs if s["t1"] > a and s["t0"] < b]
            if subs:
                for s in subs:
                    sa = max(a, s["t0"]); sb = min(b, s["t1"])
                    if sb <= sa:
                        continue
                    tid = s.get("track_id")
                    # split the sub-span around any extended off-frame
                    # gaps for the talker. The gap pieces become `off_frame_hold`
                    # records that keep the previous crop x; the visible pieces
                    # go through the normal dead-zone / cut path.
                    off_frame_gaps = _speaker_off_frame_gaps(samples, tid, sa, sb,
                                                              min_gap=1.0)
                    piece_spans = _split_around_gaps(sa, sb, off_frame_gaps)
                    for (ps, pe, is_gap) in piece_spans:
                        if is_gap:
                            # during an off-frame gap for the diarized
                            # talker, prefer framing the OTHER face that IS in
                            # frame (reaction shot cutaway) — the rule "never
                            # frame a region without a face unless it is a
                            # deliberate card/product shot" rule. Only if NO
                            # face is visible do we hold the previous x.
                            reaction_cx = _dominant_face_cx_in(samples, ps, pe,
                                                                 exclude_track_id=tid)
                            if reaction_cx is not None:
                                if smooth_track:
                                    # smoothed trajectory on the dominant OTHER face
                                    smoothed = _smoothed_track_for_cutaway(
                                        samples, tid, ps, pe, src_w, cw,
                                        fallback_cx=reaction_cx, tag_speaker=tid)
                                    for rec in _clip_segments_by_shots(smoothed, shots):
                                        out.append(rec)
                                    continue
                                x = _speaker_x(reaction_cx)
                                rec = {"t0": round(ps, 3), "t1": round(pe, 3),
                                       "x": x, "cut": False,
                                       "speaker": tid, "reaction_cutaway": True}
                                out.append(rec)
                                continue
                            hold_x = out[-1]["x"] if out else None
                            if hold_x is None:
                                is_gap = False
                            else:
                                rec = {"t0": round(ps, 3), "t1": round(pe, 3),
                                       "x": hold_x, "cut": False,
                                       "speaker": tid, "off_frame_hold": True}
                                out.append(rec)
                                continue
                        if smooth_track:
                            # zero-phase smoothed trajectory on this speaker's face
                            smoothed = _smoothed_track_for_speaker(
                                samples, tid, ps, pe, src_w, cw,
                                fallback_cx=s["cx"])
                            for rec in _clip_segments_by_shots(smoothed, shots):
                                out.append(rec)
                            continue
                        if samples is not None and (pe - ps) > PAN_S + 0.4:
                            pieces = _speaker_dead_zone_relocks(ps, pe, s["cx"], tid,
                                                                samples, dead_zone)
                        else:
                            pieces = [{"t0": round(ps, 3), "t1": round(pe, 3),
                                       "cx": s["cx"], "cut": True, "speaker": tid}]
                        # coalesce back-to-back same-speaker sub-pieces that didn't trigger
                        # a re-lock (shouldn't happen but be defensive)
                        for p in pieces:
                            x = _speaker_x(p["cx"])
                            rec = {"t0": p["t0"], "t1": p["t1"], "x": x,
                                   "cut": bool(p.get("cut", True)),
                                   "speaker": tid}
                            if p.get("relock"):
                                rec["relock"] = True
                            if (out and out[-1].get("speaker") == tid
                                    and out[-1].get("off_frame_hold") is not True
                                    and out[-1]["x"] == x
                                    and abs(out[-1]["t1"] - rec["t0"]) < 0.01):
                                out[-1]["t1"] = rec["t1"]
                            else:
                                out.append(rec)
            else:
                out.append({**seg, "t0": round(a, 3), "t1": round(b, 3)})
            cursor = b
        if cursor < seg["t1"]:
            out.append({**seg, "t0": round(cursor, 3), "t1": round(seg["t1"], 3)})
    return out


def analyze(src, start, dur, force=None, cw=None, speaker_cfg=None, words=None,
            largest_face_fallback=False, smooth_track=False):
    """Reframe plan for a window: {"layout", "segments" | "split", "src_w", "src_h",
    "face_ratio", "shots"}. `force` ('track'|'split'|'broadcast'|'hudband') skips the layout
    decision. `cw` overrides the top-band crop width (used by hudband where the caller decides
    the top-band height from HUD detection). `speaker_cfg` overrides render.speaker settings
    (None -> use the config file).

    when `words` is passed (word-level timestamps from the ASR sidecar), the
    speaker path uses audio-only diarization + single-face-shot cluster->track mapping
    to decide the active talker. Mouth motion is only a tie-break. When `words` is
    None or diarization fails, we fall back to the mouth-motion planner."""
    src_w, src_h = probe_size(src)
    samples, hists = [], []
    for t, frame in iter_frames(src, start, dur, src_size=(src_w, src_h)):
        samples.append({"t": t, "faces": detect_faces(frame)})
        hists.append((t, _hist(frame)))
    shots = _cut_times(hists)
    layout = force or decide_layout(samples)
    if layout in ("track", "broadcast", "hudband") and shots:
        fps = probe_fps(src)
        shots = [refine_cut(src, start, t, fps, (src_w, src_h)) for t in shots]
    plan = {"layout": layout, "src_w": src_w, "src_h": src_h,
            "face_ratio": round(sum(1 for s in samples if s["faces"]) / len(samples), 3)
            if samples else 0.0,
            "shots": shots, "samples": len(samples),
            # faces while the hook title is up, so it can be placed clear of them
            "hook_faces": [s for s in samples if s["t"] <= HOOK_FACE_WINDOW and s["faces"]]}
    if layout == "track":
        if smooth_track:
            # dense zero-phase smoothed trajectory on the dominant face per
            # shot. Uses the 10 fps native-res face samples (same signal
            # speaker.active_speaker uses) so the crop follows handheld camera
            # motion in real time without the per-shot median drift bug of
            # track_crops / largest_face_track_crops.
            try:
                from clipengine.render import speaker as _speaker
                # Use active_speaker to get per-shot talker ID via mouth motion;
                # it also returns the dense samples with assigned track_ids so we
                # don't need to re-sample.
                scfg = {**_speaker_cfg(), **(speaker_cfg or {})}
                sp_segs_as, dense = [], None
                try:
                    sp_segs_as, dense = _speaker.active_speaker(
                        src, start, dur, cfg=scfg, return_samples=True)
                except Exception as e:
                    plan["smooth_track_speaker_error"] = f"{type(e).__name__}: {e}"
                if not dense:
                    dense = []
                    for t, frame in _speaker.iter_frames_native(src, start, dur,
                                                                 fps=SMOOTH_SAMPLE_FPS):
                        dense.append({"t": t,
                                       "faces": _speaker.detect_faces_with_landmarks(frame)})
                    _speaker.assign_tracks(dense)
                plan["segments"] = smooth_tracked_crops(
                    dense, shots, dur, src_w, src_h,
                    speaker_segs=sp_segs_as)
                plan["smooth_track"] = {
                    "sample_fps": SMOOTH_SAMPLE_FPS,
                    "sigma_s": SMOOTH_SIGMA_S,
                    "max_vel_pct": SMOOTH_MAX_VEL_PCT,
                    "n_dense_samples": len(dense),
                    "speaker_segs_count": len(sp_segs_as or []),
                }
                if sp_segs_as:
                    plan["speaker_segments"] = sp_segs_as
            except Exception as e:
                plan["smooth_track_error"] = f"{type(e).__name__}: {e}"
                plan["segments"] = track_crops(samples, shots, dur, src_w, src_h)
        elif largest_face_fallback:
            plan["segments"] = largest_face_track_crops(samples, shots, dur, src_w, src_h)
            plan["largest_face_fallback"] = True
        else:
            plan["segments"] = track_crops(samples, shots, dur, src_w, src_h)
    elif layout == "broadcast":
        plan["segments"] = track_crops(samples, shots, dur, src_w, src_h,
                                       cw=broadcast_crop_width(src_w, src_h))
    elif layout == "hudband":
        plan["segments"] = track_crops(samples, shots, dur, src_w, src_h,
                                       cw=int(cw) if cw else crop_width(src_w, src_h))
    elif layout == "split":
        plan["split"] = split_crops(samples, src_w, src_h)
    # active-speaker replacement for tied-face track shots (3+ hosts in a wide shot, or
    # any two-face tie inside a track segment). Everything else is untouched.
    # the largest-face fallback bypasses the speaker path entirely — we're
    # trying to pass the frame-level face gate, not frame "the actual talker".
    if layout == "track" and not largest_face_fallback and not smooth_track:
        scfg = {**_speaker_cfg(), **(speaker_cfg or {})}
        if scfg.get("enabled", True):
            tie_tol = float(scfg.get("tie_tol", TIE_TOL))
            min_faces = int(scfg.get("min_faces_multi", 3))
            tied = _tied_shots(samples, shots, dur, tie_tol=tie_tol, min_faces=min_faces)
            if not tied:                          # fall back: any track shot with >= 2 tied faces
                tied = _tied_shots(samples, shots, dur, tie_tol=tie_tol, min_faces=2)
            # when words are available, start whole-window (not only tied shots)
            # because off-tied moments still need the talker's crop. Shots without a
            # proper tie fall through to the same track_crops output.
            use_audio_first = bool(words) and len(samples) > 0
            path_tried = "none"
            if tied or use_audio_first:
                from clipengine.render import speaker as _speaker
                sp_segs, sp_samples = [], None
                diar_meta = None
                if use_audio_first:
                    try:
                        sp_segs, sp_samples = _speaker.active_speaker_audio_first(
                            src, start, dur, words=words, cfg=scfg, return_samples=True)
                        if sp_segs:
                            path_tried = "audio_first"
                            diar_meta = sp_segs[0].pop("_diarization_meta", None)
                    except Exception as e:
                        plan["speaker_audio_first_error"] = f"{type(e).__name__}: {e}"
                        sp_segs, sp_samples = [], None
                # static two-shot sources never have single-face moments
                # to anchor voice clusters. Use the audio-visual SYNC detector that
                # correlates motion-stabilised mouth motion with the audio envelope per
                # voice cluster — far more discriminating than the pixel-diff VAD
                # for the listener case (both faces move mouths; only one tracks the
                # cluster's audio). Falls through to the pixel VAD when the AV-sync
                # scores are too close to call (identical implementation shape so the
                # downstream _speaker_replace path is unchanged).
                if not sp_segs and use_audio_first:
                    try:
                        sp_segs, sp_samples = _speaker.active_speaker_av_sync(
                            src, start, dur, words=words, cfg=scfg, return_samples=True)
                        if sp_segs:
                            path_tried = "av_sync"
                            diar_meta = sp_segs[0].pop("_diarization_meta", None)
                    except Exception as e:
                        plan["speaker_av_sync_error"] = f"{type(e).__name__}: {e}"
                        sp_segs, sp_samples = [], None
                if not sp_segs and use_audio_first:
                    try:
                        sp_segs, sp_samples = _speaker.active_speaker_per_face_vad(
                            src, start, dur, words=words, cfg=scfg, return_samples=True)
                        if sp_segs:
                            path_tried = "pixel_vad"
                            diar_meta = sp_segs[0].pop("_diarization_meta", None)
                    except Exception as e:
                        plan["speaker_pixel_vad_error"] = f"{type(e).__name__}: {e}"
                        sp_segs, sp_samples = [], None
                if not sp_segs:
                    try:
                        sp_segs, sp_samples = _speaker.active_speaker(src, start, dur,
                                                                      cfg=scfg,
                                                                      return_samples=True)
                        if sp_segs:
                            path_tried = "mouth_motion"
                    except Exception as e:
                        plan["speaker_error"] = f"{type(e).__name__}: {e}"
                        sp_segs, sp_samples = [], None
                if sp_segs:
                    dz = float(scfg.get("dead_zone", SPEAKER_DEAD_ZONE))
                    # when the diarized talker's face isn't visible in a window,
                    # hold the previous crop (don't drift to the median position of
                    # samples that don't exist).
                    tied_or_window = tied if tied else [(0.0, dur)]
                    plan["segments"] = _speaker_replace(plan["segments"], tied_or_window,
                                                        sp_segs,
                                                        src_w, crop_width(src_w, src_h),
                                                        samples=sp_samples, dead_zone=dz,
                                                        smooth_track=smooth_track,
                                                        shots=shots)
                    if smooth_track:
                        plan["smooth_track"] = {
                            "sample_fps": SMOOTH_SAMPLE_FPS,
                            "sigma_s": SMOOTH_SIGMA_S,
                            "max_vel_pct": SMOOTH_MAX_VEL_PCT,
                        }
                    plan["speaker_segments"] = sp_segs
                    plan["speaker_tied_shots"] = tied
                    plan["speaker_path"] = path_tried
                    if diar_meta:
                        plan["diarization"] = {
                            "k": diar_meta["diarization"].get("k"),
                            "mapping": {str(k): v for k, v in diar_meta["mapping"].items()},
                            "evidence": {str(k): v for k, v in diar_meta["evidence"].items()},
                            "n_words": len(diar_meta["diarization"].get("words") or []),
                        }
                    # per-clip metrics for QC
                    plan["speaker_switches"] = len(sp_segs) - 1
                    plan["same_speaker_relocks"] = sum(1 for s in plan["segments"]
                                                        if s.get("relock"))
                    plan["speaker_cfg"] = {k: scfg.get(k) for k in
                                           ("min_hold_s", "switch_margin",
                                            "min_candidate_run_s", "dead_zone")}
    return plan


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        sys.exit("usage: reframe.py SRC START DUR")
    print(json.dumps(analyze(sys.argv[1], float(sys.argv[2]), float(sys.argv[3])), indent=2))
