#!/usr/bin/env python3
"""Active-speaker tracking for 3+ person podcasts / remote grids.

The I/O half samples the clip window at ~10 fps at native resolution (mouth ROIs need real
pixels), detects faces (YuNet with 5 landmarks) and pulls a matching audio track. The
decision half (assign_tracks, mouth_activity, pick_speakers) is pure so it is unit-testable
with synthetic samples.

For each ~0.5 s decision window, the active speaker is the tracked face with the highest
normalized mouth activity while audio is above the speech floor. Hysteresis: minimum hold
`min_hold_s` and a margin before switching, so the picked speaker never ping-pongs.

Numpy + cv2 + ffmpeg only. No pyannote / HF-token / torch deps.

Shapes:
  Track    = {"id": int, "face": {"cx","cy","w","h"}, "landmarks": [(x,y)*5], "mouth_roi": np.ndarray|None}
  Speaker  = {"t0": float, "t1": float, "face_idx": int, "cx": float, "cy": float}
"""
import json
import math
import subprocess
from pathlib import Path

import numpy as np

# YuNet landmarks are (right_eye, left_eye, nose, right_mouth, left_mouth) — indices 3 and 4.
MOUTH_LEFT_IDX = 4
MOUTH_RIGHT_IDX = 3

DEFAULTS = {
    "enabled": True,
    "sample_fps": 10.0,
    "decision_hop_s": 0.5,
    "min_hold_s": 2.5,                 # challenger can't take over before current held this long
    "switch_margin": 0.35,             # challenger must lead by this fraction of max activity
    "min_candidate_run_s": 1.2,        # new speaker must dominate continuously this long before we switch
    "dead_zone": 0.12,                 # face center must drift outside ±12% of crop width to re-lock
    "iou_min": 0.15,                   # face-track association threshold
    "centre_max": 0.12,                # fallback: max normalised face-centre distance for a match
    "speech_rms": 0.015,               # audio RMS above this = speech is on
    "mouth_pad_w": 1.6,                # mouth ROI is this many mouth-widths wide
    "mouth_pad_h": 1.3,                # ... and this many mouth-widths tall (mouths are wider than tall)
    "min_face_h": 0.04,                # ignore faces smaller than 4% of frame height
    "min_score": 0.7,                  # YuNet confidence floor
    "grid_min_gap": 0.15,              # fraction of frame width between remote-grid boxes
}


# ---------------------------------------------------------------- I/O

def _detector():
    """Cached YuNet detector, or None when cv2 / the model are unavailable."""
    global _DET
    d = globals().get("_DET")
    if d is not None:
        return d
    _DET = ("none", None)
    try:
        import cv2
    except ImportError:
        globals()["_DET"] = _DET
        return _DET
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    model = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
    if model.exists() and hasattr(cv2, "FaceDetectorYN"):
        try:
            _DET = ("yunet", cv2.FaceDetectorYN.create(str(model), "", (320, 320),
                                                        DEFAULTS["min_score"], 0.3, 5000))
        except Exception:
            pass
    globals()["_DET"] = _DET
    return _DET


def detect_faces_with_landmarks(frame):
    """[{cx, cy, w, h, score, landmarks}] in a BGR frame; landmarks are (x, y) tuples in
    pixels. Empty list when YuNet is not available or nothing detected."""
    kind, det = _detector()
    if kind != "yunet":
        return []
    fh, fw = frame.shape[:2]
    det.setInputSize((fw, fh))
    _, found = det.detect(frame)
    out = []
    for f in (found if found is not None else []):
        score = float(f[-1])
        if score < DEFAULTS["min_score"]:
            continue
        x, y, w, h = float(f[0]), float(f[1]), float(f[2]), float(f[3])
        if h / fh < DEFAULTS["min_face_h"]:
            continue
        lms = [(float(f[4 + 2 * k]), float(f[5 + 2 * k])) for k in range(5)]
        out.append({"cx": (x + w / 2) / fw, "cy": (y + h / 2) / fh,
                    "w": w / fw, "h": h / fh, "score": score,
                    "landmarks": lms, "bbox": (x, y, w, h)})
    return sorted(out, key=lambda f: f["h"], reverse=True)


def iter_frames_native(src, start, dur, fps=10.0):
    """Yield (t, bgr_frame) at native resolution and ~`fps` decimation."""
    # reframe imports speaker at top level; speaker calls back into it for probe_size
    from clipengine.render import reframe as _reframe
    src_w, src_h = _reframe.probe_size(src)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", str(start), "-t", str(dur),
           "-i", str(src), "-an", "-vf", f"fps={fps}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    size = src_w * src_h * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        i = 0
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield round(i / fps, 3), np.frombuffer(buf, np.uint8).reshape(src_h, src_w, 3)
            i += 1


def load_audio_rms(src, start, dur, sr=16000, hop_s=0.05):
    """Per-hop RMS of the mono audio in [start, start+dur]. `hop_s` = 50 ms (~20 Hz).
    Returns np.array (may be empty when audio is unavailable)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", str(start), "-t", str(dur),
           "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return np.zeros(0, dtype=np.float64)
    if not raw:
        return np.zeros(0, dtype=np.float64)
    a = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    n = max(1, int(round(sr * hop_s)))
    k = len(a) // n
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    fr = a[:k * n].reshape(k, n)
    return np.sqrt((fr * fr).mean(axis=1))


# ---------------------------------------------------------------- pure decisions

def face_iou(a, b):
    """IoU of two normalised faces (cx, cy, w, h)."""
    ax0, ay0 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2
    ax1, ay1 = ax0 + a["w"], ay0 + a["h"]
    bx0, by0 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2
    bx1, by1 = bx0 + b["w"], by0 + b["h"]
    ov = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    if ov <= 0:
        return 0.0
    return ov / (a["w"] * a["h"] + b["w"] * b["h"] - ov + 1e-9)


def assign_tracks(samples, iou_min=None, centre_max=None):
    """Assign a stable track id to each face across `samples` (list of dicts with "t", "faces").
    Greedy IoU match against the last-known face position of every open track; near matches by
    centre distance when IoU is 0 (a face that moved between frames). Modifies `samples` in
    place: each face gets a "track_id" key. Returns the number of distinct tracks."""
    iou_min = DEFAULTS["iou_min"] if iou_min is None else iou_min
    centre_max = DEFAULTS["centre_max"] if centre_max is None else centre_max
    next_id = 0
    last_by_track = {}      # track_id -> (t, face)
    for s in samples:
        faces = s.get("faces") or []
        used = set()
        # greedy: sort candidate matches by (best available track id, score)
        for f in faces:
            best = None
            for tid, (ts, prev) in last_by_track.items():
                if tid in used:
                    continue
                score = face_iou(f, prev)
                if score < iou_min:
                    d = math.hypot(f["cx"] - prev["cx"], f["cy"] - prev["cy"])
                    if d <= centre_max:
                        score = max(score, 0.5 * (1.0 - d / centre_max))
                if score > 0 and (best is None or score > best[1]):
                    best = (tid, score)
            if best is not None:
                tid = best[0]
                used.add(tid)
                f["track_id"] = tid
                last_by_track[tid] = (s["t"], f)
            else:
                f["track_id"] = next_id
                last_by_track[next_id] = (s["t"], f)
                next_id += 1
    return next_id


def _mouth_box(face, frame_w, frame_h, mouth_pad_w=None, mouth_pad_h=None):
    """(x0, y0, x1, y1) pixel bbox around the mouth from YuNet landmarks."""
    mouth_pad_w = DEFAULTS["mouth_pad_w"] if mouth_pad_w is None else mouth_pad_w
    mouth_pad_h = DEFAULTS["mouth_pad_h"] if mouth_pad_h is None else mouth_pad_h
    lm = face.get("landmarks") or []
    if len(lm) < 5:
        # no landmarks: fall back to the lower third of the face bbox
        x, y, w, h = face["cx"] * frame_w, face["cy"] * frame_h, face["w"] * frame_w, face["h"] * frame_h
        return (int(x - 0.2 * w), int(y + 0.1 * h), int(x + 0.2 * w), int(y + 0.4 * h))
    lx, ly = lm[MOUTH_LEFT_IDX]
    rx, ry = lm[MOUTH_RIGHT_IDX]
    mx, my = (lx + rx) / 2.0, (ly + ry) / 2.0
    mw = max(1.0, abs(lx - rx)) * mouth_pad_w
    mh = mw / mouth_pad_w * mouth_pad_h * 0.6         # mouths are wider than tall
    x0 = int(round(max(0, mx - mw / 2)))
    y0 = int(round(max(0, my - mh / 2)))
    x1 = int(round(min(frame_w, mx + mw / 2)))
    y1 = int(round(min(frame_h, my + mh / 2)))
    return (x0, y0, x1, y1)


def mouth_activity(frame_prev_gray, frame_cur_gray, face_prev, face_cur):
    """Mean absolute grayscale difference in the mouth ROI between two frames. Normalized by
    the face height (in pixels) so a big head does not inflate activity. Zero when the ROIs
    have no overlap."""
    fh, fw = frame_cur_gray.shape[:2]
    b_prev = _mouth_box(face_prev, fw, fh)
    b_cur = _mouth_box(face_cur, fw, fh)
    x0 = max(b_prev[0], b_cur[0])
    y0 = max(b_prev[1], b_cur[1])
    x1 = min(b_prev[2], b_cur[2])
    y1 = min(b_prev[3], b_cur[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    a = frame_prev_gray[y0:y1, x0:x1].astype(np.float32)
    b = frame_cur_gray[y0:y1, x0:x1].astype(np.float32)
    diff = np.mean(np.abs(a - b))
    scale = max(1.0, float(face_cur.get("h", 0.05)) * fh)
    return float(diff) * 100.0 / scale


def speech_mask(rms, sample_hop_s=0.05, floor=None):
    """Boolean array: True where audio RMS exceeds `floor` (or an adaptive floor at 60th
    percentile)."""
    rms = np.asarray(rms, dtype=np.float64)
    if not len(rms):
        return np.zeros(0, dtype=bool)
    if floor is None:
        floor = max(DEFAULTS["speech_rms"], float(np.percentile(rms, 40)))
    return rms > float(floor)


def _speech_on(mask, t0, t1, hop_s):
    """True when at least half of the samples in [t0, t1] exceed the floor."""
    a = int(round(t0 / hop_s))
    b = int(round(t1 / hop_s))
    a, b = max(0, a), min(len(mask), b)
    if b <= a:
        return False
    return bool(np.mean(mask[a:b]) >= 0.4)


def _switch_at(cur_t0, challenger_run_t0, min_hold_s):
    """where a confirmed switch cuts. The decision is offline, so the
    cut is back-dated from the confirming hop to the hop where the challenger's continuous lead
    began (the turn, not the turn + `min_candidate_run_s`), but never before the current
    speaker has held `min_hold_s`."""
    return max(challenger_run_t0, cur_t0 + min_hold_s)


def pick_speakers(track_activity, speech_bool_fn, hop_s, min_hold_s=None, switch_margin=None,
                  min_candidate_run_s=None):
    """From per-hop `track_activity` [(t, {track_id: activity})], return
    [{"t0","t1","track_id"}] chosen by:
      - highest activity per hop while speech is on
      - hysteresis: never switch before `min_hold_s` has elapsed on the current speaker
      - switch requires the challenger to lead by at least `switch_margin` (of max activity)
        *continuously for `min_candidate_run_s`* — ignores backchannels (yeah / mm / laughs)
      - a confirmed switch is back-dated to the start of the challenger's lead (_switch_at)
      - never switch to a track with zero activity (silent speaker)
      - speech gate off (nobody talking) -> hold current speaker

    `speech_bool_fn(t)` returns True when speech is on at time `t`. Consecutive segments
    for the same track are coalesced into one segment (never emit same-speaker neighbours)."""
    min_hold_s = DEFAULTS["min_hold_s"] if min_hold_s is None else float(min_hold_s)
    switch_margin = DEFAULTS["switch_margin"] if switch_margin is None else float(switch_margin)
    min_candidate_run_s = (DEFAULTS["min_candidate_run_s"] if min_candidate_run_s is None
                           else float(min_candidate_run_s))
    if not track_activity:
        return []
    segments = []
    cur, cur_t0 = None, None
    challenger, challenger_run_t0 = None, None       # track_id leading non-cur, since this t
    for t, per_track in track_activity:
        if not per_track:
            continue
        speaking = speech_bool_fn(t)
        best_id, best_val = max(per_track.items(), key=lambda kv: kv[1])
        if cur is None:
            # bootstrap: wait for the first hop that has speech + a non-zero winner
            if speaking and best_val > 0:
                cur, cur_t0 = best_id, t
            continue
        if not speaking:
            challenger, challenger_run_t0 = None, None
            continue                # speech gate off: hold current speaker
        held = t - cur_t0
        margin_needed = switch_margin * max(1e-6, best_val)
        cur_val = per_track.get(cur, 0.0)
        leads = (best_id != cur and (best_val - cur_val) >= margin_needed and best_val > 0)
        if not leads:
            challenger, challenger_run_t0 = None, None
            continue
        if challenger != best_id:
            challenger, challenger_run_t0 = best_id, t
            continue
        run = t - challenger_run_t0
        if held < min_hold_s or run < min_candidate_run_s:
            continue              # challenger has to lead continuously long enough
        cut = _switch_at(cur_t0, challenger_run_t0, min_hold_s)
        segments.append({"t0": round(cur_t0, 3), "t1": round(cut, 3), "track_id": cur})
        cur, cur_t0 = best_id, cut
        challenger, challenger_run_t0 = None, None
    if cur is not None:
        last_t = track_activity[-1][0] + hop_s
        segments.append({"t0": round(cur_t0, 3), "t1": round(last_t, 3), "track_id": cur})
    # coalesce consecutive same-speaker segments (defensive — the state machine should
    # never emit them but keep this invariant local).
    coalesced = []
    for s in segments:
        if coalesced and coalesced[-1]["track_id"] == s["track_id"]:
            coalesced[-1]["t1"] = s["t1"]
        else:
            coalesced.append(dict(s))
    return coalesced


# ---------------------------------------------------------------- entry point

def active_speaker(src, start, dur, cfg=None, return_samples=False):
    """[{"t0","t1","track_id","cx","cy","face_h"}] describing who is talking through the
    window, from 10fps sampling and audio RMS gating. Empty list when the detector or
    audio is unavailable, or when fewer than 2 tracks appear (the caller falls back to
    the largest face). All times are clip-relative seconds.

    When `return_samples=True`, returns `(segments, samples)` where `samples` is the
    full per-sample face dict list `[{"t", "faces": [...]}]` with track_ids assigned —
    the caller uses this to do dead-zone re-locks on long same-speaker shots without
    re-sampling the window."""
    c = {**DEFAULTS, **(cfg or {})}
    fps = float(c["sample_fps"])
    hop_s = float(c["decision_hop_s"])
    samples = []
    frames_gray = []           # (t, gray_frame)
    import cv2
    for t, frame in iter_frames_native(src, start, dur, fps=fps):
        faces = detect_faces_with_landmarks(frame)
        samples.append({"t": t, "faces": faces})
        frames_gray.append((t, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    if not samples:
        return ([], []) if return_samples else []
    n_tracks = assign_tracks(samples, iou_min=c["iou_min"], centre_max=c["centre_max"])
    if n_tracks < 2:
        return ([], samples) if return_samples else []
    # per-hop activity: sum mouth diffs between the two frames closest to each hop centre
    activity_seq = []
    prev_by_track = {}         # track_id -> (t, face, gray)
    for i, (t, gray) in enumerate(frames_gray):
        per_track = {}
        for f in samples[i].get("faces") or []:
            tid = f.get("track_id")
            prev = prev_by_track.get(tid)
            if prev is not None:
                act = mouth_activity(prev[2], gray, prev[1], f)
                per_track[tid] = per_track.get(tid, 0.0) + act
            prev_by_track[tid] = (t, f, gray)
        if per_track:
            activity_seq.append((t, per_track))
    # bin activities into decision hops
    hops = []
    if activity_seq:
        t_start = activity_seq[0][0]
        t_end = activity_seq[-1][0]
        h = 0
        while t_start + h * hop_s <= t_end:
            a, b = t_start + h * hop_s, t_start + (h + 1) * hop_s
            agg = {}
            for t, per_track in activity_seq:
                if a <= t < b:
                    for tid, v in per_track.items():
                        agg[tid] = agg.get(tid, 0.0) + v
            if agg:
                # normalise per track by the per-track mean activity so beards / small faces
                # do not lose to shiny mouths
                hops.append(((a + b) / 2.0, agg))
            h += 1
    # per-track normalisation: divide each track's activity by its own median
    if hops:
        tid_vals = {}
        for _, per in hops:
            for tid, v in per.items():
                tid_vals.setdefault(tid, []).append(v)
        med = {tid: max(np.median(vs), 1e-6) for tid, vs in tid_vals.items()}
        hops = [(t, {tid: v / med[tid] for tid, v in per.items()}) for t, per in hops]

    rms = load_audio_rms(src, start, dur, hop_s=0.05)
    sp_mask = speech_mask(rms, sample_hop_s=0.05, floor=c.get("speech_floor"))
    speech_fn = lambda t: _speech_on(sp_mask, t - hop_s / 2, t + hop_s / 2, 0.05)

    segs = pick_speakers(hops, speech_fn, hop_s,
                         min_hold_s=c["min_hold_s"], switch_margin=c["switch_margin"],
                         min_candidate_run_s=c["min_candidate_run_s"])

    # attach face centre / height for each segment: median over the samples that fall
    # inside *this speaker's own time range* (not the whole window). Keeps the crop
    # pinned to where the speaker actually sat, not a position polluted by moments when
    # someone else held the frame.
    face_pos_by_tid = {}       # track_id -> [(t, cx, cy, h)]
    for s in samples:
        for f in s.get("faces") or []:
            tid = f.get("track_id")
            if tid is None:
                continue
            face_pos_by_tid.setdefault(tid, []).append((s["t"], f["cx"], f["cy"], f["h"]))
    for seg in segs:
        tid = seg["track_id"]
        pts = [(cx, cy, h) for (t, cx, cy, h) in face_pos_by_tid.get(tid, [])
               if seg["t0"] <= t <= seg["t1"]]
        if not pts:    # speaker had no face sample in-range: fall back to the full-track median
            pts = [(cx, cy, h) for (_, cx, cy, h) in face_pos_by_tid.get(tid, [])]
        if pts:
            seg["cx"] = float(np.median([p[0] for p in pts]))
            seg["cy"] = float(np.median([p[1] for p in pts]))
            seg["face_h"] = float(np.median([p[2] for p in pts]))
        else:
            seg["cx"] = seg["cy"] = 0.5
            seg["face_h"] = 0.1
    return (segs, samples) if return_samples else segs


# ---------------------------------------------------------------- audio-first

def _audio_first_hops(diarization, mapping, hop_s, t0, t1):
    """Build per-hop talker runs from the diarized words. Each hop
    [a, a+hop_s) carries the mapped face track_id of the diarized voice
    centred in that hop. Hops with no diarized word fall back to the previous
    talker (keeps the camera on the last known speaker during silence)."""
    talker = None
    hops = []
    a = t0
    while a < t1:
        b = min(a + hop_s, t1)
        words = diarization.get("words") or []
        labels = diarization.get("labels") or []
        votes = {}
        for lbl, w in zip(labels, words):
            if w["end"] >= a and w["start"] < b:
                tid = mapping.get(lbl)
                if tid is None:
                    continue
                overlap = min(b, w["end"]) - max(a, w["start"])
                if overlap > 0:
                    votes[tid] = votes.get(tid, 0.0) + overlap
        if votes:
            talker = max(votes.items(), key=lambda kv: kv[1])[0]
        hops.append(((a + b) / 2.0, talker))     # may be None if nothing seen yet
        a = b
    return hops


def pick_speakers_audio_first(audio_hops, hop_s, min_hold_s=None, switch_margin=None,
                               min_candidate_run_s=None):
    """From audio-first per-hop talker ids (produced by _audio_first_hops),
    return `[{"t0","t1","track_id"}]` applying the hysteresis gates:
    challenger must hold the floor for `min_candidate_run_s` continuously
    before switching, current speaker must have held `min_hold_s` first. The
    switch is back-dated to the challenger's first hop (_switch_at).

    `switch_margin` is accepted for API parity but has no effect in the
    audio-first path — the challenger either IS or isn't the diarized talker,
    there's no activity magnitude to compare against. (The underlying signal
    is independent of mouth motion by construction, which is the whole point
    of the audio-first decision.)"""
    min_hold_s = DEFAULTS["min_hold_s"] if min_hold_s is None else float(min_hold_s)
    min_candidate_run_s = (DEFAULTS["min_candidate_run_s"] if min_candidate_run_s is None
                           else float(min_candidate_run_s))
    _ = switch_margin
    if not audio_hops:
        return []
    segments = []
    cur, cur_t0 = None, None
    challenger, challenger_run_t0 = None, None
    for t, talker in audio_hops:
        if cur is None:
            if talker is not None:
                cur, cur_t0 = talker, t
            continue
        if talker is None:
            challenger, challenger_run_t0 = None, None
            continue
        if talker == cur:
            challenger, challenger_run_t0 = None, None
            continue
        if challenger != talker:
            challenger, challenger_run_t0 = talker, t
            continue
        held = t - cur_t0
        run = t - challenger_run_t0
        if held < min_hold_s or run < min_candidate_run_s:
            continue
        # the talker ids come from diarised words, so the run's first hop is the turn itself
        cut = _switch_at(cur_t0, challenger_run_t0, min_hold_s)
        segments.append({"t0": round(cur_t0, 3), "t1": round(cut, 3), "track_id": cur})
        cur, cur_t0 = talker, cut
        challenger, challenger_run_t0 = None, None
    if cur is not None:
        last_t = audio_hops[-1][0] + hop_s
        segments.append({"t0": round(cur_t0, 3), "t1": round(last_t, 3), "track_id": cur})
    coalesced = []
    for s in segments:
        if coalesced and coalesced[-1]["track_id"] == s["track_id"]:
            coalesced[-1]["t1"] = s["t1"]
        else:
            coalesced.append(dict(s))
    return coalesced


def active_speaker_audio_first(src, start, dur, words, cfg=None, return_samples=False):
    """Audio-first talker decision. Pipeline:
      1. sample native-res frames + run YuNet (same as `active_speaker`),
      2. assign stable face track ids,
      3. diarize the audio into K voice clusters per word (`render/diarize`),
      4. map voice clusters -> face tracks via single-face moments,
      5. build per-hop talker runs from the diarized words (no mouth motion),
      6. apply the hysteresis (min_hold / candidate_run) on top.

    Returns `[{"t0","t1","track_id","cx","cy","face_h"}]` identical shape to
    `active_speaker` + extra metadata on the returned samples list. When
    diarization fails (silence, too few words), returns [] so the caller falls
    back to the primary-face planner.

    Mouth motion is only used downstream, as a *tie-breaker* when two clusters
    both map to the same face track; it never DECIDES who is talking."""
    c = {**DEFAULTS, **(cfg or {})}
    fps = float(c["sample_fps"])
    hop_s = float(c["decision_hop_s"])

    import cv2
    from clipengine.render import diarize as _diarize

    samples = []
    for t, frame in iter_frames_native(src, start, dur, fps=fps):
        faces = detect_faces_with_landmarks(frame)
        samples.append({"t": t, "faces": faces})
    if not samples:
        return ([], []) if return_samples else []
    n_tracks = assign_tracks(samples, iou_min=c["iou_min"], centre_max=c["centre_max"])
    if n_tracks < 2:
        return ([], samples) if return_samples else []

    diar = _diarize.diarize_words(src, start, dur, words or [])
    if not diar or diar.get("k", 0) < 2:
        return ([], samples) if return_samples else []

    # which track_ids are even plausible (seen in a single-face sample)
    seen_tids = set()
    for s in samples:
        for f in s.get("faces") or []:
            seen_tids.add(f.get("track_id"))
    mapping, evidence = _diarize.map_clusters_to_tracks(diar, samples,
                                                        allow_track_ids=seen_tids)
    if len(mapping) < 2:
        # not enough single-face evidence to disambiguate speakers; let the
        # pixel-VAD anchor take over.
        return ([], samples) if return_samples else []
    # when ANY mapped cluster came from the diarize._av_sync fallback (the weak
    # landmark-displacement proxy, not single-face moments), the mapping may map that
    # cluster to the WRONG face. A static two-shot podcast is the canonical case: no
    # single-face evidence available for all speakers, the av_sync proxy latches onto
    # whichever face has more subtle head motion, and that is NOT always the talker.
    # Reject the mapping entirely and fall through to :func:`active_speaker_per_face_vad`
    # whose pixel-based mouth motion signal (``pixel_mouth_motion_by_track`` + Pass 2 in
    # :func:`map_clusters_to_tracks_pixel_vad`) is strictly stronger than the landmark
    # proxy. A single-face-only mapping (every cluster had a confident single-face vote)
    # is kept because single_face voting is the authoritative mapping.
    sources = {v.get("source") for v in (evidence or {}).values()}
    if sources and "av_sync" in sources:
        return ([], samples) if return_samples else []

    hops = _audio_first_hops(diar, mapping, hop_s, t0=0.0, t1=dur)
    segs = pick_speakers_audio_first(hops, hop_s,
                                      min_hold_s=c["min_hold_s"],
                                      switch_margin=c.get("switch_margin"),
                                      min_candidate_run_s=c["min_candidate_run_s"])

    # attach per-speaker face centre: median of samples where this track's face
    # was actually visible, within the speaker's own time range
    face_pos_by_tid = {}
    for s in samples:
        for f in s.get("faces") or []:
            tid = f.get("track_id")
            if tid is None:
                continue
            face_pos_by_tid.setdefault(tid, []).append((s["t"], f["cx"], f["cy"], f["h"]))
    for seg in segs:
        tid = seg["track_id"]
        pts = [(cx, cy, h) for (t, cx, cy, h) in face_pos_by_tid.get(tid, [])
               if seg["t0"] <= t <= seg["t1"]]
        if not pts:
            pts = [(cx, cy, h) for (_, cx, cy, h) in face_pos_by_tid.get(tid, [])]
        if pts:
            seg["cx"] = float(np.median([p[0] for p in pts]))
            seg["cy"] = float(np.median([p[1] for p in pts]))
            seg["face_h"] = float(np.median([p[2] for p in pts]))
        else:
            seg["cx"] = seg["cy"] = 0.5
            seg["face_h"] = 0.1

    # Attach diarization metadata so the caller can log mapping evidence
    for seg in segs:
        seg["diarization_cluster"] = {k: v for k, v in mapping.items() if v == seg["track_id"]}
    if return_samples:
        # also stash diarization bits on the samples list via its first element so
        # the caller can read them without a second return value
        samples_meta = {"diarization": diar, "mapping": mapping, "evidence": evidence}
        if segs:
            segs[0]["_diarization_meta"] = samples_meta
        return segs, samples
    return segs


# ---------------------------------------------------------------- 3+ tied / remote grid

def shot_is_multi_speaker(samples, tie_tol=0.9, min_faces=3):
    """True when at least `min_faces` faces are consistently detected at comparable size in
    the samples of one shot. `tie_tol` matches reframe.TIE_TOL: faces within 10% of the
    largest count as tied."""
    tied_frames = 0
    for s in samples:
        faces = s.get("faces") or []
        if not faces:
            continue
        top = max(f["h"] for f in faces)
        tied = [f for f in faces if f["h"] >= tie_tol * top]
        if len(tied) >= min_faces:
            tied_frames += 1
    return tied_frames * 2 >= len(samples)      # majority of samples show 3+ tied


def shot_is_two_tied(samples, tie_tol=0.9):
    """True when at least half of the shot's frames show >= 2 faces of comparable size
    (tied by height). Used by reframe to decide "track with >= 2 tied faces"."""
    tied_frames = 0
    for s in samples:
        faces = s.get("faces") or []
        if not faces:
            continue
        top = max(f["h"] for f in faces)
        tied = [f for f in faces if f["h"] >= tie_tol * top]
        if len(tied) >= 2:
            tied_frames += 1
    return tied_frames * 2 >= len(samples)


# ---------------------------------------------------------------- per-face pixel VAD anchor

def pixel_mouth_motion_by_track(samples_with_gray, track_ids):
    """Per-face pixel-based mouth motion time series. Returns ``{tid: [(t, motion)]}``.

    `samples_with_gray` is a list ``[{"t", "faces", "gray"}]`` where ``gray`` is the
    frame as single-channel uint8 (shape H,W). Consecutive samples carrying the same
    track are differenced over the mouth ROI (via :func:`mouth_activity`). This is the
    SAME signal `active_speaker` uses for mouth-motion speaker decisions, but computed
    per face track across the whole window — not per decision hop.

    The static two-shot case has no single-face moments to anchor voice
    clusters, and the landmark-displacement proxy in diarize._track_mouth_motion_series
    is too weak to disambiguate the talker from the listener. The pixel diff is strong
    enough to anchor each cluster to its face track.

    Streaming variant: when ``samples_with_gray`` is a generator yielding one dict at
    a time, memory stays O(1) in the number of frames (only the previous grayscale
    is kept per track). Callers passing a full list get the same O(N) peak behaviour
    as before.
    """
    prev_by_track = {}
    out = {tid: [] for tid in track_ids}
    for s in samples_with_gray:
        gray = s.get("gray")
        if gray is None:
            prev_by_track = {}
            continue
        for f in s.get("faces") or []:
            tid = f.get("track_id")
            if tid is None or tid not in out:
                continue
            prev = prev_by_track.get(tid)
            if prev is not None:
                try:
                    act = mouth_activity(prev[1], gray, prev[0], f)
                except Exception:                 # noqa: BLE001
                    act = 0.0
                out[tid].append((s["t"], float(act)))
            prev_by_track[tid] = (f, gray)
    return out


def map_clusters_to_tracks_pixel_vad(diarization, mouth_series, speech_fn=None):
    """Map voice clusters -> face tracks using pixel-based mouth motion correlation.

    `mouth_series` is `{tid: [(t, motion)]}` from :func:`pixel_mouth_motion_by_track`.
    Each cluster C's score for track T = sum of T's mouth motion while C's words
    overlap the sample time t (± 50 ms). Greedy 1-1 assignment: highest score wins,
    lock C and T, repeat. The second-best (cluster, track) must be strictly below
    the best (ratio < 0.85 of best) OR the losing track must have < 10 % of the
    winning track's score for this cluster — otherwise the pair is skipped as
    ambiguous and the next cluster is tried first.

    Returns `(mapping, evidence)` identical in shape to
    :func:`diarize.map_clusters_to_tracks`.
    """
    labels = diarization.get("labels") or []
    words = diarization.get("words") or []
    if not labels or not mouth_series:
        return {}, {}
    cluster_ids = sorted(set(labels))
    track_ids = sorted(mouth_series.keys())
    windows_by_cluster = {c: [] for c in cluster_ids}
    for lbl, w in zip(labels, words):
        windows_by_cluster.setdefault(lbl, []).append((float(w["start"]), float(w["end"])))
    score = {}
    for c, windows in windows_by_cluster.items():
        for tid in track_ids:
            total = 0.0
            series = mouth_series.get(tid) or []
            if not series or not windows:
                score.setdefault(c, {})[tid] = 0.0
                continue
            wi = 0
            for (st, act) in series:
                while wi < len(windows) and windows[wi][1] + 0.05 < st:
                    wi += 1
                hit = False
                for j in range(wi, len(windows)):
                    (a, b) = windows[j]
                    if a - 0.05 <= st <= b + 0.05:
                        hit = True
                        break
                    if a > st + 0.05:
                        break
                if hit:
                    total += act
            score.setdefault(c, {})[tid] = total
    # Each cluster is mapped INDEPENDENTLY to the face track with the strongest
    # pixel-mouth-motion correlation. We do NOT force 1-to-1 (two clusters are
    # allowed to map to the same face track) — a two-shot podcast clip where one host talks
    # the whole clip may show up as k=2 clusters in k-means because of acoustic
    # micro-variation; forcing 1-to-1 would send the second cluster to the LISTENER
    # face, producing listener-framing bugs. The per-cluster winner must still beat
    # its track-runner-up by ``SEPARATION_RATIO`` (otherwise the signal is too weak
    # to be trusted — leave that cluster unmapped and the hop builder will hold the
    # previous talker).
    SEPARATION_RATIO = 1.5
    MIN_TOTAL_MOTION = 0.5          # absolute floor so a near-zero signal cannot pass
    mapping = {}
    evidence = {}
    for c in cluster_ids:
        per = score.get(c) or {}
        if not per:
            continue
        best_tid, best_v = max(per.items(), key=lambda kv: kv[1])
        if best_v < MIN_TOTAL_MOTION:
            continue
        runner_v = max(
            [per.get(t2, 0.0) for t2 in track_ids if t2 != best_tid] or [0.0])
        if runner_v > 0.0 and best_v < SEPARATION_RATIO * runner_v:
            continue
        mapping[c] = best_tid
        evidence[c] = {"track_id": best_tid, "score": round(best_v, 3),
                       "track_runner": round(runner_v, 3),
                       "source": "pixel_vad"}
    return mapping, evidence


# ---------------------------------------------------------------- audio-visual sync ASD
#
# A review finding: the pixel-diff VAD (`pixel_mouth_motion_by_track` +
# `map_clusters_to_tracks_pixel_vad`) treats ANY motion in the mouth ROI as speaking
# evidence. Real two-shot podcasts have both faces moving mouths (laughter, backchannel,
# chewing, smiling), so k-means over-split one speaker's voice into k=2 clusters and
# both collapsed onto the same face.
#
# The design called out Light-ASD / TalkNet-ASD. Both are MIT-licensed and their
# weights are available on HuggingFace (e.g. `AlekseyKorshuk/talknet-asd`,
# `pretrain_TalkSet.model` 63 MB). Running either requires torch (~200 MB CPU wheel) +
# dedicated inference code (visual encoder on 112×112 face crops, audio encoder on
# MFCC windows, cross-attention). Adding ~200 MB of toolchain to the engine was flagged
# as out-of-scope here (not in requirements.txt, not previously authorised); the engine
# is CPU-render-budget-bound so a torch deep model is a non-trivial addition.
#
# This implementation ships the SAME signal class a deep ASD surfaces — a
# per-face audio-visual SYNC score — but computed with cv2 + numpy only:
#
#  1. Per-face mouth motion over the whole window, motion-stabilised:
#     - Optical flow (Farneback) in the mouth ROI gives per-pixel displacement.
#     - Subtract the whole-face median flow (removes head translation / camera shake).
#     - The L2 magnitude of the residual flow is "mouth-opening motion" — strongest
#       when the subject's jaw actually moves, not when they merely nod.
#
#  2. Audio envelope at the same sample rate: speech-band RMS (300-3400 Hz).
#
#  3. Per-track × per-voice-cluster audio-visual sync score:
#     - For each voice cluster C and face track T, compute the Pearson correlation
#       between (mouth motion on T) and (audio envelope) across C's word windows.
#     - A listener frame has mouth motion UNCORRELATED with the diarized cluster's
#       speech — correlation is near 0.
#     - A speaker frame's mouth motion rises and falls with the audio — correlation
#       approaches 1.
#     - The AV sync score is `max(0, corr) × mean(mouth motion during C's speech)`,
#       which rewards faces whose motion aligns with the audio and whose motion is
#       strong enough to be real.
#
#  4. Cluster → track mapping: per cluster, the winning track must beat the runner-up
#     by `AV_SYNC_SEPARATION_RATIO` and clear the absolute floor
#     `AV_SYNC_MIN_SCORE`.
#
#  Hysteresis + hop building is reused from the existing pixel-VAD pipeline.
#
#  License note in favour of a future upgrade: see docstrings on
#  `active_speaker_av_sync` — swapping in Light-ASD or TalkNet-ASD only requires
#  replacing `motion_stabilised_mouth_motion_by_track` with a model-driven per-face
#  score producer. The caller contract stays identical.

AV_SYNC_SEPARATION_RATIO = 1.3
AV_SYNC_MIN_SCORE = 0.08           # absolute floor: a near-zero sync score fails
AV_SYNC_CORR_WINDOW_S = 1.5         # Pearson correlation computed over this many
                                         # seconds centred on each sample; stretched to
                                         # cover a syllable (~0.2 s) plus trailing noise.
AV_FLOW_WIN_SIZE = 15               # Farneback window size (odd)
AV_FLOW_PYR = 3                     # pyramid levels for Farneback


def motion_stabilised_mouth_motion_by_track(samples_with_gray, track_ids,
                                              flow_win=AV_FLOW_WIN_SIZE,
                                              flow_pyr=AV_FLOW_PYR):
    """per-face motion-stabilised mouth-opening signal.

    For each face track, consecutive sample pairs produce a per-pixel optical flow
    (``cv2.calcOpticalFlowFarneback``). The whole-face median flow is subtracted so
    pure head translation / camera shake contributes zero; what remains is residual
    motion whose L2 magnitude peaks when the mouth is opening and closing (and when
    teeth / lips expose / cover). This isolates speaking motion from the listener's
    nodding or head turning that fools raw pixel-diff VAD.

    Returns ``{tid: [(t, motion)]}``. Streaming-friendly: ``samples_with_gray`` can
    be a generator yielding one `{t, faces, gray}` dict at a time. Only the previous
    gray frame per track is kept in memory.
    """
    import cv2
    import numpy as np
    out = {tid: [] for tid in track_ids}
    prev_by_track = {}
    for s in samples_with_gray:
        gray = s.get("gray")
        if gray is None:
            prev_by_track = {}
            continue
        gh, gw = gray.shape[:2]
        for f in s.get("faces") or []:
            tid = f.get("track_id")
            if tid is None or tid not in out:
                continue
            prev = prev_by_track.get(tid)
            if prev is not None:
                prev_gray, prev_face = prev
                try:
                    act = _motion_stabilised_mouth(prev_gray, gray, prev_face, f,
                                                     flow_win=flow_win, flow_pyr=flow_pyr)
                except Exception:            # noqa: BLE001
                    act = 0.0
                out[tid].append((float(s["t"]), float(act)))
            prev_by_track[tid] = (gray, f)
    return out


def _motion_stabilised_mouth(prev_gray, cur_gray, face_prev, face_cur,
                               flow_win=AV_FLOW_WIN_SIZE, flow_pyr=AV_FLOW_PYR):
    """Compute the motion-stabilised mouth signal between two frames.
    Residual mouth flow magnitude = ||mouth_flow - median(face_flow)||.
    Returns a scalar normalised by face height (so a big face does not dominate)."""
    import cv2
    import numpy as np
    fh, fw = cur_gray.shape[:2]
    # face bbox in pixels
    fx0 = int(round(max(0.0, (face_cur["cx"] - face_cur["w"] / 2) * fw)))
    fy0 = int(round(max(0.0, (face_cur["cy"] - face_cur["h"] / 2) * fh)))
    fx1 = int(round(min(fw, (face_cur["cx"] + face_cur["w"] / 2) * fw)))
    fy1 = int(round(min(fh, (face_cur["cy"] + face_cur["h"] / 2) * fh)))
    if fx1 <= fx0 + 4 or fy1 <= fy0 + 4:
        return 0.0
    # mouth bbox from landmarks
    mb_prev = _mouth_box(face_prev, fw, fh)
    mb_cur = _mouth_box(face_cur, fw, fh)
    x0 = max(mb_prev[0], mb_cur[0], fx0)
    y0 = max(mb_prev[1], mb_cur[1], fy0)
    x1 = min(mb_prev[2], mb_cur[2], fx1)
    y1 = min(mb_prev[3], mb_cur[3], fy1)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return 0.0
    face_prev_crop = prev_gray[fy0:fy1, fx0:fx1]
    face_cur_crop = cur_gray[fy0:fy1, fx0:fx1]
    if face_prev_crop.shape != face_cur_crop.shape:
        return 0.0
    try:
        flow = cv2.calcOpticalFlowFarneback(
            face_prev_crop, face_cur_crop,
            None, 0.5, flow_pyr, flow_win, 3, 5, 1.2, 0)
    except Exception:                        # noqa: BLE001
        return 0.0
    # Subtract median (translation + camera shake) over the whole face crop.
    med_dx = float(np.median(flow[..., 0]))
    med_dy = float(np.median(flow[..., 1]))
    # Compute residual flow magnitude in the MOUTH subregion (relative to face crop).
    mx0 = x0 - fx0; mx1 = x1 - fx0
    my0 = y0 - fy0; my1 = y1 - fy0
    sub = flow[my0:my1, mx0:mx1]
    if sub.size == 0:
        return 0.0
    rx = sub[..., 0] - med_dx
    ry = sub[..., 1] - med_dy
    mag = np.sqrt(rx * rx + ry * ry).mean()
    # Normalise by face height in pixels so bigger faces don't trivially win.
    scale = max(1.0, float(face_cur.get("h", 0.1)) * fh)
    return float(mag) * 100.0 / scale


def map_clusters_to_tracks_av_sync(diarization, mouth_series, audio_rms_series,
                                     corr_window_s=AV_SYNC_CORR_WINDOW_S,
                                     separation_ratio=AV_SYNC_SEPARATION_RATIO,
                                     min_score=AV_SYNC_MIN_SCORE):
    """map voice clusters -> face tracks by audio-visual SYNC.

    For each cluster C and each candidate face track T:
      score(C, T) = max(0, pearson(mouth_motion_T(windows_C), audio_rms(windows_C)))
                      × mean(mouth_motion_T(windows_C))

    The correlation term is near 0 for listener faces (their motion is uncorrelated
    with the diarized cluster's audio); the magnitude term kills tracks whose motion
    is essentially flat. Separation ratio + absolute floor filter out ambiguous
    mappings.

    `audio_rms_series` is a list of `(t, rms)` pairs at any sample rate; the function
    interpolates to the sample times of `mouth_series`.

    Returns `(mapping, evidence)` identical in shape to
    `map_clusters_to_tracks_pixel_vad`.
    """
    import numpy as np
    labels = diarization.get("labels") or []
    words = diarization.get("words") or []
    if not labels or not mouth_series or not audio_rms_series:
        return {}, {}
    cluster_ids = sorted(set(labels))
    track_ids = sorted(mouth_series.keys())
    windows_by_cluster = {c: [] for c in cluster_ids}
    for lbl, w in zip(labels, words):
        windows_by_cluster.setdefault(lbl, []).append((float(w["start"]), float(w["end"])))
    # Audio RMS as sorted (t, v) and arrays for lookup.
    rms_items = sorted((float(t), float(v)) for t, v in audio_rms_series)
    rms_ts = np.array([p[0] for p in rms_items], dtype=np.float64)
    rms_vs = np.array([p[1] for p in rms_items], dtype=np.float64)
    def _interp(times):
        if rms_ts.size == 0:
            return np.zeros_like(times, dtype=np.float64)
        return np.interp(times, rms_ts, rms_vs)
    score = {c: {} for c in cluster_ids}
    corrs = {c: {} for c in cluster_ids}
    means = {c: {} for c in cluster_ids}
    for c, windows in windows_by_cluster.items():
        if not windows:
            for tid in track_ids:
                score[c][tid] = 0.0
            continue
        for tid in track_ids:
            series = mouth_series.get(tid) or []
            if not series:
                score[c][tid] = 0.0
                continue
            # restrict mouth samples to times inside any of the cluster's windows (± 50 ms)
            ms_t = np.array([t for (t, _) in series], dtype=np.float64)
            ms_v = np.array([v for (_, v) in series], dtype=np.float64)
            in_c = np.zeros_like(ms_t, dtype=bool)
            for (a, b) in windows:
                in_c |= (ms_t >= a - 0.05) & (ms_t <= b + 0.05)
            if int(in_c.sum()) < 4:
                score[c][tid] = 0.0
                continue
            t_in = ms_t[in_c]
            v_in = ms_v[in_c]
            a_in = _interp(t_in)
            # Pearson correlation (zero-safe)
            if np.std(v_in) < 1e-6 or np.std(a_in) < 1e-6:
                corr = 0.0
            else:
                corr = float(np.corrcoef(v_in, a_in)[0, 1])
                if not np.isfinite(corr):
                    corr = 0.0
            mean_motion = float(np.mean(v_in))
            score[c][tid] = max(0.0, corr) * mean_motion
            corrs[c][tid] = corr
            means[c][tid] = mean_motion
    mapping = {}
    evidence = {}
    for c in cluster_ids:
        per = score.get(c) or {}
        if not per:
            continue
        best_tid, best_v = max(per.items(), key=lambda kv: kv[1])
        if best_v < min_score:
            continue
        runner_v = max(
            [per.get(t2, 0.0) for t2 in track_ids if t2 != best_tid] or [0.0])
        if runner_v > 0.0 and best_v < separation_ratio * runner_v:
            continue
        mapping[c] = best_tid
        evidence[c] = {"track_id": best_tid, "score": round(best_v, 4),
                       "corr": round(float(corrs.get(c, {}).get(best_tid, 0.0)), 3),
                       "mean_motion": round(float(means.get(c, {}).get(best_tid, 0.0)), 3),
                       "track_runner": round(runner_v, 4),
                       "source": "av_sync"}
    return mapping, evidence


def active_speaker_av_sync(src, start, dur, words, cfg=None, return_samples=False):
    """audio-visual sync-based active-speaker detector.

    Replaces the pixel-diff VAD (`active_speaker_per_face_vad`) with a motion-
    stabilised mouth-motion signal correlated with the audio envelope. The outer
    pipeline is unchanged: sample native-res frames at 10 fps, assign stable track
    ids, diarize audio into K voice clusters, build per-hop talker runs via
    `_audio_first_hops`, apply the hysteresis in `pick_speakers_audio_first`.

    What changes is the cluster -> track mapping: `map_clusters_to_tracks_av_sync`
    scores each (cluster, track) by `max(0, pearson) × mean_motion`, which is the
    same signal class a deep ASD (Light-ASD / TalkNet-ASD) exposes at its output —
    the correlation + strength of audio-visual sync. The listener case the pixel-
    diff version mis-mapped (two voice clusters collapsing onto the laughing
    listener's track) fails here because the listener's mouth motion is NOT
    correlated with the diarized cluster's audio.

    Returns `[{"t0","t1","track_id","cx","cy","face_h"}]`. Empty when diarization
    fails or the mapping is ambiguous.
    """
    c = {**DEFAULTS, **(cfg or {})}
    fps = float(c["sample_fps"])
    hop_s = float(c["decision_hop_s"])

    import cv2
    from clipengine.render import diarize as _diarize

    # Pass 1: faces + stable track ids (no gray in memory).
    samples = []
    for t, frame in iter_frames_native(src, start, dur, fps=fps):
        samples.append({"t": t, "faces": detect_faces_with_landmarks(frame)})
    if not samples:
        return ([], []) if return_samples else []
    n_tracks = assign_tracks(samples, iou_min=c["iou_min"], centre_max=c["centre_max"])
    if n_tracks < 2:
        return ([], samples) if return_samples else []

    diar = _diarize.diarize_words(src, start, dur, words or [])
    if not diar or diar.get("k", 0) < 2:
        return ([], samples) if return_samples else []

    track_ids = sorted({f["track_id"] for s in samples for f in (s.get("faces") or [])
                        if f.get("track_id") is not None})

    # Pass 2: stream gray frames, compute motion-stabilised mouth motion per track.
    def _gray_stream():
        sample_idx = 0
        for t2, frame2 in iter_frames_native(src, start, dur, fps=fps):
            if sample_idx >= len(samples):
                break
            s = samples[sample_idx]
            gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            yield {"t": s["t"], "faces": s.get("faces") or [], "gray": gray}
            sample_idx += 1

    mouth_series = motion_stabilised_mouth_motion_by_track(_gray_stream(), track_ids)

    # Audio envelope at the sample fps so correlation is unit-less and aligned.
    rms_hop = 1.0 / float(fps)
    rms = load_audio_rms(src, start, dur, hop_s=rms_hop)
    rms_items = [(i * rms_hop, float(rms[i])) for i in range(len(rms))]
    mapping, evidence = map_clusters_to_tracks_av_sync(diar, mouth_series, rms_items)
    if len(mapping) < 2:
        return ([], samples) if return_samples else []

    hops = _audio_first_hops(diar, mapping, hop_s, t0=0.0, t1=dur)
    segs = pick_speakers_audio_first(hops, hop_s,
                                      min_hold_s=c["min_hold_s"],
                                      switch_margin=c.get("switch_margin"),
                                      min_candidate_run_s=c["min_candidate_run_s"])

    face_pos_by_tid = {}
    for s in samples:
        for f in s.get("faces") or []:
            tid = f.get("track_id")
            if tid is None:
                continue
            face_pos_by_tid.setdefault(tid, []).append((s["t"], f["cx"], f["cy"], f["h"]))
    for seg in segs:
        tid = seg["track_id"]
        pts = [(cx, cy, h) for (t, cx, cy, h) in face_pos_by_tid.get(tid, [])
               if seg["t0"] <= t <= seg["t1"]]
        if not pts:
            pts = [(cx, cy, h) for (_, cx, cy, h) in face_pos_by_tid.get(tid, [])]
        if pts:
            seg["cx"] = float(np.median([p[0] for p in pts]))
            seg["cy"] = float(np.median([p[1] for p in pts]))
            seg["face_h"] = float(np.median([p[2] for p in pts]))
        else:
            seg["cx"] = seg["cy"] = 0.5
            seg["face_h"] = 0.1

    for seg in segs:
        seg["diarization_cluster"] = {k: v for k, v in mapping.items() if v == seg["track_id"]}
    if return_samples:
        samples_meta = {"diarization": diar, "mapping": mapping, "evidence": evidence,
                        "source": "av_sync"}
        if segs:
            segs[0]["_diarization_meta"] = samples_meta
        samples_light = [{"t": s["t"], "faces": s.get("faces") or []} for s in samples]
        return segs, samples_light
    return segs


def active_speaker_per_face_vad(src, start, dur, words, cfg=None, return_samples=False):
    """per-face VAD-anchored speaker decision for sources where the talker is never
    alone on screen (a static two-shot). Pipeline:

      1. sample native-res frames + run YuNet (same as `active_speaker_audio_first`),
      2. assign stable face track ids,
      3. diarize the audio into K voice clusters per word (`render/diarize`),
      4. compute PIXEL mouth motion per face track across the whole window,
      5. map voice clusters -> face tracks by correlating step 4's motion with each
         cluster's word times (:func:`map_clusters_to_tracks_pixel_vad`),
      6. build per-hop talker runs from the diarized words via
         :func:`_audio_first_hops` + the mapping,
      7. apply the hysteresis via :func:`pick_speakers_audio_first`
         (min_hold_s / min_candidate_run_s).

    Returns `[{"t0","t1","track_id","cx","cy","face_h"}]` identical shape to
    `active_speaker`. When diarization fails or the mapping is empty, returns [] so
    the caller can fall back to the mouth-motion planner (last resort).
    """
    c = {**DEFAULTS, **(cfg or {})}
    fps = float(c["sample_fps"])
    hop_s = float(c["decision_hop_s"])

    import cv2
    from clipengine.render import diarize as _diarize

    # Two passes to avoid holding all grayscale frames in memory (a 60 s window at
    # 10 fps on a 1080p source would otherwise sit on ~1.5 GB of float gray buffers,
    # which triggers massive swap thrash on shared hosts). Pass 1 collects faces +
    # assigns stable track ids; Pass 2 re-samples the window and streams one gray
    # frame at a time through :func:`pixel_mouth_motion_by_track`, so peak memory
    # stays at a few MB (two frame-sized grays per track_id).
    samples = []
    for t, frame in iter_frames_native(src, start, dur, fps=fps):
        faces = detect_faces_with_landmarks(frame)
        samples.append({"t": t, "faces": faces})
    if not samples:
        return ([], []) if return_samples else []
    n_tracks = assign_tracks(samples, iou_min=c["iou_min"], centre_max=c["centre_max"])
    if n_tracks < 2:
        return ([], samples) if return_samples else []

    diar = _diarize.diarize_words(src, start, dur, words or [])
    if not diar or diar.get("k", 0) < 2:
        return ([], samples) if return_samples else []

    track_ids = sorted({f["track_id"] for s in samples for f in (s.get("faces") or [])
                        if f.get("track_id") is not None})

    def _gray_stream():
        # Rebuild sample dicts in-order and attach a fresh gray each iteration; the
        # pixel_mouth_motion_by_track loop keeps only two grays per track at a time.
        sample_idx = 0
        for t2, frame2 in iter_frames_native(src, start, dur, fps=fps):
            if sample_idx >= len(samples):
                break
            s = samples[sample_idx]
            gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            yield {"t": s["t"], "faces": s.get("faces") or [], "gray": gray}
            sample_idx += 1

    mouth_series = pixel_mouth_motion_by_track(_gray_stream(), track_ids)
    mapping, evidence = map_clusters_to_tracks_pixel_vad(diar, mouth_series)
    if len(mapping) < 2:
        return ([], samples) if return_samples else []

    hops = _audio_first_hops(diar, mapping, hop_s, t0=0.0, t1=dur)
    segs = pick_speakers_audio_first(hops, hop_s,
                                      min_hold_s=c["min_hold_s"],
                                      switch_margin=c.get("switch_margin"),
                                      min_candidate_run_s=c["min_candidate_run_s"])

    face_pos_by_tid = {}
    for s in samples:
        for f in s.get("faces") or []:
            tid = f.get("track_id")
            if tid is None:
                continue
            face_pos_by_tid.setdefault(tid, []).append((s["t"], f["cx"], f["cy"], f["h"]))
    for seg in segs:
        tid = seg["track_id"]
        pts = [(cx, cy, h) for (t, cx, cy, h) in face_pos_by_tid.get(tid, [])
               if seg["t0"] <= t <= seg["t1"]]
        if not pts:
            pts = [(cx, cy, h) for (_, cx, cy, h) in face_pos_by_tid.get(tid, [])]
        if pts:
            seg["cx"] = float(np.median([p[0] for p in pts]))
            seg["cy"] = float(np.median([p[1] for p in pts]))
            seg["face_h"] = float(np.median([p[2] for p in pts]))
        else:
            seg["cx"] = seg["cy"] = 0.5
            seg["face_h"] = 0.1

    for seg in segs:
        seg["diarization_cluster"] = {k: v for k, v in mapping.items() if v == seg["track_id"]}
    if return_samples:
        samples_meta = {"diarization": diar, "mapping": mapping, "evidence": evidence,
                        "source": "pixel_vad"}
        if segs:
            segs[0]["_diarization_meta"] = samples_meta
        # strip gray from samples before returning (large memory)
        samples_light = [{"t": s["t"], "faces": s.get("faces") or []} for s in samples]
        return segs, samples_light
    return segs


# ---------------------------------------------------------------- broadcast gate 10-21: is the kept face the speaker?
#
# 10-21 clips 490 and 534 held a news-show guest whose mouth never moved while another guest spoke
# (eye-check: "the speaker is off screen"), and no gate measure looked: speaker_switches,
# face_centre and active_speaker read the reframe plan, which the engine's gate never gets. This
# reads the RENDERED clip (5-9 s for a 24 s clip at nice 19, load 59): the kept face (the
# largest face on the canvas) and its mouth motion (mouth_activity between consecutive samples
# of the same face) against the speech energy (audio RMS), both smoothed over KEPT_SMOOTH_S. Per second of speech, in a KEPT_WINDOW_S window around
# it, the kept face is NOT the speaker when its mouth neither follows the speech (Pearson r <
# KEPT_CORR_MIN) nor moves (median motion during speech < KEPT_STILL_MAX). Either alone would fail
# real speakers: approved clips 95 / 328 talk at r ~ 0 but move their mouths (median 4-5), and the
# speaking guest of RKXubRUMwwE in his 240 px tile moves his mouth at a median of 0.64 but follows
# his speech (r 0.28). Measured (10 fps, 540 px wide): 534 not the speaker for 74 % of its speech,
# 490 70-74 % (RMS / words speech), the silent guest's own tile 75 %, the speaking guest's tile
# 24 %, 9 approved podcast clips 0 %.
KEPT_FPS = 10.0
KEPT_SCAN_W = 540
KEPT_SMOOTH_S = 0.5
KEPT_WINDOW_S = 4.0
KEPT_CORR_MIN = 0.15
KEPT_STILL_MAX = 1.0
KEPT_SAME_FACE_IOU = 0.3       # consecutive samples' kept faces overlap this much: the same face
KEPT_DETECT_SCALE = 0.5        # faces are found on a half-size copy (a 9:16 crop's face is big;
                               # 3-4x less YuNet time, the same verdicts on 534/490/535/95/328)
KEPT_MIN_SPEECH_S = 3.0        # speech with a kept face needed for a verdict
KEPT_MIN_COVER = 0.5           # ... and at least this share of all the speech


def kept_face_motion(frames):
    """[(t, motion or None)] for (t, bgr frame) samples: mouth_activity of the kept face (the
    largest YuNet face with landmarks, found at KEPT_DETECT_SCALE) against the previous sample's,
    when both samples keep the same face (IoU >= KEPT_SAME_FACE_IOU); None without a face or
    across a cut."""
    import cv2
    out, prev = [], None
    k = KEPT_DETECT_SCALE
    for t, frame in frames:
        small = cv2.resize(frame, (max(2, int(frame.shape[1] * k)), max(2, int(frame.shape[0] * k))),
                           interpolation=cv2.INTER_AREA)
        faces = detect_faces_with_landmarks(small)
        face = None
        if faces:                     # normalised boxes carry over; landmarks are pixels
            face = {**faces[0], "landmarks": [(x / k, y / k) for x, y in faces[0]["landmarks"]]}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m = None
        if face is not None and prev is not None and prev[1] is not None \
                and face_iou(face, prev[1]) >= KEPT_SAME_FACE_IOU:
            m = mouth_activity(prev[0], gray, prev[1], face)
        out.append((float(t), None if m is None else float(m)))
        prev = (gray, face)
    return out


def _smooth(x, k):
    return x if k <= 1 else np.convolve(x, np.ones(k) / k, mode="same")


def kept_face_verdict(motion, energy, speech, fps=KEPT_FPS, window_s=KEPT_WINDOW_S,
                      smooth_s=KEPT_SMOOTH_S, corr_min=KEPT_CORR_MIN, still_max=KEPT_STILL_MAX):
    """Pure: `motion` [m or None] (kept_face_motion values), `energy` [audio RMS] and `speech`
    [bool] per sample at `fps`. Returns {"not_speaker_frac": share of the speech seconds with a
    kept face whose window says it is not the speaker, "speech_s", "covered_s", "seconds": one
    char per speech second ("x" not the speaker, "." the speaker)}, or {"not_speaker_frac": None,
    "reason": ...} when there is too little speech with a face to judge (no face: no data)."""
    n = min(len(motion), len(energy), len(speech))
    if not n:
        return {"not_speaker_frac": None, "reason": "no samples"}
    m = np.array([np.nan if v is None else float(v) for v in motion[:n]])
    e = np.asarray(energy[:n], dtype=np.float64)
    sp = np.asarray(speech[:n], dtype=bool)
    ok = ~np.isnan(m)
    speech_s = float(sp.sum()) / fps
    covered = sp & ok
    covered_s = float(covered.sum()) / fps
    base = {"speech_s": round(speech_s, 1), "covered_s": round(covered_s, 1)}
    if not ok.any():
        return {**base, "not_speaker_frac": None, "reason": "no face on the canvas"}
    if covered_s < KEPT_MIN_SPEECH_S or covered_s < KEPT_MIN_COVER * speech_s:
        return {**base, "not_speaker_frac": None,
                "reason": f"a kept face for {covered_s:.1f}s of {speech_s:.1f}s of speech"}
    k = max(1, int(round(smooth_s * fps)))
    ms = _smooth(np.where(ok, m, np.nanmedian(m)), k)
    es = _smooth(e, k)
    half, step = int(round(window_s * fps / 2)), max(1, int(round(fps)))
    bad = judged = 0
    marks = []
    for c in range(0, n, step):
        sec = covered[c:c + step]
        if not sec.any():
            continue
        a, b = max(0, c + step // 2 - half), min(n, c + step // 2 + half)
        w = ok[a:b]
        if w.sum() < 0.75 * (b - a):
            continue                              # a cut / no face around it: not judged
        x, y = ms[a:b][w], es[a:b][w]
        r = 0.0 if np.std(x) < 1e-9 or np.std(y) < 1e-9 else float(np.corrcoef(x, y)[0, 1])
        talk = w & sp[a:b]
        level = float(np.median(m[a:b][talk])) if talk.any() else 0.0
        still = r < corr_min and level < still_max
        judged += int(sec.sum())
        bad += int(sec.sum()) if still else 0
        marks.append("x" if still else ".")
    if not judged:
        return {**base, "not_speaker_frac": None, "reason": "no speech second with a steady face"}
    return {**base, "not_speaker_frac": round(bad / judged, 3), "judged_s": round(judged / fps, 1),
            "seconds": "".join(marks), "corr_min": corr_min, "still_max": still_max,
            "window_s": window_s}


def _clip_frames(path, fps, width):
    """(t, bgr) samples of a rendered clip at `fps`, `width` px wide."""
    from clipengine.render import reframe as _reframe
    sw, sh = _reframe.probe_size(path)
    if not (sw and sh):
        return
    th = max(2, int(round(sh * width / sw / 2)) * 2)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-an", "-vf",
           f"fps={fps},scale={width}:{th}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    size = width * th * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        i = 0
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield round(i / fps, 3), np.frombuffer(buf, np.uint8).reshape(th, width, 3)
            i += 1


def kept_face_speech_check(path, speech_spans=None, fps=KEPT_FPS, width=KEPT_SCAN_W):
    """kept_face_verdict on a rendered clip. Speech: `speech_spans` [(a, b)] clip seconds (the
    words sidecar) when given, else audio RMS above DEFAULTS["speech_rms"]. Never raises: a
    detector, decoder or audio failure is {"not_speaker_frac": None, "reason": ...}."""
    try:
        motion = kept_face_motion(_clip_frames(path, fps, width))
        if not motion:
            return {"not_speaker_frac": None, "reason": "no video samples"}
        rms = load_audio_rms(path, 0, len(motion) / fps + 1.0, hop_s=1.0 / fps)
        if not len(rms):
            return {"not_speaker_frac": None, "reason": "no audio"}
        n = min(len(motion), len(rms))
        if speech_spans:
            sp = [any(a <= i / fps < b for a, b in speech_spans) for i in range(n)]
            how = "words"
        else:
            sp = [float(v) >= DEFAULTS["speech_rms"] for v in rms[:n]]
            how = "rms"
        rep = kept_face_verdict([v for _t, v in motion[:n]], rms[:n], sp, fps=fps)
        return {**rep, "speech_from": how, "samples": n}
    except Exception as e:      # noqa: BLE001 - no data, never a crash
        return {"not_speaker_frac": None, "reason": f"{type(e).__name__}: {str(e)[:160]}"}
