#!/usr/bin/env python3
"""Hard quality gate: runs on the RENDERED mp4 between the render stage and the queue stage.

Fail = re-render once with the fallback plan, then DROP (never queue) with a reason stored
on the clip row. A dropped clip is never shipped.

Measurements (all accumulate into one ``report`` dict so a status view can show them):

* ``frame_gate``        — >=98% of 10 fps frames contain a full face fully inside the frame
  AND 100% of frames in the first/last 1.0 s pass. Card-pull windows (from the transcript)
  are exempt by design.
* ``face_center``       — talker face centre within +-10% of crop width on >=95% of frames.
* ``crop_step``         — crop_x(t) step <= 1.5% of source width/frame except at `cut=True`
  segment boundaries.
* ``same_subject_jumps``— zero (every mid-shot x jump beyond SWITCH_MIN must be a hard cut).
* ``speaker_switches``  — every switch target_tid matches the diarization segment at that t.
* ``av_sync``           — abs(video_duration - audio_duration) < 40 ms.
* ``word_boundaries``   — clip start/end sit within +-0.05 s of a word boundary (clean start
  AND clean end). This mirrors the dangling-``Like,`` rule.
* ``caption``           — captions_tiktok.validate returns no errors against the campaign's
  required_elements.
* ``boxed_slices`` / ``cover_frame`` / ``kept_face_speaker`` (broadcast gate 10-21) — no
  neighbouring video box sliced by the crop, no cover frame (first 0.5 s) off the clip's steady
  layout, and the kept face is the speaker for >= 60% of the speech.

The pure helpers do the measurements; ``gate(video_path, plan, clip_meta, campaign)`` is the
orchestrator. ``decide(report)`` collapses the whole report to a pass/fail + ``drop_reason``.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from clipengine import config
from clipengine.render import finish                 # face_fully_inside, head_tail check (shared defs)
from clipengine import captions_tiktok        # caption validator
from clipengine.render import edgebands              # flat edge bands (render/edgebands.py, shared with the render)

FRAME_FPS = 10.0
EDGE_PX = finish.EDGE_PX_DEFAULT
MIN_FACE_H = 0.06                 # smaller than this is not "the subject"
HEAD_TAIL_WINDOW_S = 1.0          # first/last window where every frame must pass
FAIL_PCT_MAX = 2.0                # fail_pct <= this value overall
FACE_CENTRE_OFFSET_MAX = 0.10     # +-10% of crop width
FACE_CENTRE_PASS_FRAC = 0.95
CROP_STEP_MAX_PCT = 0.015         # 1.5% of src_w per frame outside cuts
AV_SYNC_MS = 40.0
WORD_BOUNDARY_INSIDE_TOL_S = 0.03    # a cut this close to a word edge (from the inside)
                                     # counts as cutting INTO the word; outside this margin the cut
                                     # sits in true silence/inter-word gap and passes
WORD_BOUNDARY_TOL_S = WORD_BOUNDARY_INSIDE_TOL_S

# Auto b-roll detector: a contiguous no-face span is treated as b-roll (and exempted) when
# >= this many seconds. Tightened from the original value of 1.5 s so handheld shots
# where the subject drifts out of frame are NOT silently exempted.
AUTO_BROLL_MIN_SPAN_S = 3.0
AUTO_BROLL_MAX_TOTAL_FRACTION = 0.5    # total auto-exempt time can't exceed half the clip

# downscale factors tried on a ``none`` frame before it counts as subject loss
# (see _rescue_none_frame). 0.5 alone recovered 53/55 misses on a real 9:16 track close-up.
NONE_RETRY_SCALES = (0.5, 1 / 3)
# Rendered layouts with no subject crop: the full source frame (``general``) or a game-only
# crop. Nothing in the render can lose a speaker, so a no-face frame is not a failure there;
# edge-cut faces still fail.
NO_SPEAKER_LAYOUTS = ("general", "game_only")
# The speaker crop a talk clip renders as; with no campaign layout its frame gate reads config
# quality_gate.layouts.track (edge-cut transitions, >= 1 s cutaways)
PLAIN_TRACK_LAYOUT = "track"




def _load_config_layouts():
    """Return the ``quality_gate.layouts`` block from the config, plus its default
    row. These are the production thresholds that live in the engine config — the
    calibration manifest and campaign records only choose which layout to run under, they
    do not define thresholds locally."""
    try:
        cfg = config.load()
    except (OSError, ValueError):
        return {}, _hardcoded_default()
    qg = cfg.get("quality_gate") or {}
    layouts = qg.get("layouts") or {}
    default = {**_hardcoded_default(), **(qg.get("default") or {})}
    return layouts, default


def _hardcoded_default():
    return {
        "gate_fail_pct_max": FAIL_PCT_MAX,
        "gate_head_tol_frames": 0,
        "gate_tail_tol_frames": 0,
        "gate_broll_min_span_s": AUTO_BROLL_MIN_SPAN_S,
        "gate_cutaway_min_span_s": None,
        "gate_skip_frame_gate": False,
        "gate_exempt_edge_cut_transitions": False,
        "gate_exempt_low_motion_frames": False,
        "gate_exempt_head_tail_track_carry": False,
    }


def layout_params(campaign):
    """Resolve the quality-gate thresholds for a clip.

    Order of precedence:

    1. The campaign's layout block in the record (``campaign.layout``) — only if the keys
       are explicit overrides. The convention is: a campaign record NEVER sets thresholds
       directly; it sets ``layout: {"layout": "track"}``. The thresholds come from (2).
    2. `the config` -> ``quality_gate.layouts.<layout_name>``.
    3. `the config` -> ``quality_gate.default``.
    4. Hard-coded module defaults.

    Any explicit ``gate_*`` key on the campaign.layout dict wins last. This keeps thresholds
    out of the calibration manifest and inside the engine config, so production and
    calibration cannot drift.
    """
    layouts, default = _load_config_layouts()
    layout_name = None
    layout_block = {}
    if campaign is not None:
        if isinstance(campaign, dict):
            layout_block = campaign.get("layout") or {}
        else:
            layout_block = getattr(campaign, "layout", None) or {}
        layout_name = str(layout_block.get("layout") or "").strip() or None
    cfg_layout = layouts.get(layout_name) if layout_name else {}
    merged = {**default, **(cfg_layout or {}),
              # per-campaign explicit overrides stay possible, but calibration manifests
              # must NEVER set them (see tests/calibration/manifest.json: no gate_* keys)
              **{k: v for k, v in layout_block.items() if k.startswith("gate_")}}
    return {
        "fail_pct_max":      float(merged["gate_fail_pct_max"]),
        "head_tol_frames":   int(merged["gate_head_tol_frames"]),
        "tail_tol_frames":   int(merged["gate_tail_tol_frames"]),
        "broll_min_span_s":  float(merged["gate_broll_min_span_s"]),
        "cutaway_min_span_s": merged.get("gate_cutaway_min_span_s"),
        "skip_frame_gate":   bool(merged.get("gate_skip_frame_gate", False)),
        "exempt_edge_cut_transitions": bool(merged.get("gate_exempt_edge_cut_transitions", False)),
        "exempt_low_motion_frames":    bool(merged.get("gate_exempt_low_motion_frames", False)),
        "exempt_head_tail_track_carry": bool(merged.get("gate_exempt_head_tail_track_carry", False)),
        "layout_name":       layout_name,
    }


def _layout_defaults():
    """Default params when no campaign/layout info is available. Reads from config so the
    overall gate behaviour is still tunable without editing code."""
    _, default = _load_config_layouts()
    return {
        "fail_pct_max":      float(default["gate_fail_pct_max"]),
        "head_tol_frames":   int(default["gate_head_tol_frames"]),
        "tail_tol_frames":   int(default["gate_tail_tol_frames"]),
        "broll_min_span_s":  float(default["gate_broll_min_span_s"]),
        "cutaway_min_span_s": default.get("gate_cutaway_min_span_s"),
        "skip_frame_gate":   bool(default.get("gate_skip_frame_gate", False)),
        "exempt_edge_cut_transitions": bool(default.get("gate_exempt_edge_cut_transitions", False)),
        "exempt_low_motion_frames":    bool(default.get("gate_exempt_low_motion_frames", False)),
        "exempt_head_tail_track_carry": bool(default.get("gate_exempt_head_tail_track_carry", False)),
        "layout_name":       None,
    }


# ---------------------------------------------------------------- small shared helpers

def face_fully_inside(bbox_xywh, src_w, src_h, pad=EDGE_PX):
    """Re-export for callers that only want quality_gate. Delegates to render.finish."""
    return finish.face_fully_inside(bbox_xywh, src_w, src_h, pad=pad)


# ---------------------------------------------------------------- card-pull / b-roll windows

CARD_PULL_PAT = re.compile(
    r"\b(pull|pulls|pulled|pulling|card|cards|hit|hits|value|psa|box|boxes|"
    r"open|opens|opened|opening|rip|rips|flip|flips|chase|dollar|dollars|"
    r"reveal|revealed|show|showing|hobby|break|breaks|charizard|"
    r"pokemon|pok[eé]mon|rookie|refractor|mojo|magic|topps|chrome|"
    r"serial|numbered|prism|bowman|slab|holo|parallel|auto|"
    r"wolverine|sapphire|optic|donruss|panini|pack|packs|insert|"
    r"sgc|bgs|beckett|graded|gem|mint|loosey|loose|collect|collection|"
    r"collecting|edition|series|debut|prizm|prism|rare|rarest|rarity)\b|"
    r"\$|\d+/\d+|/\d+", re.IGNORECASE)


def auto_card_windows(words, start_s, dur, *, pad=1.2, lead=0.4):
    """Infer card/product-in-hand windows from the clip-relative transcript.

    ``words`` is the words sidecar in SOURCE seconds; ``start_s`` + ``dur`` describe the clip
    window. Returns a list of ``(a, b)`` clip-relative seconds where card words appear."""
    hits = []
    for w in words or []:
        if not CARD_PULL_PAT.search(str(w.get("w", ""))):
            continue
        t = float(w["start"]) - float(start_s)
        if 0.0 <= t < float(dur):
            hits.append(t)
    if not hits:
        return []
    hits.sort()
    wins = [(max(0.0, hits[0] - lead), min(dur, hits[0] + pad))]
    for t in hits[1:]:
        a = max(0.0, t - lead); b = min(dur, t + pad)
        if a <= wins[-1][1]:
            wins[-1] = (wins[-1][0], max(wins[-1][1], b))
        else:
            wins.append((a, b))
    return wins


def auto_broll_windows(fail_times, min_span=AUTO_BROLL_MIN_SPAN_S, pad=0.2):
    """Treat any contiguous span of face-less frames >= ``min_span`` seconds as a b-roll /
    product cutaway: product shots, map/location cuts, etc. (tightened).

    Default ``min_span`` is :data:`AUTO_BROLL_MIN_SPAN_S` (3.0 s) rather than the onboarding
    scripts' 1.5 s. Shorter contiguous no-face spans are more likely to be subject-drift
    failures (the camera panned, the crop didn't follow) than true b-roll, so they stay in
    the fail count."""
    return [(max(0.0, a - pad), b + pad) for (a, b) in _fail_runs(fail_times)
            if (b - a) >= min_span]


def _fail_runs(fail_times):
    """[[first, last]] clip times of each contiguous run of failing frames."""
    if not fail_times:
        return []
    ft = sorted(fail_times)
    runs = [[ft[0], ft[0]]]
    for t in ft[1:]:
        if t - runs[-1][1] <= 0.15:        # ~1 frame apart at 10 fps
            runs[-1][1] = t
        else:
            runs.append([t, t])
    return runs


def cutaway_windows(fail_times, times, min_span, pad=0.2):
    """The no-face spans >= ``min_span`` seconds that lie INSIDE the clip, with a
    sampled subject frame before and after (a cut away from the speaker and back: a map, a
    b-roll insert). The b-roll rule takes 3 s, so a 1-2 s insert failed a track clip. A span
    that opens or closes the clip is a dead opening / closing, never a cutaway. ``fail_times``
    are the raw failing frames' times, ``times`` every sampled frame's."""
    if not fail_times or not times:
        return []
    lo, hi = min(times), max(times)
    return [(max(0.0, a - pad), b + pad) for (a, b) in _fail_runs(fail_times)
            if (b - a) >= min_span and a > lo + 1e-6 and b < hi - 1e-6]


def _merge_windows(windows):
    out = []
    for a, b in sorted(windows):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


# ---------------------------------------------------------------- ffprobe helpers

def probe_duration(path, kind="format"):
    sel = []
    if kind == "video":
        sel = ["-select_streams", "v:0"]; q = "stream=duration"
    elif kind == "audio":
        sel = ["-select_streams", "a:0"]; q = "stream=duration"
    else:
        q = "format=duration"
    r = subprocess.run(["ffprobe", "-v", "error", *sel, "-show_entries", q,
                        "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                       capture_output=True, text=True)
    try:
        return float((r.stdout or "0").strip() or 0.0)
    except ValueError:
        return 0.0


def probe_size(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "json", str(path)],
                       capture_output=True, text=True)
    try:
        s = json.loads(r.stdout or "{}")["streams"][0]
        return int(s["width"]), int(s["height"])
    except (ValueError, KeyError, IndexError):
        return 0, 0


# ---------------------------------------------------------------- frame iteration

def iter_frames(path, fps=FRAME_FPS):
    """Yield ``(t, frame_bgr, sw, sh)`` at ``fps``. Imports numpy lazily."""
    import numpy as np
    sw, sh = probe_size(path)
    if sw == 0 or sh == 0:
        return
    size = sw * sh * 3
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
           "-an", "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        i = 0
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield round(i / fps, 3), np.frombuffer(buf, np.uint8).reshape(sh, sw, 3), sw, sh
            i += 1


# ---------------------------------------------------------------- measurement: frame gate

def _bbox_touches_edge(bbox_xywh, src_w, src_h, pad=EDGE_PX):
    """True when ANY face bbox edge is within pad px of a frame edge (= edge-cut face)."""
    x, y, w, h = bbox_xywh
    return (x <= pad or y <= pad or (x + w) >= (src_w - pad) or (y + h) >= (src_h - pad))


def _classify_frame(faces_raw, sw, sh):
    """Classify a frame as:
        ``full``      — a face is fully inside the frame (passes)
        ``edge_cut``  — a face is detected but its bbox touches a frame edge (= mid-pan)
        ``none``      — no detectable big face at all

    Big-face threshold is ``MIN_FACE_H`` of frame height. Score < ``MIN_SCORE`` filtered by
    the detector."""
    big = [f for f in faces_raw if f.get("h", 0.0) >= MIN_FACE_H]
    if not big:
        return "none"
    for f in big:
        fw = f.get("w", 0.0) * sw
        fh = f.get("h", 0.0) * sh
        x = (f["cx"] * sw) - fw / 2
        y = (f["cy"] * sh) - fh / 2
        if face_fully_inside((x, y, fw, fh), sw, sh):
            return "full"
    # at least one big face present, but every one is edge-cut
    return "edge_cut"


def _rescue_none_frame(frame, sw, sh, detect, scales=NONE_RETRY_SCALES):
    """re-detect a ``none`` frame on downscaled copies. YuNet's anchors top
    out well below the face of a 9:16 close-up (a track-cropped speaker fills ~half of the
    1080 px width), so at native res it returns nothing on frames where the face is plainly
    there. Downscaling puts the face back inside the anchor range; the bbox is normalised so
    it maps straight back onto the native ``sw`` x ``sh`` frame and ``_classify_frame`` applies
    the unchanged full/edge-cut rule. Only ``none`` frames come here — an ``edge_cut`` frame
    is never retried, so a face-cut frame still fails. Returns the first non-``none`` label,
    or ``none``."""
    try:
        import cv2   # noqa: WPS433
    except ImportError:
        return "none"
    for s in scales:
        w, h = max(2, int(round(sw * s))), max(2, int(round(sh * s)))
        label = _classify_frame(detect(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)),
                                sw, sh)
        if label != "none":
            return label
    return "none"


def _exempt_short_noise_gaps(classified, fps, max_gap_s=0.5):
    """Exempt very short no-face gaps (<= ``max_gap_s``) that are bounded by face frames on
    both sides. These are detector noise (one or two frames the face detector missed),
    not actual subject loss. A clip-boundary gap is NOT exempted here — those require the
    layout-specific logic (bounded transition / bounded product cutaway)."""
    exempt = set()
    n = len(classified)
    i = 0
    while i < n:
        if classified[i] != "none":
            i += 1
            continue
        j = i
        while j < n and classified[j] == "none":
            j += 1
        left_face = (i > 0 and classified[i - 1] in ("full", "edge_cut"))
        right_face = (j < n and classified[j] in ("full", "edge_cut"))
        span_s = (j - i) / fps
        if left_face and right_face and span_s <= max_gap_s:
            exempt.update(range(i, j))
        i = j if j > i else i + 1
    return exempt


def _stable_anchor_before(classified, i, min_run_frames):
    """True when classified[i-min_run_frames : i] is all ``full``. A single stray face
    detection does NOT count as an anchor — otherwise an isolated detector hit at frame 0
    would make a 3 s dead opening look like a bounded transition."""
    if i < min_run_frames:
        return False
    return all(classified[k] == "full" for k in range(i - min_run_frames, i))


def _stable_anchor_after(classified, j, min_run_frames, n):
    """True when classified[j : j+min_run_frames] is all ``full``."""
    if j + min_run_frames > n:
        return False
    return all(classified[k] == "full" for k in range(j, j + min_run_frames))


def _apply_edge_cut_transitions(classified, fps, max_transition_s=4.0,
                                 max_no_face_within_transition_s=3.0,
                                 max_boundary_transition_s=1.5,
                                 min_anchor_s=0.1,
                                 min_subject_established_s=1.0):
    """Smart exemption for podcast_two_shot layouts.

    A ``edge_cut`` frame means a face is on-screen but the crop is mid-pan between speakers.
    A contiguous run of ``edge_cut`` or ``none`` frames that is BOUNDED on both sides by a
    face anchor (>= ``min_anchor_s``) within ``max_transition_s`` is a legitimate transition
    window and gets exempted — PROVIDED the cluster does not start within the first
    ``min_subject_established_s`` of the clip. A dead opening (cluster starting before the
    subject is established) is always a fail, no matter what anchor is detected around it.
    The sub-run of pure ``none`` within one transition is capped at
    ``max_no_face_within_transition_s``.

    Clip start / end are implicit boundaries: a brief (<= ``max_boundary_transition_s``)
    transition at either end with a face anchor on the inside is exempt.

    Returns a set of frame indices to exempt."""
    exempt = set()
    n = len(classified)
    min_anchor_frames = max(1, int(round(min_anchor_s * fps)))
    established_i = int(round(min_subject_established_s * fps))
    i = 0
    while i < n:
        if classified[i] == "full":
            i += 1
            continue
        # find next 'full' marker (right boundary)
        j = i
        while j < n and classified[j] != "full":
            j += 1
        left_anchor = _stable_anchor_before(classified, i, min_anchor_frames)
        right_anchor = _stable_anchor_after(classified, j, min_anchor_frames, n)
        # Reject exemption for clusters that start before the subject is established.
        # A 1-frame "full" at index 0 doesn't count as establishing the subject.
        subject_established = i >= established_i
        span_s = (j - i) / fps
        cur_none, max_none = 0, 0
        for k in range(i, j):
            if classified[k] == "none":
                cur_none += 1
                max_none = max(max_none, cur_none)
            else:
                cur_none = 0
        no_face_ok = max_none / fps <= max_no_face_within_transition_s
        if (left_anchor and right_anchor and span_s <= max_transition_s
                and no_face_ok and subject_established):
            exempt.update(range(i, j))
        elif no_face_ok and span_s <= max_boundary_transition_s and (
             (i == 0 and right_anchor) or (j == n and left_anchor)):
            exempt.update(range(i, j))
        i = j if j > i else i + 1
    return exempt


LOW_MOTION_THRESHOLD = 15.0   # mean |delta| per pixel/channel (0..255 scale); below = static


# a stacked chart stream opens on a small cam-panel face that YuNet sometimes misses in the first
# 0.5 s — motion blur during the pour-in, the face in a slightly off angle, etc. The spec
# forbids widening head_tol; instead we add two detection fixes:
#
#   1. Upscale-retry: for each frame classified ``none`` within the head/tail window,
#      decode the single frame, upscale it (1.5x by default) and re-run face detection
#      at native score threshold. YuNet's confidence picks up a half-stop on upscale and
#      often catches a face it missed at the native sample.
#   2. Track-carry-forward: if the retry still shows ``none`` but a stable face anchor
#      appears RIGHT AFTER (>= ``min_anchor_s`` of ``full`` frames), the subject IS there
#      — the detector just missed the opening frames. Exempt them.
#
# Both are opt-in via the layout flag ``exempt_head_tail_track_carry`` (chart_top_cam_bottom).

HEAD_TAIL_RETRY_SCALE = 1.5
HEAD_TAIL_CARRY_FORWARD_S = 1.0           # how far after a 'none' frame to look for an anchor
HEAD_TAIL_CARRY_ANCHOR_S = 0.5            # length of the stable anchor needed to carry

def _retry_classify_frame_at(video_path, t, scale=HEAD_TAIL_RETRY_SCALE):
    """decode the SINGLE frame at time ``t`` from ``video_path``, upscale it by
    ``scale`` using the ffmpeg scale filter + lanczos (same quality as the main
    render path), and return the raw face list from ``reframe.detect_faces``. The
    upscale catches small faces YuNet misses at native sample res (small cam-panel
    streamers in the opening second of a chart_top_cam_bottom clip). Returns an
    empty list on any decoder / detector error.
    """
    try:
        import numpy as np                   # noqa: F401
    except ImportError:
        return []
    try:
        from clipengine.render import reframe   # noqa: WPS433
    except ImportError:
        return []
    try:
        sw, sh = probe_size(video_path)
    except Exception:                        # noqa: BLE001
        return []
    if sw == 0 or sh == 0:
        return []
    import numpy as np
    out_w = max(1, int(round(sw * float(scale))))
    out_h = max(1, int(round(sh * float(scale))))
    size = out_w * out_h * 3
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{float(t):.3f}",
           "-i", str(video_path), "-frames:v", "1",
           "-vf", f"scale={out_w}:{out_h}:flags=lanczos",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    try:
        buf = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    if len(buf) < size:
        return []
    frame = np.frombuffer(buf[:size], np.uint8).reshape(out_h, out_w, 3)
    try:
        return reframe.detect_faces(frame)
    except Exception:                        # noqa: BLE001
        return []


def _apply_head_tail_track_carry(classified, times, fps, video_path,
                                  head_s=HEAD_TAIL_WINDOW_S,
                                  tail_s=HEAD_TAIL_WINDOW_S,
                                  carry_s=HEAD_TAIL_CARRY_FORWARD_S,
                                  min_anchor_s=HEAD_TAIL_CARRY_ANCHOR_S,
                                  retry_scale=HEAD_TAIL_RETRY_SCALE,
                                  src_wh=None):
    """two-pass head/tail detection recovery.

    Pass A (upscale-retry): for each frame in [0, head_s] or [dur-tail_s, dur] whose
    initial classification is ``none``, decode the frame at native res, upscale it
    by ``retry_scale`` with lanczos, run YuNet again. On a hit, flip ``classified[i]``
    to ``full``. The initial pass used the native-res sample via :func:`iter_frames`
    which is good enough for talking-head sources but misses small cam-panel faces
    at the start of chart_top_cam_bottom clips (one chart clip's 0.3/0.4/0.5 drops).

    Pass B (track-carry-forward): for each ``none`` frame remaining in the head/tail
    window, check whether a stable anchor of ``min_anchor_s`` ``full`` frames begins
    within ``carry_s`` seconds. If yes, add the frame index to the exempt set so the
    gate credits the opening to the detector's miss rather than a missing subject.

    ``src_wh`` lets callers pass the already-probed frame size; falls through to
    :func:`probe_size` when None.

    Returns ``(flips, exempt_idx)`` — ``flips`` is a set of indices whose
    ``classified[i]`` was flipped from ``none`` to ``full`` in Pass A (the caller
    mutates ``classified`` with these), ``exempt_idx`` is a set of indices to add to
    the smart-exempt set.
    """
    n = len(classified)
    flips = set()
    exempt = set()
    if n == 0:
        return flips, exempt
    dur_s = times[-1] + 1.0 / fps if times else 0.0
    tail_start_s = max(0.0, dur_s - tail_s)
    anchor_frames = max(1, int(round(min_anchor_s * fps)))
    def _in_head_tail(i):
        t = times[i]
        return t < head_s or t > tail_start_s
    # Pass A: upscale retry on 'none' head/tail frames.
    for i in range(n):
        if classified[i] != "none" or not _in_head_tail(i):
            continue
        faces = _retry_classify_frame_at(video_path, times[i], scale=retry_scale)
        if not faces:
            continue
        big = [f for f in faces if f.get("h", 0.0) >= MIN_FACE_H]
        if not big:
            continue
        # If any face is fully inside the (upscaled) frame, call it ``full``. The
        # bbox is in normalized coordinates so the fit check is scale-invariant.
        for f in big:
            if (f["cx"] - f["w"] / 2) >= 0.002 and (f["cy"] - f["h"] / 2) >= 0.002 \
               and (f["cx"] + f["w"] / 2) <= 0.998 and (f["cy"] + f["h"] / 2) <= 0.998:
                classified[i] = "full"
                flips.add(i)
                break
        else:
            # face present but edge-cut on the upscaled frame — still better than 'none'
            classified[i] = "edge_cut"
            flips.add(i)
    # Pass B: carry-forward exemption for remaining 'none' head/tail frames.
    # Rule: a 'none' head frame is exempt ONLY when the next stable 'full' anchor
    # (``min_anchor_s`` of continuous 'full' frames) begins within ``min_anchor_s``
    # of this frame. A 1 s dead opening does NOT qualify: a long 'none' span before
    # the anchor is a real subject miss, not a brief detector slip. Mirror logic at
    # the tail (anchor must END within ``min_anchor_s`` AFTER the 'none' frame).
    gap_frames = max(1, int(round(min_anchor_s * fps)))
    carry_frames = max(1, int(round(carry_s * fps)))
    def _find_anchor_start_within(i, max_lookahead):
        """Return the frame index where a 'full' anchor of ``anchor_frames`` starts
        between i+1 and i+1+max_lookahead, or None."""
        last = min(n, i + 1 + max_lookahead)
        for start in range(i + 1, last):
            if start + anchor_frames > n:
                return None
            if all(classified[k] == "full" for k in range(start, start + anchor_frames)):
                return start
        return None
    def _find_anchor_end_within(i, max_lookbehind):
        """Return the LAST frame index of a 'full' anchor ending between i-1 and
        i-1-max_lookbehind, or None."""
        first = max(-1, i - 1 - max_lookbehind)
        for end in range(i - 1, first, -1):
            if end - anchor_frames + 1 < 0:
                return None
            if all(classified[k] == "full" for k in range(end - anchor_frames + 1,
                                                           end + 1)):
                return end
        return None
    for i in range(n):
        if classified[i] != "none" or not _in_head_tail(i):
            continue
        in_tail = times[i] > tail_start_s
        if not in_tail:
            start = _find_anchor_start_within(i, carry_frames)
            if start is not None and (start - i) <= gap_frames:
                exempt.add(i)
            continue
        end = _find_anchor_end_within(i, carry_frames)
        if end is not None and (i - end) <= gap_frames:
            exempt.add(i)
    return flips, exempt



def _apply_bounded_product_cutaways(classified, motion, fps, max_cutaway_s=5.0,
                                     max_boundary_cutaway_s=2.0,
                                     low_motion_threshold=LOW_MOTION_THRESHOLD,
                                     min_anchor_s=0.5):
    """Smart exemption for product_full_bleed layouts.

    Two kinds of frames count as subject-visible for a product campaign:
      1. A frame with a face (full or edge-cut).
      2. A frame with low scene motion (the camera is holding on the product).

    Exempt any failing frame that is EITHER low-motion OR inside a bounded cutaway: a
    contiguous no-face run bounded on both sides by a STABLE run of face-present frames
    (>= ``min_anchor_s``) within ``max_cutaway_s``. Clip boundaries (start / end) are
    implicit anchors — a brief product opening or closing (<= ``max_boundary_cutaway_s``)
    with a stable face-run inside is also exempt. A dead opening with no face in the first
    2 s still fails because the inside anchor is missing.
    """
    exempt = set()
    n = len(classified)
    min_anchor_frames = max(1, int(round(min_anchor_s * fps)))
    # Rule 1: low-motion frames are subject-visible even without a face.
    for i, c in enumerate(classified):
        if c == "full":
            continue
        if i < len(motion) and float(motion[i]) < low_motion_threshold:
            exempt.add(i)
    # Rule 2: bounded cutaways. A sweep over non-face stretches that are flanked by face
    # presence (full OR edge-cut) within max_cutaway_s. Clip boundaries get a shorter
    # max_boundary_cutaway_s. Anchors must be stable runs (not single stray detections).
    face_present = [c in ("full", "edge_cut") for c in classified]
    def _stable_face_before(idx):
        return idx >= min_anchor_frames and all(
            face_present[k] for k in range(idx - min_anchor_frames, idx))
    def _stable_face_after(idx):
        return idx + min_anchor_frames <= n and all(
            face_present[k] for k in range(idx, idx + min_anchor_frames))
    i = 0
    while i < n:
        if face_present[i]:
            i += 1
            continue
        j = i
        while j < n and not face_present[j]:
            j += 1
        left_face = _stable_face_before(i)
        right_face = _stable_face_after(j)
        span_s = (j - i) / fps
        if left_face and right_face and span_s <= max_cutaway_s:
            exempt.update(range(i, j))
        elif span_s <= max_boundary_cutaway_s and (
             (i == 0 and right_face) or (j == n and left_face)):
            exempt.update(range(i, j))
        i = j if j > i else i + 1
    return exempt


def _frame_motion_series(video_path, fps=FRAME_FPS):
    """Mean absolute pixel difference between consecutive 10 fps frames (scaled to 320 wide
    for speed). Returns a list where index ``i`` is the motion AT sample i (i.e. diff vs
    sample i-1; index 0 is 0.0). Smaller values => a more static shot."""
    import numpy as np
    sw, sh = probe_size(video_path)
    if sw == 0 or sh == 0:
        return []
    target_w = 320
    target_h = max(2, int(round(sh * target_w / sw / 2)) * 2)
    size = target_w * target_h * 3
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video_path),
           "-an", "-vf", f"fps={fps},scale={target_w}:{target_h}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    diffs = [0.0]
    prev = None
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(target_h, target_w, 3).astype(np.int16)
            if prev is not None:
                diffs.append(float(np.abs(frame - prev).mean()))
            prev = frame
    return diffs


LOW_MOTION_THRESHOLD = 4.0        # mean |delta| per pixel/channel (0-255 scale); below = static




def measure_frame_gate(video_path, card_pull_windows=None, fps=FRAME_FPS, params=None):
    """Scan the rendered 9:16 mp4 and classify every sampled frame as full/edge_cut/none,
    then apply the smart exemptions declared by the layout.

    When ``params['skip_frame_gate']`` is True the function returns ``{"passed": None,
    "reason": "skip_frame_gate"}`` without reading pixels (gamestack layouts).
    """
    params = params or _layout_defaults()
    if params.get("skip_frame_gate"):
        return {"passed": None, "reason": "skip_frame_gate", "params": params,
                "note": "layout explicitly skips the frame gate (gameplay footage, etc.)"}
    from clipengine.render import reframe   # noqa: WPS433
    cards = card_pull_windows or []
    dur = probe_duration(video_path, "video") or probe_duration(video_path)
    classified = []           # per-frame label
    times = []                # per-frame clip time
    rescued = 0               # 'none' frames recovered by the downscaled re-detect
    for t, frame, sw, sh in iter_frames(video_path, fps=fps):
        times.append(round(t, 3))
        if any(a <= t < b for (a, b) in cards):
            classified.append("full")   # card-pull windows are subject-visible by rule
            continue
        faces_raw = reframe.detect_faces(frame)
        label = _classify_frame(faces_raw, sw, sh)
        if label == "none":
            label = _rescue_none_frame(frame, sw, sh, reframe.detect_faces)
            rescued += label != "none"
        classified.append(label)
    total = len(classified)
    if total == 0:
        return {"passed": False, "errors": ["no frames read"], "params": params}

    # Raw fail indices: anything not "full". A speaker-less layout (params require_subject
    # False) only fails on edge-cut faces: there is no subject crop to lose.
    fail_labels = ("none", "edge_cut") if params.get("require_subject", True) else ("edge_cut",)
    raw_fails_idx = [i for i, c in enumerate(classified) if c in fail_labels]

    # Auto b-roll windows from the TIMES of raw fails (contiguous no-face / edge-cut spans
    # >= broll_min_span_s). Tightened + capped total to AUTO_BROLL_MAX_TOTAL_FRACTION of dur.
    fail_times = [times[i] for i in raw_fails_idx]
    broll = auto_broll_windows(fail_times, min_span=float(params["broll_min_span_s"]), pad=0.2)
    if params.get("cutaway_min_span_s"):
        # plain track's shorter cutaways, inside the clip only
        broll = _merge_windows(broll + cutaway_windows(
            fail_times, times, float(params["cutaway_min_span_s"]), pad=0.2))
    broll = _cap_broll_total(broll, dur)
    broll_exempt_idx = {i for i, t in enumerate(times) if any(a <= t <= b for (a, b) in broll)}

    # Layout-specific smart exemptions. Thresholds stay strict; these decide which failing
    # frames are "legit cutaways" vs "lost subject".
    smart_exempt_idx = set()
    if params.get("exempt_edge_cut_transitions"):
        # A podcast two-shot cutaway (speaker-to-speaker pan, broll insert) is bounded on
        # both sides by STABLE runs of face frames (>= min_anchor_s). A single stray
        # detector hit at frame 0 does NOT count as an anchor, so a dead opening still
        # fails. Allow 4 s outer span and up to 3 s of pure no-face inside — enough for
        # legit cutaways, still rejects clips that OPEN or CLOSE on no-face stretches.
        smart_exempt_idx |= _apply_edge_cut_transitions(classified, fps,
                                                         max_transition_s=4.0,
                                                         max_no_face_within_transition_s=3.0,
                                                         max_boundary_transition_s=1.5,
                                                         min_anchor_s=0.1,
                                                         min_subject_established_s=1.0)
        # Also: short detector-noise gaps (<= 0.5 s) bounded by face on both sides. These
        # are not transitions, just the detector missing a frame or two — exempt them
        # unconditionally (brief gaps don't affect subject presence).
        smart_exempt_idx |= _exempt_short_noise_gaps(classified, fps, max_gap_s=0.5)
    motion = []
    if params.get("exempt_low_motion_frames"):
        motion = _frame_motion_series(video_path, fps=fps)
        if motion:
            smart_exempt_idx |= _apply_bounded_product_cutaways(classified, motion, fps,
                                                                 max_cutaway_s=5.0,
                                                                 max_boundary_cutaway_s=2.0,
                                                                 min_anchor_s=0.1)
        # Short detector-noise gaps apply to product content too.
        smart_exempt_idx |= _exempt_short_noise_gaps(classified, fps, max_gap_s=0.5)

    # a stacked chart stream opens on a small cam-panel face that YuNet sometimes misses in the first
    # 0.5 s. Opt-in via ``exempt_head_tail_track_carry`` (chart_top_cam_bottom). Two layered
    # fixes: upscale the frame and re-run the detector (Pass A); if still 'none', exempt
    # when a stable face anchor appears right after (Pass B). Mutates ``classified`` with
    # Pass A flips so the head/tail guard credits them as subject-visible. Short noise gaps
    # (<= 0.5 s bounded by face on both sides) are universal detector artifacts and get
    # exempted together.
    head_tail_flips, head_tail_carry = set(), set()
    if params.get("exempt_head_tail_track_carry"):
        try:
            head_tail_flips, head_tail_carry = _apply_head_tail_track_carry(
                classified, times, fps, video_path)
        except Exception as e:                    # noqa: BLE001
            head_tail_flips, head_tail_carry = set(), set()
            params = {**params, "head_tail_track_carry_error":
                      f"{type(e).__name__}: {e}"}
        smart_exempt_idx |= head_tail_carry
        smart_exempt_idx |= _exempt_short_noise_gaps(classified, fps, max_gap_s=0.5)
        # Pass A flipped some 'none' frames to 'full'; raw_fails_idx was computed BEFORE
        # the flip, so recompute kept_fails_idx by removing the flipped indices.
        if head_tail_flips:
            raw_fails_idx = [i for i in raw_fails_idx if i not in head_tail_flips]
            fail_times = [times[i] for i in raw_fails_idx]

    exempt_idx = broll_exempt_idx | smart_exempt_idx
    kept_fails_idx = [i for i in raw_fails_idx if i not in exempt_idx]
    kept_fail_times = [times[i] for i in kept_fails_idx]
    # Head/tail guard: a dead opening or closing is a hard fail. We DO credit the smart
    # exemptions (low-motion product frames, edge-cut-with-stable-anchor transitions)
    # because they prove the frame is subject-visible via another signal. The crucial point
    # is that a bounded "transition" needs a STABLE INSIDE ANCHOR (see min_anchor_s in the
    # helpers) — a stray single-frame face detection does NOT count. This stops a 3 s
    # face-less opening from exempting itself just because frame 0 happened to register.
    first = [i for i in kept_fails_idx if times[i] < HEAD_TAIL_WINDOW_S]
    last = [i for i in kept_fails_idx if times[i] > max(0.0, dur - HEAD_TAIL_WINDOW_S)]
    fail_pct = round(len(kept_fails_idx) * 100.0 / max(1, total), 2)

    # Pass criterion: STRICT thresholds on every layout. The smart exemptions above are what
    # let legitimate layouts (podcast_two_shot, product_full_bleed) pass without raising
    # the fail_pct cap.
    passed = (fail_pct <= float(params["fail_pct_max"])
              and len(first) <= int(params["head_tol_frames"])
              and len(last) <= int(params["tail_tol_frames"]))

    return {
        "video": str(video_path),
        "dur_s": round(dur, 3),
        "fps_sampled": fps,
        "total_frames": total,
        "fail_count": len(kept_fails_idx),
        "fail_pct": fail_pct,
        "fail_times": kept_fail_times,
        "first_sec_fail_count": len(first),
        "last_sec_fail_count": len(last),
        "passed": passed,
        "card_pull_windows": cards,
        "broll_windows_auto": broll,
        "raw_fail_count": len(raw_fails_idx),
        "edge_cut_count": sum(1 for c in classified if c == "edge_cut"),
        "none_count": sum(1 for c in classified if c == "none"),
        "rescaled_rescue_count": rescued,
        "smart_exempt_count": len(smart_exempt_idx),
        "broll_exempt_count": len(broll_exempt_idx),
        "params": params,
    }


def _cap_broll_total(windows, dur):
    """Trim auto-broll windows so their total exempt time stays below
    ``AUTO_BROLL_MAX_TOTAL_FRACTION`` of the clip duration. Longest windows are kept first.
    """
    if not windows or dur <= 0:
        return list(windows)
    cap = float(dur) * AUTO_BROLL_MAX_TOTAL_FRACTION
    sorted_wins = sorted(windows, key=lambda w: w[1] - w[0], reverse=True)
    kept, used = [], 0.0
    for a, b in sorted_wins:
        span = b - a
        if used + span > cap:
            continue
        used += span
        kept.append((a, b))
    kept.sort()
    return kept


# ---------------------------------------------------------------- measurement: face centre

def measure_face_centre(plan, src_w):
    """Pure: given a reframe plan's smoothed segments and the detected face track, verify that
    the talker face centre lands within +-10% of crop width on >= 95% of samples.

    The plan is expected to carry ``smooth_track``-style segments (dense per-100 ms steps). If
    the plan does not include per-sample face centres (older ``track`` layouts), return
    ``{"passed": None, "reason": "no dense tracker"}`` so this measure is informational, not
    gating. The gate's authoritative check for same-subject jumps is :func:`measure_crop_step`.
    """
    face_samples = (plan or {}).get("face_samples") or []
    segments = (plan or {}).get("segments") or []
    cw = int(plan.get("crop_width") or 0) if plan else 0
    if not face_samples or not segments or cw <= 0:
        return {"passed": None, "reason": "no dense tracker samples in plan"}
    def crop_x_at(t):
        for s in segments:
            if s["t0"] <= t < s["t1"]:
                return s["x"]
        return segments[-1]["x"]
    ok, total = 0, 0
    for fs in face_samples:
        total += 1
        x = crop_x_at(float(fs["t"]))
        centre_px = x + cw / 2.0
        off = abs(float(fs["cx"]) * src_w - centre_px) / cw
        if off <= FACE_CENTRE_OFFSET_MAX:
            ok += 1
    frac = ok / max(1, total)
    return {"passed": frac >= FACE_CENTRE_PASS_FRAC, "frac_ok": round(frac, 4),
            "samples": total}


# ---------------------------------------------------------------- measurement: crop step

def measure_crop_step(plan, src_w):
    """Return the maximum per-step crop_x delta as a fraction of ``src_w`` across non-cut
    segment boundaries. Also counts any same-subject jump: consecutive segments with cut=False
    whose delta still exceeds the 1.5% cap."""
    if not plan or src_w <= 0:
        return {"passed": None, "reason": "no plan or no src_w"}
    segs = plan.get("segments") or []
    worst = 0.0
    jumps = 0
    for a, b in zip(segs, segs[1:]):
        if b.get("cut"):
            continue
        d = abs(float(b["x"]) - float(a["x"])) / float(src_w)
        worst = max(worst, d)
        if d > CROP_STEP_MAX_PCT:
            jumps += 1
    return {"passed": jumps == 0, "max_step_pct": round(worst, 5),
            "same_subject_jumps": jumps}


# ---------------------------------------------------------------- measurement: speaker switches match diarization

def measure_speaker_switches(plan):
    """Every speaker-switch segment's track_id must have been the diarization winner at that
    time; verified by the reframe plan's ``speaker_segments`` + ``diarization`` fields."""
    sp = (plan or {}).get("speaker_segments") or []
    if not sp:
        return {"passed": None, "reason": "no speaker segments"}
    # When reframe populates speaker_segments it has already enforced the margin + hold rules
    # upstream (see render/speaker.pick_speakers); the gate here just asserts the shape is
    # consistent: no empty track_id, strictly monotonic t0, and <= one switch per 1 s.
    bad = []
    prev_t = -1.0
    switches_1s = 0
    last_sw = -10.0
    for s in sp:
        if s.get("track_id") is None:
            bad.append("missing track_id")
        if s["t0"] < prev_t:
            bad.append(f"non-monotonic t0 {s['t0']} < {prev_t}")
        if s["t0"] - last_sw < 1.0:
            switches_1s += 1
        last_sw = s["t0"]
        prev_t = s["t0"]
    return {"passed": not bad and switches_1s <= max(0, len(sp) // 4),
            "n_segments": len(sp), "errors": bad, "rapid_switches": switches_1s}


# ---------------------------------------------------------------- measurement: A/V sync drift

def measure_av_sync(video_path):
    v = probe_duration(video_path, "video")
    a = probe_duration(video_path, "audio")
    drift_ms = abs(v - a) * 1000.0
    return {"passed": drift_ms < AV_SYNC_MS, "drift_ms": round(drift_ms, 1),
            "v_dur_s": round(v, 4), "a_dur_s": round(a, 4)}


# ---------------------------------------------------------------- measurement: word-boundary start/end

def _cut_inside_word(t, words, tol_s=WORD_BOUNDARY_INSIDE_TOL_S):
    """True when the cut time ``t`` lands STRICTLY inside a word AND is at least
    ``tol_s`` from either edge of that word. A cut at the exact word boundary or within
    ``tol_s`` of the edge (effectively at the boundary — detectors + boundary planners
    quantize to a few ms) returns False. A cut strictly inside a word with more than
    ``tol_s`` of margin from either edge is a true mid-word cut and returns True.
    """
    t = float(t); tol = float(tol_s)
    for w in words or []:
        ws, we = float(w["start"]), float(w["end"])
        if ws + tol < t < we - tol:
            return True, w
    return False, None


# The cut's audio: 12 of 15 measured "cut inside word" rejects sat
# in more than 15 dB of silence (the gate's whisper words stretched over the pause: 'I' 0.48 s,
# 'and' 0.60 s, 'like' 1.12 s), the pause boundaries.seat_cuts put them in. A cut whose 20 ms
# frame is quiet_db under the clip's median frame, inside a quiet run of quiet_run_s, is in a
# pause: never inside a word, whatever a transcript's word times say. A word stretched over it is
# read on the side of the cut its voice is on (seat_on_cut_audio), by every cut check: on the
# the reviewed clips the same stretched word that failed word_boundaries also failed the sentence
# start (sentence_bounds read it as the word before the clip's first).
CUT_AUDIO = {"enabled": True, "quiet_db": -15.0, "quiet_run_s": 0.1}
CUT_AUDIO_HOP_S = 0.02
CUT_AUDIO_PAD_S = 1.0             # audio read around the clip, for the frames on both sides of a cut


def cut_audio_settings(cfg=None):
    """config quality_gate.cut_audio over CUT_AUDIO; env CUT_AUDIO=0 turns it off."""
    if cfg is None:
        try:
            cfg = config.load()
        except (OSError, ValueError):
            cfg = {}
    st = {**CUT_AUDIO, **(((cfg.get("quality_gate") or {}).get("cut_audio")) or {})}
    st.pop("_doc", None)
    if os.environ.get("CUT_AUDIO", "").strip() == "0":
        st["enabled"] = False
    return st


def cut_audio_enabled(cfg=None):
    return bool(cut_audio_settings(cfg).get("enabled"))


def _audio_frames(times, db, start_s, end_s):
    """{"times", "db"}: frames (start times, absolute s) with dB relative to the median frame
    inside [start_s, end_s] (the clip's level, as cut_energy's)."""
    import numpy as np
    times, db = np.asarray(times, dtype=np.float64), np.asarray(db, dtype=np.float64)
    inside = (times >= float(start_s) - 1e-9) & (times < float(end_s))
    if len(db) < 2 or not inside.any():
        raise RuntimeError("no audio inside the clip")
    ref = float(np.median(db[inside]))
    return {"times": times.tolist(), "db": (db - ref).tolist(), "hop": CUT_AUDIO_HOP_S}


def cut_audio(src_media, start_s, end_s, pad_s=CUT_AUDIO_PAD_S):
    """cut_in_quiet's frames for the clip [start_s, end_s] of ``src_media``: 20 ms RMS frames from
    ``pad_s`` before the start cut to ``pad_s`` after the end cut, dB relative to the clip's median
    frame. Raises when the audio can't be read."""
    import numpy as np
    sr = 16000
    hop = int(CUT_AUDIO_HOP_S * sr)
    lo = max(0.0, float(start_s) - pad_s)
    hi = float(end_s) + pad_s
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{lo:.3f}",
                          "-t", f"{hi - lo:.3f}", "-i", str(src_media), "-vn", "-ac", "1",
                          "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, timeout=120).stdout
    a = np.frombuffer(raw[:len(raw) // 4 * 4], np.float32)
    n = len(a) // hop
    if n < 2:
        raise RuntimeError("no source audio")
    db = 20 * np.log10(np.sqrt((a[:n * hop].reshape(n, hop).astype(np.float64) ** 2)
                               .mean(axis=1)) + 1e-9)
    return _audio_frames(lo + np.arange(n) * CUT_AUDIO_HOP_S, db, start_s, end_s)


def cut_audio_from_rms(rms, t0, frame_s, start_s, end_s, pad_s=CUT_AUDIO_PAD_S):
    """cut_audio's frames from a window's frame RMS (boundaries.frame_rms: rms[0] starts at t0;
    10 ms frames are joined in pairs into 20 ms ones), so a caller that read the window audio
    already (stages._final_cut) needs no new read. Raises when it covers too little."""
    import numpy as np
    r = np.asarray(rms if rms is not None else [], dtype=np.float64)
    k = max(1, int(round(CUT_AUDIO_HOP_S / float(frame_s))))
    i0 = max(0, int(np.floor((float(start_s) - pad_s - t0) / frame_s)))
    i1 = min(len(r), int(np.ceil((float(end_s) + pad_s - t0) / frame_s)))
    seg = r[i0:i1]
    n = len(seg) // k
    if n < 2:
        raise RuntimeError("no window audio around the clip")
    p = np.sqrt((seg[:n * k] ** 2).reshape(n, k).mean(axis=1))
    db = 20 * np.log10(p + 1e-9)
    times = t0 + (i0 + np.arange(n) * k) * float(frame_s)
    return _audio_frames(times, db, start_s, end_s)


def _quiet_run(times, db, t, hop, quiet_db):
    """(p0, p1) of the run of frames `quiet_db` under the median that holds the frame nearest
    `t` (frames start at `times`, `hop` long): p0 = where its first frame starts, p1 = where the
    voice is back. None when that frame is not quiet or the frames do not reach `t`."""
    if not times:
        return None
    i = min(range(len(times)), key=lambda k: abs(float(times[k]) + hop / 2.0 - float(t)))
    if abs(float(times[i]) + hop / 2.0 - float(t)) > hop or float(db[i]) > quiet_db:
        return None
    a = b = i
    while a > 0 and float(db[a - 1]) <= quiet_db:
        a -= 1
    while b + 1 < len(db) and float(db[b + 1]) <= quiet_db:
        b += 1
    return float(times[a]), float(times[b]) + hop


def _quiet_at(times, db, t, hop, quiet_db, min_run):
    """True when the frame nearest `t` is `quiet_db` under the median, in a quiet run at least
    `min_run` long (_quiet_run)."""
    run = _quiet_run(times, db, t, hop, quiet_db)
    return run is not None and run[1] - run[0] >= min_run - 1e-9


def cut_in_quiet(audio, t, quiet_db=None, min_run=None):
    """True when the cut at `t` (absolute s) sits in a pause on `audio` (cut_audio /
    cut_audio_from_rms frames): its 20 ms frame quiet_db under the clip's median frame, inside a
    quiet run of min_run s (CUT_AUDIO defaults). False without audio there."""
    if not audio:
        return False
    st = CUT_AUDIO
    return _quiet_at(audio["times"], audio["db"], t, float(audio.get("hop", CUT_AUDIO_HOP_S)),
                     float(st["quiet_db"] if quiet_db is None else quiet_db),
                     float(st["quiet_run_s"] if min_run is None else min_run))


def energy_end_in_quiet(energy, quiet_db=None, min_run=None):
    """cut_in_quiet for the end cut on cut_energy()'s frames (times relative to the end cut, dB
    relative to the clip's median frame: the same frames)."""
    if not energy:
        return False
    times, db = energy
    st = CUT_AUDIO
    return _quiet_at(list(times), list(db), 0.0, ENDING_ENERGY_HOP_S,
                     float(st["quiet_db"] if quiet_db is None else quiet_db),
                     float(st["quiet_run_s"] if min_run is None else min_run))


def seat_on_cut_audio(words, cuts, audio, tol_s=WORD_BOUNDARY_INSIDE_TOL_S):
    """(words, seated): every word a cut in a pause (cut_in_quiet on `audio`) sits more than
    `tol_s` inside, moved to the side of the cut its voice is on (more voiced 20 ms frames of its
    claimed span there): its start onto where the voice comes back after the pause, or its end
    onto where the voice stops before it. A word wholly in the pause goes after the cut. Other
    words are as they were (copies; the input is not mutated); a moved word keeps its times as
    asr_start / asr_end. seated = [{"w", "cut", "side", "from", "to"}]."""
    out = [dict(w) for w in words or []]
    if not audio or not out:
        return out, []
    times, db = audio["times"], audio["db"]
    hop = float(audio.get("hop", CUT_AUDIO_HOP_S))
    qdb = float(CUT_AUDIO["quiet_db"])
    mids = [(float(x) + hop / 2.0, float(d)) for x, d in zip(times, db)]
    seated = []
    for side, t in cuts:
        if t is None or not cut_in_quiet(audio, float(t)):
            continue
        t = float(t)
        p0, p1 = _quiet_run(times, db, t, hop, qdb)
        for w in out:
            ws, we = float(w["start"]), float(w["end"])
            if not (ws + tol_s < t < we - tol_s):
                continue
            before = sum(1 for m, d in mids if ws <= m < p0 and d > qdb)
            after = sum(1 for m, d in mids if p1 <= m < we and d > qdb)
            if before > after:
                w.update(end=round(max(p0, ws), 3), asr_end=w.get("asr_end", we))
            else:
                w.update(start=round(min(p1, we), 3), asr_start=w.get("asr_start", ws))
            seated.append({"w": str(w.get("w", "")), "cut": side, "side": "before" if before > after
                           else "after", "from": [ws, we], "to": [w["start"], w["end"]]})
    return out, seated


def safe_cut_audio(src_media, start_s, end_s):
    """cut_audio when the setting is on and the audio reads, else None (the word times alone)."""
    if not src_media or start_s is None or end_s is None or not cut_audio_enabled():
        return None
    try:
        return cut_audio(src_media, float(start_s), float(end_s))
    except Exception:             # noqa: BLE001 - no audio: the word times decide, as before
        return None


def measure_word_boundaries(words, start_s, end_s, tol_s=WORD_BOUNDARY_INSIDE_TOL_S, audio=None):
    """``words`` is the SOURCE-time words sidecar. Fail only when the clip start / end
    cut lands INSIDE a word (or within ``tol_s`` of its edge, i.e. effectively still inside
    the word). A cut that sits in a true inter-word gap passes regardless of how far it is
    from either neighbouring word — ``boundaries.clean_clip`` places cuts at the gap's
    midpoint (or RMS-silence minimum), which is in silence by construction.

    Earlier revisions required the cut to be within 50 ms of a word boundary (treating the
    midpoint-of-gap cut as a near-miss). That measured the wrong thing: the gap IS silence,
    the cut was never INSIDE a word, so the clip did not cut off a word.

    ``audio`` (cut_audio frames): a cut the audio shows in a pause is
    not inside the word a transcript stretched over it; that word is read on the side of the cut
    its voice is on (seat_on_cut_audio; recorded as start_quiet / end_quiet).
    """
    if not words:
        return {"passed": None, "reason": "no words sidecar"}
    words, seated = seat_on_cut_audio(words, (("start", start_s), ("end", end_s)), audio, tol_s)
    s_hit, s_word = _cut_inside_word(start_s, words, tol_s)
    e_hit, e_word = _cut_inside_word(end_s, words, tol_s)
    # distance to the NEAREST boundary (informational: smaller is tighter)
    starts = [float(w["start"]) for w in words]
    ends = [float(w["end"]) for w in words]
    ds = min((abs(float(start_s) - s) for s in starts), default=1e9)
    de = min((abs(float(end_s) - e) for e in ends), default=1e9)
    errs = []
    if s_hit:
        errs.append(f"start cut inside word {str(s_word.get('w','?'))!r} "
                    f"(start={float(s_word['start']):.3f} end={float(s_word['end']):.3f})")
    if e_hit:
        errs.append(f"end cut inside word {str(e_word.get('w','?'))!r} "
                    f"(start={float(e_word['start']):.3f} end={float(e_word['end']):.3f})")
    rep = {"passed": not (s_hit or e_hit),
           "d_start_s": round(ds, 3), "d_end_s": round(de, 3),
           "tol_s": float(tol_s),
           "start_in_word": bool(s_hit), "end_in_word": bool(e_hit),
           "errors": errs}
    for x in seated:
        rep[f"{x['cut']}_quiet"] = f"in a pause: {x['w']!r} read {x['side']} the cut"
    return rep


# ---------------------------------------------------------------- measurement: caption

def measure_caption(caption_text, campaign):
    """``caption_text`` is the exact text that will be posted; ``campaign`` is a Campaign dict
    (or dataclass) carrying the campaign's required_elements. Returns the validator errors.
    When ``caption_text`` is None, the gate holds (``passed=False``) with reason=missing."""
    if caption_text is None:
        return {"passed": False, "errors": ["no caption built"]}
    errs = captions_tiktok.validate_from_campaign(caption_text, campaign)
    return {"passed": not errs, "errors": errs}


# ---------------------------------------------------------------- measurement: sentence-complete boundaries

SENTENCE_TERMINALS = (".", "?", "!")
SENTENCE_CLOSERS = "\"'”’)]}»"
SENTENCE_ABBREV = {"mr", "mrs", "ms", "dr", "vs"}
SENTENCE_PAUSE_S = 0.35          # a gap this long between words is a sentence boundary

# tokens that NEVER constitute a complete clause end even if a pause follows them.
# "I was like," + 0.35 s pause is NOT a sentence end — "like" is a quotative tag, the next
# phrase is the actual payload. Likewise dangling conjunctions / prepositions ("and", "to",
# ...) and bare commas.
TRAILING_DANGLING = {
    # quotative tag — "I was like, [actual thing]"
    "like",
    # conjunctions
    "and", "but", "so", "because", "or", "cause", "cuz",
    # prepositions
    "to", "of", "with", "in", "on", "at", "by", "for", "from", "as",
    # articles
    "the", "a", "an",
    # subordinators that must be followed by a clause
    "that", "if", "when", "while",
    # hesitations
    "um", "uh",
}


def _strip_closers(tok):
    return str(tok or "").strip().rstrip(SENTENCE_CLOSERS)


def _normalize_core(tok):
    """Lowercase alphanumeric core of a token for dangling / abbrev matching."""
    t = _strip_closers(tok).lower()
    return re.sub(r"[^a-z']", "", t)


def _tok_ends_sentence(tok):
    """True when a token ends a sentence — terminal ``. ? !`` after optional closing
    quotes, not an abbreviation, and not a dangling filler like ``like.``. A trailing comma
    / colon / semicolon never counts as a sentence end regardless of what follows."""
    t = _strip_closers(tok)
    if not t:
        return False
    if t[-1] in (",", ";", ":"):
        return False
    if t[-1] not in SENTENCE_TERMINALS:
        return False
    core = t.rstrip(".").lower()
    if core in SENTENCE_ABBREV:
        return False
    norm = _normalize_core(core)
    if norm in TRAILING_DANGLING:
        return False
    return True


def _is_dangling_tail(tok):
    """True when the token itself is a dangling clause continuation — trailing comma /
    colon / semicolon OR a filler word (``like``, ``and``, ``but``, ...). An explicit
    sentence terminator (``. ? !``) overrides the dangling status for every token EXCEPT
    the quotative tag ``like`` (``I was like.`` is still a mid-thought handoff — speakers
    use ``like`` to introduce a quoted phrase and the transcript sometimes drops the comma).
    """
    t = _strip_closers(tok)
    if not t:
        return True
    if t[-1] in (",", ";", ":"):
        return True
    norm = _normalize_core(t)
    if norm == "like":
        return True
    if t[-1] in SENTENCE_TERMINALS:
        return False
    return norm in TRAILING_DANGLING


def measure_sentence_bounds(words, start_s, end_s, *, pause_s=SENTENCE_PAUSE_S, tol_s=0.05,
                            audio=None):
    """enforce sentence-complete boundaries.

    * START condition: the clip's first word is a true sentence start — the previous word
      (in the source) ends with ``. ? !`` (not an abbreviation), OR there is a pause of at
      least ``pause_s`` between that word and the first clip word, OR the clip starts at the
      first source word.
    * END condition: the clip's last word ends with ``. ? !``, OR there is a pause of at
      least ``pause_s`` after it, OR the clip ends at the last source word.

    ``words`` is the SOURCE-time words sidecar (absolute seconds). ``audio`` (cut_audio frames): a word stretched over a cut in a pause is read on the side of the cut
    its voice is on (seat_on_cut_audio), as word_boundaries reads it."""
    if not words:
        return {"passed": None, "reason": "no words sidecar"}
    words, _seated = seat_on_cut_audio(words, (("start", start_s), ("end", end_s)), audio)
    ws = sorted(words, key=lambda w: float(w["start"]))
    i = next((k for k, w in enumerate(ws) if float(w["start"]) >= float(start_s) - tol_s), None)
    if i is None:
        return {"passed": None, "reason": "clip start past last word"}
    j = None
    for k in range(len(ws) - 1, -1, -1):
        if float(ws[k]["end"]) <= float(end_s) + tol_s:
            j = k
            break
    if j is None or j < i:
        return {"passed": None, "reason": "end before start"}
    first = ws[i]
    last = ws[j]
    # Start side
    start_ok = False
    if i == 0:
        start_ok = True
    else:
        prev = ws[i - 1]
        if _tok_ends_sentence(prev.get("w", "")):
            start_ok = True
        elif float(first["start"]) - float(prev["end"]) >= pause_s:
            start_ok = True
    # End side
    end_ok = False
    last_tok = last.get("w", "")
    if _tok_ends_sentence(last_tok):
        end_ok = True
    elif _is_dangling_tail(last_tok):
        # trailing comma / "like," / "and" / "to" never ends a complete clause,
        # even if a long pause follows. The engine must pick a different sentence end.
        end_ok = False
    elif j + 1 >= len(ws):
        end_ok = True
    else:
        nxt = ws[j + 1]
        if float(nxt["start"]) - float(last["end"]) >= pause_s:
            end_ok = True
    errs = []
    if not start_ok:
        prev_w = ws[i - 1].get("w", "?") if i > 0 else "(none)"
        errs.append(f"start mid-sentence: prev={prev_w!r} first={first.get('w','?')!r}")
    if not end_ok:
        nxt_w = ws[j + 1].get("w", "?") if j + 1 < len(ws) else "(none)"
        tag = ""
        if _is_dangling_tail(last_tok):
            tag = " (dangling tail)"
        errs.append(f"end mid-sentence{tag}: last={last_tok!r} next={nxt_w!r}")
    return {"passed": start_ok and end_ok,
            "start_sentence_start": start_ok, "end_sentence_end": end_ok,
            "first_word": first.get("w", ""), "last_word": last.get("w", ""),
            "errors": errs}


# ---------------------------------------------------------------- measurement: hook support

HOOK_JUDGE_PROMPT = """You are a short-form video editor QA for TikTok clips.
Each clip has a HOOK (an overlay title that appears on the clip in the first few seconds)
and a TRANSCRIPT (what the speakers actually say in the clip, in order).

Task: decide whether the HOOK is supported by the transcript.
Rules:
- "Supported" means the transcript contains a line (or set of lines) that justifies the hook
  claim. A paraphrase is fine but the specific CLAIM in the hook must be present.
- If the hook names an outcome or event the transcript does not describe, mark unsupported.
- If the transcript is only an opening setup (no payoff) and the hook promises a payoff,
  mark unsupported.
- Short "I think so" / "yeah" / listener reactions do not count as support.

Return ONLY compact JSON (no code fences, no commentary):
{{"supported": true|false, "quote": "<=120 chars quote from the transcript (empty when unsupported)", "reason": "<=160 chars one line why>"}}

HOOK: {hook}
TRANSCRIPT:
{transcript}
"""


HOOK_GROUND_PROMPT = """You check a short-form clip's HOOK (the title burned onto the clip) against the
TRANSCRIPT of what is said in the clip (a speech-to-text pass; it can mishear a word).

A hook is a curiosity title. It may paraphrase, compress, tease ("Why ...", "How ...",
"What everyone gets wrong about ...", "The X that ..."), and it may overstate a little. A "Why" or
"How" hook does NOT need the clip to give the full explanation, as long as the clip is about
that topic.

The hook may also describe what is SEEN in the clip (gameplay, an action, a reaction, a crash)
that a transcript cannot show: a plain description of the action passes when nothing said
contradicts it.

The hook FAILS when one of these is true:
1. WRONG SUBJECT: it pins the claim on a person or thing the transcript shows it is not about
   (e.g. a remark about "his hair" made right after greeting someone else by name).
2. CONTRADICTED: the transcript explicitly says the opposite of the hook's specific claim.
3. OFF TOPIC: the clip is about something else entirely.
4. NOT MENTIONED: it states a specific fact that nobody says in the transcript: a number or an
   amount, a name, a person's role or job, an event or an outcome. "Not mentioned" is a failure
   for these, even when it might be true or might be on screen (e.g. "the $25,000 card" when
   nobody says a price, "Wrestler demonstrates ..." when nobody says he wrestles, "even dealers
   suffer" when nobody mentions dealers).

Return ONLY compact JSON (no code fences, no commentary):
{{"supported": true|false, "quote": "<=120 chars quote from the transcript", "reason": "<=160 chars>"}}

HOOK: {hook}
TRANSCRIPT:
{transcript}
"""


def make_hook_ground_judge(cfg_select=None, model=None, timeout_s=120):
    """The semantic hook judge: :func:`make_claude_judge` with
    HOOK_GROUND_PROMPT (wrong subject / contradicted claim / off topic; a paraphrase, a tease,
    a mild overstatement or a claim about what is seen passes). The first prompt failed the
    approved clip 119 ("Why he gave up on 300 pounds": "never explains why"); a first version
    of this one that also failed claims the transcript "never touches" failed two approved
    10-13 batch clips whose hooks describe gameplay / the screen (03_142 "Zipline cross goes
    horribly wrong", 03_14). A specific fact nobody says (a number, a name, a role, an event)
    fails again, a description of the action still passes: hooks this judge had passed named a
    price, a job and a group nobody in the clip mentions."""
    from clipengine import select_semantic as ss     # noqa: WPS433
    opts = dict(cfg_select or {})
    mdl = model or opts.get("semantic_model") or "opus"

    def judge(hook, transcript):
        from clipengine import second_asr                # a window deadline bounds it too
        left = second_asr.time_left()
        t = float(timeout_s) if left is None else min(float(timeout_s), left - 5.0)
        if t < 10.0:
            return {"error": f"window cap near ({left:.0f} s left): judge not asked",
                    "skipped": "time"}
        prompt = HOOK_GROUND_PROMPT.format(hook=str(hook or "").strip(),
                                           transcript=str(transcript or "").strip())
        try:
            text = ss._claude_cli(prompt, mdl, t, opts)
        except Exception as e:          # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}
        obj = _parse_judge_json(text)
        if obj is None:
            return {"error": "judge produced no parseable JSON"}
        return {"supported": bool(obj.get("supported")),
                "quote": str(obj.get("quote") or "")[:200],
                "reason": str(obj.get("reason") or "")[:400]}
    return judge


def _parse_judge_json(text):
    """Pull the first JSON object with a ``supported`` key from the model's reply (tolerates
    fences / leading prose). Returns None when nothing parses."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", s)
        s = re.sub(r"```\s*$", "", s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "supported" in obj:
            return obj
    except ValueError:
        pass
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", s):
        try:
            obj, _ = dec.raw_decode(s, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and "supported" in obj:
            return obj
    return None


def make_claude_judge(cfg_select=None, model=None, timeout_s=120):
    """Build a judge callable backed by the engine's existing ``select_semantic._claude_cli``
    helper. ``cfg_select`` is the ``select`` block of the config; the judge inherits
    the same claude_config_dir / claude_env_file settings.

    Returns a callable ``(hook, transcript) -> dict`` with keys
    ``{supported: bool, quote: str, reason: str}`` or ``{"error": "..."}`` on failure.
    """
    from clipengine import select_semantic as ss     # noqa: WPS433
    opts = dict(cfg_select or {})
    mdl = model or opts.get("semantic_model") or "opus"

    def judge(hook, transcript):
        prompt = HOOK_JUDGE_PROMPT.format(hook=str(hook or "").strip(),
                                          transcript=str(transcript or "").strip())
        try:
            text = ss._claude_cli(prompt, mdl, float(timeout_s), opts)
        except Exception as e:          # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}
        obj = _parse_judge_json(text)
        if obj is None:
            return {"error": "judge produced no parseable JSON"}
        return {"supported": bool(obj.get("supported")),
                "quote": str(obj.get("quote") or "")[:200],
                "reason": str(obj.get("reason") or "")[:400]}
    return judge


def _words_to_transcript(words):
    if words is None:
        return ""
    if isinstance(words, str):
        return words
    return " ".join(str(w.get("w", "")).strip() for w in words if w.get("w"))


def measure_hook_support(hook, words, *, judge=None):
    """verify the HOOK (the overlay title) is supported by what the speakers actually
    say in the clip. ``words`` is either a words sidecar (list of ``{w,start,end}``) or a raw
    transcript string. ``judge`` is a callable ``(hook, transcript) -> dict``; see
    :func:`make_claude_judge`. When ``judge`` is None, the measure is informational.
    """
    if not hook or not str(hook).strip():
        return {"passed": None, "reason": "no hook text"}
    transcript = _words_to_transcript(words)
    if not transcript:
        return {"passed": None, "reason": "no transcript for the clip"}
    if judge is None:
        return {"passed": None, "reason": "no judge configured",
                "hook": str(hook).strip(),
                "transcript_chars": len(transcript)}
    try:
        verdict = judge(hook, transcript)
    except Exception as e:      # noqa: BLE001
        return {"passed": None, "reason": f"judge raised: {type(e).__name__}: {e}"}
    if not isinstance(verdict, dict) or verdict.get("error"):
        return {"passed": None,
                "reason": (verdict or {}).get("error") if isinstance(verdict, dict) else "no verdict"}
    supported = bool(verdict.get("supported"))
    quote = str(verdict.get("quote") or "")
    why = str(verdict.get("reason") or "")
    return {"passed": supported, "supported": supported, "quote": quote,
            "reason": why,
            "errors": [] if supported else
                      [f"hook not supported by transcript: {why or 'no supporting quote found'}"]}


# ---------------------------------------------------------------- measurement: active-speaker framing

ACTIVE_SPEAKER_MIN_FRAC = 0.80          # fraction of SPEECH time the framed face must be
                                         # the diarized speaker
# spec restores the audio-visual mouth-motion gate to the >=80% from section 8;
# It shipped at 0.60 (loosened to let listener-framing be informational), which
# section 10 explicitly orders us to restore before proving the engine fix on calibration.
ACTIVE_SPEAKER_MOUTH_FRAC = 0.80         # fraction of speech-time samples where the framed
                                         # face must show mouth motion above its own silence
                                         # median (used by audio-visual check)
ACTIVE_SPEAKER_SAMPLE_FPS = 4.0          # downsample rate for the audio-visual check
ACTIVE_SPEAKER_SPEECH_FLOOR = 0.015      # RMS floor identifying speech (same as speaker.py)


def measure_active_speaker_plan(plan, min_frac=ACTIVE_SPEAKER_MIN_FRAC):
    """Plan side: assert the plan's crop targeted the DIARIZED speaker for at least
    ``min_frac`` of the clip's speech time. For every ``speaker_segment`` with ``track_id``,
    the plan segments overlapping it must carry ``speaker == track_id`` (OR the plan has no
    speaker tagging at all, in which case this measure is informational).

    This catches a static two-shot podcast case where ``speaker.active_speaker`` picked the LISTENER
    (track_id 2 with its own single speaker_segment); here the plan is tautologically
    "aligned" with its own picked speaker, so the measure returns ``passed=True`` on the
    plan side alone. The audio-visual check in :func:`measure_active_speaker_audio_visual`
    is the authoritative cross-validation.
    """
    sp = (plan or {}).get("speaker_segments") or []
    segs = (plan or {}).get("segments") or []
    if not sp:
        return {"passed": None, "reason": "no speaker_segments in plan"}
    total_speech = sum(float(s["t1"]) - float(s["t0"]) for s in sp)
    if total_speech <= 0:
        return {"passed": None, "reason": "zero-length speech"}
    any_speaker_tag = any("speaker" in s for s in segs)
    if not any_speaker_tag:
        return {"passed": None, "reason": "plan segments have no speaker tag"}
    matched = 0.0
    for sp_seg in sp:
        tid = sp_seg.get("track_id")
        for seg in segs:
            a = max(float(seg["t0"]), float(sp_seg["t0"]))
            b = min(float(seg["t1"]), float(sp_seg["t1"]))
            if b <= a:
                continue
            if seg.get("speaker") == tid:
                matched += (b - a)
    frac = matched / total_speech if total_speech > 0 else 0.0
    passed = frac >= float(min_frac)
    return {"passed": passed, "fraction_on_speaker": round(frac, 3),
            "min_fraction_required": float(min_frac),
            "total_speech_s": round(total_speech, 2),
            "matched_s": round(matched, 2),
            "errors": [] if passed else
                      [f"framed speaker matched diarization on {frac*100:.0f}% of speech "
                       f"(need {min_frac*100:.0f}%)"]}


def _audio_rms_samples(video_path, fps):
    """Mono audio RMS at the sample rate ``fps`` (one value per 1/fps second)."""
    try:
        import numpy as np
    except ImportError:
        return []
    sr = 16000
    hop = max(1, int(round(sr / float(fps))))
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video_path),
           "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    try:
        buf = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    a = np.frombuffer(buf[:len(buf) // 4 * 4], dtype=np.float32)
    if not len(a):
        return []
    n = len(a) // hop
    if n == 0:
        return []
    a = a[:n * hop].reshape(n, hop)
    return (np.sqrt((a.astype(np.float64) ** 2).mean(axis=1))).tolist()


def measure_active_speaker_audio_visual(video_path, *,
                                         fps=ACTIVE_SPEAKER_SAMPLE_FPS,
                                         mouth_frac_min=ACTIVE_SPEAKER_MOUTH_FRAC,
                                         speech_floor=ACTIVE_SPEAKER_SPEECH_FLOOR):
    """Audio-visual cross-check on the RENDERED mp4: sample the rendered 9:16 clip at
    ``fps``, detect faces + mouth landmarks on each frame, and track the LARGEST face across
    frames (this is the framed subject). Compute mouth-region motion between successive
    samples and compare to audio RMS at the same times. If audio is above the speech floor
    but the framed face's mouth motion is below its own silence-time median for more than
    ``1 - mouth_frac_min`` of the speech samples, the engine is framing a LISTENER.

    Returns ``passed=None`` when the detector isn't available or the clip has no detectable
    face (nothing to validate).
    """
    try:
        import numpy as np                     # noqa: F401
        import cv2                              # noqa: F401
    except ImportError:
        return {"passed": None, "reason": "numpy/cv2 unavailable"}
    try:
        from clipengine.render import speaker as _speaker              # noqa: WPS433
    except ImportError:
        return {"passed": None, "reason": "render.speaker unavailable"}
    rms = _audio_rms_samples(video_path, fps)
    if not rms:
        return {"passed": None, "reason": "no audio samples"}
    samples = []
    for t, frame, sw, sh in iter_frames(video_path, fps=fps):
        faces = _speaker.detect_faces_with_landmarks(frame)
        samples.append({"t": t, "faces": faces, "gray": None, "frame": frame,
                        "sw": sw, "sh": sh})
    if not samples:
        return {"passed": None, "reason": "no video samples"}
    try:
        _speaker.assign_tracks(samples)
    except Exception as e:   # noqa: BLE001
        return {"passed": None, "reason": f"track assignment failed: {type(e).__name__}: {e}"}
    import cv2
    for s in samples:
        s["gray"] = cv2.cvtColor(s["frame"], cv2.COLOR_BGR2GRAY)
    # Primary face per sample = the largest face (framed subject in a 9:16 output).
    def _primary(faces):
        return max(faces, key=lambda f: float(f.get("h") or 0.0)) if faces else None
    prev = None
    mouth_motion = []        # per sample (first sample is None)
    rms_at = []
    for i, s in enumerate(samples):
        f = _primary(s["faces"])
        a = None
        if f is not None and prev is not None and prev.get("face") is not None:
            try:
                a = _speaker.mouth_activity(prev["gray"], s["gray"], prev["face"], f)
            except Exception:                     # noqa: BLE001
                a = None
        mouth_motion.append(a)
        rms_i = rms[min(i, len(rms) - 1)]
        rms_at.append(rms_i)
        prev = {"gray": s["gray"], "face": f}
    valid = [i for i, m in enumerate(mouth_motion) if m is not None]
    if len(valid) < 4:
        return {"passed": None, "reason": "not enough face samples to validate",
                "n_valid": len(valid), "total_frames": len(samples)}
    speech_idx = [i for i in valid if rms_at[i] >= float(speech_floor)]
    silence_idx = [i for i in valid if rms_at[i] < float(speech_floor)]
    if len(speech_idx) < 3:
        return {"passed": None, "reason": "not enough speech samples",
                "n_speech": len(speech_idx)}
    import numpy as np
    speech_motion = np.array([mouth_motion[i] for i in speech_idx], dtype=np.float64)
    if silence_idx:
        silence_median = float(np.median([mouth_motion[i] for i in silence_idx]))
    else:
        # all speech: compare to the clip's overall mouth-motion floor
        silence_median = float(np.quantile(speech_motion, 0.2))
    # Threshold: mouth motion must exceed the silence median by any amount (the simplest
    # unbiased floor). We ALSO require an absolute minimum so a dead-still clip with a
    # noise-floor silence_median doesn't trivially pass.
    thr = max(silence_median * 1.25, 0.5)
    active = (speech_motion > thr).sum()
    frac_active = float(active) / len(speech_idx)
    passed = frac_active >= float(mouth_frac_min)
    return {"passed": passed, "frac_speech_with_mouth_motion": round(frac_active, 3),
            "min_fraction_required": float(mouth_frac_min),
            "n_speech_samples": int(len(speech_idx)),
            "n_silence_samples": int(len(silence_idx)),
            "silence_mouth_median": round(silence_median, 3),
            "threshold": round(float(thr), 3),
            "errors": [] if passed else
                      [f"framed face shows mouth motion in {frac_active*100:.0f}% of speech "
                       f"samples (need {mouth_frac_min*100:.0f}%); likely framing a listener"]}


# ---------------------------------------------------------------- measurement: layout match

# Strict layout match: the rendered plan must carry the configured name. Each campaign-
# visible layout (``product_full_bleed``, ``chart_top_cam_bottom``, …) is a FIRST-CLASS
# name that recompose stamps onto its emitted plan — a plan carrying the internal planner's
# name (``track`` / ``broadcast``) proves the pipeline took the earlier generic path and is
# a rejection by design (a review finding: a stacked chart stream was rendered as a squeezed
# full stream instead of chart_top_cam_bottom). The one unavoidable alias is
# ``chart_top_cam_bottom`` ↔ ``broadcast`` because the internal planner's name is still
# emitted when ``force="broadcast"`` is called without the recompose wrapper; keeping it
# in the allowed set avoids false fails when a non-pipeline caller plans directly against
# reframe.analyze. Legacy layouts (track / split / broadcast / etc.) match themselves only.
LAYOUT_ALIASES = {
    "product_full_bleed":    {"product_full_bleed"},
    "full_bleed_speaker":    {"full_bleed_speaker"},
    "podcast_two_shot":      {"podcast_two_shot"},
    "chart_top_cam_bottom":  {"chart_top_cam_bottom", "broadcast"},
    "stream_split":          {"stream_split", "split"},
    "gamestack":             {"gamestack"},
    "track":                 {"track"},
    "broadcast":             {"broadcast"},
    "split":                 {"split"},
    "hudband":               {"hudband"},
    "general":               {"general"},
    "reaction":              {"reaction"},
    "screencast":            {"screencast", "screencast_auto"},
    "screencast_auto":       {"screencast_auto", "screencast"},
}


def _campaign_layout_name(campaign):
    if campaign is None:
        return None
    if isinstance(campaign, dict):
        block = campaign.get("layout") or {}
    else:
        block = getattr(campaign, "layout", None) or {}
    name = str((block.get("layout") or "")).strip()
    return name or None


def measure_layout_match(plan, campaign):
    """assert the rendered layout matches the campaign's configured layout.

    ``plan`` carries a ``layout`` field written by :mod:`render.recompose` (either the
    canonical campaign layout like ``chart_top_cam_bottom`` or an internal alias like
    ``broadcast``). Returns ``passed=None`` when either side is missing (informational)."""
    configured = _campaign_layout_name(campaign)
    rendered = str((plan or {}).get("layout") or "").strip() or None
    if configured is None:
        return {"passed": None, "reason": "no campaign layout configured",
                "rendered": rendered}
    if rendered is None:
        return {"passed": None, "reason": "no rendered layout in plan",
                "configured": configured}
    if configured == rendered:
        return {"passed": True, "configured": configured, "rendered": rendered}
    allowed = LAYOUT_ALIASES.get(configured, {configured})
    if rendered in allowed:
        return {"passed": True, "configured": configured, "rendered": rendered,
                "note": "alias match"}
    return {"passed": False, "configured": configured, "rendered": rendered,
            "errors": [f"rendered layout {rendered!r} != configured {configured!r}"]}


# ---------------------------------------------------------------- measurement: gameplay

# A gameplay clip never ships a face-follow crop. The gamestack fallback used to give a stream whose
# cam is burned in mid-frame the track crop: centred on the cam, the game cropped tight, the action
# of the game out of the frame. Gameplay renders the game band (render/gamelayout.py: gamestack
# with its cam card, or game only).
FACE_FOLLOW_LAYOUTS = ("track", "split", "product_full_bleed", "full_bleed_speaker",
                       "podcast_two_shot")
GAMEPLAY_VERTICALS = ("gaming", "stream")
ACTION_CROP = "full_bleed_action"   # recompose.full_bleed_action: a track-shaped 9:16 window over
                                    # the action of gameplay with no face (no face crop)


def gameplay_clip(vertical, niche=None, kind=None, layout=None):
    """Whether a clip is gameplay: a gamestack request, or a gaming / stream vertical clip whose
    source is gameplay by its kind ("screen": engine.source_kind of facecam / full-screen
    gameplay) or by its niche (gaming). The stream vertical's talking heads and IRL vlogs
    (kind "talk") keep their speaker crop; so does a stream clip with neither a kind nor the
    gaming niche (a manual render)."""
    if (layout or "") == "gamestack":
        return True
    vertical = (vertical or "").lower()
    if vertical not in GAMEPLAY_VERTICALS or kind == "talk":
        return False
    return vertical == "gaming" or kind == "screen" or niche == "gaming"


def face_follow(layout, guard_action=None):
    """Whether a rendered layout is a face-follow crop (the full-bleed action crop is not)."""
    return layout in FACE_FOLLOW_LAYOUTS and guard_action != ACTION_CROP


def measure_gameplay_layout(gameplay, rendered_layout, guard_action=None):
    """A gameplay clip (gameplay_clip) whose rendered layout is a face-follow crop fails (code
    gameplay_face_follow). `guard_action` = the render's banner_guard action (QC line)."""
    if not gameplay:
        return {"passed": None, "reason": "not gameplay", "layout": rendered_layout}
    if not rendered_layout:
        return {"passed": None, "reason": "no rendered layout", "layout": None}
    if face_follow(rendered_layout, guard_action):
        return {"passed": False, "layout": rendered_layout,
                "errors": [f"gameplay rendered as {rendered_layout!r}, a face-follow crop: the "
                           f"game band (gamestack / game only) or no clip"]}
    return {"passed": True, "layout": rendered_layout}


# ---------------------------------------------------------------- measurement: the streamer cam

# A stream / gaming clip whose source has a streamer cam shows it as an inset in a bottom corner of
# the frame. The render (recompose.keep_cam) reports the source's cam and the inset it drew in its
# QC line ("cam_inset"); a stream / gaming clip from a source with a cam rendered without an inset
# fails (code cam_missing), so it is never shipped silently.
CAM_VERTICALS = ("stream", "gaming")


def measure_cam_inset(vertical, cam_qc):
    """The cam check of a stream / gaming clip: fails when its source has a streamer cam
    (cam_qc["source_cam"]) and the render drew no inset (cam_qc["inset"]). Unmeasured (passed None)
    outside the verticals, with no cam report (an older render) or when the render could not
    check; passes a source with no cam (game only) and a speaker crop (the person is the picture)."""
    if (vertical or "").lower() not in CAM_VERTICALS:
        return {"passed": None, "reason": "not a stream / gaming clip"}
    if not isinstance(cam_qc, dict) or not cam_qc.get("rule"):
        return {"passed": None, "reason": "no cam report from the render"}
    if cam_qc.get("error"):
        return {"passed": None, "reason": f"the render's cam check failed: {cam_qc['error']}"}
    out = {k: cam_qc.get(k) for k in ("source_cam", "inset", "corner", "area", "width", "over",
                                       "how", "layout", "cap_clear") if k in cam_qc}
    if cam_qc.get("source_cam") and not cam_qc.get("inset"):
        return dict(out, passed=False, errors=[
            f"cam_missing: the source has a streamer cam {cam_qc['source_cam']} and the "
            f"{cam_qc.get('layout')} render shows no cam inset ({cam_qc.get('why') or 'dropped'})"])
    return dict(out, passed=True, reason=cam_qc.get("why"))


# ---------------------------------------------------------------- measurement: layout quality

CHART_TOP_H = 1312                     # must match render.recompose/reframe BROADCAST_TOP_H
CHART_BLACK_BAR_FRAC_MAX = 0.03        # <= 3% of top-panel area may be black BARS
                                        # (letterbox / pillarbox edges, not dark content)
CHART_BOTTOM_FACE_FRAC_MIN = 0.25      # cam face height >= 25% of bottom panel height
CHART_BLACK_THRESHOLD = 15             # pixel intensity <= this counts as "near-black"
CHART_BAR_ROW_BLACK_FRAC = 0.98        # a row/column counts as a "black bar" when this
                                        # fraction of its pixels are near-black
# sample layout_quality at >= 4 fps so a sub-second black-screen episode or a
# content-vanishing moment is caught. The earlier 1 fps sampling missed the last 6 s of
# one chart clip going fully black.
CHART_SAMPLE_FPS = 4.0
# chart content must fill at least this fraction of the top panel's vertical extent.
# This is enforced per 1 s window (not just the clip median) because one chart clip
# averaged 57% while whole seconds were under 30%.
CHART_CONTENT_HEIGHT_FRAC_MIN = 0.50
CHART_CONTENT_WINDOW_FRAC_MIN = 0.30   # fail if ANY 1 s sliding window has mean
                                        # content < this. Catches localized content loss
                                        # even when the clip average passes.
CHART_WINDOW_BAR_FRAC_MAX = 0.03       # fail if ANY 1 s sliding window has mean
                                        # black-bars > this.
CHART_PANEL_BLACK_FRAC_MIN = 0.90      # a top-panel frame with >= this fraction of
                                        # near-black pixels is treated as "panel went black"
                                        # (chart replaced by a UI overlay / black fade). Any
                                        # single such frame fails the gate.
CHART_CONTENT_SAT_FLOOR = 25.0         # per-pixel HSV saturation to count as "content"
                                        # (same floor as recompose.detect_chart_content_bbox)


def _black_bar_fraction(panel, threshold=CHART_BLACK_THRESHOLD,
                         row_frac=CHART_BAR_ROW_BLACK_FRAC):
    """Fraction of ``panel`` area occupied by black BARS on its edges — rows at the top /
    bottom and columns at the left / right where >= ``row_frac`` of pixels are near-black.
    Walks inward from each edge until it hits a non-black row/column. A chart UI with dark
    content in the middle (lots of black pixels) does NOT trigger because the inward walk
    stops at the first non-black row.
    """
    import numpy as np
    h, w = panel.shape[:2]
    max_c = panel.max(axis=2)                              # (h, w) brightest channel
    row_black = (max_c <= threshold).mean(axis=1)          # per-row black-pixel frac
    col_black = (max_c <= threshold).mean(axis=0)          # per-col black-pixel frac
    top_bar = 0
    while top_bar < h and row_black[top_bar] >= row_frac:
        top_bar += 1
    bot_bar = 0
    while bot_bar < h - top_bar and row_black[h - 1 - bot_bar] >= row_frac:
        bot_bar += 1
    left_bar = 0
    while left_bar < w and col_black[left_bar] >= row_frac:
        left_bar += 1
    right_bar = 0
    while right_bar < w - left_bar and col_black[w - 1 - right_bar] >= row_frac:
        right_bar += 1
    bar_area = (top_bar + bot_bar) * w + (left_bar + right_bar) * (h - top_bar - bot_bar)
    return float(bar_area) / max(1, h * w), {
        "top_bar_px": int(top_bar), "bot_bar_px": int(bot_bar),
        "left_bar_px": int(left_bar), "right_bar_px": int(right_bar),
    }


def _chart_content_vertical_fraction(panel, sat_floor=CHART_CONTENT_SAT_FLOOR):
    """fraction of the top panel's vertical extent that contains chart "content".

    Converts the panel to HSV, builds a saturation mask (``S >= sat_floor``), then returns
    the vertical span of the content bbox / panel height. The spec says the price action
    must fill >= 50 % of the panel's height; this measure is what feeds that check.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None, {"reason": "cv2/numpy unavailable"}
    h, w = panel.shape[:2]
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    mask = (sat >= float(sat_floor)).astype("uint8") * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, {"n_contours": 0, "panel_h": h}
    cnt = max(cnts, key=lambda c: cv2.contourArea(c))
    _, y0, _, ch = cv2.boundingRect(cnt)
    return float(ch) / max(1, h), {"content_y0": int(y0), "content_h": int(ch),
                                     "panel_h": int(h), "n_contours": len(cnts)}


def _panel_near_black_fraction(panel, threshold=CHART_BLACK_THRESHOLD):
    """fraction of pixels in ``panel`` whose brightest channel is <= ``threshold``.
    When this is above :data:`CHART_PANEL_BLACK_FRAC_MIN` the top chart panel has gone
    fully dark (chart replaced by a UI overlay, a black fade, or similar). A review
    finding: one chart clip went fully black for ~6 s at the end and the median-based
    content fraction still passed.
    """
    try:
        import numpy as np
    except ImportError:
        return 0.0
    max_c = panel.max(axis=2)
    return float((max_c <= threshold).mean())


def _window_fail_1s(series, fps, cmp, limit):
    """return (fail_idx, bad_mean_at_fail) if any 1 s sliding window of ``series``
    (one value per sample at ``fps``) has a mean that violates ``cmp(mean, limit)``, else
    None. The window is a fixed count of ``max(1, round(fps))`` samples and advances by
    one sample at a time."""
    import numpy as np
    arr = np.asarray(series, dtype=np.float64)
    if arr.size == 0:
        return None
    win = max(1, int(round(float(fps))))
    if arr.size < win:
        m = float(arr.mean())
        if cmp(m, limit):
            return (0, m)
        return None
    # rolling mean via cumulative sum
    cs = np.concatenate(([0.0], arr.cumsum()))
    means = (cs[win:] - cs[:-win]) / win
    for i, m in enumerate(means):
        if cmp(float(m), limit):
            return (int(i), float(m))
    return None


def measure_layout_quality(video_path, campaign, *, fps=CHART_SAMPLE_FPS):
    """layout-specific pixel-level quality checks on the rendered
    mp4. Currently enforces ``chart_top_cam_bottom``:

      * every sampled frame's top panel: black BARS (letterbox / pillarbox edges) summed
        into any 1 s window mean <= ``CHART_WINDOW_BAR_FRAC_MAX``;
      * every sampled frame's top panel: chart CONTENT vertical extent summed into any
        1 s window mean >= ``CHART_CONTENT_WINDOW_FRAC_MIN`` (this catches
        a chart clip where the clip-median content was 57% but whole seconds were under
        30%);
      * **any single frame** with the top panel >= ``CHART_PANEL_BLACK_FRAC_MIN`` of
        near-black pixels fails outright (this catches a chart clip's ~6 s all-black
        tail);
      * bottom panel's main face median >= ``CHART_BOTTOM_FACE_FRAC_MIN`` of panel height.

    Sampling is at :data:`CHART_SAMPLE_FPS` (4 fps) so sub-second episodes do not slip
    between samples. Dark CONTENT (black trading chart background with candles on top)
    does not count as a black bar — ``_black_bar_fraction`` walks inward from each edge
    and stops at the first non-black row/column.
    """
    name = _campaign_layout_name(campaign)
    if name != "chart_top_cam_bottom":
        return {"passed": None, "reason": f"no layout quality check for {name!r}"}
    try:
        import numpy as np                      # noqa: F401
    except ImportError:
        return {"passed": None, "reason": "numpy unavailable"}
    try:
        from clipengine.render import reframe   # noqa: WPS433
    except ImportError:
        return {"passed": None, "reason": "render.reframe unavailable"}
    top_h = CHART_TOP_H
    bar_fracs = []
    bar_details = []
    face_fracs = []
    content_fracs = []
    content_details = []
    panel_black_fracs = []
    times = []
    total = 0
    for t, frame, sw, sh in iter_frames(video_path, fps=fps):
        total += 1
        times.append(float(t))
        top = frame[:top_h]
        bot = frame[top_h:]
        bf, det = _black_bar_fraction(top)
        bar_fracs.append(bf)
        bar_details.append(det)
        cf, cd = _chart_content_vertical_fraction(top)
        content_fracs.append(0.0 if cf is None else float(cf))
        content_details.append(cd)
        panel_black_fracs.append(_panel_near_black_fraction(top))
        faces = reframe.detect_faces(bot)
        if faces:
            face_fracs.append(float(max(f.get("h", 0.0) for f in faces)))
        else:
            face_fracs.append(0.0)
    if total == 0:
        return {"passed": None, "reason": "no frames read"}
    bar_med = float(sorted(bar_fracs)[len(bar_fracs) // 2])
    face_med = float(sorted(face_fracs)[len(face_fracs) // 2])
    content_med = float(sorted(content_fracs)[len(content_fracs) // 2])
    errs = []
    # any single frame where the chart panel went fully black is a drop.
    black_frames = [(times[i], panel_black_fracs[i]) for i, pb in enumerate(panel_black_fracs)
                     if pb >= CHART_PANEL_BLACK_FRAC_MIN]
    if black_frames:
        t0 = black_frames[0][0]; p0 = black_frames[0][1]
        errs.append(f"top panel went black at {t0:.2f}s ({p0*100:.0f}% near-black pixels; "
                    f"{len(black_frames)}/{total} black frames)")
    # fail if any 1 s sliding window has mean black bars > CHART_WINDOW_BAR_FRAC_MAX.
    bar_window_fail = _window_fail_1s(bar_fracs, fps, lambda m, lim: m > lim,
                                        CHART_WINDOW_BAR_FRAC_MAX)
    if bar_window_fail is not None:
        wi, wm = bar_window_fail
        errs.append(f"top panel 1 s window at {times[wi]:.2f}s has mean black bars "
                    f"{wm*100:.1f}% (> {CHART_WINDOW_BAR_FRAC_MAX*100:.0f}% cap)")
    # fail if any 1 s sliding window has mean chart content < CHART_CONTENT_WINDOW_FRAC_MIN.
    content_window_fail = _window_fail_1s(content_fracs, fps, lambda m, lim: m < lim,
                                            CHART_CONTENT_WINDOW_FRAC_MIN)
    if content_window_fail is not None:
        wi, wm = content_window_fail
        errs.append(f"top panel 1 s window at {times[wi]:.2f}s has mean chart content "
                    f"{wm*100:.0f}% (< {CHART_CONTENT_WINDOW_FRAC_MIN*100:.0f}% min)")
    # Clip-level medians remain enforced for the slower, structural checks:
    if face_med < CHART_BOTTOM_FACE_FRAC_MIN:
        errs.append(f"bottom cam face height {face_med*100:.0f}% of panel "
                    f"(< {CHART_BOTTOM_FACE_FRAC_MIN*100:.0f}% cap)")
    if content_med < CHART_CONTENT_HEIGHT_FRAC_MIN:
        errs.append(f"top panel chart content fills {content_med*100:.0f}% of height "
                    f"(< {CHART_CONTENT_HEIGHT_FRAC_MIN*100:.0f}% required)")
    return {"passed": not errs, "layout": name,
            "sample_fps": float(fps),
            "top_black_bar_frac_median": round(bar_med, 4),
            "bottom_face_frac_median": round(face_med, 3),
            "top_chart_content_frac_median": round(content_med, 3),
            "top_black_bar_frac_max": round(max(bar_fracs), 4),
            "bottom_face_frac_min": round(min(face_fracs), 3),
            "top_chart_content_frac_min": round(min(content_fracs), 3),
            "panel_black_frac_max": round(max(panel_black_fracs), 3),
            "black_panel_frames": len(black_frames),
            "black_panel_first_t": round(black_frames[0][0], 3) if black_frames else None,
            "sample_bar_detail": bar_details[len(bar_details) // 2],
            "sample_content_detail": content_details[len(content_details) // 2],
            "total_frames": total, "errors": errs,
            "top_black_bar_frac_cap": CHART_BLACK_BAR_FRAC_MAX,
            "bottom_face_frac_min_cap": CHART_BOTTOM_FACE_FRAC_MIN,
            "top_chart_content_frac_min_cap": CHART_CONTENT_HEIGHT_FRAC_MIN,
            "top_chart_content_window_frac_min_cap": CHART_CONTENT_WINDOW_FRAC_MIN,
            "top_black_bar_window_frac_max_cap": CHART_WINDOW_BAR_FRAC_MAX,
            "panel_black_frac_min_cap": CHART_PANEL_BLACK_FRAC_MIN}


# ---------------------------------------------------------------- measurement: overlay edge-cut

OVERLAY_EDGE_PX = 3                  # within this many px (in downsampled frame) of the edge
OVERLAY_MIN_W_FRAC = 0.20            # overlay must be at least this wide (frac of frame width)
OVERLAY_MIN_H_PX = 10                # overlay must be at least this tall (downsampled px)
OVERLAY_MAX_H_FRAC = 0.20            # ignore big fills above this height frac
OVERLAY_MIN_ASPECT = 4.0             # banner-like: w / h >= this (lower-third / score strip)
OVERLAY_FAIL_FRAC_MAX = 0.02         # <= 2% of frames can show an edge-cut overlay
OVERLAY_SAMPLE_FPS = 2.0             # a banner that is only up for < 0.5 s is noise
OVERLAY_SCAN_WIDTH = 540             # downsample frames to this width before running the detector


def _find_overlay_boxes(frame_bgr, min_w_frac=OVERLAY_MIN_W_FRAC, min_h_px=OVERLAY_MIN_H_PX,
                         max_h_frac=OVERLAY_MAX_H_FRAC, min_aspect=OVERLAY_MIN_ASPECT):
    """Detect burned-in text / graphic banner rectangles on a frame.

    Heuristic: Sobel-x gradient magnitude + morphological closing yields horizontal text
    bands (the signature of burned-in captions / lower thirds / score banners). Connected
    components with banner-like aspect ratio (w/h >= ``min_aspect``) and at least
    ``min_w_frac`` of the frame width are candidates. Deliberately tuned conservatively to
    avoid flagging opus-style word-by-word captions (short tokens that don't span width).
    Returns ``[(x0, y0, x1, y1)]`` in frame pixels; empty list when cv2 / numpy is missing.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    mag = cv2.convertScaleAbs(gx)
    _, bw = cv2.threshold(mag, 80, 255, cv2.THRESH_BINARY)
    # wider kernel: bind characters of a line into one box, but demand real width
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 30), 3))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    min_w = int(round(min_w_frac * w))
    max_h = int(round(max_h_frac * h))
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < min_w or ch < min_h_px or ch > max_h:
            continue
        if cw < min_aspect * ch:              # banner-like aspect ratio only
            continue
        out.append((int(x), int(y), int(x + cw), int(y + ch)))
    return out


def _overlay_cut_by_edge(bbox, src_w, src_h, pad_px=OVERLAY_EDGE_PX):
    """An overlay bbox is cut by the frame edge when it touches one side within ``pad_px``
    AND does not reach the opposite side (so part of it is clearly off-frame). An overlay
    that touches BOTH sides is a full-width band and is NOT a cut."""
    x0, y0, x1, y1 = bbox
    touches_left = x0 <= pad_px
    touches_right = x1 >= src_w - pad_px
    touches_top = y0 <= pad_px
    touches_bottom = y1 >= src_h - pad_px
    cut_horiz = (touches_left and not touches_right) or (touches_right and not touches_left)
    # Vertical cut is rare for text banners; count it only when the box is narrow and tall.
    cut_vert = (touches_top and not touches_bottom and (y1 - y0) > (x1 - x0)) \
               or (touches_bottom and not touches_top and (y1 - y0) > (x1 - x0))
    return cut_horiz or cut_vert


def _iter_downsampled_frames(path, fps=OVERLAY_SAMPLE_FPS, width=OVERLAY_SCAN_WIDTH):
    """Yield ``(t, frame_bgr, scaled_w, scaled_h)`` sampled at ``fps`` and scaled to ``width``
    by ffmpeg. Shaves orders of magnitude of decode + numpy time compared with reading full-
    resolution frames at 10 fps."""
    try:
        import numpy as np
    except ImportError:
        return
    sw, sh = probe_size(path)
    if sw == 0 or sh == 0:
        return
    tw = int(width)
    th = max(2, int(round(sh * tw / sw / 2)) * 2)
    size = tw * th * 3
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
           "-an", "-vf", f"fps={fps},scale={tw}:{th}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        i = 0
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield round(i / fps, 3), np.frombuffer(buf, np.uint8).reshape(th, tw, 3), tw, th
            i += 1


def measure_overlay_edge_cut(video_path, *, fps=OVERLAY_SAMPLE_FPS,
                              fail_frac_max=OVERLAY_FAIL_FRAC_MAX, params=None):
    """scan the rendered mp4 for burned-in overlay banners that are cut by the frame
    edge. Returns ``{"passed": bool, "fail_pct": float, "fail_frames": [...], "params": ...}``.

    When the heuristic detector finds no overlay in ANY frame of the clip, the measure stays
    informational (``passed: None``) rather than claiming a pass — the detector is not
    foolproof. Only when cuts ARE detected do we take a hard stance. Frames are sampled at
    ``OVERLAY_SAMPLE_FPS`` and downscaled to ``OVERLAY_SCAN_WIDTH`` for speed.
    """
    params = params or {}
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        return {"passed": None, "reason": "numpy unavailable"}
    try:
        import cv2  # noqa: F401
    except ImportError:
        return {"passed": None, "reason": "cv2 unavailable"}
    cuts = []           # (t, bbox) pairs
    total = 0
    boxes_found = 0
    pad = OVERLAY_EDGE_PX
    for t, frame, sw, sh in _iter_downsampled_frames(video_path, fps=fps):
        total += 1
        boxes = _find_overlay_boxes(frame)
        if not boxes:
            continue
        boxes_found += len(boxes)
        for b in boxes:
            if _overlay_cut_by_edge(b, sw, sh, pad_px=pad):
                cuts.append((round(float(t), 3), b))
                break
    if total == 0:
        return {"passed": None, "reason": "no frames read"}
    if boxes_found == 0:
        return {"passed": None, "reason": "no overlay candidates detected",
                "total_frames": total}
    fail_pct = round(100.0 * len(cuts) / max(1, total), 2)
    passed = (len(cuts) / max(1, total)) <= float(fail_frac_max)
    return {"passed": passed, "fail_pct": fail_pct, "fail_count": len(cuts),
            "total_frames": total, "boxes_detected": boxes_found,
            "fail_frames": [c[0] for c in cuts][:20],
            "fail_frac_max": float(fail_frac_max),
            "errors": [] if passed else
                      [f"{len(cuts)} frame(s) show an edge-cut overlay (>{fail_frac_max*100:.1f}%)"]}


# ---------------------------------------------------------------- measurement: visual quality

# a streamer clip passed every gate while most of the canvas was the
# blurred letterbox fill around a dark game frame. This reads the rendered pixels the viewer
# sees (brightness, contrast, sharpness) and takes the blur-fill share of the canvas from the
# render's own layout (recompose QC `blur_fill_frac`): flat-area statistics cannot tell a blur
# fill from a clean wall or a cinematic shadow. Calibration against review
# verdicts: approved clips sit at luma >= 60, contrast >= 47,
# sharpness >= 1378; the rejected streamer clip at luma 11.9, contrast 26.9, fill 68 %.
VQ_SAMPLE_FPS = 2.0
VQ_SCAN_W = 540                      # 540x960 gray
VQ_MIN_LUMA = 35.0                   # median frame mean luma (0-255)
VQ_MIN_CONTRAST = 30.0               # median frame luma std
VQ_MIN_SHARPNESS = 400.0             # median variance of the Laplacian (540 px wide)
VQ_MAX_BLUR_FILL = 0.5               # blurred letterbox / pillarbox fill share of the canvas


def frame_visual_stats(gray):
    """{"luma", "contrast", "sharpness"} for one gray frame (scaled to VQ_SCAN_W wide first so
    the thresholds are resolution-independent)."""
    import cv2
    import numpy as np
    h, w = gray.shape[:2]
    if w != VQ_SCAN_W:
        gray = cv2.resize(gray, (VQ_SCAN_W, max(2, int(round(h * VQ_SCAN_W / w)))),
                          interpolation=cv2.INTER_AREA)
    g = gray.astype(np.float32)
    return {"luma": float(g.mean()), "contrast": float(g.std()),
            "sharpness": float(cv2.Laplacian(g, cv2.CV_32F, ksize=3).var())}


def visual_quality_verdict(stats, blur_fill_frac=None, sliver_frac=None):
    """Collapse per-frame stats (+ the layout's blur-fill share) to the measure dict. Pure:
    tests feed synthetic stats.

    Broadcast gate 10-21: under a blurred pad the sliced neighbouring video boxes the canvas keeps
    (`sliver_frac`, measure_boxed_slices) are not content either: 534's 0.488 blurred fill kept
    two sliced boxes (0.20 of the canvas) and passed the 0.5 cap. They count as fill when the
    layout pads (blur_fill_frac > 0); a full-bleed crop's slivers are the boxed check's alone."""
    import numpy as np
    if not stats:
        return {"passed": None, "reason": "no frames read"}
    med = {k: float(np.median([s[k] for s in stats])) for k in ("luma", "contrast", "sharpness")}
    slivers = 0.0
    if blur_fill_frac is not None and float(blur_fill_frac) > 0 and sliver_frac:
        slivers = float(sliver_frac)
    errors = []
    if med["luma"] < VQ_MIN_LUMA:
        errors.append(f"too dark: median luma {med['luma']:.1f} < {VQ_MIN_LUMA:.0f}")
    if med["contrast"] < VQ_MIN_CONTRAST:
        errors.append(f"low contrast: median luma std {med['contrast']:.1f} < "
                      f"{VQ_MIN_CONTRAST:.0f}")
    if med["sharpness"] < VQ_MIN_SHARPNESS:
        errors.append(f"blurry: median Laplacian var {med['sharpness']:.0f} < "
                      f"{VQ_MIN_SHARPNESS:.0f}")
    if blur_fill_frac is not None and float(blur_fill_frac) > VQ_MAX_BLUR_FILL:
        errors.append(f"blurred fill covers {float(blur_fill_frac):.0%} of the canvas "
                      f"(> {VQ_MAX_BLUR_FILL:.0%})")
    elif slivers and float(blur_fill_frac) + slivers > VQ_MAX_BLUR_FILL:
        errors.append(f"blurred fill and sliced neighbour boxes cover "
                      f"{float(blur_fill_frac) + slivers:.0%} of the canvas (fill "
                      f"{float(blur_fill_frac):.0%} + sliced boxes {slivers:.0%}; "
                      f"> {VQ_MAX_BLUR_FILL:.0%})")
    return {"passed": not errors, "errors": errors, "frames": len(stats),
            "luma": round(med["luma"], 1), "contrast": round(med["contrast"], 1),
            "sharpness": round(med["sharpness"], 1),
            "blur_fill_frac": None if blur_fill_frac is None else round(float(blur_fill_frac), 3),
            "sliver_frac": round(slivers, 3) if slivers else None,
            "thresholds": {"min_luma": VQ_MIN_LUMA, "min_contrast": VQ_MIN_CONTRAST,
                           "min_sharpness": VQ_MIN_SHARPNESS,
                           "max_blur_fill": VQ_MAX_BLUR_FILL}}


def measure_visual_quality(video_path, blur_fill_frac=None, fps=VQ_SAMPLE_FPS, sliver_frac=None):
    """Brightness / contrast / sharpness of the rendered mp4, plus the blur-fill rule (sliced
    neighbour boxes under a blurred pad count as fill: visual_quality_verdict)."""
    try:
        import numpy as np
        import cv2  # noqa: F401
    except ImportError:
        return {"passed": None, "reason": "numpy/cv2 unavailable"}
    sw, sh = probe_size(video_path)
    if sw == 0 or sh == 0:
        return {"passed": None, "reason": "ffprobe size unavailable"}
    th = max(2, int(round(sh * VQ_SCAN_W / sw / 2)) * 2)
    size = VQ_SCAN_W * th
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video_path), "-an",
           "-vf", f"fps={fps},scale={VQ_SCAN_W}:{th}", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    stats = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            stats.append(frame_visual_stats(np.frombuffer(buf, np.uint8).reshape(th, VQ_SCAN_W)))
    return visual_quality_verdict(stats, blur_fill_frac, sliver_frac)


# ---------------------------------------------------------------- measurement: flat edge band

# 10-20 clip 108 (a 9:16 vertical stream rendered full-frame) passed every gate with the bottom
# 216 px (11.2 %) of every frame solid black. Per sampled frame, the band on an edge is the run
# of rows (columns) from that edge that are each flat (luma std across the line
# <= EDGE_BAND_LINE_STD) at the luma of the outermost line (+- EDGE_BAND_DRIFT). A band is a
# defect when the SAME band (depth within +-EDGE_BAND_STABLE_PX scan px) is there in at least
# EDGE_BAND_FRAME_FRAC of the sampled frames and is >= EDGE_BAND_MIN_FRAC of the frame height
# (width for the sides): a pad, letterbox or dead source band sits at a fixed row, while a dark
# backdrop's flat region moves with the speaker and the noise. Measured (540x960
# scan): clip 108 = 11.1 % bottom band, same depth in 100 % of frames; the eye-check-approved
# clips with dark backdrops (10-16 clip 30, 10-17 clips 60/61, 10-19 clip 69, 10-15 clip 3)
# have 8.6-10.7 % flat top regions, but at one depth in at most 45 % of frames.
EDGE_BAND_FPS = 2.0
EDGE_BAND_SCAN_W = 540
EDGE_BAND_MIN_FRAC = 0.04         # band >= 4 % of the frame height (width for left/right) fails
EDGE_BAND_FRAME_FRAC = edgebands.FRAME_FRAC   # ... when it is there, at one depth, in 60 % of
EDGE_BAND_STABLE_PX = edgebands.STABLE_PX     # frames ("one depth": +-2 scan px, ~4 of 1920)
EDGES = edgebands.EDGES
flat_edge_depths = edgebands.flat_edge_depths   # per-frame band depths (LINE_STD 3, DRIFT 4)


def edge_band_verdict(depths, frame_w, frame_h, min_frac=EDGE_BAND_MIN_FRAC,
                      frame_frac=EDGE_BAND_FRAME_FRAC, stable_px=EDGE_BAND_STABLE_PX,
                      out_size=None, what="canvas"):
    """Collapse per-frame ``flat_edge_depths`` dicts to the measure dict. Pure (tests feed
    synthetic depths). An edge's band is the depth shared (+-stable_px) by the most frames; it
    fails at >= min_frac of the frame when at least ``frame_frac`` of the frames share it.
    ``out_size`` / ``what`` only name the region in the error (the dual view's top panel)."""
    out_w, out_h = out_size or (OUT_FRAME_W, OUT_FRAME_H)
    if not depths:
        return {"passed": None, "reason": "no frames read"}
    n = len(depths)
    persistent = edgebands.persistent_bands(depths, frame_frac=frame_frac, tol=stable_px)
    bands, errors = {}, []
    for edge in EDGES:
        size = frame_h if edge in ("top", "bottom") else frame_w
        frac = persistent[edge] / float(size)
        bands[edge] = round(frac, 4)
        if frac >= min_frac:
            px_out = int(round(frac * (out_h if edge in ("top", "bottom") else out_w)))
            errors.append(f"flat {edge} band {frac:.1%} of the frame (~{px_out} px of the "
                          f"{out_w}x{out_h} {what}) in >= {frame_frac:.0%} of frames")
    return {"passed": not errors, "errors": errors, "band_frac": bands, "frames": n,
            "min_frac": min_frac, "frame_frac": frame_frac}


OUT_FRAME_W, OUT_FRAME_H = 1080, 1920


def _iter_gray_frames(path, fps, width):
    try:
        import numpy as np
    except ImportError:
        return
    sw, sh = probe_size(path)
    if sw == 0 or sh == 0:
        return
    th = max(2, int(round(sh * width / sw / 2)) * 2)
    size = width * th
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-an",
           "-vf", f"fps={fps},scale={width}:{th}", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        i = 0
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield round(i / fps, 3), np.frombuffer(buf, np.uint8).reshape(th, width)
            i += 1


def measure_edge_bands(video_path, fps=EDGE_BAND_FPS, width=EDGE_BAND_SCAN_W):
    """a solid / flat band on any edge of the rendered canvas."""
    try:
        import numpy as np   # noqa: F401
    except ImportError:
        return {"passed": None, "reason": "numpy unavailable"}
    depths, wh = [], None
    for _t, g in _iter_gray_frames(video_path, fps, width):
        wh = (g.shape[1], g.shape[0])
        depths.append(flat_edge_depths(g))
    if not depths:
        return {"passed": None, "reason": "no frames read"}
    return edge_band_verdict(depths, *wh)


# ---------------------------------------------------------------- dual view (HUD niches 10-21)

# recompose's dual view (gaming / poker / streamers) stacks a blurred hook header, the FULL source
# frame fit to the canvas width (the "top" panel) and a zoom of the same frame (the facecam, the
# speaking player, the action); QC "panels" gives their canvas rows. The top panel must slice
# nothing and show no bar, so the checks read it as they read any frame: the edge-band rule on its
# own four edges (a source letterbox / pillarbox / dead band, at 4 % of the panel) plus the
# canvas's top edge (a flat blurred header: 10-21 clip 311), and crop_slices on what it shows
# (the whole frame: only text the source itself runs into its edge). The zoom panel is a
# deliberate crop of what the top panel shows whole: its windows (crop views "panel": "zoom")
# and edges are not read. No threshold changes for any other layout.
DUAL_VIEW_LAYOUT = "dual_view"


def slice_views(views, layout=None):
    """The crop views crop_slices reads: a dual view's zoom windows are left out."""
    if layout != DUAL_VIEW_LAYOUT or views is None:
        return views
    return [v for v in views if v.get("panel") != "zoom"]


def dual_view_edge_depths(gray, panels, canvas_h=OUT_FRAME_H):
    """(flat_edge_depths of the top panel, flat top-edge depth of the canvas) of one gray canvas
    frame at any scale; ``panels`` = QC "panels" (canvas rows)."""
    k = gray.shape[0] / float(canvas_h)
    y0, y1 = panels["top"]
    panel = gray[int(round(y0 * k)):int(round(y1 * k))]
    head = panels.get("header") or [0, 0]
    top = flat_edge_depths(gray)["top"] if head[1] > head[0] else 0
    return flat_edge_depths(panel), top


def dual_view_edge_verdict(panel_depths, canvas_tops, scan_w, panel_h, canvas_h, panels):
    """Pure: the edge-band verdict of a dual view from per-frame dual_view_edge_depths (scan px)."""
    y0, y1 = panels["top"]
    top = edge_band_verdict(panel_depths, scan_w, panel_h, out_size=(OUT_FRAME_W, y1 - y0),
                            what="top panel")
    head = edge_band_verdict([{"top": d, "bottom": 0, "left": 0, "right": 0} for d in canvas_tops],
                             scan_w, canvas_h)
    if top.get("passed") is None:
        return top
    errors = [f"top panel: {e}" for e in top["errors"]] + \
             [f"canvas: {e}" for e in head.get("errors") or []]
    return {"passed": not errors, "errors": errors, "band_frac": top["band_frac"],
            "canvas_top_frac": (head.get("band_frac") or {}).get("top"), "frames": top["frames"],
            "min_frac": top["min_frac"], "frame_frac": top["frame_frac"], "panels": panels,
            "layout": DUAL_VIEW_LAYOUT}


def measure_dual_view_edge_bands(video_path, panels, fps=EDGE_BAND_FPS, width=EDGE_BAND_SCAN_W):
    """The edge-band check of a dual view render: its top panel's four edges and the canvas top."""
    try:
        import numpy as np   # noqa: F401
    except ImportError:
        return {"passed": None, "reason": "numpy unavailable"}
    pd, tops, shape = [], [], None
    for _t, g in _iter_gray_frames(video_path, fps, width):
        d, top = dual_view_edge_depths(g, panels)
        pd.append(d)
        tops.append(top)
        k = g.shape[0] / float(OUT_FRAME_H)
        shape = (g.shape[1], int(round(panels["top"][1] * k)) - int(round(panels["top"][0] * k)),
                 g.shape[0])
    if not pd:
        return {"passed": None, "reason": "no frames read"}
    return dual_view_edge_verdict(pd, tops, shape[0], shape[1], shape[2], panels)


# ---------------------------------------------------------------- graphics sliced by a crop edge

# 10-20 clips 126/127/128 (hudband tiles cut "POT 107,00", "21%", "HIGH STAKES CASH GA..."),
# 133/135 (a lower third and a news crawl cut by the 9:16 crop), 137 (a traderTV vertical
# broadcast: crawls and panels running off the frame) and 108 (a sliced HUD strip) passed every
# gate. The render reports every source window the canvas shows (recompose.crop_views -> QC
# "crop_views"; [] = whole frames). The gate finds the burned-in TEXT of the source window:
# persistent text-line graphics (render/banners.scan, the detector behind the render's banner
# guard) that OCR confirms hold >= SLICE_MIN_ALNUM letters/digits (the line detector also boxes
# a shirt or a hand), plus the OCR's own text lines (it reads a scrolling crawl the persistence
# clustering misses). A text item is cut when
#   * a crop window shows only part of it: its columns (banners.sliced) or rows ("crop"), or
#   * it runs into the source frame edge and that edge is on screen ("frame").
# Each cut item weighs its on-screen share: its presence (share of sampled frames) x the share
# of the clip during which it is cut. Fail when crop cuts sum > SLICE_CROP_SHARE_MAX (the render
# itself must never slice burned-in text) or all cuts sum > SLICE_SHARE_MAX (more than one cut
# text line on screen on average: broadcast clutter; one HUD number at the frame edge is not).
# Measured (report): eye-check-approved clips 0.00 / 0.00; 10-20 rejects crop
# 0.46-3.7 (128 lowest) or total 2.5-7.7.
SLICE_CROP_SHARE_MAX = 0.4
SLICE_SHARE_MAX = 1.0
SLICE_OCR_TIMES = (0.15, 0.5, 0.85)   # source frames OCR'd to confirm / find text
SLICE_OCR_UPSCALE = 2
SLICE_MIN_ALNUM = 4                   # confident OCR letters/digits a text item must hold
SLICE_EDGE_PX = 3                     # within this many src px of the frame edge = runs into it


def ocr_source_text(src, start_s, dur, src_size, times=SLICE_OCR_TIMES):
    """[(x0, y0, x1, y1, text, frame_i)] src px: confident OCR words on a few source frames
    (2x upscale, as banners.ocr_source_words); [] without tesseract."""
    import cv2
    from clipengine.render import reframe
    from clipengine.render import textzones
    if not textzones._ocr_available():
        return []
    sw, sh = src_size
    out = []
    for i, f in enumerate(times):
        fr = next(iter(reframe.iter_frames(src, float(start_s) + f * float(dur), 0.5, fps=1,
                                           width=sw, src_size=src_size)), None)
        if fr is None:
            continue
        g = cv2.cvtColor(fr[1], cv2.COLOR_BGR2GRAY)
        big = cv2.resize(g, (sw * SLICE_OCR_UPSCALE, sh * SLICE_OCR_UPSCALE),
                         interpolation=cv2.INTER_CUBIC)
        out += [tuple(v // SLICE_OCR_UPSCALE for v in w[:4]) + (w[4], i)
                for w in textzones.ocr_words(big)]
    return out


def text_alnum_in(box, words, pad=4):
    """Letters/digits of the OCR words whose centre lies in ``box`` (+pad)."""
    x0, y0, x1, y1 = box
    n = 0
    for w in words or []:
        wx0, wy0, wx1, wy1, text = w[:5]
        cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        if x0 - pad <= cx <= x1 + pad and y0 - pad <= cy <= y1 + pad:
            n += sum(c.isalnum() for c in str(text))
    return n


def ocr_text_lines(words, n_frames=len(SLICE_OCR_TIMES)):
    """OCR words -> text lines [{"box", "frac", "alnum"}]: words on one row of one frame
    (vertical overlap >= half the shorter word, gap <= 3 word heights) join a line; lines on
    the same row across frames merge (banners._same_row), frac = frames seen / n_frames."""
    from clipengine.render import banners
    per = {}
    for w in words or []:
        per.setdefault(w[5] if len(w) > 5 else 0, []).append(w)
    lines = []                                           # [box, frames set, alnum]
    for fi, ws in per.items():
        rows = []
        for x0, y0, x1, y1, text, *_ in sorted(ws, key=lambda w: (w[1], w[0])):
            h = max(1, y1 - y0)
            for r in rows:
                b = r[0]
                if min(b[3], y1) - max(b[1], y0) >= 0.5 * min(h, b[3] - b[1]) and \
                        max(b[0], x0) - min(b[2], x1) <= 3 * h:
                    r[0] = [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]
                    r[1] += sum(c.isalnum() for c in str(text))
                    break
            else:
                rows.append([[x0, y0, x1, y1], sum(c.isalnum() for c in str(text))])
        for box, alnum in rows:
            hit = next((ln for ln in lines if banners._same_row(ln[0], box)), None)
            if hit is None:
                lines.append([box, {fi}, alnum])
            else:
                b = hit[0]
                hit[0] = [min(b[0], box[0]), min(b[1], box[1]), max(b[2], box[2]),
                          max(b[3], box[3])]
                hit[1].add(fi)
                hit[2] = max(hit[2], alnum)
    return [{"box": b, "frac": round(len(f) / max(1, n_frames), 3), "alnum": a}
            for b, f, a in lines]


def text_items(sc, words, min_alnum=SLICE_MIN_ALNUM):
    """Burned-in text of the source window: OCR-confirmed persistent graphics + OCR lines not
    already inside one. ``words`` None = no OCR: every graphic counts (tests, no tesseract)."""
    items = []
    for g in sc.get("graphics") or []:
        alnum = text_alnum_in(g["box"], words) if words is not None else None
        if words is None or alnum >= min_alnum:
            items.append({"box": list(g["box"]), "frac": float(g.get("frac", 1.0)),
                          "banner": bool(g.get("banner")), "alnum": alnum, "src": "graphic"})
    for ln in ocr_text_lines(words) if words else []:
        if ln["alnum"] < min_alnum:
            continue
        cx, cy = (ln["box"][0] + ln["box"][2]) / 2, (ln["box"][1] + ln["box"][3]) / 2
        if any(i["box"][0] <= cx <= i["box"][2] and i["box"][1] <= cy <= i["box"][3]
               for i in items):
            continue
        items.append({**ln, "banner": False, "src": "ocr"})
    return items


def _union_s(spans):
    covered, end = 0.0, float("-inf")
    for a, b in sorted(spans):
        if b > end:
            covered += b - max(a, end)
            end = b
    return covered


def text_extent(box, words, pad=4):
    """The union box of the OCR words whose centre lies in ``box`` (+pad), or ``box`` itself."""
    x0, y0, x1, y1 = box
    hit = [w[:4] for w in words or []
           if x0 - pad <= (w[0] + w[2]) / 2 <= x1 + pad and y0 - pad <= (w[1] + w[3]) / 2 <= y1 + pad]
    if not hit:
        return list(box)
    return [min(w[0] for w in hit), min(w[1] for w in hit), max(w[2] for w in hit),
            max(w[3] for w in hit)]


def crop_slices(sc, views, dur, words=None, min_alnum=SLICE_MIN_ALNUM, edge_px=SLICE_EDGE_PX,
                text_edges=False):
    """Pure: cut text items of scan ``sc`` (banners.scan dict) + OCR ``words`` under the crop
    ``views`` (recompose.crop_views; [] = whole frames on screen). Returns
    [{"box", "frac", "banner", "share", "how", "alnum"}], biggest share first, ``how`` "crop"
    (a window slices it) or "frame" (it runs into the source frame edge, on screen).
    ``text_edges`` (the dual view): an item runs into the frame edge only where its OCR'd text
    does (text_extent), not its line-detector box (10-21 clip 377: the FIELD and board graphics
    end 10-35 px inside the frame, their detector boxes at its edge)."""
    from clipengine.render import banners
    sw, sh = sc["src_w"], sc["src_h"]
    tol_y = banners.SLICE_TOL_FRAC * sh
    dur = float(dur)
    views = views or [{"t0": 0.0, "t1": dur, "x": 0, "y": 0, "w": sw, "h": sh}]
    out = []
    for it in text_items(sc, words, min_alnum):
        x0, y0, x1, y1 = it["box"]
        ex0, ey0, ex1, ey1 = text_extent(it["box"], words) if text_edges and words else it["box"]
        edges = {"left": ex0 <= edge_px, "right": ex1 >= sw - edge_px,
                 "top": ey0 <= edge_px, "bottom": ey1 >= sh - edge_px}
        crop, frame = [], []
        for v in views:
            vx, vy, vw, vh = v["x"], v["y"], v["w"], v["h"]
            if x1 <= vx or x0 >= vx + vw or y1 <= vy or y0 >= vy + vh:
                continue                                    # not on screen in this window
            span = (max(0.0, float(v["t0"])), min(dur, float(v["t1"])))
            if banners.sliced(it["box"], vx, vw, sw) or y0 < vy - tol_y or \
                    y1 > vy + vh + tol_y:
                crop.append(span)
            elif (edges["left"] and vx <= edge_px) or (edges["right"] and vx + vw >= sw - edge_px) \
                    or (edges["top"] and vy <= edge_px) or (edges["bottom"] and vy + vh >= sh - edge_px):
                frame.append(span)
        for how, spans in (("crop", crop), ("frame", frame)):
            share = it["frac"] * _union_s(spans) / max(1e-6, dur) if spans else 0.0
            if share > 0:
                out.append({"box": it["box"], "frac": it["frac"], "banner": it["banner"],
                            "alnum": it["alnum"], "share": round(share, 3), "how": how,
                            "src": it["src"]})
                break                                       # a crop cut outranks a frame cut
    return sorted(out, key=lambda s: -s["share"])


def crop_slice_verdict(slices, crop_max=SLICE_CROP_SHARE_MAX, share_max=SLICE_SHARE_MAX):
    crop = round(sum(s["share"] for s in slices if s["how"] == "crop"), 3)
    total = round(sum(s["share"] for s in slices), 3)
    errors = []
    if crop > crop_max:
        errors.append(f"the crop slices burned-in text: {sum(s['how'] == 'crop' for s in slices)}"
                      f" item(s), on-screen share {crop:.2f} (> {crop_max:.2f})")
    if total > share_max:
        errors.append(f"text cut at the frame / crop edges: {len(slices)} item(s), on-screen "
                      f"share {total:.2f} (> {share_max:.2f})")
    return {"passed": not errors, "errors": errors, "crop_share": crop, "sliced_share": total,
            "sliced": slices[:12], "crop_share_max": crop_max, "share_max": share_max}


def measure_crop_slices(src_media, start_s, end_s, views, scan=None, words=None, layout=None):
    """Graphics sliced by the crop, on the source window + the render's crop views. A dual view
    (``layout``) is read on its top panel: no zoom window (slice_views), and frame edges by the
    text that reaches them (crop_slices ``text_edges``)."""
    views = slice_views(views, layout)
    if views is None:
        return {"passed": None, "reason": "render reported no crop views"}
    if not src_media or not Path(src_media).exists():
        return {"passed": None, "reason": "source media unavailable"}
    if start_s is None or end_s is None:
        return {"passed": None, "reason": "no clip window"}
    from clipengine.render import banners
    dur = float(end_s) - float(start_s)
    sc = scan if scan is not None else banners.scan(str(src_media), float(start_s), dur)
    if words is None:
        words = ocr_source_text(str(src_media), start_s, dur, (sc["src_w"], sc["src_h"]))
    return crop_slice_verdict(crop_slices(sc, views, dur, words=words,
                                          text_edges=layout == DUAL_VIEW_LAYOUT))


# ---------------------------------------------------------------- boxed broadcast (broadcast gate 10-21)

# Three clips of a news show (a row of guest video boxes over a network
# backdrop) passed every gate and failed eye-check. crop_slices reads burned-in TEXT and a video box
# holds none; overlay_edge_cut looks for banners. render/broadcast.canvas_boxes finds the boxes'
# tile borders on the rendered canvas, and which side keeps a neighbouring box the crop sliced.
# Measured on the canvas at 2 fps: 490 / 534 slice a neighbour in 49/49 frames (also 10-20 133
# 106/106, 10-21 331 65/90, 290 6/38, 359 6/98: all eye-check rejects); the 23 posted clips of
# 10-15..10-21 in 0 frames (stream clip 258, lean gate, 1/48).
BOX_SCAN_FPS = 2.0
BOX_SLICE_FRAC_MAX = 0.05     # share of the sampled frames that may keep a sliced neighbouring box
# The cover: TikTok's default cover is frame 0, and a viewer sees the first frames before the
# show cuts. One cover frame whose layout is not the clip's steady layout fails: 535's 6 boxed
# frames (0.00-0.20 s) were 1.2 % of its frames, under any share-based cap (10-20 clip 255: 5
# frames of an outlined guest box on a dark backdrop).
COVER_S = 0.5                 # every frame of the first 0.5 s is a cover frame
STEADY_FROM_S = 1.0           # the steady layout: most common label of the 2 fps frames from here


def _gray_frames(path, width, fps=None, dur=None):
    """Yield (t, gray) frames scaled to `width`: at `fps`, or every frame (t from the stream's
    frame rate) of the first `dur` seconds."""
    import numpy as np
    sw, sh = probe_size(path)
    if sw == 0 or sh == 0:
        return
    th = max(2, int(round(sh * width / sw / 2)) * 2)
    rate = float(fps) if fps else _probe_rate(path)
    vf = (f"fps={fps}," if fps else "") + f"scale={width}:{th}"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path)] + \
          (["-t", f"{float(dur):.3f}"] if dur else []) + \
          ["-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        i = 0
        while True:
            buf = p.stdout.read(width * th)
            if len(buf) < width * th:
                break
            yield round(i / rate, 3), np.frombuffer(buf, np.uint8).reshape(th, width)
            i += 1


def _probe_rate(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
                        str(path)], capture_output=True, text=True)
    try:
        num, _, den = (r.stdout or "").strip().partition("/")
        return float(num) / float(den or 1) or 25.0
    except (ValueError, ZeroDivisionError):
        return 25.0


def scan_canvas_boxes(video_path, fps=BOX_SCAN_FPS, dur=None):
    """[(t, render.broadcast.canvas_boxes)] of the rendered canvas: at `fps`, or every frame of
    the first `dur` seconds (fps None)."""
    from clipengine.render import broadcast
    return [(t, broadcast.canvas_boxes(g))
            for t, g in _gray_frames(video_path, broadcast.CANVAS_SCAN_W, fps=fps, dur=dur)]


def boxed_slice_verdict(scan, frac_max=BOX_SLICE_FRAC_MAX):
    """Pure: fail when more than `frac_max` of the scanned frames keep a sliced neighbouring video
    box. "sliver_frac" (median canvas share of the sliced boxes) feeds the blur-fill rule."""
    import numpy as np
    if not scan:
        return {"passed": None, "reason": "no frames read"}
    n = len(scan)
    sliced = [(t, v) for t, v in scan if v.get("sliced")]
    frac = len(sliced) / n
    sides = sorted({s for _t, v in sliced for s in v["sliced"]})
    errors = []
    if frac > frac_max:
        errors.append(f"boxed broadcast: a neighbouring video box is sliced at the "
                      f"{' and '.join(sides)} canvas edge{'s' if len(sides) > 1 else ''} in "
                      f"{len(sliced)}/{n} frames ({frac:.0%} > {frac_max:.0%})")
    return {"passed": not errors, "errors": errors, "sliced_frac": round(frac, 3),
            "sliced_frames": len(sliced), "frames": n, "sides": sides,
            "panel_frames": sum(1 for _t, v in scan if v.get("label") == "panel"),
            "first_sliced_t": sliced[0][0] if sliced else None,
            "sliver_frac": round(float(np.median([v.get("sliver_frac") or 0.0
                                                  for _t, v in scan])), 3),
            "frac_max": frac_max}


def cover_frame_verdict(cover, scan, steady_from=STEADY_FROM_S):
    """Pure: `cover` [(t, canvas_boxes)] = every frame of the first COVER_S, `scan` the 2 fps
    frames. The steady layout is the most common label ("panel" / "framed" / "clean") of the scan
    from `steady_from` on; one cover frame with another label fails (not share-based)."""
    from collections import Counter
    body = [v["label"] for t, v in scan if t >= steady_from] or [v["label"] for _t, v in scan]
    if not cover or not body:
        return {"passed": None, "reason": "no frames read"}
    steady = Counter(body).most_common(1)[0][0]
    bad = [(t, v["label"]) for t, v in cover if v["label"] != steady]
    errors = []
    if bad:
        errors.append(f"cover frame layout: {len(bad)}/{len(cover)} frame(s) of the first "
                      f"{COVER_S:g} s (from {bad[0][0]:.2f} s) show a {bad[0][1]!r} layout, "
                      f"the clip's steady layout is {steady!r}")
    return {"passed": not errors, "errors": errors, "steady": steady,
            "cover_labels": "".join(v["label"][0] for _t, v in cover),
            "bad_frames": len(bad), "cover_frames": len(cover),
            "first_bad_t": bad[0][0] if bad else None, "cover_s": COVER_S}


def measure_broadcast_boxes(video_path):
    """(boxed_slices, cover_frame) measures of the rendered mp4 (one 2 fps scan feeds both)."""
    scan = scan_canvas_boxes(video_path)
    cover = scan_canvas_boxes(video_path, fps=None, dur=COVER_S)
    return boxed_slice_verdict(scan), cover_frame_verdict(cover, scan)


# ---------------------------------------------------------------- kept face = the speaker (broadcast gate 10-21)

BROADCAST_KEYS = ("boxed_slices", "cover_frame", "kept_face_speaker")
KEPT_FACE_NOT_SPEAKER_MAX = 0.40   # the kept face may be a listener for at most 40 % of the speech
# Layouts with no single speaker crop: the whole frame, two stacked crops, a game / screen stack
# (their largest face need not be the one the render chose).
KEPT_FACE_SKIP_LAYOUTS = NO_SPEAKER_LAYOUTS + ("split", "gamestack", "reaction",
                                               "screencast_auto", "hudband", DUAL_VIEW_LAYOUT)


def speech_spans(words, start_s, end_s, join_s=0.3):
    """[(a, b)] clip seconds of speech from the SOURCE-time words sidecar (every word inside the
    clip, captioned or not); gaps under `join_s` are joined."""
    dur = float(end_s) - float(start_s)
    spans = []
    for w in sorted(words or [], key=lambda w: float(w["start"])):
        a, b = float(w["start"]) - float(start_s), float(w["end"]) - float(start_s)
        if b <= 0 or a >= dur:
            continue
        a, b = max(0.0, a), min(dur, b)
        if spans and a - spans[-1][1] <= join_s:
            spans[-1] = (spans[-1][0], max(spans[-1][1], b))
        else:
            spans.append((a, b))
    return spans


def measure_kept_face_speaker(video_path, words=None, start_s=None, end_s=None,
                              rendered_layout=None):
    """Broadcast gate 10-21: the face the crop keeps must be the one speaking
    (render/speaker.kept_face_speech_check: mouth motion vs speech energy). Fails when it is not
    the speaker for more than KEPT_FACE_NOT_SPEAKER_MAX of the speech time; no data (passed None)
    without a face, without enough speech, or on a layout with no single speaker crop."""
    if rendered_layout in KEPT_FACE_SKIP_LAYOUTS:
        return {"passed": None, "reason": f"layout {rendered_layout!r} keeps no single speaker"}
    from clipengine.render import speaker
    spans = speech_spans(words, start_s, end_s) if words and start_s is not None \
        and end_s is not None else None
    rep = speaker.kept_face_speech_check(str(video_path), speech_spans=spans)
    frac = rep.get("not_speaker_frac")
    if frac is None:
        return {"passed": None, **rep}
    errors = []
    if frac > KEPT_FACE_NOT_SPEAKER_MAX:
        errors.append(f"the kept face is not the speaker for {frac:.0%} of the speech time "
                      f"(> {KEPT_FACE_NOT_SPEAKER_MAX:.0%}): its mouth is still and does not "
                      f"follow the speech")
    return {"passed": not errors, "errors": errors, **rep,
            "max_frac": KEPT_FACE_NOT_SPEAKER_MAX}


# ---------------------------------------------------------------- transcript checks

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "first": "1st", "second": "2nd", "third": "3rd",
}
_TOKEN_SPLIT = re.compile(r"[\s\-–—/]+")


def norm_tokens(words):
    """Normalised word tokens of a words list ({w, ...}) or a string: lowercase, punctuation and
    apostrophes dropped, hyphenated / spaced pieces split, number words as digits. Both
    transcripts go through this so tokenisation differences ("300 -pound", "tik -tok") do not
    count as disagreement."""
    if isinstance(words, str):
        raw = [words]
    else:
        raw = [str(w.get("w", "")) for w in (words or [])]
    out = []
    for text in raw:
        for piece in _TOKEN_SPLIT.split(text.lower()):
            t = re.sub(r"[^a-z0-9]", "", piece.replace("’", "'").replace("'", ""))
            if t:
                out.append(_NUMBER_WORDS.get(t, t))
    return out


def _tok_match(a, b):
    """Token equality with a little slack for spelling variants of the same word (whisper
    models spell names / slang differently: "anthropics"/"anthropiks", "moissanite"): equal,
    or both >= 5 chars with a difflib ratio >= 0.8."""
    if a == b:
        return True
    if len(a) >= 5 and len(b) >= 5:
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio() >= 0.8
    return False


def align_tokens(a, b):
    """Longest common (fuzzy) subsequence of token lists ``a`` and ``b``; returns the boolean
    list "a[i] is matched". O(len(a) x len(b)) DP: clip transcripts are a few hundred words."""
    n, m = len(a), len(b)
    if not n or not m:
        return [False] * n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, nxt = dp[i], dp[i + 1]
        ai = a[i]
        for j in range(m - 1, -1, -1):
            if _tok_match(ai, b[j]):
                row[j] = nxt[j + 1] + 1
            else:
                row[j] = max(nxt[j], row[j + 1])
    matched = [False] * n
    i = j = 0
    while i < n and j < m:
        if _tok_match(a[i], b[j]) and dp[i][j] == dp[i + 1][j + 1] + 1:
            matched[i] = True
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return matched


def clip_words(words, start_s, end_s):
    """Source-time words sidecar -> clip-relative words inside the clip, as recompose burns
    them (recompose.load_words: shift, then clamp/drop by captions.clean_words)."""
    out = []
    dur = float(end_s) - float(start_s)
    for w in words or []:
        if w.get("cap", True) is False:          # crosstalk: kept for cuts, not burned
            continue
        text = " ".join(str(w.get("w", "")).split())
        if not text:
            continue
        s, e = float(w["start"]) - float(start_s), float(w["end"]) - float(start_s)
        if s >= dur or e <= 0:
            continue
        out.append({"w": text, "start": max(0.0, s), "end": max(max(0.0, s), min(e, dur))})
    return sorted(out, key=lambda w: w["start"])


# Caption agreement: share of the burned caption words the second ASR also heard, over the
# whole clip and in the worst window of CAPTION_AGREE_WINDOW caption tokens; the share of the
# words the second ASR heard inside the clip that the captions carry, over the clip and in the
# worst window of heard tokens (an uncaptioned stretch). Measured with the shipped second ASR
# (whisper small) on the eye-checked clips that still have their words sidecar: the 9 approved clips have precision >= 0.839 (10-20 clip 95, whose "BGO" mishearing
# the eye-check accepted), coverage >= 0.825 and both worst windows >= 0.667 (8/12: clips 66,
# 95); each caption reject sits under a floor: 10-20 110 precision 0.771 / heard window 0.583,
# 135 precision 0.701 / window 0.083, 125 (crosstalk both models half-hear alike) window 0.50,
# 109 coverage 0.55 / heard window 0.0 (its last ~6 s uncaptioned), 10-19 70 heard window 0.417
# ("you're wearing Nike sneakers, I'm participating in" never captioned).
CAPTION_AGREE_WINDOW = 12
CAPTION_AGREE_MIN = 0.80          # whole-clip caption precision
CAPTION_AGREE_WINDOW_MIN = 0.60   # worst 12-token window, either side (fails at <= 7/12)
CAPTION_COVERAGE_MIN = 0.75       # heard words that are captioned


def caption_agreement_verdict(caption_words, asr_words, dur, *,
                              agree_min=CAPTION_AGREE_MIN,
                              window=CAPTION_AGREE_WINDOW,
                              window_min=CAPTION_AGREE_WINDOW_MIN,
                              coverage_min=CAPTION_COVERAGE_MIN):
    """Pure: compare the burned caption words (clip-relative) with the second ASR's words that
    start inside the clip. Returns the measure dict."""
    cap = norm_tokens(caption_words)
    heard = norm_tokens([w for w in asr_words or [] if float(w["start"]) < float(dur)])
    if not heard and not cap:
        return {"passed": None, "reason": "no speech in either transcript"}
    if not heard:
        return {"passed": None, "reason": "second ASR heard no words",
                "caption_tokens": len(cap)}
    if not cap:
        return {"passed": False, "errors": [f"no captions, second ASR heard {len(heard)} words"],
                "asr_tokens": len(heard)}
    m_cap = align_tokens(cap, heard)
    m_heard = align_tokens(heard, cap)
    precision = sum(m_cap) / len(cap)
    coverage = sum(m_heard) / len(heard)

    def _worst(matched):
        if len(matched) < window:
            return sum(matched) / max(1, len(matched)), 0
        best = (1.0, 0)
        for i in range(0, len(matched) - window + 1):
            f = sum(matched[i:i + window]) / window
            if f < best[0]:
                best = (f, i)
        return best
    worst, worst_at = _worst(m_cap)
    worst_heard, heard_at = _worst(m_heard)
    errors = []
    if precision < agree_min:
        errors.append(f"captions agree with the second ASR on {precision:.0%} of words "
                      f"(< {agree_min:.0%})")
    if worst < window_min:
        errors.append(f"garbled caption stretch: {worst:.0%} of {window} words confirmed "
                      f"({' '.join(cap[worst_at:worst_at + window])!r})")
    if coverage < coverage_min:
        errors.append(f"captions carry {coverage:.0%} of the words the second ASR heard "
                      f"(< {coverage_min:.0%})")
    if worst_heard < window_min:
        errors.append(f"uncaptioned stretch: {worst_heard:.0%} of {window} heard words "
                      f"captioned ({' '.join(heard[heard_at:heard_at + window])!r})")
    return {"passed": not errors, "errors": errors, "precision": round(precision, 3),
            "worst_window": round(worst, 3), "coverage": round(coverage, 3),
            "worst_heard_window": round(worst_heard, 3),
            "caption_tokens": len(cap), "asr_tokens": len(heard),
            "thresholds": {"agree_min": agree_min, "window": window, "window_min": window_min,
                           "coverage_min": coverage_min}}


# Caption tie-break: the second ASR (whisper `small`) is not ground
# truth. It failed the gate's own thresholds against large-v3-turbo on 9 of 10 disputed spans, and
# 4 of the 9 caption rejects a third, independent model could check were false (the captions
# agreed with turbo / `medium` at 0.85-0.96 where `small` missed words). When the captions fail
# against the second ASR, a third model no other pass uses (`medium`) hears the same span; the
# check fails only when the burned words also fail against it. The reason strings stay the second
# ASR's, so the reject codes do not move.
CAPTION_TIEBREAK = {"enabled": False, "model": "medium"}


def caption_tiebreak_settings(cfg=None):
    """config quality_gate.caption_tiebreak over CAPTION_TIEBREAK; env CAPTION_TIEBREAK=0 off."""
    if cfg is None:
        try:
            cfg = config.load()
        except (OSError, ValueError):
            cfg = {}
    st = {**CAPTION_TIEBREAK, **(((cfg.get("quality_gate") or {}).get("caption_tiebreak")) or {})}
    st.pop("_doc", None)
    if os.environ.get("CAPTION_TIEBREAK", "").strip() == "0":
        st["enabled"] = False
    return st


def _verdict_brief(v):
    return {k: v.get(k) for k in ("passed", "precision", "coverage", "worst_window",
                                  "worst_heard_window", "errors", "reason") if k in v}


def caption_tiebreak_verdict(caption_words, asr_words, third_words, dur, *, base=None,
                             model=None):
    """Pure: the caption check with a tie-break. `base` = caption_agreement_verdict against the
    second ASR (computed when None); when it fails, the same check against `third_words` (the
    third model's clip-relative words over the same span) decides: a pass there passes (its
    numbers, decided_by the model), else the second ASR's verdict and errors stand. No third
    verdict (it heard nothing, no words given): the second ASR's verdict stands."""
    if base is None:
        base = caption_agreement_verdict(caption_words, asr_words, dur)
    if base.get("passed") is not False or third_words is None:
        return base
    third = caption_agreement_verdict(caption_words, third_words, dur)
    tb = {"model": model, "second_asr": _verdict_brief(base), "third": _verdict_brief(third)}
    if third.get("passed") is True:
        return {**third, "decided_by": model or "tie-break", "tiebreak": {
            **tb, "verdict": f"the captions agree with {model or 'the third model'}; the second "
                             f"ASR missed or misheard words"}}
    tb["verdict"] = (f"the captions also fail against {model or 'the third model'}"
                     if third.get("passed") is False else
                     f"no verdict from {model or 'the third model'} ({third.get('reason')})")
    return {**base, "tiebreak": tb}


def run_caption_tiebreak(src_media, start_s, end_s, model=None):
    """The tie-break model's words for the clip (second_asr.transcribe on the source window +
    its tail, cached like the second ASR's own pass), clip-relative. Raises when it cannot run or
    is not independent of every caption model and the second ASR."""
    from clipengine import second_asr
    model = model or caption_tiebreak_settings()["model"]
    why = second_asr.independent(model)
    if why is None and model == second_asr.settings()["model"]:
        why = f"tie-break model {model!r} is the second ASR"
    if why:
        raise RuntimeError(why)
    if not (src_media and Path(src_media).exists()) or start_s is None or end_s is None:
        raise RuntimeError("no source media for the tie-break")
    return second_asr.transcribe(src_media, start_s, end_s, model=model)["words"]


def caption_check_with_tiebreak(cap, asr_words, dur, src_media=None, start_s=None, end_s=None,
                                third_fn=None):
    """caption_agreement_verdict against the second ASR, and when it fails and
    quality_gate.caption_tiebreak is on, caption_tiebreak_verdict with the tie-break model's
    words (run_caption_tiebreak, or `third_fn()`). A tie-break that cannot run (off, no source,
    the window's time cap, any error) leaves the second ASR's verdict as it was."""
    base = caption_agreement_verdict(cap, asr_words, dur)
    if base.get("passed") is not False:
        return base
    st = caption_tiebreak_settings()
    if not st.get("enabled"):
        return base
    try:
        third = third_fn() if third_fn else run_caption_tiebreak(src_media, start_s, end_s,
                                                                 st["model"])
    except Exception as e:        # noqa: BLE001 - the second ASR's verdict stands
        return {**base, "tiebreak": {"model": st["model"],
                                     "error": f"{type(e).__name__}: {str(e)[:160]}"}}
    return caption_tiebreak_verdict(cap, asr_words, third, dur, base=base, model=st["model"])


# Hook support: a missing hook fails (10-20 clip 113 shipped with hook=None). Otherwise the
# semantic judge (make_hook_ground_judge: wrong subject / contradicted or absent claim / off
# topic) decides; it caught clip 109's wrong subject (3 of 3 runs) and passed all 18 approved
# hooks it was given. Without a judge answer the lexical floor decides: at least one hook
# content word said in the clip (second ASR words inside the clip + captions; stems and spelling
# variants count). It is only a floor because hooks paraphrase: approved 10-15 clip 3 "Why I
# bet against the crowd" shares 0/2 words with "take the opposite side of the majority". A
# judge skipped for the window's time cap leaves a missed floor unjudged.
HOOK_STOPWORDS = frozenset("""
a an the and or but so if then than of to in on at by for from with without about into over
under after before up down out off is are was were be been being am do does did done doing have
has had having it its this that these those he she they them his her their him we our you your
i me my mine who whom whose what which why how when where there here not no yes just only very
really can could will would should may might must shall get gets got getting make makes made
one ones s t isnt arent wasnt dont doesnt didnt cant wont aint
""".split())
HOOK_GROUND_MIN_WORDS = 1         # hook content words that must be said in the clip


STEM_SUFFIXES = ("ingly", "edly", "ing", "ers", "ies", "ied", "es", "ed", "er", "ly", "s")


def _stem(t):
    """Every stem of token `t` (a frozenset): `t` itself and `t` less each suffix it ends with
    (a stem keeps >= 3 letters); after -ing / -ed / -er / -es also the stem with its silent e
    (agreed -> agree, dating -> date) and, when 4+ letters are left, with a doubled last consonant
    undone (stepping -> stepp -> step); -ies / -ied -> -y. Two words match when their stems meet.
    The one-stem rule took "water" to "wat" and "watering" to "water", "stepping" to "stepp",
    so a hook saying "watering" or "stepping" missed a clip that says "water" or "step"."""
    out = {t}
    for suf in STEM_SUFFIXES:
        if not t.endswith(suf) or len(t) - len(suf) < 3:
            continue
        s = t[: -len(suf)]
        out.add(s)
        if suf[0] in "ie":
            out.add(s + "e")
            if len(s) >= 5 and s[-1] == s[-2] and s[-1] not in "aeiouls":
                out.add(s[:-1])
        if suf in ("ies", "ied"):
            out.add(s + "y")
    return frozenset(out)


def hook_content_tokens(hook):
    return [t for t in norm_tokens(str(hook or "")) if t not in HOOK_STOPWORDS]


def hook_grounding(hook, transcript_tokens):
    """(share, matched, content): hook content words found (stem / fuzzy) in the clip."""
    content = hook_content_tokens(hook)
    if not content:
        return 0.0, [], content
    stems = set().union(*(_stem(t) for t in transcript_tokens)) if transcript_tokens else set()
    toks = set(transcript_tokens)
    matched = [c for c in content
               if c in toks or _stem(c) & stems
               or any(_tok_match(c, t) for t in toks if abs(len(t) - len(c)) <= 3)]
    return len(matched) / len(content), matched, content


def hook_support_verdict(hook, asr_words, caption_words, dur, *, judge=None,
                         min_words=HOOK_GROUND_MIN_WORDS):
    """Pure (the judge is injected): missing hook -> fail; else the semantic judge's answer
    decides; without one (no judge, or it errored) the lexical floor does."""
    if not hook or not str(hook).strip():
        return {"passed": False, "errors": ["no hook: the clip has no hook title"]}
    inside = [w for w in asr_words or [] if float(w["start"]) < float(dur)]
    tokens = norm_tokens(inside) + norm_tokens(caption_words)
    if not tokens:
        return {"passed": None, "reason": "no transcript for the clip", "hook": str(hook)}
    share, matched, content = hook_grounding(hook, tokens)
    rep = {"hook": str(hook), "ground_share": round(share, 3), "matched": matched,
           "content_words": content, "min_words": min_words}
    errors = []
    v = None
    if judge is not None:
        text = " ".join(str(w["w"]) for w in inside) or _words_to_transcript(caption_words)
        try:
            v = judge(str(hook), text)
        except Exception as e:      # noqa: BLE001
            v = {"error": f"{type(e).__name__}: {e}"}
        rep["judge"] = v
    if isinstance(v, dict) and not v.get("error") and "supported" in v:
        rep["decided_by"] = "judge"
        if v.get("supported") is False:
            errors.append(f"hook not supported by what is said: {v.get('reason') or 'judge'}")
    else:
        rep["decided_by"] = "lexical"
        if len(matched) < min_words:
            if judge_skipped_for_time(v):
                # the floor alone never rejects a hook the judge had no time for (most such
                # wrong_hook rejects matched the clip's topic): no verdict, a later review
                # judges the hook
                rep["decided_by"] = "deferred"
                return {**rep, "passed": None, "errors": [],
                        "reason": "the judge was not asked (window time cap) and the lexical "
                                  "floor is not met: deferred to the eye-check"}
            errors.append(f"hook not grounded in the clip: {len(matched)}/{len(content)} "
                          f"content words said ({', '.join(content)})")
    return {**rep, "passed": not errors, "errors": errors}


def judge_skipped_for_time(v):
    """True for a hook judge answer that is the window's time cap, not a verdict or a fault."""
    return isinstance(v, dict) and (v.get("skipped") == "time"
                                    or "judge not asked" in str(v.get("error") or ""))


# Ending: the clip must end on a complete sentence, not on a filler, not mid-word, and the next
# source word must not start before the cut. Uses the second ASR, which ran past the cut.
ENDING_ONSET_TOL_S = 0.12         # = recompose FADE_OUT: a word starting inside the final audio
                                  # fade is attenuated from its onset; earlier, its onset plays
ENDING_MIDWORD_TOL_S = 0.15       # a last word ending this far past the cut was cut inside ...
ENDING_MAX_WORD_S = 1.0           # ... unless it "lasts" longer than any word: whisper stretches
                                  # a segment's last word over the music / noise after it
ENDING_MIN_TAIL_S = 0.3           # source audio needed after the cut to see the next word
# Never a complete ending, whatever punctuation a transcript gives them (brief: "Oh", "um",
# "like", "so", "and" ...): interjections, conjunctions, the quotative "like".
END_FILLERS = frozenset({"oh", "ohh", "ah", "ahh", "uh", "um", "umm", "hmm", "mm", "mhm", "er",
                         "erm", "huh", "eh", "like", "and", "but", "so", "because", "or", "cause",
                         "cuz", "cos"})


def _ends_with_terminal(tok):
    t = _strip_closers(tok)
    return bool(t) and t[-1] in SENTENCE_TERMINALS and \
        t.rstrip(".").lower() not in SENTENCE_ABBREV


def _caption_word_at(cap, w, pad=0.3):
    """The last caption word overlapping heard word ``w`` in time (+-pad), or None."""
    for c in reversed(cap):
        if float(c["start"]) <= float(w["end"]) + pad and float(c["end"]) >= float(w["start"]) - pad:
            return c
    return None


ENDING_ENERGY_HOP_S = 0.02
ENDING_ONSET_PROBE_S = 0.1        # audio right after a word's claimed start that must hold speech
ENDING_SILENCE_DB = -15.0         # a 20 ms frame this far under the clip's median frame level
                                  # is a pause, not a word: 10-19 clip 69's audio sits at -19 to
                                  # -48 dB where whisper put "And" 0.21 s before the cut, while
                                  # speech frames run -10 to +15 dB (item-2 report)
# the second ASR (small) times the word after a pause from the end of the
# word before it, so its claimed onset sits on that word's voice and the probe above hears speech
# there (10-26 1252 / 1253: 'and' claimed 0.21 / 0.24 s before the cut, the voice back 0.31 /
# 0.68 s after it). The word is seated where the audio says its voice starts.
ENDING_ENERGY_AFTER_S = 2.0       # audio read past the cut (= second_asr.DEFAULT_TAIL_S): the
                                  # voice under every word the second ASR heard after the cut
ENDING_VOICE_DB = -10.0           # a frame at least this loud (vs the median) is voice: the
                                  # floor of the speech frames above
ENDING_PAUSE_MIN_S = 0.15         # = boundaries.PAUSE_MIN_S: a shorter quiet run is a stop
                                  # inside a word, not a pause between two
ENDING_CLICK_S = 0.02             # a voiced run this short inside a pause is a click / breath
                                  # (1252: one -8 dB frame 0.2 s into its 0.36 s pause)


def ending_seat_enabled(cfg=None):
    """config quality_gate.ending_seat.enabled (default True): the ending seats the second ASR's
    last heard word on the audio (ending_verdict `seat`). Off: the 10-26 probe alone."""
    if cfg is None:
        try:
            cfg = config.load()
        except (OSError, ValueError):
            cfg = {}
    block = ((cfg.get("quality_gate") or {}).get("ending_seat") or {})
    return bool(block.get("enabled", True))


def cut_energy(src_media, start_s, end_s, before_s=1.0, after_s=ENDING_ENERGY_AFTER_S):
    """(times, db): 20 ms RMS frames of the source audio from ``before_s`` before the cut to
    ``after_s`` after it, times relative to the cut, dB relative to the clip's median frame."""
    import numpy as np
    sr = 16000
    hop = int(ENDING_ENERGY_HOP_S * sr)
    s, e = float(start_s), float(end_s)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{s:.3f}",
                          "-t", f"{e - s + after_s:.3f}", "-i", str(src_media), "-vn", "-ac", "1",
                          "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, timeout=120).stdout
    a = np.frombuffer(raw[:len(raw) // 4 * 4], np.float32)
    n = len(a) // hop
    if n < 2:
        raise RuntimeError("no source audio")
    db = 20 * np.log10(np.sqrt((a[:n * hop].reshape(n, hop).astype(np.float64) ** 2)
                               .mean(axis=1)) + 1e-9)
    cut_i = int(round((e - s) / ENDING_ENERGY_HOP_S))
    ref = float(np.median(db[:max(1, cut_i)]))
    lo = max(0, cut_i - int(round(before_s / ENDING_ENERGY_HOP_S)))
    times = (np.arange(lo, n) - cut_i) * ENDING_ENERGY_HOP_S
    return times.tolist(), (db[lo:] - ref).tolist()


def _silent_between(energy, a, b, silence_db=ENDING_SILENCE_DB):
    """True when the audio frames in [a, b) (cut-relative s) average under silence_db."""
    if not energy or b <= a:
        return False
    times, db = energy
    vals = [d for t, d in zip(times, db) if a <= t < b]
    return bool(vals) and sum(vals) / len(vals) < silence_db


def _pauses(energy, a, b):
    """[(p0, p1)]: the pauses in the cut_energy() frames between a and b (cut-relative s) that the
    voice comes back after inside [a, b): frames under ENDING_VOICE_DB (a voiced run of up to
    ENDING_CLICK_S between two quiet ones is a click), averaging under ENDING_SILENCE_DB, at least
    ENDING_PAUSE_MIN_S long. p0 = where the first quiet frame starts, p1 = where the voice is
    back."""
    if not energy:
        return []
    fr = [(float(t), float(d)) for t, d in zip(*energy) if a - 1e-9 <= float(t) < b]
    runs = []                                       # [voiced, first frame, past the last]
    for k, (_t, d) in enumerate(fr):
        v = d >= ENDING_VOICE_DB
        if runs and runs[-1][0] == v:
            runs[-1][2] = k + 1
        else:
            runs.append([v, k, k + 1])
    click = int(round(ENDING_CLICK_S / ENDING_ENERGY_HOP_S))
    for before, r, after in zip(runs, runs[1:], runs[2:]):
        if r[0] and r[2] - r[1] <= click and not before[0] and not after[0]:
            r[0] = False
    merged = []
    for r in runs:
        if merged and merged[-1][0] == r[0]:
            merged[-1][2] = r[2]
        else:
            merged.append(list(r))
    out = []
    for r, voice in zip(merged, merged[1:]):
        vals = [d for _t, d in fr[r[1]:r[2]]]
        if not r[0] and (r[2] - r[1]) * ENDING_ENERGY_HOP_S >= ENDING_PAUSE_MIN_S - 1e-9 and \
                sum(vals) / len(vals) < ENDING_SILENCE_DB:
            out.append((round(fr[r[1]][0], 3), round(fr[voice[1]][0], 3)))
    return out


def _seat_on_pause(heard, cap, energy, dur, onset_tol):
    """(p0, p1) of the pause the second ASR's last heard word is seated after, else None.
    The word claims a span the audio has a pause in, and the voice comes back after it still
    inside that span, no earlier than `onset_tol` before the cut; the burned captions end on the
    second ASR's word before it (they place the voice before the pause on that word, as select's
    words do for the start-fix's seat: boundaries.seat_words). Its voice then starts at p1."""
    if len(heard) < 2 or not cap or not energy:
        return None
    w, prev = heard[-1], heard[-2]
    cw = _caption_word_at(cap, prev)
    if cw is not cap[-1] or _normalize_core(cw["w"]) != _normalize_core(prev["w"]) or \
            _normalize_core(w["w"]) == _normalize_core(prev["w"]):
        return None
    ws, we = float(w["start"]) - dur, float(w["end"]) - dur
    inside = [p for p in _pauses(energy, ws, we + ENDING_ONSET_PROBE_S) if p[1] < we]
    if inside and inside[-1][1] >= -onset_tol:
        return inside[-1]
    return None


def ending_verdict(asr_words, caption_words, dur, *, tail_s=0.0,
                   onset_tol=ENDING_ONSET_TOL_S, midword_tol=ENDING_MIDWORD_TOL_S,
                   min_tail=ENDING_MIN_TAIL_S, energy=None, seat=True, quiet_end=None):
    """Pure: ``asr_words`` clip-relative second-ASR words (run ``tail_s`` past the cut; None =
    only the rendered clip was heard), ``caption_words`` clip-relative burned caption words,
    ``energy`` = cut_energy() frames (optional). Rules:

      * the captions do not end on a filler / dangling word ("Oh", "um", "like", "and", ...);
      * the last heard word (onset before the final fade, so a next word starting more than
        ``onset_tol`` before the cut IS the last heard word) is not a filler / conjunction;
        whisper's word starts drift by 0.2-1 s around a pause (10-20 clip 116: "And" at -0.03 /
        +0.23 s by two models, the audio is silent until +0.26 s), so a late word whose onset
        the ``energy`` frames show to be silence counts as the next word, not the last; with
        ``seat``, so does one timed from the end of the word before it over a pause
        the voice only comes back after in the final fade or past the cut, when the captions
        end on that word before it (_seat_on_pause: 10-26 1252 / 1253's 'and');
      * it is not cut: it ends <= ``midword_tol`` after the cut (stretched > 1 s words aside;
        ``quiet_end``: the end cut sits in a pause on the audio
        (cut_in_quiet / energy_end_in_quiet), so the word was stretched over it, not cut);
      * the second ASR (which heard past the cut) does not mark it as continuing (", ; : -");
      * it ends a sentence: ". ? !" in the second ASR or on the caption word at that time (one
        model drops a period the other keeps: 10-20 clip 123 "...something of that[.]");
      * there is >= ``min_tail`` of source after the cut, else what follows the cut is unknown
        (10-19 clips 68 / 83 and 10-20 clip 114 were cut at the window edge mid-thought).
    """
    dur = float(dur)
    errors = []
    cap = [w for w in caption_words or [] if str(w.get("w", "")).strip()]
    if cap:
        last_cap = str(cap[-1]["w"])
        if _normalize_core(last_cap) in END_FILLERS or _is_dangling_tail(last_cap):
            errors.append(f"captions end on a filler/dangling word {last_cap!r}")
    heard = [w for w in asr_words or [] if float(w["start"]) < dur - onset_tol]
    nxt = [w for w in asr_words or [] if float(w["start"]) >= dur - onset_tol]
    rep = {"tail_s": None if tail_s is None else round(float(tail_s), 3)}
    while heard and energy:
        ws = float(heard[-1]["start"]) - dur
        if ws > -1.0 and _silent_between(energy, ws, min(ws + ENDING_ONSET_PROBE_S, -onset_tol)):
            nxt.insert(0, heard.pop())              # its onset is a pause: it starts later
            rep["moved_by_energy"] = rep.get("moved_by_energy", 0) + 1
            continue
        p = _seat_on_pause(heard, cap, energy, dur, onset_tol) if seat else None
        if p is None:
            break
        w = heard.pop()                             # its voice starts after the pause
        nxt.insert(0, {**w, "start": round(dur + p[1], 3), "asr_start": w["start"]})
        rep.setdefault("seated", []).append({"w": str(w["w"]), "from": round(ws, 3),
                                             "to": p[1], "pause": [p[0], p[1]]})
    if not heard:
        if not cap:
            return {"passed": None, "reason": "no words in either transcript"}
        return {**rep, "passed": not errors, "errors": errors,
                "note": "second ASR heard no words; caption rules only"}
    last = heard[-1]
    lt = str(last["w"])
    ls, le = float(last["start"]) - dur, float(last["end"]) - dur
    rep.update(last_word=lt, last_word_start_s=round(ls, 3), last_word_end_s=round(le, 3))
    if nxt:
        rep["next_word"] = str(nxt[0]["w"])
        rep["next_word_onset_s"] = round(float(nxt[0]["start"]) - dur, 3)
    core = _normalize_core(lt)
    cw = _caption_word_at(cap, last)
    terminal = _ends_with_terminal(lt) or bool(cw and _ends_with_terminal(cw["w"]))
    # whisper sometimes returns a whole pass lower-case and unpunctuated: its missing period
    # then says nothing, and without captions there is no other punctuation to read. A pass
    # over the rendered clip alone (tail_s None) never heard what follows the cut, so its last
    # word's punctuation is a guess too: only the captions' counts then.
    if tail_s is None:
        terminal = bool(cw and _ends_with_terminal(cw["w"]))
    punctuated = cap or (tail_s is not None and any(_ends_with_terminal(w["w"]) for w in heard))
    if not punctuated:
        rep["note_punct"] = "no punctuation from a pass that heard past the cut, no captions: " \
                            "sentence end not judged"
    # the second ASR heard past the cut: a comma / colon / dash it puts on the last word says
    # the sentence goes on, whatever the caption pass punctuated (10-20 clip 124: captions
    # "losers." / second ASR "losers, because ...")
    continues = tail_s is not None and _strip_closers(lt)[-1:] in (",", ";", ":", "-")
    midword = le > midword_tol and (le - ls) <= ENDING_MAX_WORD_S
    if midword and quiet_end:
        midword = False
        rep["quiet_end"] = f"the cut sits in a pause the second ASR's {lt!r} is stretched over"
    if core in END_FILLERS:
        errors.append(f"ends on a filler/conjunction {lt!r} (starts {ls:+.2f} s from the cut)")
    elif midword:
        errors.append(f"cut inside the word {lt!r} ({ls:+.2f} to {le:+.2f} s around the cut)")
    elif continues:
        errors.append(f"ends mid-sentence: the second ASR hears it continue after {lt!r}"
                      + (f" ({rep['next_word']!r} at {rep['next_word_onset_s']:+.2f} s)"
                         if nxt else ""))
    elif not terminal and punctuated:
        tag = " (dangling)" if _is_dangling_tail(lt) else ""
        errors.append(f"ends mid-sentence{tag}: last word {lt!r}"
                      + (f", next {rep['next_word']!r} at {rep['next_word_onset_s']:+.2f} s"
                         if nxt else ""))
    if tail_s is None:
        rep["note"] = "no source media: the words after the cut were not heard"
    elif float(tail_s) < min_tail:
        errors.append(f"ending unverifiable: the source ends {float(tail_s):.2f} s after the cut "
                      f"(< {min_tail:.1f} s), so the next word cannot be checked")
    return {**rep, "passed": not errors, "errors": errors, "terminal": terminal,
            "onset_tol_s": onset_tol, "midword_tol_s": midword_tol, "min_tail_s": min_tail}


# ---------------------------------------------------------------- the five render checks

CLIP_CHECK_KEYS = ("edge_bands", "crop_slices", "caption_asr", "hook_grounding", "ending")


def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:        # noqa: BLE001 - a measure that cannot run is informational
        return {"passed": None, "reason": f"{type(e).__name__}: {str(e)[:160]}"}


def run_second_asr(video_path, src_media, start_s, end_s):
    """Second-ASR words for the clip: on the source window with its tail when the source is
    there, else on the rendered clip. Returns (rec, tail_s) where tail_s is None when only the
    rendered clip could be transcribed (the words after the cut are unknown)."""
    from clipengine import second_asr
    s = second_asr.settings()
    why = second_asr.independent(s["model"])
    if why:
        raise RuntimeError(why)
    if src_media and Path(src_media).exists() and start_s is not None and end_s is not None:
        rec = second_asr.transcribe(src_media, start_s, end_s)
        return rec, rec["tail_s"]
    d = probe_duration(video_path)
    if not d:
        raise RuntimeError("no source media and the rendered clip has no duration")
    return second_asr.transcribe(video_path, 0.0, d, tail_s=0.0, media_dur=d), None


def clip_checks(video_path, *, words=None, start_s=None, end_s=None, hook=None,
                  src_media=None, crop_views=None, hook_judge=None, asr=None, layout=None,
                  panels=None):
    """the five checks the 10-20 eye-check needed, for EVERY vertical (the full
    gate and the stream lean gate both call this). ``asr`` = (rec, tail_s) injects the second
    ASR (tests); otherwise it runs per quality_gate.second_asr in config.json. ``layout`` /
    ``panels`` = the render's QC layout and panel rows: a dual view's edge bands and slices are
    read on its top panel (DUAL_VIEW_LAYOUT); without its panels the whole canvas is read."""
    from clipengine import second_asr
    if layout == DUAL_VIEW_LAYOUT and (panels or {}).get("top"):
        bands = _safe(measure_dual_view_edge_bands, video_path, panels)
    else:
        bands = _safe(measure_edge_bands, video_path)
    rep = {"edge_bands": bands,
           "crop_slices": _safe(measure_crop_slices, src_media, start_s, end_s, crop_views,
                                layout=layout)}
    cap = clip_words(words, start_s, end_s) if words and start_s is not None \
        and end_s is not None else []
    if start_s is not None and end_s is not None:
        dur = float(end_s) - float(start_s)
    else:
        dur = probe_duration(video_path)
    info = {"model": second_asr.settings()["model"]}
    if asr is None and not second_asr.settings()["enabled"]:
        asr_words, tail, info["reason"] = None, None, "second ASR disabled in config"
    else:
        try:
            rec, tail = asr if asr is not None else run_second_asr(video_path, src_media,
                                                                  start_s, end_s)
            asr_words = rec["words"]
            info.update(model=rec.get("model", info["model"]), words=len(asr_words),
                        tail_s=tail, cached=rec.get("cached"))
        except Exception as e:    # noqa: BLE001
            asr_words, tail = None, None
            info["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    rep["second_asr"] = info
    if asr_words is None:
        why = info.get("error") or info.get("reason")
        rep["caption_asr"] = {"passed": None, "reason": f"no second ASR ({why})"}
        rep["ending"] = _safe(ending_verdict, [], cap, dur or 0.0, tail_s=None) if cap else \
            {"passed": None, "reason": f"no second ASR ({why})"}
    else:
        # a failed check goes to the tie-break model before it stands
        rep["caption_asr"] = _safe(caption_check_with_tiebreak, cap, asr_words, dur,
                                   src_media=src_media if tail is not None else None,
                                   start_s=start_s, end_s=end_s) if cap else \
            {"passed": None, "reason": "no words sidecar (no burned captions to check)"}
        energy = None
        if tail is not None and src_media:
            try:
                energy = cut_energy(src_media, start_s, end_s)
            except Exception:     # noqa: BLE001 - the ending still runs on the words alone
                energy = None
        rep["ending"] = _safe(ending_verdict, asr_words, cap, dur, tail_s=tail, energy=energy,
                              seat=ending_seat_enabled(),
                              quiet_end=cut_audio_enabled() and energy_end_in_quiet(energy))
    rep["hook_grounding"] = _safe(hook_support_verdict, hook, asr_words or [], cap, dur or 0.0,
                                  judge=hook_judge)
    return rep


# ---------------------------------------------------------------- orchestration

def _passing(m):
    return (m or {}).get("passed") is not False       # None (informational) does not fail


def gate(video_path, *, plan=None, clip_meta=None, campaign=None, words=None,
         caption_text=None, start_s=None, end_s=None, src_w=None, hook=None,
         hook_judge=None, run_overlay_check=True, rendered_layout=None,
         blur_fill_frac=None, src_media=None, crop_views=None, run_clip_checks=True,
         clip_checks_judge=None, second_asr_words=None, run_broadcast=True, panels=None):
    """Run every measurement and return a report. ``clip_meta`` is the clip row dict (for
    logging); ``campaign`` is the Campaign dataclass or dict. ``hook`` is the overlay title
    text; when provided together with ``words`` and ``hook_judge`` the hook-support
    measure runs. ``run_overlay_check`` toggles the overlay-edge-cut measure.

    ``blur_fill_frac`` is the blurred-fill share of the canvas recompose reports in its QC
    line; the visual-quality measure fails it above VQ_MAX_BLUR_FILL.

    ``rendered_layout`` is the layout recompose actually rendered (its QC line). When it is
    speaker-less (``NO_SPEAKER_LAYOUTS``) and no campaign layout is configured, the frame gate
    drops the face-presence rule and only fails edge-cut faces. A campaign that names a layout
    (full_bleed_speaker, podcast_two_shot, ...) always keeps the full rule, whatever rendered.
    A ``track`` render with no campaign layout reads the ``layouts.track`` row.

    Any measure that cannot run on the inputs stays as ``{"passed": null, "reason": "..."}``
    and does NOT cause a fail — it is informational. The gate only fails on an explicit False
    from one of the required measures.

    flat edge band, graphics sliced by the crop (needs
    ``src_media`` + the render's ``crop_views``), caption vs second-ASR agreement, hook
    grounding (a missing ``hook`` fails) and the ending, via :func:`clip_checks`.
    ``second_asr_words`` = (rec, tail_s) injects the second ASR (tests).

    Broadcast gate 10-21 (``run_broadcast``): a neighbouring video box sliced by the crop
    (``boxed_slices``), cover frames whose layout is not the clip's steady one (``cover_frame``)
    and a kept face that is not the speaker (``kept_face_speaker``).
    """
    report = {}
    # 1. frame gate — only one that actually reads the mp4 pixels
    card_wins = []
    if words and start_s is not None and end_s is not None:
        try:
            card_wins = auto_card_windows(words, float(start_s),
                                           float(end_s) - float(start_s))
        except Exception as e:
            report["card_windows_error"] = f"{type(e).__name__}: {e}"
    lp = layout_params(campaign)
    if rendered_layout in NO_SPEAKER_LAYOUTS and not lp.get("layout_name"):
        lp = {**lp, "require_subject": False, "rendered_layout": rendered_layout}
    elif rendered_layout == PLAIN_TRACK_LAYOUT and not lp.get("layout_name"):
        # a track render with no campaign layout reads the layouts.track row (its smart
        # measures)
        lp = {**layout_params({"layout": {"layout": PLAIN_TRACK_LAYOUT}}),
              "rendered_layout": rendered_layout}
    try:
        report["frame_gate"] = measure_frame_gate(video_path, card_pull_windows=card_wins,
                                                  params=lp)
    except Exception as e:
        report["frame_gate"] = {"passed": False, "errors": [f"{type(e).__name__}: {e}"]}
    report["layout_params"] = lp
    # 2. crop step + 3. same-subject jumps (from the plan)
    _src_w = src_w or (plan or {}).get("src_w")
    report["crop_step"] = measure_crop_step(plan, int(_src_w or 0))
    # 4. speaker switches
    report["speaker_switches"] = measure_speaker_switches(plan)
    # 5. face centre (informational unless the plan has face samples)
    report["face_centre"] = measure_face_centre(plan, int(_src_w or 0))
    # 6. A/V sync
    try:
        report["av_sync"] = measure_av_sync(video_path)
    except Exception as e:
        report["av_sync"] = {"passed": None, "reason": f"ffprobe error: {e}"}
    # 7. word boundaries (per-word tolerance clean-start/clean-end); a cut the source
    #    audio shows in a pause is in no word
    cut_audio_frames = safe_cut_audio(src_media, start_s, end_s) if words else None
    report["word_boundaries"] = measure_word_boundaries(
        words, start_s, end_s, audio=cut_audio_frames) if words else \
        {"passed": None, "reason": "no words sidecar"}
    # 7b. sentence-complete bounds
    if words is not None and start_s is not None and end_s is not None:
        try:
            report["sentence_bounds"] = measure_sentence_bounds(words, start_s, end_s,
                                                                audio=cut_audio_frames)
        except Exception as e:        # noqa: BLE001
            report["sentence_bounds"] = {"passed": None, "reason": f"{type(e).__name__}: {e}"}
    else:
        report["sentence_bounds"] = {"passed": None, "reason": "no words sidecar"}
    # 8. caption
    report["caption"] = measure_caption(caption_text, campaign) if campaign else \
        {"passed": None, "reason": "no campaign bound"}
    # 9. hook support — needs an LLM judge; informational when not provided
    report["hook_support"] = measure_hook_support(hook, words, judge=hook_judge)
    # 10. layout-by-construction
    report["layout_match"] = measure_layout_match(plan, campaign)
    # 10b. active-speaker framing — plan-side alignment first, then audio-visual
    #     cross-check on the rendered mp4 (opt-in via gate_check_active_speaker on the campaign
    #     layout; default informational so a noisy mouth-motion heuristic cannot drop an
    #     approved clip).
    try:
        report["active_speaker_plan"] = measure_active_speaker_plan(plan)
    except Exception as e:        # noqa: BLE001
        report["active_speaker_plan"] = {"passed": None,
                                          "reason": f"{type(e).__name__}: {e}"}
    opt_in_av = False
    if campaign is not None:
        layout_block = (campaign.get("layout") if isinstance(campaign, dict)
                        else getattr(campaign, "layout", None)) or {}
        opt_in_av = bool(layout_block.get("gate_check_active_speaker"))
    # Only RUN the audio-visual check when a campaign opts in — the measure is heavy (YuNet
    # face detection + mouth motion on every sampled frame of the rendered mp4) and most
    # campaigns do not need it. Opt-in via ``campaign.layout.gate_check_active_speaker``.
    if opt_in_av:
        try:
            av = measure_active_speaker_audio_visual(video_path)
        except Exception as e:        # noqa: BLE001
            av = {"passed": None, "reason": f"{type(e).__name__}: {e}"}
    else:
        av = {"passed": None, "reason": "not opted in (gate_check_active_speaker=False)"}
    report["active_speaker_av"] = av
    # 10c. layout quality — pixel-level checks for layouts with hard sub-panels.
    # Opt-in via ``campaign.layout.gate_check_layout_quality`` because the measure runs
    # YuNet detection on every sampled frame of the rendered mp4 (slow) AND the semantics
    # changed later (chart_top_cam_bottom now covers chart on top + cam zoomed on bottom):
    # historical clips rendered under the old broadcast_vf would be flagged by the new rule,
    # which is correct for live renders but wrong for calibration against existing files.
    opt_in_lq = False
    if campaign is not None:
        layout_block = (campaign.get("layout") if isinstance(campaign, dict)
                        else getattr(campaign, "layout", None)) or {}
        opt_in_lq = bool(layout_block.get("gate_check_layout_quality"))
    if opt_in_lq:
        try:
            report["layout_quality"] = measure_layout_quality(video_path, campaign)
        except Exception as e:        # noqa: BLE001
            report["layout_quality"] = {"passed": None,
                                         "reason": f"{type(e).__name__}: {e}"}
    else:
        report["layout_quality"] = {"passed": None,
                                     "reason": "not opted in (gate_check_layout_quality=False)"}
    # 11. overlay edge-cut. Opt-in per campaign via
    # ``campaign.layout.gate_check_overlay_edge_cut`` (default False). Still RUNS by default
    # but demotes the verdict to informational (``passed: None``) so a noisy detector cannot
    # fail an approved clip until the campaign explicitly asks for it.
    opt_in_overlay = False
    if campaign is not None:
        layout_block = (campaign.get("layout") if isinstance(campaign, dict)
                        else getattr(campaign, "layout", None)) or {}
        opt_in_overlay = bool(layout_block.get("gate_check_overlay_edge_cut"))
    if run_overlay_check:
        try:
            m = measure_overlay_edge_cut(video_path)
        except Exception as e:        # noqa: BLE001
            m = {"passed": None, "reason": f"{type(e).__name__}: {e}"}
        if not opt_in_overlay and m.get("passed") is False:
            m = {**m, "passed": None,
                 "note": "informational: campaign did not opt in via gate_check_overlay_edge_cut"}
        report["overlay_edge_cut"] = m
    else:
        report["overlay_edge_cut"] = {"passed": None, "reason": "overlay check disabled"}

    # 11b. boxed broadcast (broadcast gate 10-21): a neighbouring video box the crop sliced (blur
    #      pad or not), and cover frames that are not the clip's steady layout
    if run_broadcast:
        try:
            report["boxed_slices"], report["cover_frame"] = measure_broadcast_boxes(video_path)
        except Exception as e:        # noqa: BLE001
            why = {"passed": None, "reason": f"{type(e).__name__}: {str(e)[:160]}"}
            report["boxed_slices"], report["cover_frame"] = why, dict(why)
        # 11c. the face the crop keeps is the one speaking
        report["kept_face_speaker"] = _safe(measure_kept_face_speaker, video_path, words,
                                            start_s, end_s, rendered_layout)
    else:
        for k in BROADCAST_KEYS:
            report[k] = {"passed": None, "reason": "broadcast checks disabled"}

    # 12. visual quality: brightness / contrast / sharpness / blur fill (sliced
    #     neighbour boxes under a blurred pad count as fill: broadcast gate 10-21)
    try:
        report["visual_quality"] = measure_visual_quality(
            video_path, blur_fill_frac=blur_fill_frac,
            sliver_frac=(report.get("boxed_slices") or {}).get("sliver_frac"))
    except Exception as e:        # noqa: BLE001
        report["visual_quality"] = {"passed": None, "reason": f"{type(e).__name__}: {e}"}

    # 13. edge band, crop slices, caption vs second ASR, hook, ending
    if run_clip_checks:
        report.update(clip_checks(video_path, words=words, start_s=start_s, end_s=end_s,
                                    hook=hook, src_media=src_media, crop_views=crop_views,
                                    hook_judge=clip_checks_judge, asr=second_asr_words,
                                    layout=rendered_layout, panels=panels))

    verdict, reason = decide(report)
    report["verdict"] = verdict
    report["drop_reason"] = reason
    return report


def decide(report):
    """Collapse a measurement report to ``(verdict, drop_reason)``. Verdict is ``'pass'`` or
    ``'fail'``. ``drop_reason`` is a short string, empty when passing."""
    reasons = []
    def _fails(k, label):
        m = report.get(k) or {}
        if m.get("passed") is False:
            errs = m.get("errors") or []
            tail = f": {'; '.join(errs)}" if errs else ""
            reasons.append(f"{label}{tail}")
    _fails("frame_gate", "frame_gate")
    _fails("crop_step", "crop_step")
    _fails("speaker_switches", "speaker_switches")
    _fails("face_centre", "face_centre")
    _fails("av_sync", "av_sync")
    _fails("word_boundaries", "word_boundaries")
    _fails("sentence_bounds", "sentence_bounds")
    _fails("caption", "caption")
    _fails("hook_support", "hook_support")
    _fails("layout_match", "layout_match")
    _fails("overlay_edge_cut", "overlay_edge_cut")
    _fails("active_speaker_plan", "active_speaker_plan")
    _fails("active_speaker_av", "active_speaker_av")
    _fails("layout_quality", "layout_quality")
    _fails("visual_quality", "visual_quality")
    for k in CLIP_CHECK_KEYS + BROADCAST_KEYS:
        _fails(k, k)
    return ("fail" if reasons else "pass"), "; ".join(reasons)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--plan", help="reframe plan JSON")
    ap.add_argument("--words", help="SOURCE-time words sidecar JSON")
    ap.add_argument("--start-s", type=float)
    ap.add_argument("--end-s", type=float)
    ap.add_argument("--caption", help="text file with the posted caption")
    ap.add_argument("--campaign", help="JSON file of a Campaign-like dict")
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text()) if args.plan else None
    words = json.loads(Path(args.words).read_text()) if args.words else None
    if isinstance(words, dict):
        words = words.get("words") or []
    cap = Path(args.caption).read_text() if args.caption else None
    camp = json.loads(Path(args.campaign).read_text()) if args.campaign else None
    rep = gate(args.video, plan=plan, words=words, start_s=args.start_s, end_s=args.end_s,
               caption_text=cap, campaign=camp)
    print(json.dumps(rep, indent=2, default=str))
    sys.exit(0 if rep["verdict"] == "pass" else 3)
