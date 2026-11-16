"""Tracking tests. Pure helpers so no cv2 / ffmpeg needed.

Covers:
- Zero-phase gaussian smoothing has zero lag (symmetric around a step).
- Velocity clamp keeps per-sample moves <= 1.2% src width.
- Synthetic panning: the smoothed crop keeps the face centre within +-10% of crop width.
- Silent-listener switch is blocked by pick_speakers (track with zero activity never wins).
- Edge-cut face: a face whose bbox touches the frame edge is rejected by the gate's
  face_fully_inside check.
- Same-subject jump: track_crops marks `cut=True` whenever consecutive segments differ by
  more than SWITCH_MIN.
- Overlay-aware crop: a burned-in source band either stays whole inside or outside the crop.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from clipengine.render import reframe
from clipengine.render import speaker


def face(cx, h=0.25, cy=0.4, track_id=None):
    f = {"cx": float(cx), "cy": float(cy), "w": h * 0.56, "h": float(h), "score": 0.95}
    if track_id is not None:
        f["track_id"] = int(track_id)
    return f


# ---------------------------------------------------------------- zero-phase smoothing

def test_zero_phase_gaussian_zero_lag_on_step():
    # A perfect step in cx should smooth symmetrically: the smoothed value at the step
    # is the mean of the pre/post values, not shifted later.
    xs = [0.0] * 50 + [1.0] * 50
    sm = reframe._zero_phase_gaussian(xs, sigma=3.0)
    assert abs(sm[49] + sm[50] - (sm[48] + sm[51])) < 1e-9    # mirror symmetry around the step
    # The midpoint value is roughly 0.5 by symmetry.
    assert 0.4 < (sm[49] + sm[50]) / 2 < 0.6


def test_zero_phase_gaussian_preserves_endpoints():
    xs = [0.3] * 20
    sm = reframe._zero_phase_gaussian(xs, sigma=2.0)
    assert all(abs(v - 0.3) < 1e-6 for v in sm)


# ---------------------------------------------------------------- velocity clamp

def test_velocity_clamp_bounds_max_step():
    xs = [0.0, 0.0, 10.0, 10.0, 20.0]
    out = reframe._clamp_velocity_px(xs, max_step_px=2.0)
    # Forward pass caps the jump at 2.0 per step.
    diffs = [b - a for a, b in zip(out, out[1:])]
    assert all(abs(d) <= 2.0 + 1e-9 for d in diffs)


# ---------------------------------------------------------------- panning face: ±10% centre

def test_smooth_tracked_crop_keeps_face_centre_within_10pct():
    """Simulate a handheld shot where the face pans slowly across the frame. The smoothed
    crop should keep the face centre within +-10% of the crop width every sample."""
    src_w, src_h = 1920, 1080
    cw = reframe.crop_width(src_w, src_h)      # 608 for 9:16 at 1080 height
    dur = 10.0
    fps = reframe.SMOOTH_SAMPLE_FPS
    n = int(dur * fps) + 1
    # face pans linearly from cx=0.35 to cx=0.65 over the clip (0.5 Hz worth of movement)
    samples = []
    for i in range(n):
        t = i / fps
        cx = 0.35 + 0.30 * (t / dur)
        samples.append({"t": round(t, 3), "faces": [face(cx, track_id=1)]})
    segs = reframe.smooth_tracked_crops(samples, [], dur, src_w, src_h)
    # evaluate crop centre at each sample and check face_cx stays within +-10% crop width
    def crop_x_at(t):
        for s in segs:
            if s["t0"] <= t < s["t1"]:
                return s["x"]
        return segs[-1]["x"]
    bad = []
    for i in range(n):
        t = i / fps
        cx_px = (0.35 + 0.30 * (t / dur)) * src_w
        x = crop_x_at(t)
        centre_px = x + cw / 2
        off = abs(cx_px - centre_px) / cw
        if off > 0.10 + 1e-6:
            bad.append((round(t, 2), round(off, 3)))
    assert not bad, f"face slid outside ±10% at {bad[:4]}"


def test_smooth_tracked_crop_velocity_cap_satisfies_15pct_per_frame():
    """The smoothed trajectory never jumps by more than SMOOTH_MAX_VEL_PCT * src_w per 100 ms
    step — gate limit is 1.5% / frame; we come in at <=1.2% to have headroom."""
    src_w, src_h = 1920, 1080
    dur = 6.0
    fps = reframe.SMOOTH_SAMPLE_FPS
    n = int(dur * fps) + 1
    # worst case: face hops instantly from cx=0.1 to cx=0.9 at t=3 s (shouldn't happen without a
    # shot cut, but if it does, the clamp should pull it apart)
    samples = []
    for i in range(n):
        t = i / fps
        cx = 0.1 if t < 3.0 else 0.9
        samples.append({"t": round(t, 3), "faces": [face(cx, track_id=1)]})
    segs = reframe.smooth_tracked_crops(samples, [], dur, src_w, src_h)
    # Within a single shot every segment except the first is a "cut=False" smoothed step; the
    # velocity clamp only applies to those non-cut steps.
    diffs = []
    for a, b in zip(segs, segs[1:]):
        if b.get("cut"):
            continue
        diffs.append(abs(b["x"] - a["x"]))
    # segments store integer pixel x, so allow 1 px for int(round(...)) coming out of the clamp
    cap_px = int(round(reframe.SMOOTH_MAX_VEL_PCT * src_w)) + 1
    assert diffs, segs
    max_step = max(diffs)
    assert max_step <= cap_px, f"step {max_step} > cap {cap_px}"


# ---------------------------------------------------------------- silent listener switch blocked

def _speech_bool_always_on(_t):
    return True


def test_pick_speakers_never_switches_to_silent_listener():
    # Two tracks: 1 talks (activity 0.9), 2 is a silent nod (activity 0.0). pick_speakers must
    # keep track 1 and never switch to the silent face even if present.
    hops = [(t, {1: 0.9, 2: 0.0}) for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]]
    segs = speaker.pick_speakers(hops, _speech_bool_always_on, hop_s=0.5)
    assert all(s["track_id"] == 1 for s in segs)


def test_pick_speakers_requires_continuous_lead_before_switch():
    # Track 1 talks, track 2 wins a single hop by a huge margin (laughs/backchannel) then
    # falls back. pick_speakers must not switch.
    hops = []
    for i in range(10):
        t = i * 0.5
        hops.append((t, {1: 1.0, 2: 2.0 if i == 5 else 0.0}))
    segs = speaker.pick_speakers(hops, _speech_bool_always_on, hop_s=0.5,
                                 min_hold_s=1.0, switch_margin=0.1,
                                 min_candidate_run_s=1.2)
    assert all(s["track_id"] == 1 for s in segs), [s["track_id"] for s in segs]


# ---------------------------------------------------------------- edge-cut face

def test_face_fully_inside_rejects_edge_touching_bbox():
    # Shared helper used by render/finish.py and clipengine/quality_gate.py so both agree on what
    # "the frame cuts a face" means.
    from clipengine.render import finish
    assert finish.face_fully_inside((5, 50, 100, 100), 1080, 1920, pad=10) is False
    assert finish.face_fully_inside((50, 50, 100, 100), 1080, 1920, pad=10) is True
    assert finish.face_fully_inside((1075 - 100, 50, 100, 100), 1080, 1920, pad=10) is False


# ---------------------------------------------------------------- same-subject jump

def test_track_crops_marks_hard_cut_at_subject_switch():
    # In a 12-second shot (longer than DRIFT_MIN_SHOT=6.0), a mid-shot primary-face hop larger
    # than SWITCH_MIN produces a hard-cut segment boundary — track_crops splits the shot into
    # 3-second windows and the >20%-of-width x delta is flagged cut=True.
    n = 120
    s = []
    for i in range(n):
        t = i / 10.0
        cx = 0.1 if t < 6.0 else 0.9
        s.append({"t": round(t, 3), "faces": [face(cx, track_id=1)]})
    segs = reframe.track_crops(s, [], 12.0, 1920, 1080)
    assert len(segs) >= 2, segs
    # Every subsequent segment with an x delta > SWITCH_MIN * src_w is a hard cut.
    for a, b in zip(segs, segs[1:]):
        if abs(b["x"] - a["x"]) > reframe.SWITCH_MIN * 1920:
            assert b.get("cut") is True


# ---------------------------------------------------------------- overlay-aware crop

def test_overlay_aware_prefers_containing_fitting_overlay():
    """A banner that fits inside the crop width is swallowed, not clipped."""
    src_w = 1920
    cw = 608
    # Overlay centred around x=900, 300 px wide; crop should contain it whole.
    overlay = (750, 1050)
    x = reframe.adjust_crop_for_overlay(x_px=700, cw=cw, overlay_box=overlay, src_w=src_w)
    assert x <= overlay[0] and x + cw >= overlay[1], (x, overlay, cw)


def test_overlay_aware_excludes_fully_when_too_wide():
    """A banner that is wider than the crop (impossible to contain) must be entirely outside the
    crop — never half-visible — provided an exclusion position exists in the source frame."""
    src_w = 1920
    cw = 608
    overlay = (400, 1200)   # 800 px wide; too wide to fit 608 crop but can be excluded on right
    x = reframe.adjust_crop_for_overlay(x_px=600, cw=cw, overlay_box=overlay, src_w=src_w)
    # Either fully to the right of the overlay, or (impossible here) fully to its left.
    assert x >= overlay[1] or x + cw <= overlay[0], (x, overlay, cw)


def test_overlay_aware_passthrough_when_already_clean():
    src_w = 1920
    cw = 608
    overlay = (0, 100)     # banner fully to the left of a right-biased crop
    x = reframe.adjust_crop_for_overlay(x_px=800, cw=cw, overlay_box=overlay, src_w=src_w)
    assert x == 800
