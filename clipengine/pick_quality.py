#!/usr/bin/env python3
"""pre-render pick-quality filters.

The LLM selector happily picks viral moments that the frame_gate will drop at
render time — card-pull windows of a handheld vlog are the canonical case: the HOST's
face is occluded by hands + cards for 15-20 % of the window, so even a correctly
rendered full_bleed_speaker clip fails the strict 2.0 % frame_gate cap.

The rule: score each candidate pick by how often the host face is
detected on the SOURCE (1 fps samples) and SKIP picks below the frame_gate bar
BEFORE the LLM pick lands on disk. This module is the engine side of that rule.

Call shape:
    from clipengine import pick_quality
    frac = pick_quality.face_visibility_fraction(src, start_s, end_s, fps=1.0)
    picks = pick_quality.filter_picks_by_face_visibility(src, picks, min_frac=0.98)

Soft dependencies on cv2 / render.reframe. When they are not importable the
module returns ``None`` fractions and keeps every pick (fail-open), so a build
without OpenCV does not silently drop every candidate.
"""
from __future__ import annotations

from typing import Iterable, List, Optional


def _reframe():
    try:
        from clipengine.render import reframe as _rf          # noqa: WPS433
    except ImportError:
        return None
    return _rf


def face_visibility_fraction(src, start_s: float, end_s: float, fps: float = 1.0):
    """Fraction of 1 fps samples in ``[start_s, end_s]`` where at least one face is
    detected on the source. Returns ``{"frac": float|None, "n_total": int,
    "n_with_face": int}``. ``frac`` is None when cv2 / the detector is unavailable;
    callers must fail open on None.
    """
    rf = _reframe()
    if rf is None:
        return {"frac": None, "n_total": 0, "n_with_face": 0,
                "reason": "render.reframe unavailable"}
    try:
        src_w, src_h = rf.probe_size(str(src))
    except Exception as e:                        # noqa: BLE001
        return {"frac": None, "n_total": 0, "n_with_face": 0,
                "reason": f"probe failed: {type(e).__name__}: {e}"}
    dur = max(0.0, float(end_s) - float(start_s))
    if dur <= 0:
        return {"frac": None, "n_total": 0, "n_with_face": 0,
                "reason": "zero or negative duration"}
    n_total = 0
    n_with_face = 0
    try:
        for _t, frame in rf.iter_frames(str(src), float(start_s), dur, fps=fps,
                                         src_size=(src_w, src_h)):
            n_total += 1
            try:
                faces = rf.detect_faces(frame)
            except Exception:                     # noqa: BLE001
                faces = []
            if faces:
                n_with_face += 1
    except Exception as e:                        # noqa: BLE001
        return {"frac": None, "n_total": n_total, "n_with_face": n_with_face,
                "reason": f"sample loop failed: {type(e).__name__}: {e}"}
    if n_total == 0:
        return {"frac": None, "n_total": 0, "n_with_face": 0,
                "reason": "no samples decoded"}
    return {"frac": round(n_with_face / n_total, 4),
            "n_total": n_total, "n_with_face": n_with_face}


def filter_picks_by_face_visibility(src, picks: Iterable[dict],
                                     min_frac: float = 0.98,
                                     fps: float = 1.0,
                                     start_key: str = "start_s",
                                     end_key: str = "end_s") -> List[dict]:
    """Keep only picks whose source face-visibility fraction is >= ``min_frac``.
    Each pick is tagged with its visibility measurement in
    ``pick["face_visibility"] = {"frac", "n_total", "n_with_face"}`` so the
    driver can show the reason in the drop list.

    Behaviour on measurement failure (``frac is None``): KEEP the pick (fail
    open) — a broken detector should not silently empty the selector. The driver
    still gets the gate's authoritative verdict at render time.
    """
    out = []
    for p in picks:
        start_s = p.get(start_key)
        end_s = p.get(end_key)
        if start_s is None or end_s is None:
            p = dict(p)
            p["face_visibility"] = {"frac": None, "reason": "missing start/end"}
            out.append(p)
            continue
        vis = face_visibility_fraction(src, float(start_s), float(end_s), fps=fps)
        p = dict(p)
        p["face_visibility"] = vis
        frac = vis.get("frac")
        if frac is None:
            # measurement failed — keep the pick (fail open)
            out.append(p)
            continue
        if frac + 1e-9 >= float(min_frac):
            out.append(p)
    return out


def predict_frame_gate(src, start_s: float, end_s: float, fps: float = 2.0,
                       fail_pct_max: float = 2.0, broll_min_span_s: float = 3.0,
                       broll_max_fraction: float = 0.5):
    """predict the render-time frame gate for a speaker-crop pick from the
    SOURCE, before anything is rendered. A source sample with no detectable face can't show a
    face after a 9:16 crop either, so the no-face share of the source is a floor on the
    rendered fail share. Applies the gate's own exemption shape: a contiguous no-face run of
    at least ``broll_min_span_s`` is b-roll (exempt), capped at ``broll_max_fraction`` of the
    clip. ``passed`` is False when the remaining no-face share exceeds ``fail_pct_max`` (the
    gate's cap), None when the source can't be measured — callers must fail open on None.
    """
    rf = _reframe()
    dur = max(0.0, float(end_s) - float(start_s))
    if rf is None or dur <= 0:
        return {"passed": None, "frac": None, "reason": "reframe unavailable or empty window"}
    flags = []
    try:
        src_w, src_h = rf.probe_size(str(src))
        for _t, frame in rf.iter_frames(str(src), float(start_s), dur, fps=fps,
                                         src_size=(src_w, src_h)):
            try:
                flags.append(bool(rf.detect_faces(frame)))
            except Exception:                     # noqa: BLE001
                flags.append(False)
    except Exception as e:                        # noqa: BLE001
        return {"passed": None, "frac": None, "reason": f"sample failed: {type(e).__name__}: {e}"}
    n = len(flags)
    if n == 0:
        return {"passed": None, "frac": None, "reason": "no samples decoded"}
    runs, i = [], 0                               # contiguous no-face sample runs (lo, hi)
    while i < n:
        if flags[i]:
            i += 1
            continue
        j = i
        while j < n and not flags[j]:
            j += 1
        runs.append((i, j))
        i = j
    cap = broll_max_fraction * n
    exempt = 0
    for lo, hi in sorted(runs, key=lambda r: r[1] - r[0], reverse=True):
        if (hi - lo) / float(fps) >= broll_min_span_s and exempt + (hi - lo) <= cap:
            exempt += hi - lo
    misses = n - sum(flags)
    fail_pct = round((misses - exempt) * 100.0 / n, 2)
    return {"passed": fail_pct <= float(fail_pct_max), "fail_pct": fail_pct,
            "frac": round(sum(flags) / n, 4), "n_total": n, "n_with_face": sum(flags),
            "broll_exempt": exempt, "fps": fps}


__all__ = ["face_visibility_fraction", "filter_picks_by_face_visibility", "predict_frame_gate"]
