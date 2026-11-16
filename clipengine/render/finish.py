#!/usr/bin/env python3
"""Finish stage: snap a selected clip's start/end to clean boundaries, and apply the campaign's
rendered-on-frame overlays (watermark, top chip, etc.) on the already-rendered 9:16 output.

Two things live here because they go together — both operate on an existing clip (either the
selection boundaries in the job store, or the rendered mp4 file on disk) and both are driven
by the campaign's layout/finish config, not by per-campaign branches in Python.

Public API:

* ``snap_bounds(words, start_s, end_s, *, source_face_series=None, min_dur_s=15.0,
  max_shift_s=6.0, hold_s=1.0, min_big_face_h=0.20, bigface_ratio=0.9)`` picks a new
  (start_s, end_s) on word boundaries where (optionally) the next ``hold_s`` seconds of the
  SOURCE at the start/end have >=``bigface_ratio`` of frames showing a face at least
  ``min_big_face_h`` tall. Pure: no ffmpeg; the caller passes the pre-sampled face series.
  If no shift satisfies both ends while keeping ``end - start >= min_dur_s`` the originals are
  returned together with a note.
* ``source_face_series(src, abs_start, abs_end, fps=10.0)`` samples the SOURCE media at
  ``fps`` and returns ``[(t_src, has_face, biggest_h, biggest_cx)]`` — the probe used by
  snap_bounds. Imports render.speaker only when called so headless unit tests do not need cv2.
* ``apply_finish(in_mp4, out_mp4, layout_cfg)`` burns the campaign's overlays onto an existing
  rendered mp4. Supported overlay kinds:
    * ``watermark`` — bottom-right image at a given opacity + relative width
    * ``top_chip``  — top safe-zone text badge with a dark band and fade-in
    * ``none``      — passthrough (copy)

No subprocess is run when ``layout_cfg['overlays']`` is empty: ``in_mp4`` is simply linked /
copied to ``out_mp4`` so the finish stage is a safe no-op for layouts that need no overlay.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

# Defaults picked to match the onboarding scripts. They are a sane fallback when
# the campaign config does not override them; the engine does NOT branch per-campaign in code.
DEFAULT_SNAP = {
    "min_dur_s": 15.0,
    "max_shift_s": 6.0,
    "hold_s": 1.0,
    "min_big_face_h": 0.20,
    "bigface_ratio": 0.9,
    "head_noface_s": 1.0,     # no-face frames in first head_noface_s of the clip is a hard fail
    "tail_noface_s": 1.0,
}

WATERMARK_WIDTH_FRAC = 0.22       # watermark scaled to 22% of 1080
WATERMARK_PAD = 24                # pixels inset from each edge
WATERMARK_ALPHA = 0.80

CHIP_Y = 120
CHIP_W = 720
CHIP_H = 100
CHIP_HOLD_T = 3.4              # chip appears after the hook title fades
CHIP_FADE_S = 0.3


# ---------------------------------------------------------------- word-boundary snapping

def _nearest_word_boundary(words, target_s, prefer="start"):
    """Return the ``start`` or ``end`` timestamp of the word in ``words`` closest to
    ``target_s``. Returns ``target_s`` when ``words`` is empty."""
    if not words:
        return float(target_s)
    best, best_dt = None, None
    key = "start" if prefer == "start" else "end"
    for w in words:
        t = float(w[key])
        dt = abs(t - float(target_s))
        if best_dt is None or dt < best_dt:
            best, best_dt = t, dt
    return float(best if best is not None else target_s)


def _bigface_ratio(series, lo, hi, min_big_face_h):
    """Fraction of frames in ``series`` with ``lo <= t <= hi`` that have a face at least
    ``min_big_face_h`` tall. Returns 0.0 for an empty window."""
    pts = [1 if (has and h >= min_big_face_h) else 0
           for (t, has, h, _cx) in series if lo <= t <= hi]
    return (sum(pts) / len(pts)) if pts else 0.0


def snap_bounds(words, start_s, end_s, *, source_face_series=None,
                min_dur_s=None, max_shift_s=None, hold_s=None,
                min_big_face_h=None, bigface_ratio=None):
    """Snap (start_s, end_s) to a nearby word boundary where the SOURCE head and tail hold a
    big face for ``hold_s`` seconds.

    Returns ``(new_start, new_end, info)``. ``info`` is a dict with the big-face ratios at
    both ends, the applied shifts (``shift_start_s``, ``shift_end_s``), and a ``note`` string
    when the originals had to be kept.

    ``source_face_series`` is the list produced by :func:`source_face_series`; absolute SOURCE
    seconds. When ``None``, the function returns the originals snapped to word boundaries
    (no face check). ``words`` are absolute SOURCE seconds, same shape as the pipeline sidecar.
    """
    cfg = {**DEFAULT_SNAP,
           **{k: v for k, v in {
               "min_dur_s": min_dur_s, "max_shift_s": max_shift_s, "hold_s": hold_s,
               "min_big_face_h": min_big_face_h, "bigface_ratio": bigface_ratio,
           }.items() if v is not None}}
    orig_start, orig_end = float(start_s), float(end_s)
    if not words:
        return orig_start, orig_end, {"note": "no words available"}

    # Candidate boundaries: starts near orig_start, ends near orig_end, inside shift windows.
    cand_starts = sorted({float(w["start"]) for w in words
                          if orig_start - 1.0 <= float(w["start"]) <= orig_start + cfg["max_shift_s"]})
    cand_ends = sorted({float(w["end"]) for w in words
                        if orig_end - cfg["max_shift_s"] <= float(w["end"]) <= orig_end + 0.5},
                       reverse=True)
    if not cand_starts:
        cand_starts = [_nearest_word_boundary(words, orig_start, "start")]
    if not cand_ends:
        cand_ends = [_nearest_word_boundary(words, orig_end, "end")]

    if source_face_series is None:
        new_s, new_e = cand_starts[0], cand_ends[0]
        if new_e - new_s < cfg["min_dur_s"]:
            return orig_start, orig_end, {"note": "word-only snap would under-cut min_dur"}
        return new_s, new_e, {
            "shift_start_s": round(new_s - orig_start, 3),
            "shift_end_s": round(new_e - orig_end, 3),
            "note": "no face series; word-boundary only",
        }

    hold_s = float(cfg["hold_s"])
    min_h = float(cfg["min_big_face_h"])
    want = float(cfg["bigface_ratio"])

    chosen_start = orig_start
    for ts in cand_starts:
        if _bigface_ratio(source_face_series, ts, ts + hold_s, min_h) >= want:
            chosen_start = ts
            break
    chosen_end = orig_end
    for te in cand_ends:
        if _bigface_ratio(source_face_series, te - hold_s, te, min_h) >= want:
            chosen_end = te
            break
    info_base = {
        "ratio_start_1s_bigface": round(_bigface_ratio(source_face_series,
                                                        chosen_start, chosen_start + hold_s,
                                                        min_h), 3),
        "ratio_end_1s_bigface": round(_bigface_ratio(source_face_series,
                                                      chosen_end - hold_s, chosen_end,
                                                      min_h), 3),
    }
    if chosen_end - chosen_start < cfg["min_dur_s"]:
        return orig_start, orig_end, {
            **info_base,
            "note": "snap rejected (would under-cut min_dur)",
        }
    return chosen_start, chosen_end, {
        **info_base,
        "shift_start_s": round(chosen_start - orig_start, 3),
        "shift_end_s": round(chosen_end - orig_end, 3),
    }


def source_face_series(src, abs_start, abs_end, fps=10.0):
    """Sample the SOURCE media at ``fps`` and return
    ``[(t_src, has_face, biggest_h, biggest_cx)]``.

    Imports render.speaker lazily so unit tests that pass a pre-built series via
    :func:`snap_bounds` don't need cv2 / ffmpeg at import time."""
    from clipengine.render import speaker as _speaker  # type: ignore
    out = []
    for t_rel, frame in _speaker.iter_frames_native(str(src), float(abs_start),
                                                     float(abs_end) - float(abs_start),
                                                     fps=float(fps)):
        faces = _speaker.detect_faces_with_landmarks(frame) or []
        big = max((f for f in faces if f.get("h", 0.0) >= 0.08),
                  key=lambda f: f["h"], default=None)
        out.append((float(abs_start) + float(t_rel),
                    big is not None,
                    float(big["h"]) if big else 0.0,
                    float(big["cx"]) if big else 0.0))
    return out


# ---------------------------------------------------------------- frame-level face check

EDGE_PX_DEFAULT = 10          # face bbox this close to any edge is "cut by the edge"


def face_fully_inside(bbox_xywh, src_w, src_h, pad=EDGE_PX_DEFAULT):
    """True when a face bbox ``(x, y, w, h)`` in SOURCE pixels sits ``pad`` pixels inside every
    frame edge. Shared by :mod:`render.finish` and :mod:`clipengine.quality_gate` so the two
    stages agree on what "the frame cuts a face" means."""
    x, y, w, h = bbox_xywh
    return (x > pad and y > pad and (x + w) < (src_w - pad) and (y + h) < (src_h - pad))


# ---------------------------------------------------------------- head / tail no-face checks

def head_tail_face_ok(series, start_s, end_s, *, head_noface_s=None, tail_noface_s=None,
                     min_big_face_h=None):
    """Pure helper that returns ``True`` when the first ``head_noface_s`` seconds AND the last
    ``tail_noface_s`` seconds of ``[start_s, end_s]`` on the SOURCE contain a big face at every
    sampled frame. The caller feeds the same ``source_face_series``.

    Used by the engine's gate to confirm an open/close that will not frame an empty couch, a
    wall, the back of a head, etc. (open and close on a face)."""
    cfg = {**DEFAULT_SNAP,
           **{k: v for k, v in {"head_noface_s": head_noface_s,
                                 "tail_noface_s": tail_noface_s,
                                 "min_big_face_h": min_big_face_h}.items() if v is not None}}
    h = float(cfg["head_noface_s"]); t = float(cfg["tail_noface_s"])
    mh = float(cfg["min_big_face_h"])
    head = [1 for (ts, has, hh, _cx) in series
            if start_s <= ts <= start_s + h and has and hh >= mh]
    tail = [1 for (ts, has, hh, _cx) in series
            if end_s - t <= ts <= end_s and has and hh >= mh]
    # a window shorter than one sample is not a fail; we only reject when we have samples
    head_total = sum(1 for (ts, *_r) in series if start_s <= ts <= start_s + h)
    tail_total = sum(1 for (ts, *_r) in series if end_s - t <= ts <= end_s)
    ok_head = head_total == 0 or len(head) == head_total
    ok_tail = tail_total == 0 or len(tail) == tail_total
    return ok_head and ok_tail


# ---------------------------------------------------------------- overlays on the finished mp4

def _pick_font():
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"):
        if Path(p).exists():
            return p
    return None


def _ffmpeg(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-6:]
        raise RuntimeError(f"ffmpeg exit {r.returncode}: {' | '.join(tail)}")
    return r


def _overlays(layout_cfg):
    return (layout_cfg or {}).get("overlays") or []


def apply_finish(in_mp4, out_mp4, layout_cfg):
    """Burn ``layout_cfg['overlays']`` onto the rendered mp4.

    Each overlay is one of:

      ``{"kind": "watermark", "path": "/abs/watermark.png", "width_frac": 0.22,
         "pad": 24, "alpha": 0.80}``
      ``{"kind": "top_chip", "text": "BRAND", "fontsize": 60, "fontcolor": "yellow",
         "y": 120, "w": 720, "h": 100, "hold_t": 3.4, "fade_s": 0.3}``
      ``{"kind": "none"}``

    An unsupported overlay kind raises :class:`ValueError`. An empty list copies ``in_mp4`` to
    ``out_mp4`` and returns.
    """
    in_mp4, out_mp4 = Path(in_mp4), Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ovs = [o for o in _overlays(layout_cfg) if (o.get("kind") or "none") != "none"]
    if not ovs:
        shutil.copy2(in_mp4, out_mp4)
        return {"overlays": [], "copied": True}

    # One encode per clip. Build the filterchain in order; stack inputs for overlays that need
    # external files.
    inputs = ["-i", str(in_mp4)]
    filter_parts = []
    label = "0:v"
    for i, ov in enumerate(ovs):
        kind = (ov.get("kind") or "").lower()
        nxt = f"v{i+1}"
        if kind == "watermark":
            path = ov.get("path")
            if not path or not Path(path).exists():
                raise ValueError(f"watermark overlay missing path: {path!r}")
            idx = (len(inputs) // 2)  # input stream index of this overlay image
            inputs += ["-i", str(path)]
            width_frac = float(ov.get("width_frac", WATERMARK_WIDTH_FRAC))
            # 1080 is the canonical output width — the recompose stage always outputs 1080x1920.
            wm_w = max(16, int(round(1080 * width_frac)))
            pad = int(ov.get("pad", WATERMARK_PAD))
            alpha = float(ov.get("alpha", WATERMARK_ALPHA))
            filter_parts.append(
                f"[{idx}:v]scale={wm_w}:-1,format=rgba,"
                f"colorchannelmixer=aa={alpha:.3f}[wm{i}]"
            )
            filter_parts.append(f"[{label}][wm{i}]overlay=W-w-{pad}:H-h-{pad}[{nxt}]")
            label = nxt
        elif kind == "top_chip":
            font = _pick_font()
            if not font:
                raise RuntimeError("top_chip overlay needs a bold TTF (none found)")
            text = str(ov.get("text") or "").replace("'", r"\\'").replace(":", r"\\:")
            w = int(ov.get("w", CHIP_W))
            h = int(ov.get("h", CHIP_H))
            y = int(ov.get("y", CHIP_Y))
            size = int(ov.get("fontsize", 60))
            color = str(ov.get("fontcolor") or "yellow")
            band = float(ov.get("band_alpha", 0.55))
            hold_t = float(ov.get("hold_t", CHIP_HOLD_T))
            fade_s = float(ov.get("fade_s", CHIP_FADE_S))
            on = f"gt(t,{hold_t})"
            alpha_ramp = (f"if(gt(t,{hold_t + fade_s}),1,(t-{hold_t})/{fade_s})"
                          if fade_s > 0 else "1")
            # drawbox + drawtext chained on the current label
            filter_parts.append(
                f"[{label}]drawbox=x=(iw-{w})/2:y={y}:w={w}:h={h}:"
                f"color=black@{band:.2f}:t=fill:enable='{on}',"
                f"drawtext=fontfile={font}:text='{text}':fontcolor={color}:fontsize={size}:"
                f"bordercolor=black:borderw=2:x=(w-text_w)/2:"
                f"y={y}+({h}-text_h)/2:alpha='{alpha_ramp}':enable='{on}'[{nxt}]"
            )
            label = nxt
        else:
            raise ValueError(f"unsupported overlay kind: {kind!r}")
    # If we built any filter, map the final label; otherwise straight copy.
    if filter_parts:
        cmd = ["ffmpeg", "-y", "-nostdin", *inputs,
               "-filter_complex", ";".join(filter_parts),
               "-map", f"[{label}]", "-map", "0:a?",
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high",
               "-pix_fmt", "yuv420p", "-c:a", "copy", str(out_mp4)]
        _ffmpeg(cmd)
    else:
        shutil.copy2(in_mp4, out_mp4)
    return {"overlays": [o.get("kind") for o in ovs], "copied": False}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--layout-cfg", required=True,
                    help="path to a JSON file with the campaign's layout block "
                         "(a finish config object)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.layout_cfg).read_text())
    res = apply_finish(a.input, a.output, cfg)
    print(json.dumps(res))
