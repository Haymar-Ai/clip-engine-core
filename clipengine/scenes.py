#!/usr/bin/env python3
"""Shot-change times from ffmpeg's scene score, so a cut can land on a visual edit. Only ever run
on small windows around picks (never the whole file)."""
import re, subprocess

_PTS = re.compile(r"pts_time:\s*([0-9.]+)")
_CACHE = {}                    # (media, lo, hi, threshold) -> [times]; failures are not cached


def scene_cuts(media, lo, hi, threshold=0.3):
    """Absolute times (s) of scene changes in [lo, hi]. [] if ffmpeg fails (retried next call)."""
    key = (str(media), float(lo), float(hi), float(threshold))
    if key in _CACHE:
        return list(_CACHE[key])
    s = max(0.0, float(lo))
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{s:.3f}", "-to", f"{float(hi):.3f}",
           "-i", str(media), "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo",
           "-an", "-f", "null", "-"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        return []
    if p.returncode != 0:
        return []
    # with input seeking, pts restart at 0: showinfo times are relative to the seek point
    times = sorted({round(s + float(t), 3) for t in _PTS.findall(p.stderr or "")})
    _CACHE[key] = [t for t in times if s <= t <= float(hi)]
    return list(_CACHE[key])
