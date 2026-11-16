"""Contact-sheet sampling: a 1 fps strip per clip written to a review folder, so a person can
spot-check clips without being a gate.

Each sheet is a horizontal filmstrip (1 fps row of frames) saved as JPEG; filename encodes the
clip identity. Independent of the auto-quality gate — this is a human-eyes sanity check that
runs REGARDLESS of the gate verdict (both passes and drops get a sheet, labeled). The sheets
pile up in `review/<date>/<account>/`; `batch_cleanup.py` doesn't touch them.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def build_contact_sheet(media_path: str, out_path: Path, *, fps: float = 1.0,
                        tile_rows: int = 1, max_cols: int = 20,
                        width_per_tile: int = 240, duration_s: Optional[float] = None,
                        runner: Optional[Callable[..., int]] = None) -> bool:
    """Render a 1 fps filmstrip of `media_path` to `out_path` (JPEG). Returns True on success.
    ffmpeg call: `-vf fps=1,scale=W:-2,tile=COLSxROWS` — simplest filter chain that works.
    `tile` pads a short clip's missing cells with black, so the column count is capped at the
    frames the clip has when its duration is known (`duration_s`, else ffprobe).
    `runner` lets tests pass a stub (default: subprocess.call)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur = duration_s if duration_s is not None else probe_duration(media_path)
    if dur:
        max_cols = max(1, min(max_cols, int(dur * fps) // max(1, tile_rows)))
    tile = f"{max_cols}x{tile_rows}"
    vf = f"fps={fps},scale={width_per_tile}:-2,tile={tile}"
    cmd = ["ffmpeg", "-y", "-i", str(media_path), "-frames:v", "1", "-vf", vf,
           "-q:v", "4", str(out_path)]
    rc = int((runner or _default_runner)(cmd))
    return rc == 0 and out_path.exists() and out_path.stat().st_size > 0


def _default_runner(cmd):
    return subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def probe_duration(media_path) -> Optional[float]:
    """Container duration in seconds, or None when ffprobe can't read the file."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", str(media_path)],
                           capture_output=True, text=True, timeout=30)
        d = float((r.stdout or "").strip() or 0)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return d if d > 0 else None


GRID_TAIL_S = 0.5    # never seek closer than this to the end: a seek past the last frame
                     # yields no frame and `tile` fills the cell with black


def sheet_times(duration_s: float, n: int = 6) -> list[float]:
    """`n` evenly spaced sample times: the centre of each of `n` equal slices of the clip,
    clamped so no seek lands past the last decodable frame."""
    hi = max(0.0, float(duration_s) - GRID_TAIL_S)
    return [round(min(hi, float(duration_s) * (i + 0.5) / n), 3) for i in range(n)]


def grid_sheet_cmd(media_path, out_path, times, *, cols: int = 3, width: int = 320) -> list:
    """ffmpeg command for an exactly-len(times) tile grid: one input-seeked frame per time,
    each scaled to `width`, concatenated and tiled COLSxROWS.

    The go-live sheets used `select='eq(n,0)+...+gt(scene,0.4)',thumbnail=100,...,tile=3x2`:
    `thumbnail=100` emits ONE frame per 100 selected frames, so `tile` got a single frame and
    padded the other five cells black. Seeking each frame explicitly cannot under-fill."""
    n = len(times)
    rows = (n + cols - 1) // cols
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin"]
    for t in times:
        cmd += ["-ss", f"{t:.3f}", "-i", str(media_path)]
    chains = [f"[{i}:v]trim=end_frame=1,setpts=PTS-STARTPTS,scale={width}:-2,setsar=1[f{i}]"
              for i in range(n)]
    graph = ";".join(chains) + ";" + "".join(f"[f{i}]" for i in range(n)) + \
        f"concat=n={n}:v=1:a=0,tile={cols}x{rows}[sheet]"
    return cmd + ["-filter_complex", graph, "-map", "[sheet]", "-frames:v", "1",
                  "-q:v", "3", str(out_path)]


def build_grid_sheet(media_path, out_path, *, n: int = 6, cols: int = 3, width: int = 320,
                     duration_s: Optional[float] = None,
                     runner: Optional[Callable[..., int]] = None) -> bool:
    """Review contact sheet: `n` frames evenly spaced over the clip in a COLS-wide grid
    (default 3x2, 320 px tiles). Returns True when the JPEG was written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur = duration_s if duration_s is not None else probe_duration(media_path)
    if not dur:
        return False
    cmd = grid_sheet_cmd(media_path, out_path, sheet_times(dur, n), cols=cols, width=width)
    rc = int((runner or _default_runner)(cmd))
    return rc == 0 and out_path.exists() and out_path.stat().st_size > 0


def review_dir(root: Path, date_iso: str, account: str) -> Path:
    """`<root>/<date>/<account_slug>/`. `account_slug` strips `@` + punctuation so a Linux fs
    is happy. The slug is deterministic so a reviewer finds yesterday's sheets by eye."""
    slug = (account or "").replace("@", "").replace(".", "_").replace("/", "_")
    p = Path(root) / date_iso / (slug or "misc")
    p.mkdir(parents=True, exist_ok=True)
    return p


def sheet_name(clip: dict, verdict: str = "") -> str:
    """Encode clip id + verdict in the filename so a glance tells you what the gate said."""
    cid = clip.get("clip_id") or 0
    sid = clip.get("source_id") or "src"
    tag = ("_" + verdict) if verdict else ""
    return f"c{cid}_{sid}{tag}.jpg"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="3x2 review contact sheet for a rendered clip")
    ap.add_argument("media")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--width", type=int, default=320)
    args = ap.parse_args()
    ok = build_grid_sheet(args.media, args.out, n=args.n, cols=args.cols, width=args.width,
                          runner=lambda cmd: subprocess.call(cmd))
    raise SystemExit(0 if ok else 1)
