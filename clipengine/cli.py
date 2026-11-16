"""Command line: `python -m clipengine run <url-or-file> --niche <niche> --out <dir>`.

Exit codes: 0 when at least one clip passed the quality gate, 1 when the run finished with no
passing clip (or harvest / select failed), 2 for a usage or config error.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

LAYOUTS = ("auto", "general", "track", "split", "screencast", "broadcast", "hudband",
           "reaction", "gamestack")
VERTICALS = ("podcast", "stream", "poker", "general")
NICHE_RE = re.compile(r"[a-z0-9_-]{1,40}")


def _packs():
    d = Path(__file__).resolve().parent / "prompt_packs"
    return sorted(p.stem for p in d.glob("*.json"))


def _check_target(target):
    """An existing file (returned resolved) or an http(s) URL with a host; else ValueError."""
    p = Path(target).expanduser()
    if p.exists():
        if not p.is_file():
            raise ValueError(f"{target}: not a file")
        return str(p.resolve())
    u = urlparse(str(target))
    if u.scheme in ("http", "https") and u.netloc and not str(target).startswith("-"):
        return str(target)
    raise ValueError(f"{target}: neither an existing file nor an http(s) URL")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m clipengine",
        description="Cut short vertical clips from a long video: transcript, picked moments, "
                    "clean cuts, 9:16 reframe, captions and a hook title, quality gate.")
    sub = ap.add_subparsers(dest="cmd", metavar="command")
    r = sub.add_parser("run", help="make clips from one video (URL or local file)",
                       description="Make clips from one source video. Writes "
                                   "<out>/<source_id>/clip_NN.mp4 for every clip that passes the "
                                   "quality gate, clip_NN.json for every selected clip and run.json.")
    r.add_argument("target", help="local video file, or an http(s) URL yt-dlp can download")
    r.add_argument("--niche", help="prompt pack and render rules for the source: "
                                   + ", ".join(_packs()) + " (default: none, the vertical's prompts)")
    r.add_argument("--out", default="out", help="output folder (default: ./out)")
    r.add_argument("--title", help="the source's title (whisper's prompt for the caption passes: "
                                   "names and punctuation come out better)")
    r.add_argument("--layout", default="auto", choices=LAYOUTS,
                   help="render layout (default: auto, from the vertical and the frames)")
    r.add_argument("--vertical", choices=VERTICALS,
                   help="override the niche's vertical (podcasts -> podcast, gaming / streamers "
                        "-> stream, poker -> poker, else general)")
    r.add_argument("--cookies", help="Netscape cookies file for YouTube downloads (optional; "
                                     "the file is copied, never written)")
    r.add_argument("--config", help="JSON config merged over config.example.json")
    r.add_argument("--scratch", help="cache / working folder (default: scratch_dir from the "
                                     "config, ./.cache)")
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.cmd != "run":
        ap.print_help()
        return 0 if a.cmd is None else 2
    try:
        target = _check_target(a.target)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.niche is not None and not NICHE_RE.fullmatch(a.niche):
        print(f"error: --niche {a.niche!r}: lowercase letters, digits, '_' or '-' only",
              file=sys.stderr)
        return 2
    if a.niche and a.niche not in _packs():
        print(f"warning: no prompt pack for niche {a.niche!r}; the vertical's prompts are used",
              file=sys.stderr)
    if a.cookies and not Path(a.cookies).expanduser().is_file():
        print(f"error: --cookies {a.cookies}: no such file", file=sys.stderr)
        return 2
    from clipengine import config
    try:
        config.init(a.config, a.scratch)
    except config.ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    from clipengine import engine        # after config.init: the engine reads it on import
    rec = engine.run(target, niche=a.niche, out_dir=a.out, title=a.title, layout=a.layout,
                     vertical=a.vertical,
                     cookies=str(Path(a.cookies).expanduser().resolve()) if a.cookies else None)
    summary = {k: rec.get(k) for k in ("source_id", "niche", "vertical", "made", "selected",
                                       "out_dir", "seconds")}
    print(json.dumps(summary))
    return 0 if rec.get("made") else 1


if __name__ == "__main__":
    sys.exit(main())
