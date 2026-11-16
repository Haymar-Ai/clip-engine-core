#!/usr/bin/env python3
"""Generate opus-clips style ASS subtitles from whisper word timestamps.

Thin CLI over captions.build_ass (the single caption builder).

Usage: build_ass.py WHISPER.json OUT.ass [STYLE]
STYLE: 'opus' (default), 'karaoke', 'minimal'
"""
import json
import sys
from pathlib import Path

from clipengine.render import captions


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        sys.exit(__doc__)
    words = captions.words_from_whisper(json.loads(Path(argv[0]).read_text()))
    out = captions.build_ass(words, argv[1], style=argv[2] if len(argv) > 2 else "opus")
    print(f"wrote {out}: {len(words)} words", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
