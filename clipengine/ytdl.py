"""yt-dlp helpers: which binary to run, the JS runtime on PATH, and a private cookies copy.

  get_ytdlp_bin()     YTDLP_BIN when set, else "yt-dlp" from PATH
  ensure_deno_path()  ~/.deno/bin on PATH: yt-dlp solves YouTube's JS challenge with deno
  scratch_copy(path)  a 0600 temp copy of a cookies file, made once per process

yt-dlp writes refreshed cookies back into the --cookies file on exit, so a bad response can
clobber a saved login. It is always handed a private per-process copy; the original stays intact.
"""
import atexit
import os
import shutil
import tempfile
import time
from pathlib import Path

_COPIES = {}
# atexit never runs when a process is killed (a timeout's SIGTERM, OOM), so copies can pile up in
# the temp dir. No run lasts a day, so anything older is orphaned.
STALE_AFTER_S = 24 * 3600


def get_ytdlp_bin():
    """The yt-dlp binary every call uses: the YTDLP_BIN env var, else "yt-dlp" from PATH."""
    return os.environ.get("YTDLP_BIN") or "yt-dlp"


def ensure_deno_path():
    """Prepend ~/.deno/bin to os.environ["PATH"] if it is not there yet. Idempotent."""
    deno_bin = os.path.expanduser("~/.deno/bin")
    cur = os.environ.get("PATH", "")
    if deno_bin not in cur.split(os.pathsep):
        os.environ["PATH"] = deno_bin + os.pathsep + cur
    return os.environ["PATH"]


def sweep_stale(tmpdir=None, now=None):
    """Delete ytc-*.txt copies untouched for STALE_AFTER_S. Returns how many were removed."""
    now = time.time() if now is None else now
    removed = 0
    for f in Path(tmpdir or tempfile.gettempdir()).glob("ytc-*.txt"):
        try:
            if now - f.stat().st_mtime > STALE_AFTER_S:
                f.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def scratch_copy(path):
    """Path of a 0600 temp copy of `path` (made once per process), or None if it doesn't exist."""
    if not path or not Path(path).exists():
        return None
    if path not in _COPIES:
        sweep_stale()
        fd, tmp = tempfile.mkstemp(prefix="ytc-", suffix=".txt")
        os.close(fd)
        shutil.copyfile(path, tmp)
        os.chmod(tmp, 0o600)
        atexit.register(lambda p=tmp: Path(p).unlink(missing_ok=True))
        _COPIES[path] = tmp
    return _COPIES[path]
