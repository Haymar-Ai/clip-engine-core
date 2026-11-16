"""clipengine/ytdl.py: which yt-dlp binary runs, the deno runtime on PATH, the private cookies
copy, and harvest's use of all three."""
from __future__ import annotations

import os
import subprocess
import time

from clipengine import engine
from clipengine import runstate
from clipengine import ytdl


# ---------------------------------------------------------------------------
# get_ytdlp_bin / ensure_deno_path
# ---------------------------------------------------------------------------

def test_get_ytdlp_bin_env_var_wins(monkeypatch):
    monkeypatch.setenv("YTDLP_BIN", "/custom/yt-dlp")
    assert ytdl.get_ytdlp_bin() == "/custom/yt-dlp"


def test_get_ytdlp_bin_last_resort_is_path_lookup(monkeypatch):
    """No env -> hands back to PATH via bare 'yt-dlp'."""
    monkeypatch.delenv("YTDLP_BIN", raising=False)
    assert ytdl.get_ytdlp_bin() == "yt-dlp"


def test_ensure_deno_path_prepends_and_is_idempotent(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    first = ytdl.ensure_deno_path()
    assert first.split(os.pathsep)[0] == os.path.expanduser("~/.deno/bin")
    second = ytdl.ensure_deno_path()
    assert first == second
    # exactly one `~/.deno/bin` entry
    assert second.count(os.path.expanduser("~/.deno/bin")) == 1


# ---------------------------------------------------------------------------
# the private cookies copy
# ---------------------------------------------------------------------------

def test_sweep_removes_only_stale_copies(tmp_path):
    old, fresh, other = tmp_path / "ytc-old.txt", tmp_path / "ytc-new.txt", tmp_path / "keep-old.txt"
    for f in (old, fresh, other):
        f.write_text("x")
    day_ago = time.time() - ytdl.STALE_AFTER_S - 60
    os.utime(old, (day_ago, day_ago))
    os.utime(other, (day_ago, day_ago))
    assert ytdl.sweep_stale(tmp_path) == 1
    assert not old.exists() and fresh.exists() and other.exists()


def test_sweep_tolerates_missing_dir(tmp_path):
    assert ytdl.sweep_stale(tmp_path / "nope") == 0


def test_scratch_copy_is_private_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(ytdl, "_COPIES", {})
    src = tmp_path / "cookies.txt"
    src.write_text("login")
    a = ytdl.scratch_copy(str(src))
    try:
        assert a != str(src) and open(a).read() == "login"
        assert os.stat(a).st_mode & 0o777 == 0o600
        assert ytdl.scratch_copy(str(src)) == a
    finally:
        os.unlink(a)
    assert ytdl.scratch_copy(str(tmp_path / "missing.txt")) is None


# ---------------------------------------------------------------------------
# harvest: the resolved binary, cookies for YouTube only
# ---------------------------------------------------------------------------

def _harvest(monkeypatch, tmp_path, url, cookies):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setenv("YTDLP_BIN", "/env/yt-dlp")
    monkeypatch.setattr(engine, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(engine, "COOKIES", cookies)
    monkeypatch.setattr(engine.subprocess, "run", run)
    monkeypatch.setattr(engine.subprocess, "check_output", lambda cmd, **kw: b"600.0\n")
    runstate.add_source("s1", url=url)
    assert engine.harvest(runstate.source("s1")) is True
    assert runstate.source("s1")["state"] == "harvested"
    return calls[0]


def test_harvest_runs_the_resolved_binary_with_a_cookies_copy_for_youtube(monkeypatch, tmp_path):
    ck = tmp_path / "cookies.txt"
    ck.write_text("login")
    monkeypatch.setattr(ytdl, "_COPIES", {})
    cmd = _harvest(monkeypatch, tmp_path, "https://www.youtube.com/watch?v=abcdefghijk", str(ck))
    assert cmd[0] == "/env/yt-dlp"
    assert cmd[1] == "--cookies" and cmd[2] != str(ck) and open(cmd[2]).read() == "login"


def test_harvest_never_passes_cookies_to_another_host(monkeypatch, tmp_path):
    ck = tmp_path / "cookies.txt"
    ck.write_text("login")
    cmd = _harvest(monkeypatch, tmp_path, "https://www.twitch.tv/videos/123", str(ck))
    assert cmd[0] == "/env/yt-dlp" and "--cookies" not in cmd


def test_harvest_without_cookies(monkeypatch, tmp_path):
    cmd = _harvest(monkeypatch, tmp_path, "https://www.youtube.com/watch?v=abcdefghijk", "")
    assert "--cookies" not in cmd and cmd[-1].startswith("https://")
