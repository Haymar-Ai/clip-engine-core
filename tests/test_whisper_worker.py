"""Tests for clipengine/whisper_worker.py and the warm-worker path of transcript._cli_words.
Offline: fake whisper/torch modules and a fake worker script; the real-model
equivalence check is tests/test_whisper_worker_equiv.py (-m slow)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import types
import uuid

import pytest

from clipengine import transcript
from clipengine import whisper_worker as ww


# ---- the worker side ----
def test_install_caches_models_and_reseeds_every_transcribe():
    calls = []
    torch = types.SimpleNamespace(seed=lambda: calls.append("seed"))
    whisper = types.SimpleNamespace(load_model=lambda name, **k: calls.append(("load", name, k))
                                    or object())
    wt = types.SimpleNamespace(transcribe=lambda m, a, **k: calls.append(("tr", a, k)) or {"a": a})
    models = ww.install(whisper, wt, torch)
    m1 = whisper.load_model("small.en", device="cpu", download_root=None)
    m2 = whisper.load_model("small.en", device="cpu", download_root=None)
    m3 = whisper.load_model("base.en", device="cpu", download_root=None)
    assert m1 is m2 and m1 is not m3 and len(models) == 2
    assert [c for c in calls if c[0] == "load"] == [
        ("load", "small.en", {"device": "cpu", "download_root": None, "in_memory": False}),
        ("load", "base.en", {"device": "cpu", "download_root": None, "in_memory": False})]
    calls.clear()
    assert wt.transcribe(m1, "a.wav", temperature=0) == {"a": "a.wav"}
    assert wt.transcribe(m1, "b.wav", temperature=0) == {"a": "b.wav"}
    assert calls == ["seed", ("tr", "a.wav", {"temperature": 0}),
                     "seed", ("tr", "b.wav", {"temperature": 0})]


def test_serve_runs_the_cli_once_per_request_and_replies_its_exit_code(monkeypatch):
    seen, replies = [], []

    def cli():
        seen.append(list(sys.argv))
        arg = sys.argv[1]
        if arg == "usage":
            raise SystemExit(2)
        if arg == "boom":
            raise RuntimeError("model file corrupt")
        if arg == "exit0":
            raise SystemExit(None)
    monkeypatch.setattr(sys, "argv", ["pytest"])
    lines = [json.dumps({"id": 1, "argv": ["a.wav", "--model", "small.en"]}), "",
             json.dumps({"id": 2, "argv": ["usage"]}), "not json",
             json.dumps({"id": 3, "argv": ["boom"]}), json.dumps({"id": 4, "argv": ["exit0"]})]
    ww.serve(cli, replies.append, iter(lines))
    assert seen[0] == ["whisper", "a.wav", "--model", "small.en"]
    assert replies == [{"id": 1, "rc": 0}, {"id": 2, "rc": 2},
                       {"id": None, "rc": 2, "error": "bad request"},
                       {"id": 3, "rc": 1}, {"id": 4, "rc": 0}]


def test_worker_exits_at_once_when_its_parent_is_already_gone():
    out = subprocess.run([sys.executable, ww.__file__, "--parent", "1"], capture_output=True,
                         timeout=60, stdin=subprocess.DEVNULL)
    assert out.returncode == 0 and out.stdout == b""        # pid 1 is never our parent


def test_a_worker_dies_with_its_parent_even_mid_call(tmp_path):
    """The window process is SIGKILLed at its cap: its worker must not keep transcribing."""
    marker = f"ww-{uuid.uuid4().hex}"
    parent = tmp_path / "parent.py"
    parent.write_text(textwrap.dedent(f"""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", '''
        import os, sys, time
        from clipengine import whisper_worker
        assert whisper_worker.die_with_parent({{ppid}})
        print("armed", flush=True)
        time.sleep(300)
        '''.replace("{{ppid}}", str(__import__("os").getpid())), {marker!r}], stdout=subprocess.PIPE)
        print(child.stdout.readline().decode().strip(), flush=True)
        time.sleep(300)
    """))
    p = subprocess.Popen([sys.executable, str(parent)], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "armed"
        p.send_signal(signal.SIGKILL)
        p.wait(timeout=30)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and subprocess.run(
                ["pgrep", "-f", marker], capture_output=True).stdout.strip():
            time.sleep(0.1)
        assert subprocess.run(["pgrep", "-f", marker], capture_output=True).stdout.strip() == b""
    finally:
        if p.poll() is None:
            p.kill()


# ---- the transcript side: _cli_words on a warm worker ----
FAKE_WORKER = textwrap.dedent("""
    import json, os, pathlib, sys, time
    for line in sys.stdin:
        req = json.loads(line)
        argv = req["argv"]
        audio, out = argv[0], argv[argv.index("--output_dir") + 1]
        name = pathlib.Path(audio).name
        if name.startswith("sleep"):
            time.sleep(60)
        if name.startswith("die"):
            os._exit(9)
        rc = 1 if name.startswith("fail") else 0
        if not rc:
            words = [{"word": " " + name, "start": 0.0, "end": 0.5}, {"word": " pid" + str(os.getpid()),
                      "start": 0.5, "end": 0.9}]
            (pathlib.Path(out) / (pathlib.Path(audio).stem + ".json")).write_text(
                json.dumps({"segments": [{"words": words}]}))
        print(json.dumps({"id": req["id"], "rc": rc}), flush=True)
""")


@pytest.fixture()
def fake_worker(tmp_path, monkeypatch):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    monkeypatch.setattr(transcript, "WORKER_SCRIPT", script)
    monkeypatch.setattr(transcript, "_whisper_python", lambda: sys.executable)
    monkeypatch.setattr(transcript, "WORKER", True)
    transcript.close_worker()
    yield
    transcript.close_worker()


def _pid(words):
    return int(words[1]["w"][3:])


def test_cli_words_reuse_one_warm_worker(fake_worker):
    a = transcript._cli_words("/x/a.wav", "small.en")
    b = transcript._cli_words("/x/b.wav", "base.en")
    assert [w["w"] for w in a][0] == "a.wav" and [w["w"] for w in b][0] == "b.wav"
    assert _pid(a) == _pid(b) == transcript._WORKER["p"].pid      # no new process per call
    p = transcript._WORKER["p"]
    transcript.close_worker()
    assert p.poll() is not None and transcript._WORKER["p"] is None
    transcript.close_worker()                                      # idempotent


def test_worker_failure_raises_like_the_cli_and_keeps_the_worker(fake_worker):
    a = transcript._cli_words("/x/a.wav", "small.en")
    with pytest.raises(subprocess.CalledProcessError) as e:
        transcript._cli_words("/x/fail.wav", "small.en")
    assert e.value.returncode == 1 and e.value.cmd[0] == "whisper"
    assert _pid(transcript._cli_words("/x/c.wav", "small.en")) == _pid(a)


def test_worker_timeout_kills_it_and_the_next_call_starts_a_fresh_one(fake_worker):
    a = transcript._cli_words("/x/a.wav", "small.en")
    old = transcript._WORKER["p"]
    t0 = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        transcript._cli_words("/x/sleep.wav", "small.en", timeout=1.0)
    assert time.monotonic() - t0 < 10 and old.poll() is not None    # killed, like the CLI was
    b = transcript._cli_words("/x/b.wav", "small.en")
    assert _pid(b) != _pid(a)


def test_a_dead_worker_fails_the_call_and_is_replaced(fake_worker):
    a = transcript._cli_words("/x/a.wav", "small.en")
    with pytest.raises(subprocess.CalledProcessError):
        transcript._cli_words("/x/die.wav", "small.en")
    assert _pid(transcript._cli_words("/x/b.wav", "small.en")) != _pid(a)


def test_worker_off_keeps_one_cli_process_per_call(monkeypatch, tmp_path):
    monkeypatch.setattr(transcript, "WORKER", None)
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        out = cmd[cmd.index("--output_dir") + 1]
        (tmp_path / "x").mkdir(exist_ok=True)
        with open(os.path.join(out, "a.json"), "w") as f:
            json.dump({"segments": [{"words": [{"word": " hi", "start": 0, "end": 1}]}]}, f)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(transcript.subprocess, "run", fake_run)
    assert transcript._cli_words("/x/a.wav", "small.en") == [{"w": "hi", "start": 0.0, "end": 1.0}]
    assert seen[0][:2] == ["whisper", "/x/a.wav"] and transcript._WORKER["p"] is None


def test_worker_env_keeps_numba_off_a_second_openmp_runtime():
    """whisper's numba-parallel dtw_cpu on numba's default OpenMP layer loads a second libgomp
    and resets torch to 16 threads: every later call in a warm process cost ~40% more CPU. The
    one-shot CLI path keeps its env (it only ever makes one call)."""
    env = transcript._worker_env()
    assert env["NUMBA_THREADING_LAYER"] == "workqueue"
    assert env["NUMBA_NUM_THREADS"] == env["OMP_NUM_THREADS"] == str(transcript.THREADS)
    assert env["OMP_WAIT_POLICY"] == "PASSIVE"
    assert transcript._cli_env().get("NUMBA_THREADING_LAYER") == os.environ.get("NUMBA_THREADING_LAYER")


def test_the_worker_process_gets_the_worker_env(fake_worker, tmp_path, monkeypatch):
    seen = {}
    real = subprocess.Popen

    def spy(cmd, **kw):
        seen.update(kw.get("env") or {})
        return real(cmd, **kw)
    monkeypatch.setattr(transcript.subprocess, "Popen", spy)
    transcript._cli_words("/x/a.wav", "small.en")
    assert seen["NUMBA_THREADING_LAYER"] == "workqueue"


def test_worker_runs_on_the_interpreter_of_the_whisper_cli(tmp_path, monkeypatch):
    def script(first_line):
        f = tmp_path / f"whisper{len(list(tmp_path.iterdir()))}"
        f.write_text(first_line + "\nprint()\n")
        return str(f)
    py = script("#!" + sys.executable)
    monkeypatch.setattr(transcript.shutil, "which", lambda n: py if n == "whisper" else sys.executable)
    assert transcript._whisper_python() == sys.executable
    env = script("#!/usr/bin/env python3")
    monkeypatch.setattr(transcript.shutil, "which",
                        lambda n: env if n == "whisper" else "/opt/py/bin/python3")
    assert transcript._whisper_python() == "/opt/py/bin/python3"
    sh = script("#!/bin/sh")
    monkeypatch.setattr(transcript.shutil, "which", lambda n: sh if n == "whisper" else None)
    assert transcript._whisper_python() == sys.executable
    monkeypatch.setattr(transcript.shutil, "which", lambda n: None)
    assert transcript._whisper_python() == sys.executable
