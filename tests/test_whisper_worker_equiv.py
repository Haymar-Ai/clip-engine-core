"""The warm whisper worker (clipengine/whisper_worker.py) writes the same JSON as
one `whisper` CLI process per call. Real whisper + tiny.en, so opt-in: `pytest -m slow`.

The CLI's decode is deterministic at temperature 0 (beam search), the path every call takes
unless whisper's temperature fallback fires; there the worker's JSON (text, tokens, logprobs,
word times and probabilities) must equal the CLI's byte for byte, call after call in one
process. The fallback samples from torch's RNG, which torch seeds at random per process, so the
CLI itself never repeats those decodes; test_torch_seeds_each_process_at_random pins that fact
(the worker reseeds per call, tests/test_whisper_worker.py)."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from clipengine import transcript

pytestmark = pytest.mark.slow

MODEL = "tiny.en"


def _need():
    if not shutil.which("whisper") or not shutil.which("ffmpeg"):
        pytest.skip("whisper CLI / ffmpeg not installed")
    if not (Path.home() / ".cache" / "whisper" / f"{MODEL}.pt").exists():
        pytest.skip(f"{MODEL} not cached")


def _wavs(tmp_path):
    out = []
    for name, src in (("speech", "flite=text='The quick brown fox jumps over the lazy dog. "
                                 "Then it ran away into the forest and was never seen again.'"),
                      ("noise", "anoisesrc=d=6:c=pink:a=0.3")):
        wav = tmp_path / f"{name}.wav"
        r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                            "-i", src, "-ar", "16000", "-ac", "1", str(wav)], capture_output=True)
        if r.returncode == 0:
            out.append(str(wav))
    if not out:
        pytest.skip("ffmpeg can't synthesize audio")
    return out


def _argv(wav, out):
    return [wav, "--model", MODEL, "--language", "en", "--word_timestamps", "True",
            "--output_format", "json", "--output_dir", out, "--threads", "2"]


def _cli(wav):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["whisper", *_argv(wav, tmp)], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, env=transcript._cli_env())
        return json.loads(next(Path(tmp).glob("*.json")).read_text())


def _worker(wav):
    with tempfile.TemporaryDirectory() as tmp:
        transcript._worker_run(_argv(wav, tmp))
        return json.loads(next(Path(tmp).glob("*.json")).read_text())


def test_worker_json_equals_one_cli_process_per_call(tmp_path):
    _need()
    ref = {w: _cli(w) for w in _wavs(tmp_path)}
    # only inputs the CLI decodes deterministically (no temperature fallback) have one answer
    wavs = [w for w, d in ref.items() if {s["temperature"] for s in d["segments"]} <= {0.0}]
    if not wavs:
        pytest.skip("every input hit the temperature fallback")
    try:
        for w in wavs + wavs[::-1] + wavs:          # one warm process, repeated + interleaved
            assert _worker(w) == ref[w], w
    finally:
        transcript.close_worker()


def test_torch_seeds_each_process_at_random():
    """On the interpreter the whisper CLI and the warm worker run on (its torch)."""
    _need()
    seeds = {subprocess.run([transcript._whisper_python(), "-c",
                             "import torch; print(torch.initial_seed())"],
                            capture_output=True, text=True, check=True).stdout.strip()
             for _ in range(2)}
    assert len(seeds) == 2
