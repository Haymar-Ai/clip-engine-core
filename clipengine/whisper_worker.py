#!/usr/bin/env python3
"""A warm `whisper` CLI: one process that runs whisper's own command line once per
request, so the daily run pays the interpreter start, the torch import and the model load (which
re-reads and SHA-256s the checkpoint) once per window instead of once per call. A candidate clip
costs 5 small.en calls (precision pass + 4 edge-verify windows); on the 10-19 10:07 run those
plus the base.en pass were half of every niche's time.

Each request is `whisper <argv>` run through whisper.transcribe.cli() (same parser, defaults,
writer and output file); only whisper.load_model is cached (by its arguments). The output is the
CLI's byte for byte wherever the CLI is deterministic: beam search at temperature 0, i.e. every
decode unless whisper's temperature fallback fires (all 12 recorded-audio calls replayed for
 were temperature 0). The fallback samples from torch's RNG, which torch seeds at
random in every process, so the CLI never reproduced those decodes run to run; the worker draws
a fresh random seed per call (torch.seed()), so each call samples independently, as a new CLI
process does.

transcript._worker_env runs it with numba's workqueue thread pool: whisper's word alignment
(dtw_cpu, numba-parallel) on numba's default OpenMP layer loads a second libgomp next to
torch's, and every call after the first then cost ~40% more CPU for the same transcription.

Protocol: one JSON request per stdin line, {"id": n, "argv": [...]}; one JSON reply per line on
the original stdout, {"id": n, "rc": exit code}. The CLI's own stdout/stderr go to /dev/null, as
transcript._cli_words sends them. EOF on stdin ends the worker, and so does its parent's death
(PR_SET_PDEATHSIG; `--parent PID`): a worker never outlives the window that started it, even
mid-transcription.
"""
from __future__ import annotations

import ctypes
import json
import os
import signal
import sys

PR_SET_PDEATHSIG = 1


def die_with_parent(parent):
    """SIGKILL this process when `parent` exits (Linux; a no-op elsewhere). True when armed and
    the parent is still alive."""
    try:
        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)
    except (OSError, AttributeError):
        return False
    return os.getppid() == int(parent)


def install(whisper_mod, wt_mod, torch_mod):
    """Patch whisper's load_model (cached by its arguments) and transcribe (a fresh random torch
    seed per call, as a new CLI process has) in place; returns the model cache for tests."""
    models = {}
    real_load, real_transcribe = whisper_mod.load_model, wt_mod.transcribe

    def load_model(name, device=None, download_root=None, in_memory=False):
        key = (name, str(device), download_root, bool(in_memory))
        if key not in models:
            models[key] = real_load(name, device=device, download_root=download_root,
                                    in_memory=in_memory)
        return models[key]

    def transcribe(model, audio, **kw):
        torch_mod.seed()                        # independent fallback sampling per call
        return real_transcribe(model, audio, **kw)

    whisper_mod.load_model = load_model
    wt_mod.transcribe = transcribe
    return models


def serve(cli, reply, lines):
    """Run cli() once per request line with sys.argv = ["whisper", *argv]; reply(dict) each."""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            reply({"id": None, "rc": 2, "error": "bad request"})
            continue
        sys.argv = ["whisper", *[str(a) for a in req.get("argv") or []]]
        rc = 0
        try:
            cli()
        except SystemExit as e:                 # argparse errors exit 2, like the CLI
            rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except Exception:                        # noqa: BLE001 - the CLI would have died: rc 1
            rc = 1
        reply({"id": req.get("id"), "rc": rc})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--parent" in argv:
        parent = argv[argv.index("--parent") + 1]
        if not die_with_parent(parent) and os.getppid() != int(parent):
            return 0                            # the parent is already gone: nothing to serve
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    requests = os.fdopen(os.dup(0), "r")        # private: no child of the CLI can read requests
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)                         # the CLI's prints and tracebacks: /dev/null
    os.dup2(devnull, 2)
    import torch
    import whisper
    wt = sys.modules["whisper.transcribe"]      # `whisper.transcribe` is the function
    install(whisper, wt, torch)

    def reply(d):
        proto.write(json.dumps(d) + "\n")
        proto.flush()
    serve(wt.cli, reply, requests)
    return 0


if __name__ == "__main__":
    sys.exit(main())
