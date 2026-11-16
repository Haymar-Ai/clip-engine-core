#!/usr/bin/env python3
"""Word-level transcripts. Two-stage: a cheap full pass (base.en, cached) for selection, then a
precise pass (small.en) only on the picked windows. faster-whisper is imported lazily; without it
the whisper CLI is used, so this module imports fine under a bare /usr/bin/python3.

Shapes (absolute source seconds):
  Word     = {"w": str, "start": float, "end": float}
  Sentence = {"i": int, "start": float, "end": float, "text": str, "w0": int, "w1": int}
"""
import contextlib, fcntl, os, json, hashlib, selectors, shutil, subprocess, sys, tempfile, time
import wave
from pathlib import Path
import numpy as np

from clipengine import config

_HERE = Path(__file__).resolve().parent
try:
    _CFG = config.load()
except (OSError, ValueError):
    _CFG = {}
SCRATCH = Path(_CFG.get("scratch_dir", str(_HERE / "scratch")))
# CTranslate2 thrashes when threads exceed free cores: 16 threads on a loaded 16-core box ran
# base.en at <1x realtime vs ~12x with 4 (measured 2026-09-24). Keep it small and fixed.
THREADS = int(_CFG.get("transcribe_threads", min(4, os.cpu_count() or 4)))
CLI_TIMEOUT_BASE_S = 120.0        # whisper CLI cap = base + per_audio_s x audio seconds ...
CLI_TIMEOUT_PER_AUDIO_S = 6.0
CLI_DEADLINE = None               # ... and never past this time.monotonic() (a caller's per-window deadline)
WORKER = None                     # True: CLI calls go to one warm whisper_worker.py (a run;
                                  # same JSON as the CLI, no per-call interpreter + model load)
WORKER_SCRIPT = _HERE / "whisper_worker.py"
_WORKER = {"p": None, "n": 0, "buf": b""}
HEAVY_MODELS = ("large", "turbo")  # a whisper model named with one of these is a heavy job ...
HEAVY_SLOTS = 3                   # ... and at most this many run at once host-wide (env
                                  # HEAVY_WHISPER_SLOTS, else WINDOW_CONCURRENCY: the window pool)
HEAVY_LOCK_DIR = "/tmp/clip-engine-heavy-whisper"   # the slots' flock files (env override)
HEAVY_POLL_S = 0.5
CLOSERS = "\"'”’)]}»"
OPENERS = "\"'“‘([{«"
ABBREV = {"mr", "mrs", "ms", "dr", "vs"}      # "Dr." is not a sentence end
_MODELS = {}                                  # model name -> loaded WhisperModel
# whisper hallucination guards (segment_ok), on the per-segment fields
# whisper's JSON and faster-whisper's segments already carry.
SEG_NO_SPEECH_MAX = 0.6           # dropped when no_speech_prob > this ...
SEG_NO_SPEECH_LOGPROB_MIN = -0.8  # ... and avg_logprob < this (whisper heard no speech, unsure)
SEG_COMPRESSION_MAX = 2.6         # dropped when compression_ratio > this (repetitive text)
SEG_LOOP_LOGPROB_MAX = -0.5       # a loop is dropped only when avg_logprob < this ...
SEG_LOOP_NO_SPEECH_MIN = 0.3      # ... or no_speech_prob > this: real chants ("whoa" x10) are heard confidently
SEG_LOOP_NGRAM = 3                # dropped when a word 3-gram occurs this many times ...
SEG_LOOP_REPEATS = 3
SEG_LOOP_MIN_TOKENS = 8           # ... or a segment of 8+ tokens has under this unique share
SEG_LOOP_UNIQUE_MIN = 0.3


def _word(text, start, end):
    t = str(text).strip()
    return {"w": t, "start": round(float(start), 3), "end": round(float(end), 3)} if t else None


def _seg_get(seg, name):
    """`name` of a whisper JSON segment (dict) or a faster-whisper Segment; None when absent."""
    return seg.get(name) if isinstance(seg, dict) else getattr(seg, name, None)


def _seg_num(seg, name):
    try:
        v = _seg_get(seg, name)
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _loops(tokens):
    """True when the tokens loop: a word 3-gram occurs SEG_LOOP_REPEATS+ times (non-overlapping,
    so a chanted "go go go go go" is one (go, go, go), not three), or 8+ tokens are under 30 %
    unique."""
    n = len(tokens)
    if n >= SEG_LOOP_MIN_TOKENS and len(set(tokens)) / n < SEG_LOOP_UNIQUE_MIN:
        return True
    seen = {}                                     # 3-gram -> (count, index its last one ends)
    for i in range(n - SEG_LOOP_NGRAM + 1):
        g = tuple(tokens[i:i + SEG_LOOP_NGRAM])
        k, end = seen.get(g, (0, 0))
        if i >= end:
            seen[g] = (k + 1, i + SEG_LOOP_NGRAM)
            if k + 1 >= SEG_LOOP_REPEATS:
                return True
    return False


def segment_ok(seg):
    """False for a whisper segment that is likely a hallucination: no_speech_prob > 0.6 with
    avg_logprob < -0.8, compression_ratio > 2.6, or looping text (_loops) that whisper was unsure of. `seg` is a whisper
    JSON segment dict or a faster-whisper Segment; a missing field never drops it."""
    nsp, lp = _seg_num(seg, "no_speech_prob"), _seg_num(seg, "avg_logprob")
    if nsp is not None and lp is not None and nsp > SEG_NO_SPEECH_MAX \
            and lp < SEG_NO_SPEECH_LOGPROB_MIN:
        return False
    cr = _seg_num(seg, "compression_ratio")
    if cr is not None and cr > SEG_COMPRESSION_MAX:
        return False
    unsure = (lp is not None and lp < SEG_LOOP_LOGPROB_MAX) or \
        (nsp is not None and nsp > SEG_LOOP_NO_SPEECH_MIN)
    if not unsure:
        return True
    toks = [t for t in (_tok_core(x) for x in str(_seg_get(seg, "text") or "").split()) if t]
    return not _loops(toks)


def _model(name):
    from faster_whisper import WhisperModel
    if name not in _MODELS:
        _MODELS[name] = WhisperModel(name, device="cpu", compute_type="int8",
                                     cpu_threads=THREADS)
    return _MODELS[name]


def _fw_words(path, model):
    segs, _ = _model(model).transcribe(str(path), language="en", word_timestamps=True,
                                       vad_filter=True, condition_on_previous_text=False)
    return [w for s in segs if segment_ok(s)
            for w in (_word(x.word, x.start, x.end) for x in (s.words or [])) if w]


def _media_seconds(path):
    """Length of an audio/video file in seconds, None if unknown. A .wav (words_window) is read
    from its header; anything else goes through ffprobe."""
    try:
        if str(path).lower().endswith(".wav"):
            with wave.open(str(path), "rb") as w:
                return w.getnframes() / float(w.getframerate())
        p = subprocess.Popen(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                              "default=noprint_wrappers=1:nokey=1", str(path)],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, _ = p.communicate(timeout=30)
        return float(out.decode().strip())
    except Exception:
        return None


def cli_timeout(audio_s):
    """Wall-clock cap for one whisper CLI call on `audio_s` seconds of audio, also bounded by
    CLI_DEADLINE (None = no cap: unknown length and no deadline). Without a timeout
    and one 37 s small.en precision window ran 2092 s on 10-17 (replayed: >1435 s, 93 CPU-min)."""
    caps = []
    if audio_s is not None:
        caps.append(CLI_TIMEOUT_BASE_S + CLI_TIMEOUT_PER_AUDIO_S * max(0.0, float(audio_s)))
    if CLI_DEADLINE is not None:
        caps.append(max(1.0, CLI_DEADLINE - time.monotonic()))
    return min(caps) if caps else None


def _cli_env():
    """The CLI's torch runs on OpenMP/MKL: by default 8+8 threads that spin-wait. At nice 19 on a
    loaded box that burned 667 CPU-s for a 35 s base.en pass (88 s wall); capped + passive it was
    82 CPU-s (74 s wall), same words and timestamps (2026-10-17)."""
    return {**os.environ, "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS),
            "OMP_WAIT_POLICY": "PASSIVE"}


def _worker_env():
    """_cli_env plus numba's own thread pool for whisper's word alignment (dtw_cpu is
    numba-parallel). Its default OpenMP layer loads a second libgomp next to torch's and resets
    torch to 16 threads; in a long-lived process every later call then paid ~40% more CPU
    (2.5x the sys time) for the same transcription (measured 2026-10-19).
    dtw_cpu's parallel part only fills its cost/trace arrays with constants, so its output is
    the same whichever pool runs it."""
    return {**_cli_env(), "NUMBA_THREADING_LAYER": "workqueue", "NUMBA_NUM_THREADS": str(THREADS)}


def _whisper_python():
    """The interpreter the `whisper` CLI script runs on (its shebang), so the warm worker imports
    the same whisper and torch; sys.executable when the script can't be read."""
    exe = shutil.which("whisper")
    try:
        with open(exe, "rb") as f:
            first = f.readline().decode("utf-8", "replace").strip()
    except (OSError, TypeError):
        first = ""
    parts = first[2:].split() if first.startswith("#!") else []
    if parts and os.path.basename(parts[0]) == "env":
        parts = parts[1:]
    if parts and "python" in os.path.basename(parts[0]):
        return shutil.which(parts[0]) or parts[0]
    return sys.executable


def heavy_model(model):
    m = str(model or "").lower()
    return any(k in m for k in HEAVY_MODELS)


def heavy_slots():
    """How many heavy whisper jobs may run at once on the host (>= 1)."""
    for name in ("HEAVY_WHISPER_SLOTS", "WINDOW_CONCURRENCY"):
        try:
            if os.environ.get(name):
                return max(1, int(os.environ[name]))
        except ValueError:
            pass
    return HEAVY_SLOTS


@contextlib.contextmanager
def heavy_slot(model, timeout=None):
    """Throughput 10-22: windows run in parallel, and a large-v3-turbo job (the strong captions,
    the repair lane's recaption) takes ~3 s per audio second of 4 threads; hold one of
    heavy_slots() host-wide flock slots while one runs, so there are never more of them than the
    window pool has windows. Yields the call's timeout less the wait (None: no timeout); raises
    subprocess.TimeoutExpired when no slot frees in time (callers treat it as a failed pass).
    Any other model passes straight through."""
    if not heavy_model(model):
        yield timeout
        return
    d = Path(os.environ.get("HEAVY_WHISPER_LOCK_DIR") or HEAVY_LOCK_DIR)
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    while True:
        for i in range(heavy_slots()):
            f = open(d / f"slot.{i}", "a+")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            try:
                yield None if timeout is None else max(1.0, timeout - (time.monotonic() - t0))
            finally:
                try:
                    fcntl.flock(f, fcntl.LOCK_UN)
                finally:
                    f.close()
            return
        if timeout is not None and time.monotonic() - t0 >= timeout:
            raise subprocess.TimeoutExpired(["whisper", "--model", str(model)], timeout)
        time.sleep(HEAVY_POLL_S)


def _argv_model(argv):
    argv = list(argv)
    return argv[argv.index("--model") + 1] if "--model" in argv[:-1] else None


def close_worker():
    """Stop the warm worker (EOF, then SIGKILL after 5 s). Safe to call when none runs."""
    p, _WORKER["p"], _WORKER["buf"] = _WORKER.get("p"), None, b""
    if p is None:
        return
    try:
        p.stdin.close()
    except OSError:
        pass
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def _worker_kill():
    p, _WORKER["p"], _WORKER["buf"] = _WORKER.get("p"), None, b""
    if p is not None and p.poll() is None:
        p.kill()
        p.wait()


def _worker_line(p, deadline):
    """The worker's next reply line; None when `deadline` (monotonic) passes first, b"" on EOF."""
    fd = p.stdout.fileno()
    with selectors.DefaultSelector() as sel:
        sel.register(fd, selectors.EVENT_READ)
        while b"\n" not in _WORKER["buf"]:
            wait = None if deadline is None else deadline - time.monotonic()
            if wait is not None and wait <= 0:
                return None
            if not sel.select(timeout=wait):
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                return b""
            _WORKER["buf"] += chunk
    line, _WORKER["buf"] = _WORKER["buf"].split(b"\n", 1)
    return line


def _worker_run(argv, timeout=None):
    """`whisper <argv>` in the warm worker (started on first use, restarted after it died or
    timed out). Raises as subprocess.run(check=True, timeout=timeout) would: TimeoutExpired (the
    worker is killed, like the CLI process was) or CalledProcessError. A heavy model runs in a
    heavy_slot (its wait counts against `timeout`)."""
    with heavy_slot(_argv_model(argv), timeout) as left:
        return _worker_call(argv, left)


def _worker_call(argv, timeout=None):
    cmd = ["whisper", *argv]
    p = _WORKER.get("p")
    if p is None or p.poll() is not None:
        _WORKER["buf"] = b""
        p = _WORKER["p"] = subprocess.Popen(
            [_whisper_python(), "-u", str(WORKER_SCRIPT), "--parent", str(os.getpid())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=_worker_env())
    _WORKER["n"] += 1
    rid = _WORKER["n"]
    deadline = None if timeout is None else time.monotonic() + timeout
    try:
        p.stdin.write((json.dumps({"id": rid, "argv": argv}) + "\n").encode())
        p.stdin.flush()
        line = _worker_line(p, deadline)
    except OSError:
        line = b""
    if line is None:
        _worker_kill()
        raise subprocess.TimeoutExpired(cmd, timeout)
    try:
        reply = json.loads(line) if line else {}
    except ValueError:
        reply = {}
    if reply.get("id") != rid:                  # died, or out of step: never trust it again
        rc = p.poll()
        _worker_kill()
        raise subprocess.CalledProcessError(rc if rc is not None else -1, cmd)
    if reply.get("rc"):
        raise subprocess.CalledProcessError(int(reply["rc"]), cmd)


def _cli_words(path, model, timeout=None):
    # --threads + _cli_env: see THREADS / _cli_env. The timeout raises subprocess.TimeoutExpired;
    # callers already treat a transcription failure as "no words" / "keep the base cuts".
    # WORKER: the same command line in the warm worker (same output, no per-call model load).
    with tempfile.TemporaryDirectory() as tmp:
        argv = [str(path), "--model", model, "--language", "en", "--word_timestamps", "True",
                "--output_format", "json", "--output_dir", tmp, "--threads", str(THREADS)]
        if WORKER:
            _worker_run(argv, timeout=timeout)
        else:
            with heavy_slot(model, timeout) as left:
                subprocess.run(["whisper", *argv], check=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=left, env=_cli_env())
        data = json.loads(next(Path(tmp).glob("*.json")).read_text())
    return [w for s in data.get("segments", []) if segment_ok(s) for w in
            (_word(x.get("word", ""), x["start"], x["end"]) for x in s.get("words", [])) if w]


def _transcribe(path, model):
    """Words of an audio/video file, times relative to the file. CLI fallback if no faster-whisper
    (that call is capped by cli_timeout of the file's probed length)."""
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return _cli_words(path, model, timeout=cli_timeout(_media_seconds(path)))
    return _fw_words(path, model)


def get_words(media, model="base.en", cache_dir=None):
    """Full-media word timestamps, cached as JSON keyed by md5(media|mtime|model)."""
    media = str(media)
    d = Path(cache_dir) if cache_dir else SCRATCH / "transcripts"
    key = hashlib.md5(f"{media}|{os.path.getmtime(media)}|{model}".encode()).hexdigest()
    f = d / f"{key}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except ValueError:
            pass                                  # torn/corrupt cache: re-transcribe
    words = _transcribe(media, model)
    d.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(words))
    tmp.replace(f)
    return words


def words_window(media, lo, hi, model="small.en"):
    """Precise word timestamps for [lo, hi] of `media`, in absolute source seconds."""
    lo = max(0.0, float(lo))
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "win.wav"
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{lo:.3f}",
                        "-t", f"{max(0.0, float(hi) - lo):.3f}", "-i", str(media), "-vn", "-ac", "1",
                        "-ar", "16000", str(wav)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        words = _transcribe(wav, model)
    return [{"w": w["w"], "start": round(w["start"] + lo, 3), "end": round(w["end"] + lo, 3)}
            for w in words]


def _ends_sentence(tok):
    t = str(tok).strip().rstrip(CLOSERS)
    return t.endswith((".", "?", "!")) and t.rstrip(".").lstrip(OPENERS).lower() not in ABBREV


# tokens that never complete a clause even when followed by a pause / punctuation.
# A pause after "like," or "and" is NOT a sentence end — it's a mid-thought handoff.
_TRAILING_DANGLING = frozenset({
    # quotative tag — "I was like, [actual thing]"
    "like",
    # conjunctions
    "and", "but", "so", "because", "or", "cause", "cuz",
    # prepositions
    "to", "of", "with", "in", "on", "at", "by", "for", "from", "as",
    # articles
    "the", "a", "an",
    # subordinators that must be followed by a clause
    "that", "if", "when", "while",
    # hesitations
    "um", "uh",
})


def _tok_core(tok):
    """Lowercase alphanumeric core of a token (strips punctuation + closers)."""
    t = str(tok).strip().rstrip(CLOSERS).lower()
    return "".join(ch for ch in t if ch.isalnum() or ch == "'")


def _is_dangling_tail(tok):
    """True when the token ends mid-clause — trailing comma / colon / semicolon or a
    filler word like ``like,`` / ``and`` that cannot stand as a sentence end even if a long
    pause follows it.

    When the token carries an explicit sentence terminator (``. ? !``) we honor that
    (``He shoved all in.`` is a complete sentence even though ``in`` is a preposition). The
    ONE exception is the quotative tag ``like`` — speakers often say ``I was like, [actual
    thing]`` and the transcript sometimes drops the comma, so a bare ``like.`` is still a
    mid-thought handoff."""
    t = str(tok).strip().rstrip(CLOSERS)
    if not t:
        return True
    if t[-1] in (",", ";", ":"):
        return True
    core = _tok_core(t)
    if core == "like":
        return True
    if t[-1] in ".?!":
        return False
    return core in _TRAILING_DANGLING


def _split_long(words, a, b, max_words):
    """Split words[a..b] at its largest internal gap until no span exceeds max_words.
    Equal gaps split nearest the middle. Iterative, so a gapless run can't blow the stack."""
    out, todo = [], [(a, b)]
    while todo:
        a, b = todo.pop()
        if b - a + 1 <= max_words:
            out.append((a, b))
            continue
        mid = (a + b) / 2.0
        k = max(range(a, b), key=lambda j: (round(words[j + 1]["start"] - words[j]["end"], 3),
                                            -abs(j - mid)))
        todo += [(k + 1, b), (a, k)]
    return sorted(out)


def sentences(words, gap_s=0.7, max_words=40):
    """Group words into sentences: split after . ? ! (ignoring trailing quotes/brackets), at gaps
    >= gap_s, and split any sentence longer than max_words at its largest internal gap."""
    spans, a, n = [], 0, len(words)
    for k, w in enumerate(words):
        if k == n - 1 or _ends_sentence(w["w"]) or words[k + 1]["start"] - w["end"] >= gap_s:
            spans += _split_long(words, a, k, max(1, int(max_words)))
            a = k + 1
    return [{"i": i, "start": float(words[a]["start"]), "end": float(words[b]["end"]),
             "text": " ".join(str(words[j]["w"]).strip() for j in range(a, b + 1)), "w0": a, "w1": b}
            for i, (a, b) in enumerate(spans)]


# a punctuated end the speaker runs straight on from is not a sentence end. 10-20 clips
# 110 ("tonight." + "Hey" 0.08 s later) and 100 ("actually." + "So" 0.24 s later) were cut there;
# 124 ended on "losers." with "Because" right after (whisper base.en put a period before the
# continuation); 109 ended on "Oh".
CONTINUATION_OPENERS = frozenset({"because", "cause", "cuz", "cos", "which", "whereas", "unless",
                                  "until", "than", "whom", "whose"})
END_INTERJECTIONS = frozenset({"oh", "ohh", "ah", "ahh", "uh", "um", "umm", "hmm", "mm", "mhm",
                               "huh", "eh", "er", "erm"})
CONTINUATION_GAP_S = 1.0     # a continuation opener this close after an end makes it a run-on


def clean_edges(sents, hard_gap_s=1.5, edges_clean=True, min_end_gap_s=0.0, strict_ends=False):
    """(clean_start, clean_end) bool lists, one entry per sentence. A sentence ends cleanly when its
    last token closes a sentence (. ? !, see _ends_sentence) or a pause >= hard_gap_s follows it; it
    starts cleanly when the one before it ends cleanly. sentences() also splits at short pauses
    (gap_s) and inside long runs, so not every sentence it returns is a true sentence start/end.
    The first/last sentence counts as clean only when `edges_clean` (a whole transcript, not a
    window cut out of the middle of one); a (start, end) pair sets the two edges apart (a window
    whose speech starts after a pause has a clean first start).

 `strict_ends` (the daily selector): a punctuated end also needs >= min_end_gap_s of
    pause before the next sentence, no continuation opener ("because", "which", ...) right after
    it, and no interjection ("Oh", "um") as its last word; and the last sentence of a window
    (edge end not clean) never ends cleanly, punctuated or not."""
    if isinstance(edges_clean, (tuple, list)):
        edge_start, edge_end = (bool(v) for v in edges_clean)
    else:
        edge_start = edge_end = bool(edges_clean)
    n, ends = len(sents), []
    for k, s in enumerate(sents):
        toks = str(s.get("text", "")).split()
        term = bool(toks) and _ends_sentence(toks[-1])
        # a sentence that ends on a dangling tail (``like,`` / ``and`` / ``to``) does
        # NOT count as "clean end" even when a hard gap follows it — the next phrase is the
        # actual payload, and cutting here produces a trailing-filler end on the clip.
        trailing_dangle = bool(toks) and _is_dangling_tail(toks[-1])
        if strict_ends and toks and _tok_core(toks[-1]) in END_INTERJECTIONS:
            trailing_dangle = True
        if k + 1 < n:
            gap = float(sents[k + 1]["start"]) - float(s["end"])
            long_gap = gap >= hard_gap_s
            if strict_ends and term and not long_gap:
                nxt = str(sents[k + 1].get("text", "")).split()
                run_on = gap < float(min_end_gap_s) or (
                    bool(nxt) and _tok_core(nxt[0]) in CONTINUATION_OPENERS
                    and gap < CONTINUATION_GAP_S)
                term = term and not run_on
            ends.append((term or long_gap) and not trailing_dangle)
        elif strict_ends and not edge_end:
            # whisper closes the last sentence of a cut-out window with a period because the
            # audio stops there (10-20 clip 114: "... at one point." at the window end, mid-thought)
            ends.append(False)
        else:
            ends.append((term or edge_end) and not trailing_dangle)
    starts = [edge_start if k == 0 else ends[k - 1] for k in range(n)]
    return starts, ends


def load_audio(media, lo, hi, sr=16000):
    """Mono float32 samples of [lo, hi] via an ffmpeg pipe. Sample 0 is at time `lo`: a negative
    `lo` is zero-padded so callers can use lo as t0."""
    lo, hi = float(lo), float(hi)
    pad = int(round((min(hi, 0.0) - lo) * sr)) if lo < 0 else 0
    s = max(0.0, lo)
    out = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{s:.3f}",
                          "-t", f"{max(0.0, hi - s):.3f}", "-i", str(media), "-vn", "-ac", "1",
                          "-ar", str(sr), "-f", "f32le", "pipe:1"],
                         check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    a = np.frombuffer(out[:len(out) // 4 * 4], dtype=np.float32)
    return np.concatenate([np.zeros(pad, np.float32), a]) if pad else a.copy()


def speech_regions(audio, sr, offset=0.0, **vad):
    """Speech spans [(start, end)] in absolute seconds via Silero VAD (ships with faster-whisper;
    expects sr=16000). `vad` overrides VadOptions. [] when faster-whisper is unavailable."""
    try:
        from faster_whisper.vad import get_speech_timestamps, VadOptions
    except ImportError:
        return []
    a = np.asarray(audio, dtype=np.float32).ravel()
    if not len(a):
        return []
    ts = get_speech_timestamps(a, VadOptions(**vad), sampling_rate=sr)
    return [(offset + t["start"] / sr, offset + t["end"] / sr) for t in ts]
