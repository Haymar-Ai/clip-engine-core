#!/usr/bin/env python3
"""Independent second ASR pass for the quality gate.

The burned captions come from the words sidecar the render used (select's base.en full pass, or
its small.en precision pass). The gate cross-checks them against a DIFFERENT whisper model
(``quality_gate.second_asr.model``, default ``small``: the multilingual small, not small.en)
run on the clip's own audio. large-v3-turbo is the stronger judge but ran 5-16 s per audio
second on the shared box (10-19, load 30-50: four of 25 clips hit a 400-780 s cap); small ran
2.5-5 s per audio second and separated the eye-check verdicts the same way (item-2 report):

  * caption agreement: are the burned words what is actually said (10-20 clips 109/110/125/135
    had nonsense captions from crosstalk / event audio);
  * the ending: whisper punctuates from the audio it hears, so a pass that runs ``tail_s`` past
    the cut sees the continuation the clip cuts off (clip 124: "losers[.] because" - base.en put
    a period before the continuation, and "because" starts 0.17 s before the cut);
  * hook support: the hook must be grounded in what is said inside the clip.

The pass runs on the SOURCE media over [start_s, end_s + tail_s] (clamped to the media end), so
the words after the cut are visible; without the source it runs on the rendered clip (no tail).
Times returned are clip-relative. The whisper CLI (openai-whisper) is used on purpose: it reads
the cached ``~/.cache/whisper/<model>.pt`` and never needs a download, and the production python
has no faster-whisper. Results are cached by (media, mtime, window, model).
"""
import contextlib
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from clipengine import config

HERE = Path(__file__).resolve().parent

DEFAULT_MODEL = "small"
DEFAULT_TAIL_S = 2.0
DEFAULT_THREADS = 4
TIMEOUT_BASE_S = 120.0           # CLI cap = base + per_audio_s x audio seconds
TIMEOUT_PER_AUDIO_S = 15.0       # small: 2.5-5 s per audio second on the loaded box (10-19)
DEADLINE = None                  # time.monotonic() the pass must end by (a caller's window
                                 # cap less a gate tail, so a slow pass skips
                                 # checks 3/5 instead of the window being killed with its clip)
MIN_RUN_S = 20.0                 # less time than this left: the pass is not started


def time_left():
    """Seconds until DEADLINE, or None without one (campaign renders, manual runs)."""
    return None if DEADLINE is None else DEADLINE - time.monotonic()


def _timeout(audio_s):
    """The pass's cap: its own (base + per audio second) and never past DEADLINE. Raises
    TimeoutError when the window has no room left for it; the gate turns that into
    informational checks."""
    cap = TIMEOUT_BASE_S + TIMEOUT_PER_AUDIO_S * max(0.0, float(audio_s))
    left = time_left()
    if left is not None:
        if left < MIN_RUN_S:
            raise TimeoutError(f"window cap near ({left:.0f} s left): second ASR not started")
        cap = min(cap, left)
    return cap


def _cfg():
    try:
        return config.load()
    except (OSError, ValueError):
        return {}


def settings(cfg=None):
    """The ``quality_gate.second_asr`` block merged over the defaults."""
    cfg = _cfg() if cfg is None else cfg
    block = ((cfg.get("quality_gate") or {}).get("second_asr") or {})
    scratch = Path(cfg.get("scratch_dir", str(HERE / "scratch")))
    return {
        "enabled": bool(block.get("enabled", True)),
        "model": str(block.get("model", DEFAULT_MODEL)),
        "tail_s": float(block.get("tail_s", DEFAULT_TAIL_S)),
        "threads": int(block.get("threads", DEFAULT_THREADS)),
        "cache_dir": str(block.get("cache_dir") or scratch / "second_asr"),
    }


def caption_models(cfg=None):
    """Models whose words can end up burned in as captions: select's full pass + precision pass
    (the words sidecar), render.caption_model (recompose's own transcription fallback) and
    render.caption_strong_model (clip-quality stage 3: the caption words of every render)."""
    cfg = _cfg() if cfg is None else cfg
    sel = cfg.get("select") or {}
    ren = cfg.get("render") or {}
    out = {str(sel.get("transcribe_model", "base.en")), str(sel.get("refine_model", "small.en")),
           str(ren.get("caption_model", "small.en"))}
    if ren.get("caption_strong_model"):
        out.add(str(ren["caption_strong_model"]))
    return out


def independent(model, cfg=None):
    """None when ``model`` differs from every caption model, else the reason it is not."""
    if model in caption_models(cfg):
        return f"second ASR model {model!r} is also a caption model; it cannot check them"
    return None


def media_duration(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                            "default=noprint_wrappers=1:nokey=1", str(path)],
                           capture_output=True, text=True, timeout=30)
        return float((r.stdout or "").strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def extract_wav(media, start_s, dur_s, wav):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
                    "-ss", f"{max(0.0, float(start_s)):.3f}", "-t", f"{max(0.0, float(dur_s)):.3f}",
                    "-i", str(media), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)


def _worker_on():
    try:
        from clipengine import transcript
    except ImportError:
        return None
    return transcript if transcript.WORKER else None


def _heavy_slot(model, timeout):
    """transcript.heavy_slot (a large model waits for one of the host's heavy-whisper slots), or
    a pass-through without transcript."""
    try:
        from clipengine import transcript
    except ImportError:
        return contextlib.nullcontext(timeout)
    return transcript.heavy_slot(model, timeout)


def _segment_ok():
    """transcript.segment_ok (drop whisper's hallucinated segments), or
    keep every segment without transcript."""
    try:
        from clipengine import transcript
    except ImportError:
        return lambda seg: True
    return transcript.segment_ok


def whisper_cli_words(wav, model, threads=DEFAULT_THREADS, timeout=None):
    """Word timestamps ({w, start, end}, file-relative) from the openai-whisper CLI. OpenMP/MKL
    capped and passive (transcript._cli_env): the box shares its cores with the pipeline. In a
    window (transcript.WORKER on) the same argv runs in the window's warm whisper worker
    (no interpreter start + model load per call)."""
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
           "OMP_WAIT_POLICY": "PASSIVE"}
    with tempfile.TemporaryDirectory(prefix="asr2_") as tmp:
        argv = [str(wav), "--model", model, "--language", "en", "--word_timestamps", "True",
                "--output_format", "json", "--output_dir", tmp, "--threads", str(threads),
                "--condition_on_previous_text", "False"]
        worker = _worker_on()
        if worker is not None:
            worker._worker_run(argv, timeout=timeout)       # takes a heavy slot itself
        else:
            with _heavy_slot(model, timeout) as left:
                subprocess.run(["whisper", *argv], check=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=left, env=env)
        data = json.loads(next(Path(tmp).glob("*.json")).read_text())
    out, ok = [], _segment_ok()
    for seg in data.get("segments", []):
        if not ok(seg):
            continue
        for x in seg.get("words", []) or []:
            t = " ".join(str(x.get("word", "")).split())
            if t:
                out.append({"w": t, "start": round(float(x["start"]), 3),
                            "end": round(float(x["end"]), 3)})
    return out


def _cache_key(media, start_s, end_s, tail_s, model):
    try:
        mt = os.path.getmtime(media)
    except OSError:
        mt = 0
    raw = f"{media}|{mt}|{float(start_s):.3f}|{float(end_s):.3f}|{float(tail_s):.3f}|{model}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_file(media, start_s, end_s, model, tail_s, media_dur, cache_dir):
    """(cache file, tail heard) of the pass over [start_s, end_s] + up to tail_s."""
    tail = max(0.0, min(tail_s, (media_dur - end_s) if media_dur else tail_s))
    return Path(cache_dir) / f"{_cache_key(media, start_s, end_s, tail, model)}.json", tail


def cached(media, start_s, end_s, *, model=None, tail_s=None, cache_dir=None, media_dur=None):
    """The record transcribe() cached for this window ({..., "cached": True}), or None. Never
    runs whisper (the render's re-cut reads the gate's own pass back instead of a new
    one)."""
    s = settings()
    if media_dur is None:
        media_dur = media_duration(media)
    cf, _tail = _cache_file(media, float(start_s), float(end_s), model or s["model"],
                            s["tail_s"] if tail_s is None else float(tail_s), media_dur,
                            cache_dir or s["cache_dir"])
    try:
        return {**json.loads(cf.read_text()), "cached": True}
    except (OSError, ValueError):
        return None


def transcribe(media, start_s, end_s, *, model=None, tail_s=None, cache_dir=None, runner=None,
               threads=None, media_dur=None):
    """Second-ASR words for the clip window [start_s, end_s] of ``media`` plus up to ``tail_s``
    after the cut. Returns {"words": [{w, start, end}] clip-relative (0 = the cut start),
    "dur": end_s - start_s, "tail_s": the tail actually transcribed, "model", "media",
    "cached"}. ``runner(wav, model) -> words`` replaces the whisper CLI (tests). Raises on any
    failure: the gate turns that into an informational measure."""
    s = settings()
    model = model or s["model"]
    tail_s = s["tail_s"] if tail_s is None else float(tail_s)
    threads = threads or s["threads"]
    start_s, end_s = float(start_s), float(end_s)
    dur = end_s - start_s
    if dur <= 0:
        raise ValueError(f"empty window {start_s}-{end_s}")
    if media_dur is None:
        media_dur = media_duration(media)
    cdir = Path(cache_dir or s["cache_dir"])
    cf, tail = _cache_file(media, start_s, end_s, model, tail_s, media_dur, cdir)
    if runner is None and cf.exists():
        try:
            rec = json.loads(cf.read_text())
            return {**rec, "cached": True}
        except ValueError:
            pass
    timeout = _timeout(dur + tail)
    with tempfile.TemporaryDirectory(prefix="asr2wav_") as tmp:
        wav = Path(tmp) / "clip.wav"
        extract_wav(media, start_s, dur + tail, wav)
        if runner is not None:
            words = runner(wav, model)
        else:
            words = whisper_cli_words(wav, model, threads=threads, timeout=timeout)
    rec = {"words": words, "dur": round(dur, 3), "tail_s": round(tail, 3), "model": model,
           "media": str(media), "window": [round(start_s, 3), round(end_s, 3)]}
    if runner is None:
        try:
            cdir.mkdir(parents=True, exist_ok=True)
            tmpf = cf.with_suffix(".tmp")
            tmpf.write_text(json.dumps(rec))
            tmpf.replace(cf)
        except OSError:
            pass
    return {**rec, "cached": False}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="second ASR pass on a clip window")
    ap.add_argument("media")
    ap.add_argument("start_s", type=float)
    ap.add_argument("end_s", type=float)
    ap.add_argument("--model")
    ap.add_argument("--tail-s", type=float)
    ap.add_argument("--cache-dir")
    args = ap.parse_args()
    print(json.dumps(transcribe(args.media, args.start_s, args.end_s, model=args.model,
                                tail_s=args.tail_s, cache_dir=args.cache_dir), indent=1))
