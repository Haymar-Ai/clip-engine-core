"""The strong whisper pass: word timestamps with word probabilities from the openai-whisper CLI.

The caption step, the caption check's re-hearing, the caption repair's tie-break model and the
optional window pass all hear their spans through transcribe_strong (large-v3-turbo by default,
stronger than every selection and caption model).
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

STRONG_MODEL = "large-v3-turbo"     # stronger than every caption model (base/small.en)


def transcribe_strong(media, lo, hi, model=STRONG_MODEL, threads=4, timeout=None, prompt=None,
                      extra_args=None):
    """Word timestamps (absolute) with whisper's word probability `p`, from the openai-whisper CLI
    on [lo, hi] of `media` (`prompt`: whisper --initial_prompt). With transcript.WORKER on, the
    same argv runs in a warm whisper worker: the model loads once per run, not per call."""
    from clipengine import transcript
    lo = max(0.0, float(lo))
    with tempfile.TemporaryDirectory(prefix="strong_asr_") as tmp:
        wav = Path(tmp) / "span.wav"
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
                        "-ss", f"{lo:.3f}", "-t", f"{max(0.0, float(hi) - lo):.3f}", "-i",
                        str(media), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                       check=True, timeout=120)
        env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
               "OMP_WAIT_POLICY": "PASSIVE"}
        argv = ["whisper", str(wav), "--model", model, "--language", "en",
                "--word_timestamps", "True", "--output_format", "json", "--output_dir", tmp,
                "--threads", str(threads), "--condition_on_previous_text", "False"]
        if prompt:
            argv += ["--initial_prompt", prompt]
        argv += [str(x) for x in extra_args or ()]     # the caption check's extra CLI options
        if transcript.WORKER:
            transcript._worker_run(argv[1:], timeout=timeout or 1800)
        else:
            with transcript.heavy_slot(model, timeout or 1800) as left:
                subprocess.run(argv, check=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, env=env, timeout=left)
        data = json.loads(next(Path(tmp).glob("span.json")).read_text())
    out = []
    for seg in data.get("segments", []):
        for x in seg.get("words", []) or []:
            t = " ".join(str(x.get("word", "")).split())
            if t:
                out.append({"w": t, "start": round(float(x["start"]) + lo, 3),
                            "end": round(float(x["end"]) + lo, 3),
                            "p": round(float(x.get("probability", 1.0)), 3)})
    return out
