# clip-engine-core

Cuts short vertical clips out of a long video: transcript, picked moments, clean cuts, a 9:16
reframe, burned captions and a hook title, loudness normalisation, and a quality gate that
judges every rendered clip. One command takes one source (a local file or a URL) and writes the
clips that pass the gate.

```
source video -> transcript -> LLM picks sentence ranges -> cuts placed in word gaps
             -> editor pass + caption pass -> 1080x1920 render -> quality gate -> repairs -> clip
```

## What it does not do

No posting, no accounts, no scheduling, no campaigns, no analytics. The engine makes clips and
JSON sidecars in a folder; what happens to them after that is up to you.

## Install

Python 3.10+ and these tools on `PATH`:

| tool | used for |
|---|---|
| `ffmpeg`, `ffprobe` | audio, frames, the render |
| `whisper` (openai-whisper CLI) | every transcription pass (selection, precision, captions, the gate's second ASR) |
| `tesseract` (optional) | on-screen text checks; without it those checks read the frame scan only |
| `yt-dlp` (optional) | downloading a URL; a local file needs none |
| `claude` or a local `ollama` | the LLM that picks moments and reviews cuts (see below) |

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .                 # numpy, opencv-python-headless, pillow
pip install -e '.[whisper]'      # the whisper CLI, if it is not installed already
python -m clipengine --help
```

Whisper models are downloaded by whisper on first use (`base.en`, `small.en`, `small`,
`large-v3-turbo`). If `faster-whisper` is importable, the selection and precision passes use it
instead of the CLI; the caption passes and the gate always use the CLI.

## Run

```bash
python -m clipengine run talk.mp4 --niche podcasts --out out/
python -m clipengine run "https://www.twitch.tv/videos/123456789" --niche gaming --out out/
```

Options: `--niche` (finance, gaming, lifestyle, podcasts, poker, streamers, tech: picks the
prompt pack, the vertical and the render rules), `--title` (the source's title: whisper's prompt
for the caption passes, so names and punctuation come out right), `--layout`, `--vertical`,
`--cookies <file>` (a Netscape cookies file for YouTube downloads; copied, never written),
`--config <file>`, `--scratch <dir>`.

Output, per source:

```
out/<source_id>/clip_01.mp4     every clip that passed the quality gate (1080x1920, H.264/AAC)
out/<source_id>/clip_01.json    every selected clip: start, end, duration, score, title, layout,
                                the gate's verdict and reason codes, QC, selection features
out/<source_id>/run.json        the run: source, niche, steps, clip list, event log
```

Exit code 0 when at least one clip passed, 1 when none did, 2 for a usage or config error.

The run is heavy (several whisper passes per clip, a large-v3-turbo caption pass, OCR). On a
shared machine run it under `nice`. `TRANSCRIBE_WORKER=0` turns off the warm whisper process the
run otherwise keeps for all its whisper calls.

## Config

`config.example.json` holds every default. A file passed with `--config` (or env
`CLIPENGINE_CONFIG`) is merged over it key by key, so it only names what it changes:

```json
{"select": {"semantic_backend": "ollama", "semantic_model": "qwen2.5:7b"},
 "render": {"font_dir": "/path/to/fonts"}}
```

- `select`: models (`transcribe_model`, `refine_model`), the LLM backend, the number of clips
  (`top_n`), head / tail skips, per-vertical duration bands and score weights (`verticals`), the
  topic map, the end rule, prompt packs, the optional whole-source strong pass.
- `quality_gate`: the frame-gate thresholds and per-layout opt-ins, the second ASR, the audio cut
  checks, the caption tie-break, the semantic hook judge.
- `render`: caption models, the caption check and context, loudness, speaker tracking, the
  vertical -> layout map, gamestack / reaction / screencast geometry, game genres, HUD profiles,
  `font_dir`.
- `layout_crops`: manual crop rectangles for the `screencast` layout.
- `scratch_dir`: caches and working files (default `.cache/`, relative to where you run).

`render.crop_feasibility` probes the whole source before transcription and refuses it when no
clean 9:16 speaker crop exists on most sampled frames. Its thresholds are tuned for windows of a
minute or two; a long stream with a persistent cam card or logo bug can be refused as a whole.
`{"render": {"crop_feasibility": {"enabled": false}}}` skips the probe; the render and the gate
still refuse any single clip whose crop would slice a graphic.

Fonts: captions and the hook use Inter (`Inter-VF.ttf`) from `render.font_dir` or env
`CLIPENGINE_FONT_DIR`; without it a system bold font (DejaVu Sans Bold, Liberation Sans Bold) is
used. `CLIP_FONT_PATH` names a font file directly.

Environment switches (each `=0` turns a step off): `TOPIC_MAP`, `END_RULE`, `PROMPT_PACKS`,
`CUT_REVIEW`, `CUT_SEAT`, `FINAL_CUT`, `CAPTION_CHECK`, `CAPTION_PROMPT`, `CROSSTALK_CONFIRM`,
`CUT_AUDIO`, `CAPTION_TIEBREAK`, `WINDOW_TRANSCRIPT`. `YTDLP_BIN` picks the yt-dlp binary.

## LLM backends

`select.semantic_backend` (picks, topic map, cut review, hook judge):

- `claude_cli`: runs `claude -p --model <semantic_model> --output-format json --tools ""` with the
  prompt on stdin. It uses whatever Claude account the environment selects;
  `select.claude_config_dir` and/or `select.claude_env_file` (a KEY=VALUE file) pick one
  explicitly. `--tools ""` turns every tool off for the call: the prompt carries the video's
  transcript, and words spoken in a video must not be able to make the model read a file or run
  a command. The answer only needs the prompt. `tests/test_claude_tools.py` fails on any
  `claude -p` call in the package that does not name its tools.
- `ollama`: POSTs to `http://localhost:11434/api/generate` with `semantic_model`.

`semantic_fallback_backend` / `semantic_fallback_model` are tried once when the primary fails.
When no backend answers, selection falls back to audio energy peaks; when the model answers that
nothing is clip-worthy, no clip is made. Every answer is parsed as JSON and checked against the
transcript (sentence numbers in range, scores clamped) before it is used.

## Tests

```bash
pip install -e '.[dev]'
python -m pytest
```

Unit tests never touch the network, an LLM or a real whisper model. Tests marked `slow` (a real
model or real media tools) run only with `-m slow`.

## Layout

```
clipengine/
  cli.py            the command line
  engine.py         harvest -> select -> render -> gate, and the repairs between them
  config.py         defaults + user file + env
  runstate.py       the run's sources and clips (in memory)
  transcript.py, boundaries.py, select_semantic.py, topic_map.py, cut_review.py, edge_verify.py,
  quality_gate.py, second_asr.py, crosstalk.py, caption_repair.py, captions_*.py, reason_codes.py,
  scenes.py, pick_quality.py, slurs.py, sampling.py, strong_asr.py, whisper_worker.py, ytdl.py
  prompt_packs/     per-niche rubrics
  render/           recompose.py (the renderer) and its layout / caption / face modules
  render/models/    the YuNet face detector
```

## Third-party model

`clipengine/render/models/face_detection_yunet_2023mar.onnx` is the YuNet face detector from the
OpenCV Zoo (opencv_zoo, MIT license); see `clipengine/render/models/NOTICE.md`.
