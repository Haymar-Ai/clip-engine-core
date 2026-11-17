#!/usr/bin/env python3
"""9:16 recompose renderer: subject-aware reframe + captions + hook + loudness, ONE encode.

Layouts:
  auto       reframe.analyze picks track / split / general from sampled faces (default).
  track      9:16 crop locked per shot on the primary face (talking heads).
  split      two face crops stacked 1080x960 each (two-speaker podcasts). With --top/--bottom
             the manual crop rectangles are used instead (unchanged behaviour).
  general    full frame -> blurred-bg letterbox. Right for full-screen graphics (poker table).
  screencast content crop on top + presenter/facecam crop below (needs --top/--bottom).

Everything (seek, layout graph, ASS captions + hook via `subtitles=`, two-pass loudnorm apply,
micro-fades) happens in a single libx264 encode, then the output is QC'd with ffprobe/ebur128.

Usage:
  recompose.py SRC START DUR OUT [--layout auto|general|track|split|screencast]
      [--top CW:CH:CX:CY --bottom FW:FH:FX:FY --split PX --bottom-mode cover|fit]
      [--captions] [--words WORDS.json] [--caption-model small.en] [--caption-style opus]
      [--loudnorm LUFS] [--hook TEXT] [--emit-plan PLAN.json]
Exit: 0 rendered + QC passed, 3 QC failed, other non-zero render failed.
Last stdout line: QC {json}.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from clipengine import config

HERE = Path(__file__).resolve().parent
from clipengine.render import banners   # noqa: E402
from clipengine.render import captions  # noqa: E402
from clipengine.render import cleancrop  # noqa: E402
from clipengine.render import cropcheck  # noqa: E402
from clipengine.render import edgebands  # noqa: E402
from clipengine.render import facecam   # noqa: E402
from clipengine.render import gamelayout  # noqa: E402
from clipengine.render import hud       # noqa: E402
from clipengine.render import overlays  # noqa: E402
from clipengine.render import reframe   # noqa: E402
from clipengine.render import textzones  # noqa: E402

OUT_W, OUT_H = 1080, 1920
MAX_FPS = 60
LOUD_TP, LOUD_LRA = -1.0, 11
FADE_IN, FADE_OUT = 0.04, 0.12
QC_DUR_TOL = 0.15
QC_LUFS_TOL = 1.5
EXIT_QC = 3
# RENDER_ENCODER picks the video encoder: unset = libx264; a macOS host can set
# h264_videotoolbox or hevc_videotoolbox (8M video, maxrate/bufsize scaled from 10M/14M/20M)
ENCODERS = {
    "libx264": ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high",
                "-pix_fmt", "yuv420p", "-g", "60"],
    "h264_videotoolbox": ["-c:v", "h264_videotoolbox", "-profile:v", "high", "-b:v", "8M",
                          "-maxrate", "11200k", "-bufsize", "16M", "-pix_fmt", "yuv420p",
                          "-g", "60", "-allow_sw", "0", "-realtime", "0"],
    "hevc_videotoolbox": ["-c:v", "hevc_videotoolbox", "-profile:v", "main", "-b:v", "7M",
                          "-maxrate", "10M", "-bufsize", "14M", "-pix_fmt", "yuv420p",
                          "-tag:v", "hvc1", "-g", "60", "-allow_sw", "0", "-realtime", "0"],
}
_ENC = os.environ.get("RENDER_ENCODER", "libx264")
if _ENC not in ENCODERS:
    sys.exit(f"recompose: unknown RENDER_ENCODER {_ENC!r}")   # exit 1 -> mini 'env' fallback
VIDEO_CODEC = ENCODERS[_ENC]
AUDIO_CODEC = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]


def _vertical_layouts():
    """Vertical -> layout mapping from the config's render.vertical_layouts block.
    Broken/missing config is not fatal: layouts fall back to `decide_layout`."""
    try:
        cfg = config.load()
    except (OSError, ValueError):
        return {}
    m = (cfg.get("render") or {}).get("vertical_layouts") or {}
    return {str(k): str(v) for k, v in m.items() if isinstance(v, str)}


class RenderError(Exception):
    pass


def log(msg):
    print(msg, file=sys.stderr)


def run(cmd, cwd=None):
    """Run a tool with captured text output; raise RenderError on a non-zero exit."""
    cmd = [str(c) for c in cmd]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                       errors="replace", cwd=cwd)
    if r.returncode != 0:
        raise RenderError(f"{Path(cmd[0]).name} exited {r.returncode}: "
                          f"{(r.stderr or r.stdout or '').strip()[-800:]}")
    return r


def rect(s):
    w, h, x, y = (int(v) for v in s.split(":"))
    return w, h, x, y


def _rate(s):
    try:
        num, _, den = str(s or "0").partition("/")
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path):
    """{"width", "height", "fps", "duration", "has_audio"} via ffprobe."""
    d = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                        "-of", "json", path]).stdout or "{}")
    streams = d.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    return {"width": int(v.get("width") or 0), "height": int(v.get("height") or 0),
            "fps": _rate(v.get("avg_frame_rate")) or _rate(v.get("r_frame_rate")),
            "duration": float(d.get("format", {}).get("duration") or 0.0),
            "has_audio": any(s.get("codec_type") == "audio" for s in streams)}


# ---------------------------------------------------------------- captions

def load_words(path, start, dur):
    """WORDS.json (absolute source seconds) -> clip-relative words inside [0, dur]."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("words", [])
    shifted = [dict(w, start=float(w["start"]) - start, end=float(w["end"]) - start)
               for w in data]
    return captions.clean_words(shifted, dur)


def extract_audio(src, start, dur, wav):
    run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-ss", start, "-t", dur, "-i", src,
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav])


def _transcribe_faster_whisper(wav, model):
    from faster_whisper import WhisperModel
    from clipengine.transcript import segment_ok   # whisper hallucination guards
    segs, _ = WhisperModel(model, device="cpu", compute_type="int8").transcribe(
        str(wav), language="en", word_timestamps=True, condition_on_previous_text=False)
    return [{"w": w.word.strip(), "start": float(w.start), "end": float(w.end)}
            for s in segs if segment_ok(s) for w in (s.words or [])]


def _transcribe_whisper_cli(wav, model, tmp):
    run(["whisper", wav, "--model", model, "--word_timestamps", "True", "--output_format",
         "json", "--output_dir", tmp, "--language", "en"])
    return captions.words_from_whisper(json.loads((Path(tmp) / f"{Path(wav).stem}.json").read_text()))


def transcribe(src, start, dur, model, tmp):
    """Clip-relative words for the window: audio-only extract, then faster_whisper if
    importable, else the whisper CLI."""
    wav = Path(tmp) / "window.wav"
    extract_audio(src, start, dur, wav)
    try:
        return _transcribe_faster_whisper(wav, model)
    except ImportError:
        pass
    except Exception as e:
        log(f"faster_whisper failed ({e}); trying whisper CLI")
    return _transcribe_whisper_cli(wav, model, tmp)


def caption_words(a, src, tmp):
    if a.words:
        return load_words(a.words, a.start, a.dur)
    try:
        return captions.clean_words(transcribe(src, a.start, a.dur, a.caption_model, tmp), a.dur)
    except Exception as e:   # captions are an enhancement: render without them
        log(f"transcription failed ({e}); rendering without captions")
        return []


# ---------------------------------------------------------------- layout graphs

def general_vf(plan=None):
    # blurred bg (cover) + foreground fit, centered. A screen fit's content_box is
    # cropped first, so the box (the screen / chart region its text sits in whole) fills the
    # canvas width over a blurred fill of the same frame.
    box = (plan or {}).get("content_box")
    head = (f"[0:v]crop={int(box['w'])}:{int(box['h'])}:{int(box['x'])}:{int(box['y'])},"
            if box else "[0:v]")
    return (f"{head}split=2[bg][fg];"
            f"[bg]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{OUT_H},gblur=sigma=22[bgb];"
            f"[fg]scale={OUT_W}:-2{':flags=lanczos' if box else ''}[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]")


def stack_vf(top, bottom, top_h, bottom_mode="cover"):
    # top crop cover-scaled to its band; bottom crop either cover-scaled (two landscape
    # speakers) or fit-by-height over a blurred fill (portrait facecam — avoids the vertical
    # face-slice that cover forces when a portrait source goes into a landscape band).
    bot_h = OUT_H - top_h
    tw, th, tx, ty = top
    fw, fh, fx, fy = bottom
    top_f = (f"[0:v]crop={tw}:{th}:{tx}:{ty},"
             f"scale={OUT_W}:{top_h}:force_original_aspect_ratio=increase,"
             f"crop={OUT_W}:{top_h}[top]")
    if bottom_mode == "fit":
        bot_f = (f"[0:v]crop={fw}:{fh}:{fx}:{fy},split=2[fb][ff];"
                 f"[fb]scale={OUT_W}:{bot_h}:force_original_aspect_ratio=increase,"
                 f"crop={OUT_W}:{bot_h},gblur=sigma=22[fbb];"
                 f"[ff]scale=-2:{bot_h}[ffs];"
                 f"[fbb][ffs]overlay=(W-w)/2:0[bot]")
    else:
        bot_f = (f"[0:v]crop={fw}:{fh}:{fx}:{fy},"
                 f"scale={OUT_W}:{bot_h}:force_original_aspect_ratio=increase,"
                 f"crop={OUT_W}:{bot_h}[bot]")
    return f"{top_f};{bot_f};[top][bot]vstack=inputs=2,setsar=1[v0]"


def track_crop_window(plan):
    """(crop w, crop h or "ih", crop y) of a track-like plan. A `crop_box` (banners.fix_track_plan
    trimmed a source banner's rows out) replaces the full-height 9:16 window."""
    box = plan.get("crop_box")
    if box:
        return int(box["w"]), int(box["h"]), int(box["y"])
    return reframe.crop_width(plan["src_w"], plan["src_h"]), "ih", 0


def track_boxed(plan):
    """True when the banner guard boxed the crop: its window is fit to the canvas width over a
    blurred fill instead of filling the 9:16 canvas (banners.fix_track_plan step 3)."""
    return (plan.get("crop_box") or {}).get("fit") == "box"


def track_vf(plan):
    """The track crop's graph; with plan["text_shots"] (render yield 11-09: a shot whose burned-in
    text the crop would slice, class 3, or that shows a logo bug the crop would cut, class 2) each
    shot's own window, or the frame fit to width, overlays the track for its span."""
    base = _track_base_vf(plan)
    shots = plan.get("text_shots") or []
    if not shots:
        return base
    n = len(shots)
    base = base.replace("[0:v]", "[tsb]", 1)[:-len("[v0]")] + "[tsm]"
    parts = ["[0:v]split=" + str(n + 1) + "[tsb]" + "".join(f"[ts{i}]" for i in range(n)), base]
    for i, s in enumerate(shots):
        if s["mode"] == "fit":
            parts.append(f"[ts{i}]split=2[tf{i}b][tf{i}f];"
                         f"[tf{i}b]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
                         f"crop={OUT_W}:{OUT_H},gblur=sigma=22[tf{i}g];"
                         f"[tf{i}f]scale={OUT_W}:-2:flags=lanczos[tf{i}s];"
                         f"[tf{i}g][tf{i}s]overlay=(W-w)/2:(H-h)/2,setsar=1[tso{i}]")
        else:
            parts.append(f"[ts{i}]crop={int(s['w'])}:{int(s['h'])}:{int(s['x'])}:{int(s['y'])},"
                         f"scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1[tso{i}]")
    prev = "tsm"
    for i, s in enumerate(shots):
        out = "v0" if i == n - 1 else f"tsx{i}"
        parts.append(f"[{prev}][tso{i}]overlay=0:0:"
                     f"enable='between(t,{float(s['t0']):.3f},{float(s['t1']):.3f})'[{out}]")
        prev = out
    return ";".join(parts)


def _track_base_vf(plan):
    cw, ch, cy = track_crop_window(plan)
    # when the plan carries a sendcmd script (dense zero-phase smoothed crop
    # trajectory, hundreds of segments), drive the crop filter's x parameter via
    # sendcmd instead of a nested-if expression. ffmpeg's expression parser trips
    # on expressions with >~200 nested if()s; sendcmd scales freely.
    cmd_file = plan.get("sendcmd_file")
    if cmd_file and os.path.exists(cmd_file):
        crop = f"[0:v]sendcmd=f={cmd_file},crop={cw}:{ch}:0:{cy}"
    else:
        crop = f"[0:v]crop={cw}:{ch}:'{reframe.crop_x_expr(plan['segments'])}':{cy}"
    if track_boxed(plan):
        return (f"{crop},split=2[bxbg][bxfg];"
                f"[bxbg]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
                f"crop={OUT_W}:{OUT_H},gblur=sigma=22[bxbgb];"
                f"[bxfg]scale={OUT_W}:-2:flags=lanczos[bxfgs];"
                f"[bxbgb][bxfgs]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]")
    return f"{crop},scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1[v0]"


def broadcast_vf(plan):
    """Broadcast layout: subject-tracking top band (1080x1312) stacked on the full 16:9 source
    frame scaled to 1080x608, so every HUD graphic on the bottom stays visible."""
    top_h = reframe.BROADCAST_TOP_H
    bot_h = OUT_H - top_h
    cw = reframe.broadcast_crop_width(plan["src_w"], plan["src_h"])
    top_x = reframe.crop_x_expr(plan["segments"])
    # bottom is fit-to-container so a non-16:9 source letterboxes instead of cropping the HUD off.
    return (f"[0:v]split=2[vt][vb];"
            f"[vt]crop={cw}:ih:'{top_x}':0,scale={OUT_W}:{top_h}:flags=lanczos,setsar=1[top];"
            f"[vb]scale={OUT_W}:{bot_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={OUT_W}:{bot_h}:(ow-iw)/2:(oh-ih)/2,setsar=1[bot];"
            f"[top][bot]vstack=inputs=2,setsar=1[v0]")


def write_sendcmd_file(segments, out_path, max_entries=4000):
    """Helper: translate a dense list of crop ``segments`` into an ffmpeg
    ``sendcmd`` command file. Each entry sets the ``crop`` filter's ``x`` parameter at the
    segment's t0. The nested-``if()`` expression that :func:`reframe.crop_x_expr` emits trips
    ffmpeg's parser past ~200 nested calls; sendcmd scales to thousands of entries.

    ``max_entries`` is a safety cap in case the segment list is truly massive.
    """
    from pathlib import Path as _P
    lines = []
    seen = set()
    for s in segments[:max_entries]:
        t = float(s.get("t0", 0.0))
        x = int(round(float(s.get("x", 0))))
        key = round(t, 3)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{t:.3f} crop x {x};")
    _P(out_path).write_text("\n".join(lines))
    return str(out_path)


def chart_top_cam_bottom_vf(plan):
    """chart_top_cam_bottom: top panel = full source cover-scaled to 1080x1312 so the
    chart fills with no black bars (horizontal crop may trim mild side regions); bottom
    panel = cam box crop zoomed to 1080x608 so the streamer's face is >= 25 % of the panel.

    The planner (``_plan_chart_top_cam_bottom``) sets ``cam_box=[x,y,w,h]`` in source px.
    When no cam box could be detected the planner returns None and the caller falls back to
    ``broadcast_vf`` (the earlier behaviour).

    when the planner also detected the chart content bbox (``chart_bbox=[x,y,w,h]``
    in source px), the top-panel source is first cropped to that bbox before cover-scaling.
    This zooms the price action to fill the panel instead of showing a thin flat line of
    candles on a mostly-black screen. When no bbox is set the full source
    goes to the top panel (earlier behaviour).
    """
    top_h = reframe.BROADCAST_TOP_H
    bot_h = OUT_H - top_h
    cx, cy, cw, ch = plan["cam_box"]
    chart_bbox = plan.get("chart_bbox")
    if chart_bbox:
        chx, chy, chw, chh = chart_bbox
        top_f = (f"[vt]crop={chw}:{chh}:{chx}:{chy},"
                 f"scale={OUT_W}:{top_h}:force_original_aspect_ratio=increase:flags=lanczos,"
                 f"crop={OUT_W}:{top_h},setsar=1[top]")
    else:
        top_f = (f"[vt]scale={OUT_W}:{top_h}:force_original_aspect_ratio=increase:flags=lanczos,"
                 f"crop={OUT_W}:{top_h},setsar=1[top]")
    return (f"[0:v]split=2[vt][vb];"
            f"{top_f};"
            f"[vb]crop={cw}:{ch}:{cx}:{cy},"
            f"scale={OUT_W}:{bot_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={OUT_W}:{bot_h},setsar=1[bot];"
            f"[top][bot]vstack=inputs=2,setsar=1[v0]")


CHART_BBOX_SAMPLE_FPS = 1.0            # sample at >= 1 Hz across the WHOLE window
CHART_BBOX_PANEL_BLACK_FRAC = 0.90     # a sampled source frame whose entire visible
                                        # area is near-black means the chart overlay is gone
                                        # — the planner reports this as a hard fault so the
                                        # pick can be dropped before render time.
CHART_BBOX_MIN_CANDLES = 20            # minimum distinct candles the union mask must
                                        # contain. A readable price window has >= 20 vertical
                                        # candle columns; a thin flat-line chart has 2-6.


def detect_chart_content_bbox(src, start, dur, sample_fps=CHART_BBOX_SAMPLE_FPS,
                               min_area_frac=0.03, max_area_frac=0.85, pad_frac=0.03):
    """detect the "chart content" region on the source so the top panel of
    chart_top_cam_bottom can be ZOOMED to it instead of cover-scaled to the full frame.

    Pipeline (replaces an earlier 5-sample pipeline):
      1. sample the WHOLE window at ``sample_fps`` (default 1 Hz) at a 320-px scan
         width — a 38 s clip gets ~38 samples instead of 5. The earlier 5-sample path
         missed the last 6 s of one chart clip going fully black.
      2. per-pixel TEMPORAL STANDARD DEVIATION across the WHOLE sample set: candles
         update throughout the clip (not just between the first 5 frames of the clip).
      3. per-sample NEAR-BLACK detection: when any sample's whole visible area is
         near-black (brightest channel <= 15 for >= 90 % of pixels) the function
         returns a special sentinel ``{"black_at_s": [...]}`` structure so the planner
         can refuse to render through the bad window instead of silently zooming into
         a chart that disappears.
      4. UNION the per-sample content masks (saturation + motion). The union — not
         any single frame's bbox — is what the engine renders.
      5. CANDLE COUNT: count distinct vertical colored components (price candles) in
         the union mask; a readable chart requires >= 20. ``None`` returned (planner
         falls back to the broadcast scale) when fewer than 20 are detected.
      6. threshold + bbox + validate as before: the padded bbox of the union mask is
         returned in SOURCE pixel coordinates (``[x, y, w, h]``) when the content
         area is between ``min_area_frac`` and ``max_area_frac`` of the frame.

    Returns ``[x, y, w, h]`` in source pixels, OR ``{"error": ..., "black_at_s": [...]}``
    when the source goes black during the window, OR ``None`` when detection fails for
    any other reason (caller falls back to broadcast cover-scale).
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    try:
        src_w, src_h = reframe.probe_size(str(src))
    except Exception as e:                               # noqa: BLE001
        log(f"detect_chart_content_bbox: probe failed ({e})")
        return None
    if src_w == 0 or src_h == 0 or dur <= 0:
        return None
    scan_w = 320
    scan_h = max(2, int(round(src_h * scan_w / src_w / 2)) * 2)
    # sample across the WHOLE window at ``sample_fps`` so a late-window black
    # episode or a shifting chart is caught. iter_frames does the ffmpeg decode with
    # ``-ss start -t dur`` so this stays single-pass.
    try:
        frames_t = list(reframe.iter_frames(str(src), float(start), float(dur),
                                              fps=float(sample_fps), width=scan_w,
                                              src_size=(src_w, src_h)))
    except Exception as e:                               # noqa: BLE001
        log(f"detect_chart_content_bbox: iter_frames failed ({e})")
        return None
    if len(frames_t) < 2:
        return None
    frames = [f[1] for f in frames_t]
    times = [float(f[0]) for f in frames_t]
    # per-sample near-black fault — a frame is "panel black" when it is
    # dominated by near-black pixels AND has essentially no saturated color content
    # (so the chart overlay has truly disappeared, not just gone sparse). This
    # separates "live chart with small candles on dark bg" (saturated pixels present)
    # from "overlay missing / black fade" (no color anywhere).
    black_at = []
    for i, f in enumerate(frames):
        maxc = f.max(axis=2)
        dark_frac = float((maxc <= 15).mean())
        hsv_i = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        sat_i = hsv_i[..., 1]
        colored_frac = float((sat_i >= 60).mean())
        if dark_frac >= CHART_BBOX_PANEL_BLACK_FRAC and colored_frac < 0.005:
            black_at.append(round(times[i], 2))
    if black_at:
        # The planner's caller treats this as a hard fault — the pick cannot be rendered
        # as chart_top_cam_bottom because the top panel would go black. Returning the
        # sentinel instead of None preserves the diagnostic reason for the drop.
        return {"error": "source_chart_went_black", "black_at_s": black_at,
                "sample_fps": float(sample_fps), "src_wh": [src_w, src_h]}
    stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
    tstd = stack.std(axis=0).max(axis=-1)                # (H, W)
    med = np.median(stack, axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32)
    sat_floor = max(25.0, float(sat.mean()) * 1.5)
    tstd_floor = max(2.0, float(tstd.mean()) * 1.5)
    # UNION mask — a pixel is "chart content" if ANY sample saw strong
    # saturation OR temporal std was above the noise floor across the window.
    sat_union = (sat >= sat_floor)
    tstd_union = (tstd >= tstd_floor)
    raw_mask = (sat_union & tstd_union).astype("uint8") * 255
    # OR in the per-sample saturation mask: a candle that only appears in a few
    # samples would otherwise get killed by the median-based path. This is the real
    # "union over samples" the spec asks for.
    for f in frames:
        hsv_i = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        sat_i = hsv_i[..., 1].astype(np.float32)
        raw_mask |= ((sat_i >= sat_floor) & (tstd >= tstd_floor)).astype("uint8") * 255
    # count distinct candle columns from the RAW (pre-morph) mask — morph
    # close merges adjacent candles into one blob and would hide a thin flat-line
    # chart. Candle counting walks the raw column histogram.
    col_hits_raw = (raw_mask > 0).any(axis=0)
    cnt_candles = 0
    in_run = False
    run_len = 0
    for c in col_hits_raw:
        if c:
            run_len += 1
            if not in_run:
                in_run = True
        else:
            if in_run and run_len >= 1:
                cnt_candles += 1
            in_run = False
            run_len = 0
    if in_run:
        cnt_candles += 1
    if cnt_candles < CHART_BBOX_MIN_CANDLES:
        log(f"detect_chart_content_bbox: only {cnt_candles} candle columns detected "
            f"(< {CHART_BBOX_MIN_CANDLES} min); returning None so planner falls back")
        return None
    # Only AFTER candle count passes do we close + bbox the content.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=lambda c: cv2.contourArea(c))
    x0, y0, w0, h0 = cv2.boundingRect(cnt)
    area_frac = (w0 * h0) / float(scan_w * scan_h)
    if area_frac < min_area_frac or area_frac > max_area_frac:
        return None
    pad_px_x = int(round(pad_frac * scan_w))
    pad_px_y = int(round(pad_frac * scan_h))
    x0 = max(0, x0 - pad_px_x); y0 = max(0, y0 - pad_px_y)
    x1 = min(scan_w, x0 + w0 + 2 * pad_px_x)
    y1 = min(scan_h, y0 + h0 + 2 * pad_px_y)
    # Scale back to source pixels
    sx = src_w / float(scan_w); sy = src_h / float(scan_h)
    sx0 = int(round(x0 * sx)); sy0 = int(round(y0 * sy))
    sw0 = int(round((x1 - x0) * sx)); sh0 = int(round((y1 - y0) * sy))
    sx0 -= sx0 % 2; sy0 -= sy0 % 2
    sw0 -= sw0 % 2; sh0 -= sh0 % 2
    sw0 = max(2, min(sw0, src_w - sx0))
    sh0 = max(2, min(sh0, src_h - sy0))
    return [sx0, sy0, sw0, sh0]


HUDBAND_BG = hud.HUDBAND_BG

GAMESTACK_TOP_H = 1344     # default game-band height; overridden by render.gamestack.top_h in config
GAMESTACK_CAM_GUARD_FRAC = 0.02  # extra pixels (per side, as fraction of src width) added around
                                  # the RAW grow box before the game crop picks its x. The cam's
                                  # dark frame border + drop shadow sit just outside the raw grow
                                  # box; without this buffer the game crop hugs the raw box and
                                  # the border bleeds into the game band as a vertical sliver.
REACTION_TOP_H = 960       # default reaction content-band height; overridden by render.reaction.top_h.
REACTION_CONTENT_FIT = "fit"  # fit-over-blur so the reacted video is never cropped; cover=hard crop.
SCREENCAST_TOP_H = 1300    # default auto-screencast content-band height; overridden by render.screencast.top_h.


def _gamestack_cfg():
    """`render.gamestack` block from the config (or {} when unset/broken)."""
    try:
        cfg = config.load()
    except (OSError, ValueError):
        return {}
    return ((cfg.get("render") or {}).get("gamestack") or {})


def _config():
    try:
        return config.load()
    except (OSError, ValueError):
        return {}


def gamestack_vf(plan):
    """Two-band gamestack: game on top (full source height crop, cover-scaled or fit-with-blur),
    webcam inset on the bottom (cover-cropped around the median face, or fit-with-blur for a
    portrait cam that's taller than the band's aspect)."""
    if plan.get("v2"):
        return gamestack_v2_vf(plan)
    top_h = int(plan["top_h"])
    bot_h = OUT_H - top_h
    gw, gx, g_fit = plan["game"]
    cw, ch, cx, cy, c_fit = plan["cam"]
    # render yield 11-09: the streamer split's game window (game_box) need not be full height
    gb = plan.get("game_box")
    gcrop = f"{int(gb[0])}:{int(gb[1])}:{int(gb[2])}:{int(gb[3])}" if gb else f"{gw}:ih:{gx}:0"
    tags = []
    if g_fit:
        tags += ["gbg", "gfg"]
    else:
        tags += ["gc"]
    if c_fit:
        tags += ["cbg", "cfg"]
    else:
        tags += ["cc"]
    parts = ["[0:v]split=" + str(len(tags)) + "".join(f"[{t}]" for t in tags)]
    if g_fit:
        parts.append(f"[gbg]crop={gw}:ih:{gx}:0,"
                     f"scale={OUT_W}:{top_h}:force_original_aspect_ratio=increase,"
                     f"crop={OUT_W}:{top_h},gblur=sigma=22[topbg]")
        parts.append(f"[gfg]crop={gw}:ih:{gx}:0,scale={OUT_W}:-2[topfg]")
        parts.append("[topbg][topfg]overlay=(W-w)/2:(H-h)/2[top]")
    else:
        parts.append(f"[gc]crop={gcrop},"
                     f"scale={OUT_W}:{top_h}:force_original_aspect_ratio=increase,"
                     f"crop={OUT_W}:{top_h}[top]")
    if c_fit:
        parts.append(f"[cbg]crop={cw}:{ch}:{cx}:{cy},"
                     f"scale={OUT_W}:{bot_h}:force_original_aspect_ratio=increase,"
                     f"crop={OUT_W}:{bot_h},gblur=sigma=22[botbg]")
        parts.append(f"[cfg]crop={cw}:{ch}:{cx}:{cy},scale=-2:{bot_h}[botfg]")
        parts.append("[botbg][botfg]overlay=(W-w)/2:0[bot]")
    else:
        parts.append(f"[cc]crop={cw}:{ch}:{cx}:{cy},"
                     f"scale={OUT_W}:{bot_h}:force_original_aspect_ratio=increase,"
                     f"crop={OUT_W}:{bot_h}[bot]")
    parts.append("[top][bot]vstack=inputs=2,setsar=1[v0]")
    return ";".join(parts)


def gamestack_v2_vf(plan):
    """Gamestack / game only v2 and v3: the game band (its crop x, and y for
    a narrow source, on the action path) at canvas row band_y (v3: under the hook header; v2
    plans: 0), the card in its bottom corner (the cam crop; a v2 game only card is the whole
    frame; v3 game only has none) with a light frame, over an ambient blur of the band (blurred
    at a quarter of the canvas: the same look for a fraction of a full-size gblur)."""
    b = plan["band"]
    gw, ch, gh = int(b["w"]), int(b["h"]), int(b["out_h"])
    x = gamelayout.pan_expr(plan["pan"])
    y = gamelayout.pan_expr(plan["pan_y"]) if plan.get("pan_y") else str(int(b.get("y", 0)))
    band_y = int(plan.get("band_y", 0))
    fw, fh = OUT_W // gamelayout.FILL_SCALE, OUT_H // gamelayout.FILL_SCALE
    # a mid-frame cam the band shows is painted out of the band only (the card keeps it)
    mask = overlays.cover_filter([plan["cam_mask"]], plan["src_w"], plan["src_h"]) \
        if plan.get("cam_mask") else ""
    parts = ["[0:v]split=2[gsrc][csrc]" if plan.get("card") else "[0:v]null[gsrc]",
             f"[gsrc]{mask + ',' if mask else ''}crop={gw}:{ch}:'{x}':'{y}',"
             f"scale={OUT_W}:{gh}:flags=lanczos,setsar=1,split=2[band][bandf]",
             f"[bandf]scale=-2:{fh},crop={fw}:{fh},gblur=sigma={gamelayout.FILL_SIGMA},"
             f"scale={OUT_W}:{OUT_H},setsar=1[fill]"]
    if not plan.get("card"):
        return ";".join(parts + [f"[fill][band]overlay=0:{band_y},setsar=1[v0]"])
    kx, ky, kw, kh = (int(v) for v in plan["card"])
    if plan["layout"] == "game_only":
        card = f"[csrc]scale={kw}:{kh}:flags=lanczos"
    else:
        cw, cch, cx, cy, _fit = plan["cam"]
        card = f"[csrc]crop={cw}:{cch}:{cx}:{cy},scale={kw}:{kh}:flags=lanczos"
    return ";".join(parts + [
        f"{card},setsar=1,drawbox=x=0:y=0:w=iw:h=ih:color=white@0.85:"
        f"t={gamelayout.INSET_BORDER}[card]",
        f"[fill][band]overlay=0:{band_y}[gsb]",
        f"[gsb][card]overlay={kx}:{ky},setsar=1[v0]"])


def _game_only_vf(plan):
    """Fallback for a source with no face at all: tight 9:16 crop of the full source, centred."""
    if plan.get("v2"):
        return gamestack_v2_vf(plan)
    src_w = plan["src_w"]
    src_h = plan["src_h"]
    cw = reframe.crop_width(src_w, src_h)
    x = max(0, (src_w - cw) // 2)
    return (f"[0:v]crop={cw}:ih:{x}:0,"
            f"scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1[v0]")


def _plan_gamestack(src, a):
    """Build a gamestack plan or return None (caller falls back).

    Kept on the original facecam.get_or_detect signature so existing mocks keep working.
    Reaction / auto-screencast go through `_resolve_cam` (same shape, wider signature).

    Steps:
      1. Resolve or detect the streamer's webcam box (per-channel profile if we have one,
         multi-window detection on the source otherwise). Save the profile on first success.
      2. Verify: face inside the box in >= facecam.FACE_ACCEPT of samples; below that,
         gamestack isn't the right layout for this clip.
      3. Plan the top game crop (mode from `render.game_genres[game]`, avoiding the cam's
         x-range) and the bottom cam crop (cover on landscape cam, fit-with-blur on portrait).
    """
    try:
        src_w, src_h = reframe.probe_size(str(src))
    except Exception as e:
        log(f"gamestack: probe failed ({e}); falling back")
        return None
    cfg = _config()
    gscfg = (cfg.get("render") or {}).get("gamestack") or {}
    if gscfg.get("v2"):
        return _plan_gamestack_v2(src, a, cfg, gscfg, (src_w, src_h))
    top_h = int(gscfg.get("top_h", GAMESTACK_TOP_H))
    if top_h <= 0 or top_h >= OUT_H:
        top_h = GAMESTACK_TOP_H
    channel = getattr(a, "channel", None) or None
    src_duration = float(getattr(a, "src_duration", 0) or 0)
    if not src_duration:
        try:
            src_duration = float(probe(str(src))["duration"])
        except Exception:
            src_duration = 0.0
    cache_dir = getattr(a, "facecam_cache", None) or None
    box_norm, raw_box_norm, from_profile, face_frac = facecam.get_or_detect(
        channel, str(src), a.start, a.dur, source_duration=src_duration, log=log,
        cache_dir=cache_dir)
    if box_norm is None:
        log(f"gamestack: no cam inset (face_frac=0); falling back")
        return None
    if face_frac < facecam.FACE_ACCEPT:
        log(f"gamestack: face-in-box {face_frac:.0%} < "
            f"{facecam.FACE_ACCEPT:.0%}; falling back")
        return None
    bx0 = int(round(box_norm[0] * src_w)); by0 = int(round(box_norm[1] * src_h))
    bx1 = int(round(box_norm[2] * src_w)); by1 = int(round(box_norm[3] * src_h))
    bx0 -= bx0 % 2; by0 -= by0 % 2
    bx1 = (bx1 // 2) * 2; by1 = (by1 // 2) * 2
    cam_box = [max(0, bx0), max(0, by0), min(src_w, bx1), min(src_h, by1)]
    nb, why = bound_cam(cam_box, a, src, (src_w, src_h))
    if why:
        log(f"gamestack: {why}")
    if nb is None:
        log("gamestack: falling back")
        return None
    cam_box = nb
    rx0 = int(round(raw_box_norm[0] * src_w)); rx1 = int(round(raw_box_norm[2] * src_w))
    ry0 = int(round(raw_box_norm[1] * src_h)); ry1 = int(round(raw_box_norm[3] * src_h))
    guard = int(round(GAMESTACK_CAM_GUARD_FRAC * src_w))
    guard_box = [max(0, rx0 - guard), max(0, ry0 - guard),
                 min(src_w, rx1 + guard), min(src_h, ry1 + guard)]
    frames = facecam.sample_native(str(src), a.start, a.dur, n=15)
    face_med_norm = None
    if frames:
        import numpy as np
        faces_px, _ = facecam._collect_faces(frames)
        if faces_px:
            face_med_norm = [float(np.median([f[0] / src_w for f in faces_px])),
                             float(np.median([f[1] / src_h for f in faces_px])),
                             float(np.median([f[2] / src_h for f in faces_px]))]
    game = getattr(a, "game", None) or None
    mode = facecam.game_genre_mode(cfg, game)
    gw, gx, g_fit = facecam.game_crop(src_w, src_h, guard_box, mode=mode, top_h=top_h)
    cw, ch, cx, cy, c_fit = facecam.cam_crop(src_w, src_h, cam_box, face_med_norm,
                                              bot_h=OUT_H - top_h)
    return {"layout": "gamestack", "src_w": src_w, "src_h": src_h, "top_h": top_h,
            "game": [gw, gx, g_fit], "cam": [cw, ch, cx, cy, c_fit],
            "cam_box": cam_box, "cam_box_norm": box_norm,
            "raw_cam_box": guard_box, "raw_cam_box_norm": raw_box_norm,
            "from_profile": bool(from_profile), "face_frac": float(face_frac),
            "game_mode": mode, "face_median": face_med_norm, "channel": channel}


V2_SLICE_MARGIN = 0.6       # a v2 band x whose gate slice share stays under this x the cap first
V2_SHARE_STEP = 0.01        # band shares tried below render.gamestack.game_share, to the floor
V2_GATE_RULES = 5           # _v2_band's first rules: every banner whole, the gate's verdict passes
V2_HARD_RULES = 7           # _v2_band's rules that keep every banner whole
V2_SCENE_TEXT_FRAC = 0.35   # a banner row on screen in fewer of the scan frames is scene text the
                            # camera moves through (2620's wall sign: 23 %), no overlay
V2_WIDEN_SHARES = (0.53, 0.56, 0.59, 0.62, 0.65, 0.68)   # v3: tried when no band at the share
                            # passes (2670's 700 px subtitle row); the narrowest clean one wins
V2_WIDEN_MAX = 0.68         # ... up to v2's share for a gamestack (blur ~41 %, gate cap 50 %)
V2_WIDEN_MAX_GAME_ONLY = 0.56   # ... a game only band has no card: 0.56 keeps the blur <= 45 %


def _v2_band(src, a, sw, sh, share, ranges_fn, exclude=(), min_share=gamelayout.MIN_SHARE,
             max_h=gamelayout.BAND_MAX_H, widen_max=0.0):
    """The game band of a v2 plan (gamelayout): {"band", "pan", "pan_y", "samples", "peak"}, or
    None when no x range holds the band. ranges_fn(gw) gives the x ranges the band may take (left
    / right of the cam); inside them a band edge never slices a banner plate (banners.sliced, the
    v1 guard's rule), the gate's own crop-slice verdict holds and, where it can, no text panel,
    corner overlay or persistent HUD plate is cut; the share steps down from `share` to the
    `min_share` floor while a narrower band is cleaner (a band is at most `max_h` canvas rows).
    The motion scan masks the cam (`exclude`) and every HUD plate, panel and overlay; the band's
    run is the legal one holding the most action, its path follows the action
    (gamelayout.plan_path) and is pinned to the event's peak when the smooth path leaves it out.
    v3, for a band narrower than v2's:
      - when no x of any share keeps every banner whole, the banner rows on screen in under
        V2_SCENE_TEXT_FRAC of the scan frames are scene text (a 50 % band cannot hold 2620's
        810 px wall sign, whose union box the camera widens): the band may cut those where the
        gate's own crop-slice verdict passes ("scene_text" in the band);
      - when still no x passes, the V2_WIDEN_SHARES up to `widen_max` are tried and the narrowest
        clean one is taken ("widened_from" in the band: 2670's subtitle row needs ~0.6), rather
        than rejecting a clip v2 rendered."""
    import cv2
    sc = _scan(src, a, (sw, sh))
    hard = [banners.plate(g["box"], sw, sh) for g in sc["graphics"] if g["banner"]]
    scene = [banners.plate(g["box"], sw, sh) for g in sc["graphics"]
             if g["banner"] and float(g.get("frac", 1.0)) < V2_SCENE_TEXT_FRAC]
    overlay = [b for b in hard if b not in scene]
    panels = [list(b) for b in sc.get("panels") or []]
    ovs = [list(b) for b in _overlays(src, a, (sw, sh))]
    soft = panels + ovs
    from clipengine import quality_gate as qg
    words = _ocr(src, a, (sw, sh))
    # the HUD plates no crop edge of a HUD niche may cross (keep_plates: the minimap's
    # street label in 3172), the persistent ones, weighted by how long each is up
    fr = {tuple(pl["box"]): pl["frac"] for pl in keep_plates(
        sc, words, sw, sh, pad=LADDER_TEXT_PAD, min_frac=cropcheck.PROBE_MIN_FRAC)}
    plates = [list(b) for b in fr]
    clean = (0.0, V2_SLICE_MARGIN * qg.SLICE_SHARE_MAX)
    passing = (qg.SLICE_CROP_SHARE_MAX, qg.SLICE_SHARE_MAX)

    def legal(gw, ch, ranges):
        """(rank, plate cost, runs): the band's x runs under the first rule that leaves any (a
        lower rank is cleaner), then those cutting the fewest plates."""
        verdicts = {}

        def gate(x):
            # the gate's own crop-slice verdict (quality_gate.crop_slices: text a window slices
            # or shows running into the frame edge) of the band held at x: 2620's HUD runs into
            # the right frame edge, so a band reaching that edge shows it cut (2.85 > 1.00)
            if x not in verdicts:
                verdicts[x] = qg.crop_slice_verdict(qg.crop_slices(
                    sc, [{"t0": 0.0, "t1": float(a.dur), "x": x, "y": 0, "w": gw, "h": ch}],
                    a.dur, words=words))
            return verdicts[x]

        def ok(boxes, share=None, whole=()):
            def f(x):
                if any(banners.sliced(b, x, gw, sw) for b in boxes):
                    return False
                # a corner overlay the band cuts at all gets painted out (cover_overlays' rule)
                if any(cropcheck.cuts([x, 0, x + gw, ch], o) for o in whole):
                    return False
                if share is None:
                    return True
                v = gate(x)
                return v["passed"] and v["crop_share"] <= share[0] and \
                    v["sliced_share"] <= share[1]
            return f

        rules = ((hard + panels, clean, ovs), (hard, clean, ovs), (hard, clean),
                 (hard, passing, ovs), (hard, passing), (hard,),
                 ([g["box"] for g in sc["graphics"] if g["banner"]],))
        if scene:          # v3: the scene text left to the gate's crop-slice verdict
            rules += ((overlay, clean, ovs), (overlay, clean), (overlay, passing, ovs),
                      (overlay, passing))
        for rank, rule in enumerate(rules):
            runs = [r for lo, hi in ranges for r in gamelayout.legal_runs(lo, hi, ok(*rule))]
            if runs:
                break
        else:
            return len(rules), 0.0, []

        def cost(x):
            return round(sum(f for b, f in fr.items() if cropcheck.cuts([x, 0, x + gw, ch], b)), 3)
        best = min(cost(x) for lo, hi in runs for x in range(lo, hi + 1, 2))
        return rank, best, [r for lo, hi in runs
                            for r in gamelayout.legal_runs(lo, hi, lambda x: cost(x) <= best)]

    # the configured share first, narrower down to the floor when that keeps the HUD whole
    # (3172: at 68 % the wall sign or the minimap is cut at every x, at 65 % neither is)
    best = None
    k = max(float(share), float(min_share))
    while True:
        geo = gamelayout.band_geometry(sw, sh, k, max_h=max_h, min_share=min_share)
        ranges = ranges_fn(geo["gw"])
        if ranges:
            rank, cost, runs = legal(geo["gw"], geo["ch"], ranges)
            key = (rank, cost, -geo["gw"])
            if best is None or key < best[0]:
                best = (key, geo, runs, ranges)
            if rank == 0 and cost == 0:
                break
        if geo["gw"] / float(sw) <= min_share + 0.005:
            break
        k = max(float(min_share), k - V2_SHARE_STEP)
    if best is not None and not (best[2] and (best[0][0] < V2_GATE_RULES or
                                              best[0][0] >= V2_HARD_RULES)):
        for k in (w for w in V2_WIDEN_SHARES if share < w <= widen_max + 1e-9):
            geo = gamelayout.band_geometry(sw, sh, k, max_h=max_h, min_share=min_share)
            ranges = ranges_fn(geo["gw"])
            if not ranges:
                continue
            rank, cost, runs = legal(geo["gw"], geo["ch"], ranges)
            if runs and (rank < V2_GATE_RULES or rank >= V2_HARD_RULES):
                log(f"game band: no {share:.0%} band passes the gate's crop-slice verdict; "
                    f"widened to {geo['share']:.0%}")
                best = ((rank, cost, -geo["gw"]), dict(geo, widened_from=share), runs, ranges)
                break
    if best is None:
        return None
    key, geo, runs, ranges = best
    gw, ch, gh = geo["gw"], geo["ch"], geo["gh"]
    sliced = not runs
    cut_scene = [list(b) for b in scene] if runs and key[0] >= V2_HARD_RULES else []
    if cut_scene:
        log(f"game band: no {geo['share']:.0%} band keeps the scene text {cut_scene} whole; the "
            f"band may cut it where the gate's crop-slice verdict passes")
    if sliced:
        runs = list(ranges)                    # the banner guard rejects it, as it did v1's
    mask = list(exclude) + [banners.plate(g["box"], sw, sh) for g in sc["graphics"]] + soft + \
        plates
    frames = [(t, cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)) for t, fr in reframe.iter_frames(
        str(src), a.start, a.dur, fps=gamelayout.TRACK_FPS, width=gamelayout.TRACK_W,
        src_size=(sw, sh))]
    samples = gamelayout.motion_samples(frames, sw, sh, exclude=mask)
    live = [s for s in samples if s["cx"] is not None]
    lo, hi = max(runs, key=lambda r: (sum(s["e"] for s in live if r[0] <= s["cx"] < r[1] + gw),
                                      r[1] - r[0]))
    pan, _xs = gamelayout.plan_path(samples, gw, lo, hi, sw, a.dur)
    peak = gamelayout.peak_sample(samples)
    if peak and lo <= peak["cx"] < hi + gw and not \
            gamelayout.x_at(pan, peak["t"]) <= peak["cx"] < gamelayout.x_at(pan, peak["t"]) + gw:
        pan, _xs = gamelayout.plan_path(samples, gw, lo, hi, sw, a.dur,
                                        pin=(peak["t"], peak["cx"]))
    pan_y, y0 = None, 0
    if ch < sh:
        pan_y, _ys = gamelayout.plan_path(samples, gw, 0, sh - ch, sh, a.dur, axis="cy", ch=ch)
        y0 = int(pan_y[0][1])
    peak_rep = None
    if peak:
        x = gamelayout.x_at(pan, peak["t"])
        peak_rep = {"t": peak["t"], "cx": peak["cx"], "cy": peak["cy"],
                    "inside": bool(x <= peak["cx"] < x + gw)}
    xs = sorted(x for _t, x in pan)
    return {"band": {"w": gw, "h": ch, "out_h": gh, "share": geo["share"], "y": y0,
                     "lo": int(lo), "hi": int(hi), "runs": [list(r) for r in runs],
                     "sliced": sliced, "scene_text": cut_scene,
                     "widened_from": geo.get("widened_from")},
            "x_static": int(xs[len(xs) // 2]), "pan": pan, "pan_y": pan_y,
            "samples": [[s["t"], s["cx"], s["cy"], s["e"]] for s in samples],
            "peak": peak_rep, "inside_share": gamelayout.inside_share(samples, pan, gw)}


def _v2_cfg(gscfg):
    """render.gamestack's band / card knobs (gamelayout defaults): game_share, cam_max_area (never
    above INSET_HARD_CAP), game_only_share and header_h (the hook header, <= HEADER_MAX_H)."""
    return {"share": float(gscfg.get("game_share", gamelayout.GAME_SHARE)),
            "area": min(float(gscfg.get("cam_max_area", gamelayout.INSET_MAX_AREA)),
                        gamelayout.INSET_HARD_CAP),
            "game_only_share": float(gscfg.get("game_only_share", gamelayout.GAME_ONLY_SHARE)),
            "header_h": int(gamelayout.clamp(int(gscfg.get("header_h", gamelayout.HEADER_H)), 0,
                                             gamelayout.HEADER_MAX_H))}


def _v3_caption_y(src, a, size, band, lay):
    """The v3 captions' centre line: on the band's lower rows (canvas_layout), moved off the band's
    banner rows (a game's subtitle line, a ticker) where they would sit under the captions."""
    sw, sh = size
    plates = [banners.plate(g["box"], sw, sh) for g in _scan(src, a, size)["graphics"]
              if g["banner"]]
    rows = gamelayout.band_text_rows(plates, lay["band"][0], band["out_h"], band["h"],
                                     crop_y=band.get("y", 0),
                                     x_range=(band["lo"], band["hi"] + band["w"]))
    y = gamelayout.caption_y_clear(lay["cap_y"], rows, lay["band"], lay["card"])
    if y != lay["cap_y"]:
        log(f"captions: y {lay['cap_y']} -> {y}, off the band's text rows {rows}")
    return y


def _plan_gamestack_v2(src, a, cfg, gscfg, size):
    """Gamestack v2 (render.gamestack.v2): the streamer's cam as a small card under
    a zoomed-out game band that follows the action (gamelayout). The cam: the v1 detection (the
    channel profile, the busy rule), then the calm rule when the busy growth ran into a dark or
    calm game (2670 / 2500: the box grew to the whole frame, so the clip fell back to the track
    crop on the streamer's face), then the facecam's own box when the planned one reaches past it
    (bound_cam). A box inside the scene (a face in an IRL shot, not an inset at a frame edge) is
    no cam: the caller falls back as before. None -> fallback.
    For gameplay (game_band_only) a box inside the scene whose overlay borders are read round its
    face is a cam burned in mid-frame: the card is cut from where it is and the band takes an x
    clear of it, else shows it painted out (cam_mask), else the clip is game only (no card);
    RenderError when no clean game band is left (_mid_cam_band)."""
    src_w, src_h = size
    channel = getattr(a, "channel", None) or None
    src_duration = float(getattr(a, "src_duration", 0) or 0)
    if not src_duration:
        try:
            src_duration = float(probe(str(src))["duration"])
        except Exception:
            src_duration = 0.0
    cache_dir = getattr(a, "facecam_cache", None) or None
    box_norm, raw_box_norm, from_profile, face_frac = facecam.get_or_detect(
        channel, str(src), a.start, a.dur, source_duration=src_duration, log=log,
        cache_dir=cache_dir)
    how = "profile" if from_profile else "busy"
    if box_norm is None or face_frac < facecam.FACE_ACCEPT:
        r = facecam.detect(str(src), a.start, a.dur, mode="auto")
        if r is None or r["face_frac"] < facecam.FACE_ACCEPT:
            log("gamestack: no cam inset (busy and calm rules); falling back")
            return None
        box_norm, raw_box_norm, face_frac, how = r["box"], r["raw_box"], r["face_frac"], r["mode"]
        log(f"gamestack: cam inset by the calm rule ({how}, face-in-box {face_frac:.0%})")

    def px(b):
        return [_even(b[0] * src_w), _even(b[1] * src_h), _even(b[2] * src_w), _even(b[3] * src_h)]

    cam_box = px(box_norm)
    # a box inside the scene (no frame edge) is a person in an IRL shot, unless the clip is
    # gameplay and the box is a cam overlay burned in mid-frame (its borders read round the face
    # below): then the card is cut from where the cam is
    mid = not gamelayout.is_inset(cam_box, src_w, src_h)
    if mid and not game_band_only(a):
        log(f"gamestack: cam box {cam_box} sits inside the scene (no frame edge): no inset; "
            f"falling back")
        return None
    import numpy as np
    face, face_med_norm, rect = None, None, None
    frames = facecam.sample_native(str(src), a.start, a.dur, n=15)

    def cam_face(box):
        fs = [f for f in (facecam._collect_faces(frames)[0] if frames else [])
              if box[0] <= f[0] <= box[2] and box[1] <= f[1] <= box[3]]
        return tuple(float(np.median([f[k] for f in fs])) for k in range(3)) if fs else None

    face = cam_face(cam_box)
    snapped = False
    if face:
        # the overlay's own borders toward the frame centre (a calm or grown box runs past them
        # into the game, a trimmed one stops short); a mid-frame cam has no edge side: cut to all
        # four borders
        rect = gamelayout.overlay_rect(frames, face)
        if rect:
            nb = gamelayout.inside_rect(cam_box, rect) if mid else \
                gamelayout.snap_to_rect(cam_box, rect, face, src_w, src_h)
            if nb != cam_box:
                log(f"gamestack: cam box {cam_box} -> {nb} on the overlay's borders {rect}")
                cam_box, snapped = nb, True
    if mid and not rect:
        log(f"gamestack: box {cam_box} inside the scene shows no overlay borders round a face: "
            f"no cam; falling back")
        return None
    nb, why = bound_cam(cam_box, a, src, (src_w, src_h))
    if why:
        log(f"gamestack: {why}")
    if nb is None:
        log("gamestack: falling back")
        return None
    gb = nb if (nb != cam_box or snapped) else px(raw_box_norm)
    if snapped or nb != cam_box:
        face = cam_face(nb) or face
    cam_box = nb
    if mid:
        gb = gamelayout.union(rect, cam_box)     # the overlay whole: its borders, the cam's box
    guard = int(round(GAMESTACK_CAM_GUARD_FRAC * src_w))
    guard_box = [max(0, gb[0] - guard), max(0, gb[1] - guard), min(src_w, gb[2] + guard),
                 min(src_h, gb[3] + guard)]
    if face:
        face_med_norm = [face[0] / src_w, face[1] / src_h, face[2] / src_h]
    knobs = _v2_cfg(gscfg)
    share, band_y = knobs["share"], knobs["header_h"]
    cover = mask_box = None
    if mid:
        mask_box = gamelayout.pad_box(gb, MID_CAM_MASK_PAD, src_w, src_h)
        b, cover, why = _mid_cam_band(src, a, src_w, src_h, share, guard_box, mask_box,
                                      widen_max=V2_WIDEN_MAX)
        if b is None:
            return _mid_cam_game_only(src, a, gscfg, size, guard_box, mask_box, why)
    else:
        b = _v2_band(src, a, src_w, src_h, share,
                     lambda gw: gamelayout.cam_free_ranges(src_w, gw, guard_box), [guard_box],
                     widen_max=V2_WIDEN_MAX)
    if b is None:
        log(f"gamestack: no {share:.0%} band clear of the cam box {guard_box}; falling back")
        return None
    # v3: the card as tall as the rows under the band, so it sits right against
    # the game and never covers it (the brief's 120 px overlap is never needed at a >= 48 % band)
    card = gamelayout.card_size(gamelayout.card_room(b["band"]["out_h"], band_y),
                                max_area=knobs["area"])
    if card is None and mid:
        return _mid_cam_game_only(src, a, gscfg, size, guard_box, mask_box,
                                  f"no room for the cam card under the {b['band']['out_h']} px "
                                  f"band")
    if card is None:
        log(f"gamestack: no room for the cam card under the {b['band']['out_h']} px band; "
            f"falling back")
        return None
    kw, kh = card
    cw, ch, cx, cy = gamelayout.inset_crop(cam_box, face, kw, kh)
    side = "left" if cam_box[0] + cam_box[2] < src_w else "right"
    lay = gamelayout.canvas_layout(b["band"]["out_h"], (kw, kh), side, band_y=band_y)
    lay["cap_y"] = _v3_caption_y(src, a, size, b["band"], lay)
    shares = gamelayout.canvas_shares(b["band"]["out_h"], lay["card"], band_y)
    log(f"gamestack v3: band {b['band']['w']}x{b['band']['h']} ({b['band']['share']:.0%} of the "
        f"width) x {b['band']['lo']}..{b['band']['hi']} at canvas rows {lay['band']} "
        f"({shares['band']:.0%}), {len(b['pan'])} knot(s), action inside {b['inside_share']}; "
        f"cam card {kw}x{kh} {side} ({kw * kh / float(OUT_W * OUT_H):.0%}); "
        f"blur {shares['blur']:.0%}")
    if mid:
        log(f"gamestack: mid-frame cam {mask_box}: " +
            (f"painted out of the band ({cover})" if cover else "the band clear of it"))
    plan = {"layout": "gamestack", "v2": True, "src_w": src_w, "src_h": src_h,
            "top_h": b["band"]["out_h"], "game": [b["band"]["w"], b["x_static"], False],
            "band": b["band"], "pan": b["pan"], "pan_y": b["pan_y"],
            "band_y": band_y, "header": lay["header"],
            "cam": [cw, ch, cx, cy, False], "card": lay["card"], "card_side": side,
            "cap_y": lay["cap_y"], "fg_h": lay["cap_y"] - 50,
            "cam_box": cam_box, "cam_box_norm": box_norm, "cam_how": how, "cam_rect": rect,
            "raw_cam_box": guard_box, "raw_cam_box_norm": raw_box_norm,
            "from_profile": bool(from_profile), "face_frac": float(face_frac),
            "face_median": face_med_norm, "channel": channel,
            "action": {"fps": gamelayout.TRACK_FPS, "samples": b["samples"], "peak": b["peak"],
                       "inside_share": b["inside_share"]}}
    if mid:
        plan.update(_mid_cam_keys(mask_box, cover, b))
    return plan


def _plan_game_only_v2(src, a, gscfg, size, cam=None):
    """Game only v3: no face, so no card; the band following the action takes
    render.gamestack.game_only_share (0.42: 1446 rows) under the hook header, the middle of the
    canvas (blurred fill <= 25 %). `cam` = (guard, mask) of a gameplay cam burned in mid-frame
    that no gamestack band could take: the band clear of it or with it painted out
    (_mid_cam_band); None when neither keeps the banners whole."""
    sw, sh = size
    knobs = _v2_cfg(gscfg)
    band_y = knobs["header_h"]
    kw = dict(min_share=min(gamelayout.GAME_ONLY_MIN_SHARE, knobs["game_only_share"]),
              max_h=gamelayout.GAME_ONLY_BAND_MAX_H, widen_max=V2_WIDEN_MAX_GAME_ONLY)
    cover = None
    if cam:
        b, cover, why = _mid_cam_band(src, a, sw, sh, knobs["game_only_share"], cam[0], cam[1],
                                      **kw)
        if b is None:
            log(f"game only: {why}")
    else:
        b = _v2_band(src, a, sw, sh, knobs["game_only_share"], lambda gw: [(0, sw - gw)], **kw)
    if b is None:
        return None
    lay = gamelayout.canvas_layout(b["band"]["out_h"], None, band_y=band_y)
    lay["cap_y"] = _v3_caption_y(src, a, size, b["band"], lay)
    blur = gamelayout.fill_frac(b["band"]["out_h"], None, band_y)
    log(f"game only v3: band {b['band']['w']}x{b['band']['h']} ({b['band']['share']:.0%} of the "
        f"width) at canvas rows {lay['band']}, {len(b['pan'])} knot(s), action inside "
        f"{b['inside_share']}; no card; blur {blur:.0%}")
    plan = {"layout": "game_only", "v2": True, "src_w": sw, "src_h": sh,
            "top_h": b["band"]["out_h"], "game": [b["band"]["w"], b["x_static"], False],
            "band": b["band"], "pan": b["pan"], "pan_y": b["pan_y"], "card": None,
            "band_y": band_y, "header": lay["header"],
            "card_side": None, "cap_y": lay["cap_y"], "fg_h": lay["cap_y"] - 50,
            "action": {"fps": gamelayout.TRACK_FPS, "samples": b["samples"], "peak": b["peak"],
                       "inside_share": b["inside_share"]}}
    if cam:
        plan.update(_mid_cam_keys(cam[1], cover, b))
    return plan


MID_CAM_MASK_PAD = 12      # src px the painted-out mid-frame cam reaches past its overlay
                           # (the border line and its shadow)
MID_CAM_MASK_MAX = 0.35    # a band window the cam would paint more of is no clean game band (a
                           # 640x360 cam is ~31 % of a 0.50 band's 960x1080 window with its pad)


def _mid_cam_band(src, a, sw, sh, share, guard, mask, **kw):
    """The game band of gameplay whose cam overlay is burned in mid-frame
    (`guard`: the cam with the v2 guard margin; `mask`: the cam to paint out). (band, None, None)
    at an x clear of the cam's columns; else (band, cover, None) anywhere with the cam painted out
    of it (cover: the part of `mask` its windows show, at most MID_CAM_MASK_MAX of a window); else
    (None, None, why). The motion scan leaves the cam out either way. `kw` -> _v2_band."""
    b = _v2_band(src, a, sw, sh, share, lambda gw: gamelayout.cam_free_ranges(sw, gw, guard),
                 [guard], **kw)
    if b is not None and not b["band"]["sliced"]:
        return b, None, None
    b = _v2_band(src, a, sw, sh, share, lambda gw: [(0, sw - gw)], [guard], **kw)
    if b is None or b["band"]["sliced"]:
        return None, None, f"every {share:.0%} band x slices a banner (the mid-frame cam {mask})"
    w, h = b["band"]["w"], b["band"]["h"]
    wins = [[x, y, x + w, y + h] for _t0, _t1, x, y in v2_views(b, a.dur)]
    part = gamelayout.shown_part(mask, wins)
    if part is None:
        return b, None, None                   # the band's path leaves the cam out after all
    share_shown = gamelayout.cover_share(part, wins)
    if share_shown > MID_CAM_MASK_MAX:
        return None, None, (f"the mid-frame cam {mask} would paint out {share_shown:.0%} of the "
                            f"band (cap {MID_CAM_MASK_MAX:.0%})")
    return dict(b, cover_share=share_shown), part, None


def _mid_cam_keys(mask, cover, b):
    """The plan keys of a mid-frame cam: cam_mask = the part of it painted out of the
    band (None: the band is clear of it), cam_mask_share = its largest share of a band window."""
    return {"cam_mid": list(mask), "cam_mask": list(cover) if cover else None,
            "cam_mask_share": b.get("cover_share", 0.0) if cover else 0.0}


def _mid_cam_game_only(src, a, gscfg, size, guard, mask, why):
    """Gameplay whose mid-frame cam no gamestack band takes (`why`): game only (no card),
    its band clear of the cam or with it painted out; RenderError (the clip is dropped) when that
    has no clean band either: never a face crop."""
    log(f"gamestack: {why}; game only (no card)")
    g = _plan_game_only_v2(src, a, gscfg, size, cam=(guard, mask))
    if g is None:
        raise RenderError(f"gameplay: no clean game band round the mid-frame cam {list(mask)}: "
                          f"{why}")
    return g


def v2_views(plan, dur, step=0.5):
    """[(t0, t1, x, y)] the v2 band's source window along its path: one per held knot span, a
    ramp cut into `step` s pieces (x at each piece's middle)."""
    pan, pan_y = plan["pan"], plan.get("pan_y")
    cuts = sorted({0.0, float(dur)} | {float(t) for t, _x in pan + (pan_y or [])
                                        if 0.0 < float(t) < float(dur)})
    out = []
    for t0, t1 in zip(cuts, cuts[1:]):
        n = 1 if gamelayout.x_at(pan, t0) == gamelayout.x_at(pan, t1) and (
            not pan_y or gamelayout.x_at(pan_y, t0) == gamelayout.x_at(pan_y, t1)) \
            else max(1, int(math.ceil((t1 - t0) / step)))
        for i in range(n):
            a_, b_ = t0 + (t1 - t0) * i / n, t0 + (t1 - t0) * (i + 1) / n
            m = (a_ + b_) / 2.0
            y = gamelayout.x_at(pan_y, m) if pan_y else int(plan["band"].get("y", 0))
            out.append((a_, b_, gamelayout.x_at(pan, m), y))
    return out


def _plan_reaction(src, a):
    """Build a reaction plan (content video on top over blur, cam on bottom) or None.

    Steps:
      1. Resolve or detect the streamer's cam box (calm-mode detection; larger max face —
         reaction cams often run bigger than gaming cams). Profile keyed by channel:reaction.
      2. Verify face_frac against the box; below FACE_ACCEPT -> track fallback.
      3. Detect the content rectangle on the clip (`facecam.detect_content` with the raw
         cam guard box excluded). No content -> caller falls back to `track`.
      4. Plan: top = content rect fit over blurred fill; bottom = cam_crop on trimmed box.
    """
    cfg = _config()
    rcfg = (cfg.get("render") or {}).get("reaction") or {}
    top_h = int(rcfg.get("top_h", REACTION_TOP_H))
    if top_h <= 0 or top_h >= OUT_H:
        top_h = REACTION_TOP_H
    content_fit = str(rcfg.get("content_fit", REACTION_CONTENT_FIT)).lower()
    if content_fit not in ("fit", "cover"):
        content_fit = "fit"
    r = _resolve_cam(src, a, "reaction", top_h,
                     max_face_h=facecam.MAX_FACE_H_CALM, mode="auto")
    if r is None:
        return None
    src_w, src_h = r["src_w"], r["src_h"]
    frames = r.get("clip_frames") or facecam.sample_native(str(src), a.start, a.dur, n=15)
    # Re-run the streamer-face cluster picker at clip time to get any secondary cam
    # insets currently on screen (kai_speed R2: two PIP corner cams). Their bboxes are
    # masked out of the content-change map so the largest CC doesn't lock onto them.
    exclude_boxes = []
    if frames:
        _, exclude_boxes = facecam._pick_streamer_face(frames,
                                                       max_face_h=facecam.MAX_FACE_H_CALM)
    content_box = facecam.detect_content(frames, cam_guard_box_px=r["guard_box"],
                                          exclude_boxes=exclude_boxes)
    if content_box is None:
        log("reaction: no confident content region; caller falls back to track")
        return None
    # Safety net: even if a caller / monkeypatch returns a content rect that intersects the
    # guard, carve out the largest-area sub-rectangle that clears the guarded cam box. When
    # no sub-rect fits (cam sits central + carve loses >50%), fall back (return None).
    gx0, gy0, gx1, gy1 = r["guard_box"]
    cx0, cy0, cx1, cy1 = content_box
    if min(cx1, gx1) > max(cx0, gx0) and min(cy1, gy1) > max(cy0, gy0):
        carved = facecam._largest_rect_clearing_box(cx0, cy0, cx1, cy1,
                                                     [gx0, gy0, gx1, gy1])
        if carved is None:
            log("reaction: content region intersects cam guard on every side; falling back")
            return None
        orig_area = (cx1 - cx0) * (cy1 - cy0)
        carved_area = (carved[2] - carved[0]) * (carved[3] - carved[1])
        if carved_area < 0.5 * orig_area:
            log("reaction: guard carve-out loses more than half the content region; falling back")
            return None
        content_box = list(carved)
    tw = content_box[2] - content_box[0]
    th = content_box[3] - content_box[1]
    tw -= tw % 2; th -= th % 2
    tx = content_box[0]; ty = content_box[1]
    tx -= tx % 2; ty -= ty % 2
    tw = max(2, tw); th = max(2, th)
    cw, ch, cx, cy, c_fit = facecam.cam_crop(src_w, src_h, r["cam_box"], r["face_median"],
                                              bot_h=OUT_H - top_h)
    return {"layout": "reaction", "src_w": src_w, "src_h": src_h, "top_h": top_h,
            "content_fit": content_fit,
            "content": [tw, th, tx, ty], "content_box": content_box,
            "cam": [cw, ch, cx, cy, c_fit],
            "cam_box": r["cam_box"], "cam_box_norm": r["box_norm"],
            "raw_cam_box": r["guard_box"], "raw_cam_box_norm": r["raw_box_norm"],
            "from_profile": r["from_profile"], "face_frac": r["face_frac"],
            "face_median": r["face_median"], "channel": r["channel"]}


SCREENCAST_WIDE_MULT = 1.5   # `content_zoom: "wide"` widens the content crop by this
                              # multiple of the cover width, fit to 1080 wide over blur,
                              # clamped to never cross the cam guard. Prevents the cover
                              # crop from clipping every line at the left edge (primeagen
                              # defect in round 1: "ar_ticket>", "ion>" — text ran off).


def _plan_screencast_auto(src, a):
    """Build a screencast plan without manual --top/--bottom.

    Steps:
      1. Resolve or detect the presenter cam (calm-mode, MAX_FACE_H_CALM). Profile key
         channel:screencast. No cam -> caller falls back to `general` (full screen).
      2. Content band = crop over the source clearing the raw cam guard box, sized by
         `render.screencast.content_zoom`:
         - "cover": source-height crop at cover width (fills OUT_W x top_h, no blur).
         - "wide" : ~1.5x cover width, fit to OUT_W wide over a blurred vertical fill.
                    Trades a small blur band for legible text near the crop's edges.
      3. Cam band = trimmed cam via facecam.cam_crop (landscape -> cover, portrait -> fit).
    """
    cfg = _config()
    scfg = (cfg.get("render") or {}).get("screencast") or {}
    top_h = int(scfg.get("top_h", SCREENCAST_TOP_H))
    if top_h <= 0 or top_h >= OUT_H:
        top_h = SCREENCAST_TOP_H
    # --content-zoom (CLI) overrides the config default; else config; else "cover".
    zoom = str(getattr(a, "content_zoom", None) or scfg.get("content_zoom", "cover")).lower()
    if zoom not in ("cover", "wide"):
        zoom = "cover"
    r = _resolve_cam(src, a, "screencast", top_h,
                     max_face_h=facecam.MAX_FACE_H_CALM, mode="auto")
    if r is None:
        return None
    src_w, src_h = r["src_w"], r["src_h"]
    # cover width: source-height crop scaled to fill OUT_W x top_h.
    cw_cover = _even(min(src_w, int(round(src_h * OUT_W / max(1, top_h)))))
    cw_cover = max(2, cw_cover)
    if zoom == "wide":
        cw_content = _even(min(src_w, int(round(SCREENCAST_WIDE_MULT * cw_cover))))
    else:
        cw_content = cw_cover
    cw_content = max(2, cw_content)
    centroid_x = facecam.activity_centroid(str(src), a.start, a.dur,
                                            guard_box_px=r["guard_box"])
    if centroid_x is None:
        centroid_x = src_w / 2.0
    tx = _even(int(round(centroid_x - cw_content / 2.0)))
    tx = max(0, min(src_w - cw_content, tx))
    gx0, _, gx1, _ = r["guard_box"]
    fit_blur = (zoom == "wide")
    if tx < gx1 and tx + cw_content > gx0:
        left_room = gx0
        right_room = src_w - gx1
        if left_room >= cw_content:
            tx = _even(max(0, min(gx0 - cw_content,
                                  int(round(centroid_x - cw_content / 2.0)))))
            tx = _even(max(0, min(gx0 - cw_content, tx)))
        elif right_room >= cw_content:
            tx = _even(max(gx1, min(src_w - cw_content,
                                     int(round(centroid_x - cw_content / 2.0)))))
            tx = _even(max(gx1, min(src_w - cw_content, tx)))
        elif zoom == "wide":
            # widen mode + cam central: shrink the wide crop back until it fits on the
            # bigger side, keeping fit-with-blur active.
            side = max(left_room, right_room)
            if side >= 2:
                cw_content = _even(min(cw_content, side))
                if left_room >= right_room:
                    tx = _even(max(0, gx0 - cw_content))
                else:
                    tx = _even(gx1)
            else:
                # neither side has any room — degrade to game_crop
                gw_avail, gxa, _ = facecam.game_crop(src_w, src_h, r["guard_box"],
                                                      mode="wide", top_h=top_h)
                tx = _even(gxa); cw_content = _even(gw_avail); fit_blur = True
        else:
            # cover mode + cam central: pick the wider side's full width, fit-with-blur.
            gw_avail, gxa, _ = facecam.game_crop(src_w, src_h, r["guard_box"],
                                                  mode="wide", top_h=top_h)
            tx = _even(gxa); cw_content = _even(gw_avail); fit_blur = True
    cw, ch, cx, cy, c_fit = facecam.cam_crop(src_w, src_h, r["cam_box"], r["face_median"],
                                              bot_h=OUT_H - top_h)
    return {"layout": "screencast_auto", "src_w": src_w, "src_h": src_h, "top_h": top_h,
            "content": [cw_content, src_h, tx, 0], "cam": [cw, ch, cx, cy, c_fit],
            "content_zoom": zoom, "content_fit_blur": bool(fit_blur),
            "cam_box": r["cam_box"], "cam_box_norm": r["box_norm"],
            "raw_cam_box": r["guard_box"], "raw_cam_box_norm": r["raw_box_norm"],
            "from_profile": r["from_profile"], "face_frac": r["face_frac"],
            "face_median": r["face_median"], "channel": r["channel"],
            "activity_x": float(centroid_x),
            "cutout": bool(r.get("cutout", False))}


def reaction_vf(plan):
    """Two-band reaction: content (top, top_h) fit over blurred fill + cam (bot). The
    content region is fit to 1080 wide over a blurred cover fill of itself, so the whole
    reacted video stays visible (no cropping of the video). `content_fit: cover` swaps
    that for a hard cover crop."""
    top_h = int(plan["top_h"])
    bot_h = OUT_H - top_h
    tw, th, tx, ty = plan["content"]
    cw, ch, cx, cy, c_fit = plan["cam"]
    fit_mode = plan.get("content_fit", REACTION_CONTENT_FIT)
    parts = []
    top_tags = ["tbg", "tfg"] if fit_mode == "fit" else ["tc"]
    cam_tags = ["cbg", "cfg"] if c_fit else ["cc"]
    tags = top_tags + cam_tags
    parts.append("[0:v]split=" + str(len(tags)) + "".join(f"[{t}]" for t in tags))
    if fit_mode == "fit":
        # a screen stack sits on the cam band, leaving the hook a clean blurred strip on top
        oy = "H-h" if plan.get("content_valign") == "bottom" else "(H-h)/2"
        parts.append(f"[tbg]crop={tw}:{th}:{tx}:{ty},"
                     f"scale={OUT_W}:{top_h}:force_original_aspect_ratio=increase,"
                     f"crop={OUT_W}:{top_h},gblur=sigma=22[topbg]")
        parts.append(f"[tfg]crop={tw}:{th}:{tx}:{ty},"
                     f"scale={OUT_W}:{top_h}:force_original_aspect_ratio=decrease[topfg]")
        parts.append(f"[topbg][topfg]overlay=(W-w)/2:{oy}[top]")
    else:
        parts.append(f"[tc]crop={tw}:{th}:{tx}:{ty},"
                     f"scale={OUT_W}:{top_h}:force_original_aspect_ratio=increase,"
                     f"crop={OUT_W}:{top_h}[top]")
    if c_fit:
        parts.append(f"[cbg]crop={cw}:{ch}:{cx}:{cy},"
                     f"scale={OUT_W}:{bot_h}:force_original_aspect_ratio=increase,"
                     f"crop={OUT_W}:{bot_h},gblur=sigma=22[botbg]")
        parts.append(f"[cfg]crop={cw}:{ch}:{cx}:{cy},scale=-2:{bot_h}[botfg]")
        parts.append("[botbg][botfg]overlay=(W-w)/2:0[bot]")
    else:
        parts.append(f"[cc]crop={cw}:{ch}:{cx}:{cy},"
                     f"scale={OUT_W}:{bot_h}:force_original_aspect_ratio=increase,"
                     f"crop={OUT_W}:{bot_h}[bot]")
    parts.append("[top][bot]vstack=inputs=2,setsar=1[v0]")
    return ";".join(parts)


def screencast_auto_vf(plan):
    """Two-band auto screencast. Top band = content crop (cover-scaled OR fit-to-width
    over a blurred fill when `content_fit_blur` is true / `content_zoom` == "wide").
    Bottom band = cam via `facecam.cam_crop` (landscape covers, portrait fits-with-blur).
    """
    top_h = int(plan["top_h"])
    bot_h = OUT_H - top_h
    tw, th, tx, ty = plan["content"]
    cw, ch, cx, cy, c_fit = plan["cam"]
    fit_blur = bool(plan.get("content_fit_blur", False))
    parts = []
    top_tags = ["tbg", "tfg"] if fit_blur else ["gc"]
    cam_tags = ["cbg", "cfg"] if c_fit else ["cc"]
    tags = top_tags + cam_tags
    parts.append("[0:v]split=" + str(len(tags)) + "".join(f"[{t}]" for t in tags))
    if fit_blur:
        parts.append(f"[tbg]crop={tw}:{th}:{tx}:{ty},"
                     f"scale={OUT_W}:{top_h}:force_original_aspect_ratio=increase,"
                     f"crop={OUT_W}:{top_h},gblur=sigma=22[topbg]")
        parts.append(f"[tfg]crop={tw}:{th}:{tx}:{ty},"
                     f"scale={OUT_W}:-2[topfg]")
        parts.append("[topbg][topfg]overlay=(W-w)/2:(H-h)/2[top]")
    else:
        parts.append(f"[gc]crop={tw}:{th}:{tx}:{ty},"
                     f"scale={OUT_W}:{top_h}:force_original_aspect_ratio=increase,"
                     f"crop={OUT_W}:{top_h}[top]")
    if c_fit:
        parts.append(f"[cbg]crop={cw}:{ch}:{cx}:{cy},"
                     f"scale={OUT_W}:{bot_h}:force_original_aspect_ratio=increase,"
                     f"crop={OUT_W}:{bot_h},gblur=sigma=22[botbg]")
        parts.append(f"[cfg]crop={cw}:{ch}:{cx}:{cy},scale=-2:{bot_h}[botfg]")
        parts.append("[botbg][botfg]overlay=(W-w)/2:0[bot]")
    else:
        parts.append(f"[cc]crop={cw}:{ch}:{cx}:{cy},"
                     f"scale={OUT_W}:{bot_h}:force_original_aspect_ratio=increase,"
                     f"crop={OUT_W}:{bot_h}[bot]")
    parts.append("[top][bot]vstack=inputs=2,setsar=1[v0]")
    return ";".join(parts)


def dual_view_vf(plan):
    """HUD-niche dual view (10-21): the hook header (the frame cover-scaled and blurred), then the
    FULL source frame fit to the canvas width (no crop, so no burned-in graphic is sliced), then
    below the seam the zoom panel: one fixed-size source window (the facecam, or the speaking
    player's face / the busiest region per shot) cover-scaled to fill it."""
    hh, ph, seam = int(plan["header_h"]), int(plan["panel_h"]), int(plan["seam_y"])
    bh = OUT_H - seam
    zw, zh = int(plan["zoom"]["w"]), int(plan["zoom"]["h"])
    segs = plan["zoom_segments"]
    xe = reframe.crop_x_expr(segs)
    ye = reframe.crop_x_expr([dict(s, x=s["y"]) for s in segs])
    if plan["zoom"].get("fit") == "inside":
        # a small facecam shown whole over its own blurred fill (dual_view_cam_zoom)
        zoom = (f"[dvz]crop={zw}:{zh}:'{xe}':'{ye}',split=2[zbg][zfg];"
                f"[zbg]scale={OUT_W}:{bh}:force_original_aspect_ratio=increase,"
                f"crop={OUT_W}:{bh},gblur=sigma=22[zbgb];"
                f"[zfg]scale={OUT_W}:{bh}:force_original_aspect_ratio=decrease:flags=lanczos[zfgs];"
                f"[zbgb][zfgs]overlay=(W-w)/2:(H-h)/2,setsar=1[zm]")
    else:
        zoom = (f"[dvz]crop={zw}:{zh}:'{xe}':'{ye}',"
                f"scale={OUT_W}:{bh}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={OUT_W}:{bh},setsar=1[zm]")
    fr = plan.get("frame")         # the frame inside its source bars
    inner = f"crop={fr['w']}:{fr['h']}:{fr['x']}:{fr['y']}," if fr else ""
    return (f"[0:v]split=3[dvh][dvt][dvz];"
            f"[dvh]{inner}scale={OUT_W}:{hh}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{hh},gblur=sigma=22[hdr];"
            f"[dvt]{inner}scale={OUT_W}:{ph}:flags=lanczos,setsar=1[top];"
            f"{zoom};"
            f"[hdr][top][zm]vstack=inputs=3,setsar=1[v0]")


def _plan_game_only(src, a):
    """No-face fallback for a gamestack request: full-source 9:16 centre crop."""
    try:
        src_w, src_h = reframe.probe_size(str(src))
    except Exception as e:
        log(f"gamestack game-only fallback: probe failed ({e})")
        return None
    gscfg = _gamestack_cfg()
    if gscfg.get("v2"):
        try:
            p = _plan_game_only_v2(src, a, gscfg, (src_w, src_h))
        except Exception as e:    # noqa: BLE001 - unplannable: the v1 centre crop
            log(f"game only v2: planning failed ({type(e).__name__}: {e})")
            p = None
        if p is not None:
            return p
    return {"layout": "game_only", "src_w": src_w, "src_h": src_h}


def _even(v):
    v = int(v)
    return v - (v % 2)


def _resolve_cam(src, a, layout, top_h, max_face_h, mode):
    """Shared cam-detection + planning helper for gamestack / reaction / screencast.

    Returns a dict with the resolved boxes, or None (caller falls back).
      {"src_w", "src_h", "top_h", "cam_box", "guard_box", "raw_box_norm", "box_norm",
       "face_frac", "from_profile", "face_median", "channel", "profile_key"}
    """
    try:
        src_w, src_h = reframe.probe_size(str(src))
    except Exception as e:
        log(f"{layout}: probe failed ({e}); falling back")
        return None
    channel = getattr(a, "channel", None) or None
    # gamestack keeps the bare channel key (no migration). Other layouts scope by layout so
    # a streamer who games and reacts doesn't share one box across scenes.
    profile_key = channel if layout == "gamestack" else (f"{channel}:{layout}" if channel else None)
    src_duration = float(getattr(a, "src_duration", 0) or 0)
    if not src_duration:
        try:
            src_duration = float(probe(str(src))["duration"])
        except Exception:
            src_duration = 0.0
    cache_dir = getattr(a, "facecam_cache", None) or None
    box_norm, raw_box_norm, from_profile, face_frac = facecam.get_or_detect(
        channel, str(src), a.start, a.dur, source_duration=src_duration, log=log,
        mode=mode, max_face_h=max_face_h, profile_key=profile_key,
        cache_dir=cache_dir)
    if box_norm is None:
        log(f"{layout}: no cam inset (face_frac=0); falling back")
        return None
    if face_frac < facecam.FACE_ACCEPT:
        log(f"{layout}: face-in-box {face_frac:.0%} < "
            f"{facecam.FACE_ACCEPT:.0%}; falling back")
        return None
    bx0 = int(round(box_norm[0] * src_w)); by0 = int(round(box_norm[1] * src_h))
    bx1 = int(round(box_norm[2] * src_w)); by1 = int(round(box_norm[3] * src_h))
    bx0 -= bx0 % 2; by0 -= by0 % 2
    bx1 = (bx1 // 2) * 2; by1 = (by1 // 2) * 2
    cam_box = [max(0, bx0), max(0, by0), min(src_w, bx1), min(src_h, by1)]
    rx0 = int(round(raw_box_norm[0] * src_w)); rx1 = int(round(raw_box_norm[2] * src_w))
    ry0 = int(round(raw_box_norm[1] * src_h)); ry1 = int(round(raw_box_norm[3] * src_h))
    guard = int(round(GAMESTACK_CAM_GUARD_FRAC * src_w))
    guard_box = [max(0, rx0 - guard), max(0, ry0 - guard),
                 min(src_w, rx1 + guard), min(src_h, ry1 + guard)]
    frames = facecam.sample_native(str(src), a.start, a.dur, n=15)
    face_med_norm = None
    if frames:
        # Use the streamer-face cluster picker (same rule as facecam.detect) so a react
        # stream with faces in the reacted video doesn't drag the cam-band centre off
        # the streamer's cam-inset face. Gamestack streams have a single cam-face
        # cluster and the picker's median matches the previous median-of-all median.
        streamer, _ = facecam._pick_streamer_face(frames, max_face_h=max_face_h)
        if streamer is not None:
            fcx, fcy, fh_px = streamer
            face_med_norm = [float(fcx / src_w), float(fcy / src_h),
                             float(fh_px / src_h)]
    # `detect()` records `cutout=True` when it swapped the grow box for a head-and-
    # shoulders rectangle (chroma-keyed cam). Read it back from the saved profile so
    # downstream reporting sees the flag.
    profiles = facecam.load_profiles(cache_dir=cache_dir)
    cutout = bool((profiles.get(profile_key) or {}).get("cutout", False)) if profile_key else False
    return {"src_w": src_w, "src_h": src_h, "top_h": top_h,
            "cam_box": cam_box, "guard_box": guard_box, "box_norm": box_norm,
            "raw_box_norm": raw_box_norm, "face_frac": float(face_frac),
            "from_profile": bool(from_profile), "face_median": face_med_norm,
            "channel": channel, "profile_key": profile_key, "clip_frames": frames,
            "cutout": cutout}


def hudband_vf(plan):
    """hudband layout: subject-tracked crop on top + per-region HUD overlays packed into a dark
    band below. Every region comes from a fixed per-show HUD profile; presence intervals decide
    when each region draws. No drawbox, no blur — when a region is absent (close-up shot,
    full-screen graphic) the band shows plain band colour in that spot.

    Top-crop rows start below the outs_strip (`y_off` = outs_strip bottom edge, full remaining
    height as `eff_h`) so the outs strip never appears in the top crop.

    plan carries: src_w/src_h, segments (top-crop x by time), cw (top-crop width),
    y_off / eff_h (source rows sampled by the top crop), band_height,
    band_items (each {"name","src":(x,y,w,h),"dst":(x,y,w,h),"intervals":[(a,b),...]})."""
    band_h = plan["band_height"]
    top_h = OUT_H - band_h
    src_h = plan["src_h"]
    y_off = int(plan.get("y_off", 0))
    eff_h = int(plan.get("eff_h", src_h - y_off))
    cw = plan.get("cw") or reframe.hudband_crop_width(plan["src_w"], src_h, top_h, eff_h)
    top_x = reframe.crop_x_expr(plan["segments"])
    items = plan["band_items"]
    n_hud = len(items)
    n_splits = 1 + n_hud
    parts = []
    outs = [f"src{i}" for i in range(n_splits)]
    parts.append(f"[0:v]split={n_splits}" + "".join(f"[{o}]" for o in outs))

    # top crop: rows start at y_off (below the outs strip), full remaining height. cw was
    # picked to keep 1080:top_h aspect against eff_h so nothing squishes.
    parts.append(f"[{outs[0]}]crop={cw}:{eff_h}:'{top_x}':{y_off},"
                 f"scale={OUT_W}:{top_h}:flags=lanczos,setsar=1[top]")

    # dark band background — visible wherever no region overlay is enabled at time t.
    parts.append(f"color=c={HUDBAND_BG}:s={OUT_W}x{band_h}:d=1[band0]")

    band_chain = "band0"
    for i, item in enumerate(items):
        sx, sy, sw, sh = item["src"]
        dx, dy, dw, dh = item["dst"]
        src_idx = 1 + i
        parts.append(f"[{outs[src_idx]}]crop={sw}:{sh}:{sx}:{sy},"
                     f"scale={dw}:{dh}:flags=lanczos[tile{i}]")
        enable = hud.enable_expr(item.get("intervals") or [])
        overlay = f"overlay={dx}:{dy}:enable='{enable}'"
        parts.append(f"[{band_chain}][tile{i}]{overlay}[bnd{i}]")
        band_chain = f"bnd{i}"
    parts.append(f"[{band_chain}]setsar=1[bandf]")

    parts.append(f"[top][bandf]vstack=inputs=2,setsar=1[v0]")
    return ";".join(parts)


def split_vf(plan):
    band = OUT_H // 2
    parts = ["[0:v]split=2[sa][sb]"]
    for pad, out, c in (("sa", "top", plan["split"]["left"]), ("sb", "bot", plan["split"]["right"])):
        parts.append(f"[{pad}]crop={c['w']}:{c['h']}:{c['x']}:{c['y']},"
                     f"scale={OUT_W}:{band}:flags=lanczos,setsar=1[{out}]")
    parts.append("[top][bot]vstack=inputs=2[v0]")
    return ";".join(parts)


def _hud_profile(vertical):
    """Load the HUD profile mapped to `vertical` from the config. None when unmapped
    or the config is unreadable / the named profile is missing."""
    try:
        cfg = config.load()
    except (OSError, ValueError):
        return None
    name = hud.profile_for_vertical(cfg, vertical)
    if not name:
        return None
    try:
        return hud.load_profile(cfg, name)
    except (KeyError, ValueError) as e:
        log(f"hudband: profile {name!r} invalid ({e}); falling back")
        return None


def _broadcast_profile(broadcast):
    """the HUD profile measured on this broadcast (render.hud_profile_by_broadcast),
    or None when the broadcast has none (the vertical's profile is used then)."""
    if not broadcast:
        return None
    try:
        cfg = config.load()
    except (OSError, ValueError):
        return None
    name = hud.profile_for_broadcast(cfg, broadcast)
    if not name:
        return None
    try:
        return hud.load_profile(cfg, name)
    except (KeyError, ValueError) as e:
        log(f"hudband: broadcast profile {name!r} invalid ({e}); falling back")
        return None


def _hud_profile_mismatch(src, a, profile, src_w, src_h):
    """Why `profile` does not fit this window's feed (hud.profile_mismatch against the graphics
    that stay put across its camera shots), or None. An unreadable window keeps the profile."""
    try:
        frames = [fr for _t, fr in reframe.iter_frames(str(src), a.start, a.dur, fps=hud.SAMPLE_HZ,
                                                       width=320, src_size=(src_w, src_h))]
        cuts = reframe._cut_times([(i / hud.SAMPLE_HZ, reframe._hist(f))
                                   for i, f in enumerate(frames)])
        mask = hud.overlay_mask(frames, cuts)
    except Exception as e:        # noqa: BLE001
        log(f"hudband: profile match not verified ({type(e).__name__}: {e})")
        return None
    if mask is None:
        log("hudband: one camera shot in the window; profile match not verified")
    return hud.profile_mismatch(profile, mask)


def _region_presence(src, a, profile, src_w, src_h):
    """Sample the clip window and return {region_name: [(a,b),...]} of present intervals for
    each named region in `profile`. Uses hud.presence_intervals with the module defaults."""
    samples = hud.sample_frames(str(src), a.start, a.dur)
    if not samples:
        return {name: [] for name in profile}
    out = {}
    for name, region in profile.items():
        crops = hud.region_crops(samples, region)
        marks = hud.presence_marks(crops)
        out[name] = hud.presence_intervals(marks)
    return out


HUD_REGION_MIN_ALNUM = 3     # OCR'd letters / digits a band region must hold to be a HUD


def _blank_regions(src, a, profile, names, src_w, src_h):
    """The band regions in `names` whose source rect holds no OCR'd HUD text (fewer than
    HUD_REGION_MIN_ALNUM letters / digits over the OCR'd frames). An empty OCR read (no text, or
    tesseract timed out) falls back to the scan's text lines: a region with no persistent one
    (frac >= cropcheck.PROBE_MIN_FRAC) is blank too (10-18 clip 1316: a
    channel's HUD profile on its webcam podcast, no word on screen, kept the band). [] when
    neither can be read (the band as before)."""
    words = _ocr(src, a, (src_w, src_h))
    graphics = None
    if not words:
        try:
            graphics = [g["box"] for g in _scan(src, a, (src_w, src_h)).get("graphics") or []
                        if float(g.get("frac") or 0.0) >= cropcheck.PROBE_MIN_FRAC]
        except Exception as e:    # noqa: BLE001 - unreadable: the band as before
            log(f"hudband: band regions unreadable ({type(e).__name__}: {e})")
            return []
    blank = []
    for name in sorted(names):
        x, y, w, h = hud.region_pixels(profile[name], src_w, src_h)
        if graphics is not None:
            if not any(x <= (b[0] + b[2]) / 2.0 <= x + w and y <= (b[1] + b[3]) / 2.0 <= y + h
                       for b in graphics):
                blank.append(name)
            continue
        n = sum(cropcheck.alnum(wd[4]) for wd in words
                if x <= (wd[0] + wd[2]) / 2.0 <= x + w and y <= (wd[1] + wd[3]) / 2.0 <= y + h)
        if n < HUD_REGION_MIN_ALNUM:
            blank.append(name)
    return blank


def _plan_hudband(src, a):
    """Build a hudband plan using the per-show HUD profile (per PHASE4h).

    - profile lookup: vertical -> profile name -> normalized region rects (players, pot,
      outs_strip). No detection.
    - presence gating: sample the window every 0.5 s; for each region mark PRESENT samples
      (temporal-median reference, MAD threshold); build merged intervals.
    - band: `players` + `pot` in one row; each region's overlay carries the ffmpeg `enable=`
      built from its intervals. outs_strip is not shown in the band.
    - top crop: y_off = outs_strip bottom edge, full remaining height; cw keeps 1080:top_h.
      x is chosen with hud.safe_crop_x so the crop never SLICES `players` or `pot` (each is
      either fully outside or fully inside; prefer outside when the subject keeps >= 70% of its
      box inside the crop).
    - fallback: if `players` is present in < FALLBACK_PLAYERS_MIN_FRAC of the window, return
      None (caller falls back to `track` and logs it).
    """
    vertical = getattr(a, "vertical", None) or "poker"
    broadcast = getattr(a, "broadcast", None)
    profile = _broadcast_profile(broadcast) or _hud_profile(vertical)
    if not profile:
        return None
    try:
        src_w, src_h = reframe.probe_size(src)
    except Exception as e:
        log(f"hudband: probe failed ({e}); falling back")
        return None
    # a profile measured on another feed slices this one's graphics (10-20 clips
    # 126/127/128): the speaker crop (track, banner-guarded) instead
    why = _hud_profile_mismatch(src, a, profile, src_w, src_h)
    if why:
        log(f"hudband: HUD profile does not fit {broadcast or 'this feed'} ({why}); "
            f"falling back to the speaker crop")
        return None

    intervals = _region_presence(src, a, profile, src_w, src_h)
    total_present = sum(b - x for x, b in intervals.get("players", []))
    frac = total_present / max(0.001, float(a.dur))
    if frac < hud.FALLBACK_PLAYERS_MIN_FRAC:
        log(f"hudband: players present {frac:.1%} of window (< "
            f"{hud.FALLBACK_PLAYERS_MIN_FRAC:.0%}); falling back to track")
        return None

    present_regions = {n for n, iv in intervals.items() if iv and n in ("players", "pot")}
    # a band tile shows a HUD panel only where the OCR reads one: the generic profile
    # on a talking-head podcast (10-18 clip 1323) filled the band with a mic and a hand
    blank = _blank_regions(src, a, profile, present_regions, src_w, src_h)
    if blank:
        log(f"hudband: no HUD text in the profile's {', '.join(sorted(blank))} region(s); "
            f"falling back to the speaker crop")
        return None
    pack = hud.plan_band(profile, src_w, src_h, present_regions=present_regions)
    if not pack or not pack.get("items"):
        return None

    band_h = pack["height"]
    top_h = OUT_H - band_h

    # outs_strip bottom (source pixels) sets the y-off for the tracked top crop. When the strip
    # is missing from the profile OR isn't present at all, y_off = 0.
    y_off = 0
    if "outs_strip" in profile:
        _, sy, _, sh = hud.region_pixels(profile["outs_strip"], src_w, src_h)
        y_off = sy + sh
    bottom = src_h
    if "bottom_strip" in profile:          # a feed's bottom HUD rows (KPC's bar, the Lodge's tiles
                                           # + board + bar) stay out of the top crop
        _, by, _, _ = hud.region_pixels(profile["bottom_strip"], src_w, src_h)
        bottom = min(src_h, by)
    eff_h = max(2, bottom - y_off)
    cw = reframe.hudband_crop_width(src_w, src_h, top_h, effective_src_h=eff_h)

    plan = reframe.analyze(str(src), a.start, a.dur, force="hudband", cw=cw)

    # crop-never-slices: each of players/pot must be fully outside or fully inside every crop
    # window. Prefer outside; when that would drop the subject > 30% out, allow inside.
    region_x_ranges = []
    for name in ("players", "pot"):
        if name not in profile:
            continue
        rx, _, rw, _ = hud.region_pixels(profile[name], src_w, src_h)
        region_x_ranges.append((rx, rx + rw))
    if region_x_ranges:
        for seg in plan.get("segments") or []:
            subject = (int(seg["x"]), int(cw))
            seg["x"] = hud.safe_crop_x(seg["x"], cw, src_w, region_x_ranges,
                                       subject_box=subject)

    # attach per-region intervals onto band items
    for item in pack["items"]:
        item["intervals"] = intervals.get(item["name"], [])

    plan.update(layout="hudband", band_height=band_h, band_items=pack["items"], cw=cw,
                y_off=y_off, eff_h=eff_h, profile=profile,
                region_intervals=intervals, players_present_frac=round(frac, 3))
    return plan


def _words_from_arg(a):
    """Return the SOURCE-time words list from ``a.words`` (or None when not available).

    both ``reframe.analyze`` and downstream ``diarize.diarize_words`` take
    SOURCE-time words — they subtract ``a.start`` themselves when building the
    clip-relative view. Returning CLIP-relative words (as the previous version did)
    produced a double-shift that silently killed the audio-first speaker path for
    every podcast_two_shot clip ending up on the mouth-motion fallback (which picked
    the listener). We now return the raw source-time list filtered to the clip
    window; ``load_words`` is retained for captions where clip-relative IS what we
    want."""
    if not getattr(a, "words", None):
        return None
    try:
        data = json.loads(Path(a.words).read_text())
        if isinstance(data, dict):
            data = data.get("words") or []
        start, end = float(a.start), float(a.start) + float(a.dur)
        return [dict(w) for w in data if start <= float(w["start"]) < end]
    except Exception as e:        # noqa: BLE001
        log(f"load_words source-time read failed ({e}); skipping audio-first speaker path")
        return None


def _plan_chart_top_cam_bottom(src, a):
    """Planner for ``chart_top_cam_bottom``: detect the streamer's face position in
    the source, compute a cam-box crop around it that will zoom the face to >= 25 % of the
    bottom panel (608 px) height, and emit a plan with ``cam_box=[x, y, w, h]``. Returns
    None when no face is detected (caller falls back to broadcast_vf).
    """
    import statistics as _stat
    try:
        src_w, src_h = reframe.probe_size(str(src))
    except Exception as e:
        log(f"chart_top_cam_bottom: probe failed ({e}); fallback")
        return None
    try:
        plan = reframe.analyze(str(src), a.start, a.dur, force="broadcast")
    except Exception as e:
        log(f"chart_top_cam_bottom: reframe.analyze failed ({e}); fallback")
        return None
    # Collect every detected face (hook_faces covers the first few seconds, but we also
    # sample a few frames across the clip for a more robust median when the face is in a
    # fixed corner).
    faces = []
    for s in plan.get("hook_faces") or []:
        for f in s.get("faces") or []:
            faces.append(f)
    if len(faces) < 3:
        try:
            n_steps = 6
            dur = float(a.dur)
            step = max(1.0, dur / n_steps)
            for k in range(n_steps):
                t = min(dur - 0.1, k * step)
                for _t, frame in reframe.iter_frames(str(src), a.start + t, 0.4,
                                                     fps=2, src_size=(src_w, src_h)):
                    for f in reframe.detect_faces(frame):
                        faces.append(f)
                    break
        except Exception:                     # noqa: BLE001
            pass
    if not faces:
        log("chart_top_cam_bottom: no face detected; falling back to broadcast")
        return None
    face_cx = _stat.median(f["cx"] for f in faces)
    face_cy = _stat.median(f["cy"] for f in faces)
    face_h = _stat.median(f["h"] for f in faces)
    top_h = reframe.BROADCAST_TOP_H
    bot_h = OUT_H - top_h
    target_face_frac = 0.30             # aim slightly above the 25 % floor
    face_h_px_src = max(1.0, float(face_h) * src_h)
    # The bottom panel scales a (cam_w, cam_h) crop to (OUT_W, bot_h) with cover scaling.
    # Pick cam_h such that the face's height after scaling = target_face_frac * bot_h.
    # scale_factor = bot_h / cam_h  →  face_scaled_h = face_h_px_src * scale_factor.
    # Solve for cam_h: cam_h = face_h_px_src * bot_h / (target_face_frac * bot_h)
    #                 = face_h_px_src / target_face_frac
    cam_h = int(round(face_h_px_src / float(target_face_frac)))
    cam_h = max(96, min(cam_h, src_h))        # clamp to source height
    cam_h -= cam_h % 2
    cam_w = int(round(cam_h * (OUT_W / float(bot_h))))
    cam_w = max(96, min(cam_w, src_w))
    cam_w -= cam_w % 2
    cx = int(round(float(face_cx) * src_w - cam_w / 2.0))
    cy = int(round(float(face_cy) * src_h - cam_h * 0.42))   # face slightly above centre
    cx = max(0, min(cx, src_w - cam_w))
    cy = max(0, min(cy, src_h - cam_h))
    cx -= cx % 2; cy -= cy % 2
    # detect the chart content bbox on the source so chart_top_cam_bottom_vf
    # can zoom the top panel into the price-action area instead of cover-scaling the full
    # frame. The detector samples the WHOLE window at 1 Hz (not 5 samples), unions per-sample
    # content, requires >= 20 candles, and refuses to render through windows where the
    # source goes black (returns a sentinel dict). When detection fails, chart_bbox stays
    # None and the vf falls back to the full-source cover scale.
    chart_bbox = None
    chart_bbox_fault = None
    try:
        bbox = detect_chart_content_bbox(str(src), a.start, a.dur)
    except Exception as e:                               # noqa: BLE001
        log(f"detect_chart_content_bbox failed ({e})")
        bbox = None
    if isinstance(bbox, dict) and bbox.get("error"):
        # the source window has frames where the chart overlay is gone (fully black
        # pixels for most of the frame). Record the fault on the plan so the pipeline can
        # drop the pick before rendering — the live gate's layout_quality will also catch
        # it after render, but failing upstream saves the render.
        chart_bbox_fault = {
            "reason": bbox.get("error"),
            "black_at_s": bbox.get("black_at_s") or [],
        }
        log(f"chart_top_cam_bottom: chart overlay missing at "
            f"{len(chart_bbox_fault['black_at_s'])} sampled seconds "
            f"({chart_bbox_fault['black_at_s'][:6]}); flagging plan.chart_bbox_fault")
    elif isinstance(bbox, (list, tuple)):
        chart_bbox = list(bbox)
    out = {
        "layout": "chart_top_cam_bottom",
        "canonical_layout": "chart_top_cam_bottom",
        "internal_layout": "chart_top_cam_bottom_v2",
        "src_w": src_w, "src_h": src_h,
        "cam_box": [cx, cy, cam_w, cam_h],
        "chart_bbox": chart_bbox,
        "face_cx": round(float(face_cx), 4),
        "face_cy": round(float(face_cy), 4),
        "face_h_src_frac": round(float(face_h), 4),
        "face_h_bottom_frac_expected": round(float(face_h_px_src) * bot_h / cam_h / bot_h, 3),
        "hook_faces": plan.get("hook_faces") or [],
        # keep track of segments only for diagnostics; the vf doesn't use them
        "segments": plan.get("segments") or [],
        "shots": plan.get("shots") or [],
    }
    if chart_bbox_fault:
        out["chart_bbox_fault"] = chart_bbox_fault
    return out


def single_view_layout(a):
    """The layout to plan for a tech/finance clip (full-bleed single view following
    the speaker): a stacked request (campaign / content_type / vertical mapping) becomes track."""
    lay = a.layout
    if lay in STACKED_LAYOUTS:
        log(f"{a.niche}: {lay} is a stacked layout; single view (track) instead")
        return "track"
    if lay == "auto":
        mapped = _vertical_layouts().get(getattr(a, "vertical", None) or "")
        if mapped in STACKED_LAYOUTS:
            log(f"{a.niche}: vertical {a.vertical} maps to {mapped}; single view (track) instead")
            return "track"
    return lay


def dual_view_fallback(a):
    """a tech / finance clip whose single view sliced the screen
    (the gate's crop_slices / edge_slice, a screen-content eye-check reject, or no clean crop at
    all) is rendered once more as the dual view: the whole frame fit to width on top, the speaker
    below. Only on --dual-view-fallback: the first render of every clip is the single view."""
    return bool(getattr(a, "dual_view_fallback", False)) and \
        getattr(a, "niche", None) in SINGLE_VIEW_NICHES


def enforce_single_view(plan, a):
    """Last check before the encode: a tech/finance clip never renders a stacked layout (the dual
    view fallback aside)."""
    if dual_view_fallback(a) and plan.get("layout") == DUAL_VIEW:
        return
    if plan.get("streamer_split") and getattr(a, "niche", None) in SPLIT_SCREEN_NICHES:
        return                     # render yield 11-09 class 4: a screen with a facecam
    if getattr(a, "niche", None) in SINGLE_VIEW_NICHES and (
            plan.get("manual") or plan.get("layout") in STACKED_LAYOUTS):
        raise RenderError(f"{a.niche} clips render a single view; the planner chose "
                          f"{plan.get('layout')}")


def plan_layout(a, src):
    """Reframe plan dict; always has "layout"."""
    if getattr(a, "niche", None) in SINGLE_VIEW_NICHES:
        a.layout = single_view_layout(a)
        a.top = a.bottom = None              # no manual screencast/split stack either
    # manual screencast (both --top and --bottom set) keeps the existing behaviour
    # byte-for-byte. Manual split is the same. --layout screencast WITHOUT --top/--bottom
    # falls through to the auto-screencast planner below.
    manual = (a.layout == "screencast" and a.top and a.bottom) \
        or (a.layout == "split" and a.top and a.bottom)
    if manual:
        return {"layout": a.layout, "manual": True, "top": a.top, "bottom": a.bottom,
                "split_px": OUT_H // 2 if a.layout == "split" else a.split}
    if a.layout == "general":
        return {"layout": "general"}
    if a.layout == DUAL_VIEW:
        p = plan_dual_view(src, a)
        if p is not None:
            return p
        log("dual view: no room for a zoom panel; planning as --layout auto")
        a.layout = "auto"
    # named campaign layouts that recompose implements by delegating to an existing
    # internal planner, then stamping the plan with the canonical campaign-visible name so
    # the quality gate's layout_match measure can verify "rendered layout == configured
    # layout". The video graph for each is reused (chart_top_cam_bottom <- broadcast_vf,
    # product_full_bleed <- track_vf) in :func:`video_graph`.
    if a.layout == "chart_top_cam_bottom":
        # use the dedicated chart/cam planner — top panel = chart cover-scaled (no
        # black bars), bottom panel = cam zoomed so the face takes >= 25 % of panel height.
        p = _plan_chart_top_cam_bottom(src, a)
        if p is not None:
            return p
        try:
            plan = reframe.analyze(str(src), a.start, a.dur, force="broadcast")
        except Exception as e:
            log(f"chart_top_cam_bottom planning failed ({e}); falling back to broadcast")
            return {"layout": "general", "error": str(e)}
        plan["layout"] = "chart_top_cam_bottom"
        plan["canonical_layout"] = "chart_top_cam_bottom"
        plan["internal_layout"] = "broadcast"
        return plan
    if a.layout == "product_full_bleed":
        # product_full_bleed stays on the plain `track` planner so the generated crop_x
        # expression stays short enough for ffmpeg (smooth_track emits ~hundreds of segments
        # that overflow ffmpeg's expression parser without a sendcmd_file sidecar).
        try:
            plan = reframe.analyze(str(src), a.start, a.dur, force="track")
        except Exception as e:
            log(f"product_full_bleed planning failed ({e})")
            return {"layout": "general", "error": str(e)}
        plan["layout"] = "product_full_bleed"
        plan["canonical_layout"] = "product_full_bleed"
        plan["internal_layout"] = "track"
        return plan
    if a.layout == "full_bleed_speaker":
        # use the zero-phase smoothed track for handheld sources. The
        # per-shot median path (default ``track``) + ``_speaker_replace`` cutaway created
        # 57 % same-subject jumps because the "dominant other face" cutaway moved the
        # crop hundreds of pixels mid-shot without a cut flag. ``smooth_track=True``
        # bypasses that path and emits a dense, velocity-clamped trajectory that holds
        # the subject at <= 1.2 % per-100 ms step (under the gate's 1.5 % cap).
        try:
            plan = reframe.analyze(str(src), a.start, a.dur, force="track",
                                    smooth_track=True)
        except Exception as e:
            log(f"full_bleed_speaker planning failed ({e})")
            return {"layout": "general", "error": str(e)}
        plan["layout"] = "full_bleed_speaker"
        plan["canonical_layout"] = "full_bleed_speaker"
        plan["internal_layout"] = "track"
        return plan
    if a.layout == "podcast_two_shot":
        # pass the words sidecar (when provided) so reframe.analyze uses the audio-
        # first diarization path to pick the active speaker instead of the mouth-motion
        # fallback. Mouth motion got a static two-shot podcast clip wrong — picked the listener. Audio-first
        # ties speaker identity to the actual voice track, which is what we want.
        words = _words_from_arg(a)
        try:
            plan = reframe.analyze(str(src), a.start, a.dur, force="track", words=words)
        except Exception as e:
            log(f"podcast_two_shot planning failed ({e}); falling back to track")
            return {"layout": "general", "error": str(e)}
        plan["layout"] = "podcast_two_shot"
        plan["canonical_layout"] = "podcast_two_shot"
        plan["internal_layout"] = "track"
        return plan
    force = a.layout if a.layout in ("track", "split", "broadcast", "hudband",
                                      "gamestack", "reaction", "screencast") else None
    # --layout auto on a source with a mapped vertical (config: render.vertical_layouts) resolves
    # to that layout so a poker source picks `hudband` without hard-coding it here.
    if force is None and a.layout == "auto" and getattr(a, "vertical", None):
        mapped = _vertical_layouts().get(a.vertical)
        if mapped in ("track", "split", "broadcast", "hudband", "gamestack", "reaction",
                       "screencast", "general"):
            if mapped == "general":
                return {"layout": "general"}
            force = mapped
    if force == "hudband":
        p = _plan_hudband(src, a)
        if p is not None:
            return p
        log("hudband: no confident HUD; falling back to track")
        force = "track"
    if force == "gamestack":
        p = _plan_gamestack(src, a)
        if p is not None:
            return p
        if game_band_only(a):
            # gameplay never falls back to a face-follow crop (track
            # centred on a cam burned in mid-frame cropped the game tight): no cam card, so the
            # game band alone, face or no face
            log("gamestack: gameplay with no cam card; game-only band (never a face crop)")
            return _plan_game_only(src, a) or {"layout": "general"}
        # per-clip fallback (a talking-head or IRL stream): face somewhere in the frame -> track;
        # nothing -> game only (tight 9:16)
        try:
            analysed = reframe.analyze(str(src), a.start, a.dur)
        except Exception as e:
            log(f"gamestack fallback probe failed ({e})")
            return {"layout": "general"}
        if analysed.get("layout") == "general":
            g = _plan_game_only(src, a)
            log("gamestack: no face at all; game-only 9:16 crop")
            return g or {"layout": "general"}
        log(f"gamestack: face outside a cam inset; falling back to {analysed['layout']}")
        return analysed
    if force == "reaction":
        p = _plan_reaction(src, a)
        if p is not None:
            return p
        # per-clip fallback: no cam -> general (full-frame); face outside inset or no
        # content region -> track; face-in-box below FACE_ACCEPT -> track.
        try:
            analysed = reframe.analyze(str(src), a.start, a.dur)
        except Exception as e:
            log(f"reaction fallback probe failed ({e})")
            return {"layout": "general"}
        if analysed.get("layout") == "general":
            log("reaction: no face detected; general layout")
            return {"layout": "general"}
        log(f"reaction: falling back to {analysed['layout']}")
        return analysed
    if force == "screencast":
        # auto screencast: no --top/--bottom given
        p = _plan_screencast_auto(src, a)
        if p is not None:
            return p
        try:
            analysed = reframe.analyze(str(src), a.start, a.dur)
        except Exception as e:
            log(f"screencast fallback probe failed ({e})")
            return {"layout": "general"}
        if analysed.get("layout") == "general":
            log("screencast: no cam detected; general layout")
            return {"layout": "general"}
        log(f"screencast: falling back to {analysed['layout']}")
        return analysed
    try:
        plan = reframe.analyze(str(src), a.start, a.dur, force=force)
    except Exception as e:
        log(f"reframe analysis failed ({e}); falling back to general")
        return {"layout": "general", "error": str(e)}
    if plan.get("layout") == "split" and force is None and getattr(a, "niche", None) \
            and not stack_allowed(a):
        # two steady faces: a split screen is for gaming/streamers only
        log(f"{a.niche}: two faces would split the frame; single view (track) instead")
        try:
            return reframe.analyze(str(src), a.start, a.dur, force="track")
        except Exception as e:
            log(f"track re-plan failed ({e}); falling back to general")
            return {"layout": "general", "error": str(e)}
    return plan


def _fit_fill(w, h, band_w, band_h, mode):
    """Pixels of a band_w x band_h band left to the blurred fill when a w x h crop is scaled
    to fit it ("width": to band_w wide, "height": to band_h tall, "inside": both)."""
    if not (w and h):
        return 0.0
    k = band_w / w if mode == "width" else band_h / h if mode == "height" else \
        min(band_w / w, band_h / h)
    return max(0.0, band_w * band_h - min(band_w, w * k) * min(band_h, h * k))


def blur_fill_frac(plan, a, info):
    """Share of the 1080x1920 canvas that shows a gblur fill instead of picture, from the
    layout geometry (the quality gate fails a canvas the blur fill
    dominates). 0.0 for full-bleed layouts; None when the layout's fill is not modelled."""
    layout = plan.get("layout")
    sw = plan.get("src_w") or info.get("width")
    sh = plan.get("src_h") or info.get("height")
    fill = 0.0
    if plan.get("v2"):                          # gamestack / game only v2: band + card
        fill = gamelayout.fill_frac(plan["band"]["out_h"], plan.get("card"),
                                    plan.get("band_y", 0))
        if plan.get("cam_mask"):                # the cam painted out of the band is no
            fill = round(fill + float(plan.get("cam_mask_share") or 0.0)   # picture either
                         * int(plan["band"]["out_h"]) / float(OUT_H), 3)
        return fill
    if plan.get("manual"):
        bmode = a.bottom_mode or ("fit" if layout == "screencast" else "cover")
        if bmode == "fit":
            fw, fh = rect(a.bottom)[:2]
            fill = _fit_fill(fw, fh, OUT_W, OUT_H - plan["split_px"], "height")
    elif layout == "general" and plan.get("content_box"):
        box = plan["content_box"]
        fill = _fit_fill(int(box["w"]), int(box["h"]), OUT_W, OUT_H, "width")
    elif layout == "general":
        fill = _fit_fill(sw, sh, OUT_W, OUT_H, "width")
    elif layout in TRACK_LIKE_LAYOUTS and track_boxed(plan):
        box = plan["crop_box"]
        fill = _fit_fill(int(box["w"]), int(box["h"]), OUT_W, OUT_H, "width")
    elif layout in TRACK_LIKE_LAYOUTS and plan.get("text_shots"):
        # render yield 11-09: a text shot fit to width is blurred fill for its span only
        span = sum(float(s["t1"]) - float(s["t0"]) for s in plan["text_shots"]
                   if s["mode"] == "fit")
        fill = _fit_fill(sw, sh, OUT_W, OUT_H, "width") * span / max(1e-6, float(a.dur))
    elif layout in ("gamestack", "reaction", "screencast_auto"):
        top_h = int(plan["top_h"])
        cw, ch, _cx, _cy, c_fit = plan["cam"]
        if layout == "gamestack" and plan["game"][2]:
            fill += _fit_fill(plan["game"][0], sh, OUT_W, top_h, "width")
        elif layout == "reaction" and plan.get("content_fit", REACTION_CONTENT_FIT) == "fit":
            fill += _fit_fill(plan["content"][0], plan["content"][1], OUT_W, top_h, "inside")
        elif layout == "screencast_auto" and plan.get("content_fit_blur"):
            fill += _fit_fill(plan["content"][0], plan["content"][1], OUT_W, top_h, "width")
        if c_fit:
            fill += _fit_fill(cw, ch, OUT_W, OUT_H - top_h, "height")
    elif layout == DUAL_VIEW:
        fill = int(plan["header_h"]) * OUT_W            # the hook header; both panels are picture
        if plan["zoom"].get("fit") == "inside":         # ... but a small cam fit inside its panel
            fill += _fit_fill(plan["zoom"]["w"], plan["zoom"]["h"], OUT_W,
                              OUT_H - int(plan["seam_y"]), "inside")
    elif layout not in TRACK_LIKE_LAYOUTS + BROADCAST_LIKE_LAYOUTS + (
            "split", "hudband", "game_only"):
        return None
    return round(fill / (OUT_W * OUT_H), 3)


DARK_LUMA = 55.0             # a window darker than this (median frame luma) gets levels
LEVELS_MAX_GAIN = 1.5        # mild: the white point never drops below 255 / this ...
LEVELS_MAX_BLACK = 16.0      # ... and the black point never rises above this
LEVELS_GAMMA = 1.15          # midtone lift, only when the stretched window is still dark
# every vertical, not only streams (config select.verticals + render.vertical_layouts; "" = no
# --vertical): most too_dark rejects were dark studios (podcasts, interviews) that got no lift
# because only stream windows were measured.
LEVELS_VERTICALS = ("stream", "gaming", "poker", "podcast", "interview", "general", "product",
                    "reaction", "screencast", "")


def auto_levels(frames):
    """a mild levels filter for a dark window, or None. `frames`: gray uint8 arrays.
    Black/white points from the 1st / 99.5th luma percentile, bounded so the stretch stays
    mild (gain <= LEVELS_MAX_GAIN); a small gamma lift only if the stretch leaves it dark.
    10-13..10-19: streamer clips 2, 3, 41, 63 were dropped or rejected as too dark."""
    import numpy as np
    if not frames:
        return None
    med = float(np.median([float(f.mean()) for f in frames]))
    if med >= DARK_LUMA:
        return None
    lum = np.concatenate([f.ravel() for f in frames])
    lo = min(float(np.percentile(lum, 1)), LEVELS_MAX_BLACK)
    top = float(np.percentile(lum, 99.5))
    if top - lo < 32:
        return None                           # a flat (black) window: nothing to stretch
    hi = max(top, lo + 255.0 / LEVELS_MAX_GAIN)       # gain 255 / (hi - lo) <= the cap
    gamma = LEVELS_GAMMA if (med - lo) * 255.0 / (hi - lo) < DARK_LUMA else 1.0
    a, b = lo / 255.0, min(1.0, hi / 255.0)
    f = (f"colorlevels=rimin={a:.3f}:gimin={a:.3f}:bimin={a:.3f}"
         f":rimax={b:.3f}:gimax={b:.3f}:bimax={b:.3f}")
    if gamma > 1.0:
        f += f",eq=gamma={gamma:.2f}"
    return {"filter": f, "src_luma": round(med, 1), "black": round(lo, 1), "white": round(hi, 1),
            "gamma": gamma}


def with_levels(vgraph, levels_filter):
    """The layout graph reading a levelled source: every [0:v] use becomes a split output."""
    n = vgraph.count("[0:v]")
    if not levels_filter or not n:
        return vgraph
    labels = [f"[lv{i}]" for i in range(n)]
    head = f"[0:v]{levels_filter}" + (f",split={n}" if n > 1 else "") + "".join(labels) + ";"
    for lab in labels:
        vgraph = vgraph.replace("[0:v]", lab, 1)
    return head + vgraph


def measure_levels(a, src, plan=None):
    """auto_levels for a LEVELS_VERTICALS window (sampled at 0.5 fps, 160 px), else None.
    --levels (repair of a too-dark reject) measures any vertical, and so does a streamer
    split (render yield 11-09: a tech screen split's game panel is the stream's dark game view,
    2876 / 2875 at median luma 29 where the gate takes 35)."""
    if (getattr(a, "vertical", None) or "") not in LEVELS_VERTICALS \
            and not getattr(a, "levels", False) and not (plan or {}).get("streamer_split"):
        return None
    import cv2
    frames = [cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
              for _t, fr in reframe.iter_frames(str(src), a.start, a.dur, fps=0.5, width=160)]
    return auto_levels(frames)


SCREEN_STACK_TOP_H = 1100    # screen stack: 608 px of screen at the bottom of this band leave
                             # the hook 230..468 px (2 lines; 960 left 98 px: 10-19 clip 73)
SCREEN_CAM_MAX_AREA = 0.15   # screen-stack reframe: the cam inset covers at most this share
SCREEN_FACE_MAX_H = 0.25     # of the frame and its face at most this share of the frame height


STACK_NICHES = ("gaming", "streamers")   # the only niches whose clips may stack
SINGLE_VIEW_NICHES = ("tech", "finance")  # full-bleed single view following the speaker, always
STACKED_LAYOUTS = ("reaction", "screencast", "screencast_auto", "gamestack", "broadcast",
                   "chart_top_cam_bottom", "hudband", "split", "dual_view")
SCREEN_BOX_MAX_FILL = 0.5    # = quality_gate.VQ_MAX_BLUR_FILL: a box the gate would fail is not made


def stack_allowed(a):
    """Whether this clip may use a stacked layout: only gaming and streamers clips.
    Without --niche (campaign clips, manual renders) the stream vertical stands in for them."""
    niche = getattr(a, "niche", None)
    if niche:
        return niche in STACK_NICHES
    return (getattr(a, "vertical", None) or "") == "stream"


def game_band_only(a):
    """Whether this clip is gameplay (quality_gate.gameplay_clip: a gamestack request, or a
    gaming / stream clip whose source is gameplay by its --content-kind or the gaming niche;
    talking heads and IRL vlogs are not). Gameplay never renders a face-follow crop: the game band
    with the cam card, the game band alone, or no clip."""
    from clipengine import quality_gate as qg
    return qg.gameplay_clip(getattr(a, "vertical", None), getattr(a, "niche", None),
                            getattr(a, "content_kind", None), getattr(a, "layout", None))


def enforce_game_band(plan, a):
    """The last check before the encode: a gameplay clip never renders a face-follow crop (a
    --layout track request, or the slice ladder's speaker crop): RenderError (the clip is dropped;
    the gate would fail it: gameplay_face_follow). The full-bleed action crop (gameplay with no
    face, centred on the action) is no face crop."""
    from clipengine import quality_gate as qg
    if plan.get("manual") or not qg.face_follow(plan.get("layout"),
                                                (plan.get("banner_guard") or {}).get("action")):
        return
    if game_band_only(a):
        raise RenderError(f"gameplay planned as {plan.get('layout')!r}, a face-follow crop: "
                          f"gameplay renders the game band (gamestack / game only) or no clip")


def small_cam_inset(a, src, top_h):
    """The presenter cam of a screen share (trading platform, editor, whiteboard, chart) when it
    is a small inset (<= SCREEN_CAM_MAX_AREA of the frame, face <= SCREEN_FACE_MAX_H of its
    height), else None (a talking-head shot: the 'cam' is the speaker)."""
    r = _resolve_cam(src, a, "reaction", top_h, max_face_h=facecam.MAX_FACE_H_CALM, mode="auto")
    if r is None:
        return None
    bx0, by0, bx1, by1 = r["cam_box"]
    face = r.get("face_median")
    if (bx1 - bx0) * (by1 - by0) > SCREEN_CAM_MAX_AREA * r["src_w"] * r["src_h"] or not face \
            or face[2] > SCREEN_FACE_MAX_H:
        log(f"screen share: cam {r['cam_box']} is not a small inset")
        return None
    return r


def screen_box_plan(a, src):
    """A non-stack niche's screen share with a tiny cam whose 9:16 crop
    would slice its UI: the whole screen (or chart) fit to the canvas width over its blurred
    fill, no cam band (the general layout). Returns (plan or None, reason). None when the frame
    is not a screen share with a small cam, or when the box would be more blurred fill than the
    quality gate accepts (a 16:9 screen: 68 % > 50 %) - the clip is then rejected rather than
    rendered to fail the gate."""
    r = small_cam_inset(a, src, SCREEN_STACK_TOP_H)
    if r is None:
        return None, "not a screen share with a small cam"
    fill = _fit_fill(r["src_w"], r["src_h"], OUT_W, OUT_H, "width") / (OUT_W * OUT_H)
    if fill > SCREEN_BOX_MAX_FILL:
        return None, (f"{SCREEN_TOO_WIDE} {fill:.0%} blurred fill (quality gate cap "
                      f"{SCREEN_BOX_MAX_FILL:.0%})")
    return {"layout": "general", "screen_box": True}, f"screen box, {fill:.0%} blurred fill"


SCREEN_TOO_WIDE = "screen box would be"   # screen_box_plan found a small-cam screen share whose
                                          # whole frame is too wide to fit (the screen fit reads it)


def screen_stack_plan(a, src, sc):
    """a screen share (trading platform, editor, whiteboard) with a small presenter cam
    whose 9:16 track crop has to slice its UI text (10-19: 5 of the 8 banner-guard rejects).
    Reframe instead of rejecting: the whole screen fit to width over a blurred fill on top (no
    crop edge crosses the UI) and the cam zoomed in the band below - the reaction layout with
    the full frame as its content. Gaming and streamers clips only (stack_allowed). None when
    the frame has no small cam inset (a talking-head shot whose banners sit on the speaker) or
    when the cam crop itself would slice a banner."""
    top_h = SCREEN_STACK_TOP_H
    r = small_cam_inset(a, src, top_h)
    if r is None:
        return None
    if _gamestack_cfg().get("v2") and not gamelayout.is_inset(r["cam_box"], r["src_w"], r["src_h"]):
        # gamestack v2: a face inside the scene (an IRL shot's person) is no
        # streamer cam; the stack would show the scene's own face as an 820 px cam band
        log(f"screen stack: cam {r['cam_box']} sits inside the scene (no frame edge)")
        return None
    src_w, src_h = r["src_w"], r["src_h"]
    face = r["face_median"]
    cam = _cam_clear_of_text(
        facecam.cam_crop(src_w, src_h, r["cam_box"], face, bot_h=OUT_H - top_h),
        [g["box"] for g in sc["graphics"] if g["banner"]], face, src_w, src_h)
    if cam is None:
        log("screen stack: the cam crop would slice a banner; no reframe")
        return None
    cw, ch, cx, cy, c_fit = cam
    return {"layout": "reaction", "src_w": src_w, "src_h": src_h, "top_h": top_h,
            "content_fit": "fit", "content_valign": "bottom",
            "content": [_even(src_w), _even(src_h), 0, 0],
            "content_box": [0, 0, src_w, src_h], "cam": [cw, ch, cx, cy, c_fit],
            "cam_box": r["cam_box"], "cam_box_norm": r["box_norm"],
            "raw_cam_box": r["guard_box"], "raw_cam_box_norm": r["raw_box_norm"],
            "from_profile": r["from_profile"], "face_frac": r["face_frac"],
            "face_median": face, "channel": r["channel"], "screen_stack": True}


def _cam_clear_of_text(cam, boxes, face, src_w, src_h):
    """The cam crop with its rows cut off any burned-in text line it would slice (a screen's
    status bar running under a corner cam: 10-19 clips 74/76), the face kept whole and at least
    half the crop's height left; fit-by-height once trimmed. None when that is impossible."""
    cw, ch, cx, cy, fit = cam
    fy0 = (face[1] - face[2] / 2) * src_h
    fy1 = (face[1] + face[2] / 2) * src_h
    for b in sorted(boxes, key=lambda b: b[1]):
        if b[3] <= cy or b[1] >= cy + ch or not banners.sliced(b, cx, cw, src_w):
            continue
        if b[1] >= fy1:                                   # below the face: end the crop above it
            ch = _even(b[1] - banners.CUT_MARGIN - cy)
        elif b[3] <= fy0:                                 # above the face: start below it
            y = b[3] + banners.CUT_MARGIN
            y += y % 2
            ch, cy = _even(cy + ch - y), y
        else:
            return None
        fit = True
        if ch < 0.5 * cam[1] or cy > fy0 or cy + ch < fy1:
            return None
    return [cw, ch, cx, cy, fit]


def guard_banners(plan, a, src):
    """a 9:16 crop never slices a persistent source banner (lower
    third, ticker). Track-like plans are shifted, trimmed or boxed by banners.fix_track_plan,
    and a screen share whose crop still slices its UI is reframed: a screen stack for gaming and
    streamers (screen_stack_plan), the whole-screen box for every other niche when the quality
    gate could pass it (screen_box_plan), else rejected; the gamestack game crop
    is shifted off banners and text
    panels (chat) by banners.fix_gamestack_plan; split crops are checked as-is. Returns the
    report stored on the plan (a reframe replaces the plan's contents in place); raises
    RenderError when the crop cannot avoid slicing one (clip rejected)."""
    try:
        return _guard_banners(plan, a, src)
    except RenderError:
        raise
    except Exception as e:        # noqa: BLE001 - an unverifiable crop is not shipped
        raise RenderError(f"banner scan failed: {type(e).__name__}: {e}")


def _guard_banners(plan, a, src):
    layout = plan.get("layout")
    if plan.get("manual") or "src_w" not in plan:
        return None
    if layout in TRACK_LIKE_LAYOUTS and plan.get("segments"):
        size = (plan["src_w"], plan["src_h"])
        sc = _scan(src, a, size)
        rep = banners.fix_track_plan(
            plan, sc, words_fn=lambda: banners.ocr_source_words(str(src), a.start, a.dur, size))
        if rep["action"] != "reject":
            # a source's own dead band (10-20 clip 108: a 9:16 vertical stream
            # ending in 72 black rows) never reaches the canvas
            dead = banners.trim_dead_bands(plan, sc)
            if dead["action"] != "none":
                rep = {**rep, "dead_band": dead, "crop_box": plan.get("crop_box")}
                if rep["action"] == "none":
                    rep["action"] = f"dead_band_{dead['action']}"
        if rep["action"] == "reject":
            # gaming/streamers: screen stack; every other niche: the whole-screen box, or reject
            if stack_allowed(a):
                alt, why = screen_stack_plan(a, src, sc), "no screen stack"
            else:
                alt, why = screen_box_plan(a, src)
                if alt is None and why.startswith(SCREEN_TOO_WIDE):
                    plan["screen_share"] = True    # the screen box goes first
            if alt is not None:
                rep = {"banners": rep.get("banners"), "frames": rep.get("frames"),
                       "action": "reframe", "layout": alt["layout"],
                       "rejected_crop": rep.get("reason")}
                plan.clear()
                plan.update(alt)
            else:
                rep["reason"] = f"{rep.get('reason')}; {why}"
    elif layout in ("gamestack", "game_only") and plan.get("v2"):
        # the v2 band was planned inside its legal runs; every window on its path is checked
        sw = int(plan["src_w"])
        sc = _scan(src, a, (sw, int(plan["src_h"])))
        bs = [banners.plate(g["box"], sw, sc["src_h"]) for g in sc["graphics"] if g["banner"]]
        scene = [list(b) for b in plan["band"].get("scene_text") or []]
        bs = [b for b in bs if list(b) not in scene]     # the planner left these to the gate
        gw = int(plan["band"]["w"])
        cut = sorted({tuple(b) for _t0, _t1, x, _y in v2_views(plan, a.dur) for b in bs
                      if banners.sliced(b, x, gw, sw)})
        rep = {"banners": bs, "frames": sc["frames"], "action": "reject" if cut else "none"}
        if cut:
            rep["reason"] = f"game band {gw}px slices banner(s) {[list(b) for b in cut]} on its path"
    elif layout == "gamestack" and plan.get("game"):
        sc = banners.scan(str(src), a.start, a.dur, src_size=(plan["src_w"], plan["src_h"]))
        rep = banners.fix_gamestack_plan(plan, sc)
    elif layout == "split" and plan.get("split"):
        sc = banners.scan(str(src), a.start, a.dur, src_size=(plan["src_w"], plan["src_h"]))
        bs = [banners.plate(g["box"], sc["src_w"], sc["src_h"]) for g in sc["graphics"]
              if g["banner"]]
        cut = [b for c in plan["split"].values() for b in bs
               if b[3] > c["y"] and b[1] < c["y"] + c["h"]
               and banners.sliced(b, c["x"], c["w"], sc["src_w"])]
        rep = {"banners": bs, "frames": sc["frames"],
               "action": "reject" if cut else "none"}
        if cut:
            rep["reason"] = f"split crop slices banner(s) {cut}"
    else:
        return None
    plan["banner_guard"] = rep
    if rep["action"] == "reject":
        raise RenderError(f"crop would slice a source banner: {rep.get('reason')}")
    if rep["action"] != "none":
        log(f"banner guard: {rep['action']} "
            f"{rep.get('crop_box') or rep.get('x') or rep.get('layout')}")
    return rep


_SCANS = {}


def _scan(src, a, size):
    """banners.scan of the clip window, once per render (the banner guard and the slice ladder
    read the same scan)."""
    key = (str(src), float(a.start), float(a.dur), tuple(size))
    if key not in _SCANS:
        _SCANS[key] = banners.scan(str(src), a.start, a.dur, src_size=size)
    return _SCANS[key]


def _overlays(src, a, size):
    """render yield 11-09 class 2: the window's static corner overlays (overlays.scan: a network's
    logo bug), boxes src px padded, once per render; [] outside the daily niches (campaign and
    manual renders as before) or when the frames cannot be read."""
    if not getattr(a, "niche", None):
        return []
    key = ("overlays", str(src), float(a.start), float(a.dur), tuple(size))
    if key not in _SCANS:
        try:
            _SCANS[key] = [o["box"] for o in overlays.scan(src, a.start, a.dur, *size)]
        except Exception as e:    # noqa: BLE001 - unreadable: the crop as planned
            log(f"overlays: unreadable ({type(e).__name__}: {e})")
            _SCANS[key] = []
        if _SCANS[key]:
            log(f"overlays: static corner overlay(s) {_SCANS[key]}")
    return _SCANS[key]


def cover_overlays(plan, a, src):
    """render yield 11-09 class 2, the last resort: a static corner overlay a final crop window
    still cuts (no shift kept it whole or out) is covered: plan["cover"] holds the part of it the
    windows show (overlays.visible_parts), painted out of the source before any crop
    (overlays.cover_graph: delogo), so no partial logo reaches the canvas. A part over a sampled
    face is never painted (left as planned, logged). Returns the covered boxes."""
    if plan.get("manual") or not plan.get("src_w") or not getattr(a, "niche", None):
        return []
    sw, sh = int(plan["src_w"]), int(plan["src_h"])
    ovs = _overlays(src, a, (sw, sh))
    parts = overlays.visible_parts(crop_views(plan, a.dur), ovs) if ovs else []
    if not parts:
        return []
    # the face itself (the detector's box, eyes to mouth), not its hair room: the Playmaker bug
    # sits over the guest's cap (2889), a cover there leaves the face as it is
    faces = [[(f["cx"] - f["w"] / 2) * sw, (f["cy"] - f["h"] / 2) * sh, (f["cx"] + f["w"] / 2) * sw,
              (f["cy"] + f["h"] / 2) * sh]
             for _t, fs in _scan(src, a, (sw, sh)).get("faces") or [] for f in fs]
    cov = overlays.coverable(parts, faces)
    if len(cov) < len(parts):
        log(f"overlays: {len(parts) - len(cov)} cut overlay(s) over a face are not covered")
    if cov:
        plan["cover"] = cov
        log(f"overlays: no crop keeps the overlay(s) whole or out; their shown part {cov} covered")
    return cov


_CAMS = {}


def cam_overlay(src, a, size):
    """cropcheck.cam_overlay of the clip window (the facecam's recurring face and
    its box, read on the probe's frames: 1 fps at banners.SCAN_W), once per render; None when
    the window has no recurring small face."""
    key = (str(src), float(a.start), float(a.dur), tuple(size))
    if key not in _CAMS:
        import cv2
        sc = _scan(src, a, size)
        grays = [cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) for _t, fr in reframe.iter_frames(
            str(src), a.start, a.dur, fps=cropcheck.PROBE_FPS, width=banners.SCAN_W,
            src_size=tuple(size))]
        tiles = cropcheck.tile_finder(grays, int(size[0])) if grays else None
        _CAMS[key] = cropcheck.cam_overlay(sc.get("faces") or [], tiles, int(size[0]),
                                           int(size[1]))
    return _CAMS[key]


CAM_BOX_TOL = 16          # src px a planned cam box may reach past the cam's own
                          # box before it is cut back to it
CAM_BOX_MAX_FACE = 4.0    # with no box read, a planned cam box over this many face heights tall
                          # grew past the cam (10-26 1396 / 1390: the busy-pixel growth ran down a
                          # calm game to the frame's bottom: [0, 14, 340, 720] for a 208 px cam)


def _even_in(box):
    """Box [x0, y0, x1, y1] with even edges, rounded inward (never onto the cam's border)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    return [x0 + x0 % 2, y0 + y0 % 2, x1 - x1 % 2, y1 - y1 % 2]


def _holds(box, pt, tol=0):
    return box[0] - tol <= pt[0] <= box[2] + tol and box[1] - tol <= pt[1] <= box[3] + tol


def _painted(win, tile, mask, tol=CAM_BOX_TOL):
    """Whether every part of the cam's box `tile` the window `win` shows is inside the painted-out
    `mask` (a mid-frame cam's band)."""
    x0, y0 = max(win[0], tile[0]), max(win[1], tile[1])
    x1, y1 = min(win[2], tile[2]), min(win[3], tile[3])
    return x1 <= x0 or y1 <= y0 or (_holds(mask, (x0, y0), tol) and _holds(mask, (x1, y1), tol))


def _cam_pt(ov, size):
    """The facecam face's centre (src px) of a cam_overlay."""
    return ov["face"]["cx"] * int(size[0]), ov["face"]["cy"] * int(size[1])


def plausible_cam(cam_box, ov, size):
    """Whether a planned cam box (src px) can be the facecam's own box when that box could not be
    read: it holds the cam's face and is at most CAM_BOX_MAX_FACE face heights tall."""
    if not cam_box or not ov:
        return False
    fh = float(ov["face"]["h"]) * int(size[1])
    return _holds(cam_box, _cam_pt(ov, size)) and cam_box[3] - cam_box[1] <= CAM_BOX_MAX_FACE * fh


def bound_cam(cam_box, a, src, size):
    """(cam_box, why) for a stack's cam panel in gaming / streamers when the planned box holds
    the window's facecam (cam_overlay): the facecam's own box when the planned one reaches past
    it (by > CAM_BOX_TOL); (None, why) when that box could not be read and the planned one is
    over CAM_BOX_MAX_FACE face heights tall; else as planned (no facecam, another face's box,
    or a box inside the cam's)."""
    if not cam_box or not dual_view_allowed(a) or not getattr(a, "niche", None):
        return cam_box, None
    try:
        ov = cam_overlay(src, a, size)
    except Exception as e:        # noqa: BLE001 - unreadable: the box as planned
        log(f"cam bound: cam box unreadable ({type(e).__name__}: {e})")
        return cam_box, None
    if not ov or not _holds(cam_box, _cam_pt(ov, size)):
        return cam_box, None
    tile = ov.get("tile")
    bx0, by0, bx1, by1 = [int(v) for v in cam_box]
    if tile:
        if _holds(tile, (bx0, by0), CAM_BOX_TOL) and _holds(tile, (bx1, by1), CAM_BOX_TOL):
            return cam_box, None
        nb = _even_in(tile)
        return nb, f"cam box {list(cam_box)} reaches past the facecam's box {tile}: the cam's box"
    if not plausible_cam(cam_box, ov, size):
        fh = float(ov["face"]["h"]) * int(size[1])
        return None, (f"the facecam's box cannot be read and the planned cam box {list(cam_box)} "
                      f"is {(by1 - by0) / fh:.1f} face heights tall: its panel would show the game")
    return cam_box, None


def _ocr(src, a, size):
    """quality_gate.ocr_source_text of the clip window, once per render (kept with the scans);
    None when the OCR is unavailable."""
    key = ("ocr", str(src), float(a.start), float(a.dur), tuple(size))
    if key not in _SCANS:
        from clipengine import quality_gate as qg
        try:
            _SCANS[key] = qg.ocr_source_text(str(src), a.start, a.dur, tuple(size))
        except Exception as e:    # noqa: BLE001 - the line detector's graphics still count
            log(f"OCR unavailable ({type(e).__name__}: {e})")
            _SCANS[key] = None
    return _SCANS[key]


def keep_plates(sc, words, sw, sh, pad=0, min_frac=0.0):
    """the plates no crop edge of a HUD niche (and no dual view zoom) may cross: the
    gate's text items and every OCR'd word of >= cropcheck.PLATE_MIN_ALNUM letters, grown to their
    plates (hud_panels), plus the scan's docked text panels (a chat column the OCR reads a line
    of at a time). [{"box", "frac"}] src px, `pad` px around each; `min_frac` keeps the plates
    that persist (a decision reads those: a shirt print or a wall sign the OCR read on one frame
    triggers nothing; where a crop may go still keeps clear of every plate)."""
    from clipengine import quality_gate as qg
    items = qg.text_items(sc, words)
    pls = [p for p in cropcheck.plates(items, words or [], sw, sh, pad=pad) if p["frac"] >= min_frac]
    return pls + cropcheck.panel_plates(sc, sw, sh, pad=pad)


BODY_W = 1.0                 # a shirt print's centre is within this many face widths of a
BODY_MAX_W = 2.5             # face's centre, it is at most this many face widths wide, and it starts
                             # below the chin


def off_body(plates, faces, sw, sh, min_frac=None):
    """The plates minus the ones on a speaker's body: a plate that does not persist (frac <
    cropcheck.PROBE_MIN_FRAC), under a sampled face's chin, centred within BODY_W face widths of
    it and no wider than BODY_MAX_W of them, moves with the speaker (10-18 clip 1357: 'CRUSH' on
    Bart's shirt, read on 1 of 3 OCR frames, left no 9:16 window that holds his face as he
    leans). A board graphic beside a player (10-21 clip 472) is no shirt."""
    min_frac = cropcheck.PROBE_MIN_FRAC if min_frac is None else min_frac
    fs = [f for _t, ff in faces or [] for f in ff]
    out = []
    for p in plates:
        b = p["box"]
        cx = (b[0] + b[2]) / 2.0
        body = p["frac"] < min_frac and any(
            b[1] >= (f["cy"] + f["h"] / 2.0) * sh and abs(cx - f["cx"] * sw) <= BODY_W * f["w"] * sw
            and b[2] - b[0] <= BODY_MAX_W * f["w"] * sw for f in fs)
        if not body:
            out.append(p)
    return out


def tiled_broadcast(src, a):
    """Whether the window shows video boxes (render/broadcast.detect_broadcast_panel: a panel, or
    any video box in a sampled frame: 10-18 clip 1309's guest box, seen once, made the open recrop
    show her empty room); True when it cannot be read (the tile rule stands)."""
    key = ("tiles", str(src), float(a.start), float(a.dur))
    if key not in _SCANS:
        from clipengine.render import broadcast
        v = broadcast.detect_broadcast_panel(str(src), a.start, a.dur)
        _SCANS[key] = v.get("panel") is not False or int(v.get("tiles_max") or 0) >= 1
        log(f"slice ladder: video boxes {'seen' if _SCANS[key] else 'none'} ({v.get('why') or v.get('reason')})")
    return _SCANS[key]


def item_plates(sc, words, pad=0):
    """The gate's own text items (quality_gate.text_items: what crop_slices counts) as plates,
    tight to their boxes: the fallback a zoom keeps whole when every plate cannot be."""
    from clipengine import quality_gate as qg
    sw, sh = int(sc["src_w"]), int(sc["src_h"])
    return [{"box": [max(0, int(it["box"][0]) - pad), max(0, int(it["box"][1]) - pad),
                     min(sw, int(it["box"][2]) + pad), min(sh, int(it["box"][3]) + pad)],
             "frac": float(it.get("frac") or 1.0)} for it in qg.text_items(sc, words)]


LADDER_MIN_SHARE = 0.05      # clip-quality 10-21 stage 2: a speaker crop slicing burned-in text for
                             # this on-screen share (quality_gate.crop_slices units) is re-cropped
                             # (10-21 clip 472 sliced a POT board for 0.67 of the clip and passed
                             # the gate's 0.4 cap; the eye-check rejected it)
LADDER_BOX_SHARE = 0.15      # ... and boxed (ladder step c) when no clean recrop exists and it
                             # slices at least this share (the gate fails 0.4; the eye-check
                             # rejected 472 below it). Smaller slices keep the planned crop
WINDOW_TOL = 4               # src px a crop may reach past the speaker's boxed window
LADDER_TEXT_PAD = 6          # src px around each text item the ladder keeps a crop edge out of:
                             # cleancrop lets a box poke 3 px over an edge, the gate calls an item
                             # cut when any row of it shows (10-21 clip 315: a line 2 px into the
                             # recrop's top edge measured 0.24 sliced)
LADDER_WIN_INSET = 6         # src px more the ladder keeps inside a tile (cleancrop's 3 px inset
                             # left 10-21 clip 331's white tile border on the box edge)
LADDER_BAND_FPS = 1.0        # source frames the ladder reads a crop's flat bands on
LADDER_BAND_W = 640
LADDER_BAND_RETRIES = 2      # a ladder crop that still holds a flat band is re-searched this often
TILE_BOX_MIN_H_FRAC = 0.5    # (b) a speaker's tile fit to width keeps at least this much of the
                             # source height (10-21 clip 490's news boxes are 380 of 720 px)


def _leaves_window(plan, wins):
    """Segments whose crop reaches outside the speaker's boxed window (its border and the
    neighbouring tile on screen: 10-21 clips 331 / 490 / 491)."""
    cw, ch, cy = track_crop_window(plan)
    ch = plan["src_h"] if ch == "ih" else ch
    out = []
    for i, (s, b) in enumerate(zip(plan.get("segments") or [], wins or [])):
        if b and (s["x"] < b[0] - WINDOW_TOL or s["x"] + cw > b[2] + WINDOW_TOL
                  or cy < b[1] - WINDOW_TOL or cy + ch > b[3] + WINDOW_TOL):
            out.append(i)
    return out


def _seg_at(segs, t):
    """Index of the plan segment shown at clip time t (crop_views' convention)."""
    i = 0
    for k, s in enumerate(segs):
        if k and float(s["t0"]) <= t:
            i = k
    return i


BAND_LINE_STD = 6.0          # the ladder's band rows: luma sd across a row at most this (the gate's
                             # edge_bands takes 3.0: the grey gradient under 10-21 clip 315 / 385's
                             # desk reads sd 3-5 and passed it, the eye-check rejected it) ...
BAND_STEP = 2.5              # ... changing at most this much from row to row (a smooth gradient:
BAND_MAX_DRIFT = 45.0        # 385 ran luma 51 -> 26) and this much in all
BAND_WIDEN = 0.5             # a graded band must also go on this share of the crop's width past
                             # it on one side: a desk or a headliner spans the scene, a dark
                             # hoodie ends on both sides of the speaker (10-21 PASS clip 465)


def soft_band_depths(gray, line_std=BAND_LINE_STD, step=BAND_STEP, max_drift=BAND_MAX_DRIFT,
                     edges=("top", "bottom")):
    """{edge: depth px} of smoothly graded rows (columns) from each of `edges` of one gray frame:
    luma sd <= line_std across each, <= step from the one before, <= max_drift in all."""
    import numpy as np
    g = np.asarray(gray, dtype=np.float32)
    out = {}
    for edge in edges:
        lines = g if edge in ("top", "bottom") else g.T
        if edge in ("bottom", "right"):
            lines = lines[::-1]
        std, mean = lines.std(axis=1), lines.mean(axis=1)
        d = 0
        while d < len(mean) and std[d] <= line_std and abs(mean[d] - mean[0]) <= max_drift \
                and (d == 0 or abs(mean[d] - mean[d - 1]) <= step):
            d += 1
        out[edge] = d
    return out


def crop_bands(plan, frames, soft=True):
    """The quality gate's edge-band verdict (quality_gate.edge_band_verdict: >= 4 % of the side in
    >= 60 % of the frames) for a full-bleed track crop, measured on gray source frames
    [(t, gray)] cropped to it: the gate's flat bands (edgebands.flat_edge_depths), and with
    `soft` graded top / bottom bands (soft_band_depths) that go on BAND_WIDEN past the crop on a
    side, plus
    the band depths
    (src px) per segment: {i: {edge: px}}. A boxed crop is measured on its central 9:16
    slice: the blurred fill is the box scaled to cover the canvas, so that slice's top and
    bottom rows are the canvas's top and bottom (10-21 clip 311: an unlit headliner across the
    box top became a flat blurred band over the top 20 % of the canvas). (None, {}) without
    frames. A frame that is flat all over (a fade) holds no band."""
    from clipengine import quality_gate as qg
    if not frames or not plan.get("segments"):
        return None, {}
    sw, sh = int(plan["src_w"]), int(plan["src_h"])
    k = sw / float(frames[0][1].shape[1])
    cw, ch, cy = track_crop_window(plan)
    ch = sh if ch == "ih" else int(ch)
    segs = plan["segments"]
    per, by_seg, wh = [], {}, None
    inset = (cw - ch * 9 / 16) / 2 if track_boxed(plan) and cw > ch * 9 / 16 else 0
    for t, g in frames:
        i = _seg_at(segs, t)
        x = int(segs[i]["x"] + inset)
        region = g[int(cy / k):int((cy + ch) / k), int(x / k):int((x + cw - 2 * inset) / k)]
        if region.size == 0 or min(region.shape) < 8:
            continue
        wh = wh or (region.shape[1], region.shape[0])
        if float(region.std()) < 8.0:
            d = {e: 0 for e in edgebands.EDGES}
        else:
            d = edgebands.flat_edge_depths(region)
        if soft and float(region.std()) >= 8.0:
            rw = cw - 2 * inset
            pad = BAND_WIDEN * rw
            rows = slice(int(cy / k), int((cy + ch) / k))
            left = g[rows, int(max(0, x - pad) / k):int((x + rw) / k)]
            right = g[rows, int(x / k):int(min(sw, x + rw + pad) / k)]
            graded = soft_band_depths(region)
            # the band goes on past the crop on at least one side (a desk or a show background
            # under a webcam box: 10-21 clip 315's has a decorative circle on its left)
            on = [soft_band_depths(side) for side in (left, right)
                  if side.shape[1] > region.shape[1]]
            for e in ("top", "bottom"):
                d[e] = max(d[e], min(graded[e], max([o[e] for o in on] or [0])))
        per.append(d)
        by_seg.setdefault(i, []).append(d)
    if not per:
        return None, {}
    v = qg.edge_band_verdict(per, *wh)
    bands = {i: {e: int(round(px * k)) for e, px in edgebands.persistent_bands(ds).items()}
             for i, ds in by_seg.items()}
    return v, bands


BAND_KEEP_FRAC = 0.03         # an excluded band leaves this much of the crop side to the crop (under
                              # the gate's 4 %): a speaker whose head sits right under the band
                              # still fits (10-21 clip 311's cap under the headliner)


def band_regions(plan, bands, min_frac, keep=BAND_KEEP_FRAC):
    """exclude boxes (src px) over each segment's flat crop bands of >= min_frac of its side, all
    but `keep` of the side, so the next search moves the crop (nearly) off them (cleancrop:
    excluded regions stay outside)."""
    cw, ch, cy = track_crop_window(plan)
    ch = int(plan["src_h"]) if ch == "ih" else int(ch)
    kh, kw = int(keep * ch), int(keep * cw)
    out = []
    for i, d in bands.items():
        x = int(plan["segments"][i]["x"])
        if d.get("top", 0) >= min_frac * ch:
            out.append([x, cy, x + cw, cy + d["top"] - kh])
        if d.get("bottom", 0) >= min_frac * ch:
            out.append([x, cy + ch - d["bottom"] + kh, x + cw, cy + ch])
        if d.get("left", 0) >= min_frac * cw:
            out.append([x, cy, x + d["left"] - kw, cy + ch])
        if d.get("right", 0) >= min_frac * cw:
            out.append([x + cw - d["right"] + kw, cy, x + cw, cy + ch])
    return out


FULL_BLEED_STEPS = ("a_recrop", "b_tile")    # ladder steps whose crop fills the canvas


OVERLAY_HEAD_UP = 0.05        # render yield 11-09: a crop leaving a logo bug out above the speaker
                              # may trim the head room over the forehead to this (x the face
                              # height; cleancrop keeps HEAD_UP 0.2): the hair, never the face


def _head_room(sc, head_up):
    """A copy of scan `sc` whose faces get only `head_up` (x the face height) of head room over
    the forehead in cleancrop's face boxes, the chin room unchanged: each face's height scaled
    and its centre moved down so that cy - (0.5 + HEAD_UP) h' = cy - (0.5 + head_up) h and the
    box's bottom stays."""
    hu, cd = cleancrop.HEAD_UP, cleancrop.CHIN_DOWN
    k = (1.0 + head_up + cd) / (1.0 + hu + cd)
    faces = [(t, [dict(f, h=f["h"] * k, cy=f["cy"] + (0.5 + cd) * f["h"] * (1.0 - k))
                  for f in fs]) for t, fs in sc.get("faces") or []]
    return dict(sc, faces=faces)


def overlay_shots(plan, planned, fn, sc, exclude, no_slice, windows, frames, a, ovs):
    """render yield 11-09 class 2: the ladder step's crop that keeps the logo bugs whole or out
    (`plan`) shown only in the shots that show a bug. The same step without them (the crop as it
    would have been) is the base; over each span where that crop cuts a bug the clean crop's
    window takes over (plan["text_shots"], mode "crop"). 2884: the crop moved under the guest's
    Playmaker bug cut the host's NIGHT CAP sign in his shots, which the crop as it was kept.
    Returns the base plan, or None (keep `plan`: the step fails without the bugs, holds a flat
    band, or every shot shows a bug)."""
    import copy
    base = copy.deepcopy(planned)
    try:
        crop = fn(base, sc, list(exclude), no_slice, windows=windows)
    except cleancrop.CleanCropError:
        return None
    cleancrop.apply(base, crop)
    bv, _bands = crop_bands(base, frames)
    if bv is not None and bv["passed"] is False:
        return None
    dur = float(a.dur)
    clean = crop_views(plan, dur)
    shots = []
    for v in crop_views(base, dur):
        if not overlays.cut([v], ovs):
            continue
        for c in clean:
            t0, t1 = max(v["t0"], c["t0"]), min(v["t1"], c["t1"])
            if t1 - t0 > 1e-3:
                shots.append({"mode": "crop", "w": c["w"], "h": c["h"], "x": c["x"], "y": c["y"],
                              "t0": round(t0, 3), "t1": round(t1, 3), "why": "overlay"})
    if shots and sum(s_["t1"] - s_["t0"] for s_ in shots) >= dur - 1e-3:
        return None
    if shots:
        base["text_shots"] = shots
        log(f"overlays: the clean crop only in the {len(shots)} shot(s) that show a logo bug; "
            f"the crop as it was elsewhere")
    return base


def hud_panels(boxes, src_w, src_h):
    """HUD niches: the panels a full-bleed crop must keep whole (cropcheck.hud_panels: each text
    box grown to the plate drawn around it, overlapping plates merged). The OCR reads a HUD
    panel's words, not the panel: 10-21 clip 472's recrop kept clear of '$24,500' and cut the
    board's last card (a suit the OCR does not read) and the panel's border."""
    return cropcheck.hud_panels(boxes, src_w, src_h)


def slice_ladder(plan, a, src, guard, rejected=None, dual_view=None):
    """clip-quality 10-21 stage 2: a speaker crop (track-like plan) that slices a graphic is
    edited, not rejected. The planned crop is measured the way the quality gate will measure the
    render (quality_gate.crop_slices on the banner scan + OCR'd text; the speaker's boxed window,
    cleancrop.window_boxes). When it slices text (>= LADDER_MIN_SHARE), leaves the speaker's
    box, or the banner guard could not fix it (`rejected`), the ladder runs:

      (a) open layout: the clean 9:16 recrop that slices nothing and keeps the faces
          (cleancrop.clean_crop: same size and height for the whole clip, >= 720 px of canvas)
      (b) boxed broadcast layout with a face: the speaker's tile, full-bleed (clean_crop inside
          the window) or fit to width (clean_box inside the window)
      (c) the largest slice-free box (<= cleancrop.BOX_MAX_ASPECT:1) fit to the canvas width
          over a blurred fill of the same frame; then the whole frame the same way when its
          blurred fill is within the quality gate's cap
          (charts and screens fit to width: memory clip-layout-full-bleed-speaker)

    A full-bleed crop holding a flat band the gate's edge_bands would fail (a desk, an unlit
    headliner, a tile's backdrop: 10-21 clips 315 / 311 / 331) also runs the ladder; every
    full-bleed step is re-measured on the source frames and searched again off its band (up to
    LADDER_BAND_RETRIES times), else it counts as failed.

    (c) runs when the planned crop cannot ship as it is (the guard rejected it, a flat band or
    its slices fail the gate, it leaves the speaker's box) or slices >= LADDER_BOX_SHARE; a
    smaller slice that (a)/(b) cannot better keeps the planned crop. The gate still checks the final render with its own
    thresholds. Returns the report stored as plan["banner_guard"]; raises RenderError when
    nothing on the ladder fits (the clip is rejected as before).

    `dual_view(why, tried)` (HUD niches, hud_layout): when the crop slices text, the guard
    rejected or boxed it, or the gate would fail it, and no full-bleed step (a / b tile) is
    clean, it replaces the plan with the dual view and returns its report, before any box
    (the screen fit's box too); None (no room for a zoom panel) lets the ladder go on as
    above, the screen box first."""
    if plan.get("layout") not in TRACK_LIKE_LAYOUTS or not plan.get("segments") \
            or "src_w" not in plan:
        if rejected:
            raise RenderError(rejected)
        return guard
    from clipengine import quality_gate as qg
    sw, sh = int(plan["src_w"]), int(plan["src_h"])
    sc = _scan(src, a, (sw, sh))
    words = _ocr(src, a, (sw, sh))
    hud = hud_niche(a)
    plates = off_body(keep_plates(sc, words, sw, sh), sc.get("faces"), sw, sh) if hud else []
    ovs = _overlays(src, a, (sw, sh))         # render yield 11-09: logo bugs, whole or out
    try:
        wins = cleancrop.window_boxes(src, a.start, a.dur, plan, sc.get("faces") or [])
    except Exception as e:        # noqa: BLE001 - an open layout then
        log(f"slice ladder: window scan failed ({type(e).__name__}: {e})")
        wins = None
    boxed = bool(wins) and any(wins)
    if boxed:
        k = LADDER_WIN_INSET
        wins = [[b[0] + k, b[1] + k, b[2] - k, b[3] - k] if b else None for b in wins]
    try:
        frames = [(t, edgebands.to_gray(fr)) for t, fr in reframe.iter_frames(
            str(src), a.start, a.dur, fps=LADDER_BAND_FPS, width=LADDER_BAND_W, src_size=(sw, sh))]
    except Exception as e:        # noqa: BLE001 - no band check then
        log(f"slice ladder: band frames unavailable ({type(e).__name__}: {e})")
        frames = []
    why, must, sliced = [], bool(rejected), False
    if dual_view is not None and track_boxed(plan):
        why.append("the banner guard boxed the crop")
        sliced = True
    if not rejected:
        bv, _bands = crop_bands(plan, frames)
        if bv is not None and bv["passed"] is False:
            why.append("flat crop band: " + "; ".join(bv["errors"]))
            # only a band the gate itself fails may push the clip to a box (c); a graded band
            # the gate passes is worth a recrop, not a box (10-21 clip 306: a dark night-time
            # car interior went to a box showing the stream's chat overlay)
            strict, _ = crop_bands(plan, frames, soft=False)
            must = must or (strict is not None and strict["passed"] is False)
    if rejected:
        why.append(str(rejected))
    else:
        v = qg.crop_slice_verdict(qg.crop_slices(sc, crop_views(plan, a.dur), a.dur,
                                                 words=words))
        if v["crop_share"] >= LADDER_MIN_SHARE:
            why.append(f"slices burned-in text (on-screen share {v['crop_share']:.2f})")
            must = must or not v["passed"] or v["crop_share"] >= LADDER_BOX_SHARE
            sliced = True
        if plates and not stack_allowed(a):
            # a poker HUD plate the OCR's 4-letter items miss ('BB', a card, a CHECK
            # tag: 10-18 clips 1315 / 1357) is a sliced graphic to the eye-check; the plates
            # that persist decide (calibration: one-frame reads of shirts and signs on posted
            # clips). Not in gaming / streamers: a posted split's wall logo flipped it (1017)
            share = cropcheck.cut_share(crop_views(plan, a.dur),
                                        [p for p in plates
                                         if p["frac"] >= cropcheck.PROBE_MIN_FRAC], a.dur)
            if share >= LADDER_MIN_SHARE and v["crop_share"] < LADDER_MIN_SHARE:
                why.append(f"slices HUD plates (on-screen share {share:.2f})")
                must = must or share >= LADDER_BOX_SHARE
                sliced = True
        out = _leaves_window(plan, wins) if boxed else []
        if out:
            why.append(f"segment(s) {out} leave the speaker's boxed window")
            must = True
        # render yield 11-09 class 2: a crop through a logo bug is recropped (whole or out); a
        # cut no step can avoid keeps the crop and is covered (cover_overlays), never boxed
        oc = overlays.cut(crop_views(plan, a.dur), ovs)
        if oc:
            why.append(f"cuts {len(oc)} static corner overlay(s) (a logo bug: {oc[:2]})")
    # a chart / screen / slide (no face, a screen-kind source with only small
    # faces, or a screen share the banner guard found) is fit to width where its crop would
    # slice (the layout rule: charts and screens fit to width; gaming / streamers keep their
    # stacks), and a crop through one of its content blocks (a chart panel, a post card: no
    # OCR'd text to count) is a sliced graphic too (10-21 fixtures: a face found in a chart or a
    # defocused room planned a 9:16 sliver of it, and the gate failed it for no face)
    screen_src = fit_width_allowed(a) and bool(plan.get("screen_share") or screen_content(a, sc))
    blocks = None
    if screen_src and frames:
        try:
            blocks = content_blocks(src, a, sw, sh, frames=[g for _t, g in frames])
        except Exception as e:    # noqa: BLE001 - screen_fit_plan reads them again or refuses
            log(f"slice ladder: content blocks unavailable ({type(e).__name__}: {e})")
        cut = _cut_blocks(plan, a, blocks or []) if not rejected else []
        if cut:
            why.append(f"slices {len(cut)} content block(s) of a screen ({cut[:3]})")
            must = True
    if not why and getattr(a, "clean_first", False):
        why.append("a known edge_slice source: the clean recrop first")
    if not why:
        return guard
    tried = []
    dual = dual_view if (must or sliced) else None     # a graded band alone: the ladder as was
    # the screen box before any 9:16 recrop (which would show a slice-free sliver of the screen,
    # or a tiny cam blown up), unless the speaker sits in a boxed tile (a broadcast two-up: the
    # tile steps first, the screen box last and never through a tile). A HUD niche's dual view
    # comes before any box (poker is one, and fit_width_allowed): its full-bleed steps, then
    # the dual view, then the screen box (screen_next) when the dual view has no room
    screen_first = screen_src and not boxed and dual is None
    screen_next = screen_src and not boxed and dual is not None
    screen_last = screen_src and boxed
    tiles = [list(map(int, b)) for b in (wins or []) if b] if boxed else []

    def screen_fit():
        try:
            alt = screen_fit_plan(a, sw, sh, sc, words, extra=tiles, src=src, blocks=blocks,
                                  frames=frames or None)    # the ladder's frames: no 2nd decode
        except cleancrop.CleanCropError as e:
            tried.append(f"d_screen_fit: {e}")
            return None
        rep = {"action": "ladder_d_screen_fit", "layout": "general", "why": "; ".join(why),
               "tried": list(tried), "blur_fill": alt["blur_fill"],
               "crop_box": dict(alt["content_box"]),
               "previous": guard.get("action") if guard else None}
        plan.clear()
        plan.update({**alt, "banner_guard": rep})
        return rep
    if screen_first:
        rep = screen_fit()
        if rep:
            return rep
    # what may not be sliced is what the gate will check: its text items (persistent graphics
    # OCR confirms + OCR'd lines); every line-detector box only when OCR is unavailable (a shirt
    # print or a shelf edge is no graphic: 10-21 clips 377 / 491 had 14 / 9 such boxes and no
    # crop or box could leave them all whole). Dead bands stay outside.
    exclude, no_slice = cleancrop.regions_from_scan(sc, strict=False)
    if words is not None:
        no_slice = [list(map(int, it["box"])) for it in qg.text_items(sc, words)]
    if hud and (words is not None or dual_view is not None):
        no_slice = [p["box"] for p in plates]          # whole persistent plates
    pad = LADDER_TEXT_PAD
    no_slice = [[max(0, b[0] - pad), max(0, b[1] - pad), min(sw, b[2] + pad), min(sh, b[3] + pad)]
                for b in no_slice]
    # render yield 11-09 class 2: each step keeps the logo bugs whole or out first (they come
    # padded), then the same with less head room over the speaker (a bug just above a guest's
    # cap: 2889); a step that cannot is tried again without them, its cut covered
    # (cover_overlays)
    whole = no_slice + [list(b) for b in ovs]
    variants = [(whole, sc), (whole, _head_room(sc, OVERLAY_HEAD_UP)),
                (no_slice, sc)] if ovs else [(no_slice, sc)]
    import copy
    planned = copy.deepcopy(plan)
    import functools
    tile_box = functools.partial(cleancrop.clean_box, min_h_frac=TILE_BOX_MIN_H_FRAC)
    # poker is full-bleed (a HUD niche that may not stack): its speaker is never boxed
    # over a blurred fill either; no clean full-bleed crop rejects the clip before the encode.
    # render yield 11-09 class 1: a daily gaming / streamers clip neither (no blurred or black
    # band: the streamer split or full-bleed, else rejected)
    bleed_only = hud and (not stack_allowed(a) or no_band_stack(a))
    steps = ([("b_tile", cleancrop.clean_crop, wins), ("b_tile_box", tile_box, wins)]
             if boxed else [("a_recrop", cleancrop.clean_crop, None)])
    if boxed and not tiled_broadcast(src, a):
        # a 'boxed window' on a source that shows no video boxes is the room behind the
        # speaker (10-18 clip 926: shelves read as a box around Sam Parr): the open recrop after
        # the tile, before any box
        steps.insert(1, ("a_recrop", cleancrop.clean_crop, None))
    if bleed_only:
        steps = [st for st in steps if st[0] in FULL_BLEED_STEPS]
    if must and not boxed and not bleed_only:
        steps.append(("c_box", cleancrop.clean_box, None))
    for name, fn, w, (ns, scv) in [(n, f, w, v) for n, f, w in steps for v in variants]:
        if dual is not None and name not in FULL_BLEED_STEPS:
            rep = dual("; ".join(why), tried) or (screen_fit() if screen_next else None)
            if rep is not None:
                return rep
            dual = None
        loose = ns is not variants[0][0]       # the logo bugs left to the cover
        ex = list(exclude)
        for attempt in range(LADDER_BAND_RETRIES + 1):
            plan.clear()
            plan.update(copy.deepcopy(planned))
            try:
                crop = fn(plan, scv, ex, ns, windows=w)
            except cleancrop.CleanCropError as e:
                crop = None
                tried.append(f"{name}{' (logo bugs covered)' if loose else ''}"
                             f"{' (head room trimmed)' if scv is not sc else ''}: {e}")
                break
            cleancrop.apply(plan, crop)
            bv, bands = crop_bands(plan, frames)
            if bv is None or bv["passed"] is not False:
                break
            more = band_regions(plan, bands, bv["min_frac"])
            if attempt == LADDER_BAND_RETRIES or not more:
                # the gate fails this band: the step failed, the next one is tried
                tried.append(f"{name}: flat band kept ({'; '.join(bv['errors'])})")
                crop = None
                break
            ex += more                                # move off the band and search again
        if crop is None:
            plan.clear()
            plan.update(copy.deepcopy(planned))
            continue
        if ovs and not loose:
            only = overlay_shots(plan, planned, fn, sc, exclude, no_slice, w, frames, a, ovs)
            if only is not None:
                plan.clear()
                plan.update(only)
        rep = {"action": f"ladder_{name}", "crop_box": plan["crop_box"], "why": "; ".join(why),
               "canvas_w": crop.get("canvas_w"), "windows": wins if boxed else None,
               "tried": tried, "previous": guard.get("action") if guard else None}
        if plan.get("text_shots"):
            rep["overlay_shots"] = len(plan["text_shots"])
        if loose:
            rep["overlays_cover"] = True
        if scv is not sc:
            rep["head_room"] = OVERLAY_HEAD_UP
        plan["banner_guard"] = rep
        log(f"slice ladder {name}{' (logo bugs covered)' if loose else ''}: "
            f"{crop['w']}x{crop['h']} at y={crop['y']}"
            + (f" fit to width ({crop.get('blur_fill', 0):.0%} blurred fill)"
               if crop.get("fit") == "box" else f" ({crop['canvas_w']} px of canvas)")
            + f" ({'; '.join(why)})")
        return rep
    if dual is not None:
        rep = dual("; ".join(why), tried) or (screen_fit() if screen_next else None)
        if rep is not None:
            return rep
    if not must:
        log(f"slice ladder: kept the planned crop ({'; '.join(why)}; {'; '.join(tried)})")
        if guard is not None:
            guard["ladder_tried"] = tried
        return guard
    fill = _fit_fill(sw, sh, OUT_W, OUT_H, "width") / (OUT_W * OUT_H)
    if fill <= SCREEN_BOX_MAX_FILL and not bleed_only:
        rep = {"action": "ladder_c_frame", "layout": "general", "why": "; ".join(why),
               "tried": tried, "blur_fill": round(fill, 3)}
        plan.clear()
        plan.update({"layout": "general", "slice_ladder": True, "src_w": sw, "src_h": sh,
                     "banner_guard": rep})
        log(f"slice ladder: the whole frame fit to width ({fill:.0%} blurred fill)")
        return rep
    if screen_last or (bleed_only and screen_src and not screen_first):
        rep = screen_fit()
        if rep:
            return rep
    rep = _tolerant_full_bleed(plan, planned, a, sc, words, exclude, steps, frames, why, tried,
                               guard)
    if rep is not None:
        return rep
    raise RenderError(f"crop would slice a source banner or graphic: {'; '.join(why)}; "
                      f"ladder: {'; '.join(tried)}; the whole frame fit to width would be "
                      f"{fill:.0%} blurred fill (gate cap {SCREEN_BOX_MAX_FILL:.0%})")


# ---- a full-bleed crop the gate's own slice measure passes ----
# render 'other' was 163 rejects in 7 days, every one the slice ladder refusing every step: 81
# ended on the speaker's tile (b_tile / b_tile_box: podcasts 46, tech 17, finance 15) and 55 on
# the open layout's box (c_box: lifestyle 22, finance 11, tech 10). The ladder's steps keep EVERY
# OCR'd text item whole or out, while the gate fails a crop only when the text it slices is on
# screen for a share over SLICE_CROP_SHARE_MAX (0.4: item persistence x the time it is cut). The
# HEAD replay's refusals were held by such items: a 55x40 px item beside the speaker in his tile
# (TCJav5-LrIw), a studio sign the crop cuts in one 4 s shot only (kytapOss9mc, share ~0.15).
# Last, before the refusal: the full-bleed steps once more, keeping whole only the items the gate
# would weigh most (the least persistent left out one at a time), and a crop is taken only when
# the gate's own measure on it stays at LADDER_TOLERANT_SHARE or under, with no flat band. The
# gate still measures the render (gate_slice_check before the encode, the gate after it).
LADDER_TOLERANT_SHARE = 0.2


def _tolerant_full_bleed(plan, planned, a, sc, words, exclude, steps, frames, why, tried, guard):
    """The ladder's last step (see above): a report like the full-bleed steps' with action
    'ladder_<step>_tolerant', the plan cropped; None (the plan back as planned) when no crop
    keeps the gate's slice share under LADDER_TOLERANT_SHARE."""
    import copy
    from clipengine import quality_gate as qg
    if words is None:
        return None
    sw, sh = int(planned["src_w"]), int(planned["src_h"])
    items = sorted(qg.text_items(sc, words), key=lambda it: (float(it["frac"]), it["box"]))
    full = [(n, f, w) for n, f, w in steps if n in FULL_BLEED_STEPS]
    if not items or not full:
        return None
    pad = LADDER_TEXT_PAD
    boxed = lambda b: [max(0, b[0] - pad), max(0, b[1] - pad), min(sw, b[2] + pad),   # noqa: E731
                       min(sh, b[3] + pad)]
    for k in range(1, len(items) + 1):
        left_out = items[:k]
        keep = [boxed(list(map(int, it["box"]))) for it in items[k:]]
        for name, fn, w in full:
            plan.clear()
            plan.update(copy.deepcopy(planned))
            try:
                crop = fn(plan, sc, list(exclude), keep, windows=w,
                          soft=[boxed(list(map(int, it["box"]))) for it in left_out])
            except cleancrop.CleanCropError:
                continue
            cleancrop.apply(plan, crop)
            bv, _b = crop_bands(plan, frames)
            if bv is not None and bv["passed"] is False:
                continue
            v = qg.crop_slice_verdict(qg.crop_slices(sc, crop_views(plan, a.dur), a.dur,
                                                     words=words))
            if not v["passed"] or v["crop_share"] > LADDER_TOLERANT_SHARE:
                tried.append(f"{name} (tolerant, {k} item(s) left out): the gate's slice share "
                             f"{v['crop_share']:.2f}")
                continue
            rep = {"action": f"ladder_{name}_tolerant", "crop_box": plan["crop_box"],
                   "why": "; ".join(why), "canvas_w": crop.get("canvas_w"), "tried": list(tried),
                   "left_out": [list(map(int, it["box"])) for it in left_out],
                   "crop_share": v["crop_share"],
                   "previous": guard.get("action") if guard else None}
            plan["banner_guard"] = rep
            log(f"slice ladder {name} (tolerant): {crop['w']}x{crop['h']} at y={crop['y']} "
                f"({crop['canvas_w']} px of canvas), {k} text item(s) of persistence <= "
                f"{left_out[-1]['frac']:.2f} left out, the gate's slice share "
                f"{v['crop_share']:.2f} (<= {LADDER_TOLERANT_SHARE}) ({'; '.join(why)})")
            return rep
    plan.clear()
    plan.update(copy.deepcopy(planned))
    return None


# ---------------------------------------------------------------- screen fit

SCREEN_FIT_TEXT_PAD = LADDER_TEXT_PAD    # src px kept between the box edge and any text item
BLOCK_FPS = 1.0              # content blocks: source frames read per second ...
BLOCK_SCAN_W = 640           # ... at this width
BLOCK_DETAIL = 20.0          # a pixel shows content when its smoothed gradient (|Sobel x| + |y|)
                             # reaches this in any sampled frame (a flat backdrop stays under it)
BLOCK_CLOSE_FRAC = 0.012     # content closer than this share of the width is one block (a panel's
                             # lines, a chart's candles); a wider gap separates two blocks
BLOCK_MIN_AREA = 0.015       # blocks under this share of the frame are text items (OCR has them)


def content_blocks(src, a, sw, sh, frames=None):
    """the source's content blocks (src px [x0, y0, x1, y1]): a panel, a chart's
    plot, a speaker's tile, a slide's body. A screen box keeps each whole or leaves it out: a
    box edge through a block is a sliced graphic to the eye-check (10-21 previews: boxes clear
    of every OCR'd line still cut a dashboard's tables and a two-up's second tile). Raises when
    no frame can be read (an unverifiable box is not made)."""
    import cv2
    import numpy as np
    if frames is None:
        frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _t, f in reframe.iter_frames(
            str(src), a.start, a.dur, fps=BLOCK_FPS, width=BLOCK_SCAN_W, src_size=(sw, sh))]
    if not frames:
        raise RenderError("screen fit: no source frame to find its content blocks on")
    h, w = frames[0].shape[:2]
    acc = np.zeros((h, w), np.float32)
    for g in frames:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        acc = np.maximum(acc, cv2.GaussianBlur(np.abs(gx) + np.abs(gy), (5, 5), 0))
    k = max(3, int(BLOCK_CLOSE_FRAC * w))
    mask = cv2.morphologyEx((acc >= BLOCK_DETAIL).astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((k, k), np.uint8))
    n, _lab, st, _c = cv2.connectedComponentsWithStats(mask)
    fx, fy = sw / float(w), sh / float(h)
    return [[int(x * fx), int(y * fy), int(min(sw, (x + bw) * fx)), int(min(sh, (y + bh) * fy))]
            for x, y, bw, bh, area in (st[i] for i in range(1, n))
            if area >= BLOCK_MIN_AREA * w * h]


def fit_width_allowed(a):
    """Whether the layout rule lets this clip fit a screen to width (full-bleed follows the
    speaker, split screen only for streamers, charts and screens fit to width): a daily niche clip
    of every niche but the stacking ones (gaming / streamers keep their stacks and splits).
    Without --niche (campaign and manual renders) nothing changes."""
    return bool(getattr(a, "niche", None)) and not stack_allowed(a)


def _cut_blocks(plan, a, blocks):
    """Content blocks the plan's crop windows cut (partly in a window, not whole in it) that a
    screen box could hold whole (no wider than cleancrop.BOX_MAX_ASPECT x the frame height): a
    block the size of the screen is cut by any crop, and no box can do better."""
    tol = cleancrop.SLICE_TOL
    sh = int(plan.get("src_h") or 0)
    fits = [b for b in blocks if b[2] - b[0] <= cleancrop.BOX_MAX_ASPECT * sh]
    cut = []
    for v in crop_views(plan, float(getattr(a, "dur", 0) or 0)):
        x0, y0, x1, y1 = v["x"], v["y"], v["x"] + v["w"], v["y"] + v["h"]
        for b in fits:
            overlap = b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0
            whole = b[0] >= x0 - tol and b[2] <= x1 + tol and b[1] >= y0 - tol and b[3] <= y1 + tol
            if overlap and not whole and b not in cut:
                cut.append(b)
    return cut


def screen_content(a, sc):
    """A screen share, chart or slide rather than a speaker shot: the scan found no face at all,
    or the source's kind says screen (discover.source_kind, --content-kind) and every face is a
    small cam (< SCREEN_FACE_MAX_H of the frame height): a talking head keeps its speaker crop."""
    faces = [f for _t, fs in (sc.get("faces") or []) for f in fs]
    if not faces:
        return True
    return getattr(a, "content_kind", None) == "screen" and \
        max(float(f.get("h") or 0) for f in faces) < SCREEN_FACE_MAX_H


def _face_boxes(sc, sw, sh):
    """Every sampled face (grown to hair and chin) as a src-px box: a screen fit keeps a cam's
    face whole in its box or leaves it out, never cut."""
    out = []
    for _t, fs in sc.get("faces") or []:
        for f in fs:
            fh = f["h"] * sh
            out.append([int(max(0, (f["cx"] - f["w"] / 2) * sw)),
                        int(max(0, f["cy"] * sh - (0.5 + cleancrop.HEAD_UP) * fh)),
                        int(min(sw, (f["cx"] + f["w"] / 2) * sw)),
                        int(min(sh, f["cy"] * sh + (0.5 + cleancrop.CHIN_DOWN) * fh))])
    return out


def screen_fit_plan(a, sw, sh, sc, words=None, extra=(), blocks=None, src=None, frames=None):
    """the screen box of a chart / screen / slide: the largest box (at most
    cleancrop.BOX_MAX_ASPECT:1, so the blurred fill stays under the quality gate's cap) that
    slices none of the text the gate will check (its OCR'd text items, else the scan's
    graphics, padded SCREEN_FIT_TEXT_PAD), no face (a cam's face is whole in the box or out),
    no content block (content_blocks: a panel, a plot, a tile; `blocks` given, else read from
    `src`), no docked panel and none of the `extra` boxes (a speaker's tile), centred on the
    text, leaving the source's dead bands out. A 16:9 dashboard or a full-width chart has no
    such box: it is still rejected. It is rendered as the general layout (no speaker crop)
    cropped to the box: fit to the canvas width over a blurred fill of the same frame. The fill
    is the box cover-scaled, so its top and bottom rows are the box's: a box whose fill would
    hold a flat band the gate's edge_bands fails (a slide's flat margin) is searched again off
    it, as the slice ladder does for its boxes, else refused (`frames`: gray source frames at
    LADDER_BAND_W, the ladder's own; read from `src` when not given). Returns the plan; raises
    cleancrop.CleanCropError when no box fits."""
    if frames is None and src is not None:
        try:
            frames = [(t, edgebands.to_gray(fr)) for t, fr in reframe.iter_frames(
                str(src), a.start, a.dur, fps=LADDER_BAND_FPS, width=LADDER_BAND_W,
                src_size=(sw, sh))]
        except Exception as e:      # noqa: BLE001 - an unverifiable box is not made
            raise cleancrop.CleanCropError(f"source frames unreadable ({type(e).__name__}: {e})")
    if blocks is None:
        try:
            blocks = content_blocks(src, a, sw, sh,
                                    frames=[g for _t, g in frames] if frames else None)
        except Exception as e:      # noqa: BLE001 - an unverifiable box is not made
            raise cleancrop.CleanCropError(f"content blocks unreadable ({type(e).__name__}: {e})")
    import copy
    from clipengine import quality_gate as qg
    exclude, no_slice = cleancrop.regions_from_scan(sc, strict=False)
    panels = [list(map(int, p)) for p in sc.get("panels") or []]
    if words is not None:
        no_slice = [list(map(int, it["box"])) for it in qg.text_items(sc, words)]
    pad = SCREEN_FIT_TEXT_PAD
    no_slice = [[max(0, b[0] - pad), max(0, b[1] - pad), min(sw, b[2] + pad), min(sh, b[3] + pad)]
                for b in no_slice] + _face_boxes(sc, sw, sh)
    # a block ends where the source's dead bands begin (its edge is read on blurred gradients,
    # a few px wide: a card between two flat margins would otherwise never fit between them)
    db = {k: int(v or 0) for k, v in (sc.get("dead_bands") or {}).items()}
    m = cleancrop.MARGIN                    # the box keeps this clear of an excluded band
    lo_x, hi_x = db.get("left", 0) + m, sw - db.get("right", 0) - m
    lo_y, hi_y = db.get("top", 0) + m, sh - db.get("bottom", 0) - m
    blocks = [[max(b[0], lo_x), max(b[1], lo_y), min(b[2], hi_x), min(b[3], hi_y)]
              for b in blocks or ()]
    blocks = [b for b in blocks if b[2] > b[0] and b[3] > b[1]]
    shapes = panels + [list(map(int, b)) for b in list(extra or ()) + blocks]
    # centre of interest: the text, weighted by its area (a chart's axis labels and title, a
    # slide's lines, a terminal's output); the frame centre without any
    area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in no_slice) or 0
    cx = (sum((b[0] + b[2]) / 2 * (b[2] - b[0]) * (b[3] - b[1]) for b in no_slice) / area
          if area else sw / 2)
    cw = reframe.crop_width(sw, sh)
    static = {"layout": "track", "src_w": sw, "src_h": sh,
              "segments": [{"t0": 0.0, "t1": float(getattr(a, "dur", 0) or 0),
                            "x": int(max(0, min(sw - cw, cx - cw / 2)))}]}
    faceless = copy.deepcopy(sc)
    faceless["faces"] = []
    ex = list(exclude)
    for attempt in range(LADDER_BAND_RETRIES + 1):
        box = cleancrop.clean_box(copy.deepcopy(static), faceless, ex, no_slice + shapes)
        if not frames:
            break
        boxed = cleancrop.apply(copy.deepcopy(static), box)       # the fill = the box covered
        bv, bands = crop_bands(boxed, frames, soft=False)
        if bv is None or bv["passed"] is not False:
            break
        more = band_regions(boxed, bands, bv["min_frac"])
        if attempt == LADDER_BAND_RETRIES or not more:
            raise cleancrop.CleanCropError(
                f"screen box {box['w']}x{box['h']}: its blurred fill would show a flat band the "
                f"gate's edge_bands fails ({'; '.join(bv['errors'])})")
        ex += more                                  # move off the band and search again
    x = int(box["x"][0])
    fill = _fit_fill(box["w"], box["h"], OUT_W, OUT_H, "width") / (OUT_W * OUT_H)
    if fill > SCREEN_BOX_MAX_FILL:
        raise cleancrop.CleanCropError(f"screen box {box['w']}x{box['h']} would be {fill:.0%} "
                                       f"blurred fill (gate cap {SCREEN_BOX_MAX_FILL:.0%})")
    plan = {"layout": "general", "screen_fit": True, "src_w": sw, "src_h": sh,
            "content_box": {"x": x, "y": int(box["y"]), "w": int(box["w"]), "h": int(box["h"])},
            "blur_fill": round(fill, 3)}
    log(f"screen fit: {box['w']}x{box['h']} at x={x} y={box['y']} fit to width "
        f"({fill:.0%} blurred fill; {len(no_slice)} text / face region(s) left whole)")
    return plan


def screen_fit_general(plan, a, src, info):
    """a general plan (no speaker found) on a frame whose whole-frame fit the
    quality gate would fail for blurred fill (a 16:9 screen: 68 % > 50 %) gets the screen box
    instead, where the layout rule allows it. Returns the report, or None (plan unchanged)."""
    if plan.get("layout") != "general" or plan.get("manual") or plan.get("content_box") \
            or not fit_width_allowed(a):
        return None
    sw, sh = int(info.get("width") or 0), int(info.get("height") or 0)
    if not (sw and sh):
        return None
    whole = _fit_fill(sw, sh, OUT_W, OUT_H, "width") / (OUT_W * OUT_H)
    if whole <= SCREEN_BOX_MAX_FILL:
        return None
    from clipengine import quality_gate as qg
    sc = _scan(src, a, (sw, sh))
    try:
        words = qg.ocr_source_text(str(src), a.start, a.dur, (sw, sh))
    except Exception as e:        # noqa: BLE001 - the scan's graphics still count
        log(f"screen fit: OCR unavailable ({type(e).__name__}: {e})")
        words = None
    try:
        alt = screen_fit_plan(a, sw, sh, sc, words, src=src)
    except cleancrop.CleanCropError as e:
        log(f"screen fit: none ({e}); the whole frame fit to width ({whole:.0%} blurred fill)")
        return None
    rep = {"action": "screen_fit", "layout": "general", "blur_fill": alt["blur_fill"],
           "crop_box": dict(alt["content_box"]),
           "why": f"the whole frame fit to width is {whole:.0%} blurred fill"}
    plan.clear()
    plan.update({**alt, "banner_guard": rep})
    return rep


# ---------------------------------------------------------------- dual view (HUD niches, 10-21)

# Gaming, poker and streamer sources are covered in burned-in graphics (pot / board bugs,
# scoreboards, chat, alerts, socials bars): any 9:16 crop of a 16:9 frame cuts one (10-21: 17 of
# 44 rejects edge_slice, nearly all these niches), and the ladder's fit-to-width box is mostly
# blurred fill the gate refuses. The dual view shows the WHOLE frame fit to the canvas width (the
# action stays readable, nothing is cropped away: memory clip-layout-full-bleed-speaker) and
# fills the rest with a zoom of the same frame: the facecam (the cam goes on the bottom), else
# the speaking player's face, else the busiest region. Captions sit on the seam, the hook in the
# header above the frame (clear of the platform's top UI), never over the frame's graphics.
DUAL_VIEW = "dual_view"
# the dual view is a stack, and only gaming and streamers clips may stack (STACK_NICHES,
# the eye-check's layout rule). 10-18..25: every gate-passed poker dual view (30) was
# rejected at the eye-check, 28 of them for the stack itself (layout_hud_mismatch: poker is
# full-bleed and follows the speaker). Poker keeps the HUD rule below (whole plates), not the stack.
DUAL_VIEW_NICHES = STACK_NICHES
DUAL_VIEW_VERTICALS = ("stream",)    # ... the stream vertical stands in without --niche
HUD_NICHES = ("gaming", "poker", "streamers")   # a full-bleed crop keeps every HUD plate whole
HUD_VERTICALS = ("stream", "poker")             # ... their verticals stand in without --niche
DV_SEAM_Y = 1100          # canvas row of the panels' seam: a 16:9 frame (608 px) under a 492 px
                          # hook header (= SCREEN_STACK_TOP_H: the hook's 230..468 px)
DV_HEADER_MIN = 484       # the header always holds a 3-line hook at the smallest size (230 px)
                          # + captions.TEXT_PAD under the platform's top UI (captions.HOOK_TOP)
DV_MIN_ZOOM_H = 560       # a frame whose fit-to-width panel leaves less (narrower than ~1.25:1:
                          # a vertical stream) gets no dual view
DV_SAMPLE_FPS = 2.0       # source frames the zoom is planned on
DV_SAMPLE_W = 640
DV_MOTION_W = 160         # width of the motion map the busiest region is found on
DV_FACE_SHARE = 0.5       # a shot is a face shot when the speaker's face is in >= half its samples
DV_FACE_FRAC = 0.30       # the zoomed face fills this share of the zoom panel's height ...
DV_FACE_Y = 0.40          # ... its centre this far down the panel
DV_MOTION_FRAC = 0.6      # a zoom without a face shows this share of the source height, and
                          # no zoom more: a big face's window is capped here, or the zoom panel
                          # shows the top panel again (10-21 clips 569 / 561: a face 0.35 of the
                          # frame asked for a window taller than the frame)
DV_MAX_UPSCALE = 3.0      # the zoom shows at least (panel height / this) source rows
DV_TEXT_PAD = 6           # src px around a scanned graphic / panel kept out of the motion map
DV_MIN_SHOT_S = 1.5       # a 'shot' shorter than this (a muzzle flash or an explosion read as a
                          # cut: 10-21 clip 570 had 17 in 15.7 s) joins its neighbour: no zoom jump
DV_CAM_FACE_MAX = 0.15    # a facecam inset's face is at most this share of the frame height
                          # (10-21: 570's 0.05, 569's 0.13); a bigger 'cam' face is a camera shot
                          # the face zoom frames better (309: a carved pumpkin at 0.17 on
                          # a shelf; 561: a car interior at 0.21)


def dual_view_allowed(a):
    """Whether this clip may render the dual view: gaming, poker and streamers clips; without
    --niche (campaign clips, manual renders) the stream and poker verticals stand in."""
    niche = getattr(a, "niche", None)
    if niche:
        return niche in DUAL_VIEW_NICHES
    return (getattr(a, "vertical", None) or "") in DUAL_VIEW_VERTICALS


def no_band_stack(a):
    """render yield 11-09 class 1: a daily gaming / streamers clip ships no blurred or black band
    (the dual view's blurred hook header was the 11-09 black_band reject of 15 of 20 xQc clips):
    the streamer split replaces the dual view and the slice ladder keeps to its full-bleed steps.
    Campaign and manual renders (no --niche) keep the dual view."""
    return bool(getattr(a, "niche", None)) and dual_view_allowed(a)


def hud_niche(a):
    """Whether this clip's source carries a HUD (gaming, poker, streamers): its full-bleed crop
    keeps every HUD plate whole (hud_panels), not only the text the OCR reads."""
    niche = getattr(a, "niche", None)
    if niche:
        return niche in HUD_NICHES
    return (getattr(a, "vertical", None) or "") in HUD_VERTICALS


def _is_poker(a):
    niche = getattr(a, "niche", None)
    return niche == "poker" if niche else (getattr(a, "vertical", None) or "") == "poker"


def dual_view_geometry(src_w, src_h):
    """(header_h, panel_h, seam_y) canvas px of the dual view for a src_w x src_h frame, or None
    when the frame fit to the canvas width leaves under DV_MIN_ZOOM_H for the zoom panel."""
    ph = max(2, int(round(OUT_W * float(src_h) / float(src_w) / 2)) * 2)
    seam = max(DV_SEAM_Y, ph + DV_HEADER_MIN)
    if OUT_H - seam < DV_MIN_ZOOM_H:
        return None
    return seam - ph, ph, seam


def dual_view_panels(plan):
    """{"header", "top", "zoom"}: canvas row ranges [y0, y1) of a dual view plan's three bands
    (QC "panels": the gate reads its edge bands on "top" and the canvas top)."""
    hh, seam = int(plan["header_h"]), int(plan["seam_y"])
    return {"header": [0, hh], "top": [hh, seam], "zoom": [seam, OUT_H]}


def dual_view_cam_crop(box, face, ar, src_w, src_h):
    """(w, h, x, y) src px: the cover crop of cam `box` [x0, y0, x1, y1] at the zoom panel's aspect
    `ar` (facecam.cam_crop's cover rule, for a portrait cam too): a cam wider than the panel keeps
    its height, centred on the face; a taller one keeps its width, the face DV_FACE_Y down."""
    x0, y0, x1, y1 = [int(v) for v in box]
    bw, bh = max(2, x1 - x0), max(2, y1 - y0)
    fx = face[0] * src_w if face else (x0 + x1) / 2.0
    fy = face[1] * src_h if face else (y0 + y1) / 2.0
    if bw / float(bh) >= ar:
        h = _even(bh)
        w = max(2, min(_even(round(ar * h)), _even(bw)))
        x, y = min(max(fx - w / 2.0, x0), x1 - w), y0
    else:
        w = _even(bw)
        h = max(2, min(_even(round(w / ar)), _even(bh)))
        x, y = x0, min(max(fy - DV_FACE_Y * h, y0), y1 - h)
    x = _even(max(0, min(src_w - w, x)))
    y = _even(max(0, min(src_h - h, y)))
    return w, h, x, y


def dual_view_cam_zoom(box, face, bh, header_h, src_w, src_h):
    """The zoom window {"w", "h", "x", "y"} (src px) of a facecam: its cover crop
    (dual_view_cam_crop), or when that would upscale past DV_MAX_UPSCALE (a small inset: 10-21
    clip 570's 164 px tall cam, 5x) the whole cam box fit inside the panel over its blurred fill
    ("fit": "inside"), as long as the canvas's blurred fill stays within the gate's cap."""
    w, h, x, y = dual_view_cam_crop(box, face, OUT_W / float(bh), src_w, src_h)
    if bh / float(h) <= DV_MAX_UPSCALE:
        return {"w": w, "h": h, "x": x, "y": y}
    x0, y0, x1, y1 = [int(v) for v in box]
    bw, bhb = _even(x1 - x0), _even(y1 - y0)
    fill = int(header_h) * OUT_W + _fit_fill(bw, bhb, OUT_W, bh, "inside")
    if bw < 2 or bhb < 2 or fill / float(OUT_W * OUT_H) > SCREEN_BOX_MAX_FILL:
        return {"w": w, "h": h, "x": x, "y": y}
    return {"w": bw, "h": bhb, "x": _even(x0), "y": _even(y0), "fit": "inside"}


def _cam_candidate(src, a, prior, bh):
    """{"cam_box", "face_median"} of the facecam a gamestack / reaction / screencast plan already
    found, else of a small cam inset (small_cam_inset) on the window; None without one."""
    if prior.get("layout") in ("gamestack", "reaction", "screencast_auto") and prior.get("cam_box"):
        return {"cam_box": list(prior["cam_box"]), "face_median": prior.get("face_median")}
    try:
        r = small_cam_inset(a, src, OUT_H - bh)
    except Exception as e:        # noqa: BLE001 - no cam: the zoom then
        log(f"dual view: cam detection failed ({type(e).__name__}: {e})")
        return None
    if r is None:
        return None
    return {"cam_box": list(r["cam_box"]), "face_median": r.get("face_median")}


def small_cam_face(cam):
    """Whether a cam candidate's face is small enough for a facecam inset (DV_CAM_FACE_MAX)."""
    face = (cam or {}).get("face_median")
    return bool(face) and float(face[2]) <= DV_CAM_FACE_MAX


def _in_box(f, box):
    """Whether face `f` (normalized) has its centre in `box` (normalized [x0, y0, x1, y1])."""
    return box[0] <= f["cx"] <= box[2] and box[1] <= f["cy"] <= box[3]


def cam_holds(samples, cuts, dur, box):
    """Whether the cam `box` (normalized) holds a face in >= DV_FACE_SHARE of the samples of every
    camera shot (of >= 2 samples). A cam found on part of the window does not hold where the
    stream cuts to its full camera (10-21 clip 569: the bottom showed a plush toy) or on an IRL
    stream's moving picture (561: a 'cam' the gamestack detector found in a car)."""
    spans = reframe._intervals(cuts, float(dur))
    for k, (t0, t1) in enumerate(spans):
        last = k == len(spans) - 1
        idx = [s for s in samples if t0 <= s["t"] < t1 or (last and s["t"] >= t1)]
        if len(idx) < 2:
            continue
        hit = sum(1 for s in idx if any(_in_box(f, box) for f in s.get("faces") or []))
        if hit < DV_FACE_SHARE * len(idx):
            return False
    return True


def _speaker_windows(prior):
    """[(t0, x0, x1)] normalized: the speaker crop of a planned track / hudband / broadcast plan
    over time (who the speaker crop followed), or None."""
    segs = prior.get("segments") or []
    if not segs or not prior.get("src_w"):
        return None
    sw, sh = float(prior["src_w"]), float(prior["src_h"])
    lay = prior.get("layout")
    if lay in TRACK_LIKE_LAYOUTS:
        cw = track_crop_window(prior)[0]
    elif lay == "hudband" and prior.get("cw"):
        cw = prior["cw"]
    elif lay in BROADCAST_LIKE_LAYOUTS:
        cw = reframe.broadcast_crop_width(int(sw), int(sh))
    else:
        return None
    return [(float(s["t0"]), s["x"] / sw, (s["x"] + float(cw)) / sw) for s in segs]


def _target_face(faces, t, speaker, ref_cx, cam=None):
    """The face the zoom follows in one sample: the streamer's in the cam box `cam` (normalized)
    when it shows, else the largest inside the speaker crop at t, else (no speaker crop planned)
    the primary face (reframe._primary)."""
    if cam:
        inside = [f for f in faces if _in_box(f, cam)]
        if inside:
            return max(inside, key=lambda f: f["h"])
    if speaker:
        i = _seg_at([{"t0": s[0]} for s in speaker], t)
        _t0, x0, x1 = speaker[i]
        inside = [f for f in faces if x0 <= f["cx"] <= x1]
        return max(inside, key=lambda f: f["h"]) if inside else None
    return reframe._primary(faces, ref_cx)


def _busiest_window(energy, ww, wh):
    """(x, y) of the ww x wh window of motion map `energy` with the most motion, or None when
    nothing moves."""
    import numpy as np
    mh, mw = energy.shape
    ww, wh = max(1, min(mw, int(round(ww)))), max(1, min(mh, int(round(wh))))
    if float(energy.sum()) <= 1e-6:
        return None
    ii = np.pad(energy.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = ii[wh:, ww:] - ii[:-wh, ww:] - ii[wh:, :-ww] + ii[:-wh, :-ww]
    y, x = np.unravel_index(int(np.argmax(s)), s.shape)
    return int(x), int(y)


def zoom_shots(cuts, dur, min_s=DV_MIN_SHOT_S):
    """The shots [(t0, t1)] the zoom is planned on: reframe._intervals with a shot shorter than
    `min_s` joined to the one before it (the first to the one after it)."""
    out = []
    for a_, b_ in reframe._intervals(cuts, float(dur)):
        if out and (b_ - a_ < min_s or out[-1][1] - out[-1][0] < min_s):
            out[-1] = (out[-1][0], b_)
        else:
            out.append((a_, b_))
    return out


def dual_view_zoom_plan(samples, grays, cuts, dur, src_w, src_h, bh, speaker=None, mask=None,
                        cam=None):
    """The zoom panel of a dual view without a facecam: {"zoom": {"w", "h"}, "zoom_segments":
    [{"t0", "t1", "x", "y", "cut", "how"}]} src px. Pure (tests feed synthetic samples).

    `samples` [{"t", "faces"}] (reframe.detect_faces), `grays` the same frames as float gray
    arrays (any one size), `cuts` the shot cuts (clip s), `speaker` _speaker_windows, `mask` a
    0/1 array the size of a gray: text the motion map ignores (a scrolling chat, a HUD counter),
    `cam` a facecam box (normalized) that does not hold all clip: its face is followed when shown.

    Per shot (zoom_shots): a face shot (the speaker's face in >= DV_FACE_SHARE of its samples) zooms on the
    face (re-locked every reframe.DRIFT_STEP s when it drifts, as track_crops); any other shot on
    its busiest window (mean frame difference); a shot without either holds its neighbour's
    window. One zoom size for the clip (an ffmpeg crop cannot resize): the face shots' face at
    DV_FACE_FRAC of the panel, else DV_MOTION_FRAC of the frame; never more than DV_MOTION_FRAC of
    the frame (a real zoom) and never upscaled past DV_MAX_UPSCALE."""
    import numpy as np
    import statistics
    ar = OUT_W / float(bh)
    dur = float(dur)
    spans = zoom_shots(cuts, dur)
    order = sorted(range(len(samples)), key=lambda i: samples[i]["t"])
    picks, ref = {}, None
    for i in order:
        f = _target_face(samples[i].get("faces") or [], samples[i]["t"], speaker, ref, cam)
        if f:
            picks[i] = f
            ref = f["cx"]
    segs = []                                         # [t0, t1, how, data, cut]
    for k, (t0, t1) in enumerate(spans):
        last = k == len(spans) - 1
        idx = [i for i in order if t0 <= samples[i]["t"] < t1 or (last and samples[i]["t"] >= t1)]
        pts = [(samples[i]["t"], picks[i]) for i in idx if i in picks]
        if idx and len(pts) >= DV_FACE_SHARE * len(idx):
            cxs = [f["cx"] for _t, f in pts]
            subs = [(t0, t1)]
            if t1 - t0 > reframe.DRIFT_MIN_SHOT and max(cxs) - min(cxs) > reframe.DRIFT_MAX:
                subs, a_ = [], t0
                while a_ < t1:
                    subs.append((a_, min(a_ + reframe.DRIFT_STEP, t1)))
                    a_ += reframe.DRIFT_STEP
            for j, (a_, b_) in enumerate(subs):
                sub = [f for t, f in pts if a_ <= t < b_ or (b_ == t1 and t >= t1)]
                face = ({key: statistics.median(f.get(key, f["h"] * 0.56) for f in sub)
                         for key in ("cx", "cy", "h", "w")} if sub else None)
                segs.append([a_, b_, "face" if face else "hold", face, j == 0])
            continue
        gi = [i for i in idx if i < len(grays)]
        diffs = [np.abs(grays[q] - grays[p]) for p, q in zip(gi, gi[1:])]
        energy = np.mean(diffs, axis=0) if diffs else None
        if energy is not None and mask is not None:
            energy = energy * (1.0 - mask)
        segs.append([t0, t1, "motion" if energy is not None else "hold", energy, True])
    face_hs = [(s[1] - s[0], s[3]["h"]) for s in segs if s[2] == "face"]
    if face_hs:
        total, acc = sum(d for d, _h in face_hs), 0.0
        for d, h in sorted(face_hs, key=lambda p: p[1]):    # duration-weighted median face
            acc += d
            if acc >= total / 2:
                zh = h * src_h / DV_FACE_FRAC
                break
    else:
        zh = DV_MOTION_FRAC * src_h
    zh = min(float(src_h), max(min(zh, DV_MOTION_FRAC * src_h), bh / DV_MAX_UPSCALE))
    zw = zh * ar
    if zw > src_w:
        zw, zh = float(src_w), src_w / ar
    zw, zh = _even(round(zw)), _even(round(zh))
    out = []
    for t0, t1, how, data, cut in segs:
        if how == "face":
            x = data["cx"] * src_w - zw / 2.0
            y = data["cy"] * src_h - DV_FACE_Y * zh
        elif how == "motion":
            k = data.shape[1] / float(src_w)
            win = _busiest_window(data, zw * k, zh * k)
            if win is None:
                how, x, y = "hold", None, None
            else:
                x, y = win[0] / k, win[1] / k
        else:
            x = y = None
        out.append({"t0": round(t0, 3), "t1": round(t1, 3), "x": x, "y": y, "cut": bool(cut),
                    "how": how})
        if how == "face":           # what clean_zoom keeps whole when it moves the window
            out[-1]["faces"] = [[round(float(data[k]), 4) for k in ("cx", "cy", "w", "h")]]
    # a shot with nothing to follow holds the window before it, else the one after, else the centre
    prev = None
    for s in out:
        if s["x"] is None and prev is not None:
            s["x"], s["y"] = prev
        prev = (s["x"], s["y"]) if s["x"] is not None else prev
    nxt = ((src_w - zw) / 2.0, (src_h - zh) / 2.0)
    for s in reversed(out):
        if s["x"] is None:
            s["x"], s["y"] = nxt
        nxt = (s["x"], s["y"])
    segs_out = []
    for s in out:
        s["x"] = _even(max(0, min(src_w - zw, round(s["x"]))))
        s["y"] = _even(max(0, min(src_h - zh, round(s["y"]))))
        if segs_out and abs(s["x"] - segs_out[-1]["x"]) < reframe.MERGE_TOL * src_w \
                and abs(s["y"] - segs_out[-1]["y"]) < reframe.MERGE_TOL * src_h:
            segs_out[-1]["t1"] = s["t1"]
            if s.get("faces"):
                segs_out[-1]["faces"] = segs_out[-1].get("faces", []) + s["faces"]
            continue
        segs_out.append(s)
    if segs_out:
        segs_out[0]["t0"], segs_out[-1]["t1"] = 0.0, round(dur, 3)
    return {"zoom": {"w": zw, "h": zh}, "zoom_segments": segs_out}


DV_ZOOM_SIZES = (1.0, 0.9, 0.8, 0.7, 1.12, 1.25, 1.4, 0.6)   # zoom heights tried (x the
                          # planned one, nearest first) when the planned window cuts a plate


def _zoom_face_box(f, src_w, src_h):
    """A zoom segment's face [cx, cy, w, h] (normalized) as the src-px box the window keeps."""
    return cropcheck.face_box({"cx": f[0], "cy": f[1], "w": f[2], "h": f[3]}, src_w, src_h)


def clean_zoom(plan, plates, src_w, src_h, bh, faces=None):
    """a dual view's zoom windows keep every plate (keep_plates boxes) whole or out.
    10-18..25: the gate does not read the zoom panel, and the eye-check rejected 31 of 72
    gate-passed gaming / streamers dual views for edge_slice (1337 / 1290: the face zoom cut
    the donation alert and chat over summit1g's cam; 1341 / 1283: the sponsor bugs and a
    broadcast's own cam inset). A window that cuts one is moved (nearest clean position that
    keeps the segment's faces whole), at the planned size first, then the other DV_ZOOM_SIZES
    (one size for the clip: an ffmpeg crop cannot resize), between DV_MAX_UPSCALE and
    DV_MOTION_FRAC of the frame. A moved window also keeps whole or leaves out every face
    `faces(t0, t1)` returns for its segment (src-px boxes; 10-18 clip 1290: the move that kept
    the chat whole cut summit1g's face at the panel's bottom edge). In place; returns the number
    of windows moved (0: clean as planned), or None when no size gives every window a clean
    place (no clean dual view)."""
    z, segs = plan["zoom"], plan.get("zoom_segments") or []
    boxes = [p["box"] if isinstance(p, dict) else p for p in plates or []]
    zw, zh = int(z["w"]), int(z["h"])

    def clean_at(seg, w, h):
        return not any(cropcheck.cuts([seg["x"], seg["y"], seg["x"] + w, seg["y"] + h], b)
                       for b in boxes)
    if not boxes or all(clean_at(sg, zw, zh) for sg in segs):
        return 0
    if z.get("fit") == "inside" or plan.get("bottom") == "cam":
        return None                     # a facecam crop is the cam's own box: not moved
    ar = zw / float(zh)
    h_lo = bh / DV_MAX_UPSCALE
    h_hi = min(DV_MOTION_FRAC * src_h, float(src_h), src_w / ar)
    sizes = []
    for k in DV_ZOOM_SIZES:
        h = _even(round(min(max(zh * k, h_lo), h_hi)))
        if h not in sizes:
            sizes.append(h)
    for h in sizes:
        w = min(_even(round(h * ar)), src_w - src_w % 2)
        moved, places = 0, []
        for sg in segs:
            own = [_zoom_face_box(f, src_w, src_h) for f in sg.get("faces") or []]
            keep = [min(b[0] for b in own), min(b[1] for b in own), max(b[2] for b in own),
                    max(b[3] for b in own)] if own else None
            cx, cy = sg["x"] + zw / 2.0, sg["y"] + zh / 2.0
            if keep is None and w == zw and clean_at(sg, w, h):
                places.append((sg["x"], sg["y"]))
                continue
            seg_faces = [list(map(int, b)) for b in (faces(sg["t0"], sg["t1"]) if faces else [])
                         if keep is None or not (b[0] >= keep[0] and b[2] <= keep[2]
                                                 and b[1] >= keep[1] and b[3] <= keep[3])]
            pos = cropcheck.find_window(w, h, src_w, src_h, boxes + seg_faces, contain=keep,
                                        prefer=(cx - w / 2.0, cy - h / 2.0))
            if pos is None:
                break
            moved += (pos[0], pos[1]) != (sg["x"], sg["y"]) or w != zw
            places.append(pos)
        else:
            z.update(w=w, h=h)
            for sg, (x, y) in zip(segs, places):
                sg["x"], sg["y"] = int(x), int(y)
            return moved
    return None


def _text_mask(sc, shape, src_w):
    """0/1 float array of `shape` over the scan's graphics and text panels (padded DV_TEXT_PAD)."""
    import numpy as np
    m = np.zeros(shape, np.float32)
    k = shape[1] / float(src_w)
    for b in [g["box"] for g in sc.get("graphics") or []] + list(sc.get("panels") or []):
        x0, y0, x1, y1 = [int(round(v * k)) for v in (b[0] - DV_TEXT_PAD, b[1] - DV_TEXT_PAD,
                                                       b[2] + DV_TEXT_PAD, b[3] + DV_TEXT_PAD)]
        m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 1.0
    return m


def _dual_view_samples(src, a, src_w, src_h):
    """(samples [{"t", "faces"}], gray motion frames, shot cuts, text mask) of the window at
    DV_SAMPLE_FPS: what the cam check and the zoom are planned on."""
    import cv2
    import numpy as np
    samples, hists, grays = [], [], []
    mh = max(2, int(round(DV_MOTION_W * src_h / float(src_w))))
    for t, fr in reframe.iter_frames(str(src), a.start, a.dur, fps=DV_SAMPLE_FPS,
                                     width=DV_SAMPLE_W, src_size=(src_w, src_h)):
        samples.append({"t": t, "faces": reframe.detect_faces(fr)})
        hists.append((t, reframe._hist(fr)))
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        grays.append(cv2.resize(g, (DV_MOTION_W, mh), interpolation=cv2.INTER_AREA)
                     .astype(np.float32))
    try:
        mask = _text_mask(_scan(src, a, (src_w, src_h)), (mh, DV_MOTION_W), src_w)
    except Exception as e:        # noqa: BLE001 - the motion map then reads the text too
        log(f"dual view: no text mask ({type(e).__name__}: {e})")
        mask = None
    return samples, grays, reframe._cut_times(hists), mask


DV_BAR_LUMA = 24.0        # a source bar (pillar / letterbox) is near black ...
DV_BAR_MIN = 0.01         # ... at least this share of its side (thinner is the encoder's edge) ...
DV_BAR_MAX = 0.2          # ... and at most this (more is picture, not a bar)
DV_BAR_FRAMES = 24        # source frames the bars are read on ...
DV_BAR_FRAME_FRAC = 0.5   # ... a bar in at least this share of them is trimmed (the gate fails a
                          # band in 60 % of its frames; a logo over the bar now and then hides it
                          # from a frame or two: 10-21 576)


def bar_depths(grays):
    """{edge: px} of the near-black bars a source carries on its edges (a pillarbox or letterbox
    baked into the video), read on gray frames (h x w, one size): the gate's flat-band rule
    (edgebands.flat_edge_depths, persistent over DV_BAR_FRAME_FRAC of the frames) whose band is
    also dark (<= DV_BAR_LUMA) and DV_BAR_MIN-DV_BAR_MAX of its side; 0 for any other edge."""
    import numpy as np
    if not grays:
        return {e: 0 for e in edgebands.EDGES}
    h, w = grays[0].shape[:2]
    per = []
    for g in grays:
        g = np.asarray(g, dtype=np.float32)
        d = edgebands.flat_edge_depths(g) if float(g.std()) >= 8.0 else \
            {e: 0 for e in edgebands.EDGES}
        for e, px in list(d.items()):
            band = {"top": g[:px], "bottom": g[h - px:], "left": g[:, :px],
                    "right": g[:, w - px:]}[e] if px else None
            if not px or band.size == 0 or float(band.mean()) > DV_BAR_LUMA:
                d[e] = 0
        per.append(d)
    out = {}
    for e, px in edgebands.persistent_bands(per, frame_frac=DV_BAR_FRAME_FRAC).items():
        side = h if e in ("top", "bottom") else w
        out[e] = int(px) if DV_BAR_MIN * side <= px <= DV_BAR_MAX * side else 0
    return out


def source_bars(src, a, src_w, src_h):
    """{"x", "y", "w", "h"} src px: the source frame inside its near-black bars (bar_depths on
    DV_BAR_FRAMES frames of the window), or None when it has none."""
    fps = max(0.2, DV_BAR_FRAMES / max(1.0, float(a.dur)))
    grays = [edgebands.to_gray(fr) for _t, fr in reframe.iter_frames(
        str(src), a.start, a.dur, fps=fps, width=DV_SAMPLE_W, src_size=(src_w, src_h))]
    d = bar_depths(grays)
    if not any(d.values()):
        return None
    k = src_w / float(DV_SAMPLE_W)
    # src px off each edge: the bar (read on the sample frames) and 2 px of its blurred edge
    cut = {e: int(math.ceil(px * k)) + 2 if px else 0 for e, px in d.items()}
    x0, y0 = cut["left"] + cut["left"] % 2, cut["top"] + cut["top"] % 2
    w, h = _even(src_w - x0 - cut["right"]), _even(src_h - y0 - cut["bottom"])
    if w < 0.6 * src_w or h < 0.6 * src_h:
        return None
    return {"x": x0, "y": y0, "w": w, "h": h, "bars": d}


def plan_dual_view(src, a, prior=None):
    """The dual view plan of this window ({"layout": "dual_view", "header_h", "panel_h",
    "seam_y", "bottom": "cam" | "zoom", "zoom": {"w", "h"[, "fit"]}, "zoom_segments"}), or None
    when the frame leaves no room for a zoom panel. `prior` = the plan it replaces (its cam and
    speaker crop are reused). The bottom is the facecam when its face is an inset's
    (small_cam_face) and it holds a face in every shot (cam_holds), else the zoom
    (dual_view_zoom_plan), following that cam's face where it shows.
    Poker has no facecam: its zoom is the speaking player or the action."""
    prior = prior or {}
    if prior.get("src_w") and prior.get("src_h"):
        sw, sh = int(prior["src_w"]), int(prior["src_h"])
    else:
        sw, sh = reframe.probe_size(str(src))
    # a tech / finance source with a pillarbox or letterbox baked in shows its
    # frame inside the bars (10-21 576: a 60 px black pillar on the right of a 1280 px frame, read
    # by the gate as a flat band on the top panel); the HUD niches' dual view is unchanged
    inner = None
    if dual_view_fallback(a) and sw and sh:
        try:
            inner = source_bars(src, a, sw, sh)
        except Exception as e:    # noqa: BLE001 - the whole frame, as before
            log(f"dual view: bars unreadable ({type(e).__name__}: {e})")
    fw, fh = (inner["w"], inner["h"]) if inner else (sw, sh)
    geo = dual_view_geometry(fw, fh) if fw and fh else None
    if geo is None:
        log(f"dual view: a {fw}x{fh} frame fit to width leaves no zoom panel")
        return None
    hh, ph, seam = geo
    bh = OUT_H - seam
    plan = {"layout": DUAL_VIEW, "src_w": sw, "src_h": sh, "header_h": hh, "panel_h": ph,
            "seam_y": seam}
    if inner:
        plan["frame"] = inner
        log(f"dual view: the frame inside its source bars {inner['bars']}: "
            f"{inner['w']}x{inner['h']} at {inner['x']},{inner['y']}")
    cam = None if _is_poker(a) else _cam_candidate(src, a, prior, bh)
    if cam and not small_cam_face(cam):
        log(f"dual view: the cam {cam['cam_box']} face {cam.get('face_median')} is no inset's; zoom")
        cam = None
    # a stream with a facecam shows it below only inside the cam's own box
    # (10-26: 7 of 16 gaming eye-check rejects, every one a cam panel or face zoom that reached
    # past the cam into the game). A cam whose box was read but cannot fill the panel clean: no
    # dual view, since a face zoom would reach into the game the same way. A box that could not
    # be read changes nothing but a planned box too tall to be the cam's (bound_cam).
    ov = None
    if not _is_poker(a) and getattr(a, "niche", None):   # niche clips: campaign renders as before
        try:
            ov = cam_overlay(src, a, (sw, sh))
        except Exception as e:    # noqa: BLE001 - unreadable: the dual view as before
            log(f"dual view: facecam box unreadable ({type(e).__name__}: {e})")
    if ov and not ov.get("tile"):
        if cam:
            nb, why = bound_cam(cam["cam_box"], a, src, (sw, sh))
            if why:
                log(f"dual view: {why}")
            cam = dict(cam, cam_box=nb) if nb else None
        ov = None              # an unread box is no proof of a facecam (posted IRL streamers)
    elif ov:
        if cam and _holds(cam["cam_box"], _cam_pt(ov, (sw, sh))):
            nb, why = bound_cam(cam["cam_box"], a, src, (sw, sh))
            if why:
                log(f"dual view: {why}")
            cam = dict(cam, cam_box=nb)
        elif not cam:
            f = ov["face"]
            cam = {"cam_box": _even_in(ov["tile"]), "face_median": [f["cx"], f["cy"], f["h"]]}
            log(f"dual view: the facecam's own box {cam['cam_box']}")
        else:
            ov = None                   # the planned cam is another face's: as before
    samples, grays, cuts, mask = _dual_view_samples(src, a, sw, sh)
    plates, items = _zoom_plates(src, a, sw, sh)
    box = [cam["cam_box"][0] / sw, cam["cam_box"][1] / sh, cam["cam_box"][2] / sw,
           cam["cam_box"][3] / sh] if cam else None
    if cam and cam_holds(samples, cuts, a.dur, box):
        z = dual_view_cam_zoom(cam["cam_box"], cam["face_median"], bh, hh, sw, sh)
        cp = dict(plan, bottom="cam", zoom={k: z[k] for k in ("w", "h", "fit") if k in z},
                  zoom_segments=[{"t0": 0.0, "t1": round(float(a.dur), 3), "x": z["x"],
                                  "y": z["y"], "cut": True, "how": "cam"}], **cam)
        if clean_zoom(cp, plates, sw, sh, bh) == 0 or clean_zoom(cp, items, sw, sh, bh) == 0:
            plan.update(cp)
            return plan
        log(f"dual view: the cam crop {cam['cam_box']} cuts a graphic; zoom")
        if ov and ov.get("tile"):
            log("dual view: a face zoom would leave the facecam's box; no dual view")
            return None
    elif cam:
        log(f"dual view: the cam {cam['cam_box']} does not hold a face in every shot; zoom")
        if ov and ov.get("tile"):
            log("dual view: a face zoom would leave the facecam's box; no dual view")
            return None
    z = dual_view_zoom_plan(samples, grays, cuts, a.dur, sw, sh, bh,
                            speaker=_speaker_windows(prior), mask=mask, cam=box)
    plan.update(bottom="zoom", zoom_samples=len(samples), **z)
    if cam:
        plan["cam_box"] = cam["cam_box"]
    import copy
    planned = copy.deepcopy(plan)

    def faces(t0, t1):           # the faces sampled in a zoom segment, as kept boxes
        return [cropcheck.face_box(f, sw, sh) for smp in samples
                if t0 - 0.25 <= smp["t"] <= t1 + 0.25 for f in smp.get("faces") or []]
    moved = clean_zoom(plan, plates, sw, sh, bh, faces=faces)
    motion = any(sg.get("how") == "motion" for sg in plan.get("zoom_segments") or [])
    if moved is None and not motion:   # every plate whole is impossible: the gate's text items
        plan.clear()                   # whole (a face zoom: the face is what it shows)
        plan.update(planned)
        moved = clean_zoom(plan, items, sw, sh, bh, faces=faces)
    elif moved is None:
        log(f"dual view: a zoom on the action shows the frame's content itself, and no window "
            f"keeps its {len(plates)} plate(s) whole (scope 2: 891 cut an article's lines)")
    if moved is None:
        log(f"dual view: no zoom window of any size keeps the frame's {len(items)} text "
            f"item(s) whole; no dual view")
        return None
    if moved:
        log(f"dual view: {moved} zoom window(s) moved off the frame's graphics "
            f"({plan['zoom']['w']}x{plan['zoom']['h']} src px)")
    return plan


def _zoom_plates(src, a, sw, sh):
    """(keep_plates, item_plates) of the window for its zoom panel (padded LADDER_TEXT_PAD);
    ([], []) when the frame cannot be scanned (the zoom is planned as before)."""
    try:
        sc, words = _scan(src, a, (sw, sh)), _ocr(src, a, (sw, sh))
        return (keep_plates(sc, words, sw, sh, pad=LADDER_TEXT_PAD),
                item_plates(sc, words, pad=LADDER_TEXT_PAD))
    except Exception as e:        # noqa: BLE001 - an unscanned frame: the zoom as planned
        log(f"dual view: graphics unreadable ({type(e).__name__}: {e}); zoom as planned")
        return [], []


def to_dual_view(plan, a, src, why, tried=None, guard=None):
    """Replace `plan` (in place) with the dual view and return its report (stored as
    plan["banner_guard"]), or None (no room for a zoom panel, or planning failed: the plan
    stands)."""
    try:
        dv = plan_dual_view(src, a, prior=plan)
    except Exception as e:        # noqa: BLE001 - the planned layout stands
        log(f"dual view: planning failed ({type(e).__name__}: {e})")
        return None
    if dv is None:
        return None
    rep = {"action": "dual_view", "layout": DUAL_VIEW, "why": why, "tried": list(tried or []),
           "replaced": plan.get("layout"), "bottom": dv["bottom"],
           "previous": (guard or {}).get("action")}
    plan.clear()
    plan.update(dv)
    plan["banner_guard"] = rep
    log(f"dual view ({dv['bottom']} below, {dv['zoom']['w']}x{dv['zoom']['h']} src px) instead "
        f"of {rep['replaced']}: {why}")
    return rep


# ---------------------------------------------------------------- streamer split (yield 11-09)

# 11-09 01:55 prep: 15 of 20 xQc GTA RP clips (2844 / 2849 / 2787 / 2793 / 2792 before them) were
# rejected black_band: the dual view's blurred hook header over the frame fit to width over the
# cam. Each had planned a gamestack whose game window sliced the chat or the HUD (2992: a 578 px
# window through the speedometer), and the dual view took its place. The layout standard:
# full-bleed following the speaker; split screen only for a streamer with a real facecam, both
# panels filling the frame. The streamer split is that split, rendered as the gamestack: on top
# the game window, sized and placed so every plate (the gate's text items, OCR'd HUD plates, the
# docked chat, a corner logo bug) is whole in it or out of it and the cam stays out of it; below
# the cam's own box cover-cropped to a panel whose height follows the box's aspect, so the cam
# fills it. Nothing is blurred.
SPLIT_CAM_MIN_H = 576        # cam panel height range (canvas px): 2855's passing gamestack is 576
SPLIT_CAM_MAX_H = 864        # ... the game panel keeps at least 1056
SPLIT_CAM_STEP = 48
SPLIT_CAM_MAX_UPSCALE = 6.0  # the cam panel shows at least (panel height / this) source rows
                             # (2855's cam: 576 / 130 = 4.4x, an eye-check pass; a 212 px wide
                             # cam fills the 1080 px panel at 5.1x at the least: 2876)
SPLIT_CAM_MIN_FRAC = 0.7     # a cam window keeps at least this share of the cover crop's height
SPLIT_GAME_MIN_H = 0.7       # the game window keeps at least this share of the source height
SPLIT_GAME_STEPS = 8         # game window heights tried, the full height down to that
SPLIT_SCREEN_NICHES = ("tech",)   # class 4: a tech screen with a streamer's facecam splits too
SPLIT_CENTRE = 0.2                # the game window first keeps the frame's centre (a third-person
                                  # game's car or player: 2992's car sat on the left edge of the
                                  # widest clean window) at least this share of its width inside
SPLIT_CENTRE_TOP = 0.2            # ... starting in this top share of the frame: a centred window
                                  # pushed under the scene's signs shows the floor (2910: y 218,
                                  # the robber and the hostage cut at the chest)
SPLIT_HOOK_ROWS = (0.2, 0.48)     # a screen's game window first keeps these rows (of its height:
                                  # canvas ~230-500 of the panel, the hook's place) free of plates
                                  # (2876: the RuneLite XP column left no row clear for the hook)


def _face_keep(face, src_w, src_h):
    """The src-px box a cam panel keeps whole of a cam face (normalized [cx, cy, h]): the face
    grown to hair and chin (cropcheck.face_box), its width taken as 0.8 face heights."""
    fh = float(face[2]) * src_h
    return cropcheck.face_box({"cx": float(face[0]), "cy": float(face[1]),
                               "w": 0.8 * fh / float(src_w), "h": float(face[2])}, src_w, src_h)


def split_cam_window(box, face, bot_h, plates, src_w, src_h):
    """(w, h, x, y) src px: the cam panel's window inside the cam's box `box` [x0, y0, x1, y1] at
    the panel's aspect (OUT_W / bot_h), the face whole, cutting no plate (src-px boxes): the cover
    crop (dual_view_cam_crop) first, then smaller ones down to SPLIT_CAM_MIN_FRAC of its height.
    None when none fits or the panel would upscale the window past SPLIT_CAM_MAX_UPSCALE."""
    ar = OUT_W / float(bot_h)
    w0, h0, x0, y0 = dual_view_cam_crop(box, face, ar, src_w, src_h)
    bx0, by0, bx1, by1 = [int(v) for v in box]
    bw, bh = bx1 - bx0, by1 - by0
    rel = [[p[0] - bx0, p[1] - by0, p[2] - bx0, p[3] - by0] for p in plates
           if p[0] < bx1 and p[2] > bx0 and p[1] < by1 and p[3] > by0]
    keep = None
    if face:
        k = _face_keep(face, src_w, src_h)
        keep = [max(0.0, k[0] - bx0), max(0.0, k[1] - by0), min(bw, k[2] - bx0),
                min(bh, k[3] - by0)]
    for i in range(6):
        h = _even(round(h0 * (1.0 - (1.0 - SPLIT_CAM_MIN_FRAC) * i / 5.0)))
        w = _even(round(h * ar))
        if h < 2 or w < 2 or bot_h / float(h) > SPLIT_CAM_MAX_UPSCALE:
            break                                  # a smaller window only upscales more
        if w > bw or h > bh:
            continue
        prefer = (x0 - bx0 + (w0 - w) / 2.0, y0 - by0 + DV_FACE_Y * (h0 - h))
        pos = cropcheck.find_window(w, h, bw, bh, rel, contain=keep, prefer=prefer)
        if pos is not None:
            return w, h, _even(bx0 + pos[0]), _even(by0 + pos[1])
    return None


def split_game_window(top_h, guard, plates, src_w, src_h, prefer_x=None, hook_rows=None,
                      centre=None):
    """(w, h, x, y) src px: the game panel's window at the panel's aspect (OUT_W / top_h), the
    largest from the full source height down to SPLIT_GAME_MIN_H of it, that cuts no plate
    (src-px boxes: whole or out) and leaves the cam's `guard` box [x0, y0, x1, y1] out, nearest
    `prefer_x` (its centre; the frame's centre without one), vertically centred; with
    `hook_rows` (r0, r1) no plate in rows [y + r0 h, y + r1 h] of it either, and with `centre`
    (x, frac, top) that column at least frac of its width inside it, its top row at most `top` of
    the source height down. None when no window is clean."""
    import numpy as np
    ar = OUT_W / float(top_h)
    gx0, gy0, gx1, gy1 = [int(v) for v in guard]
    for k in range(SPLIT_GAME_STEPS + 1):
        h = _even(src_h - (src_h - SPLIT_GAME_MIN_H * src_h) * k / float(SPLIT_GAME_STEPS))
        w = _even(round(h * ar))
        if w > src_w:
            w = src_w - src_w % 2
            h = _even(round(w / ar))
        xs = np.arange(0, src_w - w + 1, 2)
        px = (src_w / 2.0 if prefer_x is None else float(prefer_x)) - w / 2.0
        py = (src_h - h) / 2.0
        ys = {0, src_h - h, int(py)}
        for b in plates:
            ys |= {b[3] + cropcheck.SLICE_TOL, b[1] - cropcheck.SLICE_TOL - h}
        ys |= {gy1, gy0 - h}
        best = None
        for y in sorted({_even(min(max(v, 0), src_h - h)) for v in ys}, key=lambda v: abs(v - py)):
            if centre and y > centre[2] * src_h:
                continue
            ok = cropcheck.valid_xs(xs, y, w, h, plates)
            if gy1 > y and gy0 < y + h:
                ok &= (xs + w <= gx0) | (xs >= gx1)
            if centre:
                ok &= (xs + centre[1] * w <= centre[0]) & (xs + (1 - centre[1]) * w >= centre[0])
            if hook_rows:
                r0, r1 = y + hook_rows[0] * h, y + hook_rows[1] * h
                for b in plates:
                    if b[3] > r0 and b[1] < r1:
                        ok &= (xs + w <= b[0] + cropcheck.SLICE_TOL) | \
                            (xs >= b[2] - cropcheck.SLICE_TOL)
            if not ok.any():
                continue
            good = xs[ok]
            x = int(good[np.argmin(np.abs(good - px))])
            d = math.hypot(x - px, y - py)
            if best is None or d < best[0]:
                best = (d, x, y)
        if best:
            return w, h, best[1], best[2]
    return None


def _split_cam_box(cam, ov, size):
    """The box the split's cam panel shows: the facecam's own box (cam_overlay's tile) when the
    planned box holds the cam's face; the planned box when that box was not read and it is no
    taller than CAM_BOX_MAX_FACE face heights (plausible_cam), else None; the planned box when the
    window has no recurring cam face to check it against."""
    box = [int(v) for v in cam["cam_box"]]
    if not ov or not _holds(box, _cam_pt(ov, size)):
        return box
    if ov.get("tile"):
        return _even_in(ov["tile"])
    return box if plausible_cam(box, ov, size) else None


def plan_streamer_split(src, a, prior=None):
    """The streamer split plan of this window (a gamestack plan with "streamer_split": True and
    the game window as "game_box"), or None: no facecam inset (a small recurring face in a box, a
    face that shows in every shot), or no clean game window / cam window for any cam panel height
    (SPLIT_CAM_MIN_H..SPLIT_CAM_MAX_H, the cam box's own aspect first). `prior` = the plan it
    replaces (its cam is reused)."""
    prior = prior or {}
    if prior.get("src_w") and prior.get("src_h"):
        sw, sh = int(prior["src_w"]), int(prior["src_h"])
    else:
        sw, sh = reframe.probe_size(str(src))
    cam = _cam_candidate(src, a, prior, SPLIT_CAM_MAX_H)
    if not cam or not small_cam_face(cam):
        log(f"streamer split: no facecam inset ({(cam or {}).get('face_median')})")
        return None
    try:
        ov = cam_overlay(src, a, (sw, sh))
    except Exception as e:        # noqa: BLE001 - unread: the planned cam box
        log(f"streamer split: facecam box unreadable ({type(e).__name__}: {e})")
        ov = None
    box = _split_cam_box(cam, ov, (sw, sh))
    if box is None or box[2] - box[0] < 16 or box[3] - box[1] < 16:
        log(f"streamer split: the cam box {cam['cam_box']} is not the facecam's")
        return None
    if getattr(a, "niche", None) in STACK_NICHES and _gamestack_cfg().get("v2") \
            and not gamelayout.is_inset(box, sw, sh):
        # gamestack v2: a face inside the scene (an IRL shot's person) is no
        # facecam; its split would stack the scene's own face as an 864 px cam panel
        log(f"streamer split: the cam box {box} sits inside the scene (no frame edge)")
        return None
    face = cam.get("face_median")
    samples, _grays, cuts, _mask = _dual_view_samples(src, a, sw, sh)
    if not cam_holds(samples, cuts, a.dur, [box[0] / sw, box[1] / sh, box[2] / sw, box[3] / sh]):
        log(f"streamer split: the cam {box} does not hold a face in every shot")
        return None
    sc, words = _scan(src, a, (sw, sh)), _ocr(src, a, (sw, sh))
    kp = keep_plates(sc, words, sw, sh, pad=LADDER_TEXT_PAD)
    plates = [p["box"] for p in kp] + [list(b) for b in _overlays(src, a, (sw, sh))]
    # inside the facecam's own box (read: a tile) everything is the cam's picture (an OCR read
    # of xQc's shirt merged with the chat under the cam: 2910); a box that was not read (a cut-out
    # streamer over a screen: 2876) keeps the screen's plates whole or out, a shirt print aside
    tile = bool(ov and ov.get("tile")) and box == _even_in(ov["tile"])
    cam_plates = [] if tile else [p["box"] for p in off_body(kp, sc.get("faces"), sw, sh)] + \
        [list(b) for b in _overlays(src, a, (sw, sh))]
    g = int(round(GAMESTACK_CAM_GUARD_FRAC * sw))
    guard = [max(0, box[0] - g), max(0, box[1] - g), min(sw, box[2] + g), min(sh, box[3] + g)]
    prefer_x = None
    try:
        prefer_x = facecam.activity_centroid(str(src), a.start, a.dur, guard_box_px=guard)
    except Exception as e:        # noqa: BLE001 - the free side's centre then
        log(f"streamer split: activity unreadable ({type(e).__name__}: {e})")
    if prefer_x is None:
        prefer_x = (guard[2] + sw) / 2.0 if guard[0] + guard[2] < sw else guard[0] / 2.0
    want = _even(min(max(OUT_W * (box[3] - box[1]) / float(box[2] - box[0]), SPLIT_CAM_MIN_H),
                     SPLIT_CAM_MAX_H))
    heights = sorted(set(range(SPLIT_CAM_MIN_H, SPLIT_CAM_MAX_H + 1, SPLIT_CAM_STEP)) | {want},
                     key=lambda v: (abs(v - want), -v))
    screen = getattr(a, "niche", None) in SPLIT_SCREEN_NICHES
    centre = (sw / 2.0, SPLIT_CENTRE, SPLIT_CENTRE_TOP)
    sweeps = [(r, c) for r in ([SPLIT_HOOK_ROWS, None] if screen else [None])
              for c in (centre, None)]
    for (hook_rows, ctr), bot_h in [(sw_, b) for sw_ in sweeps for b in heights]:
        cw = split_cam_window(box, face, bot_h, cam_plates, sw, sh)
        if cw is None:
            continue
        gw = split_game_window(OUT_H - bot_h, guard, plates, sw, sh,
                               sw / 2.0 if ctr else prefer_x, hook_rows, ctr)
        if gw is None:
            continue
        log(f"streamer split: game {gw[0]}x{gw[1]} at {gw[2]},{gw[3]} over the cam "
            f"{cw[0]}x{cw[1]} at {cw[2]},{cw[3]} (cam panel {bot_h} px; {len(plates)} plate(s) "
            f"whole or out)")
        return {"layout": "gamestack", "streamer_split": True, "src_w": sw, "src_h": sh,
                "top_h": OUT_H - bot_h, "game": [gw[0], gw[2], False], "game_box": list(gw),
                "cam": [cw[0], cw[1], cw[2], cw[3], False], "cam_box": box, "raw_cam_box": guard,
                "face_median": face, "game_mode": "split", "channel": getattr(a, "channel", None)}
    log(f"streamer split: no cam window in {box} and game window ({len(plates)} plate(s) whole or "
        f"out, the cam out) for any cam panel height")
    return None


def to_streamer_split(plan, a, src, why, tried=None, guard=None):
    """Replace `plan` (in place) with the streamer split and return its report (stored as
    plan["banner_guard"]), or None (no facecam inset or no clean split: the plan stands)."""
    try:
        sp = plan_streamer_split(src, a, prior=plan)
    except Exception as e:        # noqa: BLE001 - the planned layout stands
        log(f"streamer split: planning failed ({type(e).__name__}: {e})")
        return None
    if sp is None:
        return None
    rep = {"action": "streamer_split", "layout": "gamestack", "why": why,
           "tried": list(tried or []), "replaced": plan.get("layout"),
           "previous": (guard or {}).get("action")}
    plan.clear()
    plan.update(sp)
    plan["banner_guard"] = rep
    log(f"streamer split instead of {rep['replaced']}: {why}")
    return rep


def stack_fallback(plan, a, src, why, tried=None, guard=None):
    """The stacking niches' layout for a planned one that slices: the streamer split for a daily
    clip (no_band_stack), the dual view for campaign and manual renders (as before)."""
    if no_band_stack(a):
        return to_streamer_split(plan, a, src, why, tried, guard)
    return to_dual_view(plan, a, src, why, tried, guard)


def screen_cam_split(plan, a, src):
    """render yield 11-09 class 4: a tech clip of a screen with a streamer's facecam (10-30 2876 /
    2875: the speaker crop followed the face into the RuneLite client's corner, the client sliced
    and the speaker a small corner cam) shows the speaker as the subject: the streamer split, the
    screen on top and the cam below. Only a screen source (screen_content: the source's kind says
    screen and every face is small) with a recurring cam face (cam_overlay). Returns the split's
    report, or None (the single view as before)."""
    if getattr(a, "niche", None) not in SPLIT_SCREEN_NICHES or plan.get("manual") \
            or not plan.get("src_w") or plan.get("layout") not in TRACK_LIKE_LAYOUTS:
        return None
    sw, sh = int(plan["src_w"]), int(plan["src_h"])
    try:
        sc = _scan(src, a, (sw, sh))
        ov = cam_overlay(src, a, (sw, sh)) if screen_content(a, sc) else None
    except Exception as e:        # noqa: BLE001 - the single view then
        log(f"screen cam split: unreadable ({type(e).__name__}: {e})")
        return None
    if not ov:
        return None
    return to_streamer_split(plan, a, src, "a screen with a streamer's facecam: the speaker is "
                                           "the subject (the screen above, the cam below)")


DV_SPLIT_OVERLAP = 0.5    # a split whose two windows share this much of the smaller one stacks
                          # one picture twice (10-21 clip 309: one camera twice, 70 %,
                          # the second 'speaker' a carved pumpkin; its wall logo and socials bar
                          # cut at the edges, text the OCR cannot read)


def split_overlap(plan):
    """Share of the smaller split window the other one also shows (0.0 for any other plan)."""
    sp = plan.get("split") if plan.get("layout") == "split" else None
    if not sp or "left" not in sp or "right" not in sp:
        return 0.0
    l, r = sp["left"], sp["right"]
    iw = min(l["x"] + l["w"], r["x"] + r["w"]) - max(l["x"], r["x"])
    ih = min(l["y"] + l["h"], r["y"] + r["h"]) - max(l["y"], r["y"])
    small = min(l["w"] * l["h"], r["w"] * r["h"])
    return max(0.0, iw) * max(0.0, ih) / small if small > 0 else 0.0


def shown_defect(plan, a, src):
    """Why a planned layout other than a speaker crop should not ship in a HUD niche, or None:
    a window it shows slices burned-in text (the ladder's measure: quality_gate.crop_slices on
    the banner scan + OCR, >= LADDER_MIN_SHARE), or its blurred fill is over the gate's cap (a
    16:9 frame fit to width: 68 %)."""
    if plan.get("src_w") and plan.get("src_h"):
        sw, sh = int(plan["src_w"]), int(plan["src_h"])
    else:
        sw, sh = reframe.probe_size(str(src))
    fill = blur_fill_frac({"src_w": sw, "src_h": sh, **plan}, a, {"width": sw, "height": sh})
    if fill is not None and fill > SCREEN_BOX_MAX_FILL:
        return (f"{plan.get('layout')} is {fill:.0%} blurred fill (gate cap "
                f"{SCREEN_BOX_MAX_FILL:.0%})")
    dup = split_overlap(plan)
    if dup >= DV_SPLIT_OVERLAP:
        return f"the split shows the same picture twice ({dup:.0%} of a window overlaps the other)"
    views = crop_views({"src_w": sw, "src_h": sh, **plan}, a.dur)
    if not views:
        return None
    from clipengine import quality_gate as qg
    sc = _scan(src, a, (sw, sh))
    words = _ocr(src, a, (sw, sh))
    v = qg.crop_slice_verdict(qg.crop_slices(sc, views, a.dur, words=words))
    if v["crop_share"] >= LADDER_MIN_SHARE:
        return (f"{plan.get('layout')} slices burned-in text (on-screen share "
                f"{v['crop_share']:.2f})")
    # the gate fails text cut at the frame / crop edges over its total cap too (10-18
    # clip 1314: a gamestack whose game crop shows a chat overlay running into the frame edge,
    # 1.65), and a HUD plate the OCR's items miss is a sliced graphic to the eye-check
    if v["sliced_share"] > qg.SLICE_SHARE_MAX:
        return (f"{plan.get('layout')} cuts text at the frame / crop edges (on-screen share "
                f"{v['sliced_share']:.2f} > the gate's {qg.SLICE_SHARE_MAX:.2f})")
    if hud_niche(a) and not stack_allowed(a):      # poker's band (see slice_ladder: not 1017)
        share = cropcheck.cut_share(views, keep_plates(sc, words, sw, sh,
                                                       min_frac=cropcheck.PROBE_MIN_FRAC), a.dur)
        if share >= LADDER_MIN_SHARE:
            return f"{plan.get('layout')} cuts HUD plates (on-screen share {share:.2f})"
    return None


SHOT_TEXT_MAX_FRAC = 0.5     # render yield 11-09 class 3: a text event up for more of the clip is
                             # a persistent graphic (the banner guard and the ladder have it)
SHOT_TEXT_H_STEP = 8         # src px: punch-in heights tried below the planned crop's


def _event_faces(sc, views, t0, t1, src_w, src_h):
    """The face boxes (src px, grown to hair and chin: cleancrop's rule) sampled within [t0, t1]
    (+- half a probe step) whose centre lies in a crop view shown then: the speaker the shot
    keeps."""
    out = []
    for t, fs in sc.get("faces") or []:
        if not t0 - 0.5 <= t <= t1 + 0.5:
            continue
        for f in fs:
            cx, cy = f["cx"] * src_w, f["cy"] * src_h
            if not any(v["x"] <= cx <= v["x"] + v["w"] and v["y"] <= cy <= v["y"] + v["h"]
                       for v in views):
                continue
            fh = f["h"] * src_h
            out.append((max(0.0, cx - f["w"] * src_w / 2),
                        max(0.0, cy - (0.5 + cleancrop.HEAD_UP) * fh),
                        min(float(src_w), cx + f["w"] * src_w / 2),
                        min(float(src_h), cy + (0.5 + cleancrop.CHIN_DOWN) * fh)))
    return out


def text_shot(box, faces, cw, ch, cy, src_w, src_h, prefer_cx):
    """The window {"mode": "crop", "w", "h", "x", "y"} (src px, full-bleed 9:16) that shows a text
    event's `box` whole or leaves it out, every face in `faces` whole: the planned crop's size
    first (cw x ch at cy, then any row), then smaller ones (a punch-in) down to
    cleancrop.MIN_CANVAS_W of canvas; nearest `prefer_cx`. None when none does."""
    import numpy as np
    h_min = cleancrop.min_crop_h(src_h)
    win = [0, 0, src_w, src_h]
    tol = cleancrop.SLICE_TOL
    fy = [f[1] for f in faces]
    for h in [int(ch)] + list(range(int(ch) - SHOT_TEXT_H_STEP, h_min - 1, -SHOT_TEXT_H_STEP)):
        h = _even(h)
        w = int(cw) if h == int(ch) else min(_even(round(h * 9 / 16)), src_w - src_w % 2)
        xs = np.arange(0, src_w - w + 1, 2)
        want = int(cy) if h == int(ch) else ((min(fy) - 0.12 * h) if fy else (src_h - h) / 2.0)
        ys = {int(cy), 0, src_h - h, box[1] - tol - h, box[3] + tol}
        ys |= {f[1] - 0.15 * (f[3] - f[1]) for f in faces} | {f[1] for f in faces}
        for y in sorted({_even(min(max(v, 0), src_h - h)) for v in ys},
                        key=lambda v: abs(v - want)):
            ok = cleancrop._valid_x(xs, w, y, h, faces, [], [list(box)], win, src_w)
            if ok.any():
                good = xs[ok]
                x = int(good[np.argmin(np.abs(good + w / 2.0 - prefer_cx))])
                return {"mode": "crop", "w": int(w), "h": int(h), "x": x, "y": int(y)}
    return None


def shot_text_fix(plan, a, src):
    """render yield 11-09 class 3: per text event of the source (textzones.source_text_spans:
    burned-in kinetic captions up for a shot) whose box a speaker crop window cuts, that span is
    shown by the window that holds the text whole or leaves it out with the speaker whole
    (text_shot: the planned size, then a punch-in down to cleancrop.MIN_CANVAS_W), else by the
    frame fit to width (the whole line shown over a blurred fill for that span). When those
    fit-to-width spans would make the clip more blurred fill than the gate takes
    (SCREEN_BOX_MAX_FILL), the clip fails cropcheck (cropcheck.TEXT_SLICE): never a slice.
    plan["text_shots"] = [{"t0", "t1", "mode": "crop" | "fit", ...}]. Daily niche clips with a
    full-bleed speaker crop only (a boxed or stacked layout as before)."""
    if not getattr(a, "niche", None) or plan.get("manual") or not plan.get("src_w") \
            or plan.get("layout") not in TRACK_LIKE_LAYOUTS or not plan.get("segments") \
            or track_boxed(plan):
        return None
    sw, sh = int(plan["src_w"]), int(plan["src_h"])
    try:
        events = textzones.source_text_spans(src, a.start, a.dur, sw, sh)
    except Exception as e:        # noqa: BLE001 - unread: the crop as planned (the gate reads it)
        log(f"source text: unreadable ({type(e).__name__}: {e})")
        return None
    events = [e for e in events if e["t1"] - e["t0"] <= SHOT_TEXT_MAX_FRAC * float(a.dur)]
    if not events:
        return None
    cw, ch, cy = track_crop_window(plan)
    ch = sh if ch == "ih" else int(ch)
    views = crop_views(plan, a.dur)
    sc = _scan(src, a, (sw, sh))
    shots = []
    for ev in events:
        shown = [v for v in views if v["t1"] > ev["t0"] and v["t0"] < ev["t1"]]
        if not any(cropcheck.cuts([v["x"], v["y"], v["x"] + v["w"], v["y"] + v["h"]], ev["box"])
                   for v in shown):
            continue
        faces = _event_faces(sc, shown, ev["t0"], ev["t1"], sw, sh)
        prefer = (sum((f[0] + f[2]) / 2.0 for f in faces) / len(faces) if faces else
                  shown[0]["x"] + shown[0]["w"] / 2.0)
        shot = text_shot(ev["box"], faces, cw, ch, cy, sw, sh, prefer) or {"mode": "fit"}
        shot.update(t0=ev["t0"], t1=ev["t1"], box=list(ev["box"]), text=ev["text"])
        shots.append(shot)
        log(f"source text {ev['text']!r} {ev['box']} at {ev['t0']:.2f}-{ev['t1']:.2f} s: the crop "
            f"would slice it; " + (f"a {shot['w']}x{shot['h']} window at {shot['x']},{shot['y']}"
                                   if shot["mode"] == "crop" else "the frame fit to width"))
    if not shots:
        return None
    span = sum(s["t1"] - s["t0"] for s in shots if s["mode"] == "fit")
    fill = _fit_fill(sw, sh, OUT_W, OUT_H, "width") / (OUT_W * OUT_H) * span / float(a.dur)
    if fill > SCREEN_BOX_MAX_FILL:
        raise RenderError(cropcheck.text_slice_error(
            f"{len(shots)} text shot(s); fitting {span:.1f} s of {float(a.dur):.1f} s to width "
            f"would be {fill:.0%} blurred fill (gate cap {SCREEN_BOX_MAX_FILL:.0%})"))
    plan["text_shots"] = list(plan.get("text_shots") or []) + shots    # after any logo shots
    return shots


def gate_slice_check(plan, a, src):
    """the quality gate's own crop-slice check (quality_gate.crop_slices on the same
    banners.scan + OCR of the window and the same crop views: slice_views / text_edges as the
    gate reads them), run on the chosen plan before the encode. A plan it would fail raises
    RenderError: the clip is rejected without an encode, a gate pass, an eye-check or repair
    rounds (10-18..25: 179 gate edge_slice rejects were rendered first)."""
    if plan.get("manual") or not plan.get("src_w") or not getattr(a, "niche", None):
        return None                   # daily niche clips only: campaign / manual renders as before
    sw, sh = int(plan["src_w"]), int(plan["src_h"])
    views = crop_views(plan, a.dur)
    if not views:
        return None
    from clipengine import quality_gate as qg
    lay = plan.get("layout")
    v = qg.crop_slice_verdict(qg.crop_slices(_scan(src, a, (sw, sh)), qg.slice_views(views, lay),
                                             a.dur, words=_ocr(src, a, (sw, sh)),
                                             text_edges=lay == DUAL_VIEW))
    if not v["passed"]:
        raise RenderError(f"crop would slice a source banner or graphic: the quality gate would "
                          f"fail the {lay} crop ({'; '.join(v['errors'])})")
    return v


def replan_track(plan, a, src, why):
    """a poker HUD band whose windows cut the feed's graphics (10-18 clip 1317: the
    Lodge's tiles and ticker, text share 1.2) is planned again as the speaker crop (track), in
    place, for the banner guard and the slice ladder."""
    log(f"{plan.get('layout')}: {why}; the speaker crop instead")
    try:
        new = reframe.analyze(str(src), a.start, a.dur, force="track")
    except Exception as e:        # noqa: BLE001
        raise RenderError(f"{why}; the speaker crop could not be planned ({e})")
    plan.clear()
    plan.update(new)


def hud_layout(plan, a, src, guard, rejected):
    """The layout chooser for gaming / poker / streamers clips (dual_view_allowed). A speaker crop
    that slices a graphic runs the slice ladder's full-bleed steps (a clean recrop, the speaker's
    tile) and keeps full-bleed when one is clean; otherwise, and for any other planned layout
    whose windows slice text or whose blurred fill is over the gate's cap (or that the banner
    guard rejected), the dual view replaces it (to_dual_view). A plan that slices nothing ships
    unchanged. Returns the guard report; raises RenderError when the clip is rejected as before
    (no room for a zoom panel)."""
    lay = plan.get("layout")
    if plan.get("manual") or lay == DUAL_VIEW:
        if rejected:
            raise RenderError(rejected)
        return guard
    if lay in TRACK_LIKE_LAYOUTS and plan.get("segments") and "src_w" in plan:
        return slice_ladder(plan, a, src, guard, rejected,
                            dual_view=lambda why, tried: stack_fallback(plan, a, src, why, tried,
                                                                        guard))
    try:
        why = rejected or shown_defect(plan, a, src)
    except Exception as e:        # noqa: BLE001 - an unmeasurable plan ships as before
        log(f"dual view: the planned {lay} could not be measured ({type(e).__name__}: {e})")
        why = None
    if why:
        rep = stack_fallback(plan, a, src, why, [], guard)
        if rep is None and no_band_stack(a) and lay not in ("gamestack", "reaction",
                                                            "screencast_auto") \
                and not plan.get("cam_mid"):
            # gameplay without a cam (a mid-frame cam's 9:16 window over the activity
            # could centre on the cam, the face crop gameplay never ships)
            rep = full_bleed_action(plan, a, src, why)
        if rep is not None:
            return rep
        if no_band_stack(a):
            # render yield 11-09 class 1: no dual view takes its place (no blurred band)
            raise RenderError(f"crop would slice a source graphic or show a blurred fill: {why}; "
                              f"no clean streamer split or full-bleed crop")
    if rejected:
        raise RenderError(rejected)
    return guard


def full_bleed_action(plan, a, src, why):
    """render yield 11-09 class 1: a gaming / streamers window with no speaker and no facecam
    (gameplay the planner left as the general layout, a blurred fill the gate fails) gets a
    full-bleed 9:16 window over the action instead of the dual view: the largest clean_crop
    (down to cleancrop.MIN_CANVAS_W of canvas) whose edges cut no plate or logo bug, nearest the
    activity centroid. Replaces `plan` in place and returns the report; None when none is clean."""
    if plan.get("manual") or getattr(a, "dur", None) is None:
        return None
    if plan.get("src_w") and plan.get("src_h"):
        sw, sh = int(plan["src_w"]), int(plan["src_h"])
    else:
        sw, sh = reframe.probe_size(str(src))
    try:
        sc, words = _scan(src, a, (sw, sh)), _ocr(src, a, (sw, sh))
        plates = [p["box"] for p in keep_plates(sc, words, sw, sh, pad=LADDER_TEXT_PAD)]
        plates += [list(b) for b in _overlays(src, a, (sw, sh))]
        cx = facecam.activity_centroid(str(src), a.start, a.dur)
    except Exception as e:        # noqa: BLE001 - no verdict: no crop
        log(f"full-bleed action: unreadable ({type(e).__name__}: {e})")
        return None
    cw = reframe.crop_width(sw, sh)
    cx = sw / 2.0 if cx is None else cx
    static = {"layout": "track", "src_w": sw, "src_h": sh,
              "segments": [{"t0": 0.0, "t1": float(a.dur), "x": int(max(0, min(sw - cw,
                                                                                 cx - cw / 2)))}]}
    exclude, _ns = cleancrop.regions_from_scan(sc, strict=False)
    try:
        crop = cleancrop.clean_crop(static, dict(sc, faces=[]), exclude, plates)
    except cleancrop.CleanCropError as e:
        log(f"full-bleed action: none ({e})")
        return None
    cleancrop.apply(static, crop)
    rep = {"action": "full_bleed_action", "crop_box": static["crop_box"], "why": why,
           "replaced": plan.get("layout"), "canvas_w": crop.get("canvas_w")}
    plan.clear()
    plan.update(static)
    plan["banner_guard"] = rep
    log(f"full-bleed action {crop['w']}x{crop['h']} at x={crop['x'][0]} y={crop['y']} instead of "
        f"{rep['replaced']}: {why}")
    return rep


def clean_crop_plan(plan, a, src):
    """--clean-crop: replace the banner guard with the clean speaker crop
    (render/cleancrop.py): the largest full-bleed 9:16 window that keeps the speaker, stays in
    the speaker's boxed window (--clean-crop-windows), leaves banners / panels / dead bands /
    --avoid-boxes outside (strict) or unsliced (soft) and slices no other source graphic.
    Raises RenderError ("clean crop: ...") when none is >= cleancrop.MIN_CANVAS_W of canvas."""
    if plan.get("layout") not in TRACK_LIKE_LAYOUTS or not plan.get("segments"):
        raise RenderError(f"clean crop: the planner gave {plan.get('layout')!r}, not a speaker "
                          f"track (no face to crop to)")
    size = (plan["src_w"], plan["src_h"])
    sc = banners.scan(str(src), a.start, a.dur, src_size=size)
    extra = json.loads(Path(a.avoid_boxes).read_text()) if a.avoid_boxes else []
    strict = a.clean_crop == "strict"
    exclude, no_slice = cleancrop.regions_from_scan(sc, strict=strict, extra=extra)
    items = _gate_text_items(src, a, sc)
    if items:
        ex2, ns2 = cleancrop.regions_from_text(items, size[0], size[1], strict=strict)
        exclude, no_slice = exclude + ex2, no_slice + ns2
    wins = cleancrop.window_boxes(src, a.start, a.dur, plan, sc.get("faces") or []) \
        if a.clean_crop_windows else None
    try:
        crop = cleancrop.clean_crop(plan, sc, exclude, no_slice, windows=wins)
    except cleancrop.CleanCropError as e:
        raise RenderError(f"clean crop: {e}")
    cleancrop.apply(plan, crop)
    rep = {"action": "clean_crop", "crop_box": plan["crop_box"], "mode": a.clean_crop,
           "canvas_w": crop["canvas_w"], "windows": wins,
           "excluded": len(exclude), "no_slice": len(no_slice)}
    plan["banner_guard"] = rep
    log(f"clean crop ({a.clean_crop}): {crop['w']}x{crop['h']} at y={crop['y']} "
        f"({crop['canvas_w']} px of canvas), windows={wins}")
    return rep


def _gate_text_items(src, a, sc):
    """The burned-in text items the quality gate's crop-slice check will read for this window
    (quality_gate.text_items on the same OCR'd frames); [] when the OCR is unavailable."""
    try:
        from clipengine import quality_gate
        words = quality_gate.ocr_source_text(str(src), a.start, a.dur, (sc["src_w"], sc["src_h"]))
        return [it for it in quality_gate.text_items(sc, words) if it.get("src") == "ocr"]
    except Exception as e:        # noqa: BLE001 - the line detector's graphics still count
        log(f"clean crop: OCR text items unavailable ({type(e).__name__}: {e})")
        return []


def mute_spans(spec):
    """--mute "a-b,c-d" (clip seconds) -> [(a, b)]."""
    out = []
    for part in str(spec or "").split(","):
        if part.strip():
            a, b = part.split("-")
            out.append((float(a), float(b)))
    return out


def video_graph(plan, a):
    """The layout's graph (ends in [v0]); a plan with "cover" (render yield 11-09: the logo bugs
    no crop could keep whole or out) reads a source with those parts painted out."""
    vg = _layout_graph(plan, a)
    if plan.get("cover"):
        vg = overlays.cover_graph(vg, plan["cover"], plan["src_w"], plan["src_h"])
    return vg


def _layout_graph(plan, a):
    layout = plan["layout"]
    if plan.get("manual"):
        bmode = a.bottom_mode or ("fit" if layout == "screencast" else "cover")
        return stack_vf(rect(a.top), rect(a.bottom), plan["split_px"], bmode)
    if layout == "track":
        return track_vf(plan)
    # campaign-visible named layouts render through their internal planner's graph.
    if layout == "product_full_bleed":
        return track_vf(plan)
    if layout == "full_bleed_speaker":
        return track_vf(plan)
    if layout == "podcast_two_shot":
        return track_vf(plan)
    if layout == "chart_top_cam_bottom":
        # the dedicated chart/cam planner emits a ``cam_box``; fall back to the old
        # broadcast layout when it wasn't able to find a cam (plan carries ``segments``).
        if plan.get("cam_box"):
            return chart_top_cam_bottom_vf(plan)
        return broadcast_vf(plan)
    if layout == "broadcast":
        return broadcast_vf(plan)
    if layout == "hudband":
        return hudband_vf(plan)
    if layout == "gamestack":
        return gamestack_vf(plan)
    if layout == "reaction":
        return reaction_vf(plan)
    if layout == "screencast_auto":
        return screencast_auto_vf(plan)
    if layout == "game_only":
        return _game_only_vf(plan)
    if layout == "split":
        return split_vf(plan)
    if layout == DUAL_VIEW:
        return dual_view_vf(plan)
    return general_vf(plan)


FACE_UP, FACE_DOWN = 0.9, 0.65   # face box (forehead-chin) -> head incl. hair/cap and chin


TRACK_LIKE_LAYOUTS = ("track", "product_full_bleed", "full_bleed_speaker", "podcast_two_shot")
BROADCAST_LIKE_LAYOUTS = ("broadcast", "chart_top_cam_bottom")


def _hook_views(plan, t):
    """(src_x, src_w, src_y, src_h, out_y, out_h) source windows visible on the canvas at clip
    time t, each mapped to an output y-band."""
    layout, sh = plan["layout"], plan["src_h"]
    if layout == "split":
        l, r = plan["split"]["left"], plan["split"]["right"]
        band = OUT_H // 2
        return [(l["x"], l["w"], l["y"], l["h"], 0, band), (r["x"], r["w"], r["y"], r["h"], band, band)]
    # chart_top_cam_bottom with a cam_box uses a different plan shape (no per-time
    # segments). The top panel is the whole source cover-scaled; the bottom panel shows
    # just the cam_box region.
    if layout == "chart_top_cam_bottom" and plan.get("cam_box"):
        top_h = reframe.BROADCAST_TOP_H
        bot_h = OUT_H - top_h
        cx, cy, cw, ch = plan["cam_box"]
        return [(0, plan["src_w"], 0, sh, 0, top_h),
                (cx, cw, cy, ch, top_h, bot_h)]
    segs = plan.get("segments") or []
    if layout not in TRACK_LIKE_LAYOUTS + BROADCAST_LIKE_LAYOUTS + ("hudband",) or not segs:
        return []
    if layout in TRACK_LIKE_LAYOUTS:      # render yield 11-09: a text shot's own window
        for s in plan.get("text_shots") or []:
            if float(s["t0"]) <= t <= float(s["t1"]):
                if s["mode"] == "fit":
                    fh = int(round(OUT_W * sh / float(plan["src_w"])))
                    return [(0, plan["src_w"], 0, sh, (OUT_H - fh) // 2, fh)]
                return [(s["x"], s["w"], s["y"], s["h"], 0, OUT_H)]
    # before the first segment the crop holds the first segment's x (reframe.crop_x_expr)
    x = segs[0]["x"] if t < segs[0]["t0"] else \
        next((g["x"] for g in segs if g["t0"] <= t < g["t1"]), segs[-1]["x"])
    if layout in TRACK_LIKE_LAYOUTS:
        cw, ch, cy = track_crop_window(plan)
        ch = sh if ch == "ih" else ch
        if track_boxed(plan):
            out_h = min(OUT_H, int(round(OUT_W * ch / cw)))
            return [(x, cw, cy, ch, (OUT_H - out_h) // 2, out_h)]
        return [(x, cw, cy, ch, 0, OUT_H)]
    if layout in BROADCAST_LIKE_LAYOUTS:
        return [(x, reframe.broadcast_crop_width(plan["src_w"], sh), 0, sh, 0, reframe.BROADCAST_TOP_H)]
    return [(x, plan["cw"], plan.get("y_off", 0), plan.get("eff_h", sh), 0,
             OUT_H - int(plan["band_height"]))]


def _cover_visible(x, y, w, h, band_w, band_h):
    """The part of source rect (x, y, w, h) that a cover scale (force_original_aspect_ratio=
    increase, then a centred crop) into a band_w x band_h band leaves on screen."""
    if not (w and h and band_w and band_h):
        return x, y, w, h
    if w * band_h > h * band_w:                       # wider than the band: sides cut
        vw = h * band_w / band_h
        return x + (w - vw) / 2, y, vw, h
    vh = w * band_h / band_w                          # taller: top/bottom cut
    return x, y + (h - vh) / 2, w, vh


def _minus_spans(t0, t1, spans):
    """[(a, b)] of [t0, t1] left once every span in `spans` is taken out."""
    out = [(float(t0), float(t1))]
    for s0, s1 in spans:
        nxt = []
        for a_, b_ in out:
            if s1 <= a_ or s0 >= b_:
                nxt.append((a_, b_))
                continue
            if s0 > a_:
                nxt.append((a_, s0))
            if s1 < b_:
                nxt.append((s1, b_))
        out = nxt
    return [(a_, b_) for a_, b_ in out if b_ - a_ > 1e-3]


def crop_views(plan, dur):
    """every source rectangle the canvas shows, with when: [{"t0", "t1", "x",
    "y", "w", "h"}] (clip seconds, source px). A crop edge of one of these windows is where a
    source graphic can be sliced; the quality gate checks the source's graphics against them
    (quality_gate.measure_crop_slices). A full-frame view (general, broadcast's bottom band,
    a fit / boxed window shown whole) cannot slice anything, so it is listed only when it is
    smaller than the frame. [] for manual stacks and plans without a source size. The dual view's
    zoom windows carry "panel": "zoom" (a deliberate crop of the frame shown whole above it: the
    gate does not read them for slices)."""
    if plan.get("manual") or not plan.get("src_w"):
        return []
    layout, sw, sh = plan.get("layout"), int(plan["src_w"]), int(plan["src_h"])
    dur = float(dur)
    views = []

    def add(x, y, w, h, t0=0.0, t1=dur, panel=None):
        x, y, w, h = (round(float(v), 1) for v in (x, y, w, h))
        if w > 0 and h > 0 and (x > 0.5 or y > 0.5 or x + w < sw - 0.5 or y + h < sh - 0.5):
            views.append({"t0": round(float(t0), 3), "t1": round(float(t1), 3),
                          "x": x, "y": y, "w": w, "h": h})
            if panel:
                views[-1]["panel"] = panel

    def per_segment(cw, y, h):
        segs = plan.get("segments") or []
        for i, s in enumerate(segs):
            add(s["x"], y, cw, h, 0.0 if i == 0 else s["t0"],
                dur if i == len(segs) - 1 else segs[i + 1]["t0"])

    def cam(c, top_h):
        cw, ch, cx, cy, c_fit = c
        if c_fit:
            add(cx, cy, cw, ch)
        else:
            add(*_cover_visible(cx, cy, cw, ch, OUT_W, OUT_H - top_h))

    if layout in TRACK_LIKE_LAYOUTS:
        cw, ch, cy = track_crop_window(plan)
        per_segment(cw, cy, sh if ch == "ih" else ch)
        shots = plan.get("text_shots") or []
        if shots:                     # render yield 11-09: a shot's window replaces the track's
            base, views = views, []
            for v in base:
                for a_, b_ in _minus_spans(v["t0"], v["t1"], [(s["t0"], s["t1"]) for s in shots]):
                    views.append(dict(v, t0=round(a_, 3), t1=round(b_, 3)))
            for s in shots:
                if s["mode"] != "fit":            # a frame fit to width shows the whole frame
                    add(s["x"], s["y"], s["w"], s["h"], s["t0"], s["t1"])
            views.sort(key=lambda v: v["t0"])
    elif layout in BROADCAST_LIKE_LAYOUTS and plan.get("cam_box"):
        top_h = reframe.BROADCAST_TOP_H
        bx, by, bw, bh = plan.get("chart_bbox") or (0, 0, sw, sh)
        add(*_cover_visible(bx, by, bw, bh, OUT_W, top_h))
        cx, cy, cw, ch = plan["cam_box"]
        add(*_cover_visible(cx, cy, cw, ch, OUT_W, OUT_H - top_h))
    elif layout in BROADCAST_LIKE_LAYOUTS:
        per_segment(reframe.broadcast_crop_width(sw, sh), 0, sh)
    elif layout == "hudband":
        per_segment(plan["cw"], int(plan.get("y_off", 0)),
                    int(plan.get("eff_h", sh - int(plan.get("y_off", 0)))))
        for item in plan.get("band_items") or []:
            sx, sy, iw, ih = item["src"]
            for a, b in (item.get("intervals") or [(0.0, dur)]):
                add(sx, sy, iw, ih, max(0.0, a), min(dur, b))
    elif layout in ("gamestack", "game_only") and plan.get("v2"):
        b = plan["band"]                          # the band along its path; the card below
        for t0, t1, x, y in v2_views(plan, dur):
            add(x, y, b["w"], b["h"], t0, t1)
        if layout == "gamestack":                 # (a game only card shows the whole frame)
            cw, ch, cx, cy, _fit = plan["cam"]
            add(cx, cy, cw, ch)
    elif layout == "gamestack":
        top_h = int(plan["top_h"])
        gw, gx, g_fit = plan["game"]
        gb = plan.get("game_box")
        if gb:                                    # the streamer split's game window
            add(*_cover_visible(gb[2], gb[3], gb[0], gb[1], OUT_W, top_h))
        elif g_fit:
            add(gx, 0, gw, sh)
        else:
            add(*_cover_visible(gx, 0, gw, sh, OUT_W, top_h))
        cam(plan["cam"], top_h)
    elif layout in ("reaction", "screencast_auto"):
        top_h = int(plan["top_h"])
        tw, th, tx, ty = plan["content"]
        fit = plan.get("content_fit", REACTION_CONTENT_FIT) == "fit" if layout == "reaction" \
            else bool(plan.get("content_fit_blur"))
        if fit:
            add(tx, ty, tw, th)
        else:
            add(*_cover_visible(tx, ty, tw, th, OUT_W, top_h))
        cam(plan["cam"], top_h)
    elif layout == "split" and plan.get("split"):
        for c in plan["split"].values():
            add(c["x"], c["y"], c["w"], c["h"])
    elif layout == "game_only":
        cw = reframe.crop_width(sw, sh)
        add(max(0, (sw - cw) // 2), 0, cw, sh)
    elif layout == DUAL_VIEW:
        z, segs = plan["zoom"], plan.get("zoom_segments") or []
        for i, s in enumerate(segs):                 # the top panel is the whole frame
            add(s["x"], s["y"], z["w"], z["h"], 0.0 if i == 0 else s["t0"],
                dur if i == len(segs) - 1 else segs[i + 1]["t0"], panel="zoom")
    elif layout == "general" and plan.get("content_box"):
        b = plan["content_box"]                   # the screen fit's box edges
        add(b["x"], b["y"], b["w"], b["h"])
    return views


def hook_avoid(plan):
    """Output-canvas y-bands [(y0, y1)] covered by a head while the hook is on screen."""
    if plan.get("manual") or "src_w" not in plan:
        return []
    if plan.get("header"):
        # gamestack / game only v3: the hook sits in the header above the band (build_ass's
        # hook_box), never on the band or the card
        return [(int(plan["header"][1]), OUT_H)]
    if plan.get("v2"):
        # gamestack / game only v2: the hook stays in the zone between the band (the action) and
        # the card, as the captions do
        return [(0, int(plan["band"]["out_h"])), (int(plan["card"][1]), OUT_H)]
    if plan.get("layout") == "gamestack":
        # the whole cam band belongs to the streamer; keep the title in the game band.
        top_h = int(plan.get("top_h", GAMESTACK_TOP_H))
        return [(top_h, OUT_H)]
    if plan.get("layout") == "reaction":
        top_h = int(plan.get("top_h", REACTION_TOP_H))
        return [(top_h, OUT_H)]
    if plan.get("layout") == "screencast_auto":
        top_h = int(plan.get("top_h", SCREENCAST_TOP_H))
        return [(top_h, OUT_H)]
    if plan.get("layout") == DUAL_VIEW:
        return []                     # the hook stays in the header (main: a hard text band)
    sw, sh = plan["src_w"], plan["src_h"]
    bands = []
    for s in plan.get("hook_faces") or []:
        views = _hook_views(plan, s["t"])
        for f in s["faces"]:
            fx = f["cx"] * sw
            fy0, fy1 = (f["cy"] - FACE_UP * f["h"]) * sh, (f["cy"] + FACE_DOWN * f["h"]) * sh
            for vx, vw, vy, vh, oy, oh in views:
                if vx <= fx < vx + vw and fy1 > vy and fy0 < vy + vh:
                    k = oh / vh
                    bands.append((max(oy, oy + (fy0 - vy) * k), min(oy + oh, oy + (fy1 - vy) * k)))
    return bands


def _filter_path(p):
    # only characters that need no filtergraph escaping; anything else is dropped by the caller
    return p if p and re.fullmatch(r"[\w./+-]+", p) else None


def audio_chain(dur, loud=None, target=None, mute=None):
    parts = []
    if mute:                     # a slur bleeped (muted) over its word span
        parts.append("volume=0:enable='" + "+".join(f"between(t,{a:.3f},{b:.3f})"
                                                    for a, b in mute) + "'")
    if loud:
        parts.append(f"loudnorm=I={target}:TP={LOUD_TP}:LRA={LOUD_LRA}"
                     f":measured_I={loud['input_i']}:measured_TP={loud['input_tp']}"
                     f":measured_LRA={loud['input_lra']}:measured_thresh={loud['input_thresh']}"
                     f":offset={loud['target_offset']}:linear=true:print_format=summary")
    parts.append("aresample=48000")
    # ``apad`` extends the audio with silence to any length. Follow it with
    # ``atrim=0:dur`` and both the "input seek snapped to an AAC packet boundary before the
    # requested t0" case AND the "AAC frame boundary rounded up at the end" case produce
    # audio of EXACTLY ``dur`` seconds. The old chain only trimmed trailing samples, which
    # left the 60-100 ms shortfall from leading-sample snap intact.
    parts.append("apad")
    parts.append(f"afade=t=in:st=0:d={FADE_IN}")
    parts.append(f"afade=t=out:st={max(0.0, dur - FADE_OUT):.3f}:d={FADE_OUT}")
    parts.append(f"atrim=0:{float(dur):.3f}")
    parts.append("asetpts=PTS-STARTPTS")
    return ",".join(parts)


def filter_graph(vgraph, ass_name=None, fps_cap=False, achain=None, dur=None):
    post = []
    if fps_cap:
        post.append(f"fps={MAX_FPS}")
    if ass_name:
        sub = f"subtitles={ass_name}"
        fonts = _filter_path(captions.FONT_DIR)
        if fonts:
            sub += f":fontsdir={fonts}"
        post.append(sub)
    # hard-trim video to exactly the requested duration. ``trim``+``setpts`` resets
    # PTS so the encoder stops at ``dur`` instead of the next frame-boundary rounding up.
    if dur is not None:
        post.append(f"trim=0:{float(dur):.3f}")
        post.append("setpts=PTS-STARTPTS")
    graph = f"{vgraph};[v0]{','.join(post) or 'null'}[v]"
    if achain:
        graph += f";[0:a:0]{achain}[a]"
    return graph


# ---------------------------------------------------------------- loudness + QC

def measure_loudnorm(src, start, dur, target):
    """Pass 1: audio-only loudnorm analysis of the window. None if unmeasurable (silence)."""
    r = run(["ffmpeg", "-hide_banner", "-nostats", "-nostdin", "-ss", start, "-t", dur, "-i", src,
             "-vn", "-af", f"loudnorm=I={target}:TP={LOUD_TP}:LRA={LOUD_LRA}:print_format=json",
             "-f", "null", "-"])
    m = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr)
    if not m:
        return None
    stats = json.loads(m[-1])
    try:
        if not all(math.isfinite(float(stats[k])) for k in
                   ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")):
            return None
    except (KeyError, ValueError):
        return None
    return stats


def measure_lufs(path):
    r = run(["ffmpeg", "-hide_banner", "-nostats", "-nostdin", "-i", path, "-vn",
             "-af", "ebur128=framelog=quiet", "-f", "null", "-"])
    m = re.findall(r"\bI:\s*(-?[\d.]+|-inf)\s*LUFS", r.stderr)
    if not m or m[-1] == "-inf":
        return None
    return float(m[-1])


def qc(path, dur, target, layout):
    rep = {"width": 0, "height": 0, "duration": 0.0, "has_audio": False,
           "integrated_lufs": None, "layout": layout}
    errors = []
    try:
        info = probe(path)
        rep.update(width=info["width"], height=info["height"],
                   duration=round(info["duration"], 3), has_audio=info["has_audio"])
    except (RenderError, OSError, ValueError) as e:
        errors.append(f"probe failed: {e}")
    if target is not None and rep["has_audio"]:
        try:
            rep["integrated_lufs"] = measure_lufs(path)
        except RenderError as e:
            errors.append(f"loudness measurement failed: {e}")
    if (rep["width"], rep["height"]) != (OUT_W, OUT_H):
        errors.append(f"size {rep['width']}x{rep['height']} != {OUT_W}x{OUT_H}")
    if abs(rep["duration"] - dur) > QC_DUR_TOL:
        errors.append(f"duration {rep['duration']} vs requested {dur}")
    if not rep["has_audio"]:
        errors.append("no audio stream")
    if target is not None and rep["has_audio"] and (
            rep["integrated_lufs"] is None or abs(rep["integrated_lufs"] - target) > QC_LUFS_TOL):
        errors.append(f"integrated loudness {rep['integrated_lufs']} not within "
                      f"{QC_LUFS_TOL} LU of {target}")
    rep["passed"] = not errors
    rep["errors"] = errors
    return rep


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src"); ap.add_argument("start", type=float)
    ap.add_argument("dur", type=float); ap.add_argument("out")
    ap.add_argument("--layout", choices=["auto", "general", "track", "split", "screencast",
                                          "broadcast", "hudband", "gamestack", "reaction",
                                          "chart_top_cam_bottom", "product_full_bleed",
                                          "full_bleed_speaker", "podcast_two_shot",
                                          "dual_view"],
                    default="auto")
    ap.add_argument("--niche", default=None,
                    help="daily niche of the clip (gaming, streamers, podcasts, finance, tech, "
                         "lifestyle, poker): only gaming/streamers may stack; tech/finance "
                         "always render a single view")
    ap.add_argument("--vertical", default=None,
                    help="content vertical (poker, podcast, ...); --layout auto uses "
                         "render.vertical_layouts from the config to map it")
    ap.add_argument("--channel", default=None,
                    help="channel key for the per-channel facecam profile (gamestack layout); "
                         "typically the Twitch login or YouTube channel id")
    ap.add_argument("--game", default=None,
                    help="game name (gamestack); maps to render.game_genres = tight|wide crop")
    ap.add_argument("--broadcast", default=None,
                    help="the source's broadcast / channel name: picks its hudband HUD profile "
                         "(render.hud_profile_by_broadcast) before the vertical's")
    ap.add_argument("--top"); ap.add_argument("--bottom")
    ap.add_argument("--split", type=int, default=1300, help="top band height px (screencast)")
    ap.add_argument("--bottom-mode", choices=["cover", "fit"], default=None,
                    help="fit = portrait facecam over blur fill (screencast default); cover = two landscape crops (split default)")
    ap.add_argument("--captions", action="store_true")
    ap.add_argument("--words", help="WORDS.json: [{w, start, end}] in absolute source seconds")
    ap.add_argument("--caption-model", default="small.en")
    ap.add_argument("--caption-style", default="opus", choices=sorted(captions.STYLES))
    ap.add_argument("--loudnorm", type=float, default=None, help="target integrated LUFS, e.g. -14")
    ap.add_argument("--hook", default=None, help="hook title: first ~3s, then fades; placed clear of faces")
    ap.add_argument("--emit-plan", help="write the reframe plan JSON here")
    ap.add_argument("--facecam-cache", default=None,
                    help="override the facecam profile cache dir (per-run isolation; also "
                         "settable via FACECAM_CACHE_DIR). Never wipes other layouts' profiles.")
    ap.add_argument("--clean-crop", choices=["soft", "strict"], default=None,
                    help="clean speaker crop instead of the banner guard; strict "
                         "keeps banners / panels / dead bands outside, soft only unsliced")
    ap.add_argument("--clean-crop-windows", action="store_true",
                    help="keep the clean crop inside the speaker's boxed broadcast window")
    ap.add_argument("--avoid-boxes", default=None,
                    help="JSON [[x0,y0,x1,y1], ...] src px the clean crop leaves outside")
    ap.add_argument("--caption-zone", choices=["default", "auto"], default="default",
                    help="auto: captions move to rows the clip's on-screen text leaves free")
    ap.add_argument("--mute", default=None, help="clip-second spans 'a-b,c-d' to mute (bleep)")
    ap.add_argument("--levels", action="store_true",
                    help="measure a mild levels lift for any vertical (too-dark repair)")
    ap.add_argument("--content-zoom", choices=["cover", "wide"], default=None,
                    help="auto-screencast content crop mode (default: render.screencast.content_zoom)")
    ap.add_argument("--dual-view-fallback", action="store_true",
                    help="a tech/finance clip renders the dual view (the whole "
                         "frame fit to width on top, the speaker below) instead of its single "
                         "view; the re-render after the single view sliced the screen")
    ap.add_argument("--clean-first", action="store_true",
                    help="a source whose crops keep slicing graphics (a boxed broadcast, "
                         "a creator with repeat edge_slice rejects): the slice ladder's clean "
                         "recrop replaces the planned speaker crop even when that measures clean")
    ap.add_argument("--content-kind", choices=["talk", "screen", "unknown"], default=None,
                    help="the source's kind (engine.source_kind); a screen share / "
                         "chart whose 9:16 crop would slice its text is fit to width instead")
    return ap.parse_args(argv)


def choose_layout(a, src, info):
    """(plan, guard report): the window planned and its layout chosen, everything main does
    before the encode. Raises RenderError when the
    clip has no layout that ships (no clean crop: rejected before the encode)."""
    plan = plan_layout(a, src)
    if dual_view_fallback(a):
        guard = to_dual_view(plan, a, src, f"{a.niche} dual view fallback: the single view "
                             "sliced the screen")
        if guard is None:
            raise RenderError("dual view fallback: the frame leaves no room for the speaker "
                              "panel")
    elif a.clean_crop:
        try:
            guard = clean_crop_plan(plan, a, src)
        except RenderError as e:
            # a stacking niche without a clean speaker crop gets the streamer split (a daily
            # clip) or the dual view (hud_layout)
            guard = stack_fallback(plan, a, src, str(e)) if dual_view_allowed(a) else None
            if guard is None:
                raise
    else:
        if plan.get("layout") == "hudband" and hud_niche(a) and not dual_view_allowed(a):
            try:
                why = shown_defect(plan, a, src)
            except Exception as e:    # noqa: BLE001 - an unmeasurable band ships as before
                log(f"hudband could not be measured ({type(e).__name__}: {e})")
                why = None
            if why:
                replan_track(plan, a, src, why)
        guard = screen_cam_split(plan, a, src)      # render yield 11-09 class 4 (tech)
        if guard is None:
            try:
                guard, rejected = guard_banners(plan, a, src), None
            except RenderError as e:
                guard, rejected = None, str(e)
            if dual_view_allowed(a):
                guard = hud_layout(plan, a, src, guard, rejected)
            else:
                guard = slice_ladder(plan, a, src, guard, rejected)
            # a plan the dual view did not replace may still get the screen box: a dual view
            # plan is no general plan, so it never does
            guard = screen_fit_general(plan, a, src, info) or guard
    enforce_single_view(plan, a)
    enforce_game_band(plan, a)                      # gameplay, never a face crop
    shot_text_fix(plan, a, src)                     # render yield 11-09 class 3
    cover_overlays(plan, a, src)                    # render yield 11-09 class 2 (last resort)
    try:
        gate_slice_check(plan, a, src)
    except RenderError:
        raise
    except Exception as e:        # noqa: BLE001 - unmeasured here: the gate measures the render
        log(f"pre-encode slice check unavailable ({type(e).__name__}: {e})")
    try:
        facecam_view_check(plan, a, src)
    except RenderError:
        raise
    except Exception as e:        # noqa: BLE001 - unmeasured: the plan as chosen
        log(f"pre-encode facecam check unavailable ({type(e).__name__}: {e})")
    return plan, guard


def facecam_view_check(plan, a, src):
    """in gaming / streamers, a window whose facecam box was read
    (cropcheck.cam_overlay) ships only a plan whose every shown window stays inside that box
    (the stack's cam panel) or leaves it out whole. A window that holds the cam's face and
    reaches past its box shows the cam with the game around it, cut at the window's edges (10-26
    1408: a full-bleed recrop of the cam corner, the game's 'CAMERA' label cut to 'ERA'); one
    that cuts the box slices the cam. A facecam whose box was not read is left as planned. Raises
    RenderError (the window is replaced, as for any no-clean-crop plan); daily niche clips only."""
    if plan.get("manual") or not plan.get("src_w") or not getattr(a, "niche", None) \
            or not dual_view_allowed(a):
        return None
    sw, sh = int(plan["src_w"]), int(plan["src_h"])
    ov = cam_overlay(src, a, (sw, sh))
    if not ov:
        return None
    tile = ov.get("tile")
    if not tile:
        return None        # an unread box is no proof of a facecam: posted IRL streamers (1017,
                           # 1019, 1293) and posted gaming crops of a corner cam (1400, 1405)
    cp = _cam_pt(ov, (sw, sh))
    mask = plan.get("cam_mask")
    for v in crop_views(plan, a.dur):
        win = [v["x"], v["y"], v["x"] + v["w"], v["y"] + v["h"]]
        if _holds(tile, win[:2], CAM_BOX_TOL) and _holds(tile, win[2:], CAM_BOX_TOL):
            continue                              # the cam panel, inside the cam's own box
        if mask and _painted(win, tile, mask):
            continue                              # the band shows the cam painted out
        holds = _holds(win, cp)
        if holds or cropcheck.cuts(win, tile, tol=CAM_BOX_TOL):
            raise RenderError(
                f"crop would slice the game around the facecam's box {tile}: the "
                f"{plan.get('layout')} window {[int(x) for x in win]} "
                f"{'shows the cam with the game around it' if holds else 'cuts the cam'}")
    return None


def main(argv=None):
    a = parse_args(argv)
    src, out = Path(a.src).resolve(), Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".recompose_", dir=out.parent))
    try:
        info = probe(src)
        plan, guard = choose_layout(a, src, info)
        layout = plan["layout"]     # a screen-stack / screen-box reframe changes it
        # dense smooth-track trajectories (hundreds of segments) overflow ffmpeg's
        # expression parser if we feed them to crop_x_expr. Emit a sendcmd script in those
        # cases so track_vf can drive the crop filter from a file instead.
        segs = plan.get("segments") or []
        if plan.get("smooth_track") and segs and len(segs) > 50 and layout in (
                "track", "full_bleed_speaker", "podcast_two_shot"):
            try:
                plan["sendcmd_file"] = write_sendcmd_file(segs, tmp / "cropx.cmd")
            except Exception as e:        # noqa: BLE001
                log(f"sendcmd emit failed ({e}); falling back to expression")
        vgraph = video_graph(plan, a)
        try:
            levels = measure_levels(a, src, plan)
        except Exception as e:       # noqa: BLE001 - an enhancement: render without it
            log(f"levels probe failed ({e}); no levels")
            levels = None
        if levels:
            vgraph = with_levels(vgraph, levels["filter"])
            log(f"dark window (median luma {levels['src_luma']}): {levels['filter']}")

        ass_name = None
        if a.captions or a.hook:
            words = caption_words(a, src, tmp) if a.captions else []
            fg_h = captions.general_fg_h(info["width"], info["height"]) if info["width"] else None
            if isinstance(plan.get("content_box"), dict):    # the screen box's band
                # (a screen stack's reaction plan keeps its content_box as a list: it crashed here)
                fg_h = captions.general_fg_h(plan["content_box"]["w"], plan["content_box"]["h"])
            if layout == "hudband" and plan.get("band_height"):
                # feed the top-region height so captions land above the band seam.
                fg_h = OUT_H - int(plan["band_height"])
            if layout == "gamestack":
                # captions sit on the seam between game and cam bands.
                fg_h = int(plan.get("top_h", GAMESTACK_TOP_H))
            if plan.get("v2"):
                # gamestack / game only v2: captions low in the zone between the band and the card
                fg_h = int(plan["fg_h"])
            if layout == "reaction":
                # captions sit on the seam between the content band and the cam band.
                fg_h = int(plan.get("top_h", REACTION_TOP_H))
            if layout == "screencast_auto":
                # captions/hook: keep the existing screencast defaults (caption_y stays as
                # today's screencast target); avoid the presenter cam band with `hook_avoid`.
                fg_h = int(plan.get("top_h", SCREENCAST_TOP_H))
            if layout == DUAL_VIEW:
                # captions sit just under the seam, clear of the full frame's graphics
                fg_h = int(plan["seam_y"])
            text_avoid = None
            hook, hook_info, hook_why = a.hook, {}, None
            if a.hook:
                # the hook must clear on-screen source text/UI
                tz = textzones.hook_zone_text(src, a.start, a.dur, vgraph)
                if not tz["frames"]:
                    # never a refusal for the hook: no text read -> no hook
                    hook, hook_why = None, "the hook-zone text scan read no frames"
                else:
                    plan["hook_text"] = tz
                    text_avoid = tz["bands"]
                    if layout == DUAL_VIEW:
                        # the hook stays in the header: never over the frame's graphics or the
                        # zoom
                        text_avoid = list(text_avoid) + [(int(plan["header_h"]), OUT_H)]
            cap_y = None
            if a.caption_zone == "auto":
                # the caption band moves to the rows the
                # clip's on-screen text leaves free
                cz = textzones.clip_text_bands(src, a.start, a.dur, vgraph)
                default_y = captions.caption_y(layout, fg_h)
                # a dual view's captions only move down, into the zoom panel; a v3 gamestack's
                # only up, staying on the band (under it sits the card)
                cap_y = captions.free_caption_y(cz["bands"], default_y, **(
                    {"lo": default_y} if layout == DUAL_VIEW else
                    {"hi": int(plan["cap_y"])} if plan.get("header") else {}))
                plan["caption_zone"] = {"bands": cz["bands"], "frames": cz["frames"],
                                        "default_y": default_y, "y": cap_y}
                log(f"caption zone: y={cap_y} (default {default_y}), text bands {cz['bands']}")
            if cap_y is None and plan.get("v2"):
                cap_y = int(plan["cap_y"])           # game only v2 too (caption_y has no row)
            hook_box = None
            if plan.get("header") and plan["header"][1] > 0:
                # v3: the hook on the header above the band, for the whole clip
                hook_box = tuple(plan["header"])
            # the hook falls back (one line, smaller, plate, none), never refuses
            captions.build_ass(words, tmp / "captions.ass", style=a.caption_style,
                               hook=hook, layout=layout, dur=a.dur, fg_h=fg_h,
                               avoid=hook_avoid(plan), text_avoid=text_avoid, cap_y=cap_y,
                               hook_box=hook_box, hook_info=hook_info)
            if hook_why:
                hook_info = {"fallback": "none", "why": hook_why}
            if hook_info.get("fallback"):
                plan["hook_fallback"] = {k: v for k, v in hook_info.items() if k != "plate"}
                log(f"hook placement fallback: {hook_info['fallback']} ("
                    + (hook_info.get("why") or f"the first choice covered "
                       f"{hook_info.get('covered_px')}px of on-screen text") + ")")
            ass_name = "captions.ass"   # ffmpeg runs in tmp: no filtergraph path escaping

        if a.emit_plan:
            Path(a.emit_plan).write_text(json.dumps(dict(plan, video_graph=vgraph), indent=2,
                                                    default=str))

        loud = None
        if a.loudnorm is not None and info["has_audio"]:
            loud = measure_loudnorm(src, a.start, a.dur, a.loudnorm)
            if loud is None:
                log("loudnorm pass 1 found no measurable audio; not normalizing")
        achain = audio_chain(a.dur, loud, a.loudnorm, mute=mute_spans(a.mute)) \
            if info["has_audio"] else None

        graph = filter_graph(vgraph, ass_name, info["fps"] > MAX_FPS + 0.01, achain,
                              dur=a.dur)
        tmp_out = tmp / f"render{out.suffix or '.mp4'}"
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-ss", a.start, "-t", a.dur,
               "-i", src, "-filter_complex", graph, "-map", "[v]"]
        if achain:
            cmd += ["-map", "[a]"]
        # duration hard-caps on the output. ``-t`` matches the input constraint;
        # ``-shortest`` snaps whichever stream is briefly longer back to the other's end
        # (eliminates the ~45-65 ms AAC-frame-boundary drift an earlier render review measured).
        cmd += ["-t", f"{float(a.dur):.3f}"]
        if achain:
            cmd += ["-shortest"]
        cmd += VIDEO_CODEC + (AUDIO_CODEC if achain else []) + ["-movflags", "+faststart",
                                                                 tmp_out]
        run(cmd, cwd=tmp)

        rep = qc(tmp_out, a.dur, a.loudnorm, layout)
        rep["encoder"] = _ENC
        try:
            rep["blur_fill_frac"] = blur_fill_frac(plan, a, info)
        except (KeyError, TypeError, ValueError, IndexError) as e:
            log(f"blur fill estimate failed ({e})")
            rep["blur_fill_frac"] = None
        try:
            rep["crop_views"] = crop_views(plan, a.dur)      # gate check 2
        except (KeyError, TypeError, ValueError, IndexError) as e:
            log(f"crop views failed ({e})")
            rep["crop_views"] = None
        if plan.get("text_shots"):                      # render yield 11-09 class 3
            rep["text_shots"] = [{k: s[k] for k in ("t0", "t1", "mode", "why", "box", "w", "h", "x",
                                                    "y") if k in s} for s in plan["text_shots"]]
        if plan.get("cover"):                           # render yield 11-09 class 2
            rep["covered_overlays"] = plan["cover"]
        if plan.get("streamer_split"):                  # render yield 11-09 classes 1 / 4
            rep["streamer_split"] = {"game_box": plan.get("game_box"), "cam": plan.get("cam"),
                                     "top_h": plan.get("top_h")}
        if plan.get("v2"):                              # v2 / v3: band, card, action
            b = plan["band"]
            card = plan.get("card")
            sh = gamelayout.canvas_shares(b["out_h"], card, plan.get("band_y", 0))
            rep["gamestack_v2"] = {
                "band_share": b["share"], "band_h": b["out_h"], "band_y": plan.get("band_y", 0),
                "band_canvas": sh["band"], "card": card,
                "card_share": round(card[2] * card[3] / float(OUT_W * OUT_H), 3) if card else 0.0,
                "blur": sh["blur"], "bottom_blur": sh["bottom_blur"],
                "widened_from": b.get("widened_from"), "scene_text": b.get("scene_text"),
                "cap_y": plan["cap_y"], "knots": len(plan["pan"]),
                "action_inside": plan["action"]["inside_share"], "peak": plan["action"]["peak"]}
            if plan.get("cam_mid"):                     # a cam burned in mid-frame
                rep["gamestack_v2"].update({k: plan.get(k) for k in ("cam_mid", "cam_mask",
                                                                     "cam_mask_share")})
        if layout == DUAL_VIEW:
            rep["panels"] = dual_view_panels(plan)      # the gate's edge bands read "top"
            rep["dual_view"] = {"bottom": plan.get("bottom"), "zoom": plan.get("zoom"),
                                "segments": len(plan.get("zoom_segments") or [])}
        if guard:
            rep["banner_guard"] = {k: guard[k] for k in ("action", "crop_box", "x", "layout",
                                                          "dead_band", "canvas_w", "windows",
                                                          "mode", "why", "tried", "blur_fill",
                                                          "ladder_tried", "replaced", "bottom")
                                   if k in guard}
        if plan.get("hook_text"):
            rep["hook_text_bands"] = plan["hook_text"]["bands"]
        if plan.get("hook_fallback"):
            rep["hook_fallback"] = plan["hook_fallback"]
        if plan.get("caption_zone"):
            rep["caption_y"] = plan["caption_zone"]["y"]
            rep["caption_zone"] = plan["caption_zone"]
        if a.mute:
            rep["mute"] = mute_spans(a.mute)
        if levels:
            rep["levels"] = levels
        if rep["passed"]:
            os.replace(tmp_out, out)
            print(f"DONE {out} ({out.stat().st_size // 1024}KB) layout={layout}")
        else:
            log("QC FAILED: " + "; ".join(rep["errors"]))
        print("QC " + json.dumps(rep), flush=True)
        return 0 if rep["passed"] else EXIT_QC
    except RenderError as e:
        log(f"RENDER FAILED: {e}")
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
