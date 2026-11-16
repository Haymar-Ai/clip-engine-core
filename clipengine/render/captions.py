#!/usr/bin/env python3
"""The caption + hook-title ASS builder (build_ass.py is a thin CLI wrapper over this).

Input words are clip-relative: [{"w": str, "start": float, "end": float}].

Canvas is 1080x1920. Everything stays out of the platform UI zones:
  captions  1-3 word chunks centred at y=1270 (above the bottom 420px TikTok/Reels/Shorts UI
            band), or just below the foreground video band for the `general` letterbox layout.
  hook      bold uppercase with a heavy outline, digit-bearing words (the $ / % / count
            hook) in the caption highlight yellow, max 3 lines; the fewest, widest
            lines at a readable size (wider than the captions), then the biggest size. On screen for HOOK_HOLD seconds,
            then fades. Placed in the free vertical space between the top UI band and the
            captions that stays clear of the on-screen source text/UI (`text_avoid`, a hard
            rule: no clear position raises HookPlacementError) and then of the faces the
            reframe saw (`avoid`); centred in the top blurred band for `general`.
Widths are measured with the real font (PIL getlength), not a per-character guess.
"""
import os
import re
from pathlib import Path

PLAY_W, PLAY_H = 1080, 1920

def _config_font_dir():
    """render.font_dir from the engine config (env CLIPENGINE_FONT_DIR sets it), or None."""
    try:
        from clipengine import config
        return (config.load().get("render") or {}).get("font_dir") or None
    except Exception:          # noqa: BLE001 - no config: the system fonts
        return None


_FONT_DIR_CFG = _config_font_dir()
# (file, ASS Fontname, PIL variation, ASS Bold). Inter-VF's named "Inter Black" instance is
# what libass/fontconfig select for Fontname "Inter Black"; the fallbacks are static bolds.
# CLIP_FONT_PATH: an explicit font file; render.font_dir: a folder holding Inter-VF.ttf.
FONT_CANDIDATES = [
    (os.environ.get("CLIP_FONT_PATH", ""), "Inter Black", "Black", 0),
    (str(Path(_FONT_DIR_CFG).expanduser() / "Inter-VF.ttf") if _FONT_DIR_CFG else "",
     "Inter Black", "Black", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVu Sans", None, 1),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "Liberation Sans", None, 1),
]
FONT_PATH, FONT_NAME, FONT_VARIATION, FONT_BOLD = next(
    (c for c in FONT_CANDIDATES if c[0] and Path(c[0]).is_file()), (None, "Arial", None, 1))
FONT_DIR = str(Path(FONT_PATH).parent) if FONT_PATH else None
CHAR_W = 0.62            # per-char width estimate (x size) when PIL/the font is unavailable

STYLES = {
    "opus":    dict(size=110, min_size=84, highlight="&H0000E5FF&", outline=7, shadow=2),
    "karaoke": dict(size=110, min_size=84, highlight="&H0000FF00&", outline=7, shadow=2),
    "minimal": dict(size=92, min_size=76, highlight=None, outline=4, shadow=1),
}
WHITE = "&H00FFFFFF&"

SAFE_MARGIN = 90                          # MarginL/MarginR
SAFE_W = PLAY_W - 2 * SAFE_MARGIN         # 900: no caption line may be wider
MAX_WORDS = 3
GAP_BREAK = 0.25         # a pause this long starts a new chunk
HOLD_GAP = 0.4           # chunks bridge gaps shorter than this (no flicker)
TAIL = 0.15              # otherwise a chunk lingers this long after its last word
CAPTION_Y = 1270         # centre line for track/split/screencast
CAPTION_Y_MAX = 1500     # never inside the bottom 420px UI band
GENERAL_FG_H = 608       # 16:9 source scaled to 1080 wide
GENERAL_GAP = 110        # general: caption centre sits this far below the foreground band
BROADCAST_TOP_H = 1312   # broadcast: top band; captions sit just above the seam so the HUD stays clear
BROADCAST_CAPTION_Y = 1250
DUAL_VIEW_SEAM_Y = 1100  # dual view: the seam under the full frame (recompose.DV_SEAM_Y) ...
DUAL_VIEW_CAPTION_GAP = 85   # ... and the caption centre this far below it (half the 150 px
                             # caption block + 10)

HOOK_TOP = 230           # nothing above this: platform top UI (tabs, search)
HOOK_MARGIN = 80        # narrower than the caption margin: the hook reads wider than captions
HOOK_OUTLINE = 8
HOOK_W = PLAY_W - 2 * HOOK_MARGIN - 2 * HOOK_OUTLINE   # 868
HOOK_SIZES = tuple(range(120, 63, -4))    # 120 -> 64 (ASS units)
HOOK_MAX_LINES = 3
HOOK_WIDE_MIN = 80       # prefer fewer, wider lines as long as the size stays at least this
HOOK_LINE_H = 1.2        # line pitch as a multiple of font size
HOOK_HOLD = 3.0          # fully visible this long; a hook's job is done by then
HOOK_FADE_IN_MS = 120
HOOK_FADE_OUT_MS = 400
HOOK_FACE_PAD = 30       # keep this much air between the hook block and any face band
HOOK_STEP = 10
CAPTION_CLEAR = 90       # hook block bottom stays this far above the caption centre line
TEXT_PAD = 24            # air between the hook block and on-screen source text/UI
HOOK_BOX_PAD = 10        # a header hook (hook_box) keeps this much of the header clear each side
HOOK_BOX_LINES = 2       # ... and takes at most this many lines

_SENTENCE_END = re.compile(r"[.?!][\"'”’)\]]*$")
_PIL_FONTS = {}


def _pil_font(size):
    if size not in _PIL_FONTS:
        font = None
        if FONT_PATH:
            try:
                from PIL import ImageFont
                font = ImageFont.truetype(FONT_PATH, size)
                if FONT_VARIATION:
                    try:
                        font.set_variation_by_name(FONT_VARIATION)
                    except Exception:
                        pass
            except Exception:
                font = None
        _PIL_FONTS[size] = font
    return _PIL_FONTS[size]


def text_width(text, size):
    """Rendered width in px at font `size` (real metrics; 0.62*size per char fallback)."""
    font = _pil_font(int(size))
    if font is None:
        return CHAR_W * size * len(text)
    return font.getlength(text)


def _ass_scale(path):
    """libass sizes a font so usWinAscent + usWinDescent spans Fontsize px, so glyphs render at
    unitsPerEm / (winAscent + winDescent) of the PIL size (Inter: 0.70). 1.0 if unreadable."""
    import struct
    try:
        b = Path(path).read_bytes()
        tables = {b[12 + 16 * i:16 + 16 * i]: struct.unpack(">I", b[20 + 16 * i:24 + 16 * i])[0]
                  for i in range(struct.unpack(">H", b[4:6])[0])}
        upm = struct.unpack(">H", b[tables[b"head"] + 18:tables[b"head"] + 20])[0]
        wa, wd = struct.unpack(">HH", b[tables[b"OS/2"] + 74:tables[b"OS/2"] + 78])
        return upm / (wa + wd) if wa + wd else 1.0
    except Exception:
        return 1.0


ASS_SCALE = _ass_scale(FONT_PATH) if FONT_PATH else 1.0


def hook_width(text, size):
    """Width in px as libass actually draws it at ASS Fontsize `size` (the hook fills the line,
    and captions use it too, so their sizes are what actually lands on screen)."""
    return text_width(text, size * ASS_SCALE)


def escape(text):
    """Make any text render literally in an ASS event: braces start override blocks and
    backslash sequences (\\N, \\h, \\{) are escapes, so neutralize them."""
    text = " ".join(str(text).split())
    return text.replace("\\", "\\⁠").replace("{", "\\{").replace("}", "\\}")


def _cs(t):
    return int(round(max(t, 0.0) * 100))


def fmt_time(t):
    h, cs = divmod(_cs(t), 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def words_from_whisper(data):
    """whisper / faster-whisper JSON ({"segments": [{"words": [...]}]}) -> words."""
    return [{"w": str(w.get("word", "")).strip(), "start": float(w["start"]), "end": float(w["end"])}
            for seg in data.get("segments", []) for w in seg.get("words", []) or []]


_NUMBER_TAIL = re.compile(r"^[,.]\d")


def clean_words(words, dur=None):
    """Normalize, drop empty text and crosstalk words (`"cap": false`, see
    clipengine/crosstalk.py), clamp to [0, dur], sort by start. A number whisper split in two
    ("$60" + ",000": 10-21 clip 299 burned "60 ,000") is joined back into one word."""
    out = []
    for w in words or []:
        if w.get("cap", True) is False:
            continue
        text = " ".join(str(w.get("w", w.get("word", w.get("text", "")))).split())
        if not text:
            continue
        s, e = float(w["start"]), float(w["end"])
        if dur is not None and (s >= dur or e <= 0):
            continue
        s = max(0.0, s)
        e = max(s, e if dur is None else min(e, float(dur)))
        out.append({"w": text, "start": s, "end": e})
    out.sort(key=lambda w: w["start"])
    joined = []
    for w in out:
        if joined and _NUMBER_TAIL.match(w["w"]) and joined[-1]["w"][-1:].isdigit():
            joined[-1] = {"w": joined[-1]["w"] + w["w"], "start": joined[-1]["start"],
                          "end": max(joined[-1]["end"], w["end"])}
        else:
            joined.append(w)
    return joined


def _fits(text, size, outline):
    return hook_width(text, size) + 2 * outline <= SAFE_W


def _fit_size(text, p):
    size = p["size"]
    while size > p["min_size"] and not _fits(text, size, p["outline"]):
        size -= 1
    while size > 12 and not _fits(text, size, p["outline"]):   # last resort: one huge word
        size -= 1
    return size


def ends_sentence(text):
    return bool(_SENTENCE_END.search(text))


def chunk_words(words, style="opus"):
    """[{"words": [...], "size": int}]: 1-3 words, one line within SAFE_W, broken at pauses
    >= 0.25s and never across a sentence end."""
    p = STYLES.get(style, STYLES["opus"])
    chunks, cur = [], []
    for w in words:
        if cur:
            text = " ".join(x["w"] for x in cur + [w])
            if (len(cur) >= MAX_WORDS or w["start"] - cur[-1]["end"] >= GAP_BREAK
                    or ends_sentence(cur[-1]["w"]) or not _fits(text, p["size"], p["outline"])):
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return [{"words": c, "size": _fit_size(" ".join(x["w"] for x in c), p)} for c in chunks]


def general_fg_h(src_w, src_h):
    """Height of the letterboxed foreground band (source scaled to 1080 wide, even)."""
    return max(2, int(round(PLAY_W * src_h / src_w / 2)) * 2)


def caption_y(layout, fg_h=None):
    if layout == "general":
        fg = fg_h or GENERAL_FG_H
        return int(min(round((PLAY_H + fg) / 2 + GENERAL_GAP), CAPTION_Y_MAX))
    if layout in ("broadcast", "chart_top_cam_bottom"):
        return BROADCAST_CAPTION_Y
    if layout == "hudband":
        # captions sit just above the band seam. fg_h is the top region height (PLAY_H - band_h).
        top_h = int(fg_h) if fg_h else BROADCAST_TOP_H
        return max(200, top_h - 62)
    if layout == "gamestack":
        # captions sit ON the seam between the game band (top) and the cam band. fg_h is the
        # game band height (default 1220); centre the chunk about 50 px below it.
        top_h = int(fg_h) if fg_h else 1220
        return max(200, min(top_h + 50, CAPTION_Y_MAX))
    if layout == "reaction":
        # captions sit on the seam between the reacted-video band and the cam band.
        # fg_h is the content band height (default 960); centre the chunk just below.
        top_h = int(fg_h) if fg_h else 960
        return max(200, min(top_h + 50, CAPTION_Y_MAX))
    if layout == "screencast_auto":
        # keep the existing screencast caption default (bottom band on the presenter cam
        # is the same shape as manual screencast).
        return CAPTION_Y
    if layout == "dual_view":
        # captions sit on the seam: the whole caption block just under the full frame (fg_h is
        # the seam row), never over its graphics
        seam = int(fg_h) if fg_h else DUAL_VIEW_SEAM_Y
        return max(200, min(seam + DUAL_VIEW_CAPTION_GAP, CAPTION_Y_MAX))
    return CAPTION_Y


def _hard_cut(word, size, max_w):
    n = len(word)
    while n > 1 and hook_width(word[:n], size) > max_w:
        n -= 1
    return word[:n]


def _balanced(words, n, size, max_w):
    """`words` split into n lines, each within max_w, minimizing the widest line; None if none fit."""
    from itertools import combinations
    best = None
    for cuts in combinations(range(1, len(words)), n - 1):
        lines = [" ".join(words[i:j]) for i, j in zip((0,) + cuts, cuts + (len(words),))]
        w = max(hook_width(l, size) for l in lines)
        if w <= max_w and (best is None or w < best[0]):
            best = (w, lines)
    return best and best[1]


def hook_fits(text, max_w=HOOK_W, sizes=HOOK_SIZES, max_lines=HOOK_MAX_LINES):
    """[(lines, size)] largest size first: at each size, the fewest balanced lines that fit."""
    words = " ".join(str(text).split()).split(" ")
    out = []
    for size in sizes:
        lines = next((l for n in range(1, min(max_lines, len(words)) + 1)
                      if (l := _balanced(words, n, size, max_w))), None)
        if lines:
            out.append((lines, size))
    return out


def wrap_hook(text, max_w=HOOK_W, sizes=HOOK_SIZES, max_lines=HOOK_MAX_LINES):
    """(lines, size): the largest size at which the hook fits in <= max_lines balanced lines;
    at the smallest size, fill greedily and truncate with an ellipsis."""
    fits = hook_fits(text, max_w, sizes, max_lines)
    if fits:
        return fits[0]
    words = " ".join(str(text).split()).split(" ")
    size = sizes[-1]
    lines, cur, queue = [], "", list(words)
    while queue and len(lines) < max_lines:
        w = queue[0]
        cand = f"{cur} {w}" if cur else w
        if hook_width(cand, size) <= max_w:
            cur = cand
            queue.pop(0)
        elif cur:
            lines.append(cur)
            cur = ""
        else:                                  # a single word wider than the line
            cut = _hard_cut(w, size, max_w)
            lines.append(cut)
            queue[0] = w[len(cut):]
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if queue:                                  # truncated
        last = lines[-1]
        while last and hook_width(last + "…", size) > max_w:
            last = last[:-1].rstrip()
        lines[-1] = last + "…"
    return lines, size


def _header(p, hook_size):
    ow, bold = p["outline"], FONT_BOLD
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PLAY_W}
PlayResY: {PLAY_H}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{FONT_NAME},{p["size"]},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,{bold},0,0,0,100,100,0,0,1,{ow},{p["shadow"]},5,{SAFE_MARGIN},{SAFE_MARGIN},0,1
Style: Hook,{FONT_NAME},{hook_size},&H00FFFFFF,&H000000FF,&H00000000,&H99000000,{bold},0,0,0,100,100,0,0,1,{HOOK_OUTLINE},3,5,{HOOK_MARGIN},{HOOK_MARGIN},0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


CAPTION_BLOCK_H = 150    # a caption chunk's height incl. outline (110 px font)
CAPTION_TEXT_PAD = 20
CAPTION_Y_MIN = 640      # a moved caption stays in the lower two thirds of the canvas


def free_caption_y(bands, default, lo=CAPTION_Y_MIN, hi=CAPTION_Y_MAX, block_h=CAPTION_BLOCK_H,
                   step=10):
    """the caption centre line nearest `default` whose caption
    block clears every on-screen text band; `default` when it already does or nothing does."""
    if not band_overlap(default, block_h, bands, CAPTION_TEXT_PAD):
        return int(default)
    ys = sorted(range(int(lo), int(hi) + 1, step), key=lambda y: (abs(y - default), y))
    return next((y for y in ys if not band_overlap(y, block_h, bands, CAPTION_TEXT_PAD)),
                int(default))


def caption_events(words, style="opus", layout="track", dur=None, fg_h=None, cap_y=None):
    """Dialogue lines for the captions (one event per active word). `cap_y` overrides the
    layout's caption centre line (free_caption_y)."""
    p = STYLES.get(style, STYLES["opus"])
    y = int(cap_y) if cap_y is not None else caption_y(layout, fg_h)
    chunks = chunk_words(clean_words(words, dur), style)
    events = []
    for i, ch in enumerate(chunks):
        ws = ch["words"]
        nxt = chunks[i + 1]["words"][0]["start"] if i + 1 < len(chunks) else None
        end = ws[-1]["end"] + TAIL
        if nxt is not None:
            end = nxt if nxt - ws[-1]["end"] < HOLD_GAP else min(end, nxt)
        if dur is not None:
            end = min(end, float(dur))
        for j, w in enumerate(ws):
            s = w["start"]
            e = ws[j + 1]["start"] if j + 1 < len(ws) else end
            if _cs(e) <= _cs(s):
                continue                       # zero-length after centisecond rounding
            parts = []
            for k, x in enumerate(ws):
                t = escape(x["w"])
                if p["highlight"] and k == j:
                    t = f"{{\\c{p['highlight']}}}{t}{{\\c{WHITE}}}"
                parts.append(t)
            events.append(f"Dialogue: 0,{fmt_time(s)},{fmt_time(e)},Caption,,0,0,0,,"
                          f"{{\\an5\\pos({PLAY_W // 2},{y})\\fs{ch['size']}}}{' '.join(parts)}")
    return events


class HookPlacementError(ValueError):
    """No hook position clears the on-screen text/UI the source shows while the hook is up."""


def band_overlap(y, block_h, bands, pad):
    """Pixels of the y-bands inside a `block_h` block centred on y, padded by `pad` each side."""
    a, b = y - block_h / 2 - pad, y + block_h / 2 + pad
    return sum(max(0.0, min(b, y1) - max(a, y0)) for y0, y1 in bands or ())


def place_hook(block_h, avoid=(), top=HOOK_TOP, bottom=CAPTION_Y - CAPTION_CLEAR, text_avoid=()):
    """(centre y, face overlap px) for a hook block of `block_h` px: the highest position in
    [top, bottom] with the least overlap against the `avoid` y-bands (faces). Positions clear
    of the `text_avoid` bands (source text/UI) win over positions clear of faces."""
    lo, hi = top + block_h / 2, bottom - block_h / 2

    def overlap(y):
        return band_overlap(y, block_h, avoid, HOOK_FACE_PAD)

    def text(y):
        return band_overlap(y, block_h, text_avoid, TEXT_PAD)

    if hi <= lo:
        return int(round(lo)), overlap(lo) + (lo - hi)     # block taller than the free zone
    y = min(range(int(lo), int(hi) + 1, HOOK_STEP),
            key=lambda y: (text(y) > 0, text(y), overlap(y), y))
    return y, overlap(y)


def _hook_line(line):
    return " ".join(f"{{\\c{STYLES['opus']['highlight']}}}{escape(w)}{{\\c{WHITE}}}"
                    if any(c.isdigit() for c in w) else escape(w) for w in line.split(" "))


def hook_event(hook, layout="track", dur=None, fg_h=None, avoid=None, text_avoid=None,
               cap_y=None, box=None):
    """(Dialogue line, font size) for the hook title, or (None, HOOK_SIZES[0]).
    `text_avoid`: canvas y-bands holding source text/UI while the hook is up; a hook that
    cannot clear them raises HookPlacementError (the caller rejects the clip). `box` = (y0, y1):
    a header the hook sits on, centred, for the whole clip (gamestack v3)."""
    if not hook or not " ".join(str(hook).split()):
        return None, HOOK_SIZES[0]
    text = " ".join(str(hook).split()).upper()
    fade_out = HOOK_FADE_OUT_MS
    end = HOOK_HOLD + HOOK_FADE_OUT_MS / 1000
    if box is not None:
        room = (int(box[1]) - int(box[0])) - 2 * HOOK_BOX_PAD
        fits = [f for f in hook_fits(text, max_lines=HOOK_BOX_LINES)
                if len(f[0]) * f[1] * HOOK_LINE_H <= room]
        lines, size = fits[0] if fits else wrap_hook(text, max_lines=HOOK_BOX_LINES)
        y = int(round((int(box[0]) + int(box[1])) / 2.0))
        hit = band_overlap(y, len(lines) * size * HOOK_LINE_H, text_avoid, TEXT_PAD)
        fade_out, end = 0, (float(dur) if dur else end)
    elif layout == "general":
        band = (PLAY_H - (fg_h or GENERAL_FG_H)) / 2
        fits = [f for f in hook_fits(text) if len(f[0]) * f[1] * HOOK_LINE_H <= band - 40]
        lines, size = fits[0] if fits else wrap_hook(text)
        y = int(round(band / 2))
        hit = band_overlap(y, len(lines) * size * HOOK_LINE_H, text_avoid, TEXT_PAD)
    else:
        # clear of source text first, then of faces; then the fewest (widest) lines at a
        # readable size; then biggest
        bottom = (int(cap_y) if cap_y is not None else caption_y(layout, fg_h)) - CAPTION_CLEAR
        opts = []
        for l, sz in (hook_fits(text) or [wrap_hook(text)]):
            block_h = len(l) * sz * HOOK_LINE_H
            y, face = place_hook(block_h, avoid, bottom=bottom, text_avoid=text_avoid)
            opts.append((l, sz, y, face, band_overlap(y, block_h, text_avoid, TEXT_PAD)))
        lines, size, y, _, hit = min(opts, key=lambda o: (o[4] > 0, o[4], o[3] > 0, o[3],
                                                          o[1] < HOOK_WIDE_MIN, len(o[0]),
                                                          -o[1]))
    if hit > 0:
        raise HookPlacementError(f"no hook position clears the on-screen text: best y={y} "
                                 f"still covers {hit:.0f}px of text bands "
                                 f"{[(int(a), int(b)) for a, b in text_avoid]}")
    if dur:
        end = min(end, float(dur))
    tags = f"\\an5\\pos({PLAY_W // 2},{y})\\fad({HOOK_FADE_IN_MS},{fade_out})"
    body = "\\N".join(_hook_line(l) for l in lines)
    return f"Dialogue: 1,0:00:00.00,{fmt_time(end)},Hook,,0,0,0,,{{{tags}}}{body}", size


def build_ass(words, out_path, style="opus", hook=None, layout="track", dur=None, fg_h=None,
              avoid=None, text_avoid=None, cap_y=None, hook_box=None):
    """Write the ASS file (captions from clip-relative words + optional hook) and return it.
    `avoid`: output-canvas y-bands [(y0, y1)] the hook should stay clear of (faces).
    `text_avoid`: y-bands of on-screen source text/UI the hook must clear (HookPlacementError).
    `cap_y`: the captions' centre line when moved off on-screen text.
    `hook_box`: (y0, y1) header rows the hook sits on for the whole clip (gamestack v3)."""
    p = STYLES.get(style, STYLES["opus"])
    hook_line, hook_size = hook_event(hook, layout, dur, fg_h, avoid, text_avoid, cap_y=cap_y,
                                      box=hook_box)
    events = caption_events(words, style, layout, dur, fg_h, cap_y=cap_y)
    if hook_line:
        events.append(hook_line)
    out = Path(out_path)
    out.write_text(_header(p, hook_size) + "".join(e + "\n" for e in events), encoding="utf-8")
    return out
