#!/usr/bin/env python3
"""Caption repair: a clip whose burned captions fail the gate's caption check
(quality_gate.caption_agreement_verdict: the captions vs the second ASR) is reconciled instead of
rejected.

The gate only knows that two transcripts disagree: the caption words (render.caption_strong_model,
large-v3-turbo; select's small.en / base.en when the strong pass was skipped) and the second ASR
(whisper `small`). Each disagreeing stretch gets a tie-break vote:

  (a) the source's own YouTube subtitles, manual first, auto second (fetched once per video with
      the pipeline's yt-dlp, cached under <scratch_dir>/ytsubs, outside the worktree);
  (b) else a third transcription of just that stretch (+-1 s) with a model different from both
      (`quality_gate.caption_repair.tiebreak_model`, default `medium`: no caption model, not the
      second ASR).

Majority wins per stretch: when the vote reads like the second ASR, the captions take its words;
when it reads like the captions, or like neither, the captions stay. The caption words keep their
timings (the strong model's): a replaced stretch is laid over the old words' span, a stretch the
captions dropped as crosstalk ("cap": false) is burned again on its own words, a stretch the
captions never had is laid in the gap the second ASR heard it in, and caption words nobody else
heard stay in the sidecar unburned ("cap": false). The caller re-renders with the same crop and
layout and runs the full gate again; captions that still fail are rejected as before.

Pure helpers (stretches, vote, apply_votes) are unit-testable; the voters take seams.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from clipengine import config

HERE = Path(__file__).resolve().parent

VOTE_PAD_S = 1.0             # the tie-break transcript hears the stretch +- this
CONTEXT_TOKENS = 3           # shared (agreed) tokens on each side scored with a stretch
JOIN_TOKENS = 1              # disagreeing runs this few agreed tokens apart are one stretch
SLICE_PAD_S = 0.25           # the tie-break words read for a stretch: its context's span +- this
TIEBREAK_MODEL = "medium"    # default third model (multilingual medium: on disk, in no caption role)
YT_TIMEOUT_S = 90.0          # one yt-dlp subtitle pull
YT_NONE_TTL_S = 6 * 3600.0   # a video without subtitles is asked again after this
CRAM_SPAN_K = 1.5            # a replaced stretch whose second-ASR span is this much longer than the
CRAM_SPAN_S = 0.5            # captions' (+ this) or that would burn more than CRAM_WPS words a second
CRAM_WPS = 6.0               # over it keeps the second ASR's own word times
YT_URL = re.compile(r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|live/)|youtu\.be/)"
                    r"([A-Za-z0-9_-]{11})")
_WINDOW_FILE = re.compile(r"_(\d+)-(\d+)\.mp4$")


def _cfg():
    try:
        return config.load()
    except (OSError, ValueError):
        return {}


def settings(cfg=None):
    """The `quality_gate.caption_repair` block over the defaults."""
    cfg = _cfg() if cfg is None else cfg
    block = (cfg.get("quality_gate") or {}).get("caption_repair") or {}
    scratch = Path(cfg.get("scratch_dir", str(HERE / "scratch")))
    return {"enabled": bool(block.get("enabled", True)),
            "tiebreak_model": str(block.get("tiebreak_model", TIEBREAK_MODEL)),
            "youtube": bool(block.get("youtube", True)),
            "cache_dir": str(block.get("cache_dir") or scratch / "ytsubs")}


def independent_tiebreak(model, cfg=None):
    """None when `model` is neither a caption model nor the second ASR, else why it is not."""
    from clipengine import second_asr
    if model in second_asr.caption_models(cfg):
        return f"tie-break model {model!r} is a caption model"
    if model == second_asr.settings(cfg)["model"]:
        return f"tie-break model {model!r} is the second ASR"
    return None


# ---------------------------------------------------------------- tokens and alignment

def _tok_match(a, b):
    from clipengine import quality_gate
    return quality_gate._tok_match(a, b)


def _word_tokens(words):
    """[(token, word index)] of a words list, tokenised as the gate tokenises (norm_tokens)."""
    from clipengine import quality_gate
    out = []
    for i, w in enumerate(words):
        out += [(t, i) for t in quality_gate.norm_tokens([w])]
    return out


def align_pairs(a, b):
    """Matched index pairs [(i, j)] of the longest common (fuzzy, quality_gate._tok_match)
    subsequence of token lists `a` and `b` (quality_gate.align_tokens' DP, pairs kept)."""
    n, m = len(a), len(b)
    if not n or not m:
        return []
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, nxt = dp[i], dp[i + 1]
        for j in range(m - 1, -1, -1):
            row[j] = nxt[j + 1] + 1 if _tok_match(a[i], b[j]) else max(nxt[j], row[j + 1])
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if _tok_match(a[i], b[j]) and dp[i][j] == dp[i + 1][j + 1] + 1:
            pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def lcs_len(a, b):
    """Length of the longest common (fuzzy) subsequence of token lists `a` and `b`."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if _tok_match(x, y) else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


EXTRA_COST = 0.5             # a reading's token the tie-break transcript lacks costs half a token
                             # it has that the reading lacks: a third transcript (YouTube auto
                             # captions above all) drops repeats and fillers far more often than it
                             # invents speech, so "the captions dropped what was said" is the
                             # stronger evidence (10-21 627: "it was a great show", unburned as
                             # crosstalk; YouTube heard "great show")


def reading_cost(x, t, extra=EXTRA_COST):
    """How far reading `x` is from the tie-break tokens `t` of the same time span: the tokens of
    `t` it misses, plus `extra` x the tokens it has that `t` does not."""
    k = lcs_len(x, t)
    return (len(t) - k) + extra * (len(x) - k)


# ---------------------------------------------------------------- stretches

def caption_side(side, start_s, end_s):
    """Indexes of the sidecar words the gate reads as burned captions (quality_gate.clip_words:
    any overlap with the clip, "cap" not False), in time order."""
    dur = float(end_s) - float(start_s)
    out = []
    for k, w in enumerate(side or []):
        if w.get("cap", True) is False or not " ".join(str(w.get("w", "")).split()):
            continue
        s, e = float(w["start"]) - float(start_s), float(w["end"]) - float(start_s)
        if s >= dur or e <= 0:
            continue
        out.append(k)
    return sorted(out, key=lambda k: float(side[k]["start"]))


def stretches(cap_words, asr_words, context=CONTEXT_TOKENS, join_tokens=JOIN_TOKENS):
    """The disagreeing stretches of two word lists in the same time base. Tokens are aligned
    (fuzzy LCS, the gate's token match); each run of unmatched tokens between two agreed ones is a
    stretch, and runs at most `join_tokens` agreed tokens apart are one stretch. A stretch takes
    whole words on both sides (a word with a token in it joins, with the words its matched tokens
    pair with). Returns [{"cap": [word idx], "asr": [word idx], "cap_tokens", "asr_tokens",
    "left", "right" (up to `context` agreed tokens right before / after it), "lo", "hi" (its own
    span), "ctx_lo", "ctx_hi" (with the context)}], in time order."""
    ct, at = _word_tokens(cap_words), _word_tokens(asr_words)
    pairs = align_pairs([t for t, _ in ct], [t for t, _ in at])
    mi = {i: j for i, j in pairs}
    mj = {j: i for i, j in pairs}
    # runs of unmatched tokens between consecutive matched pairs: (pair before, pair after)
    runs, prev = [], (-1, -1, -1)
    for p, (i, j) in enumerate(pairs + [(len(ct), len(at))]):
        if i - prev[1] > 1 or j - prev[2] > 1:
            runs.append([prev[0], p])
        prev = (p, i, j)
    joined = []
    for r in runs:
        if joined and r[0] - joined[-1][1] + 1 <= join_tokens:
            joined[-1][1] = r[1]
        else:
            joined.append(list(r))
    full = pairs + [(len(ct), len(at))]
    out = []
    for p0, p1 in joined:
        i0 = pairs[p0][0] + 1 if p0 >= 0 else 0
        j0 = pairs[p0][1] + 1 if p0 >= 0 else 0
        i1, j1 = full[p1]
        cw = {ct[i][1] for i in range(i0, i1)}
        aw = {at[j][1] for j in range(j0, j1)}
        # whole words: a word split by the run's edge joins with its partner words
        changed = True
        while changed:
            changed = False
            for i, (_, w) in enumerate(ct):
                if w in cw and i in mi and at[mi[i]][1] not in aw:
                    aw.add(at[mi[i]][1])
                    changed = True
            for j, (_, w) in enumerate(at):
                if w in aw and j in mj and ct[mj[j]][1] not in cw:
                    cw.add(ct[mj[j]][1])
                    changed = True
        out.append([cw, aw])
    # stretches that now share a word are one
    merged = []
    for cw, aw in out:
        if merged and (merged[-1][0] & cw or merged[-1][1] & aw):
            merged[-1][0] |= cw
            merged[-1][1] |= aw
        else:
            merged.append([set(cw), set(aw)])
    res = []
    for cw, aw in merged:
        ci = [i for i, (_, w) in enumerate(ct) if w in cw]
        ai = [j for j, (_, w) in enumerate(at) if w in aw]
        # context: the agreed tokens right before / after it (on the caption side; on the second
        # ASR's side when the captions have nothing there), contiguous
        if ci:
            toks, lo_i, hi_i, matched, words = ct, min(ci), max(ci), mi, cap_words
        else:
            toks, lo_i, hi_i, matched, words = at, min(ai), max(ai), mj, asr_words
        left, k = [], lo_i - 1
        while k >= 0 and k in matched and len(left) < context:
            left.insert(0, k)
            k -= 1
        right, k = [], hi_i + 1
        while k < len(toks) and k in matched and len(right) < context:
            right.append(k)
            k += 1
        own = [(float(cap_words[w]["start"]), float(cap_words[w]["end"])) for w in cw] + \
              [(float(asr_words[w]["start"]), float(asr_words[w]["end"])) for w in aw]
        ctx = list(own)
        for k in left + right:
            ctx.append((float(words[toks[k][1]]["start"]), float(words[toks[k][1]]["end"])))
            other = matched[k]
            ow = asr_words if words is cap_words else cap_words
            ot = at if words is cap_words else ct
            ctx.append((float(ow[ot[other][1]]["start"]), float(ow[ot[other][1]]["end"])))
        res.append({"cap": sorted(cw), "asr": sorted(aw),
                    "cap_tokens": [ct[i][0] for i in ci], "asr_tokens": [at[j][0] for j in ai],
                    "left": [toks[k][0] for k in left], "right": [toks[k][0] for k in right],
                    "lo": round(min(a for a, _ in own), 3), "hi": round(max(b for _, b in own), 3),
                    "ctx_lo": round(min(a for a, _ in ctx), 3),
                    "ctx_hi": round(max(b for _, b in ctx), 3)})
    return sorted(res, key=lambda s: s["lo"])


# ---------------------------------------------------------------- the vote

def tie_slice(words, lo, hi, pad=SLICE_PAD_S):
    """The tie-break words whose middle lies in [lo - pad, hi + pad] (one time base)."""
    return [w for w in words or []
            if lo - pad <= (float(w["start"]) + float(w["end"])) / 2.0 <= hi + pad]


def vote(stretch, tie_tokens, extra=EXTRA_COST):
    """("asr" | "caption" | None, (cost of the captions' reading, cost of the second ASR's)):
    each reading is the stretch with its agreed context on both sides, scored against the
    tie-break tokens of the same time span (reading_cost). The lower cost is the majority; a tie,
    or a tie-break transcript that shares no token with either stretch, is none (the captions
    stay)."""
    if not tie_tokens:
        return None, None
    x_c = stretch["left"] + stretch["cap_tokens"] + stretch["right"]
    x_a = stretch["left"] + stretch["asr_tokens"] + stretch["right"]
    d_c, d_a = reading_cost(x_c, tie_tokens, extra), reading_cost(x_a, tie_tokens, extra)
    d = (round(d_c, 2), round(d_a, 2))
    if not (lcs_len(stretch["cap_tokens"], tie_tokens) or lcs_len(stretch["asr_tokens"],
                                                                   tie_tokens)) \
            and (stretch["cap_tokens"] and stretch["asr_tokens"]):
        return None, d                          # a third reading: no majority
    if d_a < d_c:
        return "asr", d
    if d_c < d_a:
        return "caption", d
    return None, d


def _text(words):
    return " ".join(str(w["w"]).strip() for w in words)


def _dangles(word):
    """The gate's caption ending rule (quality_gate.ending_verdict): a filler / dangling last word."""
    from clipengine import quality_gate
    return quality_gate._normalize_core(word) in quality_gate.END_FILLERS or \
        quality_gate._is_dangling_tail(word)


def apply_votes(side, cap_idx, asr_words, results, start_s):
    """The sidecar with every stretch the second ASR won taken from it, the caption words'
    timings kept. `cap_idx` = caption_side() indexes (the stretches' "cap" positions index it),
    `asr_words` clip-relative, `results` = [(stretch, winner)]. Returns (new sidecar, fixes):
    fixes = [{"lo", "hi", "old", "new", "how"}] (source seconds)."""
    from clipengine import quality_gate
    side = [dict(w) for w in side]
    start_s = float(start_s)
    drop, add, fixes = set(), [], []
    for st, winner in results:
        if winner != "asr":
            continue
        old = [side[cap_idx[i]] for i in st["cap"]]
        new = [asr_words[j] for j in st["asr"]]
        fix = {"lo": round(st["lo"] + start_s, 3), "hi": round(st["hi"] + start_s, 3),
               "old": _text(old), "new": _text(new)}
        if old and new:
            a, b = float(old[0]["start"]), float(old[-1]["end"])
            n0, n1 = float(new[0]["start"]), float(new[-1]["end"])
            if len(old) == len(new):
                put = [{"w": n["w"], "start": o["start"], "end": o["end"]} for o, n in zip(old, new)]
            elif (n1 - n0) > CRAM_SPAN_K * (b - a) + CRAM_SPAN_S or \
                    len(new) > CRAM_WPS * max(b - a, 0.1):
                # the captions' span is far shorter than the speech the second
                # ASR heard there (11-10 tw_2916500966 1100002: 'Okay,' 0.3 s took 40 words, and
                # the eye-check saw the dialogue of 15-31 s crammed into 30.3-31.8 s): the second
                # ASR's own times, inside the caption words around the stretch
                keep = [cap_idx[i] for i in range(len(cap_idx)) if i not in st["cap"]]
                lo_b = max((float(side[k]["end"]) for k in keep
                            if float(side[k]["end"]) <= n0 + start_s + 0.05), default=float("-inf"))
                hi_b = min((float(side[k]["start"]) for k in keep
                            if float(side[k]["start"]) >= n1 + start_s - 0.05), default=float("inf"))
                put = [{"w": n["w"],
                        "start": round(min(max(float(n["start"]) + start_s, lo_b), hi_b), 3),
                        "end": round(min(max(float(n["end"]) + start_s, lo_b), hi_b), 3)}
                       for n in new]
                fix["timing"] = "asr2"
            else:                       # the second ASR's own rhythm over the captions' span
                k = (b - a) / max(1e-3, n1 - n0)
                put = [{"w": n["w"], "start": round(a + (float(n["start"]) - n0) * k, 3),
                        "end": round(a + (float(n["end"]) - n0) * k, 3)} for n in new]
            drop.update(cap_idx[i] for i in st["cap"])
            add += put
            fix["how"] = "replaced"
        elif new:                       # the captions do not carry it: crosstalk-flagged words
            lo, hi = float(new[0]["start"]) + start_s, float(new[-1]["end"]) + start_s
            want = quality_gate.norm_tokens(new)
            flagged = [k for k, w in enumerate(side) if w.get("cap", True) is False
                       and float(w["end"]) > lo - 0.3 and float(w["start"]) < hi + 0.3]
            got = quality_gate.norm_tokens([side[k] for k in flagged])
            hit = sum(1 for t in want if any(_tok_match(t, g) for g in got))
            if flagged and hit >= 0.6 * len(want):
                for k in flagged:
                    side[k].pop("cap", None)
                    side[k]["fix"] = "uncrossed"
                fix.update(how="unflagged", new=_text([side[k] for k in flagged]))
            else:                       # never heard by the captions: the second ASR's times
                before = [float(side[k]["end"]) for k in cap_idx
                          if float(side[k]["end"]) <= lo + 0.05]
                after = [float(side[k]["start"]) for k in cap_idx
                         if float(side[k]["start"]) >= hi - 0.05]
                a = max([lo] + before)
                b = min([hi] + after)
                if b - a < 0.05 * len(new):
                    fix["how"] = "no_room"
                    fixes.append(fix)
                    continue
                n0, n1 = float(new[0]["start"]) + start_s, float(new[-1]["end"]) + start_s
                k = (b - a) / max(1e-3, n1 - n0)
                add += [{"w": n["w"], "start": round(a + (float(n["start"]) + start_s - n0) * k, 3),
                         "end": round(a + (float(n["end"]) + start_s - n0) * k, 3)} for n in new]
                fix["how"] = "inserted"
        elif old:                       # the captions' words nobody else heard: not burned
            for i in st["cap"]:
                side[cap_idx[i]]["cap"] = False
                side[cap_idx[i]]["fix"] = "unheard"
            fix["how"] = "unburned"
        else:
            continue
        fixes.append(fix)
    for w in add:
        w["fix"] = "asr2"
    out = [w for k, w in enumerate(side) if k not in drop] + add
    return sorted(out, key=lambda w: float(w["start"])), fixes


# ---------------------------------------------------------------- voters

def youtube_id(url):
    m = YT_URL.search(str(url or ""))
    return m.group(1) if m else None


def window_offset(source):
    """Seconds into the VOD where a window `<parent>_wNN` starts (its range-keyed file
    `<parent>_<start_ms>-<end_ms>.mp4`), 0.0 for a whole source."""
    if not re.search(r"_w\d+$", str(source.get("source_id") or "")):
        return 0.0
    m = _WINDOW_FILE.search(str(source.get("local_path") or ""))
    return int(m.group(1)) / 1000.0 if m else None


def parse_json3(doc):
    """[{w, start, end}] (VOD seconds) of a YouTube json3 subtitle track. Auto tracks time every
    word (segs' tOffsetMs); a manual cue's words share its span by length. Speaker marks (">>")
    and bracketed sound tags ("[Music]") are dropped."""
    out = []
    for ev in (doc or {}).get("events") or []:
        segs = [s for s in ev.get("segs") or [] if str(s.get("utf8", "")).strip()]
        if not segs or "tStartMs" not in ev:
            continue
        t0 = float(ev["tStartMs"]) / 1000.0
        t1 = t0 + float(ev.get("dDurationMs") or 0) / 1000.0
        timed = len(segs) > 1 and any("tOffsetMs" in s for s in segs)
        if timed:
            starts = [t0 + float(s.get("tOffsetMs", 0)) / 1000.0 for s in segs]
            for k, s in enumerate(segs):
                end = starts[k + 1] if k + 1 < len(segs) else max(starts[k] + 0.2, min(t1, starts[k] + 1.0))
                for piece in _clean_sub(s["utf8"]).split():
                    out.append({"w": piece, "start": round(starts[k], 3), "end": round(end, 3)})
        else:
            pieces = _clean_sub(" ".join(s["utf8"] for s in segs)).split()
            total = sum(len(p) + 1 for p in pieces) or 1
            t = t0
            for p in pieces:
                d = (t1 - t0) * (len(p) + 1) / total
                out.append({"w": p, "start": round(t, 3), "end": round(t + d, 3)})
                t += d
    return sorted(out, key=lambda w: w["start"])


def _clean_sub(text):
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", str(text))
    return text.replace(">>", " ").replace("\n", " ")


def fetch_youtube_subs(video_id, runner=None):
    """(kind, words) of the video's English subtitles: ("manual", ...) when it has them, else
    ("auto", ...), else ("none", []). One yt-dlp call per kind, the pipeline's binary and
    cookies (ytdlp_env / ytcookies). `runner(argv) -> {name: text}` replaces yt-dlp (tests)."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    for kind, flags, langs in (("manual", ["--write-subs"], "en,en-US,en-GB,en-CA,en-AU"),
                               ("auto", ["--write-auto-subs"], "en-orig,en")):
        files = (runner or _ytdlp_subs)(url, flags, langs)
        if not files:
            continue
        order = [l for l in langs.split(",")]
        name = min(files, key=lambda n: next((i for i, l in enumerate(order)
                                              if n.endswith(f".{l}.json3")), len(order)))
        try:
            words = parse_json3(json.loads(files[name]))
        except ValueError:
            continue
        if words:
            return kind, words
    return "none", []


def _ytdlp_subs(url, flags, langs):
    from clipengine import engine
    from clipengine import ytdl
    with tempfile.TemporaryDirectory(prefix="ytsubs_") as tmp:
        argv = [ytdl.get_ytdlp_bin(), "--no-warnings", "--ignore-no-formats-error",
                "--skip-download", "--no-write-playlist-metafiles", *flags,
                "--sub-langs", langs, "--sub-format", "json3", "-o", str(Path(tmp) / "s"), url]
        ck = ytdl.scratch_copy(engine.COOKIES)
        if ck:
            argv[1:1] = ["--cookies", ck]
        ytdl.ensure_deno_path()
        try:
            subprocess.run(argv, capture_output=True, text=True, timeout=YT_TIMEOUT_S,
                           env=os.environ.copy())
        except (OSError, subprocess.SubprocessError):
            return {}
        return {p.name: p.read_text(encoding="utf-8") for p in Path(tmp).glob("*.json3")}


def youtube_words(source, cache_dir, runner=None, now=None, fetch=True):
    """(kind, words in the window's own time) of the source's YouTube subtitles, or (None, [])
    for a source that is not a YouTube video / whose window offset is unknown (or not cached
    when `fetch` is False). Cached per video under `cache_dir` (no subtitles: re-asked after
    YT_NONE_TTL_S)."""
    vid = youtube_id(source.get("url"))
    off = window_offset(source)
    if not vid or off is None:
        return None, []
    now = time.time() if now is None else now
    f = Path(cache_dir) / f"{vid}.json"
    rec = None
    if f.exists():
        try:
            rec = json.loads(f.read_text())
        except ValueError:
            rec = None
        if rec and rec.get("kind") == "none" and now - float(rec.get("at") or 0) > YT_NONE_TTL_S:
            rec = None
    if rec is None and not fetch:
        return None, []
    if rec is None:
        kind, words = fetch_youtube_subs(vid, runner=runner)
        rec = {"kind": kind, "words": words, "at": now, "video": vid}
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec))
            tmp.replace(f)
        except OSError:
            pass
    words = [{"w": w["w"], "start": round(float(w["start"]) - off, 3),
              "end": round(float(w["end"]) - off, 3)} for w in rec.get("words") or []]
    return rec.get("kind"), words


def third_words(media, lo, hi, model, timeout=None):
    """Tie-break words of [lo, hi] (source seconds) from `model`, via the repair lane's whisper
    call (the window's warm worker when it runs)."""
    from clipengine import strong_asr
    return [{k: w[k] for k in ("w", "start", "end")}
            for w in strong_asr.transcribe_strong(media, lo, hi, model=model, timeout=timeout)]


# ---------------------------------------------------------------- reconcile

def reconcile(side, start_s, end_s, asr_words, *, media, source, settings_=None,
              yt_fn=None, third_fn=None, deadline=None):
    """(new sidecar, report) for a clip whose captions failed the caption check. `side` = the
    words sidecar (source s), `asr_words` = the second ASR's words (clip-relative, as
    second_asr.transcribe returns them). Votes come from the source's YouTube subtitles when it
    has them, else from the tie-break model on each stretch +- VOTE_PAD_S. The report lists every
    stretch (old -> new, vote source, winner) and `verdict` = the gate's caption check on the
    repaired captions. new sidecar is None when no stretch changed."""
    from clipengine import quality_gate
    s = settings_ or settings()
    start_s, end_s = float(start_s), float(end_s)
    dur = end_s - start_s
    cap_idx = caption_side(side, start_s, end_s)
    cap_words = [{"w": side[k]["w"], "start": float(side[k]["start"]) - start_s,
                  "end": float(side[k]["end"]) - start_s} for k in cap_idx]
    heard = [w for w in asr_words or [] if float(w["start"]) < dur]
    sts = stretches(cap_words, heard)
    rep = {"stretches": len(sts), "votes": [], "tiebreak_model": s["tiebreak_model"]}
    if not sts:
        return None, {**rep, "reason": "no disagreeing stretch"}
    kind, yt = (None, [])
    # a pull may take two yt-dlp calls: near the end of the repair's time only a cached one
    fetch = deadline is None or deadline - time.monotonic() >= 2 * YT_TIMEOUT_S
    if s.get("youtube", True):
        try:
            kind, yt = (yt_fn or (lambda src: youtube_words(src, s["cache_dir"],
                                                            fetch=fetch)))(source)
        except Exception as e:      # noqa: BLE001 - the third model votes instead
            rep["youtube_error"] = f"{type(e).__name__}: {str(e)[:160]}"
            kind, yt = None, []
    rep["youtube"] = kind
    why = None if third_fn else independent_tiebreak(s["tiebreak_model"])
    heard_third = {}
    # the tie-break model hears the stretches the subtitles do not cover (with their context,
    # +- VOTE_PAD_S)
    need = []
    for st in sts:
        lo, hi = st["ctx_lo"] + start_s, st["ctx_hi"] + start_s
        if not tie_slice(yt, lo, hi):
            need.append((max(0.0, lo - VOTE_PAD_S), hi + VOTE_PAD_S))
    # one call over all of them: whisper encodes 30 s of audio per call whatever the span, so
    # one call per stretch cost 6x one call (10-21 567: 210 s for six stretches)
    spans = [(min(a for a, _ in need), max(b for _, b in need))] if need else []
    for a, b in spans:
        if why:
            break
        if deadline is not None and time.monotonic() > deadline:
            rep["third_skipped"] = "repair time over"
            break
        try:
            left = None if deadline is None else max(20.0, deadline - time.monotonic())
            heard_third[(a, b)] = (third_fn or (lambda m, x, y, t: third_words(
                m, x, y, s["tiebreak_model"], timeout=t)))(media, a, b, left)
        except Exception as e:      # noqa: BLE001 - that stretch has no vote
            rep.setdefault("third_errors", []).append(f"{a:.1f}-{b:.1f}s: {type(e).__name__}: "
                                                      f"{str(e)[:120]}")
    if why:
        rep["third_skipped"] = why
    results = []
    for st in sts:
        lo, hi = st["ctx_lo"] + start_s, st["ctx_hi"] + start_s
        src, tie = None, tie_slice(yt, lo, hi)
        if tie:
            src = f"youtube-{kind}"
        else:
            for (a, b), words in heard_third.items():
                if a <= max(0.0, lo - VOTE_PAD_S) + 1e-6 and hi + VOTE_PAD_S <= b + 1e-6:
                    src, tie = s["tiebreak_model"], tie_slice(words, lo, hi)
                    break
        winner, d = vote(st, quality_gate.norm_tokens(tie)) if src else (None, None)
        results.append((st, winner))
        rep["votes"].append({
            "t": [round(st["lo"] + start_s, 2), round(st["hi"] + start_s, 2)],
            "caption": _text([cap_words[i] for i in st["cap"]]),
            "asr": _text([heard[j] for j in st["asr"]]),
            "context": [" ".join(st["left"]), " ".join(st["right"])],
            "tie": _text(tie), "source": src, "winner": winner or "none",
            "cost": list(d) if d else None})
    new, fixes = apply_votes(side, cap_idx, heard, results, start_s)
    # a stretch that leaves the captions ending on a filler / dangling word is not taken: the
    # second ASR hears past the cut, and its last word can be the next sentence's first (10-21
    # 558: "... need a whisker. A")
    dangling = lambda ws: bool(ws) and _dangles(ws[-1]["w"])              # noqa: E731
    while dangling(quality_gate.clip_words(new, start_s, end_s)) and \
            not dangling(quality_gate.clip_words(side, start_s, end_s)):
        last = [k for k, (st, w) in enumerate(results) if w == "asr"]
        if not last:
            break
        k = max(last, key=lambda i: results[i][0]["hi"])
        results[k] = (results[k][0], None)
        rep["votes"][k]["winner"] = "none"
        rep["votes"][k]["note"] = "left out: the captions would end on a dangling word"
        new, fixes = apply_votes(side, cap_idx, heard, results, start_s)
    rep["fixes"] = fixes
    if not [f for f in fixes if f.get("how") != "no_room"]:
        return None, {**rep, "reason": "no stretch went to the second ASR"}
    rep["verdict"] = quality_gate.caption_agreement_verdict(
        quality_gate.clip_words(new, start_s, end_s), asr_words, dur)
    return new, rep


def fix_lines(clip_id, rep):
    """One log line per stretch: clip, stretch, old -> new, vote source, winner."""
    out = []
    for v in (rep or {}).get("votes") or []:
        out.append(f"clip {clip_id}: caption stretch {v['t'][0]:.2f}-{v['t'][1]:.2f}s "
                   f"{v['caption']!r} -> {v['asr'] if v['winner'] == 'asr' else v['caption']!r}"
                   f" (vote {v['source'] or 'none'}: {v['tie']!r}, winner {v['winner']})")
    return out
