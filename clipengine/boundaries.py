#!/usr/bin/env python3
"""Boundary engine — where exactly a clip starts and stops. Pure (numpy + stdlib, no subprocess)
so every rule is unit-testable on synthetic data.

A clip is a sentence range. Its cuts are placed inside the inter-word gap on either side:
  start cut in [prev_word.end - 0.02, first_word.start]   (never after the first word begins)
  end cut   in [last_word.end, next_word.start + 0.02]    (never before the last word finishes)
With audio, the cut goes to the quietest 10 ms frame in that gap — this still works over a music
or crowd bed that never drops below a fixed silence threshold. Without audio, a small pad is used.

Shapes (absolute source seconds):
  Word     = {"w": str, "start": float, "end": float}
  Sentence = {"i": int, "start": float, "end": float, "text": str, "w0": int, "w1": int}
"""
import numpy as np

from clipengine.transcript import (_TRAILING_DANGLING, _ends_sentence, _tok_core,  # pure helpers (no
                        clean_edges, sentences)                         # subprocess at import)

FILLERS = ("so", "um", "uh", "yeah", "and", "like", "okay", "ok", "well", "right", "anyway",
           "you know", "i mean")
DANGLING = {"and", "but", "so", "because", "or", "the", "a", "to", "of", "with"}
NEVER_END = frozenset({"and", "so", "but", "like", "because", "or", "cause", "cuz", "cos", "oh",
                       "ohh", "ah", "ahh", "uh", "um", "umm", "hmm", "mm", "mhm", "er", "erm",
                       "huh", "eh"})   # = quality_gate.END_FILLERS: never a clip's last word
# filler endings: never a clip's last word either. The gate's ending check reads a
# NEVER_END word as a filler with or without a period (10-25 clip 979 ended on "So."); an article
# never ends a sentence. An END_OPENERS word ends a clip only with . ? ! ("do it." does; a bare
# "It" is the next sentence's first word, 10-25 clip 984).
END_DANGLING = NEVER_END | {"the", "a", "an"}
END_OPENERS = frozenset({"it"})
DEFAULT_CFG = {"min_dur_s": 15, "max_dur_s": 75, "lead_pad_s": 0.15, "tail_pad_s": 0.35}
SCENE_RMS_K = 1.5        # a shot change wins if its frame is within 1.5x of the gap's quietest frame
MAX_EXTEND = 3           # a too-short pick may grow by up to this many following sentences
HARD_GAP_S = 1.5         # a pause this long is a sentence boundary even without . ? !
TAG_GAP_S = 0.45         # tail-tag must start within this gap of the chosen end sentence
SNAP_MAX_S = 8.0         # clip-quality 10-21 stage 1: a cut moves to a sentence edge at most this far
END_ROOM_S = 0.5         # source kept after an end cut (the gate needs 0.3 s to hear the next word)
PAUSE_FLOOR_DB = 30.0    # start-fix 10-22: a frame this far under the loud (p90) frame is silence,
PAUSE_MIN_S = 0.15       # ... and a silent run this long is a pause between words, not a stop
                         # closure inside one
TAIL_TAGS = frozenset((
    "to be honest", "honestly", "for real", "you know", "you know what i mean",
    "right", "though", "i think", "or something", "at all", "man", "bro",
    "for sure", "i swear",
))


def _norm(tok):
    """Lower-case, punctuation-free token ("Okay," -> "okay")."""
    return "".join(ch for ch in str(tok).lower() if ch.isalnum() or ch == "'").strip("'")


def dangling_end(tok):
    """True when a clip must not end on `tok`: an END_DANGLING word (punctuated or not), an
    END_OPENERS word without . ? !, or a clause that runs on (a trailing , ; :)."""
    t = str(tok).strip().rstrip("\"'”’)]}»")
    core = _norm(t)
    return core in END_DANGLING or t.endswith((",", ";", ":")) or (
        core in END_OPENERS and not t.endswith((".", "?", "!")))


# ---- the end rule: a clip never ends on the trailing wrap-up ----
# autoclip_mvp's timeline prompt ends a topic on its last core sentence, "excluding the summary or
# transition lines that follow"; Vizard and CaptionX refuse a throwaway last line the same way. In
# the 11-07..09 sample the eye-check's cut_end rejects named endings that drift into the next
# topic or close on a tag ("Yeah, you know what I'm saying"). Matched on the normalized sentence
# (lower case, punctuation off); a niche's prompt pack adds its own openers (`wrapup_extra`).
WRAPUP_OPENERS = (
    "anyway", "anyways", "anyhow", "so anyway", "so anyways", "but anyway", "but anyways",
    "and anyway", "yeah anyway", "okay anyway", "ok anyway", "alright anyway", "all right anyway",
    "moving on", "let's move on", "lets move on", "so moving on", "but moving on",
    "on to the next", "onto the next", "next question", "next topic", "next up",
    "speaking of which", "on another note", "on a different note", "changing the subject",
    "switching gears", "that's all for", "that's it for", "that's all we have", "that's all i have",
    "thanks for watching", "thank you for watching", "thank you guys for watching",
    "thank you so much for watching", "thanks for tuning in", "thanks for listening",
    "thank you for listening", "see you next", "see you in the next", "see you tomorrow",
    "see you guys next", "see you guys later", "see you guys tomorrow", "see ya", "catch you later",
    "catch you guys", "peace out", "until next time", "till next time", "like and subscribe",
    "don't forget to like", "don't forget to subscribe", "don't forget to hit",
    "don't forget to comment", "make sure to like", "make sure you like", "make sure to subscribe",
    "make sure you subscribe", "make sure to hit", "make sure you hit", "smash that like",
    "smash the like", "hit that like", "hit the like", "hit that subscribe", "hit subscribe",
    "link in the description", "links in the description", "link in my bio", "link in bio",
    "we'll be right back", "we will be right back", "be right back", "let's take a break",
    "let's take a quick break", "so yeah", "but yeah", "yeah so yeah",
)
WRAPUP_WHOLE = frozenset((
    "so that's that", "that's that", "that's about it", "that's pretty much it",
    "you know what i'm saying", "you know what i mean", "know what i'm saying",
    "yeah you know what i'm saying", "yeah you know what i mean", "if that makes sense",
    "and yeah", "anyway yeah",
))
WRAPUP_WORDS = frozenset(("yeah", "yep", "yup", "okay", "ok", "alright", "all", "right", "anyway",
                          "anyways", "so", "um", "uh", "mhm", "mm", "hmm", "cool", "well", "and",
                          "but"))   # a line of only these (at most 4, no "!") closes nothing


def _phrase(text):
    return " ".join(t for t in (_norm(w) for w in str(text).replace("…", " ").split()) if t)


def wrapup_sentence(text, extra=()):
    """True when the sentence `text` is a trailing wrap-up a clip must not end on: a sign-off or
    subscribe line, "anyway" / "so yeah" / "moving on", a turn to the next topic ("next question",
    "speaking of which"), a tag-only line ("You know what I'm saying?") or a line of only fillers
    ("Yeah.", "Okay so."). `extra`: a niche's own openers (prompt pack `wrapup`). A shouted
    reaction ("Yeah!") is not one."""
    ph = _phrase(text)
    if not ph:
        return False
    if ph in WRAPUP_WHOLE:
        return True
    for o in tuple(WRAPUP_OPENERS) + tuple(_phrase(x) for x in extra or ()):
        if o and (ph == o or ph.startswith(o + " ")):
            return True
    toks = ph.split()
    return len(toks) <= 4 and all(t in WRAPUP_WORDS for t in toks) and \
        not str(text).rstrip().rstrip("\"'”’)]}»").endswith("!")


def _end_rule(c):
    """(wrap(text) -> bool, inside(sentence) -> bool) for cfg `c`: `drop_wrapup` turns the wrap-up
    rule on (with `wrapup_extra`), `span_s` = (lo, hi) keeps a clip inside one topic (a sentence
    whose middle is inside it). Both off when the keys are absent: every caller outside select
    and the cut review keeps today's cuts."""
    extra = tuple(c.get("wrapup_extra") or ())
    wrap = (lambda t: wrapup_sentence(t, extra)) if c.get("drop_wrapup") else (lambda t: False)
    span = c.get("span_s")
    if span:
        lo, hi = float(span[0]), float(span[1])
        inside = lambda s: lo <= (float(s["start"]) + float(s["end"])) / 2.0 <= hi   # noqa: E731
    else:
        inside = lambda s: True                                                      # noqa: E731
    return wrap, inside


def _tail_tag_next(sents, s):
    """Index of a tag sentence (short phrase from TAIL_TAGS) right after `s`, or None. Whisper
    sometimes punctuates a period before conversational tails like "to be honest" that belong to
    the previous thought; when the gap is < TAG_GAP_S and the tail matches, it can be absorbed."""
    if s + 1 >= len(sents):
        return None
    b, nxt = sents[s], sents[s + 1]
    if float(nxt["start"]) - float(b["end"]) > TAG_GAP_S:
        return None
    toks = [t for t in (_norm(w) for w in str(nxt.get("text", "")).split()) if t]
    if not (0 < len(toks) <= 5):
        return None
    return s + 1 if " ".join(toks) in TAIL_TAGS else None


# ---- audio ----
def frame_rms(audio, sr, frame_s=0.01):
    """RMS per `frame_s` frame of a mono signal. Trailing partial frame is dropped."""
    a = np.asarray(audio, dtype=np.float64).ravel()
    n = max(1, int(round(sr * frame_s)))
    k = len(a) // n
    if k == 0:
        return np.zeros(0, dtype=np.float32)
    fr = a[:k * n].reshape(k, n)
    return np.sqrt((fr * fr).mean(axis=1)).astype(np.float32)


def _frame_range(rms, t0, frame_s, lo, hi):
    """Indexes [i_lo, i_hi] of frames whose centre lies in [lo, hi] (empty -> i_lo > i_hi)."""
    i_lo = max(0, int(np.ceil((lo - t0) / frame_s - 0.5)))
    i_hi = min(len(rms) - 1, int(np.floor((hi - t0) / frame_s - 0.5)))
    return i_lo, i_hi


def quietest_point(rms, t0, frame_s, lo, hi):
    """Absolute time of the minimum-RMS frame in [lo, hi]; `t0` is the absolute time of rms[0].
    Ties resolve to the frame closest to the centre of the range. A range narrower than one frame
    (or outside the audio) returns its midpoint. Result is always clamped to [lo, hi]."""
    if hi < lo:
        lo, hi = hi, lo
    rms = np.asarray(rms, dtype=np.float64)
    i_lo, i_hi = _frame_range(rms, t0, frame_s, lo, hi)
    if i_lo > i_hi or hi - lo < frame_s:
        return (lo + hi) / 2.0
    seg = rms[i_lo:i_hi + 1]
    ties = np.flatnonzero(seg <= seg.min() + 1e-12) + i_lo
    centre = (lo + hi) / 2.0
    times = t0 + (ties + 0.5) * frame_s
    t = float(times[np.argmin(np.abs(times - centre))])
    return min(max(t, lo), hi)


def _cut_in(rms, t0, frame_s, lo, hi, scene_times=None):
    """Quietest point in [lo, hi], unless a scene cut inside the range is nearly as quiet
    (frame RMS <= SCENE_RMS_K x range minimum) — a cut on a shot change looks intentional."""
    q = quietest_point(rms, t0, frame_s, lo, hi)
    if not scene_times:
        return q
    r = np.asarray(rms, dtype=np.float64)
    i_lo, i_hi = _frame_range(r, t0, frame_s, lo, hi)
    if i_lo > i_hi:
        return q
    floor = float(r[i_lo:i_hi + 1].min())
    best = None
    for s in scene_times:
        if not (lo <= s <= hi):
            continue
        i = min(max(int((s - t0) / frame_s), i_lo), i_hi)
        if r[i] <= SCENE_RMS_K * floor + 1e-12:
            key = (r[i], abs(s - q))
            if best is None or key < best[0]:
                best = (key, float(s))
    return best[1] if best else q


# ---- cuts ----
def start_bounds(words, wi, max_lead=0.6):
    """Hard search range for a start cut before words[wi]."""
    hi = float(words[wi]["start"])
    lo = hi - max_lead
    if wi > 0:
        lo = max(float(words[wi - 1]["end"]) - 0.02, lo)
    lo = max(0.0, lo)
    return min(lo, hi), hi                 # overlapping timestamps: never start after the word


def end_bounds(words, wj, max_tail=0.8):
    """Hard search range for an end cut after words[wj]."""
    lo = float(words[wj]["end"])
    hi = lo + max_tail
    if wj + 1 < len(words):
        hi = min(float(words[wj + 1]["start"]) + 0.02, hi)
    return lo, max(lo, hi)                 # overlapping timestamps: never end before the word


def start_cut(words, wi, rms=None, t0=0.0, frame_s=0.01, lead_pad=0.15, max_lead=0.6,
              scene_times=None):
    """Start cut for a clip whose first word is words[wi]. Always within start_bounds()."""
    lo, hi = start_bounds(words, wi, max_lead)
    if rms is not None and len(rms):
        t = _cut_in(rms, t0, frame_s, lo, hi, scene_times)
    else:
        ws = float(words[wi]["start"])
        t = ws - lead_pad
        if wi > 0:
            t = max(t, (float(words[wi - 1]["end"]) + ws) / 2.0)
    return min(max(t, lo), hi)


def end_cut(words, wj, rms=None, t0=0.0, frame_s=0.01, tail_pad=0.35, max_tail=0.8,
            scene_times=None):
    """End cut for a clip whose last word is words[wj]. Always within end_bounds()."""
    lo, hi = end_bounds(words, wj, max_tail)
    if rms is not None and len(rms):
        t = _cut_in(rms, t0, frame_s, lo, hi, scene_times)
    else:
        we = float(words[wj]["end"])
        t = we + tail_pad
        if wj + 1 < len(words):
            t = min(t, (we + float(words[wj + 1]["start"])) / 2.0)
    return min(max(t, lo), hi)


def trim_leading_fillers(words, w0, w1, fillers=FILLERS):
    """Advance w0 past leading filler tokens ("so", "um", "you know", ...) while there is a
    usable gap (>= 0.06 s) before the next word. Never advances past w1 - 3."""
    singles = {f for f in fillers if " " not in f}
    multis = [tuple(f.split()) for f in fillers if " " in f]
    while True:
        n = 0
        for m in multis:
            if w0 + len(m) - 1 <= w1 and tuple(_norm(words[w0 + k]["w"]) for k in range(len(m))) == m:
                n = len(m)
                break
        if not n and _norm(words[w0]["w"]) in singles:
            n = 1
        if not n:
            return w0
        nxt = w0 + n
        if nxt > w1 - 3 or float(words[nxt]["start"]) - float(words[nxt - 1]["end"]) < 0.06:
            return w0
        w0 = nxt


def clip_from_sentences(words, sents, s0, s1, rms=None, t0=0.0, cfg=None, frame_s=0.01,
                        scene_times=None):
    """Cut a clip spanning sentences s0..s1 (list positions). Returns
    {"start","end","w0","w1","s0","s1","boundary_type":"sentence"} or None.
    Too short -> extend s1 by up to MAX_EXTEND following sentences. Too long -> None
    (never truncate mid-thought)."""
    c = {**DEFAULT_CFG, **(cfg or {})}
    min_dur, max_dur = float(c["min_dur_s"]), float(c["max_dur_s"])
    if not (0 <= s0 <= s1 < len(sents)):
        return None
    w0 = trim_leading_fillers(words, sents[s0]["w0"], sents[s1]["w1"])
    start = start_cut(words, w0, rms, t0, frame_s, lead_pad=float(c["lead_pad_s"]),
                      scene_times=scene_times)
    for s in range(s1, min(len(sents) - 1, s1 + MAX_EXTEND) + 1):
        w1 = sents[s]["w1"]
        end = end_cut(words, w1, rms, t0, frame_s, tail_pad=float(c["tail_pad_s"]),
                      scene_times=scene_times)
        dur = end - start
        if dur > max_dur:
            return None
        if dur >= min_dur:
            tag = _tail_tag_next(sents, s)
            if tag is not None:
                t_w1 = sents[tag]["w1"]
                t_end = end_cut(words, t_w1, rms, t0, frame_s, tail_pad=float(c["tail_pad_s"]),
                                scene_times=scene_times)
                if t_end - start <= max_dur:
                    w1, end, s = t_w1, t_end, tag
            return {"start": start, "end": end, "w0": w0, "w1": w1, "s0": s0, "s1": s,
                    "boundary_type": "sentence"}
    return None


def clean_clip(words, sents, s0, s1, rms=None, t0=0.0, cfg=None, frame_s=0.01, scene_times=None,
               edges_clean=True, trim_long=False):
    """clip_from_sentences, but the clip starts on a true sentence start and ends on a true
    sentence end (transcript.clean_edges with cfg hard_gap_s, default HARD_GAP_S; never on a
    dangling_end word, _strict_edges).

    Options, cheapest first: s0 moves BACK to the nearest clean start and s1 FORWARD to the nearest
    clean end (further forward if still too short). When that overshoots max_dur_s, s0 moves
    FORWARD / s1 BACKWARD to a clean boundary inside the span. Cost = seconds added + 2x seconds
    dropped, so the pick changes as little as possible. When no clean pair fits the duration band,
    the side that cannot be clean falls back to clip_from_sentences behaviour (s0 as picked, s1
    extended by up to MAX_EXTEND sentences) and its flag is False; clean starts win over clean ends.
    `edges_clean` may be a (start, end) pair (transcript.clean_edges); cfg `strict_ends` /
    `min_end_gap_s` turn on the run-on / continuation / interjection end rules.

    A pick that is already longer than max_dur_s as given is rejected like clip_from_sentences does
    (the LLM chose a moment that does not fit), unless `trim_long` (precision pass: the refined
    transcript may merge sentences the base one split) lets it shrink to clean boundaries inside.

    Returns clip_from_sentences' dict plus "clean_start", "clean_end" (bools) and "clean_note"
    (why a side is not clean, else None), or None when nothing fits [min_dur_s, max_dur_s]."""
    c = {**DEFAULT_CFG, **(cfg or {})}
    min_dur, max_dur = float(c["min_dur_s"]), float(c["max_dur_s"])
    n = len(sents)
    if not (0 <= s0 <= s1 < n):
        return None
    # cfg span_s keeps every start and end inside the pick's topic (sentences lo..hi),
    # cfg drop_wrapup never ends on a trailing wrap-up (wrapup_sentence): the pick's own wrap-up
    # lines are dropped first, and the end never runs on past one
    wrap, inside = _end_rule(c)
    lo_k, hi_k = 0, n - 1
    if c.get("span_s"):
        ins = [k for k in range(n) if inside(sents[k])]
        if not ins:
            return None
        lo_k, hi_k = ins[0], ins[-1]
        s0, s1 = max(s0, lo_k), min(s1, hi_k)
        if s0 > s1:
            return None
    is_wrap = [wrap(s.get("text", "")) for s in sents]
    dropped = []
    while s1 > s0 and is_wrap[s1]:
        dropped.insert(0, str(sents[s1].get("text", "")))
        s1 -= 1
    cs, ce = _strict_edges(sents, c, edges_clean)
    st = lambda k: float(sents[k]["start"])          # noqa: E731
    en = lambda k: float(sents[k]["end"])            # noqa: E731

    starts = []                                       # (a, clean, cost in seconds)
    a = s0
    while a >= lo_k and not cs[a] and en(s0) - st(a) <= max_dur:
        a -= 1
    if a >= lo_k and cs[a] and en(s0) - st(a) <= max_dur:
        starts.append((a, True, st(s0) - st(a)))
    starts += [(k, True, 2.0 * (st(k) - st(s0))) for k in range(s0 + 1, s1 + 1) if cs[k]]
    if not cs[s0]:
        starts.append((s0, False, 0.0))
    ends = [(k, True, 2.0 * (en(s1) - en(k))) for k in range(s0, s1) if ce[k]]
    k = s1
    while k <= hi_k and en(k) - st(s1) <= max_dur:
        if is_wrap[k]:
            break                                     # the thought closed: never end past it
        if ce[k]:
            ends.append((k, True, en(k) - en(s1)))
        elif k <= s1 + MAX_EXTEND:
            ends.append((k, False, en(k) - en(s1)))
        k += 1

    start_at, end_at = {}, {}

    def cuts(a, b):
        w0 = trim_leading_fillers(words, sents[a]["w0"], sents[b]["w1"])
        if w0 not in start_at:
            start_at[w0] = start_cut(words, w0, rms, t0, frame_s, lead_pad=float(c["lead_pad_s"]),
                                     scene_times=scene_times)
        w1 = sents[b]["w1"]
        if w1 not in end_at:
            end_at[w1] = end_cut(words, w1, rms, t0, frame_s, tail_pad=float(c["tail_pad_s"]),
                                 scene_times=scene_times)
        return w0, w1, end_at[w1] - start_at[w0]

    if not trim_long and cuts(s0, s1)[2] > max_dur:
        return None
    best = None
    for a, a_ok, a_cost in starts:
        for b, b_ok, b_cost in ends:
            if a > b:
                continue
            w0, w1, dur = cuts(a, b)
            if not (min_dur <= dur <= max_dur):
                continue
            # ties keep the picked start (the hook), then the picked end
            key = ((not a_ok) * 2 + (not b_ok), round(a_cost + b_cost, 3), abs(a - s0), abs(b - s1))
            if best is None or key < best[0]:
                best = (key, a, b, a_ok, b_ok, w0, w1)
    if best is None:
        return None
    _, a, b, a_ok, b_ok, w0, w1 = best
    tag = _tail_tag_next(sents, b)
    if tag is not None and (tag > hi_k or is_wrap[tag]):
        tag = None
    if tag is not None:
        t_w1 = sents[tag]["w1"]
        if t_w1 not in end_at:
            end_at[t_w1] = end_cut(words, t_w1, rms, t0, frame_s, tail_pad=float(c["tail_pad_s"]),
                                   scene_times=scene_times)
        if end_at[t_w1] - start_at[w0] <= max_dur:
            b, w1, b_ok = tag, t_w1, ce[tag]
    why = [f"no clean {side} fits {min_dur:g}-{max_dur:g}s" for side, ok in (("start", a_ok), ("end", b_ok))
           if not ok]
    out = {"start": start_at[w0], "end": end_at[w1], "w0": w0, "w1": w1, "s0": a, "s1": b,
           "boundary_type": "sentence", "clean_start": a_ok, "clean_end": b_ok,
           "clean_note": "; ".join(why) or None}
    if dropped:
        out["wrapup_dropped"] = dropped
    return out


def _strict_edges(sents, c, edges_clean):
    """transcript.clean_edges with cfg hard_gap_s / min_end_gap_s / strict_ends, and (cfg
    strict_dangling, default on: select.strict_dangling in config) a sentence whose last word is
    dangling_end never ending cleanly (clean_edges lets a period make "So." an end; the gate's
    ending check does not). The starts are clean_edges' own."""
    cs, ce = clean_edges(sents, float(c.get("hard_gap_s", HARD_GAP_S)), edges_clean,
                         min_end_gap_s=float(c.get("min_end_gap_s", 0.0)),
                         strict_ends=bool(c.get("strict_ends", False)))
    if c.get("strict_dangling", True):
        ce = [e and not dangling_end((str(s.get("text", "")).split() or [""])[-1])
              for e, s in zip(ce, sents)]
    if c.get("drop_wrapup"):              # a wrap-up line is never a clean end
        wrap, _inside = _end_rule(c)
        ce = [e and not wrap(s.get("text", "")) for e, s in zip(ce, sents)]
    return cs, ce


def _sent_of(sents, k):
    return next(i for i, s in enumerate(sents) if s["w0"] <= k <= s["w1"])


def snap_clip(words, start, end, cfg=None, edges_clean=True, media_dur=None, rms=None, t0=0.0,
              frame_s=0.01, max_move=SNAP_MAX_S, end_room=END_ROOM_S):
    """The last word on a clip's cuts (clip-quality 10-21 stage 1), on the precise transcript
    `words`: after the precision pass and edge-verify (which moves cuts on its own short
    transcripts: 10-21 clip 306 ran on for a lone "Okay.", 314 opened mid-sentence after an
    "extend", 311 cut into its first word) the clip must still end on a clean sentence end and
    open on a clean sentence start (transcript.clean_edges: cfg hard_gap_s / strict_ends /
    min_end_gap_s; `edges_clean` as for clean_clip).

      end    its last sentence ends cleanly: kept; else the next clean end at most `max_move` s
             later (inside max_dur_s, `end_room` s before `media_dur`); else back to the last clean
             end inside the clip that keeps min_dur_s; else kept (flagged "none"). A clip whose
             last word is dangling_end ("So.", a bare "It") tries back first, then on (filler
             endings: drop the filler before adding a sentence)
      start  mirrored (a start is clean after any sentence end, strict or not): back to the
             clean start at most `max_move` s earlier, else forward
      a kept side whose cut is outside its word gap (placed on another transcript's word times)
      is re-cut in the gap (start_cut / end_cut, on `rms` when given)

    Returns {"start", "end", "snap": {"start": a, "end": a}}, a in ok | recut | extend |
    pull_back | none; the input cuts unchanged when the result would leave the duration band."""
    c = {**DEFAULT_CFG, **(cfg or {})}
    min_dur, max_dur = float(c["min_dur_s"]), float(c["max_dur_s"])
    start, end = float(start), float(end)
    same = {"start": start, "end": end}
    inside = [k for k, w in enumerate(words or [])
              if start <= (float(w["start"]) + float(w["end"])) / 2.0 <= end]
    if not inside:
        return {**same, "snap": {"start": "none", "end": "none"}, "note": "no words in the clip"}
    sents = sentences(words, gap_s=float(c.get("sentence_gap_s", 0.7)))
    _cs, ce = _strict_edges(sents, c, edges_clean)
    # a start only needs the sentence before it to end (. ? ! or a hard pause), as the gate's
    # sentence_bounds reads it: the strict run-on rule is for ends (10-21 PASS clip 427 opens on
    # "All right" 0.2 s after "That's awesome.", which strict ends would have pulled in)
    cs, _ce = _strict_edges(sents, {**c, "strict_ends": False, "min_end_gap_s": 0.0}, edges_clean)
    st = lambda k: float(sents[k]["start"])          # noqa: E731
    en = lambda k: float(sents[k]["end"])            # noqa: E731
    wi, wj = inside[0], inside[-1]
    sa, sb = _sent_of(sents, wi), _sent_of(sents, wj)
    limit = (float(media_dur) - end_room) if media_dur else float("inf")
    act = {"start": "ok", "end": "ok"}
    # an end never moves on past a wrap-up line
    # or out of the clip's topic, a start never back out of it
    wrap, in_span = _end_rule(c)
    stop = next((k for k in range(sb, len(sents)) if wrap(sents[k].get("text", ""))
                 or not in_span(sents[k])), len(sents))
    # end
    b = sb
    if not (wj == sents[sb]["w1"] and ce[sb] and en(sb) <= limit):
        fwd = [k for k in range(sb, stop) if ce[k] and end < en(k) <= end + max_move
               and en(k) <= limit and en(k) - start <= max_dur]
        back = [k for k in range(sa, sb + (wj == sents[sb]["w1"])) if ce[k] and en(k) < end
                and en(k) - start >= min_dur and en(k) <= limit]
        tries = [(fwd[:1], "extend"), (back[-1:], "pull_back")]
        if dangling_end(words[wj]["w"]):
            tries.reverse()
        b, act["end"] = next(((ks[0], how) for ks, how in tries if ks), (sb, "none"))
    # start
    a = sa
    filler_ok = wi <= trim_leading_fillers(words, sents[sa]["w0"], sents[b]["w1"])
    if not (filler_ok and cs[sa]):
        back = [k for k in range(0, sa + (wi > sents[sa]["w0"])) if cs[k]
                and float(words[wi]["start"]) - st(k) <= max_move and en(b) - st(k) <= max_dur
                and in_span(sents[k])]
        fwd = [k for k in range(sa + 1, b + 1) if cs[k] and en(b) - st(k) >= min_dur]
        if back:
            a, act["start"] = back[-1], "extend"
        elif fwd:
            a, act["start"] = fwd[0], "pull_back"
        else:
            act["start"] = "none"
    w0 = trim_leading_fillers(words, sents[a]["w0"], sents[b]["w1"]) if act["start"] != "none" \
        else wi
    w1 = sents[b]["w1"] if act["end"] != "none" else wj
    new_s, new_e = start, end
    lo, hi = start_bounds(words, w0)
    if act["start"] in ("extend", "pull_back") or not (lo - 1e-6 <= start <= hi + 1e-6):
        new_s = start_cut(words, w0, rms, t0, frame_s, lead_pad=float(c["lead_pad_s"]))
        act["start"] = "recut" if act["start"] == "ok" else act["start"]
    lo, hi = end_bounds(words, w1)
    if act["end"] in ("extend", "pull_back") or not (lo - 1e-6 <= end <= hi + 1e-6):
        new_e = end_cut(words, w1, rms, t0, frame_s, tail_pad=float(c["tail_pad_s"]))
        if media_dur:
            new_e = min(new_e, float(media_dur) - min(end_room, 0.3))
        act["end"] = "recut" if act["end"] == "ok" else act["end"]
    if not (min_dur <= new_e - new_s <= max_dur):
        return {**same, "snap": {"start": "none", "end": "none"},
                "note": f"snapped cut {new_s:.2f}-{new_e:.2f}s leaves {min_dur:g}-{max_dur:g}s"}
    return {"start": start if new_s == start else round(new_s, 3),
            "end": end if new_e == end else round(new_e, 3), "snap": act}


def recut_end(words, start, end, cfg=None, media_dur=None, edge_end=False, rms=None, t0=0.0,
              frame_s=0.01, max_move=SNAP_MAX_S, end_room=END_ROOM_S, min_shift=0.25):
    """A new end cut for a clip the gate failed on its ending only (clip-quality 10-21 stage 1:
    re-cut, don't reject). `words` = the precise transcript around the old end (absolute s; it
    may begin anywhere before the cut). The nearest OTHER clean sentence end (strict ends) at
    most `max_move` s away: later (inside max_dur_s and `end_room` s before `media_dur`) or
    earlier (keeping min_dur_s); cost = seconds added + 2x seconds dropped (clean_clip's), so a
    short completion beats a long trim. `edge_end`: the words run
    to the media end and silence follows them there (their last sentence may then end clean).
    Returns (end_s, info) or (None, why)."""
    c = {**DEFAULT_CFG, "strict_ends": True, "min_end_gap_s": 0.25, **(cfg or {})}
    min_dur, max_dur = float(c["min_dur_s"]), float(c["max_dur_s"])
    start, end = float(start), float(end)
    if not words:
        return None, "no words around the end"
    sents = sentences(words, gap_s=float(c.get("sentence_gap_s", 0.7)))
    _cs, ce = _strict_edges(sents, c, (False, bool(edge_end)))
    limit = (float(media_dur) - end_room) if media_dur else float("inf")
    best = None
    for k, s in enumerate(sents):
        if not ce[k]:
            continue
        w1 = s["w1"]
        e = end_cut(words, w1, rms, t0, frame_s, tail_pad=float(c["tail_pad_s"]))
        if media_dur:
            e = min(e, float(media_dur) - min(end_room, 0.3))
        if abs(e - end) < min_shift or float(s["end"]) > limit:
            continue
        if not (min_dur <= e - start <= max_dur) or abs(e - end) > max_move:
            continue
        if float(s["end"]) <= start:
            continue
        cost = (e - end) if e > end else 2.0 * (end - e)
        if best is None or cost < best[0]:
            best = (cost, e, k)
    if best is None:
        return None, (f"no other clean sentence end within {max_move:g} s of the cut that "
                      f"keeps the clip {min_dur:g}-{max_dur:g} s")
    _, e, k = best
    return round(e, 3), {"old_end": round(end, 3), "new_end": round(e, 3),
                         "action": "extend" if e > end else "trim",
                         "last": sents[k]["text"][-60:]}


def recut_start(words, start, end, cfg=None, edge_start=False, rms=None, t0=0.0, frame_s=0.01,
                max_move=SNAP_MAX_S, min_shift=0.25):
    """recut_end's mirror (start-fix 10-22): a new start cut for a clip the gate failed on its
    start. `words` = the precise transcript around the old start (absolute s; it may end anywhere
    after the cut). The nearest OTHER clean sentence start at most `max_move` s away: earlier
    (inside max_dur_s) or later (keeping min_dur_s), leading fillers trimmed as select trims them;
    cost = seconds added + 2x seconds dropped. A start is clean after any sentence end, strict or
    not (snap_clip's start rule, as the gate's sentence_bounds reads it). `edge_start`: the words
    begin at the media start and silence precedes them (their first sentence may then start
    clean). Returns (start_s, info) or (None, why)."""
    c = {**DEFAULT_CFG, **(cfg or {})}
    min_dur, max_dur = float(c["min_dur_s"]), float(c["max_dur_s"])
    start, end = float(start), float(end)
    if not words:
        return None, "no words around the start"
    sents = sentences(words, gap_s=float(c.get("sentence_gap_s", 0.7)))
    cs, _ce = _strict_edges(sents, {**c, "strict_ends": False, "min_end_gap_s": 0.0},
                            (bool(edge_start), False))
    best = None
    for k, s in enumerate(sents):
        if not cs[k]:
            continue
        w0 = trim_leading_fillers(words, s["w0"], len(words) - 1)
        if float(words[w0]["start"]) >= end:
            continue
        t = start_cut(words, w0, rms, t0, frame_s, lead_pad=float(c["lead_pad_s"]))
        if abs(t - start) < min_shift or abs(t - start) > max_move:
            continue
        if not (min_dur <= end - t <= max_dur):
            continue
        cost = (start - t) if t < start else 2.0 * (t - start)
        if best is None or cost < best[0]:
            best = (cost, t, k)
    if best is None:
        return None, (f"no other clean sentence start within {max_move:g} s of the cut that "
                      f"keeps the clip {min_dur:g}-{max_dur:g} s")
    _, t, k = best
    return round(t, 3), {"old_start": round(start, 3), "new_start": round(t, 3),
                         "start_action": "extend" if t < start else "trim",
                         "first": sents[k]["text"][:60]}


def pause_at(rms, t0, frame_s, t, floor_db=PAUSE_FLOOR_DB, min_s=PAUSE_MIN_S, slack=0.0):
    """(p0, p1) of a silent run of frames at time `t` (absolute s; rms[0] starts at t0), or
    within `slack` s of it: every frame at least `floor_db` under the 90th-percentile frame of
    `rms`, the run at least `min_s` long with voiced audio on both sides of it inside `rms`.
    p0 = where its first frame starts, p1 = where its last one ends (the voice comes back there).
    None when `t` is voiced, in a shorter dip, or the audio does not show both ends of a pause."""
    r = np.asarray(rms if rms is not None else [], dtype=np.float64)
    if not len(r):
        return None
    loud = float(np.percentile(r, 90))
    if loud <= 0:
        return None
    quiet = r <= loud * 10.0 ** (-float(floor_db) / 20.0)
    # the frames whose span meets [t - slack, t + slack]
    k_lo = max(0, int(np.floor((float(t) - slack - t0) / frame_s)))
    k_hi = min(len(r) - 1, int(np.floor((float(t) + slack - t0) / frame_s)))
    near = [k for k in range(k_lo, k_hi + 1) if quiet[k]]
    if not near:
        return None
    i = min(near, key=lambda k: abs(t0 + (k + 0.5) * frame_s - float(t)))
    a, b = i, i
    while a > 0 and quiet[a - 1]:
        a -= 1
    while b + 1 < len(r) and quiet[b + 1]:
        b += 1
    if a == 0 or b == len(r) - 1 or (b + 1 - a) * frame_s < min_s - 1e-9:
        return None
    return t0 + a * frame_s, t0 + (b + 1) * frame_s


def seat_words(words, ref, t, rms, t0, frame_s=0.01, tol=0.0, floor_db=PAUSE_FLOOR_DB,
               min_s=PAUSE_MIN_S):
    """`words` (a clip's burned caption words) with no word claiming the pause a cut at `t` sits
    in (start-fix 10-22). large-v3-turbo times the first word after a pause from the end of the
    word before it: 10-22 clip 529's 'I' ran 2.995-3.475 s over 0.37 s of silence (the voice comes
    back at 3.43 s, small.en heard it at 3.28 s), so its start cut at 3.275 s, in that silence,
    read to the gate as a cut inside 'I'. A word with the cut more than `tol` inside it (the gate's
    word_boundaries rule) moves off the cut only when
      - the audio at `t` is a pause (pause_at; the cut may sit up to `tol` outside it, as the
        gate lets a cut sit `tol` inside a word: 531's cut was 0.01 s before its pause), and
      - `ref` (the precise words the cut was placed on) has the same word overlapping it in time,
        wholly on one side of the cut;
    the word's edge then goes to the pause's edge on that side: start = where the voice comes back
    after the pause, or end = where it stops before it. A cut in voiced audio and words `ref`
    does not place are left alone (the gate judges them). Returns (words, moved): moved =
    [{"w", "from": [start, end], "to": [start, end]}]; the ASR time stays on the word as
    asr_start / asr_end."""
    t = float(t)
    hit = [k for k, w in enumerate(words or [])
           if float(w["start"]) + tol < t < float(w["end"]) - tol]
    if not hit:
        return words, []
    p = pause_at(rms, t0, frame_s, t, floor_db, min_s, slack=tol)
    if p is None:
        return words, []
    p0, p1 = p
    out, moved = list(words), []
    for k in hit:
        w = words[k]
        ws, we = float(w["start"]), float(w["end"])
        twins = [r for r in ref or [] if _norm(r["w"]) == _norm(w["w"])
                 and float(r["start"]) < we and float(r["end"]) > ws]
        if twins and all(float(r["start"]) >= t - 1e-6 for r in twins) and t < p1 < we:
            new = {**w, "start": round(p1, 3), "asr_start": w["start"]}
        elif twins and all(float(r["end"]) <= t + 1e-6 for r in twins) and ws < p0 < t:
            new = {**w, "end": round(p0, 3), "asr_end": w["end"]}
        else:
            continue
        out[k] = new
        moved.append({"w": w["w"], "from": [ws, we], "to": [new["start"], new["end"]]})
    return out, moved


# ---- clean cut points on the strong-model words ----
EXTEND_MAX_S = 6.0       # a cut may move out by up to this (an end to finish its sentence, a start
TRIM_MAX_S = 4.0         # back to its sentence's start) or in by up to this
ONSET_SLACK_S = 0.12     # a word whose claimed span opens (or closes) on a pause this close to its
                         # edge is seated on the voice
FALSE_START_MARKS = ("-", "—", "–", "...", "…")


LOCAL_AUDIO_S = 2.0      # pause_at reads its loudness reference this far around a word (start-fix's
                         # seat read +-2 s of window audio around the cut)


def local_pause(rms, t0, frame_s, t, half=LOCAL_AUDIO_S, **kw):
    """pause_at on the frames within `half` s of `t` only: the loud reference is the voice around
    it, not a span that may hold music or a quieter speaker."""
    r = np.asarray(rms if rms is not None else [], dtype=np.float64)
    i0 = max(0, int(np.floor((float(t) - half - t0) / frame_s)))
    i1 = min(len(r), int(np.ceil((float(t) + half - t0) / frame_s)))
    if i1 - i0 < 3:
        return None
    return pause_at(r[i0:i1], t0 + i0 * frame_s, frame_s, t, **kw)


END_GAP_DB = 15.0        # a clean end's gap: its audio this far under the voice around it (the
                         # gate's ending check reads a word's onset as silence 15 dB under the
                         # clip's median frame: quality_gate.ENDING_SILENCE_DB)
END_GAP_S = 0.25         # ... over this long after the last word (= select's min_end_gap_s)


def pause_between(rms, t0, frame_s, a, b, half=LOCAL_AUDIO_S, under_db=END_GAP_DB):
    """Whether [a, b] is a pause in the audio: its median frame at least `under_db` under the
    median frame of the audio within `half` s around it. None when that audio has no such quiet
    frames at all (its 10th percentile within `under_db` of its median: a music or crowd bed),
    so the audio cannot tell and the words decide."""
    r = np.asarray(rms if rms is not None else [], dtype=np.float64)
    if not len(r):
        return None
    db = 20.0 * np.log10(r + 1e-9)
    mid = (float(a) + float(b)) / 2.0
    i0 = max(0, int(np.floor((mid - half - t0) / frame_s)))
    i1 = min(len(db), int(np.ceil((mid + half - t0) / frame_s)))
    seg = db[i0:i1]
    j0 = max(0, int(np.floor((float(a) - t0) / frame_s)))
    j1 = min(len(db), int(np.ceil((float(b) - t0) / frame_s)))
    gap = db[j0:j1]
    if len(seg) < 10 or not len(gap):
        return None
    med = float(np.median(seg))
    if med - float(np.percentile(seg, 10)) < under_db:
        return None
    return float(np.median(gap)) <= med - under_db


def seat_onsets(words, rms, t0, frame_s=0.01, slack=ONSET_SLACK_S, floor_db=PAUSE_FLOOR_DB,
                min_s=PAUSE_MIN_S):
    """`words` with every word that claims a pause at its edge seated on the voice (start-fix
    10-22's seat_words, for every word rather than the one under a cut): large-v3-turbo times the
    first word after a pause from the end of the word before it (10-22 clip 529: 'I' 2.995-3.475
    s over 0.36 s of silence, the voice back at 3.42 s), so a sentence gap read on its words is
    0 s where the audio has a pause. A pause (pause_at) that begins within `slack` of a word's
    start and ends before the word does moves the start to where the voice comes back; one that
    ends within `slack` of its end and begins after its start moves the end to where the voice
    stops. Returns (words, moved); a moved word keeps its ASR time as asr_start / asr_end."""
    if rms is None or not len(rms):
        return list(words or []), []
    out, moved = [], []
    for w in words or []:
        ws, we = float(w["start"]), float(w["end"])
        new = dict(w)
        p = None
        for t in np.arange(ws, min(ws + slack, we) + 1e-9, frame_s):
            p = local_pause(rms, t0, frame_s, float(t), floor_db=floor_db, min_s=min_s)
            if p:
                break
        if p and p[0] <= ws + slack and p[1] < we - 0.04:
            new.update(start=round(p[1], 3), asr_start=w.get("asr_start", w["start"]))
        else:
            q = None
            for t in np.arange(we, max(we - slack, ws) - 1e-9, -frame_s):
                q = local_pause(rms, t0, frame_s, float(t), floor_db=floor_db, min_s=min_s)
                if q:
                    break
            if q and q[1] >= we - slack and q[0] > ws + 0.04:
                new.update(end=round(q[0], 3), asr_end=w.get("asr_end", w["end"]))
        if (new["start"], new["end"]) != (w["start"], w["end"]):
            moved.append({"w": w["w"], "from": [ws, we], "to": [new["start"], new["end"]]})
        out.append(new)
    return out, moved


# ---- the cuts seated on the audio, not on word times ----
CUT_LEAD_S = 0.15        # a start cut sits this far before the voice comes back ...
CUT_BREATH_S = 0.30      # ... an end cut this far after it stops: the last word's real offset
                         # plus a breath (SupoClip 0.35 s, autoclip 0.28 s: word end times land
                         # before the final consonant decays)
CUT_GUARD_S = 0.08       # ... and never closer than this to the voice on the other side
CUT_PAD_MAX_S = 0.6      # an end cut inside the voice moves out at most this far to where it stops
CUT_LEAD_MAX_S = 0.5     # a start cut inside the voice moves back at most this far to its onset
CUT_AUDIO_S = 3.0        # the loud reference: the audio within this of the cut


SHORT_WORD_S = 0.10      # whisper collapses a word to 0-0.1 s (often a sentence's last before a pause)
WORD_VOICE_DB = 15.0     # ... its voice runs on while the frames stay within this of the loud frame
SHORT_WORD_MAX_S = 1.2   # ... for at most this long
SHORT_ONSET_S = 0.08     # a quiet stop this short at its start (a plosive's closure) is the word's


def span_short_words(words, rms, t0, frame_s=0.01, short=SHORT_WORD_S, floor_db=WORD_VOICE_DB,
                     max_s=SHORT_WORD_MAX_S):
    """(words, moved): every word shorter than `short` s given the span its voice has on the
    audio: its end runs on from its start while the frames stay within `floor_db` of the loud
    frame around it, up to 0.02 s before the next word starts and at most `max_s`. 10-25 3172: the
    window's small.en pass timed 'hair?' 69.10-69.10 s and 'Bro,' 69.84 s, a 0.74 s "pause" the
    retrim took for a clean sentence end; the voice runs from 69.10 s into 'Bro,' with no pause, so
    the end it cut at 69.125 s was inside the word. A moved word keeps its ASR end as asr_end."""
    if rms is None or not len(rms):
        return list(words or []), []
    out, moved = [], []
    words = list(words or [])
    for k, w in enumerate(words):
        ws, we = float(w["start"]), float(w["end"])
        if we - ws >= short:
            out.append(w)
            continue
        cap = min(ws + max_s, float(words[k + 1]["start"]) - 0.02 if k + 1 < len(words)
                  else ws + max_s)
        got = _quiet_runs(rms, t0, frame_s, ws, CUT_AUDIO_S, floor_db)
        if got is None or cap <= we:
            out.append(w)
            continue
        quiet, base = got
        i = min(max(0, int(np.floor((ws - base) / frame_s))), len(quiet) - 1)
        j = i
        while j < len(quiet) and quiet[j] and (j - i) * frame_s < SHORT_ONSET_S:
            j += 1
        if j < len(quiet) and quiet[j]:
            out.append(w)                  # no voice at its start: nothing to give it
            continue
        while j < len(quiet) and not quiet[j]:
            j += 1
        end = round(min(base + j * frame_s, cap), 3)
        if end > we + frame_s:
            new = {**w, "end": end, "asr_end": w.get("asr_end", w["end"])}
            moved.append({"w": w["w"], "from": [ws, we], "to": [ws, end]})
            out.append(new)
        else:
            out.append(w)
    return out, moved


def _quiet_runs(rms, t0, frame_s, t, half, floor_db):
    """(quiet mask, slice t0) of the frames within `half` s of `t`: a frame at least `floor_db`
    under the slice's 90th-percentile frame is quiet. None when the slice is empty or silent."""
    r = np.asarray(rms if rms is not None else [], dtype=np.float64)
    i0 = max(0, int(np.floor((float(t) - half - t0) / frame_s)))
    i1 = min(len(r), int(np.ceil((float(t) + half - t0) / frame_s)))
    seg = r[i0:i1]
    if len(seg) < 3:
        return None
    loud = float(np.percentile(seg, 90))
    if loud <= 0:
        return None
    return seg <= loud * 10.0 ** (-float(floor_db) / 20.0), t0 + i0 * frame_s


def _run_at(quiet, k):
    """[a, b] frame indexes of the run of equal quiet[] values that holds frame k."""
    a = b = k
    while a > 0 and quiet[a - 1] == quiet[k]:
        a -= 1
    while b + 1 < len(quiet) and quiet[b + 1] == quiet[k]:
        b += 1
    return a, b


def seat_cuts(start, end, rms, t0=0.0, frame_s=0.01, lead=CUT_LEAD_S, breath=CUT_BREATH_S,
              guard=CUT_GUARD_S, pad_max=CUT_PAD_MAX_S, lead_max=CUT_LEAD_MAX_S,
              floor_db=PAUSE_FLOOR_DB, min_s=PAUSE_MIN_S, media_dur=None, min_tail=0.3,
              words=None):
    """((start, end), info): the cuts moved onto the window audio. Every pass the engine cuts on
    times its words its own way, and none of them is the audio: 10-25 3172's retrim cut 25 ms after
    'hair?', which the window's small.en pass timed 69.10-69.10 s, inside the word as spoken (the
    gate's second ASR heard it start 0.04 s before the cut and failed 'ends mid-sentence: last
    word your'). So:
      end    in a pause: at least `breath` after the voice stops, at least `guard` before it comes
             back; inside the voice: out to where the voice stops (<= pad_max on) plus `breath`;
      start  in a pause: at least `lead` before the voice comes back, `guard` after it stopped;
             inside the voice: back to where it began (<= lead_max back) less `lead`.
    A pause is a run of frames `floor_db` under the loud frame around the cut, `min_s` long. A cut
    never moves inward, never past media_dur - min_tail, and over a music or crowd bed with no
    such quiet (or with no audio) nothing moves. `words` (the words around the cut, any pass): a
    cut inside the voice does not move over a word that starts after the end cut / ends before
    the start cut (the voice there is the next / previous word, not the clip's own: 3062's
    'taxpayer.' and), nor over a word that starts after a pause the cut skips over.
    info["start"] / info["end"]: what moved and why."""
    s, e = float(start), float(end)
    info = {}
    if rms is None or not len(rms):
        return (s, e), info
    ws = sorted(words or [], key=lambda w: float(w["start"]))

    def word_in(a, b):
        """A word of `words` that is not the clip's own and sits in [a, b]."""
        for w in ws:
            m = (float(w["start"]) + float(w["end"])) / 2.0
            if a < m < b and not (s <= m <= e):
                return w
        return None
    n_min = max(1, int(round(min_s / frame_s)))
    cap = float(media_dur) - min_tail if media_dur else float("inf")
    got = _quiet_runs(rms, t0, frame_s, e, CUT_AUDIO_S, floor_db)
    if got is not None:
        quiet, base = got
        k = min(max(0, int(np.floor((e - base) / frame_s))), len(quiet) - 1)
        a, b = _run_at(quiet, k)
        new = None
        if quiet[k] and (b - a + 1 >= n_min or b == len(quiet) - 1):
            stops = base + a * frame_s
            back = base + (b + 1) * frame_s if b < len(quiet) - 1 else float("inf")
            new, how, ref = min(stops + breath, back - guard), "breath", (stops, back)
        elif not quiet[k]:
            j, limit = b + 1, k + int(round(pad_max / frame_s))
            while j < len(quiet) and j <= limit:
                qa, qb = _run_at(quiet, j)
                if qb - qa + 1 >= n_min or qb == len(quiet) - 1:
                    stops = base + qa * frame_s
                    back = base + (qb + 1) * frame_s if qb < len(quiet) - 1 else float("inf")
                    new, how, ref = min(stops + breath, back - guard), "offset", (stops, back)
                    break
                j = qb + 1
                if j < len(quiet):
                    j = _run_at(quiet, j)[1] + 1     # skip the voiced run after a short dip
        if new is not None and how == "offset" and word_in(e, ref[0]) is not None:
            info["end_kept"] = f"{word_in(e, ref[0])['w']!r} is said before the voice stops"
            new = None
        if new is None and not quiet[k]:
            # no pause the audio shows, but the clip's own last word runs past the cut (its
            # span_short_words span): its end + breath, when the gap to the next word allows
            own = [w for w in ws if s <= (float(w["start"]) + float(w["end"])) / 2.0 <= e]
            last = own[-1] if own else None
            if last is not None and float(last["end"]) > e + 0.02:
                nxt = next((w for w in ws if float(w["start"]) > float(last["start"]) + 1e-6
                            and w is not last), None)
                hi = float(nxt["start"]) - guard if nxt else float(last["end"]) + breath
                want = min(float(last["end"]) + breath, hi)
                if want >= float(last["end"]) + 0.1:
                    new, how = want, "word_end"
                    ref = (float(last["end"]), float(nxt["start"]) if nxt else float("inf"))
        if new is not None:
            new = min(new, cap)
            if new > e + 1e-6:
                info["end"] = {"how": how, "from": round(e, 3), "to": round(new, 3),
                               "voice_stops": round(ref[0], 3),
                               "voice_back": None if ref[1] == float("inf") else round(ref[1], 3)}
                e = new
    got = _quiet_runs(rms, t0, frame_s, s, CUT_AUDIO_S, floor_db)
    if got is not None:
        quiet, base = got
        k = min(max(0, int(np.floor((s - base) / frame_s))), len(quiet) - 1)
        a, b = _run_at(quiet, k)
        new = None
        if quiet[k] and (b - a + 1 >= n_min or a == 0):
            back = base + (b + 1) * frame_s
            stopped = base + a * frame_s if a > 0 else float("-inf")
            new, how, ref = max(back - lead, stopped + guard), "lead", (stopped, back)
        elif not quiet[k]:
            j, limit = a - 1, k - int(round(lead_max / frame_s))
            while j >= 0 and j >= limit:
                qa, qb = _run_at(quiet, j)
                if qb - qa + 1 >= n_min or qa == 0:
                    back = base + (qb + 1) * frame_s
                    stopped = base + qa * frame_s if qa > 0 else float("-inf")
                    new, how, ref = max(back - lead, stopped + guard), "onset", (stopped, back)
                    break
                j = qa - 1
                if j >= 0:
                    j = _run_at(quiet, j)[0] - 1
        if new is not None and how == "onset" and word_in(ref[1], s) is not None:
            info["start_kept"] = f"{word_in(ref[1], s)['w']!r} is said after the voice starts"
            new = None
        if new is not None:
            new = max(new, 0.0)
            if new < s - 1e-6:
                info["start"] = {"how": how, "from": round(s, 3), "to": round(new, 3),
                                 "voice_back": round(ref[1], 3),
                                 "voice_stopped": None if ref[0] == float("-inf") else round(ref[0], 3)}
                s = new
    return (round(s, 3), round(e, 3)), info


def false_start(words, w0, w1):
    """The first word of a clean start past a false start at words[w0] ("I- I think", "the, the
    thing", "we were, we were going"): a cut-off word (FALSE_START_MARKS) or a word / two-word run
    the speaker says again right after it. w0 when there is none; None when the restart has no
    usable gap (>= 0.06 s) before it, so no clean start is there."""
    k = w0
    for _ in range(3):
        core = _norm(words[k]["w"])
        tok = str(words[k]["w"]).strip()
        n = 0
        if tok.endswith(FALSE_START_MARKS) and k + 1 <= w1:
            n = 1
        elif k + 1 <= w1 and core and core == _norm(words[k + 1]["w"]):
            n = 1
        elif k + 3 <= w1 and (core, _norm(words[k + 1]["w"])) == \
                (_norm(words[k + 2]["w"]), _norm(words[k + 3]["w"])):
            n = 2
        if not n:
            return k
        if float(words[k + n]["start"]) - float(words[k + n - 1]["end"]) < 0.06:
            return None
        k += n
    return k


def snap_sentences(words, start, end, cfg=None, edges_clean=True, media_dur=None, rms=None,
                   t0=0.0, frame_s=0.01, extend_max=EXTEND_MAX_S, trim_max=TRIM_MAX_S,
                   end_room=END_ROOM_S):
    """Cut points on sentence boundaries of the strong-model words: the clip
    opens on a clean sentence start (the one before it ends: . ? ! or a hard pause, leading
    fillers trimmed, never on a false start) and ends on a clean sentence end (strict ends: a
    pause after it, no continuation opener, never a dangling_end word), each cut
    in its word gap. A cut moves out by at most `extend_max` s (to finish the sentence it cuts, or
    to open the one it opens inside) or in by at most `trim_max` s; the pair that moves least
    wins (seconds added + 2x seconds dropped, clean_clip's cost), inside the vertical's duration
    band and `end_room` s before `media_dur`. With `rms`, an end also needs a pause in the audio
    right after its last word (pause_between: END_GAP_S at END_GAP_DB under the voice around it;
    an audio bed without quiet frames leaves it to the words). `words` should be seated on the audio first (seat_onsets). Returns {"start", "end", "snap": {"start": a, "end": a}} (a in ok | recut |
    extend | trim) or {"drop": why} when no clean pair fits."""
    c = {**DEFAULT_CFG, **(cfg or {})}
    min_dur, max_dur = float(c["min_dur_s"]), float(c["max_dur_s"])
    start, end = float(start), float(end)
    inside = [k for k, w in enumerate(words or [])
              if start <= (float(w["start"]) + float(w["end"])) / 2.0 <= end]
    if not inside:
        return {"drop": "no words in the clip"}
    sents = sentences(words, gap_s=float(c.get("sentence_gap_s", 0.7)))
    _cs, ce = _strict_edges(sents, {**c, "strict_ends": True,
                                    "min_end_gap_s": float(c.get("min_end_gap_s", 0.25))},
                            edges_clean)
    cs, _ce = _strict_edges(sents, {**c, "strict_ends": False, "min_end_gap_s": 0.0}, edges_clean)
    limit = (float(media_dur) - end_room) if media_dur else float("inf")
    wi, wj = inside[0], inside[-1]
    starts = []                                  # (cut, w0, sentence)
    for k, s in enumerate(sents):
        if not cs[k] or s["w0"] > wj:
            continue
        w0 = trim_leading_fillers(words, s["w0"], s["w1"])
        w0 = false_start(words, w0, s["w1"])
        if w0 is None:
            continue
        if w0 == wi and start_bounds(words, w0)[0] - 1e-6 <= start <= start_bounds(words, w0)[1] + 1e-6:
            t = start                            # the cut already opens it, in its gap
        elif w0 == wi and rms is not None and local_pause(rms, t0, frame_s, start):
            t = start                            # ... in an audio pause (start-fix)
        else:
            t = start_cut(words, w0, rms, t0, frame_s, lead_pad=float(c["lead_pad_s"]))
        if start - extend_max - 1e-6 <= t <= start + trim_max + 1e-6:
            starts.append((t, w0, k))
    ends = []
    for k, s in enumerate(sents):
        if not ce[k] or s["w1"] < wi or dangling_end(words[s["w1"]]["w"]):
            continue
        w1 = s["w1"]
        we = float(words[w1]["end"])
        if rms is not None and pause_between(rms, t0, frame_s, we + 0.02,
                                             we + END_GAP_S) is False:
            continue                             # the voice runs on: no pause after it (10-21 650:
                                                 # '82.' and 0.1 s later 'but', which the words
                                                 # timed 0.9 s later)
        if w1 == wj and end_bounds(words, w1)[0] - 1e-6 <= end <= end_bounds(words, w1)[1] + 1e-6:
            t = end
        elif w1 == wj and rms is not None and local_pause(rms, t0, frame_s, end):
            t = end
        else:
            t = end_cut(words, w1, rms, t0, frame_s, tail_pad=float(c["tail_pad_s"]))
            if media_dur:
                t = min(t, float(media_dur) - min(end_room, 0.3))
        if float(s["end"]) > limit:
            continue
        if end - trim_max - 1e-6 <= t <= end + extend_max + 1e-6:
            ends.append((t, w1, k))
    best = None
    for ts, w0, a in starts:
        for te, w1, b in ends:
            if w0 > w1 or not (min_dur <= te - ts <= max_dur):
                continue
            cost = (start - ts if ts < start else 2.0 * (ts - start)) + \
                   (te - end if te > end else 2.0 * (end - te))
            key = (round(cost, 3), abs(a - b))
            if best is None or key < best[0]:
                best = (key, ts, te, w0, w1)
    if best is None:
        side = "start" if not starts else "end" if not ends else "pair"
        why = {"start": f"no clean sentence start within -{extend_max:g}/+{trim_max:g} s",
               "end": f"no clean sentence end within +{extend_max:g}/-{trim_max:g} s",
               "pair": f"no clean start/end pair fits {min_dur:g}-{max_dur:g} s"}[side]
        return {"drop": why, "side": side}
    _, ts, te, w0, w1 = best

    def act(old, new, out_is_lower):
        if abs(new - old) < 1e-6:
            return "ok"
        moved_out = new < old if out_is_lower else new > old
        return "extend" if moved_out and abs(new - old) > 0.75 else \
            "trim" if not moved_out and abs(new - old) > 0.75 else "recut"
    return {"start": start if abs(ts - start) < 1e-6 else round(ts, 3),
            "end": end if abs(te - end) < 1e-6 else round(te, 3),
            "snap": {"start": act(start, ts, True), "end": act(end, te, False)},
            "first": words[w0]["w"], "last": words[w1]["w"]}


# ---- filler endings: the cuts on the words the gate reads ----
TIDY_TOL_S = 0.03        # = quality_gate.WORD_BOUNDARY_INSIDE_TOL_S: a cut this close inside a
                         # word's edge is on the edge
SENTENCE_PAUSE_S = 0.35  # = quality_gate.SENTENCE_PAUSE_S: a gap this long opens a sentence
TIDY_GAP_S = 0.8         # a cut sits at most this far after its last word / before its first


def _burned(w):
    """A sidecar word the clip burns as a caption (quality_gate.clip_words: not crosstalk)."""
    return w.get("cap", True) is not False and bool(str(w.get("w", "")).strip())


def _ends_clean(tok):
    """`tok` ends a sentence as the gate's sentence checks read one: . ? ! (no abbreviation), not
    a trailing-dangling word ("to." / "of."), not dangling_end."""
    return _ends_sentence(tok) and _tok_core(tok) not in _TRAILING_DANGLING and \
        not dangling_end(tok)


def _quiet(rms, t0, frame_s, a, b):
    """pause_between read as a yes unless the audio shows the voice running in [a, b]."""
    return pause_between(rms, t0, frame_s, a, b) is not False


def tidy_cut(words, start, end, cfg=None, rms=None, t0=0.0, frame_s=0.01, media_dur=None,
             max_move=SNAP_MAX_S, tol=TIDY_TOL_S, end_room=END_ROOM_S, start_forward=False):
    """The last look at a clip's cuts before its render, on the words the gate will read (filler
    endings). `words` = the clip's words sidecar as it is burned (absolute s; words
    outside the cut are context). large-v3-turbo times the first word after a pause from the end
    of the word before it, so a cut placed in the pause on select's words can sit inside the next
    word's claimed span (10-25 clip 979: 'So.' 37.775-38.175 s over the end cut at 37.965 s, the
    audio silent from 37.69 to 37.99 s; 984: 'It' 40.895-41.235 s over 41.09 s). The gate then
    fails the cut inside the word, and an ending whose last burned word is a filler.

      end    when the cut is inside a word or the last burned word is dangling_end: re-cut after
             the clip's last word (by midpoint) when it ends a sentence; else back to the sentence
             end before it (keeping min_dur_s); else on to the next one (inside max_dur_s, its word
             `end_room` s before `media_dur`); each at most `max_move` s; else the cut stays
      start  when the cut is inside a word: re-cut before the clip's first word when it opens a
             sentence; else back to the sentence start before it (inside max_dur_s), as snap_clip
             mends a start; else on to the next one (keeping min_dur_s)

    `start_forward` (the editor chose this start): a start inside a word looks on to
    the next sentence start before it looks back (11-09 3097 / 3119: back went 1-8 s onto the
    filler and the fragment the editor had cut off).

    A sentence end is a word _ends_clean with the audio pausing after it (END_GAP_S, pause_between);
    a sentence start a word whose previous word _ends_clean or lies SENTENCE_PAUSE_S before it (with
    no word before it in `words`: a pause in the audio before it), with no voice between the cut
    and it. The cut goes on the quietest frame between the two words (at most TIDY_GAP_S from its
    own) and that frame must be quiet. Needs `rms` (the window audio, rms[0] at `t0`).
    Returns {"start", "end", "tidy": {"start": a, "end": a}, "why": {side: str}, "first", "last"}
    with a in ok | recut | pull_back | extend | none (the side fired and nothing valid was found:
    the cut stays for the gate to judge)."""
    c = {**DEFAULT_CFG, **(cfg or {})}
    min_dur, max_dur = float(c["min_dur_s"]), float(c["max_dur_s"])
    start, end = float(start), float(end)
    ws = sorted(words or [], key=lambda w: float(w["start"]))
    S = lambda k: float(ws[k]["start"])              # noqa: E731
    E = lambda k: float(ws[k]["end"])                # noqa: E731
    mid = lambda k: (S(k) + E(k)) / 2.0              # noqa: E731
    out = {"start": start, "end": end, "tidy": {"start": "ok", "end": "ok"}, "why": {}}

    def straddled(t):
        return next((k for k in range(len(ws)) if S(k) + tol < t < E(k) - tol), None)

    burned = [k for k in range(len(ws)) if _burned(ws[k]) and S(k) < end and E(k) > start]
    k_s, k_e = straddled(start), straddled(end)
    if k_s is not None:
        out["why"]["start"] = f"cut inside {ws[k_s]['w']!r} ({S(k_s):.3f}-{E(k_s):.3f} s)"
    if k_e is not None:
        out["why"]["end"] = f"cut inside {ws[k_e]['w']!r} ({S(k_e):.3f}-{E(k_e):.3f} s)"
    elif burned and dangling_end(ws[burned[-1]]["w"]):
        out["why"]["end"] = f"ends on {ws[burned[-1]]['w']!r}"
    if not out["why"]:
        return out
    if rms is None or not len(rms):
        for side in out["why"]:
            out["tidy"][side] = "none"
        out["note"] = "no audio: the cuts stay"
        return out

    def quiet_frame(t):
        return _quiet(rms, t0, frame_s, t - frame_s, t + frame_s)

    def start_at(k):
        """The cut that opens the clip on ws[k], or None."""
        if not _burned(ws[k]):
            return None
        lo, hi = S(k) - TIDY_GAP_S, S(k)
        if k > 0:
            if not (_ends_clean(ws[k - 1]["w"]) or S(k) - E(k - 1) >= SENTENCE_PAUSE_S):
                return None
            lo = max(lo, E(k - 1))
        elif not pause_between(rms, t0, frame_s, S(k) - END_GAP_S, S(k) - 0.02):
            return None                      # nothing heard before it: the audio must pause there
        if hi < lo - tol:
            return None
        t = quietest_point(rms, t0, frame_s, min(lo, hi), hi)
        if not quiet_frame(t) or (S(k) - t >= 0.05 and not _quiet(rms, t0, frame_s, t, S(k))):
            return None                      # in speech, or a word nobody transcribed follows
        return max(0.0, round(t, 3))

    def end_at(k):
        """The cut that ends the clip after ws[k], or None."""
        if not (_burned(ws[k]) and _ends_clean(ws[k]["w"])):
            return None
        if media_dur and E(k) > float(media_dur) - end_room:
            return None
        lo, hi = E(k), E(k) + TIDY_GAP_S
        if k + 1 < len(ws):
            hi = min(hi, S(k + 1))           # the next word starts after the cut: not burned
        if media_dur:
            hi = min(hi, float(media_dur) - min(end_room, 0.3))
        if hi < lo - tol or not _quiet(rms, t0, frame_s, E(k) + 0.02, E(k) + END_GAP_S):
            return None                      # no gap, or the voice runs on after it
        t = quietest_point(rms, t0, frame_s, min(lo, hi), hi)
        return round(t, 3) if quiet_frame(t) else None

    def pick(here, before, after, at, ok, forward=False):
        """(index, cut, action): `here` re-cut, else the nearest of `before`, then `after`
        (`forward`: `after` first)."""
        order = ((after, None), (before[::-1], None)) if forward else \
            ((before[::-1], None), (after, None))
        for ks, how in (([here] if here is not None else [], "recut"),) + order:
            for k in ks:
                t = at(k)
                if t is not None and ok(t):
                    return k, t, how
        return None

    new_s = start
    if "start" in out["why"]:
        inside = [k for k in burned if start <= mid(k) <= end]
        f = inside[0] if inside else None
        ok = lambda t: abs(t - start) <= max_move and min_dur <= end - t <= max_dur   # noqa: E731
        got = f is not None and pick(f, list(range(f)),
                                     [k for k in range(f + 1, len(ws)) if S(k) < end], start_at, ok,
                                     forward=start_forward)
        if got:
            k, new_s, how = got
            out["tidy"]["start"] = how or ("pull_back" if new_s > start else "extend")
            out["first"] = ws[k]["w"]
        else:
            out["tidy"]["start"] = "none"
    if "end" in out["why"]:
        inside = [k for k in burned if new_s <= mid(k) <= end]
        last = inside[-1] if inside else None
        ok = lambda t: abs(t - end) <= max_move and min_dur <= t - new_s <= max_dur   # noqa: E731
        got = last is not None and pick(last, [k for k in range(last) if S(k) >= new_s],
                                        list(range(last + 1, len(ws))), end_at, ok)
        if got:
            k, t, how = got
            out["end"] = t
            out["tidy"]["end"] = how or ("pull_back" if t < end else "extend")
            out["last"] = ws[k]["w"]
        else:
            out["tidy"]["end"] = "none"
    out["start"] = new_s
    return out


# ---- the final cut: both cuts read on the words the clip burns ----
# The editor (cut_review) picks whole lines on select's base / small words; the caption step then
# burns large-v3-turbo's words, seated on the audio, and the gate reads its own second ASR. On the
# HEAD replay of 11-07..10 (59 clips) 13 clips failed the gate on cut_end
# ('skydance' + 'and', 'universe,' + 'the', a cut inside 'Indiana' / 'like'), and the eye-check
# rejected 10 of the 22 it judged on cut_start / cut_end; half of those were mechanical: a first
# word cut off ('We'), the tail of the sentence before ('Norwegian.'), the next sentence's first
# word ('Forrest', 'There's'). The final cut reads the burned words with their context (the strong
# pass hears 1.5 s before the cut and 2 s after it) and moves a cut that opens or ends inside a
# sentence, on a lone interjection, a false start or a wrap-up line to the nearest clean sentence
# edge: back to the sentence's start when it opens within the heard lead, on past a fragment
# otherwise; out to the sentence's end when it finishes within the heard tail, back to the last
# complete one otherwise.
INTERJECTIONS = frozenset((
    "yes", "yeah", "yep", "yup", "no", "nope", "nah", "okay", "ok", "oh", "ohh", "ah", "ahh", "uh",
    "um", "umm", "hmm", "mm", "mhm", "huh", "eh", "wow", "whoa", "right", "exactly", "sure",
    "true", "alright", "all", "cool", "nice", "damn", "dang", "bro", "dude", "man", "lol", "haha",
    "ha", "what", "really", "well", "so", "and", "but", "hey", "yo", "classic", "correct",
    "absolutely", "definitely", "totally", "thanks", "thank", "you", "god", "my", "gosh", "jeez"))
FINAL_LEAD_S = 1.6       # a start moves back at most this far (the strong pass hears 1.5 s before)
FINAL_TAIL_S = 1.6       # an end moves out at most this far (it hears 2 s after; the gate 2 s more)
FINAL_TRIM_S = 12.0      # a cut moves in at most this far
FINAL_GAP_S = 0.06       # two words this close leave no gap a cut can sit in
FINAL_PAUSE_S = 1.0      # an unpunctuated word this far before the next one closes its sentence
FINAL_NEXT_TOL_S = 0.05  # the next word may start this close before the end cut (the gate: 0.12)
FINAL_MAX_CANDS = 12
FINAL_IN_WORD_S = 0.03   # = quality_gate.WORD_BOUNDARY_INSIDE_TOL_S: a cut this far inside a word cuts it
# a sentence's first word an editor drops ("So | in my opinion, ...": 11-10 lvJjUyKnOl0, an
# eye-check pass): a start right after one of these, itself a sentence's first word, is clean
LEADING_CONNECTORS = frozenset(("so", "and", "but", "well", "like", "um", "uh", "okay", "ok",
                                "yeah", "now", "then", "also", "plus", "or", "oh", "alright"))


def _word_text(ws, a, b):
    return " ".join(str(w["w"]).strip() for w in ws[a:b + 1])


def lone_interjection(words):
    """True when `words` (one sentence) are only INTERJECTIONS, at most three ("Yes.", "Oh my
    gosh.", "Yeah. Right." is two sentences): a reply or a reaction, no line of its own."""
    toks = [t for t in (_norm(w["w"]) for w in words) if t]
    return 0 < len(toks) <= 3 and all(t in INTERJECTIONS for t in toks)


# a word a long pause after it does not close a sentence with (11-11 replay 3000005, an
# unpunctuated track: 'you' + 2 s + 'know exactly what these people', burned as a dangling 'you')
NO_PAUSE_END = frozenset(("you", "i", "we", "they", "he", "she", "it", "my", "your", "our",
                          "their", "his", "her", "this", "these", "those", "is", "are", "was",
                          "were", "be", "been", "have", "has", "had", "do", "does", "did", "can",
                          "could", "will", "would", "should", "just", "really", "very", "gonna",
                          "wanna", "gotta", "what", "who", "how", "why", "where", "when"))
CLOSE_TOL_S = 0.3        # another transcript's sentence end this close to a word's end closes it


def mark_closes(words, closes_at, tol=CLOSE_TOL_S):
    """`words` (copies) each marked "_close": whether it closes a sentence, with `closes_at` =
    the end times of the words that close one in the other transcripts of the same audio (the
    second ASR, select's): . ? ! on the word itself; never a dangling word; else another
    transcript's sentence end within `tol`, or a FINAL_PAUSE_S pause after a word not in
    NO_PAUSE_END. Unpunctuated tracks (large-v3-turbo: 6 of 10 HEAD replay clips, 11-11) are
    judged by the others' punctuation instead of by their pauses alone."""
    out = [dict(w) for w in words or []]
    ends = [float(w["end"]) for w in out]
    near = set()
    for x in closes_at or ():          # each sentence end goes to the one word ending nearest it
        if not ends:
            break
        k = min(range(len(ends)), key=lambda i: abs(ends[i] - float(x)))
        if abs(ends[k] - float(x)) <= tol:
            near.add(k)
    for k, w in enumerate(out):
        t = str(w["w"]).strip()
        if _ends_clean(t):
            w["_close"] = True
        elif dangling_end(t) or t.endswith(FALSE_START_MARKS):
            w["_close"] = False
        else:
            pause = k + 1 < len(out) and float(out[k + 1]["start"]) - ends[k] >= FINAL_PAUSE_S
            w["_close"] = bool(k in near or (pause and _norm(t) not in NO_PAUSE_END))
    return out


def _closes(ws, k):
    """ws[k] closes a sentence: its "_close" mark (mark_closes) when it has one; else . ? !
    (_ends_clean), or a long pause after a word that does not dangle (no punctuation from the
    model on it)."""
    if "_close" in ws[k]:
        return bool(ws[k]["_close"])
    if _ends_clean(ws[k]["w"]):
        return True
    t = str(ws[k]["w"]).strip()
    if dangling_end(t) or t.endswith(FALSE_START_MARKS):
        return False
    return k + 1 < len(ws) and float(ws[k + 1]["start"]) - float(ws[k]["end"]) >= FINAL_PAUSE_S


def _sents(ws, i0, i1):
    """[(a, b)] the sentences of ws[i0..i1], split after every word that closes one."""
    out, a = [], i0
    for k in range(i0, i1 + 1):
        if k == i1 or _closes(ws, k):
            out.append((a, k))
            a = k + 1
    return out


def sentence_spans(words, closes_at=None):
    """[(start, end)] (s) of the sentences of `words` (absolute), split after every word that
    closes one (_closes; mark_closes with `closes_at`): the final cut's lines."""
    ws = sorted((w for w in words or [] if str(w.get("w", "")).strip()),
                key=lambda w: float(w["start"]))
    if not ws:
        return []
    if closes_at is not None:
        ws = mark_closes(ws, closes_at)
    return [(float(ws[a]["start"]), float(ws[b]["end"])) for a, b in _sents(ws, 0, len(ws) - 1)]


def _opens(ws, k):
    """ws[k] opens a sentence: the word before it closes one (or there is none in `ws`), or it
    is a leading connector that opens one ("So | in my opinion")."""
    if k == 0 or _closes(ws, k - 1):
        return True
    return _norm(ws[k - 1]["w"]) in LEADING_CONNECTORS and \
        not str(ws[k - 1]["w"]).strip().endswith(FALSE_START_MARKS) and (k == 1 or _closes(ws, k - 2))


def _restart(ws, a, b):
    """The word the speaker restarts on after a false start at ws[a] (false_start's rule, also
    when no gap separates the two: the cut then sits on their seam), else a."""
    k = a
    for _ in range(3):
        core, tok, n = _norm(ws[k]["w"]), str(ws[k]["w"]).strip(), 0
        if tok.endswith(FALSE_START_MARKS) and k + 1 <= b:
            n = 1
        elif k + 1 <= b and core and core == _norm(ws[k + 1]["w"]):
            n = 1
        elif k + 3 <= b and (core, _norm(ws[k + 1]["w"])) == (_norm(ws[k + 2]["w"]),
                                                                _norm(ws[k + 3]["w"])):
            n = 2
        if not n:
            return k
        k += n
    return k


def _bad_first(ws, a, b, more):
    """Why the sentence ws[a..b] cannot open a clip (`more`: other lines follow it), or None."""
    t = str(ws[b]["w"]).strip()
    if t.endswith(FALSE_START_MARKS) or false_start(ws, a, b) != a:
        return f"opens on a false start {_word_text(ws, a, b)!r}"
    if more and lone_interjection(ws[a:b + 1]):
        return f"opens on a lone interjection {_word_text(ws, a, b)!r}"
    return None


def _bad_last(ws, a, b, more, extra=()):
    """Why the sentence ws[a..b] cannot close a clip (`more`: lines precede it), or None."""
    text = _word_text(ws, a, b)
    if more and (lone_interjection(ws[a:b + 1]) or wrapup_sentence(text, extra)):
        return f"ends on {text!r}"
    return None


def final_cut_issues(words, start, end, extra=(), tol=0.05, closes_at=None):
    """{"start": why, "end": why} of the cut [start, end] read on `words` (the clip's burned
    words with their context, absolute s, sorted): a start inside a sentence (the word before it
    closes none, and the first word is lower case, follows it within SENTENCE_PAUSE_S, or opens a
    fragment of at most two words: 'Norwegian.'), a false start or a lone interjection before the
    first line; an end whose next word starts before the cut, inside a sentence (no . ? !, no long
    pause after it), or on a lone interjection or a wrap-up line after the last one. {} when both
    are clean, or no word is inside the cut. `closes_at`: mark_closes' evidence from the other
    transcripts."""
    ws = sorted(words or [], key=lambda w: float(w["start"]))
    if closes_at is not None:
        ws = mark_closes(ws, closes_at)
    mid = [(float(w["start"]) + float(w["end"])) / 2.0 for w in ws]
    inside = [k for k in range(len(ws)) if start - tol <= mid[k] <= end + tol]
    if not inside:
        return {}
    i0, i1 = inside[0], inside[-1]
    sents = _sents(ws, i0, i1)
    out = {}
    if float(ws[i0]["start"]) < float(start) - FINAL_IN_WORD_S:
        out["start"] = f"the cut is inside the first word {str(ws[i0]['w']).strip()!r}"
    elif i0 > 0 and not _opens(ws, i0):
        first = str(ws[i0]["w"]).strip()
        gap = float(ws[i0]["start"]) - float(ws[i0 - 1]["end"])
        lower = first[:1].islower()
        if lower or gap < SENTENCE_PAUSE_S or sents[0][1] - sents[0][0] < 2:
            out["start"] = (f"opens mid-sentence: {str(ws[i0 - 1]['w']).strip()!r} before "
                            f"{first!r} closes no sentence")
    if "start" not in out:
        why = _bad_first(ws, sents[0][0], sents[0][1], len(sents) > 1)
        if why:
            out["start"] = why
    nxt = ws[i1 + 1] if i1 + 1 < len(ws) else None
    last = str(ws[i1]["w"]).strip()
    if float(ws[i1]["end"]) > float(end) + FINAL_IN_WORD_S:
        out["end"] = f"the cut is inside the last word {last!r}"
    elif nxt is not None and float(nxt["start"]) < float(end) - FINAL_NEXT_TOL_S:
        out["end"] = f"the next word {str(nxt['w']).strip()!r} starts before the cut"
    elif not _closes(ws, i1) and (nxt is not None or dangling_end(last)):
        out["end"] = f"ends mid-sentence on {last!r}"
    else:
        why = _bad_last(ws, sents[-1][0], sents[-1][1], len(sents) > 1, extra)
        if why:
            out["end"] = why
    return out


FINAL_NEAR_S = 0.4       # the quietest frame is looked for this close to the pad's point


def _gap_cut(rms, t0, frame_s, lo, hi, want):
    """A cut in [lo, hi]: the quietest frame within FINAL_NEAR_S of `want` with audio, else
    `want` clamped into it. None when hi < lo (callers widen a tight gap to the two words' edges
    +- 0.02 s: the gate's word check allows FINAL_IN_WORD_S)."""
    if hi < lo:
        return None
    t = min(max(want, lo), hi)
    if rms is not None and len(rms):
        a, b = max(lo, t - FINAL_NEAR_S), min(hi, t + FINAL_NEAR_S)
        try:
            t = quietest_point(rms, t0, frame_s, a, b)
        except Exception:                 # noqa: BLE001 - the pad then
            pass
    return round(min(max(t, lo), hi), 3)


def final_cut_candidates(words, start, end, issues, cfg=None, rms=None, t0=0.0, frame_s=0.01,
                         media_dur=None, cover=None, extra=(), lead_s=FINAL_LEAD_S,
                         tail_s=FINAL_TAIL_S, trim_s=FINAL_TRIM_S, tol=0.05, closes_at=None,
                         end_at_least=None):
    """[(start, end, info)] cuts on clean sentence edges for a cut with `issues`
    (final_cut_issues), best first: only the sides with an issue move. A start goes on a word
    that opens a sentence (back to the opening of the sentence the cut splits when that is at
    most `lead_s` before the cut, else on past the fragment, at most `trim_s`); an end after a
    word that closes one (out to the end of the sentence the cut splits within `tail_s`, else back
    to the last complete line, at most `trim_s`); the first line is no false start or lone
    interjection, the last no lone interjection or wrap-up line. Both words must lie in `cover`
    (lo, hi: the span the burned words cover, so a move never burns words nobody heard), the cut
    sits in the gap before / after them (CUT_LEAD_S / CUT_BREATH_S, the quietest frame with
    `rms`), and the clip keeps cfg min_dur_s..max_dur_s and END_ROOM_S before `media_dur`.
    `end_at_least` (s): the last word ends no earlier (the line chat reacted to after the cut).
    info: {"start": "kept" | "back" | "on", "end": "kept" | "out" | "back", "first", "last"}."""
    c = {**DEFAULT_CFG, **(cfg or {})}
    min_dur, max_dur = float(c["min_dur_s"]), float(c["max_dur_s"])
    ws = sorted(words or [], key=lambda w: float(w["start"]))
    if closes_at is not None:
        ws = mark_closes(ws, closes_at)
    S = lambda k: float(ws[k]["start"])            # noqa: E731
    E = lambda k: float(ws[k]["end"])              # noqa: E731
    mid = [(S(k) + E(k)) / 2.0 for k in range(len(ws))]
    inside = [k for k in range(len(ws)) if start - tol <= mid[k] <= end + tol]
    if not inside:
        return []
    i0, i1 = inside[0], inside[-1]
    lo_c, hi_c = cover if cover else (float("-inf"), float("inf"))
    in_cover = lambda k: lo_c - tol <= mid[k] <= hi_c + tol      # noqa: E731

    def start_cut(k):
        if k == 0:
            return None                        # nothing heard before it: no gap to judge
        lo, hi = E(k - 1) + 0.02, S(k) - 0.02
        if hi - lo < FINAL_GAP_S - 0.04:       # words that abut: on the seam between them
            lo, hi = min(E(k - 1), S(k)) - 0.0, max(E(k - 1), S(k))
            if S(k) - E(k - 1) < -FINAL_IN_WORD_S:
                return None                    # they overlap: no seam a cut can sit on
            return round((lo + hi) / 2.0, 3)
        return _gap_cut(rms, t0, frame_s, lo, hi, S(k) - CUT_LEAD_S)

    def end_cut(k):
        lo = E(k) + 0.02
        if k + 1 < len(ws):
            hi = S(k + 1) - CUT_GUARD_S
            if hi - lo < FINAL_GAP_S - 0.04:
                hi = S(k + 1) - 0.02           # a tight gap: up to the next word's onset
            if hi < lo:                        # words that abut: on the seam between them
                if S(k + 1) - E(k) < -FINAL_IN_WORD_S:
                    return None
                t = round((E(k) + S(k + 1)) / 2.0, 3)
                return t if not media_dur or t <= float(media_dur) - END_ROOM_S else None
        else:
            hi = E(k) + CUT_BREATH_S
        if media_dur:
            hi = min(hi, float(media_dur) - END_ROOM_S)
        return _gap_cut(rms, t0, frame_s, lo, hi, E(k) + CUT_BREATH_S)

    starts = [(i0, float(start), "kept")]
    if "start" in issues:
        starts = []
        # a false start ("Do Do you remember...", "Why why is there..."): its restart opens too
        restarts = set()
        for a, b in _sents(ws, max(0, i0 - 1), i1)[:2]:
            r = _restart(ws, a, b)
            if r != a:
                restarts.add(r)
        for k in range(len(ws)):
            if not (in_cover(k) and (_opens(ws, k) or k in restarts)) or k > i1:
                continue
            if not (start - lead_s <= S(k) <= start + trim_s):
                continue
            ss = _sents(ws, k, i1)
            if _bad_first(ws, ss[0][0], ss[0][1], len(ss) > 1):
                continue
            t = start_cut(k)
            if t is not None:
                starts.append((k, t, "back" if t < float(start) else "on"))
    ends = [(i1, float(end), "kept")]
    if "end" in issues:
        ends = []
        for k in range(len(ws)):
            if not (in_cover(k) and _closes(ws, k)) or k < i0:
                continue
            if not (end - trim_s <= E(k) <= end + tail_s):
                continue
            if end_at_least is not None and E(k) < float(end_at_least) - tol:
                continue
            t = end_cut(k)
            if t is not None:
                ends.append((k, t, "out" if t > float(end) else "back"))
    out = []
    for ks, ts, hs in starts:
        for ke, te, he in ends:
            if ke < ks or not (min_dur <= te - ts <= max_dur):
                continue
            ss = _sents(ws, ks, ke)
            if "start" in issues and _bad_first(ws, ss[0][0], ss[0][1], len(ss) > 1):
                continue
            if "end" in issues and _bad_last(ws, ss[-1][0], ss[-1][1], len(ss) > 1, extra):
                continue
            # an end that finishes the split sentence beats one that drops it; a start that
            # takes in the split sentence's opening beats one that skips its tail
            cost = abs(ts - start) * (0.5 if hs == "back" else 1.0) + \
                abs(te - end) * (0.5 if he == "out" else 1.0)
            out.append((cost, ts, te, {"start": hs, "end": he, "first": str(ws[ks]["w"]).strip(),
                                       "last": str(ws[ke]["w"]).strip()}))
    out.sort(key=lambda x: x[0])
    return [(ts, te, info) for _c, ts, te, info in out[:FINAL_MAX_CANDS]]


def sentences_around(sents, peak, pre_s=12.0, post_s=8.0):
    """(s0, s1) for an energy peak with no LLM: s0 = sentence starting closest to peak-pre_s
    (and <= peak), s1 = sentence ending closest to peak+post_s (and >= peak). None if absent."""
    before = [k for k, s in enumerate(sents) if s["start"] <= peak]
    after = [k for k, s in enumerate(sents) if s["end"] >= peak]
    if not before or not after:
        return None
    s0 = min(before, key=lambda k: abs(sents[k]["start"] - (peak - pre_s)))
    s1 = min(after, key=lambda k: abs(sents[k]["end"] - (peak + post_s)))
    return (s0, s1) if s0 <= s1 else None


# ---- low-speech (no words) ----
def energy_clip(env, t0, frame_s, peak, median, min_dur, max_dur, pre=(20.0, 3.0), quiet_s=1.5,
                scene_times=None):
    """Cuts for a peak in footage without speech, from an energy envelope (`env[i]` covers
    t0 + i*frame_s). Start: the quietest point (or a near-quiet scene cut) in
    [peak-pre[0], peak-pre[1]], searched from the last moment the envelope was below `median`.
    End: the first moment after the peak where the envelope stays below `median` for >= quiet_s.
    Duration must land in [min_dur, max_dur]: if the natural end is too early/late, the end moves
    to the next quiet stretch, else to the quietest point of the allowed end range. Every end is
    an envelope feature. Returns {"start","end","boundary_type":"energy"} or None."""
    env = np.asarray(env, dtype=np.float64)
    t_end = t0 + len(env) * frame_s
    lo, hi = max(t0, peak - pre[0]), min(t_end, peak - pre[1])
    if hi <= lo:
        return None
    i_lo, i_hi = _frame_range(env, t0, frame_s, lo, hi)
    quiet = np.flatnonzero(env[i_lo:i_hi + 1] < median) if i_lo <= i_hi else np.array([], int)
    if len(quiet):
        lo = max(lo, t0 + (i_lo + quiet[-1]) * frame_s)
    start = _cut_in(env, t0, frame_s, lo, hi, scene_times)

    need = max(1, int(np.ceil(quiet_s / frame_s)))
    below = env < median
    ends, run = [], 0
    for i in range(max(0, int((peak - t0) / frame_s)), len(env)):
        run = run + 1 if below[i] else 0
        if run == need:
            ends.append(t0 + (i - need + 1) * frame_s)     # the quiet stretch begins here
    ok = [e for e in ends if min_dur <= e - start <= max_dur]
    if ok:
        return {"start": start, "end": ok[0], "boundary_type": "energy"}
    # search range, not a fixed end; never before the peak (a pre-peak lull must not end the clip)
    elo, ehi = max(start + min_dur, peak), min(t_end, start + max_dur)
    if ehi <= elo:
        return None
    return {"start": start, "end": quietest_point(env, t0, frame_s, elo, ehi),
            "boundary_type": "energy"}


# ---- ranking ----
def iou(a, b):
    """Overlap of two clips divided by the length of the shorter one (0..1)."""
    ov = min(a["end"], b["end"]) - max(a["start"], b["start"])
    if ov <= 0:
        return 0.0
    short = min(a["end"] - a["start"], b["end"] - b["start"])
    return ov / short if short > 0 else 1.0


def suppress(clips, max_overlap=0.3, top_n=6):
    """Rank by score desc; drop any clip overlapping a kept, higher-scored clip by > max_overlap."""
    kept = []
    for c in sorted(clips, key=lambda c: -float(c.get("score") or 0.0)):
        if all(iou(c, k) <= max_overlap for k in kept):
            kept.append(c)
        if top_n and len(kept) >= top_n:
            break
    return kept


def ends_cleanly(sent_text):
    """True if the text ends in . ? ! and its final token is not a dangling conjunction or
    preposition. A scoring penalty signal, not a hard reject.

    ``like,`` / trailing commas / dangling fillers also fail this check (``like`` is
    a quotative tag — ``I was like, [actual thing]`` — so a clip ending on ``like`` is
    mid-clause even if a period follows)."""
    t = str(sent_text).strip().rstrip("\"'”’)]}»")
    if not t:
        return False
    toks = t.split()
    if not toks:
        return False
    if t[-1] in (",", ";", ":"):
        return False
    if t[-1] not in ".?!":
        return False
    last = _norm(toks[-1])
    if last in DANGLING:
        return False
    # quotative / dangling filler at the tail
    return last not in {"like", "cause", "cuz", "um", "uh"}
