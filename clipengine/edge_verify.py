#!/usr/bin/env python3
"""Real-audio edge verifier for selected clips.

After the precision pass has placed a clip's cuts, this stage transcribes the last few seconds
inside the clip and the first few seconds after it (mirror for the start) with the refine model.
End rules:
  1. If the in-clip TAIL does not end a sentence (last word not _ends_sentence, or last word in
     DANGLING), OR the OUT-AFTER audio begins within TAG_HEAD_S of the cut with a short
     (<= TAG_MAX_WORDS words) sentence ending in . ? !, then extend the clip end to that
     sentence's end (using end_cut over the new words) IF it still fits max_dur_s.
  2. Else if the current end is not a clean sentence end, pull the end BACK to the last clean
     sentence end still inside the clip if that keeps min_dur_s.
  3. Else keep the end and flag.
Start rule mirror: if the audio just before the start runs into the first in-clip word with a
gap < LEAD_GAP_S and the preceding word does not end a sentence, move the start back to that
sentence's start if it still fits.

Public API: verify(clip, words_window_fn, cfg) -> dict with
  {"start", "end", "action": {"start": "...", "end": "..."}} — every clip has both, values are one
of "noop", "extend", "pull_back", "blocked".
At most one action per side (no loops). Pure: the words_window_fn is injected so tests can
stub the transcript.
"""
from __future__ import annotations
from clipengine import boundaries
from clipengine import transcript

END_TAIL_S = 5.0             # seconds of in-clip tail to re-transcribe
END_LOOK_S = 4.0             # seconds after the cut to look at
START_HEAD_S = 5.0
START_LOOK_S = 4.0
TAG_HEAD_S = 0.5             # a short reaction tail must begin this soon after the cut
TAG_MAX_WORDS = 5
LEAD_GAP_S = 0.3             # if the pre-start word ends this close to the first in-clip word,
                             # the sentence carries over -- consider extending the start back.
DANGLING = boundaries.DANGLING


def _ends_sentence(text):
    return transcript._ends_sentence(text)


def _clean_stop(text):
    """A sentence end a clip may stop on. A question is not one: in interviews it is the host's
    next setup, so stopping there cuts before the answer (DSH #230: extend ran into "Were you
    profitable from the first year?", pull_back landed on "That's from a review?")."""
    return _ends_sentence(text) and not str(text).rstrip().endswith("?")


def _short_reaction(words, start_t):
    """True and (end_time, w_index) if a short (<=5 word) sentence ending in .?! begins within
    TAG_HEAD_S seconds of start_t and its last word ends within [start_t, start_t + max]."""
    if not words:
        return None
    if float(words[0]["start"]) - float(start_t) > TAG_HEAD_S:
        return None
    for i, w in enumerate(words[:TAG_MAX_WORDS]):
        if _clean_stop(w["w"]):
            return (i, float(w["end"]))
    return None


def _last_clean_end(words, end_t):
    """Latest word in `words` whose end is <= end_t and whose text ends a sentence. Returns
    (i, end_time) or None."""
    best = None
    for i, w in enumerate(words):
        if float(w["end"]) > end_t:
            break
        if _clean_stop(w["w"]):
            best = (i, float(w["end"]))
    return best


def _norm_lower(text):
    return "".join(ch for ch in str(text).lower() if ch.isalnum() or ch == "'")


def _last_word_dangling(words, before_t):
    """True when the last transcribed word ending at or before `before_t` is in DANGLING."""
    ws = [w for w in words if float(w["end"]) <= before_t + 1e-6]
    if not ws:
        return False
    return _norm_lower(ws[-1]["w"]) in DANGLING


def _first_word_after(words, t):
    """First word whose start is >= t (or overlaps it), or None."""
    for w in words:
        if float(w["start"]) >= t - 1e-6:
            return w
    return None


def _last_word_before(words, t):
    """Last word whose end is <= t + eps, or None."""
    best = None
    for w in words:
        if float(w["end"]) <= t + 1e-6:
            best = w
    return best


def verify(clip, words_window_fn, cfg):
    """Apply the edge rules to `clip = {"start", "end"}`. `words_window_fn(lo, hi)` returns
    Word dicts (absolute source seconds) for [lo, hi]. cfg needs min_dur_s and max_dur_s.

    Returns {"start", "end", "action": {"start": ..., "end": ...}}.
    Each action is one of:
      "noop": nothing to do or nothing changed
      "extend": moved outward to absorb a tail / preceding sentence (fits duration band)
      "pull_back": moved inward to the last clean sentence boundary that keeps duration band
      "blocked": would have moved but the duration band prevented it (kept in place, flagged)
    """
    min_dur = float(cfg.get("min_dur_s", 15))
    max_dur = float(cfg.get("max_dur_s", 75))
    s0, s1 = float(clip["start"]), float(clip["end"])
    new_s, new_e = s0, s1
    action = {"start": "noop", "end": "noop"}

    # ---- end side ----
    tail = words_window_fn(max(0.0, s1 - END_TAIL_S), s1)
    after = words_window_fn(s1, s1 + END_LOOK_S)
    tail_word = _last_word_before(tail, s1)
    tail_text = tail_word["w"] if tail_word else ""
    tail_dangles = _norm_lower(tail_text) in DANGLING if tail_text else True
    tail_ended = bool(tail_text) and _clean_stop(tail_text) and not tail_dangles
    reaction = _short_reaction(after, s1) if after else None
    if reaction and cfg.get("drop_wrapup") and boundaries.wrapup_sentence(
            " ".join(str(w["w"]) for w in after[:reaction[0] + 1]), cfg.get("wrapup_extra") or ()):
        reaction = None             # "Anyway." after the cut is no reaction to take in
    span = cfg.get("span_s")        # the clip's topic; no move takes it out of it
    hi_t = float(span[1]) if span else float("inf")
    lo_t = float(span[0]) if span else float("-inf")

    if not tail_ended or reaction:
        # try to extend to the end of the reaction sentence (or end of the tail's own thought
        # if the reaction is what's live). We use end_cut with the new tail words to find the
        # quietest gap; without audio access here we just use the word's end + a small pad.
        target_end = None
        if reaction:
            _, r_end = reaction
            target_end = float(r_end) + float(cfg.get("tail_pad_s", 0.35))
        else:
            # no reaction: extend to the next sentence-ending word inside `after`
            for w in after:
                if _clean_stop(w["w"]):
                    target_end = float(w["end"]) + float(cfg.get("tail_pad_s", 0.35))
                    break
        if target_end is not None and target_end - new_s <= max_dur and target_end <= hi_t:
            new_e = target_end
            action["end"] = "extend"
        elif not tail_ended:
            # pull back to the last clean sentence end inside the clip
            back = _last_clean_end(tail, s1)
            if back is not None:
                _, b_end = back
                b_end = float(b_end) + float(cfg.get("tail_pad_s", 0.35))
                if b_end - new_s >= min_dur and b_end < s1:
                    new_e = b_end
                    action["end"] = "pull_back"
                else:
                    action["end"] = "blocked"
            else:
                action["end"] = "blocked"

    # ---- start side (mirror) ----
    before = words_window_fn(max(0.0, s0 - START_LOOK_S), s0)
    head = words_window_fn(s0, s0 + START_HEAD_S)
    first_in = head[0] if head else None
    last_before = _last_word_before(before, s0)
    if first_in and last_before:
        gap = float(first_in["start"]) - float(last_before["end"])
        prev_ends = _ends_sentence(last_before["w"])
        if gap < LEAD_GAP_S and not prev_ends:
            # find the sentence-starting word: the first word after the last preceding sentence
            # break in `before`
            sent_start = None
            for i in range(len(before) - 1, -1, -1):
                if _ends_sentence(before[i]["w"]):
                    if i + 1 < len(before):
                        sent_start = before[i + 1]
                    break
            if sent_start is None and before:
                sent_start = before[0]
            if sent_start is not None:
                # start moves to LEAD_PAD before the sentence's first word
                target_start = float(sent_start["start"]) - float(cfg.get("lead_pad_s", 0.15))
                target_start = max(0.0, target_start)
                if new_e - target_start <= max_dur and target_start >= lo_t:
                    new_s = target_start
                    action["start"] = "extend"
                else:
                    action["start"] = "blocked"

    return {"start": new_s if new_s == s0 else round(new_s, 3),
            "end": new_e if new_e == s1 else round(new_e, 3),
            "action": action}


def make_words_fn(media, model):
    """A words_window callable that uses transcript.words_window with `model`. Kept out of
    `verify` so tests can substitute a stub."""
    def fn(lo, hi):
        try:
            return transcript.words_window(media, lo, hi, model=model)
        except Exception:
            return []
    return fn
