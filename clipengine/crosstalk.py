#!/usr/bin/env python3
"""Caption crosstalk filter.

10-20 eye-check: clips 109, 110, 125 and 135 shipped captions that were not what anyone said:
two people talking over each other (125, 135), event audio under a stream (109), a garbled line
(110). One whisper pass turns overlapping speech into a confident-looking line of nonsense.

A caption phrase is kept only when it is confirmed:
  - by an independent transcript of the same audio (`ref_words`: select's base.en pass for the
    small.en precision words the sidecar holds; in the repair lane the precise small.en words for
    a large-v3-turbo re-transcription): >= MIN_AGREE of its tokens are heard there too;
  - and, when the words carry whisper's own word probability (`p`), its mean probability is
    >= MIN_PROB.
A phrase that fails is crosstalk: its words stay in the sidecar (the cut logic and the gate's
sentence checks still see them) but get `"cap": false`, so the render does not burn them and the
gate's caption check compares exactly what is burned (quality_gate.clip_words skips them too).
The gate is unchanged: a clip whose crosstalk leaves an uncaptioned stretch still fails it.

`single_speaker` (repair lane): with per-word voice labels (render/diarize), a minority voice's
words inside the dominant speaker's run are crosstalk too.

`confirm_crosstalk`: before the render, a flagged phrase the gate's
second ASR also hears is burned again ("crosstalk": true stays for the cut logic).
"""
from __future__ import annotations

PHRASE_GAP_S = 0.3           # a pause this long ends a phrase
PHRASE_MAX_WORDS = 12
MIN_TOKENS = 3               # shorter phrases are not judged (a "yeah" the ref missed is fine)
MIN_AGREE = 0.5              # share of a phrase's tokens the independent transcript also heard
MIN_PROB = 0.45              # mean whisper word probability of a phrase
REF_PAD_S = 0.6              # the ref words looked at around a phrase
INTERRUPT_S = 0.6            # single_speaker: dominant-voice words this close on both sides


def _tokens(words):
    from clipengine import quality_gate
    return quality_gate.norm_tokens(words)


def phrases(words, gap_s=PHRASE_GAP_S, max_words=PHRASE_MAX_WORDS):
    """[(i0, i1)] index ranges of `words` split at pauses >= gap_s, sentence ends and max_words."""
    from clipengine.transcript import _ends_sentence
    out, a = [], 0
    for k, w in enumerate(words):
        last = k == len(words) - 1
        if last or _ends_sentence(w["w"]) or k - a + 1 >= max_words or \
                float(words[k + 1]["start"]) - float(w["end"]) >= gap_s:
            out.append((a, k))
            a = k + 1
    return out


def phrase_agreement(phrase_words, ref_words, pad_s=REF_PAD_S):
    """Share of the phrase's tokens found (in order, fuzzy) among the ref words around it."""
    from clipengine import quality_gate
    if not phrase_words:
        return 1.0
    t0 = float(phrase_words[0]["start"]) - pad_s
    t1 = float(phrase_words[-1]["end"]) + pad_s
    ref = [w for w in ref_words or [] if float(w["end"]) >= t0 and float(w["start"]) <= t1]
    a, b = _tokens(phrase_words), _tokens(ref)
    if not a:
        return 1.0
    return sum(quality_gate.align_tokens(a, b)) / len(a)


def flag_crosstalk(words, ref_words=None, *, min_agree=MIN_AGREE, min_prob=MIN_PROB,
                   min_tokens=MIN_TOKENS):
    """(words with "cap": False on crosstalk phrases, report). `ref_words` None and no word
    probabilities: nothing to judge against, every word kept. Never mutates its input."""
    out = [dict(w) for w in words or []]
    rep = {"phrases": 0, "dropped": [], "judged_by": []}
    if ref_words is not None:
        rep["judged_by"].append("ref")
    if any("p" in w for w in out):
        rep["judged_by"].append("prob")
    if not rep["judged_by"]:
        return out, rep
    for i0, i1 in phrases(out):
        ph = out[i0:i1 + 1]
        rep["phrases"] += 1
        if len(_tokens(ph)) < min_tokens:
            continue
        why = []
        if ref_words is not None:
            agree = phrase_agreement(ph, ref_words)
            if agree < min_agree:
                why.append(f"ref agrees on {agree:.0%}")
        probs = [float(w["p"]) for w in ph if w.get("p") is not None]
        if probs and sum(probs) / len(probs) < min_prob:
            why.append(f"mean word probability {sum(probs) / len(probs):.2f}")
        if why:
            for w in ph:
                w["cap"] = False
            rep["dropped"].append({"start": round(float(ph[0]["start"]), 2),
                                   "end": round(float(ph[-1]["end"]), 2),
                                   "text": " ".join(str(w["w"]) for w in ph),
                                   "why": "; ".join(why)})
    return out, rep


def single_speaker(words, labels, *, interrupt_s=INTERRUPT_S):
    """Mark a minority voice's words that land inside the dominant voice's run (both neighbours
    within interrupt_s are the dominant voice) as crosstalk: "cap": False. `labels` is one voice
    label per word (None = unknown, kept). Returns (words, n_dropped)."""
    out = [dict(w) for w in words or []]
    known = [lb for lb in labels or [] if lb is not None]
    if len(out) != len(labels or []) or not known:
        return out, 0
    dom = max(set(known), key=known.count)
    n = 0
    for i, (w, lb) in enumerate(zip(out, labels)):
        if lb is None or lb == dom:
            continue
        prev = next((j for j in range(i - 1, -1, -1) if labels[j] is not None), None)
        nxt = next((j for j in range(i + 1, len(out)) if labels[j] is not None), None)
        if prev is None or nxt is None or labels[prev] != dom or labels[nxt] != dom:
            continue
        if float(w["start"]) - float(out[prev]["end"]) <= interrupt_s and \
                float(out[nxt]["start"]) - float(w["end"]) <= interrupt_s:
            w["cap"] = False
            n += 1
    return out, n


def captioned(words):
    """The words the render burns: every word not flagged `"cap": false`."""
    return [w for w in words or [] if w.get("cap", True) is not False]


# 39 of the 90 weekly gate caption rejects were "uncaptioned stretch",
# and in 8 of the 35 that still had their sidecar the stretch was crosstalk this filter left
# unburned (3097: 24 of 91 words) while the gate's second ASR, and the viewer, heard it. A
# flagged phrase the second ASR hears where the burned captions carry nothing is speech, not
# garble: it is burned again, and keeps "crosstalk": true for the cut logic. Only the heard words
# the burned captions do not already carry count: on the 11-11 replay of production's crosstalk
# clips a flagged repeat of a line the captions burn ('They all say a Supra.', 3237) matched the
# same heard words and cost the captions precision; single flagged words ('you', 'need') matched
# any word nearby.
def _uncarried(heard, burned):
    """The `heard` words none of whose tokens the `burned` words carry (in order, fuzzy)."""
    from clipengine import quality_gate
    hs = sorted(heard or [], key=lambda w: float(w["start"]))
    toks, owner = [], []
    for k, w in enumerate(hs):
        for t in quality_gate.norm_tokens([w]):
            toks.append(t)
            owner.append(k)
    matched = quality_gate.align_tokens(toks, _tokens(sorted(burned, key=lambda w: float(w["start"]))))
    carried = {owner[i] for i, m in enumerate(matched) if m}
    return [w for k, w in enumerate(hs) if k not in carried]


def confirm_crosstalk(words, heard, *, min_agree=MIN_AGREE, min_tokens=MIN_TOKENS,
                      gap_s=PHRASE_GAP_S):
    """(words, report): every run of at least `min_tokens` crosstalk-flagged words ("cap": False;
    a run ends at a gap_s pause or a burned word) whose tokens the `heard` transcript (same time
    base: the second ASR) confirms among the words the burned captions do not carry
    (phrase_agreement >= min_agree) burned again: "cap" removed, "crosstalk": True. Words the
    caption repair unburned as heard by nobody ("fix": "unheard") stay unburned. Never mutates
    its input. report = {"burned": [{start, end, text, agree}], "kept": [...]}."""
    out = [dict(w) for w in words or []]
    rep = {"burned": [], "kept": []}
    if not heard:
        return out, rep
    order = sorted(range(len(out)), key=lambda k: float(out[k]["start"]))
    runs, cur = [], []
    for k in order:
        w = out[k]
        flagged = w.get("cap", True) is False and w.get("fix") != "unheard"
        if flagged and cur and float(w["start"]) - float(out[cur[-1]]["end"]) < gap_s:
            cur.append(k)
            continue
        if cur:
            runs.append(cur)
        cur = [k] if flagged else []
    if cur:
        runs.append(cur)
    runs = [r for r in runs if len(_tokens([out[k] for k in r])) >= min_tokens]
    if not runs:
        return out, rep
    free = _uncarried(heard, [w for w in out if w.get("cap", True) is not False])
    for run in runs:
        ph = [out[k] for k in run]
        agree = phrase_agreement(ph, free)
        rec = {"start": round(float(ph[0]["start"]), 2), "end": round(float(ph[-1]["end"]), 2),
               "text": " ".join(str(w["w"]) for w in ph), "agree": round(agree, 2)}
        if agree >= min_agree:
            for w in ph:
                w.pop("cap", None)
                w["crosstalk"] = True
            rep["burned"].append(rec)
        else:
            rep["kept"].append(rec)
    return out, rep
