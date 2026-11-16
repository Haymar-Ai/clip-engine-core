#!/usr/bin/env python3
"""Slur censoring for burned captions and audio (repair lane, code `slur`).

10-20 clip 125 (Full Send) shipped uncensored slurs. A slur in a clip is bleeped: the word's
audio is muted (recompose --mute) and its caption shows the first letter and asterisks. The
list is deliberately short and exact: hate slurs only, not profanity (the niches' clips swear;
the eye-check rejects slurs). Matching is on the whole normalised token (case, punctuation and
a plural / -ing / -ed ending stripped), so "Snigger" never matches.
"""
from __future__ import annotations

import re

# stems, lower case; a token matches its stem plus an optional s / es / z / a / ed / ing / ers
_STEMS = (
    "nigger", "nigga", "niggah", "nigguh",
    "faggot", "fag", "faggit", "fagot", "dyke", "tranny", "trannie",
    "retard", "retarded", "spic", "spick", "wetback", "beaner",
    "chink", "gook", "kike", "raghead", "towelhead", "sandnigger", "coon", "jigaboo",
    "porchmonkey",
)
_SUFFIX = r"(?:s|es|z|a|ed|ing|ers?)?"
_RX = re.compile(r"^(?:" + "|".join(sorted(map(re.escape, _STEMS), key=len, reverse=True))
                 + r")" + _SUFFIX + r"$")
MUTE_PAD_S = 0.04


def _core(token):
    return re.sub(r"[^a-z]", "", str(token).lower())


def is_slur(token):
    core = _core(token)
    return bool(core) and bool(_RX.match(core))


def mask(token):
    """'Word,' -> 'W***,': first letter kept, letters masked, punctuation kept."""
    out, first = [], True
    for ch in str(token):
        if ch.isalpha():
            out.append(ch if first else "*")
            first = False
        else:
            out.append(ch)
    return "".join(out)


def censor(words, start_s=0.0, end_s=None):
    """(words with slurs masked, [(a, b)] mute spans clip-relative to start_s, [found]). Only the
    words inside [start_s, end_s] count (the clip); never mutates its input."""
    out, spans, found = [], [], []
    for w in words or []:
        w = dict(w)
        s, e = float(w["start"]), float(w["end"])
        inside = e > start_s and (end_s is None or s < end_s)
        if inside and is_slur(w.get("w", "")):
            found.append(str(w["w"]))
            w["w"] = mask(w["w"])
            spans.append((round(max(0.0, s - start_s - MUTE_PAD_S), 3),
                          round(e - start_s + MUTE_PAD_S, 3)))
        out.append(w)
    return out, spans, found
