"""the caption crosstalk filter. 10-20 clips 109/110/125/135 burned captions
nobody said; a phrase is now burned only when an independent transcript of the same audio (and
whisper's own word probability, when known) confirms it. Offline."""
from __future__ import annotations

import importlib
import json
import pathlib
import sys

from clipengine import crosstalk
from clipengine import quality_gate

ROOT = pathlib.Path(__file__).resolve().parents[1]


def W(text, t0=0.0, step=0.3, **extra):
    return [{"w": w, "start": round(t0 + i * step, 3), "end": round(t0 + i * step + 0.25, 3),
             **extra} for i, w in enumerate(text.split())]


# precision words: a clean line, a crosstalk line (two voices merged), a clean line
SIDECAR = (W("He told me the price was fine.", 0.0)
           + W("Wait so the bring car sold bump yeah it on", 3.0)
           + W("Then we drove home.", 7.0))
# base pass of the same audio: hears the clean lines, half-hears the crosstalk differently
REF = (W("He told me the price was fine.", 0.0) + W("wait no the car was fine yeah", 3.0)
       + W("Then we drove home.", 7.0))


def test_phrases_split_at_pauses_and_sentence_ends():
    ph = crosstalk.phrases(SIDECAR)
    texts = [" ".join(w["w"] for w in SIDECAR[a:b + 1]) for a, b in ph]
    assert texts == ["He told me the price was fine.", "Wait so the bring car sold bump yeah it on",
                     "Then we drove home."]


def test_unconfirmed_phrase_is_flagged_not_dropped():
    out, rep = crosstalk.flag_crosstalk(SIDECAR, REF)
    assert len(out) == len(SIDECAR)                                   # the cut logic keeps them
    flagged = [w["w"] for w in out if w.get("cap") is False]
    assert flagged == "Wait so the bring car sold bump yeah it on".split()
    assert rep["dropped"][0]["why"].startswith("ref agrees on")
    assert [w["w"] for w in crosstalk.captioned(out)][:3] == ["He", "told", "me"]
    assert all("cap" not in w for w in SIDECAR)                       # input untouched


def test_low_word_probability_flags_a_phrase_without_a_ref():
    words = W("this is fine", 0.0, p=0.9) + W("blah murmur wobble mumble", 2.0, p=0.2)
    out, rep = crosstalk.flag_crosstalk(words)
    assert [w["w"] for w in crosstalk.captioned(out)] == ["this", "is", "fine"]
    assert rep["judged_by"] == ["prob"]
    out, rep = crosstalk.flag_crosstalk(W("no probabilities here at all"))
    assert rep["judged_by"] == [] and crosstalk.captioned(out) == out


def test_short_phrases_are_never_judged():
    out, _ = crosstalk.flag_crosstalk(W("yeah right", 0.0), ref_words=[])
    assert crosstalk.captioned(out) == out


def test_single_speaker_drops_an_interjection_inside_the_main_run():
    words = W("so I went down there and", 0.0) + W("what", 1.8) + W("they had it", 2.1)
    labels = [0] * 6 + [1] + [0] * 3
    out, n = crosstalk.single_speaker(words, labels)
    assert n == 1 and out[6]["cap"] is False and out[6]["w"] == "what"
    out, n = crosstalk.single_speaker(words, [0] * 6 + [1] + [None] * 3)
    assert n == 0                                       # the other side is unknown: kept


def test_render_and_gate_skip_flagged_words_alike():
    from clipengine.render import captions
    out, _ = crosstalk.flag_crosstalk(SIDECAR, REF)
    burned = captions.clean_words(out, 10.0)
    seen = quality_gate.clip_words(out, 0.0, 10.0)
    assert [w["w"] for w in burned] == [w["w"] for w in seen] == \
        [w["w"] for w in crosstalk.captioned(out)]


def test_persist_flags_crosstalk_against_the_base_pass(tmp_path, monkeypatch):
    from clipengine import runstate
    runstate.reset()
    from clipengine import engine
    monkeypatch.setattr(engine, "SCRATCH", tmp_path)
    runstate.init()
    runstate.add_source("X_w00")
    c = {"start": 0.0, "end": 8.5, "words": SIDECAR, "mean_z": 0.1, "boundary_type": "sentence",
         "score": 0.5, "refined": True}
    cid = engine._persist("X_w00", "auto", c, "general", 0.8, ref_words=REF)
    clip = runstate.clip(cid)
    side = json.loads(pathlib.Path(clip["words_path"]).read_text())
    assert len(side) == len(SIDECAR) and sum(w.get("cap") is False for w in side) == 10
    assert json.loads(clip["features_json"])["crosstalk_dropped"][0]["start"] == 3.0
