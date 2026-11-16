"""whisper hallucination guards, offline.

transcript.segment_ok drops a segment when whisper heard no speech and was unsure of the text
(no_speech_prob > 0.6 with avg_logprob < -0.8), when its text compresses too well
(compression_ratio > 2.6), or when it loops (a word 3-gram 3+ times, or under 30 % unique tokens
on 8+ tokens). A missing field never drops a segment. The four readers that see segments apply
it: transcript._cli_words, transcript._fw_words, second_asr.whisper_cli_words and
recompose._transcribe_faster_whisper; the faster-whisper paths also run with
condition_on_previous_text=False. Whisper and faster-whisper are faked: no model, no network.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from clipengine import second_asr
from clipengine import transcript

from clipengine.render import recompose


def _seg(text, no_speech_prob=0.05, avg_logprob=-0.3, compression_ratio=1.4, words=None):
    seg = {"text": text, "no_speech_prob": no_speech_prob, "avg_logprob": avg_logprob,
           "compression_ratio": compression_ratio}
    if words is not None:
        seg["words"] = words
    return seg


NORMAL = " So I walked in and the whole room just went quiet, nobody said a word."


# ---- segment_ok: each rule ----------------------------------------------------------------

def test_a_normal_segment_is_kept():
    assert transcript.segment_ok(_seg(NORMAL))


def test_no_speech_with_low_logprob_is_dropped():
    assert not transcript.segment_ok(_seg(" Thank you.", no_speech_prob=0.85, avg_logprob=-1.1))


def test_no_speech_needs_both_signals():
    # whisper unsure it heard speech but confident of the words, or the reverse: kept
    assert transcript.segment_ok(_seg(NORMAL, no_speech_prob=0.85, avg_logprob=-0.5))
    assert transcript.segment_ok(_seg(NORMAL, no_speech_prob=0.3, avg_logprob=-1.4))
    # the thresholds are strict
    assert transcript.segment_ok(_seg(NORMAL, no_speech_prob=0.6, avg_logprob=-1.4))
    assert transcript.segment_ok(_seg(NORMAL, no_speech_prob=0.9, avg_logprob=-0.8))


def test_high_compression_ratio_is_dropped():
    assert not transcript.segment_ok(_seg(NORMAL, compression_ratio=2.61))
    assert transcript.segment_ok(_seg(NORMAL, compression_ratio=2.6))


def test_a_repeated_3gram_is_dropped():
    looped = " I'm going home. I'm going home. I'm going home and that's it."
    assert not transcript.segment_ok(_seg(looped, avg_logprob=-0.7))
    # twice is speech
    assert transcript.segment_ok(_seg(" I'm going home. I'm going home and that's it."))


def test_a_low_unique_share_is_dropped():
    # 8 tokens, 2 unique (25 %), and no 3-gram three times
    assert not transcript.segment_ok(_seg(" yeah okay okay yeah yeah okay okay yeah", avg_logprob=-0.7))
    # 7 tokens of the same: under the 8-token floor, and (go, go, go) occurs twice
    # non-overlapping, so a chant is kept
    assert transcript.segment_ok(_seg(" go go go go go go go"))
    # 9 of the same: both rules
    assert not transcript.segment_ok(_seg(" go go go go go go go go go", avg_logprob=-0.7))


def test_a_confident_loop_is_real_speech_and_kept():
    # 11-11 sample: both loop drops were real chants that base.en also heard
    whoa = " Whoa, whoa, whoa, whoa, whoa, whoa, whoa, whoa, whoa, whoa."
    assert transcript.segment_ok(_seg(whoa, avg_logprob=-0.3, no_speech_prob=0.05))
    assert not transcript.segment_ok(_seg(whoa, avg_logprob=-0.6))
    assert not transcript.segment_ok(_seg(whoa, no_speech_prob=0.4))
    assert transcript.segment_ok({"text": whoa})                                  # no fields


def test_loop_tokens_ignore_case_and_punctuation():
    assert not transcript.segment_ok(_seg(" Thank you, for watching! thank YOU for watching. "
                                          "Thank you for watching...", avg_logprob=-0.7))


def test_a_segment_without_the_fields_is_kept():
    assert transcript.segment_ok({})
    assert transcript.segment_ok({"words": [{"word": " hi", "start": 0.0, "end": 0.3}]})
    assert transcript.segment_ok({"text": NORMAL, "no_speech_prob": 0.95})       # no avg_logprob
    assert transcript.segment_ok({"text": NORMAL, "avg_logprob": -2.0})          # no no_speech
    assert transcript.segment_ok({"text": NORMAL, "compression_ratio": None})
    assert transcript.segment_ok({"text": NORMAL, "compression_ratio": "n/a"})
    assert transcript.segment_ok(types.SimpleNamespace(words=[]))                 # fw, no fields


def test_a_faster_whisper_segment_object_is_read_by_attribute():
    W = types.SimpleNamespace
    assert transcript.segment_ok(W(text=NORMAL, no_speech_prob=0.1, avg_logprob=-0.2,
                                   compression_ratio=1.3, words=[]))
    assert not transcript.segment_ok(W(text=NORMAL, no_speech_prob=0.1, avg_logprob=-0.2,
                                       compression_ratio=3.4, words=[]))


# ---- the readers ----------------------------------------------------------------------------

def _w(word, start, end):
    return {"word": word, "start": start, "end": end, "probability": 0.9}


SEGMENTS = [
    _seg(" Hello there.", words=[_w(" Hello", 0.0, 0.4), _w(" there.", 0.4, 0.8)]),
    _seg(" Thank you.", no_speech_prob=0.9, avg_logprob=-1.2,
         words=[_w(" Thank", 1.0, 1.2), _w(" you.", 1.2, 1.4)]),
    _seg(" la la la la", compression_ratio=3.1,
         words=[_w(" la", 2.0, 2.1), _w(" la", 2.1, 2.2), _w(" la", 2.2, 2.3), _w(" la", 2.3, 2.4)]),
    {"words": [_w(" Bye.", 3.0, 3.3)]},                         # no segment fields: kept
]
KEPT = [("Hello", 0.0, 0.4), ("there.", 0.4, 0.8), ("Bye.", 3.0, 3.3)]


def _fake_cli_json(cmd, **kw):
    out = Path(cmd[cmd.index("--output_dir") + 1])
    (out / (Path(cmd[1]).stem + ".json")).write_text(json.dumps({"segments": SEGMENTS}))
    return subprocess.CompletedProcess(cmd, 0)


def test_cli_words_drops_hallucinated_segments(monkeypatch):
    monkeypatch.setattr(transcript, "WORKER", None)
    monkeypatch.setattr(transcript.subprocess, "run", _fake_cli_json)
    words = transcript._cli_words("/x/a.wav", "base.en")
    assert [(w["w"], w["start"], w["end"]) for w in words] == KEPT


def test_second_asr_drops_hallucinated_segments(monkeypatch, tmp_path):
    monkeypatch.setattr(transcript, "WORKER", None)
    monkeypatch.setattr(subprocess, "run", _fake_cli_json)
    words = second_asr.whisper_cli_words(tmp_path / "clip.wav", "small", threads=2)
    assert [(w["w"], w["start"], w["end"]) for w in words] == KEPT


def _fw_segments():
    W = types.SimpleNamespace
    return [W(text=s.get("text"), no_speech_prob=s.get("no_speech_prob"),
              avg_logprob=s.get("avg_logprob"), compression_ratio=s.get("compression_ratio"),
              words=[W(word=x["word"], start=x["start"], end=x["end"]) for x in s["words"]])
            for s in SEGMENTS]


class _FakeModel:
    calls = []

    def __init__(self, name, **kw):
        pass

    def transcribe(self, path, **kw):
        _FakeModel.calls.append(kw)
        return iter(_fw_segments()), None


@pytest.fixture
def fake_fw(monkeypatch):
    _FakeModel.calls = []
    monkeypatch.setitem(sys.modules, "faster_whisper",
                        types.SimpleNamespace(WhisperModel=_FakeModel))
    return _FakeModel


def test_fw_words_drops_hallucinated_segments_and_does_not_condition(fake_fw, monkeypatch):
    monkeypatch.setattr(transcript, "_MODELS", {})
    words = transcript._fw_words("/x/a.wav", "base.en")
    assert [(w["w"], w["start"], w["end"]) for w in words] == KEPT
    assert fake_fw.calls[-1]["condition_on_previous_text"] is False


def test_recompose_faster_whisper_drops_hallucinated_segments_and_does_not_condition(fake_fw):
    words = recompose._transcribe_faster_whisper("/x/window.wav", "small.en")
    assert [(w["w"], w["start"], w["end"]) for w in words] == KEPT
    assert fake_fw.calls[-1]["condition_on_previous_text"] is False
