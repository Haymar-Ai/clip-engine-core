"""Edge verifier tests (clipengine/edge_verify.py) — pure, no ffmpeg / no whisper."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from clipengine import edge_verify as EV


def W(w, s, e):
    return {"w": w, "start": float(s), "end": float(e)}


def stub(mapping):
    """mapping: (lo_round, hi_round) -> list[Word]. Rounds to 2 dp for lookup convenience.
    Test writers usually just build one flat words list and slice it."""
    def fn(lo, hi):
        eps = 1e-6
        return [w for w in mapping if float(w["start"]) >= lo - eps and float(w["end"]) <= hi + eps]
    return fn


def test_noop_when_tail_ends_clean_and_no_reaction():
    tail = [W("Yes.", 24.0, 24.4)]                                        # sentence end
    after = [W("Basically", 25.5, 26.0)]                                  # no reaction in 0.5s
    words = tail + after
    v = EV.verify({"start": 0.0, "end": 25.0}, stub(words),
                  {"min_dur_s": 5.0, "max_dur_s": 60.0})
    assert v["end"] == 25.0 and v["action"]["end"] == "noop"


def test_extends_end_on_mid_thought_out_after_sentence():
    # tail is dangling ("some..."); OUT-AFTER begins fast with a short complete sentence
    tail = [W("some", 24.6, 24.9)]                                        # incomplete
    after = [W("Well", 25.05, 25.15), W("done.", 25.15, 25.45)]           # 4-word ready sentence
    v = EV.verify({"start": 0.0, "end": 25.0}, stub(tail + after),
                  {"min_dur_s": 5.0, "max_dur_s": 60.0, "tail_pad_s": 0.35})
    assert v["action"]["end"] == "extend"
    assert v["end"] > 25.0
    assert v["end"] < 25.0 + (25.45 - 25.0) + 0.4


def test_extends_absorbs_reaction_tail_after_clean_end():
    # tail ends with a full sentence, but a short reaction tail ("Nice hand, boom.") fires <0.5s
    tail = [W("go.", 24.6, 24.9)]                                         # clean
    after = [W("Nice", 25.1, 25.3), W("hand,", 25.3, 25.6), W("boom.", 25.6, 25.9)]
    v = EV.verify({"start": 0.0, "end": 25.0}, stub(tail + after),
                  {"min_dur_s": 5.0, "max_dur_s": 60.0, "tail_pad_s": 0.35})
    assert v["action"]["end"] == "extend"
    assert v["end"] >= 25.9


def test_pulls_back_to_clean_when_extension_would_exceed_max_dur():
    # tail has a clean sentence a few words back, current end is mid-thought,
    # extending would exceed max_dur_s, so pull back.
    tail = [W("okay.", 22.0, 22.4), W("but", 22.7, 22.9), W("some", 24.6, 24.9)]
    after = [W("More", 30.1, 30.4), W("words.", 30.4, 30.7)]              # far future
    # max_dur small enough that extending is blocked but pulling back to 22.4 still passes min
    v = EV.verify({"start": 0.0, "end": 25.0}, stub(tail + after),
                  {"min_dur_s": 15.0, "max_dur_s": 25.5, "tail_pad_s": 0.35})
    # okay. -> pull back
    assert v["action"]["end"] == "pull_back"
    assert 22.0 < v["end"] < 25.0


def test_end_blocked_when_no_clean_alternative():
    tail = [W("um", 24.0, 24.5), W("and", 24.5, 24.9)]                    # dangles all the way
    after = []                                                             # nothing to extend into
    v = EV.verify({"start": 0.0, "end": 25.0}, stub(tail + after),
                  {"min_dur_s": 15.0, "max_dur_s": 26.0})
    assert v["action"]["end"] == "blocked"
    assert v["end"] == 25.0                                                # unchanged


def test_start_moves_back_when_first_in_word_runs_into_prev():
    before = [W("The", 9.0, 9.3), W("thing", 9.3, 9.7), W("is,", 9.7, 9.9)]  # unfinished sentence
    head = [W("some", 9.95, 10.2), W("guys", 10.2, 10.5), W("bluff.", 10.5, 10.9)]
    # last word before "is," is not a sentence end; gap 0.05 < 0.3
    v = EV.verify({"start": 10.0, "end": 30.0}, stub(before + head),
                  {"min_dur_s": 5.0, "max_dur_s": 60.0, "lead_pad_s": 0.15})
    assert v["action"]["start"] == "extend"
    assert v["start"] < 10.0
    # pulled back to before "The" (start of the sentence) minus lead pad
    assert v["start"] <= 9.0


def test_start_noop_when_previous_word_ends_sentence():
    before = [W("Done.", 9.5, 9.9)]
    head = [W("New", 10.05, 10.3)]                                         # gap < 0.3 but prev is clean
    v = EV.verify({"start": 10.0, "end": 30.0}, stub(before + head),
                  {"min_dur_s": 5.0, "max_dur_s": 60.0})
    assert v["action"]["start"] == "noop"
    assert v["start"] == 10.0


def test_at_most_one_action_per_side():
    # even if end and start both need moving, each side moves at most once
    before = [W("The", 8.0, 8.3), W("thing", 8.3, 8.6), W("is,", 8.6, 8.85)]
    head = [W("some", 9.05, 9.3)]                                          # first in-clip word
    tail = [W("um", 24.5, 24.8)]                                           # incomplete
    after = [W("Good.", 25.1, 25.4)]                                       # short reaction
    v = EV.verify({"start": 9.0, "end": 25.0}, stub(before + head + tail + after),
                  {"min_dur_s": 5.0, "max_dur_s": 60.0})
    assert v["action"]["start"] == "extend" and v["action"]["end"] == "extend"
    assert v["start"] < 9.0 and v["end"] > 25.0


def test_end_never_extends_into_host_question():
    # DSH #230 clip 5: tail ends on a host's filler, next sentence is the host's question
    tail = [W("eight", 295.9, 296.1), W("years.", 296.1, 296.5), W("Nice", 296.6, 296.9)]
    after = [W("Walk", 297.1, 297.3), W("me", 297.3, 297.4), W("through", 297.4, 297.6),
             W("first", 299.9, 300.2), W("year?", 300.2, 300.7)]
    v = EV.verify({"start": 238.7, "end": 296.97}, stub(tail + after),
                  {"min_dur_s": 15.0, "max_dur_s": 75.0, "tail_pad_s": 0.35})
    assert v["action"]["end"] == "pull_back"
    assert 296.5 <= v["end"] < 296.97


def test_pull_back_skips_question_to_last_statement():
    # DSH #230 clip 1: crosstalk "That's from a review? Yeah," -- a question is not a clean end
    tail = [W("week.", 219.5, 219.86), W("That's", 220.0, 220.2), W("review?", 220.4, 220.74),
            W("Yeah,", 220.8, 221.0)]
    v = EV.verify({"start": 181.4, "end": 221.1}, stub(tail),
                  {"min_dur_s": 15.0, "max_dur_s": 39.0, "tail_pad_s": 0.35})
    assert v["action"]["end"] == "pull_back"
    assert abs(v["end"] - (219.86 + 0.35)) < 1e-6


def test_tail_ending_on_question_extends_to_answer():
    tail = [W("profitable", 299.5, 299.9), W("year?", 300.2, 300.7)]
    after = [W("No,", 301.0, 301.2), W("not", 301.2, 301.4), W("at", 301.4, 301.5), W("all.", 301.5, 301.8)]
    v = EV.verify({"start": 250.0, "end": 300.8}, stub(tail + after),
                  {"min_dur_s": 15.0, "max_dur_s": 75.0, "tail_pad_s": 0.35})
    assert v["action"]["end"] == "extend" and v["end"] >= 301.8
