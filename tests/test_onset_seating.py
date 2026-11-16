"""Seat the ending's onset on the window audio, not on the second ASR's stretched word times.

The gate's ending check (quality_gate.ending_verdict) reads the second ASR's (small) last word
before the cut as the clip's last, and checks only the 100 ms after its claimed onset for silence.
small times the word after a pause from the end of the word before it, so that probe can read the
previous word's tail and fail a clip as ending on 'and' when the audio only voices the 'and' after
the cut.
- With the seat, a word whose claimed span holds a pause is seated where the voice comes back
  (quality_gate._seat_on_pause), when that is in the final fade or past the cut and the burned
  captions end on the word before it.
- Kill switch: quality_gate.ending_seat.enabled.

Every case is synthetic. `period` / `no_period` are test_filler_endings' second-ASR cases: their
pass, the burned captions inside the clip and the gate's cut_energy frames on their window audio.
The others hold the last words of the second ASR and of the captions around the cut, and
cut_energy-shaped frames (_frames: 20 ms, dB vs the clip's median, -1 s .. +2 s around the cut).
"""
from __future__ import annotations

import json

import pytest

from clipengine import quality_gate
import test_filler_endings as tf
from test_filler_endings import db  # noqa: F401 - the render tests' runstate / engine fixture

_core = quality_gate._normalize_core


def _frames(*runs):
    """cut_energy-shaped frames from -1 s: [(seconds, dB), ...] runs."""
    db = []
    for secs, level in runs:
        db += [float(level)] * int(round(secs / 0.02))
    return [round(-1.0 + 0.02 * k, 2) for k in range(len(db))], db


def _case(asr, cap, energy, gate_error, dur=20.0):
    """An ending on a `dur` s clip: `asr` / `cap` = the second ASR's / the burned captions' last
    words as (word, start, end) around the cut (cut-relative s), `energy` = _frames runs."""
    def rel(ws):
        return [{"w": w, "start": round(dur + a, 3), "end": round(dur + b, 3)} for w, a, b in ws]
    times, levels = _frames(*energy)
    return {"start_s": 0.0, "end_s": dur, "asr2": {"words": rel(asr), "tail_s": 2.0},
            "caption_words": rel(cap), "energy": {"times": times, "db": levels},
            "gate_error": gate_error}


def _from_render_case(c):
    """test_filler_endings' case as the gate's ending reads it on its cut."""
    s, e = c["start_s"], c["end_s"]
    times, levels = tf._cut_energy(c, s, e)
    return {"start_s": s, "end_s": e, "asr2": c["asr2"],
            "caption_words": quality_gate.clip_words(c["caption_words"], s, e),
            "energy": {"times": times, "db": levels}, "gate_error": c["gate_errors"][0]}


def _filler(word, at):
    return f"ending: ends on a filler/conjunction {word!r} (starts {at:+.2f} s from the cut)"


END = {
    "period": _from_render_case(tf.ASR2["period"]),
    "no_period": _from_render_case(tf.ASR2["no_period"]),
    # 'But' claimed from the end of 'bend.'; quiet from -0.14 s, the voice back at +0.18 s
    "but": _case([("around", -1.10, -0.90), ("the", -0.90, -0.60), ("bend.", -0.60, -0.22),
                  ("But", -0.22, 0.28), ("nobody", 0.28, 0.60), ("stopped", 0.60, 0.95)],
                 [("around", -1.10, -0.92), ("the", -0.90, -0.78), ("bend.", -0.77, -0.21)],
                 [(0.86, 0), (0.32, -50), (1.82, 0)], _filler("But", -0.22)),
    # 'like' claimed from 0.14 s before the cut; quiet from -0.06 s, the voice back at +0.34 s
    "like": _case([("same", -0.95, -0.70), ("old", -0.70, -0.40), ("game", -0.40, -0.14),
                   ("like", -0.14, 0.42), ("I", 0.42, 0.58), ("said", 0.58, 0.90)],
                  [("same", -0.95, -0.72), ("old", -0.70, -0.45), ("game.", -0.43, -0.19)],
                  [(0.94, 0), (0.40, -50), (1.66, 0)], _filler("like", -0.14)),
    # 'but' claimed from 0.54 s before the cut; quiet from -0.30 s, the voice back at +0.02 s
    "late_but": _case([("water", -1.30, -1.10), ("was", -1.10, -0.82), ("cold", -0.82, -0.54),
                       ("but", -0.54, 0.12), ("we", 0.12, 0.30), ("swam", 0.30, 0.62)],
                      [("water", -1.30, -1.12), ("was", -1.10, -0.82), ("cold.", -0.80, -0.44)],
                      [(0.70, 0), (0.32, -50), (1.98, 0)], _filler("but", -0.54)),
    # unpunctuated captions; quiet from -0.04 s, the voice back at +0.26 s
    "so": _case([("kept", -1.30, -1.00), ("going", -1.00, -0.78), ("downhill", -0.78, -0.20),
                 ("So", -0.20, 0.32), ("they", 0.32, 0.50), ("waited", 0.50, 0.90)],
                [("kept", -1.30, -1.02), ("going", -1.02, -0.80), ("downhill", -0.76, -0.18)],
                [(0.96, 0), (0.30, -50), (1.74, 0)], _filler("So", -0.20)),
    # the second ASR hears 'too' between the captions' last word and 'but'
    "other_word": _case([("out", -1.00, -0.80), ("of", -0.80, -0.70), ("them", -0.70, -0.40),
                         ("too", -0.40, -0.22), ("but", -0.22, 0.60), ("the", 0.60, 0.78),
                         ("rest", 0.78, 1.02)],
                        [("out", -1.00, -0.82), ("of", -0.80, -0.70), ("them", -0.70, -0.48)],
                        [(0.88, 0), (0.42, -50), (1.70, 0)], _filler("but", -0.22)),
    # 'Mm' voiced from -0.26 to -0.12 s, the pause only after it
    "inside": _case([("a", -1.00, -0.90), ("close", -0.90, -0.72), ("finish.", -0.72, -0.26),
                     ("Mm", -0.26, -0.12), ("-hmm,", -0.10, 0.16), ("but", 0.16, 0.32),
                     ("then", 0.32, 0.60)],
                    [("a", -1.00, -0.90), ("close", -0.90, -0.70), ("finish.", -0.66, -0.30)],
                    [(0.90, 0), (0.26, -50), (1.84, 0)], _filler("Mm", -0.26)),
    # the audio is voiced all through the cut
    "voiced": _case([("who", -1.26, -1.04), ("was", -1.04, -0.88), ("that?", -0.88, -0.42),
                     ("And", -0.18, 0.20), ("why", 0.20, 0.40), ("now", 0.40, 0.70)],
                    [("who", -1.15, -1.05), ("was", -1.05, -0.87), ("that?", -0.87, -0.61)],
                    [(3.0, 0)], _filler("And", -0.18)),
    # after 'winter' the audio drops to -8 / -13 dB, never to a pause
    "quiet": _case([("ready", -1.20, -0.90), ("for", -0.90, -0.55), ("winter.", -0.55, -0.15),
                    ("And", -0.15, 0.29), ("then", 0.29, 0.59), ("spring", 0.59, 0.90)],
                   [("ready", -1.20, -0.92), ("for", -0.71, -0.53), ("winter", -0.53, -0.07)],
                   [(0.90, 0)] + [(0.02, -13), (0.02, -8)] * 7 + [(0.02, -13), (1.80, 0)],
                   _filler("And", -0.15)),
}


def _energy(c):
    return c["energy"]["times"], c["energy"]["db"]


def _ending(c, seat=True, cap=None, energy=True):
    """The gate's ending on the case's cut, from its own pass, captions and audio frames."""
    return quality_gate.ending_verdict(
        c["asr2"]["words"], c["caption_words"] if cap is None else cap,
        c["end_s"] - c["start_s"], tail_s=c["asr2"]["tail_s"],
        energy=_energy(c) if energy else None, seat=seat)


# ---------------------------------------------------------------- the cases, as the gate read them

@pytest.mark.parametrize("cid", sorted(END))
def test_with_the_seat_off_every_case_fails_on_its_filler(cid):
    c = END[cid]
    assert ["ending: " + e for e in _ending(c, seat=False)["errors"]] == [c["gate_error"]]


# cid: (the word seated, where its voice comes back vs the cut, the gate's ending errors now)
SEATED = {
    "period": ("and", 0.30, []),
    "no_period": ("and", 0.68, ["ends mid-sentence: last word 'listen', next 'and' at +0.68 s"]),
    "but": ("But", 0.18, []),
    "like": ("like", 0.34, []),
    "late_but": ("but", 0.02, []),
    "so": ("So", 0.26, ["ends mid-sentence: last word 'downhill', next 'So' at +0.26 s"]),
}
# cid: why the word stays the clip's last (the gate's verdict without the seat stands)
KEPT = {
    "other_word": "captions end on 'them', the second ASR's word before 'but' is 'too'",
    "inside": "'Mm' ends 0.12 s before the cut: inside the clip",
    "voiced": "the audio runs voiced through the cut: no pause to seat 'And' after",
    "quiet": "the quiet after 'winter' (-8 to -13 dB) is no pause: 'And' may be voiced there",
}


def test_every_case_is_either_seated_or_kept():
    assert set(SEATED) | set(KEPT) == set(END) and not set(SEATED) & set(KEPT)


@pytest.mark.parametrize("cid", sorted(SEATED))
def test_the_word_is_seated_where_its_voice_comes_back_on_the_audio(cid):
    word, onset, errors = SEATED[cid]
    c = END[cid]
    m = _ending(c)
    assert m["errors"] == errors and m["passed"] is (not errors)
    [s] = m["seated"]
    assert s["w"] == word and s["to"] == pytest.approx(onset, abs=1e-6)
    p0, p1 = s["pause"]
    assert p1 == s["to"] and p1 - p0 >= quality_gate.ENDING_PAUSE_MIN_S
    assert s["from"] <= -quality_gate.ENDING_ONSET_TOL_S <= s["to"]     # heard before, now not
    # the audio: quiet from p0 until the voice comes back at p1
    t, d = _energy(c)
    quiet = [x for tt, x in zip(t, d) if p0 <= tt < p1]
    assert sum(quiet) / len(quiet) < quality_gate.ENDING_SILENCE_DB
    back = [x for tt, x in zip(t, d) if p1 - 0.01 <= tt < p1 + 0.03]   # its first two frames
    assert len(back) == 2 and min(back) >= quality_gate.ENDING_VOICE_DB
    # the gate now reads the word as the next one and the word before it as the clip's last
    assert _core(m["next_word"]) == _core(word) and m["next_word_onset_s"] == pytest.approx(onset)
    assert _core(m["last_word"]) == _core(c["caption_words"][-1]["w"])


@pytest.mark.parametrize("cid", sorted(KEPT))
def test_the_gate_keeps_the_word_where_the_audio_or_the_captions_do_not_place_it(cid):
    c = END[cid]
    m = _ending(c)
    assert "seated" not in m
    assert ["ending: " + e for e in m["errors"]] == [c["gate_error"]]


def test_on_selects_own_words_the_seated_ending_passes():
    """no_period's burned words are the caption pass's ('listen', no period). select's precise
    words (the cut placed on them) end 'listen.': had the clip burned those, as a render does when
    the strong pass has no time, the seated ending passes."""
    c, case = END["no_period"], tf.ASR2["no_period"]
    cap = quality_gate.clip_words(case["select_words"], c["start_s"], c["end_s"])
    assert cap[-1]["w"] == "listen."
    m = _ending(c, cap=cap)
    assert m["passed"] is True and m["last_word"] == "listen" and m["seated"][0]["to"] == 0.68
    assert _ending(c, cap=cap, seat=False)["errors"] == [case["gate_errors"][0][len("ending: "):]]


# ---------------------------------------------------------------- the seat's limits

def test_no_seat_when_the_captions_end_on_the_word_itself():
    """The captions burn 'and' as the clip's last word: they place its voice before the cut."""
    c = END["period"]
    cap = c["caption_words"] + [{"w": "and", "start": 22.80, "end": 22.88}]
    m = _ending(c, cap=cap)
    assert "seated" not in m and m["last_word"] == "and"
    assert "captions end on a filler/dangling word 'and'" in m["errors"]


def test_no_seat_without_captions_or_audio_or_with_the_seat_off():
    c = END["period"]
    for m in (_ending(c, cap=[]), _ending(c, energy=False), _ending(c, seat=False)):
        assert "seated" not in m and m["last_word"] == "and" and m["passed"] is False


def test_a_click_inside_a_pause_does_not_end_it_a_word_does():
    one = _frames((0.8, 0), (0.2, -20), (0.02, -5), (0.2, -20), (0.2, 0))
    assert quality_gate._pauses(one, -1.0, 0.42) == [(-0.2, 0.22)]
    two = _frames((0.8, 0), (0.2, -20), (0.04, -5), (0.1, -20), (0.2, 0))
    assert quality_gate._pauses(two, -1.0, 0.34) == [(-0.2, 0.0)]          # 0.1 s after: a stop
    assert quality_gate._pauses(_frames((0.8, 0), (0.12, -30), (0.2, 0)), -1.0, 0.12) == []
    assert quality_gate._pauses(_frames((0.8, 0), (0.4, -12), (0.2, 0)), -1.0, 0.4) == []


def _synthetic(word_end, voice_back, cap_last="mess."):
    """'... a mess. and then' cut at 10 s: small's 'and' claimed from the end of 'mess.' (-0.5 s)
    to `word_end`; the audio voiced to -0.4 s ('mess.'), quiet until `voice_back`."""
    asr = [{"w": "a", "start": 9.0, "end": 9.2}, {"w": "mess.", "start": 9.2, "end": 9.5},
           {"w": "and", "start": 9.5, "end": 10.0 + word_end},
           {"w": "then", "start": 10.0 + word_end, "end": 10.5 + word_end}]
    cap = [{"w": "a", "start": 9.0, "end": 9.2}, {"w": cap_last, "start": 9.2, "end": 9.6}]
    energy = _frames((0.6, 5), (voice_back + 0.4, -25), (1.5, 5))
    return quality_gate.ending_verdict(asr, cap, 10.0, tail_s=2.0, energy=energy)


def test_the_voice_must_come_back_in_the_fade_or_after_the_cut_inside_the_claimed_word():
    assert _synthetic(0.5, 0.3)["seated"][0]["to"] == 0.3 and _synthetic(0.5, 0.3)["passed"]
    assert _synthetic(0.5, -0.12)["passed"]                     # back in the final fade: unheard
    m = _synthetic(0.5, -0.14)                                  # back before the fade: 'and' heard
    assert "seated" not in m and m["last_word"] == "and"
    m = _synthetic(0.2, 0.3)                                    # 'and' claimed to end in the pause
    assert "seated" not in m and m["last_word"] == "and"
    m = _synthetic(0.5, 0.3, cap_last="and")                    # captions end on 'and' itself
    assert "seated" not in m


def test_the_kill_switch():
    assert quality_gate.ending_seat_enabled({}) is True
    assert quality_gate.ending_seat_enabled({"quality_gate": {"ending_seat": {}}}) is True
    assert quality_gate.ending_seat_enabled(
        {"quality_gate": {"ending_seat": {"enabled": False}}}) is False
    assert quality_gate.ending_seat_enabled() is True          # config.json: no block, on


def test_the_cut_check_before_the_render_reads_the_seat_and_its_switch(monkeypatch):
    """engine.gate_cut_errors (the caption step's / re-cut lane's read of the gate's checks)
    seats as the gate does, and follows the switch."""
    from clipengine import second_asr
    from clipengine import engine
    c, case = END["period"], tf.ASR2["period"]
    monkeypatch.setattr(second_asr, "settings", lambda cfg=None: {
        "enabled": True, "model": "small", "tail_s": 2.0, "threads": 4, "cache_dir": "x"})
    monkeypatch.setattr(second_asr, "transcribe", lambda m, s, e, **kw: {
        **c["asr2"], "cached": True})
    monkeypatch.setattr(quality_gate, "cut_energy", lambda m, s, e: _energy(c))
    args = (case["caption_words"], c["start_s"], c["end_s"], "w.mp4")
    assert engine.gate_cut_errors(*args) == []
    monkeypatch.setattr(quality_gate, "ending_seat_enabled", lambda cfg=None: False)
    assert engine.gate_cut_errors(*args) == [c["gate_error"]]


# ---------------------------------------------------------------- the render, end to end

def _render(cid, tmp_path, monkeypatch, **kw):
    return tf._render_asr2(cid, tmp_path, monkeypatch, **kw)


def _caps(row):
    return json.loads(row["features_json"])["captions"]


@pytest.mark.parametrize("left", [tf.AMPLE, tf.TIGHT, None], ids=["ample", "tight", "no_window"])
def test_a_seated_filler_ships_on_its_first_render_at_its_own_cut(db, tmp_path, monkeypatch,
                                                                  left):
    """period in a window with time, on a tight budget and outside a window: the gate's checks read
    'and' after the cut, the cut stays at 39.00 s and the first render passes. No pull-back, no
    re-cut, one second-ASR pass (the gate reads the caption step's back from its cache)."""
    ok, row, seen, calls, runs = _render("period", tmp_path, monkeypatch, left=left, seat=True)
    assert ok is True and row["state"] == "rendered" and len(calls) == 1 and seen == [[]]
    assert (row["start_s"], row["end_s"]) == (16.1, 39.0)
    assert "asr2_end" not in _caps(row) and "recut" not in json.loads(row["features_json"])
    assert runs == [("second_asr", 16.1, 39.0)]


def test_a_seated_filler_still_fails_on_a_last_word_with_no_sentence_end(db, tmp_path,
                                                                         monkeypatch):
    """The seat moves no_period's 'and' to +0.68 s, so the filler error is gone; the captions and
    small both leave 'listen' unpunctuated, so the gate now fails it as ending mid-sentence. The
    caption step reads only filler endings before the render, so the cut goes to the render; the
    re-cut after the gate (time enough here) plans the lane's 26.405 s, which ends on 'And' and
    fails too."""
    ok, row, seen, calls, runs = _render("no_period", tmp_path, monkeypatch, rerun=True,
                                         left=tf.AMPLE, seat=True)
    assert ok is False and row["state"] == "failed"
    assert seen[0] == ["ending: ends mid-sentence: last word 'listen', next 'and' at +0.68 s"]
    assert "asr2_end" not in _caps(row)
    assert len(calls) == 2 and "captions end on a filler/dangling word 'And'" in seen[1][0]


# ---------------------------------------------------------------- the re-cut lane's plan, once

def test_the_recut_after_the_gate_never_repeats_the_lanes_plan_that_failed_before_it(
        db, tmp_path, monkeypatch, capsys):
    """Seat off (a filler the audio cannot seat): no_period's caption step runs the re-cut lane,
    whose plan (26.405 s) fails the gate's checks before the render; the gate then fails the cut
    as before. The re-cut after it does not plan and render that very cut again (a second precise
    pass, render and gate); it stops there."""
    ok, row, seen, calls, runs = _render("no_period", tmp_path, monkeypatch, rerun=True,
                                         left=tf.AMPLE)
    out = capsys.readouterr().out
    plan = _caps(row)["asr2_end"]["recut"]["plan"]
    assert plan == {"from": [5.6, 22.6], "sides": ["end"], "cut": [5.6, 26.405]}
    assert ok is False and len(calls) == 1 and seen == [[END["no_period"]["gate_error"]]]
    assert [r[0] for r in runs].count("words_window") == 1
    assert "no re-cut (the re-cut lane's plan for this cut failed before the render" in out


def test_recut_failed_before_render_matches_only_that_cut_and_sides():
    from clipengine import engine
    tried = {"captions": {"asr2_end": {"recut": {
        "none": "the re-cut 5.59-27.12s fails them too (x)",
        "plan": {"from": [5.59, 22.65], "sides": ["end"], "cut": [5.59, 27.125]}}}}}
    clip = {"start_s": 5.59, "end_s": 22.650000000000002, "features_json": json.dumps(tried)}
    assert "fails them too" in engine._recut_failed_before_render(clip, ["end"])
    assert engine._recut_failed_before_render(clip, ["start", "end"]) is None
    assert engine._recut_failed_before_render({**clip, "end_s": 22.7}, ["end"]) is None
    for rec in ({"skipped": "40s left"}, {"none": "TimeoutError: x"}, {}):
        feats = json.dumps({"captions": {"asr2_end": {"recut": rec}}})
        assert engine._recut_failed_before_render({**clip, "features_json": feats}, ["end"]) \
            is None
