"""Filler endings: a clip never ends on a dangling connective or filler, and its cuts never sit
inside a word of the transcripts the gate reads.

The strong caption pass (large-v3-turbo) and the gate's second ASR (whisper small) often time the
first word after a pause from the end of the word before it. A cut select placed in that pause on
its own words can then sit inside a caption word stretched back over it, and the second ASR can
hear a filler before a cut the audio only voices it after. Every case below is synthetic: invented
sentences (_talk: a word every STEP s), the transcripts the engine reads made from them, and window
audio voiced exactly where the words are said (_window_db).

  so_end / it_end   the caption pass stretches the word after the end pause back over the cut
                    ('So.', a bare 'It'); the start-fix seat cannot move it (select's sidecar holds
                    no word past the cut). boundaries.tidy_cut re-cuts the end before it, or goes
                    back to the sentence end before it: both then pass the gate's cut checks
  start_in_word     the start cut sits inside the first caption word, in running speech, with no
                    sentence start in reach on the (unpunctuated) caption words: left for the gate
                    (the "no valid boundary" edge)
  but_heard / so_heard  the caption words end clean; the second ASR stretches 'but' / 'So' back
                    before the cut, and the captions carry no . ? ! at all, so no cut there passes
                    the gate's ending: left for the gate

The second ASR's 'and' (ASR2 cases): the burned words end clean, so tidy_cut keeps the cut, and the
gate's second ASR hears 'and' 0.21 / 0.24 s before it (stretched back over the pause: the voice
comes back 0.30 / 0.68 s after the cut). Each case holds the gate's report on its cut, and
(`replays`) the second-ASR passes on the cuts the engine moves it to and the re-cut lane's precise
words.

  period     the end goes back past 'and' to after 'mess.'; a pass that hears 'mess,' there fails
             it, and the re-cut lane's extension past the next sentence passes: fixed before the
             render when the window has the time for that ASR
  no_period  'listen' carries no . ? ! in either transcript, so no end back past 'and' passes; the
             re-cut lane's extension ends on a stretched 'And' and fails: left for the gate
"""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from clipengine import boundaries
from clipengine import quality_gate
from clipengine import runstate
from clipengine import transcript
from test_stage1_cuts import _fake_recompose, words_from
from helpers import fake_store_clip

VC = {"min_dur_s": 15, "max_dur_s": 75, "lead_pad_s": 0.15, "tail_pad_s": 0.35,
      "strict_ends": True, "min_end_gap_s": 0.25, "hard_gap_s": 1.5, "sentence_gap_s": 0.7}
CUT_CHECKS = ("word_boundaries", "sentence_bounds", "ending")


# ---------------------------------------------------------------- synthetic speech

STEP, WORD = 0.3, 0.25          # a word every STEP s inside a sentence, voiced for WORD s of it
VOICE_DB, QUIET_DB = -20.0, -70.0


def _talk(t0, *script):
    """The precise words of `script` from `t0` (absolute s): a string is a sentence (a word every
    STEP s, each voiced for WORD s), a number the pause from the end of the last word to the first
    word of the next sentence."""
    out, t = [], float(t0)
    for part in script:
        if isinstance(part, (int, float)):
            t += float(part) - (STEP - WORD)
            continue
        for tok in part.split():
            out.append({"w": tok, "start": round(t, 3), "end": round(t + WORD, 3)})
            t += STEP
    return out


BODY = ("We rented a small boat down at the harbor.", 0.5,
        "The engine coughed twice and then it ran just fine.", 0.5,
        "My cousin steered while I kept an eye on the gulls.", 0.5,
        "Halfway out a thick fog rolled in from the north.", 0.5,
        "We could barely see the end of our own rope.", 0.5,
        "Nobody said a word for a long time.")          # 19.6 s from its first word to its last


def _within(words, a, b, p=None):
    """Copies of the words whose middle lies in [a, b] (with word probability `p`)."""
    return [dict(w, **({} if p is None else {"p": p})) for w in words
            if a <= (w["start"] + w["end"]) / 2.0 <= b]


def _bare(words):
    """`words` as a pass that came back lower case with no punctuation has them."""
    return [{**w, "w": w["w"].strip(".,?!").lower()} for w in words]


def _edit(words, word, at, **new):
    """The word `word` that starts at `at` s in `words`, changed to `new` (in place)."""
    [x] = [x for x in words if x["w"] == word and abs(x["start"] - at) < 1e-6]
    x.update(new)
    return x


def _pass(words, start, end, tail=2.0):
    """A second-ASR record for the cut [start, end]: `words` heard from the start to `tail` s past
    the end, clip-relative."""
    return {"words": [{"w": w["w"], "start": round(w["start"] - start, 3),
                       "end": round(w["end"] - start, 3)}
                      for w in words if w["end"] > start and w["start"] < end + tail],
            "dur": round(end - start, 3), "tail_s": tail, "model": "small",
            "window": [round(start, 3), round(end, 3)]}


def _window_db(c):
    """The case's window audio: 10 ms frame levels (dBFS) from 0 s, VOICE_DB where a word of
    c["voice"] is said, QUIET_DB elsewhere, then c["levels"] ([(a, b, dB)]) on top."""
    t = (np.arange(int(round(c["window_s"] / 0.01))) + 0.5) * 0.01
    db = np.full(len(t), QUIET_DB)
    for w in c["voice"]:
        db[(t >= w["start"]) & (t < w["end"])] = VOICE_DB
    for a, b, level in c.get("levels", ()):
        db[(t >= a) & (t < b)] = level
    return db


def _audio(c):
    """(rms, t0): the case's window audio as 10 ms frame RMS from 0 s."""
    return 10.0 ** (_window_db(c) / 20.0), 0.0


# ---------------------------------------------------------------- the cases the tidy reads

def _so_end():
    """'... finally dry.' (ends 37.05 s), quiet until 'So' at 37.35 s; the caption pass hears
    'So.' from 37.13 s, over the cut at 37.32 s."""
    voice = _talk(14.80, *BODY) + _talk(35.00, "By noon the shed was finally dry.", 0.30,
                                         "So we carried the boxes back inside.")
    start, end = 14.65, 37.32
    select = _within(voice, start, end)
    caption = _within(voice, start, end, p=0.99) + [
        {"w": "So.", "start": 37.13, "end": 37.60, "p": 0.99}]
    return {"source_id": "win_so_end", "window_s": 44.0, "start_s": start, "end_s": end,
            "refined": True, "voice": voice, "select_words": select, "caption_words": caption,
            "asr2": _pass(voice, start, end),
            "gate_errors": ["word_boundaries: end cut inside word 'So.' (start=37.130 end=37.600)",
                            "ending: captions end on a filler/dangling word 'So.'"]}


def _it_end():
    """'... the night crew.' (ends 33.15 s), quiet until 'It' at 33.59 s; the caption pass hears
    a bare 'It' from 33.39 s, over the cut at 33.58 s."""
    voice = _talk(10.00, *BODY) + _talk(30.20, "She started a club and named it the night crew.",
                                         0.44, "It was busy every single night.")
    start, end = 9.85, 33.58
    select = _within(voice, start, end)
    caption = _within(voice, start, end, p=0.99) + [
        {"w": "It", "start": 33.39, "end": 33.70, "p": 0.99}]
    return {"source_id": "win_it_end", "window_s": 44.0, "start_s": start, "end_s": end,
            "refined": True, "voice": voice, "select_words": select, "caption_words": caption,
            "asr2": _pass(voice, start, end),
            "gate_errors": ["word_boundaries: end cut inside word 'It' (start=33.390 end=33.700)"]}


def _start_in_word():
    """The speaker runs on into the clip: select's first word 'nobody' starts at 3.70 s, right
    after 'because'; the caption pass (lower case, unpunctuated but for its last word, each word
    timed to the next) hears 'nobody' from 3.565 s, over the cut at 3.625 s."""
    voice = _talk(1.00, "We had to leave the house before dawn because nobody wanted to miss the "
                        "early ferry.", 0.5, *BODY, 0.5,
                  "Did anyone know what would happen next?", 0.9, "We found out soon enough.")
    start, end = 3.625, 28.80
    select = _within(voice, start, end)
    caption = _bare(select[:-1]) + [dict(select[-1])]
    for a, b in zip(caption, caption[1:]):
        a["end"] = b["start"]
    caption[0]["start"] = 3.565
    return {"source_id": "win_start", "window_s": 44.0, "start_s": start, "end_s": end,
            "refined": True, "voice": voice, "select_words": select,
            "caption_words": [dict(w, p=0.99) for w in caption],
            "asr2": _pass(voice, start, end),
            "gate_errors": ["word_boundaries: start cut inside word 'nobody' (start=3.565 "
                            "end=4.000)",
                            "sentence_bounds: start mid-sentence: prev='nobody' first='wanted'",
                            "visual_quality: too dark: median luma 28.0 < 35"]}


def _filler_heard(sid, start, end, voice, last, last_at, filler, filler_at, claim):
    """Unpunctuated captions that end on `last`; the second ASR holds `last` on to `claim[0]` and
    hears `filler` from there to `claim[1]` (clip-relative), over the pause before the cut."""
    asr2 = _pass(voice, start, end)
    _edit(asr2["words"], last, last_at, end=claim[0])
    _edit(asr2["words"], filler, filler_at, start=claim[0], end=claim[1])
    ls = claim[0] - round(end - start, 3)
    return {"source_id": sid, "window_s": 44.0, "start_s": start, "end_s": end, "refined": True,
            "voice": voice, "select_words": _within(voice, start, end),
            "caption_words": _bare(_within(voice, start, end, p=0.99)), "asr2": asr2,
            "gate_errors": [f"ending: ends on a filler/conjunction {filler!r} (starts {ls:+.2f} s "
                            f"from the cut)"]}


def _but_heard():
    """'... out of them' (ends 35.05 s), 'but' voiced from 35.85 s; the second ASR hears it from
    0.22 s before the cut at 35.53 s."""
    voice = _talk(10.50, *BODY, 0.5, "I ask a lot of questions and I still get good stuff out of "
                                     "them", 0.80, "but the other thing we do is listen.")
    c = _filler_heard("win_but", 10.40, 35.53, voice, "them", 24.40, "but", 25.45, (24.91, 25.73))
    assert c["gate_errors"] == ["ending: ends on a filler/conjunction 'but' (starts -0.22 s from "
                                "the cut)"]
    return c


def _so_heard():
    """'... kept going down' (ends 34.35 s), 'So' voiced from 34.80 s; the second ASR hears it
    from 0.20 s before the cut at 34.54 s."""
    voice = _talk(12.20, *BODY, 0.5, "Every week the price kept going down", 0.45,
                  "So they changed the plan again.")
    c = _filler_heard("win_so", 12.05, 34.54, voice, "down", 22.05, "So", 22.75, (22.29, 22.81))
    assert c["gate_errors"] == ["ending: ends on a filler/conjunction 'So' (starts -0.20 s from "
                                "the cut)"]
    return c


TIDY = {"so_end": _so_end(), "it_end": _it_end(), "start_in_word": _start_in_word(),
        "but_heard": _but_heard(), "so_heard": _so_heard()}


def _asr2(c, start, end):
    """The gate's second ASR for a cut [start, end], clip-relative: the case's pass re-based on
    `start`, heard to its tail past `end` (the gate re-runs it on a moved cut; its word times there
    are taken to be the same)."""
    t = float(c["start_s"])
    return [{"w": w["w"], "start": t + w["start"] - start, "end": t + w["end"] - start}
            for w in c["asr2"]["words"] if t + w["start"] < end + c["asr2"]["tail_s"]]


def _gate_cut_errors(c, words, start, end):
    """The gate's own cut checks on a cut: word_boundaries and sentence_bounds on the sidecar,
    the ending on the second ASR and the burned captions (engine.gate_cut_errors' set)."""
    errs = []
    for k, m in (("word_boundaries", quality_gate.measure_word_boundaries(words, start, end)),
                 ("sentence_bounds", quality_gate.measure_sentence_bounds(words, start, end)),
                 ("ending", quality_gate.ending_verdict(
                     _asr2(c, start, end), quality_gate.clip_words(words, start, end),
                     end - start, tail_s=c["asr2"]["tail_s"]))):
        if m.get("passed") is False:
            errs += [f"{k}: {e}" for e in m.get("errors") or []]
    return errs


def _cut_errors(c):
    """The gate's errors on the case's cut that come from its cut checks."""
    return [e for e in c["gate_errors"] if e.split(":")[0] in CUT_CHECKS]


def _tidy(c, words=None, **kw):
    rms, t0 = _audio(c)
    return boundaries.tidy_cut(c["caption_words"] if words is None else words, c["start_s"],
                               c["end_s"], cfg={**VC, **kw.pop("cfg", {})}, rms=rms, t0=t0,
                               media_dur=c["window_s"], **kw)


# ---------------------------------------------------------------- the cases, as the gate reads them

@pytest.mark.parametrize("cid", sorted(TIDY))
def test_each_case_reproduces_the_gates_cut_errors(cid):
    c = TIDY[cid]
    assert _gate_cut_errors(c, c["caption_words"], c["start_s"], c["end_s"]) == _cut_errors(c)


@pytest.mark.parametrize("cid, name, word", [("so_end", "end", "So."), ("it_end", "end", "It"),
                                             ("start_in_word", "start", "nobody")])
def test_the_start_fix_seat_leaves_the_stretched_word_on_the_cut(cid, name, word):
    """The start-fix seat (engine._seat, select's sidecar as the reference): the caption word over
    the cut stays, so a render without the tidy takes the bad boundary."""
    c = TIDY[cid]
    rms, t0 = _audio(c)
    t = c[f"{name}_s"]
    words, moved = boundaries.seat_words(c["caption_words"], c["select_words"], t, rms, t0,
                                         tol=quality_gate.WORD_BOUNDARY_INSIDE_TOL_S)
    assert moved == [] and words == c["caption_words"]
    hit = [w["w"] for w in words if w["start"] + 0.03 < t < w["end"] - 0.03]
    assert hit == [word]
    # select's words never had a word past the end cut to place it by
    assert not [w for w in c["select_words"] if w["start"] >= c["end_s"]]


# ---------------------------------------------------------------- one test per case

def test_the_end_is_recut_before_the_stretched_so():
    c = TIDY["so_end"]
    r = _tidy(c)
    assert r["tidy"] == {"start": "ok", "end": "recut"} and r["last"] == "dry."
    assert r["start"] == c["start_s"]
    assert 37.05 <= r["end"] <= 37.13                    # after 'dry.', before 'So.' as captioned
    assert _gate_cut_errors(c, c["caption_words"], r["start"], r["end"]) == []
    assert quality_gate.clip_words(c["caption_words"], r["start"], r["end"])[-1]["w"] == "dry."


def test_the_end_goes_back_to_the_sentence_end_before_a_bare_it():
    c = TIDY["it_end"]
    r = _tidy(c)
    assert r["tidy"] == {"start": "ok", "end": "pull_back"} and r["last"] == "crew."
    assert 33.15 <= r["end"] <= 33.39                    # after 'crew.', before the bare 'It'
    assert r["end"] - r["start"] >= VC["min_dur_s"]
    assert _gate_cut_errors(c, c["caption_words"], r["start"], r["end"]) == []


def test_no_sentence_start_in_reach_is_left_for_the_gate():
    """The cut at 3.625 s sits inside the caption pass's 'nobody' in running speech: the voice runs
    right before it (no sentence start there), and the caption words carry no . ? ! or pause until
    the clip's last word. The start stays and the gate still fails it."""
    c = TIDY["start_in_word"]
    r = _tidy(c)
    assert r["tidy"] == {"start": "none", "end": "ok"}
    assert "nobody" in r["why"]["start"]
    assert (r["start"], r["end"]) == (c["start_s"], c["end_s"])
    assert _gate_cut_errors(c, c["caption_words"], r["start"], r["end"]) == _cut_errors(c)


@pytest.mark.parametrize("cid, filler", [("but_heard", "but"), ("so_heard", "So")])
def test_the_burned_words_end_clean_and_the_cut_stays(cid, filler):
    """The connective the gate heard is the second ASR's, stretched back over the pause before
    the cut; the burned words show nothing to fix, and they carry no sentence end any cut could
    move to (the caption pass came back unpunctuated), so the gate judges them as before."""
    c = TIDY[cid]
    r = _tidy(c)
    assert r["tidy"] == {"start": "ok", "end": "ok"} and r["why"] == {}
    assert (r["start"], r["end"]) == (c["start_s"], c["end_s"])
    asr = [w for w in _asr2(c, c["start_s"], c["end_s"]) if w["w"] == filler]
    assert asr and asr[-1]["start"] < (c["end_s"] - c["start_s"]) - 0.12   # heard before the cut
    burned = [w for w in c["caption_words"] if c["start_s"] <= w["start"] < c["end_s"]]
    assert not [w for w in burned if boundaries._ends_clean(w["w"])]


# ---------------------------------------------------------------- happy path, edges

@pytest.mark.parametrize("cid", ["so_end", "it_end"])
def test_a_clean_cut_is_unchanged(cid):
    """The tidied cuts are clean: a second look changes nothing (no audio read, no move)."""
    c = TIDY[cid]
    r = _tidy(c)
    again = boundaries.tidy_cut(c["caption_words"], r["start"], r["end"], cfg=VC)
    assert again["tidy"] == {"start": "ok", "end": "ok"} and again["why"] == {}
    assert (again["start"], again["end"]) == (r["start"], r["end"])


def test_a_clean_synthetic_cut_is_unchanged():
    words = words_from("We drove out to the desert that night. Nobody said a word the whole "
                       "way. Then my dad pulled over and told me everything.", 10.0,
                       gaps={8: 0.6, 15: 0.6})
    start, end = words[0]["start"] - 0.1, words[-1]["end"] + 0.2
    r = boundaries.tidy_cut(words, start, end, cfg={**VC, "min_dur_s": 5})
    assert r == {"start": start, "end": end, "tidy": {"start": "ok", "end": "ok"}, "why": {}}


def test_no_valid_end_leaves_the_cut_for_the_gate():
    """it_end with a 23.6 s minimum: going back to 'crew.' (at most 33.39 - 9.85 = 23.54 s) breaks
    it and no word follows the cut to move on to, so the cut stays."""
    c = TIDY["it_end"]
    r = _tidy(c, cfg={"min_dur_s": 23.6})
    assert r["tidy"]["end"] == "none" and (r["start"], r["end"]) == (c["start_s"], c["end_s"])


def test_without_audio_nothing_moves():
    c = TIDY["so_end"]
    r = boundaries.tidy_cut(c["caption_words"], c["start_s"], c["end_s"], cfg=VC)
    assert r["tidy"]["end"] == "none" and r["end"] == c["end_s"] and "audio" in r["note"]


def _voice_rms(words, t1):
    """10 ms frames from 0 s: -20 dBFS where a word is voiced, -70 dBFS between them."""
    t = (np.arange(int(round(t1 / 0.01))) + 0.5) * 0.01
    on = np.zeros(len(t), bool)
    for w in words:
        on |= (t >= w["start"]) & (t < w["end"])
    return np.where(on, 0.1, 0.0003)


# so_end's shape on words_from words: the word after the pause that ends "... the whole way." is
# timed from the end of 'way.' by the strong pass
STORY = words_from("We drove out to the desert that night. Nobody said a word the whole way. "
                   "So we sat there in the dark for an hour. Then my dad pulled over.", 10.0,
                   gaps={8: 0.6, 15: 0.5, 25: 0.6})
STRETCHED = [dict(w) for w in STORY]
STRETCHED[15]["start"] = STORY[14]["end"]                          # 'So' over the pause


def test_the_end_moves_on_when_going_back_breaks_min_length():
    cut = STORY[14]["end"] + 0.25                                   # in the pause after 'way.'
    rms = _voice_rms(STORY, 30.0)
    back = boundaries.tidy_cut(STRETCHED, 9.9, cut, cfg={**VC, "min_dur_s": 5}, rms=rms)
    assert back["tidy"]["end"] == "recut" and back["last"] == "way."
    assert STORY[14]["end"] <= back["end"] <= STRETCHED[15]["start"]
    longer = {**VC, "min_dur_s": cut - 9.9 + 0.5}                   # 'way.' cannot keep it
    r = boundaries.tidy_cut(STRETCHED, 9.9, cut, cfg=longer, rms=rms)
    assert r["tidy"]["end"] == "extend" and r["last"] == "hour."
    assert STORY[24]["end"] <= r["end"] <= STORY[25]["start"]


def test_an_end_whose_last_word_is_a_filler_goes_back():
    """the cut after a bare 'So' (its own word, not stretched): back to 'way.'"""
    cut = STORY[15]["end"] + 0.02
    r = boundaries.tidy_cut(STORY, 9.9, cut, cfg={**VC, "min_dur_s": 5},
                            rms=_voice_rms(STORY, 30.0))
    assert r["why"]["end"] == "ends on 'So'" and r["tidy"]["end"] == "pull_back"
    assert STORY[14]["end"] <= r["end"] <= STORY[15]["start"]


@pytest.mark.parametrize("tok, out", [("So.", True), ("but", True), ("And,", True),
                                      ("the", True), ("um.", True), ("It", True),
                                      ("out,", True), ("it.", False), ("crew.", False),
                                      ("them", False), ("resistance", False), ("It's", False)])
def test_dangling_end(tok, out):
    assert boundaries.dangling_end(tok) is out


# ---------------------------------------------------------------- the snap and the re-cut

SO = words_from("We had to fix the roof. So. My brother held the ladder and I climbed up there. "
                "We left.", 30.0, gaps={6: 0.35, 7: 0.3, 17: 0.6})


def test_snap_drops_a_trailing_so_for_the_sentence_end_before_it():
    """'... the roof. So.' with the cut after 'So.': the period no longer makes it an end, and the
    end goes back first (it used to stand as a clean sentence end)."""
    end = SO[6]["end"] + 0.1                                 # after 'So.'
    r = boundaries.snap_clip(SO, 29.9, end, cfg={**VC, "min_dur_s": 1}, edges_clean=(True, False),
                             media_dur=60.0)
    assert r["snap"]["end"] == "pull_back"
    assert SO[5]["end"] <= r["end"] <= SO[6]["start"]        # after 'roof.', before 'So.'


def test_snap_moves_on_past_a_trailing_so_when_going_back_breaks_min_length():
    end = SO[6]["end"] + 0.1
    r = boundaries.snap_clip(SO, 29.9, end, cfg={**VC, "min_dur_s": 2.5},
                             edges_clean=(True, False), media_dur=60.0)
    assert r["snap"]["end"] == "extend" and SO[16]["end"] <= r["end"] <= SO[17]["start"]


def test_recut_end_never_picks_a_dangling_end():
    end = SO[16]["end"] + 0.1                                # after '... climbed up there.'
    got, info = boundaries.recut_end(SO, 29.9, end, cfg={**VC, "min_dur_s": 1}, media_dur=60.0)
    assert got is not None and info["last"].endswith("the roof.")


def test_clean_clip_never_ends_on_so():
    sents = transcript.sentences(SO, gap_s=0.7)
    k = next(i for i, s in enumerate(sents) if s["text"] == "So.")
    c = boundaries.clean_clip(SO, sents, 0, k, cfg={**VC, "min_dur_s": 1, "max_dur_s": 3.0},
                              edges_clean=(True, False))
    assert c is not None and SO[c["w1"]]["w"] == "roof."
    off = boundaries.clean_clip(SO, sents, 0, k, cfg={**VC, "min_dur_s": 1, "max_dur_s": 3.0,
                                                      "strict_dangling": False},
                                edges_clean=(True, False))
    assert SO[off["w1"]]["w"] == "So."                       # config select.strict_dangling off


# ---------------------------------------------------------------- the render lane, end to end

@pytest.fixture
def db(tmp_path, monkeypatch):
    from clipengine import engine
    runstate.reset()
    monkeypatch.setattr(engine, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(engine, "vertical_cfg", lambda v: dict(VC))
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", None)
    runstate.init()
    return tmp_path


def _load_audio(c):
    """transcript.load_audio on the case's window audio (16 kHz samples of its 10 ms frames)."""
    rms, t0 = _audio(c)

    def load_audio(m, lo, hi, sr=16000):
        n = int(round((hi - lo) / 0.01))
        k = np.clip(np.floor((lo - t0) / 0.01 + np.arange(n) + 1e-6).astype(int), 0, len(rms) - 1)
        return np.repeat(rms[k], sr // 100).astype(np.float32)
    return load_audio


def _render_tidy(cid, tmp_path, monkeypatch, tidy=True, context=False):
    """Render case `cid` on select's sidecar, the strong pass's words (the captions) and the window
    audio; the gate = its own cut checks on the burned words and the second ASR."""
    from clipengine import second_asr
    from clipengine import engine
    c = TIDY[cid]
    media = tmp_path / f"{c['source_id']}.mp4"
    media.write_bytes(b"\0")
    runstate.add_source(c["source_id"], local_path=str(media), content_type="auto",
                        vertical="general")
    runstate.update_source(c["source_id"], state="selected", duration_s=c["window_s"])
    wp = tmp_path / "select.json"
    wp.write_text(json.dumps(c["select_words"]))
    clip_id = runstate.add_clip(c["source_id"], c["start_s"], c["end_s"], 0.4, "auto", hook="h",
                                words_path=str(wp), features_json={"refined": c["refined"]})
    monkeypatch.setattr(engine, "CAPTION_STRONG_MODEL", "large-v3-turbo")
    monkeypatch.setattr(engine.transcript, "WORKER", None)
    # the strong pass on the cut +-0.5 s with no seat (render.caption_context off), the cut seat
    # off: the caption words reach the tidy as heard
    if not context:
        monkeypatch.setattr(engine, "caption_context", lambda cfg=None: (0.5, 0.5, False))
        monkeypatch.setattr(engine, "cut_seat_enabled", lambda cfg=None: False)
    monkeypatch.setattr(engine, "_strong_pass", lambda m, lo, hi, timeout: [
        {k: v for k, v in w.items() if k != "cap"} for w in c["caption_words"]])
    monkeypatch.setattr(second_asr, "settings", lambda cfg=None: {"enabled": False})
    monkeypatch.setattr(engine, "CFG", {**engine.CFG, "daily": {
        **(engine.CFG.get("daily") or {}), "cut_tidy": {"enabled": tidy}}})
    monkeypatch.setattr(engine.transcript, "load_audio", _load_audio(c))
    monkeypatch.setattr(engine.transcript, "words_window", lambda m, a, b, model=None: [])
    calls, seen = [], []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    monkeypatch.setattr(engine, "_store_clip", fake_store_clip)

    def gate(out, clip, qc=None):
        errs = _gate_cut_errors(c, engine._load_words_sidecar(clip["words_path"]),
                                float(clip["start_s"]), float(clip["end_s"]))
        seen.append(errs)
        return {"verdict": "fail" if errs else "pass", "drop_reason": "; ".join(errs),
                **{k: {"passed": False, "errors": [e.split(": ", 1)[1] for e in errs
                                                   if e.startswith(k)]}
                   for k in CUT_CHECKS if any(e.startswith(k) for e in errs)}}
    monkeypatch.setattr(engine, "_run_quality_gate", gate)
    ok = engine.render(runstate.clip(clip_id))
    return ok, runstate.clip(clip_id), seen, calls


@pytest.mark.parametrize("cid, last", [("so_end", "dry."), ("it_end", "crew.")])
def test_render_tidies_the_end_and_passes_the_gate_on_its_first_render(db, tmp_path,
                                                                      monkeypatch, cid, last):
    from clipengine import engine
    c = TIDY[cid]
    ok, row, seen, calls = _render_tidy(cid, tmp_path, monkeypatch)
    assert ok is True and seen == [[]] and len(calls) == 1 and row["state"] == "rendered"
    assert row["start_s"] == c["start_s"] and row["end_s"] < c["end_s"]
    tidy = json.loads(row["features_json"])["captions"]["tidy"]
    assert tidy["tidy"]["start"] == "ok" and tidy["last"] == last
    assert tidy["new"] == [row["start_s"], row["end_s"]]
    burned = quality_gate.clip_words(engine._load_words_sidecar(row["words_path"]),
                                     row["start_s"], row["end_s"])
    assert burned[-1]["w"] == last


@pytest.mark.parametrize("cid", ["so_end", "it_end"])
def test_render_with_the_tidy_off_fails_on_the_cases_cut_errors(db, tmp_path, monkeypatch, cid):
    c = TIDY[cid]
    ok, row, seen, _calls = _render_tidy(cid, tmp_path, monkeypatch, tidy=False)
    assert ok is False and seen[0] == _cut_errors(c)
    assert (row["start_s"], row["end_s"]) == (c["start_s"], c["end_s"])


def test_with_the_caption_context_it_end_ends_on_crew_without_a_tidy(db, tmp_path, monkeypatch):
    """render.caption_context on: the strong words are seated on the window audio before the
    clip's own are picked, so 'It', the next sentence's first word stretched back over the end
    cut, leaves the clip; the burned words end on 'crew.' and the cut stands, tidy off."""
    from clipengine import engine
    c = TIDY["it_end"]
    ok, row, seen, _calls = _render_tidy("it_end", tmp_path, monkeypatch, tidy=False,
                                         context=True)
    assert ok is True and seen == [[]]
    assert row["end_s"] == c["end_s"]
    assert c["start_s"] - 0.03 <= row["start_s"] <= c["start_s"]   # the seat's lead (cut_seat)
    burned = quality_gate.clip_words(engine._load_words_sidecar(row["words_path"]),
                                     row["start_s"], row["end_s"])
    assert burned[-1]["w"] == "crew."


def test_render_leaves_a_start_with_no_sentence_start_in_reach_for_the_gate(db, tmp_path,
                                                                            monkeypatch):
    c = TIDY["start_in_word"]
    ok, row, seen, _calls = _render_tidy("start_in_word", tmp_path, monkeypatch)
    assert ok is False and seen[0] == _cut_errors(c)
    tidy = json.loads(row["features_json"])["captions"]["tidy"]
    assert tidy["tidy"]["start"] == "none" and tidy["new"] == tidy["old"]


# ---------------------------------------------------------------- the second ASR's 'and'
# These pin the ending probe that reads only the 100 ms after a word's claimed onset. The ending
# seat (quality_gate.ending_seat) seats a word the second ASR timed over a pause on the audio, and
# hears both cases' 'and' after the cut; the tests below run with that seat off (its kill switch,
# `seat=False`), so the pull-back / re-cut / time check they test stay pinned for a filler the
# audio cannot seat. test_onset_seating.py has the same cases with the seat on.

def _period():
    """'... was a mess.' (voiced to 38.92 s), quiet until 'And' at 39.30 s; the cut at 39.00 s. The
    second ASR ends 'mess.' at 38.79 s and hears 'and' from there; a pass on the cut it is pulled
    back to hears 'mess,'. The next sentence ends 'mattered most.' (44.05 s)."""
    voice = _talk(16.27, *BODY, 0.10, "When we got back the whole kitchen was a mess.", 0.38,
                  "And then we spent a week sorting through the boxes and kept only what "
                  "mattered most.", 0.71, "Some of it went to the neighbors.")
    start, end = 16.10, 39.00
    heard = _pass(voice, start, end)
    _edit(heard["words"], "mess.", 22.57, end=22.69)
    _edit(heard["words"], "And", 23.20, w="and", start=22.69, end=23.45)
    back = _pass(voice, start, 38.905)
    _edit(back["words"], "mess.", 22.57, w="mess,")
    _edit(back["words"], "And", 23.20, w="and", start=23.105)
    error = "ends on a filler/conjunction 'and' (starts -0.21 s from the cut)"
    return {"source_id": "win_period", "window": "win_period.mp4", "window_s": 48.0,
            "start_s": start, "end_s": end, "refined": True, "voice": voice,
            "select_words": _within(voice, start, end),
            "caption_words": _within(voice, start, end, p=0.99), "asr2": heard,
            "gate": {"word_boundaries": {"passed": True, "errors": []},
                     "sentence_bounds": {"passed": True, "errors": []},
                     "ending": {"passed": False, "last_word": "and", "last_word_start_s": -0.21,
                                "errors": [error]},
                     "second_asr": {"model": "small", "tail_s": 2.0, "cached": False}},
            "gate_errors": ["ending: " + error],
            "replays": {"pull_back": {"cut": [16.1, 38.905], "asr2": back},
                        "recut": {"refine": {"lo": 24.0, "hi": 48.0,
                                             "words": [dict(w) for w in voice
                                                       if w["start"] >= 24.0]},
                                  "cut": [16.1, 44.415], "asr2": _pass(voice, start, 44.415)}}}


def _no_period():
    """'... you really listen' (voiced to 22.50 s), quiet until 'and' at 23.28 s; the cut at
    22.60 s. The caption pass and the second ASR leave 'listen' unpunctuated (select's precise
    words have 'listen.'); the second ASR ends it at 22.36 s and hears 'and' from there. The re-cut
    lane's precise words time the 'And' after the next sentence end ('hum.', 25.98 s) from 26.40 s,
    0.2 s before its voice, where the pause is quietest."""
    voice = _talk(5.75, "When my aunt calls she wants to talk about her whole week.", 0.5,
                  "The bus was late and the boss was rude and the cat got out again.", 0.5,
                  "Some of it is funny and some of it is not.", 0.35,
                  "You nod and smile and laugh in the right places and you really listen", 0.78,
                  "and just nod along.", 0.10, "Most days you only hum.", 0.62,
                  "And half the time I hum too.", 1.0, "It never takes long.")
    start, end = 5.60, 22.60
    select = _within(voice, start, end)
    _edit(select, "listen", 22.25, w="listen.")
    caption = _bare(_within(voice, start, end, p=0.99))
    _edit(caption, "listen", 22.25, end=22.44)
    heard = _pass(voice, start, end)
    _edit(heard["words"], "listen", 16.65, end=16.76)
    _edit(heard["words"], "and", 17.68, start=16.76, end=17.78)
    refine = [dict(w) for w in voice if 7.6 <= w["start"] and w["end"] <= 32.1]
    _edit(refine, "and", 23.28, start=23.01)
    _edit(refine, "And", 26.60, start=26.40)
    error = "ends on a filler/conjunction 'and' (starts -0.24 s from the cut)"
    return {"source_id": "win_no_period", "window": "win_no_period.mp4", "window_s": 42.0,
            "start_s": start, "end_s": end, "refined": False, "voice": voice,
            "levels": [(26.40, 26.60, -75.0)], "select_words": select, "caption_words": caption,
            "asr2": heard,
            "gate": {"word_boundaries": {"passed": True, "errors": []},
                     "sentence_bounds": {"passed": True, "errors": []},
                     "ending": {"passed": False, "last_word": "and", "last_word_start_s": -0.24,
                                "errors": [error]},
                     "second_asr": {"model": "small", "tail_s": 2.0, "cached": False}},
            "gate_errors": ["ending: " + error],
            "replays": {"recut": {"refine": {"lo": 7.6, "hi": 32.1, "words": refine},
                                  "cut": [5.6, 26.405], "asr2": _pass(voice, start, 26.405)}}}


ASR2 = {"period": _period(), "no_period": _no_period()}


def _cut_energy(c, start, end, before_s=1.0, after_s=quality_gate.ENDING_ENERGY_AFTER_S):
    """quality_gate.cut_energy on the case's window audio: 20 ms frames (two 10 ms ones) from
    `start`, dB relative to the clip's median frame, times relative to the cut."""
    rms, t0 = _audio(c)
    i0 = int(round((start - t0) / 0.01))
    r = rms[i0:i0 + 2 * int((end - start + after_s) / 0.02)]
    n = len(r) // 2
    db = 20 * np.log10(np.sqrt((r[:2 * n].reshape(n, 2) ** 2).mean(axis=1)) + 1e-9)
    cut_i = int(round((end - start) / 0.02))
    lo = max(0, cut_i - int(round(before_s / 0.02)))
    return ((np.arange(lo, n) - cut_i) * 0.02).tolist(), \
        (db[lo:] - float(np.median(db[:max(1, cut_i)]))).tolist()


def _asr2_pass(c, start, end, rerun):
    """The gate's second ASR on the cut [start, end]: the case's pass, or (`rerun`) its recorded
    pass on that very cut; else the case's pass re-based on `start`, heard to its tail past `end`
    (None when that runs past what the case's pass heard: no model for it)."""
    cut = [round(start, 3), round(end, 3)]
    for r in [c["asr2"]] + ([r["asr2"] for r in c["replays"].values()] if rerun else []):
        if r["window"] == cut:
            return {**r, "cached": False}
    t, tail = float(c["start_s"]), c["asr2"]["tail_s"]
    if abs(start - t) > 1e-6 or end > float(c["end_s"]) + 1e-6:
        return None
    return {"words": [{"w": w["w"], "start": round(t + w["start"] - start, 3),
                       "end": round(t + w["end"] - start, 3)}
                      for w in c["asr2"]["words"] if t + w["start"] < end + tail],
            "dur": round(end - start, 3), "tail_s": tail, "model": "small", "window": cut,
            "cached": False}


def _gate_ending(c, words, start, end, rerun=False, seat=False):
    rec = _asr2_pass(c, start, end, rerun)
    return quality_gate.ending_verdict(rec["words"], quality_gate.clip_words(words, start, end),
                                       end - start, tail_s=rec["tail_s"],
                                       energy=_cut_energy(c, start, end), seat=seat)


@pytest.mark.parametrize("cid", sorted(ASR2))
def test_each_case_reproduces_the_gates_ending(cid):
    c = ASR2[cid]
    s, e = c["start_s"], c["end_s"]
    m = _gate_ending(c, c["caption_words"], s, e)
    assert ["ending: " + x for x in m["errors"]] == c["gate_errors"]
    assert m["last_word"] == "and" == c["gate"]["ending"]["last_word"]
    assert m["last_word_start_s"] == c["gate"]["ending"]["last_word_start_s"]
    for k in ("word_boundaries", "sentence_bounds"):          # the burned words end clean
        assert c["gate"][k]["passed"] is True
        assert getattr(quality_gate, f"measure_{k}")(c["caption_words"], s, e)["passed"] is True


def test_the_end_goes_back_past_the_one_word():
    from clipengine import engine
    c = ASR2["period"]
    s, e = c["start_s"], c["end_s"]
    t, info = engine.filler_end_pull_back(c["caption_words"], s, e, c["asr2"], c["gate"]["ending"],
                                          cfg=VC)
    # 'and' at 38.79 s + the gate's 0.12 s onset tolerance - one 5 ms step
    assert t == 38.905 and info == {"old_end": 39.0, "new_end": 38.905, "action": "pull_back",
                                    "dropped": "and", "last": "mess."}
    # on the case's pass the gate's cut checks pass it, 10 ms later 'and' is heard again
    assert _gate_ending(c, c["caption_words"], s, t)["passed"] is True
    assert engine.asr2_filler_end(_gate_ending(c, c["caption_words"], s, 38.915)) == "and"
    # the stream lean gate reads the ending only: the same end
    assert engine.filler_end_pull_back(c["caption_words"], s, e, c["asr2"], c["gate"]["ending"],
                                       cfg=VC, lean=True)[0] == 38.905


def test_no_end_back_past_and_when_the_word_before_ends_no_sentence():
    """no_period: 'listen' has no . ? ! in the second ASR or the captions."""
    from clipengine import engine
    c = ASR2["no_period"]
    t, why = engine.filler_end_pull_back(c["caption_words"], c["start_s"], c["end_s"], c["asr2"],
                                         c["gate"]["ending"], cfg=VC)
    assert t is None and "ends mid-sentence: last word 'listen'" in why
    assert engine.filler_end_pull_back(c["caption_words"], c["start_s"], c["end_s"], c["asr2"],
                                       {"passed": True}, cfg=VC)[0] is None


def test_only_a_filler_ending_alone_qualifies():
    from clipengine import engine
    c = ASR2["period"]
    rep = {**c["gate"], "verdict": "fail"}
    assert engine.filler_end_only(rep)
    assert engine.filler_end_only({**rep, "caption_asr": {"passed": False, "errors": ["x"]}})
    assert not engine.filler_end_only({**rep, "word_boundaries": {"passed": False,
                                                                  "errors": ["end cut inside"]}})
    two = {**c["gate"]["ending"], "errors": ["captions end on a filler/dangling word 'and'",
                                             c["gate"]["ending"]["errors"][0]]}
    assert not engine.filler_end_only({**rep, "ending": two})
    mid = {**c["gate"]["ending"], "errors": ["ends mid-sentence: last word 'mess'"]}
    assert not engine.filler_end_only({**rep, "ending": mid})


COST = {"strong": 45.4, "asr2_rtf": 2.5, "refine": 50.0, "render": 60.0}   # s on the fake clock


def _render_asr2(cid, tmp_path, monkeypatch, rerun=False, left=None, waiting=0, seat=False):
    """Render case `cid` (the strong pass's words = its captions, the window audio) with the
    second ASR on: the gate = its own cut checks on the burned words and its second ASR
    (_asr2_pass: `rerun` = the case's recorded passes on the moved cuts where they exist), cached
    per cut as second_asr caches it; the re-cut lane's precise words = the case's (`rerun`) or
    none. `left` = seconds of the window left at the render (None: no window); the clock moves by
    COST for each pass and render; `waiting` = clips of the window still to render; `seat` = the
    ending seat (quality_gate.ending_seat) on. Returns (ok, row, gate errors per render, renders,
    new ASR runs)."""
    import types
    from clipengine import second_asr
    from clipengine import engine
    c = ASR2[cid]
    clock = {"t": 1000.0}
    monkeypatch.setattr(engine, "time", types.SimpleNamespace(
        monotonic=lambda: clock["t"], time=__import__("time").time))
    media = tmp_path / c["window"]
    media.write_bytes(b"\0")
    runstate.add_source(c["source_id"], local_path=str(media), content_type="auto",
                        vertical="general")
    runstate.update_source(c["source_id"], state="selected", duration_s=c["window_s"])
    wp = tmp_path / "select.json"
    wp.write_text(json.dumps(c["select_words"]))
    clip_id = runstate.add_clip(c["source_id"], c["start_s"], c["end_s"], 0.7, "auto", hook="h",
                                words_path=str(wp), features_json={"refined": c["refined"]})
    for _ in range(waiting):
        runstate.add_clip(c["source_id"], c["end_s"] + 1.0, c["end_s"] + 2.0, 0.5, "auto",
                          hook="w")
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", None if left is None else clock["t"] + left)
    monkeypatch.setattr(engine, "CAPTION_STRONG_MODEL", "large-v3-turbo")
    monkeypatch.setattr(engine.transcript, "WORKER", None)

    def strong_pass(m, lo, hi, timeout):
        clock["t"] += COST["strong"]
        return [{"w": w["w"], "start": w.get("asr_start", w["start"]),
                 "end": w.get("asr_end", w["end"]), "p": w["p"]} for w in c["caption_words"]]
    monkeypatch.setattr(engine, "_strong_pass", strong_pass)
    monkeypatch.setattr(engine, "CFG", {**engine.CFG, "daily": {
        **(engine.CFG.get("daily") or {}), "cut_tidy": {"enabled": True}}})
    monkeypatch.setattr(engine.transcript, "load_audio", _load_audio(c))
    refine = c["replays"]["recut"]["refine"]
    asr_runs = []

    def words_window(m, lo, hi, model=None):
        asr_runs.append(("words_window", round(lo, 3), round(hi, 3)))
        clock["t"] += COST["refine"]
        ok = rerun and abs(lo - refine["lo"]) < 0.01 and abs(hi - refine["hi"]) < 0.01
        return [dict(w) for w in refine["words"]] if ok else []
    monkeypatch.setattr(engine.transcript, "words_window", words_window)
    passes, unheard = {}, []

    def transcribe(m, start, end, **kw):
        key = (round(float(start), 3), round(float(end), 3))
        if key in passes:
            return {**passes[key], "cached": True}
        rec = _asr2_pass(c, float(start), float(end), rerun)
        if rec is None:
            unheard.append(key)
            raise RuntimeError(f"no second ASR recorded for {key}")
        asr_runs.append(("second_asr",) + key)
        passes[key] = rec
        clock["t"] += COST["asr2_rtf"] * (key[1] - key[0] + rec["tail_s"])
        return rec
    monkeypatch.setattr(second_asr, "settings", lambda cfg=None: {
        "enabled": True, "model": "small", "tail_s": 2.0, "threads": 4,
        "cache_dir": str(tmp_path / "asr2")})
    monkeypatch.setattr(second_asr, "transcribe", transcribe)
    monkeypatch.setattr(second_asr, "cached", lambda m, start, end, **kw: (
        lambda r: None if r is None else {**r, "cached": True})(
        passes.get((round(float(start), 3), round(float(end), 3)))), raising=False)
    monkeypatch.setattr(quality_gate, "cut_energy", lambda m, s, e: _cut_energy(c, s, e))
    monkeypatch.setattr(quality_gate, "ending_seat_enabled", lambda cfg=None: seat)
    calls, seen = [], []
    render = _fake_recompose(calls)

    def run(cmd, **kw):
        clock["t"] += COST["render"]
        return render(cmd, **kw)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(engine, "_store_clip", fake_store_clip)

    def gate(out, clip, qc=None):
        s, e = float(clip["start_s"]), float(clip["end_s"])
        words = engine._load_words_sidecar(clip["words_path"])
        rec = second_asr.transcribe(str(media), s, e)
        rep = {"word_boundaries": quality_gate.measure_word_boundaries(words, s, e),
               "sentence_bounds": quality_gate.measure_sentence_bounds(words, s, e),
               "ending": quality_gate.ending_verdict(
                   rec["words"], quality_gate.clip_words(words, s, e), e - s,
                   tail_s=rec["tail_s"], energy=_cut_energy(c, s, e),
                   seat=quality_gate.ending_seat_enabled())}
        errs = [f"{k}: {x}" for k, m in rep.items() if m.get("passed") is False
                for x in m.get("errors") or []]
        seen.append(errs)
        return {**rep, "second_asr": {"model": "small", "tail_s": rec["tail_s"],
                                      "cached": rec["cached"]},
                "verdict": "fail" if errs else "pass", "drop_reason": "; ".join(errs)}
    monkeypatch.setattr(engine, "_run_quality_gate", gate)
    ok = engine.render(runstate.clip(clip_id))
    assert unheard == [], "a cut with no recorded second ASR was judged"
    return ok, runstate.clip(clip_id), seen, calls, asr_runs


def _asr2_end(row):
    return json.loads(row["features_json"])["captions"]["asr2_end"]


def _sidecar(row):
    from clipengine import engine
    return engine._load_words_sidecar(row["words_path"])


AMPLE = 10_000.0        # a window with the time for every pass
TIGHT = 322.0           # a window with the time for the caption step, the render and the gate's
                        # pass, not for a pulled-back cut's pass or a re-cut's new ASR


def test_the_end_goes_back_past_and_before_the_render(db, tmp_path, monkeypatch):
    """The gate's pass on the moved cut taken to be the case's (re-based): the tidy branch reads
    the 'and', ends the clip after 'mess.' and the first render passes."""
    ok, row, seen, calls, runs = _render_asr2("period", tmp_path, monkeypatch, left=AMPLE)
    assert ok is True and len(calls) == 1 and seen == [[]] and row["state"] == "rendered"
    assert (row["start_s"], row["end_s"]) == (16.1, 38.905)
    end = _asr2_end(row)
    assert end["word"] == "and" and end["pull_back"]["verified"] is True
    assert end["pull_back"]["last"] == "mess." and "recut" not in end
    # the gate's own pass on the old and the new cut, read back by the gate: no other ASR
    assert runs == [("second_asr", 16.1, 39.0), ("second_asr", 16.1, 38.905)]


def test_when_the_pulled_back_end_fails_its_pass_the_recut_lane_extends_it_before_the_render(
        db, tmp_path, monkeypatch):
    """The pass on 38.905 s hears 'mess,' and fails it; the re-cut lane's plan on its precise
    words ends after 'mattered most.' (44.415 s) and the gate's pass there passes it: the clip
    ships on its first render (the re-cut after the gate would need a second one)."""
    ok, row, seen, calls, runs = _render_asr2("period", tmp_path, monkeypatch, rerun=True,
                                              left=AMPLE)
    assert ok is True and len(calls) == 1 and seen == [[]] and row["state"] == "rendered"
    assert row["end_s"] == ASR2["period"]["replays"]["recut"]["cut"][1] == 44.415
    end = _asr2_end(row)
    assert end["pull_back"]["verified"] is False
    assert "continue after 'mess,'" in "; ".join(end["pull_back"]["errors"])
    assert end["recut"]["action"] == "extend" and end["recut"]["last"].endswith("mattered most.")
    burned = quality_gate.clip_words(_sidecar(row), row["start_s"], row["end_s"])
    assert burned[-1]["w"] == "most."
    assert [r[0] for r in runs] == ["second_asr", "second_asr", "words_window", "second_asr"]


def test_no_clean_end_is_left_for_the_gate(db, tmp_path, monkeypatch):
    """Nothing back past 'and' ends a sentence, and the re-cut lane's extension ends on 'And':
    the cut goes to the render as it is (no drop before it) and the gate judges it."""
    c = ASR2["no_period"]
    ok, row, seen, calls, _runs = _render_asr2("no_period", tmp_path, monkeypatch, rerun=True,
                                               left=AMPLE)
    assert ok is False and row["state"] == "failed"
    assert seen[0] == c["gate_errors"] and calls              # rendered at 22.60 s as it was
    end = _asr2_end(row)
    assert "ends mid-sentence: last word 'listen'" in end["pull_back"]["none"]
    assert "captions end on a filler/dangling word 'And'" in end["recut"]["none"]


def test_outside_a_daily_window_the_caption_step_does_not_read_the_second_asr(
        db, tmp_path, monkeypatch):
    """Manual / campaign renders (no window): the re-cut after the gate has no time check there,
    so the caption step leaves the ending to the gate as before; the clip re-cuts after its
    gate."""
    ok, row, seen, calls, runs = _render_asr2("period", tmp_path, monkeypatch, rerun=True)
    assert ok is True and len(calls) == 2 and seen == [ASR2["period"]["gate_errors"], []]
    assert "asr2_end" not in json.loads(row["features_json"])["captions"]
    assert row["end_s"] == 44.415 and runs[0] == ("second_asr", 16.1, 39.0)


def test_recut_skips_the_window_time_check_for_a_pull_back_on_the_gates_own_pass(
        db, tmp_path, monkeypatch, capsys):
    """On a tight budget the caption step reads the 'and' but has no time to hear the pulled-back
    cut (or a re-cut) before the render; the gate fails the ending, and a re-cut with new words
    does not fit the window. The end goes back past 'and' on the gate's own cached pass (no new
    ASR) and re-renders without the time check."""
    ok, row, seen, calls, runs = _render_asr2("period", tmp_path, monkeypatch, left=TIGHT)
    out = capsys.readouterr().out
    assert ok is True and len(calls) == 2 and seen == [ASR2["period"]["gate_errors"], []]
    assert row["end_s"] == 38.905 and json.loads(row["features_json"])["recut"]["action"] == \
        "pull_back"
    end = _asr2_end(row)
    assert end["pull_back"]["verified"] is None and "skipped" in end["recut"]
    assert "no re-cut (" not in out and "no new ASR, no time check" in out
    # the gate's own passes only (the old cut, then the re-render's): no precise pass
    assert runs == [("second_asr", 16.1, 39.0), ("second_asr", 16.1, 38.905)]


def test_on_a_tight_budget_a_pulled_back_end_that_fails_its_pass_still_drops_the_clip(
        db, tmp_path, monkeypatch):
    """The same with the pass on 38.905 s that hears 'mess,': the re-render fails too. No end back
    past 'and' saves it; only the re-cut lane's extension passes (above), which needs new ASR the
    window has no time for."""
    ok, row, seen, calls, runs = _render_asr2("period", tmp_path, monkeypatch, rerun=True,
                                              left=TIGHT)
    assert ok is False and len(calls) == 2 and seen[0] == ASR2["period"]["gate_errors"]
    assert "continue after 'mess,'" in seen[1][0]
    assert not [r for r in runs if r[0] == "words_window"]


@pytest.mark.parametrize("cid, waiting", [("no_period", 0), ("period", 1)])
def test_a_budget_short_recut_that_needs_asr_is_still_skipped(db, tmp_path, monkeypatch, capsys,
                                                              cid, waiting):
    """no_period on a tight budget: no end back past 'and' passes, so its re-cut needs the precise
    pass, and the window has no time for it. period with a clip of its window still to render:
    that clip comes first, no time check is skipped."""
    ok, row, seen, calls, runs = _render_asr2(cid, tmp_path, monkeypatch, rerun=True, left=TIGHT,
                                              waiting=waiting)
    out = capsys.readouterr().out
    assert ok is False and len(calls) == 1 and seen == [ASR2[cid]["gate_errors"]]
    assert f"clip {row['clip_id']}: no re-cut (" in out
    assert ("still to render come first" if waiting else "no pull-back: no end from") in out
    assert not [r for r in runs if r[0] == "words_window"]
    assert (row["start_s"], row["end_s"]) == (ASR2[cid]["start_s"], ASR2[cid]["end_s"])


def test_recut_never_repeats_a_pull_back_the_gates_pass_already_failed(db, tmp_path,
                                                                       monkeypatch):
    from clipengine import second_asr
    from clipengine import engine
    c = ASR2["period"]
    runstate.add_source(c["source_id"], local_path="w.mp4", content_type="auto",
                        vertical="general")
    feats = {"captions": {"asr2_end": {"pull_back": {"new_end": 38.905, "verified": False}}}}
    wp = tmp_path / "cap.json"
    wp.write_text(json.dumps(c["caption_words"]))
    cid = runstate.add_clip(c["source_id"], c["start_s"], c["end_s"], 0.7, "auto", hook="h",
                            words_path=str(wp), features_json=feats)
    monkeypatch.setattr(second_asr, "cached", lambda *a, **k: dict(c["asr2"]))
    monkeypatch.setattr(engine, "CFG", {**engine.CFG, "daily": {
        **(engine.CFG.get("daily") or {}), "cut_tidy": {"enabled": True}}})
    report = {**c["gate"], "verdict": "fail"}
    plan, why = engine._pull_back_after_gate(runstate.clip(cid), {"vertical": "general"}, "w.mp4",
                                             report, 0)
    assert plan is None and "failed the gate's checks before the render" in why
    runstate.update_clip(cid, features_json=json.dumps({}))
    plan, why = engine._pull_back_after_gate(runstate.clip(cid), {"vertical": "general"}, "w.mp4",
                                             report, 0)
    assert why is None and plan[0] == (c["start_s"], 38.905) and plan[2]["dropped"] == "and"


def test_second_asr_cached_reads_the_cache_and_never_runs_whisper(tmp_path, monkeypatch):
    from clipengine import second_asr
    media = tmp_path / "w.mp4"
    media.write_bytes(b"\0")
    monkeypatch.setattr(second_asr, "whisper_cli_words", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("whisper ran")))
    monkeypatch.setattr(second_asr, "settings", lambda cfg=None: {
        "enabled": True, "model": "small", "tail_s": 2.0, "threads": 4,
        "cache_dir": str(tmp_path / "asr2")})
    assert second_asr.cached(media, 1.0, 20.0, media_dur=60.0) is None
    cf, tail = second_asr._cache_file(media, 1.0, 20.0, "small", 2.0, 60.0, tmp_path / "asr2")
    cf.parent.mkdir(parents=True)
    cf.write_text(json.dumps({"words": [{"w": "so.", "start": 0.1, "end": 0.4}], "tail_s": tail}))
    rec = second_asr.cached(media, 1.0, 20.0, media_dur=60.0)
    assert rec["cached"] is True and rec["words"][0]["w"] == "so." and rec["tail_s"] == 2.0
    assert second_asr.transcribe(media, 1.0, 20.0, media_dur=60.0)["cached"] is True
    assert second_asr.cached(media, 1.0, 20.5, media_dur=60.0) is None
