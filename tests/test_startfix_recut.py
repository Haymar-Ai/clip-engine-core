"""A clip's start cut and the gate's words agree, and a clip the gate fails on its start (or on
both cuts) is re-cut once like a failed ending.

The strong caption model times the first word after a pause from the end of the word before it,
so a start cut placed in a pause on select's words (and on the audio) reads to the gate as a cut
inside that word. The synthetic cases below model the three shapes seen in practice: case 529
(the first word 'I' stretched back over a 0.36 s silence the cut sits in), case 531 (a cut one
frame before its pause) and case 533 (a cut in a 20 ms dip inside running speech, which must
still fail). Each holds the caption words, select's words, the gate's start errors and the audio's
10 ms frame RMS around the start."""
from __future__ import annotations

import importlib
import json
import pathlib
import subprocess

import numpy as np
import pytest

from clipengine import boundaries
from clipengine import quality_gate
from clipengine import runstate
from test_stage1_cuts import CFG, END_OF, WORDS, W, _fake_recompose, words_from
from helpers import codes_for, fake_store_clip

def _spread(text, t0, t1, gap=0.45, p=None):
    """Words of `text` spread evenly over [t0, t1], `gap` s of pause after each sentence end."""
    toks = text.split()
    n_gaps = sum(1 for t in toks[:-1] if t.endswith((".", "?", "!")))
    step = (t1 - t0 - gap * n_gaps) / len(toks)
    out, t = [], t0
    for tok in toks:
        w = {"w": tok, "start": round(t, 3), "end": round(t + step * 0.85, 3)}
        if p is not None:
            w["p"] = p
        out.append(w)
        t += step + (gap if tok.endswith((".", "?", "!")) else 0.0)
    return out


def _rms_db(t0, n, quiet, voiced_db=-20.0, quiet_db=-80.0, frame_s=0.01):
    """n 10 ms frames from t0 in dB: voiced except inside the (a, b) spans of `quiet`."""
    t = t0 + (np.arange(n) + 0.5) * frame_s
    db = np.full(n, voiced_db)
    for a, b in quiet:
        db[(t >= a) & (t < b)] = quiet_db
    return [round(float(x), 1) for x in db]


REST_529 = ("guess we could start over and build it again from scratch. That was the plan from day "
            "one. Nobody believed it would work, but we kept going anyway. Every single week we "
            "shipped something new. Some of it failed and some of it stuck. By the end of the "
            "year it was finally making money.")
_SEL_529 = [{"w": "I", "start": 3.28, "end": 3.5}] + _spread(REST_529, 3.5, 30.4)
_CAP_529 = [{"w": "I", "start": 2.995, "end": 3.475, "p": 0.17}] + [
    {**w, "start": round(w["start"] - 0.025, 3), "end": round(w["end"] - 0.025, 3), "p": 0.95}
    for w in _SEL_529[1:]]
_SEL_531 = [{"w": "So", "start": 5.0, "end": 5.1}] + _spread("we tried it twice and it broke.",
                                                             5.1, 7.0)
_CAP_531 = [{"w": "so", "start": 4.73, "end": 5.11}] + [
    {"w": w["w"].lower(), "start": w["start"] + 0.01, "end": w["end"] + 0.01} for w in _SEL_531[1:]]
_SEL_533 = [{"w": "Ah,", "start": 15.03, "end": 15.03}] + _spread("right. So that's the trick.",
                                                                 15.31, 17.0)
_CAP_533 = [{"w": "ah", "start": 14.645, "end": 14.985},
            {"w": "right", "start": 14.985, "end": 15.485}] + _spread("so that's the trick", 15.485,
                                                                      17.0)
LIVE = {
    "529": {"source_id": "Syn_w01", "window_s": 42.233, "start_s": 3.275, "end_s": 30.675,
            "refined": True, "select_words": _SEL_529[:8], "caption_words": _CAP_529[:8],
            "select_sidecar": _SEL_529, "caption_sidecar": _CAP_529,
            "gate_start_errors": ["start cut inside word 'I' (start=2.995 end=3.475)",
                                  "start mid-sentence: prev='I' first='guess'"],
            "t0": 1.275, "frame_s": 0.01, "rms_db": _rms_db(1.275, 400, [(3.06, 3.42)])},
    "531": {"start_s": 4.85, "end_s": 25.01, "select_words": _SEL_531, "caption_words": _CAP_531,
            "gate_start_errors": ["start cut inside word 'so' (start=4.730 end=5.110)",
                                  "start mid-sentence: prev='so' first='we'"],
            "t0": 2.85, "frame_s": 0.01, "rms_db": _rms_db(2.85, 400, [(4.86, 5.02)])},
    "533": {"start_s": 14.725, "end_s": 35.245, "select_words": _SEL_533,
            "caption_words": _CAP_533,
            "gate_start_errors": ["start cut inside word 'ah' (start=14.645 end=14.985)",
                                  "start mid-sentence: prev='ah' first='right'"],
            "t0": 12.725, "frame_s": 0.01, "rms_db": _rms_db(12.725, 400, [(14.72, 14.74)])},
}
TOL = quality_gate.WORD_BOUNDARY_INSIDE_TOL_S
# WORDS with a pause and two more sentences after "... in a single year."
LONG = WORDS + words_from("It was. Then we drove home and slept until noon the next day.",
                          WORDS[-1]["end"] + 0.8)


def _rms(c):
    return 10.0 ** (np.asarray(c["rms_db"], dtype=np.float64) / 20.0)


def _start_errors(words, start, end):
    return [e for m in (quality_gate.measure_word_boundaries(words, start, end),
                        quality_gate.measure_sentence_bounds(words, start, end))
            for e in m.get("errors") or [] if e.startswith("start")]


# ---------------------------------------------------------------- the three start shapes

def test_529s_words_reproduce_the_live_gate_failure():
    c = LIVE["529"]
    assert _start_errors(c["caption_words"], c["start_s"], c["end_s"]) == c["gate_start_errors"]
    # select's words, the ones the snap placed the cut on, had the cut in the pause before 'I'
    assert _start_errors(c["select_words"], c["start_s"], c["end_s"]) == []


def test_529_the_caption_I_is_seated_on_the_voice_and_the_start_passes():
    c = LIVE["529"]
    words, moved = boundaries.seat_words(c["caption_words"], c["select_words"], c["start_s"],
                                         _rms(c), c["t0"], tol=TOL)
    assert [m["w"] for m in moved] == ["I"]
    first = words[0]
    assert first["w"] == "I" and first["asr_start"] == 2.995
    assert 3.40 <= first["start"] <= 3.44 and first["end"] == 3.475    # the voice comes back at 3.42
    assert words[1:] == c["caption_words"][1:]                         # nothing else moved
    assert _start_errors(words, c["start_s"], c["end_s"]) == []


def test_531_a_cut_a_frame_before_its_pause_is_seated_too():
    c = LIVE["531"]
    assert _start_errors(c["caption_words"], c["start_s"], c["end_s"]) == c["gate_start_errors"]
    words, moved = boundaries.seat_words(c["caption_words"], c["select_words"], c["start_s"],
                                         _rms(c), c["t0"], tol=TOL)
    assert [m["w"] for m in moved] == ["so"] and 5.0 <= words[0]["start"] <= 5.05
    assert _start_errors(words, c["start_s"], c["end_s"]) == []


def test_533_a_cut_in_speech_still_fails():
    """533 cut at 14.725 s in a 20 ms dip of running speech: no pause, the words stay."""
    c = LIVE["533"]
    assert boundaries.pause_at(_rms(c), c["t0"], 0.01, c["start_s"], slack=TOL) is None
    words, moved = boundaries.seat_words(c["caption_words"], c["select_words"], c["start_s"],
                                         _rms(c), c["t0"], tol=TOL)
    assert moved == [] and words == c["caption_words"]
    assert _start_errors(words, c["start_s"], c["end_s"]) == c["gate_start_errors"]


def test_seat_needs_the_precise_words_to_place_the_word():
    c = LIVE["529"]
    for ref in ([], [dict(w, w="you") for w in c["select_words"]],      # no twin / another word
                [dict(w, start=3.2) for w in c["select_words"]]):      # the twin straddles the cut
        words, moved = boundaries.seat_words(c["caption_words"], ref, c["start_s"], _rms(c),
                                             c["t0"], tol=TOL)
        assert moved == [] and words == c["caption_words"]


def test_seat_without_audio_changes_nothing():
    c = LIVE["529"]
    assert boundaries.seat_words(c["caption_words"], c["select_words"], c["start_s"], None,
                                 c["t0"], tol=TOL) == (c["caption_words"], [])


def _frames(t0, spans, n, frame_s=0.01, loud=0.1, quiet=1e-4):
    """Frame RMS from t0: loud except inside the (a, b) quiet spans."""
    t = t0 + (np.arange(n) + 0.5) * frame_s
    r = np.full(n, loud)
    for a, b in spans:
        r[(t >= a) & (t < b)] = quiet
    return r


def test_seat_moves_a_last_word_stretched_over_the_end_pause_back():
    """The end mirror: the last word runs on over the pause after it to past the cut."""
    words = W(("that's", 10.0, 10.3), ("it.", 10.3, 11.1), ("And", 11.15, 11.4))
    ref = W(("that's", 10.0, 10.3), ("it.", 10.3, 10.6))
    rms = _frames(9.0, [(10.62, 11.12)], 300)
    out, moved = boundaries.seat_words(words, ref, 10.9, rms, 9.0, tol=TOL)
    assert moved and out[1]["end"] == pytest.approx(10.62, abs=0.011) and out[1]["asr_end"] == 11.1
    assert quality_gate.measure_word_boundaries(out, 9.9, 10.9)["end_in_word"] is False


def test_a_short_dip_is_not_a_pause():
    rms = _frames(9.0, [(10.0, 10.08)], 300)                     # an 80 ms stop closure
    assert boundaries.pause_at(rms, 9.0, 0.01, 10.04) is None
    assert boundaries.pause_at(_frames(9.0, [(10.0, 10.3)], 300), 9.0, 0.01, 10.04) == \
        pytest.approx((10.0, 10.3), abs=0.011)


# ---------------------------------------------------------------- boundaries.recut_start

def test_recut_start_goes_back_to_the_sentence_the_cut_opens_inside():
    start = END_OF["dad"] + 0.02                                 # "Then my dad | pulled over ..."
    new, info = boundaries.recut_start(WORDS, start, WORDS[-1]["end"] + 0.3, cfg=CFG)
    then = next(w for w in WORDS if w["w"] == "Then")
    assert info["start_action"] == "extend" and then["start"] - 0.6 <= new <= then["start"]
    assert info["first"].startswith("Then my dad")


def test_recut_start_moves_on_when_the_sentence_start_is_out_of_reach():
    start = END_OF["dad"] + 0.02
    new, info = boundaries.recut_start(WORDS, start, WORDS[-1]["end"] + 0.3, cfg=CFG,
                                       max_move=0.5)
    assert new is None                         # "Then" is 0.9 s back and nothing starts after
    start = next(w for w in LONG if w["w"] == "back")["start"] + 0.1   # "Then" is 6 s back
    new, info = boundaries.recut_start(LONG, start, LONG[-1]["end"] + 0.3,
                                       cfg={**CFG, "min_dur_s": 3}, max_move=3.0)
    nxt = next(w for w in LONG if w["w"] == "It")
    assert info["start_action"] == "trim" and nxt["start"] - 0.6 <= new <= nxt["start"]
    assert info["first"] == "It was."


def test_recut_start_never_returns_the_failed_start():
    then = next(w for w in WORDS if w["w"] == "Then")
    new, info = boundaries.recut_start(WORDS, then["start"] - 0.1, WORDS[-1]["end"] + 0.3,
                                       cfg=CFG)
    nobody = next(w for w in WORDS if w["w"] == "Nobody")
    assert info["start_action"] == "extend" and new <= nobody["start"]


def test_recut_start_refuses_when_no_clean_start_is_near():
    words = words_from("and so we kept going and going and going and going and going and going "
                       "and going and going and going", 10.0)
    new, why = boundaries.recut_start(words, 12.0, 30.0, cfg=CFG)
    assert new is None and "no other clean sentence start" in why
    assert boundaries.recut_start([], 1.0, 20.0, cfg=CFG)[0] is None


# ---------------------------------------------------------------- engine: which cuts re-cut

def _gate_fail(*errors, **extra):
    """A failed gate report with word_boundaries / sentence_bounds errors (the gate's strings)."""
    wb = [e for e in errors if "cut inside word" in e]
    sb = [e for e in errors if "mid-sentence" in e]
    rep = {"verdict": "fail", "drop_reason": "; ".join(errors)}
    if wb:
        rep["word_boundaries"] = {"passed": False, "errors": wb}
    if sb:
        rep["sentence_bounds"] = {"passed": False, "errors": sb}
    rep.update(extra)
    return rep


START_529 = _gate_fail(*LIVE["529"]["gate_start_errors"])
DARK = {"visual_quality": {"passed": False, "errors": ["too dark: median luma 30.9 < 35"]}}


def test_recut_sides_from_the_gate_codes():
    from clipengine import engine
    assert engine.recut_sides(START_529) == (["start"], None)
    both = _gate_fail(*LIVE["529"]["gate_start_errors"], "end cut inside word 'and' "
                      "(start=34.985 end=35.385)")
    assert engine.recut_sides(both) == (["start", "end"], None)
    sides, why = engine.recut_sides(_gate_fail(*LIVE["531"]["gate_start_errors"], **DARK))
    assert sides is None and "too_dark" in why                  # 531: a re-cut can't fix that
    assert engine.recut_sides({"verdict": "fail"})[0] is None


@pytest.fixture
def db(tmp_path, monkeypatch):
    runstate.reset()
    from clipengine import reason_codes
    from clipengine import engine
    monkeypatch.setattr(engine, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(engine, "vertical_cfg", lambda v: dict(CFG))
    monkeypatch.setattr(engine, "CAPTION_STRONG_MODEL", None)
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", None)
    runstate.init()
    return tmp_path


# a 529-shaped clip on WORDS: the cut opens inside "Then my dad | pulled over ...", its first
# caption word is stretched back over the cut (its middle a little before it)
START = END_OF["dad"] + 0.02
END = END_OF["single"] + 0.02


def _clip(tmp_path, start=START, end=END, side=None):
    media = tmp_path / "W_w00.mp4"
    media.write_bytes(b"\0")
    runstate.add_source("W_w00", local_path=str(media), content_type="auto", vertical="general")
    runstate.update_source("W_w00", state="selected", duration_s=60.0)
    if side is None:
        side = [dict(w) for w in WORDS if start <= (w["start"] + w["end"]) / 2 <= end]
        side[0] = {**side[0], "start": round(START - 0.12, 3)}       # 'pulled', stretched
    wp = tmp_path / "side.json"
    wp.write_text(json.dumps(side))
    cid = runstate.add_clip("W_w00", start, end, 0.7, "auto", hook="h", words_path=str(wp))
    return cid, media


def _heard(m, a, b):
    return [w for w in LONG if a <= w["start"] <= b]


def test_recut_plan_recuts_a_failed_start_and_keeps_the_clips_words(db, tmp_path):
    from clipengine import engine
    cid, media = _clip(tmp_path)
    passes = []

    def heard(m, a, b):
        passes.append((a, b))
        return _heard(m, a, b)
    cut, side, info = engine.recut_plan(runstate.clip(cid), str(media), 60.0, START_529, CFG,
                                        words_fn=heard, rms_fn=lambda *a: None)
    then = next(w for w in WORDS if w["w"] == "Then")
    assert info["sides"] == ["start"] and info["start_action"] == "extend"
    assert then["start"] - 0.6 <= cut[0] <= then["start"] and cut[1] == END
    assert len(passes) == 1 and passes[0][0] <= START - 8.0          # the pad before the start
    texts = [w["w"] for w in side]
    assert texts.count("pulled") == 1 and texts.count("dad") == 1     # no seam duplicate / loss
    assert texts[texts.index("Then"):texts.index("pulled") + 1] == ["Then", "my", "dad", "pulled"]
    assert texts[0] == WORDS[0]["w"]                                  # context before the start
    assert [e for e in _start_errors(side, cut[0], cut[1])] == []


def test_recut_plan_recuts_both_cuts_on_one_plan(db, tmp_path):
    from clipengine import engine
    cid, media = _clip(tmp_path)
    rep = _gate_fail(*LIVE["529"]["gate_start_errors"], "end mid-sentence: last='single' "
                     "next='year.'")
    passes = []
    cut, side, info = engine.recut_plan(runstate.clip(cid), str(media), 60.0, rep, CFG,
                                        words_fn=lambda m, a, b: passes.append((a, b)) or
                                        _heard(m, a, b), rms_fn=lambda *a: None)
    assert info["sides"] == ["start", "end"] and len(passes) == 2
    assert cut[0] < START and info["action"] == "extend" and cut[1] >= END_OF["year."]
    assert quality_gate.measure_sentence_bounds(side, *cut)["passed"] is True


def test_recut_plan_gives_up_when_a_failed_cut_has_nowhere_to_go(db, tmp_path):
    from clipengine import engine
    cid, media = _clip(tmp_path)
    cut, side, why = engine.recut_plan(runstate.clip(cid), str(media), 60.0, START_529, CFG,
                                       words_fn=lambda *a: [], rms_fn=lambda *a: None)
    assert cut is None and side is None and "no words around the start" in why


# ---------------------------------------------------------------- engine: render

def _no_audio(monkeypatch, engine):
    monkeypatch.setattr(engine.transcript, "words_window", lambda m, a, b, model=None:
                        _heard(m, a, b))
    monkeypatch.setattr(engine.transcript, "load_audio", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("no audio in tests")))
    monkeypatch.setattr(engine, "_store_clip", fake_store_clip)


def test_render_recuts_a_failed_start_and_ships_the_pass(db, tmp_path, monkeypatch):
    from clipengine import engine
    cid, _media = _clip(tmp_path)
    calls, gates = [], []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    _no_audio(monkeypatch, engine)

    def gate(out, clip, qc=None):
        gates.append(float(clip["start_s"]))
        return START_529 if len(gates) == 1 else {"verdict": "pass"}
    monkeypatch.setattr(engine, "_run_quality_gate", gate)
    assert engine.render(runstate.clip(cid)) is True
    row = runstate.clip(cid)
    assert row["state"] == "rendered" and len(calls) == 2
    assert gates[0] == pytest.approx(START) and row["start_s"] == gates[1] < START
    assert row["end_s"] == pytest.approx(END)
    assert float(calls[1][4]) == row["start_s"]                       # recompose got the new cut
    feats = json.loads(row["features_json"])
    assert feats["recut"]["sides"] == ["start"] and feats["gate_report"]["recut"]["new_start"]
    assert row["words_path"].endswith("_recut.json")


def test_render_rejects_when_the_start_recut_still_fails(db, tmp_path, monkeypatch):
    from clipengine import reason_codes
    from clipengine import engine
    cid, _media = _clip(tmp_path)
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    _no_audio(monkeypatch, engine)
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: START_529)
    assert engine.render(runstate.clip(cid)) is False
    assert len(calls) == 2                                            # re-rendered once, no more
    assert runstate.clip(cid)["state"] == "failed"
    assert set(codes_for(cid)) == {"cut_start"}


def test_render_does_not_recut_a_start_that_failed_with_more(db, tmp_path, monkeypatch):
    """531 also failed too_dark: a re-cut can't fix that, so it is not tried."""
    from clipengine import engine
    cid, _media = _clip(tmp_path)
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    _no_audio(monkeypatch, engine)
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: _gate_fail(
        *LIVE["531"]["gate_start_errors"], **DARK))
    assert engine.render(runstate.clip(cid)) is False and len(calls) == 1
    assert runstate.clip(cid)["start_s"] == pytest.approx(START)


def test_a_failed_recut_render_restores_both_cuts(db, tmp_path, monkeypatch):
    from clipengine import engine
    cid, _media = _clip(tmp_path)
    calls = []
    ok = _fake_recompose(calls)

    def run(cmd, **kw):
        if calls:
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ffmpeg died")
        return ok(cmd, **kw)
    monkeypatch.setattr(subprocess, "run", run)
    _no_audio(monkeypatch, engine)
    rep = _gate_fail(*LIVE["529"]["gate_start_errors"], "end mid-sentence: last='single' "
                     "next='year.'")
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: rep)
    assert engine.render(runstate.clip(cid)) is False and len(calls) == 2
    row = runstate.clip(cid)
    assert (row["start_s"], row["end_s"]) == (pytest.approx(START), pytest.approx(END))


def test_a_both_cuts_recut_needs_time_for_both_passes(db, tmp_path, monkeypatch):
    import time
    from clipengine import engine
    cid, _media = _clip(tmp_path)
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    _no_audio(monkeypatch, engine)
    rep = _gate_fail(*LIVE["529"]["gate_start_errors"], "end mid-sentence: last='single' "
                     "next='year.'")
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: rep)
    # room for one precise pass, not for two
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", time.monotonic() + engine.RECUT_ASR_S * 1.5)
    assert engine.render(runstate.clip(cid)) is False and len(calls) == 1


# ---------------------------------------------------------------- 529 end to end

def test_render_529_passes_the_gate_on_its_first_render(db, tmp_path, monkeypatch):
    """529 as it ran live: select's precise words, the strong pass's words, the window audio
    around the start. The gate's own word_boundaries / sentence_bounds judge the burned words."""
    from clipengine import engine
    c = LIVE["529"]
    media = tmp_path / f"{c['source_id']}.mp4"
    media.write_bytes(b"\0")
    runstate.add_source(c["source_id"], local_path=str(media), content_type="auto", vertical="podcast")
    runstate.update_source(c["source_id"], state="selected", duration_s=c["window_s"])
    wp = tmp_path / "select.json"
    wp.write_text(json.dumps(c["select_sidecar"]))
    cid = runstate.add_clip(c["source_id"], c["start_s"], c["end_s"], 0.58, "auto", hook="h",
                         words_path=str(wp), features_json={"refined": c["refined"]})
    monkeypatch.setattr(engine, "CAPTION_STRONG_MODEL", "large-v3-turbo")
    monkeypatch.setattr(engine.transcript, "WORKER", None)
    # the start-fix's own seat (engine._seat) as it ran: render.caption_context off (
    # seats every strong word on the audio first; test_cut_review covers that)
    monkeypatch.setattr(engine, "caption_context", lambda cfg=None: (0.5, 0.5, False))
    monkeypatch.setattr(engine, "_strong_pass", lambda m, lo, hi, timeout: [
        dict(w) for w in c["caption_sidecar"]])
    from clipengine import second_asr          # the clean-cut checks before the render read the words alone here
    monkeypatch.setattr(second_asr, "settings", lambda cfg=None: {"enabled": False})
    rms = _rms(c)

    def load_audio(m, lo, hi, sr=16000):
        """samples whose 10 ms frames have the live window's RMS (t0 = c['t0'])"""
        n = int(round((hi - lo) / 0.01))
        k = np.clip(np.floor((lo - c["t0"]) / 0.01 + np.arange(n) + 1e-6).astype(int), 0,
                    len(rms) - 1)
        return np.repeat(rms[k], sr // 100).astype(np.float32)
    monkeypatch.setattr(engine.transcript, "load_audio", load_audio)
    calls, seen = [], []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    monkeypatch.setattr(engine, "_store_clip", fake_store_clip)

    def gate(out, clip, qc=None):
        words = engine._load_words_sidecar(clip["words_path"])
        rep = {"word_boundaries": quality_gate.measure_word_boundaries(
                   words, clip["start_s"], clip["end_s"]),
               "sentence_bounds": quality_gate.measure_sentence_bounds(
                   words, clip["start_s"], clip["end_s"])}
        errs = [e for m in rep.values() for e in m["errors"]]
        seen.append(errs)
        return {**rep, "verdict": "fail" if errs else "pass", "drop_reason": "; ".join(errs)}
    monkeypatch.setattr(engine, "_run_quality_gate", gate)
    assert engine.render(runstate.clip(cid)) is True
    row = runstate.clip(cid)
    assert seen == [[]] and len(calls) == 1 and row["state"] == "rendered"
    assert (row["start_s"], row["end_s"]) == (c["start_s"], c["end_s"])   # the cut stands
    feats = json.loads(row["features_json"])
    seat = feats["captions"]["seat"]
    assert [(m["cut"], m["w"]) for m in seat] == [("start", "I")] and seat[0]["to"][0] >= 3.40
    burned = engine._load_words_sidecar(row["words_path"])
    assert [w["w"] for w in burned[:3]] == ["I", "guess", "we"]


def test_render_529_without_the_seat_fails_the_way_it_did_live(db, tmp_path, monkeypatch):
    """The same render with the audio unreadable: the strong words stay as heard and the gate
    fails 529's start with the live errors (then the start re-cut gets its one try)."""
    from clipengine import engine
    c = LIVE["529"]
    media = tmp_path / f"{c['source_id']}.mp4"
    media.write_bytes(b"\0")
    runstate.add_source(c["source_id"], local_path=str(media), content_type="auto", vertical="podcast")
    runstate.update_source(c["source_id"], state="selected", duration_s=c["window_s"])
    wp = tmp_path / "select.json"
    wp.write_text(json.dumps(c["select_sidecar"]))
    cid = runstate.add_clip(c["source_id"], c["start_s"], c["end_s"], 0.58, "auto", hook="h",
                         words_path=str(wp), features_json={"refined": c["refined"]})
    monkeypatch.setattr(engine, "CAPTION_STRONG_MODEL", "large-v3-turbo")
    monkeypatch.setattr(engine.transcript, "WORKER", None)
    # the start-fix's own seat (engine._seat) as it ran: render.caption_context off (
    # seats every strong word on the audio first; test_cut_review covers that)
    monkeypatch.setattr(engine, "caption_context", lambda cfg=None: (0.5, 0.5, False))
    monkeypatch.setattr(engine, "_strong_pass", lambda m, lo, hi, timeout: [
        dict(w) for w in c["caption_sidecar"]])
    monkeypatch.setattr(engine.transcript, "load_audio", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("no audio")))
    monkeypatch.setattr(engine.transcript, "words_window", lambda m, a, b, model=None: [])
    calls, seen = [], []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))

    def gate(out, clip, qc=None):
        words = engine._load_words_sidecar(clip["words_path"])
        rep = {"word_boundaries": quality_gate.measure_word_boundaries(
                   words, clip["start_s"], clip["end_s"]),
               "sentence_bounds": quality_gate.measure_sentence_bounds(
                   words, clip["start_s"], clip["end_s"])}
        errs = [e for m in rep.values() for e in m["errors"]]
        seen.append(errs)
        return {**rep, "verdict": "fail" if errs else "pass", "drop_reason": "; ".join(errs)}
    monkeypatch.setattr(engine, "_run_quality_gate", gate)
    assert engine.render(runstate.clip(cid)) is False
    assert seen == [c["gate_start_errors"]] and len(calls) == 1   # no precise words: no re-cut
    assert runstate.clip(cid)["state"] == "failed"
