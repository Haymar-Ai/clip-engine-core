"""clean cut points. Before the render the cuts snap to sentence boundaries of
the strong-model words (large-v3-turbo, heard EXTEND_MAX_S around the cut and seated on the window
audio, since turbo times a word after a pause from the end of the word before it): never open
mid-sentence or on a false start, never end on a conjunction / filler or mid-word. An end may move
out by up to 6 s or in by up to 4 s to finish a sentence (a start likewise); a clip with no clean
pair inside its limits is dropped before the render. The repair lane's retrim uses the same rule.
Shapes are modelled on the 10-21/22 cut reason_codes: 590 ("ends on a filler/conjunction 'and'"), 650
("ends mid-sentence ... '82,' ('but' at +0.06 s)"), 644 ("start cut inside word 'and' ... prev='and'
first='means'"), 755 ("still opens on its false start")."""
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
from test_stage1_cuts import _fake_recompose
from helpers import codes_for, fake_store_clip
from clipengine import config

CFG = {"min_dur_s": 3, "max_dur_s": 60, "lead_pad_s": 0.15, "tail_pad_s": 0.35,
       "strict_ends": True, "min_end_gap_s": 0.25, "hard_gap_s": 1.5, "sentence_gap_s": 0.7}


def words_from(text, t0, step=0.3, gaps=None):
    """Words of `text`, one every `step` s from t0; gaps {index: extra pause before that word}."""
    out, t = [], float(t0)
    for i, tok in enumerate(text.split()):
        t += (gaps or {}).get(i, 0.0)
        out.append({"w": tok, "start": round(t, 3), "end": round(t + step - 0.05, 3)})
        t += step
    return out


TEXT = ("We drove out to the desert that night. Nobody said a word the whole way there and "
        "nobody slept. Then my dad pulled over and told me the story of how he lost everything. "
        "He got it all back in a single year. It still amazes me.")
WORDS = words_from(TEXT, 10.0, gaps={8: 0.6, 19: 0.6, 34: 0.6, 43: 0.6})
AT = {}
for _w in WORDS:
    AT.setdefault(_w["w"], _w)


def _rms(words, t0=0.0, t1=40.0, frame_s=0.01):
    """10 ms frame RMS: voice inside every word, near silence between them."""
    n = int(round((t1 - t0) / frame_s))
    r = np.full(n, 1e-4, dtype=np.float32)
    for w in words:
        a, b = int((w["start"] - t0) / frame_s), int((w["end"] - t0) / frame_s)
        r[max(0, a):max(0, b)] = 0.1
    return r


RMS = _rms(WORDS)


# ---------------------------------------------------------------- seat + false starts

def test_a_word_stretched_back_over_a_pause_is_seated_on_the_voice():
    """529: 'I' claimed 2.995-3.475 s over 0.36 s of silence; the voice is back at 3.42 s."""
    words = [{"w": "fine.", "start": 2.5, "end": 2.99}, {"w": "I", "start": 2.995, "end": 3.475},
             {"w": "think", "start": 3.475, "end": 3.7}]
    voice = [{"start": 2.5, "end": 3.06}, {"start": 3.42, "end": 3.7}]
    seated, moved = boundaries.seat_onsets(words, _rms(voice, 0.0, 6.0), 0.0)
    assert seated[1]["start"] == pytest.approx(3.42, abs=0.011) and seated[1]["asr_start"] == 2.995
    assert [m["w"] for m in moved] == ["I"]
    assert seated[0] == words[0] and seated[2] == words[2]


def test_a_short_stop_inside_a_word_is_not_a_pause():
    words = [{"w": "apple", "start": 1.0, "end": 1.5}]
    voice = [{"start": 0.5, "end": 1.02}, {"start": 1.1, "end": 1.6}]      # an 80 ms closure
    assert boundaries.seat_onsets(words, _rms(voice, 0.0, 3.0), 0.0)[1] == []


def test_false_starts_are_skipped_to_the_restart():
    w = words_from("I- I think the the thing is we were we were going home.", 0.0,
                   gaps={1: 0.2, 4: 0.2, 9: 0.2})
    assert w[boundaries.false_start(w, 0, len(w) - 1)]["w"] == "I"
    assert boundaries.false_start(w, 0, len(w) - 1) == 1
    assert boundaries.false_start(w, 3, len(w) - 1) == 4             # "the the"
    assert boundaries.false_start(w, 7, len(w) - 1) == 9             # "we were we were"
    assert boundaries.false_start(w, 2, len(w) - 1) == 2             # "think": none
    tight = words_from("so so what", 0.0)                           # no gap before the restart
    tight[1]["start"] = tight[0]["end"]
    assert boundaries.false_start(tight, 0, 2) is None


# ---------------------------------------------------------------- snap_sentences

def _snap(start, end, **kw):
    return boundaries.snap_sentences(WORDS, start, end, cfg=CFG, edges_clean=(True, True),
                                     rms=RMS, t0=0.0, **kw)


def test_a_clean_cut_stays_exactly_where_it_is():
    start = AT["Nobody"]["start"] - 0.2
    end = AT["slept."]["end"] + 0.2
    r = _snap(start, end)
    assert (r["start"], r["end"]) == (start, end) and r["snap"] == {"start": "ok", "end": "ok"}


def test_an_end_on_a_conjunction_goes_on_to_finish_the_sentence():
    """590: the clip ended on 'and' (... the whole way there and | nobody slept.)."""
    start = AT["Nobody"]["start"] - 0.2
    end = AT["and"]["end"] + 0.02
    r = _snap(start, end)
    assert r["snap"]["end"] == "extend" and r["last"] == "slept."
    assert AT["slept."]["end"] <= r["end"] <= AT["Then"]["start"]
    assert r["start"] == start


def test_a_start_mid_sentence_goes_back_to_its_sentence_start():
    """644: opened on 'means' after 'and'."""
    start = AT["pulled"]["start"] - 0.05
    end = AT["everything."]["end"] + 0.2
    r = _snap(start, end)
    assert r["snap"]["start"] == "extend" and r["first"] == "Then"
    assert AT["nobody"]["end"] <= r["start"] <= AT["Then"]["start"]


def test_an_end_out_of_reach_is_trimmed_back_to_the_last_clean_end():
    """The sentence the cut ends inside runs > 6 s on; the cut comes back <= 4 s instead."""
    long = words_from("It was late. " + " ".join(["and we kept on driving"] * 6) + " home.", 0.0,
                      gaps={3: 0.6})
    end = long[12]["end"] + 0.02                        # 'driving' mid-sentence, 3 s after 'late.'
    r = boundaries.snap_sentences(long, 0.0, end, cfg={**CFG, "min_dur_s": 0.5},
                                  edges_clean=(True, True), rms=_rms(long, 0, 20), t0=0.0)
    assert r["snap"]["end"] == "trim" and r["last"] == "late."
    assert end - r["end"] <= boundaries.TRIM_MAX_S


def test_no_clean_end_within_the_limits_drops_the_clip():
    """650: the sentence goes on past +6 s and nothing ends cleanly within -4 s."""
    run_on = words_from(" ".join(["and then we went on and on"] * 8) + " forever.", 0.0)
    end = run_on[30]["end"] + 0.02
    r = boundaries.snap_sentences(run_on, 0.0, end, cfg={**CFG, "min_dur_s": 1},
                                  edges_clean=(True, True), rms=_rms(run_on, 0, 30), t0=0.0)
    assert "drop" in r and r["side"] == "end" and "+6/-4" in r["drop"]


def test_a_start_on_a_false_start_opens_on_the_restart():
    """755: 'still opens on its false start'."""
    w = words_from("Okay that's it. I- I think the plan worked out fine for all of us in the end.",
                   0.0, gaps={3: 0.8, 4: 0.25})
    r = boundaries.snap_sentences(w, w[3]["start"] - 0.1, w[-1]["end"] + 0.2,
                                  cfg={**CFG, "min_dur_s": 2}, edges_clean=(True, True),
                                  rms=_rms(w, 0, 10), t0=0.0)
    assert r["first"] == "I" and r["start"] > w[3]["end"] and r["start"] <= w[4]["start"]


def test_the_gate_reads_a_snapped_cut_as_clean():
    start = AT["pulled"]["start"] - 0.05                # mid-sentence ...
    end = WORDS[24]["end"] + 0.02                       # ... to "pulled over and | told me"
    r = _snap(start, end)
    assert (r["first"], r["last"]) == ("Then", "everything.")
    errs = [e for m in (quality_gate.measure_word_boundaries(WORDS, r["start"], r["end"]),
                        quality_gate.measure_sentence_bounds(WORDS, r["start"], r["end"]))
            for e in m.get("errors") or []]
    assert errs == []


# ---------------------------------------------------------------- engine.render

@pytest.fixture
def db(tmp_path, monkeypatch):
    runstate.reset()
    from clipengine import reason_codes
    from clipengine import engine
    monkeypatch.setattr(engine, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(engine, "CAPTION_STRONG_MODEL", "large-v3-turbo")
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", None)
    monkeypatch.setattr(engine, "vertical_cfg", lambda v: dict(CFG))
    monkeypatch.setattr(engine.transcript, "WORKER", None)
    monkeypatch.setattr(engine, "_store_clip", fake_store_clip)
    from clipengine import second_asr                     # the gate's cut checks before the render: the words alone
    monkeypatch.setattr(second_asr, "settings", lambda cfg=None: {"enabled": False})
    monkeypatch.setattr(engine, "cut_snap_enabled", lambda cfg=None: True)   # the clean-cut step on (config off)

    def load_audio(media, lo, hi, sr=16000):
        n = int(round((hi - lo) / 0.01))
        k = np.clip(np.floor(lo / 0.01 + np.arange(n)).astype(int), 0, len(RMS) - 1)
        return np.repeat(RMS[k], sr // 100).astype(np.float32)
    monkeypatch.setattr(engine.transcript, "load_audio", load_audio)
    runstate.init()
    return tmp_path


def _clip(tmp_path, start, end):
    media = tmp_path / "W_w00.mp4"
    media.write_bytes(b"\0")
    runstate.add_source("W_w00", local_path=str(media), content_type="auto", vertical="podcast")
    runstate.update_source("W_w00", state="selected", duration_s=60.0)
    side = [dict(w) for w in WORDS if start <= (w["start"] + w["end"]) / 2 <= end]
    wp = tmp_path / "side.json"
    wp.write_text(json.dumps(side))
    return runstate.add_clip("W_w00", start, end, 0.7, "auto", hook="h", words_path=str(wp),
                          boundary_type="sentence")


def test_render_finishes_the_sentence_before_the_render(db, tmp_path, monkeypatch):
    from clipengine import engine
    start, end = AT["Nobody"]["start"] - 0.2, AT["and"]["end"] + 0.02
    cid = _clip(tmp_path, start, end)
    heard, calls = [], []
    monkeypatch.setattr(engine, "_strong_pass", lambda m, lo, hi, t: heard.append((lo, hi)) or [
        dict(w, p=0.9) for w in WORDS if lo <= w["start"] <= hi])
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: {"verdict": "pass"})
    assert engine.render(runstate.clip(cid)) is True
    row = runstate.clip(cid)
    assert heard == [(pytest.approx(start - 6.5), pytest.approx(end + 6.5))]
    assert len(calls) == 1 and float(calls[0][4]) == row["start_s"] == start
    assert AT["slept."]["end"] <= row["end_s"] <= AT["Then"]["start"]
    assert float(calls[0][5]) == pytest.approx(row["end_s"] - row["start_s"])
    side = engine._load_words_sidecar(row["words_path"])
    assert quality_gate.measure_sentence_bounds(side, row["start_s"], row["end_s"])["passed"]
    assert any(w["start"] > row["end_s"] for w in side)                 # context for the gate
    feats = json.loads(row["features_json"])
    assert feats["captions"]["clean_cut"]["snap"]["end"] == "extend"


def test_render_drops_a_clip_with_no_clean_cut_before_rendering_it(db, tmp_path, monkeypatch):
    from clipengine import reason_codes
    from clipengine import engine
    start, end = AT["Nobody"]["start"] - 0.2, AT["and"]["end"] + 0.02
    cid = _clip(tmp_path, start, end)
    run_on = [dict(w, w=w["w"].rstrip(".")) for w in WORDS]          # one endless sentence
    monkeypatch.setattr(engine, "_strong_pass", lambda m, lo, hi, t: [
        dict(w, p=0.9) for w in run_on if lo <= w["start"] <= hi])
    calls, checked = [], []
    monkeypatch.setattr(engine, "gate_cut_errors", lambda side, a, b, m, lean: checked.append(
        (a, b, lean)) or [
        "ending: ends mid-sentence: the second ASR hears it continue after 'and'"])
    monkeypatch.setattr(engine, "recut_plan", lambda c, m, d, rep, vc: (
        None, None, "no clean sentence end within 8 s" if rep["ending"]["passed"] is False
        else "?"))
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: pytest.fail("no render"))
    assert engine.render(runstate.clip(cid)) is False
    row = runstate.clip(cid)
    assert checked == [(start, end, False)]   # the (full) gate's cut checks read it first
    assert calls == [] and row["state"] == "failed" and row["error"].startswith("gate: snap:")
    assert "the gate's cut checks fail the cut (ending: ends mid-sentence" in row["error"]
    assert "the re-cut finds no cut they pass (no clean sentence end within 8 s)" in row["error"]
    assert "cut_end" in codes_for(cid)
    assert json.loads(row["features_json"])["captions"]["clean_cut"]["drop"]


def test_no_clean_cut_but_the_gate_passes_the_cut_it_renders_as_it_is(db, tmp_path,
                                                                      monkeypatch):
    """The snap is stricter than the gate, which stays the judge: 10-21 595 (podcasts) passed the
    gate with no clean end in reach on the strong words. Dropping it would lose a clip in stock."""
    from clipengine import engine
    start, end = AT["Nobody"]["start"] - 0.2, AT["and"]["end"] + 0.02
    cid = _clip(tmp_path, start, end)
    run_on = [dict(w, w=w["w"].rstrip(".")) for w in WORDS]          # one endless sentence
    monkeypatch.setattr(engine, "_strong_pass", lambda m, lo, hi, t: [
        dict(w, p=0.9) for w in run_on if lo <= w["start"] <= hi])
    monkeypatch.setattr(engine, "gate_cut_errors", lambda side, a, b, m, lean: [])
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: {"verdict": "pass"})
    assert engine.render(runstate.clip(cid)) is True
    row = runstate.clip(cid)
    assert len(calls) == 1 and (row["start_s"], row["end_s"]) == (start, end)
    cc = json.loads(row["features_json"])["captions"]["clean_cut"]
    assert cc["snap"] == {"start": "kept", "end": "kept"} and "pass the cut" in cc["kept"]
    side = engine._load_words_sidecar(row["words_path"])       # as without the snap: the strong
    assert side and all(start - 0.1 <= (w["start"] + w["end"]) / 2 <= end + 0.1   # words inside
                        for w in side)                          # the cut, select's around it


def test_no_audio_no_snap_the_cut_stands(db, tmp_path, monkeypatch):
    from clipengine import engine
    monkeypatch.setattr(engine, "caption_context", lambda cfg=None: (0.5, 0.5, False))
    start, end = AT["Nobody"]["start"] - 0.2, AT["and"]["end"] + 0.02
    cid = _clip(tmp_path, start, end)
    monkeypatch.setattr(engine.transcript, "load_audio", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("no audio")))
    heard, calls = [], []
    monkeypatch.setattr(engine, "_strong_pass", lambda m, lo, hi, t: heard.append((lo, hi)) or [
        dict(w, p=0.9) for w in WORDS if lo <= w["start"] <= hi])
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: {"verdict": "pass"})
    assert engine.render(runstate.clip(cid)) is True
    row = runstate.clip(cid)
    assert (row["start_s"], row["end_s"]) == (start, end)
    assert heard == [(pytest.approx(start - 0.5), pytest.approx(end + 0.5))]


def test_the_lean_gates_cut_check_is_its_ending_alone(monkeypatch):
    """A stream / gaming clip's gate (the lean gate) reads no word_boundaries or sentence_bounds:
    the check before the render reads only what that gate reads (10-21 571, gaming: the turbo
    words run 'well' over the end cut, the lean gate passed it on its ending)."""
    from clipengine import second_asr
    from clipengine import engine
    words = [dict(w) for w in WORDS]
    start, end = AT["Nobody"]["start"] - 0.2, AT["and"]["end"] + 0.02
    monkeypatch.setattr(second_asr, "settings", lambda cfg=None: {"enabled": False})
    full = engine.gate_cut_errors(words, start, end, "m.mp4")
    assert any(e.startswith("sentence_bounds") for e in full)        # ends on 'and'
    assert engine.gate_cut_errors(words, start, end, "m.mp4", lean=True) == []


def test_no_clean_cut_the_recut_lanes_plan_runs_before_the_render(db, tmp_path, monkeypatch):
    """The cut fails the gate's cut checks: before dropping it, the re-cut lane's plan (the
    precise words) runs as it would after a failed gate; a cut the checks pass renders."""
    from clipengine import engine
    start, end = AT["Nobody"]["start"] - 0.2, AT["and"]["end"] + 0.02
    cid = _clip(tmp_path, start, end)
    run_on = [dict(w, w=w["w"].rstrip(".")) for w in WORDS]          # one endless sentence
    monkeypatch.setattr(engine, "_strong_pass", lambda m, lo, hi, t: [
        dict(w, p=0.9) for w in run_on if lo <= w["start"] <= hi])
    new_end = AT["slept."]["end"] + 0.05
    checks = iter([["ending: ends on a filler/conjunction 'and'"], []])
    monkeypatch.setattr(engine, "gate_cut_errors", lambda side, a, b, m, lean: next(checks))
    plans = []
    monkeypatch.setattr(engine, "recut_plan", lambda c, m, d, rep, vc: plans.append(rep) or (
        (start, new_end), [dict(w) for w in WORDS if w["end"] <= new_end + 1], {"action": "extend"}))
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: {"verdict": "pass"})
    assert engine.render(runstate.clip(cid)) is True
    row = runstate.clip(cid)
    assert plans[0] == {"ending": {"passed": False,
                                   "errors": ["ends on a filler/conjunction 'and'"]}}
    assert (row["start_s"], row["end_s"]) == (start, new_end) and len(calls) == 1
    assert json.loads(row["features_json"])["captions"]["clean_cut"]["recut"] == {"action": "extend"}


def test_a_cut_the_gate_passes_is_never_snapped(db, tmp_path, monkeypatch):
    """10-21 647 (tech, gate-passed): the snap would have moved its clean cut, and its hook no longer
    matched what the trimmed clip says. A cut the clip's own gate's cut checks pass stays."""
    from clipengine import engine
    start, end = AT["Nobody"]["start"] - 0.2, AT["and"]["end"] + 0.02
    cid = _clip(tmp_path, start, end)
    monkeypatch.setattr(engine, "_strong_pass", lambda m, lo, hi, t: [
        dict(w, p=0.9) for w in WORDS if lo <= w["start"] <= hi])
    monkeypatch.setattr(engine, "gate_cut_errors", lambda side, a, b, m, lean: [])
    monkeypatch.setattr(engine, "strong_snap", lambda *a, **k: pytest.fail("no snap"))
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: {"verdict": "pass"})
    assert engine.render(runstate.clip(cid)) is True
    row = runstate.clip(cid)
    assert (row["start_s"], row["end_s"]) == (start, end) and len(calls) == 1
    cc = json.loads(row["features_json"])["captions"]["clean_cut"]
    assert cc["snap"] == {"start": "kept", "end": "kept"}


# ---------------------------------------------------------------- the flag (daily.cut_snap)

def test_cut_snap_is_off_by_default_and_in_the_shipped_config():
    """The clean-cut step is off by default; renders run without it until the flag is on."""
    from clipengine import engine
    assert engine.cut_snap_enabled({}) is False
    assert engine.cut_snap_enabled({"daily": {"cut_snap": {}}}) is False
    assert engine.cut_snap_enabled({"daily": {"cut_snap": {"enabled": True}}}) is True
    shipped = config.load()
    assert engine.cut_snap_enabled(shipped) is False


def test_flag_off_the_render_is_e8be8bcs(db, tmp_path, monkeypatch):
    """With daily.cut_snap off, a cut the step would fix (it ends on 'and', mid-sentence) renders as it
    did before the step: the strong pass hears the cut +-0.5 s, the caption words inside the cut are its
    words (select's around it), and no cut check, snap, re-cut or drop runs before the render."""
    from clipengine import engine
    monkeypatch.setattr(engine, "cut_snap_enabled", lambda cfg=None: False)
    # e8be8bc's pass: render.caption_context off (hears 1.5 s before / 2 s after)
    monkeypatch.setattr(engine, "caption_context", lambda cfg=None: (0.5, 0.5, False))
    start, end = AT["Nobody"]["start"] - 0.2, AT["and"]["end"] + 0.02
    cid = _clip(tmp_path, start, end)
    select = engine._load_words_sidecar(runstate.clip(cid)["words_path"])
    heard = []
    monkeypatch.setattr(engine, "_strong_pass", lambda m, lo, hi, t: heard.append((lo, hi)) or [
        dict(w, w=w["w"].upper(), p=0.9) for w in WORDS if lo <= w["start"] <= hi])
    for name in ("gate_cut_errors", "strong_snap", "recut_plan", "_recut_before_render",
                 "_fix_cut_before_render", "_drop_unclean"):
        monkeypatch.setattr(engine, name, lambda *a, _n=name, **k: pytest.fail(f"{_n} ran"))
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: {"verdict": "pass"})
    assert engine.render(runstate.clip(cid)) is True
    row = runstate.clip(cid)
    assert heard == [(pytest.approx(start - 0.5), pytest.approx(end + 0.5))]   # the narrow pass
    assert (row["start_s"], row["end_s"]) == (start, end) and len(calls) == 1
    assert float(calls[0][4]) == start and float(calls[0][5]) == pytest.approx(end - start)
    side = engine._load_words_sidecar(row["words_path"])
    inside = [w["w"] for w in side if start <= (w["start"] + w["end"]) / 2 <= end]
    assert inside and all(w.isupper() for w in inside)                    # the strong words
    assert [w for w in side if not start <= (w["start"] + w["end"]) / 2 <= end] == [
        w for w in select if not start <= (w["start"] + w["end"]) / 2 <= end]   # select's
    caps = json.loads(row["features_json"])["captions"]
    assert caps["mode"] == "full" and "clean_cut" not in caps and "snap_skipped" not in caps
