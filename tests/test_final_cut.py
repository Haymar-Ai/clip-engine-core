"""the final cut (boundaries.final_cut_issues / final_cut_candidates,
engine._final_cut). Offline: the second ASR (gate_cut_errors) and the window audio are seams.
The words are modelled on real reject reasons (11-07..10): 'Norwegian.' (the tail of the
sentence before the cut), a first word cut off ('We'), 'skydance' + 'and' (an end inside a
sentence), 'Yeah. Right.' (a lone interjection after the point)."""
from __future__ import annotations

import importlib
import json

import pytest

from clipengine import boundaries
from clipengine import runstate


def W(text, t0, step=0.4, dur=0.3):
    out, t = [], float(t0)
    for tok in text.split(" "):
        if tok == "|":                      # a 0.8 s pause
            t += 0.8
            continue
        out.append({"w": tok, "start": round(t, 3), "end": round(t + dur, 3)})
        t += step
    return out


def at(ws, tok, k=0):
    return [w for w in ws if w["w"] == tok][k]


CFG = {"min_dur_s": 3, "max_dur_s": 60}


# ---------------------------------------------------------------- boundaries

def test_a_start_on_the_tail_of_the_sentence_before_is_a_fragment():
    ws = W("You are not even | Norwegian. So I went to the store and bought milk. It was great.", 10)
    s, e = at(ws, "Norwegian.")["start"] - 0.1, at(ws, "great.")["end"] + 0.2
    iss = boundaries.final_cut_issues(ws, s, e)
    assert "opens mid-sentence" in iss["start"] and "end" not in iss
    (s2, e2, how), *_ = boundaries.final_cut_candidates(ws, s, e, iss, cfg=CFG)
    assert how == {"start": "on", "end": "kept", "first": "So", "last": "great."}
    assert at(ws, "Norwegian.")["end"] < s2 < at(ws, "So")["start"] and e2 == e


def test_a_first_word_cut_off_goes_back_to_it():
    ws = W("Okay. We have different rules here. They work every time.", 5)
    s, e = at(ws, "We")["end"] + 0.02, at(ws, "time.")["end"] + 0.2
    iss = boundaries.final_cut_issues(ws, s, e)
    assert "'We' before 'have'" in iss["start"]
    (s2, _e2, how), *_ = boundaries.final_cut_candidates(ws, s, e, iss, cfg=CFG)
    assert how["start"] == "back" and how["first"] == "We"
    assert at(ws, "Okay.")["end"] < s2 < at(ws, "We")["start"]


def test_an_end_inside_a_sentence_finishes_it_when_it_ends_in_the_tail_else_goes_back():
    ws = W("We talked about it. The deal went to skydance and they agreed.", 0)
    s, e = -0.1, at(ws, "skydance")["end"] + 0.05
    iss = boundaries.final_cut_issues(ws, s, e)
    assert iss == {"end": "the next word 'and' starts before the cut"} or \
        iss == {"end": "ends mid-sentence on 'skydance'"}
    cands = boundaries.final_cut_candidates(ws, s, e, iss, cfg=CFG)
    assert cands[0][2]["end"] == "out" and cands[0][2]["last"] == "agreed."
    # the end of the sentence is past the heard tail: back to the last complete one
    cands = boundaries.final_cut_candidates(ws, s, e, iss, cfg={"min_dur_s": 1}, tail_s=0.5)
    assert cands[0][2] == {"start": "kept", "end": "back", "first": "We", "last": "it."}


def test_a_lone_interjection_after_the_point_is_cut_off():
    ws = W("He shot the ball from deep. It went straight through. Yeah. Right.", 5)
    s, e = 4.9, ws[-1]["end"] + 0.2
    iss = boundaries.final_cut_issues(ws, s, e)
    assert iss["end"] == "ends on 'Right.'"
    (s2, e2, how), *_ = boundaries.final_cut_candidates(ws, s, e, iss, cfg=CFG)
    assert how["last"] == "through." and at(ws, "through.")["end"] < e2 < at(ws, "Yeah.")["start"]


def test_a_lone_interjection_or_a_false_start_never_opens_a_clip():
    ws = W("Yes. Here is a fun one about cats. They sleep all day.", 1)
    assert "lone interjection" in boundaries.final_cut_issues(ws, 0.9, ws[-1]["end"] + 0.2)["start"]
    ws = W("I- I think the plan works. It really does work.", 1)
    assert "false start" in boundaries.final_cut_issues(ws, 0.9, ws[-1]["end"] + 0.2)["start"]


def test_a_clean_cut_has_no_issue_and_no_candidate_moves_a_clean_side():
    ws = W("This is the start. Then the middle part. And the end of it.", 1)
    s, e = 0.9, at(ws, "it.")["end"] + 0.2
    assert boundaries.final_cut_issues(ws, s, e) == {}
    ws2 = W("This is the start. Then the middle part. And the end of", 1)
    iss = boundaries.final_cut_issues(ws2 + W("it today.", 6.2), s, at(ws2, "of")["end"] + 0.05)
    assert set(iss) == {"end"}
    for s2, _e2, how in boundaries.final_cut_candidates(
            ws2 + W("it today.", 6.2), s, at(ws2, "of")["end"] + 0.05, iss, cfg=CFG):
        assert s2 == s and how["start"] == "kept"


def test_a_move_stays_inside_the_words_the_clip_burns_and_its_length_limits():
    ws = W("Okay. We have different rules here. They work every time.", 5)
    s, e = at(ws, "We")["end"] + 0.02, at(ws, "time.")["end"] + 0.2
    iss = boundaries.final_cut_issues(ws, s, e)
    cover = (at(ws, "have")["start"], ws[-1]["end"])      # 'We' was never burned
    for _s2, _e2, how in boundaries.final_cut_candidates(ws, s, e, iss, cfg=CFG, cover=cover):
        assert how["first"] != "We"
    assert boundaries.final_cut_candidates(ws, s, e, iss, cfg={"min_dur_s": 30, "max_dur_s": 60}) == []


# ---------------------------------------------------------------- engine._final_cut

WINDOW = (W("You are not even | Norwegian.", 0.0)
          + W("So I went down to the store on Main Street and bought a lot of milk.", 3.2)
          + W("It was the best milk I have ever had in my whole life.", 9.8)
          + W("Then we left the store.", 16.0))


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("FINAL_CUT", "1")
    runstate.reset()
    from clipengine import engine
    monkeypatch.setattr(engine, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", None)
    monkeypatch.setattr(engine, "_niche_of", lambda sid: "podcasts")
    return tmp_path


def _clip(tmp_path, start, end, sid="Pod_w01", words=WINDOW, vertical="podcast"):
    media = tmp_path / f"{sid}.mp4"
    media.write_bytes(b"\0")
    runstate.add_source(sid, local_path=str(media), vertical=vertical)
    wp = tmp_path / f"{sid}_side.json"
    wp.write_text(json.dumps(words))
    cid = runstate.add_clip(sid, start, end, 0.6, "auto", hook="Best milk ever", words_path=str(wp),
                         features_json={"refined": True}, boundary_type="sentence")
    return runstate.clip(cid), str(media)


class Gate:
    """gate_cut_errors' seam: errors per cut (None = pass), a fake second-ASR record."""

    def __init__(self, fail=()):
        self.fail, self.calls = list(fail), []

    def __call__(self, side, start, end, media, lean=False, seen=None):
        self.calls.append((round(start, 3), round(end, 3)))
        if seen is not None:
            seen["rec"] = {"words": [], "tail_s": 2.0}
        return [f for (a, b), f in self.fail if abs(a - start) < 1e-3 and abs(b - end) < 1e-3]


def _run(clip, media, gate):
    from clipengine import engine
    return engine._final_cut(clip, {"vertical": "podcast"}, media, 30.0,
                             words_fn=lambda m: [], rms_fn=lambda m, d: None, errs_fn=gate)


def test_the_final_cut_moves_a_fragment_start_on_and_records_it(db, monkeypatch):
    from clipengine import engine
    vc = engine.vertical_cfg
    monkeypatch.setattr(engine, "vertical_cfg", lambda v: {**vc(v), "min_dur_s": 5})
    s, e = at(WINDOW, "Norwegian.")["start"] - 0.1, at(WINDOW, "life.")["end"] + 0.3
    clip, media = _clip(db, s, e)
    gate = Gate()
    new = _run(clip, media, gate)
    assert at(WINDOW, "Norwegian.")["end"] < new["start_s"] < at(WINDOW, "So")["start"]
    assert new["end_s"] == pytest.approx(e)
    row = runstate.clip(clip["clip_id"])
    f = json.loads(row["features_json"])["final_cut"]
    assert row["start_s"] == pytest.approx(new["start_s"]) and f["how"]["first"] == "So"
    assert "opens mid-sentence" in f["issues"]["start"]
    assert len(gate.calls) == 2            # the cut as it was, then the gate's pass on the new one


def test_a_clean_cut_stays_and_costs_one_pass(db):
    s, e = at(WINDOW, "So")["start"] - 0.1, at(WINDOW, "life.")["end"] + 0.3
    clip, media = _clip(db, s, e)
    gate = Gate()
    new = _run(clip, media, gate)
    assert (new["start_s"], new["end_s"]) == (s, e) and len(gate.calls) == 1
    assert json.loads(runstate.clip(clip["clip_id"])["features_json"])["final_cut"]["kept"] == "clean"


def test_a_cut_the_gate_fails_with_no_clean_edge_goes_on_to_the_gate_and_its_repairs(db):
    """The final cut never drops a clip: the render, the gate and its repairs (the editor's re-cut
    with the gate's errors) judge it as on HEAD (11-11 replay 3100002: dropped before its render,
    while the first frozen run's cut on other burned words passed the eye-check)."""
    s, e = at(WINDOW, "So")["start"] - 0.1, at(WINDOW, "life.")["end"] + 0.3
    clip, media = _clip(db, s, e)

    class All(Gate):
        def __call__(self, side, start, end, media, lean=False, seen=None):
            super().__call__(side, start, end, media, lean, seen)
            return ["ending: ends mid-sentence"]
    new = _run(clip, media, All())
    assert (new["start_s"], new["end_s"]) == (s, e)
    f = json.loads(runstate.clip(clip["clip_id"])["features_json"])["final_cut"]
    assert f["kept"] == "no clean edge the gate's checks pass" and f["gate"]


def test_a_cut_the_gate_passes_stays_when_no_clean_edge_fits(db):
    ws = W("You are not even | Norwegian. So I went down to the store on Main Street and bought milk.", 0)
    s, e = at(ws, "Norwegian.")["start"] - 0.1, ws[-1]["end"] + 0.3
    clip, media = _clip(db, s, e, words=ws)
    from clipengine import engine
    new = engine._final_cut(clip, {"vertical": "podcast"}, media, 30.0, words_fn=lambda m: [],
                            rms_fn=lambda m, d: None, errs_fn=Gate())
    # 'So ... milk.' alone is under the podcast vertical's 15 s: no candidate, the gate passes it
    assert (new["start_s"], new["end_s"]) == (s, e)
    f = json.loads(runstate.clip(clip["clip_id"])["features_json"])["final_cut"]
    assert f["kept"] == "no clean edge" and f["candidates"] == 0


def test_off_by_switch(db, monkeypatch):
    monkeypatch.setenv("FINAL_CUT", "0")
    s, e = at(WINDOW, "Norwegian.")["start"] - 0.1, at(WINDOW, "life.")["end"] + 0.3
    clip, media = _clip(db, s, e)
    gate = Gate()
    assert _run(clip, media, gate) is clip and gate.calls == []


# ---------------------------------------------------------------- the next candidate

def test_propose_keeps_the_dropped_overlapping_picks_as_the_kept_picks_alternates(monkeypatch):
    from clipengine import select_semantic as ss
    sents = [{"i": i, "start": i * 5.0, "end": i * 5.0 + 4.5, "text": f"Sentence {i}."}
             for i in range(10)]
    monkeypatch.setattr(ss, "_llm", lambda *a, **k: json.dumps({"clips": [
        {"s0": 1, "s1": 5, "hook": "best", "score": 90},
        {"s0": 2, "s1": 6, "hook": "second", "score": 80},     # overlaps the best: its alternate
        {"s0": 8, "s1": 9, "hook": "other", "score": 70}]}))
    picks = ss.propose(sents, "podcast", {"semantic_backend": "ollama", "semantic_model": "x"})
    assert [p["hook"] for p in picks] == ["best", "other"]
    assert [a["hook"] for a in picks[0]["alternates"]] == ["second"]
    assert "alternates" not in picks[1]


def test_a_clip_with_no_clean_cut_takes_its_next_candidate(db):
    from clipengine import engine
    s, e = at(WINDOW, "So")["start"] - 0.1, at(WINDOW, "life.")["end"] + 0.3
    clip, media = _clip(db, s, e)
    alt_words = db / "alt.json"
    alt_words.write_text(json.dumps(WINDOW))
    feats = {"refined": True, "cut_review": [{"verdict": "move"}],
             "alternates": [{"start": 3.1, "end": 16.0, "hook": "Second", "score": 80,
                             "words_path": str(alt_words)}]}
    runstate.update_clip(clip["clip_id"], features_json=json.dumps(feats))
    clip = runstate.clip(clip["clip_id"])
    new = engine._next_alternate(clip, "final cut: no clean edge")
    assert (new["start_s"], new["end_s"], new["hook"]) == (3.1, 16.0, "Second")
    row = runstate.clip(clip["clip_id"])
    f = json.loads(row["features_json"])
    assert row["start_s"] == 3.1 and row["words_path"] == str(alt_words)
    assert f["replaced"][0]["from"] == [round(s, 3), round(e, 3)] and "cut_review" not in f
    assert f["alternates"] == [] and engine._next_alternate(runstate.clip(clip["clip_id"]), "x") is None


def test_an_alternate_over_another_clip_of_the_window_is_skipped(db):
    from clipengine import engine
    s, e = at(WINDOW, "So")["start"] - 0.1, at(WINDOW, "life.")["end"] + 0.3
    clip, media = _clip(db, s, e)
    runstate.add_clip("Pod_w01", 20.0, 40.0, 0.5, "auto", hook="Other", words_path=None,
                   features_json={}, boundary_type="sentence")
    feats = {"alternates": [{"start": 18.0, "end": 38.0, "hook": "Clash", "score": 80}]}
    runstate.update_clip(clip["clip_id"], features_json=json.dumps(feats))
    assert engine._next_alternate(runstate.clip(clip["clip_id"]), "x") is None


# ---------------------------------------------------------------- the caption check

def test_caption_confidence_reads_the_gates_check_and_the_tracks_timing():
    from clipengine import engine
    side = W("so I went down to the store and bought some milk today", 10.0)
    asr = [{**w, "start": w["start"] - 10.0, "end": w["end"] - 10.0} for w in side]
    score, problems, v = engine.caption_confidence(side, 9.9, 15.0, asr)
    assert problems == [] and score == 1.0 and v["passed"] is True
    crammed = [{"w": w["w"], "start": 10.0 + i * 0.05, "end": 10.0 + i * 0.05}
               for i, w in enumerate(side)]
    score, problems, _v = engine.caption_confidence(crammed, 9.9, 15.0, asr)
    assert any("inside one second" in p for p in problems)
    assert any("last no time" in p for p in problems) and score < 1.0
    gap = side[:2]                                                    # the rest never burned
    score, problems, _v = engine.caption_confidence(gap, 9.9, 15.0, asr)
    assert any("in a row have no caption" in p for p in problems)


def test_a_low_confidence_track_is_reheard_and_the_better_one_burned(db, monkeypatch):
    from clipengine import engine
    monkeypatch.setenv("CAPTION_CHECK", "1")
    talk = W("so I went down to the store and bought some milk and then I drove all the way home "
             "to see my mom.", 3.2)
    s, e = 3.1, talk[-1]["end"] + 0.3
    clip, media = _clip(db, s, e, words=talk[:3])            # the strong pass kept 3 words
    asr = [{**w, "start": w["start"] - s, "end": w["end"] - s} for w in talk]
    seen = []
    new = engine._caption_check(
        clip, {"vertical": "podcast"}, media, 30.0,
        strong_fn=lambda m, lo, hi, t: seen.append((lo, hi)) or [dict(w) for w in talk],
        rec_fn=lambda m, a, b: {"words": asr, "tail_s": 2.0},
        vote_fn=lambda sd: (None, {"reason": "nothing to vote"}), rms_fn=lambda m, lo, hi: None)
    assert seen and seen[0][0] == 0.0
    burned = [w for w in json.loads(open(new["words_path"]).read()) if w.get("cap", True) is not False]
    assert len(burned) == len(talk) and (new["start_s"], new["end_s"]) == (s, e)
    f = json.loads(runstate.clip(clip["clip_id"])["features_json"])["caption_check"]
    assert f["chose"].startswith("the window re-heard") and f["new_score"] > f["score"]


def test_a_clean_track_is_kept_without_a_new_pass(db, monkeypatch):
    from clipengine import engine
    monkeypatch.setenv("CAPTION_CHECK", "1")
    talk = W("so I went down to the store and bought some milk today.", 3.2)
    s, e = 3.1, talk[-1]["end"] + 0.3
    clip, media = _clip(db, s, e, words=talk)
    asr = [{**w, "start": w["start"] - s, "end": w["end"] - s} for w in talk]
    new = engine._caption_check(clip, {"vertical": "podcast"}, media, 30.0,
                                strong_fn=lambda *a: pytest.fail("no re-hearing"),
                                rec_fn=lambda m, a, b: {"words": asr, "tail_s": 2.0})
    assert new["words_path"] == clip["words_path"]
    assert json.loads(runstate.clip(clip["clip_id"])["features_json"])["caption_check"]["kept"] == "clean"


def test_the_vocab_prompt_is_the_sources_title():
    from clipengine import engine
    assert engine.vocab_prompt({"title": "Jane Doe | The Late Talk w/ Sam Rivers #692"}) == \
        "Jane Doe The Late Talk w/ Sam Rivers #692."
    # curly quotes straightened (a curly ' came back inside words), emoji out, '..' -> '.'
    assert engine.vocab_prompt({"title": "This is EXACTLY what I\u2019ve been saying.. \U0001F525"}) == \
        "This is EXACTLY what I've been saying."
    assert engine.vocab_prompt({"title": "North Pier Podcast Clips | watch?v=abcDEF12345"}) == \
        "North Pier Podcast Clips."
    assert engine.vocab_prompt({}) is None


def test_an_unpunctuated_track_reads_its_sentence_ends_from_the_other_transcripts():
    """11-11 replay 3000005: large-v3-turbo gave 'listening you' + 2 s + 'know exactly what', no
    . ? !; the pause read as a sentence end burned a dangling 'you'."""
    ws = W("i never explain myself to somebody who has no intention of listening you", 30.0)
    ws += W("know exactly what these people want from me", ws[-1]["end"] + 2.0)
    s, e = 29.9, at(ws, "me")["end"] + 0.3
    plain = boundaries.final_cut_issues(ws, s, at(ws, "people")["end"] + 0.05)
    assert "end" in plain
    old = boundaries.final_cut_candidates(ws, s, at(ws, "people")["end"] + 0.05, plain,
                                          cfg={"min_dur_s": 1})
    assert any(h["last"] == "you" for _a, _b, h in old)           # the pause alone closed 'you'
    new = boundaries.final_cut_candidates(ws, s, at(ws, "people")["end"] + 0.05, plain,
                                          cfg={"min_dur_s": 1}, closes_at=[])
    assert not any(h["last"] == "you" for _a, _b, h in new)
    # the second ASR punctuates 'listening.' and 'me.': both close
    marks = boundaries.mark_closes(ws, [at(ws, "listening")["end"] + 0.1, at(ws, "me")["end"]])
    closed = [w["w"] for w in marks if w["_close"]]
    assert closed == ["listening", "me"]


def test_the_strong_pass_gets_the_titles_prompt(db, monkeypatch):
    from clipengine import engine
    monkeypatch.setenv("CAPTION_PROMPT", "1")
    talk = W("so I went down to the store and bought some milk today.", 3.2)
    clip, media = _clip(db, 3.1, talk[-1]["end"] + 0.3, words=talk)
    runstate.update_source("Pod_w01", title="Joe Rogan Experience #2579 - Gary Sinise")
    seen = []

    def real(m, lo, hi, timeout, prompt=None):
        seen.append(prompt)
        return [dict(w) for w in talk]
    monkeypatch.setattr(engine, "_strong_pass", real)
    monkeypatch.setattr(engine, "_STRONG_PASS", real)
    monkeypatch.setattr(engine, "CAPTION_STRONG_MODEL", "large-v3-turbo")
    side, info = engine.strong_captions(runstate.clip(clip["clip_id"]), media, media_dur=30.0)
    assert seen == ["Joe Rogan Experience #2579 - Gary Sinise."]
    assert info["prompt"] == seen[0]
    monkeypatch.setenv("CAPTION_PROMPT", "0")
    seen.clear()
    engine.strong_captions(runstate.clip(clip["clip_id"]), media, media_dur=30.0)
    assert seen == [None]


def test_a_clip_the_gate_fails_on_its_cuts_alone_is_rendered_again_on_its_next_candidate(db,
                                                                                         monkeypatch):
    import subprocess
    import pathlib
    from clipengine import engine
    monkeypatch.setenv("FINAL_CUT", "0")
    monkeypatch.setenv("CAPTION_CHECK", "0")
    s, e = at(WINDOW, "So")["start"] - 0.1, at(WINDOW, "life.")["end"] + 0.3
    clip, media = _clip(db, s, e)
    runstate.update_source("Pod_w01", state="selected", duration_s=30.0)
    alt_words = db / "alt.json"
    alt_words.write_text(json.dumps(WINDOW))
    runstate.update_clip(clip["clip_id"], features_json=json.dumps(
        {"alternates": [{"start": 9.7, "end": 20.0, "hook": "Second", "score": 80,
                         "words_path": str(alt_words)}]}))
    monkeypatch.setattr(engine, "_cut_review", lambda c, *a, **k: (c, None))
    monkeypatch.setattr(engine, "_seat_cuts", lambda c, *a, **k: c)
    monkeypatch.setattr(engine, "_caption_step", lambda c, m, d=None: c)
    monkeypatch.setattr(engine, "_review_recut", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_recut", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_recaption", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_store_clip", lambda f, clip: str(f))

    def recompose(cmd, run=None, deadline=None, **kw):
        pathlib.Path(cmd[6]).write_bytes(b"fake mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout='QC {"layout": "track"}\n', stderr="")
    monkeypatch.setattr(engine, "_run_recompose", recompose)
    gates = []

    def gate(o, c, qc=None):
        gates.append((c["start_s"], c["end_s"]))
        if len(gates) == 1:
            return {"verdict": "fail", "drop_reason": "ending: ends mid-sentence",
                    "ending": {"passed": False, "errors": ["ends mid-sentence: last word 'x'"]}}
        return {"verdict": "pass"}
    monkeypatch.setattr(engine, "_run_quality_gate", gate)
    assert engine.render(runstate.clip(clip["clip_id"])) is True
    row = runstate.clip(clip["clip_id"])
    assert (row["start_s"], row["end_s"], row["hook"], row["state"]) == (9.7, 20.0, "Second", "rendered")
    assert gates == [(s, e), (9.7, 20.0)]
    f = json.loads(row["features_json"])
    assert f["replaced"][0]["why"].startswith("the gate: ending")
