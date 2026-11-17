"""the editor's pass over a clip's cut (clipengine/cut_review.py), its place before the
render and after a gate reject (engine), the repair lane's use of it, the strong caption pass's
context and seat. Offline: the LLM, whisper, recompose and the gate are seams. Transcripts are
modelled on the 11-07..09 reason_codes: 2756 opens on "How so?" (a reply to a remark before the cut),
2634 on "Just kidding, never selling that car" (the punchline of a setup before it), 2652 is a
chat-peak pick with no title."""
from __future__ import annotations

import importlib
import json
import pathlib
import subprocess
import time

import numpy as np
import pytest

from clipengine import runstate


def W(text, t0, step=0.3):
    return [{"w": w, "start": round(t0 + i * step, 3), "end": round(t0 + i * step + 0.25, 3)}
            for i, w in enumerate(text.split())]


# a window: the setup (S1-S2), the reply that needs it (S3), the story (S4-S6), a new topic (S7)
WINDOW = (W("We were at the border last week.", 1.0)
          + W("He said Sneako is the worst streamer he has ever met.", 4.0)
          + W("How so?", 8.0)
          + W("He looked like he got over the border with no papers at all.", 9.5)
          + W("Then he started yelling at the guards for free food.", 14.0)
          + W("So they kicked him out and banned him for life.", 18.0)
          + W("Anyway what game are we playing tonight?", 22.0))
CUT = (7.85, 21.0)           # select's cut: opens on "How so?", ends after "...for life."


def _sents(words):
    from clipengine import cut_review
    return cut_review.lines(words, *CUT)


def answer(**kw):
    a = {"start": 2, "end": 5, "start_fault": "none", "end_fault": "none", "title_ok": True,
         "title": "Old title"}
    a.update(kw)
    return json.dumps(a)


class LLM:
    def __init__(self, *texts):
        self.texts, self.prompts = list(texts), []

    def __call__(self, prompt, timeout):
        self.prompts.append(prompt)
        t = self.texts[min(len(self.prompts) - 1, len(self.texts) - 1)]
        if isinstance(t, Exception):
            raise t
        return t


VC = {"min_dur_s": 10, "max_dur_s": 60, "lead_pad_s": 0.15, "tail_pad_s": 0.35}


# ---------------------------------------------------------------- cut_review pieces

def test_lines_number_the_sentences_and_mark_the_clip():
    ls = _sents(WINDOW)
    assert [s["text"].split()[0] for s in ls] == ["We", "He", "How", "He", "Then", "So", "Anyway"]
    assert [s["inside"] for s in ls] == [False, False, True, True, True, True, False]
    from clipengine import cut_review
    assert cut_review.current_span(ls) == (2, 5)


def test_a_cut_inside_a_sentence_splits_it_at_the_cut():
    from clipengine import cut_review
    ls = cut_review.lines(WINDOW, 10.6, 21.0)        # opens inside "He looked like he got ..."
    first = next(s for s in ls if s["inside"])
    before = ls[first["i"] - 1]
    assert before["seam"] and before["text"].endswith(cut_review.SEAM)
    assert first["seam"] and first["text"].startswith(cut_review.SEAM)
    assert not before["inside"]


def test_the_prompt_shows_the_lines_the_title_and_the_reviewers_reason():
    from clipengine import cut_review
    p = cut_review.prompt(_sents(WINDOW), *CUT, "stream", "Why border sneakers earn respect",
                          10, 60, feedback="opens on 'How so?', a reply")
    assert "> [S2 8.0-" in p and "s] How so?" in p and "  [S1 4.0-" in p
    assert '"Why border sneakers earn respect"' in p and "REJECTED" in p and "How so?" in p
    assert "livestream" in p
    assert "(none: write one)" in cut_review.prompt(_sents(WINDOW), *CUT, "stream", None, 10, 60)


def test_parse_takes_the_first_object_with_start_and_end():
    from clipengine import cut_review
    assert cut_review.parse('Sure.\n```json\n{"start": 1, "end": 5}\n```')["end"] == 5
    assert cut_review.parse("no json here") is None
    assert cut_review.parse('{"title": "x"}') is None


def test_the_editor_moves_the_start_back_to_the_setup():
    from clipengine import cut_review
    llm = LLM(answer(start=1, start_fault="a reply to a remark the clip leaves out"))
    clip = {"start_s": CUT[0], "end_s": CUT[1], "hook": "Old title"}
    r = cut_review.review(clip, WINDOW, "stream", VC, llm=llm, st=cut_review.DEFAULTS)
    assert r["verdict"] == "move" and r["lines"] == [1, 5]
    assert 3.6 < r["start"] <= 4.0 and r["end"] == pytest.approx(CUT[1], abs=0.4)
    assert r["hook"] == "Old title" and not r["hook_changed"]
    assert "S2-S5 -> S1-S5" in r["why"]


def test_the_editor_ends_before_the_new_topic_and_keeps_a_right_cut():
    from clipengine import cut_review
    clip = {"start_s": 3.85, "end_s": 24.5, "hook": "Old title"}      # runs into "Anyway ..."
    r = cut_review.review(clip, WINDOW, "stream", VC, llm=LLM(answer(start=1, end=5)),
                          st=cut_review.DEFAULTS)
    assert r["verdict"] == "move" and r["end"] < 22.0
    r = cut_review.review({"start_s": CUT[0], "end_s": CUT[1], "hook": "Old title"}, WINDOW,
                          "stream", VC, llm=LLM(answer()), st=cut_review.DEFAULTS)
    assert r["verdict"] == "keep" and (r["start"], r["end"]) == CUT


def test_a_clip_with_no_title_gets_the_editors_title():
    from clipengine import cut_review
    clip = {"start_s": CUT[0], "end_s": CUT[1], "hook": None}
    r = cut_review.review(clip, WINDOW, "stream", VC,
                          llm=LLM(answer(title_ok=False, title='"Guards banned him for life"')),
                          st=cut_review.DEFAULTS)
    assert r["verdict"] == "keep" and r["hook"] == "Guards banned him for life"
    assert r["hook_changed"]


def test_a_moved_cut_takes_a_new_title_unless_the_old_one_still_holds():
    from clipengine import cut_review
    clip = {"start_s": CUT[0], "end_s": CUT[1], "hook": "Old title"}
    r = cut_review.review(clip, WINDOW, "stream", VC,
                          llm=LLM(answer(start=1, title_ok=False, title="Sneako got banned")),
                          st=cut_review.DEFAULTS)
    assert r["verdict"] == "move" and r["hook"] == "Sneako got banned"


@pytest.mark.parametrize("text, why", [
    ("garbage", "no JSON"),
    (answer(start=9, end=12), "not a span"),
    (answer(start=4, end=2), "not a span"),
    (RuntimeError("claude exited 1"), "RuntimeError"),
])
def test_a_bad_answer_keeps_the_cut_and_the_title(text, why):
    from clipengine import cut_review
    clip = {"start_s": CUT[0], "end_s": CUT[1], "hook": "Old title"}
    r = cut_review.review(clip, WINDOW, "stream", VC, llm=LLM(text), st=cut_review.DEFAULTS)
    assert r["verdict"] == "error" and why in r["why"]
    assert (r["start"], r["end"], r["hook"]) == (CUT[0], CUT[1], "Old title")


def test_a_move_outside_the_band_or_the_cap_is_refused():
    from clipengine import cut_review
    clip = {"start_s": CUT[0], "end_s": CUT[1], "hook": "Old title"}
    r = cut_review.review(clip, WINDOW, "stream", {**VC, "max_dur_s": 12},
                          llm=LLM(answer(start=0)), st=cut_review.DEFAULTS)
    assert r["verdict"] == "keep" and "does not fit" in r["why"]
    r = cut_review.review(clip, WINDOW, "stream", VC, llm=LLM(answer(start=0)),
                          st={**cut_review.DEFAULTS, "max_move_s": 3.0})
    assert r["verdict"] == "keep" and "cap" in r["why"]


def test_merged_words_keep_the_clips_own_words_inside_their_span():
    from clipengine import cut_review
    side = [dict(w, w=w["w"].upper()) for w in WINDOW if CUT[0] <= w["start"] <= CUT[1]]
    m = cut_review.merged_words(WINDOW, side, *CUT)
    assert len(m) == len(WINDOW)
    assert all(w["w"].isupper() for w in m if CUT[0] <= w["start"] <= CUT[1] - 0.3)
    assert m[0]["w"] == "We" and m[-1]["w"] == "tonight?"


def test_settings_switch_off_by_config_and_env(monkeypatch):
    from clipengine import cut_review
    monkeypatch.delenv("CUT_REVIEW", raising=False)
    assert cut_review.settings({})["enabled"] is True
    assert cut_review.settings({"daily": {"cut_review": {"enabled": False}}})["enabled"] is False
    monkeypatch.setenv("CUT_REVIEW", "0")
    assert cut_review.settings({})["enabled"] is False


# ---------------------------------------------------------------- engine: before the render

@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("CUT_REVIEW", raising=False)
    monkeypatch.delenv("CUT_SEAT", raising=False)
    runstate.reset()
    from clipengine import engine
    monkeypatch.setattr(engine, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(engine, "OUT_DIR", str(tmp_path / "clips"))
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", None)
    return tmp_path


def _clip(tmp_path, hook="Old title", start=CUT[0], end=CUT[1], sid="Tw_w01", bt="sentence"):
    media = tmp_path / f"{sid}.mp4"
    media.write_bytes(b"\0")
    runstate.add_source(sid, local_path=str(media), vertical="stream")
    wp = tmp_path / f"{sid}_side.json"
    wp.write_text(json.dumps([w for w in WINDOW if start <= w["start"] <= end]))
    cid = runstate.add_clip(sid, start, end, 0.6, "auto", hook=hook, words_path=str(wp),
                         features_json={"refined": True}, boundary_type=bt)
    return runstate.clip(cid), str(media)


def test_the_review_moves_the_clip_row_and_writes_its_sidecar(db):
    from clipengine import engine
    clip, media = _clip(db, hook=None)
    llm = LLM(answer(start=1, title_ok=False, title="Guards banned him for life"))
    new, r = engine._cut_review(clip, {"vertical": "stream"}, media, 40.0, llm=llm,
                                words_fn=lambda m: WINDOW, rms_fn=lambda m, d: None)
    row = runstate.clip(clip["clip_id"])
    assert r["verdict"] == "move" and row["start_s"] == pytest.approx(r["start"])
    assert row["hook"] == "Guards banned him for life" == new["hook"]
    assert row["words_path"].endswith("_rev1.json") and new["words_path"] == row["words_path"]
    side = json.loads(pathlib.Path(row["words_path"]).read_text())
    assert side[0]["w"] == "We" and side[-1]["w"] == "tonight?"     # context for the gate
    rec = json.loads(row["features_json"])["cut_review"]
    assert len(rec) == 1 and rec[0]["verdict"] == "move" and rec[0]["old_hook"] is None


def test_the_review_is_skipped_off_for_energy_cuts_and_without_time(db, monkeypatch):
    from clipengine import engine
    clip, media = _clip(db)
    no = LLM(RuntimeError("must not be asked"))
    monkeypatch.setenv("CUT_REVIEW", "0")
    assert engine._cut_review(clip, {}, media, 40.0, llm=no, words_fn=lambda m: WINDOW)[1] is None
    monkeypatch.delenv("CUT_REVIEW")
    energy, media2 = _clip(db, sid="En_w00", bt="energy")
    assert engine._cut_review(energy, {}, media2, 40.0, llm=no)[1] is None
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", time.monotonic() + 5.0)
    assert engine._cut_review(clip, {}, media, 40.0, llm=no, words_fn=lambda m: WINDOW)[1] is None
    assert no.prompts == [] and runstate.clip(clip["clip_id"])["start_s"] == CUT[0]


def test_an_llm_error_leaves_the_clip_as_it_was(db):
    from clipengine import engine
    clip, media = _clip(db)
    new, r = engine._cut_review(clip, {}, media, 40.0, llm=LLM(RuntimeError("claude exited 1")),
                                words_fn=lambda m: WINDOW, rms_fn=lambda m, d: None)
    row = runstate.clip(clip["clip_id"])
    assert r["verdict"] == "error" and (row["start_s"], row["end_s"], row["hook"]) == \
        (clip["start_s"], clip["end_s"], "Old title")
    assert row["words_path"] == clip["words_path"]


def test_review_feedback_only_for_cut_and_title_faults():
    from clipengine import engine
    rep = {"verdict": "fail", "drop_reason": "ending: ends mid-sentence",
           "ending": {"passed": False, "errors": ["ends mid-sentence: last word 'him'"]},
           "caption_asr": {"passed": False, "errors": ["garbled"]}}
    fb = engine.review_feedback(rep)
    assert "ending: ends mid-sentence: last word 'him'" in fb and "garbled" not in fb
    rep["edge_bands"] = {"passed": False, "errors": ["band"]}
    assert engine.review_feedback(rep) is None
    assert engine.review_feedback({"verdict": "fail", "hook_grounding": {
        "passed": False, "errors": ["no hook: the clip has no hook title"]}}).startswith("hook")


class Recompose:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, run=None, deadline=None, **kw):
        self.calls.append(cmd)
        pathlib.Path(cmd[6]).write_bytes(b"fake mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout='QC {"layout": "track"}\n', stderr="")


def test_a_gate_reject_on_the_cut_goes_back_to_the_editor_and_renders_once_more(db, monkeypatch):
    from clipengine import engine
    clip, media = _clip(db)
    out = db / "scratch" / f"clip_{clip['clip_id']}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = Recompose()
    monkeypatch.setattr(engine, "_run_recompose", rec)
    monkeypatch.setattr(engine, "_caption_step", lambda c, m, d=None: c)
    monkeypatch.setattr(engine, "_window_words", lambda m: WINDOW)
    monkeypatch.setattr(engine, "_window_rms", lambda m, d: None)
    llm = LLM(answer(start=1))
    from clipengine import cut_review
    monkeypatch.setattr(cut_review, "claude_llm", lambda vc, st=None: llm)
    monkeypatch.setattr(engine, "_run_quality_gate", lambda o, c, qc=None: {
        "verdict": "pass", "seen": [c["start_s"], c["end_s"]]})
    first = {"verdict": "fail", "drop_reason": "word_boundaries: start mid-sentence",
             "sentence_bounds": {"passed": False, "errors": ["start mid-sentence: prev='so?'"]}}
    got = engine._review_recut(clip, {"vertical": "stream"}, media, 40.0, first, out, 30.0)
    assert got is not None and got[2]["verdict"] == "pass" and len(rec.calls) == 1
    assert got[0]["start_s"] < CUT[0] and got[2]["review_recut"]["from"][0] == CUT[0]
    assert "REJECTED" in llm.prompts[0] and "start mid-sentence" in llm.prompts[0]
    # a crop fault is not the editor's: no review, no render
    crop = {"verdict": "fail", "edge_bands": {"passed": False, "errors": ["band"]}}
    assert engine._review_recut(clip, {}, media, 40.0, crop, out, 30.0) is None
    assert len(rec.calls) == 1 and len(llm.prompts) == 1


def test_the_editor_keeping_the_cut_after_a_reject_costs_no_render(db, monkeypatch):
    from clipengine import engine
    from clipengine import cut_review
    clip, media = _clip(db)
    rec = Recompose()
    monkeypatch.setattr(engine, "_run_recompose", rec)
    monkeypatch.setattr(engine, "_window_words", lambda m: WINDOW)
    monkeypatch.setattr(engine, "_window_rms", lambda m, d: None)
    monkeypatch.setattr(cut_review, "claude_llm", lambda vc, st=None: LLM(answer()))
    first = {"verdict": "fail", "ending": {"passed": False, "errors": ["ends mid-sentence"]}}
    assert engine._review_recut(clip, {}, media, 40.0, first, db / "x.mp4", 30.0) is None
    assert rec.calls == []


# ---------------------------------------------------------------- the strong pass's context

def test_caption_context_defaults_and_the_off_switch():
    from clipengine import engine
    assert engine.caption_context({}) == (1.5, 2.0, True)
    assert engine.caption_context({"render": {"caption_context": {
        "lead_s": 0.5, "tail_s": 0.5, "seat": False}}}) == (0.5, 0.5, False)


def test_a_first_word_stretched_over_the_silence_before_the_cut_stays_burned(db, monkeypatch):
    """2742: large-v3-turbo timed "We've" from the silence before it (6.50-7.20 s); the cut sat at
    6.945 s, the voice comes back at 6.95 s. Its middle (6.85 s) fell before the cut and the
    captions opened on "had"; seated on the audio, it stays the clip's first word."""
    from clipengine import engine
    lo = 5.445
    frames = np.full(int((10.0 - lo) / 0.01), 0.2, dtype=np.float32)
    frames[int((6.40 - lo) / 0.01):int((6.95 - lo) / 0.01)] = 0.0005   # silence 6.40-6.95 s
    side = W("We've had six years of the pandemic.", 6.96)
    wp = db / "side.json"
    wp.write_text(json.dumps(side))
    runstate.add_source("Ml_w01", local_path=str(db / "w.mp4"))
    cid = runstate.add_clip("Ml_w01", 6.945, 9.0, 0.5, "auto", hook="h", words_path=str(wp),
                         features_json={"refined": True})
    strong = [{"w": "We've", "start": 6.50, "end": 7.20, "p": 0.9}] + \
        [dict(w, p=0.9) for w in W("had six years of the pandemic.", 7.25)]
    side2, info = engine.strong_captions(runstate.clip(cid), "w.mp4", media_dur=40.0,
                                         words_fn=lambda m, a, b, t: strong,
                                         rms_fn=lambda m, a, b: frames)
    burned = [w for w in side2 if 6.945 <= (w["start"] + w["end"]) / 2 <= 9.0
              and w.get("cap") is not False]
    assert burned[0]["w"] == "We've" and burned[0]["start"] == pytest.approx(6.95, abs=0.011)
    assert info["context"]["seated"] >= 1


# ---------------------------------------------------------------- the cuts seated on the audio

def _frames(t0, t1, voiced, level=0.2, floor=0.0005):
    """10 ms frames over [t0, t1): `level` inside the (a, b) spans of `voiced`, `floor` outside."""
    n = int(round((t1 - t0) / 0.01))
    f = np.full(n, floor, dtype=np.float32)
    for a, b in voiced:
        f[max(0, int(round((a - t0) / 0.01))):min(n, int(round((b - t0) / 0.01)))] = level
    return f


def test_an_end_tight_on_the_last_word_gets_a_breath_before_the_next_voice():
    from clipengine import boundaries
    rms = _frames(0.0, 10.0, [(1.0, 4.0), (4.6, 8.0)])          # a pause 4.0-4.6 s
    (s, e), info = boundaries.seat_cuts(1.0 - 0.15, 4.03, rms)
    assert e == pytest.approx(4.30, abs=0.011) and info["end"]["how"] == "breath"
    (s, e), info = boundaries.seat_cuts(0.85, 4.40, rms)        # already a breath out: stays
    assert e == 4.40 and "end" not in info
    short = _frames(0.0, 10.0, [(1.0, 4.0), (4.25, 8.0)])       # a 0.25 s pause: guard first
    (s, e), info = boundaries.seat_cuts(0.85, 4.02, short)
    assert e == pytest.approx(4.25 - boundaries.CUT_GUARD_S, abs=0.011)


def test_an_end_inside_the_voice_moves_to_where_it_stops_unless_a_word_is_said_there():
    from clipengine import boundaries
    rms = _frames(0.0, 10.0, [(1.0, 4.4), (5.2, 8.0)])          # the voice runs on to 4.4 s
    (s, e), info = boundaries.seat_cuts(0.85, 4.1, rms)
    assert e == pytest.approx(4.70, abs=0.011) and info["end"]["how"] == "offset"
    words = W("so we went home", 1.0) + [{"w": "and", "start": 4.15, "end": 4.38}]
    (s, e), info = boundaries.seat_cuts(0.85, 4.1, rms, words=words)
    assert e == 4.1 and "and" in info["end_kept"]
    far = _frames(0.0, 10.0, [(1.0, 5.0), (5.6, 8.0)])          # it runs on 0.9 s: not this word
    assert boundaries.seat_cuts(0.85, 4.1, far)[0][1] == 4.1


def test_a_start_on_the_voice_onset_moves_back_by_a_lead():
    """2742: the start cut at 6.945 s, the voice back at 6.95 s ('We've' cut off)."""
    from clipengine import boundaries
    rms = _frames(5.0, 10.0, [(5.0, 6.40), (6.95, 9.0)])
    (s, e), info = boundaries.seat_cuts(6.945, 9.5, rms, t0=5.0)
    assert s == pytest.approx(6.95 - boundaries.CUT_LEAD_S, abs=0.011)
    assert info["start"]["how"] == "lead"
    # the cut inside the voice since 6.60 s, which is the last word of the sentence before
    words = W("it was gone.", 6.0) + W("We've had six years", 6.96)
    rms2 = _frames(5.0, 10.0, [(5.0, 6.40), (6.60, 9.0)])
    (s, e), info = boundaries.seat_cuts(6.80, 9.5, rms2, t0=5.0, words=words)
    assert s == 6.80 and "gone." in info["start_kept"]
    (s, e), info = boundaries.seat_cuts(6.80, 9.5, rms2, t0=5.0)  # no words: the onset rule
    # the voice back at 6.60 s less the lead, but no closer than the guard to where it stopped
    assert s == pytest.approx(6.40 + boundaries.CUT_GUARD_S, abs=0.011)
    assert info["start"]["how"] == "onset"


def test_nothing_moves_over_a_music_bed_or_without_audio():
    from clipengine import boundaries
    bed = np.full(1000, 0.2, dtype=np.float32) * (1 + 0.1 * np.sin(np.arange(1000)))
    assert boundaries.seat_cuts(2.0, 6.0, bed) == ((2.0, 6.0), {})
    assert boundaries.seat_cuts(2.0, 6.0, None) == ((2.0, 6.0), {})


def test_a_collapsed_last_word_gets_its_voice_and_is_no_clean_end():
    """3172: small.en timed 'hair?' 69.10-69.10 s, 'Bro,' at 69.84 s; the voice runs from 69.10 s
    into 'Bro,' with no pause. Seated, the gap the retrim read as a clean end is gone."""
    from clipengine import boundaries
    from clipengine import transcript
    words = (W("What happened to your", 68.36, 0.15)
             + [{"w": "hair?", "start": 69.10, "end": 69.10}, {"w": "Bro,", "start": 69.84,
                                                               "end": 69.84}]
             + W("bro, that is crazy.", 70.38))
    rms = _frames(66.0, 73.0, [(68.36, 69.06), (69.12, 69.90), (70.38, 72.0)])
    out, moved = boundaries.span_short_words(words, rms, 66.0)
    hair = next(w for w in out if w["w"] == "hair?")
    assert hair["end"] == pytest.approx(69.82, abs=0.011) and hair["asr_end"] == 69.10
    assert moved and moved[0]["w"] == "hair?"
    before = transcript.sentences(words, gap_s=0.7)
    after = transcript.sentences(out, gap_s=0.7)
    cs_b, ce_b = transcript.clean_edges(before, 1.5, True, min_end_gap_s=0.25, strict_ends=True)
    cs_a, ce_a = transcript.clean_edges(after, 1.5, True, min_end_gap_s=0.25, strict_ends=True)
    k = next(i for i, s in enumerate(before) if s["text"].endswith("hair?"))
    assert ce_b[k] and not ce_a[k]


def test_the_render_seats_the_cut_before_the_caption_pass(db, monkeypatch):
    from clipengine import engine
    clip, media = _clip(db, start=7.85, end=20.97)               # 0.02 s after 'life.'
    rms = _frames(0.0, 40.0, [(1.0, 3.05), (4.0, 7.25), (8.0, 8.55), (9.5, 13.15), (14.0, 17.25),
                              (18.0, 20.95), (22.0, 24.05)])     # the voice = the words
    seen = []

    def caption_step(c, m, d=None):
        seen.append((c["start_s"], c["end_s"]))
        raise engine.NoCleanCut("stop here", "end")
    monkeypatch.setattr(engine, "_caption_step", caption_step)
    monkeypatch.setattr(engine, "_drop_unclean", lambda c, e: None)
    monkeypatch.setattr(engine, "_cut_review", lambda c, *a, **k: (c, None))
    monkeypatch.setattr(engine, "_cached_window_words", lambda m, model=None: WINDOW)
    monkeypatch.setattr(engine.transcript, "load_audio", lambda m, lo, hi, sr=16000: np.repeat(
        rms[int(round(lo / 0.01)):int(round(hi / 0.01))], 160).astype(np.float32))
    runstate.update_source(clip["source_id"], state="selected", duration_s=40.0)
    engine.render(runstate.clip(clip["clip_id"]))
    assert seen and seen[0][1] == pytest.approx(20.95 + 0.30, abs=0.011)   # offset + breath
    feats = json.loads(runstate.clip(clip["clip_id"])["features_json"])
    assert feats["cut_seat"]["end"]["how"] == "breath"


def _undo_render(db, monkeypatch, llm, hook=None):
    """A clip with no title the review moves back onto S1; the crop refuses the moved cut,
    select's cut renders. Returns (row, recompose starts)."""
    from clipengine import cut_review
    from clipengine import engine
    clip, media = _clip(db, hook=hook)
    runstate.update_source(clip["source_id"], state="selected", duration_s=40.0)
    monkeypatch.setattr(cut_review, "claude_llm", lambda vc, st=None: llm)
    monkeypatch.setattr(engine, "_window_words", lambda m: WINDOW)
    monkeypatch.setattr(engine, "_cached_window_words", lambda m, model=None: WINDOW)
    monkeypatch.setattr(engine, "_window_rms", lambda m, d: None)
    monkeypatch.setattr(engine, "_caption_step", lambda c, m, d=None: c)
    monkeypatch.setattr(engine, "cut_seat_enabled", lambda cfg=None: False)
    starts = []

    def recompose(cmd, run=None, deadline=None, **kw):
        starts.append(float(cmd[4]))
        if float(cmd[4]) < 7.0:                     # the moved cut (from ~3.85 s)
            return subprocess.CompletedProcess(cmd, 1, stdout='QC {"errors": ["no clean 9:16 '
                                               'crop keeps the speaker clear without slicing 4 '
                                               'graphic(s)"]}\n', stderr="")
        pathlib.Path(cmd[6]).write_bytes(b"fake mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout='QC {"layout": "track"}\n', stderr="")
    monkeypatch.setattr(engine, "_run_recompose", recompose)
    monkeypatch.setattr(engine, "_run_quality_gate", lambda o, c, qc=None: {"verdict": "pass"})
    monkeypatch.setattr(engine, "_store_clip", lambda f, clip: str(f))
    assert engine.render(runstate.clip(clip["clip_id"])) is True
    return runstate.clip(clip["clip_id"]), starts


def test_a_moved_cut_the_crop_refuses_goes_back_to_selects_cut_keeping_the_reviews_title(
        db, monkeypatch):
    """The review moved the start back 4 s onto a sliced graphic and the crop planner refused the
    render; select's cut rendered, so the clip goes back to it. The clip had no title of its
    own, so it keeps the review's."""
    llm = LLM(answer(start=1, title_ok=False, title="Banned from the border for life"),
              RuntimeError("the kept title needs no second review"))
    row, starts = _undo_render(db, monkeypatch, llm)
    assert (row["start_s"], row["end_s"]) == CUT and len(starts) == 2 and starts[0] < 7.0
    assert row["hook"] == "Banned from the border for life" and len(llm.prompts) == 1
    rec = json.loads(row["features_json"])["cut_review"]
    assert len(rec) == 1 and rec[0]["verdict"] == "move" and "refused" in rec[0]["undone"]


def test_a_kept_title_said_only_outside_the_old_cut_is_written_again(db, monkeypatch):
    """The review's title for the moved cut quotes S1 ("... the worst streamer he has ever met"),
    which select's cut leaves out: on the way back the review writes the title for that cut."""
    llm = LLM(answer(start=1, title_ok=False, title="The worst streamer he ever met"),
              answer(title_ok=False, title="Title for the old cut"))
    row, starts = _undo_render(db, monkeypatch, llm)
    assert (row["start_s"], row["end_s"]) == CUT and len(starts) == 2
    assert row["hook"] == "Title for the old cut"
    rec = json.loads(row["features_json"])["cut_review"]
    assert rec[0]["verdict"] == "move" and rec[0]["undone"] and rec[1]["title_only"]


def test_a_clip_with_its_own_title_gets_it_back_on_the_undo(db, monkeypatch):
    """A clip titled before the review (select's pick) gets that title back on undo, as before."""
    llm = LLM(answer(start=1, title_ok=False, title="Moved title"),
              RuntimeError("no second review"))
    row, starts = _undo_render(db, monkeypatch, llm, hook="Select's own title")
    assert (row["start_s"], row["end_s"]) == CUT and len(starts) == 2
    assert row["hook"] == "Select's own title" and len(llm.prompts) == 1


def test_a_title_built_from_words_outside_the_clip_is_asked_for_again():
    """11-09 2786 (replay): 'He's dancing for a cash prize' - 'prize of money' is said after the
    cut; the eye-check rejected the title. A leaked word gets one more title-only question."""
    from clipengine import cut_review
    clip = {"start_s": CUT[0], "end_s": CUT[1], "hook": None}
    llm = LLM(answer(title_ok=False, title="Sneako is the worst streamer"),
              answer(title_ok=False, title="Banned from the border for life"))
    r = cut_review.review(clip, WINDOW, "stream", VC, llm=llm, st=cut_review.DEFAULTS)
    assert r["verdict"] == "keep" and r["hook"] == "Banned from the border for life"
    assert "sneako" in r["title_leak"]["words"] and "FIXED" in llm.prompts[1]
    assert "Sneako is the worst streamer" in llm.prompts[1]
    # still leaking: a clip with a title keeps its own
    clip = {"start_s": CUT[0], "end_s": CUT[1], "hook": "Old title"}
    llm = LLM(answer(title_ok=False, title="Sneako is the worst streamer"))
    r = cut_review.review(clip, WINDOW, "stream", VC, llm=llm, st=cut_review.DEFAULTS)
    assert r["hook"] == "Old title" and not r["hook_changed"] and len(llm.prompts) == 2


def test_a_title_for_lines_that_could_not_be_placed_does_not_replace_the_old_one():
    from clipengine import cut_review
    clip = {"start_s": CUT[0], "end_s": CUT[1], "hook": "Old title"}
    r = cut_review.review(clip, WINDOW, "stream", {**VC, "max_dur_s": 12},
                          llm=LLM(answer(start=0, title_ok=False, title="We were at the border")),
                          st=cut_review.DEFAULTS)
    assert r["verdict"] == "keep" and r["hook"] == "Old title" and not r["hook_changed"]


def test_a_title_the_caption_step_cut_out_from_under_is_written_again(db, monkeypatch):
    """11-09 2786 (replay): the review titled the cut from 'prize of money'; the tidy then pulled
    the end back before those words. The title is written again for the cut as it is."""
    from clipengine import cut_review
    from clipengine import engine
    clip, media = _clip(db, hook="Old title")
    runstate.update_source(clip["source_id"], state="selected", duration_s=40.0)
    llm = LLM(answer(title_ok=False, title="Banned from the border for life"),
              answer(start=2, end=4, title_ok=False, title="Yelling at the guards for food"))
    monkeypatch.setattr(cut_review, "claude_llm", lambda vc, st=None: llm)
    monkeypatch.setattr(engine, "_window_words", lambda m: WINDOW)
    monkeypatch.setattr(engine, "_cached_window_words", lambda m, model=None: WINDOW)
    monkeypatch.setattr(engine, "_window_rms", lambda m, d: None)
    monkeypatch.setattr(engine, "cut_seat_enabled", lambda cfg=None: False)

    def caption_step(c, m, d=None):          # the tidy pulls the end back before S5
        runstate.update_clip(c["clip_id"], end_s=17.6)
        return {**c, "end_s": 17.6}
    monkeypatch.setattr(engine, "_caption_step", caption_step)
    seen = []

    def recompose(cmd, run=None, deadline=None, **kw):
        seen.append(cmd)
        pathlib.Path(cmd[6]).write_bytes(b"fake mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout='QC {"layout": "track"}\n', stderr="")
    monkeypatch.setattr(engine, "_run_recompose", recompose)
    monkeypatch.setattr(engine, "_run_quality_gate", lambda o, c, qc=None: {"verdict": "pass"})
    monkeypatch.setattr(engine, "_store_clip", lambda f, clip: str(f))
    assert engine.render(runstate.clip(clip["clip_id"])) is True
    row = runstate.clip(clip["clip_id"])
    assert row["hook"] == "Yelling at the guards for food" and len(llm.prompts) == 2
    assert "FIXED" in llm.prompts[1] and "--hook" in seen[0]
    assert seen[0][seen[0].index("--hook") + 1] == "Yelling at the guards for food"


def test_the_leak_check_reads_every_transcript_of_the_window():
    """2786: base.en heard 'prize of money' just after the cut; the burned large-v3-turbo words
    heard other words there. A word one pass hears only outside the cut is a leak."""
    from clipengine import cut_review
    turbo = W("I need somebody to grab me", 10.0) + W("If I try to grab somebody you", 14.2)
    base = W("I need somebody to grab me", 10.0) + W("let's have a prize of money", 14.2)
    ins, out = cut_review.split_by_cut(10.0, 14.1, turbo)
    assert cut_review.leaked_words("Dancing for prize money", ins, out) == []
    ins, out = cut_review.split_by_cut(10.0, 14.1, turbo, base)
    assert cut_review.leaked_words("Dancing for prize money", ins, out) == ["prize", "money"]
    assert cut_review.leaked_words("Somebody grab me", ins, out) == []


def test_the_undo_finds_the_move_behind_a_later_title_only_review(db, monkeypatch):
    """11-09 2786 (replay): moved, then retitled after the tidy (a title-only record after the
    move), then refused by the crop planner: the undo still finds the move."""
    from clipengine import cut_review
    from clipengine import engine
    clip, media = _clip(db, hook=None)
    runstate.update_source(clip["source_id"], state="selected", duration_s=40.0)
    llm = LLM(answer(start=1, title_ok=False, title="Banned from the border for life"),
              answer(start=1, end=4, title_ok=False, title="Yelling at the guards for food"),
              answer(title_ok=False, title="Banned from the border for life"))
    monkeypatch.setattr(cut_review, "claude_llm", lambda vc, st=None: llm)
    monkeypatch.setattr(engine, "_window_words", lambda m: WINDOW)
    monkeypatch.setattr(engine, "_cached_window_words", lambda m, model=None: WINDOW)
    monkeypatch.setattr(engine, "_window_rms", lambda m, d: None)
    monkeypatch.setattr(engine, "cut_seat_enabled", lambda cfg=None: False)

    def caption_step(c, m, d=None):          # the tidy pulls a moved end back before S5
        if float(c["start_s"]) < 7.0:
            runstate.update_clip(c["clip_id"], end_s=17.6)
            return {**c, "end_s": 17.6}
        return c
    monkeypatch.setattr(engine, "_caption_step", caption_step)
    starts = []

    def recompose(cmd, run=None, deadline=None, **kw):
        starts.append(float(cmd[4]))
        if float(cmd[4]) < 7.0:
            return subprocess.CompletedProcess(cmd, 1, stdout='QC {"errors": ["no clean 9:16 '
                                               'crop without slicing 4 graphic(s)"]}\n', stderr="")
        pathlib.Path(cmd[6]).write_bytes(b"fake mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout='QC {"layout": "track"}\n', stderr="")
    monkeypatch.setattr(engine, "_run_recompose", recompose)
    monkeypatch.setattr(engine, "_run_quality_gate", lambda o, c, qc=None: {"verdict": "pass"})
    monkeypatch.setattr(engine, "_store_clip", lambda f, clip: str(f))
    assert engine.render(runstate.clip(clip["clip_id"])) is True
    row = runstate.clip(clip["clip_id"])
    assert (row["start_s"], row["end_s"]) == CUT and len(starts) == 2
    rec = json.loads(row["features_json"])["cut_review"]
    assert rec[0]["verdict"] == "move" and rec[0]["undone"] and rec[1]["title_only"]


def test_on_a_start_the_editor_chose_the_tidy_looks_forward_first():
    """11-09 3097 (replay): the editor started after the filler 'Come on.'; the start cut sat
    inside a stretched word of a run-on passage, and the tidy went back to the sentence before,
    onto the filler again. With start_forward it takes the next clean sentence start instead."""
    from clipengine import boundaries
    words = (W("Earlier sentence here.", 1.0) + W("and you know what I mean so basically yeah", 2.0)
             + W("The real story starts now and goes on for a while longer today.", 6.0)
             + W("Then it ends here.", 20.0))
    rms = _frames(0.0, 30.0, [(1.0, 1.9), (2.0, 4.7), (6.0, 9.9), (20.0, 21.2)])
    cfg = {"min_dur_s": 5, "max_dur_s": 60}
    back = boundaries.tidy_cut(words, 3.25, 21.5, cfg=cfg, rms=rms)       # inside 'I'
    assert back["tidy"]["start"] == "extend" and back["start"] < 2.0
    fwd = boundaries.tidy_cut(words, 3.25, 21.5, cfg=cfg, rms=rms, start_forward=True)
    assert fwd["tidy"]["start"] == "pull_back" and 4.7 <= fwd["start"] <= 6.0
    assert fwd["first"] == "The"


def test_the_editor_chose_the_start_unless_it_was_undone_or_only_titled():
    from clipengine import engine
    rec = lambda **k: {"features_json": json.dumps({"cut_review": [k]})}    # noqa: E731
    assert engine._editor_chose_start(rec(verdict="keep"))
    assert engine._editor_chose_start(rec(verdict="move"))
    assert not engine._editor_chose_start(rec(verdict="move", undone="refused"))
    assert not engine._editor_chose_start(rec(verdict="error"))
    assert not engine._editor_chose_start(rec(verdict="keep", title_only=True))
    assert not engine._editor_chose_start({"features_json": "{}"})


def test_an_end_moved_onto_the_windows_last_line_is_refused():
    """11-09 replay 2679 / 2682 / 3187: the review re-cut after a gate reject ended on the
    window's last line, 0.04-0.17 s before the window ends; the gate cannot hear past it
    ('ending unverifiable'). Such a move is refused; the end stays."""
    from clipengine import cut_review
    clip = {"start_s": CUT[0], "end_s": CUT[1], "hook": "Old title"}
    r = cut_review.review(clip, WINDOW, "stream", VC, llm=LLM(answer(end=6)), media_dur=24.6,
                          st=cut_review.DEFAULTS)
    assert r["verdict"] == "keep" and "window's end" in r["why"] and r["end"] == CUT[1]
    r = cut_review.review(clip, WINDOW, "stream", VC, llm=LLM(answer(end=6)), media_dur=30.0,
                          st=cut_review.DEFAULTS)
    assert r["verdict"] == "move" and r["end"] > 24.0


# ---------------------------------------------------------------- hooks for clips with no title

def test_a_title_only_review_runs_without_the_windows_time(db, monkeypatch):
    """The full review needs REVIEW_LLM_S of the window; a title-only one is one short call
    (6-15 s) and skips the time check (a clip with no title would ship with no hook)."""
    from clipengine import engine
    clip, media = _clip(db, hook=None)
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", time.monotonic() + 5.0)
    no = LLM(RuntimeError("must not be asked"))
    assert engine._cut_review(clip, {}, media, 40.0, llm=no, words_fn=lambda m: WINDOW)[1] is None
    llm = LLM(answer(title_ok=False, title="Banned from the border for life"))
    new, r = engine._cut_review(clip, {}, media, 40.0, llm=llm, words_fn=lambda m: WINDOW,
                                rms_fn=lambda m, d: None, title_only=True)
    assert r is not None and new["hook"] == "Banned from the border for life"
    assert runstate.clip(clip["clip_id"])["hook"] == "Banned from the border for life"
    assert no.prompts == [] and len(llm.prompts) == 1 and "FIXED" in llm.prompts[0]


def test_the_render_titles_a_clip_the_full_review_had_no_time_for(db, monkeypatch):
    from clipengine import cut_review
    from clipengine import engine
    clip, media = _clip(db, hook=None)
    runstate.update_source(clip["source_id"], state="selected", duration_s=40.0)
    llm = LLM(answer(title_ok=False, title="Banned from the border for life"))
    monkeypatch.setattr(cut_review, "claude_llm", lambda vc, st=None: llm)
    monkeypatch.setattr(engine, "_window_words", lambda m: WINDOW)
    monkeypatch.setattr(engine, "_cached_window_words", lambda m, model=None: WINDOW)
    monkeypatch.setattr(engine, "_window_rms", lambda m, d: None)
    monkeypatch.setattr(engine, "_caption_step", lambda c, m, d=None: c)
    monkeypatch.setattr(engine, "cut_seat_enabled", lambda cfg=None: False)
    monkeypatch.setattr(engine, "_review_room", lambda c, need: "5s of the window left, ~45s needed")
    cmds = []

    def recompose(cmd, run=None, deadline=None, **kw):
        cmds.append(cmd)
        pathlib.Path(cmd[6]).write_bytes(b"fake mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout='QC {"layout": "track"}\n', stderr="")
    monkeypatch.setattr(engine, "_run_recompose", recompose)
    monkeypatch.setattr(engine, "_run_quality_gate", lambda o, c, qc=None: {"verdict": "pass"})
    monkeypatch.setattr(engine, "_store_clip", lambda f, clip: str(f))
    assert engine.render(runstate.clip(clip["clip_id"])) is True
    row = runstate.clip(clip["clip_id"])
    assert row["hook"] == "Banned from the border for life" and len(llm.prompts) == 1
    assert "--hook" in cmds[0] and cmds[0][cmds[0].index("--hook") + 1] == row["hook"]
    rec = json.loads(row["features_json"])["cut_review"]
    assert len(rec) == 1 and rec[0]["title_only"] and (row["start_s"], row["end_s"]) == CUT


def test_a_titled_clip_with_no_time_for_the_review_asks_nothing(db, monkeypatch):
    from clipengine import engine
    clip, media = _clip(db, hook="Old title")
    monkeypatch.setattr(engine, "_review_room", lambda c, need: "no time")
    no = LLM(RuntimeError("must not be asked"))
    assert engine._cut_review(clip, {}, media, 40.0, llm=no, words_fn=lambda m: WINDOW)[1] is None
    assert no.prompts == [] and runstate.clip(clip["clip_id"])["hook"] == "Old title"
