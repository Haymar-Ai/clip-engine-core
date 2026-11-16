"""Cut endings: the final cuts snap to sentence edges of the precise transcript (the end to the
next clean end <= 8 s on), and a clip the gate fails on its ending alone is re-cut to the nearest
clean sentence end and re-rendered once before it is rejected. The word timings model real
failures: a clip that ended 0.4 s before its window end, one that ran on for a lone "Okay.", one
cut into its first word, one opening mid-sentence, one with no clean end inside its window."""
from __future__ import annotations

import importlib
import json
import pathlib
import subprocess

import pytest

from clipengine import boundaries
from clipengine import runstate
from helpers import codes_for, fake_store_clip


def W(*triples):
    return [{"w": w, "start": s, "end": e} for w, s, e in triples]


def words_from(text, t0, step=0.3, gaps=None):
    """Words of `text`, one every `step` s from t0; gaps {index: extra pause before that word}."""
    out, t = [], float(t0)
    for i, tok in enumerate(text.split()):
        t += (gaps or {}).get(i, 0.0)
        out.append({"w": tok, "start": round(t, 3), "end": round(t + step - 0.05, 3)})
        t += step
    return out


CFG = {"min_dur_s": 5, "max_dur_s": 40, "lead_pad_s": 0.15, "tail_pad_s": 0.35,
       "strict_ends": True, "min_end_gap_s": 0.25, "hard_gap_s": 1.5, "sentence_gap_s": 0.7}

# three sentences with real pauses between them, then a long one
TEXT = ("We drove out to the desert that night. Nobody said a word the whole way. "
        "Then my dad pulled over and told me the story of how he lost everything and how he "
        "got it all back in a single year.")
WORDS = words_from(TEXT, 10.0, gaps={8: 0.6, 15: 0.6})
END_OF = {w["w"]: w["end"] for w in WORDS}


# ---------------------------------------------------------------- select: snap

def test_snap_extends_a_mid_sentence_end_to_the_next_sentence_end():
    end = END_OF["said"] + 0.02                       # "Nobody said | a word the whole way."
    r = boundaries.snap_clip(WORDS, 9.9, end, cfg=CFG, edges_clean=(True, False),
                             media_dur=60.0)
    assert r["snap"]["end"] == "extend"
    assert END_OF["way."] <= r["end"] <= END_OF["way."] + 0.8


def test_snap_pulls_back_when_no_sentence_end_is_in_reach():
    words = WORDS + words_from("and then we sat there for hours and hours and nobody "
                               "moved at all for a very long time", WORDS[-1]["end"] + 0.1)
    end = END_OF["single"] + 0.02                     # long run-on: no clean end within 8 s
    r = boundaries.snap_clip(words, 9.9, end, cfg=CFG, edges_clean=(True, False),
                             media_dur=80.0, max_move=3.0)
    assert r["snap"]["end"] == "pull_back"
    assert END_OF["way."] <= r["end"] < END_OF["Then"]


def test_snap_keeps_room_before_the_window_end():
    """261: the clip ended 0.4 s before the window end. The last sentence of a cut-out window
    is never a clean end; the cut goes back to one with source after it."""
    media_dur = WORDS[-1]["end"] + 0.2
    r = boundaries.snap_clip(WORDS, 9.9, media_dur - 0.05, cfg=CFG, edges_clean=(True, False),
                             media_dur=media_dur)
    assert r["snap"]["end"] == "pull_back"
    assert r["end"] <= END_OF["way."] + 0.8 and media_dur - r["end"] > 0.5


def test_snap_moves_a_mid_sentence_start_back_to_its_sentence():
    """314 opened mid-sentence ('...when I did the the | Las Vegas Muscle contest')."""
    start = END_OF["dad"] + 0.02                      # "Then my dad | pulled over ..."
    r = boundaries.snap_clip(WORDS, start, WORDS[-1]["end"] + 0.3, cfg=CFG,
                             edges_clean=(True, True), media_dur=60.0)
    assert r["snap"]["start"] == "extend"
    then = next(w for w in WORDS if w["w"] == "Then")
    assert then["start"] - 0.6 <= r["start"] <= then["start"]


def test_snap_recuts_a_start_inside_the_first_word():
    """311: the cut sat at 11.735 s inside 'To' (11.60-11.88 s on the precise words)."""
    words = W(("Okay.", 10.0, 10.4), ("To", 11.6, 11.88), ("be", 11.9, 12.0),
              ("fair,", 12.0, 12.3)) + words_from("it was the best day of my life.", 12.4)
    r = boundaries.snap_clip(words, 11.735, words[-1]["end"] + 0.3,
                             cfg={**CFG, "min_dur_s": 1}, edges_clean=(True, True),
                             media_dur=30.0)
    assert r["snap"]["start"] == "recut" and r["start"] <= 11.6


def test_snap_leaves_clean_cuts_alone():
    end = END_OF["way."] + 0.3
    r = boundaries.snap_clip(WORDS, 9.9, end, cfg=CFG, edges_clean=(True, False), media_dur=60.0)
    assert r["snap"] == {"start": "ok", "end": "ok"} and (r["start"], r["end"]) == (9.9, end)


def test_snap_keeps_a_start_right_after_a_quick_sentence_end():
    """10-21 PASS clip 427 opens on 'All right' 0.2 s after 'That's awesome.': a start needs the
    sentence before it to end, not the strict end's pause, so the opening stays."""
    words = W(("That's", 10.0, 10.3), ("awesome.", 10.3, 10.8), ("All", 11.0, 11.2),
              ("right.", 11.2, 11.5)) + words_from("You guys have convinced me to try it.", 11.8)
    r = boundaries.snap_clip(words, 10.9, words[-1]["end"] + 0.3, cfg={**CFG, "min_dur_s": 1},
                             edges_clean=(True, True), media_dur=30.0)
    assert r["snap"]["start"] == "ok" and r["start"] == 10.9


def test_snap_without_words_changes_nothing():
    r = boundaries.snap_clip([], 1.0, 20.0, cfg=CFG)
    assert (r["start"], r["end"]) == (1.0, 20.0) and r["snap"]["end"] == "none"


# ---------------------------------------------------------------- render: re-cut

def test_recut_end_completes_a_sentence_the_cut_runs_into():
    end = END_OF["word"] + 0.02                       # gate: "ends mid-sentence: last word 'word'"
    new, info = boundaries.recut_end(WORDS, 9.9, end, cfg=CFG, media_dur=60.0)
    assert info["action"] == "extend" and END_OF["way."] <= new <= END_OF["way."] + 0.8


def test_recut_end_trims_a_lone_okay_tail():
    """306: after '...throughout it?' the clip ran on for a lone 'Okay.' with speech after it."""
    words = words_from("Is that not just lemon seeds in throughout it?", 30.0) + \
        W(("Okay.", 33.6, 33.9), ("That'll", 34.0, 34.3), ("be", 34.3, 34.5), ("it", 34.5, 34.7),
          ("then", 34.7, 34.9), ("I", 35.0, 35.1), ("think", 35.1, 35.4), ("so", 35.4, 35.6),
          ("anyway", 35.6, 36.0), ("we", 36.0, 36.2), ("left", 36.2, 36.5), ("and", 36.5, 36.7))
    new, info = boundaries.recut_end(words, 20.0, 34.0, cfg=CFG, media_dur=37.0)
    assert info["action"] == "trim" and new < 33.6


def test_recut_end_refuses_when_no_clean_end_is_near():
    words = words_from("and so we kept going and going and going and going and going and going "
                       "and going and going and going", 10.0)
    new, why = boundaries.recut_end(words, 0.0, 15.0, cfg=CFG, media_dur=40.0)
    assert new is None and "no other clean sentence end" in why


@pytest.fixture
def db(tmp_path, monkeypatch):
    runstate.reset()
    from clipengine import reason_codes
    from clipengine import engine
    monkeypatch.setattr(engine, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(engine, "vertical_cfg", lambda v: dict(CFG))
    monkeypatch.setattr(engine, "CAPTION_STRONG_MODEL", None)   # stage 3 has its own tests
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", None)   # a run sets no window deadline
    runstate.init()
    return tmp_path


def _ending_fail(msg="ends mid-sentence: last word 'word', next 'the' at -0.04 s"):
    return {"verdict": "fail", "drop_reason": f"ending: {msg}",
            "ending": {"passed": False, "errors": [msg]}}


def _clip_on_window(tmp_path, end):
    media = tmp_path / "W_w00.mp4"
    media.write_bytes(b"\0")
    runstate.add_source("W_w00", local_path=str(media), content_type="auto", vertical="general")
    runstate.update_source("W_w00", state="selected", duration_s=60.0)
    side = tmp_path / "side.json"
    side.write_text(json.dumps([w for w in WORDS if (w["start"] + w["end"]) / 2 <= end]))
    cid = runstate.add_clip("W_w00", 9.9, end, 0.7, "auto", hook="h", words_path=str(side))
    return cid, media


def test_recut_plan_only_for_an_ending_alone(db):
    from clipengine import engine
    clip = {"clip_id": 1, "start_s": 9.9, "end_s": 20.0, "words_path": None}
    both = {**_ending_fail(), "edge_bands": {"passed": False, "errors": ["flat top band"]}}
    new, _side, why = engine.recut_plan(clip, "m.mp4", 60.0, both, CFG, words_fn=lambda *a: WORDS)
    assert new is None and "black_band" in why


def test_recut_plan_builds_the_extended_sidecar(db, tmp_path):
    from clipengine import engine
    end = END_OF["word"] + 0.02
    cid, media = _clip_on_window(tmp_path, end)
    clip = runstate.clip(cid)
    new, side, info = engine.recut_plan(clip, str(media), 60.0, _ending_fail(), CFG,
                                        words_fn=lambda m, a, b: [w for w in WORDS
                                                                  if a <= w["start"] <= b],
                                        rms_fn=lambda *a: None)
    assert info["action"] == "extend" and new[0] == 9.9 and new[1] > END_OF["way."]
    texts = [w["w"] for w in side]
    assert texts.count("word") == 1 and "way." in texts          # no seam duplicate
    assert texts[-1] == WORDS[-1]["w"]                            # context after the new end


def _fake_recompose(calls):
    def run(cmd, **kw):
        calls.append(cmd)
        pathlib.Path(cmd[6]).write_bytes(b"mp4")          # recompose SRC START DUR OUT
        return subprocess.CompletedProcess(cmd, 0, stdout='QC {"layout": "track"}\n', stderr="")
    return run


def test_render_recuts_a_failed_ending_and_ships_the_pass(db, tmp_path, monkeypatch):
    from clipengine import engine
    end = END_OF["word"] + 0.02
    cid, _media = _clip_on_window(tmp_path, end)
    calls, gates = [], []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    monkeypatch.setattr(engine.transcript, "words_window", lambda m, a, b, model=None:
                        [w for w in WORDS if a <= w["start"] <= b])
    monkeypatch.setattr(engine.transcript, "load_audio", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("no audio in tests")))

    def gate(out, clip, qc=None):
        gates.append(float(clip["end_s"]))
        return _ending_fail() if len(gates) == 1 else {"verdict": "pass"}
    monkeypatch.setattr(engine, "_run_quality_gate", gate)
    monkeypatch.setattr(engine, "_store_clip", fake_store_clip)
    assert engine.render(runstate.clip(cid)) is True
    row = runstate.clip(cid)
    assert row["state"] == "rendered" and len(calls) == 2
    assert gates[0] == pytest.approx(end) and row["end_s"] > END_OF["way."]
    feats = json.loads(row["features_json"])
    assert feats["recut"]["action"] == "extend" and feats["gate_report"]["recut"]["new_end"]
    assert row["words_path"].endswith("_recut.json")


def test_render_rejects_when_the_recut_still_fails(db, tmp_path, monkeypatch):
    from clipengine import reason_codes
    from clipengine import engine
    cid, _media = _clip_on_window(tmp_path, END_OF["word"] + 0.02)
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    monkeypatch.setattr(engine.transcript, "words_window", lambda m, a, b, model=None:
                        [w for w in WORDS if a <= w["start"] <= b])
    monkeypatch.setattr(engine.transcript, "load_audio", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("no audio in tests")))
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: _ending_fail())
    assert engine.render(runstate.clip(cid)) is False
    assert len(calls) == 2                                        # re-rendered once, no more
    assert runstate.clip(cid)["state"] == "failed"
    assert set(codes_for(cid)) == {"cut_end"}


def test_render_does_not_recut_other_failures_or_past_the_window_cap(db, tmp_path, monkeypatch):
    import time
    from clipengine import engine
    cid, _media = _clip_on_window(tmp_path, END_OF["word"] + 0.02)
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    # every subprocess.run is the fake recompose here: no audio read for the cut seat
    monkeypatch.setattr(engine, "cut_seat_enabled", lambda cfg=None: False)
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: {
        **_ending_fail(), "crop_slices": {"passed": False, "errors": ["sliced OCR box"]}})
    assert engine.render(runstate.clip(cid)) is False and len(calls) == 1
    cid2 = runstate.add_clip("W_w00", 30.0, 45.0, 0.7, "auto", hook="h")
    calls.clear()
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: _ending_fail())
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", time.monotonic() + 5.0)
    assert engine.render(runstate.clip(cid2)) is False and len(calls) == 1


def test_recut_transcription_is_not_capped_by_selects_spent_deadline(monkeypatch):
    """At render time transcript.CLI_DEADLINE (select's) is in the past: the re-cut's pass runs
    under its own cap within the window deadline, and the old deadline is put back."""
    import time
    from clipengine import engine
    seen = {}
    past = time.monotonic() - 100.0
    monkeypatch.setattr(engine.transcript, "CLI_DEADLINE", past)
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", time.monotonic() + 500.0)

    def words_window(media, lo, hi, model=None):
        seen["left"] = engine.transcript.CLI_DEADLINE - time.monotonic()
        seen["model"] = model
        return []
    monkeypatch.setattr(engine.transcript, "words_window", words_window)
    engine._recut_words("m.mp4", 0.0, 10.0)
    assert 0 < seen["left"] <= engine.RECUT_ASR_S and seen["model"] == engine.REFINE_MODEL
    assert engine.transcript.CLI_DEADLINE == past
