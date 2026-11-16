"""captions the gate's caption check fails are repaired, not rejected. Every
stretch where the burned captions (large-v3-turbo) and the second ASR (small) disagree gets a
tie-break vote: the source's YouTube subtitles (manual, else auto), else a third model on the
stretch +- 1 s; the majority's words go on the captions' timings, the clip is re-rendered once and
the full gate judges it again. Shapes are modelled on the 10-21/22 reason_codes: 623 (a repeat the
captions dropped: "where these elk are at, where these elk are at"), 558 (an opening the strong
pass never captioned), 627 ("It was a great show." unburned as crosstalk), 762 (captions that
YouTube confirms and the second ASR missed: they stay)."""
from __future__ import annotations

import importlib
import json
import subprocess

import pytest

from clipengine import caption_repair as cr
from clipengine import quality_gate
from clipengine import runstate
from test_stage1_cuts import _fake_recompose
from helpers import codes_for, fake_store_clip


def W(*triples, **extra):
    return [{"w": w, "start": s, "end": e, **extra} for w, s, e in triples]


def words_from(text, t0, step=0.3):
    out, t = [], float(t0)
    for tok in text.split():
        out.append({"w": tok, "start": round(t, 3), "end": round(t + step - 0.05, 3)})
        t += step
    return out


START = 100.0
SAID = ("So when you're hunting you wait for the wind to switch because where these elk are at, "
        "where these elk are at, the sun is hitting the side of this mountain and the thermals "
        "are rising from the bottom of the valley all day long.")
HEARD = words_from(SAID, 0.5)                         # the second ASR, clip-relative
# the captions dropped the repeat ("where these elk are at," x2 -> x1) and misheard "thermals"
CAPS = [dict(w, start=w["start"] + START, end=w["end"] + START)
        for k, w in enumerate(HEARD) if not 17 <= k <= 21]
CAPS = [dict(w, w="formals") if w["w"] == "thermals" else w for w in CAPS]
END = START + HEARD[-1]["end"] + 0.3
DUR = END - START


def _yt(words, offset=0.0):
    """A YouTube voter returning `words` (clip-relative) in window time."""
    return lambda src: ("auto", [dict(w, start=w["start"] + START + offset,
                                      end=w["end"] + START + offset) for w in words])


def _verdict(side):
    return quality_gate.caption_agreement_verdict(quality_gate.clip_words(side, START, END),
                                                  HEARD, DUR)


# ---------------------------------------------------------------- stretches + vote

def test_the_dropped_repeat_and_the_misheard_word_are_two_stretches():
    cap = [dict(w, start=w["start"] - START, end=w["end"] - START) for w in CAPS]
    sts = cr.stretches(cap, HEARD)
    assert [(s["cap_tokens"], s["asr_tokens"]) for s in sts] == [
        ([], ["where", "these", "elk", "are", "at"]), (["formals"], ["thermals"])]
    assert sts[0]["left"] == ["elk", "are", "at"] and sts[0]["right"] == ["the", "sun", "is"]


def test_the_majority_takes_the_second_asrs_words_on_the_captions_timings():
    assert _verdict(CAPS)["passed"] is False                     # the live failure
    new, rep = cr.reconcile(CAPS, START, END, HEARD, media="w.mp4", source={},
                            yt_fn=_yt(HEARD), third_fn=lambda *a: pytest.fail("no third pass"))
    assert rep["youtube"] == "auto" and [v["winner"] for v in rep["votes"]] == ["asr", "asr"]
    assert rep["verdict"]["passed"] is True and _verdict(new)["passed"] is True
    old = next(w for w in CAPS if w["w"] == "formals")
    fixed = next(w for w in new if w["w"] == "thermals")
    assert (fixed["start"], fixed["end"]) == (old["start"], old["end"])     # timings kept
    assert [f["how"] for f in rep["fixes"]] == ["inserted", "replaced"]
    assert rep["fixes"][1]["old"] == "formals" and rep["fixes"][1]["new"] == "thermals"
    lines = cr.fix_lines(7, rep)
    assert len(lines) == 2 and "'formals' -> 'thermals'" in lines[1] and "youtube-auto" in lines[1]


def test_a_vote_for_the_captions_keeps_them():
    """762: YouTube hears what the captions say; the second ASR missed it. Nothing changes."""
    said = [dict(w, start=w["start"] - START, end=w["end"] - START) for w in CAPS]
    new, rep = cr.reconcile(CAPS, START, END, HEARD, media="w.mp4", source={},
                            yt_fn=_yt(said), third_fn=lambda *a: pytest.fail("no third pass"))
    assert new is None and rep["reason"] == "no stretch went to the second ASR"
    assert [v["winner"] for v in rep["votes"]] == ["caption", "caption"]


def test_no_subtitles_the_third_model_hears_each_stretch_with_a_second_around_it():
    heard_spans = []

    def third(media, lo, hi, timeout):
        heard_spans.append((lo, hi))
        return [dict(w, start=w["start"] + START, end=w["end"] + START) for w in HEARD
                if lo <= w["start"] + START <= hi]
    new, rep = cr.reconcile(CAPS, START, END, HEARD, media="w.mp4", source={},
                            yt_fn=lambda src: (None, []), third_fn=third)
    assert rep["verdict"]["passed"] is True
    assert {v["source"] for v in rep["votes"]} == {cr.settings()["tiebreak_model"]}
    st = cr.stretches([dict(w, start=w["start"] - START, end=w["end"] - START) for w in CAPS],
                      HEARD)
    lo = min(s["ctx_lo"] for s in st) + START - cr.VOTE_PAD_S
    assert heard_spans and heard_spans[0][0] == pytest.approx(lo, abs=1e-6)


def test_a_third_reading_is_no_majority():
    """The tie-break heard something else entirely: the captions stay."""
    st = {"left": ["a"], "right": ["b"], "cap_tokens": ["formals"], "asr_tokens": ["thermals"]}
    assert cr.vote(st, ["a", "warmth", "b"])[0] is None
    assert cr.vote(st, ["a", "thermals", "b"])[0] == "asr"
    assert cr.vote(st, ["a", "formals", "b"])[0] == "caption"
    assert cr.vote(st, [])[0] is None


def test_crosstalk_flagged_words_the_vote_confirms_are_burned_again():
    """627: 'It was a great show.' was unburned as crosstalk; the second ASR and the vote heard it."""
    said = words_from("Thank you very much. It was a great show. He got blackballed for it.", 0.5)
    side = [dict(w, start=w["start"] + START, end=w["end"] + START) for w in said]
    for w in side[4:9]:
        w["cap"] = False
    end = START + said[-1]["end"] + 0.3
    new, rep = cr.reconcile(side, START, end, said, media="w.mp4", source={},
                            yt_fn=_yt(said), third_fn=lambda *a: pytest.fail("no third pass"))
    assert [f["how"] for f in rep["fixes"]] == ["unflagged"] and rep["verdict"]["passed"] is True
    assert all(w.get("cap", True) is not False for w in new)
    assert [w["start"] for w in new] == [w["start"] for w in side]       # their own timings


def test_caption_words_nobody_else_heard_stay_unburned():
    said = words_from("We drove out to the desert that night and nobody said a word all the "
                      "way there until the lights.", 0.5)
    ghost = W(("Go,", 7.2, 7.4), ("go,", 7.4, 7.6), ("go,", 7.6, 7.8), ("go.", 7.8, 8.0))
    side = sorted([dict(w, start=w["start"] + START, end=w["end"] + START) for w in said + ghost],
                  key=lambda w: w["start"])
    end = START + 8.3
    new, rep = cr.reconcile(side, START, end, said, media="w.mp4", source={},
                            yt_fn=_yt(said), third_fn=lambda *a: pytest.fail("no third pass"))
    assert [f["how"] for f in rep["fixes"]] == ["unburned"]
    assert [w["w"] for w in new if w.get("cap") is False] == ["Go,", "go,", "go,", "go."]


# ---------------------------------------------------------------- the YouTube voter

AUTO = {"events": [
    {"tStartMs": 1000, "dDurationMs": 3000, "segs": [
        {"utf8": ">> where"}, {"utf8": " these", "tOffsetMs": 300}, {"utf8": " elk", "tOffsetMs": 600}]},
    {"tStartMs": 3990, "dDurationMs": 10, "aAppend": 1, "segs": [{"utf8": "\n"}]},
    {"tStartMs": 4000, "dDurationMs": 1000, "segs": [{"utf8": "[Music]"}]}]}
MANUAL = {"events": [{"tStartMs": 2000, "dDurationMs": 2000, "segs": [{"utf8": "Where these elk"}]}]}


def test_json3_auto_words_are_timed_manual_cues_share_their_span():
    auto = cr.parse_json3(AUTO)
    assert [(w["w"], w["start"]) for w in auto] == [("where", 1.0), ("these", 1.3), ("elk", 1.6)]
    man = cr.parse_json3(MANUAL)
    assert [w["w"] for w in man] == ["Where", "these", "elk"]
    assert man[0]["start"] == 2.0 and man[-1]["end"] == pytest.approx(4.0)


def test_youtube_words_manual_first_cached_per_video_in_window_time(tmp_path):
    calls = []

    def runner(url, flags, langs):
        calls.append(flags[0])
        return {"s.en.json3": json.dumps(MANUAL)} if flags == ["--write-subs"] else {}
    src = {"source_id": "abcdefghijk_w02", "url": "https://www.youtube.com/watch?v=abcdefghijk",
           "local_path": "/x/abcdefghijk_1500-61500.mp4"}
    kind, words = cr.youtube_words(src, tmp_path, runner=runner)
    assert kind == "manual" and calls == ["--write-subs"]
    assert words[0]["start"] == pytest.approx(0.5)                     # 2.0 s VOD - 1.5 s offset
    kind2, _ = cr.youtube_words(src, tmp_path, runner=lambda *a: pytest.fail("not cached"))
    assert kind2 == "manual" and (tmp_path / "abcdefghijk.json").exists()


def test_auto_subtitles_when_there_are_no_manual_ones_and_none_for_twitch(tmp_path):
    def runner(url, flags, langs):
        return {"s.en-orig.json3": json.dumps(AUTO)} if flags == ["--write-auto-subs"] else {}
    src = {"source_id": "abcdefghijk", "url": "https://youtu.be/abcdefghijk", "local_path": "/x/a.mp4"}
    assert cr.youtube_words(src, tmp_path, runner=runner)[0] == "auto"
    tw = {"source_id": "tw_1_w00", "url": "https://www.twitch.tv/videos/1",
          "local_path": "/x/tw_1_0-60000.mp4"}
    assert cr.youtube_words(tw, tmp_path, runner=lambda *a: pytest.fail("no fetch")) == (None, [])


def test_the_tiebreak_model_is_neither_a_caption_model_nor_the_second_asr():
    from clipengine import second_asr
    assert cr.independent_tiebreak(cr.settings()["tiebreak_model"]) is None
    assert cr.independent_tiebreak("large-v3-turbo") is not None
    assert cr.independent_tiebreak(second_asr.settings()["model"]) is not None


# ---------------------------------------------------------------- engine.render (Amendment 1)

CAPTION_ERR = "uncaptioned stretch: 58% of 12 heard words captioned ('where these elk')"


def _caption_fail(**extra):
    """The engine's gate report for a clip whose captions alone disagree (Amendment 1: the
    caption check is a warning, the verdict is the other checks')."""
    rep = {"verdict": "pass", "drop_reason": "", "caption_warning": CAPTION_ERR,
           "caption_asr": {"passed": False, "errors": [CAPTION_ERR]}}
    rep.update(extra)
    return rep


def test_a_failed_caption_check_never_rejects_a_clip():
    """: the gate still measures the captions; the engine reads it as a warning."""
    from clipengine import engine
    gate = {"caption_asr": {"passed": False, "errors": [CAPTION_ERR]},
            "frame_gate": {"passed": True}, "ending": {"passed": True}}
    rep = engine._captions_never_reject({**gate, "verdict": "fail",
                                         "drop_reason": f"caption_asr: {CAPTION_ERR}"})
    assert rep["verdict"] == "pass" and rep["drop_reason"] == ""
    assert rep["caption_warning"] == CAPTION_ERR and rep["caption_asr"]["passed"] is False
    both = engine._captions_never_reject({**gate, "ending": {"passed": False, "errors": ["x"]},
                                          "verdict": "fail"})
    assert both["verdict"] == "fail" and both["drop_reason"] == "ending: x"
    assert engine.non_caption_codes(both) == {"cut_end"}
    assert engine.recut_sides(both) == (["end"], None)          # captions never block a re-cut


def test_the_lean_stream_gate_reads_captions_as_a_warning_too(monkeypatch, tmp_path):
    from clipengine import engine
    from clipengine import quality_gate
    hay = {k: {"passed": True} for k in quality_gate.CLIP_CHECK_KEYS}
    hay["caption_asr"] = {"passed": False, "errors": [CAPTION_ERR]}
    monkeypatch.setattr(quality_gate, "clip_checks", lambda *a, **k: dict(hay))
    monkeypatch.setattr(quality_gate, "measure_visual_quality", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(engine, "_measure_black_frozen", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(engine, "_hook_judge", lambda: None)
    clip = {"start_s": 1.0, "end_s": 21.0, "hook": "h"}
    rep = engine._run_stream_lean_gate(tmp_path / "x.mp4", clip, {"integrated_lufs": -14.0},
                                       [{"w": "hi", "start": 1.0, "end": 3.0}])
    assert rep["verdict"] == "pass" and rep["caption_warning"] == CAPTION_ERR


@pytest.fixture
def db(tmp_path, monkeypatch):
    runstate.reset()
    from clipengine import reason_codes
    from clipengine import engine
    monkeypatch.setattr(engine, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(engine, "CAPTION_STRONG_MODEL", None)
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", None)
    monkeypatch.setattr(engine, "_store_clip", fake_store_clip)
    from clipengine import second_asr
    monkeypatch.setattr(second_asr, "transcribe",
                        lambda media, a, b, **k: {"words": HEARD, "cached": True})
    runstate.init()
    return tmp_path


def _clip(tmp_path):
    media = tmp_path / "abcdefghijk_w00.mp4"
    media.write_bytes(b"\0")
    runstate.add_source("abcdefghijk_w00", local_path=str(media), content_type="auto",
                     vertical="podcast", url="https://www.youtube.com/watch?v=abcdefghijk")
    runstate.update_source("abcdefghijk_w00", state="selected", duration_s=200.0)
    wp = tmp_path / "side.json"
    wp.write_text(json.dumps(CAPS))
    return runstate.add_clip("abcdefghijk_w00", START, END, 0.7, "auto", hook="h", words_path=str(wp))


def _votes(monkeypatch, said):
    real = cr.reconcile
    monkeypatch.setattr(cr, "reconcile", lambda *a, **k: real(
        *a, **{**k, "yt_fn": _yt(said), "third_fn": lambda *x: pytest.fail("no third pass")}))


def test_render_burns_the_best_vote_captions_and_ships_the_pass(db, tmp_path, monkeypatch):
    from clipengine import engine
    cid = _clip(tmp_path)
    calls, gates = [], []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    _votes(monkeypatch, HEARD)

    def gate(out, clip, qc=None):
        side = engine._load_words_sidecar(clip["words_path"])
        gates.append(_verdict(side)["passed"])
        return _caption_fail() if len(gates) == 1 else {"verdict": "pass"}
    monkeypatch.setattr(engine, "_run_quality_gate", gate)
    assert engine.render(runstate.clip(cid)) is True
    row = runstate.clip(cid)
    assert row["state"] == "rendered" and len(calls) == 2 and gates == [False, True]
    first, second = calls
    assert first[:6] == second[:6]                                    # same window and cut
    assert second[6].endswith("_capfix.mp4")                          # rendered beside the first
    flags = lambda c: [a for a in c[7:] if not a.endswith(".json")]  # noqa: E731
    assert flags(first) == flags(second)                              # same layout: only the
    assert second[second.index("--words") + 1].endswith("_capfix.json")  # sidecar differs
    assert row["words_path"].endswith("_capfix.json")
    feats = json.loads(row["features_json"])
    assert [f["how"] for f in feats["caption_repair"]["captions"]["fixes"]] == ["inserted",
                                                                               "replaced"]
    assert feats["gate_report"]["caption_repair"]["fixes"] and "caption_unresolved" not in feats


def test_captions_the_vote_cannot_settle_ship_as_caption_unresolved(db, tmp_path, monkeypatch):
    """762: YouTube confirms the captions; the second ASR still disagrees. No reject: the clip
    ships with its captions (the best vote) and a caption_unresolved warning."""
    from clipengine import reason_codes
    from clipengine import engine
    cid = _clip(tmp_path)
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    _votes(monkeypatch, [dict(w, start=w["start"] - START, end=w["end"] - START) for w in CAPS])
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: _caption_fail())
    assert engine.render(runstate.clip(cid)) is True
    row = runstate.clip(cid)
    assert len(calls) == 1 and row["state"] == "rendered"            # nothing to burn again
    assert codes_for(cid) == {}
    feats = json.loads(row["features_json"])
    assert feats["caption_unresolved"]["warning"] == CAPTION_ERR
    assert feats["caption_repair"]["captions"]["reason"] == "no stretch went to the second ASR"


def test_a_caption_rerender_that_fails_another_check_keeps_the_first_render(db, tmp_path,
                                                                            monkeypatch):
    from clipengine import engine
    cid = _clip(tmp_path)
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    _votes(monkeypatch, HEARD)
    reps = [_caption_fail(), {"verdict": "fail", "drop_reason": "frame_gate: flaky",
                              "frame_gate": {"passed": False}}]
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: reps.pop(0))
    assert engine.render(runstate.clip(cid)) is True
    row = runstate.clip(cid)
    assert len(calls) == 2 and row["state"] == "rendered"
    assert row["words_path"].endswith("side.json")                    # the first render's
    assert not list((tmp_path / "scratch").glob("*_capfix.mp4"))
    assert json.loads(row["features_json"])["caption_unresolved"]


def test_a_clip_is_rejected_for_its_other_checks_only(db, tmp_path, monkeypatch):
    from clipengine import reason_codes
    from clipengine import engine
    cid = _clip(tmp_path)
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_recompose(calls))
    monkeypatch.setattr(cr, "reconcile", lambda *a, **k: pytest.fail("not tried"))
    rep = _caption_fail(verdict="fail", drop_reason="crop_slices: the crop slices burned-in text",
                        crop_slices={"passed": False, "errors": ["the crop slices burned-in text"]})
    assert engine.recaption_sides(rep) == (None, "gate failed on edge_slice too")
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: rep)
    assert engine.render(runstate.clip(cid)) is False and len(calls) == 1
    assert "edge_slice" in codes_for(cid)
    assert runstate.clip(cid)["error"] == "gate: crop_slices: the crop slices burned-in text"


def test_a_repair_never_leaves_the_captions_ending_on_a_dangling_word():
    """558: the second ASR (and YouTube) heard the next sentence's 'A' just before the cut; taking
    it would end the captions on 'A', which the gate's ending check fails."""
    said = words_from("Take three years off our life. Oh you need a whisker.", 0.5)
    tail = W(("A", said[-1]["end"] + 0.4, said[-1]["end"] + 0.5))
    side = [dict(w, start=w["start"] + START, end=w["end"] + START) for w in said]
    del side[5]                                        # 'Oh' never captioned: a real fix
    end = START + tail[0]["end"] + 0.05
    heard = said + tail
    new, rep = cr.reconcile(side, START, end, heard, media="w.mp4", source={},
                            yt_fn=_yt(heard), third_fn=lambda *a: pytest.fail("no third pass"))
    burned = [w["w"] for w in quality_gate.clip_words(new, START, end)]
    assert burned[-1] == "whisker." and "Oh" in burned
    assert any(v.get("note", "").startswith("left out") for v in rep["votes"])


def test_a_short_caption_span_never_crams_the_second_asrs_stretch():
    """11-10 tw_2916500966 1100002 burned the second ASR's 15 s of dialogue over
    the captions' one 0.3 s word: the stretch keeps the second ASR's own times."""
    side = [{"w": "Hey.", "start": 10.0, "end": 10.3}, {"w": "Okay,", "start": 20.0, "end": 20.3},
            {"w": "Bye.", "start": 32.0, "end": 32.3}]
    asr = [{"w": f"w{i}", "start": 5.0 + i * 0.5, "end": 5.3 + i * 0.5} for i in range(20)]
    st = {"cap": [1], "asr": list(range(20)), "lo": 5.0, "hi": 15.0}
    out, fixes = cr.apply_votes(side, [0, 1, 2], asr, [(st, "asr")], 10.0)
    new = [w for w in out if w.get("fix") == "asr2"]
    assert len(new) == 20 and new[0]["start"] == 15.0 and new[-1]["end"] == 24.8
    assert fixes[0]["how"] == "replaced" and fixes[0]["timing"] == "asr2"
    assert all(w["end"] - w["start"] >= 0.29 for w in new)
