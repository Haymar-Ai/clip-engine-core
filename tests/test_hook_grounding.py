"""The gate's side of hooks:

  * a hook whose semantic judge was skipped for the window's time cap is deferred (no verdict),
    not failed on the lexical floor;
  * _stem: "water" / "watering", "step" / "stepping" meet;
  * HOOK_GROUND_PROMPT: a specific fact nobody says fails (a price, a job, a group nobody in the
    clip mentions).
"""
import pytest

from clipengine import quality_gate as qg


def W(text, t0, step=0.4):
    return [{"w": w, "start": round(t0 + i * step, 3), "end": round(t0 + i * step + 0.3, 3)}
            for i, w in enumerate(text.split())]


SKIPPED = {"error": "window cap near (-0 s left): judge not asked", "skipped": "time"}


# ---------------------------------------------------------------- _stem
@pytest.mark.parametrize("a,b", [("water", "watering"), ("step", "stepping"),
                                 ("stop", "stopped"), ("streamer", "streamers"),
                                 ("party", "parties"), ("agree", "agreed")])
def test_stems_meet_across_suffixes(a, b):
    assert qg._stem(a) & qg._stem(b), (a, b)


@pytest.mark.parametrize("a,b", [("water", "watts"), ("matter", "mats"), ("step", "steam"),
                                 ("card", "car")])
def test_stems_do_not_overreach(a, b):
    assert not (qg._stem(a) & qg._stem(b)), (a, b)


def test_the_two_stem_misses_now_ground():
    # "... the empty watering can": the clip says "water"
    toks = qg.norm_tokens(W("thats my intuition saying water I need the can", 0.0))
    _s, matched, _c = qg.hook_grounding("Gardener panics over the empty watering can", toks)
    assert "watering" in matched
    # "Stepping in ...": the clip says "step in"
    toks = qg.norm_tokens(W("I had to step in between them", 0.0))
    _s, matched, _c = qg.hook_grounding("Stepping in to stop the fight goes wrong", toks)
    assert "stepping" in matched


# ---------------------------------------------------------------- the judge skipped for time
CLIP = W("chat is going crazy over what I am wearing today honestly", 0.0)


def test_a_judge_skipped_for_time_defers_a_missed_floor():
    v = qg.hook_support_verdict("Viewers roast the streamer's jacket", CLIP, [], 20.0,
                                judge=lambda h, t: dict(SKIPPED))
    assert v["passed"] is None and v["decided_by"] == "deferred" and v["errors"] == []
    assert "eye-check" in v["reason"] and v["judge"]["skipped"] == "time"
    # the stored reports' judge answers carry only the error text: the same
    v = qg.hook_support_verdict("Viewers roast the streamer's jacket", CLIP, [], 20.0,
                                judge=lambda h, t: {"error": SKIPPED["error"]})
    assert v["passed"] is None and v["decided_by"] == "deferred"


def test_a_judge_skipped_for_time_with_the_floor_met_still_passes():
    v = qg.hook_support_verdict("Chat goes crazy over his outfit", CLIP, [], 20.0,
                                judge=lambda h, t: dict(SKIPPED))
    assert v["passed"] is True and v["decided_by"] == "lexical"


def test_other_judge_faults_and_answers_are_unchanged():
    hook = "Viewers roast the streamer's jacket"
    v = qg.hook_support_verdict(hook, CLIP, [], 20.0, judge=lambda h, t: {"error": "no JSON"})
    assert v["passed"] is False and v["decided_by"] == "lexical"
    v = qg.hook_support_verdict(hook, CLIP, [], 20.0)
    assert v["passed"] is False                                     # no judge at all: the floor
    no = lambda h, t: {"supported": False, "quote": "", "reason": "nobody says jacket"}
    v = qg.hook_support_verdict("Chat goes crazy over his outfit", CLIP, [], 20.0, judge=no)
    assert v["passed"] is False and v["decided_by"] == "judge"
    v = qg.hook_support_verdict(None, CLIP, [], 20.0, judge=lambda h, t: dict(SKIPPED))
    assert v["passed"] is False and "no hook" in v["errors"][0]     # a missing hook still fails


def test_the_judge_says_when_it_was_skipped_for_time(monkeypatch):
    from clipengine import second_asr
    from clipengine import select_semantic
    monkeypatch.setattr(second_asr, "time_left", lambda: 6.0)
    monkeypatch.setattr(select_semantic, "_claude_cli",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("not asked")))
    v = qg.make_hook_ground_judge({"semantic_model": "opus"})("A hook", "a transcript")
    assert v["skipped"] == "time" and "judge not asked" in v["error"]
    assert qg.judge_skipped_for_time(v) and not qg.judge_skipped_for_time({"error": "x"})
    assert not qg.judge_skipped_for_time({"supported": True}) and not qg.judge_skipped_for_time(None)


def test_a_deferred_hook_does_not_fail_the_gate():
    rep = {"hook_grounding": {"passed": None, "decided_by": "deferred", "errors": []},
           "frame_gate": {"passed": True}}
    assert qg.decide(rep) == ("pass", "")


# ---------------------------------------------------------------- the judge's prompt
def test_the_prompt_fails_a_specific_fact_nobody_says(monkeypatch):
    from clipengine import select_semantic
    seen = {}

    def fake_cli(prompt, model, timeout, opts):
        seen["prompt"] = prompt
        return '{"supported": false, "quote": "", "reason": "nobody says a price"}'
    monkeypatch.setattr(select_semantic, "_claude_cli", fake_cli)
    v = qg.make_hook_ground_judge({"semantic_model": "opus"})(
        "He pulled the $25,000 rookie card", "oh my god it's the numbered rookie auto")
    assert v["supported"] is False
    p = seen["prompt"]
    assert "NOT MENTIONED" in p and "a number or an" in p and "a person's role or job" in p
    assert "simply does not mention is NOT a failure" not in p
    assert "WRONG SUBJECT" in p and "CONTRADICTED" in p and "OFF TOPIC" in p
    # a description of the action still passes (gameplay hooks)
    assert "a plain description of the action passes" in p
