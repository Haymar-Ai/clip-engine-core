"""topic map first (clipengine/topic_map.py), the hard "drop the trailing wrap-up" end rule
(boundaries.wrapup_sentence, select's cuts and the cut review), and the per-niche selection prompt packs
(clipengine/prompt_packs). Offline: the outline / pick / editor LLMs, the transcript and the audio are
seams. The windows are modelled on 11-07..09 reason_codes: a clip that drifts into the next topic
("Anyway, what game are we playing tonight?") and one that closes on a tag ("Yeah, you know what I'm
saying")."""
from __future__ import annotations

import importlib
import json
import subprocess
import time

import numpy as np
import pytest

from clipengine import boundaries
from clipengine import runstate
from clipengine import topic_map


def W(text, t0, step=0.4):
    return [{"w": w, "start": round(t0 + i * step, 3), "end": round(t0 + i * step + 0.3, 3)}
            for i, w in enumerate(text.split())]


# two topics with a wrap-up and a turn between them (a hard pause after every sentence)
SENT = ["He started the company in his garage.",                 # S0  T1
        "Nobody believed it would work at all.",                 # S1  T1
        "The first investor said it was a toy.",                 # S2  T1
        "Ten years later it was worth a billion dollars.",       # S3  T1
        "So yeah, that's the story.",                            # S4  wrap-up
        "Anyway, what game are we playing tonight?",             # S5  the turn
        "We are playing the new horror game tonight.",           # S6  T2
        "It is supposed to be the scariest game this year.",     # S7  T2
        "The last streamer who played it screamed for an hour.",  # S8  T2
        "Chat says I will not last ten minutes in it.",          # S9  T2
        "Yeah, you know what I'm saying."]                       # S10 tag


def window_words(sents=SENT, gap=1.6):
    out, t = [], 1.0
    for s in sents:
        for w in s.split():
            out.append({"w": w, "start": round(t, 3), "end": round(t + 0.3, 3)})
            t += 0.4
        t += gap
    return out


WORDS = window_words()
SENTS = boundaries.sentences(WORDS, gap_s=0.7)
OUTLINE = {"topics": [{"s0": 0, "s1": 3, "title": "Garage company worth a billion", "continues": False},
                      {"s0": 6, "s1": 9, "title": "The scariest game this year", "continues": False}]}
VC = {"min_dur_s": 5, "max_dur_s": 60, "lead_pad_s": 0.15, "tail_pad_s": 0.35}
RULE = {**VC, "drop_wrapup": True}
# yield 11-11: the prompt-side rules are switched on separately (off by default)
RULE_PROMPTS = {**RULE, "end_rule_prompt": True, "topic_editor_rule": True}


def topics():
    return topic_map.validate(OUTLINE, SENTS)


# ---------------------------------------------------------------- the end rule
@pytest.mark.parametrize("text", [
    "Anyway.", "Anyways, let's get back to it.", "So anyway, what else?", "But yeah.", "So yeah, that's the story.",
    "Moving on.", "Next question.", "Speaking of which, have you seen this?", "Thanks for watching!",
    "Don't forget to subscribe.", "Link in the description.", "Yeah, you know what I'm saying.",
    "You know what I mean?", "Yeah.", "Okay so.", "All right.", "We'll be right back.", "See you guys tomorrow.",
    "That's that."])
def test_wrapups_are_recognised(text):
    assert boundaries.wrapup_sentence(text)


@pytest.mark.parametrize("text", [
    "Yeah!", "YEAH!!", "That's that guy from the show.", "So they kicked him out for life.",
    "He said anyway he would go.", "Next time he will win.", "Right before the river came.",
    "You know what, I'm going all in.", "Ten years later it was worth a billion dollars.", ""])
def test_points_reactions_and_lines_that_go_on_are_not_wrapups(text):
    assert not boundaries.wrapup_sentence(text)


def test_a_niche_pack_adds_its_own_openers():
    assert not boundaries.wrapup_sentence("Next hand, Hellmuth opens.")
    assert boundaries.wrapup_sentence("Next hand, Hellmuth opens.", extra=("next hand",))


def test_select_drops_a_picks_trailing_wrapup():
    c = boundaries.clean_clip(WORDS, SENTS, 1, 4, cfg=RULE)
    assert c["s1"] == 3 and c["wrapup_dropped"] == ["So yeah, that's the story."]
    old = boundaries.clean_clip(WORDS, SENTS, 1, 4, cfg=VC)              # rule off: today's cut
    assert old["s1"] == 4 and "wrapup_dropped" not in old


def test_a_too_short_pick_never_grows_past_a_wrapup():
    # S3 alone is under 5 s: today it grows into the wrap-up; with the rule it may not
    c = boundaries.clean_clip(WORDS, SENTS, 3, 3, cfg={**VC, "min_dur_s": 8})
    assert c["s1"] >= 4
    assert boundaries.clean_clip(WORDS, SENTS, 3, 3, cfg={**RULE, "min_dur_s": 8}) is None


def test_a_tag_line_is_not_taken_as_a_tail_tag():
    words = window_words(["He went all in on the river.", "You know what I mean?"], gap=0.2)
    sents = boundaries.sentences(words, gap_s=0.7)
    assert boundaries.clean_clip(words, sents, 0, 0, cfg={**VC, "min_dur_s": 1})["s1"] == 1
    assert boundaries.clean_clip(words, sents, 0, 0, cfg={**RULE, "min_dur_s": 1})["s1"] == 0


def test_snap_never_extends_onto_a_wrapup_or_out_of_the_topic():
    # a cut ending inside S3 ("...worth a billion"): snap extends to S3's end, never on to S4 / S5
    end_in = SENTS[3]["end"] - 1.0
    r = boundaries.snap_clip(WORDS, SENTS[1]["start"] - 0.1, end_in, cfg=RULE, edges_clean=True)
    assert r["end"] <= SENTS[3]["end"] + 1.0
    # ending on the wrap-up S4 itself pulls back to S3
    r = boundaries.snap_clip(WORDS, SENTS[1]["start"] - 0.1, SENTS[4]["end"] + 0.2, cfg=RULE)
    assert r["snap"]["end"] == "pull_back" and r["end"] < SENTS[4]["start"]
    t1 = topics()[0]
    r = boundaries.snap_clip(WORDS, SENTS[2]["start"] + 0.5, SENTS[3]["end"] + 0.2,
                             cfg={**VC, "span_s": (t1["lo"], t1["hi"])})
    assert r["start"] >= t1["lo"]


def test_edge_verify_takes_no_wrapup_reaction_and_stays_in_the_topic():
    from clipengine import edge_verify
    end = SENTS[3]["end"] + 0.35
    # the line right after the cut is a short "So yeah." (a reaction to edge-verify)
    words = [w for w in WORDS if w["start"] < end] + [
        {"w": "So", "start": end + 0.1, "end": end + 0.3}, {"w": "yeah.", "start": end + 0.35, "end": end + 0.6}]
    fn = lambda lo, hi: [w for w in words if lo <= w["start"] < hi]                   # noqa: E731
    clip = {"start": SENTS[1]["start"] - 0.15, "end": end}
    assert edge_verify.verify(clip, fn, VC)["action"]["end"] == "extend"
    assert edge_verify.verify(clip, fn, RULE)["action"]["end"] == "noop"
    t1 = topics()[0]
    assert edge_verify.verify(clip, fn, {**VC, "span_s": (t1["lo"], end)})["action"]["end"] == "noop"


# ---------------------------------------------------------------- the map
def test_validate_orders_topics_fixes_overlaps_and_spans_the_pauses():
    data = {"topics": [{"s0": 6, "s1": 9, "title": "B"}, {"s0": 0, "s1": 6, "title": "A", "continues": True},
                       {"s0": "S2", "s1": 1}, {"s0": 3}, "x", {"s0": 9, "s1": 99}]}
    ts = topic_map.validate(data, SENTS)
    assert [(t["s0"], t["s1"], t["title"]) for t in ts] == [(0, 6, "A"), (7, 9, "B")]
    assert ts[0]["continues"] is True and ts[1]["continues"] is False
    assert ts[0]["lo"] == pytest.approx(max(0.0, SENTS[0]["start"] - topic_map.EDGE_PAD_S))
    assert ts[0]["hi"] == pytest.approx((SENTS[6]["end"] + SENTS[7]["start"]) / 2)
    assert ts[1]["hi"] == pytest.approx(SENTS[9]["end"] + 0.5 * (SENTS[10]["start"] - SENTS[9]["end"]))
    assert topic_map.validate({"topics": []}, SENTS) == []


def test_fit_puts_a_pick_inside_the_topic_it_overlaps_most():
    ts = topics()
    t, a, b, note = topic_map.fit(1, 7, SENTS, ts)              # S1-S7 runs over the turn into T2
    assert t["i"] == 1 and (a, b) == (1, 3) and "end S7->S3" in note
    t, a, b, note = topic_map.fit(3, 8, SENTS, ts)              # ... mostly T2's lines
    assert t["i"] == 2 and (a, b) == (6, 8)
    t, a, b, note = topic_map.fit(5, 9, SENTS, ts)              # opens on the turn
    assert t["i"] == 2 and (a, b) == (6, 9) and "start S5->S6" in note
    assert topic_map.fit(6, 8, SENTS, ts)[3] is None            # already inside
    assert topic_map.fit(4, 5, SENTS, ts)[0] is None            # between the topics: no topic
    t, a, b, _ = topic_map.fit(4, 5, SENTS, ts, near_s=5.0)     # a chat peak there: the nearest
    assert t is not None and a == b


def test_the_prompt_block_names_the_topics_and_the_lines_in_none():
    block = topic_map.prompt_block(topics(), SENTS)
    assert 'T1 S0-S3 "Garage company worth a billion"' in block and "T2 S6-S9" in block
    assert "In no topic: S4, S5, S10" in block and "ONE topic" in block


class LLM:
    def __init__(self, *texts):
        self.texts, self.prompts = list(texts), []
        self.model, self.backend = "opus", "claude_cli"

    def __call__(self, prompt, timeout):
        self.prompts.append(prompt)
        t = self.texts[min(len(self.prompts) - 1, len(self.texts) - 1)]
        if isinstance(t, Exception):
            raise t
        return t


ON = dict(topic_map.DEFAULTS, enabled=True)


def test_for_source_asks_once_and_caches_per_source(tmp_path):
    llm = LLM("Here it is:\n```json\n" + json.dumps(OUTLINE) + "\n```")
    m, rec = topic_map.for_source("abc_w01", SENTS, "podcasts", "podcast", VC, cache_dir=tmp_path,
                                  llm=llm, st=ON)
    assert rec["outcome"] == "answered" and [t["i"] for t in m["topics"]] == [1, 2]
    assert "mapping the structure of a podcast / talk transcript" in llm.prompts[0]
    assert "A topic is one question and its whole answer" in llm.prompts[0]       # the podcasts pack
    assert "[S5 " in llm.prompts[0] and "EXCLUDE the wrap-up" in llm.prompts[0]
    m2, rec2 = topic_map.for_source("abc_w01", SENTS, "podcasts", "podcast", VC, cache_dir=tmp_path,
                                    llm=LLM(RuntimeError("must not be asked")), st=ON)
    assert rec2["outcome"] == "cached" and m2["topics"] == m["topics"]
    # a re-transcribed window (other sentences) asks again
    other = boundaries.sentences(window_words(SENT[:8]), gap_s=0.7)
    m3, rec3 = topic_map.for_source("abc_w01", other, "podcasts", "podcast", VC, cache_dir=tmp_path,
                                    llm=LLM(json.dumps(OUTLINE)), st=ON)
    assert rec3["outcome"] == "answered"
    assert topic_map.cached("abc_w01", tmp_path)["n_sents"] == 8


@pytest.mark.parametrize("text, outcome", [
    (RuntimeError("claude exited 1"), "error"), ("no json", "garbage"),
    (json.dumps({"topics": [{"s0": 40, "s1": 50}]}), "empty")])
def test_no_map_when_the_outline_fails(tmp_path, text, outcome):
    m, rec = topic_map.for_source("x_w00", SENTS, None, "general", VC, cache_dir=tmp_path,
                                  llm=LLM(text), st=ON)
    assert m is None and rec["outcome"] == outcome
    again = LLM(json.dumps(OUTLINE))
    m2, rec2 = topic_map.for_source("x_w00", SENTS, None, "general", VC, cache_dir=tmp_path, llm=again, st=ON)
    # an error is asked again; a verdict (an empty map) is cached
    assert (rec2["outcome"], len(again.prompts)) == (("cached", 0) if outcome == "empty" else ("answered", 1))


def test_off_short_and_late_ask_nothing(tmp_path, monkeypatch):
    no = LLM(RuntimeError("must not be asked"))
    assert topic_map.for_source("x", SENTS, None, "general", VC, cache_dir=tmp_path, llm=no,
                                st={**ON, "enabled": False})[1]["outcome"] == "off"
    assert topic_map.for_source("x", SENTS[:2], None, "general", VC, cache_dir=tmp_path, llm=no,
                                st=ON)[1]["outcome"] == "short"
    assert topic_map.for_source("x", SENTS, None, "general", VC, cache_dir=tmp_path, llm=no, st=ON,
                                deadline=time.monotonic() + 5.0)[1]["outcome"] == "no_time"
    assert no.prompts == []
    monkeypatch.setenv("TOPIC_MAP", "0")
    assert topic_map.settings()["enabled"] is False
    monkeypatch.delenv("TOPIC_MAP")
    assert topic_map.settings({"select": {"topic_map": {"enabled": True}}})["enabled"] is True


# ---------------------------------------------------------------- prompt packs
NICHES = ("podcasts", "streamers", "tech", "lifestyle", "gaming", "poker", "finance")


@pytest.mark.parametrize("niche", NICHES)
def test_every_niche_has_a_pack(niche):
    p = topic_map.pack(niche)
    assert p["niche"] == niche
    for k in ("kind", "rubric", "outline", "review"):
        assert isinstance(p[k], str) and len(p[k]) > 40 or k == "kind"
    assert all(isinstance(w, str) and w for w in p["wrapup"])
    assert topic_map.select_cfg(niche)["wrapup_extra"] == tuple(p["wrapup"])


def test_packs_are_off_for_unknown_niches_and_by_switch(monkeypatch):
    assert topic_map.pack(None) is None and topic_map.pack("unknownniche") is None
    assert topic_map.pack("../config") is None
    monkeypatch.setenv("PROMPT_PACKS", "0")
    assert topic_map.pack("podcasts") is None
    monkeypatch.setenv("END_RULE", "0")
    assert topic_map.select_cfg("podcasts") == {"drop_wrapup": False, "end_rule_prompt": False,
                                                "topic_editor_rule": False, "wrapup_extra": ()}


def test_the_pick_prompt_is_todays_with_every_switch_off():
    from clipengine import select_semantic as ss
    base = ss.build_prompt(SENTS, "podcast", 3, 15, 75)
    assert "TOPICS" not in base and "END RULE" not in base
    assert base == ss.PROMPT.format(rubric=ss.RUBRICS["podcast"], k=3, mind=15, maxd=75, end_rule="",
                                    topics="", body="\n".join(ss._line(s) for s in SENTS))
    rich = ss.build_prompt(SENTS, "podcast", 3, 15, 75, rubric=topic_map.pack("finance")["rubric"],
                           topics=topics(), end_rule=True)
    assert "finance or investing transcript" in rich and ss.RUBRICS["podcast"] not in rich
    assert "END RULE (hard)" in rich and 'T2 S6-S9 "The scariest game this year"' in rich
    assert rich.index("TOPICS") < rich.index("TRANSCRIPT:")


def test_propose_shows_each_chunk_its_own_topics(monkeypatch):
    from clipengine import select_semantic as ss
    seen = []

    def fake(prompt, model, backend="ollama", timeout=0, opts=None):
        seen.append(prompt)
        return json.dumps({"clips": [{"s0": 6, "s1": 9, "hook": "x", "score": 80}]})
    monkeypatch.setattr(ss, "_llm", fake)
    picks = ss.propose(SENTS, "stream", {"semantic_backend": "claude_cli", "semantic_model": "opus",
                                         "topics": topics(), "drop_wrapup": True, "rubric": "RUBRIC X"})
    assert picks and "RUBRIC X" in seen[0] and "T1 S0-S3" in seen[0] and "END RULE" not in seen[0]
    seen.clear()
    ss.propose(SENTS, "stream", {"semantic_backend": "claude_cli", "semantic_model": "opus",
                                 "topics": topics(), "drop_wrapup": True, "end_rule_prompt": True})
    assert "END RULE" in seen[0]


def test_the_prompt_rules_are_off_by_default_and_the_code_guards_on():
    cfg = topic_map.select_cfg("podcasts")
    assert cfg["drop_wrapup"] is True
    assert cfg["end_rule_prompt"] is False and cfg["topic_editor_rule"] is False
    on = {"select": {"end_rule": {"enabled": True, "prompt_line": True},
                     "topic_map": {"editor_rule": True}}}
    cfg = topic_map.select_cfg("podcasts", cfg=on)
    assert cfg["end_rule_prompt"] is True and cfg["topic_editor_rule"] is True


# ---------------------------------------------------------------- select with the map
@pytest.fixture
def db(tmp_path, monkeypatch):
    runstate.reset()
    runstate.init()
    return tmp_path


class Sel:
    """engine.select with the transcript, the energy, chat and the LLM faked: the outline answers
    OUTLINE, the picks are given as sentence spans; no precision pass (base cuts)."""

    def __init__(self, tmp, monkeypatch, picks, outline=OUTLINE, niche="podcasts"):
        from clipengine import engine
        self.engine, self.tmp, self.picks, self.prompts = engine, tmp, picks, []
        self.outline = outline
        monkeypatch.setattr(engine, "SCRATCH", tmp / "scratch")
        monkeypatch.setattr(engine, "CROP_PROBE", False)
        monkeypatch.setattr(engine, "SELECT_DEADLINE", None)
        monkeypatch.setattr(engine, "_speaker_crop_layout", lambda layout, vertical: False)
        monkeypatch.setattr(engine, "_niche_of", lambda sid: niche)
        vcfg = engine.vertical_cfg
        monkeypatch.setattr(engine, "vertical_cfg", lambda v: {**vcfg(v), "min_dur_s": 5})
        monkeypatch.setattr(engine, "energy_envelope", lambda media: np.full(80, 0.05))
        monkeypatch.setattr(engine.transcript, "get_words", lambda m, model="base.en", cache_dir=None:
                            [dict(w) for w in WORDS])
        monkeypatch.setattr(engine, "_refine", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no audio")))
        monkeypatch.setattr(engine.edge_verify, "make_words_fn", lambda media, model: lambda lo, hi: [])
        monkeypatch.setattr(engine, "_snap", lambda media, dur, c, vc, sid: c.update(snap={"start": "ok", "end": "ok"}))
        monkeypatch.setattr(engine.select_semantic, "_llm", self._llm)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail(f"subprocess {a}"))
        monkeypatch.setenv("TOPIC_MAP", "1")

    def _llm(self, prompt, model, backend="ollama", timeout=0, opts=None):
        self.prompts.append(prompt)
        if "mapping the structure" in prompt:
            if isinstance(self.outline, Exception):
                raise self.outline
            return json.dumps(self.outline)
        return json.dumps({"clips": [{"s0": a, "s1": b, "hook": f"pick {a}-{b}", "score": 90 - a}
                                     for a, b in self.picks]})

    def source(self, sid="Pod_w01"):
        media = self.tmp / f"Pod_100000-180000.mp4"
        media.write_bytes(b"\0")
        runstate.add_source(sid, local_path=str(media), vertical="podcast")
        runstate.update_source(sid, duration_s=80.0, state="harvested")
        return runstate.source(sid)

    def events(self):
        return [e["message"] for e in runstate.events() if e["stage"] == "select"]


def test_select_keeps_every_pick_inside_one_topic(db, monkeypatch):
    sel = Sel(db, monkeypatch, [(1, 4), (5, 9), (10, 10)])     # propose drops overlapping picks
    src = sel.source()
    assert sel.engine.select(src) is True
    outline, pick = sel.prompts
    assert "mapping the structure" in outline and "TOPICS" in pick and "END RULE" not in pick
    assert "podcast / interview transcript" in pick                     # the podcasts pack's rubric
    clips = runstate.clips_for_source(src["source_id"])
    assert len(clips) == 2                                              # S10 is in no topic
    by = {c["hook"]: c for c in clips}
    a, b = by["pick 1-4"], by["pick 5-9"]
    assert SENTS[1]["start"] - 0.5 < a["start_s"] and a["end_s"] < SENTS[4]["start"]
    assert SENTS[5]["end"] < b["start_s"] < SENTS[6]["start"] and b["end_s"] < SENTS[10]["start"]
    fa = json.loads(a["features_json"])
    assert fa["topic"]["i"] == 1 and fa["topic"]["title"] == "Garage company worth a billion"
    assert len(fa["topics"]) == 2 and "into T1" in fa["topic_fit"] and fa["prompt_pack"] == "podcasts"
    ev = sel.events()
    assert any("topic map 2 topic(s) (answered)" in m for m in ev)
    assert any("S10-S10" in m and "in no topic of the map: dropped" in m for m in ev)
    assert any("outline:claude_cli/opus=answered" in m for m in ev)
    # a second select of the window reads the cached map: one pick call, no outline call
    sel.prompts.clear()
    runstate._CLIPS.clear()
    assert sel.engine.select(src) is True
    assert len(sel.prompts) == 1 and "mapping the structure" not in sel.prompts[0]


def test_without_a_map_select_picks_as_before_but_drops_the_wrapup(db, monkeypatch):
    sel = Sel(db, monkeypatch, [(1, 4)], outline=RuntimeError("claude exited 1"), niche=None)
    src = sel.source()
    assert sel.engine.select(src) is True
    assert "TOPICS" not in sel.prompts[-1] and "END RULE" not in sel.prompts[-1]
    (c,) = runstate.clips_for_source(src["source_id"])
    assert c["end_s"] < SENTS[4]["start"]                               # "So yeah, that's the story." off
    f = json.loads(c["features_json"])
    assert f["end_rule"]["dropped"] == ["So yeah, that's the story."] and "topic" not in f
    assert any("no topic map (error" in m for m in sel.events())


def test_every_switch_off_is_todays_select(db, monkeypatch):
    sel = Sel(db, monkeypatch, [(1, 4)])
    for k in ("TOPIC_MAP", "END_RULE", "PROMPT_PACKS"):
        monkeypatch.setenv(k, "0")
    src = sel.source()
    assert sel.engine.select(src) is True
    (pick,) = sel.prompts
    assert "TOPICS" not in pick and "END RULE" not in pick and "podcast / talk transcript" in pick
    (c,) = runstate.clips_for_source(src["source_id"])
    assert c["end_s"] > SENTS[4]["start"]
    assert not {"topic", "topics", "end_rule", "prompt_pack"} & set(json.loads(c["features_json"]))


# ---------------------------------------------------------------- the cut review
def _lines(start, end, words=WORDS):
    from clipengine import cut_review
    return cut_review.lines(words, start, end)


def test_apply_rules_clamps_into_the_topic_and_drops_the_wrapup():
    from clipengine import cut_review
    ls = _lines(SENTS[1]["start"] - 0.1, SENTS[5]["end"] + 0.2)
    t1 = topics()[0]
    wrap = boundaries.wrapup_sentence
    assert cut_review.apply_rules(ls, 1, 3, t1, wrap) == (1, 3, [])
    a, b, notes = cut_review.apply_rules(ls, 1, 5, t1, wrap)
    assert (a, b) == (1, 3) and "topic T1: S1-S5 -> S1-S3" in notes[0]
    a, b, notes = cut_review.apply_rules(ls, 1, 4, None, wrap)
    assert (a, b) == (1, 3) and notes[0].startswith('end rule: dropped "So yeah')
    assert cut_review.apply_rules(ls, 4, 5, t1, wrap)[0] is None


def _review(answer, start, end, vc=None, **kw):
    from clipengine import cut_review
    llm = LLM(json.dumps({"start_fault": "none", "end_fault": "none", "title_ok": True,
                          "title": "Old", **answer}))
    clip = {"start_s": start, "end_s": end, "hook": "Old"}
    return cut_review.review(clip, WORDS, "podcast", {**(vc or RULE), "min_dur_s": 5}, llm=llm,
                             st=dict(cut_review.DEFAULTS, max_move_s=40), media_dur=80.0, **kw), llm


def test_the_review_never_moves_a_cut_out_of_its_topic():
    ts = topics()
    s, e = SENTS[1]["start"] - 0.15, SENTS[3]["end"] + 0.3
    r, llm = _review({"start": 0, "end": 6}, s, e, topics=ts, topic=ts[0], pack=topic_map.pack("podcasts"))
    assert r["verdict"] == "move" and r["lines"] == [0, 3] and "topic T1" in r["why"]
    p = llm.prompts[0]
    # yield 11-11: no tags, no topic sentence and no END RULE line in the prompt; the clamp holds
    assert "TOPICS: the tag" not in p and "T1]" not in p and " -]" not in p and "END RULE" not in p
    assert "For this podcast / talk video:" in p
    r, llm = _review({"start": 0, "end": 6}, s, e, vc=RULE_PROMPTS, topics=ts, topic=ts[0],
                     pack=topic_map.pack("podcasts"))
    assert r["verdict"] == "move" and r["lines"] == [0, 3]
    p = llm.prompts[0]
    assert "TOPICS: the tag" in p and "T1]" in p and " -]" in p and "END RULE (hard)" in p


def test_a_cut_that_ends_on_a_wrapup_moves_even_when_the_editor_keeps_it():
    s, e = SENTS[1]["start"] - 0.15, SENTS[4]["end"] + 0.3
    r, _ = _review({"start": 1, "end": 4}, s, e)
    assert r["verdict"] == "move" and r["end"] < SENTS[4]["start"] and "end rule" in r["why"]
    r, llm = _review({"start": 1, "end": 4}, s, e, title_only=True)       # a fixed cut: no rule
    assert r["verdict"] == "keep" and "END RULE" not in llm.prompts[0]


def test_the_review_with_no_map_and_no_rule_is_stage_cs():
    from clipengine import cut_review
    s, e = SENTS[1]["start"] - 0.15, SENTS[4]["end"] + 0.3
    llm = LLM(json.dumps({"start": 1, "end": 4, "title_ok": True, "title": "Old"}))
    r = cut_review.review({"start_s": s, "end_s": e, "hook": "Old"}, WORDS, "podcast", VC, llm=llm,
                          st=cut_review.DEFAULTS, media_dur=80.0)
    assert r["verdict"] == "keep"
    assert "TOPICS" not in llm.prompts[0] and "END RULE" not in llm.prompts[0]
    assert "[S1 " in llm.prompts[0] and "s] Nobody believed" in llm.prompts[0]


def test_the_render_review_reads_the_clips_topic_from_its_features(db, monkeypatch):
    from clipengine import engine
    monkeypatch.setattr(engine, "SCRATCH", db / "scratch")
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", None)
    monkeypatch.setattr(engine, "_niche_of", lambda sid: "podcasts")
    monkeypatch.setenv("TOPIC_MAP", "1")
    monkeypatch.delenv("CUT_REVIEW", raising=False)
    media = db / "Pod_w01.mp4"
    media.write_bytes(b"\0")
    runstate.add_source("Pod_w01", local_path=str(media), vertical="podcast")
    ts = topics()
    feats = {"topics": topic_map.compact(ts), "topic": {k: ts[0][k] for k in ("i", "lo", "hi", "title")}}
    s, e = SENTS[1]["start"] - 0.15, SENTS[3]["end"] + 0.3
    side = db / "side.json"
    side.write_text(json.dumps([w for w in WORDS if s <= w["start"] <= e]))
    cid = runstate.add_clip("Pod_w01", s, e, 0.6, "auto", hook="Old", words_path=str(side),
                         features_json=feats, boundary_type="sentence")
    llm = LLM(json.dumps({"start": 1, "end": 7, "title_ok": True, "title": "Old"}))
    new, r = engine._cut_review(runstate.clip(cid), {"vertical": "podcast"}, str(media), 80.0, llm=llm,
                                words_fn=lambda m: WORDS, rms_fn=lambda m, d: None)
    assert r["verdict"] == "keep" and r["lines"] == [1, 3]               # S7 is T2: clamped back
    rec = json.loads(runstate.clip(cid)["features_json"])["cut_review"][0]
    assert rec["topic"] == 1 and rec["rules"] == ["topic T1: S1-S7 -> S1-S3"]
    assert "T2]" not in llm.prompts[0] and "For this podcast / talk video" in llm.prompts[0]


# GTA week 11-14: a pack's dated title examples reach both title prompts in their window only
HOOKS = {"rubric": "R.", "review": "V.", "hook_examples": {
    "label": "GTA week", "from": "2026-11-14", "to": "2026-11-25", "rules": "7 words max.",
    "lines": ["{streamer} rage quit GTA 6", " "]}}


def test_hook_examples_only_inside_their_window():
    block = topic_map.hook_examples(HOOKS, today="2026-11-19")
    assert "TITLE EXAMPLES (GTA week)" in block and "- {streamer} rage quit GTA 6" in block
    assert "{slot}" in block and "7 words max." in block and block.count("\n- ") == 1
    assert topic_map.hook_examples(HOOKS, today="2026-11-13") == ""
    assert topic_map.hook_examples(HOOKS, today="2026-11-26") == ""
    assert topic_map.hook_examples({"rubric": "R."}, today="2026-11-19") == ""
    assert topic_map.hook_examples(None) == ""


def test_hook_examples_reach_the_cut_review_and_the_pick_prompt(monkeypatch):
    from clipengine import cut_review
    from clipengine import select_semantic
    monkeypatch.setattr(topic_map, "hook_examples",
                        lambda pk, today=None: "\nTITLE EXAMPLES (x)" if pk and "hook_examples" in pk else "")
    ls = [{"i": 0, "start": 0.0, "end": 2.0, "text": "Hi.", "inside": True}]
    on = cut_review.prompt(ls, 0, 2, "stream", "Old", 15, 60, pack=HOOKS)
    off = cut_review.prompt(ls, 0, 2, "stream", "Old", 15, 60, pack={"review": "V."})
    assert "TITLE EXAMPLES (x)" in on and "TITLE EXAMPLES" not in off
    assert off == on.replace("\nTITLE EXAMPLES (x)", "")
    sents = [{"i": 0, "start": 0.0, "end": 2.0, "text": "Hi."}]
    rub = HOOKS["rubric"] + topic_map.hook_examples(HOOKS)
    assert "TITLE EXAMPLES (x)" in select_semantic.build_prompt(sents, "stream", rubric=rub)
