"""One strong, aligned transcript per window for select (engine.window_pass): the sentences,
the LLM's transcript and the precision cuts read it, and the caption step slices the clip's words
from it; off, no time left or an error keep select's base pass. Offline, on test_select_v2's rig
(transcript, LLM, audio, scenes and subprocess all faked; the strong pass is a fake of
strong_asr.transcribe_strong)."""
import json

import pytest

from test_select_v2 import WORDS, llm_answer, rig, spans  # noqa: F401 - the rig fixture

from clipengine import engine
from clipengine import strong_asr
from clipengine import runstate


def strong_words(words=WORDS):
    """The strong pass's reading of the same speech: the same words with word probabilities, and
    its own sentence ends (a period on every 7th word the base pass ran on past)."""
    out = []
    for k, w in enumerate(words):
        x = {**w, "p": 0.93}
        if k % 7 == 3 and w["w"][-1] not in ".?!,":
            x["w"] = w["w"] + "."
        out.append(x)
    return out


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(engine, "window_transcript_settings", lambda cfg=None: {
        "enabled": True, "model": "large-v3-turbo", "prompt": True, "seat": True})


def test_select_reads_the_window_pass_and_its_precision_cuts_need_no_new_pass(rig, on,
                                                                             monkeypatch):
    calls = []
    strong = strong_words()
    monkeypatch.setattr(strong_asr, "transcribe_strong", lambda media, lo, hi, model=None,
                        timeout=None, prompt=None, **k: calls.append((lo, hi, model, prompt))
                        or [dict(w) for w in strong])
    rig.window = "same"
    clips = rig.run()
    assert calls == [(0.0, 480.0, "large-v3-turbo", None)]        # the rig's source has no title
    assert rig.window_calls == []                                  # no small.en precision pass
    assert clips and all(json.loads(c["features_json"])["refined"] for c in clips)
    side = json.loads(open(clips[0]["words_path"]).read())
    assert side and all(w["w"] in {x["w"] for x in strong} for w in side)
    added = [w["w"] for w in strong if w["w"] not in {x["w"] for x in WORDS}]
    assert added and any(a in rig.prompts[0] for a in added)       # the LLM read its sentences
    msgs = [e["message"] for e in rig_events()]
    assert any("window pass large-v3-turbo" in m for m in msgs)
    # a second select of the same window reads the pass from the cache
    calls.clear()
    engine.window_pass(runstate.source("src1")["local_path"], runstate.source("src1"))
    assert calls == []


def rig_events():
    return [{k: e[k] for k in ("level", "stage", "message")} for e in runstate.events()]


def test_a_window_pass_that_fails_or_does_not_fit_leaves_selects_base_pass(rig, on, monkeypatch):
    def broken(*a, **k):
        raise FileNotFoundError("large-v3-turbo.pt")
    monkeypatch.setattr(strong_asr, "transcribe_strong", broken)
    rig.window = "same"
    clips = rig.run()
    assert clips and rig.window_calls                             # the precision pass as before
    assert any("no window pass (FileNotFoundError" in e["message"] for e in rig_events())
    # select's time does not fit the pass: it never starts
    monkeypatch.setattr(engine, "SELECT_DEADLINE", engine.time.monotonic() + 30.0)
    monkeypatch.setattr(strong_asr, "transcribe_strong", lambda *a, **k: pytest.fail("no time for it"))
    src = runstate.source("src1")
    words, info = engine.window_pass(src["local_path"], {**src, "source_id": "other"})
    assert words is None and "select's time left" in info["skipped"]


def test_off_is_todays_select(rig, monkeypatch):
    monkeypatch.setattr(strong_asr, "transcribe_strong", lambda *a, **k: pytest.fail("off"))
    monkeypatch.setenv("WINDOW_TRANSCRIPT", "0")
    rig.window = "same"
    assert rig.run() and rig.window_calls
    assert engine.window_transcript_settings({"select": {"window_transcript": {
        "enabled": True}}})["enabled"] is False


def test_the_caption_step_slices_the_clips_words_from_the_window_pass(rig, on, monkeypatch):
    strong = strong_words()
    monkeypatch.setattr(strong_asr, "transcribe_strong", lambda *a, **k: [dict(w) for w in strong])
    rig.window = "same"
    clip = rig.run()[0]
    monkeypatch.setattr(engine, "CAPTION_STRONG_MODEL", "large-v3-turbo")
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", None)
    src = runstate.source(clip["source_id"])
    side, info = engine.strong_captions(clip, src["local_path"],
                                        words_fn=lambda *a, **k: pytest.fail("no per-clip pass"))
    s, e = clip["start_s"], clip["end_s"]
    burned = [w for w in side if s <= engine._mid(w) <= e]
    assert info["mode"] == "window" and info["judged_by"] == ["prob"]
    assert burned and all(w.get("p") == 0.93 for w in burned)
    # no window pass in the cache (another model, or never heard): the per-clip pass as before
    monkeypatch.setattr(engine, "CAPTION_STRONG_MODEL", "large-v3")
    heard = []
    side, info = engine.strong_captions(clip, src["local_path"],
                                        words_fn=lambda m, a, b, t: heard.append((a, b)) or [])
    assert heard and info["mode"] == "full"
