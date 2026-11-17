"""A slur is bleeped in the first render. The caption step masks the slurs inside the cut in its
sidecar and keeps their source seconds in features "censor"; recompose_cmd passes them as --mute
(clip seconds); recompose merges repeated --mute. Synthetic words: two slurs inside a 20.0-26.0 s
cut, profanity that stays, and one more slur after the cut. Offline: the strong caption pass is
stubbed."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipengine import runstate
from clipengine.render import recompose


def W(*triples):
    return [{"w": w, "start": s, "end": e} for w, s, e in triples]


# a slur at 21.0-21.3 s and one at 23.7-24.3 s inside a 20.0-26.0 s cut; another one after the cut
WORDS = W(("So", 20.0, 20.2), ("then,", 20.2, 20.6), ("my", 20.7, 20.95), ("nigga,", 21.0, 21.3),
          ("we", 21.4, 21.5), ("lost", 21.5, 21.8), ("the", 21.8, 21.9), ("last", 21.9, 22.1),
          ("big", 22.1, 22.2), ("round.", 22.2, 22.6), ("Is", 23.0, 23.2), ("he", 23.2, 23.3),
          ("fucking", 23.3, 23.7), ("retarded?", 23.7, 24.3), ("Anyway.", 25.0, 25.5),
          ("faggot", 27.0, 27.5))
CUT = (20.0, 26.0)


@pytest.fixture
def db(tmp_path, monkeypatch):
    from clipengine import engine
    runstate.reset()
    monkeypatch.setattr(engine, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", None)
    monkeypatch.setattr(engine.transcript, "WORKER", None)
    return tmp_path


def _clip(tmp_path, side, shape="list"):
    runstate.add_source("S_w00", local_path=str(tmp_path / "w.mp4"), vertical="stream")
    wp = tmp_path / "side.json"
    wp.write_text(json.dumps(side if shape == "list" else {"words": side, "v": 1}))
    cid = runstate.add_clip("S_w00", CUT[0], CUT[1], 0.5, "auto", hook="h", words_path=str(wp),
                            features_json={"refined": True})
    return runstate.clip(cid)


def _mute(cmd):
    return cmd[cmd.index("--mute") + 1] if "--mute" in cmd else None


def test_the_caption_step_masks_the_slurs_inside_the_cut_and_keeps_their_seconds(db, monkeypatch):
    from clipengine import engine
    clip = _clip(db, WORDS)
    monkeypatch.setattr(engine, "_strong_pass", lambda *a: [dict(w, p=0.9) for w in WORDS])
    new = engine._caption_step(clip, "w.mp4")
    row = runstate.clip(clip["clip_id"])
    side = json.loads(Path(row["words_path"]).read_text())
    burned = [w["w"] for w in side if CUT[0] <= w["start"] < CUT[1]]
    assert "n****," in burned and "r*******?" in burned and "fucking" in burned  # not profanity
    assert not any(engine.slurs.is_slur(w) for w in burned)
    cen = json.loads(row["features_json"])["censor"]
    assert cen["words"] == 2 and cen["spans"] == [[20.96, 21.34], [23.66, 24.34]]
    assert json.loads(new["features_json"])["censor"] == cen
    cmd = engine.recompose_cmd(new, {"vertical": "stream"}, "w.mp4", "o.mp4")
    assert _mute(cmd) == "0.960-1.340,3.660-4.340"                  # clip seconds
    assert cmd[cmd.index("--words") + 1] == row["words_path"]       # the masked sidecar


def test_select_words_kept_on_a_caption_error_are_masked_too(db, monkeypatch):
    from clipengine import engine
    clip = _clip(db, WORDS, shape="dict")
    monkeypatch.setattr(engine, "_strong_pass", lambda *a: (_ for _ in ()).throw(
        RuntimeError("whisper died")))
    new = engine._caption_step(clip, "w.mp4")
    assert new["words_path"] == clip["words_path"]
    d = json.loads(Path(clip["words_path"]).read_text())
    assert d["v"] == 1 and [w["w"] for w in d["words"]][3] == "n****,"        # shape kept
    assert d["words"][-1]["w"] == "faggot"                         # outside the cut: untouched
    assert _mute(engine.recompose_cmd(new, {}, "w.mp4", "o.mp4")) == "0.960-1.340,3.660-4.340"


def test_a_clip_with_no_slur_is_unchanged(db, monkeypatch):
    from clipengine import engine
    clean = [w for w in WORDS if not engine.slurs.is_slur(w["w"])]
    clip = _clip(db, clean)
    monkeypatch.setattr(engine, "_strong_pass", lambda *a: [dict(w, p=0.9) for w in clean])
    new = engine._caption_step(clip, "w.mp4")
    assert "censor" not in json.loads(runstate.clip(clip["clip_id"])["features_json"])
    cmd = engine.recompose_cmd(new, {}, "w.mp4", "o.mp4")
    assert "--mute" not in cmd and cmd[cmd.index("--words") + 1] == new["words_path"]


def test_a_sidecar_rewritten_after_the_caption_step_still_burns_masked(db):
    """A later step re-hears the captions (the caption check) or re-cuts: the slur is back in
    the sidecar; the render burns a masked copy and mutes it."""
    from clipengine import engine
    clip = _clip(db, WORDS)
    cmd = engine.recompose_cmd(clip, {}, "w.mp4", "o.mp4")
    wp = cmd[cmd.index("--words") + 1]
    assert wp.endswith("side_censored.json") and wp != clip["words_path"]
    assert [w["w"] for w in json.loads(Path(wp).read_text())][3] == "n****,"
    assert _mute(cmd) == "0.960-1.340,3.660-4.340"
    assert [w["w"] for w in json.loads(Path(clip["words_path"]).read_text())][3] == "nigga,"


def test_a_slur_outside_the_moved_cut_is_not_muted(db):
    from clipengine import engine
    clip = _clip(db, WORDS)
    clip = engine._censor_slurs(clip)
    later = {**clip, "start_s": 22.5, "end_s": 26.0}               # the final cut moved the start
    cmd = engine.recompose_cmd(later, {}, "w.mp4", "o.mp4")
    assert _mute(cmd) == "1.160-1.840"                              # only "retarded?" is inside


def test_recompose_merges_a_repeated_mute():
    a = recompose.parse_args(["s.mp4", "0", "9", "o.mp4", "--mute", "0.960-1.340",
                              "--mute", "3.660-4.340,5.0-5.2"])
    assert recompose.mute_spans(a.mute) == [(0.96, 1.34), (3.66, 4.34), (5.0, 5.2)]
    assert recompose.mute_spans("1.2-1.6") == [(1.2, 1.6)] and recompose.mute_spans(None) == []
    chain = recompose.audio_chain(9.0, mute=recompose.mute_spans(a.mute))
    assert "between(t,0.960,1.340)+between(t,3.660,4.340)+between(t,5.000,5.200)" in chain
