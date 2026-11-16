"""Gate checks that read the audio and a second / third ASR, offline.

The cut checks read the audio: a cut in a pause is in no word (quality_gate.cut_audio /
cut_in_quiet in word_boundaries, the ending's mid-word rule, gate_cut_errors and the final cut).
The audio is a synthetic tone with silences; the words are ASR words stretched over a pause the
way a fast model times them (a short word stretched over the cut's silence).

The caption check's tie-break: captions that fail against the second ASR (`small`) are read
against a third model (`medium`) before the check fails (quality_gate.caption_tiebreak_verdict /
caption_check_with_tiebreak, engine._caption_check). The words model a second ASR that dropped a
repeated phrase the captions and the third model both have.

Crosstalk the second ASR also hears is burned (crosstalk.confirm_crosstalk,
engine._confirm_crosstalk): a stretch of words never burned while the gate's second ASR heard
them, an "uncaptioned stretch" reject.
"""
from __future__ import annotations

import json
import wave

import numpy as np
import pytest

from clipengine import boundaries
from clipengine import quality_gate
from clipengine import engine
from clipengine import runstate

SR = 16000


def _wav(path, secs, quiet=(), level=0.3, sr=SR):
    """A 220 Hz tone `secs` long, silent over each (a, b) in `quiet`."""
    t = np.arange(int(secs * sr)) / sr
    a = np.sin(2 * np.pi * 220 * t) * level
    for lo, hi in quiet:
        a[(t >= lo) & (t < hi)] = 0.0
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((a * 32767).astype(np.int16).tobytes())
    return path


def W(text, t0, step=0.3):
    return [{"w": w, "start": round(t0 + i * step, 3), "end": round(t0 + i * step + step * 0.8, 3)}
            for i, w in enumerate(text.split())]


@pytest.fixture
def cut_audio_on(monkeypatch):
    monkeypatch.setattr(quality_gate, "cut_audio_enabled", lambda cfg=None: True)


# ---------------------------------------------------------------- the cut's audio

def test_a_cut_in_a_pause_is_in_no_word_the_transcript_stretched_over_it(tmp_path):
    # the end cut at 3.0 s sits in 2.8-3.25 s of silence; the gate's words stretch 'and' over it
    src = _wav(tmp_path / "s.wav", 5.0, quiet=[(2.8, 3.25)])
    words = W("so we went to the", 1.0) + [{"w": "and", "start": 2.75, "end": 3.35}]
    base = quality_gate.measure_word_boundaries(words, 1.0, 3.0)
    assert base["passed"] is False and base["end_in_word"]
    audio = quality_gate.cut_audio(src, 1.0, 3.0)
    got = quality_gate.measure_word_boundaries(words, 1.0, 3.0, audio=audio)
    assert got["passed"] is True and not got["end_in_word"] and "'and'" in got["end_quiet"]
    # the same frames from a window's 10 ms RMS (what the final cut reads)
    rms = boundaries.frame_rms(np.frombuffer(
        __import__("subprocess").run(["ffmpeg", "-v", "error", "-i", str(src), "-ac", "1", "-ar",
                                      str(SR), "-f", "f32le", "-"], capture_output=True).stdout,
        np.float32), SR, 0.01)
    audio2 = quality_gate.cut_audio_from_rms(rms, 0.0, 0.01, 1.0, 3.0)
    assert quality_gate.cut_in_quiet(audio2, 3.0) and not quality_gate.cut_in_quiet(audio2, 2.5)


def test_a_cut_in_the_voice_or_a_short_dip_is_still_inside_the_word(tmp_path):
    words = W("so we went to the", 1.0) + [{"w": "and", "start": 2.75, "end": 3.35}]
    voiced = _wav(tmp_path / "v.wav", 5.0)
    m = quality_gate.measure_word_boundaries(words, 1.0, 3.0,
                                             audio=quality_gate.cut_audio(voiced, 1.0, 3.0))
    assert m["passed"] is False and "'and'" in m["errors"][0]
    dip = _wav(tmp_path / "d.wav", 5.0, quiet=[(2.97, 3.03)])        # 60 ms: a stop, no pause
    m = quality_gate.measure_word_boundaries(words, 1.0, 3.0,
                                             audio=quality_gate.cut_audio(dip, 1.0, 3.0))
    assert m["passed"] is False


def test_no_audio_or_the_switch_off_leaves_the_word_times_to_decide(tmp_path, monkeypatch):
    words = W("so we went to the", 1.0) + [{"w": "and", "start": 2.75, "end": 3.35}]
    monkeypatch.setattr(quality_gate, "cut_audio_enabled", lambda cfg=None: True)
    assert quality_gate.safe_cut_audio(tmp_path / "missing.mp4", 1.0, 3.0) is None
    assert quality_gate.safe_cut_audio(None, 1.0, 3.0) is None
    m = quality_gate.measure_word_boundaries(words, 1.0, 3.0, audio=None)
    assert m["passed"] is False
    src = _wav(tmp_path / "s.wav", 5.0, quiet=[(2.8, 3.25)])
    assert quality_gate.safe_cut_audio(src, 1.0, 3.0) is not None
    monkeypatch.setattr(quality_gate, "cut_audio_enabled", lambda cfg=None: False)
    assert quality_gate.safe_cut_audio(src, 1.0, 3.0) is None
    monkeypatch.setenv("CUT_AUDIO", "0")
    assert quality_gate.cut_audio_settings({"quality_gate": {"cut_audio": {"enabled": True}}})[
        "enabled"] is False


def test_the_ending_does_not_read_a_word_stretched_over_the_pause_as_cut():
    heard = W("that was the best game", 0.0) + [{"w": "ever.", "start": 1.5, "end": 2.2}]
    v = quality_gate.ending_verdict(heard, heard, 1.9, tail_s=2.0)
    assert v["passed"] is False and "inside the word 'ever.'" in v["errors"][0]
    v = quality_gate.ending_verdict(heard, heard, 1.9, tail_s=2.0, quiet_end=True)
    assert v["passed"] is True and "'ever.'" in v["quiet_end"]
    # the end cut's frames: 0.2 s of quiet around it in cut_energy's frames
    times = [round(-1.0 + 0.02 * i, 2) for i in range(150)]
    pause = [(-40.0 if -0.1 <= t < 0.1 else 3.0) for t in times]
    assert quality_gate.energy_end_in_quiet((times, pause))
    short = [(-40.0 if -0.02 <= t < 0.04 else 3.0) for t in times]
    assert not quality_gate.energy_end_in_quiet((times, short))
    assert not quality_gate.energy_end_in_quiet(None)


def _frames(t0, t1, quiet=(), hop=0.02):
    """cut_audio-shaped frames over [t0, t1): -40 dB in each (a, b) of `quiet`, else +3 dB."""
    times = [round(t0 + hop * i, 3) for i in range(int(round((t1 - t0) / hop)))]
    db = [(-40.0 if any(a <= t < b for a, b in quiet) else 3.0) for t in times]
    return {"times": times, "db": db, "hop": hop}


def test_a_word_stretched_over_a_quiet_cut_is_read_on_the_side_its_voice_is_on():
    # the cut on the aligned onset; `small` times the first word 0.2 s earlier, over the pause
    # before it, so it reads as the word before the clip's first
    ws = W("That was great.", 0.0) + [{"w": "And", "start": 1.0, "end": 1.45}] + \
        W("we won the game tonight.", 1.6)
    audio = _frames(0.0, 4.0, quiet=[(0.9, 1.25)])
    s, e = 1.2, ws[-1]["end"] + 0.1
    m = quality_gate.measure_sentence_bounds(ws, s, e)
    assert m["passed"] is False and "prev='And'" in m["errors"][0]
    assert quality_gate.measure_word_boundaries(ws, s, e)["start_in_word"]
    m = quality_gate.measure_sentence_bounds(ws, s, e, audio=audio)
    assert m["passed"] is True and m["first_word"] == "And"
    m = quality_gate.measure_word_boundaries(ws, s, e, audio=audio)
    assert m["passed"] is True and "'And' read after the cut" in m["start_quiet"]
    seated, how = quality_gate.seat_on_cut_audio(ws, (("start", s), ("end", e)), audio)
    assert seated[3]["start"] == 1.26 and seated[3]["asr_start"] == 1.0 and ws[3]["start"] == 1.0
    assert how == [{"w": "And", "cut": "start", "side": "after", "from": [1.0, 1.45],
                    "to": [1.26, 1.45]}]
    # a word whose voice is before the pause (its end stretched over it) closes the line before
    ws2 = W("That was", 0.0) + [{"w": "great.", "start": 0.6, "end": 1.3}] + W("We won.", 1.6)
    seated, how = quality_gate.seat_on_cut_audio(ws2, (("start", 1.2), ("end", 2.2)),
                                                 _frames(0.0, 4.0, quiet=[(0.9, 1.5)]))
    assert how[0]["side"] == "before" and seated[2]["end"] == 0.9
    # the final cut reads the seated words: no "inside the first word"
    assert boundaries.final_cut_issues(ws, s, e).get("start") == "the cut is inside the first word 'And'"
    assert "start" not in boundaries.final_cut_issues(
        quality_gate.seat_on_cut_audio(ws, (("start", s), ("end", e)), audio)[0], s, e)


def test_gate_cut_errors_read_the_window_audio(tmp_path, monkeypatch, cut_audio_on):
    from clipengine import second_asr
    monkeypatch.setattr(second_asr, "settings", lambda cfg=None: {"enabled": False})
    src = _wav(tmp_path / "s.wav", 5.0, quiet=[(2.8, 3.25)])
    words = W("So we went to the store.", 1.05)
    words[-1].update(start=2.6, end=3.045)                 # 'store.' stretched into the pause
    assert engine.gate_cut_errors(words, 1.0, 3.0, str(src)) == []
    monkeypatch.setattr(quality_gate, "cut_audio_enabled", lambda cfg=None: False)
    errs = engine.gate_cut_errors(words, 1.0, 3.0, str(src))
    assert errs and "end cut inside word 'store.'" in errs[0]


# ---------------------------------------------------------------- the caption tie-break

TALK = ("I mean yeah you could throw it you could throw it right into the bin and it still "
        "works every single time so that is what we did")


def _rel(text, t0=0.0, step=0.3):
    return W(text, t0, step)


def _small_drops(words, drop):
    """The second ASR's reading: `words` without the ones at indexes `drop` (a repeat it missed)."""
    return [w for k, w in enumerate(words) if k not in drop]


def test_captions_only_small_disputes_pass_on_the_tiebreak():
    cap = _rel(TALK)
    small = _small_drops(cap, set(range(3, 11)))          # 'you could throw it' x2 never heard
    base = quality_gate.caption_agreement_verdict(cap, small, 9.0)
    assert base["passed"] is False
    v = quality_gate.caption_tiebreak_verdict(cap, small, _rel(TALK), 9.0, base=base,
                                              model="medium")
    assert v["passed"] is True and v["decided_by"] == "medium" and v["errors"] == []
    assert v["tiebreak"]["second_asr"]["passed"] is False and "agree with medium" in \
        v["tiebreak"]["verdict"]


def test_garbled_captions_still_fail_with_the_second_asrs_reasons():
    cap = _rel("I mean yeah blue throw window at right into the bin and it still works every "
               "single time so that is what we did")
    small = _rel(TALK)
    base = quality_gate.caption_agreement_verdict(cap, small, 9.0)
    assert base["passed"] is False
    v = quality_gate.caption_tiebreak_verdict(cap, small, _rel(TALK), 9.0, base=base,
                                              model="medium")
    assert v["passed"] is False and v["errors"] == base["errors"]       # reject codes stable
    assert "also fail against medium" in v["tiebreak"]["verdict"]
    # a third model that heard nothing decides nothing
    v = quality_gate.caption_tiebreak_verdict(cap, small, [], 9.0, base=base, model="medium")
    assert v["passed"] is False and "no verdict" in v["tiebreak"]["verdict"]


def test_no_tiebreak_model_or_the_switch_off_leaves_the_second_asrs_verdict(monkeypatch):
    cap = _rel(TALK)
    small = _small_drops(cap, set(range(3, 11)))
    monkeypatch.setenv("CAPTION_TIEBREAK", "1")
    monkeypatch.setattr(quality_gate, "caption_tiebreak_settings",
                        lambda cfg=None: {"enabled": True, "model": "medium"})

    def unavailable():
        raise FileNotFoundError("~/.cache/whisper/medium.pt")
    v = quality_gate.caption_check_with_tiebreak(cap, small, 9.0, third_fn=unavailable)
    assert v["passed"] is False and "medium.pt" in v["tiebreak"]["error"]
    v = quality_gate.caption_check_with_tiebreak(cap, small, 9.0, third_fn=lambda: _rel(TALK))
    assert v["passed"] is True
    # no source media: the model never runs
    v = quality_gate.caption_check_with_tiebreak(cap, small, 9.0, src_media=None, start_s=0.0,
                                                 end_s=9.0)
    assert v["passed"] is False and "no source media" in v["tiebreak"]["error"]
    monkeypatch.setattr(quality_gate, "caption_tiebreak_settings",
                        lambda cfg=None: {"enabled": False, "model": "medium"})
    v = quality_gate.caption_check_with_tiebreak(cap, small, 9.0, third_fn=lambda: _rel(TALK))
    assert v["passed"] is False and "tiebreak" not in v


def test_the_tiebreak_model_must_be_independent(tmp_path, monkeypatch):
    from clipengine import second_asr
    src = _wav(tmp_path / "s.wav", 3.0)
    for model in ("small", "large-v3-turbo", "base.en"):
        with pytest.raises(RuntimeError):
            quality_gate.run_caption_tiebreak(src, 0.0, 2.0, model)
    seen = []
    monkeypatch.setattr(second_asr, "transcribe",
                        lambda m, a, b, model=None: seen.append(model) or {"words": []})
    assert quality_gate.run_caption_tiebreak(src, 0.0, 2.0, "medium") == [] and seen == ["medium"]


def test_the_gate_reads_the_tiebreak(tmp_path, monkeypatch):
    monkeypatch.setattr(quality_gate, "caption_tiebreak_settings",
                        lambda cfg=None: {"enabled": True, "model": "medium"})
    monkeypatch.setattr(quality_gate, "measure_edge_bands", lambda *a, **k: {"passed": None})
    src = _wav(tmp_path / "s.wav", 14.0)
    side = _rel(TALK, 2.0)
    small = _small_drops(_rel(TALK), set(range(3, 11)))
    calls = []
    monkeypatch.setattr(quality_gate, "run_caption_tiebreak",
                        lambda m, a, b, model=None: calls.append((a, b)) or _rel(TALK))
    rep = quality_gate.clip_checks(str(src), words=side, start_s=2.0, end_s=11.0,
                                     hook="throw it in the bin", src_media=str(src),
                                     asr=({"words": small}, 2.0))
    assert rep["caption_asr"]["passed"] is True and calls == [(2.0, 11.0)]
    monkeypatch.setattr(quality_gate, "run_caption_tiebreak",
                        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("window cap near")))
    rep = quality_gate.clip_checks(str(src), words=side, start_s=2.0, end_s=11.0,
                                     hook="throw it in the bin", src_media=str(src),
                                     asr=({"words": small}, 2.0))
    assert rep["caption_asr"]["passed"] is False and "window cap" in \
        rep["caption_asr"]["tiebreak"]["error"]


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", None)
    return tmp_path


def _clip(tmp_path, start, end, words, sid="Pod_w01"):
    media = tmp_path / f"{sid}.mp4"
    media.write_bytes(b"\0")
    runstate.add_source(sid, local_path=str(media), vertical="podcast", niche="podcasts")
    wp = tmp_path / f"{sid}_side.json"
    wp.write_text(json.dumps(words))
    cid = runstate.add_clip(sid, start, end, 0.6, "auto", hook="Throw it in the bin",
                         words_path=str(wp), features_json={"refined": True},
                         boundary_type="sentence")
    return runstate.clip(cid), str(media)


def test_a_track_only_small_disputes_is_kept_without_a_rehear(db, monkeypatch):
    monkeypatch.setenv("CAPTION_CHECK", "1")
    monkeypatch.setattr(quality_gate, "caption_tiebreak_settings",
                        lambda cfg=None: {"enabled": True, "model": "medium"})
    talk = _rel(TALK + ".", 3.2)
    s, e = 3.1, talk[-1]["end"] + 0.3
    clip, media = _clip(db, s, e, talk)
    rel = [{**w, "start": w["start"] - s, "end": w["end"] - s} for w in talk]
    small = _small_drops(rel, set(range(3, 11)))
    asked = []
    new = engine._caption_check(clip, {"vertical": "podcast"}, media, 30.0,
                                strong_fn=lambda *a: pytest.fail("no re-hearing"),
                                rec_fn=lambda m, a, b: {"words": small, "tail_s": 2.0},
                                third_fn=lambda m, a, b: asked.append((a, b)) or rel)
    assert asked == [(s, e)] and new["words_path"] == clip["words_path"]
    f = json.loads(runstate.clip(clip["clip_id"])["features_json"])["caption_check"]
    assert f["kept"] == "clean" and f["tiebreak"]["model"] == "medium"
    # the tie-break model fails to load: the track is read as before (a problem, re-heard)
    clip2, media2 = _clip(db, s, e, talk, sid="Pod_w02")
    heard = []
    engine._caption_check(clip2, {"vertical": "podcast"}, media2, 30.0,
                          strong_fn=lambda m, lo, hi, t: heard.append(lo) or [dict(w) for w in talk],
                          rec_fn=lambda m, a, b: {"words": small, "tail_s": 2.0},
                          vote_fn=lambda sd: (None, {"reason": "nothing to vote"}),
                          rms_fn=lambda m, lo, hi: None,
                          third_fn=lambda *a: (_ for _ in ()).throw(OSError("no medium.pt")))
    f = json.loads(runstate.clip(clip2["clip_id"])["features_json"])["caption_check"]
    assert heard and "no medium.pt" in f["tiebreak"]["error"] and f["problems"]


# ---------------------------------------------------------------- confirmed crosstalk

from clipengine import crosstalk

HOSTS = W("so the guy just kept talking about it and I said", 10.0) + \
    W("no way he made that shot from half court", 13.6) + W("and we all lost it.", 16.4)


def _flag(words, i0, i1, fix=None):
    out = [dict(w) for w in words]
    for w in out[i0:i1]:
        w["cap"] = False
        if fix:
            w["fix"] = fix
    return out


def test_crosstalk_the_second_asr_hears_is_burned_and_keeps_its_flag():
    side = _flag(HOSTS, 11, 20)                          # 'no way he made ... half court' unburned
    cap = quality_gate.clip_words(side, 9.9, 18.0)
    heard = [{**w, "start": w["start"] - 9.9, "end": w["end"] - 9.9} for w in HOSTS]
    before = quality_gate.caption_agreement_verdict(cap, heard, 8.1)
    assert before["passed"] is False and any("uncaptioned" in e for e in before["errors"])
    new, rep = crosstalk.confirm_crosstalk(side, HOSTS)
    assert [b["text"] for b in rep["burned"]] == ["no way he made that shot from half court"]
    assert all(w.get("crosstalk") is True and "cap" not in w for w in new[11:20])
    assert side[11]["cap"] is False and "crosstalk" not in side[11]           # input untouched
    after = quality_gate.caption_agreement_verdict(quality_gate.clip_words(new, 9.9, 18.0),
                                                   heard, 8.1)
    assert after["passed"] is True


def test_crosstalk_nobody_else_hears_stays_unburned():
    side = _flag(HOSTS, 11, 20)
    other = [w for w in HOSTS if not 13.5 < w["start"] < 16.3] + \
        W("yeah yeah yeah", 13.6, 0.8)                   # the second ASR heard something else
    new, rep = crosstalk.confirm_crosstalk(side, other)
    assert not rep["burned"] and rep["kept"] and all(w["cap"] is False for w in new[11:20])
    # words the caption repair unburned as heard by nobody are not crosstalk to confirm
    new, rep = crosstalk.confirm_crosstalk(_flag(HOSTS, 11, 20, fix="unheard"), HOSTS)
    assert rep == {"burned": [], "kept": []} and all(w["cap"] is False for w in new[11:20])
    assert crosstalk.confirm_crosstalk(side, [])[1] == {"burned": [], "kept": []}


def test_the_render_burns_confirmed_crosstalk_on_the_gates_pass(db, monkeypatch):
    monkeypatch.setattr(engine, "crosstalk_confirm_enabled", lambda cfg=None: True)
    side = _flag(HOSTS, 11, 20)
    clip, media = _clip(db, 9.9, 18.0, side)
    asr = {"words": [{**w, "start": w["start"] - 9.9, "end": w["end"] - 9.9} for w in HOSTS],
           "tail_s": 2.0}
    new = engine._confirm_crosstalk(clip, media, rec_fn=lambda m, a, b: asr)
    burned = json.loads(open(new["words_path"]).read())
    assert new["words_path"] != clip["words_path"]
    assert all(w.get("cap", True) is not False for w in burned)
    f = json.loads(runstate.clip(clip["clip_id"])["features_json"])["crosstalk_confirm"]
    assert f["flagged"] == 9 and len(f["burned"]) == 1
    # no second ASR: the flags stand, the clip as it was
    clip2, media2 = _clip(db, 9.9, 18.0, side, sid="Pod_w02")
    same = engine._confirm_crosstalk(clip2, media2, rec_fn=lambda *a: (_ for _ in ()).throw(
        TimeoutError("window cap near")))
    assert same["words_path"] == clip2["words_path"]
    # nothing flagged: no pass at all; off: nothing either
    clip3, media3 = _clip(db, 9.9, 18.0, HOSTS, sid="Pod_w03")
    assert engine._confirm_crosstalk(clip3, media3, rec_fn=lambda *a: pytest.fail("no pass")) \
        == clip3
    monkeypatch.setattr(engine, "crosstalk_confirm_enabled", lambda cfg=None: False)
    assert engine._confirm_crosstalk(clip, media, rec_fn=lambda *a: pytest.fail("off")) == clip
