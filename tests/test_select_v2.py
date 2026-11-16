"""PHASE4a unified select() tests (clipengine/engine.py). Fully offline: transcript, LLM, audio,
scene detection, ffmpeg/yt-dlp and storage are mocked; the SQLite DB and SCRATCH live in tmp_path.

Acceptance criteria covered:
  4. mocked LLM picks of different sentence spans -> variable-length clips, each starting at
     start_cut of its first post-filler-trim word (base cuts, and frame-RMS cuts with audio)
  5. LLM backend raises -> energy-peak fallback, every cut in a word gap, durations vary
  6. fixed-window selection is gone (greps engine.py and clipengine/*.py)
  7. the LLM runs for every vertical (general, poker, stream, podcast) with its rubric
 11. render() passes --words <sidecar> and --layout auto through to recompose
plus the low-speech path, persistence (words sidecar, features_json, boundary_type, layout) and
the precision pass (refined words used; failure keeps base cuts and warns).
"""
import importlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PIPE = Path(__file__).resolve().parent.parent / "clipengine"

from clipengine import boundaries as B
from clipengine import edge_verify
from clipengine import select_semantic
from clipengine import engine
from clipengine import runstate
from clipengine import transcript

SR = 16000
DUR = 480.0
SHIFT = 0.05
VOCAB = ("the river card came and he just shoved all in with nothing at that was crazy call fold "
         "money never again table pot huge bluff dealer turned over aces kings everyone stood up "
         "nobody believed it chips flew across felt final hand million dollars commentator "
         "screamed crowd went wild").split()
FILLER_OPENERS = ("So", "Um,", "Yeah,", "Well,", "Okay,")
PEAKS = ((130, 0.6, 4), (210, 0.45, 6), (290, 0.8, 3), (370, 0.35, 5))   # (second, height, half-width)
PEAK_TIMES = [c + 0.5 for c, _, _ in PEAKS]
FEATURE_KEYS = {"llm_score", "sub", "mean_z", "duration", "boundary_type", "ends_cleanly",
                "speech_density", "vertical", "backend", "model", "clean_start", "clean_end", "refined"}


def make_words(dur=DUR, seed=11):
    """Sentences of 4-28 words ending in . ? !, ~35% opening on a filler, 0.15-0.45 s words,
    40-220 ms gaps inside a sentence and 0.3-1.1 s pauses between sentences. Absolute times."""
    rng = random.Random(seed)
    words, t = [], 1.0
    while True:
        n = rng.randint(4, 28)
        if t + 0.7 * n > dur - 2.0:
            return words
        for k in range(n):
            tok = rng.choice(FILLER_OPENERS) if k == 0 and rng.random() < 0.35 else rng.choice(VOCAB)
            if k == n - 1:
                tok += rng.choice(".?!.")
            d = rng.uniform(0.15, 0.45)
            words.append({"w": tok, "start": round(t, 3), "end": round(t + d, 3)})
            t += d + (rng.uniform(0.3, 1.1) if k == n - 1 else rng.uniform(0.04, 0.22))


WORDS = make_words()
SENTS = transcript.sentences(WORDS, gap_s=0.7)


def shifted(words, dt=SHIFT):
    return [{"w": w["w"], "start": round(w["start"] + dt, 3), "end": round(w["end"] + dt, 3)}
            for w in words]


def peaky_env(dur=DUR, seed=3):
    """Per-second envelope: a 0.05-0.06 floor with four triangular peaks (PEAKS)."""
    env = 0.05 + 0.01 * np.random.default_rng(seed).random(int(dur))
    for c, h, w in PEAKS:
        for i in range(c - w, c + w + 1):
            env[i] = max(env[i], 0.05 + h * (1 - abs(i - c) / (w + 1)))
    return env


_WS = np.array([w["start"] for w in WORDS])
_WE = np.array([w["end"] for w in WORDS])
_DIP = np.append(_WE[:-1] + 0.3 * (_WS[1:] - _WE[:-1]), np.inf)   # 40 ms dip 30% into each gap


def music_bed_audio(media, lo, hi, sr=SR):
    """Function of absolute time: a two-tone music bed, 6x louder while a word is spoken, with a
    40 ms dip 30% of the way into every inter-word gap (not at the gap centre)."""
    t = lo + np.arange(int(round((hi - lo) * sr))) / sr
    bed = 0.05 * np.sin(2 * np.pi * 97 * t) + 0.04 * np.sin(2 * np.pi * 311 * t + 1.0)
    k = np.searchsorted(_WS, t, side="right") - 1
    speaking = (k >= 0) & (t < _WE[np.maximum(k, 0)])
    j = np.searchsorted(_WE, t, side="right") - 1
    dip = (j >= 0) & (np.abs(t - _DIP[np.maximum(j, 0)]) < 0.02)
    return (np.where(speaking, 6.0, np.where(dip, 0.02, 1.0)) * bed).astype(np.float32)


def env_audio(env):
    """load_audio stand-in whose 100 ms frame RMS equals the per-second envelope."""
    def load(media, lo, hi, sr=SR):
        t = lo + np.arange(int(round((hi - lo) * sr))) / sr
        amp = env[np.clip(t.astype(int), 0, len(env) - 1)] * np.sqrt(2)
        return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return load


def spans(targets, t0=100.0):
    """Non-overlapping sentence spans (s0, s1) whose raw length reaches each target (s). The first
    span opens on a trimmable filler so filler trimming is exercised."""
    k = next(i for i, s in enumerate(SENTS) if s["start"] >= t0
             and B.trim_leading_fillers(WORDS, s["w0"], s["w1"]) != s["w0"])
    out = []
    for tgt in targets:
        s1 = k
        while SENTS[s1]["end"] - SENTS[k]["start"] < tgt:
            s1 += 1
        out.append((k, s1))
        k = s1 + 2
    return out


def llm_answer(sp):
    return json.dumps({"clips": [
        {"s0": a, "s1": b, "hook": f"Hook {n}", "score": 90 - 5 * n, "reason": "test pick",
         "sub": {"hook": 80, "standalone": 70, "payoff": 60, "emotion": 50}}
        for n, (a, b) in enumerate(sp)]})


def expected_start(s0, s1, words=WORDS, **cut):
    """start_cut of the first post-filler-trim word of sentences s0..s1 -> (time, word index)."""
    w0 = B.trim_leading_fillers(words, SENTS[s0]["w0"], SENTS[s1]["w1"])
    return B.start_cut(words, w0, lead_pad=0.15, **cut), w0


def events():
    return [{k: e[k] for k in ("level", "stage", "message")} for e in runstate.events()]


def by_hook(clips):
    return {c["hook"]: c for c in clips}


def _no_subprocess(*a, **kw):
    raise AssertionError(f"unexpected subprocess call: {a}")


class Rig:
    """Mocks every external dependency of engine.select and records the calls."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp = tmp_path
        self.words, self.env = WORDS, peaky_env()
        self.answer = llm_answer(spans((16, 24, 34)))      # str, or an exception to raise
        self.window = "raise"                               # words_window: raise|same|shift|garbage
        self.audio = None                                   # load_audio impl; None -> raises
        self.prompts, self.window_calls, self.audio_calls, self.scene_calls = [], [], [], []
        monkeypatch.setattr(transcript, "get_words", self._get_words)
        monkeypatch.setattr(transcript, "words_window", self._words_window)
        monkeypatch.setattr(transcript, "load_audio", self._load_audio)
        monkeypatch.setattr(select_semantic, "_llm", self._llm)
        monkeypatch.setattr(engine.scenes, "scene_cuts", self._scene_cuts)
        monkeypatch.setattr(engine, "energy_envelope", lambda media: self.env)
        monkeypatch.setattr(subprocess, "run", _no_subprocess)
        monkeypatch.setattr(subprocess, "Popen", _no_subprocess)
        # PHASE4f edge-verify has its own dedicated tests (test_edge_verify.py); this Rig tests
        # the classic select flow, so make verify a no-op here.
        monkeypatch.setattr(edge_verify, "verify", lambda clip, fn, cfg: {
            "start": clip["start"], "end": clip["end"],
            "action": {"start": "noop", "end": "noop"}})

    def _get_words(self, media, model="base.en", cache_dir=None):
        assert model == engine.TRANSCRIBE_MODEL
        return [dict(w) for w in self.words]

    def _words_window(self, media, lo, hi, model="small.en"):
        self.window_calls.append((lo, hi, model))
        if self.window == "raise":
            raise RuntimeError("small.en unavailable")
        inside = [w for w in self.words if w["start"] >= lo and w["end"] <= hi]
        if self.window == "garbage":
            return [{"w": "zzz.", "start": w["start"], "end": w["end"]} for w in inside]
        return shifted(inside) if self.window == "shift" else [dict(w) for w in inside]

    def _load_audio(self, media, lo, hi, sr=SR):
        self.audio_calls.append((lo, hi))
        if self.audio is None:
            raise RuntimeError("ffmpeg unavailable")
        return self.audio(media, lo, hi, sr)

    def _llm(self, prompt, model, backend="ollama", timeout=None, opts=None):
        self.prompts.append(prompt)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer

    def _scene_cuts(self, media, lo, hi, threshold=0.3):
        self.scene_calls.append((lo, hi))
        return []

    def source(self, sid="src1", vertical="general", layout="auto"):
        media = self.tmp / f"{sid}.mp4"
        media.write_bytes(b"\0")
        runstate.add_source(sid, local_path=str(media), content_type=layout, vertical=vertical)
        runstate.update_source(sid, duration_s=DUR, state="harvested")
        return runstate.source(sid)

    def run(self, **kw):
        src = self.source(**kw)
        assert engine.select(src) is True
        assert runstate.source(src["source_id"])["state"] == "selected"
        return runstate.clips_for_source(src["source_id"])


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    runstate.reset()
    runstate.init()
    monkeypatch.setattr(engine, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(engine, "CAPTION_STRONG_MODEL", None)   # stage 3 has its own tests
    return Rig(tmp_path, monkeypatch)


def test_synthetic_transcript_is_realistic():
    assert len(WORDS) > 50 and len(SENTS) > 40
    assert sum(w["end"] - w["start"] for w in WORDS) / DUR > 0.5
    assert len({s["w1"] - s["w0"] for s in SENTS}) > 10                 # varied sentence lengths
    assert [s["i"] for s in SENTS] == list(range(len(SENTS)))
    assert sum(1 for s in SENTS if s["start"] >= 90 and s["end"] <= DUR - 30) > 20


# ---------------------------------------------------------------------------
# Criterion 4: variable-length clips, start == start_cut of the first post-trim word
# ---------------------------------------------------------------------------
TARGETS = (16, 24, 34, 45, 58)


def test_criterion_4_variable_length_clips_start_on_start_cut(rig):
    sp = spans(TARGETS)
    rig.answer = llm_answer(sp)
    clips = rig.run()
    assert len(clips) == 5 and len(rig.prompts) == 1
    got, durs = by_hook(clips), []
    for n, (s0, s1) in enumerate(sp):
        c = got[f"Hook {n}"]
        exp, w0 = expected_start(s0, s1)
        assert c["start_s"] == pytest.approx(exp, abs=1e-9)
        assert c["end_s"] in [pytest.approx(B.end_cut(WORDS, SENTS[s]["w1"], tail_pad=0.35), abs=1e-9)
                              for s in range(s1, s1 + 4)]              # a sentence end, maybe extended
        durs.append(c["end_s"] - c["start_s"])
    assert not all(abs(a - b) <= 5 for a in durs for b in durs)
    assert max(durs) - min(durs) > 20
    s0, s1 = sp[0]
    assert expected_start(s0, s1)[1] > SENTS[s0]["w0"]                   # a filler really was trimmed
    assert got["Hook 0"]["start_s"] > WORDS[SENTS[s0]["w0"]]["end"]


def test_criterion_4_with_audio_start_is_start_cut_on_frame_rms(rig):
    sp = spans(TARGETS)
    rig.answer, rig.window, rig.audio = llm_answer(sp), "same", music_bed_audio
    clips = rig.run()
    vc = engine.vertical_cfg("general")
    got, moved = by_hook(clips), 0
    assert len(clips) == 5
    for n, (s0, s1) in enumerate(sp):
        c = got[f"Hook {n}"]
        # filler endings: a pick whose last sentence ends on a dangling word (the
        # 5th one's "... aces and.") ends on the next clean sentence end instead
        s1 = next(s for s in range(s1, len(SENTS))
                  if not B.dangling_end(SENTS[s]["text"].split()[-1]))
        base = B.clip_from_sentences(WORDS, SENTS, s0, s1, cfg=vc)
        lo, hi = max(0.0, base["start"] - 8), min(DUR, base["end"] + 8)
        assert (lo, hi, "small.en") in rig.window_calls and (lo, hi) in rig.audio_calls
        rms = B.frame_rms(music_bed_audio(None, lo, hi), SR, 0.01)
        exp, w0 = expected_start(s0, s1, rms=rms, t0=lo, scene_times=[])
        assert c["start_s"] == pytest.approx(exp, abs=1e-9)
        blo, bhi = B.start_bounds(WORDS, w0)
        assert blo <= c["start_s"] <= bhi
        moved += abs(c["start_s"] - base["start"]) > 1e-3
    assert moved >= 3                                                     # audio moved cuts into the dips
    assert not [e for e in events() if "precision pass failed" in e["message"]]
    durs = [c["end_s"] - c["start_s"] for c in clips]
    assert max(durs) - min(durs) > 5


# ---------------------------------------------------------------------------
# Criterion 5: LLM down -> energy-peak fallback, cuts in word gaps, durations vary
# ---------------------------------------------------------------------------
def _assert_cuts_in_word_gaps(c, words=WORDS):
    st, en = c["start_s"], c["end_s"]
    first = next(k for k, w in enumerate(words) if (w["start"] + w["end"]) / 2 >= st)
    last = max(k for k, w in enumerate(words) if (w["start"] + w["end"]) / 2 <= en)
    lo, hi = B.start_bounds(words, first)
    assert lo <= st <= hi
    lo, hi = B.end_bounds(words, last)
    assert lo <= en <= hi
    assert not any(w["start"] < t < w["end"] for w in words for t in (st, en))


def test_criterion_5_llm_down_falls_back_to_energy_peaks_on_word_gaps(rig):
    rig.answer = ConnectionError("ollama down")
    clips = rig.run()
    assert rig.prompts                                                    # the LLM was tried
    assert len(clips) == len(PEAKS)
    for c in clips:
        _assert_cuts_in_word_gaps(c)
        assert any(c["start_s"] - 1 <= p <= c["end_s"] + 1 for p in PEAK_TIMES)
        f = json.loads(c["features_json"])
        assert (f["backend"], f["model"], f["llm_score"]) == ("energy", None, None)
        assert c["boundary_type"] == "sentence" and c["hook"] is None
        assert 15 <= c["end_s"] - c["start_s"] <= 75
    durs = sorted(c["end_s"] - c["start_s"] for c in clips)
    assert len({round(d, 1) for d in durs}) == len(durs) and durs[-1] - durs[0] > 1.0
    assert any(e["level"] == "warn" and "LLM unavailable" in e["message"] for e in events())


def test_out_of_band_llm_picks_rejected_then_energy_fallback(rig):
    k = spans((1,))[0][0]
    s1 = next(s for s in range(k, len(SENTS)) if SENTS[s]["end"] - SENTS[k]["start"] > 90)
    rig.answer = llm_answer([(k, s1)])                                    # > max_dur 75: never truncated
    clips = rig.run()
    assert clips and "Hook 0" not in by_hook(clips)
    assert all(json.loads(c["features_json"])["backend"] == "energy" for c in clips)
    assert any("no usable LLM picks" in e["message"] for e in events())


# ---------------------------------------------------------------------------
# Criterion 6: fixed-window selection is gone
# ---------------------------------------------------------------------------
def test_criterion_6_fixed_window_code_is_gone():
    src = (PIPE / "engine.py").read_text()
    assert "start + (en - st)" not in src
    assert "window_s" not in src
    assert not re.search(r"\bWIN\b", src)
    for f in sorted(PIPE.glob("*.py")):
        text = f.read_text()
        assert "window_s" not in text, f.name
        assert not re.search(r"\bWIN\b", text), f.name
    for gone in ("WIN", "_energy_candidates", "_silences", "_snap_start", "_snap_end", "_select_semantic"):
        assert not hasattr(engine, gone), gone


# ---------------------------------------------------------------------------
# Criterion 7: the semantic path runs for every vertical
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("vertical", ["general", "poker", "stream", "podcast"])
def test_criterion_7_semantic_path_runs_for_every_vertical(rig, vertical):
    clips = rig.run(vertical=vertical)
    vc = engine.vertical_cfg(vertical)
    assert len(rig.prompts) == 1
    prompt = rig.prompts[0]
    assert select_semantic.RUBRICS[vertical] in prompt
    assert f"hard limits {vc['min_dur_s']}-{vc['max_dur_s']}s" in prompt
    assert "⚡" in prompt                                                 # sentences carry mean z
    assert "[S0 " not in prompt and f"[S{len(SENTS) - 1} " not in prompt  # skip_head / skip_tail
    assert {c["hook"] for c in clips} == {"Hook 0", "Hook 1", "Hook 2"}
    for c in clips:
        f = json.loads(c["features_json"])
        assert (f["vertical"], f["backend"], f["model"]) == (vertical, vc["semantic_backend"],
                                                             vc["semantic_model"])
        assert vc["min_dur_s"] <= c["end_s"] - c["start_s"] <= vc["max_dur_s"]
        want = (vc["w_llm"] * f["llm_score"] / 100 + vc["w_energy"] * engine.sigmoid(f["mean_z"])
                - (0.0 if f["ends_cleanly"] else 0.1))
        assert c["score"] == pytest.approx(want, abs=1e-3)
        assert f["sub"] == {"hook": 80, "standalone": 70, "payoff": 60, "emotion": 50}


# ---------------------------------------------------------------------------
# Criterion 11: render passes --words and --layout auto
# ---------------------------------------------------------------------------
def _render(rig, monkeypatch, layout="auto", words_path="exists"):
    media = rig.tmp / "r1.mp4"
    media.write_bytes(b"\0")
    runstate.add_source("r1", local_path=str(media), content_type=layout)
    runstate.update_source("r1", state="selected", duration_s=DUR)
    wp = rig.tmp / "r1_words.json"
    if words_path == "exists":
        wp.write_text(json.dumps(WORDS[:5]))
    cid = runstate.add_clip("r1", 10.0, 31.5, 0.7, layout, hook="A hook",
                         words_path=None if words_path is None else str(wp))
    calls = []
    qc = 'QC {"width": 1080, "height": 1920, "layout": "track", "passed": true, "errors": []}'
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd) or
                        subprocess.CompletedProcess(cmd, 0, stdout=f"DONE x\n{qc}\n", stderr=""))
    monkeypatch.setattr(engine, "_store_clip", lambda f, clip: f"file:///store/{clip['source_id']}/clip_{clip['clip_id']}.mp4")
    assert engine.render(runstate.clip(cid)) is True
    assert runstate.clip(cid)["state"] == "rendered"
    assert len(calls) == 1
    return calls[0], str(wp)


def test_criterion_11_render_passes_words_and_layout_auto(rig, monkeypatch):
    cmd, wp = _render(rig, monkeypatch)
    assert cmd[cmd.index("--words") + 1] == wp
    assert cmd[cmd.index("--layout") + 1] == "auto"
    assert cmd[cmd.index("--hook") + 1] == "A hook"
    assert cmd[:3] == engine.RECOMPOSE and cmd[4:6] == ["10.0", "21.5"]


@pytest.mark.parametrize("words_path", ["missing", None])
def test_render_omits_words_without_sidecar(rig, monkeypatch, words_path):
    cmd, _ = _render(rig, monkeypatch, layout="track", words_path=words_path)
    assert "--words" not in cmd
    assert cmd[cmd.index("--layout") + 1] == "track"


def test_render_qc_failure_marks_clip_failed_with_reason(rig, monkeypatch):
    media = rig.tmp / "r2.mp4"
    media.write_bytes(b"\0")
    runstate.add_source("r2", local_path=str(media), content_type="auto")
    runstate.update_source("r2", state="selected", duration_s=DUR)
    cid = runstate.add_clip("r2", 10.0, 31.5, 0.7, "auto")
    qc = 'QC {"width": 1080, "height": 1080, "passed": false, "errors": ["size 1080x1080 != 1080x1920"]}'
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw:
                        subprocess.CompletedProcess(cmd, 3, stdout=qc + "\n", stderr=""))
    put = []
    monkeypatch.setattr(engine, "_store_clip", lambda f, clip: put.append(clip))
    assert engine.render(runstate.clip(cid)) is False
    c = runstate.clip(cid)
    assert c["state"] == "failed" and "exit 3" in c["error"] and "1080x1080" in c["error"]
    assert put == []


# ---------------------------------------------------------------------------
# Low-speech path
# ---------------------------------------------------------------------------
def _assert_energy_clips(rig, clips, vertical):
    vc = engine.vertical_cfg(vertical)
    assert clips and rig.prompts == []                                    # no transcript-driven LLM call
    for c in clips:
        assert c["boundary_type"] == "energy" and c["hook"] is None
        assert vc["min_dur_s"] <= c["end_s"] - c["start_s"] <= vc["max_dur_s"]
        assert any(c["start_s"] < p < c["end_s"] for p in PEAK_TIMES)
        f = json.loads(c["features_json"])
        assert FEATURE_KEYS <= set(f)
        assert (f["boundary_type"], f["backend"], f["llm_score"], f["ends_cleanly"]) == ("energy", "energy", None, None)
        assert c["score"] == pytest.approx(engine.sigmoid(f["mean_z"]), abs=1e-3)
        assert Path(c["words_path"]).exists()


def test_low_speech_few_words_uses_energy_envelope(rig):
    rig.words, rig.audio = WORDS[:30], env_audio(rig.env)
    clips = rig.run(vertical="stream")
    _assert_energy_clips(rig, clips, "stream")
    assert len(clips) == len(PEAKS)
    assert rig.audio_calls and rig.scene_calls                            # fine 100 ms envelope + shot changes
    assert not [e for e in events() if "fine envelope failed" in e["message"]]


def test_low_speech_without_audio_uses_one_second_envelope(rig):
    rig.words = WORDS[:30]                                                # load_audio raises
    clips = rig.run(vertical="poker")
    _assert_energy_clips(rig, clips, "poker")
    assert any(e["level"] == "warn" and "fine envelope failed" in e["message"] for e in events())


def test_low_speech_sparse_speech_density(rig):
    rig.words = [{"w": f"word{k}.", "start": 5.0 + 7 * k, "end": 5.3 + 7 * k} for k in range(60)]
    rig.audio = env_audio(rig.env)
    clips = rig.run(vertical="general")                                   # >= 50 words, density 0.04
    _assert_energy_clips(rig, clips, "general")
    for c in clips:
        f = json.loads(c["features_json"])
        assert f["speech_density"] < 0.15
        ws = json.loads(Path(c["words_path"]).read_text())
        assert all(c["start_s"] <= (w["start"] + w["end"]) / 2 <= c["end_s"] for w in ws)


def test_transcription_failure_goes_low_speech(rig, monkeypatch):
    monkeypatch.setattr(transcript, "get_words", lambda *a, **k: (_ for _ in ()).throw(OSError("no whisper")))
    rig.audio = env_audio(rig.env)
    clips = rig.run()
    _assert_energy_clips(rig, clips, "general")
    assert any("transcription failed" in e["message"] for e in events())


# ---------------------------------------------------------------------------
# Persistence: words sidecar, features_json, boundary_type, layout
# ---------------------------------------------------------------------------
def test_persist_words_sidecar_features_and_boundary_type(rig):
    clips = rig.run(sid="pers")
    assert len(clips) == 3
    for c in clips:
        p = Path(c["words_path"])
        assert p.parent == rig.tmp / "scratch" / "words" and p.name == f"pers_{c['start_s']:.2f}.json"
        ws = json.loads(p.read_text())
        assert ws and all(set(w) == {"w", "start", "end"} for w in ws)
        assert ws == [w for w in WORDS if c["start_s"] <= (w["start"] + w["end"]) / 2 <= c["end_s"]]
        assert c["start_s"] <= ws[0]["start"] and ws[-1]["end"] <= c["end_s"]   # no word is cut
        assert [w["start"] for w in ws] == sorted(w["start"] for w in ws)
        f = json.loads(c["features_json"])
        assert FEATURE_KEYS <= set(f)
        assert c["boundary_type"] == f["boundary_type"] == "sentence"
        assert f["duration"] == pytest.approx(c["end_s"] - c["start_s"], abs=1e-3)
        assert f["speech_density"] > 0.15 and f["vertical"] == "general"
        assert isinstance(f["ends_cleanly"], bool) and isinstance(f["mean_z"], float)
        assert f["clean_start"] is True and f["clean_end"] is True and f["refined"] is False
        assert c["layout"] == "auto" and c["state"] == "selected"
    msg = [e["message"] for e in events() if e["level"] == "info" and e["stage"] == "select"][-1]
    assert "3 clips via semantic path" in msg and "durations=" in msg


def test_semantic_pick_after_comma_split_starts_on_true_sentence_start(rig):
    """A pick whose first sentence only exists because of a comma + short pause split is moved back
    to the true sentence start; flags land in features_json; a clean refined pass is used."""
    ws = [dict(w) for w in WORDS]
    sents = transcript.sentences(ws, gap_s=0.7)
    k = next(k for k in range(1, len(sents) - 8) if sents[k]["start"] > 120
             and 0.7 <= sents[k]["start"] - sents[k - 1]["end"] < 1.5
             and sents[k - 1]["w1"] - sents[k - 1]["w0"] >= 4)
    last = ws[sents[k - 1]["w1"]]
    last["w"] = last["w"].rstrip(".?!") + ","                              # "... Flow," | pause | "you're ..."
    rig.words = ws
    sents = transcript.sentences(ws, gap_s=0.7)
    assert sents[k]["w0"] == sents[k - 1]["w1"] + 1 and sents[k - 1]["text"].endswith(",")
    s1 = next(s for s in range(k, len(sents)) if sents[s]["end"] - sents[k]["start"] >= 20)
    rig.answer, rig.window = llm_answer([(k, s1)]), "same"
    [c] = rig.run()
    f = json.loads(c["features_json"])
    w0 = B.trim_leading_fillers(ws, sents[k - 1]["w0"], sents[s1]["w1"])
    assert c["start_s"] == pytest.approx(B.start_cut(ws, w0, lead_pad=0.15), abs=1e-9)
    assert c["start_s"] < sents[k - 1]["end"]                             # starts before the comma clause ends
    assert (f["clean_start"], f["clean_end"], f["refined"]) == (True, True, True)
    assert not [e for e in events() if "clean_start=false" in e["message"]]


@pytest.mark.parametrize("given,stored", [("auto", "auto"), (None, "auto"), ("track", "track"),
                                          ("screencast", "screencast"), ("split", "split"),
                                          ("general", "general"), ("weird", "auto")])
def test_layout_passthrough_never_coerced_to_general(rig, given, stored):
    clips = rig.run(layout=given)
    assert clips and all(c["layout"] == stored for c in clips)
    warned = any(e["level"] == "warn" and "unknown layout" in e["message"] for e in events())
    assert warned == (given == "weird")


def test_rerun_is_idempotent(rig):
    first = rig.run(sid="again")
    assert engine.select(runstate.source("again")) is True
    assert [c["clip_id"] for c in runstate.clips_for_source("again")] == [c["clip_id"] for c in first]


def test_select_failure_marks_source_failed(rig, monkeypatch):
    monkeypatch.setattr(transcript, "sentences", lambda *a, **k: 1 / 0)
    src = rig.source(sid="boom")
    assert engine.select(src) is False
    assert runstate.source("boom")["state"] == "failed"
    assert any(e["level"] == "error" for e in events())


# ---------------------------------------------------------------------------
# Precision pass
# ---------------------------------------------------------------------------
def test_precision_pass_uses_refined_words(rig):
    sp = spans((16, 24, 34))
    rig.answer, rig.window = llm_answer(sp), "shift"      # small.en hears every word 50 ms later; no audio
    clips = rig.run()
    sh = shifted(WORDS)
    got = by_hook(clips)
    for n, (s0, s1) in enumerate(sp):
        c = got[f"Hook {n}"]
        exp, w0 = expected_start(s0, s1, words=sh)
        assert c["start_s"] == pytest.approx(exp, abs=1e-9)
        assert c["start_s"] == pytest.approx(expected_start(s0, s1)[0] + SHIFT, abs=2e-3)
        ws = json.loads(Path(c["words_path"]).read_text())
        assert ws[0] == sh[w0]                                            # sidecar = refined words
    ev = events()
    assert not [e for e in ev if "precision pass failed" in e["message"]]
    assert any(e["level"] == "warn" and "no audio" in e["message"] for e in ev)   # rms None, words still refined


@pytest.mark.parametrize("window", ["raise", "garbage"])
def test_precision_pass_failure_keeps_base_cuts_and_warns(rig, window):
    sp = spans((16, 24, 34))
    rig.answer, rig.window = llm_answer(sp), window
    clips = rig.run()
    got = by_hook(clips)
    for n, (s0, s1) in enumerate(sp):
        assert got[f"Hook {n}"]["start_s"] == pytest.approx(expected_start(s0, s1)[0], abs=1e-9)
        ws = json.loads(Path(got[f"Hook {n}"]["words_path"]).read_text())
        assert ws and all(w in WORDS for w in ws)                         # base words kept
    fails = [e for e in events() if e["level"] == "warn" and "precision pass failed" in e["message"]]
    assert len(fails) == len(clips) == 3


def test_locate_fuzzy_match_with_time_tiebreak():
    base = {"text": "He shoved all in with nothing at all.", "start": 100.0, "end": 103.0}
    rs = [{"text": "What a hand.", "start": 96.0, "end": 97.0},
          {"text": "Um, he shoved all-in with nothing at all!", "start": 100.1, "end": 103.1},
          {"text": "Nobody believed it.", "start": 104.0, "end": 105.0}]
    assert engine._locate(rs, base, "start") == 1
    twice = [{"text": base["text"], "start": 40.0, "end": 43.0}, {"text": base["text"], "start": 100.2, "end": 103.4}]
    assert engine._locate(twice, base, "start") == 1 and engine._locate(twice, base, "end") == 1
    assert engine._locate([{"text": "completely different words", "start": 100.0, "end": 101.0}], base, "start") is None
    assert engine._locate([], base, "end") is None


def test_partial_ratio_without_rapidfuzz_uses_difflib(monkeypatch):
    fast = engine._partial_ratio("shoved all in", "he shoved all in with nothing")
    monkeypatch.setitem(sys.modules, "rapidfuzz", None)                   # /usr/bin/python3 has no rapidfuzz
    assert engine._partial_ratio("shoved all in", "he shoved all in with nothing") == fast == 100
    assert engine._partial_ratio("abc", "xyz") < 50
    assert engine._partial_ratio("", "abc") == 0


# ---------------------------------------------------------------------------
# Energy helpers, harvest
# ---------------------------------------------------------------------------
class _Pipe:
    """stdout stand-in that hands out odd-sized chunks regardless of the size asked for."""

    def __init__(self, data, sizes=(7777, 12345, 3, 40001)):
        self.data, self.sizes, self.i = data, sizes, 0

    def read(self, n):
        k = min(n, self.sizes[self.i % len(self.sizes)])
        self.i += 1
        out, self.data = self.data[:k], self.data[k:]
        return out

    def close(self):
        pass


class _Proc:
    def __init__(self, data, rc=0):
        self.stdout, self.rc = _Pipe(data), rc

    def wait(self):
        return self.rc


def test_energy_envelope_streams_per_second_rms(monkeypatch):
    x = (np.random.default_rng(0).standard_normal(SR * 3 + 1234) * 3000).astype("<i2")
    x[SR:2 * SR] //= 10                                                   # a quieter second
    seen = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: seen.append(cmd) or _Proc(x.tobytes()))
    env = engine.energy_envelope("m.mp4")
    ref = np.sqrt(((x[:3 * SR] / 32768.0).reshape(3, SR) ** 2).mean(axis=1))
    assert env.shape == (3,) and np.allclose(env, ref)                    # partial last second dropped
    assert env[1] < env[0] / 5 and env.max() < 1.0                        # float [-1, 1] scale
    cmd = seen[0]
    assert cmd[:2] == ["ffmpeg", "-nostdin"] and cmd[cmd.index("-f") + 1] == "s16le"
    assert cmd[cmd.index("-ar") + 1] == "16000" and cmd[cmd.index("-ac") + 1] == "1"
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: _Proc(b"", rc=1))
    with pytest.raises(RuntimeError):
        engine.energy_envelope("m.mp4")


def test_zscores_mean_z_sigmoid():
    z = engine.zscores([1.0, 2.0, 3.0, 4.0])
    assert z.mean() == pytest.approx(0) and z.std() == pytest.approx(1)
    assert not engine.zscores([0.3] * 5).any() and len(engine.zscores([])) == 0
    assert engine.mean_z(np.array([0.0, 1.0, 2.0, 3.0]), 1.2, 2.5) == pytest.approx(1.5)
    assert engine.mean_z(np.array([1.0]), 5, 9) == 0.0
    assert engine.sigmoid(0) == 0.5 and engine.sigmoid(-1e6) == 0.0 and engine.sigmoid(1e6) == 1.0


def test_energy_peaks_strongest_first_suppressed_and_skipped():
    env = np.full(600, 0.05)
    env[20] = env[590] = 2.0                                              # intro / outro stings
    env[200], env[230], env[400] = 1.0, 0.9, 1.5                          # 230 is within 60 s of 200
    z = engine.zscores(env)
    assert engine.energy_peaks(z, top_n=5, head_skip=90, tail_skip=30) == [400.5, 200.5]
    assert engine.energy_peaks(z, top_n=1, head_skip=90, tail_skip=30) == [400.5]
    assert engine.energy_peaks(z, top_n=5, head_skip=90, tail_skip=30, suppress_s=10) == [400.5, 200.5, 230.5]
    assert engine.energy_peaks(engine.zscores(np.full(300, 0.2)), 5) == []


def test_vertical_cfg_merges_band_and_defaults_to_general():
    p = engine.vertical_cfg("poker")
    assert (p["min_dur_s"], p["max_dur_s"], p["w_llm"], p["w_energy"]) == (15, 60, 0.6, 0.4)
    assert p["semantic_backend"] == "claude_cli" and p["semantic_fallback_backend"] == "ollama" and p["lead_pad_s"] == 0.15 and "verticals" not in p
    assert engine.vertical_cfg("nope") == engine.vertical_cfg("general")
    assert engine.VALID_LAYOUTS == ("auto", "general", "track", "split", "screencast",
                                    "broadcast", "hudband", "reaction", "gamestack")


def test_harvest_downloads_up_to_1080p_mp4(rig, monkeypatch):
    monkeypatch.setattr(engine, "COOKIES", "")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, **kw: b"123.4\n")
    runstate.add_source("yt1", url="https://youtu.be/abcdefghijk")
    assert engine.harvest(runstate.source("yt1")) is True
    cmd = calls[0]
    assert cmd[0] == "yt-dlp"
    assert cmd[cmd.index("-f") + 1] == \
        "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/b[height<=1080]"
    assert cmd[cmd.index("--merge-output-format") + 1] == "mp4"
    src = runstate.source("yt1")
    assert src["state"] == "harvested" and src["duration_s"] == 123.4


# ---------------------------------------------------------------------------
# an explicit empty answer
# ---------------------------------------------------------------------------
def test_llm_declining_every_moment_gets_no_energy_fallback(rig):
    """One window: claude answered {"clips":[]}; select logged "LLM unavailable"
    and published an energy peak of the very window the LLM rejected."""
    rig.answer = '{"clips": []}'
    assert rig.run() == []
    assert rig.prompts                                                    # the LLM was asked
    msgs = [e["message"] for e in events()]
    assert any("LLM answered with no clip-worthy moment" in m for m in msgs)
    assert not any("LLM unavailable" in m or "energy-peak fallback" in m for m in msgs)


def _fake_prediction(bad_start):
    def predict(media, start_s, end_s, fps=2.0, fail_pct_max=2.0, broll_min_span_s=3.0):
        if abs(start_s - bad_start) < 1e-6:
            return {"passed": False, "fail_pct": 14.5, "frac": 0.85}
        return {"passed": None if start_s > 300 else True, "fail_pct": 0.0, "frac": 1.0}
    return predict


def test_speaker_crop_pick_without_a_face_on_the_source_is_skipped_before_render(rig, monkeypatch):
    sp = spans((16, 24, 34))
    rig.answer = llm_answer(sp)
    bad, _ = expected_start(*sp[1])
    monkeypatch.setattr(engine.pick_quality, "predict_frame_gate", _fake_prediction(bad))
    clips = rig.run(vertical="general", layout="auto")                   # auto -> track crop
    assert set(by_hook(clips)) == {"Hook 0", "Hook 2"}
    assert json.loads(by_hook(clips)["Hook 0"]["features_json"])["source_frame_gate"]["passed"] is True
    assert any("before render" in e["message"] and "14.5%" in e["message"] for e in events())


def test_source_face_precheck_only_runs_for_speaker_crop_layouts(rig, monkeypatch):
    calls = []
    monkeypatch.setattr(engine.pick_quality, "predict_frame_gate",
                        lambda *a, **k: calls.append(a) or {"passed": False, "fail_pct": 50.0})
    assert rig.run(sid="g1", vertical="stream", layout="auto") and calls == []
    assert rig.run(sid="g2", vertical="general", layout="gamestack") and calls == []
    assert [engine._speaker_crop_layout(l, v) for l, v in (
        ("auto", "general"), ("auto", "podcast"), ("track", "general"), ("auto", "gaming"),
        ("screencast", "general"), ("auto", "unmapped"))] == [True, True, True, False, False, False]


def _render_clamp(rig, monkeypatch, start, end, duration_s, ffprobe_json=None):
    media = rig.tmp / "w.mp4"
    media.write_bytes(b"\0")
    runstate.add_source("w1", local_path=str(media), content_type="auto")
    runstate.update_source("w1", state="selected", duration_s=duration_s)
    cid = runstate.add_clip("w1", start, end, 0.6, "auto")
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(ffprobe_json or {}),
                                               stderr="")
        qc = 'QC {"width": 1080, "height": 1920, "layout": "track", "passed": true, "errors": []}'
        return subprocess.CompletedProcess(cmd, 0, stdout=f"DONE x\n{qc}\n", stderr="")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(engine, "_store_clip", lambda f, clip: f"file:///store/{clip['source_id']}/clip_{clip['clip_id']}.mp4")
    ok = engine.render(runstate.clip(cid))
    return ok, cid, [c for c in calls if c[:3] == engine.RECOMPOSE]


def test_render_clamps_cut_end_to_the_media_end_before_recompose(rig, monkeypatch):
    """One 34.0 s window: edge-verify extended the end to 34.335 s, so
    recompose was asked for 31.99 s, rendered the 31.667 s that exist, and QC rejected it."""
    ok, cid, rc = _render_clamp(rig, monkeypatch, 2.345, 34.335, 34.0,
                                ffprobe_json={"format": {"duration": "34.0"},
                                              "streams": [{"duration": "34.0"},
                                                          {"duration": "34.01"}]})
    assert ok is True and len(rc) == 1
    assert rc[0][4:6] == ["2.345", str(34.0 - 2.345)]
    assert float(rc[0][5]) == pytest.approx(31.655)
    assert runstate.clip(cid)["end_s"] == 34.0
    assert any("clamped to 34.000s" in e["message"] for e in events())


def test_render_clamp_uses_the_shortest_stream_from_ffprobe(rig, monkeypatch):
    ok, cid, rc = _render_clamp(rig, monkeypatch, 5.0, 34.5, None,
                                ffprobe_json={"format": {"duration": "34.333333"},
                                              "streams": [{"duration": "34.333333"},
                                                          {"duration": "34.310000"}]})
    assert ok is True and runstate.clip(cid)["end_s"] == 34.31
    assert float(rc[0][5]) == pytest.approx(29.31)


def test_render_leaves_cuts_inside_the_media_untouched(rig, monkeypatch):
    ok, cid, rc = _render_clamp(rig, monkeypatch, 10.0, 31.5, DUR)
    assert ok is True and rc[0][4:6] == ["10.0", "21.5"]
    assert runstate.clip(cid)["end_s"] == 31.5


def test_render_fails_cleanly_when_the_clamp_leaves_no_clip(rig, monkeypatch):
    ok, cid, rc = _render_clamp(rig, monkeypatch, 31.0, 40.0, 34.0)
    assert ok is False and rc == []                                      # recompose never ran
    assert "runs past media end" in runstate.clip(cid)["error"]


def test_clamp_cut_end_unit():
    assert engine.clamp_cut_end(2.345, 34.335, 34.0) == 34.0
    assert engine.clamp_cut_end(2.345, 30.0, 34.0) == 30.0
    assert engine.clamp_cut_end(2.345, 34.335, None) == 34.335            # unknown length: as-is
    assert engine.clamp_cut_end(0.0, 40.0, 34.3339) == 34.333            # floored, never past
    with pytest.raises(RuntimeError):
        engine.clamp_cut_end(31.0, 40.0, 34.0)


# ---------------------------------------------------------------------------
# every select() ends with one `[select] <sid>: timings ...` line + events row, and
# reports the step in flight to SELECT_STEP_HOOK (a runner's per-window cap).
# ---------------------------------------------------------------------------
def test_select_logs_per_step_timings_and_reports_steps(rig, monkeypatch, capsys):
    steps = []
    monkeypatch.setattr(engine, "SELECT_STEP_HOOK", steps.append)
    rig.window = "same"
    rig.run(sid="timed")
    msgs = [e["message"] for e in events() if e["stage"] == "select" and ": timings " in e["message"]]
    assert len(msgs) == 1
    line = msgs[0]
    assert line.startswith("timed: timings transcribe=")
    for key in ("energy=", "llm=", "precision=", "edge_verify=", "persist=", "total="):
        assert key in line, (key, line)
    be, model = engine.CFG["select"]["semantic_backend"], engine.CFG["select"]["semantic_model"]
    assert f"llm_via={be}/{model}=answered@" in line
    assert "[select] timed: timings transcribe=" in capsys.readouterr().out
    assert steps[:3] == ["select:transcribe", "select:energy", "select:llm"]
    assert "select:precision" in steps and "select:edge_verify" in steps


def test_select_timings_name_the_fallback_backend(rig, monkeypatch):
    calls = []

    def llm(prompt, model, backend="ollama", timeout=None, opts=None):
        calls.append(backend)
        if backend == "claude_cli":
            raise RuntimeError("claude timed out after 300s")
        return rig.answer
    monkeypatch.setattr(select_semantic, "_llm", llm)
    monkeypatch.setitem(engine.CFG["select"], "semantic_backend", "claude_cli")
    monkeypatch.setitem(engine.CFG["select"], "semantic_fallback_backend", "ollama")
    monkeypatch.setitem(engine.CFG["select"], "semantic_fallback_model", "qwen2.5:7b")
    rig.run(sid="fb")
    line = [e["message"] for e in events() if ": timings " in e["message"]][0]
    assert "=error@" in line and "ollama/qwen2.5:7b=answered@" in line
    assert calls[:2] == ["claude_cli", "ollama"]


def test_select_failure_still_logs_timings(rig, monkeypatch):
    monkeypatch.setattr(engine, "energy_envelope", lambda media: 1 / 0)
    monkeypatch.setattr(engine, "zscores", lambda env: (_ for _ in ()).throw(RuntimeError("boom")))
    src = rig.source(sid="broken")
    assert engine.select(src) is False
    assert any(e["message"].startswith("broken: timings ") for e in events())


def test_select_past_deadline_skips_precision_and_edge_verify_keeps_base_cuts(rig, monkeypatch):
    """near the window cap the remaining candidates keep their base cuts (the failed-
    precision-pass fallback) instead of burning the window on small.en passes."""
    import time as _time
    monkeypatch.setattr(engine, "SELECT_DEADLINE", _time.monotonic() - 1.0)
    rig.window = "same"
    clips = rig.run(sid="late")
    assert clips and rig.window_calls == []
    for c in clips:
        f = json.loads(c["features_json"])
        assert f["refined"] is False and f["refine_error"] == "skipped: window time cap near"
        assert f["verify_action"] == {"start": "skipped", "end": "skipped"}
    assert any("window time cap near" in e["message"] for e in events())


# ---------------------------------------------------------------------------
# a candidate span the crop probe finds no clean crop for leaves before its render and
# the other candidates take the slots; a window the probe passes selects exactly as before.
# ---------------------------------------------------------------------------
def test_a_sliced_candidate_span_leaves_and_the_rest_select_as_before(rig, monkeypatch):
    base = rig.run(sid="plain")                      # no niche: no probe, today's flow
    assert len(base) == 3
    bad = base[0]
    monkeypatch.setattr(engine, "_niche_of", lambda sid: "poker")
    verdicts = [[float(t), not (bad["start_s"] <= t <= bad["end_s"]), "x"]
                for t in range(int(DUR) + 1)]
    monkeypatch.setattr(engine, "crop_probe", lambda media, sid, niche: {
        "verdict": "ok", "why": "", "elapsed_s": 0.1, "verdicts": verdicts})
    got = rig.run(sid="probed")
    assert [c["hook"] for c in got] == [c["hook"] for c in base[1:]]
    assert [(c["start_s"], c["end_s"]) for c in got] == [(c["start_s"], c["end_s"])
                                                        for c in base[1:]]
    assert all(json.loads(c["features_json"])["crop_probe"]["ok_share"] == 1.0 for c in got)
    msgs = [e["message"] for e in events() if e["stage"] == "select"]
    assert any(m.startswith("probed: skipped") and "before render: no clean crop" in m
               for m in msgs)
    clean = [[float(t), True, "x"] for t in range(int(DUR) + 1)]
    monkeypatch.setattr(engine, "crop_probe", lambda media, sid, niche: {
        "verdict": "ok", "why": "", "elapsed_s": 0.1, "verdicts": clean})
    assert [(c["start_s"], c["end_s"], c["hook"]) for c in rig.run(sid="clean")] == \
        [(c["start_s"], c["end_s"], c["hook"]) for c in base]
