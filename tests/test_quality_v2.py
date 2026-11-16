"""Quality v2 tests (offline): clean sentence starts/ends (boundaries.clean_clip), the fuzzy
precision-pass locate (engine._locate / _refine), the claude_cli account env and the per-chunk
semantic fallback. No network, no whisper, no claude, no ffmpeg."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

PIPE = Path(__file__).resolve().parent.parent / "clipengine"

from clipengine import boundaries as B
from clipengine import select_semantic
from clipengine import engine
from clipengine import transcript


def _mk(segments, t=10.0, word_s=0.3, gap=0.1):
    """segments: [(text, pause_after_s)]. Words 0.3 s long, 0.1 s apart inside a segment."""
    words = []
    for text, pause in segments:
        toks = text.split()
        for k, tok in enumerate(toks):
            words.append({"w": tok, "start": round(t, 3), "end": round(t + word_s, 3)})
            t += word_s + (pause if k == len(toks) - 1 else gap)
    return words, transcript.sentences(words, gap_s=0.7)


def _cfg(lo, hi, **kw):
    return {"min_dur_s": lo, "max_dur_s": hi, "lead_pad_s": 0.15, "tail_pad_s": 0.35, **kw}


def _first(words, c):
    return words[c["w0"]]["w"]


# ---------------------------------------------------------------------------
# clean_edges
# ---------------------------------------------------------------------------
def test_clean_edges_punctuation_hard_gap_and_window_edges():
    words, sents = _mk([("Intro line here.", 0.8), ("He played Stole Your Flow,", 0.9),
                        ("and you were not happy", 1.6), ("then it got worse", 0.8)])
    assert [s["text"] for s in sents] == ["Intro line here.", "He played Stole Your Flow,",
                                          "and you were not happy", "then it got worse"]
    cs, ce = transcript.clean_edges(sents, hard_gap_s=1.5)
    assert ce == [True, False, True, True]          # ".", comma+0.9 s, 1.6 s pause, last sentence
    assert cs == [True, True, False, True]
    cs, ce = transcript.clean_edges(sents, hard_gap_s=1.5, edges_clean=False)
    assert (cs[0], ce[-1]) == (False, False)        # a window's edges are not true boundaries
    assert transcript.clean_edges([], 1.5) == ([], [])


# ---------------------------------------------------------------------------
# clean_clip: start moves back, end moves forward, over-max trims inward, fallback flags
# ---------------------------------------------------------------------------
COMMA_SPLIT = [("This is the warmup line.", 0.8),
               ("He played this record called Stole Your Flow,", 0.9),       # comma + pause split
               ("you're like, not taking so kindly to it at all.", 0.8),
               ("Then the whole table started laughing at him.", 0.8),
               ("Nobody saw that coming from him.", 0.8),
               ("That was the end of it.", 0.8)]


def test_clean_start_extends_backward_across_comma_pause_split():
    words, sents = _mk(COMMA_SPLIT)
    assert sents[2]["text"].startswith("you're like,")
    old = B.clip_from_sentences(words, sents, 2, 4, cfg=_cfg(5, 30))
    assert _first(words, old) == "you're"                               # the bug: mid-sentence start
    c = B.clean_clip(words, sents, 2, 4, cfg=_cfg(5, 30))
    assert (c["s0"], c["s1"]) == (1, 4)
    assert _first(words, c) == "He" and c["start"] < sents[1]["start"]
    assert c["clean_start"] is True and c["clean_end"] is True and c["clean_note"] is None
    assert c["end"] == old["end"]


def test_clean_end_extends_forward_past_comma_ending():
    words, sents = _mk([("Opening words here.", 0.8), ("So what did you do next?", 0.8),
                        ("I told him to fold it", 0.8), ("and he said yep,", 0.9),
                        ("that is exactly right.", 0.8), ("Another story starts here.", 0.8)])
    assert sents[3]["text"].endswith("yep,")
    c = B.clean_clip(words, sents, 1, 3, cfg=_cfg(3, 30))
    assert (c["s0"], c["s1"]) == (1, 4) and words[c["w1"]]["w"] == "right."
    assert c["end"] == B.end_cut(words, sents[4]["w1"], tail_pad=0.35)
    assert c["clean_start"] and c["clean_end"]


def test_clean_end_by_hard_pause_without_punctuation():
    words, sents = _mk([("First sentence here.", 0.8), ("we played the hand", 1.6),
                        ("next thing we know", 0.8), ("it is over.", 0.8)])
    c = B.clean_clip(words, sents, 1, 1, cfg=_cfg(1, 30))
    assert (c["s0"], c["s1"], c["clean_start"], c["clean_end"]) == (1, 1, True, True)
    c = B.clean_clip(words, sents, 2, 2, cfg=_cfg(1, 30))            # 1.6 s pause before it: clean start
    assert (c["s0"], c["s1"], c["clean_start"], c["clean_end"]) == (2, 3, True, True)


def test_over_max_extension_trims_inward_to_clean_start():
    long_comma = " ".join(["word"] * 19) + " done,"                   # 20 words, ~8 s, comma ending
    words, sents = _mk([("Intro line here.", 0.8), (long_comma, 0.9),
                        ("and then we went there.", 0.8),              # picked start: follows a comma
                        ("Then the river came in hot.", 0.8),
                        ("Nobody believed what happened.", 0.8),
                        ("It was a total disaster for him.", 0.8)])
    cfg = _cfg(5, 13)
    raw = B.clip_from_sentences(words, sents, 2, 5, cfg=cfg)
    assert raw is not None and raw["end"] - raw["start"] <= 13           # the pick fits as given
    assert sents[5]["end"] - sents[1]["start"] > 13                       # extending back would not
    c = B.clean_clip(words, sents, 2, 5, cfg=cfg)
    assert (c["s0"], c["s1"]) == (3, 5) and _first(words, c) == "Then"
    assert c["clean_start"] and c["clean_end"] and 5 <= c["end"] - c["start"] <= 13


def test_over_max_extension_trims_end_inward():
    tail = " ".join(["more"] * 19) + " talk"                            # ~8 s, no end punctuation
    words, sents = _mk([("Intro line here.", 0.8), ("We sat down at the table.", 0.8),
                        ("He shoved all in.", 0.8), ("I called him right away", 0.9),
                        (tail, 0.8), ("Finally done.", 0.8)])
    cfg = _cfg(3, 12)
    c = B.clean_clip(words, sents, 1, 3, cfg=cfg)
    assert (c["s0"], c["s1"]) == (1, 2) and words[c["w1"]]["w"] == "in."
    assert c["clean_start"] and c["clean_end"]


def test_no_clean_option_falls_back_to_old_behaviour_with_flags_false():
    seg = " ".join(["talk"] * 12)                                       # ~4.7 s, never punctuated
    words, sents = _mk([(seg, 0.9)] * 8)
    cfg = _cfg(5, 12)
    c = B.clean_clip(words, sents, 3, 4, cfg=cfg)
    old = B.clip_from_sentences(words, sents, 3, 4, cfg=cfg)
    assert (c["start"], c["end"], c["s0"], c["s1"]) == (old["start"], old["end"], old["s0"], old["s1"])
    assert c["clean_start"] is False and c["clean_end"] is False
    assert "no clean start" in c["clean_note"] and "no clean end" in c["clean_note"]


def test_clean_start_preferred_over_clean_end_when_only_one_fits():
    seg = " ".join(["talk"] * 12)
    words, sents = _mk([("Opening line.", 0.8)] + [(seg, 0.9)] * 6)
    c = B.clean_clip(words, sents, 1, 2, cfg=_cfg(5, 12))              # S1 follows a "." : clean start
    assert c["clean_start"] is True and c["clean_end"] is False and c["s0"] == 1
    assert c["clean_note"] == "no clean end fits 5-12s"


def test_pick_longer_than_max_still_rejected_unless_trim_long():
    words, sents = _mk([("Big red dogs ran very fast.", 0.8)] * 8)       # every sentence clean
    assert B.clean_clip(words, sents, 0, 5, cfg=_cfg(5, 10)) is None     # same as clip_from_sentences
    c = B.clean_clip(words, sents, 0, 5, cfg=_cfg(5, 10), trim_long=True)
    assert c and c["s0"] == 0 and c["end"] - c["start"] <= 10


def test_clean_clip_out_of_range_and_all_clean_matches_old():
    words, sents = _mk([("Big red dogs ran very fast.", 0.8)] * 8)
    assert B.clean_clip(words, sents, 3, 2, cfg=_cfg(1, 100)) is None
    for s0, s1 in ((0, 0), (1, 3), (2, 2)):
        old = B.clip_from_sentences(words, sents, s0, s1, cfg=_cfg(8, 30))
        new = B.clean_clip(words, sents, s0, s1, cfg=_cfg(8, 30))
        assert {k: new[k] for k in old} == old and new["clean_start"] and new["clean_end"]


# ---------------------------------------------------------------------------
# fuzzy locate + _refine on a differently worded / segmented refined transcript
# ---------------------------------------------------------------------------
def _s(text, a, b):
    return {"text": text, "start": a, "end": b}


def test_locate_tolerates_gonna_vs_going_to():
    base = _s("I'm gonna shove all in right here.", 100.0, 103.0)
    rs = [_s("What a spot.", 97.0, 98.0), _s("I'm going to shove all-in right here!", 100.2, 103.3),
          _s("He folds.", 104.0, 104.6)]
    assert engine._locate(rs, base, "start") == 1 and engine._locate(rs, base, "end") == 1


def test_locate_merged_and_split_sentences():
    a, b = _s("He bets.", 100.0, 101.2), _s("She calls.", 101.9, 103.0)
    merged = [_s("Okay.", 98.0, 98.5), _s("He bets, she calls.", 100.1, 103.1), _s("Wow.", 104.0, 104.4)]
    assert engine._locate(merged, a, "start") == 1 and engine._locate(merged, b, "end") == 1
    base = _s("He bets and she calls it fast.", 100.0, 104.0)
    split = [_s("He bets,", 100.1, 101.4), _s("and she calls it fast.", 102.2, 104.1)]
    assert engine._locate(split, base, "start") == 0 and engine._locate(split, base, "end") == 1


def test_locate_fails_cleanly_without_time_overlap():
    base = _s("He shoved all in with nothing.", 100.0, 103.0)
    far = [_s("He shoved all in with nothing.", 200.0, 203.0), _s("Different hand.", 90.0, 99.0)]
    assert engine._locate(far, base, "start") is None and engine._locate(far, base, "end") is None


def test_refine_succeeds_when_small_en_wording_and_segmentation_differ(monkeypatch):
    segs = [("Earlier table talk happened.", 0.8), ("So what are you gonna do here?", 0.8),
            ("I'm gonna shove it all in.", 0.8), ("He thinks for a long time,", 0.9),
            ("then he calls with nothing.", 0.8), ("The whole room went silent.", 0.8),
            ("Later chatter here.", 0.8)]
    words, sents = _mk(segs, t=100.0)
    # small.en: "going to" instead of "gonna" (+0.1 s drift) and it hears no comma break -> one sentence
    rsegs = [("Earlier table talk happened.", 0.8), ("So what are you going to do here?", 0.8),
             ("I'm going to shove it all in.", 0.8), ("He thinks for a long time", 0.2),
             ("then he calls with nothing.", 0.8), ("The whole room went silent.", 0.8),
             ("Later chatter here.", 0.8)]
    rwords, _ = _mk(rsegs, t=100.1)
    monkeypatch.setattr(transcript, "words_window",
                        lambda media, lo, hi, model="small.en": [w for w in rwords if lo <= w["start"] <= hi])
    monkeypatch.setattr(transcript, "load_audio", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(engine.scenes, "scene_cuts", lambda *a, **k: [])
    monkeypatch.setattr(engine, "_log", lambda *a, **k: None)
    vc = _cfg(5, 30)
    cand = B.clean_clip(words, sents, 1, 5, cfg=vc)
    assert (cand["s0"], cand["s1"]) == (1, 5)
    assert engine._locate(transcript.sentences(rwords), sents[5], "end") is not None
    c, rw, rs = engine._refine("m.mp4", 1000.0, cand, sents, vc)
    assert rw[c["w0"]]["w"] == "So" or rw[c["w0"]]["w"] == "what"       # filler "So" may be trimmed
    assert rw[c["w1"]]["w"] == "silent." and c["clean_start"] and c["clean_end"]
    assert c["start"] == pytest.approx(cand["start"] + 0.1, abs=0.2)


def test_refine_out_of_band_trims_to_clean_boundary(monkeypatch):
    # base: 3 short sentences (~9 s); refined merges an un-punctuated run onto the end -> too long
    words, sents = _mk([("Intro.", 0.8), ("He shoves.", 0.8), ("She calls fast.", 0.8),
                        ("They flip the cards over now.", 0.8), ("Outro words.", 0.8)], t=100.0)
    tail = "They flip the cards over now and then " + " ".join(["we"] * 20) + " talk"   # run-on
    rwords, _ = _mk([("Intro.", 0.8), ("He shoves.", 0.8), ("She calls fast.", 0.3),
                     (tail, 0.8), ("Outro words.", 0.8)], t=100.0)
    monkeypatch.setattr(transcript, "words_window", lambda media, lo, hi, model="small.en": rwords)
    monkeypatch.setattr(transcript, "load_audio", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(engine.scenes, "scene_cuts", lambda *a, **k: [])
    monkeypatch.setattr(engine, "_log", lambda *a, **k: None)
    vc = _cfg(2, 8)
    cand = B.clean_clip(words, sents, 1, 3, cfg=vc)
    c, rw, rs = engine._refine("m.mp4", 1000.0, cand, sents, vc)
    assert c["end"] - c["start"] <= 8 and c["clean_end"] and rw[c["w1"]]["w"].endswith(".")


# ---------------------------------------------------------------------------
# claude_cli on account B: child env from config, secrets never surface
# ---------------------------------------------------------------------------
SECRET = "fake-oauth-value-THIS-IS-NOT-A-TOKEN-0123456789"


@pytest.fixture()
def envfile(tmp_path):
    f = tmp_path / "account-b.env"
    f.write_text(f"# account B\nexport CLAUDE_CODE_OAUTH_TOKEN=\"{SECRET}\"\n\nOTHER_VAR='two words'\n")
    return f


def _cli_cfg(envfile, tmp_path):
    return {"semantic_backend": "claude_cli", "semantic_model": "opus", "chunk_s": 900,
            "chunk_overlap_s": 120, "top_n": 6, "min_dur_s": 15, "max_dur_s": 75,
            "claude_config_dir": str(tmp_path / ".claude-b"), "claude_env_file": str(envfile)}


def _sents(n, step=4.0):
    return [{"i": k, "start": k * step, "end": k * step + step - 0.3, "text": f"Sentence number {k}.",
             "w0": 3 * k, "w1": 3 * k + 2} for k in range(n)]


def _ok_stdout():
    inner = json.dumps({"clips": [{"s0": 2, "s1": 6, "hook": "h", "score": 80, "sub": {}}]})
    return json.dumps({"type": "result", "result": inner})


def test_claude_cli_env_sets_account_b_and_strips_session_vars(envfile, tmp_path, monkeypatch):
    for k in select_semantic.STRIP_ENV:
        monkeypatch.setenv(k, "inherited")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    seen = []

    def fake(argv, **kw):
        seen.append(kw)
        return subprocess.CompletedProcess(argv, 0, stdout=_ok_stdout(), stderr="")
    monkeypatch.setattr(select_semantic.subprocess, "run", fake)
    picks = select_semantic.propose(_sents(20), "podcast", _cli_cfg(envfile, tmp_path))
    assert [(p["s0"], p["s1"], p["backend"], p["model"]) for p in picks] == [(2, 6, "claude_cli", "opus")]
    env = seen[0]["env"]
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / ".claude-b")
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == SECRET and env["OTHER_VAR"] == "two words"
    assert not set(select_semantic.STRIP_ENV) & set(env)
    assert env["PATH"] == "/usr/bin:/bin"
    import os
    assert os.environ["CLAUDECODE"] == "inherited"                      # our own env is untouched


def test_claude_cli_token_never_in_exceptions_or_logs(envfile, tmp_path, monkeypatch, capsys):
    def leaky(argv, **kw):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr=f"auth failed for token {SECRET}")
    monkeypatch.setattr(select_semantic.subprocess, "run", leaky)
    cfg = _cli_cfg(envfile, tmp_path)
    with pytest.raises(RuntimeError) as ei:
        select_semantic._claude_cli("p", "opus", 5, cfg)
    assert SECRET not in str(ei.value) and "***" in str(ei.value) and "exited 1" in str(ei.value)
    assert ei.value.__cause__ is None and ei.value.__context__ is None or SECRET not in repr(ei.value.__context__)
    assert select_semantic.propose(_sents(20), "podcast", cfg) is None
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err and "claude_cli" in out.err

    def slow(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"], output=SECRET)
    monkeypatch.setattr(select_semantic.subprocess, "run", slow)
    with pytest.raises(RuntimeError) as ei:
        select_semantic._claude_cli("p", "opus", 5, cfg)
    assert SECRET not in str(ei.value) and "timed out" in str(ei.value)


def test_claude_env_file_errors_name_no_values(tmp_path, monkeypatch):
    bad = tmp_path / "bad.env"
    bad.write_text(f"{SECRET}\n")
    with pytest.raises(RuntimeError) as ei:
        select_semantic._read_env_file(bad)
    assert SECRET not in str(ei.value) and "line 1" in str(ei.value)
    with pytest.raises(RuntimeError) as ei:
        select_semantic._read_env_file(tmp_path / "missing.env")
    assert "unreadable" in str(ei.value)
    monkeypatch.setattr(select_semantic.subprocess, "run",
                        lambda *a, **k: pytest.fail("claude must not run without its account env"))
    with pytest.raises(RuntimeError):
        select_semantic._claude_cli("p", "opus", 5, {"claude_env_file": str(tmp_path / "missing.env")})


def test_claude_cli_without_account_keys_inherits_env(monkeypatch):
    seen = []
    monkeypatch.setattr(select_semantic.subprocess, "run",
                        lambda argv, **kw: seen.append(kw) or subprocess.CompletedProcess(argv, 0, _ok_stdout(), ""))
    select_semantic._claude_cli("p", "opus", 5, {})
    assert "env" not in seen[0]


# ---------------------------------------------------------------------------
# per-chunk fallback
# ---------------------------------------------------------------------------
FB_CFG = {"semantic_backend": "claude_cli", "semantic_model": "opus", "semantic_fallback_backend": "ollama",
          "semantic_fallback_model": "qwen2.5:7b", "semantic_timeout_s": 42, "chunk_s": 900,
          "chunk_overlap_s": 120, "top_n": 6, "min_dur_s": 15, "max_dur_s": 75}


class _Rec:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def __call__(self, prompt, model, backend="ollama", timeout=None, opts=None):
        chunk = 1 if "[S0 0s]" in prompt else 2
        self.calls.append((chunk, backend, model, timeout))
        a = self.answer(chunk, backend)
        if isinstance(a, BaseException):
            raise a
        return a


def _pick(s0, s1):
    return json.dumps({"clips": [{"s0": s0, "s1": s1, "hook": "h", "score": 70, "sub": {}}]})


def test_primary_failure_retries_that_chunk_once_on_fallback(monkeypatch):
    sents = _sents(300)                                                  # 1200 s -> 2 chunks
    assert len(select_semantic._chunks(sents)) == 2

    def answer(chunk, backend):
        if backend == "claude_cli":
            return TimeoutError("slow") if chunk == 1 else _pick(250, 256)
        return _pick(10, 16)
    rec = _Rec(answer)
    monkeypatch.setattr(select_semantic, "_llm", rec)
    picks = select_semantic.propose(sents, "podcast", FB_CFG)
    assert rec.calls == [(1, "claude_cli", "opus", 42), (1, "ollama", "qwen2.5:7b", 42),
                         (2, "claude_cli", "opus", 42)]
    got = {(p["s0"], p["backend"], p["model"]) for p in picks}
    assert got == {(10, "ollama", "qwen2.5:7b"), (250, "claude_cli", "opus")}


def test_garbage_primary_answer_also_falls_back_and_both_failing_returns_none(monkeypatch):
    sents = _sents(300)
    rec = _Rec(lambda chunk, backend: "not json at all" if backend == "claude_cli" else _pick(10, 16))
    monkeypatch.setattr(select_semantic, "_llm", rec)
    assert [p["backend"] for p in select_semantic.propose(sents, "podcast", FB_CFG)] == ["ollama"]
    assert [c[:2] for c in rec.calls] == [(1, "claude_cli"), (1, "ollama"), (2, "claude_cli"), (2, "ollama")]
    rec = _Rec(lambda chunk, backend: ConnectionError("down"))
    monkeypatch.setattr(select_semantic, "_llm", rec)
    assert select_semantic.propose(sents, "podcast", FB_CFG) is None
    assert len(rec.calls) == 4                                           # one fallback try per chunk


def test_no_fallback_configured_tries_once(monkeypatch):
    rec = _Rec(lambda chunk, backend: ConnectionError("down"))
    monkeypatch.setattr(select_semantic, "_llm", rec)
    cfg = {k: v for k, v in FB_CFG.items() if not k.startswith("semantic_fallback")}
    assert select_semantic.propose(_sents(20), "podcast", cfg) is None
    assert len(rec.calls) == 1
