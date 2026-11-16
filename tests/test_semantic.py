"""Semantic selector tests — no network, no whisper, no Ollama, no claude CLI (all mocked)."""
import json
import subprocess
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

PIPE = Path(__file__).resolve().parent.parent / "clipengine"

from clipengine import select_semantic

CFG = {"semantic_backend": "ollama", "semantic_model": "qwen2.5:7b", "chunk_s": 900,
       "chunk_overlap_s": 120, "top_n": 6, "min_dur_s": 15, "max_dur_s": 75, "overlap_max": 0.3}


def _sents(n, step=4.0, first=0):
    """n sentences, one every `step` seconds, numbered from `first`."""
    return [{"i": k, "start": k * step, "end": k * step + step - 0.3, "text": f"Sentence number {k}.",
             "w0": 3 * k, "w1": 3 * k + 2} for k in range(first, first + n)]


def _resp(*clips):
    return json.dumps({"clips": list(clips)})


def _clip(s0, s1, score=80, **kw):
    return {"s0": s0, "s1": s1, "hook": "A hook", "score": score, "reason": "why",
            "sub": {"hook": 70, "standalone": 60, "payoff": 80, "emotion": 50}, **kw}


class _Recorder:
    """Mock for _llm: records prompts/args, answers from a list (last one repeats)."""
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, prompt, model, backend="ollama", timeout=None, opts=None):
        self.calls.append({"prompt": prompt, "model": model, "backend": backend})
        a = self.answers[min(len(self.calls), len(self.answers)) - 1]
        if isinstance(a, BaseException):
            raise a
        return a


def _propose(sents, answers, vertical="podcast", cfg=CFG):
    rec = _Recorder(*answers)
    with patch.object(select_semantic, "_llm", rec):
        return select_semantic.propose(sents, vertical, cfg), rec


# ---- prompt ----
def test_prompt_lines_and_energy_flag():
    sents = _sents(10)
    sents[4]["z"], sents[5]["z"] = 1.5, 1.4
    _, rec = _propose(sents, [_resp()])
    p = rec.calls[0]["prompt"]
    assert "[S3 12s] Sentence number 3." in p.splitlines()
    assert "[S4 16s ⚡] Sentence number 4." in p.splitlines()
    assert "[S5 20s] Sentence number 5." in p.splitlines()          # 1.4 < 1.5: no flag
    assert "hard limits 15-75s; never pad with setup or trailing chatter" in p
    assert "as long as the moment needs, typically 20-60s" in p
    assert "<=8 word curiosity-gap" in p and "paraphrase" in p
    assert '"s0"' in p and '"s1"' in p and "S-number" in p
    assert rec.calls[0]["model"] == "qwen2.5:7b" and rec.calls[0]["backend"] == "ollama"


def test_rubric_per_vertical():
    R = select_semantic.RUBRICS
    assert R["general"] == R["podcast"]
    assert R["poker"] != R["podcast"] and R["stream"] != R["podcast"] and R["poker"] != R["stream"]
    assert "reveal" in R["poker"] and "reaction" in R["poker"].lower()
    assert "setup" in R["stream"] and "reaction" in R["stream"].lower()
    prompts = {}
    for v in ("podcast", "poker", "stream", "general", "unknown"):
        _, rec = _propose(_sents(5), [_resp()], vertical=v)
        prompts[v] = rec.calls[0]["prompt"]
        assert R.get(v, R["general"]) in prompts[v]
    assert prompts["general"] == prompts["podcast"] == prompts["unknown"]
    assert prompts["poker"] != prompts["podcast"] != prompts["stream"]


# ---- parsing / validation ----
def test_picks_parsed_validated_and_clamped():
    ans = _resp(
        {"s0": 2, "s1": 6, "hook": "H" * 100, "score": 150, "reason": "r" * 300,
         "sub": {"hook": -5, "standalone": "80", "payoff": "x", "emotion": 40}},
        {"s0": "10", "s1": 14.0, "hook": "No sub given", "score": "n/a"},  # missing sub, bad score
        {"s0": 8, "s1": 5, "score": 99},                                   # reversed
        {"s0": 3, "s1": 99, "score": 99},                                  # out of range
        {"s1": 4, "score": 99},                                            # missing s0
        {"s0": "abc", "s1": 4, "score": 99},                               # not int-coercible
        {"s0": 1.5, "s1": 4, "score": 99},                                 # not an integer
        "not a dict")
    picks, _ = _propose(_sents(20), [ans])
    assert [(p["s0"], p["s1"]) for p in picks] == [(2, 6), (10, 14)]  # score desc
    a, b = picks
    assert a["score"] == 100.0 and a["sub"] == {"hook": 0.0, "standalone": 80.0, "emotion": 40.0}
    assert len(a["hook"]) == 80 and len(a["reason"]) == 200
    assert b["score"] == 50.0 and b["sub"] == {} and b["hook"] == "No sub given"
    for p in picks:
        assert set(p) >= {"s0", "s1", "hook", "score", "sub", "reason"}


def test_indices_outside_filtered_subset_dropped():
    sents = _sents(20, first=5)                  # caller filtered S0-S4 (skip_head)
    picks, _ = _propose(sents, [_resp(_clip(2, 7), _clip(5, 9, 60), _clip(20, 26, 90))])
    assert [(p["s0"], p["s1"]) for p in picks] == [(5, 9)]
    assert picks[0]["start"] == sents[0]["start"] and picks[0]["end"] == sents[4]["end"]


def test_json_wrapped_in_prose_and_fences_parsed():
    ans = "Sure! Here are the clips {as asked}:\n```json\n" + _resp(_clip(1, 4)) + "\n```\nEnjoy."
    picks, _ = _propose(_sents(10), [ans])
    assert [(p["s0"], p["s1"]) for p in picks] == [(1, 4)]


# ---- chunking / dedupe ----
def test_cross_chunk_dedupe_via_suppress():
    sents = _sents(400)                          # 0..1596s > chunk_s -> 2 overlapping chunks
    picks, rec = _propose(sents, [_resp(_clip(200, 210, 80)),
                                  _resp(_clip(201, 211, 90), _clip(10, 14, 95))])
    assert len(rec.calls) == 2
    p1, p2 = rec.calls[0]["prompt"], rec.calls[1]["prompt"]
    assert "[S0 0s]" in p1 and "[S224 896s]" in p1 and "[S225 " not in p1
    assert "[S195 780s]" in p2 and "[S399 1596s]" in p2 and "[S194 " not in p2
    assert "Pick up to 5 " in p1                 # k = max(3, ceil(1.5*6/2))
    # S10 is not in chunk 2 -> dropped; the two overlapping picks collapse to the higher score
    assert [(p["s0"], p["s1"], p["score"]) for p in picks] == [(201, 211, 90.0)]


def test_chunk_k_floor_and_single_chunk():
    _, rec = _propose(_sents(50), [_resp()], cfg={**CFG, "top_n": 1})
    assert len(rec.calls) == 1 and "Pick up to 3 " in rec.calls[0]["prompt"]


def test_non_overlapping_picks_across_chunks_both_kept_sorted():
    picks, _ = _propose(_sents(400), [_resp(_clip(20, 26, 70)), _resp(_clip(300, 306, 85))])
    assert [(p["s0"], p["score"]) for p in picks] == [(300, 85.0), (20, 70.0)]


# ---- failure -> None ----
def test_empty_sentences_returns_none_without_llm_call():
    picks, rec = _propose([], [_resp(_clip(0, 1))])
    assert picks is None and rec.calls == []


def test_llm_raising_returns_none():
    for exc in (urllib.error.URLError("down"), TimeoutError(), RuntimeError("boom"), ValueError()):
        picks, rec = _propose(_sents(400), [exc])
        assert picks is None and len(rec.calls) == 2


def test_one_chunk_failing_still_uses_the_other():
    picks, _ = _propose(_sents(400), [urllib.error.URLError("blip"), _resp(_clip(300, 306))])
    assert [(p["s0"], p["s1"]) for p in picks] == [(300, 306)]


def test_garbage_returns_none():
    for ans in ("not json at all", "", "{broken", '{"clips": "nope"}', '{"other": []}', None,
                _resp(_clip(500, 510))):                             # only out-of-chunk picks
        picks, _ = _propose(_sents(10), [ans])
        assert picks is None, ans


# ---- an explicit empty answer is a verdict, not an outage ----
def test_explicit_empty_answer_returns_empty_list_not_none():
    # one window: claude_cli/opus answered {"clips":[]} in 6 s; propose
    # returned None and engine logged "LLM unavailable" before overriding it with energy peaks.
    picks, rec = _propose(_sents(10), ['{"clips": []}'])
    assert picks == [] and len(rec.calls) == 1


def test_every_chunk_declining_returns_empty_list():
    picks, rec = _propose(_sents(400), [_resp(), _resp()])
    assert picks == [] and len(rec.calls) == 2


def test_declined_chunk_plus_failed_chunk_is_still_an_outage():
    picks, _ = _propose(_sents(400), [_resp(), urllib.error.URLError("down")])
    assert picks is None


# ---- ollama backend ----
class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _ollama_body(model):
    seen = []

    def fake(req, timeout=None):
        seen.append(json.loads(req.data))
        return _Resp({"response": _resp(_clip(1, 4))})
    with patch("urllib.request.urlopen", fake):
        out = select_semantic._llm("PROMPT", model)
    assert json.loads(out)["clips"][0]["s0"] == 1
    return seen[0]


def test_ollama_qwen3_sends_think_false():
    body = _ollama_body("qwen3:30b-a3b")
    assert body["think"] is False
    assert body["model"] == "qwen3:30b-a3b" and body["prompt"] == "PROMPT"
    assert body["format"] == "json" and body["stream"] is False
    assert body["options"] == {"temperature": 0, "num_ctx": 32768}


def test_ollama_qwen25_has_no_think_key():
    assert "think" not in _ollama_body("qwen2.5:7b")


def test_propose_via_ollama_http_mocked():
    with patch("urllib.request.urlopen", lambda req, timeout=None: _Resp({"response": _resp(_clip(1, 4))})):
        picks = select_semantic.propose(_sents(10), "poker", CFG)
    assert [(p["s0"], p["s1"]) for p in picks] == [(1, 4)]


def test_ollama_unreachable_returns_none():
    def down(req, timeout=None):
        raise urllib.error.URLError("connection refused")
    with patch("urllib.request.urlopen", down):
        assert select_semantic.propose(_sents(10), "podcast", CFG) is None


# ---- claude_cli backend (criterion 12) ----
CLAUDE_CFG = {**CFG, "semantic_backend": "claude_cli", "semantic_model": "sonnet"}


def _cli_run(returncode=0, stdout=None):
    calls = []

    def fake(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="err")
    return fake, calls


def test_claude_cli_backend_parses_fenced_result():
    inner = "```json\n" + _resp(_clip(2, 6, 88), _clip(9, 3)) + "\n```"
    fake, calls = _cli_run(0, json.dumps({"type": "result", "result": inner}))
    with patch.object(select_semantic.subprocess, "run", fake):
        picks = select_semantic.propose(_sents(20), "stream", CLAUDE_CFG)
    assert [(p["s0"], p["s1"], p["score"]) for p in picks] == [(2, 6, 88.0)]
    argv, kw = calls[0]
    assert argv == ["claude", "-p", "--model", "sonnet", "--output-format", "json"]
    assert "[S2 8s] Sentence number 2." in kw["input"]
    assert select_semantic.RUBRICS["stream"] in kw["input"]
    assert kw["capture_output"] is True and kw["text"] is True and kw.get("timeout")
    assert "env" not in kw                       # inherits the process environment


def test_claude_cli_nonzero_exit_returns_none():
    fake, calls = _cli_run(1, "")
    with patch.object(select_semantic.subprocess, "run", fake):
        assert select_semantic.propose(_sents(20), "podcast", CLAUDE_CFG) is None
    assert calls


def test_claude_cli_error_envelope_and_bad_stdout_return_none():
    for out in (json.dumps({"is_error": True, "result": _resp(_clip(1, 4))}), "not json", "{}"):
        fake, _ = _cli_run(0, out)
        with patch.object(select_semantic.subprocess, "run", fake):
            assert select_semantic.propose(_sents(20), "podcast", CLAUDE_CFG) is None


def test_claude_cli_timeout_returns_none():
    def slow(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
    with patch.object(select_semantic.subprocess, "run", slow):
        assert select_semantic.propose(_sents(20), "podcast", CLAUDE_CFG) is None


def test_unknown_backend_returns_none():
    with patch("urllib.request.urlopen") as net, patch.object(select_semantic.subprocess, "run") as run:
        assert select_semantic.propose(_sents(10), "podcast", {**CFG, "semantic_backend": "x"}) is None
    net.assert_not_called()
    run.assert_not_called()
