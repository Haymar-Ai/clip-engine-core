#!/usr/bin/env python3
"""Meaning-based moment selection for every vertical. The LLM reads numbered sentences and picks
clips as sentence-index ranges (s0..s1), so every pick is sentence-aligned by construction; the
exact cut points are placed later by boundaries.py. Returns None when the backend is unreachable
or answers garbage, so the caller can fall back to the energy path.

Backends (cfg semantic_backend): "ollama" (local HTTP, $0) or "claude_cli" (`claude -p`). A chunk
that fails on the primary backend is retried once on semantic_fallback_backend/_model.

claude_cli runs on whichever Claude account its environment selects. With cfg claude_config_dir and/or
claude_env_file the child gets CLAUDE_CONFIG_DIR plus the env file's KEY=VALUE pairs, and the
inherited Claude Code session handles (STRIP_ENV) are removed: an inherited fd/session auth source
outranks the config-dir login. Values from the env file are never logged or put in an exception."""
import json, math, os, re, subprocess, sys, time, urllib.request

from clipengine import boundaries

OLLAMA_URL = "http://localhost:11434/api/generate"
TIMEOUT_S = 300
LAST_CALLS = []          # the last propose()'s backend calls: {chunk, backend, model, s, outcome}
SUB_KEYS = ("hook", "standalone", "payoff", "emotion")
EXCITED_Z = 1.5          # sentence mean energy z-score that earns the ⚡ cue
EXCITED_C = 1.5          # sentence mean chat spike score that earns the 💬 cue

_PODCAST = ("This is a podcast / talk transcript. A great clip is a bold claim, hot take or story "
            "with a payoff: it opens on a HOOK (an intriguing or bold line, not a greeting or "
            "throat-clearing setup), is SELF-CONTAINED (makes sense with zero prior context), and "
            "lands the PAYOFF (a punchline, twist, surprising claim or emotional peak). Reject "
            "intros, sponsor reads, logistics and rambling with no payoff.")
RUBRICS = {
    "podcast": _PODCAST,
    "poker": ("This is a poker broadcast transcript (commentary and table talk). A great clip is ONE "
              "HAND: start at the moment the stakes are clear (the big bet, the shove, the tough "
              "decision), run through the reveal (cards turned over, the pot shipped) and end once "
              "the commentator/table reaction has landed. Do not start on routine preflop chatter; "
              "do not cut before the reaction. Reject breaks, sponsor reads, chip-count recaps and "
              "dead table talk."),
    "stream": ("This is a livestream / gaming transcript. A great clip is a REACTION moment: include "
               "the setup that makes the reaction land (what just happened, what they were trying) "
               "plus the reaction itself (shock, rage, hype, laughter), and end once it has landed. "
               "Reject idle chatter, reading chat, donation/sub alerts and technical downtime. "
               "A 💬 after the time marks a sentence where Twitch chat blew up in the seconds "
               "just after it (spam, KEKW/LUL, CLIP IT) - a strong hint the reaction landed. "
               "Judge by meaning; 💬 is a signal, not a guarantee."),
    "general": _PODCAST,
}

PROMPT = """You are an expert short-form video editor picking viral clips from a video transcript.
Each line is one sentence: "[S<number> <seconds>s] text". A ⚡ after the time marks an unusually loud,
energetic sentence (shouting, crowd, hype) - a hint of a peak, but judge by meaning.

{rubric}

Pick up to {k} of the BEST standalone moments. Each clip is a contiguous run of sentences: "s0" is the
S-number of its FIRST sentence and "s1" is the S-number of its LAST sentence (s0 <= s1).
Length: as long as the moment needs, typically 20-60s, hard limits {mind}-{maxd}s; never pad with setup or trailing chatter.
Start on the sentence that hooks; end on the sentence that lands the payoff.{end_rule}
For "hook": write a punchy <=8 word curiosity-gap TITLE that makes someone stop scrolling. Do NOT
copy a verbatim transcript line; paraphrase into a clean title (e.g. "Then Jesus came to the village").
"score" is overall virality 0-100; "sub" rates 0-100 each: hook (grabs attention in the first
seconds), standalone (makes sense with no context), payoff (lands a punchline/twist/reveal),
emotion (energy, tension, surprise, laughter).
Return ONLY JSON: {{"clips":[{{"s0":<int>,"s1":<int>,"hook":"<title>","score":<0-100>,"sub":{{"hook":<0-100>,"standalone":<0-100>,"payoff":<0-100>,"emotion":<0-100>}},"reason":"<one line>"}}]}}

{topics}TRANSCRIPT:
{body}"""

# the hard end rule (boundaries.wrapup_sentence enforces it on the picks as well)
END_RULE = """
END RULE (hard): the last sentence (s1) lands the point. Never end on the wrap-up after it: a summary
or restatement, a sign-off, thanks / subscribe / link lines, "anyway", "so yeah", "moving on", a tag
line like "you know what I'm saying", or a line that turns to the next topic. End on the sentence
before it."""


# ---- backends ----
def _ollama(prompt, model, timeout):
    body = {"model": model, "prompt": prompt, "stream": False, "format": "json",
            "options": {"temperature": 0, "num_ctx": 32768}}
    if str(model).startswith("qwen3"):
        body["think"] = False                    # qwen3 otherwise spends the budget "thinking"
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"]


STRIP_ENV = ("CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_CHILD_SESSION",
             "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN",
             "CLAUDECODE", "CLAUDE_PID", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _read_env_file(path):
    """KEY=VALUE lines (optional `export `, optional matching quotes, # comments). Errors name the
    file and line number only, never a value."""
    path = os.path.expanduser(str(path))
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as e:
        raise RuntimeError(f"claude_env_file {path} unreadable ({type(e).__name__})") from None
    out = {}
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, eq, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not eq or not _ENV_KEY.match(key):
            raise RuntimeError(f"claude_env_file {path}: line {n} is not KEY=VALUE")
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        out[key] = val
    return out


def _claude_env(opts):
    """(child env for `claude -p`, secret values to mask). env is None (inherit ours) when opts
    names no account; the secrets are every value read from claude_env_file."""
    opts = opts or {}
    cdir, efile = opts.get("claude_config_dir"), opts.get("claude_env_file")
    if not (cdir or efile):
        return None, ()
    env = {k: v for k, v in os.environ.items() if k not in STRIP_ENV}
    extra = _read_env_file(efile) if efile else {}
    env.update(extra)
    if cdir:
        env["CLAUDE_CONFIG_DIR"] = os.path.expanduser(str(cdir))
    return env, tuple(v for v in extra.values() if len(v) >= 4)


def _scrub(text, secrets):
    """`text` with every secret value masked (longest first, so a prefix never leaks the rest)."""
    text = str(text)
    for v in sorted(set(secrets), key=len, reverse=True):
        text = text.replace(v, "***")
    return text


def _claude_cli(prompt, model, timeout, opts=None):
    """Model text for a prompt from `claude -p` with no tools (`--tools ""`): the prompt carries
    third-party transcript text and the answer only needs the prompt (with tools, a transcript
    line could make the call read a local file and burn its contents into a hook)."""
    env, secrets = _claude_env(opts)
    kw = {} if env is None else {"env": env}
    try:
        p = subprocess.run(["claude", "-p", "--model", model, "--output-format", "json",
                            "--tools", ""],
                           input=prompt, capture_output=True, text=True, timeout=timeout, **kw)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"claude timed out after {timeout:g}s") from None
    except OSError as e:
        raise RuntimeError(_scrub(f"claude not runnable: {e}", secrets)) from None
    if p.returncode != 0:
        tail = " ".join(((p.stderr or "") + " " + (p.stdout or "")).split())[-200:]
        raise RuntimeError(_scrub(f"claude exited {p.returncode}: {tail}", secrets))
    try:
        out = json.loads(p.stdout)
    except ValueError:
        raise RuntimeError("claude printed non-JSON output") from None
    if not isinstance(out, dict) or out.get("is_error"):
        raise RuntimeError("claude returned an error result")
    return _strip_fences(str(out["result"]))


BACKENDS = {"ollama": _ollama, "claude_cli": _claude_cli}


def _llm(prompt, model, backend="ollama", timeout=TIMEOUT_S, opts=None):
    """Raw model text for `prompt` from the configured backend (raises on any failure). `opts` is
    the select cfg (claude_cli reads claude_config_dir / claude_env_file from it)."""
    fn = BACKENDS.get(backend)
    if fn is None:
        raise ValueError(f"unknown semantic_backend {backend!r}")
    if backend == "claude_cli":
        return fn(prompt, model, timeout, opts)
    return fn(prompt, model, timeout)


# ---- parsing ----
_FENCE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\n?(.*?)(?:```|\Z)", re.S)


def _strip_fences(text):
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()


def _parse(text):
    """First JSON object with a "clips" key in model output (tolerates fences / leading prose)."""
    if not isinstance(text, str):
        return None
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and "clips" in obj:
            return obj
    return None


def _idx(v):
    """Sentence number as int ("12", 12.0, "S12" ok; 12.5, None, "x" -> None)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, str):
        v = v.strip().lstrip("Ss")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else None


def _pct(v, default=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return min(100.0, max(0.0, f)) if math.isfinite(f) else default


def _validate(data, chunk):
    """Picks from one chunk's parsed answer; drops bad/reversed/out-of-chunk ranges, clamps scores.
    start/end (sentence times) are attached for cross-chunk suppression."""
    by_i = {s["i"]: s for s in chunk}
    clips = data.get("clips")
    out = []
    for c in clips if isinstance(clips, list) else []:
        if not isinstance(c, dict):
            continue
        s0, s1 = _idx(c.get("s0")), _idx(c.get("s1"))
        if s0 not in by_i or s1 not in by_i or s0 > s1:
            continue
        sub = c.get("sub") if isinstance(c.get("sub"), dict) else {}
        sub = {k: _pct(sub.get(k)) for k in SUB_KEYS}
        out.append({"s0": s0, "s1": s1, "hook": str(c.get("hook") or "").strip()[:80],
                    "score": _pct(c.get("score"), 50.0),
                    "sub": {k: v for k, v in sub.items() if v is not None},
                    "reason": str(c.get("reason") or "").strip()[:200],
                    "start": float(by_i[s0]["start"]), "end": float(by_i[s1]["end"])})
    return out


# ---- prompt ----
def _line(s):
    flag = " ⚡" if float(s.get("z") or 0.0) >= EXCITED_Z else ""
    if float(s.get("chat_z") or 0.0) >= EXCITED_C:
        flag += " 💬"
    return f"[S{s['i']} {float(s['start']):.0f}s{flag}] {' '.join(str(s['text']).split())}"


def build_prompt(sents, vertical, k=3, min_dur=15, max_dur=75, rubric=None, topics=None,
                 end_rule=False):
    """The pick prompt. `rubric` = the niche's prompt pack (else the vertical's),
    `topics` = topic_map topics (a TOPICS block: every clip inside one topic), `end_rule` = the
    hard "drop the trailing wrap-up" line. All three off: today's prompt, byte for byte."""
    from clipengine import topic_map
    return PROMPT.format(rubric=rubric or RUBRICS.get(vertical, RUBRICS["general"]), k=k,
                         mind=int(round(min_dur)), maxd=int(round(max_dur)),
                         end_rule=END_RULE if end_rule else "",
                         topics=topic_map.prompt_block(topics, sents) if topics else "",
                         body="\n".join(_line(s) for s in sents))


def _chunks(sents, chunk_s=900.0, overlap_s=120.0):
    """Time chunks of ~chunk_s seconds of sentences; consecutive chunk starts advance by
    chunk_s - overlap_s. Empty windows and chunks contained in the previous one are skipped."""
    chunk_s = max(1.0, float(chunk_s))
    step = chunk_s - overlap_s if 0 <= overlap_s < chunk_s else chunk_s
    t, last, out = float(sents[0]["start"]), float(sents[-1]["start"]), []
    while True:
        ch = [s for s in sents if t <= s["start"] < t + chunk_s]
        if ch and (not out or ch[-1] is not out[-1][-1]):
            out.append(ch)
        if t + chunk_s > last:
            return out
        t += step


def propose(sentences, vertical, cfg):
    """LLM picks [{s0,s1,hook,score,sub,reason,start,end}] (s0/s1 = sentence "i"), deduped across
    chunks, best score first. [] when the model answered every chunk with an explicit empty
    pick list ({"clips": []}): it read the transcript and judged nothing clip-worthy, which is
    a verdict, not an outage. None if there are no sentences, every backend call failed or
    returned garbage, or no valid pick survived. Never raises on backend errors."""
    LAST_CALLS.clear()
    if not sentences:
        return None
    cfg = cfg or {}
    chunks = _chunks(sentences, cfg.get("chunk_s", 900), cfg.get("chunk_overlap_s", 120))
    k = max(3, math.ceil(1.5 * int(cfg.get("top_n") or 6) / len(chunks)))
    tries = [(cfg.get("semantic_backend") or "ollama", cfg.get("semantic_model") or "qwen2.5:7b")]
    fb = (cfg.get("semantic_fallback_backend"), cfg.get("semantic_fallback_model"))
    if fb[0] and fb[1] and tuple(fb) != tries[0]:
        tries.append(tuple(fb))
    timeout = float(cfg.get("semantic_timeout_s") or TIMEOUT_S)
    cands, declined = [], 0
    for n, ch in enumerate(chunks):
        lo, hi = ch[0]["i"], ch[-1]["i"]
        tops = [t for t in cfg.get("topics") or [] if t["s1"] >= lo and t["s0"] <= hi]
        prompt = build_prompt(ch, vertical, k, cfg.get("min_dur_s", 15), cfg.get("max_dur_s", 75),
                              rubric=cfg.get("rubric"), topics=tops,
                              end_rule=cfg.get("end_rule_prompt"))
        for backend, model in tries:             # primary, then the fallback once for this chunk
            t0 = time.monotonic()
            call = {"chunk": n + 1, "backend": backend, "model": model}
            LAST_CALLS.append(call)
            try:
                data = _parse(_llm(prompt, model, backend=backend, timeout=timeout, opts=cfg))
            except Exception as e:               # backend down / timeout / bad envelope
                call.update(s=round(time.monotonic() - t0, 1), outcome="error")
                print(f"[select_semantic] chunk {n + 1}/{len(chunks)} {backend}:{model} failed: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                continue
            call["s"] = round(time.monotonic() - t0, 1)
            if data is None:
                call["outcome"] = "garbage"
                print(f"[select_semantic] chunk {n + 1}/{len(chunks)} {backend}:{model} answered "
                      "no parseable JSON", file=sys.stderr)
                continue
            call["outcome"] = "answered"
            cands += [{**c, "backend": backend, "model": model} for c in _validate(data, ch)]
            declined += data.get("clips") == []
            break
    if not cands:
        return [] if declined == len(chunks) else None
    kept = boundaries.suppress(cands, max_overlap=float(cfg.get("overlap_max", 0.3)), top_n=None)
    # a pick the overlap rule drops is the next candidate for the kept pick it
    # overlaps most (the render replaces a kept clip with no clean cut by it)
    for c in sorted(cands, key=lambda c: -float(c.get("score") or 0.0)):
        if any(c is k for k in kept):
            continue
        best = max(kept, key=lambda k: boundaries.iou(c, k))
        if boundaries.iou(c, best) > 0:
            best.setdefault("alternates", []).append(
                {k: c[k] for k in ("s0", "s1", "hook", "score", "sub", "start", "end", "backend",
                                   "model") if k in c})
    return kept
