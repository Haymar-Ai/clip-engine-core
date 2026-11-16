#!/usr/bin/env python3
"""Topic map first: one outline call per source, before any clip is picked.

zhouxiaoka/autoclip_mvp outlines the whole video as
topics before it picks anything, and ends each topic on its last core sentence, "excluding the
summary or transition lines that follow". Our cut_start "needs earlier context" and cut_end
"drifted into the next topic" rejects (about 70 meaning rows in the 11-07..09 sample) are what a topic map prevents: a clip that stays inside one topic starts
and ends on that topic's own lines.

for_source() asks the select LLM (select's backend, account and model: the existing backend, no
local fallback) once per source for the topics of its transcript: each topic's CORE span as
sentence numbers (its first and last sentence that carry it; lead-in small talk and the trailing
wrap-up left out), a short title, and whether it was already under way at the first line. The map
is cached per source under <scratch>/topic_maps/<source_id>.json (keyed by the transcript, the
model and the prompt version), so the render's cut review and a second select of the same window
read it without another call. Each topic gets a time span [lo, hi] (the middles of the pauses
around it): a sentence of ANY transcript of the window (base, precise, strong pass) belongs to the
topic when its middle is inside, so the later passes keep the cut inside it by time.

select (stages.select) puts each pick inside the one topic it overlaps most (fit) and gives
boundaries the topic's span (cfg span_s) for every move after it (precision pass, edge-verify,
snap); a pick that overlaps no topic is dropped. The cut review (stages._cut_review) shows the
topics and refuses any start or end outside the clip's topic.

Never raises: no map (off, an error, a garbage answer, too few words, no time) means select and
the review run exactly as before. Config select.topic_map (enabled, model, timeout_s,
min_topic_s, min_words); env TOPIC_MAP=0 turns it off for one process tree. The end rule
(select.end_rule, env END_RULE=0) and the per-niche prompt packs (select.prompt_packs, env
PROMPT_PACKS=0, files clipengine/prompt_packs/<niche>.json) are switched here as well.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from clipengine import config

HERE = Path(__file__).resolve().parent
PACK_DIR = HERE / "prompt_packs"
VERSION = 1                 # in the cache key: a new outline prompt makes new maps
DEFAULTS = {"enabled": True, "model": None, "backend": None, "timeout_s": 60.0,
            "min_topic_s": 20.0, "min_words": 50}
EDGE_PAD_S = 0.5            # a window's first / last topic span reaches this far past its sentence
MAX_TITLE = 60

PROMPT = """You are mapping the structure of a {kind} transcript before anyone cuts clips from it.
Each line is one sentence: "[S<number> <seconds>s] text". The transcript is machine-made: punctuation,
sentence splits and a few words may be wrong.
{guide}
Split the transcript into TOPICS, in order. A topic is one subject, story, question with its answer,
argument or event, as a viewer would name it; a new topic starts where the talk switches to something
else. Do not split one story or answer into its beats: a topic usually runs {min_topic:.0f} s or more.
For each topic give its CORE span:
- "s0" = the first sentence that carries the topic. Skip the greeting, small talk, filler and the
  "so, anyway" lead-in before it.
- "s1" = the last sentence that carries it: the line that lands its point, answer or punchline.
  EXCLUDE the wrap-up after it: a summary or restatement, a sign-off, thanks / subscribe / link lines,
  "anyway", "so yeah", "moving on", tag lines like "you know what I'm saying", and the transition into
  the next topic.
Lines between two topics' core spans belong to no topic. Topics never overlap.
"continues": true when the topic was already under way at the transcript's first line (it opens
mid-thought: its setup was said before this transcript); false otherwise.
"title": what the topic is about, at most 6 words.

Return ONLY JSON: {{"topics":[{{"s0":<int>,"s1":<int>,"title":"<at most 6 words>","continues":<true|false>}}]}}

TRANSCRIPT:
{body}"""


# ---- switches ----
def _cfg(cfg=None):
    if cfg is not None:
        return cfg
    try:
        return config.load()
    except (OSError, ValueError):
        return {}


def settings(cfg=None):
    """select.topic_map over DEFAULTS; TOPIC_MAP=0 in the environment turns it off."""
    st = {**DEFAULTS, **(((_cfg(cfg).get("select") or {}).get("topic_map")) or {})}
    st.pop("_doc", None)
    if os.environ.get("TOPIC_MAP", "").strip() == "0":
        st["enabled"] = False
    return st


def _switch(block, env, cfg=None):
    if os.environ.get(env, "").strip() == "0":
        return False
    return bool((((_cfg(cfg).get("select") or {}).get(block)) or {}).get("enabled", True))


def end_rule_enabled(cfg=None):
    """select.end_rule.enabled (default True); END_RULE=0 turns it off."""
    return _switch("end_rule", "END_RULE", cfg)


def packs_enabled(cfg=None):
    """select.prompt_packs.enabled (default True); PROMPT_PACKS=0 turns it off."""
    return _switch("prompt_packs", "PROMPT_PACKS", cfg)


_PACKS = {}


def pack(niche, cfg=None, pack_dir=None):
    """The niche's prompt pack (clipengine/prompt_packs/<niche>.json: kind, rubric, outline, review,
    wrapup), or None (packs off, no niche, no file, unreadable): the vertical's old prompts."""
    if not niche or not packs_enabled(cfg) or not re.fullmatch(r"[a-z0-9_-]+", str(niche)):
        return None
    path = Path(pack_dir or PACK_DIR) / f"{niche}.json"
    if path not in _PACKS:
        try:
            p = json.loads(path.read_text())
            _PACKS[path] = p if isinstance(p, dict) else None
        except (OSError, ValueError):
            _PACKS[path] = None
    return _PACKS[path]


def _flag(block, key, cfg=None):
    """select.<block>.<key> (default False)."""
    return bool((((_cfg(cfg).get("select") or {}).get(block)) or {}).get(key, False))


def select_cfg(niche, cfg=None, pack_dir=None):
    """The boundaries cfg keys select and the cut review add for a niche: drop_wrapup (the end
    rule on) and wrapup_extra (the pack's own wrap-up openers). Not in vertical_cfg, so every
    other caller of boundaries (render, repairs) keeps today's cuts.
    yield 11-11: the end rule's prompt line (end_rule_prompt: select.end_rule.prompt_line) and the
    editor's topic tags with its "both ends on T<n> lines" sentence (topic_editor_rule:
    select.topic_map.editor_rule) are off by default. On the replay they made the editor
    trim lines HEAD kept (lifestyle / finance). The code guards (the wrap-up drop, the topic
    clamp) stay on."""
    p = pack(niche, cfg, pack_dir) or {}
    on = end_rule_enabled(cfg)
    return {"drop_wrapup": on, "end_rule_prompt": on and _flag("end_rule", "prompt_line", cfg),
            "topic_editor_rule": _flag("topic_map", "editor_rule", cfg),
            "wrapup_extra": tuple(str(x) for x in (p.get("wrapup") or ()) if str(x).strip())}


HOOK_EXAMPLES = """
TITLE EXAMPLES ({label}): title patterns for this week's topic. Use one ONLY when the clip itself
says what the title claims; fill a {{slot}} from the clip (the streamer or host by name). Not text
to copy, and no reason to pick a clip. {rules}
{lines}"""


def hook_examples(pk, today=None):
    """The pack's dated title examples (hook_examples: {label, from, to, rules, lines}) as a prompt
    block while today (PT, YYYY-MM-DD) is inside from..to, else "". The shipped packs carry no dated
    examples; out of a block's window the prompts are byte for byte the same."""
    h = (pk or {}).get("hook_examples") or {}
    lines = [str(x).strip() for x in (h.get("lines") or ()) if str(x).strip()]
    if not lines:
        return ""
    if today is None:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        today = datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
    if not (str(h.get("from") or "") <= today <= str(h.get("to") or "")):
        return ""
    return HOOK_EXAMPLES.format(label=h.get("label") or "this week",
                                rules=" ".join(str(h.get("rules") or "").split()),
                                lines="\n".join(f"- {x}" for x in lines))


# ---- the outline ----
KINDS = {"podcast": "podcast / talk", "stream": "livestream / gaming", "poker": "poker broadcast",
         "general": "talk"}


def _line(s):
    return f"[S{s['i']} {float(s['start']):.0f}s] {' '.join(str(s['text']).split())}"


def build_prompt(sents, vertical="general", pk=None, min_topic_s=20.0):
    pk = pk or {}
    guide = f"\n{pk['outline']}\n" if pk.get("outline") else ""
    return PROMPT.format(kind=pk.get("kind") or KINDS.get(vertical, KINDS["general"]), guide=guide,
                         min_topic=float(min_topic_s), body="\n".join(_line(s) for s in sents))


def parse(text):
    """The first JSON object with a "topics" list in the model's answer, or None."""
    if not isinstance(text, str):
        return None
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("topics"), list):
            return obj
    return None


def _idx(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, str):
        v = v.strip().lstrip("Ss")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else None


def validate(data, sents):
    """Topics of a parsed answer on `sents` (sentence "i" = list position), in order, without
    overlaps (a later topic starts after the one before it ends; one left empty is dropped), each
    with its time span [lo, hi]: the middle of the pause before its first sentence and after its
    last (a window's first / last sentence: EDGE_PAD_S past it). [] when nothing valid is left."""
    n = len(sents)
    raw = []
    for t in (data or {}).get("topics") or []:
        if not isinstance(t, dict):
            continue
        a, b = _idx(t.get("s0")), _idx(t.get("s1"))
        if a is None or b is None or not (0 <= a <= b < n):
            continue
        raw.append({"s0": a, "s1": b, "title": " ".join(str(t.get("title") or "").split())[:MAX_TITLE],
                    "continues": t.get("continues") is True})
    raw.sort(key=lambda t: (t["s0"], t["s1"]))
    out = []
    for t in raw:
        if out and t["s0"] <= out[-1]["s1"]:
            t = {**t, "s0": out[-1]["s1"] + 1, "continues": False}
            if t["s0"] > t["s1"]:
                continue
        out.append(t)
    for k, t in enumerate(out):
        a, b = t["s0"], t["s1"]
        st, en = float(sents[a]["start"]), float(sents[b]["end"])
        lo = (float(sents[a - 1]["end"]) + st) / 2.0 if a > 0 else max(0.0, st - EDGE_PAD_S)
        hi = (en + float(sents[b + 1]["start"])) / 2.0 if b + 1 < n else en + EDGE_PAD_S
        t.update(i=k + 1, start=round(st, 3), end=round(en, 3), lo=round(lo, 3), hi=round(hi, 3))
        if k:
            t["continues"] = False          # only the first topic can open before the transcript
    return out


def _key(sents, niche, model, vertical):
    h = hashlib.md5()
    h.update(json.dumps([VERSION, niche, model, vertical]).encode())
    for s in sents:
        h.update(f"{float(s['start']):.2f}|{float(s['end']):.2f}|{s['text']}\n".encode())
    return h.hexdigest()


def _safe(sid):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(sid))[:120] or "source"


def cached(sid, cache_dir, key=None):
    """The map cached for `sid` (any key when `key` is None), or None."""
    try:
        m = json.loads((Path(cache_dir) / f"{_safe(sid)}.json").read_text())
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(m, dict) or m.get("v") != VERSION or not isinstance(m.get("topics"), list):
        return None
    if key is not None and m.get("key") != key:
        return None
    return m


def _write(path, m):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(m, indent=1))
    tmp.replace(path)


def claude_llm(vc, st=None):
    """llm(prompt, timeout) on select's backend and account (select.semantic_backend,
    select.claude_config_dir) and model (select.topic_map.model, else select.semantic_model)."""
    from clipengine import select_semantic
    st = st or settings()
    backend = st.get("backend") or vc.get("semantic_backend") or "claude_cli"
    model = st.get("model") or vc.get("semantic_model") or "opus"

    def llm(text, timeout):
        return select_semantic._llm(text, model, backend=backend, timeout=timeout, opts=vc)
    llm.model, llm.backend = model, backend
    return llm


def for_source(sid, sents, niche, vertical, vc, *, cache_dir, llm=None, deadline=None, st=None):
    """(map or None, call record) for a source's sentences. The map:
      {"v", "key", "sid", "niche", "model", "backend", "seconds", "topics": [{"i", "s0", "s1",
       "start", "end", "lo", "hi", "title", "continues"}], "n_sents"}
    cached per source; the call record is {"step": "outline", "backend", "model", "s",
    "outcome": answered | cached | off | short | no_time | error | garbage | empty, "why"}.
    Never raises."""
    st = st or settings()
    rec = {"step": "outline", "outcome": "off"}
    try:
        if not st.get("enabled"):
            return None, rec
        n_words = sum(len(str(s.get("text", "")).split()) for s in sents or [])
        if not sents or n_words < int(st.get("min_words") or 0):
            rec.update(outcome="short", why=f"{n_words} words")
            return None, rec
        fn = llm or claude_llm(vc, st)
        model, backend = getattr(fn, "model", st.get("model")), getattr(fn, "backend", st.get("backend"))
        rec.update(backend=backend, model=model)
        key = _key(sents, niche, model, vertical)
        path = Path(cache_dir) / f"{_safe(sid)}.json"
        hit = cached(sid, cache_dir, key)
        if hit is not None:
            rec.update(outcome="cached", s=0.0)
            return (hit if hit.get("topics") else None), rec
        timeout = float(st.get("timeout_s") or DEFAULTS["timeout_s"])
        if deadline is not None:
            left = deadline - time.monotonic()
            if left < timeout / 2.0:
                rec.update(outcome="no_time", why=f"{left:.0f}s before select's deadline")
                return None, rec
            timeout = min(timeout, left)
        t0 = time.monotonic()
        pk = pack(niche) or {}
        try:
            text = fn(build_prompt(sents, vertical, pk, float(st.get("min_topic_s") or 20.0)), timeout)
        except Exception as e:            # noqa: BLE001 - no map: select as before
            rec.update(outcome="error", s=round(time.monotonic() - t0, 1),
                       why=f"{type(e).__name__}: {str(e)[:160]}")
            return None, rec
        rec["s"] = round(time.monotonic() - t0, 1)
        data = parse(text)
        if data is None:
            rec.update(outcome="garbage", why="no JSON with a topics list")
            return None, rec
        topics = validate(data, sents)
        m = {"v": VERSION, "key": key, "sid": sid, "niche": niche, "vertical": vertical,
             "model": model, "backend": backend, "seconds": rec["s"], "at": time.time(),
             "n_sents": len(sents), "topics": topics, "pack": pk.get("niche")}
        _write(path, m)                   # an empty map is a verdict too: no second call
        if not topics:
            rec.update(outcome="empty", why="no valid topic in the answer")
            return None, rec
        rec["outcome"] = "answered"
        return m, rec
    except Exception as e:                # noqa: BLE001 - never fail a select over the map
        rec.update(outcome="error", why=f"{type(e).__name__}: {str(e)[:160]}")
        return None, rec


# ---- picks inside one topic ----
def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def topic_for(t0, t1, topics, near_s=0.0):
    """The topic [lo, hi] overlapping [t0, t1] most (ties: the earlier, where the clip opens);
    with no overlap, the nearest one within `near_s` seconds; else None."""
    best = None
    for t in topics or []:
        ov = _overlap(float(t0), float(t1), float(t["lo"]), float(t["hi"]))
        if ov > 0 and (best is None or ov > best[0] + 1e-9):
            best = (ov, t)
    if best:
        return best[1]
    if near_s > 0:
        gaps = [(min(abs(float(t["lo"]) - float(t1)), abs(float(t0) - float(t["hi"]))), t)
                for t in topics or []]
        gaps = [g for g in gaps if g[0] <= near_s]
        if gaps:
            return min(gaps, key=lambda g: g[0])[1]
    return None


def fit(s0, s1, sents, topics, near_s=0.0):
    """(topic, s0, s1, note) of the pick s0..s1 (sentence positions in `sents`, the map's
    sentences) inside the one topic it overlaps most, clamped to that topic's lines; (None, s0, s1,
    why) when it overlaps none. note: None (already inside) or what was cut off."""
    t = topic_for(float(sents[s0]["start"]), float(sents[s1]["end"]), topics, near_s)
    if t is None:
        return None, s0, s1, "outside every topic"
    a, b = max(s0, int(t["s0"])), min(s1, int(t["s1"]))
    if a > b:                               # only `near_s` matched: the topic's own nearest edge
        a = b = int(t["s0"]) if float(sents[s1]["end"]) <= float(t["lo"]) else int(t["s1"])
    cut = []
    if a > s0:
        cut.append(f"start S{s0}->S{a}")
    if b < s1:
        cut.append(f"end S{s1}->S{b}")
    return t, a, b, (f"into T{t['i']}: " + ", ".join(cut)) if cut else None


def span_cfg(topic):
    """The boundaries cfg key that keeps a cut inside `topic`."""
    return {"span_s": (float(topic["lo"]), float(topic["hi"]))} if topic else {}


def compact(topics):
    """The map as a clip's features carry it: [[lo, hi, title], ...]."""
    return [[t["lo"], t["hi"], t.get("title") or ""] for t in topics or []]


def clip_topic(feats, start, end, cache_dir=None, sid=None):
    """(topics, topic) for a clip: its features' map (select), else the source's cached
    map; the topic is the clip's own (features "topic"), else the one its cut overlaps most.
    ([], None) when there is no map."""
    feats = feats or {}
    topics = [{"i": k + 1, "lo": float(t[0]), "hi": float(t[1]), "title": t[2] if len(t) > 2 else ""}
              for k, t in enumerate(feats.get("topics") or []) if isinstance(t, (list, tuple))
              and len(t) >= 2]
    if not topics and cache_dir and sid:
        m = cached(sid, cache_dir)
        topics = [{"i": t["i"], "lo": float(t["lo"]), "hi": float(t["hi"]), "title": t.get("title")}
                  for t in (m or {}).get("topics") or []]
    if not topics:
        return [], None
    own = feats.get("topic")
    if isinstance(own, dict) and own.get("lo") is not None:
        for t in topics:
            if abs(t["lo"] - float(own["lo"])) < 1e-3 and abs(t["hi"] - float(own["hi"])) < 1e-3:
                return topics, t
    return topics, topic_for(float(start), float(end), topics)


def prompt_block(topics, sents=None):
    """The TOPICS block select's prompt carries (sentence numbers of `sents`), or ""."""
    if not topics:
        return ""
    lines = ["TOPICS (an outline of this transcript). Every clip stays inside ONE topic: its s0 and s1",
             "are lines of the same topic. Lines in no topic are small talk, transitions or wrap-ups:",
             "no clip starts or ends on them."]
    for t in topics:
        note = " (already under way at the first line: its setup was said before this transcript)" \
            if t.get("continues") else ""
        lines.append(f"T{t['i']} S{t['s0']}-S{t['s1']} {json.dumps(t.get('title') or '')}{note}")
    covered = {k for t in topics for k in range(int(t["s0"]), int(t["s1"]) + 1)}
    if sents:
        free = [s["i"] for s in sents if s["i"] not in covered]
        if free:
            lines.append("In no topic: " + ", ".join(f"S{k}" for k in free[:60])
                         + (" ..." if len(free) > 60 else ""))
    return "\n".join(lines) + "\n\n"
