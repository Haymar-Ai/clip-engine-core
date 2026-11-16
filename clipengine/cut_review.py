#!/usr/bin/env python3
"""The editor's pass over a clip's cut before its render.

OpusClip, Vizard, ClipsAI and the open-source clippers pick a moment with an LLM over the
transcript and then let an editor's rule (or the user) move its edges until the clip stands on
its own; they repair a cut instead of throwing the clip away. The
engine picked its moments the same way (select_semantic) but then judged the cut only after the
render, by the quality gate and the eye-check, and threw it away there: in the 600 clips made
11-07..11-09, 205 cut_start / cut_end rejects, 37 no_hook (every one a chat-peak pick, which never
gets a title) and 19 wrong_hook.

review() shows the LLM the window's transcript around the cut, one sentence per line with the
clip's lines marked, and asks where a viewer who sees the clip cold needs it to START and END
(the eye-check's own rules: no fragment, false start, reply to an unheard remark or punchline
without its setup at the start; no mid-sentence, filler, new-topic question or drift at the end)
and whether the burned title is supported by the final cut, writing one when it is not (or the
clip has none). The answer moves the cut by whole sentences; place() puts the new cuts in the
word gaps of the words the clip burns (boundaries.clip_from_sentences: the quietest frame of the
gap, leading fillers trimmed). `feedback` (the gate's or the eye-check's reject reasons) asks the
same question again after a reject: the repair moves the cut, it does not re-judge it.

Never raises: any LLM, parse or placement problem is a "keep" verdict with the reason, and the
cut goes on exactly as select left it. Config daily.cut_review (enabled, model, timeout_s,
max_move_s); env CUT_REVIEW=0 turns it off for one process tree.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from clipengine import config

from clipengine import boundaries
from clipengine import topic_map
from clipengine import transcript

HERE = Path(__file__).resolve().parent
DEFAULTS = {"enabled": True, "model": None, "backend": None, "timeout_s": 120.0,
            "max_move_s": 25.0, "context_before_s": 40.0, "context_after_s": 25.0}
MAX_HOOK_CHARS = 80
EDGE_S = 0.7            # = stages.WINDOW_EDGE_PAUSE_S: an end this close to the window's end cannot
                        # be judged (the gate hears 0.3 s past the cut at least, and the window's
                        # last sentence may run on past it): 11-09 replay 2679 / 2682 / 3187
SEAM = "…"              # marks a sentence the current cut splits (its other part is the next line)
TOL_S = 0.05            # a word belongs to the clip when its middle is this close inside a cut

PROMPT = """You are the editor of one short vertical clip cut from a longer {kind} video. Below is the
automatic transcript around the clip, one sentence per line: "[S<n> <start>-<end>s] text". Lines
marked ">" are inside the clip as it is cut now ({start:.1f}-{end:.1f} s, {dur:.1f} s). A line ending or
starting with "{seam}" is a sentence the current cut splits. The transcript is machine-made:
punctuation, sentence splits and a few words may be wrong.

A viewer sees this clip cold, with no context. Decide where it must START and END.
- START on the first sentence a cold viewer can follow. Never start on a fragment or a word cut off,
  on a false start or throat-clearing ("And the third and the thing, I think we know the..."), on a
  reply to a question or remark the clip leaves out ("How so?", "No, I'm saying that"), or on the
  punchline of a setup that came before ("Just kidding, never selling that car"). If the setup or
  the question was said just before the clip, start earlier to include it; if the clip opens on a
  false start, start later. A complete first sentence that opens with "But", "And", "So", "It" or a
  pronoun whose referent was said just before IS fine when the clip still makes sense on its own.
- END on the sentence that lands the point (the answer, the punchline, the reaction). Never end
  mid-sentence, on a filler or tag ("so", "and", "like", "yeah", "you know what I'm saying"), on a
  question that opens a new topic, or after the talk has drifted to a new topic. If the thought
  finishes a sentence or two after the clip, end later; if the clip trails into a new topic, end
  earlier.
- The clip must stay {mind:.0f}-{maxd:.0f} s long and inside this transcript. Move a boundary only to fix
  one of these faults, and by as little as fixes it; when the cut is already right, keep it.
{rules}{feedback}
Then the TITLE burned on screen over the first seconds. It is now: {hook}.
It must be supported by what is said inside YOUR final clip: a paraphrase, a tease or a mild
overstatement is fine; a wrong subject, a claim the clip contradicts or never reaches is not.
Keep it when it is supported; otherwise write a new punchy title of at most 8 words, a curiosity
gap grounded in the final clip (paraphrase, do not copy a transcript line). Write it from the
lines of your final clip ONLY: the lines before and after it are context for the cut, never for
the title (a viewer never hears them).

Return ONLY JSON, no code fences:
{{"start": <S number of the first line>, "end": <S number of the last line>, "start_fault": "<the fault you fixed, or none>", "end_fault": "<the fault you fixed, or none>", "title_ok": true | false, "title": "<the title to burn>"}}

TRANSCRIPT:
{body}"""

FEEDBACK = """
The cut below was already rendered once and REJECTED by the reviewer for:
  {why}
Fix that fault by moving the boundary it names (or the title, for a title fault). Answering with
the same cut is only right when the fault is in the captions or the picture, not in the words."""

TITLE_ONLY = """
The cut is FIXED this time (another cut could not be rendered): answer "start" and "end" with
the first and last ">" lines as they are, and judge only the title."""

TITLE_AGAIN = """
The cut is FIXED (the ">" lines). Your last title {title} used {words}, which the viewer never
hears inside the clip: write the title again from the ">" lines only ("title_ok": false)."""

KINDS = {"podcast": "podcast / talk", "stream": "livestream / gaming", "poker": "poker broadcast",
         "general": "talk"}

# the hard end rule, the clip's topic and the niche pack's own editor criteria
END_RULE = """- END RULE (hard): never end on the wrap-up after the point: a summary or restatement, a sign-off,
  thanks / subscribe / link lines, "anyway", "so yeah", "moving on", a tag line ("you know what I'm
  saying"), or a line that turns to the next topic. End on the line before it.
"""
TOPIC_RULE = """- TOPICS: the tag after each line's times is its topic from an outline of the video ("-" = between
  topics: small talk, transitions, wrap-ups). The clip belongs to topic T{t} ({title}): its first and
  last line must both be T{t} lines.
"""
PACK_RULE = """- For this {kind} video: {review}
"""


def settings(cfg=None):
    """daily.cut_review over DEFAULTS; CUT_REVIEW=0 in the environment turns it off."""
    if cfg is None:
        try:
            cfg = config.load()
        except (OSError, ValueError):
            cfg = {}
    st = {**DEFAULTS, **(((cfg.get("daily") or {}).get("cut_review")) or {})}
    st.pop("_doc", None)
    if os.environ.get("CUT_REVIEW", "").strip() == "0":
        st["enabled"] = False
    return st


def _mid(w):
    return (float(w["start"]) + float(w["end"])) / 2.0


def merged_words(window, side, start, end):
    """The words the editor reads: the clip's own sidecar words (the precise / strong pass the
    render burns, plus whatever context it carries) over the span they cover, the window's
    transcript (select's base pass) outside it."""
    side = [dict(w) for w in side or [] if str(w.get("w", "")).strip()]
    if not side:
        return [dict(w) for w in window or []]
    lo = min(float(w["start"]) for w in side)
    hi = max(float(w["end"]) for w in side)
    out = [dict(w) for w in window or [] if not (lo - TOL_S <= _mid(w) <= hi + TOL_S)] + side
    return sorted(out, key=lambda w: (float(w["start"]), float(w["end"])))


def lines(words, start, end, gap_s=0.7):
    """Sentences of `words` (transcript.sentences) with any sentence a cut splits split at the
    cut: [{"i", "start", "end", "text", "w0", "w1", "inside", "seam"}], numbered from 0."""
    start, end = float(start), float(end)
    out = []
    for s in transcript.sentences(words, gap_s=gap_s):
        parts, a = [], s["w0"]
        for k in range(s["w0"], s["w1"]):
            here, nxt = _mid(words[k]), _mid(words[k + 1])
            if here < start - TOL_S <= nxt or here <= end + TOL_S < nxt:
                parts.append((a, k))
                a = k + 1
        parts.append((a, s["w1"]))
        for n, (a, b) in enumerate(parts):
            ws = words[a:b + 1]
            text = " ".join(str(w["w"]).strip() for w in ws)
            seam = len(parts) > 1
            if seam and n > 0:
                text = SEAM + text
            if seam and n < len(parts) - 1:
                text = text + SEAM
            inside = start - TOL_S <= _mid(ws[len(ws) // 2]) <= end + TOL_S
            out.append({"i": len(out), "start": float(ws[0]["start"]), "end": float(ws[-1]["end"]),
                        "text": text, "w0": a, "w1": b, "inside": inside, "seam": seam})
    return out


def _tag(s, topics):
    """' T<n>' (the topic whose span holds the line's middle) or ' -', '' without a map."""
    if not topics:
        return ""
    m = (s["start"] + s["end"]) / 2.0
    t = next((t for t in topics if float(t["lo"]) <= m <= float(t["hi"])), None)
    return f" T{t['i']}" if t else " -"


def body(ls, lo, hi, topics=None):
    """The transcript lines the editor sees: those overlapping [lo, hi]."""
    return "\n".join(f"{'>' if s['inside'] else ' '} [S{s['i']} {s['start']:.1f}-{s['end']:.1f}s"
                     f"{_tag(s, topics)}] {' '.join(s['text'].split())}"
                     for s in ls if s["end"] >= lo and s["start"] <= hi)


def in_topic(s, topic):
    """True when line `s` (its middle) is inside `topic`'s span; True without a topic."""
    if not topic:
        return True
    return float(topic["lo"]) <= (s["start"] + s["end"]) / 2.0 <= float(topic["hi"])


def apply_rules(ls, a, b, topic=None, wrap=None):
    """(a, b, notes) of the answer's lines after the two rules: both ends inside the
    clip's topic (the nearest lines of it), then no trailing wrap-up line (wrap(text) -> bool,
    boundaries.wrapup_sentence). (None, None, notes) when no line of the topic is left."""
    notes = []
    if topic:
        ins = [s["i"] for s in ls if in_topic(s, topic)]
        if not ins:
            return None, None, ["no transcript line inside the clip's topic"]
        na, nb = max(a, ins[0]), min(b, ins[-1])
        if na > nb:
            return None, None, [f"S{a}-S{b} is outside topic T{topic['i']}"]
        if (na, nb) != (a, b):
            notes.append(f"topic T{topic['i']}: S{a}-S{b} -> S{na}-S{nb}")
        a, b = na, nb
    if wrap:
        nb = b
        while nb > a and wrap(ls[nb]["text"].replace(SEAM, " ")):
            nb -= 1
        if nb != b:
            notes.append("end rule: dropped " + " / ".join(
                json.dumps(" ".join(ls[k]["text"].replace(SEAM, " ").split())[:60])
                for k in range(nb + 1, b + 1)))
            b = nb
    return a, b, notes


def leaked_words(title, inside, outside):
    """The title's content words said only OUTSIDE the clip (quality_gate.hook_grounding's
    stem / fuzzy match): 11-09 2786's 'He's dancing for a cash prize', where 'prize of money' is
    said after the cut. [] when every content word is said inside or nowhere near."""
    from clipengine import quality_gate as qg
    _s, m_in, content = qg.hook_grounding(title, qg.norm_tokens(inside))
    _s, m_out, _c = qg.hook_grounding(title, qg.norm_tokens(outside))
    return [c for c in content if c not in m_in and c in m_out]


def prompt(ls, start, end, vertical, hook, mind, maxd, feedback=None, lo=None, hi=None,
           title_only=False, again=None, topics=None, topic=None, pack=None, end_rule=False):
    fb = FEEDBACK.format(why=" ".join(str(feedback).split())[:700]) if feedback else ""
    if title_only:
        fb += TITLE_ONLY
    if again:
        fb += TITLE_AGAIN.format(title=json.dumps(again[0]), words=", ".join(again[1]))
    pack = pack or {}
    kind = pack.get("kind") or KINDS.get(vertical, KINDS["general"])
    rules = (END_RULE if end_rule else "") + (
        TOPIC_RULE.format(t=topic["i"], title=json.dumps(topic.get("title") or ""))
        if topics and topic else "") + (
        PACK_RULE.format(kind=kind, review=" ".join(str(pack["review"]).split()))
        if pack.get("review") else "") + topic_map.hook_examples(pack)
    return PROMPT.format(kind=kind, start=float(start),
                         end=float(end), dur=float(end) - float(start), seam=SEAM, mind=mind,
                         maxd=maxd, rules=rules, feedback=fb,
                         hook=json.dumps(hook) if hook else "(none: write one)",
                         body=body(ls, float("-inf") if lo is None else lo,
                                   float("inf") if hi is None else hi,
                                   topics if topic else None))


def parse(text):
    """The first JSON object with "start" and "end" in the model's answer, or None."""
    if not isinstance(text, str):
        return None
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and "start" in obj and "end" in obj:
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


def _clean_hook(h):
    h = " ".join(str(h or "").split()).strip().strip('"').strip()
    return h[:MAX_HOOK_CHARS] if h else None


def current_span(ls):
    """(first, last) line index inside the clip, or None."""
    ins = [s["i"] for s in ls if s["inside"]]
    return (ins[0], ins[-1]) if ins else None


def place(words, ls, a, b, vc, rms=None, t0=0.0):
    """(start, end) of the clip spanning lines a..b, each cut in its word gap as select places
    them (boundaries.start_cut / end_cut: leading fillers trimmed, the quietest audio frame
    between the words; no tail tag or extension: the editor chose the last line), or None when
    it does not fit the vertical's duration band."""
    c = {**boundaries.DEFAULT_CFG, **(vc or {})}
    w1 = ls[b]["w1"]
    w0 = boundaries.trim_leading_fillers(words, ls[a]["w0"], w1)
    s = boundaries.start_cut(words, w0, rms, t0, 0.01, lead_pad=float(c["lead_pad_s"]))
    e = boundaries.end_cut(words, w1, rms, t0, 0.01, tail_pad=float(c["tail_pad_s"]))
    if not (float(c["min_dur_s"]) - 1e-6 <= e - s <= float(c["max_dur_s"]) + 1e-6):
        return None
    return float(s), float(e)


def split_by_cut(fs, fe, *tracks, lo=float("-inf"), hi=float("inf")):
    """(inside, outside) words of every transcript in `tracks` around the cut [fs, fe]: the
    leak check reads them all, so a word one pass heard only outside the cut counts (11-09 2786:
    base.en, the eye-check's transcript, heard 'prize of money' after the cut; the burned
    large-v3-turbo words heard other words there)."""
    inside, outside = [], []
    for track in tracks:
        for w in track or []:
            m = _mid(w)
            if fs - TOL_S <= m <= fe + TOL_S:
                inside.append(w)
            elif lo <= m <= hi:
                outside.append(w)
    return inside, outside


def review(clip, words, vertical, vc, *, llm, rms=None, t0=0.0, media_dur=None, feedback=None,
           st=None, title_only=False, alt_words=None, topics=None, topic=None, pack=None):
    """The editor's verdict on `clip`'s cut over `words` (merged_words). `llm(prompt, timeout)`
    returns the model's text. Returns
      {"verdict": "keep" | "move" | "error", "start", "end", "hook", "hook_changed",
       "from": [start, end], "lines": [a, b], "answer": {...}, "why", "seconds"}
    "move" carries the new cut; "keep" the old one (and possibly a new hook); "error" the old
    cut and hook with the reason. Never raises.

    `topics` / `topic` (topic_map: the window's outline and the clip's topic) tag the
    lines and keep both ends inside the clip's topic; vc drop_wrapup (+ wrapup_extra) is the hard
    end rule: the answer's trailing wrap-up lines are dropped, and a cut that already ends on one
    moves. `pack` = the niche's prompt pack (its kind and editor criteria)."""
    st = st or settings()
    from clipengine import boundaries as _b
    extra = tuple(vc.get("wrapup_extra") or ())
    wrap = (lambda t: _b.wrapup_sentence(t, extra)) if vc.get("drop_wrapup") else None
    topic = topic if topics else None
    t_0 = time.monotonic()
    start, end = float(clip["start_s"]), float(clip["end_s"])
    hook = _clean_hook(clip.get("hook"))
    res = {"verdict": "error", "start": start, "end": end, "hook": hook, "hook_changed": False,
           "from": [round(start, 3), round(end, 3)]}

    def done(**kw):
        res.update(kw, seconds=round(time.monotonic() - t_0, 1))
        return res

    def check_title(fs, fe):
        """A title the editor wrote that uses a word said only outside the final cut [fs, fe]
        is asked for once more (title only, the final cut fixed); still leaking, a clip that had
        a title keeps it, one that had none takes the better of the two."""
        if not res.get("hook_changed") or not res.get("hook"):
            return
        inside, outside = split_by_cut(fs, fe, words, alt_words, lo=lo, hi=hi)
        leak = leaked_words(res["hook"], inside, outside)
        if not leak:
            return
        res["title_leak"] = {"title": res["hook"], "words": leak}
        ls2 = lines(words, fs, fe)
        again = llm(prompt(ls2, fs, fe, vertical, res["hook"], mind, maxd, None, lo, hi,
                           title_only=True, again=(res["hook"], leak), pack=pack),
                    float(st.get("timeout_s") or DEFAULTS["timeout_s"]))
        t2 = _clean_hook((parse(again) or {}).get("title"))
        leak2 = leaked_words(t2, inside, outside) if t2 else ["(no title)"]
        res["title_leak"].update(retry=t2, retry_leak=leak2)
        if t2 and not leak2:
            res.update(hook=t2, hook_changed=t2 != hook)
        elif hook:
            res.update(hook=hook, hook_changed=False)
        elif t2:
            res.update(hook=t2, hook_changed=True)
    try:
        if not words:
            return done(why="no transcript")
        ls = lines(words, start, end)
        span = current_span(ls)
        if span is None:
            return done(why="no transcript line inside the cut")
        mind = float(vc.get("min_dur_s", boundaries.DEFAULT_CFG["min_dur_s"]))
        maxd = float(vc.get("max_dur_s", boundaries.DEFAULT_CFG["max_dur_s"]))
        if media_dur:
            maxd = min(maxd, float(media_dur))
        lo = start - float(st.get("context_before_s", DEFAULTS["context_before_s"]))
        hi = end + float(st.get("context_after_s", DEFAULTS["context_after_s"]))
        # yield 11-11: the topic tags and the END RULE line reach the prompt only when switched on
        # (topic_map.select_cfg); apply_rules below keeps both as code guards either way
        tagged = bool(vc.get("topic_editor_rule"))
        text = llm(prompt(ls, start, end, vertical, hook, mind, maxd, feedback, lo, hi,
                          title_only=title_only, topics=topics if tagged else None,
                          topic=topic if tagged else None, pack=pack,
                          end_rule=bool(wrap) and bool(vc.get("end_rule_prompt"))
                          and not title_only),
                   float(st.get("timeout_s") or DEFAULTS["timeout_s"]))
        ans = parse(text)
        if ans is None:
            return done(why="no JSON answer")
        res["answer"] = {k: ans.get(k) for k in ("start", "end", "start_fault", "end_fault",
                                                 "title_ok", "title")}
        a, b = _idx(ans.get("start")), _idx(ans.get("end"))
        if title_only:
            a, b = span                       # whatever lines it named: the cut stays
        shown = {s["i"] for s in ls if s["end"] >= lo and s["start"] <= hi}
        if a not in shown or b not in shown or a > b:
            return done(why=f"answer lines S{ans.get('start')}-S{ans.get('end')} not a span of "
                            f"the transcript shown")
        rule_notes = []
        if not title_only:
            ra, rb, rule_notes = apply_rules(ls, a, b, topic, wrap)
            if ra is None:
                res["rules"] = rule_notes
                return done(verdict="keep", why="; ".join(rule_notes))
            if (ra, rb) != (a, b):
                res["rules"] = rule_notes
                a, b = ra, rb
        new_hook = hook
        if ans.get("title_ok") is False or not hook:
            new_hook = _clean_hook(ans.get("title")) or hook
        res.update(hook=new_hook, hook_changed=new_hook != hook, lines=[a, b])

        def keep(why):
            # a title written for lines that could not be placed was not written for this cut
            if (a, b) != span and hook:
                res.update(hook=hook, hook_changed=False)
            check_title(start, end)
            return done(verdict="keep", why=why)
        if (a, b) == span:
            return keep("the editor keeps the cut")
        cut = place(words, ls, a, b, {**vc, "min_dur_s": mind, "max_dur_s": maxd}, rms=rms, t0=t0)
        if cut is None:
            return keep(f"S{a}-S{b} does not fit {mind:.0f}-{maxd:.0f} s")
        ns, ne = cut
        if media_dur and (ns < 0 or ne > float(media_dur)):
            return keep("the new cut leaves the window")
        if media_dur and ne > end + 1e-6 and ne > float(media_dur) - EDGE_S:
            return keep(f"the new end {ne:.2f} s is at the window's end ({float(media_dur):.2f} s): "
                        f"what follows it cannot be heard")
        move = max(abs(ns - start), abs(ne - end))
        if move > float(st.get("max_move_s", DEFAULTS["max_move_s"])):
            return keep(f"the new cut moves {move:.1f} s (cap "
                        f"{float(st.get('max_move_s')):.0f} s)")
        if (round(ns, 3), round(ne, 3)) == (round(start, 3), round(end, 3)):
            return keep("the new lines give the same cut")
        if not hook or not res["hook_changed"]:
            # a cut that moved keeps its title only when the editor says it is still supported
            if ans.get("title_ok") is not True and _clean_hook(ans.get("title")):
                res.update(hook=_clean_hook(ans.get("title")), hook_changed=True)
        check_title(ns, ne)
        return done(verdict="move", start=round(ns, 3), end=round(ne, 3),
                    why=f"S{span[0]}-S{span[1]} -> S{a}-S{b}: start {ans.get('start_fault') or 'ok'}; "
                        f"end {ans.get('end_fault') or 'ok'}"
                        + (f" ({'; '.join(rule_notes)})" if rule_notes else ""))
    except Exception as e:                    # noqa: BLE001 - the cut goes on as it is
        return done(verdict="error", why=f"{type(e).__name__}: {str(e)[:200]}")


def sidecar(words, start, end):
    """The words sidecar of a reviewed cut: every merged word (the context outside the cut stays
    for the gate's sentence checks), crosstalk flags kept."""
    return [{k: w[k] for k in w if k in ("w", "start", "end", "p", "cap", "asr_start", "asr_end")}
            for w in words]


def claude_llm(vc, st=None):
    """llm(prompt, timeout) on select's claude_cli account (select.claude_config_dir) and model
    (daily.cut_review.model, else select.semantic_model): the backend select's picks use. No
    fallback to the local model: a pick it cannot review keeps today's flow."""
    from clipengine import select_semantic
    st = st or settings()
    backend = st.get("backend") or vc.get("semantic_backend") or "claude_cli"
    model = st.get("model") or vc.get("semantic_model") or "opus"

    def llm(text, timeout):
        return select_semantic._llm(text, model, backend=backend, timeout=timeout, opts=vc)
    llm.model, llm.backend = model, backend
    return llm
