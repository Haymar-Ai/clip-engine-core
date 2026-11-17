"""The engine: source video -> transcript -> picked moments -> cuts -> 9:16 render -> gate.

Each step reads rows in its input state (runstate: no database), processes, advances state.
Selection is content-aware for every vertical: transcript -> LLM picks sentence ranges ->
boundaries.py places each cut in an inter-word gap (energy peaks stand in when the LLM is down;
gameplay without speech uses the energy envelope). Render wraps render/recompose.py, then the
quality gate judges the rendered pixels; a clip that fails on its cuts or captions is repaired
(re-cut, re-captioned, re-titled) and rendered again before it is dropped.

run() is the one-source orchestrator the CLI calls: harvest, select, render every clip, and write
<out>/<source_id>/clip_NN.mp4, clip_NN.json and run.json."""
import os, sys, json, math, re, subprocess, time, contextlib
from pathlib import Path
import numpy as np
from clipengine import config, runstate, select_semantic, transcript, boundaries, scenes
from clipengine import edge_verify
from clipengine import topic_map
from clipengine import quality_gate
from clipengine import pick_quality
from clipengine import reason_codes
from clipengine import crosstalk
from clipengine import strong_asr
from clipengine import ytdl

CFG = config.load()
RECOMPOSE = [sys.executable, "-m", "clipengine.render.recompose"]
SCRATCH = Path(CFG.get("scratch_dir") or config.DEFAULT_SCRATCH)
COOKIES = ""                 # a cookies file for YouTube downloads (cli --cookies); none by default
_SEL = CFG.get("select", {})
TOP_N = int(_SEL.get("top_n", 6))
TRANSCRIBE_MODEL = _SEL.get("transcribe_model", "base.en")   # cheap full pass, for selection
REFINE_MODEL = _SEL.get("refine_model", "small.en")          # precise pass, picked windows only
SKIP_HEAD = float(_SEL.get("skip_head_s", 90))   # never clip the cold-open/intro card
SKIP_TAIL = float(_SEL.get("skip_tail_s", 30))   # never clip the outro/end card
OVERLAP_MAX = float(_SEL.get("overlap_max", 0.3))
MIN_SPEECH_DENSITY = float(_SEL.get("min_speech_density", 0.15))
SENTENCE_GAP = float(_SEL.get("sentence_gap_s", 0.7))
VALID_LAYOUTS = ("auto", "general", "track", "split", "screencast", "broadcast", "hudband",
                 "reaction", "gamestack")
SR = 16000
REFINE_PAD_S = 8.0          # precision pass re-transcribes [start - pad, end + pad]
CAPTION_MODEL = CFG.get("render", {}).get("caption_model", "small.en")
CAPTION_STRONG_MODEL = CFG.get("render", {}).get("caption_strong_model")   # None: select's words
LOUDNORM_LUFS = CFG.get("render", {}).get("loudnorm_lufs", -14)
PICK_QUALITY_FPS = float(_SEL.get("pick_quality_fps", 2.0))   # source samples/s for the pre-check
_SPEAKER_CROP_LAYOUTS = ("track", "full_bleed_speaker", "podcast_two_shot")


def _log(level, stage, msg, **ids):
    runstate.log_event(level, stage, msg, **ids)
    print(f"[{stage}] {msg}", flush=True)


# ---- HARVEST: source URL or local file -> local media, registered ----
def _is_youtube_url(url):
    """A cookies file (cli --cookies) is a logged-in YouTube export; passing it to a Twitch or
    other host confuses yt-dlp. Only attach it for YouTube."""
    u = str(url or "").lower()
    return "youtube.com" in u or "youtu.be" in u


def harvest(source):
    sid = source["source_id"]
    try:
        lp = source.get("local_path")
        if not (lp and Path(lp).exists()):
            SCRATCH.mkdir(parents=True, exist_ok=True)
            out = SCRATCH / f"{sid}.mp4"
            cmd = [ytdl.get_ytdlp_bin(), "--no-warnings", "-f",
                   "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/b[height<=1080]",
                   "--merge-output-format", "mp4", "-o", str(out), source["url"]]
            if _is_youtube_url(source["url"]) and ytdl.scratch_copy(COOKIES):
                cmd[1:1] = ["--cookies", ytdl.scratch_copy(COOKIES)]
            ytdl.ensure_deno_path()
            subprocess.run(cmd, check=True, env=os.environ.copy(),
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            lp = str(out)
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", lp]).decode().strip())
        runstate.update_source(sid, local_path=lp, duration_s=dur, state="harvested")
        _log("info", "harvest", f"{sid} harvested ({dur/60:.1f}min)", source_id=sid)
        return True
    except Exception as e:
        runstate.update_source(sid, state="failed", error=str(e)[:400])
        _log("error", "harvest", f"{sid} FAILED: {e}", source_id=sid)
        return False


# ---- SELECT: energy signals ----
def energy_envelope(media, sr=SR):
    """Per-second RMS of the mono audio (samples scaled to [-1, 1], same scale as
    transcript.load_audio). Streamed from ffmpeg, so a multi-hour VOD never sits in memory."""
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", str(media), "-vn", "-ac", "1",
                          "-ar", str(sr), "-f", "s16le", "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out, carry, tail = [], b"", np.zeros(0)
    try:
        while True:
            chunk = p.stdout.read(1 << 22)
            if not chunk:
                break
            chunk = carry + chunk
            n = len(chunk) // 2 * 2                      # an odd byte waits for the next chunk
            carry = chunk[n:]
            x = np.concatenate([tail, np.frombuffer(chunk[:n], dtype="<i2") / 32768.0])
            k = len(x) // sr
            if k:
                sec = x[:k * sr].reshape(k, sr)
                out.append(np.sqrt((sec * sec).mean(axis=1)))
            tail = x[k * sr:]                            # partial second carries over
    finally:
        p.stdout.close()
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited {rc} decoding audio")
    return np.concatenate(out) if out else np.zeros(0)


def zscores(env):
    """(env - mean) / std; all zeros when the envelope is empty or flat."""
    e = np.asarray(env, dtype=np.float64)
    sd = float(e.std()) if len(e) else 0.0
    return (e - e.mean()) / sd if sd > 0 else np.zeros_like(e)


def mean_z(z, lo, hi):
    """Mean z-score over seconds [floor(lo), ceil(hi)) (0.0 if none)."""
    a, b = max(0, int(math.floor(lo))), min(len(z), int(math.ceil(hi)))
    return float(np.mean(z[a:b])) if b > a else 0.0


def sigmoid(x):
    return 0.5 * (1.0 + math.tanh(float(x) / 2.0))     # overflow-free logistic


def _pick_windows(win_scores, top_n=TOP_N, floor_k=1.0, min_keep=0, suppress=60,
                  head_skip=0, tail_skip=0):
    """Pure ranker over per-second scores. Keep seconds that clear the adaptive floor
    (mean + floor_k*std), non-max-suppressed by `suppress` seconds, capped at top_n, but never
    fewer than min_keep. head_skip/tail_skip exclude the cold-open and outro (index == second).
    Returns [(index, score)] strongest first."""
    if not win_scores:
        return []
    mean = sum(win_scores) / len(win_scores)
    std = (sum((x - mean) ** 2 for x in win_scores) / len(win_scores)) ** 0.5
    floor = mean + floor_k * std
    last_ok = len(win_scores) - 1 - int(tail_skip)
    picks = []
    for s, sc in sorted(enumerate(win_scores), key=lambda x: -x[1]):   # strongest first
        if sc <= floor and len(picks) >= min_keep:
            break                                   # ranked desc: at/under floor is not a peak
        if s < head_skip or s > last_ok:
            continue                                 # intro/outro card — never a clip
        if all(abs(s - p[0]) >= suppress for p in picks):
            picks.append((s, sc))
        if len(picks) >= top_n:
            break
    return picks


def energy_peaks(z, top_n, head_skip=0, tail_skip=0, min_z=1.0, suppress_s=60):
    """Times (s, centre of the peak second) of z-score peaks above min_z, >= suppress_s apart,
    strongest first. On z-scores _pick_windows' floor (mean + k*std) is exactly k = min_z."""
    picks = _pick_windows([float(v) for v in z], top_n=top_n, floor_k=min_z, min_keep=0,
                          suppress=suppress_s, head_skip=int(head_skip), tail_skip=int(tail_skip))
    return [s + 0.5 for s, _ in picks]


# ---- SELECT: config / layout ----
STRICT_ENDS = True           # run-on / continuation / interjection ends are not clean
MIN_END_GAP_S = 0.25        # ... a punctuated end needs this much pause before the next word
WINDOW_EDGE_PAUSE_S = 0.7   # a window's first / last sentence is a clean edge only after / before
                            # this much silence at the window edge (= sentence_gap_s)
WINDOW_CONTEXT_S = 4.0      # clip-quality 10-21 stage 1: a harvested window's LLM-eligible sentences
                            # stay this far inside it; the rest of its pad (disk_safe_harvest.
                            # WINDOW_PAD_S) is room for the cuts to snap to a sentence edge
_WINDOW_SID = re.compile(r"_w\d+$")


def vertical_cfg(vertical):
    """The select block merged with its verticals[vertical] entry (unknown vertical -> general)."""
    sel = CFG.get("select", {})
    vs = sel.get("verticals", {})
    vc = {**{k: v for k, v in sel.items() if k != "verticals"},
          **vs.get(vertical, vs.get("general", {}))}
    vc.setdefault("strict_ends", STRICT_ENDS)
    vc.setdefault("min_end_gap_s", MIN_END_GAP_S)
    return vc


def window_edges(sid, words, dur):
    """`edges_clean` for clean_clip on a source's transcript. A whole source: its first and last
    sentence are real edges (True). A harvested window `<parent>_wNN` is cut out of the middle of
    a VOD at a caption cue, so its first / last sentence is a fragment unless the speech starts /
    ends WINDOW_EDGE_PAUSE_S inside the window (10-20 clip 127 started at the window's
    0.0 s mid-sentence, 114 ended at its end on a fragment)."""
    if not _WINDOW_SID.search(sid or "") or not words:
        return True
    first, last = float(words[0]["start"]), float(words[-1]["end"])
    return (first >= WINDOW_EDGE_PAUSE_S, bool(dur) and float(dur) - last >= WINDOW_EDGE_PAUSE_S)


def _layout(source):
    lay = source.get("content_type") or "auto"
    if lay not in VALID_LAYOUTS:
        _log("warn", "select", f"{source['source_id']}: unknown layout {lay!r}; using auto",
             source_id=source["source_id"])
        lay = "auto"
    return lay


# ---- SELECT: candidates ----
def _sentence_cand(clip, words, sents, pick=None, vc=None):
    p, vc = pick or {}, vc or {}
    return {**clip, "words": words, "base_words": words, "last": sents[clip["s1"]]["text"],
            "llm_score": p.get("score"),
            "sub": p.get("sub") or {}, "hook": p.get("hook") or None,
            "backend": (p.get("backend") or vc.get("semantic_backend")) if pick else "energy",
            "model": (p.get("model") or vc.get("semantic_model")) if pick else None}


def _score(c, z, vc):
    """w_llm*llm/100 + w_energy*sigmoid(mean_z) (no LLM score: sigmoid(mean_z)), minus 0.1
    when a sentence clip does not end cleanly."""
    c["mean_z"] = mean_z(z, c["start"], c["end"])
    e = sigmoid(c["mean_z"])
    if c.get("llm_score") is None:
        s = e
    else:
        s = float(vc.get("w_llm", 0.8)) * float(c["llm_score"]) / 100.0 + float(vc.get("w_energy", 0.2)) * e
    if c["boundary_type"] == "sentence":
        c["ends_cleanly"] = boundaries.ends_cleanly(c["last"])
        s -= 0.0 if c["ends_cleanly"] else 0.1
    c["score"] = s
    return c


def _norm_text(t):
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", str(t).lower()).split())


def _partial_ratio(a, b):
    """rapidfuzz fuzz.partial_ratio; difflib equivalent when rapidfuzz is missing (/usr/bin/python3)."""
    try:
        from rapidfuzz import fuzz
        return float(fuzz.partial_ratio(a, b))
    except ImportError:
        import difflib
        s, l = sorted((a, b), key=len)
        if not s:
            return 0.0
        best = 0.0
        for blk in difflib.SequenceMatcher(None, s, l, autojunk=False).get_matching_blocks():
            st = max(0, blk.b - blk.a)
            best = max(best, difflib.SequenceMatcher(None, s, l[st:st + len(s)], autojunk=False).ratio())
        return 100.0 * best


LOCATE_PAD_S = 0.25         # base.en vs small.en word times drift by a few hundred ms
LOCATE_MIN_TEXT = 50.0      # partial_ratio floor: the refined sentence must at least resemble it


def _locate(rs, sent, edge):
    """Index in refined sentences `rs` of base sentence `sent`, by TIME: the refined sentence whose
    span (padded by LOCATE_PAD_S) best overlaps the base span, overlap measured against the shorter
    of the two so a merged or split sentence still scores ~1. Ties (0.1 buckets) go to the one whose
    `edge` ("start"/"end") is nearest (0.5 s buckets), then to the higher difflib ratio on normalized
    text. Words are normalized (lowercase, no punctuation) so "all-in!" == "all in", and a refined
    sentence whose text shares nothing with the base one (partial_ratio < LOCATE_MIN_TEXT: a garbage
    transcript) is not a match. None when no refined sentence overlaps."""
    import difflib
    b0, b1 = float(sent["start"]), float(sent["end"])
    target, best = _norm_text(sent["text"]), None
    for k, r in enumerate(rs):
        r0, r1 = float(r["start"]) - LOCATE_PAD_S, float(r["end"]) + LOCATE_PAD_S
        ov = min(b1, r1) - max(b0, r0)
        if ov <= 0:
            continue
        frac = ov / max(0.05, min(b1 - b0, r1 - r0))
        text = _norm_text(r["text"])
        if _partial_ratio(target, text) < LOCATE_MIN_TEXT:
            continue
        ratio = difflib.SequenceMatcher(None, target, text, autojunk=False).ratio()
        key = (-round(min(frac, 1.0), 1), round(abs(float(r[edge]) - float(sent[edge])) * 2) / 2, -ratio)
        if best is None or key < best[0]:
            best = (key, k)
    return best[1] if best else None


def _refine(media, duration, cand, sents, vc, sid=None, window=None):
    """Precision pass: re-transcribe [start-8, end+8] with REFINE_MODEL, find the clip's first and
    last sentence there (by time overlap, _locate) and recut on frame RMS (+ scene cuts), snapping to
    clean sentence boundaries of the refined transcript. Returns (clip, words, sentences) on the
    refined transcript; raises on any failure (the caller keeps the base cuts). `window` (the
    window's strong pass): its words over the span are the precise words,
    no new pass."""
    lo, hi = max(0.0, cand["start"] - REFINE_PAD_S), min(duration, cand["end"] + REFINE_PAD_S)
    if window is not None:
        rw = [dict(w) for w in window if lo <= _mid(w) <= hi]
    else:
        rw = transcript.words_window(media, lo, hi, model=REFINE_MODEL)
    rs = transcript.sentences(rw, gap_s=SENTENCE_GAP)
    k0, k1 = _locate(rs, sents[cand["s0"]], "start"), _locate(rs, sents[cand["s1"]], "end")
    if k0 is None or k1 is None or k0 > k1:
        raise RuntimeError("clip sentences not found in refined transcript")
    try:
        rms = boundaries.frame_rms(transcript.load_audio(media, lo, hi), SR, 0.01)
    except Exception as e:
        rms = None
        _log("warn", "select", f"{sid}: no audio for {lo:.1f}-{hi:.1f}s ({e}); refined words, pad cuts",
             source_id=sid)
    try:
        scene_times = scenes.scene_cuts(media, lo, hi)
    except Exception:
        scene_times = []
    # the window's first/last sentence may be a fragment: edges_clean=False
    c = boundaries.clean_clip(rw, rs, k0, k1, rms=rms, t0=lo, cfg=vc, scene_times=scene_times,
                              edges_clean=False, trim_long=True)
    if c is None:
        raise RuntimeError("refined clip out of duration band")
    return c, rw, rs


def _snap(media, duration, c, vc, sid):
    """clip-quality 10-21 stage 1: the cuts end on the next clean sentence end (<= 8 s later),
    open on a clean sentence start, and sit in a word gap of the clip's own (precise when refined)
    transcript, after edge-verify moved them on its short transcripts (boundaries.snap_clip).
    The window's pad (disk_safe_harvest.WINDOW_PAD_S) is what the end extends into. Never
    raises; c["snap"] records what moved."""
    words = c.get("words") or []
    try:
        lo = max(0.0, float(c["start"]) - boundaries.SNAP_MAX_S - 1.0)
        hi = min(float(duration or 1e9), float(c["end"]) + boundaries.SNAP_MAX_S + 1.0)
        try:
            rms = boundaries.frame_rms(transcript.load_audio(media, lo, hi), SR, 0.01)
        except Exception:                 # noqa: BLE001 - pad cuts without audio
            rms, lo = None, 0.0
        # the precision words cover [start - 8, end + 8] only: their first / last sentence is
        # a fragment of that span, never a window edge
        edges = (False, False) if c.get("refined") else window_edges(sid, words, duration)
        r = boundaries.snap_clip(words, c["start"], c["end"], cfg=vc, edges_clean=edges,
                                 media_dur=duration or None, rms=rms, t0=lo)
    except Exception as e:                # noqa: BLE001 - the verified cuts stand
        c["snap"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        _log("warn", "select", f"{sid}: snap raised ({e}); verified cuts kept", source_id=sid)
        return
    c["snap"] = r["snap"]
    if r["start"] != c["start"] or r["end"] != c["end"]:
        _log("info", "select", f"{sid}: snap moved {c['start']:.2f}-{c['end']:.2f}s -> "
             f"{r['start']:.2f}-{r['end']:.2f}s ({r['snap']})", source_id=sid)
        c["start"], c["end"] = r["start"], r["end"]
    for side in ("start", "end"):
        if r["snap"][side] == "none":
            c[f"clean_{side}"] = False


def _low_speech(media, duration, words, env, z, vc, sid):
    """Gameplay without commentary: cuts from the energy envelope around each energy peak."""
    median = float(np.median(env)) if len(env) else 0.0
    mn, mx = float(vc.get("min_dur_s", 15)), float(vc.get("max_dur_s", 75))
    out = []
    for peak in energy_peaks(z, 2 * TOP_N, SKIP_HEAD, SKIP_TAIL):
        c = None
        try:
            lo, hi = max(0.0, peak - 25), min(duration, peak + mx + 25)
            fenv = boundaries.frame_rms(transcript.load_audio(media, lo, hi), SR, 0.1)
            try:
                st = scenes.scene_cuts(media, lo, hi)
            except Exception:
                st = []
            c = boundaries.energy_clip(fenv, lo, 0.1, peak, median, mn, mx, scene_times=st)
        except Exception as e:
            _log("warn", "select", f"{sid}: fine envelope failed at {peak:.0f}s ({e}); 1s envelope",
                 source_id=sid)
        c = c or boundaries.energy_clip(env, 0.0, 1.0, peak, median, mn, mx)
        if c:
            c.update(words=words, llm_score=None, sub={}, hook=None, ends_cleanly=None,
                     backend="energy", model=None)
            out.append(_score(c, z, vc))
    return boundaries.suppress(out, OVERLAP_MAX, TOP_N)


def _persist(sid, layout, c, vertical, density, ref_words=None):
    """Words sidecar (words whose midpoint is inside the clip) + the clip row. With
    `ref_words` (the base pass under the precision words) caption phrases the two transcripts do
    not share are crosstalk: kept for the cuts, flagged "cap": false so they are not burned."""
    inside = [{"w": w["w"], "start": w["start"], "end": w["end"]} for w in c.get("words") or []
              if c["start"] <= (w["start"] + w["end"]) / 2.0 <= c["end"]]
    crosstalk_rep = None
    if ref_words is not None and inside:
        try:
            inside, crosstalk_rep = crosstalk.flag_crosstalk(inside, ref_words)
        except Exception as e:      # noqa: BLE001 - an unfiltered sidecar is the old behaviour
            _log("warn", "select", f"{sid}: crosstalk filter failed ({e})", source_id=sid)
    wp = SCRATCH / "words" / f"{sid}_{c['start']:.2f}.json"
    wp.parent.mkdir(parents=True, exist_ok=True)
    wp.write_text(json.dumps(inside))
    feats = {"llm_score": c.get("llm_score"), "sub": c.get("sub") or {}, "mean_z": round(c["mean_z"], 4),
             "duration": round(c["end"] - c["start"], 3), "boundary_type": c["boundary_type"],
             "ends_cleanly": c.get("ends_cleanly"), "speech_density": round(density, 4),
             "vertical": vertical, "backend": c.get("backend"), "model": c.get("model"),
             "clean_start": c.get("clean_start"), "clean_end": c.get("clean_end"),
             "clean_note": c.get("clean_note"), "refined": c.get("refined")}
    if c.get("refine_error"):
        feats["refine_error"] = c["refine_error"]
    if c.get("verified_start") is not None:
        feats["verified_start"] = c["verified_start"]
    if c.get("verified_end") is not None:
        feats["verified_end"] = c["verified_end"]
    if c.get("verify_action"):
        feats["verify_action"] = c["verify_action"]
    if c.get("snap"):
        feats["snap"] = c["snap"]
    if c.get("source_frame_gate"):
        feats["source_frame_gate"] = c["source_frame_gate"]
    if c.get("crop_probe"):
        feats["crop_probe"] = c["crop_probe"]
    if crosstalk_rep and crosstalk_rep.get("dropped"):
        feats["crosstalk_dropped"] = crosstalk_rep["dropped"]
    alts = []
    for k, a in enumerate(c.get("alternates") or []):
        aw = [{"w": w["w"], "start": w["start"], "end": w["end"]} for w in c.get("base_words") or []
              if a["start"] <= (w["start"] + w["end"]) / 2.0 <= a["end"]]
        if not aw:
            continue
        ap = SCRATCH / "words" / f"{sid}_{c['start']:.2f}_alt{k + 1}.json"
        ap.write_text(json.dumps(aw))
        alts.append({**a, "words_path": str(ap)})
    if alts:
        feats["alternates"] = alts
    meta = c.get("topic_meta") or {}
    if meta.get("topics"):
        feats["topics"] = meta["topics"]
    if c.get("topic"):
        t = c["topic"]
        feats["topic"] = {k: t.get(k) for k in ("i", "lo", "hi", "title", "continues")}
    if c.get("topic_fit"):
        feats["topic_fit"] = c["topic_fit"]
    if meta.get("pack"):
        feats["prompt_pack"] = meta["pack"]
    if c.get("wrapup_dropped"):
        feats["end_rule"] = {"dropped": [str(x)[:120] for x in c["wrapup_dropped"]][:4]}
    return runstate.add_clip(sid, c["start"], c["end"], round(c["score"], 4), layout, hook=c.get("hook"),
                          words_path=str(wp), features_json=feats, boundary_type=c["boundary_type"])


def _speaker_crop_layout(layout, vertical):
    """True when the clip renders through a face-following crop (``auto`` resolves through
    render.vertical_layouts, as recompose does). Its frame gate fails every no-face frame, so a
    source window without a face on screen can't pass it. Stream verticals/layouts run the
    lean gate and never qualify."""
    if (vertical or "").lower() in _STREAM_VERTICALS or layout in _STREAM_LAYOUTS:
        return False
    if layout == "auto":
        layout = (CFG.get("render", {}).get("vertical_layouts") or {}).get(vertical, "auto")
    return layout in _SPEAKER_CROP_LAYOUTS


def _drop_faceless_picks(media, cands, sid):
    """(pick-quality rule) skip speaker-crop picks whose SOURCE window is already
    under the frame-gate bar (screencast, b-roll: no face on screen) before they cost a render,
    so the next candidate can take the slot. Fails open when the source can't be measured."""
    bar = quality_gate._layout_defaults()
    kept = []
    for c in cands:
        pred = pick_quality.predict_frame_gate(media, c["start"], c["end"], fps=PICK_QUALITY_FPS,
                                               fail_pct_max=bar["fail_pct_max"],
                                               broll_min_span_s=bar["broll_min_span_s"])
        c["source_frame_gate"] = {k: pred.get(k) for k in ("passed", "fail_pct", "frac")}
        if pred.get("passed") is False:
            _log("warn", "select", f"{sid}: skipped {c['start']:.1f}-{c['end']:.1f}s before render: "
                 f"no face on {pred['fail_pct']:.1f}% of source samples (frame-gate cap "
                 f"{bar['fail_pct_max']:.1f}%)", source_id=sid)
            continue
        kept.append(c)
    return kept


# ---- SELECT: crop feasibility before the window costs a select and a render ----
_CROP = CFG.get("render", {}).get("crop_feasibility") or {}
CROP_PROBE = bool(_CROP.get("enabled", True))
UNCROPPABLE = "uncroppable"          # the source's error prefix when no clean crop exists
_STACK_NICHES = ("gaming", "streamers")   # = recompose.STACK_NICHES: a dual view zoom counts


def _cropcheck():
    from clipengine.render import cropcheck
    return cropcheck


def crop_probe(media, sid, niche):
    """render/cropcheck.probe of a window for its niche (a dual view zoom counts in the
    stacking niches), or None: no niche (campaign / manual sources keep today's flow), the probe
    switched off, or the window unreadable (no verdict: the render decides as before)."""
    if not CROP_PROBE or not niche:
        return None
    try:
        return _cropcheck().probe(media, 0.0, None, niche=niche, stack=niche in _STACK_NICHES)
    except Exception as e:      # noqa: BLE001 - a probe failure never costs a window
        _log("warn", "select", f"{sid}: crop probe failed ({type(e).__name__}: {e}); no verdict",
             source_id=sid)
        return None


def _drop_uncroppable_picks(cands, probe, sid):
    """candidates whose span the probe found no clean crop for on enough of its face
    frames (< cropcheck.SPAN_MIN_OK, the 1309 class: a cover graphic sliced for 4 s) leave before
    their render, so the next candidate can take the slot. Spans with too few face frames keep
    today's flow."""
    cc = _cropcheck()
    verdicts = [tuple(v) for v in (probe or {}).get("verdicts") or []]
    if not verdicts:
        return cands
    kept = []
    for c in cands:
        share, n = cc.share_ok(verdicts, c["start"], c["end"])
        c["crop_probe"] = {"ok_share": share, "frames": n}
        if share is not None and n >= cc.MIN_VERDICT_FRAMES and share < cc.SPAN_MIN_OK:
            _log("warn", "select", f"{sid}: skipped {c['start']:.1f}-{c['end']:.1f}s before render: "
                 f"no clean crop for the speaker on {n - round(share * n)} of {n} sampled face "
                 f"frames (span bar {cc.SPAN_MIN_OK:.0%})", source_id=sid)
            continue
        kept.append(c)
    return kept


def _mark_uncroppable(sid, why):
    """The window leaves select with no clip because no clean crop exists: its child source
    carries the reason (a window runner can try the next window of the same source)."""
    runstate.update_source(sid, state="failed", error=f"{UNCROPPABLE}: {why}"[:400])


# ---- SELECT: one strong, aligned transcript per window ----
# Our in-run eye-check failed 62 of 248 gate-passed clips on cut_start (11-04..11), against 1 of 12
# of Klap's spans, which open on an aligned large-model sentence start; ours open on select's
# base.en / small.en sentences ('That's a little bit / of a tongue': 3026 opened on 'a little bit
# of'). Every comparable tool picks and cuts on one strong pass. The window is heard
# once by the captions' strong model with the source's title as its prompt (vocab_prompt), its
# words seated on the window audio (boundaries.seat_onsets, the alignment); select's sentences,
# the LLM's transcript and the precision cuts read it (no small.en precision pass), and the caption
# step slices the clip's words from it (no per-clip strong pass). Off, no time or any error:
# select's base pass and the per-clip passes as before.
WINDOW_TRANSCRIPT = {"enabled": False, "model": "large-v3-turbo", "prompt": True, "seat": True}
SELECT_LLM_RESERVE_S = 60.0      # select time kept after the window pass for the LLM and the cuts


def window_transcript_settings(cfg=None):
    """select.window_transcript over WINDOW_TRANSCRIPT; env WINDOW_TRANSCRIPT=0 turns it off."""
    st = {**WINDOW_TRANSCRIPT, **((((cfg if cfg is not None else CFG).get("select") or {})
                                   .get("window_transcript")) or {})}
    st.pop("_doc", None)
    if os.environ.get("WINDOW_TRANSCRIPT", "").strip() == "0":
        st["enabled"] = False
    return st


def _window_pass_key(media, source, st):
    """(model, prompt, cache file) of the window pass."""
    import hashlib
    prompt = vocab_prompt(source) if st.get("prompt") else None
    key = hashlib.md5(f"{media}|{os.path.getmtime(media)}|{st['model']}|{prompt or ''}|"
                      f"seat={bool(st.get('seat'))}".encode()).hexdigest()
    return st["model"], prompt, Path(transcript.SCRATCH) / "transcripts" / f"{key}.window.json"


def cached_window_pass(media, source, st=None):
    """The window pass's words from its cache ([{w, start, end, p}], seated), or None (off, not
    heard, unreadable). Never transcribes."""
    st = st or window_transcript_settings()
    if not st.get("enabled") or not media:
        return None
    try:
        _m, _p, f = _window_pass_key(media, source, st)
        return json.loads(f.read_text())
    except (OSError, ValueError, TypeError):
        return None


def window_pass(media, source, strong_fn=None, rms_fn=None, dur=None):
    """(words or None, info): the window heard once by select.window_transcript's model
    (repair.transcribe_strong: word probabilities kept, the window's warm whisper worker and its
    heavy slot), the source's title as its prompt, its words seated on the window audio (the
    alignment: boundaries.seat_onsets), cached under the transcript cache. None when off, when
    select's time does not fit it (SELECT_DEADLINE less SELECT_LLM_RESERVE_S), or on any error:
    select then reads its base pass. Never raises."""
    st = window_transcript_settings()
    if not st.get("enabled"):
        return None, {"mode": "off"}
    t0 = time.monotonic()
    try:
        model, prompt, f = _window_pass_key(media, source, st)
    except (OSError, TypeError) as e:
        return None, {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    info = {"model": model, "prompt": prompt}
    got = cached_window_pass(media, source, st)
    if got is not None:
        return got, {**info, "cached": True}
    dur = dur or float((source or {}).get("duration_s") or 0) or transcript._media_seconds(media)
    if not dur:
        return None, {**info, "error": "window length unknown"}
    timeout = None
    if SELECT_DEADLINE is not None:
        left = SELECT_DEADLINE - time.monotonic() - SELECT_LLM_RESERVE_S
        need = _strong_cost(dur)
        if left < need:
            return None, {**info, "skipped": f"{left:.0f}s of select's time left, ~{need:.0f}s "
                                             f"needed"}
        timeout = left
    try:
        if strong_fn is not None:
            words = strong_fn(media, 0.0, dur, timeout, prompt)
        else:
            words = strong_asr.transcribe_strong(media, 0.0, dur, model=model, timeout=timeout,
                                                 prompt=prompt)
            p = transcript._WORKER.get("p")
            if transcript.WORKER and p is not None:
                _STRONG["pid"] = p.pid      # the caption step's cost estimate: the model is warm
        moved = []
        if st.get("seat"):
            words, moved = _seat_heard(media, 0.0, dur, words, rms_fn=rms_fn)
    except Exception as e:                # noqa: BLE001 - select's base pass
        return None, {**info, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    words = sorted(words, key=lambda w: float(w["start"]))
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(words))
        tmp.replace(f)
    except OSError:
        pass
    return words, {**info, "seated": len(moved), "seconds": round(time.monotonic() - t0, 1)}


# ---- SELECT: per-step timings ----
SELECT_STEP_HOOK = None      # callable(step): reports the select step in flight (a caller's time cap)
SELECT_DEADLINE = None       # time.monotonic() after which precision/edge-verify are skipped (base cuts)
_STEP_ORDER = ("crop", "transcribe", "window_pass", "energy", "low_speech", "outline", "llm",
               "precision", "edge_verify", "snap", "faceless", "persist")


class _StepTimer:
    """Wall seconds per select() sub-step. Ends as one `[select] <sid>: timings ...` line plus an
    events row, so a slow window shows which step ate the time (and which LLM backend answered)."""

    def __init__(self):
        self.t, self.n, self.t0 = {}, {}, time.monotonic()

    @contextlib.contextmanager
    def step(self, name):
        if SELECT_STEP_HOOK is not None:
            SELECT_STEP_HOOK(f"select:{name}")
        t = time.monotonic()
        try:
            yield
        finally:
            self.t[name] = self.t.get(name, 0.0) + time.monotonic() - t
            self.n[name] = self.n.get(name, 0) + 1

    def line(self, llm_calls=None):
        parts = []
        for k in _STEP_ORDER:
            if k in self.t:
                parts.append(f"{k}={self.t[k]:.1f}s" + (f"/{self.n[k]}" if self.n[k] > 1 else ""))
        if llm_calls:
            parts.append("llm_via=" + ",".join(
                f"{c.get('step', '') and c['step'] + ':'}{c.get('backend')}/{c.get('model')}="
                f"{c.get('outcome', '?')}@{c.get('s', 0) or 0:.1f}s" for c in llm_calls))
        parts.append(f"total={time.monotonic() - self.t0:.1f}s")
        return " ".join(parts)


def _log_timings(sid, tm, llm_calls):
    try:
        _log("info", "select", f"{sid}: timings {tm.line(llm_calls)}", source_id=sid)
    except Exception as e:          # timing is diagnostics only; never fail a select over it
        print(f"[select] {sid}: timings unavailable ({e})")


# ---- SELECT: topic map first ----
TOPIC_NEAR_S = 3.0          # a chat peak / mention in no topic joins the nearest one this close


def _in_topic(sid, words, sents, s0, s1, topics, cfg, edges, near_s=0.0, why="pick"):
    """boundaries.clean_clip of s0..s1 inside the one topic it overlaps most (topic_map.fit), the
    topic's span kept for every later move; without a map, clean_clip as before. (clip or None,
    topic or None, note)."""
    if not topics:
        return boundaries.clean_clip(words, sents, s0, s1, cfg=cfg, edges_clean=edges), None, None
    t, a, b, note = topic_map.fit(s0, s1, sents, topics, near_s=near_s)
    if t is None:
        _log("info", "select", f"{sid}: {why} S{s0}-S{s1} ({float(sents[s0]['start']):.1f}-"
             f"{float(sents[s1]['end']):.1f}s) is in no topic of the map: dropped", source_id=sid)
        return None, None, note
    if note:
        _log("info", "select", f"{sid}: {why} S{s0}-S{s1} kept inside topic T{t['i']} "
             f"{json.dumps(t.get('title') or '')} ({note})", source_id=sid)
    c = boundaries.clean_clip(words, sents, a, b, cfg={**cfg, **topic_map.span_cfg(t)},
                              edges_clean=edges)
    if c is None and note:
        _log("info", "select", f"{sid}: {why} S{s0}-S{s1} does not fit inside topic T{t['i']}: "
             f"dropped", source_id=sid)
    return c, t, note


def _cand_cfg(c, cfg):
    """A candidate's boundaries cfg: its topic's span on top of select's."""
    return {**cfg, **topic_map.span_cfg(c.get("topic"))}


# ---- SELECT: the next candidate for a pick ----
MAX_ALTERNATES = 2          # alternates kept per clip, and tried per render


def _alternates(sid, words, sents, pick, topics, cfg, edges):
    """The picks select_semantic dropped for overlapping `pick` (its "alternates"), cut the way
    the pick was (clean_clip inside its topic), best score first: [{"start", "end", "hook",
    "score", "s0", "s1"}], at most MAX_ALTERNATES. The render takes the next one when the clip
    has no clean cut (render: "A window that can't be fixed is replaced by the next candidate,
    not shipped")."""
    out = []
    for a in pick.get("alternates") or []:
        try:
            c, _t, _n = _in_topic(sid, words, sents, a["s0"], a["s1"], topics, cfg, edges,
                                  why="alternate")
        except Exception:                 # noqa: BLE001 - no alternate then
            c = None
        if not c:
            continue
        if any(abs(c["start"] - o["start"]) < 0.05 and abs(c["end"] - o["end"]) < 0.05 for o in out):
            continue
        out.append({"start": round(float(c["start"]), 3), "end": round(float(c["end"]), 3),
                    "hook": a.get("hook") or None, "score": a.get("score"), "s0": c.get("s0"),
                    "s1": c.get("s1")})
        if len(out) >= MAX_ALTERNATES:
            break
    return out


# ---- SELECT: one flow for every vertical ----
def select(source):
    sid = source["source_id"]
    tm = _StepTimer()
    llm_calls = []
    try:
        media = source["local_path"]
        layout = _layout(source)
        vertical = source.get("vertical") or "general"
        vc = vertical_cfg(vertical)
        niche, probe = (_niche_of(sid) if CROP_PROBE else None), None
        pack_niche = niche if CROP_PROBE else _niche_of(sid)
        pack = topic_map.pack(pack_niche)
        # the end rule and the niche's wrap-up openers for every select cut
        svc = {**vc, **topic_map.select_cfg(pack_niche)}
        if niche:
            with tm.step("crop"):
                probe = crop_probe(media, sid, niche)
        if probe and probe.get("verdict") == UNCROPPABLE:
            # no clean crop for the speaker on most of the window: no transcribe, no
            # LLM, no render
            _mark_uncroppable(sid, probe["why"])
            _log("warn", "select", f"{sid}: UNCROPPABLE, window skipped before select: "
                 f"{probe['why']} (probe {probe.get('elapsed_s', 0):.1f}s)", source_id=sid)
            _log_timings(sid, tm, llm_calls)
            return False
        try:
            with tm.step("transcribe"):
                words = transcript.get_words(media, model=TRANSCRIBE_MODEL)
        except Exception as e:
            _log("warn", "select", f"{sid} transcription failed ({e}); treating as low-speech", source_id=sid)
            words = []
        # one strong, aligned pass of the window for the sentences, the
        # LLM's transcript and the cuts; select's base pass stays the context the cut review,
        # the final cut and the eye-check read (and the mention check above)
        strong, wp_info = None, {"mode": "off"}
        if window_transcript_settings().get("enabled"):
            with tm.step("window_pass"):
                strong, wp_info = window_pass(media, source)
        if strong:
            _log("info", "select", f"{sid}: window pass {wp_info.get('model')} "
                 f"({len(strong)} words, {wp_info.get('seated', 0)} seated"
                 f"{', cached' if wp_info.get('cached') else ''}) for the sentences and the cuts",
                 source_id=sid)
            words = strong
        elif wp_info.get("mode") != "off":
            why = wp_info.get("skipped") or wp_info.get("error")
            _log("info", "select", f"{sid}: no window pass ({why}); {TRANSCRIBE_MODEL} sentences",
                 source_id=sid)
        sents = transcript.sentences(words, gap_s=SENTENCE_GAP)
        try:
            with tm.step("energy"):
                env = energy_envelope(media)
        except Exception as e:
            _log("warn", "select", f"{sid} energy envelope failed ({e}); flat energy", source_id=sid)
            env = np.zeros(int(source.get("duration_s") or 0))
        dur = float(source.get("duration_s") or 0) or float(len(env))
        z = zscores(env)
        density = sum(w["end"] - w["start"] for w in words) / dur if dur > 0 else 0.0

        # Pre-sliced short windows (from a window harvester) don't have a cold-open
        # or outro — the content was already selected by the transcript score. Scale the
        # head/tail skips down proportionally so a 30-90 s window actually has eligible
        # sentences. Threshold: `dur < SKIP_HEAD + SKIP_TAIL + 30` means we're on a sliced
        # window, not a full VOD.
        eff_skip_head = SKIP_HEAD
        eff_skip_tail = SKIP_TAIL
        if dur > 0 and dur < SKIP_HEAD + SKIP_TAIL + 30:
            eff_skip_head = min(SKIP_HEAD, max(0.5, dur * 0.05))
            eff_skip_tail = min(SKIP_TAIL, max(0.5, dur * 0.05))
            if _WINDOW_SID.search(sid or ""):
                eff_skip_head = max(eff_skip_head, WINDOW_CONTEXT_S)
                eff_skip_tail = max(eff_skip_tail, WINDOW_CONTEXT_S)
            _log("info", "select", f"{sid}: short source ({dur:.1f}s < 120s) — scaling "
                 f"skip_head={eff_skip_head:.1f}s skip_tail={eff_skip_tail:.1f}s",
                 source_id=sid)

        if len(words) < 50 or density < MIN_SPEECH_DENSITY:
            with tm.step("low_speech"):
                clips = _low_speech(media, dur, words, env, z, vc, sid)
            path = "low-speech"
        else:
            for s in sents:
                s["z"] = mean_z(z, s["start"], s["end"])
            eligible = [s for s in sents if s["start"] >= eff_skip_head and s["end"] <= dur - eff_skip_tail]
            # topic map first, one outline call per source (cached per source)
            tmap, ocall = None, {"step": "outline", "outcome": "off"}
            if topic_map.settings().get("enabled"):
                with tm.step("outline"):
                    tmap, ocall = topic_map.for_source(sid, sents, pack_niche, vertical, svc,
                                                       cache_dir=SCRATCH / "topic_maps",
                                                       deadline=SELECT_DEADLINE)
            topics = (tmap or {}).get("topics") or None
            if topics:
                _log("info", "select", f"{sid}: topic map {len(topics)} topic(s) "
                     f"({ocall.get('outcome')}): " + "; ".join(
                         f"T{t['i']} S{t['s0']}-S{t['s1']} {t['lo']:.1f}-{t['hi']:.1f}s "
                         f"{json.dumps(t.get('title') or '')}" for t in topics), source_id=sid)
            elif ocall.get("outcome") not in ("off",):
                _log("info", "select", f"{sid}: no topic map ({ocall.get('outcome')}: "
                     f"{ocall.get('why') or ''}); picks as before", source_id=sid)
            try:
                with tm.step("llm"):
                    picks = select_semantic.propose(eligible, vertical, {
                        **svc, "top_n": TOP_N,
                        "rubric": ((pack or {}).get("rubric") or None)
                        and pack["rubric"] + topic_map.hook_examples(pack),
                        "topics": topics})
            except Exception as e:
                _log("warn", "select", f"{sid} semantic proposer raised ({e})", source_id=sid)
                picks = None
            llm_calls = ([ocall] if ocall.get("outcome") != "off" else []) + \
                [dict(c) for c in getattr(select_semantic, "LAST_CALLS", [])]
            path, cands = "semantic", []
            edges = window_edges(sid, words, dur)
            for p in picks or []:        # s0/s1 are sentence "i" == list position in `sents`
                c, t, note = _in_topic(sid, words, sents, p["s0"], p["s1"], topics, svc, edges)
                if c:
                    cands.append({**_sentence_cand(c, words, sents, p, vc), "topic": t,
                                  "topic_fit": note,
                                  "alternates": _alternates(sid, words, sents, p, topics, svc,
                                                            edges)})
            if not cands and picks == []:
                # the LLM answered and picked nothing ({"clips": []}). That is
                # its verdict on this transcript; an energy peak would publish exactly what it
                # rejected. Only an unreachable/garbage backend (picks None) falls back.
                path = "llm-declined"
                _log("info", "select", f"{sid}: LLM answered with no clip-worthy moment in "
                     f"{len(eligible)} eligible sentences; no energy fallback", source_id=sid)
            elif not cands:
                path = "fallback"
                why = ("no usable LLM picks" if picks else
                       "LLM unavailable" if eligible else "no eligible sentences for the LLM")
                _log("warn", "select", f"{sid}: {why}; energy-peak fallback", source_id=sid)
                for peak in energy_peaks(z, 2 * TOP_N, eff_skip_head, eff_skip_tail):
                    span = boundaries.sentences_around(sents, peak)
                    c, t, note = _in_topic(sid, words, sents, span[0], span[1], topics, svc, edges,
                                           near_s=TOPIC_NEAR_S, why="energy peak") \
                        if span else (None, None, None)
                    if c:
                        cands.append({**_sentence_cand(c, words, sents), "topic": t,
                                      "topic_fit": note})
            # cheap suppress first: the precision pass (small.en + audio) only runs on survivors
            cands = boundaries.suppress([_score(c, z, vc) for c in cands], OVERLAP_MAX, TOP_N)
            verify_fn = edge_verify.make_words_fn(media, REFINE_MODEL)
            late = 0
            for c in cands:
                if SELECT_DEADLINE is not None and time.monotonic() > SELECT_DEADLINE:
                    # the window's time cap is near: keep the base cuts (the same fallback as a
                    # failed precision pass) so the clip still reaches render and the gate
                    late += 1
                    c.update(refined=False, refine_error="skipped: window time cap near")
                    c["verify_action"] = {"start": "skipped", "end": "skipped"}
                    _score(c, z, vc)
                    continue
                try:
                    with tm.step("precision"):
                        r, rw, rs = _refine(media, dur, c, sents, _cand_cfg(c, svc), sid,
                                            window=strong)
                    lost = [k for k in ("clean_start", "clean_end") if c.get(k) and not r.get(k)]
                    if lost:
                        raise RuntimeError(f"refined cut loses {'/'.join(lost)} ({r.get('clean_note')})")
                    c.update(start=r["start"], end=r["end"], words=rw, last=rs[r["s1"]]["text"],
                             clean_start=r["clean_start"], clean_end=r["clean_end"],
                             clean_note=r.get("clean_note"), refined=True)
                    if r.get("wrapup_dropped"):
                        c["wrapup_dropped"] = (c.get("wrapup_dropped") or []) + r["wrapup_dropped"]
                except Exception as e:
                    c.update(refined=False, refine_error=str(e)[:200])
                    _log("warn", "select", f"{sid}: precision pass failed at {c['start']:.1f}s ({e}); "
                         "keeping base cuts", source_id=sid)
                try:
                    with tm.step("edge_verify"):
                        v = edge_verify.verify({"start": c["start"], "end": c["end"]}, verify_fn,
                                               _cand_cfg(c, svc))
                    c["verified_start"] = v["start"]
                    c["verified_end"] = v["end"]
                    c["verify_action"] = v["action"]
                    if v["start"] != c["start"] or v["end"] != c["end"]:
                        _log("info", "select",
                             f"{sid}: edge-verify moved {c['start']:.2f}-{c['end']:.2f}s -> "
                             f"{v['start']:.2f}-{v['end']:.2f}s ({v['action']})",
                             source_id=sid)
                        c["start"], c["end"] = v["start"], v["end"]
                except Exception as e:
                    _log("warn", "select", f"{sid}: edge-verify raised ({e})", source_id=sid)
                    c["verify_action"] = {"start": "error", "end": "error"}
                with tm.step("snap"):
                    _snap(media, dur, c, _cand_cfg(c, svc), sid)
                _score(c, z, vc)
                for side in ("start", "end"):
                    if c.get(f"clean_{side}") is False:
                        _log("warn", "select", f"{sid}: clip {c['start']:.1f}-{c['end']:.1f}s has "
                             f"clean_{side}=false ({c.get('clean_note')})", source_id=sid)
            if late:
                _log("warn", "select", f"{sid}: window time cap near; precision pass + edge-verify "
                     f"skipped for {late} of {len(cands)} candidate(s), base cuts kept", source_id=sid)
            if _speaker_crop_layout(layout, vertical):
                with tm.step("faceless"):
                    cands = _drop_faceless_picks(media, cands, sid)
            if probe and cands:
                n_cands = len(cands)
                cands = _drop_uncroppable_picks(cands, probe, sid)
                if not cands:
                    _mark_uncroppable(sid, f"no clean crop on any of {n_cands} candidate span(s)")
            clips = boundaries.suppress(cands, OVERLAP_MAX, TOP_N)
        if probe and clips and path == "low-speech":
            kept = _drop_uncroppable_picks(clips, probe, sid)
            if not kept:
                _mark_uncroppable(sid, f"no clean crop on any of {len(clips)} candidate span(s)")
            clips = kept

        with tm.step("persist"):
            for c in clips:              # the map, the clip's topic, the pack, the end rule
                if path != "low-speech":
                    c["topic_meta"] = {"topics": topic_map.compact(topics), "pack": (pack or {}).get("niche")}
            # the window pass has no independent precise pass under it: its word probabilities
            # judge crosstalk at the caption step (strong_captions)
            added = sum(1 for c in clips if _persist(
                sid, layout, c, vertical, density,
                ref_words=words if c.get("refined") and c.get("words") is not words
                and not strong else None))
            if added or not str((runstate.source(sid) or {}).get("error") or "").startswith(UNCROPPABLE):
                runstate.update_source(sid, state="selected")
        _log_timings(sid, tm, llm_calls)
        durs = ", ".join(f"{c['end'] - c['start']:.1f}" for c in clips)
        _log("info", "select", f"{sid}: {added} clips via {path} path (vertical={vertical}, layout={layout}, "
             f"words={len(words)}, speech={density:.2f}) durations=[{durs}]s", source_id=sid)
        return True
    except Exception as e:
        _log_timings(sid, tm, llm_calls)
        runstate.update_source(sid, state="failed", error=str(e)[:400])
        _log("error", "select", f"{sid} FAILED: {e}", source_id=sid)
        return False


# ---- RENDER: recompose + captions -> stored clip ----
def _qc_line(stdout):
    """recompose's last `QC {json}` line (exit 3 = QC failed); {} when absent or unparseable."""
    for line in reversed((stdout or "").splitlines()):
        if line.startswith("QC "):
            try:
                return json.loads(line[3:])
            except ValueError:
                return {}
    return {}


def _load_words_sidecar(path):
    """Return the SOURCE-time words list from a sidecar; tolerates [...] or {"words":[...]}."""
    if not path or not Path(path).exists():
        return []
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return []
    return d.get("words", []) if isinstance(d, dict) else d


_STREAM_VERTICALS = {"stream", "gaming", "poker", "screencast"}
_STREAM_LAYOUTS = {"screencast", "hudband", "gamestack", "broadcast", "dual_view"}


def _measure_black_frozen(mp4_path, *, sample_fps=2.0, black_ratio_max=0.02,
                           frozen_ratio_max=0.05):
    """Fail-closed check: black-frame fraction + frozen-frame fraction from ffmpeg filters.
    `black_ratio_max` and `frozen_ratio_max` are fractions of the clip's duration that we
    tolerate; defaults are 2 % black / 5 % frozen.

    Returns `{"passed": bool, "black_total_s": float, "frozen_total_s": float,
              "dur_s": float, "errors": [...]}` — matches the quality_gate measure schema.
    """
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", str(mp4_path)],
            capture_output=True, text=True, timeout=30, check=True)
        dur = float(probe.stdout.strip() or 0.0)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return {"passed": None, "reason": "ffprobe duration unavailable"}
    if dur <= 0:
        return {"passed": None, "reason": "zero-duration clip"}
    # Run black+freeze detect in one ffmpeg pass.
    cmd = ["ffmpeg", "-nostdin", "-v", "info", "-i", str(mp4_path),
           "-vf", "blackdetect=d=0.5:pix_th=0.10,freezedetect=n=0.003:d=1",
           "-an", "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"passed": None, "reason": "ffmpeg timeout"}
    err = proc.stderr or ""
    import re as _re
    black_total = 0.0
    for m in _re.finditer(r"black_duration:(\d+(?:\.\d+)?)", err):
        black_total += float(m.group(1))
    frozen_total = 0.0
    for m in _re.finditer(r"freeze_duration:(\d+(?:\.\d+)?)", err):
        frozen_total += float(m.group(1))
    black_ratio = black_total / dur
    frozen_ratio = frozen_total / dur
    errors = []
    if black_ratio > black_ratio_max:
        errors.append(f"black_fraction={black_ratio:.2%} > {black_ratio_max:.0%}")
    if frozen_ratio > frozen_ratio_max:
        errors.append(f"frozen_fraction={frozen_ratio:.2%} > {frozen_ratio_max:.0%}")
    return {"passed": not errors, "errors": errors,
            "black_total_s": round(black_total, 2),
            "frozen_total_s": round(frozen_total, 2), "dur_s": round(dur, 2)}


def _measure_duration_bounds(start_s, end_s, *, min_s=5.0, max_s=90.0):
    try:
        dur = float(end_s) - float(start_s)
    except (TypeError, ValueError):
        return {"passed": None, "reason": "missing bounds"}
    if dur < min_s:
        return {"passed": False, "dur_s": round(dur, 2),
                "errors": [f"too short ({dur:.1f}s < {min_s}s)"]}
    if dur > max_s:
        return {"passed": False, "dur_s": round(dur, 2),
                "errors": [f"too long ({dur:.1f}s > {max_s}s)"]}
    return {"passed": True, "dur_s": round(dur, 2)}


def _measure_lufs_bounds(lufs_from_qc, *, lo=-18.0, hi=-10.0):
    """Trust render QC's integrated_lufs reading. Fail-closed when LUFS is missing OR
    outside [lo, hi]. Target under renderer: -14 ± 1."""
    if lufs_from_qc is None:
        return {"passed": None, "reason": "no QC LUFS"}
    try:
        v = float(lufs_from_qc)
    except (TypeError, ValueError):
        return {"passed": None, "reason": f"bad LUFS: {lufs_from_qc!r}"}
    if v < lo:
        return {"passed": False, "integrated_lufs": v,
                "errors": [f"{v:.1f} LUFS < {lo} (too quiet)"]}
    if v > hi:
        return {"passed": False, "integrated_lufs": v,
                "errors": [f"{v:.1f} LUFS > {hi} (too loud)"]}
    return {"passed": True, "integrated_lufs": v}


def _measure_caption_presence(words):
    """Caption presence: at least 20 % of the clip has transcribed words. For streams this
    doubles as a "legibility" proxy — a facecam clip with no transcribed speech is a bad
    posting candidate. Fail-closed when the sidecar is empty."""
    if not words:
        return {"passed": False, "errors": ["no transcribed words in sidecar"]}
    # Density = sum word durations / clip duration
    try:
        total = sum((float(w.get("end", 0)) - float(w.get("start", 0))) for w in words)
        if total < 1.0:
            return {"passed": False, "n_words": len(words),
                    "errors": [f"transcribed speech < 1 s total ({total:.2f}s)"]}
    except Exception:
        return {"passed": None, "reason": "malformed words sidecar"}
    return {"passed": True, "n_words": len(words), "spoken_s": round(total, 2)}


def _hook_judge():
    """the semantic judge (quality_gate.make_hook_ground_judge on select's
    claude_cli settings) when quality_gate.hook_judge.enabled, else None (lexical rule only)."""
    if not ((CFG.get("quality_gate") or {}).get("hook_judge") or {}).get("enabled"):
        return None
    return quality_gate.make_hook_ground_judge(CFG.get("select"))


def _run_stream_lean_gate(mp4_path, clip, qc, words, src_media=None):
    """Lean gate for stream/gaming/poker/screencast verticals.

 full gate's face/overlay checks don't fit facecam-gameplay content, but the
    engine still needs a real verdict. Measures that run:
      - duration_bounds: 5-90 s
      - audio_lufs: -18..-10 dB (from render QC's `integrated_lufs`)
      - black_frozen: < 2 % black, < 5 % frozen (ffmpeg blackdetect/freezedetect)
      - caption_presence: words sidecar non-empty, > 1 s total spoken
      - layout_sanity: trust render QC's `layout` matches the expected stream layout
      - gameplay_layout: a gameplay clip (gameplay_clip) rendered as a face-follow crop (track,
        split, ...) fails
      - visual_quality: median luma / contrast / sharpness + blur-fill share
      - (quality_gate.clip_checks, same as the full gate): flat edge band,
        graphics sliced by the crop, captions vs a second ASR, hook grounding (no hook =
        fail), the ending. 10-20: gaming 108 and streamers 109/110/113 passed this gate.

    Returns a report with the same shape as `quality_gate.gate` (verdict, drop_reason,
    per-measure results). A `None`/`passed=null` measure is informational and does not fail;
    an explicit `False` always fails.
    """
    report = {}
    report["duration_bounds"] = _measure_duration_bounds(clip.get("start_s"),
                                                          clip.get("end_s"))
    report["audio_lufs"] = _measure_lufs_bounds(qc.get("integrated_lufs"))
    report["caption_presence"] = _measure_caption_presence(words)
    try:
        report["black_frozen"] = _measure_black_frozen(mp4_path)
    except Exception as e:        # noqa: BLE001
        report["black_frozen"] = {"passed": None,
                                   "reason": f"{type(e).__name__}: {str(e)[:120]}"}
    # layout sanity: the render emitted a layout. Just record it (informational).
    report["layout"] = {"passed": None, "reason": "informational",
                        "value": qc.get("layout")}
    # gameplay never ships a face-follow crop (track round the cam)
    report["gameplay_layout"] = quality_gate.measure_gameplay_layout(
        gameplay_clip(clip), qc.get("layout"), (qc.get("banner_guard") or {}).get("action"))
    # brightness / contrast / sharpness / blur-fill on the pixels
    try:
        report["visual_quality"] = quality_gate.measure_visual_quality(
            mp4_path, blur_fill_frac=qc.get("blur_fill_frac"))
    except Exception as e:        # noqa: BLE001
        report["visual_quality"] = {"passed": None,
                                    "reason": f"{type(e).__name__}: {str(e)[:120]}"}
    report.update(quality_gate.clip_checks(
        mp4_path, words=words, start_s=clip.get("start_s"), end_s=clip.get("end_s"),
        hook=clip.get("hook"), src_media=src_media, crop_views=qc.get("crop_views"),
        hook_judge=_hook_judge(), layout=qc.get("layout"), panels=qc.get("panels")))
    # Collapse to a single verdict (a failed caption check is a warning: Amendment 1).
    reasons = []
    for k in ("duration_bounds", "audio_lufs", "caption_presence", "black_frozen",
              "gameplay_layout", "visual_quality") + quality_gate.CLIP_CHECK_KEYS:
        if k in NON_REJECTING_GATE_KEYS:
            continue
        m = report.get(k) or {}
        if m.get("passed") is False:
            errs = m.get("errors") or []
            reasons.append(f"{k}: {'; '.join(errs)}" if errs else k)
    report["verdict"] = "fail" if reasons else "pass"
    report["drop_reason"] = "; ".join(reasons)
    report["note"] = "stream lean gate"
    return _captions_never_reject(report)


# ---- GATE: captions never reject a clip (yield 10-22 Amendment 1) ----
# The gate still measures the captions against the second ASR (quality_gate is unchanged); the
# engine reads a failed caption check as a route to the caption repair (_recaption: a tie-break
# vote, the sidecar rewritten, the captions burned again), never as a reason to drop the clip.
# Captions that still disagree after it ship as the best vote with a `caption_unresolved` warning.
NON_REJECTING_GATE_KEYS = ("caption_asr",)
CAPTION_CODE = "captions_garbled"


def caption_warning(report):
    """The failed caption check's errors (a warning since Amendment 1), else None."""
    m = (report or {}).get("caption_asr") or {}
    if m.get("passed") is False:
        return "; ".join(str(e) for e in m.get("errors") or []) or "caption_asr"
    return None


def _captions_never_reject(report):
    """The gate report with its verdict on the non-caption checks only (Amendment 1); a failed
    caption check stays in the report and is named in report["caption_warning"]."""
    warn = caption_warning(report)
    if warn is None:
        return report
    report["caption_warning"] = warn
    if report.get("note") != "stream lean gate":      # the lean gate skipped it already
        verdict, reason = quality_gate.decide(
            {**report, "caption_asr": {**report["caption_asr"], "passed": None}})
        report["verdict"], report["drop_reason"] = verdict, reason
    return report


def non_caption_codes(report):
    """The reject codes of a gate report that can reject a clip (all but the captions')."""
    return set(reason_codes.codes_from_gate_report(report)) - {CAPTION_CODE}


def _run_quality_gate(mp4_path, clip, qc=None):
    """Call :mod:`clipengine.quality_gate` for the rendered mp4. Returns the full report.

    Stream / gaming / poker / screencast verticals run the lean stream gate (duration + LUFS +
    black/frozen + caption presence + the edge, slice, caption, hook and ending checks) and emit
    a real verdict; talking-head verticals run the full gate."""
    src = runstate.source(clip["source_id"]) or {}
    vertical = (src.get("vertical") or "").lower()
    layout = (clip.get("layout") or "").lower()
    src_media = src.get("local_path")      # the window the clip was cut from (second ASR tail,
                                           # crop-slice scan)
    if vertical in _STREAM_VERTICALS or layout in _STREAM_LAYOUTS:
        words = _load_words_sidecar(clip.get("words_path"))
        return _run_stream_lean_gate(mp4_path, clip, qc or {}, words, src_media=src_media)
    words = _load_words_sidecar(clip.get("words_path"))
    return _captions_never_reject(quality_gate.gate(
        str(mp4_path), plan=None, clip_meta=clip, campaign=None,
        words=words, caption_text=None,
        start_s=clip.get("start_s"), end_s=clip.get("end_s"),
        rendered_layout=(qc or {}).get("layout"),
        blur_fill_frac=(qc or {}).get("blur_fill_frac"),
        hook=clip.get("hook"), src_media=src_media, crop_views=(qc or {}).get("crop_views"),
        clip_checks_judge=_hook_judge(), panels=(qc or {}).get("panels")))


def _drop_clip(clip, reason, stage, report):
    cid = clip["clip_id"]
    detail = str(reason)[:400] or "quality gate failed"
    feats = _features_merge(clip, gate_report=report,
                            reason_codes=reason_codes.reject_codes(report, detail, stage="gate"))
    runstate.update_clip(cid, state="failed", error=f"gate: {detail}", features_json=feats)
    _log("warn", stage, f"clip {cid} DROPPED by gate: {detail}", clip_id=cid,
         source_id=clip.get("source_id"))


def _features_merge(clip, **extra):
    """Merge the clip's existing features_json dict with ``extra`` fields and return a JSON
    string ready for the store. Numpy scalars are stringified by store.add_clip's default."""
    try:
        base = json.loads(clip.get("features_json") or "{}") or {}
    except (ValueError, TypeError):
        base = {}
    base.update(extra)
    return json.dumps(base, default=lambda o: o.item() if hasattr(o, "item") else str(o))


MEDIA_END_PROBE_S = 2.0   # cut ends this close to the stored duration get an exact ffprobe
MIN_RENDER_S = 5.0        # a clamp that leaves less than this is not a clip


def _media_duration(path):
    """Shortest of the container/stream durations of `path` in seconds; None when ffprobe
    can't tell (the caller then renders the cut unclamped). recompose renders with -shortest,
    so this is as far as a cut can reach."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration:stream=duration", "-of", "json", str(path)],
                           capture_output=True, text=True, timeout=30)
        d = json.loads(r.stdout or "{}")
    except Exception:                                  # noqa: BLE001 - probe is best-effort
        return None
    if not isinstance(d, dict):
        return None
    vals = []
    for v in [(d.get("format") or {}).get("duration")] + [s.get("duration")
                                                          for s in d.get("streams") or []]:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0 and math.isfinite(f):
            vals.append(f)
    return min(vals) if vals else None


def clamp_cut_end(start_s, end_s, media_dur):
    """`end_s` pulled back to the media end when the cut overshoots it.
    edge-verify's tail extension pads the last word by tail_pad_s and can land past the end
    of a pre-sliced window (2.35-34.34s on a 34.0s window): recompose then renders only what
    exists and its QC rejects the short file ("duration 31.667 vs requested 31.99"). Floors
    to the ms so the request never exceeds the media. Raises when the clamp leaves less than
    MIN_RENDER_S."""
    start_s, end_s = float(start_s), float(end_s)
    if not media_dur or end_s <= media_dur:
        return end_s
    new_end = math.floor(float(media_dur) * 1000) / 1000
    if new_end - start_s < MIN_RENDER_S:
        raise RuntimeError(f"cut {start_s:.2f}-{end_s:.2f}s runs past media end "
                           f"{media_dur:.3f}s; {new_end - start_s:.2f}s left")
    return new_end


def _niche_of(source_id):
    """The niche of a source (the run's --niche), or its parent's for a window `<parent>_wNN`;
    None when the run names no niche."""
    if not source_id:
        return None
    for sid in dict.fromkeys((source_id, re.sub(r"_w\d+$", "", source_id))):
        row = runstate.source(sid)
        if row and row.get("niche"):
            return row["niche"]
    return None


# talking head vs screen share: a title that says which. Tech and finance pick talking heads first:
# their screen shares slice UI text in a 9:16 crop, and they may not stack.
_SCREEN_TITLE = re.compile(
    r"\b(tutorial|how to|step[- ]by[- ]step|walkthrough|coding|code review|live coding|setup|"
    r"demo|testing|benchmarks?|live trading|day trading|trading live|watch ?list|charts?|"
    r"charting|technical analysis|pre-?market|session live|scalping|screen ?share)\b", re.I)
_TALK_TITLE = re.compile(
    r"\b(podcast|interview|conversation|episode|ep\.? ?\d+|joins|debate|panel|sit-?down|"
    r"fireside|q ?& ?a with)\b|\bw/ ", re.I)


def source_kind(title):
    """"screen" (screen share, chart, tutorial, gameplay with a cam inset) or "talk" (podcast /
    interview, a speaker on camera) when the title says which, else "unknown"."""
    t = str(title or "")
    if _SCREEN_TITLE.search(t):
        return "screen"
    if _TALK_TITLE.search(t):
        return "talk"
    return "unknown"


def _content_kind_of(source_id):
    """source_kind of a niche source's title (or its window's parent's); None when the run
    names no niche. recompose fits a screen share / chart to width where a 9:16 crop would
    slice it."""
    if not source_id:
        return None
    for sid in dict.fromkeys((source_id, re.sub(r"_w\d+$", "", source_id))):
        row = runstate.source(sid)
        if row and row.get("niche"):
            return source_kind(row.get("title"))
    return None


def gameplay_clip(clip, src=None):
    """Whether a clip is gameplay (quality_gate.gameplay_clip on its source's vertical, niche and
    kind, and the layout it asked for): recompose renders it the game band and the stream lean
    gate fails it rendered as a face-follow crop."""
    sid = (src or {}).get("source_id") or clip.get("source_id")
    if src is None:
        src = (runstate.source(sid) if sid else None) or {}
    return quality_gate.gameplay_clip(src.get("vertical"), _niche_of(sid), _content_kind_of(sid),
                                      clip.get("layout"))


def _run_recompose(cmd):
    """Run the render: recompose's CompletedProcess (stdout carries its QC line)."""
    return subprocess.run(cmd, capture_output=True, text=True)


def recompose_cmd(clip, src, media, out):
    """The render/recompose.py command line `render` runs for `clip` cut from `media`. A niche
    clip passes its niche: only gaming/streamers may stack, tech/finance always render a single
    view."""
    cmd = [*RECOMPOSE, media, str(clip["start_s"]), str(clip["end_s"] - clip["start_s"]),
           str(out), "--layout", clip["layout"], "--captions",
           "--caption-model", CAPTION_MODEL, "--loudnorm", str(LOUDNORM_LUFS)]
    niche = _niche_of((src or {}).get("source_id") or clip.get("source_id"))
    if niche:
        cmd += ["--niche", niche]
        kind = _content_kind_of((src or {}).get("source_id") or clip.get("source_id"))
        if kind:
            cmd += ["--content-kind", kind]
    vertical = (src or {}).get("vertical")
    if vertical:
        cmd += ["--vertical", vertical]
    # gamestack needs the streamer's channel (per-channel facecam profile) and the game
    # (`render.game_genres` -> tight|wide crop). Pass them through if the source has them.
    channel = (src or {}).get("channel") or (src or {}).get("uploader")
    if channel:
        cmd += ["--channel", str(channel)]
    game = (src or {}).get("game")
    if game:
        cmd += ["--game", str(game)]
    if clip.get("words_path") and Path(clip["words_path"]).exists():
        cmd += ["--words", clip["words_path"]]
    if clip.get("hook"):
        cmd += ["--hook", clip["hook"]]
    crops = CFG.get("layout_crops", {}).get(clip["layout"])
    if clip["layout"] in ("screencast", "split") and crops:
        cmd += ["--top", crops["top"], "--bottom", crops["bottom"]]
        if crops.get("split"):
            cmd += ["--split", str(crops["split"])]
    if clip.get("split_fallback"):           # tech / finance only (recompose)
        cmd += ["--dual-view-fallback"]
    return cmd


# ---- RENDER: re-cut, don't reject (clip-quality 10-21 stage 1; failed starts: start-fix 10-22) ----
WINDOW_DEADLINE = None      # time.monotonic() a window's time cap ends (set by a caller);
                            # None (manual / campaign renders): no time check
RECUT_LOOK_BACK_S = 15.0    # precise words heard inside the clip from a failed cut (its sentence +
                            # the one next to it)
RECUT_ASR_S = 60.0          # a re-cut cut's precise pass: its cap, and its share of the time check
RECUT_MARGIN = 1.25         # ... plus this x the first render + gate's time
RECUT_CODES = ("cut_start", "cut_end")   # the reject codes a re-cut can fix
SEAT_AUDIO_S = 2.0          # window audio heard on each side of a cut to seat the caption words
CLIP_WORD_TOL_S = 0.05      # a strong caption word is the clip's when its middle is this close to it


def _mid(w):
    return (float(w["start"]) + float(w["end"])) / 2.0


def _seat(media, side, ref, cuts, rms_fn=None):
    """(sidecar, moved): boundaries.seat_words at each (name, t) of `cuts` on the window's audio
    around the cut (SEAT_AUDIO_S each side: the pad before a start), with the gate's
    word_boundaries tolerance. `ref` = the precise words the cuts were placed on. The audio is only
    read for a cut a caption word straddles. Never raises: a cut whose audio can't be read keeps
    its words (the gate judges them)."""
    tol = quality_gate.WORD_BOUNDARY_INSIDE_TOL_S
    moved = []
    for name, t in cuts:
        t = float(t)
        if not any(float(w["start"]) + tol < t < float(w["end"]) - tol for w in side):
            continue
        lo, hi = max(0.0, t - SEAT_AUDIO_S), t + SEAT_AUDIO_S
        try:
            rms = rms_fn(media, lo, hi) if rms_fn else \
                boundaries.frame_rms(transcript.load_audio(media, lo, hi), SR, 0.01)
            side, got = boundaries.seat_words(side, ref, t, rms, lo, tol=tol)
        except Exception:                 # noqa: BLE001 - the words stay as heard
            continue
        moved += [{"cut": name, **m} for m in got]
    return side, moved


def _seat_note(moved):
    return ", ".join(f"{m['w']!r} {m['from'][0]:.3f}-{m['from'][1]:.3f} -> {m['to'][0]:.3f}-"
                     f"{m['to'][1]:.3f}s ({m['cut']})" for m in moved)


# ---- RENDER: filler endings ----
TIDY_AUDIO_S = boundaries.SNAP_MAX_S + 1.0   # window audio read on each side of the clip


def cut_tidy_enabled(cfg=None):
    """config daily.cut_tidy.enabled (default True): boundaries.tidy_cut on the words a clip
    burns, before its render (the caption step, the repair lane)."""
    block = (((cfg if cfg is not None else CFG).get("daily") or {}).get("cut_tidy") or {})
    return bool(block.get("enabled", True))


def _editor_chose_start(clip):
    """True when the cut review approved or chose this clip's start (its last cut / title review
    that judged the cut kept or moved it, and was not undone): the tidy then looks forward first."""
    try:
        hist = (json.loads(clip.get("features_json") or "{}") or {}).get("cut_review") or []
    except (TypeError, ValueError):
        return False
    judged = [r for r in hist if not r.get("title_only")]
    return bool(judged) and judged[-1].get("verdict") in ("move", "keep") and \
        not judged[-1].get("undone")


def tidy_cut(side, start, end, media, media_dur, vc, rms=None, rms_fn=None, start_forward=False):
    """((start, end) or None, info or None): boundaries.tidy_cut on `side` (the words sidecar the
    render burns and the gate reads) with the window audio: `rms` (the whole window's 10 ms
    frames) or `rms_fn(media)` when the caller has them (the repair lane), else the audio around
    the clip read here. The words alone say whether a cut needs it, so the audio is only read
    then. (None, None) when nothing fired; the new cut when one moved. Never raises."""
    start, end = float(start), float(end)
    if not (cut_tidy_enabled() and side) or not boundaries.tidy_cut(side, start, end)["why"]:
        return None, None
    lo, t0 = max(0.0, start - TIDY_AUDIO_S), 0.0
    hi = end + TIDY_AUDIO_S if not media_dur else min(float(media_dur), end + TIDY_AUDIO_S)
    try:
        if rms is None and rms_fn is not None:
            rms = rms_fn(media)
        elif rms is None:
            rms = boundaries.frame_rms(transcript.load_audio(media, lo, hi), SR, 0.01)
            t0 = lo
        r = boundaries.tidy_cut(side, start, end, cfg=vc, rms=rms, t0=t0,
                                media_dur=media_dur or None, start_forward=start_forward)
    except Exception as e:                # noqa: BLE001 - the cut stays for the gate
        return None, {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    info = {k: r[k] for k in ("tidy", "why", "first", "last", "note") if r.get(k)}
    info.update(old=[round(start, 3), round(end, 3)], new=[r["start"], r["end"]])
    moved = (r["start"], r["end"]) != (start, end)
    return ((r["start"], r["end"]) if moved else None), info


def _tidy_note(info):
    parts = []
    for i, side in enumerate(("start", "end")):
        how = info["tidy"][side]
        if how == "ok":
            continue
        word = info.get("first" if side == "start" else "last")
        parts.append(f"{side} {info['old'][i]:.2f} -> {info['new'][i]:.2f}s ({how}: "
                     f"{info['why'].get(side)}" + (f"; {'opens' if i == 0 else 'ends'} {word!r}"
                                                   if word else "") + ")")
    return "; ".join(parts)


def _recut_words(media, lo, hi):
    """REFINE_MODEL words for [lo, hi] under the re-cut's own cap: select's CLI_DEADLINE is long
    past at render time (every whisper call would get 1 s), the window's is not."""
    saved = transcript.CLI_DEADLINE
    cap = time.monotonic() + RECUT_ASR_S
    transcript.CLI_DEADLINE = cap if WINDOW_DEADLINE is None else min(cap, WINDOW_DEADLINE)
    try:
        return transcript.words_window(media, lo, hi, model=REFINE_MODEL)
    finally:
        transcript.CLI_DEADLINE = saved


def recut_sides(report):
    """(["start"] / ["end"] / ["start", "end"], None) for a gate report that failed on its cuts
    alone (reject codes cut_start / cut_end only, rejects.codes_from_gate_report; a caption code
    never rejects since Amendment 1), else (None, why)."""
    codes = non_caption_codes(report)
    if not codes or not codes <= set(RECUT_CODES):
        return None, f"gate failed on {', '.join(sorted(codes)) or 'nothing coded'}, " \
                     "not on its cuts alone"
    return [s for s in ("start", "end") if f"cut_{s}" in codes], None


def recut_plan(clip, media, media_dur, report, vc, words_fn=None, rms_fn=None):
    """((new_start, new_end), sidecar words, info) for a clip the gate failed on its cuts alone
    (its end, its start or both), else (None, None, why). Each failed cut gets a precise pass
    (REFINE_MODEL) over its side of the window, the pad included: [end - 15 s, end + 9.5 s] picks
    the nearest other clean sentence end (boundaries.recut_end: up to 8 s on, or back inside the
    clip), [start - 9.5 s, start + 15 s] the nearest other clean sentence start (boundaries.
    recut_start: up to 8 s back, or on inside the clip). The sidecar keeps the clip's own caption
    words between the cuts, adds the precise words of an extension, and carries the words past
    each new cut as context for the gate's sentence checks (as the repair lane's sidecars do);
    no caption word is left on the pause a new cut sits in (_seat)."""
    sides, why = recut_sides(report)
    if sides is None:
        return None, None, why
    if not media_dur:
        return None, None, "window duration unknown"
    start, end = float(clip["start_s"]), float(clip["end_s"])
    words_fn = words_fn or _recut_words

    def hear(lo, hi):
        words = words_fn(media, lo, hi) or []
        try:
            rms = rms_fn(media, lo, hi) if rms_fn else \
                boundaries.frame_rms(transcript.load_audio(media, lo, hi), SR, 0.01)
        except Exception:                 # noqa: BLE001 - pad cuts without audio
            rms = None
        return words, rms

    new_s, new_e, info, heard = start, end, {"sides": sides}, {}
    if "start" in sides:
        lo = max(0.0, start - boundaries.SNAP_MAX_S - 1.5)
        words, rms = heard["start"] = hear(lo, min(end, start + RECUT_LOOK_BACK_S))
        edge_start = bool(words) and lo <= 0.05 and float(words[0]["start"]) >= WINDOW_EDGE_PAUSE_S
        new_s, got = boundaries.recut_start(words, start, end, cfg=vc, edge_start=edge_start,
                                            rms=rms, t0=lo)
        if new_s is None:
            return None, None, got
        info.update(got)
    if "end" in sides:
        lo = max(start, end - RECUT_LOOK_BACK_S)
        hi = min(float(media_dur), end + boundaries.SNAP_MAX_S + 1.5)
        words, rms = heard["end"] = hear(lo, hi)
        edge_end = bool(words) and hi >= float(media_dur) - 0.05 and \
            float(media_dur) - float(words[-1]["end"]) >= WINDOW_EDGE_PAUSE_S
        new_e, got = boundaries.recut_end(words, new_s, end, cfg=vc, media_dur=media_dur,
                                          edge_end=edge_end, rms=rms, t0=lo)
        if new_e is None:
            return None, None, got
        info.update(got)
    old = _load_words_sidecar(clip.get("words_path"))
    # a stretched first caption word sits a little before the old start (strong_captions keeps it)
    a = max(start, new_s) - CLIP_WORD_TOL_S if "start" in sides else float("-inf")
    b = min(end, new_e) if "end" in sides else float("inf")
    kept = [w for w in old if a <= _mid(w) <= b]
    side = list(kept)
    if "start" in sides:
        seam = min([float(w["start"]) for w in kept] or [max(start, new_s)])
        side += [{k: w[k] for k in ("w", "start", "end")} for w in heard["start"][0]
                 if _mid(w) < max(start, new_s) and float(w["end"]) <= seam + 0.1]
    if "end" in sides:
        seam = max([float(w["end"]) for w in kept] or [max(start, end - RECUT_LOOK_BACK_S)])
        side += [{k: w[k] for k in ("w", "start", "end")} for w in heard["end"][0]
                 if _mid(w) > min(end, new_e) and float(w["start"]) >= seam - 0.1]
    side = sorted(side, key=lambda w: float(w["start"]))
    for name, t in (("start", new_s), ("end", new_e)):
        if name in sides:
            side, moved = _seat(media, side, heard[name][0], ((name, t),), rms_fn=rms_fn)
            if moved:
                info.setdefault("seat", []).extend(moved)
    info["model"] = REFINE_MODEL
    return (new_s, new_e), side, info


def _recut(clip, src, media, media_dur, report, out, spent_s):
    """One re-cut + re-render + re-gate of a clip whose gate failed on its cuts alone (its end,
    its start, or both). Returns (clip, qc, report) of the new render, or None (the first report
    stands).
    when the window's time check refuses it and the gate failed only on a filler /
    conjunction its second ASR heard just before the cut, the end moves back past that word on
    the gate's own pass (_pull_back_after_gate: no new ASR) without the time check; a re-cut that
    needs new ASR (recut_plan's precise pass) always keeps it.
    the re-cut lane's plan is not made again for a cut the caption
    step already ran it on and failed (_recut_failed_before_render); only that pull-back may
    run then."""
    cid = clip["clip_id"]
    sides, why = recut_sides(report)
    if sides is None:
        _log("info", "render", f"clip {cid}: no re-cut ({why})", clip_id=cid)
        return None
    plan = None
    # the caption step already ran the re-cut lane on this very cut and the
    # gate's cut checks failed what it planned; the same words would plan the same cut again
    repeat = _recut_failed_before_render(clip, sides)
    late = None
    waiting = 0
    if WINDOW_DEADLINE is not None:
        # the window's clips still waiting for their first render come first
        waiting = sum(1 for c in runstate.clips_for_source(clip["source_id"])
                      if c["state"] == "selected" and c["clip_id"] != cid)
        left = WINDOW_DEADLINE - time.monotonic()
        need = RECUT_MARGIN * spent_s * (1 + waiting) + RECUT_ASR_S * len(sides)
        asr2 = report.get("second_asr") or {}
        if asr2.get("cached"):
            # the gate read its second ASR from the caption step's pass, so the first
            # render's time holds none of it; the re-gate hears the new cut afresh
            need += ASR2_RTF * (float(clip["end_s"]) - float(clip["start_s"]) +
                                float(asr2.get("tail_s") or 0.0))
        if left < need:
            late = (left, need)
    if late or repeat:
        plan, why = _pull_back_after_gate(clip, src, media, report, waiting if late else 0)
        no_new = (f"{late[0]:.0f}s left of the window, ~{late[1]:.0f}s needed with {waiting} "
                  f"clip(s) still to render" if late else
                  f"the re-cut lane's plan for this cut failed before the render: {repeat}")
        if plan is None:
            _log("info", "render", f"clip {cid}: no re-cut ({no_new}"
                 f"{'; no pull-back: ' + why if why else ''})", clip_id=cid)
            return None
        _log("info", "render", f"clip {cid}: " + (
            f"{late[0]:.0f}s left of the window (~{late[1]:.0f}s for a re-cut with new words)"
            if late else no_new) + f"; the gate's own second ASR pass moves the end back past "
             f"{plan[2]['dropped']!r}: no new ASR" + (", no time check" if late else ""),
             clip_id=cid)
    t0 = time.monotonic()
    try:
        if plan is not None:
            cut, side, info = plan
        else:
            vc = vertical_cfg((src or {}).get("vertical") or "general")
            cut, side, info = recut_plan(clip, media, media_dur, report, vc)
    except Exception as e:                # noqa: BLE001 - the first report stands
        cut, info = None, f"{type(e).__name__}: {str(e)[:160]}"
    if cut is None:
        _log("info", "render", f"clip {cid}: no re-cut ({info})", clip_id=cid)
        return None
    new_s, new_e = cut
    wp = SCRATCH / "words" / f"{clip['source_id']}_{float(clip['start_s']):.2f}_recut.json"
    wp.parent.mkdir(parents=True, exist_ok=True)
    wp.write_text(json.dumps(side))
    info["first_gate"] = str(report.get("drop_reason") or "")[:300]
    new = {**clip, "start_s": new_s, "end_s": new_e, "words_path": str(wp)}
    new["features_json"] = _features_merge(clip, recut=info)
    runstate.update_clip(cid, start_s=new_s, end_s=new_e, words_path=str(wp),
                      features_json=new["features_json"])
    what = []
    if "start" in sides:
        what.append(f"start {float(clip['start_s']):.2f} -> {new_s:.2f}s ({info['start_action']}, "
                    f"opens '{info['first'][:40]}')")
    if "end" in sides:
        what.append(f"end {float(clip['end_s']):.2f} -> {new_e:.2f}s ({info['action']}, ends "
                    f"'{info['last']}')")
    label = {"start": "start", "end": "ending"}
    _log("info", "render", f"clip {cid}: gate failed on the "
         f"{' and the '.join(label[s] for s in sides)} only; re-cut {'; '.join(what)}"
         f"{'; seated ' + _seat_note(info['seat']) if info.get('seat') else ''}, re-render once",
         clip_id=cid)
    r = _run_recompose(recompose_cmd(new, src, media, out))
    qc = _qc_line(r.stdout)
    if r.returncode != 0 or not out.exists():
        why = "; ".join(qc.get("errors") or []) or (r.stderr or r.stdout).strip()[-200:]
        runstate.update_clip(cid, start_s=clip["start_s"], end_s=clip["end_s"],
                          words_path=clip.get("words_path"))
        _log("warn", "render", f"clip {cid}: re-cut render failed ({why}); the first gate "
             f"verdict stands", clip_id=cid)
        return None
    report2 = _run_quality_gate(out, new, qc=qc)
    report2["recut"] = {**info, "seconds": round(time.monotonic() - t0, 1)}
    return new, qc, report2


# ---- RENDER: repair the captions, don't reject ----
CAPTION_VOTE_S = 180.0      # the tie-break votes' share of the time check: `medium` loads in the
                            # window's worker (~40 s) and hears the stretches (one call, <= 30 s of
                            # audio); a YouTube subtitle vote is one cached pull per video


def recaption_sides(report):
    """(cut sides to re-cut with the captions, None) for a gate report whose caption check failed
    and whose other failures (if any) are its cuts, else (None, why). Since Amendment 1 the
    report's verdict may be a pass: the captions are repaired all the same."""
    if caption_warning(report) is None:
        return None, "the caption check did not fail"
    rest = non_caption_codes(report)
    if not rest <= set(RECUT_CODES):
        return None, f"gate failed on {', '.join(sorted(rest))} too"
    return [s for s in ("start", "end") if f"cut_{s}" in rest], None


def _recaption(clip, src, media, media_dur, report, out, spent_s):
    """One caption repair + re-render + re-gate of a clip whose caption check failed (with a
    failed cut or two, which re-cut on the same plan as _recut). Each stretch where the burned
    captions and the second ASR disagree gets a tie-break vote (caption_repair.reconcile) and the
    majority's words: the best-vote captions are burned again with the same crop and layout and
    the full gate runs again (Amendment 1: on its other checks; captions never reject). A
    re-render that fails a check the first render passed is not shipped: the first render (and
    its captions) stands. Returns (clip, qc, report) of the new render, or None (the first render
    and report stand)."""
    from clipengine import caption_repair
    from clipengine import second_asr
    cid = clip["clip_id"]
    sides, why = recaption_sides(report)
    if sides is None:
        return None
    if not caption_repair.settings()["enabled"]:
        _log("info", "render", f"clip {cid}: no caption repair (disabled)", clip_id=cid)
        return None
    deadline = None
    if WINDOW_DEADLINE is not None:
        waiting = sum(1 for c in runstate.clips_for_source(clip["source_id"])
                      if c["state"] == "selected" and c["clip_id"] != cid)
        left = WINDOW_DEADLINE - time.monotonic()
        need = RECUT_MARGIN * spent_s * (1 + waiting) + CAPTION_VOTE_S + RECUT_ASR_S * len(sides)
        if left < need:
            _log("info", "render", f"clip {cid}: no caption repair ({left:.0f}s left of the "
                 f"window, ~{need:.0f}s needed with {waiting} clip(s) still to render)",
                 clip_id=cid)
            return None
        deadline = WINDOW_DEADLINE - RECUT_MARGIN * spent_s * (1 + waiting)
    t0 = time.monotonic()
    start, end = float(clip["start_s"]), float(clip["end_s"])
    side = _load_words_sidecar(clip.get("words_path"))
    info = {}
    try:
        if sides:
            vc = vertical_cfg((src or {}).get("vertical") or "general")
            cut, side, got = recut_plan(clip, media, media_dur, report, vc)
            if cut is None:
                _log("info", "render", f"clip {cid}: no caption repair (its cuts: {got})",
                     clip_id=cid)
                return None
            (start, end), info["recut"] = cut, got
        rec = second_asr.transcribe(media, start, end)     # the gate's own call: cached for it
        new_side, rep = caption_repair.reconcile(side, start, end, rec["words"], media=media,
                                                 source=src or {}, deadline=deadline)
    except Exception as e:                # noqa: BLE001 - the first report stands
        _log("info", "render", f"clip {cid}: no caption repair ({type(e).__name__}: "
             f"{str(e)[:160]})", clip_id=cid)
        return None
    for line in caption_repair.fix_lines(cid, rep):
        _log("info", "render", line, clip_id=cid)
    info["captions"] = {k: rep.get(k) for k in ("stretches", "youtube", "tiebreak_model", "votes",
                                               "fixes", "reason", "third_skipped")}
    verdict = rep.get("verdict") or {}
    info["captions"]["verdict"] = {k: verdict.get(k) for k in ("passed", "precision",
                                                               "worst_window", "coverage",
                                                               "worst_heard_window", "errors")}
    feats = _features_merge(clip, caption_repair=info)
    runstate.update_clip(cid, features_json=feats)
    if new_side is None and not sides:
        _log("info", "render", f"clip {cid}: caption repair changed nothing "
             f"({rep.get('reason') or 'no vote went to the second ASR'}); the first render stands",
             clip_id=cid)
        return None
    side_out = new_side if new_side is not None else side
    wp = SCRATCH / "words" / f"{clip['source_id']}_{float(clip['start_s']):.2f}_capfix.json"
    wp.parent.mkdir(parents=True, exist_ok=True)
    wp.write_text(json.dumps(side_out))
    new = {**clip, "start_s": start, "end_s": end, "words_path": str(wp), "features_json": feats}
    runstate.update_clip(cid, start_s=start, end_s=end, words_path=str(wp))
    _log("info", "render", f"clip {cid}: caption repair: {len(rep.get('fixes') or [])} "
         f"stretch(es) from the vote, caption check now {verdict.get('precision')}/"
         f"{verdict.get('coverage')}; burn the best-vote captions, re-render once", clip_id=cid)
    fix = out.with_name(f"{out.stem}_capfix{out.suffix}")
    r = _run_recompose(recompose_cmd(new, src, media, fix))
    qc = _qc_line(r.stdout)

    def first_stands(why):
        fix.unlink(missing_ok=True)
        runstate.update_clip(cid, start_s=clip["start_s"], end_s=clip["end_s"],
                          words_path=clip.get("words_path"))
        _log("warn", "render", f"clip {cid}: caption repair render {why}; the first render "
             f"stands", clip_id=cid)
        return None
    if r.returncode != 0 or not fix.exists():
        return first_stands("failed (" + ("; ".join(qc.get("errors") or []) or
                                          (r.stderr or r.stdout).strip()[-200:]) + ")")
    report2 = _run_quality_gate(fix, new, qc=qc)
    if report2.get("verdict") != "pass" and report.get("verdict") == "pass":
        return first_stands(f"failed the gate where the first passed "
                            f"({str(report2.get('drop_reason'))[:160]})")
    os.replace(fix, out)
    report2["caption_repair"] = {"fixes": rep.get("fixes") or [], "first_gate": str(
        report.get("drop_reason") or report.get("caption_warning") or "")[:300],
        "seconds": round(time.monotonic() - t0, 1),
        **({"recut": info["recut"]} if info.get("recut") else {})}
    return new, qc, report2


def caption_unresolved(clip, report):
    """Amendment 1: the clip ships with captions the second ASR still disagrees with (the best
    vote); a warning on the clip row and in the events, never a reject."""
    warn = caption_warning(report)
    if warn is None:
        return clip
    cid = clip["clip_id"]
    feats = _features_merge(clip, caption_unresolved={"warning": warn[:300]})
    runstate.update_clip(cid, features_json=feats)
    _log("warn", "render", f"clip {cid}: caption_unresolved (ships with the best-vote captions): "
         f"{warn[:200]}", clip_id=cid)
    return {**clip, "features_json": feats}


# ---- RENDER: the tech / finance dual view fallback ----
# off by default (config render.crop_feasibility.dual_view_fallback_niches []). The dual
# view is a stack and tech / finance never stack (the eye-check's layout rule):
# 10-18..25 all 30 gate-passed tech / finance dual views were rejected at the eye-check, 27 for the
# stack itself, and the gate failed others on boxed_slices (1303 / 1308).
SPLIT_NICHES = tuple((CFG.get("render", {}).get("crop_feasibility") or {})
                     .get("dual_view_fallback_niches", ()))
SPLIT_CODES = ("edge_slice",)        # the gate's crop_slices / boxed_slices / overlay edge cut
SPLIT_WITH = (CAPTION_CODE,) + RECUT_CODES   # repaired after the fallback as after any render
_SLICE_RENDER = re.compile(r"slic|blurred fill", re.I)     # the slice ladder's no-clean-crop


def split_due(clip, report=None, error=None):
    """None when the clip gets the dual view fallback, else why not. Tech and finance clips render
    a single view; when it slices the screen (the gate's edge_slice codes, or a
    render the slice ladder refused: no clean crop, the fit-to-width box over the blur cap) the
    clip renders once more as the dual view, the whole frame fit to width over the speaker.
    `error` = the first render's error, else the gate `report` decides."""
    if clip.get("split_fallback"):
        return "the dual view was rendered already"
    niche = _niche_of(clip.get("source_id"))
    if niche not in SPLIT_NICHES:
        return f"{niche or 'no niche'} clips render no dual view fallback"
    if error is not None:
        return None if _SLICE_RENDER.search(str(error)) else "the render failed on no slice"
    codes = set(reason_codes.codes_from_gate_report(report))
    if not codes & set(SPLIT_CODES):
        return "the single view sliced nothing"
    rest = codes - set(SPLIT_CODES) - set(SPLIT_WITH)
    if rest:
        return f"gate failed on {', '.join(sorted(rest))} too"
    return None


def _split_fallback(clip, src, media, report, out, spent_s):
    """One dual view re-render + re-gate of a tech / finance clip whose single view sliced the
    screen (split_due). Returns (clip, qc, report) of the new render, or None (the first report
    stands)."""
    cid = clip["clip_id"]
    why = split_due(clip, report=report)
    if why:
        return None
    if WINDOW_DEADLINE is not None:
        waiting = sum(1 for c in runstate.clips_for_source(clip["source_id"])
                      if c["state"] == "selected" and c["clip_id"] != cid)
        left = WINDOW_DEADLINE - time.monotonic()
        need = RECUT_MARGIN * spent_s * (1 + waiting)
        if left < need:
            _log("info", "render", f"clip {cid}: no dual view fallback ({left:.0f}s left of the "
                 f"window, ~{need:.0f}s needed with {waiting} clip(s) still to render)",
                 clip_id=cid)
            return None
    t0 = time.monotonic()
    first = str(report.get("drop_reason") or "")[:300]
    new = {**clip, "split_fallback": True,
           "features_json": _features_merge(clip, split_fallback={"first_gate": first})}
    runstate.update_clip(cid, features_json=new["features_json"])
    _log("info", "render", f"clip {cid}: the single view sliced the screen ({first[:160]}); dual "
         f"view fallback, re-render once", clip_id=cid)
    r = _run_recompose(recompose_cmd(new, src, media, out))
    qc = _qc_line(r.stdout)
    if r.returncode != 0 or not out.exists():
        why = "; ".join(qc.get("errors") or []) or (r.stderr or r.stdout).strip()[-200:]
        _log("warn", "render", f"clip {cid}: dual view fallback render failed ({why}); the first "
             f"gate verdict stands", clip_id=cid)
        return None
    report2 = _run_quality_gate(out, new, qc=qc)
    report2["split_fallback"] = {"first_gate": first, "layout": qc.get("layout"),
                                 "seconds": round(time.monotonic() - t0, 1)}
    return new, qc, report2


# ---- RENDER: caption words from the strong model (clip-quality 10-21 stage 3) ----
CAPTION_PAD_S = 0.5           # the strong pass hears this much around the cut (= repair's)
CAPTION_LEAD_S = 1.5          # ... this much before the cut and
CAPTION_TAIL_S = 2.0          # this much after it (= the gate's second-ASR tail), its words seated
                              # on the audio before the clip's are picked by their middle
STRONG_LOAD_S = 45.0          # large-v3-turbo load in a fresh whisper worker (~40 s measured)
STRONG_RTF = 3.0              # strong-pass seconds per audio second: 2.5-4.6 measured 10-21 at
                              # nice 19 under load 22-30 (7 fixture clips, warm worker)
CAPTION_RESERVE_S = 160.0     # render + gate of one clip (10-20/21 renders: p90 156 s) kept free
_STRONG = {"pid": None}       # the worker pid that has the strong model loaded


def _strong_cost(audio_s):
    """Estimated seconds of a strong pass on `audio_s` of audio (model load when the window's
    worker has not loaded it yet)."""
    p = transcript._WORKER.get("p")
    warm = bool(transcript.WORKER) and p is not None and p.poll() is None \
        and p.pid == _STRONG["pid"]
    return (0.0 if warm else STRONG_LOAD_S) + STRONG_RTF * float(audio_s)


def _strong_pass(media, lo, hi, timeout, prompt=None):
    words = strong_asr.transcribe_strong(media, lo, hi, model=CAPTION_STRONG_MODEL,
                                         timeout=timeout, prompt=prompt)
    p = transcript._WORKER.get("p")
    if transcript.WORKER and p is not None:
        _STRONG["pid"] = p.pid
    return words


_STRONG_PASS = _strong_pass         # the real pass (tests swap stages._strong_pass for 4-arg fakes)


def caption_prompt_enabled(cfg=None):
    """config render.caption_prompt.enabled (default True); CAPTION_PROMPT=0 turns it off."""
    if os.environ.get("CAPTION_PROMPT", "").strip() == "0":
        return False
    block = (((cfg if cfg is not None else CFG).get("render") or {}).get("caption_prompt") or {})
    return bool(block.get("enabled", True))


def caption_prompt(source_id):
    """whisper's initial prompt for the clip's strong caption pass: its source's
    title, punctuated (vocab_prompt), or None (off, no title). large-v3-turbo with
    --condition_on_previous_text False came back lower case with no . ? ! on 6 of the 10 HEAD
    replay clips of 11-11 01:30; with the title as its prompt the same spans came back punctuated
    (9i5VBuUtTbg: 0 -> 11 sentence ends on 60 words) and with the names cased ('USO',
    'Baghdad'). Every sentence check reads that punctuation: tidy, the final cut, the gate's
    sentence bounds and ending, the eye-check's reading of the captions."""
    if not caption_prompt_enabled():
        return None
    try:
        return vocab_prompt(runstate.source(source_id))
    except Exception:                     # noqa: BLE001 - no prompt then
        return None


def caption_context(cfg=None):
    """(lead s, tail s, seat) of the strong caption pass, from config render.caption_context.
    Passes that heard only the cut +-0.5 s: large-v3-turbo then dropped the
    first word (it times it from the silence before it, so its middle fell before the cut: 2742
    'We've', 2686 'Pike'), punctuated the last one as a sentence end the speaker talks on past
    (the gate's second ASR, 2 s past the cut, heard 'slow, so' where the captions burned 'slow.':
    9 of 62 ending rejects) and came back unpunctuated on 109 of 278 clips. Off ({"lead_s": 0.5,
    "tail_s": 0.5, "seat": false}): the 10-21 pass."""
    block = ((cfg if cfg is not None else CFG).get("render") or {}).get("caption_context") or {}
    return (float(block.get("lead_s", CAPTION_LEAD_S)), float(block.get("tail_s", CAPTION_TAIL_S)),
            bool(block.get("seat", True)))


def _seat_heard(media, lo, hi, words, rms_fn=None):
    """`words` (a strong pass over [lo, hi]) seated on the window audio there (boundaries.
    seat_onsets: a word claiming the pause before or after it starts / ends where the voice
    does), so a first word stretched back over the silence before the cut keeps its middle inside
    the clip. Never raises: words whose audio can't be read stay as heard."""
    try:
        rms = rms_fn(media, lo, hi) if rms_fn else \
            boundaries.frame_rms(transcript.load_audio(media, lo, hi), SR, 0.01)
    except Exception:                     # noqa: BLE001 - the words stay as heard
        return words, []
    if rms is None or not len(rms):
        return words, []
    return boundaries.seat_onsets(words, rms, lo)


def _flagged_spans(words, start, end, pad=CAPTION_PAD_S):
    """[(a, b)] spans of the sidecar's crosstalk-flagged phrases ("cap": False) inside the clip."""
    spans = []
    for w in words:
        if w.get("cap") is not False or not (start <= _mid(w) <= end):
            continue
        a, b = float(w["start"]) - pad, float(w["end"]) + pad
        if spans and a <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], b))
        else:
            spans.append((a, b))
    return [(max(start - pad, a), min(end + pad, b)) for a, b in spans]


def strong_captions(clip, media, words_fn=None, span=None, media_dur=None, rms_fn=None):
    """(sidecar words, info) whose in-clip caption words come from CAPTION_STRONG_MODEL
    (large-v3-turbo, the repair lane's recaption model) instead of select's pass: base.en when
    the precision pass failed or ran out of time (10-21 clips 261 / 332 / 314 / 456 / 408),
    small.en otherwise. Crosstalk phrases select's words do not confirm stay unburned (repair.
    recaption's phrase filter). Its single-speaker voice filter is not used: it dropped the
    speaker's own words as a "minority voice" (10-21 clip 260, a recaption repair, burned "is just
    that was" for "is just something that was top of mind").
    select's words judge the strong phrases only when they are its precision pass (refined): a
    base.en sidecar misses what large-v3-turbo hears (10-21 clip 261: "What's the end?" was
    heard and then dropped for it), so without one the word probabilities judge alone.

    Time: the full pass runs when it fits the window cap with the render + gate of this clip and
    of the window's clips still to render; else only the phrases select already flagged as
    crosstalk are re-heard (the spec's fallback: the stronger pass where the transcripts
    disagree); else select's words stay. (None, info) when nothing changed.
    `span` = (lo, hi): the full pass hears that instead (the words around the
    cuts the clean-cut snap reads); every word it heard there is in info["_heard"]."""
    t0 = time.monotonic()
    start, end = float(clip["start_s"]), float(clip["end_s"])
    side = _load_words_sidecar(clip.get("words_path"))
    inside = [w for w in side if start <= _mid(w) <= end]
    info = {"model": CAPTION_STRONG_MODEL}
    if not CAPTION_STRONG_MODEL:
        return None, {**info, "mode": "off"}
    reserve = 0.0
    left = None
    if WINDOW_DEADLINE is not None:
        waiting = sum(1 for c in runstate.clips_for_source(clip["source_id"])
                      if c["state"] == "selected" and c["clip_id"] != clip["clip_id"])
        reserve = CAPTION_RESERVE_S * (1 + waiting)
        left = WINDOW_DEADLINE - time.monotonic() - reserve
    lead, tail, seat = caption_context()
    narrow = (start - lead, end + tail)
    if media_dur:
        narrow = (narrow[0], min(float(media_dur), narrow[1]))
    full = tuple(span) if span else narrow
    if span and left is not None and left < _strong_cost(full[1] - full[0]):
        full, span = narrow, None          # the clean-cut span does not fit: today's pass, no snap
        info["snap_skipped"] = f"{left:.0f}s of the window left: the cuts' +-6.5 s not heard"
    full_s = full[1] - full[0]
    # the clip's words from select's window pass (one strong pass per
    # window, seated on the audio already); its word probabilities judge crosstalk
    window = cached_window_pass(media, runstate.source(clip["source_id"]) or {}) \
        if window_transcript_settings().get("model") == CAPTION_STRONG_MODEL else None
    if window is not None:
        from clipengine import crosstalk
        heard = [w for w in window if full[0] <= _mid(w) <= full[1]
                 and start - CLIP_WORD_TOL_S <= _mid(w) <= end + CLIP_WORD_TOL_S]
        flagged, rep = crosstalk.flag_crosstalk(heard, None)
        keep = [w for w in side if not (start <= _mid(w) <= end)]
        new = sorted(keep + flagged, key=lambda w: float(w["start"]))
        info.update(mode="window", judged_by=rep.get("judged_by"),
                    seconds=round(time.monotonic() - t0, 1), heard_s=round(full_s, 1),
                    words=len(flagged), captioned=len(crosstalk.captioned(flagged)),
                    dropped=rep.get("dropped") or [], prompt=caption_prompt(clip["source_id"]))
        if span:
            info["_heard"] = [w for w in window if full[0] <= _mid(w) <= full[1]]
        return new, info
    spans = _flagged_spans(side, start, end)
    if left is None or left >= _strong_cost(full_s):
        mode, todo = "full", [full]
    elif spans and left >= _strong_cost(sum(b - a for a, b in spans)):
        mode, todo = "phrases", spans
    else:
        return None, {**info, "mode": "skipped",
                      "reason": f"{left:.0f}s of the window left after the render reserve; a "
                                f"strong pass needs ~{_strong_cost(full_s):.0f}s"}
    from clipengine import crosstalk
    heard, all_heard = [], []
    seated = 0
    for a, b in todo:
        timeout = None if left is None else max(20.0, left)
        fn = words_fn or _strong_pass
        prompt = caption_prompt(clip["source_id"]) if fn is _STRONG_PASS else None
        got = fn(media, max(0.0, a), b, timeout, prompt=prompt) if prompt else \
            fn(media, max(0.0, a), b, timeout)
        if prompt:
            info["prompt"] = prompt
        if seat:
            got, moved = _seat_heard(media, max(0.0, a), b, got, rms_fn=rms_fn)
            seated += len(moved)
        all_heard += [w for w in got if a <= _mid(w) <= b]
        heard += [w for w in got if a <= _mid(w) <= b
                  and start - CLIP_WORD_TOL_S <= _mid(w) <= end + CLIP_WORD_TOL_S]
    try:
        refined = bool((json.loads(clip.get("features_json") or "{}") or {}).get("refined"))
    except (TypeError, ValueError):
        refined = False
    flagged, rep = crosstalk.flag_crosstalk(heard, inside if refined else None)
    info["judged_by"] = rep.get("judged_by")
    if mode == "full":
        keep = [w for w in side if not (start <= _mid(w) <= end)]
    else:
        keep = [w for w in side if not any(a <= _mid(w) <= b for a, b in todo)]
    new = sorted(keep + flagged, key=lambda w: float(w["start"]))
    info.update(mode=mode, seconds=round(time.monotonic() - t0, 1), heard_s=round(
        sum(b - a for a, b in todo), 1), words=len(flagged),
        captioned=len(crosstalk.captioned(flagged)), dropped=rep.get("dropped") or [])
    if seat:
        info["context"] = {"lead_s": lead, "tail_s": tail, "seated": seated}
    if span and mode == "full":
        info["_heard"] = sorted(all_heard, key=lambda w: float(w["start"]))
    return new, info


# ---- RENDER: clean cut points on the strong words ----
# Behind config daily.cut_snap.enabled, default off (not landed after review).
# Off, the caption step, the render and the repair lane's retrim are e8be8bc's: the strong pass
# hears the cut +-0.5 s, no gate cut check before the render, no snap / re-cut / drop there.
SNAP_MIN_WORDS = 8          # a clip with fewer words (gameplay, energy cuts) keeps its cuts


def cut_snap_enabled(cfg=None):
    """config daily.cut_snap.enabled (default False): the clean cut points before the
    render (the caption step) and on the strong words in the repair lane's retrim."""
    block = (((cfg if cfg is not None else CFG).get("daily") or {}).get("cut_snap") or {})
    return bool(block.get("enabled", False))


class NoCleanCut(Exception):
    """No clean sentence start / end fits the clip's limits on the strong words: the clip is
    dropped before its render."""

    def __init__(self, why, side, info=None):
        super().__init__(why)
        self.side, self.info = side, info or {}


def snap_span(clip, media, media_dur):
    """(lo, hi, rms) the strong pass hears for the clean-cut snap: the cut +- EXTEND_MAX_S (the
    farthest a cut moves) + the caption pad, inside the window, with the window audio's 10 ms frame
    RMS there; None when the snap does not apply (off, an energy cut, too few words, or the audio
    cannot be read: large-v3-turbo's word onsets are only read on the audio)."""
    if not (cut_snap_enabled() and CAPTION_STRONG_MODEL) or \
            (clip.get("boundary_type") or "sentence") != "sentence":
        return None
    start, end = float(clip["start_s"]), float(clip["end_s"])
    side = _load_words_sidecar(clip.get("words_path"))
    if sum(1 for w in side if start <= _mid(w) <= end) < SNAP_MIN_WORDS:
        return None
    pad = boundaries.EXTEND_MAX_S + CAPTION_PAD_S
    lo, hi = max(0.0, start - pad), end + pad
    if media_dur:
        hi = min(float(media_dur), hi)
    try:
        rms = boundaries.frame_rms(transcript.load_audio(media, lo, hi), SR, 0.01)
    except Exception:                     # noqa: BLE001 - no audio: no snap
        return None
    if not len(rms):
        return None
    return lo, hi, rms


def gate_cut_errors(side, start, end, media, lean=False, seen=None):
    """The clip's own gate's cut checks on a cut before its render: the full gate's
    word_boundaries and sentence_bounds on the sidecar (not the stream lean gate's: `lean`), and
    the ending on the second ASR (both gates; the gate reads that call from its cache after the
    render). Their errors, [] when they pass (an unreadable second ASR leaves the ending to the
    gate). `seen` (a dict) gets the second ASR record ("rec") and the ending measure ("ending")
    when the ending was read. With quality_gate.cut_audio on, the cuts are read on the window
    audio as the gate reads them (a cut in a pause is in no word)."""
    from clipengine import second_asr
    errs = []
    audio = None if lean else quality_gate.safe_cut_audio(media, start, end)
    sides = () if lean else (
        ("word_boundaries", quality_gate.measure_word_boundaries(side, start, end, audio=audio)),
        ("sentence_bounds", quality_gate.measure_sentence_bounds(side, start, end, audio=audio)))
    for k, m in sides:
        if m.get("passed") is False:
            errs += [f"{k}: {e}" for e in m.get("errors") or ["failed"]]
    try:
        if second_asr.settings()["enabled"]:
            rec = second_asr.transcribe(media, start, end)
            try:
                energy = quality_gate.cut_energy(media, start, end)
            except Exception:             # noqa: BLE001 - the ending runs on the words alone
                energy = None
            m = quality_gate._safe(quality_gate.ending_verdict, rec["words"],
                                   quality_gate.clip_words(side, start, end), end - start,
                                   tail_s=rec["tail_s"], energy=energy,
                                   seat=quality_gate.ending_seat_enabled(),
                                   quiet_end=quality_gate.cut_audio_enabled()
                                   and quality_gate.energy_end_in_quiet(energy))
            if seen is not None:
                seen.update(rec=rec, ending=m)
            if m.get("passed") is False:
                errs += [f"ending: {e}" for e in m.get("errors") or ["failed"]]
    except Exception:                     # noqa: BLE001 - the gate judges the ending
        pass
    return errs


def _errs_report(errs):
    """gate_cut_errors' errors as the gate report measures they came from."""
    report = {}
    for e in errs:
        k, _, msg = e.partition(": ")
        report.setdefault(k, {"passed": False, "errors": []})["errors"].append(msg)
    return report


def _recut_before_render(clip, side, media, media_dur, errs, vc, lean, seen=None):
    """((start, end), sidecar, info) of the re-cut lane's plan (recut_plan, the precise words) for
    a cut the gate's cut checks fail (`errs`, gate_cut_errors), when the same checks pass its new
    cut; else (None, why). What the render's re-cut would do after a failed gate, before the
    render instead. `seen` (a dict) gets the failed plan as "plan" (the cut it was made from, its
    sides, the cut it planned or None) when the lane planned and failed (_recut reads it back)."""
    report = _errs_report(errs)
    wp = SCRATCH / "words" / f"{clip['source_id']}_{float(clip['start_s']):.2f}_precut.json"
    wp.parent.mkdir(parents=True, exist_ok=True)
    wp.write_text(json.dumps(side))
    cut, side2, got = recut_plan({**clip, "words_path": str(wp)}, media, media_dur, report, vc)
    plan = {"from": [round(float(clip["start_s"]), 3), round(float(clip["end_s"]), 3)],
            "sides": recut_sides(report)[0] or []}
    if cut is None:
        if seen is not None:
            seen["plan"] = {**plan, "cut": None}
        return None, str(got)
    errs2 = gate_cut_errors(side2, cut[0], cut[1], media, lean=lean)
    if errs2:
        if seen is not None:
            seen["plan"] = {**plan, "cut": [round(cut[0], 3), round(cut[1], 3)]}
        return None, f"the re-cut {cut[0]:.2f}-{cut[1]:.2f}s fails them too ({'; '.join(errs2)})"
    return (cut, side2, got), None


def strong_snap(clip, heard, span, media_dur, vc, media=None):
    """(new start, new end, sidecar, info, seat) for the clip's cuts on the strong words `heard`
    over `span` (lo, hi, rms): the words seated on the audio (boundaries.seat_onsets), the cuts snapped
    to clean sentence edges (boundaries.snap_sentences), the caption words of the new cut flagged
    for crosstalk as strong_captions flags them (select's precise words judge the old cut's words,
    the word probabilities the rest) and the strong words outside the cut kept as context for the
    gate's sentence checks; `seat` = the seated words that sat under a cut (the start-fix's
    record). When no clean pair fits: NoCleanCut without `media`, else the cut as it is with
    info["unclean"] (_caption_step has the gate's own cut checks read it once its caption words
    are seated: the snap is stricter than the gate, which stays the judge)."""
    from clipengine import crosstalk
    lo, hi, rms = span
    start, end = float(clip["start_s"]), float(clip["end_s"])
    words, moved = boundaries.seat_onsets(heard, rms, lo)
    edges = (bool(words) and float(words[0]["start"]) - lo >= WINDOW_EDGE_PAUSE_S,
             bool(words) and hi - float(words[-1]["end"]) >= WINDOW_EDGE_PAUSE_S)
    r = boundaries.snap_sentences(words, start, end, cfg=vc, edges_clean=edges,
                                  media_dur=media_dur, rms=rms, t0=lo)
    info = {"onsets_seated": len(moved), "heard": [round(lo, 3), round(hi, 3)]}
    unclean = None
    if "drop" in r:
        if media is None:
            raise NoCleanCut(r["drop"], r.get("side"), info)
        unclean = r
        inside = [w for w in words if start <= _mid(w) <= end]
        r = {"start": start, "end": end, "snap": {"start": "kept", "end": "kept"},
             "first": inside[0]["w"] if inside else "", "last": inside[-1]["w"] if inside else ""}
    new_s, new_e = float(r["start"]), float(r["end"])
    info.update(snap=r["snap"], old=[round(start, 3), round(end, 3)],
                new=[round(new_s, 3), round(new_e, 3)], first=r["first"], last=r["last"])
    side = _load_words_sidecar(clip.get("words_path"))
    try:
        refined = bool((json.loads(clip.get("features_json") or "{}") or {}).get("refined"))
    except (TypeError, ValueError):
        refined = False
    ref = [w for w in side if start <= _mid(w) <= end] if refined else None
    clip_w = [w for w in words if new_s - CLIP_WORD_TOL_S <= _mid(w) <= new_e + CLIP_WORD_TOL_S]
    old_part = [w for w in clip_w if start - CLIP_WORD_TOL_S <= _mid(w) <= end + CLIP_WORD_TOL_S]
    new_part = [w for w in clip_w if w not in old_part]
    flagged, rep = crosstalk.flag_crosstalk(old_part, ref)
    if new_part:
        more, rep2 = crosstalk.flag_crosstalk(new_part, None)
        flagged += more
        rep["dropped"] = (rep.get("dropped") or []) + (rep2.get("dropped") or [])
    context = [w for w in words if not (new_s - CLIP_WORD_TOL_S <= _mid(w) <= new_e + CLIP_WORD_TOL_S)]
    out = sorted(context + flagged, key=lambda w: float(w["start"]))
    info.update(captioned=len(crosstalk.captioned(flagged)), words=len(flagged),
                dropped=rep.get("dropped") or [])
    # the start-fix's record: a word under a cut that the audio seated
    cuts = (("start", new_s), ("end", new_e))
    seat = [{"cut": n, **m} for m in moved for n, t in cuts if m["from"][0] < t < m["from"][1]]
    if unclean is not None:     # _caption_step asks the gate's cut checks once the words sit
        info["unclean"] = {"drop": unclean["drop"], "side": unclean.get("side")}
    return new_s, new_e, out, info, seat


def _fix_cut_before_render(clip, media, media_dur, heard, span, side, select_words, errs, lean,
                           info):
    """(new cut or None, sidecar, seated words) for a cut the clip's own gate's cut checks fail
    (`errs`, gate_cut_errors on `side`, the words it renders with today): the snap on the strong
    words (strong_snap) when it finds a clean pair the same checks pass, else the re-cut lane's plan
    before the render (_recut_before_render). Raises NoCleanCut when neither finds a cut they pass:
    the gate would reject the clip, and the render is not spent on it."""
    cid = clip["clip_id"]
    s0, e0 = float(clip["start_s"]), float(clip["end_s"])
    vc = vertical_cfg((runstate.source(clip["source_id"]) or {}).get("vertical") or "general")
    new_s, new_e, wide, snap, seat = strong_snap(clip, heard, span, media_dur, vc, media=media)
    info["clean_cut"] = snap
    snap["gate_cut_errors"] = errs
    unclean = snap.pop("unclean", None)
    why = unclean["drop"] if unclean else None
    if unclean is None:
        wide, seat2 = _seat(media, wide, select_words, (("start", new_s), ("end", new_e)))
        errs2 = gate_cut_errors(wide, new_s, new_e, media, lean=lean)
        if not errs2:
            info.update(words=snap["words"], captioned=snap["captioned"], dropped=snap["dropped"])
            _log("info", "render", f"clip {cid}: the gate's cut checks fail the cut "
                 f"({'; '.join(errs)[:160]}); clean cut on the {info['model']} words "
                 f"{s0:.2f}-{e0:.2f}s -> {new_s:.2f}-{new_e:.2f}s ({snap['snap']['start']}/"
                 f"{snap['snap']['end']}: opens {snap['first']!r}, ends {snap['last']!r})",
                 clip_id=cid)
            cut = (new_s, new_e) if (new_s, new_e) != (s0, e0) else None
            return cut, wide, seat + seat2
        why = f"the snapped cut {new_s:.2f}-{new_e:.2f}s fails them too ({'; '.join(errs2)[:160]})"
    try:
        fixed, why2 = _recut_before_render(clip, side, media, media_dur, errs, vc, lean)
    except Exception as e:                # noqa: BLE001 - no re-cut: the drop stands
        fixed, why2 = None, f"{type(e).__name__}: {str(e)[:120]}"
    if fixed is None:
        codes = set(reason_codes.codes_from_gate_report(
            {e.partition(": ")[0]: {"passed": False, "errors": [e.partition(": ")[2]]}
             for e in errs}))
        side_name = (unclean or {}).get("side") or (
            "pair" if {"cut_start", "cut_end"} <= codes else
            "start" if "cut_start" in codes else "end")
        raise NoCleanCut(f"the gate's cut checks fail the cut ({'; '.join(errs)[:240]}); no clean "
                         f"cut on the strong words ({why[:160]}) and the re-cut finds no cut they "
                         f"pass ({why2[:160]})", side_name, snap)
    cut, side2, snap["recut"] = fixed
    _log("info", "render", f"clip {cid}: the gate's cut checks fail the cut "
         f"({'; '.join(errs)[:160]}); no clean cut on the strong words ({why[:120]}); the re-cut "
         f"lane's plan, before the render: {s0:.2f}-{e0:.2f}s -> {cut[0]:.2f}-{cut[1]:.2f}s",
         clip_id=cid)
    return cut, side2, []


# ---- RENDER: a filler the second ASR hears just before the cut ----
# 10-26 lifestyle 1252 / 1253: the burned words end clean ('mess.' / 'attention'), so tidy_cut
# kept the cuts, and only the clean-cut branch read the gate's ending before the render; the
# gate's second ASR (small) heard 'and' 0.21 / 0.24 s before the cut, and the re-cut after the
# gate did not fit the window (147 s left, ~222 s needed).
# that 'and' was small's stretch over the pause after the cut; the gate's ending now seats
# such a word on the audio (quality_gate.ending_seat), so what reaches the code below is a filler
# the audio cannot place after the cut.
ASR2_FILLER_END = "ends on a filler/conjunction "   # = quality_gate.ending_verdict's error
ASR2_RTF = 4.0              # second-ASR (small) seconds per audio second: 2.5-5 measured 10-19,
                            # 3.5-9 in the replays (nice 19 next to a live run)
PULL_BACK_STEP_S = 0.005    # the pulled-back end's grid


def _lean_gate(clip):
    """Whether the clip's gate is the stream lean gate (= _run_quality_gate's choice)."""
    vertical = ((runstate.source(clip["source_id"]) or {}).get("vertical") or "").lower()
    return vertical in _STREAM_VERTICALS or (clip.get("layout") or "").lower() in _STREAM_LAYOUTS


def _still_to_render(clip):
    """The window's other clips still waiting for their first render."""
    return sum(1 for c in runstate.clips_for_source(clip["source_id"])
               if c["state"] == "selected" and c["clip_id"] != clip["clip_id"])


def _asr_room(clip, asr_s):
    """None when the window has `asr_s` s of new ASR left over the render + gate of this clip and
    of its clips still to render (CAPTION_RESERVE_S each, as strong_captions keeps), else why
    not. No window (manual / campaign renders): None."""
    if WINDOW_DEADLINE is None:
        return None
    waiting = _still_to_render(clip)
    left = WINDOW_DEADLINE - time.monotonic()
    need = float(asr_s) + CAPTION_RESERVE_S * (1 + waiting)
    if left >= need:
        return None
    return f"{left:.0f}s left of the window, ~{need:.0f}s needed with {waiting} clip(s) still " \
           f"to render"


def asr2_filler_end(ending):
    """The word the gate's ending (quality_gate.ending_verdict) read as the clip's last when it
    failed it as a filler / conjunction the second ASR heard before the cut, else None."""
    m = ending or {}
    if m.get("passed") is False and any(str(e).startswith(ASR2_FILLER_END)
                                        for e in m.get("errors") or []):
        return m.get("last_word")
    return None


def filler_end_only(report):
    """Whether a gate report's only failure (the captions' aside) is that ending, alone."""
    failed = {k for k, m in (report or {}).items()
              if isinstance(m, dict) and m.get("passed") is False} - set(NON_REJECTING_GATE_KEYS)
    m = (report or {}).get("ending") or {}
    return failed == {"ending"} and len(m.get("errors") or []) == 1 and \
        asr2_filler_end(m) is not None


def _predicted_cut_errors(words, asr, tail, start, end, lean=False, audio=None):
    """gate_cut_errors on the cut [start, end] with the second ASR words `asr` (clip-relative
    from `start`) of a longer cut instead of a new pass: those heard up to `end` + `tail`. Words
    only (the ending's energy probe is left to the gate), and `audio` (quality_gate.cut_audio
    frames of this cut) when the caller has it."""
    d = end - start
    heard = [w for w in asr if float(w["start"]) < d + float(tail or 0.0)]
    checks = () if lean else (
        ("word_boundaries", quality_gate.measure_word_boundaries(words, start, end, audio=audio)),
        ("sentence_bounds", quality_gate.measure_sentence_bounds(words, start, end, audio=audio)))
    checks += (("ending", quality_gate._safe(quality_gate.ending_verdict, heard,
                                             quality_gate.clip_words(words, start, end), d,
                                             tail_s=tail,
                                             quiet_end=quality_gate.cut_in_quiet(audio, end))),)
    errs = []
    for k, m in checks:
        if m.get("passed") is False:
            errs += [f"{k}: {e}" for e in m.get("errors") or ["failed"]]
    return errs


def filler_end_pull_back(words, start, end, rec, ending, cfg=None, lean=False):
    """(new end, info) moving the end back past the one word the gate's `ending` read as the
    clip's last, a filler / conjunction the second ASR heard just before the cut, else (None,
    why). No ASR, no audio: `rec` = the second ASR record that ending read (words clip-relative
    from `start`), read as the gate reads a shorter cut (the same words, heard to the new end +
    the tail); `words` = the burned sidecar. The end goes on the latest point (PULL_BACK_STEP_S
    steps) from that word's onset + the gate's onset tolerance back to the start of the word
    before it where the gate's own cut checks pass on both (_predicted_cut_errors: the word
    before must end a sentence, . ? ! in either transcript, as the gate reads it), never under
    min_dur_s. info: old_end, new_end, action "pull_back", dropped (the filler), last (the word
    before it)."""
    word = asr2_filler_end(ending)
    if word is None:
        return None, "the gate's ending did not fail on a filler / conjunction"
    c = {**boundaries.DEFAULT_CFG, **(cfg or {})}
    start, end = float(start), float(end)
    asr, tail = list(rec.get("words") or []), rec.get("tail_s")
    at = end - start + float(ending.get("last_word_start_s") or 0.0)
    k = next((i for i in range(len(asr) - 1, -1, -1) if str(asr[i]["w"]) == str(word)
              and abs(float(asr[i]["start"]) - at) <= 0.006), None)
    if not k:
        return None, f"{word!r} is not in the second ASR's words with one before it"
    prev = asr[k - 1]
    hi = start + float(asr[k]["start"]) + quality_gate.ENDING_ONSET_TOL_S - PULL_BACK_STEP_S
    lo = max(start + float(prev["start"]), start + float(c["min_dur_s"]))
    first = None
    for i in range(int((hi - lo) / PULL_BACK_STEP_S + 1e-6) + 1):
        t = round(hi - i * PULL_BACK_STEP_S, 3)
        errs = _predicted_cut_errors(words, asr, tail, start, t, lean)
        if not errs:
            return t, {"old_end": round(end, 3), "new_end": t, "action": "pull_back",
                       "dropped": str(word), "last": str(prev["w"])}
        first = first or errs
    return None, (f"no end from {hi:.3f} back to {lo:.3f}s past {word!r} passes the gate's cut "
                  f"checks on its second ASR words ({'; '.join(first or ['min_dur_s'])[:200]})")


def _asr2_end_fix(clip, media, media_dur, side, start, end, vc, lean):
    """(new cut or None, new sidecar or None, info or None) for the caption step's tidy branch in
    a window: the clip's own gate's cut checks on the cut, the ending on the
    second ASR (the pass the gate then reads from its cache), when the window has its render
    reserve. An ending they fail on a filler / conjunction the second ASR heard before the cut is
    fixed before the render, as the clean-cut branch fixes a cut (_fix_cut_before_render): the end
    back past that word (filler_end_pull_back) when the gate's own pass on the new cut passes it,
    else the re-cut lane's plan (_recut_before_render: new precise words), each only when the
    window has the time for its new ASR. Else the cut stays and the gate judges it (no drop
    before the render). info is None when the ending did not fail that way."""
    cid = clip["clip_id"]
    late = _asr_room(clip, 0.0)
    if late:
        _log("info", "render", f"clip {cid}: the second ASR's ending not read before the render "
             f"({late})", clip_id=cid)
        return None, None, {"skipped": late}
    seen = {}
    errs = gate_cut_errors(side, start, end, media, lean=lean, seen=seen)
    word = asr2_filler_end(seen.get("ending"))
    if word is None:
        return None, None, None
    rec, ending = seen["rec"], seen["ending"]
    tail = float(rec.get("tail_s") or 0.0)
    info = {"errors": errs, "word": word, "at": ending.get("last_word_start_s")}
    heard = f"the gate's second ASR hears {word!r} {float(info['at'] or 0):+.2f} s from the end"
    t, got = filler_end_pull_back(side, start, end, rec, ending, cfg=vc, lean=lean)
    if t is None:
        info["pull_back"], back = {"none": got}, got
    else:
        info["pull_back"] = got
        late = _asr_room(clip, ASR2_RTF * (t - start + tail))
        if late:
            got.update(verified=None, skipped=late)
            back = f"{t:.2f}s not heard: {late}"
        else:
            seen2 = {}
            errs2 = gate_cut_errors(side, start, t, media, lean=lean, seen=seen2)
            got["verified"] = (not errs2) if "ending" in seen2 else None
            if got["verified"]:
                _log("info", "render", f"clip {cid}: {heard}; end {end:.2f} -> {t:.2f}s, back "
                     f"past it (ends {got['last']!r}; the gate's checks pass it on its own pass), "
                     f"before the render", clip_id=cid)
                return (start, t), None, info
            got["errors"] = errs2 or ["the second ASR did not run on the new cut"]
            back = f"{t:.2f}s fails them on its own pass: {'; '.join(got['errors'])}"
    sides, _why = recut_sides(_errs_report(errs))
    late = _asr_room(clip, RECUT_ASR_S * len(sides or ["end"]) + ASR2_RTF * (end - start + tail))
    if late:
        info["recut"] = {"skipped": late}
    else:
        planned = {}
        try:
            # the plan starts from the cut judged here (tidy_cut may have moved the clip's)
            fixed, why = _recut_before_render({**clip, "start_s": start, "end_s": end}, side,
                                              media, media_dur, errs, vc, lean, seen=planned)
        except Exception as e:            # noqa: BLE001 - the cut stays for the gate
            fixed, why = None, f"{type(e).__name__}: {str(e)[:120]}"
        if fixed is not None:
            cut, side2, info["recut"] = fixed
            _log("info", "render", f"clip {cid}: {heard}; no end back past it ({back[:140]}); "
                 f"the re-cut lane's plan, before the render: {start:.2f}-{end:.2f}s -> "
                 f"{cut[0]:.2f}-{cut[1]:.2f}s", clip_id=cid)
            return cut, side2, info
        info["recut"] = {"none": why, **planned}
    _log("info", "render", f"clip {cid}: {heard}; no cut the gate's checks pass before the render "
         f"(back past it: {back[:140]}; re-cut: {str(info['recut'])[:140]}); the gate judges it",
         clip_id=cid)
    return None, None, info


def _recut_failed_before_render(clip, sides):
    """Why the re-cut lane already failed on the clip's cut for `sides` before its render:
    _asr2_end_fix planned from this very cut and the gate's cut checks failed the
    cut it planned, or it planned none; else None (it did not run, was skipped for time, or
    raised)."""
    try:
        feats = json.loads(clip.get("features_json") or "{}") or {}
    except (TypeError, ValueError):
        return None
    tried = ((feats.get("captions") or {}).get("asr2_end") or {}).get("recut") or {}
    plan = tried.get("plan") or {}
    here = [round(float(clip["start_s"]), 3), round(float(clip["end_s"]), 3)]
    if plan.get("from") != here or sorted(plan.get("sides") or []) != sorted(sides):
        return None
    return str(tried.get("none") or "no cut the gate's cut checks pass")[:200]


def _pull_back_after_gate(clip, src, media, report, waiting):
    """(((start, end), sidecar, info), None) for a re-cut the window's time check refuses:
    the gate's only failure is a filler / conjunction its second ASR heard just before
    the cut (filler_end_only), no other clip of the window still waits for its first render, and
    the end moves back past that word on the gate's own pass, read back from its cache (no new
    ASR), to a cut the caption step did not already see fail. Else (None, why or None)."""
    if not (cut_tidy_enabled() and filler_end_only(report)):
        return None, None
    if waiting:
        return None, f"{waiting} clip(s) still to render come first"
    from clipengine import second_asr
    start, end = float(clip["start_s"]), float(clip["end_s"])
    try:
        rec = second_asr.cached(media, start, end)
    except Exception:                     # noqa: BLE001 - no pass to read: no pull-back
        rec = None
    if rec is None:
        return None, "the gate's second ASR pass is not in its cache"
    side = _load_words_sidecar(clip.get("words_path"))
    vc = vertical_cfg((src or {}).get("vertical") or "general")
    t, got = filler_end_pull_back(side, start, end, rec, report["ending"], cfg=vc,
                                  lean=_lean_gate(clip))
    if t is None:
        return None, got
    try:
        feats = json.loads(clip.get("features_json") or "{}") or {}
    except (TypeError, ValueError):
        feats = {}
    tried = ((feats.get("captions") or {}).get("asr2_end") or {}).get("pull_back") or {}
    if tried.get("new_end") == t and tried.get("verified") is False:
        return None, f"the pull-back to {t:.2f}s failed the gate's checks before the render"
    return ((start, t), side, {**got, "sides": ["end"], "time_check": "none (no new ASR)",
                               "model": f"the gate's {rec.get('model')} pass (cached)"}), None


def _caption_step(clip, media, media_dur=None):
    """strong_captions on the clip before its first render; the new sidecar goes on the clip row
    (the gate reads the burned words from it). Never raises but NoCleanCut: select's words stay
    on any error.
    start-fix 10-22: select placed the cuts in a pause of its own words; a strong word stretched
    over that pause is seated on the audio (_seat, select's words as the reference), or the gate
    reads the cut as inside it (10-22 clips 529 / 531: 'I' / 'so' timed from the end of the word
    before, over 0.17-0.37 s of silence).
    the strong pass also hears EXTEND_MAX_S around the cuts. A cut the clip's own
    gate's cut checks pass (gate_cut_errors) stays as it is; one they fail is fixed before the
    render (_fix_cut_before_render: the snap to clean sentence edges on the strong words, else the
    re-cut lane's plan), or the clip raises NoCleanCut (render drops it before the render).
    filler endings: otherwise tidy_cut checks the cuts on the burned words, and
    (in a window) the gate's ending on its second ASR is read on the strong words
    before the render too (_asr2_end_fix): a filler / conjunction heard just before the cut is
    fixed there when a cut the gate's checks pass is found, else the gate judges the cut (no drop
    before it)."""
    cid = clip["clip_id"]
    new_cut = None
    try:
        span = snap_span(clip, media, media_dur)
        side, info = strong_captions(clip, media, span=span[:2] if span else None,
                                     media_dur=media_dur)
        heard = info.pop("_heard", None)
        select_words = _load_words_sidecar(clip.get("words_path"))
        s0, e0 = float(clip["start_s"]), float(clip["end_s"])
        if side is not None:
            side, seat = _seat(media, side, select_words, (("start", s0), ("end", e0)))
            if seat:
                info["seat"] = seat
        if side is not None and span and heard is not None:
            # the clip's own gate's cut checks read the cut as it is, on the words it will read:
            # a cut they pass stays exactly as before this step (10-21 647, tech, gate-passed: snapping
            # its clean cut moved it off what its hook says, and the hook check failed it)
            lean = _lean_gate(clip)
            errs = gate_cut_errors(side, s0, e0, media, lean=lean)
            if not errs:
                info["clean_cut"] = {"snap": {"start": "kept", "end": "kept"},
                                     "kept": "the gate's cut checks pass the cut"}
            else:
                new_cut, side, seat = _fix_cut_before_render(clip, media, media_dur, heard, span,
                                                             side, select_words, errs, lean, info)
                if seat:
                    info["seat"] = info.get("seat", []) + seat
        else:
            # filler endings: a cut placed in a pause on select's words can sit
            # inside a strong word stretched over it, or leave a filler as the last burned word
            vc = vertical_cfg((runstate.source(clip["source_id"]) or {}).get("vertical") or "general")
            new_cut, tidy = tidy_cut(side if side is not None else select_words, s0, e0, media,
                                     media_dur, vc, start_forward=_editor_chose_start(clip))
            if tidy:
                info["tidy"] = tidy
                if "tidy" in tidy:
                    _log("info", "render", f"clip {cid}: " + (
                        f"cut tidied on the burned words: {_tidy_note(tidy)}" if new_cut else
                        f"the burned words fail the cut ({_tidy_note(tidy)}); no sentence edge "
                        f"to move it to, the gate judges it"), clip_id=cid)
            # tidy_cut reads the burned words only; the gate's ending also reads its
            # second ASR, which can hear a filler just before a cut the words end cleanly on. In
            # a window only: the re-cut after the gate is time-capped there (outside one it
            # always runs)
            if side is not None and cut_tidy_enabled() and WINDOW_DEADLINE is not None:
                s1, e1 = new_cut or (s0, e0)
                cut2, side2, end_info = _asr2_end_fix(clip, media, media_dur, side, s1, e1, vc,
                                                      _lean_gate(clip))
                if end_info:
                    info["asr2_end"] = end_info
                new_cut = cut2 or new_cut
                side = side if side2 is None else side2
        if info.get("seat"):
            _log("info", "render", f"clip {cid}: caption word(s) seated on the audio at the "
                 f"cuts: {_seat_note(info['seat'])}", clip_id=cid)
    except NoCleanCut as e:
        e.info = {"model": CAPTION_STRONG_MODEL, "mode": "full", "clean_cut": {**e.info,
                  "drop": str(e), "side": e.side}}
        raise
    except Exception as e:                # noqa: BLE001 - select's words stay
        side, info = None, {"model": CAPTION_STRONG_MODEL, "mode": "error",
                            "error": f"{type(e).__name__}: {str(e)[:160]}"}
        new_cut = None
    feats = _features_merge(clip, captions=info)
    cut = {"start_s": new_cut[0], "end_s": new_cut[1]} if new_cut else {}
    if side is None:
        runstate.update_clip(cid, features_json=feats, **cut)
        _log("info", "render", f"clip {cid}: caption words stay select's ({info.get('mode')}: "
             f"{info.get('reason') or info.get('error') or ''})", clip_id=cid)
        return {**clip, "features_json": feats, **cut}
    wp = SCRATCH / "words" / f"{clip['source_id']}_{float(clip['start_s']):.2f}_cap.json"
    wp.parent.mkdir(parents=True, exist_ok=True)
    wp.write_text(json.dumps(side))
    runstate.update_clip(cid, words_path=str(wp), features_json=feats, **cut)
    _log("info", "render", f"clip {cid}: caption words from {info['model']} ({info['mode']}, "
         f"{info['heard_s']}s heard in {info['seconds']}s, {info['captioned']}/{info['words']} "
         f"burned)", clip_id=cid)
    return {**clip, "words_path": str(wp), "features_json": feats, **cut}


def _next_alternate(clip, why):
    """the clip row moved onto its next alternate (select's dropped pick over the
    same moment, features "alternates") for a render that found no clean cut: its cut, title and
    sidecar, the cut review / caption / final-cut records of the old cut set aside
    ("replaced": [{from, hook, why}]). None when there is none left (MAX_ALTERNATES tried), it
    overlaps another clip of the source by more than OVERLAP_MAX or starts where one does, or the
    window has no time for one more caption step (_review_room)."""
    cid = clip["clip_id"]
    try:
        feats = json.loads(clip.get("features_json") or "{}") or {}
    except (TypeError, ValueError):
        return None
    alts = list(feats.get("alternates") or [])
    done = list(feats.get("replaced") or [])
    if not alts or len(done) >= MAX_ALTERNATES:
        return None
    need = REVIEW_LLM_S + CAPTION_RESERVE_S + _strong_cost(
        float(alts[0]["end"]) - float(alts[0]["start"]) + sum(caption_context()[:2]))
    late = _review_room(clip, need)
    if late:
        _log("info", "render", f"clip {cid}: no alternate ({late})", clip_id=cid)
        return None
    others = [c for c in runstate.clips_for_source(clip["source_id"]) if c["clip_id"] != cid]
    while alts:
        a = alts.pop(0)
        s, e = float(a["start"]), float(a["end"])
        clash = [o["clip_id"] for o in others if abs(float(o["start_s"]) - s) < 1e-6 or
                 boundaries.iou({"start": s, "end": e},
                                {"start": float(o["start_s"]), "end": float(o["end_s"])}) > OVERLAP_MAX]
        if clash:
            continue
        rec = {"from": [round(float(clip["start_s"]), 3), round(float(clip["end_s"]), 3)],
               "hook": clip.get("hook"), "why": str(why)[:240],
               "cut_review": feats.get("cut_review"), "final_cut": feats.get("final_cut")}
        for k in ("cut_review", "final_cut", "captions", "cut_seat"):
            feats.pop(k, None)
        feats.update(alternates=alts, replaced=done + [rec],
                     alternate={k: a.get(k) for k in ("start", "end", "hook", "score")})
        fj = json.dumps(feats, default=str)
        upd = {"start_s": s, "end_s": e, "hook": a.get("hook") or clip.get("hook"),
               "words_path": a.get("words_path") or clip.get("words_path"), "features_json": fj}
        try:
            runstate.update_clip(cid, **upd)
        except Exception as ex:           # noqa: BLE001 - a clash the store refuses: the next one
            _log("warn", "render", f"clip {cid}: alternate {s:.2f}-{e:.2f}s refused ({ex})",
                 clip_id=cid)
            continue
        _log("info", "render", f"clip {cid}: no clean cut on {rec['from'][0]:.2f}-{rec['from'][1]:.2f}s "
             f"({str(why)[:160]}); the next candidate {s:.2f}-{e:.2f}s {a.get('hook')!r} in its "
             f"place", clip_id=cid)
        return {**clip, **upd}
    return None


def _drop_unclean(clip, e):
    """a clip with no clean cut inside its limits leaves before its render, its
    reject coded like the gate's cut failures (cut_start / cut_end)."""
    cid = clip["clip_id"]
    why = f"no clean cut on the strong words: {e}"
    codes = {"start": ["cut_start"], "end": ["cut_end"]}.get(e.side, ["cut_start", "cut_end"])
    feats = _features_merge(clip, **(e.info if "final_cut" in (e.info or {}) else {"captions": e.info}),
                            reason_codes={c: f"snap: {why}"[:300] for c in codes})
    runstate.update_clip(cid, state="failed", error=f"gate: snap: {why}"[:400], features_json=feats)
    _log("warn", "render", f"clip {cid} DROPPED before render: {why}", clip_id=cid,
         source_id=clip.get("source_id"))


# ---- RENDER: the editor's pass over the cut, before the render and after a reject ----
REVIEW_CODES = ("cut_start", "cut_end", "no_hook", "wrong_hook")   # the faults it moves or rewrites
REVIEW_LLM_S = 45.0          # the review's share of the time check (one claude call: 6-15 s measured)


def _window_words(media):
    """select's whole-window words (TRANSCRIBE_MODEL: transcript.get_words, cached by select);
    [] when they can't be had."""
    try:
        return transcript.get_words(media, model=TRANSCRIBE_MODEL)
    except Exception:                     # noqa: BLE001 - no transcript: no review
        return []


def _cached_window_words(media, model=None):
    """select's whole-window words (or `model`'s) from the transcript cache only (never
    transcribes); []."""
    import hashlib
    try:
        key = hashlib.md5(f"{media}|{os.path.getmtime(media)}|{model or TRANSCRIBE_MODEL}"
                          .encode()).hexdigest()
        return json.loads((Path(transcript.SCRATCH) / "transcripts" / f"{key}.json").read_text())
    except (OSError, ValueError, TypeError):
        return []


def _window_tracks(media):
    """The window transcripts the cache holds (select's TRANSCRIBE_MODEL pass, the repair lane's
    REFINE_MODEL pass): the eye-check reads its window transcript from the same cache."""
    return [t for t in (_cached_window_words(media), _cached_window_words(media, REFINE_MODEL))
            if t]


def _window_rms(media, media_dur):
    """The window audio's 10 ms frame RMS from 0 s, or None."""
    try:
        hi = float(media_dur) if media_dur else 900.0
        return boundaries.frame_rms(transcript.load_audio(media, 0.0, hi), SR, 0.01)
    except Exception:                     # noqa: BLE001 - pad cuts without audio
        return None


def _review_room(clip, need_s):
    """None when the window's time check lets a review (and what it may cost) run, else why not."""
    if WINDOW_DEADLINE is None:
        return None
    waiting = sum(1 for c in runstate.clips_for_source(clip["source_id"])
                  if c["state"] == "selected" and c["clip_id"] != clip["clip_id"])
    left = WINDOW_DEADLINE - time.monotonic() - CAPTION_RESERVE_S * waiting
    return None if left >= need_s else f"{left:.0f}s of the window left, ~{need_s:.0f}s needed"


def _cut_review(clip, src, media, media_dur, feedback=None, llm=None, words_fn=None, rms_fn=None,
                need_s=REVIEW_LLM_S, title_only=False):
    """(clip, review or None): cut_review.review on the clip's cut (the window's words around it,
    its own sidecar inside), the moved cut and its sidecar or the new title on the clip row.
    Sentence clips only; a review that cannot run (off, no time, no words, an LLM error) leaves
    the clip exactly as it is. `feedback` = the reject reasons of a render that failed;
    `title_only`: the cut is fixed, only the title is asked for."""
    from clipengine import cut_review
    st = cut_review.settings()
    cid = clip["clip_id"]
    if not st.get("enabled") or (clip.get("boundary_type") or "sentence") != "sentence":
        return clip, None
    why = _review_room(clip, need_s)
    if why:
        _log("info", "render", f"clip {cid}: no cut review ({why})", clip_id=cid)
        return clip, None
    vertical = (src or {}).get("vertical") or "general"
    # the niche's pack, the end rule, and the clip's topic (select's map in its
    # features, else the source's cached map)
    niche = _niche_of(clip["source_id"])
    vc = {**vertical_cfg(vertical), **topic_map.select_cfg(niche)}
    start, end = float(clip["start_s"]), float(clip["end_s"])
    try:
        feats0 = json.loads(clip.get("features_json") or "{}") or {}
    except (TypeError, ValueError):
        feats0 = {}
    topics, topic = topic_map.clip_topic(feats0, start, end, cache_dir=SCRATCH / "topic_maps",
                                         sid=clip["source_id"]) \
        if topic_map.settings().get("enabled") else ([], None)
    side = _load_words_sidecar(clip.get("words_path"))
    window = (words_fn or _window_words)(media)
    words = cut_review.merged_words(window, side, start, end)
    rms = rms_fn(media, media_dur) if rms_fn else _window_rms(media, media_dur)
    alt = [w for t in [window] + (_window_tracks(media) if words_fn is None else []) for w in t]
    r = cut_review.review(clip, words, vertical, vc, llm=llm or cut_review.claude_llm(vc, st),
                          rms=rms, t0=0.0, media_dur=media_dur, feedback=feedback, st=st,
                          title_only=title_only, alt_words=alt, topics=topics, topic=topic,
                          pack=topic_map.pack(niche))
    rec = {k: r.get(k) for k in ("verdict", "from", "start", "end", "hook", "hook_changed",
                                 "lines", "answer", "why", "seconds", "rules")}
    if topic:
        rec["topic"] = topic.get("i")
    rec.update(old_hook=clip.get("hook"), old_words_path=clip.get("words_path"))
    if title_only:
        rec["title_only"] = True
    if feedback:
        rec["feedback"] = str(feedback)[:400]
    try:
        hist = list((json.loads(clip.get("features_json") or "{}") or {}).get("cut_review") or [])
    except (TypeError, ValueError):
        hist = []
    upd = {}
    if r["verdict"] == "move":
        wp = SCRATCH / "words" / f"{clip['source_id']}_{start:.2f}_rev{len(hist) + 1}.json"
        wp.parent.mkdir(parents=True, exist_ok=True)
        wp.write_text(json.dumps(cut_review.sidecar(words, r["start"], r["end"])))
        upd.update(start_s=r["start"], end_s=r["end"], words_path=str(wp))
    if r.get("hook_changed") and r.get("hook"):
        upd["hook"] = r["hook"]
    feats = _features_merge(clip, cut_review=hist + [rec])
    runstate.update_clip(cid, features_json=feats, **upd)
    if r["verdict"] == "move":
        _log("info", "render", f"clip {cid}: cut review moved {start:.2f}-{end:.2f}s -> "
             f"{r['start']:.2f}-{r['end']:.2f}s ({r['why']})", clip_id=cid)
    else:
        _log("info", "render", f"clip {cid}: cut review {r['verdict']} ({r.get('why')})",
             clip_id=cid)
    if upd.get("hook"):
        _log("info", "render", f"clip {cid}: hook {clip.get('hook')!r} -> {upd['hook']!r} "
             f"(cut review)", clip_id=cid)
    return {**clip, **upd, "features_json": feats}, r


def _title_after_cut_moves(clip, src, media, media_dur, before):
    """the caption step can move a cut after its title was written (11-09 2786: the
    review titled 2.45-15.97 s 'He'll dance anywhere for prize money', the tidy then pulled the end
    back to 14.13 s, before 'prize of money'). When the cut moved from `before` (start, end) and
    the title now uses a word said only outside it (cut_review.leaked_words, the clip's burned
    words inside, the window's around), the review writes the title again for the cut as it is
    (title only). Returns the clip."""
    from clipengine import cut_review
    hook = clip.get("hook")
    s, e = float(clip["start_s"]), float(clip["end_s"])
    if not hook or (round(s, 3), round(e, 3)) == tuple(round(float(x), 3) for x in before):
        return clip
    side = _load_words_sidecar(clip.get("words_path"))
    inside, outside = cut_review.split_by_cut(s, e, side, *_window_tracks(media),
                                              lo=s - 40.0, hi=e + 25.0)
    leak = cut_review.leaked_words(hook, inside, outside)
    if not leak:
        return clip
    _log("info", "render", f"clip {clip['clip_id']}: the cut moved {before[0]:.2f}-{before[1]:.2f}s "
         f"-> {s:.2f}-{e:.2f}s and the title {hook!r} now uses {', '.join(leak)}, said outside it; "
         f"the review writes it again", clip_id=clip["clip_id"])
    new, _r = _cut_review(clip, src, media, media_dur, title_only=True)
    return new


def _undo_review_move(clip, src, media, media_dur, why):
    """The clip back on the cut the review moved it from (its sidecar and its title; a clip that
    had no title asks the review for one on that cut, title_only), the cuts seated and the caption
    step run again for it. None when the review did not move this clip's cut, it was undone
    already, or the caption step drops it."""
    cid = clip["clip_id"]
    try:
        feats = json.loads(clip.get("features_json") or "{}") or {}
    except (TypeError, ValueError):
        return None
    hist = list(feats.get("cut_review") or [])
    # the latest move (a title-only review after it, 2786's, does not count as one)
    k = next((i for i in range(len(hist) - 1, -1, -1) if hist[i].get("verdict") == "move"), None)
    last = hist[k] if k is not None else None
    if not last or last.get("undone") or not last.get("from"):
        return None
    s0, e0 = (float(x) for x in last["from"])
    hist[k] = {**last, "undone": f"the render refused the moved cut: {str(why)[-200:]}"}
    feats["cut_review"] = hist
    fj = json.dumps(feats, default=str)
    wp = last.get("old_words_path") or clip.get("words_path")
    hook = last.get("old_hook")
    runstate.update_clip(cid, start_s=s0, end_s=e0, words_path=wp, hook=hook, features_json=fj)
    _log("info", "render", f"clip {cid}: the render refused the reviewed cut "
         f"{float(clip['start_s']):.2f}-{float(clip['end_s']):.2f}s ({str(why)[-120:]}); back to "
         f"{s0:.2f}-{e0:.2f}s, render once more", clip_id=cid)
    back = {**clip, "start_s": s0, "end_s": e0, "words_path": wp, "hook": hook,
            "features_json": fj}
    if not hook:
        back, _r = _cut_review(back, src, media, media_dur, title_only=True)
    back = _seat_cuts(back, media, media_dur)
    before = (back["start_s"], back["end_s"])
    try:
        back = _caption_step(back, media, media_dur)
        back = _final_cut(back, src, media, media_dur)
    except NoCleanCut:
        return None
    return _title_after_cut_moves(back, src, media, media_dur, before)


def cut_seat_enabled(cfg=None):
    """config daily.cut_seat.enabled (default True): boundaries.seat_cuts before the caption pass
    (the render lane) and before the recaption (the repair lane). CUT_SEAT=0 turns it off."""
    if os.environ.get("CUT_SEAT", "").strip() == "0":
        return False
    block = (((cfg if cfg is not None else CFG).get("daily") or {}).get("cut_seat") or {})
    return bool(block.get("enabled", True))


def seat_on_audio(start, end, media, media_dur, words=None, rms_fn=None):
    """((start, end), info) of boundaries.seat_cuts on the window audio around both cuts; the cut
    as it is (and info {}) when the audio can't be read. Never raises."""
    pad = boundaries.CUT_AUDIO_S + 1.0
    lo = max(0.0, float(start) - pad)
    hi = float(end) + pad if not media_dur else min(float(media_dur), float(end) + pad)
    try:
        rms = rms_fn(media, lo, hi) if rms_fn else \
            boundaries.frame_rms(transcript.load_audio(media, lo, hi), SR, 0.01)
        if words:
            words, _spanned = boundaries.span_short_words(words, rms, lo)
        return boundaries.seat_cuts(start, end, rms, t0=lo, media_dur=media_dur, words=words)
    except Exception as e:                # noqa: BLE001 - the cut stays
        return (float(start), float(end)), {"error": f"{type(e).__name__}: {str(e)[:120]}"}


def _seat_cuts(clip, media, media_dur, words_fn=None, rms_fn=None):
    """the clip's cuts seated on the window audio before its caption
    pass (seat_on_audio, the words around the cut from its sidecar and select's window pass), the
    moved cut on the clip row. Sentence clips only; returns the clip (updated or as it was)."""
    cid = clip["clip_id"]
    if not cut_seat_enabled() or (clip.get("boundary_type") or "sentence") != "sentence":
        return clip
    from clipengine import cut_review
    s0, e0 = float(clip["start_s"]), float(clip["end_s"])
    words = cut_review.merged_words((words_fn or _cached_window_words)(media),
                                    _load_words_sidecar(clip.get("words_path")), s0, e0)
    (s, e), info = seat_on_audio(s0, e0, media, media_dur, words=words, rms_fn=rms_fn)
    if not info:
        return clip
    feats = _features_merge(clip, cut_seat=info)
    upd = {"start_s": s, "end_s": e} if (s, e) != (s0, e0) else {}
    runstate.update_clip(cid, features_json=feats, **upd)
    if upd:
        parts = [f"{k} {v['from']:.3f} -> {v['to']:.3f}s ({v['how']})" for k, v in info.items()
                 if isinstance(v, dict) and "to" in v]
        _log("info", "render", f"clip {cid}: cut seated on the audio: {'; '.join(parts)}",
             clip_id=cid)
    return {**clip, **upd, "features_json": feats}


# ---- RENDER: crosstalk the second ASR hears is burned ----
CROSSTALK_CONFIRM = {"enabled": True}


def crosstalk_confirm_enabled(cfg=None):
    """config render.crosstalk_confirm.enabled over CROSSTALK_CONFIRM; CROSSTALK_CONFIRM=0 off."""
    if os.environ.get("CROSSTALK_CONFIRM", "").strip() == "0":
        return False
    block = (((cfg if cfg is not None else CFG).get("render") or {}).get("crosstalk_confirm") or {})
    return bool({**CROSSTALK_CONFIRM, **block}.get("enabled"))


def _confirm_crosstalk(clip, media, rec_fn=None):
    """The clip's crosstalk-flagged caption phrases ("cap": False: select's or the caption step's
    crosstalk filter) that the gate's second ASR also hears burned again (crosstalk.
    confirm_crosstalk), on the cut the render burns, just before it: the pass is the gate's own
    (second_asr.transcribe, cached for the caption check and the gate). The new sidecar goes on
    the clip row. Never raises; off, no flagged phrase, no time for the pass or no second ASR:
    the clip as it is."""
    cid = clip["clip_id"]
    if not crosstalk_confirm_enabled():
        return clip
    from clipengine import crosstalk
    from clipengine import second_asr
    s, e = float(clip["start_s"]), float(clip["end_s"])
    side = _load_words_sidecar(clip.get("words_path"))
    flagged = [w for w in side if w.get("cap", True) is False and w.get("fix") != "unheard"
               and s <= _mid(w) <= e]
    if not flagged:
        return clip
    try:
        if rec_fn is None and second_asr.cached(media, s, e) is None:
            late = _asr_room(clip, ASR2_RTF * (e - s + 2.0))
            if late:
                _log("info", "render", f"clip {cid}: crosstalk not checked on the second ASR "
                     f"({late})", clip_id=cid)
                return clip
        rec = (rec_fn or second_asr.transcribe)(media, s, e)
    except Exception as ex:               # noqa: BLE001 - the flags stand
        _log("info", "render", f"clip {cid}: crosstalk not checked on the second ASR "
             f"({type(ex).__name__}: {str(ex)[:120]})", clip_id=cid)
        return clip
    heard = [{**w, "start": float(w["start"]) + s, "end": float(w["end"]) + s}
             for w in rec.get("words") or []]
    new, rep = crosstalk.confirm_crosstalk(side, heard)
    feats = _features_merge(clip, crosstalk_confirm={**rep, "flagged": len(flagged)})
    if not rep["burned"]:
        runstate.update_clip(cid, features_json=feats)
        return {**clip, "features_json": feats}
    wp = SCRATCH / "words" / f"{clip['source_id']}_{s:.2f}_xt.json"
    wp.parent.mkdir(parents=True, exist_ok=True)
    wp.write_text(json.dumps(new))
    runstate.update_clip(cid, words_path=str(wp), features_json=feats)
    _log("info", "render", f"clip {cid}: crosstalk the second ASR hears burned again: "
         + "; ".join(f"{b['start']:.2f}-{b['end']:.2f}s {b['text'][:60]!r}" for b in rep["burned"]),
         clip_id=cid)
    return {**clip, "words_path": str(wp), "features_json": feats}


# ---- RENDER: the caption track's confidence before it is burned ----
# captions_garbled was the second-largest reject (gate 90, eye-check 48, in-run 47 in 7 days).
# Its causes, read on the rejects: (1) the strong pass skips speech under game / crowd audio and
# times words to nothing (11-10 tw_2916500966 1100002: 22 words for 34 s of talk, "no captions
# from 2.6 to 11.9 s"); (2) the post-render caption repair crammed a long stretch into a short
# span (caption_repair.apply_votes, fixed there); (3) a phrase select's words did not confirm left
# unburned (crosstalk); (4) plain mishearings (names, jargon). Before the render the burned track
# is read the way the gate will read it (its caption check on the second ASR, the pass it then
# reads from its cache) and for timing faults; a low-confidence track is re-heard: the window
# around the cut re-transcribed with the strong model (more context: 11-10 tw_2916500966 1100002
# went from 22 words heard with 1.5 s of lead to 72 with the window's), each phrase kept when select's words or the second ASR confirm it, then the
# tie-break vote of the caption repair. The best-reading track is burned; the cut never moves.
CAPTION_CHECK = {"enabled": True, "lead_s": 12.0, "tail_s": 2.5, "vote": True, "whisper_args": []}
UNCAPTIONED_RUN = 6          # this many words the second ASR heard in a row with no caption near
CRAM_WPS = 10                # caption words inside one second: crammed (fast talk reaches 8: 11-10
                             # LvAHAZOm6Zw; the vote's cramming burned ~27 a second)


def caption_check_settings(cfg=None):
    """render.caption_check over CAPTION_CHECK; CAPTION_CHECK=0 turns it off."""
    st = {**CAPTION_CHECK, **(((cfg if cfg is not None else CFG).get("render") or {})
                             .get("caption_check") or {})}
    st.pop("_doc", None)
    if os.environ.get("CAPTION_CHECK", "").strip() == "0":
        st["enabled"] = False
    return st


def caption_confidence(side, start, end, asr_words, third_words=None, third_model=None):
    """(score, problems, verdict) of the track a clip burns (its sidecar over [start, end]) against
    the second ASR's words (clip-relative): the gate's caption check (quality_gate.
    caption_agreement_verdict; with `third_words`, the tie-break model's words over the same
    span, its tie-break: quality_gate.caption_tiebreak_verdict) and the track's timing: more than
    CRAM_WPS caption words in a second, words with no duration (>= 3 and >= 10 %),
    UNCAPTIONED_RUN heard words in a row with no caption within 0.6 s. score = the lower of
    precision and coverage, 0.1 off per timing problem (1.0 when the second ASR heard nothing)."""
    start, end = float(start), float(end)
    dur = end - start
    cap = quality_gate.clip_words(side, start, end)
    heard = [w for w in asr_words or [] if float(w["start"]) < dur]
    v = quality_gate.caption_agreement_verdict(cap, heard, dur)
    if third_words is not None:
        v = quality_gate.caption_tiebreak_verdict(
            cap, heard, [w for w in third_words if float(w["start"]) < dur], dur, base=v,
            model=third_model)
    problems = list(v.get("errors") or [])
    starts = sorted(float(w["start"]) for w in cap)
    j = worst = 0
    for i in range(len(starts)):
        while starts[i] - starts[j] > 1.0:
            j += 1
        worst = max(worst, i - j + 1)
    timing = []
    if worst > CRAM_WPS:
        timing.append(f"{worst} caption words inside one second")
    zero = sum(1 for w in cap if float(w["end"]) - float(w["start"]) < 0.03)
    if cap and zero >= max(3, 0.1 * len(cap)):
        timing.append(f"{zero} of {len(cap)} caption words last no time")
    run = best = 0
    for w in heard:
        hs = float(w["start"])
        covered = any(float(c["start"]) - 0.6 <= hs <= float(c["end"]) + 0.6 for c in cap)
        run = 0 if covered else run + 1
        best = max(best, run)
    if best >= UNCAPTIONED_RUN:
        timing.append(f"{best} words the second ASR heard in a row have no caption")
    score = 1.0 if v.get("passed") is None else min(float(v.get("precision") or 0.0),
                                                     float(v.get("coverage") or 0.0))
    return round(score - 0.1 * len(timing), 3), problems + timing, v


_PROMPT_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                                "\u2013": "-", "\u2014": "-"})   # a curly ' in the prompt came
                                # back inside words ("you\u2019 think", 0VhFE_KbgDM 11-11)


def vocab_prompt(src):
    """whisper's initial prompt for a re-heard caption track: the source's title (and channel),
    which names the people, teams and products the clip talks about. The eye-check's
    captions_garbled reasons are mostly such words misheard (7 days to 11-11: 36 of 48, 'cloud'
    for Claude, 'WBA' for WNBA, 'AOC' for AMC). None when there is no title."""
    def clean(x):
        x = str(x or "").translate(_PROMPT_QUOTES)
        x = re.sub(r"\s*\|\s*watch\?v=\S+", "", x)
        x = re.sub(r"[^\w\s.,!?'&%$#:;()/+-]", " ", x)       # emoji and odd marks out
        x = re.sub(r"([.!?])[.!?]+", r"\1", " ".join(x.split()))    # 'saying..' -> 'saying.'
        return x.strip(" .")
    out = ". ".join(x for x in (clean((src or {}).get("title"))[:200],
                                clean((src or {}).get("channel"))[:60]) if x)
    return f"{out}." if out else None


def _rehear_track(clip, media, media_dur, side, rec, st, strong_fn=None, rms_fn=None, src=None):
    """(sidecar, info) of the window around the cut re-transcribed with the strong model
    (st lead_s / tail_s around it, st whisper_args, the source's title as whisper's initial
    prompt: vocab_prompt), its words seated on the audio, a phrase burned when select's words or
    the second ASR confirm it (crosstalk.flag_crosstalk on both)."""
    from clipengine import crosstalk
    s, e = float(clip["start_s"]), float(clip["end_s"])
    lo = max(0.0, s - float(st["lead_s"]))
    hi = e + float(st["tail_s"])
    if media_dur:
        hi = min(float(media_dur), hi)
    left = None if WINDOW_DEADLINE is None else max(20.0, WINDOW_DEADLINE - time.monotonic())
    if strong_fn is not None:
        got = strong_fn(media, lo, hi, left)
    else:
        got = strong_asr.transcribe_strong(media, lo, hi, model=CAPTION_STRONG_MODEL, timeout=left,
                                           prompt=vocab_prompt(src), extra_args=st.get("whisper_args"))
    got, moved = _seat_heard(media, lo, hi, got, rms_fn=rms_fn)
    inside = [w for w in got if s - CLIP_WORD_TOL_S <= _mid(w) <= e + CLIP_WORD_TOL_S]
    outside = [w for w in got if w not in inside]
    asr_abs = [{**w, "start": float(w["start"]) + s, "end": float(w["end"]) + s}
               for w in rec.get("words") or []]
    ref = sorted([w for w in side if str(w.get("w", "")).strip()] + asr_abs,
                 key=lambda w: float(w["start"]))
    flagged, rep = crosstalk.flag_crosstalk(inside, ref)
    new = sorted(outside + flagged, key=lambda w: float(w["start"]))
    return new, {"heard": [round(lo, 3), round(hi, 3)], "words": len(inside),
                 "seated": len(moved), "dropped": rep.get("dropped") or [],
                 "prompt": vocab_prompt(src)}


def _caption_check(clip, src, media, media_dur, strong_fn=None, rec_fn=None, vote_fn=None,
                   rms_fn=None, third_fn=None):
    """the clip's caption track read before it is burned (caption_confidence on
    the second ASR the gate reads from its cache); a track with a problem is re-heard
    (_rehear_track), then the caption repair's tie-break vote runs on the better one
    (caption_repair.reconcile), and the best-reading track whose cut the gate's checks still pass
    on its words is burned (the sidecar on the clip row). Never raises and never moves the cut;
    no time in the window, an off switch or no second ASR keep the track as it is.
    a track the second ASR's check fails is read against the gate's
    tie-break model too (quality_gate.run_caption_tiebreak, the pass the gate then reads from its
    cache; `third_fn(media, s, e)` in tests), so a track only `small` disputes is not re-heard."""
    cid = clip["clip_id"]
    st = caption_check_settings()
    if not st.get("enabled") or (clip.get("boundary_type") or "sentence") != "sentence":
        return clip
    s, e = float(clip["start_s"]), float(clip["end_s"])
    late = _asr_room(clip, 0.0)
    if late:
        return clip
    side = _load_words_sidecar(clip.get("words_path"))
    lean = _lean_gate(clip)
    try:
        from clipengine import second_asr
        rec = (rec_fn or second_asr.transcribe)(media, s, e)     # the gate's own (cached) pass
        base = caption_confidence(side, s, e, rec.get("words"))
    except Exception as ex:                # noqa: BLE001 - the track stays
        _log("info", "render", f"clip {cid}: no caption check ({type(ex).__name__}: "
             f"{str(ex)[:120]})", clip_id=cid)
        return clip
    third, tb = _caption_third(clip, media, s, e, base[2], rec, third_fn)
    if third is not None:
        base = caption_confidence(side, s, e, rec.get("words"), third_words=third,
                                  third_model=tb.get("model"))
    info = {"score": base[0], "problems": base[1][:6]}
    if tb:
        info["tiebreak"] = {**tb, **({"verdict": (base[2].get("tiebreak") or {}).get("verdict")}
                                     if third is not None else {})}
    if not base[1]:
        feats = _features_merge(clip, caption_check={**info, "kept": "clean"})
        runstate.update_clip(cid, features_json=feats)
        return {**clip, "features_json": feats}
    tail = float(rec.get("tail_s") or 0.0)
    audio = quality_gate.safe_cut_audio(media, s, e)

    def fits(track):
        """The cut still passes the gate's checks on `track` (words, and the ending read on the
        second ASR's words with the track's punctuation)."""
        return not _word_cut_errors(track, s, e, lean, audio=audio) and \
            not _predicted_cut_errors(track, rec.get("words") or [], tail, s, e, lean,
                                      audio=audio)
    ok_before = fits(side)
    best = (base[0], side, "as select / the caption step left it", base)
    tried = []
    need = _strong_cost(e - s + float(st["lead_s"]) + float(st["tail_s"]))
    if not _asr_room(clip, need):
        try:
            track, how = _rehear_track(clip, media, media_dur, side, rec, st, strong_fn=strong_fn,
                                       rms_fn=rms_fn, src=src)
            got = caption_confidence(track, s, e, rec.get("words"), third_words=third,
                                     third_model=tb.get("model"))
            good = fits(track) or not ok_before
            tried.append({"how": "reheard", "score": got[0], "problems": got[1][:4],
                          "fits": good, **{k: how[k] for k in ("heard", "words", "seated", "prompt")}})
            if good and got[0] > best[0]:
                best = (got[0], track, "the window re-heard by the strong model", got)
        except Exception as ex:            # noqa: BLE001 - the next step
            tried.append({"how": "reheard", "error": f"{type(ex).__name__}: {str(ex)[:120]}"})
    if best[3][1] and st.get("vote") and not _asr_room(clip, CAPTION_VOTE_S):
        try:
            from clipengine import caption_repair
            deadline = None if WINDOW_DEADLINE is None else WINDOW_DEADLINE - CAPTION_RESERVE_S
            fn = vote_fn or (lambda sd: caption_repair.reconcile(
                sd, s, e, rec.get("words") or [], media=media, source=src or {},
                deadline=deadline))
            track, rep = fn(best[1])
            if track is not None:
                got = caption_confidence(track, s, e, rec.get("words"), third_words=third,
                                         third_model=tb.get("model"))
                good = fits(track) or not ok_before
                tried.append({"how": "vote", "score": got[0], "problems": got[1][:4], "fits": good,
                              "fixes": len(rep.get("fixes") or [])})
                if good and got[0] > best[0]:
                    best = (got[0], track, "the tie-break vote", got)
            else:
                tried.append({"how": "vote", "none": str(rep.get("reason") or "")[:120]})
        except Exception as ex:            # noqa: BLE001 - the best so far stands
            tried.append({"how": "vote", "error": f"{type(ex).__name__}: {str(ex)[:120]}"})
    info["tried"] = tried
    if best[1] is side:
        feats = _features_merge(clip, caption_check={**info, "kept": "no better track"})
        runstate.update_clip(cid, features_json=feats)
        _log("info", "render", f"clip {cid}: caption check: {'; '.join(base[1])[:200]}; no "
             f"better track ({len(tried)} tried)", clip_id=cid)
        return {**clip, "features_json": feats}
    wp = SCRATCH / "words" / f"{clip['source_id']}_{s:.2f}_cc.json"
    wp.parent.mkdir(parents=True, exist_ok=True)
    wp.write_text(json.dumps(best[1]))
    info.update(chose=best[2], new_score=best[0], new_problems=best[3][1][:4])
    feats = _features_merge(clip, caption_check=info)
    runstate.update_clip(cid, words_path=str(wp), features_json=feats)
    _log("info", "render", f"clip {cid}: caption check {base[0]} -> {best[0]} ({best[2]}): "
         f"{'; '.join(base[1])[:200]}", clip_id=cid)
    return {**clip, "words_path": str(wp), "features_json": feats}


TIEBREAK_RTF = 2.0 * ASR2_RTF  # the tie-break model (medium) runs ~2x small's CPU


def _caption_third(clip, media, s, e, verdict, rec, third_fn=None):
    """(the tie-break model's clip-relative words or None, info) for a track whose caption check
    on the second ASR failed (`verdict`), when quality_gate.caption_tiebreak is on and the window
    has the time; the gate reads the same pass from its cache (quality_gate.
    run_caption_tiebreak). Never raises."""
    if (verdict or {}).get("passed") is not False:
        return None, {}
    st = quality_gate.caption_tiebreak_settings()
    if not st.get("enabled"):
        return None, {}
    info = {"model": st["model"]}
    late = _asr_room(clip, TIEBREAK_RTF * (e - s + float(rec.get("tail_s") or 0.0)))
    if late:
        return None, {**info, "skipped": late}
    try:
        words = third_fn(media, s, e) if third_fn else \
            quality_gate.run_caption_tiebreak(media, s, e, st["model"])
    except Exception as ex:               # noqa: BLE001 - the second ASR's check stands
        return None, {**info, "error": f"{type(ex).__name__}: {str(ex)[:120]}"}
    return words, info


# ---- RENDER: the final cut on the burned words ----
FINAL_VERIFY = 2             # moved cuts checked on the gate's own second-ASR pass, at most


def final_cut_enabled(cfg=None):
    """config daily.final_cut.enabled (default True); FINAL_CUT=0 turns it off."""
    if os.environ.get("FINAL_CUT", "").strip() == "0":
        return False
    block = (((cfg if cfg is not None else CFG).get("daily") or {}).get("final_cut") or {})
    return bool(block.get("enabled", True))


def _context_words(window, side):
    """The sidecar's words, with the window transcript's only where the sidecar has none: before
    its first word and after its last (a window word overlapping the sidecar's span is another
    model's timing of the same speech: 11-10 lvJjUyKnOl0's 'in' twice)."""
    sw = [dict(w) for w in side or [] if str(w.get("w", "")).strip()]
    if not sw:
        return [dict(w) for w in window or []]
    lo = min(float(w["start"]) for w in sw)
    hi = max(float(w["end"]) for w in sw)
    out = [dict(w) for w in window or [] if float(w["end"]) <= lo or float(w["start"]) >= hi] + sw
    return sorted(out, key=lambda w: (float(w["start"]), float(w["end"])))


def _closes_at(rec, s0, window, media=None):
    """End times (absolute) of the words that close a sentence (. ? !) in the other transcripts
    of the window: the second ASR pass `rec` (clip-relative from s0), select's window words and
    the cached refine pass (media given); boundaries.mark_closes reads an unpunctuated burned
    track with them."""
    out = []
    for w in (rec or {}).get("words") or []:
        if boundaries._ends_clean(w["w"]):
            out.append(float(w["end"]) + float(s0))
    tracks = [window or []] + ([_cached_window_words(media, REFINE_MODEL)] if media else [])
    for t in tracks:
        out += [float(w["end"]) for w in t if boundaries._ends_clean(w["w"])]
    return out


def _word_cut_errors(side, start, end, lean, audio=None):
    """gate_cut_errors without the second ASR: the full gate's word_boundaries and
    sentence_bounds on the sidecar ([] for the stream lean gate, which runs neither); `audio` =
    the cut's quality_gate.cut_audio frames, when the caller has them."""
    if lean:
        return []
    errs = []
    for k, m in (("word_boundaries", quality_gate.measure_word_boundaries(side, start, end,
                                                                         audio=audio)),
                 ("sentence_bounds", quality_gate.measure_sentence_bounds(side, start, end,
                                                                         audio=audio))):
        if m.get("passed") is False:
            errs += [f"{k}: {e}" for e in m.get("errors") or ["failed"]]
    return errs


def _asr_shift(rec, s0, s2):
    """The second ASR record of the cut from `s0` read as one from `s2` (its words clip-relative
    to s2; those before s2 dropped)."""
    d = float(s0) - float(s2)
    words = [{**w, "start": float(w["start"]) + d, "end": float(w["end"]) + d}
             for w in rec.get("words") or [] if float(w["start"]) + d >= -0.05]
    return {**rec, "words": words}


def _predicted_ending(side, rec, s0, e0, s2, e2, lean, audio=None):
    """The gate's cut checks on [s2, e2] read on `side` and on the second ASR record `rec` of
    [s0, e0] (no new pass), or None when that pass did not hear far enough past e2 (the gate
    needs ENDING_MIN_TAIL_S) or started after s2."""
    tail = float(e0) + float(rec.get("tail_s") or 0.0) - float(e2)
    if s2 < s0 - 0.01 or tail < quality_gate.ENDING_MIN_TAIL_S:
        return None
    shifted = _asr_shift(rec, s0, s2)
    return _predicted_cut_errors(side, shifted["words"], tail, s2, e2, lean=lean, audio=audio)


def _rms_cut_audio(rms, start, end):
    """quality_gate.cut_audio frames of the cut [start, end] from the window's 10 ms frame RMS
    from 0 s (_window_rms), or None (off, no audio there)."""
    if rms is None or not quality_gate.cut_audio_enabled():
        return None
    try:
        return quality_gate.cut_audio_from_rms(rms, 0.0, 0.01, start, end)
    except Exception:                     # noqa: BLE001 - the word times decide
        return None


def _final_cut(clip, src, media, media_dur, words_fn=None, rms_fn=None, errs_fn=None):
    """the cut read once more on the words the clip burns (its sidecar, the
    window transcripts as context where the sidecar has none) and on the gate's own cut checks
    (gate_cut_errors: word / sentence bounds on the sidecar and the ending on the second ASR, the
    pass the gate then reads from its cache), just before the render. A cut that opens or ends
    inside a sentence, on a fragment, a false start, a lone interjection or a wrap-up line
    (boundaries.final_cut_issues), or that the gate's checks fail, moves to the nearest clean
    sentence edge on those words (boundaries.final_cut_candidates) that the gate's checks pass:
    read on the first pass's words, then on the gate's own pass on the new cut when the window
    has the time (FINAL_VERIFY at most). No such edge: a cut the gate's checks pass stays as it is
    (the eye-check judges it); one they fail raises NoCleanCut, and the render never spends its
    encode on a clip the gate would drop. Sentence clips with SNAP_MIN_WORDS burned words only;
    returns the clip (moved or as it was)."""
    cid = clip["clip_id"]
    if not final_cut_enabled() or (clip.get("boundary_type") or "sentence") != "sentence":
        return clip
    from clipengine import cut_review
    s0, e0 = float(clip["start_s"]), float(clip["end_s"])
    side = _load_words_sidecar(clip.get("words_path"))
    burned = [w for w in side if s0 <= _mid(w) <= e0 and w.get("cap", True) is not False]
    if len(burned) < SNAP_MIN_WORDS:
        return clip
    window = (words_fn or _cached_window_words)(media)
    words = _context_words(window, side)
    niche = _niche_of(clip["source_id"])
    vc = {**vertical_cfg((src or {}).get("vertical") or "general"),
          **topic_map.select_cfg(niche)}
    extra = tuple(vc.get("wrapup_extra") or ())
    lean = _lean_gate(clip)
    gate_errs = errs_fn or gate_cut_errors
    seen = {}
    # the cuts read on the window audio too (a cut in a pause is in no
    # word: a word stretched over it is read on the side its voice is on), from the window RMS
    # the candidates are cut on
    rms = None
    if quality_gate.cut_audio_enabled():
        rms = rms_fn(media, media_dur) if rms_fn else _window_rms(media, media_dur)

    def seated(s, e):
        """(the cut's audio frames, `words` seated on them at the cut [s, e])."""
        audio = _rms_cut_audio(rms, s, e)
        return audio, quality_gate.seat_on_cut_audio(words, (("start", s), ("end", e)), audio)[0]
    audio0, words0 = seated(s0, e0)
    late = _asr_room(clip, 0.0)
    if late:                  # no second ASR before the render: the sidecar checks only
        errs = _word_cut_errors(side, s0, e0, lean, audio=audio0)
    else:
        errs = gate_errs(side, s0, e0, media, lean=lean, seen=seen)
    closes = _closes_at(seen.get("rec"), s0, window, None if words_fn else media)
    issues = boundaries.final_cut_issues(words0, s0, e0, extra=extra, closes_at=closes)
    info = {"from": [round(s0, 3), round(e0, 3)], "issues": issues, "gate": errs}
    if late:
        info["second_asr"] = f"not read before the render ({late})"
    if not issues and not errs:
        feats = _features_merge(clip, final_cut={**info, "kept": "clean"})
        runstate.update_clip(cid, features_json=feats)
        return {**clip, "features_json": feats}
    # a side the gate's own checks fail is a side to move, whatever the burned words say
    want = dict(issues)
    for e in errs:
        side_name = "start" if reason_codes._cut_side(e.partition(": ")[2]) == "cut_start" else "end"
        if e.startswith("ending"):
            side_name = "end"
        want.setdefault(side_name, e[:160])
    sw = [w for w in side if str(w.get("w", "")).strip()]
    cover = (min(float(w["start"]) for w in sw), max(float(w["end"]) for w in sw)) if sw else None
    if rms is None:
        rms = rms_fn(media, media_dur) if rms_fn else _window_rms(media, media_dur)
    cands = boundaries.final_cut_candidates(words, s0, e0, want, cfg=vc, rms=rms, t0=0.0,
                                            media_dur=media_dur, cover=cover, extra=extra,
                                            closes_at=closes)
    info["want"] = want
    info["candidates"] = len(cands)
    tried, verified = [], 0
    rec = seen.get("rec")
    for s2, e2, how in cands:
        audio2, words2 = seated(s2, e2) if audio0 is not None else (None, words)
        why = "; ".join(_word_cut_errors(side, s2, e2, lean, audio=audio2)) or None
        left = boundaries.final_cut_issues(words2, s2, e2, extra=extra, closes_at=closes)
        if why is None and left:
            why = f"still {left}"
        if why is None and rec is not None:
            pred = _predicted_ending(side, rec, s0, e0, s2, e2, lean, audio=audio2)
            if pred:
                why = f"predicted: {'; '.join(pred)}"
        if why is None and verified < FINAL_VERIFY and not late and \
                not _asr_room(clip, ASR2_RTF * (e2 - s2 + 2.0)):
            verified += 1
            errs2 = gate_errs(side, s2, e2, media, lean=lean)
            if errs2:
                why = f"the gate's pass: {'; '.join(errs2)}"
        tried.append({"cut": [round(s2, 3), round(e2, 3)], **how, "why": (why or "")[:200]})
        if why is not None:
            continue
        info.update(to=[round(s2, 3), round(e2, 3)], how=how, tried=tried[-4:])
        feats = _features_merge(clip, final_cut=info)
        runstate.update_clip(cid, start_s=s2, end_s=e2, features_json=feats)
        _log("info", "render", f"clip {cid}: final cut {s0:.2f}-{e0:.2f}s -> {s2:.2f}-{e2:.2f}s "
             f"(start {how['start']}, end {how['end']}: opens {how['first']!r}, ends "
             f"{how['last']!r}; {'; '.join(f'{k}: {v}' for k, v in want.items())[:240]})",
             clip_id=cid)
        return {**clip, "start_s": s2, "end_s": e2, "features_json": feats}
    info["tried"] = tried[-6:]
    kept = "no clean edge" if not errs else "no clean edge the gate's checks pass"
    feats = _features_merge(clip, final_cut={**info, "kept": kept})
    runstate.update_clip(cid, features_json=feats)
    clip = {**clip, "features_json": feats}
    # the render, the gate and its repairs (the editor's re-cut with the gate's errors, the
    # mechanical re-cut) judge it as on HEAD; a clip the gate still fails on its cuts alone is
    # replaced by its next candidate there (render)
    _log("info", "render", f"clip {cid}: final cut kept {s0:.2f}-{e0:.2f}s ({kept}; "
         f"{'; '.join(list(issues.values()) + errs)[:220]})", clip_id=cid)
    return clip


def review_feedback(report):
    """The gate's errors on the codes a review can fix, as the review's feedback; None when the
    report failed on anything else (a crop, a band: the editor cannot fix those)."""
    codes = non_caption_codes(report)
    if not codes or not codes <= set(REVIEW_CODES):
        return None
    parts = []
    for key, m in (report or {}).items():
        if isinstance(m, dict) and m.get("passed") is False and key not in NON_REJECTING_GATE_KEYS:
            parts += [f"{key}: {e}" for e in (m.get("errors") or [])]
    return "; ".join(parts) or str(report.get("drop_reason") or "")


def _review_recut(clip, src, media, media_dur, report, out, spent_s):
    """repair over reject: a clip whose gate failed only on its cuts or its title goes
    back to the editor with the gate's errors; a cut the editor moves gets its caption step again
    (the strong pass on the new cut) and one more render + gate; a new title alone re-renders with
    the same words. Returns (clip, qc, report) of the new render, or None (the first report stands:
    the mechanical re-cut may still run)."""
    cid = clip["clip_id"]
    fb = review_feedback(report)
    if fb is None:
        return None
    need = REVIEW_LLM_S + RECUT_MARGIN * spent_s + _strong_cost(
        float(clip["end_s"]) - float(clip["start_s"]) + sum(caption_context()[:2]))
    new, r = _cut_review(clip, src, media, media_dur, feedback=f"the quality gate: {fb}",
                         need_s=need)
    if r is None or (r["verdict"] != "move" and not r.get("hook_changed")):
        return None
    t0 = time.monotonic()
    if r["verdict"] == "move":
        new = _seat_cuts(new, media, media_dur)
        before = (new["start_s"], new["end_s"])
        try:
            new = _caption_step(new, media, media_dur)
            new = _final_cut(new, src, media, media_dur)
        except NoCleanCut:
            runstate.update_clip(cid, start_s=clip["start_s"], end_s=clip["end_s"],
                              words_path=clip.get("words_path"), hook=clip.get("hook"))
            return None
        new = _title_after_cut_moves(new, src, media, media_dur, before)
    _log("info", "render", f"clip {cid}: gate failed on {', '.join(sorted(non_caption_codes(report)))}"
         f"; the cut review {'moved the cut' if r['verdict'] == 'move' else 'wrote a new title'}"
         f", re-render once", clip_id=cid)
    rr = _run_recompose(recompose_cmd(new, src, media, out))
    qc = _qc_line(rr.stdout)
    if rr.returncode != 0 or not out.exists():
        why = "; ".join(qc.get("errors") or []) or (rr.stderr or rr.stdout).strip()[-200:]
        runstate.update_clip(cid, start_s=clip["start_s"], end_s=clip["end_s"],
                          words_path=clip.get("words_path"), hook=clip.get("hook"))
        _log("warn", "render", f"clip {cid}: review re-cut render failed ({why}); the first gate "
             f"verdict stands", clip_id=cid)
        return None
    report2 = _run_quality_gate(out, new, qc=qc)
    report2["review_recut"] = {"from": [float(clip["start_s"]), float(clip["end_s"])],
                               "to": [float(new["start_s"]), float(new["end_s"])],
                               "first_gate": str(report.get("drop_reason") or "")[:300],
                               "seconds": round(time.monotonic() - t0, 1)}
    return new, qc, report2


OUT_DIR = None              # where finished clips go: <OUT_DIR>/<source_id>/clip_NN.mp4 (run() sets
                            # it); None keeps them under <scratch>/clips


def _store_clip(out, clip):
    """Copy the finished render to <OUT_DIR>/<source_id>/clip_NN.mp4; returns its path."""
    import shutil
    root = Path(OUT_DIR) if OUT_DIR else SCRATCH / "clips"
    dest = root / str(clip["source_id"]) / f"clip_{int(clip['clip_id']):02d}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, dest)
    return str(dest)


def render(clip):
    cid = clip["clip_id"]
    try:
        src = next((s for s in runstate.sources_in("selected") + runstate.sources_in("done")
                    if s["source_id"] == clip["source_id"]), None)
        media = (src or {}).get("local_path")
        if not (media and Path(media).exists()):
            raise RuntimeError("source media missing (harvested VOD may have been swept)")
        end_s = float(clip["end_s"])
        media_dur = float((src or {}).get("duration_s") or 0)
        if not media_dur or end_s > media_dur - MEDIA_END_PROBE_S:
            media_dur = _media_duration(media) or media_dur
        new_end = clamp_cut_end(clip["start_s"], end_s, media_dur)
        if new_end != end_s:
            _log("warn", "render", f"clip {cid}: end {end_s:.3f}s past media end "
                 f"{media_dur:.3f}s; clamped to {new_end:.3f}s", clip_id=cid)
            runstate.update_clip(cid, end_s=new_end)
            clip = {**clip, "end_s": new_end}   # the gate measures word/sentence bounds on it
        SCRATCH.mkdir(parents=True, exist_ok=True)
        out = SCRATCH / f"clip_{cid}.mp4"
        # the editor's pass first, the cuts seated on the audio; the caption step then
        # hears the cut they chose
        while True:
            clip, _review = _cut_review(clip, src, media, media_dur)
            clip = _seat_cuts(clip, media, media_dur)
            before_captions = (clip["start_s"], clip["end_s"])
            try:
                clip = _caption_step(clip, media, media_dur)
                # the cut read once more on the words it burns and the gate's
                # checks; a clip with no clean cut is replaced by its next candidate
                before_final = (clip["start_s"], clip["end_s"])
                clip = _final_cut(clip, src, media, media_dur)
            except NoCleanCut as e:
                alt = _next_alternate(clip, str(e))
                if alt is None:
                    _drop_unclean(clip, e)
                    return False
                clip = alt
                continue
            break
        if _review is not None or (clip["start_s"], clip["end_s"]) != before_final:
            clip = _title_after_cut_moves(clip, src, media, media_dur, before_captions)
        clip = _confirm_crosstalk(clip, media)
        clip = _caption_check(clip, src, media, media_dur)
        cmd = recompose_cmd(clip, src, media, out)
        t_render = time.monotonic()
        r = _run_recompose(cmd)
        qc = _qc_line(r.stdout)
        if r.returncode != 0:
            # a cut the review moved can take in a graphic the crop cannot clear (11-09
            # 2874 moved back 4 s onto a sliced overlay); select's cut rendered before: back to it
            why = "; ".join(qc.get("errors") or []) or (r.stderr or r.stdout).strip()[-300:]
            undone = _undo_review_move(clip, src, media, media_dur, why)
            if undone is not None:
                clip = undone
                r = _run_recompose(recompose_cmd(clip, src, media, out))
                qc = _qc_line(r.stdout)
        if r.returncode != 0:
            why = "; ".join(qc.get("errors") or []) or (r.stderr or r.stdout).strip()[-300:]
            if split_due(clip, error=why) is not None:
                raise RuntimeError(f"recompose exit {r.returncode}: {why}")
            # the single view found no clean crop; the dual view, once
            _log("info", "render", f"clip {cid}: the single view found no clean crop "
                 f"({why[-160:]}); dual view fallback, render once more", clip_id=cid)
            clip = {**clip, "split_fallback": True, "features_json": _features_merge(
                clip, split_fallback={"first_render": why[-300:]})}
            runstate.update_clip(cid, features_json=clip["features_json"])
            r = _run_recompose(recompose_cmd(clip, src, media, out))
            qc = _qc_line(r.stdout)
            if r.returncode != 0:
                why = "; ".join(qc.get("errors") or []) or (r.stderr or r.stdout).strip()[-300:]
                raise RuntimeError(f"recompose exit {r.returncode} (dual view fallback): {why}")
        # Quality gate on the rendered pixels: no face-cut edges, no silent-listener switches,
        # no mid-word start/end, caption validator pass. The gate only runs when the rendered
        # file actually exists on disk — test fixtures that mock subprocess.run skip it. A
        # failing clip is DROPPED (state=failed) with the drop reason in features_json/error.
        gate_pass_note = ""
        if out.exists():
            report = _run_quality_gate(out, clip, qc=qc)
            if report.get("verdict") != "pass":
                again = _split_fallback(clip, src, media, report, out,
                                        time.monotonic() - t_render)
                if again is not None:
                    clip, qc, report = again
            if report.get("verdict") != "pass" or caption_warning(report):
                spent = time.monotonic() - t_render
                again = _recaption(clip, src, media, media_dur, report, out, spent)
                if again is None and report.get("verdict") != "pass":
                    # a cut / title reject goes back to the editor with the gate's
                    # errors first; the mechanical re-cut then runs on whatever still fails
                    again = _review_recut(clip, src, media, media_dur, report, out, spent)
                    base = again or (clip, qc, report)
                    if base[2].get("verdict") != "pass":
                        more = _recut(base[0], src, media, media_dur, base[2], out, spent)
                        again = more or again
                if again is None:     # what a repair that did not re-render recorded stays
                    clip = {**clip, "features_json": (runstate.clip(cid) or clip)["features_json"]}
                if again is not None:
                    clip, qc, report = again
            if report.get("verdict") != "pass":
                # a clip the gate fails on its cuts alone, after every repair, is
                # replaced by its next candidate (select's dropped pick over the same moment)
                codes = non_caption_codes(report)
                if codes and codes <= {"cut_start", "cut_end"}:
                    alt = _next_alternate(runstate.clip(cid) or clip,
                                          f"the gate: {str(report.get('drop_reason') or '')[:200]}")
                    if alt is not None:
                        out.unlink(missing_ok=True)
                        return render(alt)
                _drop_clip(clip, report.get("drop_reason") or "gate failed", "gate", report)
                out.unlink(missing_ok=True)
                return False
            clip = caption_unresolved(clip, report)
            feats = _features_merge(clip, gate_report=report)
            runstate.update_clip(cid, features_json=feats)
            gate_pass_note = " gate=pass"
        url = _store_clip(out, clip)
        if out.exists():
            out.unlink(missing_ok=True)
        runstate.update_clip(cid, render_path=None, out_path=url, state="rendered", qc=qc)
        _log("info", "render", f"clip {cid} rendered -> {url} layout={qc.get('layout')} "
             f"lufs={qc.get('integrated_lufs')}{gate_pass_note}", clip_id=cid,
             source_id=clip["source_id"])
        return True
    except Exception as e:
        runstate.update_clip(cid, state="failed", error=str(e)[:400], features_json=_features_merge(
            runstate.clip(cid) or clip, reason_codes=reason_codes.reject_codes(None, str(e), stage="render")))
        _log("error", "render", f"clip {cid} FAILED: {e}", clip_id=cid)
        if str(e).startswith("recompose exit") and _SLICE_RENDER.search(str(e)) \
                and _niche_of(clip.get("source_id")):
            # the render's planner found no clean crop before any encode: the window
            # is flagged like one the selection probe dropped
            runstate.update_source(clip["source_id"], error=f"{UNCROPPABLE}: render: {e}"[:400])
        return False


# ---- RUN: one source, end to end (cli `run`) ----
_NICHE_VERTICALS = {"podcasts": "podcast", "streamers": "stream", "gaming": "stream",
                    "poker": "poker"}
_YT_ID = re.compile(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})")
_TWITCH_VOD = re.compile(r"(?:twitch\.tv/(?:videos|[^/]+/v)/)(\d+)", re.I)


def vertical_for_niche(niche):
    """The vertical a niche's sources get (select's rubric, duration band and score weights; the
    render's layout): podcasts -> podcast, streamers / gaming -> stream, poker -> poker (select's
    poker rubric, the render's hudband: track when no HUD matches), any other -> general."""
    return _NICHE_VERTICALS.get(niche, "general")


def source_id_for(target):
    """A source's id: the YouTube video id, tw_<vod id> for a Twitch VOD (or a bare VOD number),
    else loc_ + the first 11 hex digits of md5(target)."""
    import hashlib
    m = _YT_ID.search(target)
    if m:
        return m.group(1)
    s = str(target).strip()
    vid = s if s.isdigit() else (_TWITCH_VOD.search(s).group(1) if _TWITCH_VOD.search(s) else None)
    if vid:
        return f"tw_{vid}"
    return "loc_" + hashlib.md5(target.encode()).hexdigest()[:11]


def worker_enabled(env=None):
    """A run's whisper CLI calls (base pass, precision passes, edge-verify windows, the strong
    caption passes, the gate's second ASR) go to one warm process (whisper_worker.py, same output
    as the CLI). TRANSCRIBE_WORKER=0 turns it off (one CLI process per call)."""
    return str((os.environ if env is None else env).get("TRANSCRIBE_WORKER", "1")).strip() != "0"


def _cleanup_source(sid):
    """A source downloaded into the scratch dir is deleted once none of its clips waits for a
    render; a local file the run was given is never touched."""
    s = runstate.source(sid) or {}
    pending = [c for c in runstate.clips_for_source(sid) if c["state"] == "selected"]
    lp = s.get("local_path")
    if not pending and lp and Path(lp).exists() and str(SCRATCH) in str(lp):
        sz = Path(lp).stat().st_size
        Path(lp).unlink()
        runstate.update_source(sid, local_path=None, state="done")
        _log("info", "cleanup", f"removed downloaded source {sid} ({sz // 1048576}MB)",
             source_id=sid)


def _sidecar(c):
    """The clip_NN.json record of one clip row."""
    try:
        feats = json.loads(c.get("features_json") or "{}") or {}
    except (TypeError, ValueError):
        feats = {}
    gate = feats.get("gate_report") or {}
    start, end = float(c.get("start_s") or 0.0), float(c.get("end_s") or 0.0)
    rendered = c.get("state") == "rendered" and c.get("out_path")
    codes = feats.get("reason_codes") or (reason_codes.codes_from_gate_report(gate) if gate else {})
    qc = c.get("qc") or {}
    return {
        "clip": int(c["clip_id"]), "source_id": c["source_id"], "state": c.get("state"),
        "file": Path(c["out_path"]).name if rendered else None,
        "start": round(start, 3), "end": round(end, 3), "duration": round(end - start, 3),
        "score": c.get("score"), "title": c.get("hook"), "layout": c.get("layout"),
        "rendered_layout": qc.get("layout"),
        "gate": {"verdict": gate.get("verdict") if gate else ("fail" if c.get("state") == "failed"
                                                               else None),
                 "drop_reason": gate.get("drop_reason") or None, "note": gate.get("note"),
                 "caption_warning": gate.get("caption_warning")},
        "reason_codes": codes, "error": c.get("error"),
        "qc": {k: qc.get(k) for k in ("width", "height", "duration", "has_audio",
                                      "integrated_lufs", "layout", "passed") if k in qc},
        "boundary_type": c.get("boundary_type"), "features": feats}


def _write_outputs(sid, dest, info):
    """clip_NN.json for every clip of the source and run.json; returns run.json's record."""
    clips = []
    for c in runstate.clips_for_source(sid):
        side = _sidecar(c)
        (dest / f"clip_{side['clip']:02d}.json").write_text(json.dumps(side, indent=1, default=str))
        clips.append({k: side[k] for k in ("clip", "state", "file", "start", "end", "duration",
                                           "title")} | {"verdict": side["gate"]["verdict"],
                                                        "reason_codes": sorted(side["reason_codes"])})
    made = [c for c in clips if c["state"] == "rendered"]
    rec = {**info, "source": runstate.source(sid), "clips": clips, "made": len(made),
           "selected": len(clips), "events": runstate.events()}
    (dest / "run.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def run(target, niche=None, out_dir="out", title=None, layout="auto", vertical=None,
        cookies=None):
    """One source end to end: harvest (download a URL, or use a local file), select, render and
    gate every clip, then write <out_dir>/<source_id>/clip_NN.mp4 (gate-passed clips),
    clip_NN.json (every selected clip) and run.json. Returns run.json's record."""
    global COOKIES, OUT_DIR
    t0 = time.time()
    is_local = Path(target).exists()
    sid = source_id_for(target)
    COOKIES = str(cookies or "")
    out_root = Path(out_dir).expanduser().resolve()
    OUT_DIR = str(out_root)
    dest = out_root / sid
    dest.mkdir(parents=True, exist_ok=True)
    runstate.reset()
    vert = vertical or vertical_for_niche(niche)
    runstate.add_source(sid, url=None if is_local else target, title=title,
                        local_path=str(Path(target).resolve()) if is_local else None,
                        content_type=layout, vertical=vert, niche=niche)
    steps = {"harvest": False, "select": False}
    transcript.WORKER = worker_enabled()
    try:
        steps["harvest"] = harvest(runstate.source(sid))
        if steps["harvest"]:
            steps["select"] = select(runstate.source(sid))
        if steps["select"]:
            for c in runstate.clips_for_source(sid):
                if c["state"] != "selected":
                    continue
                t_r = time.time()
                try:
                    render(c)
                except Exception as e:      # noqa: BLE001 - one clip never stops the others
                    runstate.update_clip(c["clip_id"], state="failed", error=str(e)[:400])
                    _log("error", "render", f"clip {c['clip_id']} raised {type(e).__name__}: {e}",
                         clip_id=c["clip_id"])
                fresh = runstate.clip(c["clip_id"]) or {}
                _log("info", "run", f"clip {c['clip_id']} {fresh.get('state')} in "
                     f"{time.time() - t_r:.1f}s{': ' + str(fresh.get('error'))[:160] if fresh.get('error') else ''}",
                     clip_id=c["clip_id"])
    finally:
        # one warm whisper per run, open through render so the gate's whisper calls use it too
        transcript.close_worker()
    _cleanup_source(sid)
    info = {"target": target, "source_id": sid, "niche": niche, "vertical": vert,
            "layout": layout, "title": title, "steps": steps, "out_dir": str(dest),
            "started_at": round(t0, 3), "seconds": round(time.time() - t0, 1)}
    rec = _write_outputs(sid, dest, info)
    _log("info", "run", f"{sid}: {rec['made']} of {rec['selected']} selected clip(s) passed the "
         f"gate -> {dest}", source_id=sid)
    return rec
