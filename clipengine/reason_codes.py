"""Reason codes: every clip the gate or the render rejects says why in codes, one per defect.

The engine reads them to decide which repair a failed clip gets (a gate that failed on its cuts
alone is re-cut, a caption failure goes to the caption repair, a title fault to the cut review),
and each clip's JSON sidecar carries them.

Codes (one per defect). `fix` names the kind of repair that fixes it; None = not fixable:

  cut_start / cut_end        opens / ends mid-sentence or on a filler        -> retrim
  captions_garbled           captions not what is said (crosstalk, garble)   -> recaption
  captions_over_graphics     captions or hook on top of on-screen graphics   -> caption_zone
  layout_hud_mismatch        poker hudband profile does not fit the feed     -> recrop
  layout_third_party_ui      broadcast / player / browser UI, sponsor panels -> recrop
  edge_slice                 a graphic, panel or window cut by the crop edge -> recrop
  watermark                  burned-in third-party watermark                 -> recrop
  black_band                 a black / flat band at a frame edge             -> recrop
  slur                       an uncensored slur                              -> censor
  no_hook / wrong_hook       no hook title / one the clip does not support   -> hook
  duplicate, off_topic, sensitive                                            -> not fixable
  engine-only (gate / render measures with no reviewer word for them):
  face_cut                   the speaker's face cut by the crop edge         -> recrop
  no_face                    no speaker on screen for a face crop            -> not fixable
  wrong_speaker              the kept face is a silent listener              -> not fixable
  cover_frame_layout         the first frames / cover are another layout     -> not fixable
  gameplay_face_follow       gameplay rendered as a face-follow crop (track: -> not fixable
                             the game cropped tight round the cam)
  too_dark                   visual quality: too dark                        -> levels
  av_sync / loudness         encode faults                                   -> rerender
  render_timeout             a time cap killed the render                    -> rerender
  other                      a reason no rule recognises                     -> not fixable

Captions never reject a clip: a caption code routes the clip to the caption repair, and a
clip whose codes are caption codes only is no reject.
"""
from __future__ import annotations

import re

RETRIM, RECAPTION, CAPTION_ZONE, RECROP, CENSOR, HOOK, LEVELS, RERENDER = (
    "retrim", "recaption", "caption_zone", "recrop", "censor", "hook", "levels", "rerender")

CODES = {
    "cut_start": (RETRIM, "opens mid-sentence / without its referent"),
    "cut_end": (RETRIM, "ends mid-sentence or on a filler"),
    "captions_garbled": (RECAPTION, "captions do not match what is said (crosstalk, garble)"),
    "captions_over_graphics": (CAPTION_ZONE, "captions or hook on top of on-screen graphics"),
    "layout_hud_mismatch": (RECROP, "poker hudband profile does not fit the broadcast"),
    "layout_third_party_ui": (RECROP, "third-party broadcast / player / browser UI in frame"),
    "edge_slice": (RECROP, "a graphic, panel or window sliced by the crop edge"),
    "watermark": (RECROP, "burned-in third-party watermark"),
    "black_band": (RECROP, "black / flat band at a frame edge"),
    "slur": (CENSOR, "uncensored slur"),
    "no_hook": (HOOK, "no hook title"),
    "wrong_hook": (HOOK, "hook the clip does not support"),
    "duplicate": (None, "duplicate of another clip"),
    "off_topic": (None, "off topic for the niche"),
    "sensitive": (None, "sensitive topic"),
    "face_cut": (RECROP, "speaker's face cut by the crop edge"),
    "no_face": (None, "no speaker on screen for a face-following crop"),
    # A cover frame is fixed by trimming the start past it or holding a later frame over it;
    # no repair here does either, so the code is reported, not repaired.
    "wrong_speaker": (None, "the kept face is not the one speaking (a silent listener)"),
    "cover_frame_layout": (None, "the first frames / cover show another layout than the clip"),
    # gameplay never ships a face-follow crop (the gate's gameplay_layout)
    "gameplay_face_follow": (None, "gameplay rendered as a face-follow crop (the game cropped "
                                   "tight)"),
    "too_dark": (LEVELS, "too dark"),
    "av_sync": (RERENDER, "audio / video out of sync"),
    "loudness": (RERENDER, "loudness off target"),
    "render_timeout": (RERENDER, "time cap killed the render"),
    "other": (None, "a reason no rule recognises"),
}

CAPTION_CODES = frozenset({"captions_garbled"})


def caption_only(codes):
    """True when a reject's codes are all caption codes (captions never reject: not a reject)."""
    codes = set(codes or [])
    return bool(codes) and codes <= CAPTION_CODES


def fixable(code):
    return CODES.get(code, (None, ""))[0] is not None


def fix_of(code):
    return CODES.get(code, (None, ""))[0]


def all_fixable(codes):
    """True when there is at least one code and every one has a repair step."""
    codes = list(codes or [])
    return bool(codes) and all(fixable(c) for c in codes)


# ---------------------------------------------------------------- prose -> codes

# (code, pattern) in priority order; a reason may carry several defects ("garbled captions;
# ends on 'Oh'"). Built on reviewer reasons and every engine error string.
_RULES = [
    ("sensitive", r"\bsensitive\b"),
    ("duplicate", r"\bduplicate(?:s|d)? (?:of|clip)|\bsame (?:moment|clip) as\b|\bre-?cut of\b"),
    ("off_topic", r"\boff[- ]topic\b|\bwrong niche\b"),
    ("slur", r"\bslurs?\b"),
    ("watermark", r"\bwatermark"),
    ("black_band", r"\bsolid black\b|\bblack (?:band|bar|strip)s?\b|\bflat (?:edge )?band\b"
                   r"|\bdead band\b|\bpx (?:solid )?black\b|\bedge_bands\b"),
    ("layout_hud_mismatch", r"\bhudband\b|\bhud profile\b"),
    ("captions_over_graphics", r"\bcaptions? (?:on|over|on top of|covering)\b"
                               r"|\bhook (?:on|over|covers)\b|\bhook placement\b"),
    ("captions_garbled", r"\b(?:garbled|broken|nonsense|crosstalk|wrong|mismatched) captions?\b"
                         r"|\bcaptions? (?:from crosstalk|do(?:es)? not match|garbled|broken)\b"
                         r"|\bcrosstalk captions\b|\bcaption_asr\b|\bgarbled caption stretch\b"
                         r"|\buncaptioned stretch\b|\bcaptions agree with the second asr\b"
                         r"|\bcaptions carry\b"),
    ("edge_slice", r"\bsliced?\b|(?<!band )\bslices\b|\bcut (?:off )?at (?:the|both) (?:crop |frame )?edges?\b"
                   r"|\bcrop between\b|\bcrop_slices\b|\bcrop would slice\b|\bcut at the crop edge\b"),
    ("layout_third_party_ui", r"\bbroadcast layout\b|\bthird-party\b|\bbrowser\b|\btaskbar\b"
                              r"|\bsubscribe panel\b|\bsponsor\b|\bticker|\bcrawls?\b|\blower third\b"
                              r"|\bpicture-frame border\b|\bbumper\b|\bclock bug\b|\bscore ?bug\b"
                              r"|\b(?:qr|ui) panel\b|\bpanels?\b|\bboxed (?:broadcast )?window\b"
                              r"|\bspeaker (?:boxed|in a small box)\b|\bplayer ui\b"),
    ("no_hook", r"\bno hook\b|\bmissing hook\b|\bhook=none\b|\bclip has no hook\b"),
    ("wrong_hook", r"\bwrong hook\b|\bhook not (?:supported|grounded)\b|\bhook_grounding\b"
                   r"|\bhook_support\b|\bmisleading hook\b"),
    ("cut_start", r"\bopens mid-sentence\b|\bstarts? mid-sentence\b|\bno referent\b"
                  r"|\bstart cut inside word\b|\bopens on a fragment\b|\bstart mid-sentence\b"),
    ("cut_end", r"\bends? mid-sentence\b|\bends? on (?:a )?(?:fragment|filler|'oh'|\"oh\"|oh\b)"
                r"|\blast sentence cut\b|\bends on '|\bending\b|\bend cut inside word\b"
                r"|\bend mid-sentence\b|\bcut mid-sentence\b|\bdangling tail\b"),
    ("face_cut", r"\bface (?:cut|sliced)\b|\bedge-cut face\b"),
    ("no_face", r"\bno face\b|\bno speaker\b|\bfaceless\b"),
    ("wrong_speaker", r"\bwrong speaker\b|\bkept face is not the speaker\b"
                      r"|\bsilent (?:listener|guest)\b|\bspeaker is off screen\b"),
    ("cover_frame_layout", r"\bcover frames?\b"),
    # the gate's and recompose's own words only: a reviewer's "the face-follow crop shows the
    # ceiling" (an IRL clip) is no gameplay reject
    ("gameplay_face_follow", r"\bgameplay_layout\b"
                             r"|\bgameplay (?:rendered|planned) as '\w+', a face-follow crop\b"),
    ("too_dark", r"\btoo dark\b"),
    ("av_sync", r"\bav_sync\b|\ba/v (?:sync|drift)\b"),
    ("loudness", r"\bloudness\b|\blufs\b"),
    ("render_timeout", r"\bwindow_timeout\b|\btimeout\b"),
]
_COMPILED = [(code, re.compile(p, re.I)) for code, p in _RULES]


def classify(reason):
    """Codes for a prose reject reason (a reviewer's words or an engine error string), in CODES
    order; ["other"] when nothing matches."""
    text = str(reason or "")
    hit = {code for code, rx in _COMPILED if rx.search(text)}
    # "hudband mismatch: player tile ... cut at the crop edge" is a slice too; "captions on the
    # pot value" an over-graphics caption, not a garbled one
    return [c for c in CODES if c in hit] or ["other"]


# ---------------------------------------------------------------- gate report -> codes

def _errs(m):
    return "; ".join(str(e) for e in (m or {}).get("errors") or [])


def _cut_side(err):
    """cut_start / cut_end for a word_boundaries / sentence_bounds error: they all open with the
    side ("start cut inside word ...", "end mid-sentence (dangling tail): ...")."""
    return "cut_start" if str(err).strip().lower().startswith("start") else "cut_end"


def codes_from_gate_report(report):
    """{code: detail} for every failing measure of a quality_gate / stream lean gate report."""
    out = {}

    def add(code, detail):
        if code not in out:
            out[code] = str(detail)[:300]

    for key, m in (report or {}).items():
        if not isinstance(m, dict) or m.get("passed") is not False:
            continue
        e = _errs(m)
        if key == "frame_gate":
            if int(m.get("none_count") or 0) > int(m.get("edge_cut_count") or 0):
                add("no_face", f"frame_gate: no face on {m.get('none_count')} frames "
                               f"({m.get('fail_pct')}% failed)")
            else:
                add("face_cut", f"frame_gate: face cut by the crop edge on "
                                f"{m.get('edge_cut_count', m.get('fail_count'))} frames "
                                f"({m.get('fail_pct')}% failed)")
        elif key in ("word_boundaries", "sentence_bounds"):
            for err in (m.get("errors") or [e or key]):
                add(_cut_side(err), f"{key}: {err}")
        elif key == "ending":
            add("cut_end", f"ending: {e}")
        elif key == "caption_asr":
            add("captions_garbled", f"caption_asr: {e}")
        elif key in ("hook_grounding", "hook_support"):
            add("no_hook" if "no hook" in e.lower() else "wrong_hook", f"{key}: {e}")
        elif key == "edge_bands":
            add("black_band", f"edge_bands: {e}")
        elif key in ("crop_slices", "overlay_edge_cut", "boxed_slices"):
            add("edge_slice", f"{key}: {e}")
        elif key == "cover_frame":
            add("cover_frame_layout", f"cover_frame: {e}")
        elif key == "gameplay_layout":
            add("gameplay_face_follow", f"gameplay_layout: {e}")
        elif key == "kept_face_speaker":
            add("wrong_speaker", f"kept_face_speaker: {e}")
        elif key == "visual_quality":
            add("too_dark" if "dark" in e.lower() else "edge_slice" if "sliced" in e.lower()
                else "other", f"visual_quality: {e}")
        elif key == "black_frozen":
            add("black_band" if "black" in e.lower() else "other", f"black_frozen: {e}")
        elif key == "av_sync":
            add("av_sync", f"av_sync: drift {m.get('drift_ms')} ms")
        elif key == "audio_lufs":
            add("loudness", f"audio_lufs: {e}")
        elif key in ("layout_match", "layout_quality"):
            add("layout_third_party_ui", f"{key}: {e}")
        elif key in ("face_centre", "crop_step", "speaker_switches", "active_speaker_plan",
                     "active_speaker_av"):
            add("face_cut", f"{key}: {e}")
        else:
            for code in classify(f"{key}: {e}"):
                add(code, f"{key}: {e}")
    return out


def codes_from_error(error):
    """{code: detail} from an error string ("gate: ...", "recompose exit 1: RENDER FAILED: ...",
    "WINDOW_TIMEOUT ...") when no structured report is at hand."""
    text = str(error or "")
    low = text.lower()
    if "window_timeout" in low:
        return {"render_timeout": text[:300]}
    if "crop would slice a source banner" in low:
        return {"edge_slice": text[:300]}
    if "hook placement" in low:
        return {"captions_over_graphics": text[:300]}
    if "integrated loudness" in low:
        return {"loudness": text[:300]}
    if low.startswith("gate:"):
        out = {}
        for part in text[5:].split(";"):
            part = part.strip()
            name = part.split(":", 1)[0].strip()
            if name == "frame_gate":
                out.setdefault("face_cut", part)
            elif name == "av_sync":
                out.setdefault("av_sync", part)
            elif name in ("sentence_bounds", "word_boundaries"):
                out.setdefault(_cut_side(part.split(":", 1)[-1]), part)
            elif part:
                for code in classify(part):
                    out.setdefault(code, part)
        return out or {"other": text[:300]}
    return {c: text[:300] for c in classify(text)}


def reject_codes(report=None, reason=None, stage="gate"):
    """{code: detail} of a dropped or failed clip: from the gate report when there is one, else
    from the error text (the codes a reject ledger would record for it). Never raises."""
    try:
        codes = codes_from_gate_report(report) if report else {}
        if not codes:
            codes = codes_from_error(reason if str(reason or "").lower().startswith(
                ("gate:", "recompose", "window_timeout")) or stage != "gate" else f"gate: {reason}")
        return codes
    except Exception:      # noqa: BLE001 - a code lookup never fails a render
        return {}
