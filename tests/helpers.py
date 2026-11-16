"""Small helpers shared by the tests."""
import json

from clipengine import runstate


def codes_for(clip_id):
    """{code: detail} recorded for a clip (the reason_codes its sidecar carries), {} for none."""
    row = runstate.clip(clip_id) or {}
    try:
        return (json.loads(row.get("features_json") or "{}") or {}).get("reason_codes") or {}
    except (TypeError, ValueError):
        return {}


def fake_store_clip(path, clip):
    """engine._store_clip without a copy: where the clip would be stored."""
    return f"local://{clip['source_id']}/clip_{int(clip['clip_id']):02d}.mp4"


def hud_text_ocr(profile, src_w, src_h, names=("players", "pot"), text="POT 12,500"):
    """A stand-in for recompose._ocr(src, a, size): one OCR'd word of HUD text centred in each
    band region of `profile` named in `names` (regions normalized [x0, y0, x1, y1]), so
    recompose._blank_regions reads those regions as HUD panels. Words are
    (x0, y0, x1, y1, text, frame) in source pixels, as quality_gate.ocr_source_text returns."""
    words = []
    for name in names:
        if name not in profile:
            continue
        x0, y0, x1, y1 = profile[name]
        cx, cy = int((x0 + x1) / 2.0 * src_w), int((y0 + y1) / 2.0 * src_h)
        words.append((cx - 30, cy - 8, cx + 30, cy + 8, text, 0))
    return lambda src, a, size: list(words)
