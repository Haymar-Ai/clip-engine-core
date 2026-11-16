"""The reject taxonomy (reason_codes): a reviewer's words, a gate report or an engine error string
become reason codes, one per defect, each with the kind of repair that fixes it; a clip the gate
drops or the render fails carries its codes in its features_json. The reviewer reasons below are
invented, modelled on the shapes reviewers write (several defects in one line)."""
from __future__ import annotations

import subprocess

import pytest

from clipengine import reason_codes as rj
from clipengine import runstate
from helpers import codes_for

# reviewer reason -> the codes a review assigns it
REVIEWER_REASONS = {
    "boxed_market_show": (
        "market show broadcast layout (speaker boxed among tickers, SUBSCRIBE panel); ends "
        "mid-sentence ('No, that is actually.')", ["cut_end", "layout_third_party_ui"]),
    "black_band": (
        "bottom 216 px solid black; HUD strip shows a sliced banner ('ZOR')",
        ["edge_slice", "black_band"]),
    "four_defects": (
        "wrong hook; garbled captions; ends on 'Oh'; fight broadcast round/clock bug burned in",
        ["cut_end", "captions_garbled", "layout_third_party_ui", "wrong_hook"]),
    "broken_captions": ("broken captions; last sentence cut", ["cut_end", "captions_garbled"]),
    "watermark": (
        "burned-in third-party app watermark (a creator handle) and browser/taskbar UI; no hook",
        ["layout_third_party_ui", "watermark", "no_hook"]),
    "crop_between_panels": (
        "crop between the source's two panels at 14.5-17.5 s; sponsor tag; ends on a fragment",
        ["cut_end", "layout_third_party_ui", "edge_slice"]),
    "both_cuts": (
        "ends mid-sentence ('because' starts 0.17 s before the cut); opens mid-sentence",
        ["cut_start", "cut_end"]),
    "crosstalk_slurs": (
        "nonsense captions from crosstalk; uncensored slurs; opening has no referent",
        ["cut_start", "captions_garbled", "slur"]),
    "hud_profile": (
        "hudband HUD profile mismatch: band slices the table graphics, black gaps",
        ["layout_hud_mismatch"]),
    "hud_tiles_cut": (
        "hudband mismatch: player tile and equity numbers cut at the crop edge; captions on the "
        "pot value", ["captions_over_graphics", "layout_hud_mismatch", "edge_slice"]),
    "hud_tiles_twice": (
        "hudband mismatch: band duplicates the player tiles; captions on the event's livestream "
        "bar; full-screen bumper frames",
        ["captions_over_graphics", "layout_hud_mismatch", "layout_third_party_ui"]),
    "boxed_window": (
        "boxed broadcast window, neighbouring windows sliced at both edges; burned-in lower "
        "third; sensitive topic", ["layout_third_party_ui", "edge_slice", "sensitive"]),
    "news_crawl": (
        "news crawl sliced by the frame edge; picture-frame border; crosstalk captions",
        ["captions_garbled", "layout_third_party_ui", "edge_slice"]),
    "market_panels": (
        "market show broadcast layout: crawls, sponsor QR panel, SUBSCRIBE panel, speaker in a "
        "small box", ["layout_third_party_ui"]),
}

# the codes a reviewer uses (the rest are engine-only: gate / render measures)
REVIEWER_CODES = ("cut_start", "cut_end", "captions_garbled", "captions_over_graphics",
                  "layout_hud_mismatch", "layout_third_party_ui", "edge_slice", "watermark",
                  "black_band", "slur", "no_hook", "duplicate", "off_topic", "sensitive")
NOT_FIXABLE = ("duplicate", "off_topic", "sensitive")


def _clip(sid="Abc_w00", start=1.0, end=20.0, state="selected", error=None, features=None):
    runstate.add_source(sid, url=f"https://example.com/watch/{sid}")
    cid = runstate.add_clip(sid, start, end, hook="h", features_json=features)
    runstate.update_clip(cid, state=state, error=error)
    return cid


# ---------------------------------------------------------------- classify

@pytest.mark.parametrize("key", sorted(REVIEWER_REASONS))
def test_every_reviewer_reason_gets_the_reviewed_codes(key):
    reason, codes = REVIEWER_REASONS[key]
    assert set(rj.classify(reason)) == set(codes)


def test_classify_unknown_reason_is_other_and_codes_come_in_codes_order():
    assert rj.classify("looks meh") == ["other"]
    got = rj.classify("uncensored slurs; opens mid-sentence; garbled captions")
    assert got == ["cut_start", "captions_garbled", "slur"]


def test_every_code_has_a_fix_or_is_explicitly_not_fixable():
    assert set(REVIEWER_CODES) <= set(rj.CODES)
    assert not any(rj.fixable(c) for c in NOT_FIXABLE + ("other",))
    assert all(rj.fixable(c) for c in set(REVIEWER_CODES) - set(NOT_FIXABLE))
    assert rj.all_fixable(["cut_end", "slur"]) and not rj.all_fixable(["cut_end", "sensitive"])
    assert not rj.all_fixable([])


# ---------------------------------------------------------------- gate reports / errors

def test_gate_report_measures_map_to_codes():
    rep = {
        "frame_gate": {"passed": False, "edge_cut_count": 3, "none_count": 0, "fail_pct": 2.5},
        "sentence_bounds": {"passed": False, "errors": [
            "start mid-sentence: prev='so' first='it'",
            "end mid-sentence (dangling tail): last='like,' next='(none)'"]},
        "caption_asr": {"passed": False, "errors": ["garbled caption stretch: 42% of 12"]},
        "hook_grounding": {"passed": False, "errors": ["no hook: the clip has no hook title"]},
        "edge_bands": {"passed": False, "errors": ["bottom band 216 px"]},
        "crop_slices": {"passed": False, "errors": ["'NEWS' sliced at x=404"]},
        "visual_quality": {"passed": False, "errors": ["too dark: median luma 28.3 < 35"]},
        "av_sync": {"passed": False, "drift_ms": 80},
        "ending": {"passed": False, "errors": ["ends on a filler/conjunction 'Oh'"]},
        "audio_lufs": {"passed": True},
        "word_boundaries": {"passed": None},
    }
    got = rj.codes_from_gate_report(rep)
    assert set(got) == {"face_cut", "cut_start", "cut_end", "captions_garbled", "no_hook",
                        "black_band", "edge_slice", "too_dark", "av_sync"}
    assert got["cut_start"].startswith("sentence_bounds: start mid-sentence")
    nf = rj.codes_from_gate_report({"frame_gate": {"passed": False, "edge_cut_count": 0,
                                                   "none_count": 40, "fail_pct": 9.0}})
    assert set(nf) == {"no_face"}
    assert rj.codes_from_gate_report({"hook_grounding": {"passed": False, "errors": [
        "hook not supported by what is said: wrong subject"]}}).keys() == {"wrong_hook"}


@pytest.mark.parametrize("error,codes", [
    ("gate: frame_gate", {"face_cut"}),
    ("gate: sentence_bounds: end mid-sentence (dangling tail): last='like,' next='(none)'",
     {"cut_end"}),
    ("gate: word_boundaries: end cut inside word \"let's\" (start=59.150 end=59.330)", {"cut_end"}),
    ("gate: frame_gate; word_boundaries: end cut inside word 'the' (start=50.610 end=50.790); "
     "sentence_bounds: end mid-sentence: last='doing' next='the'", {"face_cut", "cut_end"}),
    ("recompose exit 1: RENDER FAILED: crop would slice a source banner: game crop 578px slices",
     {"edge_slice"}),
    ("recompose exit 1: RENDER FAILED: hook placement: no hook position clears the on-screen "
     "text", {"captions_over_graphics"}),
    ("WINDOW_TIMEOUT render:clip1 after 420s", {"render_timeout"}),
    ("gate: av_sync", {"av_sync"}),
    ("gate: visual_quality: too dark: median luma 33.9 < 35", {"too_dark"}),
    ("recompose exit 3: integrated loudness -15.7 not within 1.5 LU of -14.0", {"loudness"}),
])
def test_engine_error_strings_map_to_codes(error, codes):
    assert set(rj.codes_from_error(error)) == codes


# ---------------------------------------------------------------- a failed clip's codes

def test_drop_clip_stores_the_gate_reports_codes():
    from clipengine import engine
    cid = _clip()
    rep = {"verdict": "fail", "caption_asr": {"passed": False, "errors": ["captions carry 55%"]},
           "ending": {"passed": False, "errors": ["ends mid-sentence: last word 'and'"]}}
    engine._drop_clip(runstate.clip(cid), "caption_asr: captions carry 55%", "gate", rep)
    assert runstate.clip(cid)["state"] == "failed"
    assert set(codes_for(cid)) == {"captions_garbled", "cut_end"}


def test_render_failure_stores_codes(monkeypatch, tmp_path):
    from clipengine import engine
    monkeypatch.setattr(engine, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(engine, "CAPTION_STRONG_MODEL", None)   # the caption step has its own tests
    media = tmp_path / "w.mp4"
    media.write_bytes(b"\0")
    runstate.add_source("R_w00", local_path=str(media), content_type="auto")
    runstate.update_source("R_w00", state="selected", duration_s=60.0)
    cid = runstate.add_clip("R_w00", 10.0, 31.5, 0.7, "auto")
    err = "RENDER FAILED: crop would slice a source banner: segment 0: no 328x584 crop"
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw:
                        subprocess.CompletedProcess(cmd, 1, stdout="", stderr=err))
    assert engine.render(runstate.clip(cid)) is False
    row = runstate.clip(cid)
    assert row["state"] == "failed" and row["error"].startswith("recompose exit 1")
    got = codes_for(cid)
    assert set(got) == {"edge_slice"} and "crop would slice a source banner" in got["edge_slice"]
