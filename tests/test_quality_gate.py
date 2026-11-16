"""Quality gate tests. Pure helpers only — no ffmpeg / cv2.

Covers:
- Crop step: 1.5% cap enforced; same-subject jumps (non-cut delta > cap) raise failures.
- Face centre: synthetic plan + face samples; +-10% of crop width on >= 95% of samples.
- Word boundaries: mid-word start or end fails; clean boundaries pass.
- Caption measure wraps captions_tiktok.validate_from_campaign; missing required => fail.
- Auto card-pull windows match the transcript regex and merge adjacent hits.
- decide() returns ``fail`` with a reason concatenation when any measure is False.
- The ``verdict`` is informational-only on measures that return None.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from clipengine import quality_gate as qg


# ---------------------------------------------------------------- auto card windows

def test_auto_card_windows_merges_adjacent_hits():
    words = [
        {"w": "pulled", "start": 100.0, "end": 100.4},
        {"w": "a",      "start": 100.5, "end": 100.6},
        {"w": "card",   "start": 100.7, "end": 101.1},
        {"w": "it",     "start": 110.0, "end": 110.2},
        {"w": "value",  "start": 110.3, "end": 110.8},
    ]
    wins = qg.auto_card_windows(words, start_s=100.0, dur=20.0)
    # Two merged windows, one around 0-1.3 s and another around 10-12 s.
    assert len(wins) == 2
    assert wins[0][0] < 1.0 and wins[0][1] < 2.5
    assert wins[1][0] >= 9.5 and wins[1][1] <= 12.5


def test_auto_card_windows_empty_when_no_card_words():
    assert qg.auto_card_windows([{"w": "hello", "start": 0.0, "end": 0.4}],
                                 start_s=0.0, dur=5.0) == []


# ---------------------------------------------------------------- crop step

def _plan(segments, src_w=1920):
    return {"segments": segments, "src_w": src_w}


def test_crop_step_passes_when_all_steps_within_cap():
    # 1.0% steps, well below the 1.5% cap
    xs = [100, 119, 138, 157, 176]           # 19 px / 1920 ≈ 0.99%
    segs = [{"t0": i * 0.1, "t1": (i + 1) * 0.1, "x": x, "cut": False}
            for i, x in enumerate(xs)]
    segs[0]["cut"] = True                     # shot entry
    m = qg.measure_crop_step(_plan(segs), 1920)
    assert m["passed"] is True
    assert m["same_subject_jumps"] == 0


def test_crop_step_fails_on_same_subject_jump_above_cap():
    # One 2.5% step in the middle, no cut flag
    xs = [100, 148, 196, 244]                # 48 px / 1920 ≈ 2.5%
    segs = [{"t0": i * 0.1, "t1": (i + 1) * 0.1, "x": x, "cut": False}
            for i, x in enumerate(xs)]
    segs[0]["cut"] = True
    m = qg.measure_crop_step(_plan(segs), 1920)
    assert m["passed"] is False
    assert m["same_subject_jumps"] >= 1


def test_crop_step_treats_cut_boundaries_as_free_steps():
    segs = [
        {"t0": 0.0, "t1": 3.0, "x": 100, "cut": True},
        {"t0": 3.0, "t1": 6.0, "x": 1500, "cut": True},
    ]
    m = qg.measure_crop_step(_plan(segs), 1920)
    assert m["passed"] is True


# ---------------------------------------------------------------- face centre

def test_face_centre_passes_when_tracker_keeps_face_centred():
    cw = 608
    segs = [{"t0": 0.0, "t1": 10.0, "x": 656, "cut": True}]     # centred at cx=0.5*1920
    face_samples = [{"t": t / 10.0, "cx": 0.5} for t in range(100)]
    plan = {"segments": segs, "src_w": 1920, "crop_width": cw,
            "face_samples": face_samples}
    m = qg.measure_face_centre(plan, 1920)
    assert m["passed"] is True and m["frac_ok"] == 1.0


def test_face_centre_fails_when_face_drifts_off_centre():
    cw = 608
    segs = [{"t0": 0.0, "t1": 10.0, "x": 656, "cut": True}]
    # Face at cx=0.3 (= 576 px) and crop centre at 656+304=960 -> off by 384 px / 608 = 63%
    face_samples = [{"t": t / 10.0, "cx": 0.3} for t in range(100)]
    plan = {"segments": segs, "src_w": 1920, "crop_width": cw,
            "face_samples": face_samples}
    m = qg.measure_face_centre(plan, 1920)
    assert m["passed"] is False and m["frac_ok"] == 0.0


def test_face_centre_informational_when_plan_lacks_samples():
    plan = {"segments": [{"t0": 0, "t1": 1, "x": 0}], "src_w": 1920}
    m = qg.measure_face_centre(plan, 1920)
    assert m["passed"] is None


# ---------------------------------------------------------------- word boundaries

def _words(spans):
    return [{"w": txt, "start": float(s), "end": float(e)} for s, e, txt in spans]


def test_word_boundaries_pass_at_true_boundary():
    # a cut at the exact word boundary is clean (not inside the word).
    words = _words([(10.0, 10.3, "a"), (10.4, 10.9, "b"), (18.0, 18.5, "c")])
    m = qg.measure_word_boundaries(words, 10.0, 18.5)
    assert m["passed"] is True


def test_word_boundaries_pass_at_midpoint_of_gap():
    # unit test: cut at the midpoint of an inter-word gap must pass regardless of
    # the gap width. Here the start cut sits 100 ms before word a (silence); the end cut
    # sits 100 ms after word c ends.
    words = _words([(10.0, 10.3, "a"), (10.4, 10.9, "b"), (18.0, 18.5, "c")])
    m = qg.measure_word_boundaries(words, 9.9, 18.6)
    assert m["passed"] is True
    assert m["d_start_s"] >= 0.08       # 100 ms from the nearest edge
    assert m["d_end_s"] >= 0.08


def test_word_boundaries_pass_near_edge_inside_margin():
    # a cut 20 ms past a word's end is within the 30 ms edge margin — treat as at
    # the boundary.
    words = _words([(10.0, 10.3, "a"), (10.4, 10.9, "b"), (18.0, 18.5, "c")])
    m = qg.measure_word_boundaries(words, 10.32, 18.52)
    assert m["passed"] is True


def test_word_boundaries_fail_on_mid_word_start():
    # cut 0.2 s into the first word is strictly inside with > 30 ms margin -> fail.
    words = _words([(10.0, 10.3, "a"), (10.4, 10.9, "b"), (18.0, 18.5, "c")])
    m = qg.measure_word_boundaries(words, 10.2, 18.5)
    assert m["passed"] is False
    assert m["start_in_word"] is True


def test_word_boundaries_fail_on_mid_word_end():
    # cut 0.2 s before c ends -> strictly inside word c with > 30 ms margin -> fail.
    words = _words([(10.0, 10.3, "a"), (18.0, 18.5, "c")])
    m = qg.measure_word_boundaries(words, 10.0, 18.3)
    assert m["passed"] is False
    assert m["end_in_word"] is True


def test_word_boundaries_long_offset_in_silence_still_passes():
    # this was an earlier false-alarm pattern — boundaries.clean_clip placed cuts
    # 90-150 ms from word edges in silence. The earlier gate failed those passes.
    words = _words([(10.0, 10.3, "a"), (18.0, 18.5, "c")])
    m = qg.measure_word_boundaries(words, 10.0 - 0.15, 18.5 + 0.09)
    assert m["passed"] is True


# ---------------------------------------------------------------- caption

TRADING_RULES = {
    "template": ("{hook}\n\n{streamer} live on Example Trading Club - "
                 "example-trading.com\n\n@exampletv @joinexample\n\n#ETC #Example #Ad"),
    "required_elements": [
        {"kind": "mention", "value": "@exampletv"},
        {"kind": "mention", "value": "@joinexample"},
        {"kind": "domain",  "value": "example-trading.com"},
        {"kind": "hashtag", "value": "#Ad"},
    ],
}


def test_caption_measure_passes_on_a_well_formed_caption():
    camp = {"caption": TRADING_RULES}
    s = "He traded until 3am\n\nSam live on Example Trading Club - example-trading.com\n\n@exampletv @joinexample\n\n#ETC #Example #Ad"
    m = qg.measure_caption(s, camp)
    assert m["passed"] is True


def test_caption_measure_fails_on_missing_required_element():
    camp = {"caption": TRADING_RULES}
    s = "He traded\n\n@joinexample\n\n#Ad"
    m = qg.measure_caption(s, camp)
    assert m["passed"] is False
    assert any("@exampletv" in e for e in m["errors"])


def test_caption_measure_requires_a_caption():
    camp = {"caption": TRADING_RULES}
    m = qg.measure_caption(None, camp)
    assert m["passed"] is False
    assert "no caption" in m["errors"][0]


# ---------------------------------------------------------------- decide()

def test_decide_passes_when_every_measure_is_pass_or_none():
    rep = {
        "frame_gate": {"passed": True},
        "crop_step": {"passed": True},
        "speaker_switches": {"passed": None},
        "face_centre": {"passed": None},
        "av_sync": {"passed": True},
        "word_boundaries": {"passed": True},
        "caption": {"passed": True},
    }
    v, r = qg.decide(rep)
    assert v == "pass" and r == ""


def test_decide_concatenates_fail_reasons():
    rep = {
        "frame_gate": {"passed": False, "errors": ["fail_pct 4.1"]},
        "crop_step": {"passed": True},
        "caption": {"passed": False, "errors": ["missing hashtag #Ad"]},
    }
    v, r = qg.decide(rep)
    assert v == "fail"
    assert "frame_gate" in r and "caption" in r and "missing hashtag" in r


# ---------------------------------------------------------------- sentence bounds

def test_sentence_bounds_pass_on_full_sentence():
    words = _words([(10.0, 10.3, "I"), (10.4, 10.6, "went"), (10.7, 11.1, "home."),
                    (18.0, 18.3, "Then"), (18.4, 18.6, "I"), (18.7, 19.0, "slept.")])
    m = qg.measure_sentence_bounds(words, 10.0, 19.0)
    assert m["passed"] is True
    assert m["start_sentence_start"] and m["end_sentence_end"]


def test_sentence_bounds_fail_on_mid_clause_end():
    # Clip ends on "to" with next word only 0.1 s later — a tiny gap, no punctuation.
    words = _words([(10.0, 10.3, "He"), (10.4, 10.7, "said."),
                    (14.0, 14.2, "Then"), (14.3, 14.5, "I"), (14.6, 14.8, "wanted"),
                    (14.9, 15.1, "to"), (15.2, 15.4, "go"), (15.5, 15.9, "back.")])
    # Pick a clip that ENDS at the word "to" (15.1) with next word only 0.1 s later.
    m = qg.measure_sentence_bounds(words, 14.0, 15.1)
    assert m["passed"] is False
    assert any("end mid-sentence" in e for e in m["errors"])


def test_sentence_bounds_pass_on_pause_after_last_word():
    # No punctuation on the last word, but the next sentence starts >= 0.35 s later
    # AND the last word is not a dangling filler. "money" is a full noun — complete.
    words = _words([(10.0, 10.3, "He"), (10.4, 10.9, "took"),
                    (12.0, 12.4, "money"),   # <-- clip ends here, gap of 1.0 s follows
                    (13.4, 13.6, "Then"), (13.7, 14.0, "we"), (14.1, 14.5, "left.")])
    m = qg.measure_sentence_bounds(words, 10.0, 12.4)
    assert m["passed"] is True


def test_sentence_bounds_fail_on_dangling_filler_tail_even_with_pause():
    # "like" is a quotative tag — a long pause after it does NOT make the clip
    # complete. The engine must pick a different sentence end.
    words = _words([(10.0, 10.3, "I"), (10.4, 10.9, "was"),
                    (11.0, 11.3, "like"),       # <-- clip ends here, 1.0 s gap follows
                    (12.3, 12.6, "yeah"), (12.7, 13.0, "sure.")])
    m = qg.measure_sentence_bounds(words, 10.0, 11.3)
    assert m["passed"] is False
    assert any("dangling tail" in e for e in m["errors"])


def test_sentence_bounds_fail_on_trailing_comma_end():
    # Trailing comma = continuation, never a sentence end.
    words = [{"w": "I,",   "start": 10.0, "end": 10.3},
             {"w": "said", "start": 10.4, "end": 10.9},
             {"w": "the",  "start": 11.0, "end": 11.3},
             {"w": "truth,", "start": 11.4, "end": 11.9},   # <-- clip ends here
             {"w": "and",  "start": 13.0, "end": 13.3},
             {"w": "won.", "start": 13.4, "end": 13.7}]
    m = qg.measure_sentence_bounds(words, 10.0, 11.9)
    assert m["passed"] is False


def test_sentence_bounds_fail_on_mid_sentence_start():
    words = _words([(10.0, 10.3, "He"), (10.4, 10.6, "went"),
                    (10.7, 10.9, "to"), (11.0, 11.3, "the"), (11.4, 11.8, "store.")])
    # Start at 10.7 is mid-sentence after "went" with no pause/punctuation.
    m = qg.measure_sentence_bounds(words, 10.7, 11.8)
    assert m["passed"] is False
    assert any("start mid-sentence" in e for e in m["errors"])


def test_sentence_bounds_informational_without_words():
    m = qg.measure_sentence_bounds([], 10.0, 20.0)
    assert m["passed"] is None


# ---------------------------------------------------------------- hook support

def test_hook_support_informational_without_judge():
    words = _words([(0.0, 0.3, "I"), (0.4, 0.9, "pulled"), (1.0, 1.5, "a"), (1.6, 2.0, "card.")])
    m = qg.measure_hook_support("He pulled a card", words, judge=None)
    assert m["passed"] is None
    assert "no judge" in (m.get("reason") or "")


def test_hook_support_pass_when_judge_returns_supported():
    words = _words([(0.0, 0.3, "I"), (0.4, 0.9, "pulled"), (1.0, 1.5, "a"), (1.6, 2.0, "card.")])
    def judge(hook, transcript):
        return {"supported": True, "quote": "I pulled a card", "reason": "hook matches"}
    m = qg.measure_hook_support("He pulled a card", words, judge=judge)
    assert m["passed"] is True
    assert m["quote"] == "I pulled a card"


def test_hook_support_fail_when_judge_returns_unsupported():
    words = _words([(0.0, 0.3, "I"), (0.4, 0.9, "played"), (1.0, 1.5, "games.")])
    def judge(hook, transcript):
        return {"supported": False, "quote": "", "reason": "transcript does not mention pulling"}
    m = qg.measure_hook_support("He pulled a card", words, judge=judge)
    assert m["passed"] is False
    assert any("hook not supported" in e for e in m["errors"])


def test_hook_support_informational_when_judge_raises():
    words = _words([(0.0, 0.3, "hi")])
    def bad_judge(hook, transcript):
        raise RuntimeError("backend down")
    m = qg.measure_hook_support("Hook", words, judge=bad_judge)
    assert m["passed"] is None
    assert "backend down" in (m.get("reason") or "")


def test_parse_judge_json_tolerates_fences_and_prose():
    text = "Sure! here you go:\n```json\n{\"supported\": true, \"quote\": \"q\", \"reason\": \"r\"}\n```"
    obj = qg._parse_judge_json(text)
    assert obj is not None and obj["supported"] is True and obj["quote"] == "q"


def test_hook_judge_prompt_formats_without_keyerror():
    # Regression: HOOK_JUDGE_PROMPT contains a JSON schema example {"supported": ...} that
    # must be escaped with double braces so .format(hook=, transcript=) doesn't treat
    # "supported" as a placeholder. Any un-doubled brace raises KeyError.
    rendered = qg.HOOK_JUDGE_PROMPT.format(hook="h", transcript="t")
    assert "{\"supported\":" in rendered
    assert "HOOK: h" in rendered
    assert "TRANSCRIPT:\nt" in rendered


# ---------------------------------------------------------------- layout match

def test_layout_match_pass_on_exact_match():
    plan = {"layout": "chart_top_cam_bottom"}
    camp = {"layout": {"layout": "chart_top_cam_bottom"}}
    m = qg.measure_layout_match(plan, camp)
    assert m["passed"] is True


def test_layout_match_fail_on_mismatch():
    plan = {"layout": "track"}
    camp = {"layout": {"layout": "chart_top_cam_bottom"}}
    m = qg.measure_layout_match(plan, camp)
    assert m["passed"] is False
    assert "chart_top_cam_bottom" in (m.get("errors") or ["x"])[0]


def test_layout_match_informational_when_missing():
    plan = {"layout": "track"}
    m = qg.measure_layout_match(plan, campaign=None)
    assert m["passed"] is None


def test_layout_match_product_full_bleed_requires_exact_name():
    # product_full_bleed MUST land under its own name — a plain `track` plan is a fail.
    # (a review finding: a product source was rendered through generic `track`, not
    # product_full_bleed, so the top chip never applied.)
    plan = {"layout": "track"}
    camp = {"layout": {"layout": "product_full_bleed"}}
    m = qg.measure_layout_match(plan, camp)
    assert m["passed"] is False
    plan2 = {"layout": "product_full_bleed"}
    m2 = qg.measure_layout_match(plan2, camp)
    assert m2["passed"] is True


def test_layout_match_chart_top_cam_bottom_allows_broadcast_alias():
    # chart_top_cam_bottom <-> broadcast is the one deliberate alias: a direct caller of
    # reframe.analyze(force="broadcast") still produces a legitimate stacked chart layout.
    plan = {"layout": "broadcast"}
    camp = {"layout": {"layout": "chart_top_cam_bottom"}}
    m = qg.measure_layout_match(plan, camp)
    assert m["passed"] is True
    # But `track` is not an alias — a track plan for a chart_top_cam_bottom campaign fails.
    plan2 = {"layout": "track"}
    m2 = qg.measure_layout_match(plan2, camp)
    assert m2["passed"] is False


# ---------------------------------------------------------------- overlay edge-cut

def test_overlay_cut_by_edge_detects_half_cropped_banner():
    # Banner from x=-10 (so x0=0 touching left edge) stretching to x=200 in a 1080 frame.
    assert qg._overlay_cut_by_edge((0, 500, 200, 540), src_w=1080, src_h=1920) is True


def test_overlay_cut_by_edge_passes_full_width_band():
    # A full-width band touches both edges — that is a legitimate full-width overlay.
    assert qg._overlay_cut_by_edge((0, 50, 1080, 90), src_w=1080, src_h=1920) is False


def test_overlay_cut_by_edge_passes_middle_banner():
    # A banner safely inside the frame — not cut.
    assert qg._overlay_cut_by_edge((100, 50, 980, 90), src_w=1080, src_h=1920) is False


# ---------------------------------------------------------------- active-speaker (plan side)

def test_active_speaker_plan_pass_when_segments_match_speaker():
    plan = {
        "speaker_segments": [
            {"t0": 0.0, "t1": 10.0, "track_id": 2, "cx": 0.3},
            {"t0": 10.0, "t1": 20.0, "track_id": 5, "cx": 0.7},
        ],
        "segments": [
            {"t0": 0.0, "t1": 10.0, "x": 100, "cut": True, "speaker": 2},
            {"t0": 10.0, "t1": 20.0, "x": 1000, "cut": True, "speaker": 5},
        ],
    }
    m = qg.measure_active_speaker_plan(plan)
    assert m["passed"] is True
    assert m["fraction_on_speaker"] == 1.0


def test_active_speaker_plan_fail_when_segments_frame_wrong_speaker():
    plan = {
        "speaker_segments": [
            {"t0": 0.0, "t1": 10.0, "track_id": 2, "cx": 0.3},
            {"t0": 10.0, "t1": 20.0, "track_id": 5, "cx": 0.7},
        ],
        "segments": [
            # plan frames speaker=2 for the whole clip even though diarization switches at 10 s
            {"t0": 0.0, "t1": 20.0, "x": 100, "cut": True, "speaker": 2},
        ],
    }
    m = qg.measure_active_speaker_plan(plan)
    assert m["passed"] is False
    assert m["fraction_on_speaker"] == 0.5     # 10 s matched / 20 s total speech


def test_active_speaker_plan_informational_without_speaker_segments():
    # Single-speaker clips (full_bleed_speaker on a solo vlog) have no speaker_segments.
    plan = {"segments": [{"t0": 0.0, "t1": 10.0, "x": 100, "cut": True}]}
    m = qg.measure_active_speaker_plan(plan)
    assert m["passed"] is None


# ---------------------------------------------------------------- layout_quality

def test_layout_quality_informational_for_non_chart_layouts():
    camp = {"layout": {"layout": "podcast_two_shot"}}
    m = qg.measure_layout_quality("/tmp/does-not-exist.mp4", camp)
    assert m["passed"] is None


# ---------------------------------------------------------------- head/tail track carry

def test_head_tail_track_carry_exempts_leading_nones_before_stable_anchor():
    """A chart_top_cam_bottom clip whose first 3 frames classify as 'none' but then
    runs a stable 'full' anchor for 1.0 s must have those openings exempted via Pass B
    (track-carry-forward). YuNet misses the small cam-panel face at the opening but
    the subject is clearly there a few frames later."""
    fps = 10.0
    # frames 0-2 are 'none' (bad detection on first 0.3 s); frames 3-14 are 'full'
    classified = ["none", "none", "none"] + ["full"] * 12
    times = [i / fps for i in range(len(classified))]
    flips, exempt = qg._apply_head_tail_track_carry(
        classified, times, fps, video_path="/tmp/missing",
        head_s=1.0, tail_s=1.0, carry_s=1.0, min_anchor_s=0.5)
    # Pass A upscale-retry is skipped because video_path is missing - returns [].
    # Pass B carry-forward exempts frames 0, 1, 2 because frames 3..7 (0.5 s anchor)
    # are all 'full'.
    assert exempt == {0, 1, 2}, exempt


def test_head_tail_track_carry_does_not_exempt_dead_opening():
    """When the whole first second is 'none' (no stable anchor within carry_s), the
    carry-forward exemption is correctly withheld — a dead opening still fails."""
    fps = 10.0
    classified = ["none"] * 15 + ["full"] * 10   # 1.5 s of 'none' then 'full'
    times = [i / fps for i in range(len(classified))]
    flips, exempt = qg._apply_head_tail_track_carry(
        classified, times, fps, video_path="/tmp/missing",
        head_s=1.0, carry_s=1.0, min_anchor_s=0.5)
    # frames 0..9 are in the head window; none have a 'full' anchor starting within
    # the next 1 s because the stable 'full' run only begins at frame 15.
    assert exempt == set(), exempt


def test_head_tail_track_carry_exempts_trailing_nones_after_stable_anchor():
    """Mirror case at clip tail: a stable 'full' run ending with a short 'none' tail
    must have the tail exempted via the look-behind arm of Pass B."""
    fps = 10.0
    # 2 s of 'full' then 2 trailing 'none' frames at the tail of a short clip
    classified = ["full"] * 20 + ["none", "none"]
    times = [i / fps for i in range(len(classified))]
    flips, exempt = qg._apply_head_tail_track_carry(
        classified, times, fps, video_path="/tmp/missing",
        head_s=1.0, tail_s=1.0, carry_s=1.0, min_anchor_s=0.5)
    assert exempt == {20, 21}, exempt


def test_head_tail_params_flag_defaults_false():
    """Unknown layouts (and the default) do NOT opt into head/tail track carry; the
    strict default still reason_codes a dead opening without this flag."""
    p_default = qg._layout_defaults()
    assert p_default["exempt_head_tail_track_carry"] is False
    # Explicit layouts: chart_top_cam_bottom opts in; others do not.
    chart = qg.layout_params({"layout": {"layout": "chart_top_cam_bottom"}})
    assert chart["exempt_head_tail_track_carry"] is True
    track = qg.layout_params({"layout": {"layout": "track"}})
    assert track["exempt_head_tail_track_carry"] is False
    full_bleed = qg.layout_params({"layout": {"layout": "full_bleed_speaker"}})
    assert full_bleed["exempt_head_tail_track_carry"] is False


# ---------------------------------------------------------------- chart content fraction

def test_chart_content_vertical_fraction_small_candle_strip():
    """A panel that only has candles in a 15 % tall strip in the middle (a chart
    clip case) reports a content fraction < 50 %. Measure returns the content bbox height
    divided by panel height."""
    pytest = __import__("pytest")
    try:
        import numpy as np                       # noqa: F401
        import cv2                                # noqa: F401
    except ImportError:
        pytest.skip("numpy/cv2 unavailable")
    import numpy as np
    panel = np.zeros((1312, 1080, 3), dtype=np.uint8)
    # draw a strip of green/red "candles" between y=550 and y=750 (height = 200, 15 %)
    for x in range(100, 1080 - 100, 30):
        colour = (0, 200, 0) if (x // 30) % 2 == 0 else (0, 0, 200)
        panel[550:750, x:x + 10] = colour
    frac, det = qg._chart_content_vertical_fraction(panel)
    assert frac is not None and 0.10 <= frac <= 0.25
    assert det["panel_h"] == 1312


def test_chart_content_vertical_fraction_full_bleed_candles():
    """A panel where the candles fill 70 % of the vertical extent reports a content
    fraction >= 0.5 (the pass bar)."""
    pytest = __import__("pytest")
    try:
        import numpy as np                       # noqa: F401
        import cv2                                # noqa: F401
    except ImportError:
        pytest.skip("numpy/cv2 unavailable")
    import numpy as np
    panel = np.zeros((1312, 1080, 3), dtype=np.uint8)
    # candles spanning y=100 to y=1050 (height = 950, ~72 %)
    for x in range(50, 1080 - 50, 25):
        colour = (0, 220, 0) if (x // 25) % 2 == 0 else (0, 0, 220)
        panel[100:1050, x:x + 8] = colour
    frac, det = qg._chart_content_vertical_fraction(panel)
    assert frac >= 0.5, (frac, det)


def test_chart_content_vertical_fraction_near_black_panel():
    """A near-all-black panel reports content fraction near zero (nothing qualifies)."""
    pytest = __import__("pytest")
    try:
        import numpy as np                       # noqa: F401
        import cv2                                # noqa: F401
    except ImportError:
        pytest.skip("numpy/cv2 unavailable")
    import numpy as np
    panel = np.zeros((1312, 1080, 3), dtype=np.uint8)
    frac, det = qg._chart_content_vertical_fraction(panel)
    assert frac == 0.0


# ---------------------------------------------------------------- per-frame layout_quality

def test_panel_near_black_detects_fully_dark_panel():
    """A chart panel that goes fully black (brightest channel <= 15 across the whole
    panel) reports a near-black fraction >= :data:`qg.CHART_PANEL_BLACK_FRAC_MIN` and
    thus fails the any-black-frame rule."""
    pytest = __import__("pytest")
    try:
        import numpy as np                       # noqa: F401
    except ImportError:
        pytest.skip("numpy unavailable")
    import numpy as np
    panel = np.zeros((1312, 1080, 3), dtype=np.uint8)
    f = qg._panel_near_black_fraction(panel)
    assert f >= qg.CHART_PANEL_BLACK_FRAC_MIN
    # a mostly-chart panel (candles over dark bg) must NOT trip the black-panel rule
    panel2 = np.zeros((1312, 1080, 3), dtype=np.uint8)
    panel2[100:1050, 100:980] = (0, 180, 0)       # a wide colored body
    f2 = qg._panel_near_black_fraction(panel2)
    assert f2 < qg.CHART_PANEL_BLACK_FRAC_MIN


def test_window_fail_1s_detects_sub_second_dip_in_1s_mean():
    """The 1 s sliding window picks up a localized dip in chart content that the
    clip-median hides. Simulate 20 samples at 4 fps where the clip-median content
    stays at 0.60 but a 1 s stretch collapses to 0.10 — the window gate catches it."""
    fps = 4.0
    content = [0.60] * 20
    for i in range(8, 12):      # 4 consecutive samples under the min
        content[i] = 0.10
    fail = qg._window_fail_1s(content, fps, lambda m, lim: m < lim,
                                qg.CHART_CONTENT_WINDOW_FRAC_MIN)
    assert fail is not None
    wi, wm = fail
    assert 0.0 <= wm < qg.CHART_CONTENT_WINDOW_FRAC_MIN


def test_window_fail_1s_passes_when_mean_stays_above_min():
    """Clip-wide content above the window min returns None (no failing window)."""
    fps = 4.0
    content = [0.55] * 40
    fail = qg._window_fail_1s(content, fps, lambda m, lim: m < lim,
                                qg.CHART_CONTENT_WINDOW_FRAC_MIN)
    assert fail is None


def test_window_fail_1s_bar_cap_detects_single_bad_second():
    """A 1 s window where the mean black-bar fraction exceeds 3% must fail even when
    the clip-median black-bar fraction is 0% (the earlier median-only gate).

    This simulates a chart clip case: 38 samples at 4 fps where most frames
    have no black bars but a few frames near the end have 20% black bars."""
    fps = 4.0
    bars = [0.0] * 152               # ~38 s of 4 fps samples
    for i in range(140, 148):        # 2 s of 20% black bars near the tail
        bars[i] = 0.20
    fail = qg._window_fail_1s(bars, fps, lambda m, lim: m > lim,
                                qg.CHART_WINDOW_BAR_FRAC_MAX)
    assert fail is not None
    wi, wm = fail
    assert wm > qg.CHART_WINDOW_BAR_FRAC_MAX


def test_layout_quality_sample_fps_at_least_4():
    """The design locks sampling at >= 4 fps so sub-second episodes do not slip
    between samples. The module constant must match."""
    assert qg.CHART_SAMPLE_FPS >= 4.0
