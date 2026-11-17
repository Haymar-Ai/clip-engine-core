"""Frame-gate tests: head/tail tolerance 3 frames (config quality_gate.default) and plain
track's smart measures (config quality_gate.layouts.track: edge-cut transitions, >= 1 s cutaways
inside the clip). No ffmpeg: frames come from a fake ``iter_frames`` and faces from a fake
detector keyed on the frame's fill value (as in test_frame_gate_rescue).

At tolerance 0 one frame with no face in the first or last second failed a clip, and plain track
got none of the cutaway exemptions (a short map insert failed it).
"""
import numpy as np
import pytest

from clipengine import quality_gate as qg
from clipengine.render import reframe

SW, SH = 1080, 1920
NOFACE, FAR = 0, 2
_INSIDE = {"cx": 0.5, "cy": 0.4, "w": 0.5, "h": 0.3, "score": 0.9}


def fake_detect(frame):
    return [dict(_INSIDE)] if int(frame[0, 0, 0]) == FAR else []


@pytest.fixture()
def frames(monkeypatch):
    """Install a frame sequence: ``frames([kind, ...])`` at 10 fps."""
    tiles = {k: np.full((SH, SW, 3), k, np.uint8) for k in (NOFACE, FAR)}

    def install(kinds):
        def it(path, fps=qg.FRAME_FPS):
            for i, k in enumerate(kinds):
                yield round(i / fps, 3), tiles[k], SW, SH
        monkeypatch.setattr(qg, "iter_frames", it)
        monkeypatch.setattr(qg, "probe_duration", lambda p, kind="format": len(kinds) / qg.FRAME_FPS)
        monkeypatch.setattr(reframe, "detect_faces", fake_detect)
        monkeypatch.setattr(qg, "_rescue_none_frame", lambda *a, **k: "none")
    return install


def _seq(n, noface):
    return [NOFACE if i in set(noface) else FAR for i in range(n)]


def _track():
    return qg.layout_params({"layout": {"layout": "track"}})


# ---------------------------------------------------------------- head / tail tolerance
def test_config_head_tail_tolerance_is_three_frames():
    lp = qg._layout_defaults()
    assert lp["head_tol_frames"] == 3 and lp["tail_tol_frames"] == 3
    assert qg.layout_params(None)["head_tol_frames"] == 3


@pytest.mark.parametrize("at", [(0,), (0, 1, 2), (97, 98, 99), (0, 1, 2, 97, 98, 99)])
def test_up_to_three_faceless_frames_at_the_head_or_tail_pass(frames, at):
    frames(_seq(100, at))
    r = qg.measure_frame_gate("clip.mp4", params=qg._layout_defaults())
    assert r["passed"] is True, r
    assert r["first_sec_fail_count"] <= 3 and r["last_sec_fail_count"] <= 3


@pytest.mark.parametrize("at", [(0, 1, 2, 3), (96, 97, 98, 99), (0, 2, 4, 6)])
def test_four_faceless_frames_in_the_first_or_last_second_still_fail(frames, at):
    frames(_seq(100, at))
    r = qg.measure_frame_gate("clip.mp4", params=qg._layout_defaults())
    assert r["passed"] is False
    assert max(r["first_sec_fail_count"], r["last_sec_fail_count"]) == 4


# ---------------------------------------------------------------- plain track's row
def _gate_params(monkeypatch, **kw):
    seen = {}

    def capture(video_path, card_pull_windows=None, fps=qg.FRAME_FPS, params=None):
        seen.update(params)
        return {"passed": True}
    monkeypatch.setattr(qg, "measure_frame_gate", capture)
    monkeypatch.setattr(qg, "measure_av_sync", lambda p: {"passed": None})
    qg.gate("clip.mp4", run_overlay_check=False, **kw)
    return seen


def test_a_track_render_with_no_campaign_layout_reads_the_track_row(monkeypatch):
    lp = _gate_params(monkeypatch, rendered_layout="track")
    assert lp["layout_name"] == "track" and lp["rendered_layout"] == "track"
    assert lp["exempt_edge_cut_transitions"] is True and lp["cutaway_min_span_s"] == 1.0
    assert lp["head_tol_frames"] == 3 and lp["fail_pct_max"] == 8.0
    assert "require_subject" not in lp


def test_other_renders_and_campaign_layouts_keep_their_rows(monkeypatch):
    for kw in ({}, {"rendered_layout": "dual_view"}, {"rendered_layout": "split"}):
        lp = _gate_params(monkeypatch, **kw)
        assert lp["layout_name"] is None and not lp["exempt_edge_cut_transitions"]
        assert not lp["cutaway_min_span_s"]
    assert _gate_params(monkeypatch, rendered_layout="general")["require_subject"] is False
    # a campaign that names its layout keeps it, whatever rendered: full_bleed_speaker has no
    # smart measures (a drifting subject is a hard fail)
    lp = _gate_params(monkeypatch, rendered_layout="track",
                      campaign={"layout": {"layout": "full_bleed_speaker"}})
    assert lp["layout_name"] == "full_bleed_speaker"
    assert not lp["exempt_edge_cut_transitions"] and not lp["cutaway_min_span_s"]


# ---------------------------------------------------------------- cutaways
def test_cutaway_windows_only_inside_the_clip():
    times = [round(i / 10, 3) for i in range(100)]
    inside = [round(i / 10, 3) for i in range(30, 45)]             # 3.0-4.4 s
    opening = [round(i / 10, 3) for i in range(0, 15)]             # 0.0-1.4 s
    closing = [round(i / 10, 3) for i in range(85, 100)]           # 8.5-9.9 s
    short = [round(i / 10, 3) for i in range(60, 68)]              # 0.7 s
    got = qg.cutaway_windows(inside + opening + closing + short, times, 1.0)
    assert got == [(pytest.approx(2.8), pytest.approx(4.6))]
    assert qg.cutaway_windows([], times, 1.0) == []


def test_track_exempts_a_short_cutaway_the_default_rejects(frames):
    frames(_seq(100, range(30, 45)))                               # a 1.5 s map insert
    assert qg.measure_frame_gate("clip.mp4", params=qg._layout_defaults())["passed"] is False
    r = qg.measure_frame_gate("clip.mp4", params=_track())
    assert r["passed"] is True and r["fail_count"] == 0


def test_track_cutaway_that_starts_inside_the_first_second(frames):
    # face 0-0.4 s, a 1.2 s insert from 0.5 s: the transition rule wants the subject
    # established for 1 s first; the cutaway rule exempts it (it is inside the clip)
    frames(_seq(100, range(5, 17)))
    assert qg.measure_frame_gate("clip.mp4", params=qg._layout_defaults())["passed"] is False
    only_transitions = {**_track(), "cutaway_min_span_s": None}
    assert qg.measure_frame_gate("clip.mp4", params=only_transitions)["passed"] is False
    r = qg.measure_frame_gate("clip.mp4", params=_track())
    assert r["passed"] is True and r["first_sec_fail_count"] == 0
    assert r["broll_windows_auto"] == [(pytest.approx(0.3), pytest.approx(1.8))]


@pytest.mark.parametrize("at", [range(0, 20), range(80, 100)])
def test_track_dead_opening_or_closing_still_fails(frames, at):
    frames(_seq(100, at))                                          # 2 s with no face at an edge
    r = qg.measure_frame_gate("clip.mp4", params=_track())
    assert r["passed"] is False
    assert max(r["first_sec_fail_count"], r["last_sec_fail_count"]) > 3


def test_track_lost_subject_over_the_cap_still_fails(frames):
    # 6.1 s with no face inside a 10 s clip: longer than the 4 s transition bound, and its
    # cutaway window is over the b-roll cap (half the clip)
    frames(_seq(100, range(20, 81)))
    r = qg.measure_frame_gate("clip.mp4", params=_track())
    assert r["passed"] is False and r["fail_pct"] > 8.0 and r["broll_windows_auto"] == []
