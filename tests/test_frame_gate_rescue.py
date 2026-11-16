"""Frame-gate tests: the downscaled re-detect of ``none`` frames and the speaker-less layout
rule. No ffmpeg: frames come from a fake ``iter_frames`` and faces from a fake detector keyed on
the frame's fill value, so only cv2.resize runs for real.

YuNet at native 1080x1920 misses a 9:16 close-up face (a track layout with the face plainly on
screen read as 12-19 % "none" frames); at 0.5x the same frames detect. The fake detector models
exactly that: a FAR face is found at any width, a CLOSE face only once the frame is <= 600 px
wide.

The share of failing frames a clip may have comes from config
(quality_gate.default.gate_fail_pct_max). Tests that count a few failing frames against a cap
pin it to STRICT_CAP, so they do not depend on how the config tunes it.
"""
import numpy as np
import pytest

from clipengine import config
from clipengine import quality_gate as qg
from clipengine.render import reframe

STRICT_CAP = 2.0
SW, SH = 1080, 1920
NOFACE, CLOSE, FAR, EDGE = 0, 1, 2, 3        # frame fill value -> what the fake detector sees
_INSIDE = {"cx": 0.5, "cy": 0.4, "w": 0.5, "h": 0.3, "score": 0.9}
_CUT = {"cx": 0.1, "cy": 0.4, "w": 0.3, "h": 0.3, "score": 0.9}   # bbox crosses the left edge


def fake_detect(frame):
    kind, w = int(frame[0, 0, 0]), frame.shape[1]
    if kind == CLOSE:
        return [dict(_INSIDE)] if w <= 600 else []
    if kind == FAR:
        return [dict(_INSIDE)]
    if kind == EDGE:                          # cut at native res; a downscale would look "full"
        return [dict(_CUT)] if w > 600 else [dict(_INSIDE)]
    return []


@pytest.fixture()
def frames(monkeypatch):
    """Install a frame sequence: ``frames([kind, ...])`` at 10 fps."""
    tiles = {k: np.full((SH, SW, 3), k, np.uint8) for k in (NOFACE, CLOSE, FAR, EDGE)}

    def install(kinds):
        def it(path, fps=qg.FRAME_FPS):
            for i, k in enumerate(kinds):
                yield round(i / fps, 3), tiles[k], SW, SH
        monkeypatch.setattr(qg, "iter_frames", it)
        monkeypatch.setattr(qg, "probe_duration", lambda p, kind="format": len(kinds) / qg.FRAME_FPS)
        monkeypatch.setattr(reframe, "detect_faces", fake_detect)
    return install


def _mid(n, at, kind, base=FAR):
    """`n` frames of `base` with `kind` at the indices in `at` (kept away from head/tail)."""
    out = [base] * n
    for i in at:
        out[i] = kind
    return out


def _spread(k):
    """`k` isolated frame indices of a 100-frame clip, clear of its first and last second (the
    head/tail rule) and never adjacent (no contiguous no-face run the b-roll rule could take)."""
    return [15 + 70 * i // max(1, k) for i in range(k)]


def _strict(params):
    return {**params, "fail_pct_max": STRICT_CAP}


def _config_cap():
    return float(config.load()["quality_gate"]["default"]["gate_fail_pct_max"])


def test_close_up_none_frames_are_rescued_by_downscale(frames):
    frames([CLOSE] * 100)
    r = qg.measure_frame_gate("clip.mp4", params=qg._layout_defaults())
    assert r["passed"] is True and r["fail_pct"] == 0.0
    assert r["rescaled_rescue_count"] == 100 and r["none_count"] == 0


def test_rescue_returns_none_when_no_scale_finds_a_face():
    assert qg._rescue_none_frame(np.zeros((SH, SW, 3), np.uint8), SW, SH, fake_detect) == "none"


def test_edge_cut_frames_are_never_retried(frames):
    # 3 edge-cut frames would look "full" at 0.5x; they must still count -> 3 % > a 2 % cap.
    frames(_mid(100, (30, 50, 70), EDGE))
    r = qg.measure_frame_gate("clip.mp4", params=_strict(qg._layout_defaults()))
    assert r["edge_cut_count"] == 3 and r["fail_count"] == 3
    assert r["passed"] is False and r["rescaled_rescue_count"] == 0


@pytest.mark.parametrize("over,passed", [(0, True), (1, False)])
def test_faceless_frames_still_fail_at_the_configured_cap(frames, over, passed):
    """The rescue only recovers frames a downscale finds a face in: faceless frames still count,
    against the configured cap, with or without a speaker layout."""
    cap = _config_cap()
    misses = int(cap) + over
    frames(_mid(100, _spread(misses), NOFACE))
    for camp in (None, {"layout": {"layout": "full_bleed_speaker"}}):
        lp = qg.layout_params(camp)
        assert lp["fail_pct_max"] == cap
        r = qg.measure_frame_gate("clip.mp4", params=lp)
        assert r["fail_count"] == misses and r["passed"] is passed


def test_speakerless_layout_ignores_no_face_but_fails_edge_cut(frames):
    lp = {**_strict(qg._layout_defaults()), "require_subject": False}
    frames([NOFACE] * 50 + [FAR] * 50)            # b-roll / screen with no face: nothing to lose
    assert qg.measure_frame_gate("clip.mp4", params=lp)["passed"] is True
    frames(_mid(100, (30, 50, 70), EDGE, base=NOFACE))
    r = qg.measure_frame_gate("clip.mp4", params=lp)
    assert r["fail_count"] == 3 and r["passed"] is False


def _gate_params(monkeypatch, **kw):
    seen = {}

    def capture(video_path, card_pull_windows=None, fps=qg.FRAME_FPS, params=None):
        seen.update(params)
        return {"passed": True}
    monkeypatch.setattr(qg, "measure_frame_gate", capture)
    monkeypatch.setattr(qg, "measure_av_sync", lambda p: {"passed": None})
    qg.gate("clip.mp4", run_overlay_check=False, **kw)
    return seen


def test_gate_drops_face_presence_rule_only_for_speakerless_render_without_campaign_layout(
        monkeypatch):
    assert _gate_params(monkeypatch, rendered_layout="general")["require_subject"] is False
    assert _gate_params(monkeypatch, rendered_layout="game_only")["require_subject"] is False
    assert "require_subject" not in _gate_params(monkeypatch, rendered_layout="track")
    assert "require_subject" not in _gate_params(monkeypatch)
    # a campaign that names a speaker layout keeps the full rule even if recompose fell back
    # to a full-frame render: a full_bleed_speaker clip never loses the face rule.
    lp = _gate_params(monkeypatch, rendered_layout="general",
                      campaign={"layout": {"layout": "full_bleed_speaker"}})
    assert "require_subject" not in lp and lp["fail_pct_max"] == _config_cap()
