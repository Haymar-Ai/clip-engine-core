"""Pick-quality unit tests. The real face detector + ffmpeg sampler are
monkeypatched so the test is pure and <0.1 s. Covers: visibility fraction math,
threshold filter, missing-start/end, fail-open on detector failure."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from clipengine import pick_quality


class _FakeReframe:
    """Fake render.reframe used by pick_quality via its _reframe() helper."""

    def __init__(self, src_size=(1920, 1080), face_pattern=None, raise_on_probe=False,
                 raise_on_iter=False):
        self.src_size = src_size
        self.face_pattern = face_pattern or []    # list of bool per sample
        self.raise_on_probe = raise_on_probe
        self.raise_on_iter = raise_on_iter
        self.calls = {"probe": 0, "iter": 0, "detect": 0}

    def probe_size(self, src):
        self.calls["probe"] += 1
        if self.raise_on_probe:
            raise RuntimeError("probe boom")
        return self.src_size

    def iter_frames(self, src, start, dur, fps=1.0, src_size=None):
        self.calls["iter"] += 1
        if self.raise_on_iter:
            raise RuntimeError("iter boom")
        for i, has_face in enumerate(self.face_pattern):
            yield (float(i) / float(fps), ("frame", i, has_face))

    def detect_faces(self, frame):
        self.calls["detect"] += 1
        if isinstance(frame, tuple) and len(frame) == 3 and frame[0] == "frame":
            return [{"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.2}] if frame[2] else []
        return []


@pytest.fixture
def install_fake(monkeypatch):
    def _install(fake):
        monkeypatch.setattr(pick_quality, "_reframe", lambda: fake)
        return fake
    return _install


def test_face_visibility_fraction_full_window(install_fake):
    install_fake(_FakeReframe(face_pattern=[True] * 10))
    r = pick_quality.face_visibility_fraction("x.mp4", 0.0, 10.0, fps=1.0)
    assert r["frac"] == 1.0
    assert r["n_total"] == 10 and r["n_with_face"] == 10


def test_face_visibility_fraction_card_pull_window(install_fake):
    # A handheld card-pull: host face visible for 83 % of the 1 fps samples.
    pattern = [True, True, False, False, True, True, True, True, True, True]
    install_fake(_FakeReframe(face_pattern=pattern))
    r = pick_quality.face_visibility_fraction("x.mp4", 0.0, 10.0, fps=1.0)
    assert r["n_total"] == 10 and r["n_with_face"] == 8
    assert r["frac"] == pytest.approx(0.8)


def test_filter_picks_by_face_visibility_drops_card_pull_windows(install_fake):
    """A handheld card-pull window (face visible 80 %) is dropped under a
    98 % bar; a clean talking-head window (100 %) passes."""
    fake = _FakeReframe(face_pattern=[True] * 10)
    install_fake(fake)
    picks = [{"id": "clean", "start_s": 0.0, "end_s": 10.0},
             {"id": "card_pull", "start_s": 10.0, "end_s": 20.0}]
    # Patch per-call: first call returns full visibility, second returns 0.8.
    orig = pick_quality.face_visibility_fraction
    calls = {"n": 0}

    def _fake_vis(src, s, e, fps=1.0):
        calls["n"] += 1
        return {"frac": 1.0 if calls["n"] == 1 else 0.8,
                "n_total": 10, "n_with_face": 10 if calls["n"] == 1 else 8}

    pick_quality.face_visibility_fraction = _fake_vis
    try:
        out = pick_quality.filter_picks_by_face_visibility("x.mp4", picks,
                                                            min_frac=0.98)
    finally:
        pick_quality.face_visibility_fraction = orig
    assert len(out) == 1 and out[0]["id"] == "clean"
    assert out[0]["face_visibility"]["frac"] == 1.0


def test_filter_picks_fail_open_on_measurement_failure(install_fake):
    """When probing or detection fails (frac is None), the pick stays — a
    broken detector must not silently empty the selector."""
    install_fake(_FakeReframe(raise_on_probe=True))
    picks = [{"id": "x", "start_s": 0.0, "end_s": 10.0}]
    out = pick_quality.filter_picks_by_face_visibility("x.mp4", picks,
                                                        min_frac=0.98)
    assert len(out) == 1 and out[0]["id"] == "x"
    assert out[0]["face_visibility"]["frac"] is None


def test_filter_picks_missing_start_end_is_marked_but_kept(install_fake):
    install_fake(_FakeReframe(face_pattern=[True] * 5))
    picks = [{"id": "x", "end_s": 10.0}]       # no start_s
    out = pick_quality.filter_picks_by_face_visibility("x.mp4", picks)
    assert len(out) == 1 and out[0]["face_visibility"]["frac"] is None
    assert "missing" in out[0]["face_visibility"]["reason"]


# ---- source-side frame-gate prediction (predict_frame_gate) ----
def test_predict_frame_gate_passes_full_talking_head(install_fake):
    install_fake(_FakeReframe(face_pattern=[True] * 60))
    r = pick_quality.predict_frame_gate("x.mp4", 0.0, 30.0, fps=2.0)
    assert r["passed"] is True and r["fail_pct"] == 0.0 and r["frac"] == 1.0


def test_predict_frame_gate_fails_scattered_no_face_samples_over_cap(install_fake):
    # screencast-ish window: the face drops out for single samples 4 times in 60 (6.7 %)
    pattern = [True] * 60
    for i in (10, 25, 40, 55):
        pattern[i] = False
    install_fake(_FakeReframe(face_pattern=pattern))
    r = pick_quality.predict_frame_gate("x.mp4", 0.0, 30.0, fps=2.0)
    assert r["passed"] is False and r["fail_pct"] == pytest.approx(6.67)


def test_predict_frame_gate_exempts_long_broll_run_like_the_gate(install_fake):
    # one 4 s b-roll cutaway (8 samples at 2 fps) is exempt (>= 3 s), as the render gate does
    pattern = [True] * 20 + [False] * 8 + [True] * 32
    install_fake(_FakeReframe(face_pattern=pattern))
    r = pick_quality.predict_frame_gate("x.mp4", 0.0, 30.0, fps=2.0)
    assert r["passed"] is True and r["broll_exempt"] == 8 and r["fail_pct"] == 0.0


def test_predict_frame_gate_broll_exemption_is_capped_at_half_the_clip(install_fake):
    install_fake(_FakeReframe(face_pattern=[False] * 40 + [True] * 20))
    r = pick_quality.predict_frame_gate("x.mp4", 0.0, 30.0, fps=2.0)
    assert r["passed"] is False and r["broll_exempt"] == 0


def test_predict_frame_gate_fails_open_when_source_unreadable(install_fake):
    install_fake(_FakeReframe(raise_on_probe=True))
    assert pick_quality.predict_frame_gate("x.mp4", 0.0, 30.0)["passed"] is None
    install_fake(_FakeReframe(face_pattern=[]))
    assert pick_quality.predict_frame_gate("x.mp4", 0.0, 30.0)["passed"] is None
