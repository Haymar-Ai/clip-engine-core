"""Selection-hardening tests — self-contained, no network, no ffmpeg.

Covers the adaptive yield floor + suppression (_pick_windows, the per-second ranker behind
engine.energy_peaks). Cuts are placed in inter-word gaps (tests/test_boundaries.py,
tests/test_select_v2.py).
"""
from pathlib import Path

PIPE = Path(__file__).resolve().parent.parent / "clipengine"


# ---------------------------------------------------------------------------
# 1. yield floor
# ---------------------------------------------------------------------------
def _pick(scores, **kw):
    from clipengine import engine
    return engine._pick_windows(scores, suppress=1, **kw)


def test_floor_keeps_peaks_drops_flat():
    # one clear spike among flat noise -> exactly the spike clears mean+std
    scores = [10] * 40 + [100] + [10] * 40
    picks = _pick(scores, floor_k=1.0, min_keep=0, top_n=5)
    assert len(picks) == 1
    assert picks[0][0] == 40


def test_flat_source_yields_fewer_than_peaky_source():
    peaky = [10] * 30 + [200, 10, 190, 10, 180, 10, 170]
    flat = [50] * 30                                 # truly uniform -> no peak clears the floor
    kw = dict(floor_k=1.0, min_keep=0, top_n=5)
    assert len(_pick(peaky, **kw)) > len(_pick(flat, **kw))


def test_min_keep_zero_can_emit_nothing():
    # no window is a >1-std outlier on a uniform source
    assert _pick([50] * 30, floor_k=1.0, min_keep=0, top_n=5) == []


def test_min_keep_forces_floor_override():
    picks = _pick([50] * 30, floor_k=1.0, min_keep=2, top_n=5)
    assert len(picks) == 2


def test_top_n_caps_yield():
    scores = list(range(200, 100, -1))          # 100 strong descending windows
    assert len(_pick(scores, floor_k=0.0, min_keep=0, top_n=3)) == 3


def test_suppression_spaces_picks():
    from clipengine import engine
    scores = [100, 99, 98, 97] + [10] * 50
    picks = engine._pick_windows(scores, floor_k=0.0, min_keep=0, top_n=5, suppress=10)
    idxs = sorted(p[0] for p in picks)
    assert all(b - a >= 10 for a, b in zip(idxs, idxs[1:]))


def test_head_skip_excludes_cold_open():
    from clipengine import engine
    # loudest window is the intro at index 2; head_skip must exclude it
    scores = [10] * 200
    scores[2] = 999          # intro sting
    scores[120] = 500        # real content peak
    picks = engine._pick_windows(scores, floor_k=1.0, min_keep=0, top_n=5,
                                 suppress=1, head_skip=90, tail_skip=30)
    idxs = [p[0] for p in picks]
    assert 2 not in idxs and 120 in idxs


def test_tail_skip_excludes_outro():
    from clipengine import engine
    scores = [10] * 200
    scores[195] = 999        # outro card / end sting
    scores[100] = 500
    picks = engine._pick_windows(scores, floor_k=1.0, min_keep=0, top_n=5,
                                 suppress=1, head_skip=90, tail_skip=30)
    idxs = [p[0] for p in picks]
    assert 195 not in idxs and 100 in idxs
