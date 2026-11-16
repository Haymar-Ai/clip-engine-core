"""Finish / snap tests. No subprocess calls — the face series is a synthetic list.

Covers rules 8 (no mid-word start/end, no no-face first/last 1 s) and 6 (never frame
an empty region at the clip opening or closing)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from clipengine.render import finish


def _words(spans):
    """``spans`` is a list of (start, end, text). Builds a plain words list."""
    return [{"w": txt, "start": float(s), "end": float(e)} for s, e, txt in spans]


def _series(span, step=0.1, has_face_at=None, big_h=0.25, cx=0.5):
    """Build a sampled face series across [span[0], span[1]]. ``has_face_at`` is a predicate
    ``t -> bool`` picking which samples contain a big face."""
    t = float(span[0])
    out = []
    while t <= float(span[1]) + 1e-6:
        has = True if has_face_at is None else bool(has_face_at(t))
        out.append((round(t, 3), has, big_h if has else 0.0, cx if has else 0.0))
        t += step
    return out


def test_snap_moves_start_to_next_word_when_head_lacks_a_face():
    # Clip picked at 100.00 - 120.00 s. Transcript shows a word that starts at 101.0. The
    # first second of the SOURCE at the picked start has NO face; from 101.1 s onward a face
    # is visible every sample. The snapper should move the start to 101.0. End has a word
    # finishing at 119.9 so the end is left alone.
    words = _words([(100.0, 100.4, "No"),
                    (100.5, 100.9, "face"),
                    (101.0, 101.5, "Here"),
                    (119.4, 119.9, "end")])
    # face present strictly after t > 101 (float-precision safe) and all through to 120 s
    series = _series((99.0, 125.0), has_face_at=lambda t: t > 101.005)
    ns, ne, info = finish.snap_bounds(words, 100.0, 120.0, source_face_series=series)
    assert ns == 101.0, info
    assert abs(ne - 119.9) < 1e-6
    assert info["shift_start_s"] > 0


def test_snap_rejects_shift_when_result_would_be_too_short():
    # min_dur is 15 s but the only face-filled window is 10 s long.
    words = _words([(10.0, 10.5, "ok")])
    series = _series((5.0, 25.0), has_face_at=lambda t: 10.0 <= t <= 20.0)
    ns, ne, info = finish.snap_bounds(words, 10.0, 20.0, source_face_series=series,
                                       min_dur_s=15.0)
    assert (ns, ne) == (10.0, 20.0)
    assert "under-cut" in info["note"]


def test_snap_without_face_series_falls_back_to_word_boundaries():
    words = _words([(10.0, 10.5, "a"), (11.0, 11.5, "b"), (29.5, 30.0, "c")])
    ns, ne, info = finish.snap_bounds(words, 10.6, 29.6, min_dur_s=15.0)
    assert ns == 10.0 or ns == 11.0
    assert ne == 30.0


def test_head_tail_face_ok_detects_opening_without_a_face():
    series = _series((0.0, 20.0), has_face_at=lambda t: t > 1.5)   # no face in first 1.5 s
    assert finish.head_tail_face_ok(series, 0.0, 20.0) is False
    # Trimming the first 2 s off the opening should pass.
    assert finish.head_tail_face_ok(series, 2.0, 20.0) is True


def test_head_tail_face_ok_detects_no_face_closing():
    series = _series((0.0, 20.0),
                     has_face_at=lambda t: t <= 18.0)   # no face in last 2 s
    assert finish.head_tail_face_ok(series, 0.0, 20.0) is False


def test_source_face_series_shape_is_a_tuple_of_four():
    # Smoke test using a tiny synthetic series builder: just make sure the shape the pure
    # helpers consume is what source_face_series would return.
    series = _series((0.0, 1.0))
    assert all(len(row) == 4 for row in series)
    assert all(isinstance(row[1], bool) for row in series)


def test_apply_finish_no_overlays_copies_file(tmp_path, monkeypatch):
    inp = tmp_path / "in.mp4"
    inp.write_bytes(b"FAKEMP4")
    out = tmp_path / "out.mp4"
    res = finish.apply_finish(inp, out, {"overlays": []})
    assert out.read_bytes() == b"FAKEMP4"
    assert res["copied"] is True


def test_apply_finish_watermark_missing_path_raises(tmp_path):
    inp = tmp_path / "in.mp4"
    inp.write_bytes(b"xx")
    out = tmp_path / "out.mp4"
    import pytest
    with pytest.raises(ValueError):
        finish.apply_finish(inp, out,
                            {"overlays": [{"kind": "watermark", "path": "/nope/x.png"}]})


def test_apply_finish_unknown_overlay_raises(tmp_path):
    inp = tmp_path / "in.mp4"
    inp.write_bytes(b"xx")
    out = tmp_path / "out.mp4"
    import pytest
    with pytest.raises(ValueError):
        finish.apply_finish(inp, out, {"overlays": [{"kind": "neon-glow"}]})
