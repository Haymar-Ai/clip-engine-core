"""The mild levels lift (recompose.measure_levels / auto_levels / with_levels) measures every
vertical's window, not only streams': most too-dark rejects were dark studios (podcasts,
interviews) that got no lift.
"""
import numpy as np
import pytest

from clipengine import config
from clipengine.render import recompose

CFG = config.load()


def _frames(monkeypatch, *values):
    fr = [np.full((90, 160, 3), v, np.uint8) for v in values]
    for f in fr:
        f[:8] = 230                       # a little highlight, like a lamp or a caption
    monkeypatch.setattr(recompose.reframe, "iter_frames",
                        lambda *a, **k: iter([(float(i), f) for i, f in enumerate(fr)]))


def test_every_configured_vertical_is_levelled():
    known = set(CFG["select"]["verticals"]) | set(CFG["render"]["vertical_layouts"])
    assert known <= set(recompose.LEVELS_VERTICALS)
    assert "" in recompose.LEVELS_VERTICALS            # a render with no --vertical


@pytest.mark.parametrize("vertical", ["podcast", "general", "interview", "poker", "stream"])
def test_a_dark_window_of_any_vertical_gets_the_mild_lift(monkeypatch, vertical):
    _frames(monkeypatch, 20, 24, 28, 22)               # a dark studio, median luma ~ 35
    a = recompose.parse_args(["s.mp4", "0", "8", "o.mp4", "--vertical", vertical])
    lv = recompose.measure_levels(a, "s.mp4")
    assert lv and lv["src_luma"] < recompose.DARK_LUMA
    assert lv["filter"].startswith("colorlevels=") and lv["black"] <= recompose.LEVELS_MAX_BLACK
    assert 255 / (lv["white"] - lv["black"]) <= recompose.LEVELS_MAX_GAIN + 1e-6
    graph = recompose.with_levels("[0:v]crop=607:1080:0:0,scale=1080:1920[v0]", lv["filter"])
    assert graph.startswith(f"[0:v]{lv['filter']}[lv0];[lv0]crop=")


def test_a_bright_or_black_window_is_left_alone(monkeypatch):
    a = recompose.parse_args(["s.mp4", "0", "8", "o.mp4", "--vertical", "podcast"])
    _frames(monkeypatch, 110, 120, 125)
    assert recompose.measure_levels(a, "s.mp4") is None
    flat = [np.zeros((90, 160, 3), np.uint8)] * 3      # a black window: nothing to stretch
    monkeypatch.setattr(recompose.reframe, "iter_frames",
                        lambda *a, **k: iter([(float(i), f) for i, f in enumerate(flat)]))
    assert recompose.measure_levels(a, "s.mp4") is None


def test_an_unknown_vertical_still_needs_the_levels_flag(monkeypatch):
    _frames(monkeypatch, 20, 24, 28)
    a = recompose.parse_args(["s.mp4", "0", "8", "o.mp4", "--vertical", "campfire"])
    assert recompose.measure_levels(a, "s.mp4") is None
    a = recompose.parse_args(["s.mp4", "0", "8", "o.mp4", "--vertical", "campfire", "--levels"])
    assert recompose.measure_levels(a, "s.mp4") is not None
