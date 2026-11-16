"""HUD profile handling (render/hud.py).

PHASE4h: no per-clip auto-detection. Profiles come from the config; a clip window is
sampled at low res every 0.5 s to build per-region presence intervals; those intervals drive
the ffmpeg `enable=` on each band overlay. Pure numpy for the detection code; ffmpeg wrappers
are exercised by the render integration tests.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from clipengine.render import hud
from clipengine import config


# ---------------------------------------------------------------- profile lookup

def _cfg(profiles=None, vertical_map=None):
    return {"render": {"hud_profiles": profiles or {},
                        "vertical_hud_profile": vertical_map or {}}}


def test_load_profile_returns_named_regions():
    prof = {"players": [0.01, 0.56, 0.27, 0.96],
            "pot": [0.73, 0.74, 0.90, 0.96],
            "outs_strip": [0.0, 0.03, 0.82, 0.08],
            "_note": "ignored"}
    out = hud.load_profile(_cfg({"high_stakes_poker": prof}), "high_stakes_poker")
    assert set(out) == {"players", "pot", "outs_strip"}
    assert out["players"] == [0.01, 0.56, 0.27, 0.96]


def test_load_profile_missing_raises_keyerror():
    with pytest.raises(KeyError, match="not in render.hud_profiles"):
        hud.load_profile(_cfg(), "nonexistent")


def test_load_profile_invalid_range_raises_valueerror():
    with pytest.raises(ValueError, match="outside"):
        hud.load_profile(_cfg({"x": {"players": [0.0, 0.0, 1.5, 1.0]}}), "x")
    with pytest.raises(ValueError, match="inverted|outside"):
        hud.load_profile(_cfg({"x": {"players": [0.5, 0.5, 0.4, 0.9]}}), "x")


def test_profile_for_vertical_maps_to_profile_name():
    cfg = _cfg({"high_stakes_poker": {"players": [0.0, 0.5, 0.3, 1.0]}},
               {"poker": "high_stakes_poker"})
    assert hud.profile_for_vertical(cfg, "poker") == "high_stakes_poker"
    assert hud.profile_for_vertical(cfg, "podcast") is None
    assert hud.profile_for_vertical({}, "poker") is None


def test_config_ships_the_poker_profile():
    """The shipped config wires the vertical -> profile map and defines the
    high_stakes_poker profile with the three regions."""
    import json
    cfg = config.load()
    assert hud.profile_for_vertical(cfg, "poker") == "high_stakes_poker"
    prof = hud.load_profile(cfg, "high_stakes_poker")
    assert set(prof) >= {"players", "pot", "outs_strip"}


def test_region_pixels_rounds_to_even_and_clamps():
    x, y, w, h = hud.region_pixels([0.0078, 0.5625, 0.2734, 0.9583], 1280, 720)
    assert x % 1 == 0 and w % 2 == 0 and h % 2 == 0
    # measured on the shipped profile: players box lands around x=10 y=405 w=340 h=286
    assert 8 <= x <= 12
    assert 402 <= y <= 408
    assert w >= 336
    # even sizes
    x, y, w, h = hud.region_pixels([0.0, 0.0, 0.5, 0.5], 101, 101)
    assert w % 2 == 0 and h % 2 == 0


# ---------------------------------------------------------------- presence gating

def _sample_stack_with_hud_visible(hud_visible, w=160, h=90, seed=1):
    """Build len(hud_visible) frames where a red 40x30 rectangle sits at (20, 30) — the "HUD"
    — but is only drawn in samples where hud_visible[i] is True. Everything else is moving noise."""
    rng = np.random.default_rng(seed)
    out = []
    for i, on in enumerate(hud_visible):
        f = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        if on:
            f[30:60, 20:60] = np.array([0, 0, 200], dtype=np.uint8)                 # red
        out.append((round(i * 0.5, 3), f))
    return out


def test_region_crops_slices_each_sample_at_the_region():
    samples = _sample_stack_with_hud_visible([True] * 6)
    region = [20.0 / 160, 30.0 / 90, 60.0 / 160, 60.0 / 90]      # matches the HUD rect
    crops = hud.region_crops(samples, region)
    assert len(crops) == 6
    for _, c in crops:
        assert c.shape == (30, 40, 3)


def _hud_like_stack(hud_visible, w=160, h=90, seed=1):
    """Frames where the "HUD" is a small chessboard rectangle (high edge density) at (20, 30).
    Non-HUD frames replace the box with a uniform grey — low edge density. Everything else is
    moving noise, which stays low-edge in this region because the box IS what's changing."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for i, on in enumerate(hud_visible):
        f = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        if on:                                                   # dense chessboard = HUD edges
            for yy in range(30, 60, 4):
                for xx in range(20, 60, 4):
                    f[yy:yy + 2, xx:xx + 2] = 0
                    f[yy + 2:yy + 4, xx + 2:xx + 4] = 0
        else:                                                     # flat grey = no HUD edges
            f[30:60, 20:60] = 128
        out.append((round(i * 0.5, 3), f))
    return out


def test_region_reference_is_top_edge_density_median():
    """region_reference builds its median from the top-edge-density frames — a mix of HUD and
    non-HUD frames still yields a HUD-looking reference."""
    samples = _hud_like_stack([True] * 12 + [False] * 8)
    region = [20 / 160, 30 / 90, 60 / 160, 60 / 90]
    crops = hud.region_crops(samples, region)
    ref = hud.region_reference(crops)
    assert ref.shape == (30, 40, 3)
    assert hud._edge_density(ref.astype("uint8")) > hud.EDGE_MIN_ABS


def test_presence_marks_flag_high_edge_density_frames():
    """Presence signal is Sobel edge density: HUD-visible frames pass PRESENT_EDGE_MIN, HUD-
    absent frames do not — no reference needed, and it stays correct even when the HUD template
    shifts mid-window (2-tile vs 3-tile poker stack)."""
    visible = [True] * 12 + [False] * 8
    samples = _hud_like_stack(visible)
    region = [20 / 160, 30 / 90, 60 / 160, 60 / 90]
    crops = hud.region_crops(samples, region)
    got = [p for _, p in hud.presence_marks(crops)]
    assert got == visible, got


def test_presence_intervals_merge_small_gaps_and_drop_blips():
    # samples every 0.5s
    marks = [(0.0, True), (0.5, True), (1.0, False), (1.5, True), (2.0, True),
             (2.5, False), (3.0, False), (3.5, False), (4.0, True), (4.5, False),
             (5.0, True), (5.5, True), (6.0, True)]
    ivals = hud.presence_intervals(marks, merge_gap=1.0, min_dur=0.5)
    # first stretch (0..2.25) covers the 0.5s gap -> single interval; the 4.0 blip is 0.5s
    # wide and kept (>= min_dur 0.5); the 5.0-6.25 stretch is a separate interval.
    kinds = [(round(a, 2), round(b, 2)) for a, b in ivals]
    assert kinds[0][0] <= 0.0 and kinds[0][1] >= 2.0                 # spans the bridged gap
    assert kinds[-1][1] >= 6.0                                        # last stretch present at end
    # spans respect min_dur
    for a, b in ivals:
        assert b - a >= 0.5


def test_presence_intervals_drops_tiny_blips_below_min_dur():
    """A single True sample surrounded by False produces one interval whose width equals the
    sample step (0.5s) — kept when min_dur=0.5 but dropped when min_dur=0.6."""
    marks = [(0.0, False), (0.5, True), (1.0, False)]
    assert hud.presence_intervals(marks, min_dur=0.5) == [(0.25, 0.75)]
    assert hud.presence_intervals(marks, min_dur=0.6) == []


def test_enable_expr_between_terms_joined_with_plus():
    assert hud.enable_expr([]) == "0"
    assert hud.enable_expr([(0.0, 3.5)]) == "between(t,0.000,3.500)"
    expr = hud.enable_expr([(0.0, 1.5), (4.0, 5.5)])
    assert expr == "between(t,0.000,1.500)+between(t,4.000,5.500)"


# ---------------------------------------------------------------- band packing

def _shipped_profile():
    import json
    cfg = config.load()
    return hud.load_profile(cfg, "high_stakes_poker")


def test_plan_band_puts_players_left_and_pot_right_no_outs_strip():
    pack = hud.plan_band(_shipped_profile(), 1280, 720)
    assert pack and pack["height"] > 0
    names = [it["name"] for it in pack["items"]]
    assert names == ["players", "pot"], "expected only players + pot in the band, in that order"
    # x order: players before pot
    xs = [it["dst"][0] for it in pack["items"]]
    assert xs == sorted(xs)
    # both fit inside 1080 with 24px margins
    for it in pack["items"]:
        x, y, w, h = it["dst"]
        assert 0 <= x and x + w <= 1080
        assert 0 <= y and y + h <= pack["height"]


def test_plan_band_present_regions_filters_out_missing_pot():
    pack = hud.plan_band(_shipped_profile(), 1280, 720, present_regions={"players"})
    assert pack and [it["name"] for it in pack["items"]] == ["players"]


def test_plan_band_returns_none_when_no_regions_present():
    pack = hud.plan_band(_shipped_profile(), 1280, 720, present_regions=set())
    assert pack is None


def test_plan_band_height_clamped_to_min_max():
    pack = hud.plan_band(_shipped_profile(), 1280, 720)
    assert hud.BAND_MIN_H <= pack["height"] <= hud.BAND_MAX_H
    assert pack["height"] % 2 == 0


def test_plan_band_uses_uniform_scale():
    pack = hud.plan_band(_shipped_profile(), 1280, 720)
    scales = [it["dst"][2] / it["src"][2] for it in pack["items"]]
    for s in scales[1:]:
        assert abs(s - scales[0]) < 0.05


def test_plan_band_scale_cap_prevents_absurd_upscale():
    tiny = {"players": [0.0, 0.0, 0.02, 0.02], "pot": [0.98, 0.98, 1.0, 1.0]}
    pack = hud.plan_band(tiny, 1280, 720)
    scales = [it["dst"][2] / it["src"][2] for it in pack["items"]]
    assert max(scales) <= hud.BAND_SCALE_CAP + 1e-6


# ---------------------------------------------------------------- crop safety

def test_crop_slices_region_detects_partial_overlap():
    # region 300..500, crop 400..800 -> partial overlap -> slices
    assert hud.crop_slices_region(400, 400, (300, 500)) is True
    # crop 550..950: fully outside -> not slicing
    assert hud.crop_slices_region(550, 400, (300, 500)) is False
    # crop 200..600: fully contains region -> not slicing
    assert hud.crop_slices_region(200, 400, (300, 500)) is False
    # crop 0..200: fully outside on the left -> not slicing
    assert hud.crop_slices_region(0, 200, (300, 500)) is False


def test_safe_crop_x_prefers_fully_outside_position():
    # players 10..350, pot 935..1155; cw=400; subject at x=50 w=200 (mostly in the players area)
    ranges = [(10, 350), (935, 1155)]
    # subject sits at [50, 250]. Fully-outside candidates: x in [350, 535] (right of players
    # AND left of pot). x=350 keeps subject box overlap = 250-350=... zero, well below 70%.
    # x=350 subject_inside = max(0,min(750,250)-max(350,50))/200 = 0. Falls back to inside.
    x = hud.safe_crop_x(desired_x=50, cw=400, src_w=1280,
                        region_x_ranges=ranges, subject_box=(50, 200))
    # crop [x, x+400) must be either fully outside or fully inside every region.
    for rx0, rx1 in ranges:
        overlap = max(0, min(x + 400, rx1) - max(x, rx0))
        assert overlap == 0 or (x <= rx0 and rx1 <= x + 400)


def test_safe_crop_x_falls_back_to_fully_inside_when_subject_would_be_lost():
    # tight scene: subject is on top of players tile; the only fully-outside crop would
    # abandon the subject.
    ranges = [(10, 350), (935, 1155)]
    x = hud.safe_crop_x(desired_x=100, cw=340, src_w=1280,
                        region_x_ranges=ranges, subject_box=(30, 300))
    # crop [x, x+340) must fully contain the players region [10, 350]
    assert x <= 10 and 350 <= x + 340


def test_safe_crop_x_returns_valid_x_when_no_subject_given():
    """Without a subject box, safe_crop_x still returns something in-range and non-slicing."""
    ranges = [(200, 400), (700, 900)]
    x = hud.safe_crop_x(desired_x=500, cw=300, src_w=1280, region_x_ranges=ranges)
    assert 0 <= x <= 1280 - 300
    for r in ranges:
        assert not hud.crop_slices_region(x, 300, r)


def test_safe_crop_x_property_over_desired_x(seed=17):
    """Property test: over many desired-x values, the picked x never slices either region."""
    rng = np.random.default_rng(seed)
    ranges = [(10, 350), (935, 1155)]
    cw = 480
    for _ in range(50):
        d = int(rng.integers(0, 1280 - cw + 1))
        s = (int(rng.integers(0, 1280 - 200)), 200)
        x = hud.safe_crop_x(d, cw, 1280, ranges, subject_box=s)
        assert 0 <= x <= 1280 - cw
        for r in ranges:
            assert not hud.crop_slices_region(x, cw, r), (d, s, x, r)
