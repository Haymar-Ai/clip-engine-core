"""Per-broadcast hudband HUD profiles, and the speaker-crop fallback when a profile does not fit
the feed. A profile measured on one broadcast's graphics slices another's, so a source can name
its broadcast (recompose --broadcast) and pick the profile mapped to it in
render.hud_profile_by_broadcast before the vertical's; any profile that cuts through the window's
persistent graphics falls back to the speaker crop.

The shipped config maps no broadcast (only the vertical's generic poker profile), so the
recompose tests inject a synthetic profile config with invented broadcasts. Offline: synthetic
frames, mocked planners."""
from pathlib import Path

import numpy as np
import pytest

from clipengine import config
from clipengine.render import hud
from clipengine.render import recompose
from helpers import hud_text_ocr

W, H = 1280, 720
CFG = config.load()
HUD_BOX = (40, 120, 120, 160)            # x0, y0, x1, y1 of a fixed graphic on a 320x180 frame

# two invented broadcasts with their own HUD layouts, merged over the shipped config (which keeps
# its vertical_hud_profile: poker -> high_stakes_poker)
SYNTH_RENDER = {
    "hud_profile_by_broadcast": {
        "_doc": "invented broadcasts for the tests",
        "Felt Cam Live": "felt_cam",
        "The River Room (Card Club)": "river_room",
    },
    "hud_profiles": {
        # player column on the left, pot + board in a bottom bar that spans the frame
        "felt_cam": {"_note": "a bottom bar", "players": [0.05, 0.25, 0.30, 0.85],
                     "pot": [0.40, 0.82, 0.60, 0.98], "bottom_strip": [0.0, 0.82, 1.0, 1.0]},
        # player tiles and pot both in a bottom zone, a thin top banner
        "river_room": {"players": [0.0, 0.72, 0.20, 1.0], "pot": [0.22, 0.80, 0.80, 1.0],
                       "outs_strip": [0.0, 0.0, 1.0, 0.06], "bottom_strip": [0.0, 0.72, 1.0, 1.0]},
    },
}
SYNTH_CFG = config.merge(CFG, {"render": SYNTH_RENDER})
BROADCAST_OF = {"felt_cam": "Felt Cam Live", "river_room": "The River Room (Card Club)"}


def _shot_frames(n_shots=3, per_shot=4, seed=3):
    """2 fps frames: every shot a different random scene, the same bright HUD box with text-like
    bars on top of all of them."""
    rng = np.random.default_rng(seed)
    frames, cuts = [], []
    for k in range(n_shots):
        scene = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
        scene = np.repeat(np.repeat(scene[::8, ::8], 8, axis=0), 8, axis=1)[:180, :320]
        for _ in range(per_shot):
            f = scene.copy()
            x0, y0, x1, y1 = HUD_BOX
            f[y0:y1, x0:x1] = 30
            for y in range(y0 + 6, y1 - 4, 10):
                f[y:y + 4, x0 + 6:x1 - 6] = 235
            frames.append(f)
        if k:
            cuts.append(k * per_shot / hud.SAMPLE_HZ)
    return frames, cuts


def _norm(box, w=320, h=180):
    return [box[0] / w, box[1] / h, box[2] / w, box[3] / h]


def test_config_maps_each_broadcast_to_its_own_profile():
    for broadcast, name in (("Felt Cam Live", "felt_cam"),
                            ("The River Room (Card Club)", "river_room")):
        assert hud.profile_for_broadcast(SYNTH_CFG, broadcast) == name
        prof = hud.load_profile(SYNTH_CFG, name)
        assert {"players", "pot"} <= set(prof) and not any(k.startswith("_") for k in prof)
    # case and spacing do not matter; notes, unmapped and missing broadcasts map to nothing
    assert hud.profile_for_broadcast(SYNTH_CFG, "  felt cam   LIVE ") == "felt_cam"
    assert hud.profile_for_broadcast(SYNTH_CFG, "_doc") is None
    assert hud.profile_for_broadcast(SYNTH_CFG, "Somebody Else") is None
    assert hud.profile_for_broadcast(SYNTH_CFG, None) is None
    # the shipped config: no broadcast mapped, the vertical's default profile kept
    assert hud.profile_for_broadcast(CFG, "Felt Cam Live") is None
    assert hud.profile_for_vertical(CFG, "poker") == "high_stakes_poker"
    assert hud.profile_for_vertical(SYNTH_CFG, "poker") == "high_stakes_poker"


def test_overlay_mask_keeps_the_graphic_that_survives_every_shot():
    frames, cuts = _shot_frames()
    mask = hud.overlay_mask(frames, cuts)
    x0, y0, x1, y1 = HUD_BOX
    assert mask[y0:y1, x0:x1].mean() > 0.1                 # the HUD's edges
    outside = mask.copy()
    outside[y0 - 2:y1 + 2, x0 - 2:x1 + 2] = False
    assert outside.mean() < 0.01                           # scenes change shot to shot
    assert hud.overlay_mask(frames[:4], []) is None        # one shot: cannot tell


def test_profile_mismatch_when_a_band_region_cuts_through_the_graphic():
    frames, cuts = _shot_frames()
    mask = hud.overlay_mask(frames, cuts)
    x0, y0, x1, y1 = HUD_BOX
    fits = {"players": _norm((x0 - 6, y0 - 6, x1 + 6, y1 + 6))}
    cuts_it = {"players": _norm((x0 - 6, y0 - 6, (x0 + x1) // 2, y1 + 6))}
    assert hud.profile_mismatch(fits, mask) is None
    why = hud.profile_mismatch(cuts_it, mask)
    assert why and why.startswith("players region's right side runs")
    assert hud.profile_mismatch(cuts_it, None) is None


def _plan(monkeypatch, mismatch=None, broadcast=None, hud_text_in=None):
    """Plan a 3 s poker window through the real profile lookup on SYNTH_CFG. `mismatch`: what
    the profile-vs-feed check reports; `hud_text_in`: the profile whose band regions the fake
    OCR reads HUD text in (the profile expected to be picked)."""
    monkeypatch.setattr(recompose.config, "load", lambda: SYNTH_CFG)
    monkeypatch.setattr(recompose.reframe, "probe_size", lambda src: (W, H))
    seen = {}
    monkeypatch.setattr(recompose, "_hud_profile_mismatch",
                        lambda src, a, prof, sw, sh: seen.setdefault("prof", prof) and mismatch)
    monkeypatch.setattr(recompose, "_region_presence",
                        lambda *ar, **kw: {"players": [(0.0, 3.0)], "pot": [(0.0, 3.0)]})
    monkeypatch.setattr(recompose, "_ocr", hud_text_ocr(hud_text_in or {}, W, H))
    monkeypatch.setattr(recompose.reframe, "analyze", lambda src, start, dur, force=None, cw=None:
                        {"layout": force or "track", "src_w": W, "src_h": H, "shots": [],
                         "segments": [{"t0": 0.0, "t1": 3.0, "x": 400, "cut": True}]})
    argv = ["s.mp4", "0", "3", "o.mp4", "--layout", "auto", "--vertical", "poker"]
    if broadcast:
        argv += ["--broadcast", broadcast]
    return recompose.plan_layout(recompose.parse_args(argv), Path("s.mp4")), seen


def test_broadcast_profile_wins_and_its_bottom_strip_stays_out_of_the_top_crop(monkeypatch):
    felt = hud.load_profile(SYNTH_CFG, "felt_cam")
    plan, seen = _plan(monkeypatch, broadcast="Felt Cam Live", hud_text_in=felt)
    assert plan["layout"] == "hudband" and seen["prof"] == felt
    _, by, _, _ = hud.region_pixels(felt["bottom_strip"], W, H)
    assert plan["y_off"] == 0 and plan["eff_h"] == by        # rows above the bottom bar
    # an unmapped broadcast keeps the vertical's profile
    generic = hud.load_profile(SYNTH_CFG, "high_stakes_poker")
    plan, seen = _plan(monkeypatch, broadcast="Somebody Else", hud_text_in=generic)
    assert plan["layout"] == "hudband" and seen["prof"] == generic


@pytest.mark.parametrize("broadcast", [None, "Felt Cam Live"])
def test_a_profile_that_does_not_fit_falls_back_to_the_speaker_crop(monkeypatch, broadcast):
    plan, _ = _plan(monkeypatch, mismatch="pot region's right side runs 22% through ...",
                    broadcast=broadcast)
    assert plan["layout"] == "track"


@pytest.mark.parametrize("name,bottom_regions", [("river_room", ("players", "pot")),
                                                  ("felt_cam", ("pot",))])
def test_the_top_crop_ends_above_the_bottom_hud_zone(monkeypatch, name, bottom_regions):
    """The band shows these regions; the top crop must not show them again: the profile's
    bottom_strip starts at or above each region in the feed's bottom HUD zone, and the planned
    top crop ends where the strip starts."""
    prof = hud.load_profile(SYNTH_CFG, name)
    assert all(prof["bottom_strip"][1] <= prof[r][1] for r in bottom_regions)
    plan, seen = _plan(monkeypatch, broadcast=BROADCAST_OF[name], hud_text_in=prof)
    assert plan["layout"] == "hudband" and seen["prof"] == prof
    for r in bottom_regions:
        _, ry, _, _ = hud.region_pixels(prof[r], W, H)
        assert plan["y_off"] + plan["eff_h"] <= ry
