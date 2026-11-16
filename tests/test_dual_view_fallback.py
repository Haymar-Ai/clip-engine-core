"""The dual view fallback of a single-view niche: tech and finance clips render a single view;
when it slices the screen (the gate's crop_slices / edge_slice, or a render the slice ladder
refused: no clean crop, the fit-to-width box over the blur cap) the clip renders once more as the
dual view (the whole frame fit to width on top, the speaker below). A clean single-speaker clip
keeps its single view: the first render never changes. The shipped config turns the fallback off:
a render with no clean crop then fails before any encode and flags its window as uncroppable.
The render-failure strings model the slice ladder's own messages."""
from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from clipengine import runstate
from helpers import codes_for, fake_store_clip
from clipengine import config


LADDER = ("RENDER FAILED: the speaker clear of 1 foreign region(s) without slicing 2 graphic(s); "
          "speaker window 292 px tall; b_tile_box: no box (<= 1.1:1, h >= 50% of 720 src px); the "
          "whole frame fit to width would be 68% blurred fill (gate cap 50%)")
SLICED = {"verdict": "fail", "drop_reason": "crop_slices: the crop slices burned-in text",
          "crop_slices": {"passed": False, "errors": ["the crop slices burned-in text: 4 item(s)"]}}
PASS = {"verdict": "pass", "drop_reason": ""}


# ---------------------------------------------------------------- recompose

def _args(*extra, niche="tech"):
    from clipengine.render import recompose
    return recompose.parse_args(["s.mp4", "1", "20", "o.mp4", "--niche", niche, *extra])


def test_recompose_renders_the_dual_view_for_tech_and_finance_only_when_asked():
    from clipengine.render import recompose
    assert recompose.dual_view_fallback(_args("--dual-view-fallback")) is True
    assert recompose.dual_view_fallback(_args("--dual-view-fallback", niche="finance")) is True
    assert recompose.dual_view_fallback(_args()) is False                     # first render
    assert recompose.dual_view_fallback(_args("--dual-view-fallback", niche="podcasts")) is False
    dv = {"layout": recompose.DUAL_VIEW}
    recompose.enforce_single_view(dv, _args("--dual-view-fallback"))           # allowed
    with pytest.raises(recompose.RenderError, match="single view"):
        recompose.enforce_single_view(dv, _args())                            # never by itself


def test_recompose_converts_the_planned_single_view(monkeypatch, tmp_path):
    """--dual-view-fallback: the single view is planned as before, then to_dual_view replaces it
    (its speaker windows feed the zoom below); no room for the zoom panel is a render failure."""
    from clipengine.render import recompose
    seen = {}
    monkeypatch.setattr(recompose, "probe", lambda src: {"width": 1280, "height": 720,
                                                         "fps": 30, "has_audio": True})
    monkeypatch.setattr(recompose, "plan_layout", lambda a, src: {"layout": "track"})
    monkeypatch.setattr(recompose, "guard_banners", lambda *a: pytest.fail("no single view"))

    def to_dv(plan, a, src, why, tried=None, guard=None):
        seen["why"], seen["from"] = why, plan.get("layout")
        return None
    monkeypatch.setattr(recompose, "to_dual_view", to_dv)
    src = tmp_path / "s.mp4"
    src.write_bytes(b"\0")
    rc = recompose.main([str(src), "1", "20", str(tmp_path / "o.mp4"), "--niche", "tech",
                         "--dual-view-fallback"])
    assert rc == 1 and seen["from"] == "track" and "tech dual view fallback" in seen["why"]


# ---------------------------------------------------------------- engine

@pytest.fixture
def db(tmp_path, monkeypatch):
    from clipengine import engine
    # the fallback is off by default (render.crop_feasibility.dual_view_fallback_niches []); these
    # tests pin the mechanism for a config that turns it back on
    monkeypatch.setattr(engine, "SPLIT_NICHES", ("tech", "finance"))
    monkeypatch.setattr(engine, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(engine, "CAPTION_STRONG_MODEL", None)
    monkeypatch.setattr(engine, "WINDOW_DEADLINE", None)
    monkeypatch.setattr(engine, "_store_clip", fake_store_clip)
    return tmp_path


def _clip(tmp_path, niche="tech", sid="Tch_w00"):
    """A selected clip of window `sid`; the niche is its parent source's (as a run names it)."""
    media = tmp_path / f"{sid}.mp4"
    media.write_bytes(b"\0")
    runstate.add_source(sid.rsplit("_w", 1)[0], niche=niche)
    runstate.add_source(sid, url="https://example.com/watch/tch", local_path=str(media),
                        vertical="general")
    runstate.update_source(sid, state="selected", duration_s=60.0)
    wp = tmp_path / "side.json"
    wp.write_text(json.dumps([{"w": "Hello.", "start": 10.5, "end": 11.0}]))
    return runstate.add_clip(sid, 10.0, 30.0, 0.7, "auto", hook="h", words_path=str(wp))


def _recompose(calls, fail_first=None):
    """A fake render: records the argv, fails the first call with `fail_first`, else writes the
    output file (argv[6]) and prints recompose's QC line."""
    def run(cmd, **kw):
        calls.append(cmd)
        if fail_first and len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=fail_first)
        pathlib.Path(cmd[6]).write_bytes(b"mp4")
        lay = "dual_view" if "--dual-view-fallback" in cmd else "track"
        return subprocess.CompletedProcess(cmd, 0, stdout=f'QC {{"layout": "{lay}"}}\n', stderr="")
    return run


def test_split_due_only_for_tech_and_finance_slices(db, tmp_path):
    from clipengine import engine
    tech = runstate.clip(_clip(tmp_path))
    pod = runstate.clip(_clip(tmp_path, niche="podcasts", sid="Pod_w00"))
    assert engine.split_due(tech, report=SLICED) is None
    assert engine.split_due(tech, error=LADDER) is None
    assert "podcasts" in engine.split_due(pod, report=SLICED)
    assert engine.split_due(tech, report=PASS) == "the single view sliced nothing"
    no_face = {**SLICED, "frame_gate": {"passed": False, "none_count": 30, "edge_cut_count": 0}}
    assert "no_face" in engine.split_due(tech, report=no_face)
    assert engine.split_due({**tech, "split_fallback": True}, report=SLICED)
    assert engine.split_due(tech, error="RENDER FAILED: hook placement: no room") is not None


def test_a_render_the_slice_ladder_refused_renders_the_dual_view_once(db, tmp_path, monkeypatch):
    from clipengine import engine
    cid = _clip(tmp_path)
    calls, gates = [], []
    monkeypatch.setattr(subprocess, "run", _recompose(calls, fail_first=LADDER))
    monkeypatch.setattr(engine, "_run_quality_gate", lambda out, clip, qc=None:
                        gates.append(qc.get("layout")) or PASS)
    assert engine.render(runstate.clip(cid)) is True
    assert len(calls) == 2 and "--dual-view-fallback" not in calls[0]
    assert "--dual-view-fallback" in calls[1] and gates == ["dual_view"]
    feats = json.loads(runstate.clip(cid)["features_json"])
    assert "68% blurred fill" in feats["split_fallback"]["first_render"]


def test_a_single_view_the_gate_failed_on_slices_gets_the_dual_view(db, tmp_path, monkeypatch):
    from clipengine import engine
    cid = _clip(tmp_path)
    calls, layouts = [], []
    monkeypatch.setattr(subprocess, "run", _recompose(calls))

    def gate(out, clip, qc=None):
        layouts.append((qc.get("layout"), clip.get("layout")))
        return SLICED if len(layouts) == 1 else PASS
    monkeypatch.setattr(engine, "_run_quality_gate", gate)
    assert engine.render(runstate.clip(cid)) is True
    assert len(calls) == 2 and "--dual-view-fallback" in calls[1]
    # the gate still runs as the full gate: the clip's layout stays the single view's request
    assert layouts == [("track", "auto"), ("dual_view", "auto")]
    rep = json.loads(runstate.clip(cid)["features_json"])["gate_report"]
    assert rep["split_fallback"]["layout"] == "dual_view"


def test_a_dual_view_that_still_fails_is_rejected_after_one_try(db, tmp_path, monkeypatch):
    from clipengine import engine
    cid = _clip(tmp_path)
    calls = []
    monkeypatch.setattr(subprocess, "run", _recompose(calls))
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: SLICED)
    assert engine.render(runstate.clip(cid)) is False
    assert len(calls) == 2 and runstate.clip(cid)["state"] == "failed"
    assert "edge_slice" in codes_for(cid)


def test_other_niches_keep_their_single_render(db, tmp_path, monkeypatch):
    from clipengine import engine
    cid = _clip(tmp_path, niche="podcasts", sid="Pod_w00")
    calls = []
    monkeypatch.setattr(subprocess, "run", _recompose(calls))
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: SLICED)
    assert engine.render(runstate.clip(cid)) is False and len(calls) == 1
    calls.clear()
    monkeypatch.setattr(subprocess, "run", _recompose(calls, fail_first=LADDER))
    cid2 = runstate.add_clip("Pod_w00", 31.0, 50.0, 0.7, "auto", hook="h",
                          words_path=str(tmp_path / "side.json"))
    assert engine.render(runstate.clip(cid2)) is False and len(calls) == 1


def test_a_clean_single_view_is_never_rerendered(db, tmp_path, monkeypatch):
    from clipengine import engine
    cid = _clip(tmp_path)
    calls = []
    monkeypatch.setattr(subprocess, "run", _recompose(calls))
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: PASS)
    assert engine.render(runstate.clip(cid)) is True
    assert len(calls) == 1 and "--dual-view-fallback" not in calls[0]


# ---------------------------------------------------------------- source bars

def _frame(w=640, h=360, right=0, top=0, logo=False):
    import numpy as np
    rng = np.random.default_rng(7)
    g = rng.integers(40, 220, size=(h, w)).astype(np.uint8)        # busy picture
    if right:
        g[:, w - right:] = 2                                         # a black pillar
        if logo:
            g[100:140, w - right + 4:w - 4] = 230                    # a logo over it
    if top:
        g[:top] = 3
    return g


def test_a_black_pillar_baked_into_the_source_is_read_in_most_frames():
    """A 60 px black pillar on the right of a 1280 px frame (30 px of the 640 px sample), a logo
    over it now and then: the dual view's top panel would show it whole and the gate fail it on a
    flat right band."""
    from clipengine.render import recompose
    frames = [_frame(right=30, logo=(i % 3 == 0)) for i in range(24)]
    assert recompose.bar_depths(frames)["right"] == 30
    assert recompose.bar_depths([_frame(top=36) for _ in range(24)])["top"] == 36
    assert recompose.bar_depths([_frame() for _ in range(24)]) == {
        "top": 0, "bottom": 0, "left": 0, "right": 0}


def test_a_dark_picture_edge_or_a_thin_line_is_no_bar():
    import numpy as np
    from clipengine.render import recompose
    lit = []
    for _ in range(24):
        g = _frame(right=30)
        g[:, 610:] = 90                                              # flat but lit: a wall
        lit.append(g)
    assert recompose.bar_depths(lit)["right"] == 0
    assert recompose.bar_depths([_frame(right=4) for _ in range(24)])["right"] == 0   # < 1 %
    wide = [_frame(right=200) for _ in range(24)]                    # > 20 %: picture
    assert recompose.bar_depths(wide)["right"] == 0


def test_the_dual_view_fallback_shows_the_frame_inside_its_bars(monkeypatch):
    from clipengine.render import recompose
    a = _args("--dual-view-fallback")
    monkeypatch.setattr(recompose, "source_bars", lambda src, a, w, h: {
        "x": 0, "y": 0, "w": 1218, "h": 720, "bars": {"top": 0, "bottom": 0, "left": 0,
                                                       "right": 30}})
    monkeypatch.setattr(recompose, "_cam_candidate", lambda *x, **k: None)
    monkeypatch.setattr(recompose, "_dual_view_samples", lambda *x: ([], [], [], None))
    monkeypatch.setattr(recompose, "dual_view_zoom_plan", lambda *x, **k: {
        "zoom": {"w": 600, "h": 400}, "zoom_segments": [{"t0": 0.0, "t1": 19.0, "x": 300,
                                                         "y": 100}]})
    plan = recompose.plan_dual_view("s.mp4", a, prior={"src_w": 1280, "src_h": 720})
    assert plan["frame"]["w"] == 1218
    assert plan["panel_h"] == recompose.dual_view_geometry(1218, 720)[1]       # fit to width
    vf = recompose.dual_view_vf(plan)
    assert vf.count("crop=1218:720:0:0,") == 2                      # header and top panel
    other = recompose.plan_dual_view("s.mp4", _args(niche="gaming"),
                                     prior={"src_w": 1280, "src_h": 720})
    assert "frame" not in other and "crop=1218" not in recompose.dual_view_vf(other)


def test_the_fallback_is_off_by_default():
    """The shipped config lists no dual view fallback niche: tech / finance never stack."""
    cfg = config.load()
    assert cfg["render"]["crop_feasibility"]["dual_view_fallback_niches"] == []
    from clipengine import engine
    assert engine.SPLIT_NICHES == ()
    assert engine.split_due({"source_id": "x"}, error=LADDER) is not None


def test_no_clean_crop_drops_the_clip_before_any_encode_and_flags_its_window(
        db, tmp_path, monkeypatch):
    """The shipped config: no dual view re-render; the window is flagged uncroppable
    (engine.UNCROPPABLE) and the clip's reason code is an edge_slice from the render."""
    from clipengine import engine
    monkeypatch.setattr(engine, "SPLIT_NICHES", ())
    cid = _clip(tmp_path)
    calls = []
    real = ("RENDER FAILED: crop would slice a source banner or graphic: slices burned-in text "
            "(on-screen share 0.18); ladder: a_recrop: no clean 9:16 crop ...")
    monkeypatch.setattr(subprocess, "run", _recompose(calls, fail_first=real))
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: pytest.fail("no gate"))
    assert engine.render(runstate.clip(cid)) is False
    assert len(calls) == 1
    src = runstate.source("Tch_w00")
    assert src["error"].startswith(f"{engine.UNCROPPABLE}: render: recompose exit 1")
    assert runstate.clip(cid)["state"] == "failed"
    assert "edge_slice" in codes_for(cid)


def test_a_crop_around_the_facecam_flags_its_window_as_uncroppable(db, tmp_path, monkeypatch):
    """The planner refuses a gaming window whose crop shows the facecam with the game around it;
    the clip costs no encode or gate pass, and its window is flagged like any no-clean-crop
    plan."""
    from clipengine import engine
    monkeypatch.setattr(engine, "SPLIT_NICHES", ())
    cid = _clip(tmp_path, niche="gaming", sid="Gam_w00")
    calls = []
    real = ("RENDER FAILED: crop would slice the game around the facecam's box [19, 3, 241, 213]: "
            "the track window [0, 0, 404, 720] shows the cam with the game around it")
    monkeypatch.setattr(subprocess, "run", _recompose(calls, fail_first=real))
    monkeypatch.setattr(engine, "_run_quality_gate", lambda *a, **k: pytest.fail("no gate"))
    assert engine.render(runstate.clip(cid)) is False
    assert len(calls) == 1
    assert runstate.source("Gam_w00")["error"].startswith("uncroppable: render: recompose exit 1")
    assert runstate.clip(cid)["state"] == "failed"

