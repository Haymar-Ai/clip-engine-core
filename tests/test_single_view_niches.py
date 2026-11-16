""": full-bleed single view following the speaker; split screen / stacks only for
gaming and streamers; tech and finance never stacked, whatever the planner, the clip's layout,
the vertical mapping or a manual --top/--bottom asks for.

Offline: the planners are fakes that would return a stacked plan if they were reached; the one
encode test renders tiny testsrc2 media and reads the plan back with --emit-plan.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from clipengine.render import banners
from clipengine.render import recompose
from clipengine.render import reframe

SW, SH = 1280, 720
STACKED = recompose.STACKED_LAYOUTS
REQUESTS = ("auto", "track", "split", "broadcast", "hudband", "gamestack", "reaction",
            "screencast", "chart_top_cam_bottom", "general")


@pytest.fixture
def stacking_planners(monkeypatch):
    """Every planner returns a stacked plan when it is reached; reframe.analyze honours `force`
    and, left to decide, sees two steady faces (split)."""
    calls = []

    def analyze(src, start, dur, force=None, **kw):
        calls.append(force)
        lay = force or "split"
        p = {"layout": lay, "src_w": SW, "src_h": SH, "segments": [], "shots": []}
        if lay == "split":
            p["split"] = {"top": {"x": 0, "y": 0, "w": 400, "h": 720},
                          "bottom": {"x": 800, "y": 0, "w": 400, "h": 720}}
        return p

    monkeypatch.setattr(reframe, "analyze", analyze)
    for fn, lay in (("_plan_gamestack", "gamestack"), ("_plan_reaction", "reaction"),
                    ("_plan_screencast_auto", "screencast_auto"), ("_plan_hudband", "hudband"),
                    ("_plan_chart_top_cam_bottom", "chart_top_cam_bottom")):
        monkeypatch.setattr(recompose, fn, lambda src, a, _l=lay: {"layout": _l, "src_w": SW,
                                                                    "src_h": SH})
    return calls


def _args(layout, niche=None, vertical=None, extra=()):
    argv = ["s.mp4", "0", "10", "o.mp4", "--layout", layout, *extra]
    if niche:
        argv += ["--niche", niche]
    if vertical:
        argv += ["--vertical", vertical]
    return recompose.parse_args(argv)


@pytest.mark.parametrize("niche", ["tech", "finance"])
@pytest.mark.parametrize("layout", REQUESTS)
@pytest.mark.parametrize("vertical", [None, "general", "stream", "poker", "reaction",
                                      "screencast"])
def test_tech_and_finance_never_plan_a_stack(stacking_planners, niche, layout, vertical):
    a = _args(layout, niche, vertical)
    plan = recompose.plan_layout(a, "s.mp4")
    assert plan["layout"] not in STACKED, (layout, vertical, plan)
    assert not plan.get("manual")
    recompose.enforce_single_view(plan, a)          # the last check before encode agrees


@pytest.mark.parametrize("niche", ["tech", "finance"])
@pytest.mark.parametrize("layout", ["screencast", "split"])
def test_manual_top_bottom_crops_are_ignored_for_tech_and_finance(stacking_planners, niche,
                                                                   layout):
    a = _args(layout, niche, extra=["--top", "1280:360:0:0", "--bottom", "1280:360:0:360"])
    plan = recompose.plan_layout(a, "s.mp4")
    assert not plan.get("manual") and plan["layout"] not in STACKED


@pytest.mark.parametrize("niche", ["gaming", "streamers"])
def test_gaming_and_streamers_keep_their_stacks(stacking_planners, niche):
    assert recompose.plan_layout(_args("gamestack", niche), "s.mp4")["layout"] == "gamestack"
    assert recompose.plan_layout(_args("auto", niche), "s.mp4")["layout"] == "split"
    a = _args("split", niche, extra=["--top", "1280:360:0:0", "--bottom", "1280:360:0:360"])
    assert recompose.plan_layout(a, "s.mp4")["manual"] is True


@pytest.mark.parametrize("niche", ["podcasts", "lifestyle", "tech", "finance"])
def test_two_faces_follow_the_speaker_outside_gaming_and_streamers(stacking_planners, niche):
    plan = recompose.plan_layout(_args("auto", niche), "s.mp4")
    assert plan["layout"] == "track"
    assert stacking_planners == [None, "track"]       # decided split, re-planned as track


def test_no_niche_keeps_the_old_auto_decision(stacking_planners):
    """Campaign / manual renders pass no --niche: nothing changes for them."""
    assert recompose.plan_layout(_args("auto"), "s.mp4")["layout"] == "split"


@pytest.mark.parametrize("layout", STACKED)
@pytest.mark.parametrize("niche", ["tech", "finance"])
def test_enforce_single_view_rejects_any_stacked_plan(layout, niche):
    with pytest.raises(recompose.RenderError, match="single view"):
        recompose.enforce_single_view({"layout": layout}, _args("auto", niche))
    with pytest.raises(recompose.RenderError):
        recompose.enforce_single_view({"layout": "track", "manual": True}, _args("auto", niche))
    recompose.enforce_single_view({"layout": layout}, _args("auto", "gaming"))   # allowed


def _cam(box, face):
    return {"src_w": SW, "src_h": SH, "top_h": 960, "cam_box": box, "guard_box": box,
            "box_norm": [v / d for v, d in zip(box, (SW, SH, SW, SH))], "face_frac": 1.0,
            "from_profile": False, "face_median": face, "channel": None, "profile_key": None}


def _screen_share(src_w=SW, src_h=SH):
    rows = [{"box": [0, y, src_w - 180, y + 30], "frac": 1.0, "banner": True}
            for y in range(60, src_h - 60, 80)]
    return {"src_w": src_w, "src_h": src_h, "frames": 20, "graphics": rows, "panels": [],
            "faces": [(t, [{"cx": 0.96, "cy": 0.82, "w": 0.08, "h": 0.13, "score": 0.9}])
                      for t in range(10)], "hgrad": None, "scale": 2.0}


@pytest.mark.parametrize("niche", ["tech", "finance"])
def test_screen_share_never_becomes_a_screen_stack_for_tech_and_finance(monkeypatch, niche):
    """A 16:9 screen share whose crop slices its UI: gaming/streamers get the screen stack, tech
    and finance get the whole-screen box only when the gate could pass it - a 16:9 box is 68 %
    blurred fill, so the clip is rejected instead of stacked."""
    monkeypatch.setattr(banners, "scan", lambda *a, **k: _screen_share())
    monkeypatch.setattr(recompose, "_resolve_cam",
                        lambda *a, **k: _cam([1110, 500, 1280, 720], [0.977, 0.826, 0.133]))
    plan = {"layout": "track", "src_w": SW, "src_h": SH,
            "segments": [{"t0": 0.0, "t1": 10.0, "x": 876, "cut": True}]}
    with pytest.raises(recompose.RenderError, match="blurred fill"):
        recompose.guard_banners(plan, _args("auto", niche), "s.mp4")
    assert plan["layout"] == "track"


def test_screen_box_is_a_single_view_when_the_gate_could_pass_it(monkeypatch):
    """A square screen (1080x1080) fit to width is 44 % blurred fill: boxed, never stacked."""
    monkeypatch.setattr(banners, "scan", lambda *a, **k: _screen_share(1080, 1080))
    monkeypatch.setattr(recompose, "_resolve_cam",
                        lambda *a, **k: dict(_cam([930, 900, 1080, 1080], [0.97, 0.9, 0.08]),
                                             src_w=1080, src_h=1080))
    plan = {"layout": "track", "src_w": 1080, "src_h": 1080,
            "segments": [{"t0": 0.0, "t1": 10.0, "x": 470, "cut": True}]}
    a = _args("auto", "finance")
    rep = recompose.guard_banners(plan, a, "s.mp4")
    assert rep["action"] == "reframe" and plan["layout"] == "general" and plan["screen_box"]
    recompose.enforce_single_view(plan, a)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    p = tmp_path_factory.mktemp("sv") / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=640x360:rate=30:duration=3", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=3", "-shortest", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(p)], check=True)
    return p


@pytest.mark.parametrize("niche", ["tech", "finance"])
def test_a_manual_split_render_comes_out_single_view(media, tmp_path, niche):
    """End to end: the same command line the pipeline builds for a layout_crops split clip."""
    out, plan_path = tmp_path / "clip.mp4", tmp_path / "plan.json"
    rc = recompose.main([str(media), "0", "2", str(out), "--layout", "split",
                         "--top", "320:360:0:0", "--bottom", "320:360:320:0",
                         "--niche", niche, "--emit-plan", str(plan_path)])
    assert rc == 0 and out.exists()
    plan = json.loads(plan_path.read_text())
    assert plan["layout"] not in STACKED and not plan.get("manual")
