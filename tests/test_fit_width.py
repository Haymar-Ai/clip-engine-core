"""fit-to-width fallback. Where the layout rule allows it (charts, screens,
slides; every niche but the stacking gaming / streamers) and a 9:16 crop would slice graphics or
text, the clip renders the screen box (the largest slice-free <= 1.1:1 box) fit to width over a
blurred fill of the same frame instead of being rejected. Talking heads keep full-bleed; streamers
keep their split / stack. Offline: synthetic scans, no OCR, no frames (the contact sheets of real
renders are in clip-coverage-scratch/sheets/)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from clipengine.render import cleancrop
from clipengine.render import recompose

W, H = 1280, 720


@pytest.fixture(autouse=True)
def _cv2_single_thread():
    """content_blocks' OpenCV filters would start cv2's thread pool; later tests fork (niche_run's
    windows) and a forked multi-threaded process warns. One thread here, restored after."""
    import cv2
    n = cv2.getNumThreads()
    cv2.setNumThreads(0)
    yield
    cv2.setNumThreads(n)


def _face(cx, cy, h=0.2, w=0.12, t=1.0):
    return (t, [{"cx": cx, "cy": cy, "w": w, "h": h, "score": 0.9}])


def _plan(xs=(438,)):
    return {"layout": "track", "src_w": W, "src_h": H,
            "segments": [{"t0": i * 5.0, "t1": (i + 1) * 5.0, "x": x} for i, x in enumerate(xs)]}


def _scan(graphics=(), faces=None, w=W, h=H):
    return {"src_w": w, "src_h": h, "frames": 10,
            "graphics": [{"box": list(b), "frac": 0.9, "banner": False} for b in graphics],
            "panels": [], "dead_bands": {}, "faces": faces if faces is not None else []}


def A(niche="finance", kind="screen", vertical="general"):
    return types.SimpleNamespace(start=0.0, dur=10.0, niche=niche, content_kind=kind,
                                 vertical=vertical)


# a trading screen: a ticker strip across the top, a chart title, axis labels down the right edge
# and a wide order-book line across the bottom; a 9:16 track crop (405 px) slices them all
CHART = [(40, 20, 1240, 60),          # ticker strip, 1200 px: no <= 1.1:1 box holds it: outside
         (380, 110, 760, 150),        # chart title
         (1150, 160, 1240, 600),      # price axis labels
         (330, 620, 900, 660)]        # volume line


@pytest.fixture
def ladder(monkeypatch):
    from clipengine import quality_gate

    def run(plan, sc, a, rejected=None, guard=None, blocks=(), wins=None):
        recompose._SCANS.clear()
        monkeypatch.setattr(recompose, "_scan", lambda src, a, size: sc)
        monkeypatch.setattr(quality_gate, "ocr_source_text", lambda *x, **k: None)
        monkeypatch.setattr(cleancrop, "window_boxes", lambda *x, **k: wins)
        monkeypatch.setattr(recompose, "content_blocks", lambda *x, **k: blocks)
        return recompose.slice_ladder(plan, a, "src.mp4", guard, rejected)
    return run


def _slices(plan, sc):
    from clipengine import quality_gate
    v = quality_gate.crop_slice_verdict(quality_gate.crop_slices(
        sc, recompose.crop_views(plan, 10.0), 10.0, words=None))
    return v["crop_share"]


def test_a_chart_the_crop_would_slice_is_fit_to_width_over_a_blurred_fill(ladder):
    sc = _scan(graphics=CHART)
    plan = _plan()
    assert _slices(plan, sc) > 0.4                                 # the 9:16 crop slices text
    rep = ladder(plan, sc, A(), rejected="crop would slice a source banner: chart")
    assert rep["action"] == "ladder_d_screen_fit"
    assert plan["layout"] == "general" and plan["screen_fit"] is True
    box = plan["content_box"]
    assert box["w"] <= cleancrop.BOX_MAX_ASPECT * box["h"] + 2 and box["w"] > 405
    assert _slices(plan, sc) == 0.0                                # nothing sliced any more
    fill = recompose.blur_fill_frac(plan, None, {})
    assert 0.0 < fill <= 0.5 and fill == plan["blur_fill"]         # the gate's cap holds
    vg = recompose.video_graph(plan, A())
    assert vg.startswith(f"[0:v]crop={box['w']}:{box['h']}:{box['x']}:{box['y']},")
    assert "gblur" in vg and "overlay=(W-w)/2:(H-h)/2" in vg      # fit to width, blurred fill
    # the title, the axis labels and the volume line are whole in the box (the ticker is out)
    bx0, bx1 = box["x"], box["x"] + box["w"]
    for g in CHART[1:]:
        assert (g[0] >= bx0 and g[2] <= bx1) or g[2] <= bx0 or g[0] >= bx1


def test_a_content_block_is_never_cut_by_the_box_edge(ladder):
    """a dashboard panel / a plot / a tile: whole in the box or out of it (10-21 previews)."""
    panel = [300, 80, 760, 560]                                     # fits a box whole
    sc = _scan(graphics=CHART)
    plan = _plan()
    rep = ladder(plan, sc, A(), rejected="crop would slice: x", blocks=[panel])
    assert rep["action"] == "ladder_d_screen_fit"
    b, tol = plan["content_box"], cleancrop.SLICE_TOL               # cleancrop's tolerance
    inside = b["x"] <= panel[0] + tol and b["x"] + b["w"] >= panel[2] - tol \
        and b["y"] <= panel[1] + tol and b["y"] + b["h"] >= panel[3] - tol
    outside = b["x"] + b["w"] <= panel[0] + tol or b["x"] >= panel[2] - tol \
        or b["y"] + b["h"] <= panel[1] + tol or b["y"] >= panel[3] - tol
    assert inside or outside


def test_a_plot_across_the_frame_gets_no_screen_box(ladder):
    """a 16:9 dashboard / full-width chart: no <= 1.1:1 box leaves it whole (the whole frame
    is 68 % blurred fill): no screen box; the ladder goes on as before (with a face: the
    speaker steps; nothing fits: rejected), the refused box named in its steps."""
    plan = {**_plan(xs=(875,)), "screen_share": True}
    sc = _scan(graphics=CHART, faces=[_face(0.93, 0.85, h=0.12, w=0.07)])
    rep = ladder(plan, sc, A(), rejected="crop would slice: x", blocks=[[0, 70, 1280, 610]])
    assert rep["action"] != "ladder_d_screen_fit" and "content_box" not in plan
    assert any(t.startswith("d_screen_fit: no box") for t in rep["tried"])


def _bgr_frames(flat_top_rows=0, n=8, fw=640, fh=360):
    """Textured source frames at the ladder's scan width; the top rows flat when asked."""
    import numpy as np
    rng = np.random.default_rng(5)
    out = []
    for i in range(n):
        g = rng.integers(40, 220, size=(fh, fw)).astype(np.uint8)
        if flat_top_rows:
            g[:flat_top_rows, :] = 30
        out.append((float(i), np.dstack([g, g, g])))
    return out


def test_a_screen_box_whose_blurred_fill_would_show_a_flat_band_is_refused(ladder, monkeypatch):
    """a slide's flat margin above a block that reaches near the top: the fill (the box covered)
    shows it as a flat band the gate's edge_bands fails; no box clear of it holds the block."""
    monkeypatch.setattr(recompose.reframe, "iter_frames",
                        lambda *x, **k: _bgr_frames(flat_top_rows=40))     # 80 src px flat
    plan = {**_plan(xs=(875,)), "screen_share": True}
    sc = _scan(graphics=CHART[1:], faces=[_face(0.93, 0.85, h=0.12, w=0.07)])
    rep = ladder(plan, sc, A(), rejected="crop would slice: x", blocks=[[420, 60, 900, 700]])
    assert rep["action"] != "ladder_d_screen_fit" and "content_box" not in plan
    assert any(t.startswith("d_screen_fit") for t in rep["tried"])
    monkeypatch.setattr(recompose.reframe, "iter_frames", lambda *x, **k: _bgr_frames())
    plan = {**_plan(xs=(875,)), "screen_share": True}
    rep = ladder(plan, sc, A(), rejected="crop would slice: x", blocks=[[420, 60, 900, 700]])
    assert rep["action"] == "ladder_d_screen_fit"                    # a textured fill passes


def test_content_blocks_separates_panels_and_ignores_a_flat_backdrop():
    import numpy as np
    rng = np.random.default_rng(3)
    g = np.full((360, 640), 40, np.uint8)                          # flat backdrop
    g[40:200, 20:300] = rng.integers(0, 255, size=(160, 280))      # panel A (textured)
    g[60:340, 360:620] = rng.integers(0, 255, size=(280, 260))     # panel B, 60 px away
    got = recompose.content_blocks(None, None, 1280, 720, frames=[g, g])
    assert len(got) == 2
    a, b = sorted(got)
    assert a[0] <= 40 and a[2] >= 590 and a[2] <= 640              # panel A in src px (x2)
    assert b[0] >= 700


def test_a_screen_with_a_small_cam_keeps_the_cam_face_whole_or_out(ladder):
    """the banner guard found a small-cam screen share too wide to fit whole (screen_box_plan):
    the screen box goes before any 9:16 recrop (a tiny cam blown up)."""
    sc = _scan(graphics=CHART[1:], faces=[_face(0.93, 0.85, h=0.12, w=0.07)])
    plan = {**_plan(xs=(875,)), "screen_share": True}
    rep = ladder(plan, sc, A(kind="screen"), rejected="crop would slice a source banner: x")
    assert rep["action"] == "ladder_d_screen_fit"                 # before any 9:16 recrop
    box = plan["content_box"]
    fx0, fx1 = (0.93 - 0.035) * W, (0.93 + 0.035) * W
    assert (box["x"] <= fx0 and box["x"] + box["w"] >= fx1) or box["x"] + box["w"] <= fx0 \
        or box["x"] >= fx1
    assert _slices(plan, sc) == 0.0


def test_small_faces_of_a_screen_source_get_the_speaker_steps_first_and_the_screen_box_last(
        ladder):
    """A screen-kind source with only small faces (a chart with a cam, a face found in a chart):
    the screen box first, the face whole in it or out. Benzinga-like speakers boxed in tiles:
    the tile steps first (the screen box would come last, never through a tile)."""
    plan = _plan(xs=(438,))
    sc = _scan(graphics=[(40, 250, 1240, 280), (380, 400, 760, 440)],
               faces=[_face(0.5, 0.25, h=0.12, w=0.07)])
    rep = ladder(plan, sc, A(kind="screen"), rejected="crop would slice a source banner: x")
    assert rep["action"] == "ladder_d_screen_fit"
    assert plan["content_box"]["y"] >= 280                          # below the line, face out
    assert _slices(plan, sc) == 0.0
    plan = _plan(xs=(300,))
    sc = _scan(faces=[_face(0.30, 0.45, h=0.12, w=0.07)])
    tile = [10, 40, 630, 700]
    rep = ladder(plan, sc, A(kind="screen"), rejected="crop would slice a source banner: x",
                 blocks=[tile], wins=[tile])
    assert rep["action"].startswith("ladder_b_tile")               # the speaker's tile first


def test_a_talking_head_with_a_clean_crop_stays_full_bleed(ladder):
    sc = _scan(graphics=[(40, 20, 300, 60)], faces=[_face(0.5, 0.4, h=0.3)])
    plan = _plan(xs=(438,))
    guard = {"action": "none"}
    rep = ladder(plan, sc, A(niche="podcasts", kind="talk"), guard=guard)
    assert rep is guard and plan["layout"] == "track" and "crop_box" not in plan
    assert "content_box" not in plan


def test_a_talking_head_the_ladder_cannot_save_is_not_turned_into_a_screen(ladder):
    sc = _scan(graphics=[(0, 200, 1280, 230)], faces=[_face(0.5, 0.45, h=0.3)])
    with pytest.raises(recompose.RenderError, match="crop would slice"):
        ladder(_plan(), sc, A(niche="podcasts", kind="talk"), rejected="crop would slice: x")


def test_streamers_keep_their_layouts(ladder):
    """a stacking niche never gets the screen box: the ladder works as before (here its 9:16
    recrop of the faceless frame)."""
    sc = _scan(graphics=CHART)
    plan = _plan()
    rep = ladder(plan, sc, A(niche="streamers", kind="screen", vertical="stream"),
                 rejected="crop would slice a source banner: x")
    assert rep["action"] != "ladder_d_screen_fit" and plan["layout"] == "track"
    assert "content_box" not in plan
    assert recompose.fit_width_allowed(A(niche="gaming")) is False
    assert recompose.fit_width_allowed(A(niche="tech")) is True
    assert recompose.fit_width_allowed(A(niche=None, vertical="stream")) is False
    assert recompose.fit_width_allowed(A(niche=None, vertical="general")) is False  # campaigns


def test_a_faceless_wide_frame_the_gate_would_fail_gets_the_screen_box(monkeypatch):
    from clipengine import quality_gate
    sc = _scan(graphics=CHART[1:])
    recompose._SCANS.clear()
    monkeypatch.setattr(recompose, "_scan", lambda src, a, size: sc)
    monkeypatch.setattr(quality_gate, "ocr_source_text", lambda *x, **k: None)
    monkeypatch.setattr(recompose, "content_blocks", lambda *x, **k: [])
    plan = {"layout": "general"}
    rep = recompose.screen_fit_general(plan, A(niche="tech"), "src.mp4", {"width": W, "height": H})
    assert rep["action"] == "screen_fit" and plan["content_box"]["w"] < W
    assert recompose.blur_fill_frac(plan, None, {"width": W, "height": H}) <= 0.5
    # a square frame fits whole within the cap: left alone; a streamer: left alone
    sq = {"layout": "general"}
    assert recompose.screen_fit_general(sq, A(niche="tech"), "s", {"width": 720, "height": 720}) is None
    st = {"layout": "general"}
    assert recompose.screen_fit_general(st, A(niche="streamers", vertical="stream"), "s",
                                        {"width": W, "height": H}) is None
    assert sq == {"layout": "general"} and st == {"layout": "general"}


def test_no_slice_free_box_leaves_the_frame_as_it_was(monkeypatch):
    from clipengine import quality_gate
    lines = [(0, y, 1280, y + 30) for y in (100, 300, 500)]       # full-width slide lines: every
    sc = _scan(graphics=lines)                                     # <= 1.1:1 box cuts one
    recompose._SCANS.clear()
    monkeypatch.setattr(recompose, "_scan", lambda src, a, size: sc)
    monkeypatch.setattr(quality_gate, "ocr_source_text", lambda *x, **k: None)
    monkeypatch.setattr(recompose, "content_blocks", lambda *x, **k: [])
    plan = {"layout": "general"}
    assert recompose.screen_fit_general(plan, A(), "s", {"width": W, "height": H}) is None
    assert plan == {"layout": "general"}


def test_stages_passes_the_source_kind_to_the_render(tmp_path, monkeypatch):
    from clipengine import engine, runstate
    runstate.add_source("CHARTvid001", title="Live trading the open", url="u", niche="finance")
    clip = {"source_id": "CHARTvid001_w00", "start_s": 1.0, "end_s": 20.0, "layout": "auto"}
    cmd = engine.recompose_cmd(clip, {"source_id": "CHARTvid001_w00", "vertical": "general"},
                               "m.mp4", "o.mp4")
    assert cmd[cmd.index("--content-kind") + 1] == "screen"
    assert cmd[cmd.index("--niche") + 1] == "finance"
