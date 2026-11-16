"""Crop feasibility. A window whose speaker crop cannot be made without slicing a graphic is
dropped before select costs its transcribe and a render, and a candidate span with no clean crop
leaves before its render; a window that has a clean crop gets it (the slice-free dual view zoom,
the HUD plates kept whole, the clean recrop first on --clean-first); a normal window is unchanged.
Offline: synthetic scans, plates, faces and OCR words, and fakes for the probe and the transcript."""
from __future__ import annotations

import copy

import pytest

from clipengine.render import cleancrop
from clipengine.render import cropcheck
from clipengine import quality_gate as qg
from clipengine.render import recompose
from clipengine import runstate

W, H = 1280, 720


def _face(cx, cy=0.45, h=0.4, w=0.18):
    return {"cx": cx, "cy": cy, "w": w, "h": h, "score": 0.9}


def _scan(graphics=(), faces=None, panels=()):
    return {"src_w": W, "src_h": H, "frames": 10,
            "graphics": [{"box": list(b), "frac": f, "banner": False} for b, f in graphics],
            "panels": [list(p) for p in panels], "dead_bands": {},
            "faces": faces if faces is not None else [(float(t), [_face(0.5)]) for t in range(10)]}


# a talking head between two HUD stacks of a poker stream: the action log and the hero's cards
# on the left, the room's logo and the board on the right
HUD_WORDS = [(78, 188, 150, 210, "FLOP", 0), (78, 188, 150, 210, "FLOP", 1),
             (78, 188, 150, 210, "FLOP", 2), (94, 223, 226, 242, "Hero checks", 0),
             (94, 223, 226, 242, "Hero checks", 1), (94, 223, 226, 242, "Hero checks", 2),
             (78, 534, 244, 572, "Eff-$1575", 0), (78, 534, 244, 572, "Eff-$1575", 1),
             (78, 534, 244, 572, "Eff-$1575", 2), (284, 540, 318, 568, "BB", 0),
             (284, 540, 318, 568, "BB", 1), (284, 540, 318, 568, "BB", 2),
             (1081, 113, 1233, 143, "TABLE", 0), (1081, 113, 1233, 143, "TABLE", 1),
             (1081, 113, 1233, 143, "TABLE", 2), (1102, 494, 1166, 522, "$235", 0),
             (1102, 494, 1166, 522, "$235", 1), (1102, 494, 1166, 522, "$235", 2)]


# ---------------------------------------------------------------- cropcheck geometry

def test_a_window_cuts_a_plate_it_shows_in_part_only():
    plate = [100, 100, 200, 140]
    assert cropcheck.cuts([150, 0, 600, 720], plate)            # its left half is out
    assert not cropcheck.cuts([50, 0, 600, 720], plate)         # whole inside
    assert not cropcheck.cuts([300, 0, 700, 720], plate)        # whole outside
    assert not cropcheck.cuts([199, 0, 600, 720], plate)        # 1 px in: within the tolerance


def test_find_window_keeps_the_face_and_steps_off_a_plate():
    face = [520, 100, 760, 520]
    plates = [[300, 500, 480, 580]]                             # left of the face, at chin height
    pos = cropcheck.find_window(404, 720, W, H, plates, contain=face, prefer=(400, 0))
    assert pos is not None
    x, y = pos
    assert x <= 520 and x + 404 >= 760 and not cropcheck.cuts([x, y, x + 404, y + 720], plates[0])
    wide = [[200, 300, 1100, 360]]                              # a ticker across the face's columns
    assert cropcheck.find_window(404, 720, W, H, wide, contain=face) is None


def test_full_bleed_between_two_hud_stacks_is_feasible_a_board_over_the_speaker_is_not():
    plates = [p["box"] for p in cropcheck.plates([], HUD_WORDS, W, H)]
    face = cropcheck.face_box(_face(0.5), W, H)
    assert cropcheck.full_bleed_ok(face, plates, W, H)
    board = plates + [[430, 380, 900, 460]]                     # a board graphic over his chest
    assert not cropcheck.full_bleed_ok(face, board, W, H)


def test_the_probe_gives_no_verdict_without_a_face_or_with_small_faces_only():
    pls = [{"box": [0, 0, 1280, 720], "frac": 1.0}]             # everything is a plate
    v = cropcheck.frame_verdicts([(0.0, []), (1.0, [_face(0.5, h=0.1)]), (2.0, [_face(0.5)])],
                                 pls, W, H)
    assert [ok for _t, ok, _h in v] == [None, None, False]
    assert cropcheck.share_ok(v) == (0.0, 1)


def test_a_plate_counts_only_on_the_frames_that_show_it():
    """A post card shown for part of the window, then the speaker alone."""
    tweet = [{"box": [398, 200, 1200, 600], "frac": 1.0}]
    face = [(0.0, [_face(0.5)]), (1.0, [_face(0.5)])]
    lines = [[[420, 300, 1100, 320]], []]                       # text lines on frame 0 only
    v = cropcheck.frame_verdicts(face, tweet, W, H, lines=lines)
    assert [ok for _t, ok, _h in v] == [False, True]


def test_the_probe_reads_persistent_overlays_not_one_frame_ocr_reads():
    sc = _scan(panels=[[0, 0, 300, 700]])                       # a chat column
    items = [{"box": [600, 600, 700, 640], "frac": 0.33, "src": "ocr"},     # a shirt read once
             {"box": [900, 500, 1200, 560], "frac": 1.0, "src": "ocr"}]     # a HUD panel
    boxes = [p["box"] for p in cropcheck.probe_plates(sc, items, [], W, H)]
    assert any(b[0] < 900 < b[2] for b in boxes) and not any(b[0] < 650 < b[2] for b in boxes)
    assert [p["box"] for p in cropcheck.panel_plates(sc, W, H)] == [[0, 0, 306, 706]]


# ---------------------------------------------------------------- the dual view zoom

def _dv(segs, zw=568, zh=432, bottom="zoom"):
    hh, ph, seam = recompose.dual_view_geometry(W, H)
    return {"layout": "dual_view", "src_w": W, "src_h": H, "header_h": hh, "panel_h": ph,
            "seam_y": seam, "bottom": bottom, "zoom": {"w": zw, "h": zh}, "zoom_segments": segs}


def test_a_zoom_window_that_cuts_a_donation_alert_moves_off_it_and_keeps_the_face():
    """A face zoom that cut a donation alert line over the streamer's cam."""
    face = [0.12, 0.8, 0.08, 0.14]                              # the cam face, bottom left
    plan = _dv([{"t0": 0.0, "t1": 10.0, "x": 0, "y": 240, "how": "face", "faces": [face]}])
    alert = [{"box": [0, 470, 320, 500], "frac": 1.0}]          # wholly inside the planned window
    assert recompose.clean_zoom(copy.deepcopy(plan), alert, W, H, 820) == 0
    chat = [{"box": [0, 300, 640, 330], "frac": 1.0}]           # a line the window's top edge cuts
    p = copy.deepcopy(plan)
    p["zoom_segments"][0]["y"] = 310
    moved = recompose.clean_zoom(p, chat, W, H, 820)
    s, z = p["zoom_segments"][0], p["zoom"]
    assert moved == 1 and not cropcheck.cuts([s["x"], s["y"], s["x"] + z["w"], s["y"] + z["h"]],
                                             chat[0]["box"])
    fb = recompose._zoom_face_box(face, W, H)
    assert s["x"] <= fb[0] and s["x"] + z["w"] >= fb[2] and s["y"] <= fb[1] \
        and s["y"] + z["h"] >= fb[3]


def test_a_moved_zoom_window_keeps_faces_whole_too():
    """A move that keeps the chat whole must not cut the streamer's face at the panel's edge."""
    plan = _dv([{"t0": 0.0, "t1": 10.0, "x": 0, "y": 200, "how": "motion"}])
    chat = [{"box": [0, 180, 500, 220], "frac": 1.0}]           # the planned window's top cuts it
    cam_face = [40, 420, 200, 640]                              # a face low in the frame
    p = copy.deepcopy(plan)
    assert recompose.clean_zoom(p, chat, W, H, 820, faces=lambda t0, t1: [cam_face]) == 1
    s, z = p["zoom_segments"][0], p["zoom"]
    win = [s["x"], s["y"], s["x"] + z["w"], s["y"] + z["h"]]
    assert not cropcheck.cuts(win, chat[0]["box"]) and not cropcheck.cuts(win, cam_face)


def test_no_clean_zoom_of_any_size_means_no_dual_view():
    face = [0.5, 0.5, 0.1, 0.2]
    plan = _dv([{"t0": 0.0, "t1": 10.0, "x": 356, "y": 144, "how": "face", "faces": [face]}])
    band = [{"box": [0, 330, 1280, 370], "frac": 1.0}]          # a ticker through the face
    assert recompose.clean_zoom(plan, band, W, H, 820) is None


def test_a_clean_zoom_is_left_exactly_as_planned():
    plan = _dv([{"t0": 0.0, "t1": 5.0, "x": 356, "y": 144, "how": "motion"},
                {"t0": 5.0, "t1": 10.0, "x": 600, "y": 200, "how": "motion"}])
    before = copy.deepcopy(plan)
    assert recompose.clean_zoom(plan, [{"box": [0, 0, 100, 40], "frac": 1.0}], W, H, 820) == 0
    assert plan == before


# ---------------------------------------------------------------- the speaker crop

@pytest.fixture
def ladder(monkeypatch):
    """slice_ladder with the scan, OCR, window and band reads replaced."""
    def run(plan, sc, a, words=None, guard=None):
        recompose._SCANS.clear()
        monkeypatch.setattr(recompose, "_scan", lambda src, a_, size: sc)
        monkeypatch.setattr(qg, "ocr_source_text", lambda *ar, **k: words)
        monkeypatch.setattr(cleancrop, "window_boxes", lambda *ar, **k: None)
        monkeypatch.setattr(recompose.reframe, "iter_frames", lambda *ar, **k: iter(()))
        monkeypatch.setattr(recompose, "plan_dual_view", lambda *ar, **k: pytest.fail("dual view"))
        return recompose.slice_ladder(plan, a, "src.mp4", guard or {"action": "none"})
    return run


def _args(niche, *extra, vertical=None):
    argv = ["src.mp4", "0", "10", "o.mp4", "--niche", niche, *extra]
    if vertical:
        argv += ["--vertical", vertical]
    return recompose.parse_args(argv)


def _track(x):
    return {"layout": "track", "src_w": W, "src_h": H,
            "segments": [{"t0": 0.0, "t1": 10.0, "x": x}]}


def test_a_boxed_poker_window_gets_the_clean_full_bleed_crop_between_its_hud_stacks(ladder):
    """The planned crop cuts the hero's BB / Eff plate; the clean recrop keeps every plate whole
    or out and the speaker whole: full-bleed, never a box or a stack in poker."""
    plan = _track(250)
    rep = ladder(plan, _scan(), _args("poker", vertical="poker"), words=HUD_WORDS)
    assert rep["action"] == "ladder_a_recrop" and plan["layout"] == "track"
    pls = recompose.keep_plates(_scan(), HUD_WORDS, W, H)
    views = recompose.crop_views(plan, 10.0)
    assert views and cropcheck.cut_share(views, pls, 10.0) == 0
    assert "slices" in rep["why"]


def test_clean_first_gets_the_clean_recrop_even_when_the_plan_measures_clean(ladder):
    plan = _track(438)
    rep = ladder(plan, _scan(), _args("podcasts", "--clean-first"), words=[])
    assert rep["action"] == "ladder_a_recrop" and "known edge_slice source" in rep["why"]


def test_a_normal_window_keeps_its_planned_crop(ladder):
    plan, guard = _track(438), {"action": "none"}
    before = copy.deepcopy(plan)
    assert ladder(plan, _scan(), _args("podcasts"), words=[], guard=guard) is guard
    assert plan == before


def test_a_shirt_print_moves_with_the_speaker_a_hud_panel_does_not():
    faces = [(0.0, [_face(0.5)])]
    shirt = {"box": [600, 640, 720, 700], "frac": 0.33}
    panel = {"box": [600, 640, 720, 700], "frac": 1.0}
    side = {"box": [20, 500, 200, 580], "frac": 0.33}           # the HUD stack beside him
    assert recompose.off_body([shirt, panel, side], faces, W, H) == [panel, side]


def test_a_long_segment_no_one_window_holds_is_solved_in_pieces():
    """The speaker leans inside one long segment; a print under his chin rules out a
    full-height window, and the shorter window that clears it cannot hold both of his positions
    for the whole shot, only for each 3 s of it."""
    faces = [(float(t), [_face(0.40 if t < 6 else 0.525, cy=0.45, h=0.5)]) for t in range(12)]
    sc = _scan(faces=faces)
    plan = {"layout": "track", "src_w": W, "src_h": H,
            "segments": [{"t0": 0.0, "t1": 12.0, "x": 390}]}
    no_slice = [[0, 600, 1280, 640]]                            # a strip below the chins
    crop = cleancrop.clean_crop(plan, sc, [], no_slice)
    assert len(crop["segments"]) == 2 and crop["segments"][1]["t0"] == 6.0
    cleancrop.apply(plan, crop)
    assert [s["x"] for s in plan["segments"]] == crop["x"] and crop["x"][0] < crop["x"][1]
    assert crop["y"] + crop["h"] <= 600
    assert cleancrop.split_segments([{"t0": 0.0, "t1": 4.0, "x": 1}]) == [
        {"t0": 0.0, "t1": 4.0, "x": 1}]


def test_a_poker_band_with_no_hud_text_in_its_tiles_falls_back(monkeypatch):
    """A generic HUD profile on a talking-head podcast would show a mic and a hand."""
    a = _args("poker", vertical="poker")
    profile = {"players": [0.03, 0.6, 0.225, 0.935], "pot": [0.745, 0.765, 0.968, 0.93]}
    recompose._SCANS.clear()
    monkeypatch.setattr(qg, "ocr_source_text", lambda *ar, **k: [(1100, 50, 1200, 80, "CardRoom", 0)])
    assert recompose._blank_regions("src.mp4", a, profile, {"players", "pot"}, W, H) == \
        ["players", "pot"]
    recompose._SCANS.clear()
    monkeypatch.setattr(qg, "ocr_source_text",
                        lambda *ar, **k: [(60, 480, 200, 520, "PLAYER $502", 0),
                                          (980, 580, 1180, 640, "POT 10,000", 0)])
    assert recompose._blank_regions("src.mp4", a, profile, {"players", "pot"}, W, H) == []


def test_a_band_on_a_feed_with_no_word_on_screen_falls_back_too(monkeypatch):
    """A HUD profile on a webcam podcast where the OCR reads nothing at all (an empty read must
    not keep the band). The scan's persistent text lines decide then: none in a region, the
    region is blank; a HUD panel's lines keep it."""
    a = _args("poker", vertical="poker")
    profile = {"players": [0.03, 0.6, 0.225, 0.935], "pot": [0.745, 0.765, 0.968, 0.93]}
    monkeypatch.setattr(qg, "ocr_source_text", lambda *ar, **k: [])
    for graphics, blank in (([], ["players", "pot"]),
                            ([((60, 480, 200, 520), 1.0), ((980, 580, 1180, 640), 0.9)], []),
                            ([((60, 480, 200, 520), 0.2)], ["players", "pot"])):
        recompose._SCANS.clear()
        monkeypatch.setattr(recompose, "_scan", lambda src, a_, size, g=graphics: _scan(graphics=g))
        assert recompose._blank_regions("src.mp4", a, profile, {"players", "pot"}, W, H) == blank


def test_the_gate_check_runs_before_the_encode_on_niche_clips(monkeypatch):
    sc = _scan(graphics=[((300, 600, 700, 640), 1.0)])
    words = [(310, 605, 690, 635, "LOWER THIRD NAME", 0)]
    recompose._SCANS.clear()
    monkeypatch.setattr(recompose, "_scan", lambda src, a, size: sc)
    monkeypatch.setattr(qg, "ocr_source_text", lambda *ar, **k: words)
    with pytest.raises(recompose.RenderError, match="quality gate would fail"):
        recompose.gate_slice_check(_track(438), _args("podcasts"), "src.mp4")
    assert recompose.gate_slice_check(_track(800), _args("podcasts"), "src.mp4")["passed"]
    manual = recompose.parse_args(["src.mp4", "0", "10", "o.mp4"])     # no niche: as before
    assert recompose.gate_slice_check(_track(438), manual, "src.mp4") is None


def test_a_stack_the_gate_would_fail_on_frame_edge_text_is_a_defect(monkeypatch):
    """A gamestack's game crop that shows a chat overlay running into the frame edge (over the
    gate's limit) is a defect though its crop share alone is 0."""
    gs = {"layout": "gamestack", "src_w": W, "src_h": H, "top_h": 1344,
          "game": [W, 0, True], "cam": [400, 300, 860, 20, True]}
    chat = [((0, 100 + 60 * i, 330, 130 + 60 * i), 1.0) for i in range(3)]
    sc = _scan(graphics=chat)
    words = [(0, 105 + 60 * i, 320, 125 + 60 * i, f"viewer{i}: lets goooo", 0) for i in range(3)]
    recompose._SCANS.clear()
    monkeypatch.setattr(recompose, "_scan", lambda src, a, size: sc)
    monkeypatch.setattr(qg, "ocr_source_text", lambda *ar, **k: words)
    a = _args("streamers", vertical="stream")
    gs["game"] = [578, 0, False]                                # the game crop shows x 0..578:
    why = recompose.shown_defect(gs, a, "src.mp4")              # the chat whole, the frame edge
    assert why and "frame / crop edges" in why


def test_the_open_recrop_runs_only_when_the_window_shows_no_video_box(monkeypatch):
    """Shelves can read as a box around a full-frame speaker (no video box: the open recrop
    runs); one real guest box, seen once, or an unreadable window keeps it from running."""
    from clipengine.render import broadcast
    a = _args("tech")
    for v, tiled in (({"panel": False, "tiles_max": 0}, False),
                     ({"panel": False, "tiles_max": 1}, True),
                     ({"panel": None, "reason": "unreadable"}, True)):
        recompose._SCANS.clear()
        monkeypatch.setattr(broadcast, "detect_broadcast_panel", lambda *ar, v=v, **k: v)
        assert recompose.tiled_broadcast("src.mp4", a) is tiled


def test_a_streamers_split_cutting_only_a_wall_logo_keeps_its_layout(monkeypatch):
    """The split's crops cut a two-letter wall logo; in the stacking niches plates decide
    nothing, the gate's text items do."""
    sp = {"layout": "split", "src_w": W, "src_h": H,
          "split": {"left": {"x": 0, "y": 0, "w": 600, "h": 720},
                    "right": {"x": 680, "y": 0, "w": 600, "h": 720}}}
    words = [(560, 100, 720, 130, "Cy", f) for f in range(3)]      # 2 letters: a plate, no item
    recompose._SCANS.clear()
    monkeypatch.setattr(recompose, "_scan", lambda src, a, size: _scan())
    monkeypatch.setattr(qg, "ocr_source_text", lambda *ar, **k: words)
    assert recompose.shown_defect(sp, _args("streamers", vertical="stream"), "src.mp4") is None
    assert "HUD plates" in recompose.shown_defect(sp, _args("poker", vertical="poker"), "src.mp4")


def test_only_the_stacking_niches_may_render_the_dual_view_fallback():
    from clipengine import engine
    assert engine.SPLIT_NICHES == ()
    assert not recompose.dual_view_allowed(_args("poker")) and recompose.hud_niche(_args("poker"))
    assert recompose.dual_view_allowed(_args("gaming"))


# ---------------------------------------------------------------- select

@pytest.fixture
def db(tmp_path, monkeypatch):
    from clipengine import engine
    monkeypatch.setattr(engine, "SCRATCH", tmp_path / "scratch")
    return engine


def _window(tmp_path, sid="src1_w00", niche="poker"):
    """A harvested window of a niche source (the niche on its parent's row, as a run names it)."""
    media = tmp_path / f"{sid}.mp4"
    media.write_bytes(b"\0")
    runstate.add_source(sid.rsplit("_w", 1)[0], niche=niche)
    runstate.add_source(sid, local_path=str(media), vertical="poker")
    runstate.update_source(sid, duration_s=60.0, state="harvested")
    return runstate.source(sid)


def test_an_uncroppable_window_is_skipped_before_select_transcribes_it(db, tmp_path, monkeypatch):
    engine = db
    src = _window(tmp_path)
    probed = []
    monkeypatch.setattr(engine, "crop_probe", lambda media, sid, niche: probed.append(niche) or {
        "verdict": "uncroppable", "why": "no clean crop for the speaker on 40 of 44 sampled "
        "frames (9 graphic plate(s); full-bleed tried)", "elapsed_s": 4.2, "verdicts": []})
    monkeypatch.setattr(engine.transcript, "get_words",
                        lambda *a, **k: pytest.fail("an uncroppable window is never transcribed"))
    assert engine.select(src) is False
    assert probed == ["poker"]                          # the parent's niche
    row = runstate.source(src["source_id"])
    assert row["state"] == "failed" and row["error"].startswith("uncroppable: no clean crop")
    assert row["error"].startswith(engine.UNCROPPABLE)
    msgs = [e["message"] for e in runstate.events() if e["stage"] == "select"]
    assert any("UNCROPPABLE, window skipped before select" in m for m in msgs)
    assert any(": timings crop=" in m for m in msgs)


def test_a_normal_window_goes_through_select_as_before(db, tmp_path, monkeypatch):
    engine = db
    src = _window(tmp_path)
    monkeypatch.setattr(engine, "crop_probe", lambda media, sid, niche: {
        "verdict": "ok", "why": "clean crop on 100% of 44 face frames", "elapsed_s": 4.0,
        "verdicts": [[float(t), True, "full_bleed"] for t in range(60)]})
    seen = []

    def words(*a, **k):
        seen.append("transcribe")
        raise RuntimeError("stop here")             # select's own low-speech path takes over
    monkeypatch.setattr(engine.transcript, "get_words", words)
    monkeypatch.setattr(engine, "energy_envelope", lambda media: (_ for _ in ()).throw(
        RuntimeError("no audio")))
    engine.select(src)
    row = runstate.source(src["source_id"])
    assert seen == ["transcribe"] and not str(row.get("error") or "").startswith("uncroppable")


def test_a_candidate_span_with_no_clean_crop_leaves_before_its_render(db):
    engine = db
    probe = {"verdicts": [[float(t), t >= 10, "x"] for t in range(40)]}   # sliced for 10 s
    cands = [{"start": 0.0, "end": 12.0}, {"start": 15.0, "end": 35.0}]
    kept = engine._drop_uncroppable_picks(cands, probe, "w")
    assert kept == [cands[1]] and cands[0]["crop_probe"]["ok_share"] < 0.75
    assert engine._drop_uncroppable_picks(cands, {"verdicts": []}, "w") == cands
