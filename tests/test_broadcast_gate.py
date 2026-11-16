"""The broadcast gate: boxed-broadcast and wrong-speaker clips are caught by the quality gate.

A boxed broadcast is a row of video boxes (a news show's guests) over a backdrop. Its clips can
keep a neighbouring box the crop sliced (render/broadcast.canvas_boxes -> quality_gate
boxed_slice_verdict, edge_slice; under a blurred pad the slivers count as fill), open on cover
frames of the boxed layout before the show cuts to its speaker (cover_frame_verdict,
cover_frame_layout), or keep a silent listener's face (render/speaker.kept_face_verdict ->
measure_kept_face_speaker, wrong_speaker). The source detector (broadcast.source_boxes /
detect_broadcast_panel) tells such a source from a studio with a screen on the wall.

Offline and synthetic: canvases and source frames are drawn with numpy / cv2 (boxes, faces as
ellipses, shelves, a window grid, a caption row of letter strokes), small mp4s are encoded from
them with ffmpeg into tmp_path, and the kept face is judged on generated mouth-motion and speech
energy series.
"""
from __future__ import annotations

import functools
import subprocess

import cv2
import numpy as np

from clipengine import quality_gate as qg
from clipengine import reason_codes
from clipengine.render import broadcast
from clipengine.render import speaker

CW, CH = 540, 960            # a rendered 9:16 canvas at the gate's scan size
SW, SH = 1280, 720           # a source window


# ---------------------------------------------------------------- synthetic frames

def _texture(h, w, seed, base=120.0, spread=45.0, grain=6.0):
    """A picture with structure (smooth blobs) and grain, but no long straight edges."""
    rng = np.random.default_rng(seed)
    low = rng.normal(0, 1, (max(2, h // 40), max(2, w // 40))).astype(np.float32)
    low = cv2.GaussianBlur(cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), 12)
    low /= np.abs(low).max() + 1e-6
    return base + spread * low + rng.normal(0, grain, (h, w)).astype(np.float32)


def _face(g, cx, cy, rx, ry, luma=185.0):
    """A face: an ellipse with two eyes and a mouth."""
    def ell(x, y, ax, ay, v):
        cv2.ellipse(g, (int(x), int(y)), (max(2, int(ax)), max(2, int(ay))), 0, 0, 360,
                    float(v), -1)
    ell(cx, cy, rx, ry, luma)
    ell(cx - rx * 0.35, cy - ry * 0.15, rx * 0.12, ry * 0.06, 60)
    ell(cx + rx * 0.35, cy - ry * 0.15, rx * 0.12, ry * 0.06, 60)
    ell(cx, cy + ry * 0.45, rx * 0.3, ry * 0.07, 90)


def _u8(g):
    return np.clip(g, 0, 255).astype(np.uint8)


def _speaker(seed=1):
    """A full-bleed talking head on a textured wall."""
    g = _texture(CH, CW, seed, base=105.0, spread=72.0, grain=9.6)
    _face(g, CW / 2, CH * 0.42, 95, 125, luma=200)
    g[int(CH * 0.62):, int(CW * 0.18):int(CW * 0.82)] *= 0.45                 # shoulders
    return _u8(g)


def _boxed_row(seed=2, y0=280, y1=680, tiles=((0, 128), (138, 402), (412, 540))):
    """A row of video boxes fit to width over a blurred pad: the guest's box in the middle, its
    neighbours sliced by the canvas edges."""
    g = cv2.GaussianBlur(_texture(CH, CW, seed + 10, base=90.0, grain=0.0), (0, 0), 25)
    for i, (a, b) in enumerate(tiles):
        g[y0:y1, a:b] = _texture(y1 - y0, b - a, seed + i, base=150.0 + 10 * i)
        _face(g, (a + b) / 2, y0 + (y1 - y0) * 0.45, min(60, (b - a) * 0.3), 70, luma=200)
    return _u8(g)


def _boxed_crop():
    """A crop of the guest's box with the next box of the row sliced at the right edge."""
    return _boxed_row(y0=200, y1=760, tiles=((0, 470), (480, 540)))


def _framed(seed=3, box=(60, 220, 480, 740)):
    """One remote guest's outlined box on a dark backdrop."""
    g = cv2.GaussianBlur(_texture(CH, CW, seed, base=40.0, grain=0.0), (0, 0), 20)
    x0, y0, x1, y1 = box
    g[y0:y1, x0:x1] = _texture(y1 - y0, x1 - x0, seed + 1, base=140.0)
    cv2.rectangle(g, (x0, y0), (x1 - 1, y1 - 1), 235.0, 4)
    _face(g, (x0 + x1) / 2, y0 + (y1 - y0) * 0.4, 80, 105)
    return _u8(g)


def _shelf(seed=4):
    """A speaker beside a short bookcase: its side's boards run off one side of it only."""
    g = _speaker(seed).astype(np.float32)
    x0, x1, y0, y1 = 40, 170, 330, 620
    g[y0:y1, x0:x1] = 70
    for y in range(y0, y1, 72):
        g[y:y + 8, x0:x1] = 150                                              # a board
        for k, xb in enumerate(range(x0 + 10, x1 - 10, 22)):
            g[y + 14:y + 66, xb:xb + 14] = 110 + 25 * (k % 3)                 # book spines
    return _u8(g)


def _window(seed=5):
    """A window grid bigger than the canvas behind the speaker: its mullions run off the top
    and the bottom, its muntins cross them."""
    g = 190.0 + np.random.default_rng(seed).normal(0, 6, (CH, CW))
    for x in (90, 270, 450):
        g[:, x - 5:x + 5] = 60
    for y in (160, 480, 800):
        g[y - 5:y + 5, :] = 60
    _face(g, CW / 2, CH * 0.45, 95, 125, luma=150)
    g[int(CH * 0.64):, int(CW * 0.18):int(CW * 0.82)] = 50
    return _u8(g)


def _door(seed=6, letters=True):
    """A door's casing that ends on the burned hook row. `letters`: the row is letter strokes
    (text, no box edge); else one solid bar (which would read as a box's bottom edge)."""
    g = _speaker(seed).astype(np.float32)
    g[200:760, 400:] = 28 + np.random.default_rng(seed).normal(0, 3, (560, CW - 400))
    g[200:760, 392:400] = 235
    if letters:
        for k, x in enumerate(range(40, 500, 11)):
            if k % 6 != 5:                                                   # word gaps
                g[760:800, x:x + 6] = 245
    else:
        g[760:800, 40:500] = 245
    return _u8(g)


FRAMES = {"speaker": _speaker, "boxed_row": _boxed_row, "boxed_crop": _boxed_crop,
          "framed": _framed, "shelf": _shelf, "window": _window, "hook_row": _door,
          "hook_bar": lambda: _door(letters=False)}


@functools.lru_cache(maxsize=None)
def frame(name):
    return FRAMES[name]()


@functools.lru_cache(maxsize=None)
def boxes(name):
    return broadcast.canvas_boxes(frame(name))


def scan_of(names_times):
    return [(t, boxes(n)) for n, t in names_times]


def _source_panel(seed=7, n=5):
    """A source frame of a news show: a row of `n` framed guest boxes over an animated backdrop
    graphic, a face in each box. (gray, faces)"""
    rng = np.random.default_rng(seed)
    xx, yy = np.arange(SW)[None, :], np.arange(SH)[:, None]
    g = 70 + 30 * np.sin(xx / 37.0 + yy / 53.0) + rng.normal(0, 4, (SH, SW))
    m, gap, y0, y1 = 40, 16, 200, 560
    w = (SW - 2 * m - (n - 1) * gap) // n
    faces = []
    for i in range(n):
        a = m + i * (w + gap)
        g[y0:y1, a:a + w] = np.clip(_texture(y1 - y0, w, seed + i, base=150.0), 120, 255)
        cv2.rectangle(g, (a, y0), (a + w - 1, y1 - 1), 240.0, 3)
        _face(g, a + w / 2, y0 + 150, 45, 60, luma=205)
        faces.append({"cx": round((a + w / 2) / SW, 3), "cy": round((y0 + 150) / SH, 3),
                      "h": round(120 / SH, 3)})
    return _u8(g), faces


def _source_studio(seed=8):
    """A podcast studio: two hosts at a desk, a screen on the wall between them. (gray, faces)"""
    g = _texture(SH, SW, seed, base=110.0)
    g[120:330, 520:760] = _texture(210, 240, seed + 1, base=170.0)          # the wall screen
    faces = []
    for cx in (300, 980):
        _face(g, cx, 300, 70, 92)
        faces.append({"cx": cx / SW, "cy": 300 / SH, "h": 184 / SH})
    g[520:, :] = 60                                                          # the desk
    return _u8(g), faces


def video(tmp_path, parts, name="clip.mp4", fps=25):
    """An mp4 of synthetic canvases: parts = [(frame name, seconds)]."""
    out = tmp_path / name
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for i, (f, s) in enumerate(parts):
        png = tmp_path / f"{out.stem}_{i}_{f}.png"
        cv2.imwrite(str(png), frame(f))
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", f"{s:.3f}", "-i", str(png)]
    n = len(parts)
    graph = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0,format=yuv420p[v]"
    cmd += ["-filter_complex", graph, "-map", "[v]", "-r", str(fps), "-c:v", "libx264",
            "-preset", "ultrafast", "-crf", "12", str(out)]
    subprocess.run(cmd, check=True)
    return out


# ---------------------------------------------------------------- 1. cover frame

def test_a_boxed_cover_fails_the_cover_frame_check_even_on_one_frame():
    cover = scan_of([("boxed_row", i * 0.04) for i in range(6)] +
                    [("speaker", 0.24 + i * 0.04) for i in range(7)])
    body = scan_of([("speaker", 0.5 * i) for i in range(40)])
    m = qg.cover_frame_verdict(cover, body)
    assert m["passed"] is False and m["steady"] == "clean" and m["bad_frames"] == 6
    assert m["cover_labels"] == "pppppp" + "c" * 7
    assert reason_codes.codes_from_gate_report({"cover_frame": m}) == {
        "cover_frame_layout": "cover_frame: " + m["errors"][0]}
    one = qg.cover_frame_verdict(cover[:1] + cover[6:], body)          # not share-based
    assert one["passed"] is False and one["bad_frames"] == 1


def test_cover_frame_that_is_the_steady_layout_passes():
    clean = scan_of([("speaker", 0.04 * i) for i in range(13)])
    assert qg.cover_frame_verdict(clean, scan_of([("speaker", 0.5 * i) for i in range(10)])
                                  )["passed"] is True
    boxed = scan_of([("boxed_row", 0.04 * i) for i in range(13)])   # the boxed check's job
    assert qg.cover_frame_verdict(boxed, scan_of([("boxed_row", 0.5 * i) for i in range(10)])
                                  )["passed"] is True
    assert qg.cover_frame_verdict([], [])["passed"] is None


def test_an_outlined_box_on_a_backdrop_is_a_framed_cover():
    assert boxes("framed")["label"] == "framed"
    m = qg.cover_frame_verdict(scan_of([("framed", 0.0)] + [("speaker", 0.04 * i)
                                                           for i in range(1, 13)]),
                               scan_of([("speaker", 0.5 * i) for i in range(10)]))
    assert m["passed"] is False and m["cover_labels"].startswith("f")


def test_cover_and_boxed_measures_on_a_rendered_mp4(tmp_path):
    """0.24 s of the boxed cover, then the speaker for the rest of the clip."""
    v = video(tmp_path, [("boxed_row", 0.24), ("speaker", 11.76)])
    boxed, cover = qg.measure_broadcast_boxes(v)
    assert cover["passed"] is False and cover["cover_labels"].startswith("pppppp")
    assert boxed["passed"] is True and boxed["sliced_frames"] <= 1    # the 2 fps scan misses it
    clean = video(tmp_path, [("speaker", 3.0)], name="clean.mp4")
    boxed, cover = qg.measure_broadcast_boxes(clean)
    assert boxed["passed"] is True and cover["passed"] is True and cover["steady"] == "clean"


# ---------------------------------------------------------------- 2. boxed broadcast

def test_canvas_boxes_finds_the_sliced_neighbours_of_a_boxed_row():
    row = boxes("boxed_row")
    assert row["label"] == "panel" and row["sliced"] == ["left", "right"]
    assert 0.15 <= row["sliver_frac"] <= 0.25
    crop = boxes("boxed_crop")
    assert crop["label"] == "panel" and crop["sliced"] == ["right"]


def test_shelves_window_grids_and_hook_rows_are_no_boxes():
    for name in ("speaker", "shelf", "window", "hook_row"):
        v = boxes(name)
        assert v["label"] == "clean" and not v["borders"] and not v["sliced"], name
    # the hook row's letter strokes are what keep the door's end from reading as a box's corner
    assert boxes("hook_bar")["label"] == "panel"


def test_boxed_slice_verdict_fails_a_sliced_row_as_edge_slice_and_passes_a_clean_clip():
    m = qg.boxed_slice_verdict(scan_of([("boxed_row", 0.5 * i) for i in range(10)]))
    assert m["passed"] is False and m["sides"] == ["left", "right"] and m["sliced_frac"] == 1.0
    assert "sliced at the left and right canvas edges" in m["errors"][0]
    assert list(reason_codes.codes_from_gate_report({"boxed_slices": m})) == ["edge_slice"]
    ok = qg.boxed_slice_verdict(scan_of([("speaker", 0.5 * i) for i in range(20)]))
    assert ok["passed"] is True and ok["sliver_frac"] == 0.0
    assert qg.boxed_slice_verdict([])["passed"] is None


def test_blur_padded_boxes_count_as_fill_not_content():
    stats = [{"luma": 148.0, "contrast": 65.0, "sharpness": 4100.0}] * 5
    assert qg.visual_quality_verdict(stats, 0.488)["passed"] is True     # the fill alone
    m = qg.visual_quality_verdict(stats, 0.488, sliver_frac=0.20)
    assert m["passed"] is False and "sliced neighbour boxes cover 69%" in m["errors"][0]
    assert list(reason_codes.codes_from_gate_report({"visual_quality": m})) == ["edge_slice"]
    assert qg.visual_quality_verdict(stats, 0.0, sliver_frac=0.20)["passed"] is True  # full bleed
    assert qg.visual_quality_verdict(stats, 0.30, sliver_frac=0.0)["passed"] is True


def test_gate_fails_a_boxed_row_render_and_passes_a_clean_render(tmp_path, monkeypatch):
    monkeypatch.setattr(qg, "measure_frame_gate", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(qg, "measure_av_sync", lambda p: {"passed": None})
    monkeypatch.setattr(speaker, "kept_face_speech_check",
                        lambda *a, **k: {"not_speaker_frac": 0.0})
    v = video(tmp_path, [("boxed_row", 3.0)])
    rep = qg.gate(str(v), run_overlay_check=False, run_clip_checks=False, blur_fill_frac=0.488)
    assert rep["verdict"] == "fail"
    codes = reason_codes.codes_from_gate_report(rep)
    assert "edge_slice" in codes and rep["boxed_slices"]["passed"] is False
    assert rep["visual_quality"]["passed"] is False                  # 0.488 + ~0.20 fill
    assert "sliced neighbour boxes" in rep["visual_quality"]["errors"][0]
    clean = video(tmp_path, [("speaker", 3.0)], name="clean.mp4")
    rep = qg.gate(str(clean), run_overlay_check=False, run_clip_checks=False, blur_fill_frac=0.0)
    assert rep["verdict"] == "pass", rep["drop_reason"]
    off = qg.gate(str(v), run_overlay_check=False, run_clip_checks=False, run_broadcast=False)
    assert all(off[k]["passed"] is None for k in qg.BROADCAST_KEYS)


def test_source_detector_labels_the_boxed_broadcast_not_a_podcast_studio():
    panel, guests = _source_panel()
    v = broadcast.source_boxes(panel, guests)
    assert v["panel"] is True and v["tiles"] >= 3
    studio, hosts = _source_studio()
    assert broadcast.source_boxes(studio, hosts)["panel"] is False
    # a face outside every box is a studio with a screen on the wall, not a panel
    out = broadcast.source_boxes(panel, guests + [{"cx": 0.5, "cy": 0.9, "h": 0.2}])
    assert out["panel"] is False


def test_detect_broadcast_panel_never_raises(tmp_path):
    bad = tmp_path / "nope.mp4"
    bad.write_bytes(b"x")
    assert broadcast.detect_broadcast_panel(bad)["panel"] is None
    v = video(tmp_path, [("speaker", 2.0)])
    rep = broadcast.detect_broadcast_panel(v, frames=3)
    assert rep["panel"] is False and rep["frames"] == 3


# ---------------------------------------------------------------- 3. kept face = the speaker

def _speech_rms(n, seed):
    """Speech energy per sample: syllable bursts under a phrase envelope, never silent."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / speaker.KEPT_FPS
    phrase = 0.5 + 0.5 * np.sin(2 * np.pi * t / 3.1 + rng.uniform(0, 6))
    syllables = np.abs(np.sin(2 * np.pi * t * 1.7 + rng.uniform(0, 6)))
    return 0.03 + 0.09 * phrase * syllables + rng.uniform(0, 0.01, n)


def _series(kind, seed=0, n=240):
    """(mouth motion, audio RMS) of the kept face over n samples (24 s at KEPT_FPS)."""
    rng = np.random.default_rng(seed + 100)
    rms = _speech_rms(n, seed)
    if kind == "listener":             # a still mouth, its twitches unrelated to the speech
        motion = 0.3 + 0.3 * rng.random(n)
    elif kind == "backchannel":        # a listener whose mouth moves in the speaker's pauses
        motion = 0.2 + 0.6 * (1.0 - (rms - rms.min()) / (rms.max() - rms.min()))
    elif kind == "speaker":            # the mouth follows the speech
        motion = 2.0 + 40.0 * rms + rng.normal(0, 0.3, n)
    elif kind == "busy_speaker":       # the mouth moves all the time, out of step with the energy
        motion = 3.0 + 2.0 * rng.random(n)
    else:                              # "tile_speaker": a small tile, little motion that follows
        motion = 0.2 + 5.0 * rms + rng.normal(0, 0.08, n)
    return list(np.maximum(motion, 0.0)), list(rms)


def _verdict(kind, seed=0):
    motion, rms = _series(kind, seed)
    sp = [v >= speaker.DEFAULTS["speech_rms"] for v in rms]
    return speaker.kept_face_verdict(motion, rms, sp)


def test_the_kept_face_of_a_silent_listener_is_not_the_speaker():
    for kind, seed in (("listener", 1), ("listener", 2), ("backchannel", 0)):
        v = _verdict(kind, seed)
        assert v["not_speaker_frac"] > qg.KEPT_FACE_NOT_SPEAKER_MAX, (kind, seed, v)


def test_kept_face_speakers_pass_even_in_a_low_res_tile_or_with_no_correlation():
    for kind, seed in (("speaker", 0), ("busy_speaker", 0), ("busy_speaker", 1)):
        assert _verdict(kind, seed)["not_speaker_frac"] == 0.0, (kind, seed)
    assert _verdict("tile_speaker")["not_speaker_frac"] <= qg.KEPT_FACE_NOT_SPEAKER_MAX


def test_kept_face_degrades_to_no_data_without_a_face(tmp_path):
    n = 100
    v = speaker.kept_face_verdict([None] * n, [0.1] * n, [True] * n)
    assert v["not_speaker_frac"] is None and v["reason"] == "no face on the canvas"
    short = speaker.kept_face_verdict([1.0] * 20 + [None] * 80, [0.1] * n, [True] * n)
    assert short["not_speaker_frac"] is None                       # 2 s of 10 s with a face
    assert speaker.kept_face_verdict([], [], [])["not_speaker_frac"] is None
    nobody = tmp_path / "nobody.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=gray:s=540x960:d=4",
                    "-f", "lavfi", "-i", "sine=frequency=300:d=4", "-shortest", "-c:v", "libx264",
                    "-preset", "ultrafast", str(nobody)], check=True)
    m = qg.measure_kept_face_speaker(nobody)
    assert m["passed"] is None and "no face" in m["reason"]
    assert qg.measure_kept_face_speaker(tmp_path / "missing.mp4")["passed"] is None
    assert qg.measure_kept_face_speaker(nobody, rendered_layout="general")["passed"] is None


def _clip_words(start_s):
    """Source-time words of a 24.4 s clip from `start_s`: four phrases of 12 words with pauses
    between them, plus a word that straddles the start and one after the end."""
    words, t = [{"w": "so", "start": start_s - 0.2, "end": start_s + 0.1}], start_s + 0.3
    for p in range(4):
        for k in range(12):
            words.append({"w": f"word{p}{k}", "start": round(t, 3), "end": round(t + 0.32, 3)})
            t += 0.44
        t += 0.9
    words.append({"w": "next", "start": start_s + 24.6, "end": start_s + 24.9})
    return words


def test_kept_face_measure_fails_as_wrong_speaker(monkeypatch):
    start_s, end_s = 100.0, 124.4
    seen = {}

    def fake(path, speech_spans=None, **k):
        seen["spans"] = speech_spans
        return {**_verdict("backchannel"), "speech_from": "words"}
    monkeypatch.setattr(speaker, "kept_face_speech_check", fake)
    m = qg.measure_kept_face_speaker("clip.mp4", _clip_words(start_s), start_s, end_s,
                                     rendered_layout="track")
    assert m["passed"] is False and "not the speaker" in m["errors"][0]
    assert list(reason_codes.codes_from_gate_report({"kept_face_speaker": m})) == ["wrong_speaker"]
    spans = seen["spans"]                                          # the clip's words, clip time
    assert len(spans) == 4 and spans[0][0] == 0.0 and spans[-1][1] > 23.0
    assert all(0.0 <= a < b <= end_s - start_s for a, b in spans)


def test_the_broadcast_codes_are_known_and_have_no_repair():
    for code in ("wrong_speaker", "cover_frame_layout"):
        assert code in reason_codes.CODES and not reason_codes.fixable(code)
    assert reason_codes.classify("the speaker is off screen: a silent listener") == ["wrong_speaker"]
    assert "cover_frame_layout" in reason_codes.classify("the cover frame is the boxed broadcast")
