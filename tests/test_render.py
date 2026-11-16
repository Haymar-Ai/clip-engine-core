"""Render tests: reframe decisions, caption/hook ASS, single-encode recompose + QC.

Offline. Encode-path tests render tiny synthetic media (ffmpeg testsrc2 + sine) for real;
faces are not needed in the media — track/split graphs are forced by mocking reframe.analyze.

Font metrics are checked against the caption font the engine resolved (captions.FONT_PATH /
FONT_NAME: Inter-VF from render.font_dir, else the system DejaVu Sans Bold). Line counts that
only hold for Inter's metrics are checked only when Inter is the resolved font.
"""
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RENDER = ROOT / "clipengine" / "render"

from clipengine.render import captions
from clipengine.render import hud
from clipengine.render import recompose
from clipengine.render import reframe
from clipengine import config
from helpers import hud_text_ocr

INTER_RESOLVED = captions.FONT_NAME == "Inter Black"
W, H = 1280, 720


# ---------------------------------------------------------------- helpers

def ffprobe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
                        str(path)], capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    return {"width": v["width"], "height": v["height"],
            "duration": float(d["format"]["duration"]),
            "has_audio": any(s["codec_type"] == "audio" for s in d["streams"])}


def make_media(path, dur, size="1280x720", rate=30):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}",
                    "-f", "lavfi", "-i", "sine=frequency=440", "-t", str(dur),
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path)],
                   check=True)
    return path


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    return make_media(tmp_path_factory.mktemp("media") / "src.mp4", 3)


@pytest.fixture(scope="module")
def long_media(tmp_path_factory):
    # >100s so a START=100 window exists; small + low rate keeps generation fast
    return make_media(tmp_path_factory.mktemp("media_long") / "long.mp4", 103, "320x180", 10)


def write_words(path, words):
    path.write_text(json.dumps([{"w": w, "start": s, "end": e} for w, s, e in words]))
    return path


class Spy:
    """Wraps subprocess.run, recording every command."""

    def __init__(self, real):
        self.real, self.calls = real, []

    def __call__(self, cmd, *a, **k):
        self.calls.append([str(c) for c in cmd])
        return self.real(cmd, *a, **k)


def encode_calls(calls):
    return [c for c in calls if any("libx264" in x for x in c)]


def assert_single_encode(calls, captions=False, loudnorm=False):
    enc = encode_calls(calls)
    assert len(enc) == 1, f"expected exactly one libx264 encode, got {len(enc)}"
    joined = " ".join(enc[0])
    assert "+faststart" in joined
    if captions:
        assert "subtitles=" in joined
    if loudnorm:
        assert "loudnorm" in joined and "measured_I" in joined
    return enc[0]


def face(cx, h=0.25, cy=0.4):
    return {"cx": cx, "cy": cy, "w": h * 0.56, "h": h, "score": 0.95}


def samples(n, faces_at, fps=3):
    return [{"t": round(i / fps, 3), "faces": faces_at(i)} for i in range(n)]


def eval_crop_x(expr, t):
    """Evaluate reframe.crop_x_expr output (nested if(lt(t,T),VAL,REST), VAL a literal or a
    linear ramp "(X0+(X1-X0)*(t-T0)/DUR)") at time t. Rounds ramp values like ffmpeg's crop
    filter (float x truncated toward zero via int()).
    """
    m = re.fullmatch(r"if\(lt\(t,([\d.]+)\),(.+?),(.*)\)", expr.strip())
    if not m:
        return int(expr)
    thresh, val, rest = m.group(1), m.group(2), m.group(3)
    if t >= float(thresh):
        return eval_crop_x(rest, t)
    rm = re.fullmatch(r"\((-?\d+)\+\((-?\d+)-(-?\d+)\)\*\(t-([\d.]+)\)/([\d.]+)\)", val)
    if rm:
        x0, x1, _, t0, dur = float(rm.group(1)), float(rm.group(2)), None, \
            float(rm.group(4)), float(rm.group(5))
        return int(x0 + (x1 - x0) * (t - t0) / dur)
    return int(val)


def ass_events(path, style):
    out = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("Dialogue:"):
            f = line.split(",", 9)
            if f[3] == style:
                out.append({"start": f[1], "end": f[2], "text": f[9]})
    return out


def ass_secs(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def plain(text):
    return re.sub(r"\{[^}]*\}", "", text)


def caption_font(size):
    """The caption font the engine resolved (captions.FONT_PATH, with its named instance when it
    has one) loaded fresh at PIL `size`."""
    from PIL import ImageFont
    f = ImageFont.truetype(str(captions.FONT_PATH), size)
    if captions.FONT_VARIATION:
        f.set_variation_by_name(captions.FONT_VARIATION)
    return f


# ---------------------------------------------------------------- criterion 5: decide_layout

def test_decide_layout_no_faces_is_general():
    assert reframe.decide_layout(samples(30, lambda i: [])) == "general"
    assert reframe.decide_layout([]) == "general"


def test_decide_layout_one_stable_face_is_track():
    assert reframe.decide_layout(samples(30, lambda i: [face(0.55)] if i % 10 else [])) == "track"


def test_decide_layout_two_separated_faces_is_split():
    s = samples(30, lambda i: [face(0.28, 0.26), face(0.72, 0.24)])
    assert reframe.decide_layout(s) == "split"
    # equal sizes (larger side flips) but both keep their seats -> still split
    s = samples(30, lambda i: [face(0.3, 0.25 + 0.01 * (i % 2)), face(0.7, 0.25 + 0.01 * (1 - i % 2))])
    assert reframe.decide_layout(s) == "split"


def test_decide_layout_faces_in_30_percent_is_general():
    assert reframe.decide_layout(samples(30, lambda i: [face(0.5)] if i % 10 < 3 else [])) == "general"


def test_decide_layout_split_needs_separation_size_and_frequency():
    close = samples(30, lambda i: [face(0.45), face(0.62)])            # < 0.30 apart
    small = samples(30, lambda i: [face(0.2, 0.06), face(0.8, 0.06)])  # h < 0.08
    rare = samples(30, lambda i: [face(0.3), face(0.7)] if i % 10 < 3 else [face(0.5)])
    assert [reframe.decide_layout(s) for s in (close, small, rare)] == ["track"] * 3


# ---------------------------------------------------------------- criterion 6: track_crops

CW = reframe.crop_width(W, H)


def test_crop_width_is_even_9_16():
    assert CW == 404 and reframe.crop_width(1920, 1080) == 608


@pytest.mark.parametrize("cx,expect", [(0.01, 0), (0.99, W - CW)])
def test_track_crop_stays_inside_frame_at_extremes(cx, expect):
    segs = reframe.track_crops(samples(27, lambda i: [face(cx)]), [], 9.0, W, H)
    assert segs == [{"t0": 0.0, "t1": 9.0, "x": expect, "cut": True}]


def test_track_one_segment_per_stable_shot():
    s = samples(27, lambda i: [face(0.5 + 0.01 * (i % 3))])      # detector jitter only
    segs = reframe.track_crops(s, [], 9.0, W, H)
    assert len(segs) == 1 and segs[0]["t0"] == 0 and segs[0]["t1"] == 9.0
    assert segs[0]["x"] == round(0.51 * W - CW / 2)


def test_track_new_segment_at_shot_boundary_when_subject_moves():
    s = samples(27, lambda i: [face(0.3)] if i / 3 < 4.0 else [face(0.75)])
    segs = reframe.track_crops(s, [4.0], 9.0, W, H)
    assert [(g["t0"], g["t1"]) for g in segs] == [(0.0, 4.0), (4.0, 9.0)]
    assert segs[0]["x"] == round(0.3 * W - CW / 2) and segs[1]["x"] == round(0.75 * W - CW / 2)
    # same position across the cut -> merged into one locked segment
    s = samples(27, lambda i: [face(0.5)])
    assert len(reframe.track_crops(s, [4.0], 9.0, W, H)) == 1


def test_track_drift_in_long_shot_creates_sub_segments():
    s = samples(36, lambda i: [face(0.2 + 0.4 * i / 35)])        # 12s shot drifting 40% of width
    segs = reframe.track_crops(s, [], 12.0, W, H)
    assert len(segs) > 1
    assert all(g["t0"] % 3 == 0 for g in segs)
    assert [g["x"] for g in segs] == sorted(g["x"] for g in segs)
    assert segs[0]["cut"] and not any(g["cut"] for g in segs[1:])   # re-locks ease, cuts don't
    # the primary face hops across the room mid-shot (wide shot, other speaker): hard switch
    hop = samples(36, lambda i: [face(0.1 if i < 20 else 0.9)])
    hs = reframe.track_crops(hop, [], 12.0, W, H)
    assert len(hs) > 1 and all(g["cut"] for g in hs)
    small = samples(36, lambda i: [face(0.5 + 0.08 * i / 35)])   # 8% drift: stays locked
    assert len(reframe.track_crops(small, [], 12.0, W, H)) == 1
    short = samples(15, lambda i: [face(0.2 + 0.4 * i / 14)])   # 5s shot: never sub-split
    assert len(reframe.track_crops(short, [], 5.0, W, H)) == 1


def test_track_faceless_samples_inherit_neighbour_x():
    s = samples(27, lambda i: [face(0.3)] if i / 3 < 3 else ([] if i / 3 < 6 else [face(0.8)]))
    segs = reframe.track_crops(s, [3.0, 6.0], 9.0, W, H)
    x1, x3 = round(0.3 * W - CW / 2), round(0.8 * W - CW / 2)
    assert segs == [{"t0": 0.0, "t1": 6.0, "x": x1, "cut": True}, {"t0": 6.0, "t1": 9.0, "x": x3, "cut": True}]
    lead = samples(27, lambda i: [] if i / 3 < 3 else [face(0.8)])  # leading gap takes the next x
    assert reframe.track_crops(lead, [3.0], 9.0, W, H) == [{"t0": 0.0, "t1": 9.0, "x": x3, "cut": True}]
    assert reframe.track_crops(samples(9, lambda i: []), [], 3.0, W, H)[0]["x"] == (W - CW) // 2


def test_track_tie_break_prefers_continuity():
    # near-equal faces after the cut: the one continuing the previous shot's subject wins
    s = samples(18, lambda i: [face(0.3)] if i / 3 < 3 else [face(0.8, 0.25), face(0.31, 0.245)])
    segs = reframe.track_crops(s, [3.0], 6.0, W, H)
    assert len(segs) == 1 and segs[0]["x"] == round(0.3 * W - CW / 2)
    # a clearly larger face still wins over continuity
    s = samples(18, lambda i: [face(0.3)] if i / 3 < 3 else [face(0.8, 0.4), face(0.31, 0.2)])
    segs = reframe.track_crops(s, [3.0], 6.0, W, H)
    assert segs[-1]["x"] == round(0.8 * W - CW / 2)


def test_crop_x_expr_evaluates_to_segment_x():
    segs = [{"t0": 0.0, "t1": 2.5, "x": 100}, {"t0": 2.5, "t1": 6.0, "x": 480},
            {"t0": 6.0, "t1": 9.0, "x": 876}]
    expr = reframe.crop_x_expr(segs)
    # camera-cut boundaries (the default) switch hard, exactly at the cut
    assert expr == "if(lt(t,2.5),100,if(lt(t,6),480,876))"
    for t, x in [(0, 100), (2.49, 100), (2.5, 480), (5.9, 480), (6.0, 876), (8.9, 876)]:
        assert eval_crop_x(expr, t) == x
    assert reframe.crop_x_expr(segs[:1]) == "100"
    # a real plan round-trips through the expression
    plan = reframe.track_crops(samples(27, lambda i: [face(0.2)] if i / 3 < 4 else [face(0.7)]),
                               [4.0], 9.0, W, H)
    for g in plan:
        assert eval_crop_x(reframe.crop_x_expr(plan), (g["t0"] + g["t1"]) / 2) == g["x"]


def test_crop_x_expr_eases_in_shot_relock_but_not_camera_cut():
    segs = [{"t0": 0.0, "t1": 2.5, "x": 100, "cut": True},
            {"t0": 2.5, "t1": 6.0, "x": 480, "cut": False}]
    expr = reframe.crop_x_expr(segs)
    before, mid, after = eval_crop_x(expr, 2.5 - reframe.PAN_S), \
        eval_crop_x(expr, 2.5 - reframe.PAN_S / 2), eval_crop_x(expr, 2.5)
    assert before == 100 and after == 480
    assert 100 < mid < 480   # ramping, not a hard step
    segs[1]["cut"] = True     # a pan across a camera cut would slide the outgoing shot
    assert reframe.crop_x_expr(segs) == "if(lt(t,2.5),100,480)"


def test_split_crops_face_centred_9_8_inside_frame():
    sp = reframe.split_crops(samples(20, lambda i: [face(0.08, 0.25), face(0.93, 0.25)]), W, H)
    for c in sp.values():
        assert c["w"] % 2 == 0 and c["h"] % 2 == 0
        assert abs(c["w"] / c["h"] - 9 / 8) < 0.02
        assert 0 <= c["x"] <= W - c["w"] and 0 <= c["y"] <= H - c["h"]
    assert sp["left"]["x"] == 0 and sp["right"]["x"] == W - sp["right"]["w"]
    mid = reframe.split_crops(samples(20, lambda i: [face(0.3), face(0.7)]), W, H)
    assert abs(mid["left"]["x"] + mid["left"]["w"] / 2 - 0.3 * W) <= 1


def test_shot_boundaries_on_hard_cut():
    red = np.zeros((90, 160, 3), np.uint8); red[..., 2] = 200
    blue = np.zeros((90, 160, 3), np.uint8); blue[..., 0] = 200
    frames = [(0.0, red), (0.333, red), (0.667, blue), (1.0, blue)]
    assert reframe.shot_boundaries(frames) == [0.667]


def test_shot_boundaries_drops_retrigger_within_min_gap():
    red = np.zeros((90, 160, 3), np.uint8); red[..., 2] = 200
    blue = np.zeros((90, 160, 3), np.uint8); blue[..., 0] = 200
    # red -> blue flash -> red (re-trigger 0.333s later) -> blue (real cut)
    frames = [(0.0, red), (0.333, blue), (0.667, red), (1.0, red), (1.333, blue)]
    assert reframe.shot_boundaries(frames) == [0.333, 1.333]


def test_refine_cut_lands_on_the_exact_frame(tmp_path):
    src = tmp_path / "cut.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=red:s=320x180:r=30:d=1[a];color=c=blue:s=320x180:r=30:d=1[b];"
                    "[a][b]concat=n=2:v=1:a=0", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    # 3fps sampling reports the red->blue edit (at 1.0s) at 1.333s; refined = between frames
    t = reframe.refine_cut(src, 0, 1.333, 30, (320, 180))
    assert abs(t - (1.0 - 0.5 / 30)) <= 0.5 / 30
    # the same edit reached through a window offset keeps clip-relative time
    assert abs(reframe.refine_cut(src, 0.5, 0.833, 30, (320, 180)) - (0.5 - 0.5 / 30)) <= 0.5 / 30
    # no edit near the sample: fall back to the sampled time
    assert reframe.refine_cut(src, 0, 0.4, 30, (320, 180)) == 0.4


def test_detect_faces_without_any_detector_returns_empty(monkeypatch):
    import cv2
    monkeypatch.setattr(reframe, "_DETECTOR", None)
    monkeypatch.setattr(reframe, "MODEL", Path("/nonexistent/model.onnx"))
    monkeypatch.delattr(cv2, "data", raising=False)
    assert reframe.detect_faces(np.zeros((360, 640, 3), np.uint8)) == []
    assert reframe._DETECTOR[0] == "none"


def test_analyze_real_media_without_faces_is_general(media):
    frames = reframe.sample_frames(media, 0, 3)
    assert 8 <= len(frames) <= 10 and frames[0][0] == 0.0
    assert frames[0][1].shape == (360, 640, 3)
    assert all(b[0] > a[0] for a, b in zip(frames, frames[1:])) and frames[-1][0] < 3
    plan = reframe.analyze(media, 0, 3)
    assert plan["layout"] == "general" and plan["face_ratio"] == 0.0
    assert (plan["src_w"], plan["src_h"]) == (W, H)


# ---------------------------------------------------------------- criterion 7: captions

VOCAB = ("the a we I you it so and but money poker table hand river bluff call raise fold "
         "absolutely unbelievable extraordinary championship tournament professional "
         "microphone conversation incredible opportunity understand everything basically "
         "$90K 100% 3:15 don't can't they're").split()


def random_words(n, seed=7):
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        w = rng.choice(VOCAB)
        if rng.random() < 0.15:
            w += rng.choice(".?!")
        d = rng.uniform(0.08, 0.6)
        out.append({"w": w, "start": round(t, 3), "end": round(t + d, 3)})
        t += d + rng.choice([0.0, 0.0, 0.05, 0.1, 0.3, 0.5, 1.2])
    return out


def caption_chunks(path):
    """Chunks as rendered: consecutive events share the chunk text; a new chunk starts when
    the text (without highlight tags) changes."""
    chunks = []
    for ev in ass_events(path, "Caption"):
        size = int(re.search(r"\\fs(\d+)", ev["text"]).group(1))
        text = plain(ev["text"])
        if not chunks or chunks[-1]["text"] != text or chunks[-1]["end"] != ev["start"]:
            chunks.append({"text": text, "size": size, "start": ev["start"], "end": ev["end"]})
        else:
            chunks[-1]["end"] = ev["end"]
    return chunks


def test_caption_chunks_fit_width_words_and_sentences(tmp_path):
    words = random_words(600)
    words.append({"w": "Supercalifragilisticexpialidociously", "start": 999, "end": 1000})
    ass = captions.build_ass(words, tmp_path / "c.ass", layout="track")
    chunks = caption_chunks(ass)
    assert len(chunks) > 150
    for ch in chunks:
        ws = ch["text"].split(" ")
        assert 1 <= len(ws) <= 3, ch
        assert all(not captions.ends_sentence(w) for w in ws[:-1]), ch
        assert captions.hook_width(ch["text"], ch["size"]) <= 900
        if captions.FONT_PATH:
            assert caption_font(round(ch["size"] * captions.ASS_SCALE)).getlength(ch["text"]) \
                <= 900, ch
    assert max(ch["size"] for ch in chunks) == captions.STYLES["opus"]["size"]
    assert min(ch["size"] for ch in chunks) < 92          # the monster word was shrunk to fit


def test_caption_timings_monotonic_non_overlapping(tmp_path):
    ass = captions.build_ass(random_words(400, seed=3), tmp_path / "c.ass", dur=200.0)
    evs = ass_events(ass, "Caption")
    prev_end = 0.0
    for ev in evs:
        s, e = ass_secs(ev["start"]), ass_secs(ev["end"])
        assert s < e and s >= prev_end - 1e-9 and e <= 200.0
        prev_end = e


def test_caption_chunk_rules_gap_hold_and_highlight(tmp_path):
    words = [("one", 0.0, 0.2), ("two", 0.2, 0.4), ("three", 0.7, 0.9),   # 0.3s gap -> break
             ("four", 1.0, 1.2), ("done.", 1.2, 1.5), ("next", 1.6, 1.8),  # sentence end -> break
             ("late", 3.0, 3.3)]                                            # 1.2s gap -> tail
    ass = captions.build_ass([{"w": w, "start": s, "end": e} for w, s, e in words],
                             tmp_path / "c.ass")
    chunks = caption_chunks(ass)
    assert [c["text"] for c in chunks] == ["one two", "three four done.", "next", "late"]
    assert chunks[0]["end"] == chunks[1]["start"] == "0:00:00.70"   # gap < 0.4 bridged
    assert chunks[2]["end"] == "0:00:01.95"                         # 1.8 + 0.15 tail
    assert "{\\c&H0000E5FF&}three{\\c&H00FFFFFF&}" in ass_events(ass, "Caption")[2]["text"]
    # both styles name the font the engine resolved (Inter Black when render.font_dir has it)
    text = Path(ass).read_text()
    for style in ("Caption", "Hook"):
        assert f"Style: {style},{captions.FONT_NAME}," in text


@pytest.mark.parametrize("layout", ["track", "split", "general", "screencast"])
def test_caption_position_outside_bottom_ui_band(tmp_path, layout):
    for fg_h in (None, 608, 810, 1080):
        ass = captions.build_ass(random_words(20), tmp_path / "c.ass", layout=layout, fg_h=fg_h)
        text = Path(ass).read_text()
        assert "MarginV" in text and ",90,90,0," in text                  # MarginL = MarginR = 90
        ys = [int(y) for y in re.findall(r"\\an5\\pos\(540,(\d+)\)", text)]
        assert ys and all(y <= 1500 for y in ys)
    general = captions.build_ass(random_words(5), tmp_path / "g.ass", layout="general")
    assert "\\pos(540,1374)" in Path(general).read_text()                 # (1920+608)/2 + 110
    track = captions.build_ass(random_words(5), tmp_path / "t.ass", layout="track")
    assert "\\an5\\pos(540,1270)" in Path(track).read_text()


def test_caption_words_with_ass_control_chars_render_literally(tmp_path):
    ass = captions.build_ass([{"w": "{\\b1}x", "start": 0, "end": 1}], tmp_path / "c.ass",
                             style="minimal")
    assert "\\{\\\u2060b1\\}x" in ass_events(ass, "Caption")[0]["text"]


# ---------------------------------------------------------------- criterion 8: hook

def test_hook_escaped_literally(tmp_path):
    hook = "Won $90K at 100% odds a:b {x} back\\slash \\N"
    ass = captions.build_ass([], tmp_path / "h.ass", hook=hook, layout="track", dur=3.0)
    (ev,) = ass_events(ass, "Hook")
    text = ev["text"]
    for lit in ("$90K", "100%", "A:B", "\\{X\\}", "BACK\\\u2060SLASH", "\\\u2060N"):
        assert lit in text
    assert re.match(r"\{\\an5\\pos\(540,\d+\)\\fad\(120,400\)\}", text)
    assert (ev["start"], ev["end"]) == ("0:00:00.00", "0:00:03.00")
    body = re.sub(r"\{\\c&H[0-9A-F]{8}&\}", "", text.split("}", 1)[1])
    assert re.search(r"(?<!\\)[{}]", body) is None                 # no unescaped braces
    style = next(l for l in Path(ass).read_text().splitlines() if l.startswith("Style: Hook"))
    f = style.split(",")
    assert f[15] == "1" and int(f[16]) >= 6                       # outlined text, no box


def test_hook_digit_words_highlighted_rest_white(tmp_path):
    ass = captions.build_ass([], tmp_path / "h.ass", hook="He made $500k/month reading reviews", dur=9)
    text = ass_events(ass, "Hook")[0]["text"]
    yellow = captions.STYLES["opus"]["highlight"]
    assert f"{{\\c{yellow}}}$500K/MONTH{{\\c{captions.WHITE}}}" in text
    assert text.count(yellow) == 1 and "HE MADE" in text and "READING REVIEWS" in text


def test_hook_fades_out_after_hold_on_every_layout(tmp_path):
    for layout in ("track", "split", "broadcast", "hudband", "general"):
        ass = captions.build_ass([], tmp_path / f"{layout}.ass", hook="Big hook", layout=layout,
                                 dur=60.0, fg_h=1312 if layout == "hudband" else None)
        (ev,) = ass_events(ass, "Hook")
        assert ev["start"] == "0:00:00.00" and ev["end"] == "0:00:03.40", layout
        assert "\\fad(120,400)" in ev["text"]
    (ev,) = ass_events(captions.build_ass([], tmp_path / "s.ass", hook="x", dur=2.0), "Hook")
    assert ev["end"] == "0:00:02.00"                              # never past the clip


def test_hook_wraps_to_max_lines_within_safe_width(tmp_path):
    hook = "He folded pocket aces preflop and then watched the river hand him $90K!"
    ass = captions.build_ass([], tmp_path / "h.ass", hook=hook, dur=5.0)
    (ev,) = ass_events(ass, "Hook")
    lines = re.sub(r"\{\\c&H[0-9A-F]{8}&\}", "", ev["text"].split("}", 1)[1]).split("\\N")
    style = next(l for l in Path(ass).read_text().splitlines() if l.startswith("Style: Hook"))
    size = int(style.split(",")[2])
    assert 1 <= len(lines) <= 3 and 64 <= size <= 120
    for line in lines:
        assert captions.hook_width(line, size) <= captions.HOOK_W
        if captions.FONT_PATH:
            assert caption_font(round(size * captions.ASS_SCALE)).getlength(line) \
                <= captions.HOOK_W + 4
    very_long = " ".join(["unbelievable"] * 30)          # cannot fit: truncated, still 3 lines
    lines, size = captions.wrap_hook(very_long.upper())
    assert len(lines) == 3 and size == captions.HOOK_SIZES[-1] and lines[-1].endswith("…")
    assert all(captions.hook_width(l, size) <= captions.HOOK_W for l in lines)


def _hook_y(ass):
    return int(re.search(r"\\pos\(540,(\d+)\)", ass_events(ass, "Hook")[0]["text"]).group(1))


def test_hook_placement_no_faces_sits_under_top_ui(tmp_path):
    y = _hook_y(captions.build_ass([], tmp_path / "h.ass", hook="Short hook", dur=9))
    assert captions.HOOK_TOP < y < captions.HOOK_TOP + 120
    hook = "Wrong barcode cost him $1M in inventory"
    ass = captions.build_ass([], tmp_path / "h.ass", hook=hook, dur=9)
    size = int(next(l for l in Path(ass).read_text().splitlines()
                    if l.startswith("Style: Hook")).split(",")[2])
    lines = re.sub(r"\{[^}]*\}", "", ass_events(ass, "Hook")[0]["text"]).split("\\N")
    widest = max(captions.hook_width(l, size) for l in lines)
    # fewest, widest lines at a readable size: of the wraps that fit at >= HOOK_WIDE_MIN, the one
    # with the fewest lines, at its biggest size; it spans nearly the full hook width, which is
    # wider than the caption safe width
    readable = [f for f in captions.hook_fits(hook.upper()) if f[1] >= captions.HOOK_WIDE_MIN]
    best = min(readable, key=lambda f: (len(f[0]), -f[1]))
    assert (len(lines), size) == (len(best[0]), best[1]) and size >= captions.HOOK_WIDE_MIN
    assert all(captions.hook_width(l, size) <= captions.HOOK_W for l in lines)
    assert captions.HOOK_W > captions.SAFE_W - 2 * captions.STYLES["opus"]["outline"]
    assert widest >= 0.95 * captions.HOOK_W
    if INTER_RESOLVED:
        assert len(lines) == 2                  # Inter's metrics fit it on two readable lines


def test_hook_placement_clears_faces_and_captions(tmp_path):
    hook = "He read Yeti reviews and made $500k/month"
    for face in [(150, 700), (200, 1000), (700, 1100), (300, 500)]:
        ass = captions.build_ass([], tmp_path / "h.ass", hook=hook, dur=9, avoid=[face])
        y = _hook_y(ass)
        size = int(next(l for l in Path(ass).read_text().splitlines()
                        if l.startswith("Style: Hook")).split(",")[2])
        n = ass_events(ass, "Hook")[0]["text"].count("\\N") + 1
        half = n * size * captions.HOOK_LINE_H / 2
        assert y - half >= captions.HOOK_TOP - 1
        assert y + half <= captions.CAPTION_Y - captions.CAPTION_CLEAR + 1
        assert y + half <= face[0] or y - half >= face[1], (face, y)       # never on a face
    # a tight talking head leaves a small gap: the hook shrinks to fit it rather than cover the face
    ass = captions.build_ass([], tmp_path / "h.ass", hook=hook, dur=9, avoid=[(0, 850)])
    size = int(next(l for l in Path(ass).read_text().splitlines()
                    if l.startswith("Style: Hook")).split(",")[2])
    n = ass_events(ass, "Hook")[0]["text"].count("\\N") + 1
    assert _hook_y(ass) - n * size * captions.HOOK_LINE_H / 2 >= 850
    # a face filling the whole free zone: least overlap wins, still inside the zone
    y = _hook_y(captions.build_ass([], tmp_path / "h.ass", hook=hook, dur=9, avoid=[(0, 1920)]))
    assert captions.HOOK_TOP < y < captions.CAPTION_Y


def test_hook_avoid_maps_faces_through_the_crop():
    plan = {"layout": "track", "src_w": 1920, "src_h": 1080,
            "segments": [{"t0": 0.0, "t1": 60.0, "x": 656, "cut": True}],
            "hook_faces": [{"t": 0.0, "faces": [
                {"cx": 0.5, "cy": 0.3, "w": 0.1, "h": 0.2, "score": 0.9},     # in the crop
                {"cx": 0.05, "cy": 0.7, "w": 0.1, "h": 0.2, "score": 0.9}]}]}  # cropped away
    (band,) = recompose.hook_avoid(plan)
    assert band == pytest.approx(((0.3 - 0.9 * 0.2) * 1920, (0.3 + 0.65 * 0.2) * 1920))
    split = {"layout": "split", "src_w": 1920, "src_h": 1080,
             "split": {"left": {"x": 0, "y": 0, "w": 960, "h": 540},
                       "right": {"x": 960, "y": 540, "w": 960, "h": 540}},
             "hook_faces": [{"t": 0.0, "faces": [{"cx": 0.75, "cy": 0.75, "w": 0.1, "h": 0.1}]}]}
    (y0, y1), = recompose.hook_avoid(split)
    assert 960 <= y0 < y1 <= 1920                                  # lands in the bottom band
    assert recompose.hook_avoid({"layout": "general"}) == []
    assert recompose.hook_avoid({"layout": "screencast", "manual": True}) == []


def test_hook_general_layout_centred_in_top_band(tmp_path):
    ass = captions.build_ass([], tmp_path / "h.ass", hook="Hook", layout="general", dur=2)
    assert ass_events(ass, "Hook")[0]["text"].startswith("{\\an5\\pos(540,328)")


# ---------------------------------------------------------------- criteria 2 + 3: single encode, real render

def test_single_encode_full_render(media, tmp_path, monkeypatch, capsys):
    wj = write_words(tmp_path / "words.json", [("Big", 0.2, 0.5), ("money", 0.5, 0.9),
                                               ("hand.", 0.9, 1.4), ("Wow", 1.8, 2.2)])
    out = tmp_path / "out" / "clip.mp4"
    spy = Spy(subprocess.run)
    monkeypatch.setattr(subprocess, "run", spy)
    rc = recompose.main([str(media), "0", "3", str(out), "--captions", "--words", str(wj),
                         "--loudnorm", "-14", "--hook", "Won $90K at 100%: {wow}"])
    monkeypatch.undo()
    assert rc == 0
    enc = assert_single_encode(spy.calls, captions=True, loudnorm=True)
    assert enc.index("-ss") < enc.index("-i") and enc.index("-t") < enc.index("-i")
    joined = " ".join(enc)
    for flag in ("-preset medium", "-crf 18", "-profile:v high", "-pix_fmt yuv420p", "-g 60",
                 "-b:a 192k", "-ar 48000", "-ac 2", "linear=true", "afade=t=in:st=0:d=0.04",
                 "afade=t=out:st=2.880:d=0.12"):
        assert flag in joined, flag
    assert "drawtext" not in joined
    pass1 = [c for c in spy.calls if "print_format=json" in " ".join(c)]
    assert len(pass1) == 1 and "-vn" in pass1[0] and "libx264" not in pass1[0]
    assert not any("whisper" in " ".join(c) for c in spy.calls)
    info = ffprobe(out)
    assert (info["width"], info["height"]) == (1080, 1920) and info["has_audio"]
    assert not list(out.parent.glob(".recompose_*"))                 # temp dir cleaned up
    last = capsys.readouterr().out.strip().splitlines()[-1]
    rep = json.loads(last[3:])
    assert last.startswith("QC ") and rep["passed"] and abs(rep["integrated_lufs"] + 14) <= 1.5


def _old_two_pass_flow(src, start, dur, out, caption=True, loudnorm=-14.0, hook="Big hook"):
    """Inline reconstruction of the pre-PHASE4b recompose flow: a base libx264 render, then
    finish() re-encoding again for captions/hook/loudnorm."""
    def run(cmd):
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).returncode == 0
    tmp = Path(out).parent / "recompose_tmp"
    base = tmp / f"{Path(out).stem}_base.mp4"
    vf = recompose.general_vf().replace("[v0]", "[v]")
    run(["ffmpeg", "-y", "-ss", str(start), "-t", str(dur), "-i", str(src), "-filter_complex", vf,
         "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(base)])
    vf2 = []
    if caption:
        run(["whisper", str(base), "--model", "small.en", "--word_timestamps", "True",
             "--output_format", "json", "--output_dir", str(tmp), "--language", "en"])
        run(["python3", "build_ass.py", str(tmp / f"{base.stem}.json"), str(tmp / "x.ass"), "opus"])
        vf2.append(f"subtitles={tmp / 'x.ass'}")
    if hook:
        vf2.append(f"drawtext=text='{hook}':fontsize=52:x=(w-text_w)/2:y=210")
    cmd = ["ffmpeg", "-y", "-i", str(base), "-vf", ",".join(vf2)]
    cmd += ["-af", f"loudnorm=I={loudnorm}:TP=-1.0:LRA=11"] if loudnorm is not None else []
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac", "-b:a", "192k", str(out)]
    run(cmd)


def test_single_encode_check_fails_on_old_two_pass_flow(tmp_path, monkeypatch):
    calls = []

    def fake(cmd, *a, **k):
        calls.append([str(c) for c in cmd])
        return subprocess.CompletedProcess(cmd, 0, b"", b"")
    monkeypatch.setattr(subprocess, "run", fake)
    _old_two_pass_flow("src.mp4", 0, 3, tmp_path / "clip.mp4")
    monkeypatch.undo()
    assert len(encode_calls(calls)) == 2
    with pytest.raises(AssertionError, match="exactly one libx264 encode, got 2"):
        assert_single_encode(calls, captions=True, loudnorm=True)


def test_real_render_cli_probes_1080x1920_with_audio(media, tmp_path):
    out = tmp_path / "clip.mp4"
    r = subprocess.run([sys.executable, str(RENDER / "recompose.py"), str(media), "0", "3", str(out),
                        "--layout", "auto", "--loudnorm", "-14"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    last = r.stdout.strip().splitlines()[-1]
    assert last.startswith("QC ")
    rep = json.loads(last[3:])
    assert {"width", "height", "duration", "has_audio", "integrated_lufs", "layout"} <= rep.keys()
    assert rep["layout"] == "general" and rep["passed"]
    info = ffprobe(out)
    assert (info["width"], info["height"]) == (1080, 1920)
    assert abs(info["duration"] - 3.0) <= 0.15 and info["has_audio"]


@pytest.mark.parametrize("layout", ["track", "split"])
def test_forced_track_and_split_graphs_render(media, tmp_path, monkeypatch, capsys, layout):
    plans = {
        "track": {"layout": "track", "src_w": W, "src_h": H, "shots": [1.5],
                  "segments": [{"t0": 0.0, "t1": 1.5, "x": 100}, {"t0": 1.5, "t1": 2.0, "x": 700}]},
        "split": {"layout": "split", "src_w": W, "src_h": H, "shots": [],
                  "split": reframe.split_crops(samples(9, lambda i: [face(0.3), face(0.7)]), W, H)},
    }
    analyze = Mock(return_value=plans[layout])
    monkeypatch.setattr(recompose.reframe, "analyze", analyze)
    wj = write_words(tmp_path / "w.json", [("hello", 0.1, 0.5), ("there", 0.5, 1.0)])
    spy = Spy(subprocess.run)
    monkeypatch.setattr(subprocess, "run", spy)
    out, plan_json = tmp_path / f"{layout}.mp4", tmp_path / "plan.json"
    rc = recompose.main([str(media), "0", "2", str(out), "--captions", "--words", str(wj),
                         "--emit-plan", str(plan_json)])
    monkeypatch.undo()
    assert rc == 0 and analyze.called
    enc = " ".join(assert_single_encode(spy.calls, captions=True))
    if layout == "track":
        assert "crop=404:ih:'if(lt(t,1.5),100,700)':0,scale=1080:1920:flags=lanczos,setsar=1" in enc
    else:
        assert "vstack=inputs=2" in enc and "scale=1080:960:flags=lanczos" in enc
    assert json.loads(plan_json.read_text())["layout"] == layout
    info = ffprobe(out)
    assert (info["width"], info["height"]) == (1080, 1920) and info["has_audio"]
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1][3:])["layout"] == layout


def test_general_and_screencast_flags_still_work(media, tmp_path):
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "screencast", "--top",
                              "800:450:0:0", "--bottom", "320:180:960:540", "--split", "1300",
                              "--bottom-mode", "fit"])
    plan = recompose.plan_layout(a, media)
    assert plan["manual"] and plan["split_px"] == 1300
    g = recompose.video_graph(plan, a)
    assert "crop=800:450:0:0" in g and "crop=320:180:960:540" in g and g.endswith("[v0]")
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "split", "--top",
                              "640:720:0:0", "--bottom", "640:720:640:0"])
    assert recompose.plan_layout(a, media)["split_px"] == 960
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "general"])
    assert recompose.plan_layout(a, media) == {"layout": "general"}
    assert recompose.parse_args(["s", "0", "1", "o"]).layout == "auto"


# ---------------------------------------------------------------- criterion 4: QC failure

@pytest.mark.parametrize("bad", [{"width": 1280, "height": 720, "duration": 1.0},
                                 {"width": 1080, "height": 1920, "duration": 0.4}])
def test_qc_failure_exits_3(media, tmp_path, monkeypatch, capsys, bad):
    real = recompose.probe

    def fake_probe(path):
        if Path(path).name.startswith("render"):           # the encoded output
            return dict({"fps": 30.0, "has_audio": True}, **bad)
        return real(path)
    monkeypatch.setattr(recompose, "probe", fake_probe)
    out = tmp_path / "clip.mp4"
    rc = recompose.main([str(media), "0", "1", str(out), "--layout", "general"])
    assert rc == recompose.EXIT_QC == 3
    assert not out.exists() and not list(tmp_path.glob(".recompose_*"))
    rep = json.loads(capsys.readouterr().out.strip().splitlines()[-1][3:])
    assert rep["passed"] is False and rep["errors"]


def test_qc_checks_loudness_window():
    info = {"width": 1080, "height": 1920, "fps": 30.0, "duration": 3.02, "has_audio": True}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(recompose, "probe", lambda p: info)
        mp.setattr(recompose, "measure_lufs", lambda p: -16.0)
        assert recompose.qc("x.mp4", 3.0, -14.0, "track")["passed"] is False
        mp.setattr(recompose, "measure_lufs", lambda p: -14.9)
        assert recompose.qc("x.mp4", 3.0, -14.0, "track")["passed"] is True
        mp.setattr(recompose, "probe", lambda p: dict(info, has_audio=False))
        assert "no audio stream" in recompose.qc("x.mp4", 3.0, None, "track")["errors"]


# ---------------------------------------------------------------- criterion 9: --words

def test_load_words_converts_absolute_to_clip_relative(tmp_path):
    wj = write_words(tmp_path / "w.json", [("before", 98.0, 99.5), ("edge", 99.8, 100.3),
                                           ("hi", 100.25, 100.5), ("last", 102.8, 103.4),
                                           ("after", 103.1, 103.5)])
    ws = recompose.load_words(wj, 100.0, 3.0)
    assert [w["w"] for w in ws] == ["edge", "hi", "last"]
    assert ws[0]["start"] == 0.0 and abs(ws[0]["end"] - 0.3) < 1e-9       # partial clamped
    assert abs(ws[1]["start"] - 0.25) < 1e-9 and ws[2]["end"] == 3.0


def test_words_path_is_clip_relative_and_skips_transcription(long_media, tmp_path, monkeypatch):
    wj = write_words(tmp_path / "w.json", [("way", 99.0, 99.6), ("Hello", 100.05, 100.4),
                                           ("world.", 100.4, 100.9), ("again", 101.2, 101.6)])
    mocks = {name: Mock(side_effect=AssertionError("transcription must not run"))
             for name in ("transcribe", "_transcribe_faster_whisper", "_transcribe_whisper_cli",
                          "extract_audio")}
    for name, m in mocks.items():
        monkeypatch.setattr(recompose, name, m)
    seen = {}
    real_build = captions.build_ass

    def spy_build(words, *a, **k):
        seen["words"] = words
        return real_build(words, *a, **k)
    monkeypatch.setattr(recompose.captions, "build_ass", spy_build)
    rc = recompose.main([str(long_media), "100", "2", str(tmp_path / "c.mp4"), "--layout", "general",
                         "--captions", "--words", str(wj)])
    assert rc == 0
    for m in mocks.values():
        m.assert_not_called()
    ws = seen["words"]
    assert [w["w"] for w in ws] == ["Hello", "world.", "again"]
    assert abs(ws[0]["start"] - 0.05) < 1e-6 and all(0 <= w["start"] < w["end"] <= 2 for w in ws)


def test_transcription_fallback_is_audio_only(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(recompose, "run", lambda cmd, cwd=None: calls.append([str(c) for c in cmd]))
    monkeypatch.setattr(recompose, "_transcribe_faster_whisper",
                        Mock(return_value=[{"w": "hi", "start": 0.1, "end": 0.3}]))
    assert recompose.transcribe("s.mp4", 5.0, 3.0, "small.en", tmp_path)[0]["w"] == "hi"
    (cmd,) = calls
    assert "-vn" in cmd and cmd[cmd.index("-ac") + 1] == "1" and cmd[cmd.index("-ar") + 1] == "16000"
    assert "libx264" not in cmd


# ---------------------------------------------------------------- criterion 10: no clipify, orig gone

def test_no_clipify_skill_reference_and_orig_removed():
    src = (RENDER / "recompose.py").read_text()
    assert "clipify" not in src
    assert "drawtext" not in src
    assert not (RENDER / "build_ass.orig.py").exists()


# ---------------------------------------------------------------- broadcast layout (poker HUD)

def test_broadcast_crop_width_matches_top_band_aspect():
    # top band is 1080x1312, so a full-height crop from a 1920x1080 source is 1080*(1080/1312)=889 -> 888.
    assert reframe.broadcast_crop_width(1920, 1080) == 888
    # crop stays inside the source frame
    assert reframe.broadcast_crop_width(640, 1080) == 640


def test_broadcast_plan_uses_broadcast_crop_width(monkeypatch):
    def fake_analyze(src, start, dur, force=None):
        s = samples(9, lambda i: [face(0.5)])
        segs = reframe.track_crops(s, [], 3.0, W, H,
                                    cw=reframe.broadcast_crop_width(W, H))
        return {"layout": "broadcast", "src_w": W, "src_h": H, "shots": [], "segments": segs}
    monkeypatch.setattr(reframe, "analyze", fake_analyze)
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "broadcast"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    assert plan["layout"] == "broadcast" and plan["segments"]
    cw = reframe.broadcast_crop_width(W, H)
    for seg in plan["segments"]:
        assert 0 <= seg["x"] <= W - cw


def test_broadcast_layout_forced_via_flag(monkeypatch):
    captured = {}

    def fake_analyze(src, start, dur, force=None, cw=None):
        captured["force"] = force
        return {"layout": force or "general", "src_w": W, "src_h": H, "shots": [], "segments": []}
    monkeypatch.setattr(reframe, "analyze", fake_analyze)
    # --layout broadcast still available directly
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "broadcast"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    assert captured["force"] == "broadcast" and plan["layout"] == "broadcast"
    # unmapped vertical falls back to decide_layout (force stays None)
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "auto",
                              "--vertical", "unmapped"])
    recompose.plan_layout(a, Path("s.mp4"))
    assert captured["force"] is None
    # no vertical: same fallback
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "auto"])
    recompose.plan_layout(a, Path("s.mp4"))
    assert captured["force"] is None


# ---------------------------------------------------------------- non-stream default

def test_non_stream_verticals_default_to_full_bleed_track(monkeypatch):
    """podcast / interview / general / product route to the full-bleed `track`
    layout under `--layout auto`, so active-speaker tracking takes over instead of
    split or decide_layout picking a stacked layout."""
    captured = {}

    def fake_analyze(src, start, dur, force=None, cw=None, speaker_cfg=None):
        captured["force"] = force
        return {"layout": force or "general", "src_w": W, "src_h": H, "shots": [],
                "segments": [{"t0": 0.0, "t1": float(dur), "x": 0, "cut": True}]}

    monkeypatch.setattr(reframe, "analyze", fake_analyze)
    for vertical in ("podcast", "interview", "general", "product"):
        a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "auto",
                                  "--vertical", vertical])
        plan = recompose.plan_layout(a, Path("s.mp4"))
        assert captured["force"] == "track", f"{vertical} should route to track"
        assert plan["layout"] == "track"
    # stream maps to the stacked gamestack layout; with no cam inset
    # in the source, gamestack falls back to the unforced reframe decision.
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "auto",
                              "--vertical", "stream"])
    recompose.plan_layout(a, Path("s.mp4"))
    assert captured["force"] is None


def test_vertical_layouts_config_maps_non_stream_to_track():
    cfg = config.load()
    m = (cfg.get("render") or {}).get("vertical_layouts") or {}
    for v in ("podcast", "interview", "general", "product"):
        assert m.get(v) == "track", f"vertical_layouts[{v}] must be 'track'"
    # unmapped, decide_layout never stacks (it returns general/split/
    # track), so facecam-over-game streams came out as the 68 % blur letterbox. Streams get the
    # stacked gamestack layout, which keeps its own no-cam fallbacks.
    assert m.get("stream") == "gamestack"


def test_broadcast_vf_stacks_track_over_full_frame():
    segs = [{"t0": 0.0, "t1": 3.0, "x": 100, "cut": True}]
    plan = {"layout": "broadcast", "src_w": W, "src_h": H, "shots": [], "segments": segs}
    vf = recompose.broadcast_vf(plan)
    cw = reframe.broadcast_crop_width(W, H)
    assert f"crop={cw}:ih:'100':0" in vf
    assert f"scale=1080:{reframe.BROADCAST_TOP_H}:flags=lanczos" in vf
    assert f"scale=1080:{reframe.BROADCAST_BOT_H}:force_original_aspect_ratio=decrease" in vf
    assert "vstack=inputs=2" in vf and vf.endswith("[v0]")
    assert reframe.BROADCAST_TOP_H + reframe.BROADCAST_BOT_H == 1920


def test_broadcast_caption_position_stays_above_seam(tmp_path):
    ass = captions.build_ass(random_words(20), tmp_path / "c.ass", layout="broadcast")
    text = Path(ass).read_text()
    ys = [int(y) for y in re.findall(r"\\an5\\pos\(540,(\d+)\)", text)]
    assert ys and all(y <= reframe.BROADCAST_TOP_H for y in ys)
    # with no faces the hook sits just under the top UI band, above the broadcast captions
    hook_ass = captions.build_ass([], tmp_path / "h.ass", hook="Broadcast hook", layout="broadcast", dur=3.0)
    assert captions.HOOK_TOP < _hook_y(hook_ass) < captions.BROADCAST_CAPTION_Y


def test_broadcast_full_render_is_1080x1920(media, tmp_path, monkeypatch, capsys):
    def fake_analyze(src, start, dur, force=None):
        s = samples(6, lambda i: [face(0.5)])
        cw = reframe.broadcast_crop_width(W, H)
        segs = reframe.track_crops(s, [], 2.0, W, H, cw=cw)
        return {"layout": "broadcast", "src_w": W, "src_h": H, "shots": [], "segments": segs}
    monkeypatch.setattr(recompose.reframe, "analyze", fake_analyze)
    out = tmp_path / "broadcast.mp4"
    rc = recompose.main([str(media), "0", "2", str(out), "--layout", "broadcast", "--loudnorm", "-14"])
    monkeypatch.undo()
    assert rc == 0
    info = ffprobe(out)
    assert (info["width"], info["height"]) == (1080, 1920) and info["has_audio"]
    rep = json.loads(capsys.readouterr().out.strip().splitlines()[-1][3:])
    assert rep["layout"] == "broadcast" and rep["passed"]


# ---------------------------------------------------------------- hudband layout (PHASE 4h)

POKER_PROFILE = {"players":    [0.0078, 0.5625, 0.2734, 0.9583],
                 "pot":        [0.7305, 0.7361, 0.9023, 0.9583],
                 "outs_strip": [0.0000, 0.0278, 0.8203, 0.0764]}


def _band_items_for(profile, src_w=W, src_h=H, intervals=None):
    """plan_band + attach intervals — mimics the recompose plan builder for isolated tests."""
    pack = hud.plan_band(profile, src_w, src_h)
    for it in pack["items"]:
        it["intervals"] = (intervals or {}).get(it["name"], [(0.0, 3.0)])
    return pack


def test_hudband_vf_stacks_track_over_packed_band():
    pack = _band_items_for(POKER_PROFILE)
    top_h = 1920 - pack["height"]
    _, ostrip_y, _, ostrip_h = hud.region_pixels(POKER_PROFILE["outs_strip"], W, H)
    y_off = ostrip_y + ostrip_h
    eff_h = H - y_off
    cw = reframe.hudband_crop_width(W, H, top_h, effective_src_h=eff_h)
    segs = [{"t0": 0.0, "t1": 3.0, "x": 400, "cut": True}]
    plan = {"layout": "hudband", "src_w": W, "src_h": H, "shots": [], "segments": segs,
            "band_height": pack["height"], "band_items": pack["items"], "cw": cw,
            "y_off": y_off, "eff_h": eff_h}
    vf = recompose.hudband_vf(plan)
    assert f"crop={cw}:{eff_h}:'400':{y_off}" in vf
    assert f"scale=1080:{top_h}:flags=lanczos" in vf
    assert f"color=c=0x0b0b0f:s=1080x{pack['height']}:d=1" in vf
    assert "vstack=inputs=2" in vf and vf.endswith("[v0]")
    for item in pack["items"]:
        sx, sy, sw, sh = item["src"]
        dx, dy, dw, dh = item["dst"]
        assert f"crop={sw}:{sh}:{sx}:{sy}" in vf
        assert f"scale={dw}:{dh}:flags=lanczos" in vf
        assert f"overlay={dx}:{dy}:enable=" in vf


def test_hudband_vf_no_gblur_no_drawbox():
    """PHASE4h: no gblur, no drawbox, ever, in the hudband graph."""
    pack = _band_items_for(POKER_PROFILE)
    top_h = 1920 - pack["height"]
    plan = {"layout": "hudband", "src_w": W, "src_h": H, "shots": [],
            "segments": [{"t0": 0.0, "t1": 3.0, "x": 400, "cut": True}],
            "band_height": pack["height"], "band_items": pack["items"],
            "cw": reframe.hudband_crop_width(W, H, top_h),
            "y_off": 55, "eff_h": H - 55}
    vf = recompose.hudband_vf(plan)
    assert "gblur" not in vf
    assert "drawbox" not in vf


def test_hudband_vf_band_overlays_carry_enable_from_intervals():
    """Each region overlay is gated on its presence intervals."""
    intervals = {"players": [(0.5, 2.0), (3.0, 5.5)], "pot": [(0.0, 6.0)]}
    pack = _band_items_for(POKER_PROFILE, intervals=intervals)
    top_h = 1920 - pack["height"]
    plan = {"layout": "hudband", "src_w": W, "src_h": H, "shots": [],
            "segments": [{"t0": 0.0, "t1": 6.0, "x": 400, "cut": True}],
            "band_height": pack["height"], "band_items": pack["items"],
            "cw": reframe.hudband_crop_width(W, H, top_h),
            "y_off": 55, "eff_h": H - 55}
    vf = recompose.hudband_vf(plan)
    assert "enable='between(t,0.500,2.000)+between(t,3.000,5.500)'" in vf
    assert "enable='between(t,0.000,6.000)'" in vf


def test_hudband_falls_back_to_track_when_players_absent(monkeypatch):
    """Players present < 20% of the window -> None from _plan_hudband (caller uses track)."""
    monkeypatch.setattr(recompose.reframe, "probe_size", lambda src: (W, H))
    monkeypatch.setattr(recompose, "_region_presence",
                        lambda *ar, **kw: {"players": [(0.0, 0.2)], "pot": [(0.0, 3.0)],
                                           "outs_strip": []})
    monkeypatch.setattr(recompose.reframe, "analyze", lambda *ar, **kw: {
        "layout": "track", "src_w": W, "src_h": H, "shots": [],
        "segments": [{"t0": 0.0, "t1": 3.0, "x": 100, "cut": True}]})
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "hudband"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    assert plan["layout"] == "track"


def test_hudband_falls_back_to_track_when_profile_missing(monkeypatch):
    monkeypatch.setattr(recompose, "_hud_profile", lambda vertical: None)
    monkeypatch.setattr(recompose.reframe, "analyze", lambda *ar, **kw: {
        "layout": "track", "src_w": W, "src_h": H, "shots": [],
        "segments": [{"t0": 0.0, "t1": 3.0, "x": 100, "cut": True}]})
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "hudband"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    assert plan["layout"] == "track"


def test_hudband_auto_for_poker_vertical(monkeypatch):
    """--vertical poker resolves to layout=hudband when the profile + presence pass and the OCR
    reads HUD text in the band regions; with no HUD text there, the speaker crop instead."""
    monkeypatch.setattr(recompose.reframe, "probe_size", lambda src: (W, H))
    monkeypatch.setattr(recompose, "_hud_profile", lambda vertical: dict(POKER_PROFILE))
    monkeypatch.setattr(recompose, "_region_presence",
                        lambda *ar, **kw: {"players": [(0.0, 3.0)], "pot": [(0.0, 3.0)],
                                           "outs_strip": [(0.0, 3.0)]})
    # the source shows only an on-screen word in the middle of the frame: no HUD in the band
    monkeypatch.setattr(recompose, "_ocr",
                        lambda src, a, size: [(600, 350, 680, 370, "WELCOME", 0)])
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "auto",
                              "--vertical", "poker"])
    monkeypatch.setattr(recompose.reframe, "analyze", lambda *ar, **kw: {
        "layout": "track", "src_w": W, "src_h": H, "shots": [],
        "segments": [{"t0": 0.0, "t1": 3.0, "x": 400, "cut": True}]})
    assert recompose.plan_layout(a, Path("s.mp4"))["layout"] == "track"
    monkeypatch.setattr(recompose, "_ocr", hud_text_ocr(POKER_PROFILE, W, H))
    called = {}

    def fake_analyze(src, start, dur, force=None, cw=None):
        called["force"] = force
        called["cw"] = cw
        return {"layout": force or "general", "src_w": W, "src_h": H, "shots": [],
                "segments": [{"t0": 0.0, "t1": 3.0, "x": 400, "cut": True}]}
    monkeypatch.setattr(recompose.reframe, "analyze", fake_analyze)
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "auto",
                              "--vertical", "poker"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    assert called["force"] == "hudband" and plan["layout"] == "hudband"
    assert plan["band_height"] > 0 and plan["band_items"]


def test_hudband_plan_top_crop_starts_below_outs_strip(monkeypatch):
    """y_off = outs_strip bottom edge in source pixels; eff_h = src_h - y_off; cw preserves aspect."""
    monkeypatch.setattr(recompose.reframe, "probe_size", lambda src: (W, H))
    monkeypatch.setattr(recompose, "_hud_profile", lambda vertical: dict(POKER_PROFILE))
    monkeypatch.setattr(recompose, "_region_presence",
                        lambda *ar, **kw: {"players": [(0.0, 3.0)], "pot": [(0.0, 3.0)],
                                           "outs_strip": [(0.0, 3.0)]})
    monkeypatch.setattr(recompose, "_ocr", hud_text_ocr(POKER_PROFILE, W, H))
    monkeypatch.setattr(recompose.reframe, "analyze",
                        lambda *ar, **kw: {"layout": "hudband", "src_w": W, "src_h": H,
                                            "shots": [],
                                            "segments": [{"t0": 0.0, "t1": 3.0, "x": 400,
                                                           "cut": True}]})
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "hudband"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    _, sy, _, sh = hud.region_pixels(POKER_PROFILE["outs_strip"], W, H)
    assert plan["y_off"] == sy + sh
    assert plan["eff_h"] == H - plan["y_off"]
    top_h = 1920 - plan["band_height"]
    assert plan["cw"] == reframe.hudband_crop_width(W, H, top_h,
                                                     effective_src_h=plan["eff_h"])


def test_hudband_plan_crop_never_slices_regions(monkeypatch):
    """The crop is either fully outside or fully inside every {players, pot} region."""
    monkeypatch.setattr(recompose.reframe, "probe_size", lambda src: (W, H))
    monkeypatch.setattr(recompose, "_hud_profile", lambda vertical: dict(POKER_PROFILE))
    monkeypatch.setattr(recompose, "_region_presence",
                        lambda *ar, **kw: {"players": [(0.0, 3.0)], "pot": [(0.0, 3.0)],
                                           "outs_strip": [(0.0, 3.0)]})
    monkeypatch.setattr(recompose, "_ocr", hud_text_ocr(POKER_PROFILE, W, H))

    def fake_analyze(src, start, dur, force=None, cw=None):
        return {"layout": "hudband", "src_w": W, "src_h": H, "shots": [],
                "segments": [{"t0": 0.0, "t1": 3.0, "x": 50, "cut": True}]}
    monkeypatch.setattr(recompose.reframe, "analyze", fake_analyze)
    a = recompose.parse_args(["s.mp4", "0", "3", "o.mp4", "--layout", "hudband"])
    plan = recompose.plan_layout(a, Path("s.mp4"))
    ranges = []
    for name in ("players", "pot"):
        rx, _, rw, _ = hud.region_pixels(POKER_PROFILE[name], W, H)
        ranges.append((rx, rx + rw))
    for seg in plan["segments"]:
        for r in ranges:
            assert not hud.crop_slices_region(seg["x"], plan["cw"], r)


def test_hudband_full_render_is_1080x1920(media, tmp_path, monkeypatch, capsys):
    """End-to-end hudband render at real ffmpeg — synthetic media, mocked profile + presence."""
    monkeypatch.setattr(recompose.reframe, "probe_size", lambda src: (1280, 720))
    monkeypatch.setattr(recompose, "_hud_profile", lambda vertical: dict(POKER_PROFILE))
    monkeypatch.setattr(recompose, "_region_presence",
                        lambda *ar, **kw: {"players": [(0.0, 2.0)], "pot": [(0.0, 2.0)],
                                           "outs_strip": [(0.0, 2.0)]})
    monkeypatch.setattr(recompose, "_blank_regions", lambda *ar, **kw: [])   # HUD text mocked too

    def fake_analyze(src, start, dur, force=None, cw=None):
        s = samples(6, lambda i: [face(0.5)])
        segs = reframe.track_crops(s, [], 2.0, 1280, 720,
                                    cw=cw or reframe.crop_width(1280, 720))
        return {"layout": "hudband", "src_w": 1280, "src_h": 720, "shots": [], "segments": segs}
    monkeypatch.setattr(recompose.reframe, "analyze", fake_analyze)
    out = tmp_path / "hud.mp4"
    rc = recompose.main([str(media), "0", "2", str(out), "--layout", "hudband",
                         "--loudnorm", "-14"])
    monkeypatch.undo()
    assert rc == 0
    info = ffprobe(out)
    assert (info["width"], info["height"]) == (1080, 1920) and info["has_audio"]
    rep = json.loads(capsys.readouterr().out.strip().splitlines()[-1][3:])
    assert rep["layout"] == "hudband" and rep["passed"]


def test_hudband_caption_position_above_seam(tmp_path):
    # band height 480 -> top region 1440; captions must sit above 1440.
    top_h = 1440
    ass = captions.build_ass(random_words(20), tmp_path / "c.ass", layout="hudband", fg_h=top_h)
    text = Path(ass).read_text()
    ys = [int(y) for y in re.findall(r"\\an5\\pos\(540,(\d+)\)", text)]
    assert ys and all(y < top_h for y in ys)


def test_build_ass_cli_wraps_captions(tmp_path):
    wj = tmp_path / "whisper.json"
    wj.write_text(json.dumps({"segments": [{"words": [
        {"word": " Hello", "start": 0.0, "end": 0.4}, {"word": " there.", "start": 0.4, "end": 0.8}]}]}))
    out = tmp_path / "o.ass"
    r = subprocess.run([sys.executable, str(RENDER / "build_ass.py"), str(wj), str(out), "opus"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    evs = ass_events(out, "Caption")
    assert len(evs) == 2 and plain(evs[0]["text"]) == "Hello there."


# ---------------------------------------------------------------- chart bbox detector

def _make_scan_frame(width, height, candle_cols, candle_y, candle_h, bg=(5, 5, 5)):
    """Build a synthetic 'chart' frame: dark bg + colored candle columns at the given
    x positions. Candles are alternating green/red. Returns a BGR uint8 frame."""
    f = np.full((height, width, 3), bg, dtype=np.uint8)
    for i, x in enumerate(candle_cols):
        c = (0, 200, 0) if (i % 2 == 0) else (0, 0, 200)
        f[candle_y:candle_y + candle_h, x:x + 2] = c
    return f


def _mk_iter_frames_mock(samples, width_arg):
    """Return a function matching reframe.iter_frames' signature that yields the given
    (t, frame) list, honoring the ``width`` arg so the function can request a smaller
    scan width than the samples."""
    def fake(src, start, dur, fps=10.0, width=None, src_size=None):
        for t, frame in samples:
            yield (float(t), frame)
    return fake


def test_detect_chart_bbox_rejects_source_that_goes_black(monkeypatch, tmp_path):
    """if ANY sampled frame has >= 90 % near-black pixels on the brightest
    channel, detect_chart_content_bbox returns a sentinel dict (not a bbox) so the
    planner can refuse the pick."""
    import numpy as np
    import importlib
    rc = importlib.reload(recompose)      # noqa: F841
    from clipengine.render import recompose as _rc                              # noqa: WPS433
    monkeypatch.setattr(_rc.reframe, "probe_size", lambda s: (1920, 1080))
    # 10 samples: 7 valid chart frames, 3 all-black frames at the tail
    candle_cols = list(range(40, 300, 10))               # 26 candle columns
    frames = []
    for t in range(7):
        f = _make_scan_frame(320, 180, candle_cols, 60, 60)
        frames.append((float(t), f))
    for t in range(7, 10):
        frames.append((float(t), np.zeros((180, 320, 3), dtype=np.uint8)))
    monkeypatch.setattr(_rc.reframe, "iter_frames", _mk_iter_frames_mock(frames, 320))
    out = _rc.detect_chart_content_bbox("/tmp/missing.mp4", 0.0, 10.0, sample_fps=1.0)
    assert isinstance(out, dict), out
    assert out.get("error") == "source_chart_went_black"
    assert len(out.get("black_at_s") or []) >= 3


def test_detect_chart_bbox_requires_minimum_20_candles(monkeypatch):
    """a thin-flat chart with < 20 distinct candle columns returns None so the
    planner falls back to broadcast cover-scale instead of zooming into a 6-candle
    strip."""
    import numpy as np
    import importlib
    rc = importlib.reload(recompose)      # noqa: F841
    from clipengine.render import recompose as _rc                              # noqa: WPS433
    monkeypatch.setattr(_rc.reframe, "probe_size", lambda s: (1920, 1080))
    # Only 6 candle columns at wide spacing — like a flat-line chart
    candle_cols = [60, 100, 140, 180, 220, 260]
    frames = []
    for t in range(5):
        f = _make_scan_frame(320, 180, candle_cols, 70, 40)
        # jiggle one column slightly to ensure temporal variance
        off = (t * 2) % 20
        f = _make_scan_frame(320, 180, [c + off for c in candle_cols], 70, 40)
        frames.append((float(t), f))
    monkeypatch.setattr(_rc.reframe, "iter_frames", _mk_iter_frames_mock(frames, 320))
    out = _rc.detect_chart_content_bbox("/tmp/missing.mp4", 0.0, 5.0, sample_fps=1.0)
    assert out is None, out


def test_detect_chart_bbox_accepts_dense_candles(monkeypatch):
    """a readable chart with >= 20 candle columns returns a bbox in source
    pixels, not None and not a sentinel."""
    import numpy as np
    import importlib
    rc = importlib.reload(recompose)      # noqa: F841
    from clipengine.render import recompose as _rc                              # noqa: WPS433
    monkeypatch.setattr(_rc.reframe, "probe_size", lambda s: (1920, 1080))
    candle_cols = list(range(30, 310, 10))               # 28 candle columns
    frames = []
    for t in range(10):
        off = (t * 3) % 15
        f = _make_scan_frame(320, 180, [c + off for c in candle_cols], 50, 80)
        frames.append((float(t), f))
    monkeypatch.setattr(_rc.reframe, "iter_frames", _mk_iter_frames_mock(frames, 320))
    out = _rc.detect_chart_content_bbox("/tmp/missing.mp4", 0.0, 10.0, sample_fps=1.0)
    assert isinstance(out, list) and len(out) == 4, out
    x, y, w, h = out
    assert w > 0 and h > 0 and x + w <= 1920 and y + h <= 1080


def test_chart_vf_fully_tiles_1080x1920_panels():
    """Constraint: top (1312) + bottom (608) must sum to the full output height
    1920 so no empty band appears between the chart and the cam."""
    from clipengine.render import recompose as _rc                              # noqa: WPS433
    from clipengine.render import reframe as _rf
    assert _rf.BROADCAST_TOP_H + (_rc.OUT_H - _rf.BROADCAST_TOP_H) == _rc.OUT_H == 1920
    assert _rc.OUT_W == 1080
