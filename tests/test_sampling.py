"""Tests for clipengine/sampling.py (phase 2 item 4)."""
import pathlib
from clipengine import sampling


def test_build_contact_sheet_calls_ffmpeg_with_1fps_filter(tmp_path):
    seen = {}

    def _runner(cmd):
        seen["cmd"] = cmd
        pathlib.Path(cmd[-1]).write_bytes(b"\xff\xd8" + b"\x00" * 128)  # jpeg-ish
        return 0

    out = tmp_path / "sheet.jpg"
    ok = sampling.build_contact_sheet("/tmp/x.mp4", out, runner=_runner)
    assert ok
    cmd = seen["cmd"]
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("fps=1.0,"), vf
    assert "tile=" in vf


def test_sheet_name_encodes_verdict():
    n = sampling.sheet_name({"clip_id": 42, "source_id": "abc"}, verdict="pass")
    assert "c42_abc" in n and n.endswith("_pass.jpg")


def test_review_dir_slugs_account(tmp_path):
    d = sampling.review_dir(tmp_path, "2026-10-14", "@sample.clips.tv")
    assert d.name == "sample_clips_tv"
    assert d.exists()


# ---- the 3x2 review grid shows 6 real frames, never black cells ----

import shutil
import subprocess

import pytest

_HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _synthetic_clip(path, dur_s, w=270, h=480):
    """Moving colour test pattern (no black frames anywhere), 9:16 like a rendered clip."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", f"testsrc2=size={w}x{h}:rate=30:duration={dur_s}",
                    "-pix_fmt", "yuv420p", str(path)], check=True)


def _tile_means(sheet_path, cols, rows):
    import cv2
    img = cv2.imread(str(sheet_path))
    assert img is not None, "sheet not readable"
    th, tw = img.shape[0] // rows, img.shape[1] // cols
    return [float(img[r * th:(r + 1) * th, c * tw:(c + 1) * tw].mean())
            for r in range(rows) for c in range(cols)]


def test_sheet_times_are_evenly_spaced_and_inside_the_clip():
    ts = sampling.sheet_times(24.0, 6)
    assert ts == [2.0, 6.0, 10.0, 14.0, 18.0, 22.0]
    assert all(t <= 2.0 - sampling.GRID_TAIL_S for t in sampling.sheet_times(2.0, 6))


def test_grid_cmd_seeks_one_frame_per_tile():
    cmd = sampling.grid_sheet_cmd("/x.mp4", "/o.jpg", [1, 2, 3, 4, 5, 6])
    assert cmd.count("-i") == 6 and cmd.count("-ss") == 6
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=6" in graph and "tile=3x2" in graph
    assert "thumbnail" not in graph and "select" not in graph


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
@pytest.mark.parametrize("dur_s", [3.0, 23.7])
def test_grid_sheet_has_six_non_black_tiles(tmp_path, dur_s):
    clip = tmp_path / "clip.mp4"
    _synthetic_clip(clip, dur_s)
    out = tmp_path / "sheet.jpg"
    assert sampling.build_grid_sheet(clip, out, n=6, cols=3, width=90)
    import cv2
    img = cv2.imread(str(out))
    assert img.shape[1] == 3 * 90 and img.shape[0] == 2 * 160
    means = _tile_means(out, 3, 2)
    assert len(means) == 6
    assert all(m > 40 for m in means), means      # black padding tiles are ~0


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_go_live_thumbnail_command_left_black_tiles(tmp_path):
    """Regression evidence: the go-live command's `thumbnail=100` fed `tile` one frame."""
    clip = tmp_path / "clip.mp4"
    _synthetic_clip(clip, 23.7)
    out = tmp_path / "old.jpg"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip), "-vf",
                    "select='eq(n,0)+eq(n,round(23.7*5/60*4))+gt(scene,0.4)',thumbnail=100,"
                    "scale=90:-1,tile=3x2", "-frames:v", "1", str(out)], check=True)
    means = _tile_means(out, 3, 2)
    assert sum(1 for m in means if m < 5) >= 1, means


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_filmstrip_caps_columns_for_short_clips(tmp_path):
    clip = tmp_path / "clip.mp4"
    _synthetic_clip(clip, 5.0)
    out = tmp_path / "strip.jpg"
    assert sampling.build_contact_sheet(clip, out, width_per_tile=60)
    import cv2
    img = cv2.imread(str(out))
    cols = img.shape[1] // 60
    assert cols == 5
    assert all(m > 40 for m in _tile_means(out, cols, 1))
