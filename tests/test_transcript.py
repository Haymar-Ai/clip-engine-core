"""transcript.py + scenes.py tests. Offline: no whisper model load, ffmpeg/whisper CLI mocked
(except the opt-in @slow test, which really transcribes a tiny TTS clip with base.en)."""
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PIPE = Path(__file__).resolve().parent.parent / "clipengine"
from clipengine import scenes
from clipengine import transcript


def _words(tokens, t=0.0, dur=0.3, gaps=None):
    """Word list from tokens; gaps[k] = silence after token k (default 0.1)."""
    out = []
    for k, tok in enumerate(tokens):
        out.append({"w": tok, "start": round(t, 3), "end": round(t + dur, 3)})
        t += dur + (gaps or {}).get(k, 0.1)
    return out


def _spans(sents):
    return [(s["w0"], s["w1"]) for s in sents]


def _check_shape(words, sents):
    """i numbered 0..n-1, spans contiguous and covering every word, text/start/end consistent."""
    assert [s["i"] for s in sents] == list(range(len(sents)))
    assert sents[0]["w0"] == 0 and sents[-1]["w1"] == len(words) - 1
    for a, b in zip(sents, sents[1:]):
        assert b["w0"] == a["w1"] + 1
    for s in sents:
        assert s["w0"] <= s["w1"]
        assert s["start"] == words[s["w0"]]["start"] and s["end"] == words[s["w1"]]["end"]
        assert s["text"] == " ".join(w["w"] for w in words[s["w0"]:s["w1"] + 1])


# ---- sentences ----
def test_sentences_punctuation_split():
    words = _words("Hello there. How are you? I am fine! Thanks".split())
    sents = transcript.sentences(words)
    assert _spans(sents) == [(0, 1), (2, 4), (5, 7), (8, 8)]
    assert [s["text"] for s in sents] == ["Hello there.", "How are you?", "I am fine!", "Thanks"]
    _check_shape(words, sents)


def test_sentences_split_through_closing_quotes_and_brackets():
    words = _words(['He', 'said', '"done."', 'Then', '(really?)', 'Yes!\'', 'ok'])
    assert _spans(transcript.sentences(words)) == [(0, 2), (3, 4), (5, 5), (6, 6)]


def test_sentences_no_split_on_honorific():
    words = _words("Dr. Jones said hi. Bye".split())
    assert _spans(transcript.sentences(words)) == [(0, 3), (4, 4)]


def test_sentences_gap_split():
    words = _words("one two three four five six".split(), gaps={2: 1.0})
    sents = transcript.sentences(words)
    assert _spans(sents) == [(0, 2), (3, 5)]
    _check_shape(words, sents)
    # boundary: a gap of exactly gap_s splits, just under does not (exact binary floats)
    ws = [{"w": "a", "start": 0.0, "end": 1.0}, {"w": "b", "start": 1.5, "end": 2.0},
          {"w": "c", "start": 2.25, "end": 3.0}]
    assert _spans(transcript.sentences(ws, gap_s=0.5)) == [(0, 0), (1, 2)]
    assert _spans(transcript.sentences(ws, gap_s=0.75)) == [(0, 2)]


def test_sentences_max_words_splits_at_largest_internal_gap():
    words = _words([f"w{k}" for k in range(12)], gaps={3: 0.3, 7: 0.5})
    sents = transcript.sentences(words, gap_s=0.7, max_words=10)
    assert _spans(sents) == [(0, 7), (8, 11)]          # the 0.5 s gap after w7, not the 0.3 one
    _check_shape(words, sents)
    # no split needed when within the limit
    assert _spans(transcript.sentences(words, max_words=12)) == [(0, 11)]


def test_sentences_max_words_split_is_recursive():
    words = _words([f"w{k}" for k in range(25)], gaps={5: 0.4, 14: 0.5, 20: 0.3})
    sents = transcript.sentences(words, gap_s=0.7, max_words=10)
    # 25 -> split at the 0.5 gap (after w14) -> 15 + 10; the 15 splits at its 0.4 gap (after w5)
    assert _spans(sents) == [(0, 5), (6, 14), (15, 24)]
    _check_shape(words, sents)


def test_sentences_gapless_run_splits_evenly_without_recursion_blowup():
    words = [{"w": f"w{k}", "start": k * 0.3, "end": k * 0.3 + 0.3} for k in range(5000)]
    sents = transcript.sentences(words, max_words=40)
    _check_shape(words, sents)
    assert all(s["w1"] - s["w0"] + 1 <= 40 for s in sents)
    assert min(s["w1"] - s["w0"] + 1 for s in sents) >= 10   # ties split near the middle


def test_sentences_empty_and_single():
    assert transcript.sentences([]) == []
    s = transcript.sentences([{"w": "hi", "start": 1.0, "end": 1.2}])
    assert s == [{"i": 0, "start": 1.0, "end": 1.2, "text": "hi", "w0": 0, "w1": 0}]


# ---- get_words cache ----
WORDS = [{"w": "Hello", "start": 0.5, "end": 0.9}, {"w": "world.", "start": 1.0, "end": 1.4}]


def test_get_words_caches_and_does_not_retranscribe(tmp_path):
    media = tmp_path / "ep.mp4"
    media.write_bytes(b"x")
    cache = tmp_path / "cache"
    with patch.object(transcript, "_transcribe", return_value=list(WORDS)) as tr:
        a = transcript.get_words(str(media), cache_dir=cache)
        b = transcript.get_words(str(media), cache_dir=cache)
    assert a == b == WORDS
    tr.assert_called_once_with(str(media), "base.en")
    key = hashlib.md5(f"{media}|{os.path.getmtime(media)}|base.en".encode()).hexdigest()
    assert json.loads((cache / f"{key}.json").read_text()) == WORDS
    assert not list(cache.glob("*.tmp"))


def test_get_words_cache_key_includes_model_and_mtime(tmp_path):
    media = tmp_path / "ep.mp4"
    media.write_bytes(b"x")
    with patch.object(transcript, "_transcribe", return_value=list(WORDS)) as tr:
        transcript.get_words(media, cache_dir=tmp_path)
        transcript.get_words(media, model="small.en", cache_dir=tmp_path)
        assert tr.call_count == 2
        st = os.stat(media)
        os.utime(media, (st.st_atime, st.st_mtime + 10))
        transcript.get_words(media, cache_dir=tmp_path)
        assert tr.call_count == 3
        transcript.get_words(media, cache_dir=tmp_path)
        assert tr.call_count == 3


def test_get_words_default_cache_dir_under_scratch(tmp_path, monkeypatch):
    media = tmp_path / "ep.mp4"
    media.write_bytes(b"x")
    monkeypatch.setattr(transcript, "SCRATCH", tmp_path / "scratch")
    with patch.object(transcript, "_transcribe", return_value=list(WORDS)):
        transcript.get_words(media)
    assert len(list((tmp_path / "scratch" / "transcripts").glob("*.json"))) == 1


def test_get_words_recovers_from_corrupt_cache(tmp_path):
    media = tmp_path / "ep.mp4"
    media.write_bytes(b"x")
    key = hashlib.md5(f"{media}|{os.path.getmtime(media)}|base.en".encode()).hexdigest()
    (tmp_path / f"{key}.json").write_text("{not json")
    with patch.object(transcript, "_transcribe", return_value=list(WORDS)) as tr:
        assert transcript.get_words(media, cache_dir=tmp_path) == WORDS
    tr.assert_called_once()


# ---- faster-whisper path (fake module, no model load) ----
def test_faster_whisper_path_args_strip_and_model_cache(monkeypatch):
    W = types.SimpleNamespace
    seg = W(words=[W(word=" Hello", start=0.1, end=0.5), W(word="  ", start=0.5, end=0.6),
                   W(word=" world.", start=0.6, end=1.0)])
    calls = []

    class FakeModel:
        def __init__(self, name, **kw):
            calls.append(("init", name, kw))

        def transcribe(self, path, **kw):
            calls.append(("transcribe", path, kw))
            return iter([seg, W(words=None)]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeModel))
    monkeypatch.setattr(transcript, "_MODELS", {})
    for _ in range(2):
        words = transcript._transcribe("/x/a.wav", "base.en")
        assert words == [{"w": "Hello", "start": 0.1, "end": 0.5},
                         {"w": "world.", "start": 0.6, "end": 1.0}]
    inits = [c for c in calls if c[0] == "init"]
    assert inits == [("init", "base.en", {"device": "cpu", "compute_type": "int8",
                                          "cpu_threads": transcript.THREADS})]
    assert 1 <= transcript.THREADS <= 4
    tr = [c for c in calls if c[0] == "transcribe"]
    assert len(tr) == 2 and tr[0][2] == {"language": "en", "word_timestamps": True,
                                         "vad_filter": True, "condition_on_previous_text": False}


# ---- CLI fallback when faster_whisper is missing ----
def _fake_whisper_cli(cmd, **kw):
    out = Path(cmd[cmd.index("--output_dir") + 1])
    (out / (Path(cmd[1]).stem + ".json")).write_text(json.dumps({"segments": [
        {"text": " Hello world.", "words": [
            {"word": " Hello", "start": 0.0, "end": 0.4, "probability": 0.9},
            {"word": " ", "start": 0.4, "end": 0.5},
            {"word": " world.", "start": 0.5, "end": 0.9}]},
        {"text": " Bye", "words": [{"word": " Bye", "start": 2.0, "end": 2.3}]}]}))
    return subprocess.CompletedProcess(cmd, 0)


def test_imports_without_faster_whisper_and_falls_back_to_cli(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", None)
    mod = importlib.reload(transcript)
    media = tmp_path / "ep.mp4"
    media.write_bytes(b"x")
    with patch.object(mod.subprocess, "run", side_effect=_fake_whisper_cli) as run:
        words = mod.get_words(media, cache_dir=tmp_path / "c")
        again = mod.get_words(media, cache_dir=tmp_path / "c")
    assert words == again == [{"w": "Hello", "start": 0.0, "end": 0.4},
                              {"w": "world.", "start": 0.5, "end": 0.9},
                              {"w": "Bye", "start": 2.0, "end": 2.3}]
    run.assert_called_once()
    cmd = run.call_args[0][0]
    assert cmd[:2] == ["whisper", str(media)]
    for flag, val in [("--model", "base.en"), ("--language", "en"), ("--word_timestamps", "True"),
                      ("--output_format", "json")]:
        assert cmd[cmd.index(flag) + 1] == val
    assert not Path(cmd[cmd.index("--output_dir") + 1]).exists()      # temp dir cleaned up
    assert mod.speech_regions(np.zeros(16000, np.float32), 16000, 5.0) == []


# ---- words_window ----
def test_words_window_offsets_to_absolute_times(tmp_path):
    seen = {}

    def fake_tr(path, model):
        seen["path"], seen["model"] = Path(path), model
        return [{"w": "So", "start": 0.25, "end": 0.5}, {"w": "yes.", "start": 1.0, "end": 1.5}]

    with patch.object(transcript.subprocess, "run") as run, \
            patch.object(transcript, "_transcribe", side_effect=fake_tr):
        words = transcript.words_window("/m/ep.mp4", 100.0, 130.0)
    assert words == [{"w": "So", "start": 100.25, "end": 100.5},
                     {"w": "yes.", "start": 101.0, "end": 101.5}]
    cmd = run.call_args[0][0]
    assert cmd[0] == "ffmpeg" and cmd[cmd.index("-i") + 1] == "/m/ep.mp4"
    assert float(cmd[cmd.index("-ss") + 1]) == 100.0 and float(cmd[cmd.index("-t") + 1]) == 30.0
    assert cmd[cmd.index("-ar") + 1] == "16000" and cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[-1] == str(seen["path"]) and seen["path"].suffix == ".wav"
    assert seen["model"] == "small.en"
    assert not seen["path"].parent.exists()                            # temp wav cleaned up


def test_words_window_clamps_negative_start():
    with patch.object(transcript.subprocess, "run") as run, \
            patch.object(transcript, "_transcribe",
                         return_value=[{"w": "Hi.", "start": 0.5, "end": 0.75}]):
        words = transcript.words_window("/m/ep.mp4", -8.0, 12.0)
    cmd = run.call_args[0][0]
    assert float(cmd[cmd.index("-ss") + 1]) == 0.0 and float(cmd[cmd.index("-t") + 1]) == 12.0
    assert words == [{"w": "Hi.", "start": 0.5, "end": 0.75}]


# ---- load_audio / speech_regions ----
def test_load_audio_reads_f32le_pipe():
    pcm = np.array([0.0, 0.5, -0.25, 1.0], np.float32)
    with patch.object(transcript.subprocess, "run",
                      return_value=subprocess.CompletedProcess([], 0, stdout=pcm.tobytes())) as run:
        a = transcript.load_audio("/m/ep.mp4", 10.0, 12.5)
    assert a.dtype == np.float32 and np.array_equal(a, pcm) and a.flags.writeable
    cmd = run.call_args[0][0]
    assert float(cmd[cmd.index("-ss") + 1]) == 10.0 and float(cmd[cmd.index("-t") + 1]) == 2.5
    assert cmd[cmd.index("-f") + 1] == "f32le" and cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[cmd.index("-ac") + 1] == "1" and "-vn" in cmd


def test_load_audio_negative_start_zero_pads_so_t0_stays_lo():
    pcm = np.ones(8, np.float32)
    with patch.object(transcript.subprocess, "run",
                      return_value=subprocess.CompletedProcess([], 0, stdout=pcm.tobytes())) as run:
        a = transcript.load_audio("/m/ep.mp4", -0.5, 1.0, sr=10)
    assert float(run.call_args[0][0][run.call_args[0][0].index("-ss") + 1]) == 0.0
    assert len(a) == 13 and not a[:5].any() and a[5:].all()


def _fake_vad(monkeypatch, spans):
    """faster_whisper.vad (an optional package) as a stub module whose Silero VAD returns
    `spans` (sample indexes)."""
    import types
    fake = MagicMock(return_value=spans)
    vad = types.ModuleType("faster_whisper.vad")
    vad.get_speech_timestamps = fake
    vad.VadOptions = lambda **kw: kw
    pkg = types.ModuleType("faster_whisper")
    pkg.vad = vad
    monkeypatch.setitem(sys.modules, "faster_whisper", pkg)
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    return fake


def test_speech_regions_converts_samples_to_absolute_seconds(monkeypatch):
    fake = _fake_vad(monkeypatch, [{"start": 16000, "end": 40000}, {"start": 48000, "end": 64000}])
    r = transcript.speech_regions(np.zeros(64000, np.float32), 16000, 100.0)
    assert r == [(101.0, 102.5), (103.0, 104.0)]
    assert fake.call_args.kwargs["sampling_rate"] == 16000
    assert transcript.speech_regions(np.zeros(0, np.float32), 16000, 0.0) == []


def test_speech_regions_without_faster_whisper_is_empty(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)       # import raises ImportError
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", None)
    assert transcript.speech_regions(np.ones(16000, np.float32), 16000, 0.0) == []


# ---- scenes ----
SHOWINFO = """[Parsed_showinfo_2 @ 0x1] config in time_base: 1/12800, frame_rate: 25/1
[Parsed_showinfo_2 @ 0x1] n:   0 pts:  19200 pts_time:1.5     duration:    512 fmt:yuv420p
[Parsed_showinfo_2 @ 0x1] n:   1 pts:  41600 pts_time:3.25    duration:    512 fmt:yuv420p
"""


@pytest.fixture()
def scene_cache(monkeypatch):
    monkeypatch.setattr(scenes, "_CACHE", {})


def test_scene_cuts_parses_showinfo_to_absolute_and_caches(scene_cache):
    ok = subprocess.CompletedProcess([], 0, stdout="", stderr=SHOWINFO)
    with patch.object(scenes.subprocess, "run", return_value=ok) as run:
        a = scenes.scene_cuts("/m/ep.mp4", 100.0, 110.0)
        b = scenes.scene_cuts("/m/ep.mp4", 100.0, 110.0)
    assert a == b == [101.5, 103.25]
    run.assert_called_once()
    cmd = run.call_args[0][0]
    assert cmd[0] == "ffmpeg" and cmd[cmd.index("-i") + 1] == "/m/ep.mp4"
    assert float(cmd[cmd.index("-ss") + 1]) == 100.0 and float(cmd[cmd.index("-to") + 1]) == 110.0
    assert cmd[cmd.index("-vf") + 1] == "scale=320:-2,select='gt(scene,0.3)',showinfo"
    assert cmd[-3:] == ["-f", "null", "-"] and "-an" in cmd
    with patch.object(scenes.subprocess, "run", return_value=ok) as run:
        scenes.scene_cuts("/m/ep.mp4", 100.0, 110.0, threshold=0.5)   # different key
    assert "gt(scene,0.5)" in run.call_args[0][0][run.call_args[0][0].index("-vf") + 1]


def test_scene_cuts_failure_returns_empty_and_is_not_cached(scene_cache):
    bad = subprocess.CompletedProcess([], 1, stdout="", stderr="No such file")
    with patch.object(scenes.subprocess, "run", return_value=bad) as run:
        assert scenes.scene_cuts("/m/missing.mp4", 0.0, 5.0) == []
        assert scenes.scene_cuts("/m/missing.mp4", 0.0, 5.0) == []
    assert run.call_count == 2
    with patch.object(scenes.subprocess, "run", side_effect=FileNotFoundError("ffmpeg")):
        assert scenes.scene_cuts("/m/ep.mp4", 0.0, 5.0) == []


# ---- opt-in: real base.en transcription of a tiny TTS clip (faster-whisper or the CLI) ----
@pytest.mark.slow
def test_real_transcription_tiny_clip(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    try:
        import faster_whisper  # noqa: F401 - either backend transcribes
    except ImportError:
        assert shutil.which("whisper"), "neither faster-whisper nor the whisper CLI is installed"
    wav = tmp_path / "speech.wav"
    r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "flite=text='Hello world. This is a short test of the transcriber.'",
                        "-ar", "16000", "-ac", "1", str(wav)], capture_output=True)
    if r.returncode != 0:
        pytest.skip("ffmpeg built without flite")
    words = transcript.get_words(wav, "base.en", cache_dir=tmp_path / "c")
    assert words and all(w["w"] and w["w"] == w["w"].strip() and w["end"] >= w["start"]
                         for w in words)
    assert "hello" in " ".join(w["w"].lower() for w in words)
    assert transcript.get_words(wav, "base.en", cache_dir=tmp_path / "c") == words
    sents = transcript.sentences(words)
    assert sents and sents[0]["i"] == 0
    win = transcript.words_window(wav, 1.0, 4.0, model="base.en")
    assert win and all(w["start"] >= 1.0 for w in win)


# ---- the whisper CLI call is capped and thread-limited ----
def test_cli_call_gets_a_timeout_and_thread_cap(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", None)
    mod = importlib.reload(transcript)
    monkeypatch.setattr(mod, "_media_seconds", lambda path: 37.0)
    seen = {}

    def fake(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        return _fake_whisper_cli(cmd, **kw)
    with patch.object(mod.subprocess, "run", side_effect=fake):
        words = mod._transcribe(tmp_path / "win.wav", "small.en")
    assert words[0]["w"] == "Hello"
    assert seen["kw"]["timeout"] == mod.cli_timeout(37.0) == 120.0 + 6.0 * 37.0
    assert seen["cmd"][seen["cmd"].index("--threads") + 1] == str(mod.THREADS)
    env = seen["kw"]["env"]
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == str(mod.THREADS)
    assert env["OMP_WAIT_POLICY"] == "PASSIVE" and env["PATH"] == os.environ["PATH"]
    assert mod.cli_timeout(None) is None
    # a runner's window deadline caps every call too
    monkeypatch.setattr(mod, "CLI_DEADLINE", mod.time.monotonic() + 50.0)
    assert 45.0 < mod.cli_timeout(37.0) <= 50.0 and 45.0 < mod.cli_timeout(None) <= 50.0
    monkeypatch.setattr(mod, "CLI_DEADLINE", mod.time.monotonic() - 10.0)
    assert mod.cli_timeout(37.0) == 1.0


def test_cli_timeout_propagates_to_the_caller(tmp_path, monkeypatch):
    """A stuck CLI raises TimeoutExpired instead of hanging; words_window callers (precision
    pass, edge-verify) already treat a raise as 'keep the base cuts' / 'no words'."""
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", None)
    mod = importlib.reload(transcript)
    monkeypatch.setattr(mod, "_media_seconds", lambda path: 5.0)

    def stuck(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
    with patch.object(mod.subprocess, "run", side_effect=stuck):
        with pytest.raises(subprocess.TimeoutExpired):
            mod._transcribe(tmp_path / "win.wav", "small.en")
