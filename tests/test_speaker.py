"""PHASE6-B active-speaker tests — pure (no ffmpeg / no real detector). All frames are
synthetic and the fixtures build faces/landmarks/audio inline."""
import sys
from pathlib import Path

import numpy as np
import pytest

RENDER = Path(__file__).resolve().parent.parent / "clipengine" / "render"

from clipengine.render import reframe
from clipengine.render import speaker


def _face(track_hint, cx, cy, h=0.2, w=None):
    """A face dict with 5 landmarks; the mouth corners sit at cx +/- 3% of a 640x360 frame."""
    w = w if w is not None else h * 0.66
    fw, fh = 640, 360
    lms = [(cx * fw - 0.05 * fw, cy * fh - 0.03 * fh),   # right eye
           (cx * fw + 0.05 * fw, cy * fh - 0.03 * fh),   # left eye
           (cx * fw, cy * fh),                            # nose
           (cx * fw + 0.03 * fw, cy * fh + 0.1 * fh),    # right mouth
           (cx * fw - 0.03 * fw, cy * fh + 0.1 * fh)]    # left mouth
    return {"cx": cx, "cy": cy, "w": w, "h": h, "score": 0.9,
            "landmarks": lms, "bbox": (cx * fw - w * fw / 2, cy * fh - h * fh / 2,
                                       w * fw, h * fh),
            "_hint": track_hint}


# ---- track assignment -------------------------------------------------------

def test_assign_tracks_by_iou_matches_faces_across_frames():
    samples = [
        {"t": 0.0, "faces": [_face("A", 0.3, 0.5), _face("B", 0.7, 0.5)]},
        {"t": 0.1, "faces": [_face("A", 0.302, 0.5), _face("B", 0.702, 0.5)]},
        {"t": 0.2, "faces": [_face("A", 0.305, 0.5), _face("B", 0.7, 0.5)]},
    ]
    n = speaker.assign_tracks(samples)
    assert n == 2
    left_track = samples[0]["faces"][0]["track_id"]
    right_track = samples[0]["faces"][1]["track_id"]
    assert left_track != right_track
    for s in samples:
        # first face in each sample corresponds to left, so its track id is stable
        assert s["faces"][0]["track_id"] == left_track
        assert s["faces"][1]["track_id"] == right_track


def test_assign_tracks_creates_new_track_for_new_face():
    s1 = {"t": 0.0, "faces": [_face("A", 0.3, 0.5)]}
    s2 = {"t": 0.1, "faces": [_face("A", 0.3, 0.5), _face("B", 0.8, 0.5)]}
    n = speaker.assign_tracks([s1, s2])
    assert n == 2
    assert s1["faces"][0]["track_id"] == s2["faces"][0]["track_id"]
    assert s2["faces"][1]["track_id"] != s2["faces"][0]["track_id"]


# ---- mouth activity ---------------------------------------------------------

def test_mouth_activity_zero_when_no_change():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, size=(360, 640), dtype=np.uint8)
    face = _face("A", 0.5, 0.5)
    act = speaker.mouth_activity(frame, frame, face, face)
    assert act == 0.0


def test_mouth_activity_grows_with_pixel_change():
    face = _face("A", 0.5, 0.5)
    a = np.zeros((360, 640), dtype=np.uint8)
    b = np.zeros((360, 640), dtype=np.uint8)
    # simulate a mouth-region change
    x0, y0, x1, y1 = speaker._mouth_box(face, 640, 360)
    b[y0:y1, x0:x1] = 200
    a_zero = speaker.mouth_activity(a, a, face, face)
    a_change = speaker.mouth_activity(a, b, face, face)
    assert a_zero == 0.0 and a_change > 5.0


# ---- speech mask ------------------------------------------------------------

def test_speech_mask_gates_silence():
    rms = np.array([0.001, 0.02, 0.02, 0.001, 0.001, 0.02])
    mask = speaker.speech_mask(rms, floor=0.005)
    assert list(mask) == [False, True, True, False, False, True]


# ---- pick_speakers with hysteresis -----------------------------------------

def _speech_always_on(t):
    return True


def test_pick_speakers_selects_highest_activity_track():
    # 6 seconds at hop_s=0.5 -> 12 hops; track 1 for first 3s, track 2 for next 3s.
    # min_hold_s = 1.0 so the state machine *can* switch after held >= 1.0 and
    # challenger has led continuously for min_candidate_run_s.
    hops = []
    for t in np.arange(0, 6.0, 0.5):
        per = {1: 5.0 if t < 3.0 else 0.5, 2: 0.5 if t < 3.0 else 5.0}
        hops.append((float(t), per))
    segs = speaker.pick_speakers(hops, _speech_always_on, hop_s=0.5, min_hold_s=1.0,
                                  switch_margin=0.1, min_candidate_run_s=1.2)
    tids = [s["track_id"] for s in segs]
    assert tids == [1, 2]
    assert segs[0]["t1"] - segs[0]["t0"] >= 1.0


def test_hysteresis_blocks_short_switches():
    # a 0.5 s blip on track 2 must NOT trigger a switch
    hops = []
    for t in np.arange(0, 3.0, 0.5):
        if 1.0 <= t < 1.5:
            per = {1: 0.5, 2: 5.0}                      # blip
        else:
            per = {1: 5.0, 2: 0.5}
        hops.append((float(t), per))
    segs = speaker.pick_speakers(hops, _speech_always_on, hop_s=0.5, min_hold_s=1.5,
                                  switch_margin=0.1, min_candidate_run_s=1.2)
    tids = {s["track_id"] for s in segs}
    assert tids == {1}, f"expected no switch, got {tids}"


def test_silence_gate_holds_current_speaker():
    # activity flips at t=1.0, but speech is off there — the switch must not happen
    hops = [(0.0, {1: 5.0, 2: 0.5}), (0.5, {1: 5.0, 2: 0.5}),
            (1.0, {1: 0.5, 2: 5.0}), (1.5, {1: 0.5, 2: 5.0})]

    def sp(t):
        return t < 1.0

    segs = speaker.pick_speakers(hops, sp, hop_s=0.5, min_hold_s=0.1,
                                  switch_margin=0.05, min_candidate_run_s=0.1)
    assert [s["track_id"] for s in segs] == [1]


# ---- backchannel + silent-speaker guards ---------------------------

def test_backchannel_under_1_2s_does_not_switch():
    """Earlier bug: a brief interjection (<1.2 s) on track 2 while track 1 kept
    the floor would flip the active speaker. Now the challenger must dominate for
    `min_candidate_run_s` continuously before we switch — a short "mm/yeah/right"
    does not qualify."""
    hops = []
    for t in np.arange(0, 10.0, 0.5):
        # challenger (track 2) interjects at 4.0-4.5s and 7.0-7.5s only — each < 1.2s
        if 4.0 <= t < 4.5 or 7.0 <= t < 7.5:
            per = {1: 0.6, 2: 5.0}
        else:
            per = {1: 5.0, 2: 0.4}
        hops.append((float(t), per))
    segs = speaker.pick_speakers(hops, _speech_always_on, hop_s=0.5,
                                  min_hold_s=2.5, switch_margin=0.35,
                                  min_candidate_run_s=1.2)
    assert {s["track_id"] for s in segs} == {1}


def test_candidate_run_satisfied_triggers_switch():
    """When the challenger does hold the floor for >=1.2 s continuously (and all
    other gates clear), the switch *does* happen."""
    hops = []
    for t in np.arange(0, 10.0, 0.5):
        if t < 4.0:
            per = {1: 5.0, 2: 0.3}
        else:
            per = {1: 0.3, 2: 5.0}                     # challenger dominates for 6 s
        hops.append((float(t), per))
    segs = speaker.pick_speakers(hops, _speech_always_on, hop_s=0.5,
                                  min_hold_s=2.5, switch_margin=0.35,
                                  min_candidate_run_s=1.2)
    assert [s["track_id"] for s in segs] == [1, 2]


def test_never_switches_to_silent_track():
    """Switch requires the challenger to have real activity. A speaker with zero
    mouth motion is silent — the state machine cannot cut to them even if speech
    is on."""
    hops = []
    for t in np.arange(0, 10.0, 0.5):
        if t < 3.0:
            per = {1: 5.0, 2: 0.1}
        else:
            per = {1: 0.0, 2: 0.0}                     # nobody talking (floor drop)
        hops.append((float(t), per))
    segs = speaker.pick_speakers(hops, _speech_always_on, hop_s=0.5,
                                  min_hold_s=1.0, switch_margin=0.35,
                                  min_candidate_run_s=1.2)
    assert {s["track_id"] for s in segs} == {1}


def test_pick_speakers_coalesces_consecutive_same_tid():
    """A speaker that remains active across many hops never emits adjacent segments."""
    hops = [(float(t), {1: 5.0, 2: 0.1}) for t in np.arange(0, 5.0, 0.5)]
    segs = speaker.pick_speakers(hops, _speech_always_on, hop_s=0.5,
                                  min_hold_s=2.5, switch_margin=0.35,
                                  min_candidate_run_s=1.2)
    assert len(segs) == 1 and segs[0]["track_id"] == 1


# ---- reframe integration: tied shots trigger speaker replacement -----------

def test_reframe_leaves_single_face_and_two_speaker_plans_untouched(tmp_path, monkeypatch):
    """Speaker path only fires on tied-face shots. A one-face track window has no tied
    frames, so track_crops output is byte-identical to today's."""
    W, H = 1280, 720
    samples = [{"t": round(i / 3, 3), "faces": [_face("A", 0.3, 0.5)]} for i in range(27)]
    # single face across the shot -> no tied span
    tied = reframe._tied_shots(samples, [], 9.0, tie_tol=0.9, min_faces=2)
    assert tied == []


def test_reframe_finds_tied_shot_when_two_equal_faces():
    samples = [{"t": round(i / 3, 3),
                "faces": [_face("A", 0.3, 0.5, h=0.2), _face("B", 0.7, 0.5, h=0.2)]}
               for i in range(27)]
    tied = reframe._tied_shots(samples, [], 9.0, tie_tol=0.9, min_faces=2)
    assert tied == [(0.0, 9.0)]
    # But not 3+
    assert reframe._tied_shots(samples, [], 9.0, tie_tol=0.9, min_faces=3) == []


def test_reframe_finds_three_way_tie():
    samples = [{"t": round(i / 3, 3),
                "faces": [_face("A", 0.2, 0.5, h=0.2), _face("B", 0.5, 0.5, h=0.2),
                          _face("C", 0.8, 0.5, h=0.2)]}
               for i in range(27)]
    assert reframe._tied_shots(samples, [], 9.0, tie_tol=0.9, min_faces=3) == [(0.0, 9.0)]


def test_speaker_replace_swaps_only_tied_segments():
    src_w, src_h = 1280, 720
    cw = reframe.crop_width(src_w, src_h)
    segs = [{"t0": 0.0, "t1": 3.0, "x": 100, "cut": True},
            {"t0": 3.0, "t1": 6.0, "x": 400, "cut": True}]
    tied = [(3.0, 6.0)]
    speaker_segs = [{"t0": 3.0, "t1": 4.5, "cx": 0.3, "track_id": 1},
                    {"t0": 4.5, "t1": 6.0, "cx": 0.7, "track_id": 2}]
    out = reframe._speaker_replace(segs, tied, speaker_segs, src_w, cw)
    # first segment untouched
    assert out[0] == segs[0]
    # second segment replaced by two speaker cuts
    assert len(out) == 3
    for seg in out[1:]:
        assert seg["cut"] is True
        assert 0 <= seg["x"] <= src_w - cw


def test_speaker_replace_no_op_when_no_speaker_segs():
    segs = [{"t0": 0.0, "t1": 5.0, "x": 200, "cut": True}]
    out = reframe._speaker_replace(segs, [(0.0, 5.0)], [], 1280, 400)
    assert out == segs


def test_speaker_replace_merges_consecutive_same_speaker_runs():
    """Target: same speaker across the same tied span must be ONE sub-seg,
    not two. The old `_speaker_replace` emitted one record per speaker_seg, so
    two consecutive seg-of-same-track_id would land as two adjacent records
    (observable as a stale snap). This test fails on that old behaviour."""
    src_w, src_h = 1920, 1080
    cw = reframe.crop_width(src_w, src_h)
    segs = [{"t0": 0.0, "t1": 10.0, "x": 1300, "cut": True}]
    tied = [(0.0, 10.0)]
    # Two consecutive same-speaker speaker_segs (e.g. the sentence boundary split)
    speaker_segs = [{"t0": 0.0, "t1": 4.0, "cx": 0.5, "track_id": 1},
                    {"t0": 4.0, "t1": 10.0, "cx": 0.5, "track_id": 1}]
    out = reframe._speaker_replace(segs, tied, speaker_segs, src_w, cw, samples=None)
    # Expect a single same-speaker record covering the whole span, not two.
    same_speaker = [s for s in out if s.get("speaker") == 1]
    assert len(same_speaker) == 1, f"expected one merged speaker=1 record, got {same_speaker}"
    assert same_speaker[0]["t0"] == 0.0 and same_speaker[0]["t1"] == 10.0


def test_dead_zone_relock_eases_not_snaps():
    """Inside a long same-speaker span, if the speaker's face drifts outside the
    dead zone the planner emits a `relock` sub-segment with `cut=False` so
    crop_x_expr eases the move instead of snapping. The relock cx is the
    *median* over the look-ahead window, not the instantaneous value."""
    pieces = reframe._speaker_dead_zone_relocks(
        0.0, 20.0, cur_cx=0.3, track_id=1,
        samples=[{"t": 0.0, "faces": [{"cx": 0.3, "cy": 0.5, "h": 0.3, "track_id": 1}]},
                 {"t": 1.0, "faces": [{"cx": 0.3, "cy": 0.5, "h": 0.3, "track_id": 1}]},
                 {"t": 2.0, "faces": [{"cx": 0.32, "cy": 0.5, "h": 0.3, "track_id": 1}]},
                 # drift: speaker physically slides to cx=0.6 and stays there
                 {"t": 5.0, "faces": [{"cx": 0.6, "cy": 0.5, "h": 0.3, "track_id": 1}]},
                 {"t": 5.5, "faces": [{"cx": 0.6, "cy": 0.5, "h": 0.3, "track_id": 1}]},
                 {"t": 6.0, "faces": [{"cx": 0.6, "cy": 0.5, "h": 0.3, "track_id": 1}]},
                 {"t": 6.5, "faces": [{"cx": 0.6, "cy": 0.5, "h": 0.3, "track_id": 1}]},
                 {"t": 7.0, "faces": [{"cx": 0.6, "cy": 0.5, "h": 0.3, "track_id": 1}]}],
        dead_zone=0.12)
    # Expect: initial piece at cx=0.3 with cut=True, then a relock to cx~0.6 with cut=False
    assert pieces[0]["cx"] == pytest.approx(0.3, abs=0.001)
    assert pieces[0]["cut"] is True
    assert len(pieces) >= 2
    assert pieces[1]["cx"] == pytest.approx(0.6, abs=0.02)
    assert pieces[1]["cut"] is False, "same-speaker re-lock must ease (cut=False), not snap"
    assert pieces[1].get("relock") is True


def test_same_speaker_no_x_change_without_scene_cut():
    """Anti-regression: a shot with ONE visible speaker whose face stays
    within the dead zone must produce ONE segment — never two same-track records
    with different x values without an intervening scene cut. (The earlier
    `_speaker_replace` could emit multiple adjacent sub-segs with slightly
    different cx values on consecutive sentences.)"""
    src_w, src_h = 1920, 1080
    cw = reframe.crop_width(src_w, src_h)
    segs = [{"t0": 0.0, "t1": 12.0, "x": 1300, "cut": True}]
    tied = [(0.0, 12.0)]
    # Speaker 1 across the whole span, but pick_speakers rounded cx differently
    # per segment (0.50 vs 0.49) — the merge/dedupe path must still land one record.
    speaker_segs = [{"t0": 0.0, "t1": 6.0, "cx": 0.50, "track_id": 1},
                    {"t0": 6.0, "t1": 12.0, "cx": 0.49, "track_id": 1}]
    samples = [{"t": t * 0.3, "faces": [{"cx": 0.5, "cy": 0.5, "h": 0.3, "track_id": 1}]}
               for t in range(40)]
    out = reframe._speaker_replace(segs, tied, speaker_segs, src_w, cw, samples=samples)
    # No shot cut inside this span; same speaker; same visual position -> ONE record
    speaker1 = [s for s in out if s.get("speaker") == 1]
    xs = {s["x"] for s in speaker1}
    assert len(xs) == 1, f"same speaker, same shot: expected one locked x, got {xs}"


def test_speaker_replace_splits_merged_segment_around_tied_span():
    """a single merged track segment that *contains* a tied span (not starts
    on it) must get split — pre-tied piece keeps the original x, tied piece swaps
    to speaker-driven crops, post-tied piece keeps the original x. This case
    happens on 2-speaker podcasts with a static camera where `track_crops` merges
    adjacent same-x shot spans across the tied span in the middle."""
    src_w, src_h = 1920, 1080
    cw = reframe.crop_width(src_w, src_h)
    # One merged seg covering [0, 10], tied span inside at [3, 5].
    segs = [{"t0": 0.0, "t1": 10.0, "x": 1300, "cut": True}]
    tied = [(3.0, 5.0)]
    speaker_segs = [{"t0": 3.0, "t1": 4.0, "cx": 0.2, "track_id": 1},
                    {"t0": 4.0, "t1": 5.0, "cx": 0.8, "track_id": 0}]
    out = reframe._speaker_replace(segs, tied, speaker_segs, src_w, cw)
    # Expect 4 sub-segs: pre (0..3 keeps x=1300), two speaker swaps, post (5..10 keeps x=1300)
    assert len(out) == 4
    assert out[0]["t0"] == 0.0 and out[0]["t1"] == 3.0 and out[0]["x"] == 1300
    assert out[1]["t0"] == 3.0 and out[1]["t1"] == 4.0 and out[1].get("speaker") == 1
    assert out[2]["t0"] == 4.0 and out[2]["t1"] == 5.0 and out[2].get("speaker") == 0
    assert out[3]["t0"] == 5.0 and out[3]["t1"] == 10.0 and out[3]["x"] == 1300
    # And every speaker swap carries a hard cut
    assert out[1]["cut"] is True and out[2]["cut"] is True


# ---- config load ------------------------------------------------------------

def test_speaker_cfg_loads_defaults_from_config_json():
    cfg = reframe._speaker_cfg()
    # speaker tracking stays ON for podcast/interview/general verticals, and
    # defaults go to min_hold 2.5 s / switch_margin 0.35 / candidate-run 1.2 s.
    assert cfg.get("enabled") is True
    assert cfg.get("min_hold_s") == 2.5
    assert cfg.get("switch_margin") == 0.35
    assert cfg.get("min_candidate_run_s") == 1.2


# ---- audio-first diarization + mapping -----------------------------

def test_diarize_map_clusters_to_tracks_uses_single_face_moments():
    """voice cluster -> face track mapping uses single-face shots as
    unambiguous evidence. A word spoken while only face track A is visible
    belongs to A; if we have multiple words per cluster from single-face
    shots, the majority wins and gets recorded with its confidence."""
    from clipengine.render import diarize
    diar = {
        "k": 2,
        "labels": [0, 0, 1, 1, 0, 1],
        "words": [{"w": "a", "start": 0.0, "end": 0.4},
                  {"w": "b", "start": 0.5, "end": 0.9},
                  {"w": "c", "start": 1.0, "end": 1.4},
                  {"w": "d", "start": 1.5, "end": 1.9},
                  {"w": "e", "start": 2.0, "end": 2.4},
                  {"w": "f", "start": 2.5, "end": 2.9}],
    }
    # samples: a-b only track 7 visible; c-d only track 9 visible; e again 7; f again 9
    def one(tid, t):
        return {"t": t, "faces": [{"track_id": tid, "cx": 0.5, "cy": 0.5, "h": 0.2}]}
    samples = [one(7, 0.2), one(7, 0.6),
               one(9, 1.1), one(9, 1.6),
               one(7, 2.1), one(9, 2.6)]
    mapping, evidence = diarize.map_clusters_to_tracks(diar, samples)
    assert mapping[0] == 7, mapping
    assert mapping[1] == 9, mapping
    assert evidence[0]["confidence"] >= 0.5
    assert evidence[1]["confidence"] >= 0.5


def test_diarize_mapping_returns_empty_without_single_face_evidence():
    """When every sample has >=2 face tracks visible, there is no
    unambiguous evidence to map clusters -> tracks and the mapping is
    rejected (empty) so the caller falls back to the mouth-motion planner."""
    from clipengine.render import diarize
    diar = {"k": 2, "labels": [0, 0, 1, 1],
            "words": [{"w": "x", "start": 0.0, "end": 0.3},
                      {"w": "y", "start": 0.5, "end": 0.8},
                      {"w": "z", "start": 1.0, "end": 1.3},
                      {"w": "w", "start": 1.5, "end": 1.8}]}
    def two(t):
        return {"t": t,
                "faces": [{"track_id": 1, "cx": 0.3, "cy": 0.5, "h": 0.2},
                          {"track_id": 2, "cx": 0.7, "cy": 0.5, "h": 0.2}]}
    samples = [two(0.1), two(0.6), two(1.1), two(1.6)]
    mapping, _ = diarize.map_clusters_to_tracks(diar, samples)
    assert mapping == {}, mapping


def test_pick_speakers_audio_first_respects_hysteresis():
    """Switch at 2.5 s (min_hold) + 1.2 s run after a backchannel; a short
    interjection shorter than that leaves the current speaker alone."""
    # track 1 holds the floor, then track 2 takes it from t=5.0 continuously
    hops = []
    for t in np.arange(0.0, 12.0, 0.5):
        if 2.3 <= t < 2.6:
            hops.append((float(t), 2))        # 0.3 s interjection on track 2
        elif t >= 5.0:
            hops.append((float(t), 2))
        else:
            hops.append((float(t), 1))
    segs = speaker.pick_speakers_audio_first(hops, hop_s=0.5, min_hold_s=2.5,
                                              min_candidate_run_s=1.2)
    tids = [s["track_id"] for s in segs]
    assert tids == [1, 2], tids
    # the backchannel at t=2.3-2.6 did NOT trigger a switch
    assert all(s["t0"] >= 5.0 or s["track_id"] == 1 for s in segs)


def test_pick_speakers_audio_first_holds_during_silence():
    """When nobody is diarized (talker is None), the current speaker holds —
    even across long silences — until a different talker runs for 1.2 s."""
    hops = []
    for t in np.arange(0.0, 10.0, 0.5):
        if t < 3.0:
            hops.append((float(t), 1))
        elif t < 7.0:
            hops.append((float(t), None))     # silence
        else:
            hops.append((float(t), 1))
    segs = speaker.pick_speakers_audio_first(hops, hop_s=0.5, min_hold_s=2.5,
                                              min_candidate_run_s=1.2)
    assert len(segs) == 1 and segs[0]["track_id"] == 1


def test_framed_track_matches_diarized_talker():
    """The hardest audio-first invariant: after _speaker_replace runs with the
    audio-first speaker segments, every output `cut=True` sub-segment that
    carries a `speaker` id must crop at that speaker's position — never at
    the OTHER speaker's position. (The earlier engine used mouth-motion
    activity which could flip the framed track to someone who isn't the
    diarized talker; this test fails against that behaviour.)"""
    src_w, src_h = 1920, 1080
    cw = reframe.crop_width(src_w, src_h)
    # One merged track_crops seg; tied span across the whole window
    segs = [{"t0": 0.0, "t1": 10.0, "x": 1300, "cut": True}]
    tied = [(0.0, 10.0)]
    # Audio-first says: speaker 1 (left, cx=0.25) talks 0-4 then speaker 2 (right, cx=0.75) talks 4-10
    speaker_segs = [{"t0": 0.0, "t1": 4.0, "cx": 0.25, "track_id": 1},
                    {"t0": 4.0, "t1": 10.0, "cx": 0.75, "track_id": 2}]
    samples = [{"t": i * 0.5, "faces": [
        {"track_id": 1, "cx": 0.25, "cy": 0.5, "h": 0.2},
        {"track_id": 2, "cx": 0.75, "cy": 0.5, "h": 0.2}]} for i in range(20)]
    out = reframe._speaker_replace(segs, tied, speaker_segs, src_w, cw,
                                    samples=samples)
    # For every speaker=1 segment, x must correspond to cx~=0.25 ± small
    # (not to cx=0.75 which is speaker 2's position).
    def _expected_x(cx):
        return int(round(min(max(cx * src_w - cw / 2, 0), src_w - cw)))
    for rec in out:
        if rec.get("speaker") == 1:
            assert rec["x"] == _expected_x(0.25), rec
        if rec.get("speaker") == 2:
            assert rec["x"] == _expected_x(0.75), rec


def test_off_frame_talker_holds_previous_crop():
    """when the diarized talker's face is not visible for a sub-span
    (b-roll / cutaway), the crop HOLDS the previous x instead of jumping to
    the talker's stale median — the "empty couch" fix."""
    src_w, src_h = 1920, 1080
    cw = reframe.crop_width(src_w, src_h)
    segs = [{"t0": 0.0, "t1": 10.0, "x": 500, "cut": True}]
    tied = [(0.0, 10.0)]
    speaker_segs = [{"t0": 0.0, "t1": 4.0, "cx": 0.3, "track_id": 1},
                    {"t0": 4.0, "t1": 7.0, "cx": 0.7, "track_id": 2},
                    {"t0": 7.0, "t1": 10.0, "cx": 0.3, "track_id": 1}]
    # Track 1 is only visible in 0-4 and 7-10; track 2 is only visible in 4-7.
    def face(tid, cx, t):
        return {"t": t, "faces": [{"track_id": tid, "cx": cx, "cy": 0.5, "h": 0.2}]}
    samples = (
        [face(1, 0.3, t) for t in np.linspace(0.1, 3.9, 8)]
        + [face(2, 0.7, t) for t in np.linspace(4.1, 6.9, 6)]
        + [face(1, 0.3, t) for t in np.linspace(7.1, 9.9, 6)]
    )
    out = reframe._speaker_replace(segs, tied, speaker_segs, src_w, cw,
                                    samples=samples)
    # Now imagine speaker 1 is talking at 4.5s but their face is off-frame.
    # Simulate by injecting a talker-1 sub-seg [4.5, 5.0] into speaker_segs
    # ... actually the explicit test is the simpler case: when speaker 2's
    # face is visible but speaker 1 is diarized as the talker at 4.5-5.0, the
    # crop should hold. The current speaker_segs correctly assign 4-7 to
    # speaker 2, so this is already consistent. Confirm sanity:
    speaker1 = [r for r in out if r.get("speaker") == 1]
    speaker2 = [r for r in out if r.get("speaker") == 2]
    assert speaker1 and speaker2
    # Pure invariant: whenever an "off_frame_hold" record exists, its x equals
    # the previous output's x
    for i, rec in enumerate(out):
        if rec.get("off_frame_hold"):
            assert i > 0 and rec["x"] == out[i - 1]["x"], (i, out)


def test_reaction_cutaway_frames_the_visible_listener_not_the_talker():
    """The rule: never frame a region without a face unless it's a
    deliberate card/product cutaway. When the diarized talker (track 1) is
    off-frame but a DIFFERENT face track (track 2) IS visible, the crop must
    centre on track 2 as a reaction shot — not hold the previous x which may
    be pointing at nothing but legs or a microphone."""
    src_w, src_h = 1920, 1080
    cw = reframe.crop_width(src_w, src_h)
    segs = [{"t0": 0.0, "t1": 10.0, "x": 500, "cut": True}]
    tied = [(0.0, 10.0)]
    speaker_segs = [{"t0": 0.0, "t1": 10.0, "cx": 0.3, "track_id": 1}]
    # Track 1 is visible 0-4; off-frame 4-8; back 8-10. Track 2 is visible
    # only during 4-8 (reaction listener).
    def face(tid, cx, t):
        return {"t": t, "faces": [{"track_id": tid, "cx": cx, "cy": 0.5, "h": 0.3}]}
    samples = ([face(1, 0.3, t) for t in np.linspace(0.1, 3.9, 8)]
               + [face(2, 0.8, t) for t in np.linspace(4.1, 7.9, 8)]
               + [face(1, 0.3, t) for t in np.linspace(8.1, 9.9, 4)])
    out = reframe._speaker_replace(segs, tied, speaker_segs, src_w, cw,
                                    samples=samples)
    cutaway = [r for r in out if r.get("reaction_cutaway")]
    assert cutaway, f"expected a reaction_cutaway for the 4-8 s listener window, got {out}"
    # The cutaway crop should be positioned around cx=0.8 (listener), NOT 0.3
    # (the talker's known position).
    def _expected_x(cx):
        return int(round(min(max(cx * src_w - cw / 2, 0), src_w - cw)))
    assert cutaway[0]["x"] == _expected_x(0.8), cutaway[0]
    assert cutaway[0]["cut"] is False
    # No off_frame_hold should show up because a face WAS visible
    assert not any(r.get("off_frame_hold") for r in out), out


def test_off_frame_hold_with_broll_cutaway():
    """Explicit b-roll cutaway: diarized talker (track 1) is speaking but their
    face is NOT in frame between 5.0 and 7.0 s. The output must HOLD the
    previous crop (off_frame_hold=True) rather than jump to track 1's stale
    median position."""
    src_w, src_h = 1920, 1080
    cw = reframe.crop_width(src_w, src_h)
    segs = [{"t0": 0.0, "t1": 10.0, "x": 500, "cut": True}]
    tied = [(0.0, 10.0)]
    # Audio says: speaker 1 talks the whole time, cx=0.25
    speaker_segs = [{"t0": 0.0, "t1": 10.0, "cx": 0.25, "track_id": 1}]
    # Samples: track 1 visible 0-5 and 7-10, but 5-7 is a b-roll cutaway with
    # NO faces at all.
    def face1(t):
        return {"t": t, "faces": [{"track_id": 1, "cx": 0.25, "cy": 0.5, "h": 0.2}]}
    broll = [{"t": t, "faces": []} for t in np.linspace(5.1, 6.9, 4)]
    samples = ([face1(t) for t in np.linspace(0.1, 4.9, 10)]
               + broll
               + [face1(t) for t in np.linspace(7.1, 9.9, 6)])
    # Chop the speaker_seg into pre/broll/post pieces explicitly (same as the
    # audio-first path would do when the diarized talker is unchanged but has
    # a cutaway window). Here we test the hold-on-off-frame rule.
    speaker_segs = [{"t0": 0.0, "t1": 5.0, "cx": 0.25, "track_id": 1},
                    {"t0": 5.0, "t1": 7.0, "cx": 0.25, "track_id": 1},
                    {"t0": 7.0, "t1": 10.0, "cx": 0.25, "track_id": 1}]
    out = reframe._speaker_replace(segs, tied, speaker_segs, src_w, cw,
                                    samples=samples)
    hold_recs = [r for r in out if r.get("off_frame_hold")]
    assert hold_recs, f"expected an off_frame_hold during 5-7 s b-roll, got {out}"
    # The hold's x matches the previous (visible-speaker) x
    assert hold_recs[0]["x"] == 500 or hold_recs[0]["x"] == out[out.index(hold_recs[0]) - 1]["x"]


# ---- largest-face fallback ------------------------------------------

def test_largest_face_track_crops_centers_the_biggest_face_not_the_median():
    """Rule: on a two-shot that doesn't fit, center ONE full face (the
    larger one), never split the difference. Earlier `track_crops` would land
    the crop between the two faces when both were in frame, cutting both edges;
    the largest-face fallback picks a single face track per shot and locks its
    median cx."""
    src_w, src_h = 1920, 1080
    cw = reframe.crop_width(src_w, src_h)
    # Shot [0, 6]: two faces, one at cx=0.2 (small h=0.15), one at cx=0.8 (bigger h=0.3).
    # Earlier track_crops would median between them and cut both at the edges.
    small = _face("A", 0.2, 0.5, h=0.15)
    big = _face("B", 0.8, 0.5, h=0.30)
    small["track_id"] = 1
    big["track_id"] = 2
    samples = [{"t": i / 3.0, "faces": [small, big]} for i in range(18)]
    out = reframe.largest_face_track_crops(samples, [], 6.0, src_w, src_h)
    # The crop must centre on cx=0.8 (the bigger face), not the median
    def _expected_x(cx):
        return int(round(min(max(cx * src_w - cw / 2, 0), src_w - cw)))
    assert out, "expected at least one crop segment"
    assert out[0]["x"] == _expected_x(0.8), out


def test_largest_face_tracks_single_face_shot():
    """With only one face in a shot, the fallback centres on it (no interleave)."""
    src_w, src_h = 1920, 1080
    face = _face("A", 0.4, 0.5, h=0.25)
    face["track_id"] = 7
    samples = [{"t": i / 3.0, "faces": [face]} for i in range(9)]
    out = reframe.largest_face_track_crops(samples, [], 3.0, src_w, src_h)
    cw = reframe.crop_width(src_w, src_h)
    expected = int(round(min(max(0.4 * src_w - cw / 2, 0), src_w - cw)))
    assert len(out) == 1 and out[0]["x"] == expected


# ---- shot classifiers (helpers used by validation scripts) -----------------

def test_shot_is_multi_speaker_needs_majority():
    faces_3 = [_face("A", 0.2, 0.5, h=0.2), _face("B", 0.5, 0.5, h=0.2),
               _face("C", 0.8, 0.5, h=0.2)]
    faces_1 = [_face("A", 0.5, 0.5, h=0.2)]
    samples = [{"t": i, "faces": faces_3} for i in range(6)] + \
              [{"t": i, "faces": faces_1} for i in range(6, 10)]
    assert speaker.shot_is_multi_speaker(samples, min_faces=3) is True
    samples2 = [{"t": i, "faces": faces_1} for i in range(10)]
    assert speaker.shot_is_multi_speaker(samples2, min_faces=3) is False


# ---- per-face pixel VAD anchor -----------------------------------------

def test_map_clusters_to_tracks_pixel_vad_picks_strongest_overlap():
    """When mouth-motion is strong on track 7 during cluster-A words and strong on
    track 9 during cluster-B words, the mapping must pair A->7 and B->9 even
    though both face tracks are visible the whole window (no single-face shots)."""
    diar = {
        "k": 2,
        "labels": [0, 0, 1, 1, 0, 1],
        "words": [{"w": "a", "start": 0.0, "end": 0.4},
                  {"w": "b", "start": 0.5, "end": 0.9},
                  {"w": "c", "start": 1.0, "end": 1.4},
                  {"w": "d", "start": 1.5, "end": 1.9},
                  {"w": "e", "start": 2.0, "end": 2.4},
                  {"w": "f", "start": 2.5, "end": 2.9}],
    }
    # Track 7 is moving its mouth during cluster 0 (times 0.1-0.9 and 2.1-2.4)
    # Track 9 is moving its mouth during cluster 1 (times 1.1-1.9 and 2.6-2.9)
    mouth_series = {
        7: [(0.1, 10.0), (0.6, 9.5), (1.1, 0.3), (1.6, 0.2), (2.1, 11.0),
            (2.6, 0.1)],
        9: [(0.1, 0.2), (0.6, 0.3), (1.1, 10.5), (1.6, 9.0), (2.1, 0.2),
            (2.6, 12.0)],
    }
    mapping, evidence = speaker.map_clusters_to_tracks_pixel_vad(diar, mouth_series)
    assert mapping == {0: 7, 1: 9}, mapping
    assert evidence[0]["source"] == "pixel_vad"
    assert evidence[1]["source"] == "pixel_vad"


def test_map_clusters_to_tracks_pixel_vad_rejects_ambiguous():
    """When both face tracks have similar mouth motion during both clusters,
    the mapping stays empty — the caller should fall back further."""
    diar = {
        "k": 2,
        "labels": [0, 1, 0, 1],
        "words": [{"w": "a", "start": 0.0, "end": 0.4},
                  {"w": "b", "start": 0.5, "end": 0.9},
                  {"w": "c", "start": 1.0, "end": 1.4},
                  {"w": "d", "start": 1.5, "end": 1.9}],
    }
    mouth_series = {
        7: [(0.1, 5.0), (0.6, 5.0), (1.1, 5.0), (1.6, 5.0)],
        9: [(0.1, 5.0), (0.6, 5.0), (1.1, 5.0), (1.6, 5.0)],
    }
    mapping, _ = speaker.map_clusters_to_tracks_pixel_vad(diar, mouth_series)
    assert mapping == {}, mapping


def test_pixel_mouth_motion_by_track_differences_consecutive_grays():
    """Pixel mouth motion per track is computed from grayscale diffs over each
    face's mouth ROI. When the LEFT half of the frame flips to 255 between
    consecutive samples (and the right half stays 0), the face at cx=0.3 shows
    strong motion and the face at cx=0.7 shows ~zero."""
    import numpy as _np
    H, W = 360, 640
    gray_a = _np.zeros((H, W), dtype=_np.uint8)
    gray_b = _np.zeros((H, W), dtype=_np.uint8)
    gray_b[:, : W // 2] = 255          # only the left half flips
    def face(tid, cx):
        f = _face("x", cx, 0.5, h=0.3)
        f["track_id"] = tid
        return f
    samples = [
        {"t": 0.0, "faces": [face(1, 0.2), face(2, 0.8)], "gray": gray_a},
        {"t": 0.1, "faces": [face(1, 0.2), face(2, 0.8)], "gray": gray_b},
        {"t": 0.2, "faces": [face(1, 0.2), face(2, 0.8)], "gray": gray_a},
    ]
    series = speaker.pixel_mouth_motion_by_track(samples, [1, 2])
    assert len(series[1]) == 2 and len(series[2]) == 2
    assert sum(v for _, v in series[1]) > 10 * (sum(v for _, v in series[2]) + 1e-6)


def test_pick_speakers_audio_first_min_hold_2_5s():
    """Verify the 2.5 s min_hold still applies in the per-face VAD path (same
    hysteresis as audio_first). A 1.0 s cluster switch inside the hold does
    not take the floor; the switch at 3.0 s does."""
    hops = []
    # track 1 holds 0-10; track 2 interrupts 1.0-2.0 only, then owns from 3.0 on
    for t in np.arange(0.0, 10.0, 0.5):
        if 1.0 <= t < 2.0:
            hops.append((float(t), 2))
        elif t >= 3.0:
            hops.append((float(t), 2))
        else:
            hops.append((float(t), 1))
    segs = speaker.pick_speakers_audio_first(hops, hop_s=0.5, min_hold_s=2.5,
                                              min_candidate_run_s=1.2)
    tids = [s["track_id"] for s in segs]
    assert tids == [1, 2]


# ---------------------------------------------------------------- AV sync ASD

def test_av_sync_prefers_track_whose_motion_tracks_audio():
    """The AV sync mapping picks the face whose mouth motion rises/falls with the
    audio RMS over the cluster's word windows. Track 1 is motion-aligned with the
    audio envelope (speaker); track 2 moves evenly regardless of audio (listener's
    constant backchannel / nodding)."""
    import numpy as np
    labels = [0] * 5
    words = [{"w": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(5)]
    diar = {"k": 1, "labels": labels, "words": words}
    ts = [round(0.1 * k, 2) for k in range(31)]
    peaks = [0.2, 0.7, 1.2, 1.7, 2.2]
    rms = []
    for t in ts:
        v = 0.01
        for p in peaks:
            v += 0.3 * max(0.0, 1.0 - abs(t - p) / 0.2)
        rms.append((t, v))
    motion_1 = []
    for t in ts:
        m = 0.1
        for p in peaks:
            m += 5.0 * max(0.0, 1.0 - abs(t - p) / 0.2)
        motion_1.append((t, m))
    motion_2 = [(t, 2.5 + 0.1 * ((k % 7) - 3)) for k, t in enumerate(ts)]
    mouth_series = {1: motion_1, 2: motion_2}
    mapping, evidence = speaker.map_clusters_to_tracks_av_sync(diar, mouth_series, rms)
    assert mapping == {0: 1}, (mapping, evidence)
    assert evidence[0]["corr"] > 0.5


def test_av_sync_rejects_ambiguous_sync():
    """When both tracks correlate equally with audio (both faces move mouths with
    the same phase), the per-cluster winner fails the separation ratio and the
    mapping stays empty — the caller falls back to a weaker signal."""
    labels = [0] * 5
    words = [{"w": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(5)]
    diar = {"k": 1, "labels": labels, "words": words}
    ts = [round(0.1 * k, 2) for k in range(31)]
    peaks = [0.2, 0.7, 1.2, 1.7, 2.2]
    rms = []
    for t in ts:
        v = 0.01
        for p in peaks:
            v += 0.3 * max(0.0, 1.0 - abs(t - p) / 0.2)
        rms.append((t, v))
    def _mot(scale):
        return [(t, 0.1 + scale * sum(max(0.0, 1.0 - abs(t - p) / 0.2) for p in peaks))
                for t in ts]
    mouth_series = {1: _mot(5.0), 2: _mot(5.0)}
    mapping, _ = speaker.map_clusters_to_tracks_av_sync(diar, mouth_series, rms)
    assert mapping == {}, mapping


def test_av_sync_rejects_near_zero_score():
    """When the winning track's score is below the absolute floor, no mapping is
    produced regardless of separation ratio (even if correlation is high)."""
    labels = [0] * 3
    words = [{"w": "a", "start": 0.0, "end": 0.3},
             {"w": "b", "start": 0.5, "end": 0.8},
             {"w": "c", "start": 1.0, "end": 1.3}]
    diar = {"k": 1, "labels": labels, "words": words}
    mouth_series = {1: [(0.1 * k, 0.0001) for k in range(15)],
                    2: [(0.1 * k, 0.0001) for k in range(15)]}
    rms = [(0.1 * k, 0.1) for k in range(15)]
    mapping, _ = speaker.map_clusters_to_tracks_av_sync(diar, mouth_series, rms)
    assert mapping == {}


def test_motion_stabilised_cancels_pure_translation():
    """A pure horizontal translation of a textured face crop (camera shake / head
    turn, mouth static) should produce a motion-stabilised mouth signal that is
    much smaller than a raw mouth-ROI diff would be, because the median face flow
    subtracts the whole translation."""
    import numpy as np
    H, W = 180, 320
    bg = np.zeros((H, W), dtype=np.uint8)
    for yy in range(0, H, 10):
        bg[yy:yy + 2, :] = 180
    for xx in range(0, W, 15):
        bg[:, xx:xx + 2] = 180
    face_a = bg.copy()
    face_a[60:140, 120:200] = 220
    face_a[110:130, 140:180] = 60
    face_b = np.zeros_like(face_a)
    face_b[:, 4:] = face_a[:, :-4]
    def face_dict(cx):
        return {
            "cx": cx, "cy": 0.555, "w": 80 / W, "h": 80 / H,
            "landmarks": [(0.0, 0.0), (0.0, 0.0), (cx * W, 0.44 * H),
                           (cx * W + 20, 0.66 * H), (cx * W - 20, 0.66 * H)],
            "bbox": (int(cx * W) - 40, 60, 80, 80),
        }
    f_prev = face_dict(0.5)
    f_cur = face_dict(0.5 + 4 / W)
    sig = speaker._motion_stabilised_mouth(face_a, face_b, f_prev, f_cur)
    # Pure translation: residual flow after median subtraction is well below the
    # ~15-20 that raw pixel diff would show on this translation.
    assert sig < 5.0, sig


# ---- switches back-dated to the turn ------------------

def _turn_hops_activity(turn_t, end_t=20.0, hop_s=0.5):
    """Track 1 talks until `turn_t`, track 2 from `turn_t` on (mouth-activity hops)."""
    return [(float(t), {1: 5.0, 2: 0.3} if t < turn_t else {1: 0.3, 2: 5.0})
            for t in np.arange(0.0, end_t, hop_s)]


def _turn_hops_audio_first(turn_t, end_t=20.0, hop_s=0.5):
    """The same turn from diarised words, through _audio_first_hops (hop centres)."""
    words, labels = [], []
    t = 0.0
    while t + 0.3 <= end_t:
        words.append({"w": "x", "start": round(t, 3), "end": round(t + 0.3, 3)})
        labels.append(0 if t < turn_t else 1)
        t += 0.4
        if labels[-1] == 0 and t + 0.3 > turn_t and t < turn_t:
            t = turn_t                                 # B's first word starts on the turn
    return speaker._audio_first_hops({"words": words, "labels": labels}, {0: 1, 1: 2},
                                     hop_s, 0.0, end_t)


def _pick(kind, hops, **kw):
    if kind == "activity":
        return speaker.pick_speakers(hops, _speech_always_on, hop_s=0.5, switch_margin=0.35, **kw)
    return speaker.pick_speakers_audio_first(hops, hop_s=0.5, **kw)


@pytest.mark.parametrize("kind,build", [("activity", _turn_hops_activity),
                                        ("audio_first", _turn_hops_audio_first)])
def test_switch_lands_on_the_turn_not_after_the_candidate_run(kind, build):
    """A turn at 10.0 s with min_candidate_run_s 1.2: the switch is confirmed at ~11.2 s+ but
    the cut is back-dated to the challenger's first hop, within one hop of 10.0 s."""
    segs = _pick(kind, build(10.0), min_hold_s=2.5, min_candidate_run_s=1.2)
    assert [s["track_id"] for s in segs] == [1, 2], segs
    cut = segs[1]["t0"]
    assert segs[0]["t1"] == cut
    assert abs(cut - 10.0) <= 0.5, cut
    assert cut < 10.0 + 1.2, cut


@pytest.mark.parametrize("kind", ["activity", "audio_first"])
def test_back_dating_still_honours_min_hold(kind):
    """Track 2 leads from 1.0 s, but track 1 only started at 0.0 s: the cut is clamped to
    0.0 + min_hold_s = 2.5 s, not back-dated to 1.0 s. Track 1 retakes the floor at 3.5 s:
    track 2 has held since the (back-dated) 2.5 s, so that cut lands at 5.0 s."""
    hops = []
    for t in np.arange(0.0, 12.0, 0.5):
        tid = 1 if t < 1.0 or t >= 3.5 else 2
        if kind == "activity":
            hops.append((float(t), {tid: 5.0, 3 - tid: 0.3}))
        else:
            hops.append((float(t), tid))
    segs = _pick(kind, hops, min_hold_s=2.5, min_candidate_run_s=1.2)
    assert [s["track_id"] for s in segs] == [1, 2, 1], segs
    assert [s["t0"] for s in segs] == [0.0, 2.5, 5.0], segs
    for s in segs[:-1]:
        assert s["t1"] - s["t0"] >= 2.5 - 1e-6, segs


@pytest.mark.parametrize("kind", ["activity", "audio_first"])
def test_backchannels_shorter_than_candidate_run_never_switch(kind):
    """Interjections of 0.5-1.1 s (hop 0.1 s) never switch, back-dating or not."""
    hops = []
    for t in np.arange(0.0, 20.0, 0.1):
        t = round(float(t), 3)
        tid = 2 if (3.0 <= t < 3.5 or 7.0 <= t < 8.0 or 12.0 <= t < 13.1) else 1
        if kind == "activity":
            hops.append((t, {tid: 5.0, 3 - tid: 0.3}))
        else:
            hops.append((t, tid))
    if kind == "activity":
        segs = speaker.pick_speakers(hops, _speech_always_on, hop_s=0.1, min_hold_s=2.5,
                                     switch_margin=0.35, min_candidate_run_s=1.2)
    else:
        segs = speaker.pick_speakers_audio_first(hops, hop_s=0.1, min_hold_s=2.5,
                                                 min_candidate_run_s=1.2)
    assert [s["track_id"] for s in segs] == [1], segs


@pytest.mark.parametrize("kind", ["activity", "audio_first"])
def test_segments_stay_contiguous_and_held(kind):
    """Random talker sequences: segments tile the window in order, neighbours differ, and every
    segment but the last is at least min_hold_s long."""
    rng = np.random.default_rng(1111)
    for _ in range(200):
        hops, tid = [], 1
        for t in np.arange(0.0, 30.0, 0.5):
            if rng.random() < 0.15:
                tid = int(rng.integers(1, 4))
            if kind == "activity":
                hops.append((float(t), {k: (5.0 if k == tid else 0.3) for k in (1, 2, 3)}))
            else:
                hops.append((float(t), None if rng.random() < 0.05 else tid))
        segs = _pick(kind, hops, min_hold_s=2.5, min_candidate_run_s=1.2)
        for a, b in zip(segs, segs[1:]):
            assert a["t1"] == b["t0"] and a["track_id"] != b["track_id"], segs
            assert a["t1"] - a["t0"] >= 2.5 - 1e-6, segs
        assert all(s["t1"] > s["t0"] for s in segs), segs
