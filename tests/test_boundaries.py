"""Boundary-engine tests (clipengine/boundaries.py) — pure numpy, offline, no ffmpeg.

PHASE4a acceptance criteria covered here:
  2. start_cut / end_cut never slice a word (seeded property tests over random word layouts)
  3. the quietest-frame rule finds a 120 ms dip on a music bed that never drops below -30 dBFS,
     where the old silencedetect(-30 dB, 0.35 s) rule keeps the raw (mid-word) edge
  8. out-of-band picks are extended (<= 3 sentences) or rejected, never truncated
  9. suppress drops clips overlapping a higher-scored clip by more than max_overlap
plus the helpers: frame_rms, quietest_point, scene cuts, filler trim, ends_cleanly,
sentences_around, energy_clip.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

PIPE = Path(__file__).resolve().parent.parent / "clipengine"

from clipengine import boundaries as B


def _words(tokens, t=0.0, dur=0.2, gap=0.1):
    """Evenly spaced Word dicts."""
    out = []
    for tok in tokens:
        out.append({"w": tok, "start": round(t, 3), "end": round(t + dur, 3)})
        t += dur + gap
    return out


# ---------------------------------------------------------------------------
# frame_rms
# ---------------------------------------------------------------------------
def test_frame_rms_levels_and_partial_frame():
    sr = 16000
    r = B.frame_rms(np.full(int(0.025 * sr), 0.5), sr)          # 2.5 frames -> 2
    assert len(r) == 2 and r.dtype == np.float32
    assert np.allclose(r, 0.5)
    t = np.arange(sr) / sr
    assert np.allclose(B.frame_rms(np.sin(2 * np.pi * 100 * t), sr), 1 / np.sqrt(2), atol=1e-3)
    assert len(B.frame_rms(np.zeros(10), sr)) == 0               # shorter than one frame


# ---------------------------------------------------------------------------
# Criterion 2: cut invariants over randomized layouts
# ---------------------------------------------------------------------------
N_LAYOUTS = 600
TOKENS = ["so", "Um,", "the", "river", "was", "cold.", "you", "know", "I", "mean", "wow!", "and"]


def _layout(r):
    """Random non-overlapping words: zero / tiny / normal / large gaps, zero-length words,
    sometimes whisper-style 2-decimal timestamps, sometimes the first word at t=0."""
    n = r.randint(1, 24)
    grid = r.random() < 0.5
    t = r.choice([0.0, 0.0, r.uniform(0.0, 0.5), r.uniform(0.5, 60.0)])
    words = []
    for _ in range(n):
        dur = r.choice([0.0, r.uniform(0.01, 0.1), r.uniform(0.1, 0.9)])
        gap = r.choice([0.0, 0.0, r.uniform(0.0, 0.03), r.uniform(0.03, 0.5), r.uniform(0.5, 6.0)])
        if grid:
            t, dur, gap = round(t, 2), round(dur, 2), round(gap, 2)
        s = t
        e = round(t + dur, 2) if grid else t + dur
        words.append({"w": r.choice(TOKENS), "start": s, "end": e})
        t = round(e + gap, 2) if grid else e + gap
    for a, b in zip(words, words[1:]):
        assert a["start"] <= a["end"] <= b["start"]             # generator sanity
    return words


def _rms_for(mode, words, r, g, frame_s):
    """(rms, t0) for a layout. Coverage is full, or starts/ends mid-layout."""
    if mode == "none":
        return None, 0.0
    first, last = words[0]["start"], words[-1]["end"]
    u = r.random()
    if u < 0.7:
        t0, t1 = max(0.0, first - r.uniform(0, 2)), last + r.uniform(0, 2)
    elif u < 0.85:
        t0, t1 = r.uniform(first, last) if last > first else first, last + 1.0
    else:
        t0, t1 = max(0.0, first - 1.0), r.uniform(first, last + 0.01)
    n = max(1, int((t1 - t0) / frame_s))
    if mode == "zeros":
        return np.zeros(n), t0
    if mode == "random":
        rms = g.random(n) if r.random() < 0.5 else g.integers(0, 3, n).astype(float)  # 2nd: many ties
        return (rms.astype(np.float32) if r.random() < 0.5 else rms), t0
    # synthetic: loud words over a noisy bed, with short dips dropped into random gaps
    tc = t0 + (np.arange(n) + 0.5) * frame_s
    rms = 0.1 + 0.02 * g.random(n)
    for w in words:
        m = (tc >= w["start"]) & (tc <= w["end"])
        rms[m] = 0.4 + 0.1 * g.random(int(m.sum()))
    for a, b in zip(words, words[1:]):
        if b["start"] > a["end"] and r.random() < 0.5:
            c = r.uniform(a["end"], b["start"])
            rms[(tc >= c - 0.03) & (tc <= c + 0.03)] = 0.02
    return rms, t0


def _scene_times(words, r):
    """Shot changes near every kind of boundary: inside gaps, exactly on bounds, inside words."""
    sc = []
    for k, w in enumerate(words):
        if r.random() < 0.5:
            a = words[k - 1]["end"] - 0.02 if k else max(0.0, w["start"] - 0.6)
            sc += [r.uniform(a - 0.1, w["start"] + 0.1), a, w["start"], w["end"], w["end"] + 0.02,
                   r.uniform(w["start"], w["end"]) if w["end"] > w["start"] else w["start"]]
    return sc + [r.uniform(0.0, words[-1]["end"] + 2.0) for _ in range(3)]


@pytest.mark.parametrize("with_scenes", [False, True])
@pytest.mark.parametrize("mode", ["none", "random", "synthetic", "zeros"])
def test_cut_invariants_property(mode, with_scenes):
    stats = dict.fromkeys(("zero_gap", "tiny_gap", "large_gap", "first_at_0", "cuts"), 0)
    for seed in range(N_LAYOUTS):
        r, g = random.Random(seed), np.random.default_rng(seed)
        words = _layout(r)
        frame_s = r.choice([0.01, 0.01, 0.02, 0.005])
        rms, t0 = _rms_for(mode, words, r, g, frame_s)
        sc = _scene_times(words, r) if with_scenes else None
        dflt = r.random() < 0.5
        lp, ml = (0.15, 0.6) if dflt else (r.uniform(0, 1), r.uniform(0.05, 2))
        tp, mt = (0.35, 0.8) if dflt else (r.uniform(0, 1), r.uniform(0.05, 2))
        stats["first_at_0"] += words[0]["start"] == 0.0
        for a, b in zip(words, words[1:]):
            gap = b["start"] - a["end"]
            stats["zero_gap"] += gap == 0.0
            stats["tiny_gap"] += 0.0 < gap < 0.02
            stats["large_gap"] += gap > 1.0
        n = len(words)
        for i in range(n):
            ctx = f"seed={seed} mode={mode} scenes={with_scenes} i={i} words={words}"
            ws, we = words[i]["start"], words[i]["end"]

            s = B.start_cut(words, i, rms, t0, frame_s, lead_pad=lp, max_lead=ml, scene_times=sc)
            lo, hi = B.start_bounds(words, i, ml)
            assert lo <= s <= hi, ctx
            assert s <= ws, ctx                                   # never after the first word begins
            assert s >= ws - ml, ctx
            if i > 0:
                assert s >= words[i - 1]["end"] - 0.02, ctx       # never before prev word ends
            else:
                assert s >= max(0.0, ws - ml), ctx
            if rms is None:                                       # spec formula, clamped to bounds
                want = ws - lp if i == 0 else max(ws - lp, (words[i - 1]["end"] + ws) / 2.0)
                assert s == min(max(want, lo), hi), ctx

            e = B.end_cut(words, i, rms, t0, frame_s, tail_pad=tp, max_tail=mt, scene_times=sc)
            lo, hi = B.end_bounds(words, i, mt)
            assert lo <= e <= hi, ctx
            assert e >= we, ctx                                   # never before the last word ends
            assert e <= we + mt, ctx
            if i + 1 < n:
                assert e <= words[i + 1]["start"] + 0.02, ctx     # never after next word begins
            if rms is None:
                want = we + tp if i + 1 == n else min(we + tp, (we + words[i + 1]["start"]) / 2.0)
                assert s <= ws and e == min(max(want, lo), hi), ctx
            stats["cuts"] += 2
    # the generator actually exercised the edge cases
    assert stats["zero_gap"] > 500 and stats["tiny_gap"] > 200 and stats["large_gap"] > 500, stats
    assert stats["first_at_0"] > 100 and stats["cuts"] > 10000, stats


def test_cuts_with_overlapping_timestamps_never_cut_into_word():
    """Whisper sometimes emits prev.end > next.start. The word itself still wins."""
    words = [{"w": "a", "start": 1.0, "end": 1.6}, {"w": "b", "start": 1.5, "end": 2.0},
             {"w": "c", "start": 1.95, "end": 2.5}]
    rms = np.random.default_rng(0).random(400)
    for r in (None, rms):
        assert B.start_cut(words, 1, r) == 1.5
        assert B.end_cut(words, 1, r) == 2.0
        assert B.start_bounds(words, 1) == (1.5, 1.5) and B.end_bounds(words, 1) == (2.0, 2.0)


# ---------------------------------------------------------------------------
# Criterion 3: music bed with a 120 ms dip vs the old silencedetect rule
# ---------------------------------------------------------------------------
SR = 16000
MB_WORDS = [
    {"w": "and", "start": 7.40, "end": 7.70}, {"w": "we", "start": 7.80, "end": 8.10},
    {"w": "lost.", "start": 8.20, "end": 8.90},
    {"w": "Then", "start": 9.40, "end": 9.70},        # clip's first word; gap before = [8.90, 9.40]
    {"w": "he", "start": 9.80, "end": 10.20}, {"w": "shoved", "start": 10.30, "end": 10.80},
    {"w": "all", "start": 10.90, "end": 11.20},
    {"w": "in.", "start": 11.30, "end": 11.70},       # clip's last word; gap after = [11.70, 12.30]
    {"w": "Wow", "start": 12.30, "end": 12.80}, {"w": "unreal", "start": 12.90, "end": 13.50},
]
START_DIP = (9.10, 9.22)                              # 120 ms each, inside the gaps
END_DIP = (11.95, 12.07)
RAW_START, RAW_END = 10.0, 11.5                       # coarse (whole/half second) old-style edges


def _db(x):
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1e-12))


def _music_bed(dips=(START_DIP, END_DIP), dur=16.0, seed=7):
    """Gaussian music/crowd bed at ~-14 dBFS, speech on top, 120 ms dips attenuated to ~-24 dBFS."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    level = np.full(n, 0.2)
    for w in MB_WORDS:
        level[(t >= w["start"]) & (t < w["end"])] = 0.36
    for a, b in dips:
        level[(t >= a) & (t < b)] = 0.06
    return np.clip(np.random.default_rng(seed).standard_normal(n) * level, -1, 1).astype(np.float32)


def _in(t, span):
    return span[0] <= t <= span[1]


def _inside_word(t):
    return any(w["start"] < t < w["end"] for w in MB_WORDS)


# OLD rule, reconstructed inline from the removed engine._silences/_snap_start/_snap_end
# (ffmpeg silencedetect=noise=-30dB:d=0.35 on [edge-win, edge+win]; nearest pause else raw edge).
def _old_pauses(audio, sr, lo, hi, noise_db=-30.0, d=0.35, frame_s=0.01):
    """(starts, ends) of runs of 10 ms frames below noise_db lasting >= d, absolute seconds."""
    n = int(round(sr * frame_s))
    seg = np.asarray(audio[int(round(lo * sr)):int(round(hi * sr))], dtype=np.float64)
    k = len(seg) // n
    quiet = _db(np.sqrt((seg[:k * n].reshape(k, n) ** 2).mean(axis=1))) < noise_db
    starts, ends, i = [], [], 0
    while i < k:
        j = i
        while j < k and quiet[j]:
            j += 1
        if j > i and (j - i) * frame_s >= d:
            starts.append(lo + i * frame_s)
            ends.append(lo + j * frame_s)
        i = max(j, i + 1)
    return starts, ends


def _old_snap_start(audio, sr, start, dur, win=4.0):
    lo, hi = max(0.0, start - win), min(dur, start + win)
    if hi - lo < 1:
        return start
    _, ends = _old_pauses(audio, sr, lo, hi)
    onsets = [e for e in ends if abs(e - start) <= win]
    return min(onsets, key=lambda e: abs(e - start)) if onsets else start


def _old_snap_end(audio, sr, end, dur, win=6.0):
    lo, hi = max(0.0, end - win), min(dur, end + win)
    if hi - lo < 1:
        return end
    starts, _ = _old_pauses(audio, sr, lo, hi)
    pauses = [p for p in starts if abs(p - end) <= win and p > 0]
    return min(pauses, key=lambda p: abs(p - end)) if pauses else end


def test_music_bed_never_below_minus_30_dbfs():
    a = _music_bed()
    rms = B.frame_rms(a, SR)
    assert _db(rms).min() > -30.0                                  # every 10 ms frame, incl. dips
    assert _db(np.sqrt(np.mean(a.astype(np.float64) ** 2))) > -30.0
    # the dips are real: clearly quieter than the bed around them
    for a_, b_ in (START_DIP, END_DIP):
        dip = rms[int(round(a_ * 100)):int(round(b_ * 100))]
        bed = rms[int(round(a_ * 100)) - 15:int(round(a_ * 100)) - 3]
        assert len(dip) == 12 and dip.max() < 0.5 * bed.min()
    # and they sit inside the cut search ranges
    assert B.start_bounds(MB_WORDS, 3)[0] < START_DIP[0] and START_DIP[1] < B.start_bounds(MB_WORDS, 3)[1]
    assert B.end_bounds(MB_WORDS, 7)[0] < END_DIP[0] and END_DIP[1] < B.end_bounds(MB_WORDS, 7)[1]


@pytest.mark.parametrize("t0", [0.0, 5.0])                          # 5.0: a load_audio window
def test_cuts_land_in_dip_over_music_bed(t0):
    a = _music_bed()[int(t0 * SR):]
    rms = B.frame_rms(a, SR)
    assert _in(B.start_cut(MB_WORDS, 3, rms, t0=t0), START_DIP)
    assert _in(B.end_cut(MB_WORDS, 7, rms, t0=t0), END_DIP)
    sents = [{"i": 0, "start": 7.40, "end": 8.90, "text": "and we lost.", "w0": 0, "w1": 2},
             {"i": 1, "start": 9.40, "end": 11.70, "text": "Then he shoved all in.", "w0": 3, "w1": 7},
             {"i": 2, "start": 12.30, "end": 13.50, "text": "Wow unreal", "w0": 8, "w1": 9}]
    c = B.clip_from_sentences(MB_WORDS, sents, 1, 1, rms, t0, {"min_dur_s": 1, "max_dur_s": 10})
    assert _in(c["start"], START_DIP) and _in(c["end"], END_DIP)


@pytest.mark.parametrize("win", [4.0, 8.0])                         # SNAP_WIN / SEM_SNAP
def test_old_silencedetect_rule_keeps_raw_edge_on_music_bed(win):
    a = _music_bed()
    dur = len(a) / SR
    for edge in (RAW_START, RAW_END):
        assert _old_pauses(a, SR, max(0.0, edge - win), min(dur, edge + win)) == ([], [])
    s = _old_snap_start(a, SR, RAW_START, dur, win)
    e = _old_snap_end(a, SR, RAW_END, dur, win)
    assert s == RAW_START and e == RAW_END                        # no pause -> raw edge kept
    assert not _in(s, START_DIP) and not _in(e, END_DIP)
    assert _inside_word(s) and _inside_word(e)                    # ...which slices a word
    rms = B.frame_rms(a, SR)                                      # the new rule, same audio
    assert _in(B.start_cut(MB_WORDS, 3, rms), START_DIP) and _in(B.end_cut(MB_WORDS, 7, rms), END_DIP)


def test_old_rule_reconstruction_does_snap_to_true_silence():
    """Control: the reconstructed old rule is not trivially 'keep raw edge'."""
    a = _music_bed(dips=())
    a[int(9.0 * SR):int(9.4 * SR)] = 0.0
    a[int(11.75 * SR):int(12.25 * SR)] = 0.0
    dur = len(a) / SR
    assert _old_snap_start(a, SR, RAW_START, dur, 4.0) == pytest.approx(9.40, abs=0.011)
    assert _old_snap_end(a, SR, RAW_END, dur, 6.0) == pytest.approx(11.75, abs=0.011)


# ---------------------------------------------------------------------------
# quietest_point
# ---------------------------------------------------------------------------
def test_quietest_point_minimum_and_t0_offset():
    rms = np.ones(100)
    rms[30] = 0.1
    assert B.quietest_point(rms, 0.0, 0.01, 0.0, 1.0) == pytest.approx(0.305)
    assert B.quietest_point(rms, 100.0, 0.01, 100.0, 101.0) == pytest.approx(100.305)
    assert B.quietest_point(rms, 100.0, 0.01, 101.0, 100.0) == pytest.approx(100.305)  # lo/hi swapped
    t = B.quietest_point(rms, 100.0, 0.01, 100.5, 100.9)          # minimum outside range ignored
    assert 100.5 <= t <= 100.9 and t != pytest.approx(100.305)


def test_quietest_point_ties_resolve_toward_centre():
    z = np.zeros(100)
    assert B.quietest_point(z, 0.0, 0.01, 0.2, 0.61) == pytest.approx(0.405)   # centre is a frame
    assert abs(B.quietest_point(z, 0.0, 0.01, 0.2, 0.6) - 0.4) <= 0.005 + 1e-9
    assert B.quietest_point(z, 7.0, 0.01, 7.2, 7.61) == pytest.approx(7.405)
    two = np.ones(100)
    two[25] = two[55] = 0.0                                        # centre 0.45: 0.555 is closer
    assert B.quietest_point(two, 0.0, 0.01, 0.0, 0.9) == pytest.approx(0.555)
    assert B.quietest_point(two, 0.0, 0.01, 0.0, 0.7) == pytest.approx(0.255)


def test_quietest_point_narrow_or_uncovered_range_returns_midpoint():
    rms = np.arange(100, dtype=float)
    for lo, hi in [(0.201, 0.204), (0.203, 0.208), (0.3, 0.3)]:   # narrower than one 10 ms frame
        t = B.quietest_point(rms, 0.0, 0.01, lo, hi)
        assert t == pytest.approx((lo + hi) / 2) and lo <= t <= hi
    assert B.quietest_point(rms, 0.0, 0.01, 5.0, 6.0) == pytest.approx(5.5)    # after the audio
    assert B.quietest_point(rms, 10.0, 0.01, 2.0, 3.0) == pytest.approx(2.5)   # before t0


# ---------------------------------------------------------------------------
# scene cuts
# ---------------------------------------------------------------------------
SC_WORDS = [{"w": "lost.", "start": 0.40, "end": 1.00}, {"w": "Then", "start": 1.50, "end": 1.80},
            {"w": "he", "start": 1.90, "end": 2.10}, {"w": "folded.", "start": 2.20, "end": 2.60},
            {"w": "Wow", "start": 3.20, "end": 3.50}]


def _scene_rms(start_scene_level, end_scene_level):
    rms = np.ones(400)
    rms[100:150] = 0.5
    rms[110] = 0.1                        # start gap [0.98, 1.50]: quietest at 1.105
    rms[140] = start_scene_level          # frame of a shot change at 1.405
    rms[260:320] = 0.5
    rms[300] = 0.1                        # end gap [2.60, 3.22]: quietest at 3.005
    rms[280] = end_scene_level            # frame of a shot change at 2.805
    rms[50] = rms[170] = rms[250] = rms[330] = 0.0   # silent frames outside both ranges
    return rms


def test_scene_cut_near_quiet_wins():
    for lvl in (0.14, 0.15):              # <= 1.5x the range minimum (0.1)
        rms = _scene_rms(lvl, lvl)
        assert B.start_cut(SC_WORDS, 1, rms, scene_times=[1.405]) == pytest.approx(1.405)
        assert B.end_cut(SC_WORDS, 3, rms, scene_times=[2.805]) == pytest.approx(2.805)


def test_scene_cut_too_loud_loses():
    rms = _scene_rms(0.3, 0.3)            # 3x the minimum
    assert B.start_cut(SC_WORDS, 1, rms, scene_times=[1.405]) == pytest.approx(1.105)
    assert B.end_cut(SC_WORDS, 3, rms, scene_times=[2.805]) == pytest.approx(3.005)


def test_scene_cut_outside_range_ignored():
    rms = _scene_rms(0.14, 0.14)
    assert B.start_cut(SC_WORDS, 1, rms, scene_times=[0.505, 1.705]) == pytest.approx(1.105)
    assert B.end_cut(SC_WORDS, 3, rms, scene_times=[2.505, 3.305]) == pytest.approx(3.005)
    # the in-range one still wins when mixed with out-of-range ones
    assert B.start_cut(SC_WORDS, 1, rms, scene_times=[0.505, 1.405, 1.705]) == pytest.approx(1.405)
    # no audio -> scene times cannot move the pad-based cut
    assert B.start_cut(SC_WORDS, 1, None, scene_times=[1.405]) == B.start_cut(SC_WORDS, 1)


# ---------------------------------------------------------------------------
# trim_leading_fillers
# ---------------------------------------------------------------------------
TAIL = ["the", "river", "was", "cold."]


@pytest.mark.parametrize("lead,n", [(["so"], 1), (["So,"], 1), (["Um,"], 1), (["you", "know"], 2),
                                    (["You", "know,"], 2), (["I", "mean"], 2), (["okay.", "Well,"], 2)])
def test_trim_leading_fillers_each(lead, n):
    w = _words(lead + TAIL)
    assert B.trim_leading_fillers(w, 0, len(w) - 1) == n


def test_trim_leading_fillers_chain():
    w = _words(["So", "Um,", "you", "know,", "I", "mean", "the", "river", "was", "cold."])
    assert B.trim_leading_fillers(w, 0, 9) == 6
    assert B.trim_leading_fillers(w, 2, 9) == 6                    # starts from the given w0


def test_trim_leading_fillers_stops_on_tight_gap():
    w = _words(["so", "the", "river", "was", "cold."])
    w[1]["start"] = w[0]["end"] + 0.05                             # < 0.06: no room to cut between
    assert B.trim_leading_fillers(w, 0, 4) == 0
    w = _words(["so", "um", "the", "river", "was", "cold."])
    w[2]["start"] = w[1]["end"] + 0.04
    assert B.trim_leading_fillers(w, 0, 5) == 1                    # 'so' trimmed, stops at 'um'
    w = _words(["you", "know", "the", "river", "was", "cold."])
    w[2]["start"] = w[1]["end"] + 0.05                             # gap after the whole phrase
    assert B.trim_leading_fillers(w, 0, 5) == 0
    w = _words(["so", "the", "river", "was", "cold."], gap=0.07)
    assert B.trim_leading_fillers(w, 0, 4) == 1


def test_trim_leading_fillers_never_past_w1_minus_3():
    w = _words(["so"] * 8)
    assert B.trim_leading_fillers(w, 0, 7) == 4
    assert B.trim_leading_fillers(w, 0, 4) == 1
    assert B.trim_leading_fillers(w, 0, 2) == 0                    # "So what now?" stays whole
    w = _words(["you", "know", "it", "works."])
    assert B.trim_leading_fillers(w, 0, 3) == 0                    # would leave < 4 words


@pytest.mark.parametrize("lead", [["Listen"], ["Sony"], ["likely"], ["you", "see"], ["I", "think"],
                                  ["Right-wing"]])
def test_trim_leading_fillers_leaves_non_fillers(lead):
    w = _words(lead + TAIL)
    assert B.trim_leading_fillers(w, 0, len(w) - 1) == 0


# ---------------------------------------------------------------------------
# clip_from_sentences (incl. criterion 8)
# ---------------------------------------------------------------------------
def _transcript(specs, t=10.0, gap=0.1, sent_gap=0.8):
    """specs: sentence text, or (text, word_s). Default word_s 0.3."""
    words, sents = [], []
    for sp in specs:
        text, word_s = (sp, 0.3) if isinstance(sp, str) else sp
        w0 = len(words)
        for tok in text.split():
            words.append({"w": tok, "start": round(t, 3), "end": round(t + word_s, 3)})
            t += word_s + gap
        t += sent_gap - gap
        sents.append({"i": len(sents), "start": words[w0]["start"], "end": words[-1]["end"],
                      "text": text, "w0": w0, "w1": len(words) - 1})
    return words, sents


SIX = "Big red dogs ran very fast."      # 6 words: 2.3 s; k sentences -> clip = 3.1k - 0.3 s
CFG = {"lead_pad_s": 0.15, "tail_pad_s": 0.35}


def _cfg(lo, hi):
    return {"min_dur_s": lo, "max_dur_s": hi, **CFG}


@pytest.mark.parametrize("variant", ["none", "rms", "rms+scenes"])
def test_clip_from_sentences_uses_start_cut_and_end_cut(variant):
    words, sents = _transcript(["Okay this is the warmup line.", "So, um, the river was freezing cold.",
                                "We waited there for hours.", "Nobody came.", "Then the ice cracked."])
    g = np.random.default_rng(3)
    rms, t0 = (None, 0.0) if variant == "none" else (g.random(3000), 9.0)
    sc = [words[7]["start"] - 0.1, words[16]["end"] + 0.05] if variant == "rms+scenes" else None
    cfg = {"min_dur_s": 2, "max_dur_s": 30, "lead_pad_s": 0.2, "tail_pad_s": 0.3}
    c = B.clip_from_sentences(words, sents, 1, 2, rms, t0, cfg, 0.01, sc)
    w0 = B.trim_leading_fillers(words, sents[1]["w0"], sents[2]["w1"])
    assert w0 == sents[1]["w0"] + 2 and words[w0]["w"] == "the"   # "So," "um," trimmed
    assert c["start"] == B.start_cut(words, w0, rms, t0, 0.01, lead_pad=0.2, scene_times=sc)
    assert c["end"] == B.end_cut(words, sents[2]["w1"], rms, t0, 0.01, tail_pad=0.3, scene_times=sc)
    assert c == {"start": c["start"], "end": c["end"], "w0": w0, "w1": sents[2]["w1"],
                 "s0": 1, "s1": 2, "boundary_type": "sentence"}


def test_clip_in_band_is_returned_unchanged():
    words, sents = _transcript([SIX] * 8)
    c = B.clip_from_sentences(words, sents, 1, 3, cfg=_cfg(5, 20))
    assert (c["s0"], c["s1"], c["w0"], c["w1"]) == (1, 3, sents[1]["w0"], sents[3]["w1"])
    assert c["end"] - c["start"] == pytest.approx(3.1 * 3 - 0.3)


def test_short_pick_extended_minimally():
    words, sents = _transcript([SIX] * 8)
    c = B.clip_from_sentences(words, sents, 0, 0, cfg=_cfg(8, 30))    # 2.8 s -> needs 3 sentences
    assert c["s0"] == 0 and c["s1"] == 2 and c["w1"] == sents[2]["w1"]
    assert c["end"] == B.end_cut(words, sents[2]["w1"], tail_pad=0.35)
    assert 8 <= c["end"] - c["start"] <= 30
    c = B.clip_from_sentences(words, sents, 2, 3, cfg=_cfg(8, 30))    # 5.9 s -> one more sentence
    assert (c["s0"], c["s1"]) == (2, 4)


def test_short_pick_extends_at_most_three_sentences():
    words, sents = _transcript([SIX] * 8)
    c = B.clip_from_sentences(words, sents, 0, 0, cfg=_cfg(12, 30))   # 4 sentences = 12.1 s
    assert c["s1"] == 3 and c["end"] - c["start"] >= 12
    assert B.clip_from_sentences(words, sents, 0, 0, cfg=_cfg(12.5, 30)) is None  # needs 4 extra
    assert B.clip_from_sentences(words, sents, 0, 4, cfg=_cfg(12.5, 30)) is not None  # 5 fit fine


def test_short_pick_at_transcript_end_rejected():
    words, sents = _transcript([SIX] * 8)
    assert B.clip_from_sentences(words, sents, 7, 7, cfg=_cfg(8, 30)) is None
    assert B.clip_from_sentences(words, sents, 6, 7, cfg=_cfg(8, 30)) is None


def test_long_pick_rejected_not_truncated():
    words, sents = _transcript([SIX] * 8)
    assert B.clip_from_sentences(words, sents, 0, 3, cfg=_cfg(5, 10)) is None     # 12.1 s > 10
    assert B.clip_from_sentences(words, sents, 0, 2, cfg=_cfg(5, 10)) is not None  # a truncation
    assert B.clip_from_sentences(words, sents, 0, 3, cfg=_cfg(5, 13)) is not None  # would have fit


def test_extension_overshooting_max_rejected():
    long_s = (" ".join(["word"] * 9) + " end.", 2.0)                   # ~21 s sentence
    words, sents = _transcript([SIX, long_s, SIX, SIX])
    assert B.clip_from_sentences(words, sents, 0, 0, cfg=_cfg(8, 15)) is None
    c = B.clip_from_sentences(words, sents, 0, 0, cfg=_cfg(8, 40))
    assert c["s1"] == 1 and c["end"] - c["start"] <= 40


@pytest.mark.parametrize("s0,s1", [(-1, 0), (0, 8), (3, 2), (8, 8), (-2, -1)])
def test_clip_from_sentences_out_of_range(s0, s1):
    words, sents = _transcript([SIX] * 8)
    assert B.clip_from_sentences(words, sents, s0, s1, cfg=_cfg(1, 100)) is None


def test_clip_durations_follow_sentence_structure():
    words, sents = _transcript([SIX] * 10)
    durs = {round(c["end"] - c["start"], 3) for s1 in range(1, 9)
            if (c := B.clip_from_sentences(words, sents, 0, s1, cfg=_cfg(5, 30)))}
    assert len(durs) == 8                                            # no fixed length


# ---------------------------------------------------------------------------
# tail-tag absorption
#
# Whisper sometimes punctuates a period before a conversational tail ("to be honest",
# "you know what I mean") that belongs to the previous thought. When the tag sentence
# starts within TAG_GAP_S of the chosen end sentence and its text is in TAIL_TAGS,
# extend the clip's end to include it — as long as max_dur_s still fits.
# ---------------------------------------------------------------------------
def _tag_transcript(tag_text, gap_after_body, body_text=None):
    """Body sentence(s) followed by `tag_text` after `gap_after_body`s. Uses SIX-word bodies so
    two bodies fit in a small duration band."""
    body = body_text or SIX
    words, sents = _transcript([body, body])
    t = float(sents[-1]["end"]) + float(gap_after_body)
    w0 = len(words)
    for tok in tag_text.split():
        words.append({"w": tok, "start": round(t, 3), "end": round(t + 0.25, 3)})
        t += 0.25 + 0.05
    sents.append({"i": len(sents), "start": words[w0]["start"], "end": words[-1]["end"],
                  "text": tag_text, "w0": w0, "w1": len(words) - 1})
    return words, sents


def test_tail_tag_absorbed_when_gap_and_max_dur_fit():
    words, sents = _tag_transcript("to be honest.", 0.3)
    c = B.clip_from_sentences(words, sents, 0, 1, cfg=_cfg(2, 30))
    assert c["s1"] == 2 and c["w1"] == sents[2]["w1"]                 # tag included
    assert c["end"] >= sents[2]["end"]
    cc = B.clean_clip(words, sents, 0, 1, cfg=_cfg(2, 30), edges_clean=True)
    assert cc["s1"] == 2 and cc["w1"] == sents[2]["w1"]


def test_tail_tag_normalizes_case_and_punctuation():
    words, sents = _tag_transcript("You Know, What I mean?", 0.3)     # -> "you know what i mean"
    c = B.clip_from_sentences(words, sents, 0, 1, cfg=_cfg(2, 30))
    assert c["s1"] == 2                                                # absorbed


def test_tail_tag_not_absorbed_when_gap_too_large():
    words, sents = _tag_transcript("to be honest.", 0.6)               # > TAG_GAP_S (0.45)
    c = B.clip_from_sentences(words, sents, 0, 1, cfg=_cfg(2, 30))
    assert c["s1"] == 1 and c["w1"] == sents[1]["w1"]
    cc = B.clean_clip(words, sents, 0, 1, cfg=_cfg(2, 30), edges_clean=True)
    assert cc["s1"] == 1


def test_tail_tag_not_absorbed_when_breaks_max_dur():
    words, sents = _tag_transcript("to be honest.", 0.3)
    body_only = B.clip_from_sentences(words, sents[:2], 0, 1, cfg=_cfg(2, 30))
    body_dur = body_only["end"] - body_only["start"]
    absorbed = B.clip_from_sentences(words, sents, 0, 1, cfg=_cfg(2, 30))
    tag_dur = absorbed["end"] - absorbed["start"]
    # cap fits the body but not the tag: absorption must be skipped, clip still returned
    cap = (body_dur + tag_dur) / 2.0
    assert body_dur < cap < tag_dur
    c = B.clip_from_sentences(words, sents, 0, 1, cfg=_cfg(2, cap))
    assert c is not None and c["s1"] == 1


def test_tail_tag_non_tag_short_sentence_not_absorbed():
    words, sents = _tag_transcript("This continues the argument.", 0.3)  # 4 words, not in TAIL_TAGS
    c = B.clip_from_sentences(words, sents, 0, 1, cfg=_cfg(2, 30))
    assert c["s1"] == 1
    cc = B.clean_clip(words, sents, 0, 1, cfg=_cfg(2, 30), edges_clean=True)
    assert cc["s1"] == 1


def test_tail_tag_not_absorbed_at_transcript_end():
    words, sents = _transcript([SIX, SIX])
    # if there is no following sentence, nothing to absorb even if the body ends short
    c = B.clip_from_sentences(words, sents, 0, 1, cfg=_cfg(2, 30))
    assert c["s1"] == 1


def test_tail_tag_absorbed_by_clean_clip_updates_end_and_note():
    """clean_clip absorbs the tag and its clean_end flag reflects the tag sentence."""
    words, sents = _tag_transcript("to be honest.", 0.3)
    cc = B.clean_clip(words, sents, 0, 1, cfg=_cfg(2, 30), edges_clean=True)
    assert cc["s1"] == 2 and cc["clean_end"] is True                   # tag ends on '.'
    words2, sents2 = _tag_transcript("man", 0.3)                       # unpunctuated, edges_clean=False
    cc2 = B.clean_clip(words2, sents2, 0, 1, cfg=_cfg(2, 30), edges_clean=False)
    assert cc2["s1"] == 2 and cc2["clean_end"] is False


# ---------------------------------------------------------------------------
# Criterion 9: iou / suppress
# ---------------------------------------------------------------------------
def _c(a, b, score):
    return {"start": float(a), "end": float(b), "score": score}


def test_iou_uses_shorter_clip():
    assert B.iou(_c(0, 100, 0), _c(10, 20, 0)) == 1.0                  # contained
    assert B.iou(_c(0, 10, 0), _c(5, 45, 0)) == pytest.approx(0.5)     # 5 / 10, not 5 / 45
    assert B.iou(_c(0, 10, 0), _c(10, 20, 0)) == 0.0                   # touching
    assert B.iou(_c(0, 10, 0), _c(12, 20, 0)) == 0.0
    assert B.iou(_c(5, 45, 0), _c(0, 10, 0)) == B.iou(_c(0, 10, 0), _c(5, 45, 0))


def test_suppress_drops_overlap_above_max():
    a, b, c, d = _c(0, 20, 90), _c(10, 30, 80), _c(14, 34, 70), _c(40, 60, 60)
    # b overlaps a by 0.5 (> 0.3) -> dropped; c overlaps a by exactly 0.3 -> kept
    assert B.suppress([d, c, b, a], max_overlap=0.3) == [a, c, d]
    assert B.suppress([d, c, b, a], max_overlap=0.29) == [a, d]
    assert B.suppress([d, c, b, a], max_overlap=0.6) == [a, b, d]   # c overlaps kept b by 0.8
    assert B.suppress([d, c, b, a], max_overlap=0.8) == [a, b, c, d]


def test_suppress_short_clip_inside_long_one_is_dropped():
    long_, short = _c(0, 60, 90), _c(20, 30, 50)                      # classic IoU 1/6, ours 1.0
    assert B.suppress([short, long_]) == [long_]


def test_suppress_ranks_by_score_and_caps_top_n():
    clips = [_c(i * 100, i * 100 + 30, s) for i, s in enumerate([10, 70, 40, 90, 20, None, 60])]
    ranked = B.suppress(clips, top_n=0)
    assert [c["score"] for c in ranked] == [90, 70, 60, 40, 20, 10, None]
    assert [c["score"] for c in B.suppress(clips, top_n=3)] == [90, 70, 60]
    assert len(B.suppress(clips, top_n=None)) == 7
    assert len(B.suppress(clips)) == 6                                 # default top_n
    assert B.suppress([]) == []


def test_suppress_higher_score_wins_regardless_of_order():
    lo, hi = _c(0, 30, 40), _c(5, 35, 95)
    assert B.suppress([lo, hi]) == [hi] and B.suppress([hi, lo]) == [hi]


# ---------------------------------------------------------------------------
# ends_cleanly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,ok", [
    ("It worked.", True), ("wait what?", True), ("Wow!", True), ("  It worked.  ", True),
    ("and then I went to the.", False), ("I was going to.", False), ("So we went and.", False),
    ("It worked", False), ("", False), ("They said, and", False),
    ('He said "it worked."', True), ("“It worked.”", True), ('"It worked"', False),
    ("It worked.)", True), ("It was because.", False), ("Heads up with.", False),
])
def test_ends_cleanly(text, ok):
    assert B.ends_cleanly(text) is ok


# ---------------------------------------------------------------------------
# sentences_around
# ---------------------------------------------------------------------------
def _grid_sents(n=7, step=5.0, length=4.0):
    return [{"i": k, "start": k * step, "end": k * step + length, "text": "x.", "w0": k, "w1": k}
            for k in range(n)]                         # starts 0,5,..,30  ends 4,9,..,34


def test_sentences_around_targets_peak_windows():
    s = _grid_sents()
    assert B.sentences_around(s, 22.0) == (2, 5)       # start ~10, end ~30
    assert B.sentences_around(s, 3.0) == (0, 1)        # nothing starts near -9: earliest
    assert B.sentences_around(s, 33.0) == (4, 6)       # only sentence 6 ends at/after 33


def test_sentences_around_respects_peak_side_constraints():
    s = _grid_sents()
    # target start 14.5 would pick sentence 3 (start 15) but it begins after the peak
    assert B.sentences_around(s, 12.0, pre_s=-2.5)[0] == 2
    # target end 17 would pick sentence 3 (end 19) but it ends before the peak
    assert B.sentences_around(s, 22.0, post_s=-5.0) == (2, 4)


def test_sentences_around_none_and_filtered_subset():
    s = _grid_sents()
    assert B.sentences_around(s[2:], 3.0) is None      # nothing starts before the peak
    assert B.sentences_around(s[:2], 20.0) is None     # nothing ends after the peak
    assert B.sentences_around([], 5.0) is None
    sub = s[3:]                                        # list positions, not "i"
    assert B.sentences_around(sub, 27.0) == (0, 3)     # "i" 3 and 6


# ---------------------------------------------------------------------------
# energy_clip (low-speech path)
# ---------------------------------------------------------------------------
MED = 1.0


def _env():
    env = np.full(120, 0.5)                   # quiet
    env[30:40] = np.linspace(1.2, 2.0, 10)    # ramp up
    env[40] = 5.0                             # peak
    env[41:55] = 2.0                          # sustained action ...
    env[45] = 0.3                             # ... with a 1 s lull (too short to end on)
    env[55:71] = 0.4                          # sustained quiet: natural end at 55
    env[71:91] = 2.0
    env[91:] = 0.45                           # next quiet stretch at 91
    return env


def _quiet_onsets(env, t0, fs, peak, median=MED, quiet_s=1.5):
    need = int(np.ceil(quiet_s / fs))
    out = []
    for i in range(int((peak - t0) / fs), len(env) - need + 1):
        if all(env[i:i + need] < median) and (i == 0 or env[i - 1] >= median):
            out.append(t0 + i * fs)
    return out


def _assert_envelope_clip(c, env, t0, fs, peak, min_dur, max_dur, pre=(20.0, 3.0)):
    assert c["boundary_type"] == "energy"
    assert min_dur <= c["end"] - c["start"] <= max_dur
    assert peak - pre[0] <= c["start"] <= peak - pre[1]
    below = [i for i in range(len(env)) if env[i] < MED and t0 + i * fs <= peak - pre[1]]
    if below:                                 # inside the last quiet frame before the build-up
        assert t0 + below[-1] * fs <= c["start"] <= t0 + (below[-1] + 1) * fs
    assert c["end"] >= peak
    if c["end"] in _quiet_onsets(env, t0, fs, peak):
        return "onset"
    lo = max(c["start"] + min_dur, peak)      # otherwise: the argmin of the allowed end range
    hi = min(t0 + len(env) * fs, c["start"] + max_dur)
    idx = [i for i in range(len(env)) if lo <= t0 + (i + 0.5) * fs <= hi]
    i = int((c["end"] - t0) / fs)
    assert i in idx and env[i] == env[idx].min()
    return "argmin"


def test_energy_clip_natural_end():
    env = _env()
    c = B.energy_clip(env, 0.0, 1.0, 40.0, MED, 15, 60)
    assert c["start"] == pytest.approx(29.5)  # last quiet second before the ramp
    assert c["end"] == 55.0                   # first >= 1.5 s quiet stretch (the 1 s lull ignored)
    assert _assert_envelope_clip(c, env, 0.0, 1.0, 40.0, 15, 60) == "onset"


def test_energy_clip_t0_offset():
    env = _env()
    c = B.energy_clip(env, 100.0, 1.0, 140.0, MED, 15, 60)
    assert c["start"] == pytest.approx(129.5) and c["end"] == 155.0


def test_energy_clip_too_short_moves_to_next_quiet_stretch():
    env = _env()
    c = B.energy_clip(env, 0.0, 1.0, 40.0, MED, 30, 75)
    assert c["end"] == 91.0
    assert _assert_envelope_clip(c, env, 0.0, 1.0, 40.0, 30, 75) == "onset"


def test_energy_clip_no_stretch_fits_uses_envelope_minimum():
    ends = []
    for k in (63, 68):
        env = _env()
        env[k] = 0.1
        c = B.energy_clip(env, 0.0, 1.0, 40.0, MED, 30, 45)
        assert _assert_envelope_clip(c, env, 0.0, 1.0, 40.0, 30, 45) == "argmin"
        assert c["end"] == pytest.approx(k + 0.5)
        ends.append(c["end"] - c["start"])
    assert ends[0] != ends[1]                 # follows the envelope, not start + constant


def test_energy_clip_fallback_end_never_before_peak():
    """Calm-before-the-storm: a quiet second just before the peak must not become the end."""
    env = np.full(100, 2.0)
    env[:22] = 0.5
    env[38] = 0.1
    env[39:42] = 5.0
    env[70] = 1.5                             # the quietest moment after the peak
    c = B.energy_clip(env, 0.0, 1.0, 40.0, MED, 10, 60)
    assert c["start"] == pytest.approx(21.5)
    assert c["end"] == pytest.approx(70.5)
    assert _assert_envelope_clip(c, env, 0.0, 1.0, 40.0, 10, 60) == "argmin"


def test_energy_clip_none_when_no_room():
    env = _env()
    assert B.energy_clip(env, 0.0, 1.0, 2.0, MED, 15, 60) is None     # no pre-roll
    short = np.full(30, 2.0)
    short[:20] = 0.5
    assert B.energy_clip(short, 0.0, 1.0, 25.0, MED, 15, 60) is None  # audio ends before min_dur
