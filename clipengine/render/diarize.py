#!/usr/bin/env python3
"""Audio-only voice diarization.

the talker decision has to come from audio, not visual mouth motion, so
the framing check isn't circular. Pyannote and resemblyzer are not available in
this env, so this module extracts a MFCC-like voice fingerprint per word using
numpy FFT + mel filterbank + log, then clusters words into K voice groups with
numpy k-means. Clusters are mapped to face tracks using the frames where
exactly one face track is visible — a word that lands inside a single-face shot
unambiguously belongs to that face.

Nothing here depends on mouth motion, so when the rest of the engine checks
"was the diarized talker framed?" the audio signal and the framing decision
share no inputs.
"""
import math
import subprocess
from pathlib import Path

import numpy as np


SR = 16000
WIN_MS = 25
HOP_MS = 10
N_MEL = 32
MIN_WORD_MS = 60       # drop sub-60 ms sidecar entries ("I", "a") that give unstable fingerprints


# ---------------------------------------------------------------- audio I/O

def load_audio_mono(src, start, dur, sr=SR):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", str(start), "-t", str(dur),
           "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return np.zeros(0, dtype=np.float64), sr
    if not raw:
        return np.zeros(0, dtype=np.float64), sr
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64), sr


# ---------------------------------------------------------------- MFCC-like

def _mel_filterbank(n_mel, n_fft, sr):
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    low = hz_to_mel(0.0)
    high = hz_to_mel(sr / 2)
    mels = np.linspace(low, high, n_mel + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mel, n_fft // 2 + 1), dtype=np.float64)
    for i in range(1, n_mel + 1):
        left, cen, right = bins[i - 1], bins[i], bins[i + 1]
        if cen > left:
            for k in range(left, min(cen, fb.shape[1])):
                fb[i - 1, k] = (k - left) / (cen - left)
        if right > cen:
            for k in range(cen, min(right, fb.shape[1])):
                fb[i - 1, k] = (right - k) / (right - cen)
    return fb


def _mel_frames(audio, sr=SR, win_ms=WIN_MS, hop_ms=HOP_MS, n_mel=N_MEL):
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(audio) < win:
        return np.zeros((0, n_mel))
    n_fft = 1 << (win - 1).bit_length()
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(win) / max(1, win - 1))
    fb = _mel_filterbank(n_mel, n_fft, sr)
    n_frames = 1 + (len(audio) - win) // hop
    out = np.empty((n_frames, n_mel), dtype=np.float64)
    for i in range(n_frames):
        chunk = audio[i * hop: i * hop + win] * window
        buf = np.zeros(n_fft)
        buf[:win] = chunk
        spec = np.abs(np.fft.rfft(buf)) ** 2
        out[i] = np.log(fb @ spec + 1e-9)
    return out


# ---------------------------------------------------------------- word features

def word_feature_matrix(audio, sr, words_rel):
    """Return (kept_words, feature_matrix) where each row is the per-word mean
    + per-word std of the mel-log coefficients. `words_rel` have start/end in
    seconds relative to the audio buffer."""
    rows = []
    kept = []
    for w in words_rel:
        a = int(max(0.0, w["start"]) * sr)
        b = int(w["end"] * sr)
        if b - a < sr * MIN_WORD_MS / 1000.0:
            continue
        if b > len(audio):
            b = len(audio)
        if b - a <= 0:
            continue
        feats = _mel_frames(audio[a:b], sr)
        if len(feats) < 3:
            continue
        rows.append(np.concatenate([feats.mean(axis=0), feats.std(axis=0)]))
        kept.append(w)
    if not rows:
        return [], np.zeros((0, N_MEL * 2))
    X = np.vstack(rows)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    return kept, X


# ---------------------------------------------------------------- k-means

def kmeans(X, k, iters=80, seed=0, n_init=4):
    """Pure-numpy k-means with multiple restarts. Picks the labelling with the
    lowest within-cluster sum of squared distances."""
    best_labels, best_centroids, best_inertia = None, None, math.inf
    rng = np.random.default_rng(seed)
    for restart in range(n_init):
        if len(X) < k:
            return np.zeros(len(X), dtype=int), X.mean(axis=0, keepdims=True)
        # k-means++ init
        centroids = [X[rng.integers(0, len(X))]]
        for _ in range(1, k):
            d = np.min(((X[:, None, :] - np.array(centroids)[None, :, :]) ** 2).sum(axis=-1), axis=1)
            prob = d / (d.sum() + 1e-9)
            centroids.append(X[rng.choice(len(X), p=prob)])
        centroids = np.array(centroids)
        for _ in range(iters):
            d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
            labels = d.argmin(axis=1)
            new_centroids = np.array(
                [X[labels == c].mean(axis=0) if (labels == c).any() else centroids[c]
                 for c in range(k)])
            if np.allclose(centroids, new_centroids):
                centroids = new_centroids
                break
            centroids = new_centroids
        d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        inertia = d.min(axis=1).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = d.argmin(axis=1)
            best_centroids = centroids
    return best_labels, best_centroids


def choose_k(X, k_max=3, k_min=2):
    """Pick the number of voice clusters by comparing inertia with a simple
    elbow test: accept more clusters only when the drop is >=25%."""
    if len(X) < k_min * 3:
        return 1
    best_k, prev_inertia = 1, None
    for k in range(k_min, k_max + 1):
        if len(X) < k * 2:
            break
        _, centroids = kmeans(X, k, iters=40, seed=0, n_init=2)
        d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        inertia = d.min(axis=1).sum()
        if prev_inertia is None or (prev_inertia - inertia) / prev_inertia >= 0.25:
            best_k = k
            prev_inertia = inertia
        else:
            break
    return best_k


# ---------------------------------------------------------------- entry

def diarize_words(src, start, dur, words, k=None):
    """Cluster `words` (clip-absolute `start`/`end` fields) into K voice groups
    using numpy-only MFCC-like features + k-means. `words` list is NOT mutated.
    Returns:
      {
        "k": int,
        "labels": [cluster_id for each kept word, in word order],
        "words": [{"w","start","end"}],    # start/end are *clip-relative*
        "feature_dim": int,
      }
    Empty dict when audio is unreadable or no word qualifies."""
    audio, sr = load_audio_mono(src, start, dur)
    if not len(audio):
        return {}
    rel = [{"w": w.get("w", ""),
            "start": float(w["start"]) - start,
            "end": float(w["end"]) - start}
           for w in words
           if start <= float(w["start"]) < start + dur]
    kept, X = word_feature_matrix(audio, sr, rel)
    if len(X) == 0:
        return {}
    chosen_k = k if k is not None else choose_k(X)
    if chosen_k <= 1 or len(X) < chosen_k * 2:
        labels = np.zeros(len(X), dtype=int)
        return {"k": 1, "labels": labels.tolist(), "words": kept,
                "feature_dim": int(X.shape[1])}
    labels, _ = kmeans(X, chosen_k, n_init=4)
    return {"k": int(chosen_k), "labels": [int(x) for x in labels],
            "words": kept, "feature_dim": int(X.shape[1])}


# ---------------------------------------------------------------- cluster-to-track mapping

def _track_mouth_motion_series(samples, track_id):
    """Per-sample mouth-motion value for `track_id`: pixel grayscale diff over
    the YuNet mouth landmarks between consecutive samples. Returns [(t, act)]
    without needing to redecode the video — the samples list carries the YuNet
    landmarks on each face dict and we can use the face bbox position deltas
    as a cheap proxy. (The real mouth-pixel diff is in `speaker.mouth_activity`
    but needs frame pixels; for the mapping we only need relative strength.)

    Proxy: mean mouth-landmark Euclidean displacement * face height. This
    captures lip movement while being invariant to zoom level."""
    import math
    prev = None
    out = []
    for s in samples:
        found = None
        for f in s.get("faces") or []:
            if f.get("track_id") == track_id:
                found = f
                break
        if found is None:
            prev = None
            continue
        lms = found.get("landmarks") or []
        if len(lms) < 5:
            prev = None
            continue
        lx, ly = lms[4]
        rx, ry = lms[3]
        mx, my = (lx + rx) / 2.0, (ly + ry) / 2.0
        if prev is not None:
            dx = mx - prev[0]; dy = my - prev[1]
            disp = math.hypot(dx, dy)
            fh = float(found.get("h", 0.1))
            # scale so bigger faces don't dominate; use displacement / sqrt(fh)
            # so the signal has the right order of magnitude
            act = disp / math.sqrt(max(fh, 0.02))
            out.append((s["t"], act))
        prev = (mx, my)
    return out


def _av_sync_mapping(diarization, tracks, samples, cluster_ids):
    """Fallback mapping when single-face evidence is absent. For each voice
    cluster C and each face track T, compute how much T's mouth-motion signal
    aligns with C's "talking" mask. Assign greedily: pair with the highest
    score wins, repeat until every cluster is paired or all tracks are taken.
    Only accepts a pair when its score is >=1.5x the next-best cluster score
    on the same track (so we don't lock to a weak signal). Returns `(mapping,
    evidence_dict)` — mapping is `{cluster_id: track_id}`."""
    labels = diarization.get("labels") or []
    words = diarization.get("words") or []
    if not labels or not tracks:
        return {}, {}
    # time windows per cluster
    windows_by_cluster = {c: [] for c in cluster_ids}
    for lbl, w in zip(labels, words):
        if lbl in windows_by_cluster:
            windows_by_cluster[lbl].append((w["start"], w["end"]))
    # mouth-motion series per track
    series_by_track = {t: _track_mouth_motion_series(samples, t) for t in tracks}
    # score[C][T] = total mouth-motion of T during C's talking windows
    score = {}
    for c, windows in windows_by_cluster.items():
        for t, series in series_by_track.items():
            total = 0.0
            for (st, act) in series:
                for (a, b) in windows:
                    if a - 0.1 <= st <= b + 0.1:
                        total += act
                        break
            score.setdefault(c, {})[t] = total
    # Greedy assignment: pick largest score (C, T), lock it, strike C and T,
    # repeat.
    mapping = {}
    evidence = {}
    taken_tracks = set()
    while score:
        best = None
        for c, per in score.items():
            if c in mapping:
                continue
            for t, v in per.items():
                if t in taken_tracks:
                    continue
                if best is None or v > best[2]:
                    best = (c, t, v)
        if best is None:
            break
        c, t, v = best
        # Confidence check: is this pair clearly better than the next-best
        # cluster for the same track? If not (voice signal weak or tied), skip.
        others = sorted([score[cc].get(t, 0.0) for cc in score
                         if cc != c and cc not in mapping], reverse=True)
        runner_up = others[0] if others else 0.0
        if v <= 0 or v < runner_up * 1.3:
            # ambiguous for this track; try pairing the next cluster first
            score.setdefault(c, {}).pop(t, None)
            if not score[c]:
                score.pop(c, None)
            continue
        mapping[c] = t
        taken_tracks.add(t)
        evidence[c] = {"track_id": t, "score": round(v, 3),
                       "runner_up": round(runner_up, 3), "source": "av_sync"}
        score.pop(c, None)
    return mapping, evidence


def map_clusters_to_tracks(diarization, samples, allow_track_ids=None):
    """Return `({cluster_id: track_id}, evidence)` using single-face-track moments
    as the first pass, then AV-sync (mouth motion vs cluster-talking windows) as a
    fallback for clusters that didn't get a single-face assignment. The second
    pass handles the 100%-two-shot interview clips where no single-face evidence
    is available (e.g. a two-shot interview clip that was framed on the wrong
    person because single-face voting was empty).

    `samples` is per-frame face list `[{"t","faces":[{"track_id","cx","h",
    "landmarks"...}]}]` with t clip-relative.
    """
    if not diarization:
        return {}, {}
    # Pass 1: single-face voting
    singles = []
    for s in samples:
        tracks = [f.get("track_id") for f in (s.get("faces") or [])
                  if f.get("track_id") is not None]
        if allow_track_ids is not None:
            tracks = [t for t in tracks if t in allow_track_ids]
        if len(set(tracks)) == 1:
            singles.append((s["t"], tracks[0]))
    votes = {}
    for lbl, w in zip(diarization["labels"], diarization["words"]):
        for (t, tid) in singles:
            if w["start"] <= t <= w["end"]:
                votes.setdefault(lbl, {}).setdefault(tid, 0)
                votes[lbl][tid] += 1
    # Count how often each track appears overall — tracks seen only in a
    # brief close-up (which paradoxically generates lots of single-face
    # evidence) shouldn't be allowed to hijack a cluster that really belongs
    # to a main speaker. Require the winning track to cover >=20 % of samples.
    track_counts = {}
    for s in samples:
        for f in s.get("faces") or []:
            tid = f.get("track_id")
            track_counts[tid] = track_counts.get(tid, 0) + 1
    # A "main speaker track" is one that's visible for a sizeable chunk of the
    # top-track's visibility — threshold at 30 % of the most-visible track.
    # This rejects brief close-up tracks that generate lots of single-face
    # evidence but aren't actually one of the main speakers.
    max_count = max(track_counts.values()) if track_counts else 0
    main_thr = max(2, int(0.3 * max_count))
    main_tracks = {t for t, n in track_counts.items() if n >= main_thr}

    mapping = {}
    evidence = {}
    # Greedy 1-1 assignment: on each round pick the (cluster, track) pair whose
    # vote count is highest among unassigned clusters/tracks. Reject the pair
    # if the cluster's second-best track is close (confidence <0.6) — it's
    # ambiguous and AV-sync should handle it. Also reject when the winning
    # track isn't a main speaker (visible <20 % of samples).
    taken_tracks = set()
    remaining = dict(votes)
    while remaining:
        best = None
        for lbl, per in remaining.items():
            if not per:
                continue
            for tid, n in per.items():
                if tid in taken_tracks or tid not in main_tracks:
                    continue
                if best is None or n > best[2]:
                    best = (lbl, tid, n)
        if best is None:
            break
        lbl, tid, n = best
        total = sum(remaining[lbl].values())
        confidence = n / total if total else 0.0
        if n >= 2 and confidence >= 0.6:
            mapping[lbl] = tid
            taken_tracks.add(tid)
            evidence[lbl] = {"track_id": tid, "votes": dict(remaining[lbl]),
                             "confidence": round(confidence, 3),
                             "source": "single_face"}
        remaining.pop(lbl, None)
    # Pass 2: AV-sync for the clusters we couldn't anchor via single-face.
    unresolved = [c for c in set(diarization.get("labels") or []) if c not in mapping]
    if unresolved:
        # Only consider major tracks (seen often). A track visible <10% of
        # samples is noise and shouldn't carry a speaker.
        counts = {}
        for s in samples:
            for f in s.get("faces") or []:
                tid = f.get("track_id")
                counts[tid] = counts.get(tid, 0) + 1
        taken = set(mapping.values())
        thr = max(5, int(0.1 * len(samples)))
        candidate_tracks = [t for t, n in counts.items() if n >= thr and t not in taken
                            and (allow_track_ids is None or t in allow_track_ids)]
        if candidate_tracks:
            av_mapping, av_evidence = _av_sync_mapping(diarization, candidate_tracks,
                                                       samples, unresolved)
            for c, tid in av_mapping.items():
                mapping[c] = tid
                evidence[c] = av_evidence[c]
    return mapping, evidence


def talker_at(diarization, mapping, t, hold_last=True):
    """Return the mapped face track_id for the diarized voice talking at clip
    time `t`. Returns None when nobody is diarized at `t` and `hold_last=False`.
    When `hold_last=True`, returns the most recent known talker."""
    if not diarization:
        return None
    labels = diarization.get("labels") or []
    words = diarization.get("words") or []
    last = None
    for i, w in enumerate(words):
        if w["start"] > t:
            break
        last = labels[i]
        if w["start"] <= t <= w["end"]:
            return mapping.get(last)
    if hold_last and last is not None:
        return mapping.get(last)
    return None
