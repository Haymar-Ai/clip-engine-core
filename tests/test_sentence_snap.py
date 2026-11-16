"""sentence-boundary snapping in the window selector. The 10-20 reason_codes 100, 109,
110, 114, 124 and 127 were cut mid-sentence: run-on ends whisper punctuated, a continuation
("Because ...") right after the end, an "Oh" ending, and window edges treated as sentence edges.
Word timings below are the shipped base.en transcripts of those windows around the cut."""
from __future__ import annotations

from clipengine import boundaries
from clipengine import transcript


def W(*triples):
    return [{"w": w, "start": s, "end": e} for w, s, e in triples]


# 10-20 clip 110 (Ny6SIEhgMJc_w00 24.89-37.82, "last sentence cut"): "tonight." then "Hey" 0.08 s
WORDS_110 = W(("They", 31.0, 31.2), ("need", 31.2, 31.4), ("a", 31.4, 31.5), ("VVS", 31.5, 31.9),
              ("checker.", 31.9, 32.4), ("Real", 33.2, 33.5), ("talk,", 33.5, 33.8),
              ("they", 33.8, 34.0), ("need", 34.0, 34.16), ("a", 34.16, 34.24),
              ("VVS", 34.24, 34.66), ("reader.", 34.66, 34.98), ("It", 35.56, 35.92),
              ("says", 35.92, 36.08), ("a", 36.08, 36.24), ("lot", 36.24, 36.4),
              ("of", 36.4, 36.5), ("more", 36.5, 36.68), ("is", 36.68, 36.8), ("not", 36.8, 36.96),
              ("going", 36.96, 37.16), ("on", 37.16, 37.4), ("there", 37.4, 37.54),
              ("tonight.", 37.54, 37.78), ("Hey", 37.86, 38.22), ("Drake.", 38.22, 38.42),
              ("I", 39.42, 39.78), ("love", 39.78, 40.0), ("you.", 40.0, 40.3))
# 10-20 clip 124 (H8QfAfCbOXs_w01, ends on "losers." with "Because" right after)
WORDS_124 = W(("you're", 31.58, 31.7), ("never", 31.7, 31.9), ("actually", 31.9, 32.32),
              ("suffering", 32.32, 32.82), ("from", 32.82, 33.1), ("losers.", 33.1, 33.34),
              ("Because", 33.7, 33.7), ("you're", 33.7, 33.9), ("like", 33.9, 34.0),
              ("the", 34.0, 34.14), ("true", 34.14, 34.26), ("losers,", 34.26, 34.58),
              ("you'll", 34.6, 34.94), ("get", 34.94, 35.06), ("out.", 35.06, 35.18))


def ends(words, **kw):
    sents = transcript.sentences(words, gap_s=0.7)
    return {s["text"].split()[-1]: e for s, e in zip(sents, transcript.clean_edges(sents, **kw)[1])}


def test_run_on_end_is_not_clean_under_strict_ends():
    loose = ends(WORDS_110)
    strict = ends(WORDS_110, strict_ends=True, min_end_gap_s=0.25)
    assert loose["tonight."] is True                       # what cut 110
    assert strict["tonight."] is False                     # "Hey" 0.08 s later: a run-on
    assert strict["reader."] is True and strict["checker."] is True   # real pauses


def test_continuation_opener_makes_the_end_a_run_on():
    strict = ends(WORDS_124, strict_ends=True, min_end_gap_s=0.25)
    assert ends(WORDS_124)["losers."] is True              # what cut 124
    assert strict["losers."] is False                      # "Because" 0.36 s later
    assert strict["out."] is True                          # last sentence of a whole source


def test_interjection_is_never_a_clean_end():
    words = W(("That", 0.0, 0.3), ("was", 0.3, 0.5), ("wild.", 0.5, 0.9), ("Oh.", 2.0, 2.3))
    assert ends(words)["Oh."] is True
    assert ends(words, strict_ends=True)["Oh."] is False


def test_edges_clean_pair_sets_the_window_edges_apart():
    sents = transcript.sentences(WORDS_110, gap_s=0.7)
    cs, ce = transcript.clean_edges(sents, edges_clean=(False, True))
    assert cs[0] is False and ce[-1] is True
    cs, ce = transcript.clean_edges(sents, edges_clean=(True, False))
    assert cs[0] is True and ce[-1] is True                 # "you." punctuated: trusted (loose)
    cs, ce = transcript.clean_edges(sents, edges_clean=(True, False), strict_ends=True)
    assert ce[-1] is False             # strict: whisper puts that period where the window stops


def test_window_edges_need_silence_at_the_window_edge():
    from clipengine import engine
    assert engine.window_edges("abc", WORDS_110, 60.0) is True            # whole source
    # a window whose speech runs from 0.05 s to 0.1 s before its end: both edges are fragments
    words = W(("cards", 0.05, 0.4), ("now.", 0.4, 0.7), ("Next", 2.0, 2.3), ("one", 34.5, 35.9))
    assert engine.window_edges("kflnuYj3sV4_w01", words, 36.0) == (False, False)
    words = W(("So", 1.0, 1.2), ("done.", 1.2, 1.5), ("Bye.", 30.0, 30.5))
    assert engine.window_edges("kflnuYj3sV4_w01", words, 36.0) == (True, True)
    assert engine.vertical_cfg("general")["strict_ends"] is True
    assert engine.vertical_cfg("general")["min_end_gap_s"] == engine.MIN_END_GAP_S


def test_clean_clip_on_a_window_moves_off_its_edge_fragments():
    """127: the pick started at the window's first sentence (a fragment, speech from 0.0 s);
    114: it ended on the window's last sentence 0.16 s before the window end."""
    words = W(("needed", 0.05, 0.4), ("to", 0.4, 0.5), ("call.", 0.5, 0.9),
              ("He", 1.8, 2.0), ("found", 2.0, 2.4), ("the", 2.4, 2.5), ("card.", 2.5, 3.0),
              *[(f"w{i}", 3.6 + 0.4 * i, 3.9 + 0.4 * i) for i in range(40)],
              ("Then", 20.4, 20.6), ("he", 20.6, 20.8), ("shoved.", 20.8, 21.4),
              ("We", 22.2, 22.4), ("invested", 22.4, 22.9), ("at", 22.9, 23.0), ("one", 23.0, 23.2))
    words[-1]["w"] = "point."
    sents = transcript.sentences(words, gap_s=0.7)
    cfg = {"min_dur_s": 10, "max_dur_s": 60, "strict_ends": True, "min_end_gap_s": 0.25}
    edges = (False, False)                     # window_edges: speech at 0.05 s and to 23.2/23.36
    c = boundaries.clean_clip(words, sents, 0, len(sents) - 1, cfg=cfg, edges_clean=edges)
    assert words[c["w0"]]["w"] == "He" and c["clean_start"]          # not the "needed" fragment
    assert words[c["w1"]]["w"] == "shoved." and c["clean_end"]       # not the window's last words
    loose = boundaries.clean_clip(words, sents, 0, len(sents) - 1, cfg=cfg)
    assert words[loose["w0"]]["w"] == "needed"                        # what the old selector did
