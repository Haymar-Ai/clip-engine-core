"""the gate catches what the 10-20 eye-check rejected.

Five checks, each on synthetic inputs here (the real-clip calibration is in the item-2 report):
  1. flat edge band (clip 108: bottom 216 px solid black);
  2. graphics sliced by a crop edge (126/127/128 hudband tiles, 133/135/137 broadcast layouts);
  3. burned captions vs an independent second ASR (109/110/125/135 garbled captions);
  4. hook grounded in the clip, a missing hook fails (109 wrong hook, 113 hook=None);
  5. the ending (109 ends on "Oh", 124 "because" 0.17 s before the cut, 114 cut at the window
     edge mid-thought).
"""
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from clipengine import quality_gate as qg
from clipengine.render import recompose


def W(text, t0, step=0.3):
    """Words list from a sentence, one word per `step` s from t0."""
    return [{"w": w, "start": round(t0 + i * step, 3), "end": round(t0 + i * step + step * 0.8, 3)}
            for i, w in enumerate(text.split())]


# ---------------------------------------------------------------- 1. flat edge band

def test_edge_band_fails_a_fixed_black_band_and_passes_a_wandering_dark_backdrop():
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(10):
        g = rng.integers(40, 200, (960, 540)).astype(np.uint8)
        g[-107:] = 0                                           # clip 108: 11.1 % of 960
        frames.append(qg.flat_edge_depths(g))
    v = qg.edge_band_verdict(frames, 540, 960)
    assert v["passed"] is False and v["band_frac"]["bottom"] == pytest.approx(107 / 960, abs=1e-3)
    assert "bottom" in v["errors"][0]
    # 10-16 clip 30: a dark backdrop flat over ~10 % of the top, but at a different depth in
    # every frame (it moves with the speaker and the noise)
    wander = [{"top": d, "bottom": 0, "left": 0, "right": 0}
              for d in (95, 104, 112, 99, 108, 93, 101, 110, 97, 106)]
    assert qg.edge_band_verdict(wander, 540, 960)["passed"] is True


def test_edge_band_thresholds_side_bands_use_the_width():
    side = [{"top": 0, "bottom": 0, "left": 20, "right": 0}] * 10      # 20/540 = 3.7 %
    assert qg.edge_band_verdict(side, 540, 960)["passed"] is True
    side = [{"top": 0, "bottom": 0, "left": 23, "right": 0}] * 10      # 4.3 %
    assert qg.edge_band_verdict(side, 540, 960)["passed"] is False
    assert qg.edge_band_verdict([], 540, 960)["passed"] is None


def _encode(tmp_path, name, vf):
    out = tmp_path / name
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=540x960:rate=10:duration=2", "-vf", vf, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def test_measure_edge_bands_on_rendered_pixels(tmp_path):
    banded = _encode(tmp_path, "banded.mp4", "crop=540:852:0:0,pad=540:960:0:0:black")
    clean = _encode(tmp_path, "clean.mp4", "null")
    m = qg.measure_edge_bands(banded)
    assert m["passed"] is False and m["band_frac"]["bottom"] >= 0.1
    assert qg.measure_edge_bands(clean)["passed"] is True
    assert qg.measure_edge_bands(tmp_path / "missing.mp4")["passed"] is None


# ---------------------------------------------------------------- 2. graphics sliced by a crop

SW, SH = 1280, 720
LOWER_THIRD = {"box": [244, 658, 1036, 688], "frac": 1.0, "banner": True}
TEXTURE = {"box": [22, 300, 436, 420], "frac": 0.9, "banner": False}      # a shirt, no text
OCR = [(260, 662, 400, 686, "Ravens", 0), (410, 662, 520, 686, "beat", 0),  # in LOWER_THIRD
       (260, 662, 400, 686, "Ravens", 1), (410, 662, 520, 686, "beat", 1),
       (260, 662, 400, 686, "Ravens", 2), (410, 662, 520, 686, "beat", 2)]


def _scan(*graphics):
    return {"src_w": SW, "src_h": SH, "frames": 20, "graphics": list(graphics)}


def _view(x, w, t0=0.0, t1=20.0, y=0, h=SH):
    return {"t0": t0, "t1": t1, "x": x, "y": y, "w": w, "h": h}


def test_a_crop_window_through_a_burned_in_text_line_is_a_slice():
    s = qg.crop_slices(_scan(LOWER_THIRD), [_view(855, 404)], 20.0, words=OCR)   # 10-15 clip 5
    assert len(s) == 1 and s[0]["how"] == "crop" and s[0]["share"] == 1.0
    assert s[0]["alnum"] >= qg.SLICE_MIN_ALNUM
    # the whole line inside, or wholly outside: not a slice
    assert qg.crop_slices(_scan(LOWER_THIRD), [_view(200, 900)], 20.0, words=OCR) == []
    assert qg.crop_slices(_scan(LOWER_THIRD), [_view(0, 230, h=600)], 20.0, words=OCR) == []


def test_crop_rows_through_a_graphic_are_a_slice_too():
    s = qg.crop_slices(_scan(LOWER_THIRD), [_view(0, SW, h=670)], 10.0, words=OCR)
    assert s and s[0]["how"] == "crop"                          # 10-20 clip 135's crawl


def test_text_running_into_the_frame_edge_counts_when_that_edge_is_on_screen():
    crawl = {"box": [0, 680, SW, 708], "frac": 1.0, "banner": True}   # touches both sides
    words = [(0, 682, 140, 704, "FOLLOW", i) for i in range(3)]
    whole = qg.crop_slices(_scan(crawl), [], 20.0, words=words)       # canvas = whole frame
    assert whole and whole[0]["how"] == "frame" and whole[0]["share"] == 1.0
    # a 9:16 window in the middle slices it instead (crop); one off the frame edge, frame cut
    assert qg.crop_slices(_scan(crawl), [_view(400, 404)], 20.0, words=words)[0]["how"] == "crop"


def test_texture_without_ocr_text_does_not_count():
    v = [_view(300, 404)]
    assert qg.crop_slices(_scan(TEXTURE), v, 20.0, words=[]) == []      # OCR read nothing
    assert qg.crop_slices(_scan(TEXTURE), v, 20.0, words=None)          # unfiltered: counts


def test_ocr_lines_find_text_the_line_detector_missed():
    # 10-20 clip 135: the crawl "HOUSTON IS NOW 2-4 ..." is cut by the trimmed crop's bottom row
    words = [(19, 634, 163, 661, "HOUSTON", i) for i in range(3)] + \
            [(171, 634, 279, 661, "IS NOW", i) for i in range(2)]
    lines = qg.ocr_text_lines(words)
    assert len(lines) == 1 and lines[0]["frac"] == 1.0 and lines[0]["box"][:2] == [19, 634]
    s = qg.crop_slices(_scan(), [_view(18, 368, h=654)], 30.0, words=words)
    assert s and s[0]["src"] == "ocr" and s[0]["how"] == "crop"


def test_slice_share_is_time_and_presence_weighted():
    g2 = {"box": [900, 20, 1270, 60], "frac": 0.5, "banner": True}
    words = OCR + [(920, 24, 1100, 56, "SPONSORED", 0)]
    views = [_view(855, 404, 0.0, 10.0), _view(100, 404, 10.0, 20.0)]
    s = qg.crop_slices(_scan(LOWER_THIRD, g2), views, 20.0, words=words)
    shares = sorted(x["share"] for x in s)
    assert shares == [0.25, 1.0]          # g2: frac 0.5 x half the clip; the lower third: both
                                          # windows slice it, so the whole clip


def test_slice_verdict_crop_cap_and_total_cap():
    crop = [{"share": 0.46, "how": "crop"}]                    # 10-20 clip 128
    assert qg.crop_slice_verdict(crop)["passed"] is False
    assert qg.crop_slice_verdict([{"share": 0.4, "how": "crop"}])["passed"] is True
    frame = [{"share": 0.9, "how": "frame"}]                   # one HUD number at the edge
    assert qg.crop_slice_verdict(frame)["passed"] is True
    clutter = [{"share": 1.0, "how": "frame"}, {"share": 0.5, "how": "frame"}]   # 137
    v = qg.crop_slice_verdict(clutter)
    assert v["passed"] is False and v["sliced_share"] == 1.5 and v["crop_share"] == 0.0


def test_measure_crop_slices_needs_views_and_source():
    assert qg.measure_crop_slices("x.mp4", 0, 10, None)["passed"] is None
    v = [_view(1, 10)]
    assert qg.measure_crop_slices("/nonexistent.mp4", 0, 10, v)["passed"] is None
    m = qg.measure_crop_slices(__file__, 0, 20, [_view(855, 404)], scan=_scan(LOWER_THIRD),
                               words=OCR)
    assert m["passed"] is False and m["crop_share"] == 1.0
    m = qg.measure_crop_slices(__file__, 0, 20, [_view(100, 1000)], scan=_scan(LOWER_THIRD),
                               words=OCR)
    assert m["passed"] is True and m["sliced_share"] == 0.0


def test_crop_views_per_layout():
    track = {"layout": "track", "src_w": SW, "src_h": SH,
             "segments": [{"t0": 0.5, "t1": 8.0, "x": 100}, {"t0": 8.0, "t1": 20.0, "x": 700}]}
    v = recompose.crop_views(track, 20.0)
    assert [(x["t0"], x["t1"], x["x"], x["w"]) for x in v] == [(0.0, 8.0, 100, 404),
                                                            (8.0, 20.0, 700, 404)]
    assert recompose.crop_views({"layout": "general", "src_w": SW, "src_h": SH}, 5) == []
    full = {"layout": "track", "src_w": 360, "src_h": 640,           # 9:16 source: whole frame
            "segments": [{"t0": 0.0, "t1": 5.0, "x": 0}]}
    assert recompose.crop_views(full, 5.0) == []
    hb = {"layout": "hudband", "src_w": SW, "src_h": SH, "cw": 300, "y_off": 108, "eff_h": 612,
          "band_height": 400, "segments": [{"t0": 0.0, "t1": 10.0, "x": 500}],
          "band_items": [{"name": "pot", "src": [935, 530, 220, 160], "dst": [0, 0, 1, 1],
                          "intervals": [(2.0, 6.0)]}]}
    v = recompose.crop_views(hb, 10.0)
    assert {"t0": 2.0, "t1": 6.0, "x": 935, "y": 530, "w": 220, "h": 160} in v
    assert v[0]["y"] == 108 and v[0]["w"] == 300
    gs = {"layout": "gamestack", "src_w": SW, "src_h": SH, "top_h": 1344,
          "game": [578, 380, False], "cam": [180, 96, 28, 66, False]}
    v = recompose.crop_views(gs, 10.0)
    assert len(v) == 2 and v[0]["x"] == pytest.approx(380 + (578 - 720 * 1080 / 1344) / 2, abs=1)


# ---------------------------------------------------------------- 3. captions vs second ASR

SENT = ("and something i think people misunderstand is people working in these labs have "
        "taken this problem very seriously from the beginning")


def test_matching_captions_pass():
    cap = W(SENT, 0.0)
    v = qg.caption_agreement_verdict(cap, W(SENT.replace("labs", "labs,"), 0.0), 20.0)
    assert v["passed"] is True and v["precision"] == 1.0 and v["coverage"] == 1.0


def test_one_misheard_word_is_tolerated():
    cap = W(SENT.replace("beginning", "begging"), 0.0)          # clip 116 class residual
    assert qg.caption_agreement_verdict(cap, W(SENT, 0.0), 20.0)["passed"] is True


def test_garbled_captions_fail():
    # clip 109: "I don't like Ashley way 160 it says they wait in that so I don't know they can't
    # wait" where the second ASR hears "How much does Ashley weigh? 160, it says. They weighed
    # in that. So I don't know what they gain weight."
    cap = W("hey drake i love you whats up with his hair color i dont like it bro i dont like "
            "ashley way 160 it says they wait in that so i dont know they cant wait he does "
            "whats wrong with his hair he looks like ice face oh", 0.0)
    heard = W("hey drake i love you brother drake i love you whats up with his hair color i dont "
              "like it brother how much does ashley weigh 160 it says they weighed in that so i "
              "dont know what they gain weight ashley looks huge yeah he does what the fuck is "
              "wrong with his hair he looks like ice face dumbass chadwick oh he hurt him he hurt "
              "him he hurt him oh my god", 0.0, step=0.25)
    v = qg.caption_agreement_verdict(cap, heard, 25.0)
    assert v["passed"] is False
    assert any("carry" in e for e in v["errors"])                # speech left uncaptioned


def test_a_garbled_stretch_fails_even_when_the_rest_agrees():
    # clip 125 class: crosstalk both models half-hear; 6 of 12 caption tokens confirmed
    good = SENT + " " + SENT + " " + SENT
    bad = "its not a percent the joke is the key like the creep yeah"
    cap = W(good + " " + bad, 0.0)
    heard = W(good + " its not a person but the joke is the thing you agree yeah", 0.0)
    v = qg.caption_agreement_verdict(cap, heard, 60.0)
    assert v["precision"] >= qg.CAPTION_AGREE_MIN and v["coverage"] >= qg.CAPTION_COVERAGE_MIN
    assert v["passed"] is False and any("garbled" in e for e in v["errors"])
    assert v["worst_window"] < qg.CAPTION_AGREE_WINDOW_MIN


def test_an_uncaptioned_stretch_fails():
    # 10-19 clip 70: "you're wearing Nike sneakers, I'm participating in" never captioned
    head = SENT + " " + SENT
    cap = W(head + " well i mean youre wearing capitalism because i live in a capitalist "
                   "society", 0.0)
    heard = W(head + " well i mean youre wearing youre wearing nike sneakers im participating "
                     "in capitalism because i live in a capitalist society", 0.0)
    v = qg.caption_agreement_verdict(cap, heard, 60.0)
    assert v["precision"] == 1.0 and v["coverage"] >= qg.CAPTION_COVERAGE_MIN
    assert v["passed"] is False and v["worst_heard_window"] < qg.CAPTION_AGREE_WINDOW_MIN
    assert any("uncaptioned stretch" in e for e in v["errors"])


def test_caption_agreement_edge_cases():
    assert qg.caption_agreement_verdict([], [], 10)["passed"] is None
    assert qg.caption_agreement_verdict(W("hello there", 0), [], 10)["passed"] is None
    assert qg.caption_agreement_verdict([], W("hello there", 0), 10)["passed"] is False
    # words the second ASR heard AFTER the cut are not captions it should find
    cap = W("we invested at one point", 0.0)
    heard = W("we invested at one point", 0.0) + W("and then it grew a lot", 10.0)
    assert qg.caption_agreement_verdict(cap, heard, 5.0)["coverage"] == 1.0


def test_norm_tokens_and_fuzzy_match():
    assert qg.norm_tokens([{"w": "300 -pound"}, {"w": "Anthropic's"}, {"w": "ten,"}]) == \
        ["300", "pound", "anthropics", "10"]
    assert qg._tok_match("anthropics", "anthropiks") and not qg._tok_match("him", "them")
    assert qg.align_tokens(["a", "b", "c"], ["a", "x", "c"]) == [True, False, True]


# ---------------------------------------------------------------- 4. hook grounding

CLIP = W("like we are not waiting around for 300 pound sam anymore at least not currently "
         "and i guess i cant completely write it off", 0.0)


def test_missing_hook_fails():
    for hook in (None, "", "   "):
        v = qg.hook_support_verdict(hook, CLIP, [], 10.0)
        assert v["passed"] is False and "no hook" in v["errors"][0]


def test_without_a_judge_the_lexical_floor_decides():
    assert qg.hook_support_verdict("Why he gave up on 300 pounds", CLIP, [], 20.0)["passed"]
    v = qg.hook_support_verdict("Viewer reveals they were his opponent", CLIP, [], 20.0)
    assert v["passed"] is False and v["ground_share"] == 0.0 and v["decided_by"] == "lexical"
    assert "not grounded" in v["errors"][0]
    # one content word is enough for the floor (paraphrased hooks, 10-20 clips 114 / 124)
    assert qg.hook_support_verdict("Why waiting quietly kills growth", CLIP, [], 20.0)["passed"]


def test_hook_words_after_the_cut_do_not_ground_it():
    late = [dict(w, start=w["start"] + 30, end=w["end"] + 30) for w in CLIP]
    assert qg.hook_support_verdict("Why he gave up on 300 pounds", late, [], 20.0)["passed"] \
        is None                                                     # nothing said in the clip


def test_the_judge_decides_when_it_answers():
    no = lambda h, t: {"supported": False, "quote": "", "reason": "greets Drake; 'his' is the fighter"}
    yes = lambda h, t: {"supported": True, "quote": "opposite side", "reason": "paraphrase"}
    boom = lambda h, t: (_ for _ in ()).throw(RuntimeError("cli down"))
    hook = "Why he gave up on 300 pounds"
    v = qg.hook_support_verdict(hook, CLIP, [], 20.0, judge=no)      # grounded words, wrong claim
    assert v["passed"] is False and v["decided_by"] == "judge"       # (clip 109)
    # 10-15 clip 3: "Why I bet against the crowd" shares no word with the clip, judge: fair
    v = qg.hook_support_verdict("Why I bet against the crowd", CLIP, [], 20.0, judge=yes)
    assert v["passed"] is True and v["ground_share"] == 0.0
    v = qg.hook_support_verdict(hook, CLIP, [], 20.0, judge=boom)
    assert v["passed"] is True and "error" in v["judge"] and v["decided_by"] == "lexical"
    v = qg.hook_support_verdict("Viewer reveals they were his opponent", CLIP, [], 20.0,
                                judge=lambda h, t: {"error": "no JSON"})
    assert v["passed"] is False                                      # floor still applies


def test_hook_ground_judge_prompt_and_parsing(monkeypatch):
    from clipengine import select_semantic
    seen = {}

    def fake_cli(prompt, model, timeout, opts):
        seen["prompt"], seen["model"] = prompt, model
        return '```json\n{"supported": false, "quote": "", "reason": "wrong subject"}\n```'
    monkeypatch.setattr(select_semantic, "_claude_cli", fake_cli)
    judge = qg.make_hook_ground_judge({"semantic_model": "opus"})
    v = judge("Drake's new hair color divides the room", "Hey Drake. What's up with his hair?")
    assert v == {"supported": False, "quote": "", "reason": "wrong subject"}
    assert "WRONG SUBJECT" in seen["prompt"] and "Drake's new hair" in seen["prompt"]
    assert seen["model"] == "opus"


# ---------------------------------------------------------------- 5. ending

def test_complete_sentence_with_silence_after_passes():
    heard = W("and i guess i cant completely write it off.", 0.0)     # ends 2.66 s
    v = qg.ending_verdict(heard, heard, 3.2, tail_s=2.0)
    assert v["passed"] is True and v["last_word"] == "off."


def test_captions_ending_on_oh_fail():                                # clip 109
    cap = W("he looks like ice face Oh", 0.0)
    heard = W("he looks like ice face. oh my god!", 0.0)
    v = qg.ending_verdict(heard, cap, 2.5, tail_s=2.0)
    assert v["passed"] is False and "'Oh'" in v["errors"][0]


def test_next_word_starting_before_the_cut_fails_but_inside_the_fade_passes():
    base = W("you are never actually suffering from losers.", 0.0)   # ends at 2.04
    dur = 2.5
    because = {"w": "because", "start": dur - 0.17, "end": dur + 0.1}  # clip 124
    v = qg.ending_verdict(base + [because], base, dur, tail_s=1.6)
    assert v["passed"] is False and "because" in v["errors"][0]
    so = {"w": "So", "start": dur - 0.05, "end": dur + 0.2}            # clip 123: in the fade
    v = qg.ending_verdict(base + [so], base, dur, tail_s=2.0)
    assert v["passed"] is True and v["next_word"] == "So"


def test_a_comma_from_the_pass_that_heard_past_the_cut_beats_the_caption_period():
    # clip 124: captions "...suffering from losers." / second ASR "losers, because like ..."
    cap = W("you are never actually suffering from losers.", 0.0)
    heard = W("you are never actually suffering from losers,", 0.0) + \
        [{"w": "because", "start": 2.54, "end": 2.8}]
    v = qg.ending_verdict(heard, cap, 2.5, tail_s=1.6)
    assert v["passed"] is False and "continue after 'losers,'" in v["errors"][0]
    # the same words with the ASR's period (or none at all) and a pause: complete
    heard[-2] = dict(heard[-2], w="losers.")
    assert qg.ending_verdict(heard, cap, 2.5, tail_s=1.6)["passed"] is True
    heard[-2] = dict(heard[-2], w="losers")
    assert qg.ending_verdict(heard, cap, 2.5, tail_s=1.6)["passed"] is True


def test_a_word_whisper_starts_inside_a_pause_is_the_next_word_not_the_last():
    # 10-19 clip 69 (approved): whisper put "And" 0.21 s before the cut; the audio there is
    # 19-48 dB under the clip's level, speech resumes 0.08 s after the cut
    base = W("that is your lineup.", 0.0)                             # ends 1.14
    dur = 1.6
    heard = base + [{"w": "And", "start": dur - 0.21, "end": dur - 0.05},
                    {"w": "why", "start": dur + 0.15, "end": dur + 0.3}]
    times = [round(-1.0 + 0.02 * i, 2) for i in range(75)]
    silent = [(-40.0 if -0.18 <= t < 0.08 else 5.0) for t in times]
    v = qg.ending_verdict(heard, base, dur, tail_s=2.0, energy=(times, silent))
    assert v["passed"] is True and v["last_word"] == "lineup." and v["next_word"] == "And"
    assert v["moved_by_energy"] == 1
    # the same words over speech: "And" really starts before the cut -> fail
    loud = [5.0 for _ in times]
    v = qg.ending_verdict(heard, base, dur, tail_s=2.0, energy=(times, loud))
    assert v["passed"] is False and "'And'" in v["errors"][0]
    assert qg.ending_verdict(heard, base, dur, tail_s=2.0)["passed"] is False   # no audio


def test_a_pass_over_the_rendered_clip_alone_does_not_judge_punctuation():
    heard = W("and the dog", 0.0)                       # whisper drops the final period at
    v = qg.ending_verdict(heard, [], 1.5, tail_s=None)  # an abrupt end of audio
    assert v["passed"] is True and "not heard" in v["note"]
    heard = W("that was good, and", 0.0)                # fillers still fail
    assert qg.ending_verdict(heard, [], 2.0, tail_s=None)["passed"] is False


def test_cut_energy_reads_the_source_around_the_cut(tmp_path):
    src = tmp_path / "s.wav"
    sr = 16000
    t = np.arange(int(4.0 * sr)) / sr
    a = np.where((t < 2.8) | (t > 3.2), np.sin(2 * np.pi * 220 * t) * 0.3, 0.0)
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((a * 32767).astype(np.int16).tobytes())
    times, db = qg.cut_energy(src, 1.0, 3.0)           # cut at 3.0 s, silence 2.8-3.2 s
    by_t = dict(zip([round(x, 2) for x in times], db))
    assert by_t[-0.5] > -3 and by_t[-0.1] < -40 and by_t[0.1] < -40 and by_t[0.3] > -3


def test_filler_last_word_fails_whatever_its_punctuation():
    heard = W("and that was great. and", 0.0)
    assert qg.ending_verdict(heard, heard, 2.0, tail_s=2.0)["passed"] is False
    heard = W("that was great. Oh.", 0.0)
    assert qg.ending_verdict(heard, heard, 2.0, tail_s=2.0)["passed"] is False


def test_mid_sentence_and_mid_word_fail():
    heard = W("it says a lot is not going on in", 0.0)                 # clip 110 per the plan
    v = qg.ending_verdict(heard, heard, 3.0, tail_s=2.0)
    assert v["passed"] is False and "dangling" in " ".join(v["errors"])
    heard = W("it says a lot of boys are not going there tonight", 0.0)
    v = qg.ending_verdict(heard, heard, 3.6, tail_s=2.0)
    assert v["passed"] is False and v["errors"] == ["ends mid-sentence: last word 'tonight'"]
    cut = W("we invested at one", 0.0) + [{"w": "point.", "start": 1.3, "end": 1.9}]
    v = qg.ending_verdict(cut, cut, 1.6, tail_s=2.0)
    assert v["passed"] is False and "inside the word" in v["errors"][0]
    # whisper stretching a segment's last word over the music after it is not a cut word
    long = W("it is hard to run", 0.0) + [{"w": "well", "start": 1.5, "end": 4.0}]
    assert "inside the word" not in " ".join(
        qg.ending_verdict(long, W("it is hard to run.", 0.0), 2.3, tail_s=2.0)["errors"])


def test_terminal_punctuation_from_either_transcript():
    heard = W("they get back up and make something of that", 0.0)     # ASR dropped the period
    cap = W("they get back up and make something of that.", 0.0)
    assert qg.ending_verdict(heard, cap, 3.0, tail_s=2.0)["passed"] is True
    assert qg.ending_verdict(heard, heard, 3.0, tail_s=2.0)["passed"] is False


def test_cut_at_the_window_edge_is_unverifiable():                    # 10-19 68/83, 10-20 114
    heard = W("and i look at you know legora we invested at one point.", 0.0)
    v = qg.ending_verdict(heard, heard, 3.6, tail_s=0.06)
    assert v["passed"] is False and "unverifiable" in v["errors"][-1]
    v = qg.ending_verdict(heard, heard, 3.6, tail_s=None)             # no source: not judged
    assert v["passed"] is True and "no source" in v["note"]


# ---------------------------------------------------------------- second ASR

def _wav(path, secs=3.0, sr=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.sin(np.arange(int(secs * sr)) / 8) * 3000).astype(np.int16).tobytes())
    return path


def test_second_asr_runs_on_the_window_plus_tail_and_returns_clip_times(tmp_path):
    from clipengine import second_asr
    media = _wav(tmp_path / "src.wav", secs=10.0)
    seen = {}

    def runner(wav, model):
        with wave.open(str(wav)) as w:
            seen["secs"] = w.getnframes() / w.getframerate()
        seen["model"] = model
        return [{"w": "hello.", "start": 0.1, "end": 0.5}]
    rec = second_asr.transcribe(media, 2.0, 6.0, model="small", tail_s=2.0, runner=runner,
                                cache_dir=tmp_path / "c")
    assert seen == {"secs": pytest.approx(6.0, abs=0.05), "model": "small"}
    assert rec["dur"] == 4.0 and rec["tail_s"] == 2.0 and rec["words"][0]["w"] == "hello."
    rec = second_asr.transcribe(media, 2.0, 9.0, model="small", tail_s=2.0, runner=runner,
                                cache_dir=tmp_path / "c")
    assert rec["tail_s"] == pytest.approx(1.0, abs=0.05)          # clamped at the media end


def test_second_asr_must_differ_from_the_caption_models():
    from clipengine import second_asr
    cfg = {"select": {"transcribe_model": "base.en", "refine_model": "small.en"},
           "render": {"caption_model": "small.en"}}
    assert second_asr.independent("small.en", cfg)
    assert second_asr.independent("base.en", cfg)
    assert second_asr.independent("small", cfg) is None
    assert second_asr.independent(second_asr.settings()["model"]) is None   # shipped config


# ---------------------------------------------------------------- wiring: every vertical

def test_clip_checks_with_an_injected_second_asr(tmp_path):
    words = W("i guess i cant completely write it off.", 100.0)         # source time
    heard = W("i guess i can't completely write it off.", 0.0)
    rep = qg.clip_checks(tmp_path / "none.mp4", words=words, start_s=100.0, end_s=103.0,
                           hook="Why he can't write it off", asr=({"words": heard}, 2.0))
    assert set(qg.CLIP_CHECK_KEYS) <= set(rep)
    assert rep["caption_asr"]["passed"] is True and rep["ending"]["passed"] is True
    assert rep["hook_grounding"]["passed"] is True
    assert rep["edge_bands"]["passed"] is None and rep["crop_slices"]["passed"] is None
    rep = qg.clip_checks(tmp_path / "none.mp4", words=words, start_s=100.0, end_s=103.0,
                           hook=None, asr=({"words": heard}, 2.0))
    assert rep["hook_grounding"]["passed"] is False


def test_failed_second_asr_is_informational_but_caption_rules_still_run(tmp_path, monkeypatch):
    def broken(*a, **k):
        raise subprocess.TimeoutExpired("whisper", 1)
    monkeypatch.setattr(qg, "run_second_asr", broken)
    words = W("he looks like ice face Oh", 50.0)
    rep = qg.clip_checks(tmp_path / "none.mp4", words=words, start_s=50.0, end_s=52.0,
                           hook="Drake's new hair color")
    assert rep["caption_asr"]["passed"] is None and "TimeoutExpired" in rep["second_asr"]["error"]
    assert rep["ending"]["passed"] is False                      # captions end on "Oh"


def test_decide_fails_on_every_clip_check_measure():
    for k in qg.CLIP_CHECK_KEYS:
        verdict, reason = qg.decide({k: {"passed": False, "errors": ["x"]}})
        assert verdict == "fail" and reason.startswith(f"{k}: x")
    assert qg.decide({k: {"passed": None} for k in qg.CLIP_CHECK_KEYS})[0] == "pass"


def test_full_gate_runs_the_clip_checks(monkeypatch):
    monkeypatch.setattr(qg, "measure_frame_gate", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(qg, "measure_av_sync", lambda p: {"passed": None})
    monkeypatch.setattr(qg, "measure_visual_quality", lambda *a, **k: {"passed": True})
    seen = {}

    def fake(video_path, **kw):
        seen.update(kw)
        return {"hook_grounding": {"passed": False, "errors": ["no hook: the clip has no hook"]}}
    monkeypatch.setattr(qg, "clip_checks", fake)
    rep = qg.gate("clip.mp4", run_overlay_check=False, hook=None, src_media="w.mp4",
                  crop_views=[], start_s=1.0, end_s=9.0)
    assert seen["src_media"] == "w.mp4" and seen["crop_views"] == []
    assert rep["verdict"] == "fail" and "hook_grounding: no hook" in rep["drop_reason"]
    off = qg.gate("clip.mp4", run_overlay_check=False, run_clip_checks=False)
    assert not set(qg.CLIP_CHECK_KEYS) & set(off)
