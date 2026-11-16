"""Code `slur` (10-20 clip 125): a slur is muted and masked in the
captions; ordinary words and profanity are left alone."""
from clipengine import slurs


def test_whole_token_match_with_plain_endings_only():
    assert slurs.is_slur("faggot,") and slurs.is_slur("Retards") and slurs.is_slur("N*GGA") is False
    assert slurs.is_slur("niggas") and slurs.is_slur("retarded.")
    for ok in ("snigger", "Spice", "raccoon", "fuck", "shit", "coonhound", "Gooks?"[:4] + "e"):
        assert not slurs.is_slur(ok), ok


def test_mask_keeps_the_first_letter_and_punctuation():
    assert slurs.mask("Faggot,") == "F*****," and slurs.mask("retard.") == "r*****."


def test_censor_masks_and_mutes_only_inside_the_clip():
    words = [{"w": "you", "start": 9.0, "end": 9.2}, {"w": "retard", "start": 10.0, "end": 10.4},
             {"w": "man.", "start": 10.4, "end": 10.7}, {"w": "faggot", "start": 30.0, "end": 30.3}]
    out, spans, found = slurs.censor(words, start_s=5.0, end_s=20.0)
    assert found == ["retard"] and out[1]["w"] == "r*****" and out[3]["w"] == "faggot"
    assert spans == [(round(5.0 - slurs.MUTE_PAD_S, 3), round(5.4 + slurs.MUTE_PAD_S, 3))]
    assert words[1]["w"] == "retard"                                   # input untouched
