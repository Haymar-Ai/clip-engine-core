"""TikTok caption builder + validator tests. Pure, no media."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
from clipengine import captions_tiktok as ct


TRADING_RULES = {
    # An old template used an em dash on the brand line; em dashes are banned
    # at onboarding review, so the engine template uses a plain `-` instead.
    "template": ("{hook}\n\n{streamer} live on Example Trading Club - "
                 "example-trading.com\n\n@exampletv @joinexample\n\n#ETC #Example #Ad"),
    "required_elements": [
        {"kind": "mention", "value": "@exampletv"},
        {"kind": "mention", "value": "@joinexample"},
        {"kind": "domain",  "value": "example-trading.com"},
        {"kind": "hashtag", "value": "#Ad"},
    ],
}

PRODUCT_RULES = {
    "template": "{hook}\n\nThis is a FOLDY folding desk. @foldy\n\n#FOLDY #desk #foldingdesk #ad",
    "required_elements": [
        {"kind": "mention", "value": "@foldy"},
        {"kind": "phrase",  "value": "FOLDY"},
        {"kind": "hashtag", "value": "#Ad"},
    ],
}


def test_build_substitutes_hook_and_streamer():
    s = ct.build(TRADING_RULES, hook="He traded until 3am", streamer="Sam")
    assert "He traded until 3am" in s
    assert "Sam live on Example Trading Club" in s


def test_build_missing_variable_raises():
    rules = {"template": "{hook} {streamer} {missing}", "allow_em_dash": True}
    with pytest.raises(ct.CaptionError):
        ct.build(rules, hook="x", streamer="y")


def test_lowercase_ad_hashtag_is_auto_corrected_to_capital_A():
    # Even though the brand template uses `#ad`, the builder swaps it to `#Ad`.
    s = ct.build(PRODUCT_RULES, hook="He built a $60K home")
    assert "#Ad" in s
    # No bare lowercase `#ad` left on a token boundary
    assert " #ad" not in s and "\n#ad" not in s and not s.endswith("#ad")
    # Longer tags should not be touched
    rules2 = {"template": "x #advice", "allow_em_dash": False}
    assert ct.build(rules2, hook="x").endswith("#advice")


def test_em_dash_is_rejected_by_default():
    rules = {"template": "{hook} — tagline", "required_elements": []}
    with pytest.raises(ct.CaptionError):
        ct.build(rules, hook="x")
    # explicit allow lets it through
    rules2 = {**rules, "allow_em_dash": True}
    assert "—" in ct.build(rules2, hook="x")


def test_validate_all_required_elements_present_passes():
    s = ct.build(TRADING_RULES, hook="H", streamer="Sam")
    assert ct.validate(s, TRADING_RULES["required_elements"]) == []


def test_validate_missing_required_element_fails_with_reason():
    s = "He traded until 3am\n\n@joinexample\n\n#Ad"   # @exampletv missing
    errs = ct.validate(s, TRADING_RULES["required_elements"])
    assert any("@exampletv" in e for e in errs)
    assert any("example-trading.com" in e for e in errs)


def test_validate_lowercase_ad_in_caption_is_an_error():
    s = "He pulled a card\n\n@cardshop #unpacked #ad"
    assert any("lowercase #ad" in e for e in ct.validate(s, []))


def test_validate_em_dash_in_caption_is_an_error():
    s = "He traded until 3am — big one\n\n#Ad"
    assert any("em dash" in e for e in ct.validate(s, []))


def test_hashtag_matches_as_whole_token_not_substring():
    s = "I love #advice but no advertising"
    assert ct.validate(s, [{"kind": "hashtag", "value": "#Ad"}])[0].startswith("missing")


def test_build_from_campaign_dict():
    camp = {"id": "x", "caption": PRODUCT_RULES}
    s = ct.build_from_campaign(camp, hook="He built a desk")
    assert "FOLDY" in s and "@foldy" in s and "#Ad" in s


def test_build_from_campaign_without_caption_raises():
    with pytest.raises(ct.CaptionError):
        ct.build_from_campaign({"id": "x"}, hook="h")
