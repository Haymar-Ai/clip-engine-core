"""Tests for clipengine/captions_hygiene.py (phase 2 item 3)."""
from clipengine import captions_hygiene as ch


def test_em_dash_fails_but_fixed():
    r = ch.check_caption("Watch this — you won't believe it! #Ad", require_ad_tag=True)
    assert not r.ok
    assert "em dash present" in r.reasons
    assert "—" not in r.fixed


def test_lowercase_ad_fails_but_fixed():
    r = ch.check_caption("Check #ad new drop", require_ad_tag=True)
    assert not r.ok
    assert any("lowercase" in reason for reason in r.reasons)
    assert "#Ad" in r.fixed
    assert "#ad " not in r.fixed    # the lowercase form is scrubbed


def test_missing_ad_tag_fails():
    r = ch.check_caption("No tag here", require_ad_tag=True)
    assert not r.ok
    assert any("missing #Ad" in reason for reason in r.reasons)


def test_pass_through_when_clean():
    r = ch.check_caption("All clean here #Ad @sponsor", require_ad_tag=True,
                         required=["@sponsor"])
    assert r.ok
    assert r.reasons == []


def test_required_element_check_is_case_insensitive():
    r = ch.check_caption("Love NORTH PIER tonight #Ad", required=["North Pier"])
    assert r.ok


def test_missing_required_element():
    r = ch.check_caption("Random post #Ad", required=["@mentionX"])
    assert not r.ok
    assert any("@mentionX" in reason for reason in r.reasons)


def test_optional_ad_tag_off():
    r = ch.check_caption("No disclosure needed", require_ad_tag=False)
    assert r.ok
