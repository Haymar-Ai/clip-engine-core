#!/usr/bin/env python3
"""TikTok caption builder + validator for the engine.

The caption text that will be posted alongside each clip is NOT free form — campaigns demand
specific required elements (brand mentions, hashtags, links, platform-tag disclosures) and will
reject (or hide) posts that drop them. On top of that, lowercase ``#ad`` is known to trip
TikTok's disclosure modal and hangs the post from auto-publishing; the valid casing is ``#Ad``.

This module keeps all of that in one place:

* ``build(campaign_rules, hook, streamer=...)`` returns the exact caption string the clip is
  posted with. The campaign_rules dict comes from the campaign record (template string + the
  required_elements list the gate validates on). No campaign-specific branches here — the
  engine never branches on a campaign name: the rules are data.
* ``validate(caption, required_elements)`` returns ``[errors]``. Empty list means ``post it as
  is``; non-empty means ``hold the clip``.

Required-element shapes (``required_elements`` on the campaign record):

    {"kind": "hashtag", "value": "#Ad", "case_insensitive": False}
    {"kind": "mention", "value": "@brand"}
    {"kind": "domain",  "value": "example.com"}
    {"kind": "phrase",  "value": "BRAND"}

Forbidden patterns are enforced globally for every campaign:

* lowercase ``#ad`` (even inside a longer tag like ``#adv``) → force ``#Ad``
* em dash ``—`` → use a plain ``-`` or two words. TikTok's text normalisation
  mangles em dashes on some devices and reviewers reject them on sight.
"""
import re
from pathlib import Path

EM_DASH = "—"
FORBIDDEN_HASHTAG_LOWER = "#ad"      # lowercase trips TikTok disclosure modal
PREFERRED_HASHTAG_AD = "#Ad"


class CaptionError(ValueError):
    """Raised by ``build`` when a template references a placeholder the caller did not pass."""


def build(rules, hook, **vars):
    """Render ``rules['template']`` with ``{hook}`` and any other ``vars``.

    ``rules`` is the ``caption`` sub-block of a campaign record, e.g.::

        {"template": "{hook}\\n\\n@brand #tag #Ad",
         "required_elements": [...], "allow_em_dash": False}

    * ``{hook}`` is always substituted from ``hook`` (stripped).
    * Other ``{var}`` placeholders come from ``**vars`` (e.g. ``streamer=``).
    * Missing placeholders raise :class:`CaptionError` so the pipeline holds the clip.
    * Lowercase ``#ad`` is auto-corrected to ``#Ad`` (safe: TikTok disclosure modal fix).
    * Em dashes are rejected unless ``rules.get('allow_em_dash')`` — the default is False.

    Returns the caption string with trailing whitespace stripped, no trailing newline.
    """
    tmpl = str((rules or {}).get("template") or "{hook}")
    hook = str(hook or "").strip()
    missing = []
    class _D(dict):
        def __missing__(self, key):
            missing.append(key)
            return "{" + key + "}"
    body = tmpl.format_map(_D(hook=hook, **{k: str(v or "").strip() for k, v in vars.items()}))
    if missing:
        raise CaptionError(f"caption template missing vars: {sorted(set(missing))}")
    body = _normalize_ad_hashtag(body)
    if not (rules or {}).get("allow_em_dash") and EM_DASH in body:
        raise CaptionError("em dash in caption; use '-' or rewrite")
    return body.rstrip()


def _normalize_ad_hashtag(text):
    """Replace the exact token ``#ad`` (any case) with ``#Ad`` while leaving longer tags like
    ``#advice`` alone. Tag boundary is start-of-string, whitespace, or newline on the left; and
    any non [A-Za-z0-9_] on the right."""
    def _sub(m):
        return m.group(1) + PREFERRED_HASHTAG_AD + m.group(3)
    return re.sub(r"(^|\s)(#ad)(?![A-Za-z0-9_])", r"\1" + PREFERRED_HASHTAG_AD, text,
                  flags=re.IGNORECASE | re.MULTILINE)


def validate(caption, required_elements):
    """Return a list of error strings; empty => the caption is good to post.

    Each required element must be present in the caption text (case-insensitive for ``hashtag``
    and ``mention`` kinds, case-sensitive for ``phrase`` by default)."""
    errs = []
    s = str(caption or "")
    for req in (required_elements or []):
        kind = (req.get("kind") or "").lower()
        val = str(req.get("value") or "").strip()
        if not val:
            continue
        case_sensitive = bool(req.get("case_sensitive"))
        if kind in ("hashtag", "mention"):
            if not _has_token(s, val, case_sensitive=case_sensitive):
                errs.append(f"missing {kind} {val}")
        elif kind == "domain":
            if val.lower() not in s.lower():
                errs.append(f"missing domain {val}")
        elif kind == "phrase":
            needle = val if case_sensitive else val.lower()
            hay = s if case_sensitive else s.lower()
            if needle not in hay:
                errs.append(f"missing phrase {val}")
        else:
            errs.append(f"unknown required element kind {kind!r}")
    # Forbidden patterns are enforced for every campaign — the auto-fix in ``build`` should have
    # caught them, but a hand-edited caption or a different code path might still slip through.
    if re.search(r"(^|\s)#ad(?![A-Za-z0-9_])", s):
        errs.append("lowercase #ad (use #Ad)")
    if EM_DASH in s:
        errs.append("em dash in caption")
    return errs


def _has_token(text, token, case_sensitive=False):
    """Match a hashtag or mention as a whole token (``#foo``, ``@foo``), not inside a word."""
    if not token:
        return False
    pat = re.escape(token) + r"(?![A-Za-z0-9_])"
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.search(pat, text, flags=flags) is not None


def build_from_campaign(campaign, hook, **vars):
    """Convenience wrapper used by :mod:`clipengine.engine`: pulls ``caption`` out of a
    :class:`Campaign` record (dict or dataclass) and renders it."""
    cap = _caption_block(campaign)
    if not cap:
        raise CaptionError("campaign has no caption rules")
    return build(cap, hook, **vars)


def validate_from_campaign(caption, campaign):
    cap = _caption_block(campaign)
    return validate(caption, (cap or {}).get("required_elements") or [])


def _caption_block(campaign):
    """Accept a dict, a Campaign dataclass with a ``.caption`` attr, or an arbitrary mapping
    with a ``caption`` key. Returns the caption dict or None."""
    if campaign is None:
        return None
    if isinstance(campaign, dict):
        return campaign.get("caption")
    return getattr(campaign, "caption", None)


if __name__ == "__main__":
    import argparse, json, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True, help="path to a campaign caption-rules JSON file")
    ap.add_argument("--hook", required=True)
    ap.add_argument("--streamer", default="")
    ap.add_argument("--out", default="-")
    a = ap.parse_args()
    rules = json.loads(Path(a.rules).read_text())
    body = build(rules, a.hook, streamer=a.streamer)
    (sys.stdout if a.out == "-" else open(a.out, "w")).write(body + "\n")
