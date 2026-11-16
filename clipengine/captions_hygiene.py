"""Caption hygiene: validate a post caption before it is used.

Three hard rules:
  - "#Ad" (capital A, always); lowercase "#ad" hangs TikTok posting on the disclosure modal
  - no em dash (`—`): hand-written feel, but TikTok mobile replaces it with "-" inconsistently
  - every campaign-required name / mention / link must be present (campaign record carries the
    list)

captions_tiktok.py builds captions from a template; this module only VALIDATES an existing
caption (a hook plus whatever suffix the caller adds). The engine itself never posts: this is
for a caller that does.

Returns `(ok: bool, fixed: str, reasons: list[str])`. `fixed` is a cheap auto-fix (lowercase
#ad -> #Ad, em dash -> hyphen); callers can accept it, or treat `ok=False` as drop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

EM_DASH = "—"
EN_DASH = "–"

# accepts the correctly-cased tag at a word boundary; rejects anything else
_HASHTAG_AD = re.compile(r"(?<![A-Za-z0-9_])(?:#[Aa][Dd])(?![A-Za-z0-9_])")


@dataclass(frozen=True)
class HygieneResult:
    ok: bool
    fixed: str
    reasons: list[str]


def check_caption(caption: str, required: Optional[Iterable[str]] = None,
                  *, require_ad_tag: bool = True) -> HygieneResult:
    """Validate `caption`. `required` is a sequence of literal strings (case-insensitive) each of
    which must appear somewhere in the caption; typical values are campaign mentions / links.
    If any required element is missing, return `ok=False` with a reason per missing item.

    `require_ad_tag`: when the campaign treats the post as paid promotion, "#Ad" must be
    present (capital A). We auto-upcase lowercase `#ad` into `fixed` but still fail the check
    unless the raw caption already contained the uppercase form (we don't trust auto-fixes to
    always render in a posting tool's queue; the writer should get it right).
    """
    reasons: list[str] = []
    fixed = caption or ""
    # Em dash: fix in `fixed`, flag in reasons (a clean caption is wanted; the auto-fix is a
    # courtesy for logs, not a reason to ship the original).
    if EM_DASH in fixed:
        reasons.append("em dash present")
        fixed = fixed.replace(EM_DASH, "-")
    if EN_DASH in fixed:
        reasons.append("en dash present")
        fixed = fixed.replace(EN_DASH, "-")
    # #Ad check
    if require_ad_tag:
        if "#Ad" in caption:
            pass
        elif _HASHTAG_AD.search(caption):
            reasons.append("lowercase #ad (must be #Ad)")
            fixed = _HASHTAG_AD.sub("#Ad", fixed)
        else:
            reasons.append("missing #Ad disclosure")
    missing_required: list[str] = []
    low = caption.lower()
    for req in required or ():
        if req and req.lower() not in low:
            missing_required.append(req)
    for r in missing_required:
        reasons.append(f"missing required element: {r!r}")
    return HygieneResult(ok=(not reasons), fixed=fixed, reasons=reasons)
