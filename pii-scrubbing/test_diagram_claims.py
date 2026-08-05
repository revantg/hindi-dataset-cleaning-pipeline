#!/usr/bin/env python3
"""Assert that the SVG diagrams state facts the code actually produces.

The diagrams hard-code example strings and outcomes. That is fine while it stays
true, and a silent lie the moment a pattern, threshold or allowlist rule changes.
This script re-derives every load-bearing claim from `pii_techniques` and from
Python's own `re`, so a drift breaks the build instead of the reader's trust.

    .venv/bin/python pii-scrubbing/test_diagram_claims.py
"""

from __future__ import annotations

import re
import sys
import xml.dom.minidom
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pii_diagrams as diagrams  # noqa: E402
import pii_techniques as pii  # noqa: E402

MOBILE_DEVANAGARI = "\u096f\u096e\u096a\u096b\u0966\u0967\u0968\u0969\u096a\u096b"  # ९८४५०१२३४५
CITATION = "\u0968.\u096f.\u0968\u0967.\u0967"  # २.९.२१.१
CASCADE_DOC = (
    "\u092e\u094b\u092c\u093e\u0907\u0932 " + MOBILE_DEVANAGARI + " \u092a\u0930"
    "   \u00b7   \u0906\u0927\u093e\u0930 9999 4105 7058"
    "   \u00b7   \u0927\u0930\u094d\u092e\u0938\u0942\u0924\u094d\u0930 (" + CITATION + ")"
)

failures: list[str] = []


def check(claim: str, condition: bool) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {claim}")
    if not condition:
        failures.append(claim)


print("\ndevanagari_digit_trap.svg")
# Row 1: the ASCII-only leading class makes a real mobile invisible.
check(
    r"[6-9]\d{9} does NOT match the Devanagari mobile",
    re.search(r"[6-9]\d{9}", MOBILE_DEVANAGARI) is None,
)
# Row 2: the very same digits DO match a bare \d run -- the asymmetry.
check(
    r"\d{10} DOES match the same Devanagari mobile",
    re.search(r"\d{10}", MOBILE_DEVANAGARI) is not None,
)
check(
    "folding maps the mobile to 9845012345",
    pii.fold_digits(MOBILE_DEVANAGARI) == "9845012345",
)
check(
    "folding preserves length, so offsets survive",
    pii.folding_is_offset_safe(CASCADE_DOC),
)
check(
    "the scripture citation folds to the dotted quad 2.9.21.1",
    pii.fold_digits(CITATION) == "2.9.21.1",
)
_prefix = "\u092e\u094b\u092c\u093e\u0907\u0932 "  # "मोबाइल "
check(
    "the diagram's stated span offsets 7..17 are correct",
    len(_prefix) == 7 and len(_prefix + MOBILE_DEVANAGARI) == 17,
)

print("\ncascade_internals.svg")
result = pii.scrub(CASCADE_DOC)
accepted = {(s.category, s.text_of(CASCADE_DOC)) for s in result.accepted}
suppressed = {(s.category, s.text_of(CASCADE_DOC)) for s in result.suppressed}

check("the Devanagari phone is ACCEPTED", ("phone_in", MOBILE_DEVANAGARI) in accepted)
check("the Aadhaar-shaped run is ACCEPTED", ("aadhaar_like", "9999 4105 7058") in accepted)
check("the scripture citation is SUPPRESSED", ("ipv4", CITATION) in suppressed)
check(
    "the masked text is exactly what the MASKED box shows",
    result.masked_text
    == "\u092e\u094b\u092c\u093e\u0907\u0932 [PHONE] \u092a\u0930   \u00b7   "
       "\u0906\u0927\u093e\u0930 [ID]   \u00b7   "
       "\u0927\u0930\u094d\u092e\u0938\u0942\u0924\u094d\u0930 (" + CITATION + ")",
)
check(
    "the suppression reason is the native-digit rule, not a checksum",
    any(
        "native-script" in s.reason
        for s in result.suppressed
        if s.category == "ipv4"
    ),
)
check(
    "no layer proposes a personal NAME, as the MISSED box claims",
    not any(s.category in {"person", "name"} for s in result.accepted + result.suppressed),
)
check("the drawn threshold of 0.50 is the function default", pii.scrub.__defaults__[0] == 0.50)
check(
    "layer 3's drawn arithmetic matches Presidio's constants",
    (pii.CONTEXT_BOOST, pii.CONTEXT_FLOOR) == (0.35, 0.40),
)

print("\ntechnique_2 / checksum claims")
check("Verhoeff accepts the sample Aadhaar-shaped run", pii.verhoeff_valid("999941057058"))
check("Luhn accepts the ISO reference vector", pii.luhn_valid("79927398713"))
check("Luhn rejects a corrupted vector", not pii.luhn_valid("79927398710"))

print("\nall diagrams")
for name, build in diagrams.DIAGRAMS.items():
    svg = build()
    try:
        xml.dom.minidom.parseString(svg)
        parsed = True
    except Exception:
        parsed = False
    check(f"{name} is well-formed XML", parsed)
    # `{9}` inside a displayed regex is legitimate; an unresolved f-string
    # placeholder is a Python identifier or attribute lookup in braces.
    leaked = re.findall(r"\{[A-Za-z_][A-Za-z0-9_.\[\]'\"]*\}", svg)
    check(f"{name} carries no unresolved f-string placeholder {leaked or ''}", not leaked)

print()
if failures:
    print(f"{len(failures)} CLAIM(S) NO LONGER TRUE:")
    for claim in failures:
        print("  -", claim)
    raise SystemExit(1)
print("every diagram claim verified against the implementation")
