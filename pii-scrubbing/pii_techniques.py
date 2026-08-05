"""Detector primitives for the Session 4 PII-scrubbing technique study.

This module holds the mechanics so the Marimo notebook can stay a narrative.
Everything here is deterministic, CPU-only, and has no network dependency.

Design rules that the notebook relies on:

* Detection runs on a **folded copy** of the text while every offset stays valid
  against the **original** string. Digit folding is one-to-one per code point, so
  offsets are preserved exactly and the emitted text is never the folded copy.
* Every detector returns spans, never a rewritten string. Masking is a separate,
  auditable step over resolved spans.
* Shape matches and decisions are different things. A pattern proposes a
  candidate; checksums, allowlists and context promote or reject it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Unicode digit folding
# ---------------------------------------------------------------------------
# Indic corpora carry native-script digits. Python's `\d` on a `str` pattern
# matches every Unicode category-Nd code point, not just [0-9]:
#   https://docs.python.org/3/library/re.html  ("Matches any Unicode decimal
#   digit ... Matches [0-9] if the ASCII flag is used.")
# So `\d{10}` silently matches Devanagari digits while an explicit `[0-9]` or a
# `[6-9]` prefix class does not. That asymmetry inside one pattern is the bug:
# `[6-9]\d{9}` matches nothing on Devanagari input, and `\d{12}` matches
# Devanagari years. Folding makes the behaviour intentional instead of accidental.

# Digit ranges for the scripts a Sangraha Hindi slice can realistically contain.
_DIGIT_BLOCKS = {
    "Devanagari": 0x0966,
    "Bengali": 0x09E6,
    "Gurmukhi": 0x0A66,
    "Gujarati": 0x0AE6,
    "Oriya": 0x0B66,
    "Tamil": 0x0BE6,
    "Telugu": 0x0C66,
    "Kannada": 0x0CE6,
    "Malayalam": 0x0D66,
    "ExtendedArabicIndic": 0x06F0,
    "ArabicIndic": 0x0660,
}

DIGIT_FOLD_MAP: dict[int, str] = {
    base + offset: str(offset) for base in _DIGIT_BLOCKS.values() for offset in range(10)
}


def fold_digits(text: str) -> str:
    """Map non-ASCII decimal digits onto ASCII, preserving length and offsets.

    Every substitution is a single code point for a single code point, so an
    offset in the folded string addresses the same character in the original.
    """
    return text.translate(DIGIT_FOLD_MAP)


def folding_is_offset_safe(text: str) -> bool:
    """Guard the offset-preservation invariant the whole design depends on."""
    return len(fold_digits(text)) == len(text)


def native_digit_count(text: str) -> int:
    return sum(1 for character in text if ord(character) in DIGIT_FOLD_MAP)


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------
# Verhoeff tables. Aadhaar numbers carry a Verhoeff check digit, so a 12-digit
# run either satisfies the checksum or is not an Aadhaar number.
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff_valid(digits: str) -> bool:
    """True when `digits` satisfies the Verhoeff check used by Aadhaar."""
    if not digits.isdigit():
        return False
    checksum = 0
    for index, character in enumerate(reversed(digits)):
        checksum = _VERHOEFF_D[checksum][_VERHOEFF_P[index % 8][int(character)]]
    return checksum == 0


def luhn_valid(digits: str) -> bool:
    """True when `digits` satisfies the Luhn mod-10 check used by card numbers."""
    if not digits.isdigit():
        return False
    total = 0
    doubled = False
    for character in reversed(digits):
        value = int(character)
        if doubled:
            value *= 2
            if value > 9:
                value -= 9
        total += value
        doubled = not doubled
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """One detector proposal against the ORIGINAL text offsets."""

    start: int
    end: int
    category: str
    detector: str
    score: float
    decision: str = "candidate"  # candidate | confirmed | suppressed
    reason: str = ""

    @property
    def length(self) -> int:
        return self.end - self.start

    def text_of(self, text: str) -> str:
        return text[self.start : self.end]


# ---------------------------------------------------------------------------
# Layer 1 — structured patterns
# ---------------------------------------------------------------------------
# Each entry: category, compiled pattern, base score, whether the pattern must
# see folded digits, and the group carrying the digits a checksum should read.
#
# Patterns use [0-9] explicitly. Combined with the folding pass, digit matching
# becomes a deliberate decision rather than a side effect of Unicode `\d`.

PATTERN_LAYER: tuple[tuple[str, re.Pattern[str], float, str], ...] = (
    (
        "credential",
        re.compile(
            r"(?i)\b(?:api[_ -]?key|access[_ -]?token|auth[_ -]?token|bearer|password|passwd|pwd|secret)"
            r"\s*[:=]\s*[^\s,;\"']{6,}"
        ),
        0.90,
        "",
    ),
    (
        "email",
        # Local part deliberately allows \w so mixed-script local parts such as
        # "एRoopchandrashastri@gmail.com" (a real Sangraha string) still match.
        # An ASCII-only [A-Za-z0-9._%+-]+ class silently misses that address.
        #
        # The domain repeats a dot-separated group but the final group must not be
        # followed by a word character. A trailing sentence period is therefore
        # allowed to terminate the match ("...@hotmail.com." keeps the address and
        # drops the period), which a blanket (?![\w.-]) lookahead would reject
        # outright — a real miss in the probe cache.
        re.compile(r"(?<![\w.+-])[\w.+%-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,24}(?![\w-])", re.UNICODE),
        0.95,
        "",
    ),
    (
        # Indian mobile: 10 digits starting 6-9, optionally +91/0 prefixed and
        # optionally grouped. The trailing (?![0-9]) boundary is load-bearing:
        # without it a pattern slices a 10-digit "phone" out of any longer digit
        # run, which is how a NANP-shaped regex turns bank account numbers and
        # astronomy magnitudes into phone numbers on Hindi text.
        "phone_in",
        re.compile(
            r"(?<![0-9])(?:\+?91[\s-]?|0)?"
            r"([6-9][0-9]{4})[\s-]?([0-9]{5})(?![0-9])"
        ),
        0.60,
        "digits",
    ),
    (
        # Indian landline: STD code (2-5 digits, leading 0) + 6-8 digit subscriber
        # number. Published in news copy constantly, so scored below the mobile
        # pattern and reliant on a context cue to cross the threshold.
        "phone_landline",
        re.compile(r"(?<![0-9])(0[0-9]{2,4})[\s-]{0,3}([0-9]{6,8})(?![0-9])"),
        0.35,
        "digits",
    ),
    (
        # Toll-free: 1800/1860 series. Almost always a published business line,
        # so the allowlist layer suppresses it rather than the pattern.
        "phone_tollfree",
        re.compile(r"(?<![0-9])(1(?:800|860)[\s-]?[0-9]{6,7})(?![0-9])"),
        0.30,
        "digits",
    ),
    (
        "ipv4",
        re.compile(
            r"(?<![0-9.])(?:(?:25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])\.){3}"
            r"(?:25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])(?![0-9.])"
        ),
        0.70,
        "",
    ),
    (
        "aadhaar_like",
        re.compile(r"(?<![0-9])([2-9][0-9]{3})[ -]?([0-9]{4})[ -]?([0-9]{4})(?![0-9])"),
        0.35,
        "digits",
    ),
    (
        "card_like",
        re.compile(r"(?<![0-9])((?:[0-9]{4}[ -]?){3}[0-9]{4})(?![0-9])"),
        0.35,
        "digits",
    ),
    (
        "pan_like",
        re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
        0.75,
        "",
    ),
    (
        "handle",
        re.compile(r"(?<![\w@])@([A-Za-z_][A-Za-z0-9_.]{2,29})(?![\w@])"),
        0.30,
        "",
    ),
)


def _digits_of(match: re.Match[str]) -> str:
    return re.sub(r"[^0-9]", "", match.group(0))


def detect_patterns(folded: str) -> list[Span]:
    """Layer 1: run every structured pattern over the folded detection copy."""
    spans: list[Span] = []
    for category, pattern, score, _checksum_kind in PATTERN_LAYER:
        for match in pattern.finditer(folded):
            spans.append(
                Span(
                    start=match.start(),
                    end=match.end(),
                    category=category,
                    detector="pattern",
                    score=score,
                )
            )
    return spans


# ---------------------------------------------------------------------------
# Layer 2 — checksum gates
# ---------------------------------------------------------------------------
CHECKSUM_GATES = {
    "aadhaar_like": ("verhoeff", verhoeff_valid),
    "card_like": ("luhn", luhn_valid),
}


def apply_checksums(spans: list[Span], folded: str) -> list[Span]:
    """Layer 2: promote or reject numeric shape matches arithmetically.

    A 12-digit run is only an Aadhaar number if it satisfies Verhoeff. This is
    the cheapest precision lever available: it costs microseconds and it needs no
    training data, no language assumption and no threshold.
    """
    resolved: list[Span] = []
    for span in spans:
        gate = CHECKSUM_GATES.get(span.category)
        if gate is None:
            resolved.append(span)
            continue
        name, validator = gate
        digits = re.sub(r"[^0-9]", "", folded[span.start : span.end])
        if validator(digits):
            resolved.append(
                Span(
                    span.start,
                    span.end,
                    span.category,
                    span.detector,
                    score=0.95,
                    decision="confirmed",
                    reason=f"{name} check passed",
                )
            )
        else:
            resolved.append(
                Span(
                    span.start,
                    span.end,
                    span.category,
                    span.detector,
                    score=0.0,
                    decision="suppressed",
                    reason=f"{name} check failed",
                )
            )
    return resolved


# ---------------------------------------------------------------------------
# Layer 3 — context scoring
# ---------------------------------------------------------------------------
# Presidio's LemmaContextAwareEnhancer adds a fixed 0.35 to a candidate's score
# when a supportive keyword sits in the surrounding window, then floors the
# result at 0.4 and caps it at 1.0
# (presidio-analyzer/.../lemma_context_aware_enhancer.py:35-41, 150-153).
# The same mechanism is reproduced here with Hindi and English cue words so a
# labelled number outranks a bare one.

CONTEXT_BOOST = 0.35
CONTEXT_FLOOR = 0.40
CONTEXT_WINDOW = 60

CONTEXT_CUES: dict[str, tuple[str, ...]] = {
    "phone_in": (
        "फोन", "मोबाइल", "संपर्क", "नंबर", "व्हाट्सएप", "हेल्पलाइन", "कॉल", "दूरभाष",
        "phone", "mobile", "contact", "call", "whatsapp", "helpline", "tel",
    ),
    "phone_landline": (
        "फोन", "लैंडलाइन", "दूरभाष", "संपर्क", "नंबर", "कार्यालय", "कॉल",
        "phone", "landline", "contact", "call", "office", "tel",
    ),
    "phone_tollfree": (
        "टोल", "फ्री", "हेल्पलाइन", "शिकायत", "नंबर", "ग्राहक",
        "toll", "free", "helpline", "customer", "grievance",
    ),
    "aadhaar_like": ("आधार", "यूआईडी", "aadhaar", "aadhar", "uid", "uidai"),
    "card_like": ("कार्ड", "क्रेडिट", "डेबिट", "card", "credit", "debit", "visa"),
    "pan_like": ("पैन", "पैन कार्ड", "pan", "income tax", "आयकर"),
    "email": ("ईमेल", "मेल", "संपर्क", "email", "e-mail", "mail", "contact"),
    "handle": ("ट्विटर", "ट्वीट", "twitter", "instagram", "handle", "इंस्टाग्राम"),
}


def context_window(folded: str, span: Span, width: int = CONTEXT_WINDOW) -> str:
    return folded[max(0, span.start - width) : min(len(folded), span.end + width)].casefold()


def apply_context(spans: list[Span], folded: str) -> list[Span]:
    """Layer 3: additive context boost, Presidio's mechanism, Hindi cue words."""
    adjusted: list[Span] = []
    for span in spans:
        if span.decision == "suppressed":
            adjusted.append(span)
            continue
        cues = CONTEXT_CUES.get(span.category, ())
        window = context_window(folded, span)
        matched = [cue for cue in cues if cue.casefold() in window]
        if matched:
            score = min(max(span.score + CONTEXT_BOOST, CONTEXT_FLOOR), 1.0)
            adjusted.append(
                Span(
                    span.start,
                    span.end,
                    span.category,
                    span.detector,
                    score=score,
                    decision=span.decision,
                    reason=f"context cue: {matched[0]}",
                )
            )
        else:
            adjusted.append(span)
    return adjusted


# ---------------------------------------------------------------------------
# Layer 4 — allowlist suppression
# ---------------------------------------------------------------------------
# BigCode exempt documentation/example IP addresses from redaction so that
# tutorial text is not destroyed (bigcode-project/bigcode-dataset, pii/).
# Hindi news prose needs its own exemptions: published helplines, institutional
# numbers, and year runs that look like identifiers.

RESERVED_IP_PREFIXES = (
    "0.", "10.", "127.", "169.254.", "192.168.", "224.", "255.",
    "192.0.2.", "198.51.100.", "203.0.113.",
)

PUBLISHED_SERVICE_NUMBERS = frozenset(
    {
        "1800111139", "1800115526", "1930", "1091", "1098", "1075", "112", "100", "101", "102",
        "9224992249",  # IOCL published LPG booking line, appears verbatim in Sangraha
        "18002125512",  # state grain-procurement grievance line seen in the probe cache
        "1800233355", "1800180410", "1800180420",
    }
)

# Institutional and business landlines are published contact points, not personal
# data. Suppressing the whole *category* would be wrong, so this only fires when a
# nearby cue marks the number as an organisation's own contact line.
ORGANISATION_CUES = (
    "हॉस्पिटल", "अस्पताल", "कार्यालय", "थाना", "विभाग", "बैंक", "कंपनी", "संस्थान",
    "हेल्पलाइन", "टोल फ्री", "ग्राहक सेवा", "पुलिस", "काउंसलिंग",
    "hospital", "office", "department", "bank", "company", "helpline", "customer care",
    "police", "counselling", "institute",
)

# Digit folding buys recall on native-script identifiers, but it also lets a
# dotted Sanskrit citation such as "आपस्तंब धर्मसूत्र (२.९.२१.१)" fold into
# "2.9.21.1" and satisfy a dotted-quad IPv4 pattern. Every IPv4 "match" in the
# verified Hindi probe cache was in fact a scripture reference of this shape.
# An IP address is an ASCII-only artefact of network configuration, so a dotted
# quad written in native digits is evidence AGAINST it being an IP address.
NETWORK_CUES = (
    "आईपी", "सर्वर", "नेटवर्क", "राउटर", "पता",
    "ip", "server", "network", "router", "address", "host", "dns", "tcp", "http",
)

# A four-digit group in 1000-2099 that is adjacent to another such group is far
# more likely a year list than an identifier. This is the exact failure the
# Sangraha bibliography documents produce.
_YEARISH = re.compile(r"(?<![0-9])(1[0-9]{3}|20[0-9]{2})(?![0-9])")


def _is_reserved_ip(value: str) -> bool:
    return any(value.startswith(prefix) for prefix in RESERVED_IP_PREFIXES)


def _looks_like_year_run(folded: str, span: Span) -> bool:
    window = folded[max(0, span.start - 40) : min(len(folded), span.end + 40)]
    return len(_YEARISH.findall(window)) >= 3


def apply_allowlists(spans: list[Span], folded: str, original: str) -> list[Span]:
    """Layer 4: the precision brake. Suppress known-benign shape matches."""
    resolved: list[Span] = []
    for span in spans:
        if span.decision == "suppressed":
            resolved.append(span)
            continue
        value = folded[span.start : span.end]
        digits = re.sub(r"[^0-9]", "", value)
        suppress_reason = ""

        _PHONE_CATEGORIES = {"phone_in", "phone_landline", "phone_tollfree"}

        if span.category == "ipv4" and _is_reserved_ip(value):
            suppress_reason = "reserved/documentation IP range"
        elif (
            span.category == "ipv4"
            and native_digit_count(original[span.start : span.end]) > 0
            and not any(
                cue.casefold() in context_window(folded, span, 60) for cue in NETWORK_CUES
            )
        ):
            suppress_reason = "native-script dotted number, citation not an IP"
        elif span.category in _PHONE_CATEGORIES and (
            digits in PUBLISHED_SERVICE_NUMBERS or digits[-10:] in PUBLISHED_SERVICE_NUMBERS
        ):
            suppress_reason = "published service/helpline number"
        elif span.category == "phone_tollfree":
            # The 1800/1860 series is reserved for business lines by construction.
            suppress_reason = "toll-free business number, not personal data"
        elif span.category == "phone_landline" and any(
            cue.casefold() in context_window(folded, span, 90) for cue in ORGANISATION_CUES
        ):
            suppress_reason = "institutional contact line, not personal data"
        elif (
            span.category in {"aadhaar_like", "card_like"}
            and not span.reason.startswith("context cue")
            and _looks_like_year_run(folded, span)
        ):
            # Positional evidence only. An explicit "आधार"/"card" cue is stronger
            # evidence than adjacency, so a context-boosted span is never
            # suppressed by this heuristic.
            suppress_reason = "adjacent year run, not an identifier"

        if suppress_reason:
            resolved.append(
                Span(
                    span.start, span.end, span.category, span.detector,
                    score=0.0, decision="suppressed", reason=suppress_reason,
                )
            )
        else:
            resolved.append(span)
    return resolved


# ---------------------------------------------------------------------------
# Span conflict resolution
# ---------------------------------------------------------------------------
def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Highest score wins; longer span breaks ties; overlaps are dropped.

    Presidio's `EntityRecognizer.remove_duplicates` sorts by
    `(-score, start, -length)` and only drops a span contained in another of the
    SAME entity type, so cross-type overlaps survive there
    (presidio-analyzer/.../entity_recognizer.py:276-318). For a masking pass
    that must produce non-overlapping edits, cross-type overlaps have to be
    resolved too, which is what this does.
    """
    ordered = sorted(spans, key=lambda s: (-s.score, s.start, -s.length))
    kept: list[Span] = []
    for span in ordered:
        if any(span.start < other.end and other.start < span.end for other in kept):
            continue
        kept.append(span)
    return sorted(kept, key=lambda s: s.start)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
PLACEHOLDERS = {
    "email": "[EMAIL]",
    "phone_in": "[PHONE]",
    "phone_landline": "[PHONE]",
    "phone_tollfree": "[PHONE]",
    "ipv4": "[IP]",
    "credential": "[SECRET]",
    "aadhaar_like": "[ID]",
    "card_like": "[CARD]",
    "pan_like": "[ID]",
    "handle": "[USER]",
    "person": "[NAME]",
}


def mask_spans(text: str, spans: list[Span]) -> str:
    """Replace confirmed spans back-to-front so earlier offsets stay valid."""
    output = text
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        output = output[: span.start] + PLACEHOLDERS.get(span.category, "[PII]") + output[span.end :]
    return output


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------
@dataclass
class ScrubResult:
    text: str
    masked_text: str
    accepted: list[Span] = field(default_factory=list)
    suppressed: list[Span] = field(default_factory=list)
    stage_counts: dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.accepted)


def scrub(text: str, threshold: float = 0.50, fold: bool = True) -> ScrubResult:
    """Run the four-layer cascade and return spans plus the masked text.

    `fold=False` reproduces the naive behaviour for the A/B comparison: patterns
    see the raw string, so native-script digits are invisible to `[0-9]` classes.
    """
    folded = fold_digits(text) if fold else text

    candidates = detect_patterns(folded)
    after_checksum = apply_checksums(candidates, folded)
    after_context = apply_context(after_checksum, folded)
    after_allowlist = apply_allowlists(after_context, folded, text)

    surviving = [span for span in after_allowlist if span.decision != "suppressed"]
    above = [span for span in surviving if span.score >= threshold]
    below = [
        Span(s.start, s.end, s.category, s.detector, s.score, "suppressed", "below threshold")
        for s in surviving
        if s.score < threshold
    ]
    accepted = resolve_overlaps(above)
    suppressed = [s for s in after_allowlist if s.decision == "suppressed"] + below

    return ScrubResult(
        text=text,
        masked_text=mask_spans(text, accepted),
        accepted=accepted,
        suppressed=suppressed,
        stage_counts={
            "pattern_candidates": len(candidates),
            "checksum_rejected": sum(1 for s in after_checksum if s.decision == "suppressed"),
            "context_boosted": sum(1 for s in after_context if s.reason.startswith("context cue")),
            "allowlist_suppressed": sum(
                1
                for s in after_allowlist
                if s.decision == "suppressed" and not s.reason.endswith("check failed")
            ),
            "below_threshold": len(below),
            "accepted": len(accepted),
        },
    )
