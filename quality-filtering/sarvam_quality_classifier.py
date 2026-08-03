#!/usr/bin/env python3
"""Build a Hindi/Indic quality router by distilling Sarvam-105B annotations.

This is a **SOTA-informed, assignment-scale** pipeline—not a claim of a
state-of-the-art classifier. It follows the teacher-to-student and
hard-negative-mining pattern used by FineWeb-Edu and DataComp-LM while keeping
the deployed router fast, local, and reproducible:

* Sarvam-105B supplies structured weak labels for a deterministic, stratified
  blend of trusted and candidate Hindi documents.
* A local fastText character-subword student learns the conservative routing
  decision: retain (Sarvam keep/review) versus candidate reject.
* Synthetic corruptions are training aids and separately measured controls;
  they are never presented as real corpus defects or real-world accuracy.
* Protected Indic controls and source-grouped test records detect false
  rejection of Hinglish, Devanagari marks, structured text, verse, and short
  legitimate notices.

Sarvam is an upstream teacher and audit mechanism, never a per-document live
dependency. Generated labels are weak supervision, not human gold labels, and
no reject prediction automatically deletes source text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL_ID = "sarvam-105b"
MODEL_PRICING_PER_MILLION_RUPEES = {"input": 4.0, "output": 16.0}
PIPELINE_VERSION = "sarvam-hindi-quality-classifier-2026-08-04-v2"
LABELS = frozenset({"keep", "review", "reject"})
CLASSIFIER_LABELS = frozenset({"retain", "reject"})
REVIEW_THRESHOLD = 0.70
REASON_CODES = frozenset(
    {
        "coherent_useful_content",
        "valid_structured_content",
        "valid_code_mixed_content",
        "short_but_usable",
        "uncertain_ocr_or_fragment",
        "uncertain_boilerplate",
        "residual_markup",
        "link_farm_or_seo",
        "symbol_debris",
        "repetition_or_keyword_stuffing",
        "malformed_or_unusable_text",
        "not_hindi_or_insufficient_content",
    }
)
BAD_KINDS = (
    "residual_markup",
    "link_farm_or_seo",
    "symbol_debris",
    "repetition_or_keyword_stuffing",
    "malformed_or_unusable_text",
)
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\w)")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>()]+")
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

RUBRIC = """You are annotating documents for a Hindi/Indic pretraining-corpus
quality study. Classify the supplied TEXT, not the source or its topic.

Labels:
- keep: coherent, meaningful material that can be retained.
- review: potentially useful but ambiguous: recoverable OCR, a real short
  fragment, valid list/table/verse/dialogue, unusually formatted text, or
  uncertain boilerplate.
- reject: clearly unusable corpus material: markup/script residue, link farm or
  SEO stuffing, symbol/character debris, repeated boilerplate/keyword stuffing,
  or text too malformed to understand.

Protect valid Devanagari marks, Hindi danda (।/॥), Hinglish, Romanized Hindi,
code mixing, lists, tables, verse, dialogue, numerals, and technical vocabulary.
Do not judge factual accuracy, political viewpoint, profanity, source prestige,
or whether the topic is educational. Do not use the presence of PII placeholders
as a reason to reject. Use only one listed reason_code.
"""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_input_path() -> Path:
    root = repository_root()
    full_cache = root / "extraction/data/sangraha_hi_verified_30m/raw.jsonl"
    return full_cache if full_cache.exists() else root / "extraction/data/sangraha_hi_verified_30m/sample_raw.jsonl"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "data/sarvam_quality_classifier"


def default_candidate_path() -> Path:
    return default_output_dir() / "unverified_pool.jsonl"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_bucket(value: str, buckets: int = 5, *, salt: str = "") -> int:
    """Return a reproducible bucket; callers must salt independent decisions."""
    digest_input = f"{salt}\x1f{value}".encode()
    return int(hashlib.sha256(digest_input).hexdigest(), 16) % buckets


def redact_for_annotation(text: str) -> str:
    """Mask obvious PII before any external annotation or local model training."""
    text = EMAIL_RE.sub("[MASKED_EMAIL]", text or "")
    text = IP_RE.sub("[MASKED_IPV4]", text)
    return PHONE_RE.sub("[MASKED_PHONE]", text)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", redact_for_annotation(text)).strip()


def capped_text(text: str, maximum_characters: int) -> str:
    """Preserve beginning, middle, and end evidence under an annotation cap."""
    text = normalize_text(text)
    if len(text) <= maximum_characters:
        return text
    marker = "\n[TRUNCATED_FOR_ANNOTATION]\n"
    available = maximum_characters - len(marker)
    if available < 180:
        return f"{text[:maximum_characters].rstrip()}\n[TRUNCATED_FOR_ANNOTATION]"
    head = available * 2 // 5
    tail = available * 2 // 5
    middle = available - head - tail
    middle_start = max(head, (len(text) - middle) // 2)
    return "".join((
        text[:head].rstrip(), marker,
        text[middle_start : middle_start + middle].strip(), marker,
        text[-tail:].lstrip(),
    ))


def load_records(input_path: Path, maximum_documents: int, *, origin: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id") or "")
            text = str(row.get("text") or "")
            if not doc_id or not text.strip() or doc_id in seen_ids:
                continue
            records.append(
                {
                    "doc_id": doc_id,
                    "source_doc_id": doc_id,
                    "type": str(row.get("type") or "unknown"),
                    "text": normalize_text(text),
                    "origin": origin,
                    "line_number": str(line_number),
                }
            )
            seen_ids.add(doc_id)
            if len(records) >= maximum_documents:
                break
    if not records:
        raise ValueError(f"No usable records found in {input_path}")
    return records


def heuristic_risk_band(text: str) -> str:
    """A transparent pre-screen for *sampling*, never a source deletion rule."""
    compact = re.sub(r"\s+", " ", text).strip()
    symbols = sum(not char.isalnum() and not char.isspace() for char in compact)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated_lines = len(lines) - len(set(lines))
    signals = sum((
        len(URL_RE.findall(compact)) >= 3,
        compact.count("<") + compact.count(">") >= 8,
        bool(compact) and symbols / len(compact) >= 0.32,
        repeated_lines >= 3,
        len(source_words(compact)) < 12,
    ))
    return "high" if signals >= 2 else "medium" if signals else "low"


def risk_priority(record: dict[str, str]) -> tuple[int, str]:
    """Prefer riskier candidate rows inside a stratum; used only for annotation sampling."""
    priority = {"high": 0, "medium": 1, "low": 2}[heuristic_risk_band(record["text"])]
    return priority, sha256_bytes(record["doc_id"].encode())


def length_bucket(text: str) -> str:
    length = len(text)
    if length < 500:
        return "short"
    if length < 2_000:
        return "medium"
    return "long"


def deterministic_stratified_sample(
    records: list[dict[str, str]], maximum_documents: int
) -> list[dict[str, str]]:
    """Balance origins, then sample origin × type × length × risk deterministically."""
    if maximum_documents < 1:
        raise ValueError("maximum_documents must be positive")
    by_origin: dict[str, list[dict[str, str]]] = {}
    for record in records:
        by_origin.setdefault(record["origin"], []).append(record)
    origin_names = sorted(by_origin)
    quotas = {
        origin: minimum + (1 if position < maximum_documents % len(origin_names) else 0)
        for position, (origin, minimum) in enumerate(
            (origin, maximum_documents // len(origin_names)) for origin in origin_names
        )
    }
    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()
    for origin in origin_names:
        groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for record in by_origin[origin]:
            stratum = (record["type"], length_bucket(record["text"]), heuristic_risk_band(record["text"]))
            groups.setdefault(stratum, []).append(record)
        ordered_groups = [
            sorted(group, key=risk_priority)
            for _, group in sorted(groups.items())
        ]
        origin_selected: list[dict[str, str]] = []
        for group in ordered_groups:
            if len(origin_selected) >= quotas[origin]:
                break
            origin_selected.append(group[0])
        remainder = sorted(
            (record for group in ordered_groups for record in group if record not in origin_selected),
            key=risk_priority,
        )
        origin_selected.extend(remainder[: quotas[origin] - len(origin_selected)])
        selected.extend(origin_selected)
        selected_ids.update(record["doc_id"] for record in origin_selected)
    if len(selected) < maximum_documents:
        remainder = sorted(
            (record for record in records if record["doc_id"] not in selected_ids),
            key=lambda row: sha256_bytes(row["doc_id"].encode()),
        )
        selected.extend(remainder[: maximum_documents - len(selected)])
    return sorted(selected, key=lambda row: row["doc_id"])


def annotation_plan(records: list[dict[str, str]], batch_size: int, character_cap: int) -> dict[str, Any]:
    """Estimate the request ceiling before contacting Sarvam; estimates are not spend."""
    prompt_tokens = 0
    for batch in chunks(records, batch_size):
        payload = label_request_payload(batch, character_cap)
        prompt_tokens += math.ceil(len(json.dumps(payload, ensure_ascii=False)) / 2.5)
    completion_token_ceiling = 72 * len(records)
    estimated_cost = (
        prompt_tokens * MODEL_PRICING_PER_MILLION_RUPEES["input"]
        + completion_token_ceiling * MODEL_PRICING_PER_MILLION_RUPEES["output"]
    ) / 1_000_000
    return {
        "documents": len(records),
        "requests": math.ceil(len(records) / batch_size),
        "character_cap": character_cap,
        "estimated_prompt_tokens": prompt_tokens,
        "completion_token_ceiling": completion_token_ceiling,
        "estimated_cost_rupees": estimated_cost,
        "note": "A conservative pre-request estimate from payload characters and max_tokens; billed usage is recorded from API responses.",
    }


def first_sentence_or_line(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return text.strip()


def source_words(text: str) -> list[str]:
    return [word for word in WORD_RE.findall(text) if len(word) >= 3]


def corrupt_record(record: dict[str, str], kind: str) -> dict[str, str]:
    """Create a clearly bad counterfactual; never relabel an original document."""
    seed = record["source_doc_id"]
    snippet = first_sentence_or_line(record["text"])[:180]
    words = source_words(record["text"])
    keyword = " ".join(words[:6]) or "सस्ता ऑफर"
    if kind == "residual_markup":
        text = (
            "<html><head><script>track('x')</script></head><body><nav>"
            "Home | Login | Subscribe</nav><div class='ad'>ADVERTISEMENT</div>"
            f"<article>{snippet}</article><footer>cookie policy</footer></body></html>"
        )
    elif kind == "link_farm_or_seo":
        links = " ".join(f"https://spam.example/{index}" for index in range(12))
        text = f"best deal cheap offer {keyword} {links} buy now best deal {keyword}"
    elif kind == "symbol_debris":
        text = "@@@ ### $$$ %%% ^^^ &&& ||| !!! ??? " * 14
    elif kind == "repetition_or_keyword_stuffing":
        repeated = (snippet or keyword)[:140]
        text = "\n".join([repeated] * 14)
    elif kind == "malformed_or_unusable_text":
        text = "अआअअ ### qqq 9999 %% र््् ??? " * 12
    else:
        raise ValueError(f"Unknown bad kind: {kind}")
    return {
        "doc_id": f"synthetic-{kind}-{seed}",
        "source_doc_id": seed,
        "type": "synthetic_bad_control",
        "text": normalize_text(text),
        "origin": "synthetic_bad",
        "label": "reject",
        "reason_code": kind,
    }


def protected_controls(split: str) -> list[dict[str, str]]:
    """Independent known-good and ambiguous Indic forms for train or held-out audit."""
    by_split = {
        "train": [
            ("keep", "valid_code_mixed_content", "मुझे flight की booking बदलनी है क्योंकि मौसम के कारण यात्रा स्थगित हो गई। कृपया नया समय और शुल्क साफ़-साफ़ बताइए।"),
            ("keep", "valid_structured_content", "खरीदारी सूची:\n- दाल\n- चावल\n- तेल\n- नमक\nयह सूची घर के साप्ताहिक खर्च की योजना के लिए है।"),
            ("keep", "coherent_useful_content", "जल संरक्षण के लिए वर्षा जल को छत से पाइप द्वारा टंकी तक पहुंचाया जा सकता है। इससे भूजल पर दबाव कम होता है।"),
            ("keep", "valid_code_mixed_content", "Server restart के बाद cache साफ़ करें और फिर app को दोबारा चलाइए। अगर error आए तो log का समय लिखिए।"),
            ("keep", "valid_structured_content", "प्रयोग के चरण:\n1. बीज को नम मिट्टी में रखें।\n2. प्रतिदिन पानी दें।\n3. सात दिन बाद अंकुर की लंबाई मापें।"),
            ("keep", "coherent_useful_content", "टीकाकरण का समय और स्थान स्थानीय स्वास्थ्य केंद्र से सत्यापित करना चाहिए। संदेश को पूरा पढ़ना जरूरी है।"),
            ("review", "valid_structured_content", "रात ढली, आंगन में चांद रुका।\nनीम की छाया में एक पुराना गीत जागा।\nयह छोटी कविता है; संक्षिप्त होना इसे शोर नहीं बनाता।"),
            ("review", "short_but_usable", "बैठक सोमवार को दोपहर तीन बजे होगी। कृपया समय पर पहुंचें।"),
            ("review", "uncertain_ocr_or_fragment", "विद्यालय के छात्रों के लिए पुस्तकालय में अतिरिक्त संदर्भ सामग्री उपलब्ध कराई जाएगी। कुछ अक्षर धुंधले हैं, पर मुख्य अर्थ समझ में आता है।"),
            ("review", "valid_structured_content", "तालिका: वर्ष | वर्षा\n2022 | 620 मिमी\n2023 | 580 मिमी\nइकाई और स्रोत की पुष्टि बाकी है।"),
            ("review", "short_but_usable", "प्रश्न: आवेदन की अंतिम तिथि क्या है? उत्तर अधूरा है, इसलिए मूल सूचना देखनी चाहिए।"),
            ("review", "uncertain_boilerplate", "पृष्ठ के नीचे संपर्क, गोपनीयता और सदस्यता लिंक हैं; मुख्य लेख का आरंभ उपलब्ध नहीं है।"),
        ],
        "held_out": [
            ("keep", "valid_code_mixed_content", "कल Zoom meeting में product demo है। कृपया Hindi और English दोनों में प्रश्न पूछ सकते हैं।"),
            ("keep", "valid_structured_content", "दवा लेने की योजना:\n- सुबह नाश्ते के बाद\n- रात भोजन के बाद\nनिर्देश डॉक्टर की सलाह के अनुसार बदल सकते हैं।"),
            ("keep", "coherent_useful_content", "मिट्टी की नमी बनाए रखने के लिए पौधे के आसपास सूखी पत्तियों की परत बिछाई जा सकती है।"),
            ("keep", "valid_code_mixed_content", "UPI payment सफल होने पर reference number सुरक्षित रखें; किसी अनजान व्यक्ति के साथ OTP साझा न करें।"),
            ("keep", "valid_structured_content", "पुस्तकालय समय:\nसोमवार–शुक्रवार: 9 से 5\nशनिवार: 10 से 2\nरविवार: बंद"),
            ("keep", "coherent_useful_content", "साक्षात्कार में उत्तर देने से पहले प्रश्न को ध्यान से सुनना और उदाहरण देना उपयोगी होता है।"),
            ("review", "valid_structured_content", "धूप की एक पतली रेखा\nखिड़की से होकर आई।\nशब्द कम हैं, अर्थ का निर्णय पाठक पर छोड़ती है।"),
            ("review", "short_but_usable", "ट्रेन प्लेटफॉर्म तीन से रवाना होगी। टिकट साथ रखें।"),
            ("review", "uncertain_ocr_or_fragment", "अनुसंधान प्रतिवेदन के तीसरे खंड में आंकड़े अधूरे हैं और एक वाक्य बीच में कट गया है।"),
            ("review", "valid_structured_content", "नाम | उपस्थिति\nरीना | 18\nमोहन | 16\nसंदर्भ अवधि लिखी नहीं गई है।"),
            ("review", "short_but_usable", "अगली सूचना तक कार्यालय बंद रहेगा। कारण और अवधि स्पष्ट नहीं हैं।"),
            ("review", "uncertain_boilerplate", "इस पृष्ठ पर navigation links और copyright पंक्ति दिखती है, लेकिन लेख का पूरा भाग नहीं मिला।"),
        ],
    }
    try:
        controls = by_split[split]
    except KeyError as error:
        raise ValueError(f"Unknown protected-control split: {split}") from error
    return [
        {
            "doc_id": f"protected-{split}-{index}",
            "source_doc_id": f"protected-{split}-{index}",
            "type": "protected_control",
            "text": normalize_text(text),
            "origin": "protected_control",
            "label": label,
            "reason_code": reason_code,
        }
        for index, (label, reason_code, text) in enumerate(controls)
    ]



def classifier_label(row: dict[str, str]) -> str:
    """Use a conservative binary learner; both keep and review remain retain."""
    return "reject" if row["label"] == "reject" else "retain"

def held_out_bad_controls() -> list[dict[str, str]]:
    """One fixed control per failure family; reported only as synthetic coverage."""
    seed = {
        "doc_id": "adversarial-seed",
        "source_doc_id": "adversarial-seed",
        "type": "control",
        "text": "यह उदाहरण केवल खराब डेटा के प्रकारों का परीक्षण करता है।",
        "origin": "synthetic_bad",
    }
    return [corrupt_record(seed, kind) for kind in BAD_KINDS]


def label_request_payload(records: list[dict[str, str]], cap: int) -> dict[str, Any]:
    documents = [
        {
            "doc_id": record["doc_id"],
            "type": record["type"],
            "text": capped_text(record["text"], cap),
        }
        for record in records
    ]
    item_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["doc_id", "label", "reason_code"],
        "properties": {
            "doc_id": {"type": "string"},
            "label": {"type": "string", "enum": sorted(LABELS)},
            "reason_code": {"type": "string", "enum": sorted(REASON_CODES)},
        },
    }
    response_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["labels"],
        "properties": {"labels": {"type": "array", "items": item_schema}},
    }
    return {
        "model": MODEL_ID,
        "temperature": 0,
        "seed": 20260804,
        "reasoning_effort": None,
        "max_tokens": max(320, 72 * len(documents)),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "hindi_quality_labels",
                "description": "One quality-routing label for every supplied document.",
                "schema": response_schema,
                "strict": True,
            },
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    f"{RUBRIC}\nAllowed reason_code literals: {', '.join(sorted(REASON_CODES))}. "
                    "Return one label for every supplied doc_id."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"documents": documents}, ensure_ascii=False),
            },
        ],
    }


def extract_json_value(content: str) -> Any:
    content = content.strip()
    if content.startswith("```"):
        fence_end = content.find("\n")
        content = content[fence_end + 1 :] if fence_end >= 0 else ""
        content = content.removesuffix("```")
        content = content.strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        return value
    raise ValueError(f"Sarvam did not return JSON: {content[:160]!r}")


def parse_partial_labels(content: str, expected_ids: set[str]) -> dict[str, dict[str, str]]:
    """Parse valid labels returned before a batch omits one or more IDs."""
    parsed = extract_json_value(content)
    entries = parsed.get("labels") if isinstance(parsed, dict) else parsed
    if not isinstance(entries, list):
        raise TypeError("Sarvam response has no labels array")
    labels: dict[str, dict[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        doc_id = str(entry.get("doc_id") or "")
        label = str(entry.get("label") or "").lower()
        reason_code = str(entry.get("reason_code") or "")
        if doc_id not in expected_ids or label not in LABELS or reason_code not in REASON_CODES:
            continue
        if doc_id in labels:
            continue
        labels[doc_id] = {"label": label, "reason_code": reason_code}
    return labels


def parse_labels(content: str, expected_ids: set[str]) -> dict[str, dict[str, str]]:
    labels = parse_partial_labels(content, expected_ids)
    if set(labels) != expected_ids:
        missing = sorted(expected_ids - set(labels))
        extra = sorted(set(labels) - expected_ids)
        raise ValueError(f"Sarvam label IDs mismatch; missing={missing}, extra={extra}")
    return labels


def sarvam_chat(payload: dict[str, Any], api_key: str, retries: int = 3) -> tuple[dict[str, Any], int]:
    request = Request(
        "https://api.sarvam.ai/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8")), attempt + 1
        except HTTPError as error:
            retryable = error.code in {429, 500, 502, 503, 504}
            body = error.read().decode("utf-8", errors="replace")[:400]
            if not retryable or attempt == retries - 1:
                raise RuntimeError(f"Sarvam HTTP {error.code}: {body}") from error
        except URLError as error:
            if attempt == retries - 1:
                raise RuntimeError(f"Sarvam network failure: {error}") from error
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def sarvam_labels_for_record(
    record: dict[str, str], character_cap: int, api_key: str, retries: int = 3
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]], int]:
    """Get one valid structured label, retrying malformed model content without relabeling a batch."""
    request_usages: list[dict[str, Any]] = []
    retry_count = 0
    for attempt in range(retries):
        response, request_attempts = sarvam_chat(
            label_request_payload([record], character_cap), api_key
        )
        retry_count += request_attempts - 1
        request_usages.append(response.get("usage") or {})
        content = response.get("choices", [{}])[0].get("message", {}).get("content")
        if not isinstance(content, str):
            continue
        try:
            return parse_labels(content, {record["doc_id"]}), request_usages, retry_count
        except (TypeError, ValueError):
            if attempt == retries - 1:
                raise RuntimeError(
                    f"Sarvam returned malformed structured output for {record['doc_id']} after {retries} attempts"
                )
    raise AssertionError("unreachable")


def chunks(rows: list[dict[str, str]], size: int) -> Iterable[list[dict[str, str]]]:
    for index in range(0, len(rows), size):
        yield rows[index : index + size]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def annotation_cost(usage: dict[str, Any]) -> float:
    return (
        int(usage.get("prompt_tokens", 0)) * MODEL_PRICING_PER_MILLION_RUPEES["input"]
        + int(usage.get("completion_tokens", 0)) * MODEL_PRICING_PER_MILLION_RUPEES["output"]
    ) / 1_000_000


def annotate_source_records(
    source_records: list[dict[str, str]],
    output_path: Path,
    api_key: str,
    batch_size: int,
    character_cap: int,
    budget_rupees: float,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    existing = {
        str(row["doc_id"]): row
        for row in read_jsonl(output_path)
        if row.get("pipeline_version") == PIPELINE_VERSION
    }
    cumulative_cost = sum(float(row.get("request_cost_rupees", 0.0)) for row in existing.values())
    request_count = 0
    retry_count = 0
    for batch in chunks(source_records, batch_size):
        missing = [record for record in batch if record["doc_id"] not in existing]
        if not missing:
            continue
        payload = label_request_payload(missing, character_cap)
        estimated_input = len(json.dumps(payload, ensure_ascii=False)) / 2.5
        estimated_output = 36 * len(missing)
        estimated_cost = (
            estimated_input * MODEL_PRICING_PER_MILLION_RUPEES["input"]
            + estimated_output * MODEL_PRICING_PER_MILLION_RUPEES["output"]
        ) / 1_000_000
        if cumulative_cost + estimated_cost > budget_rupees:
            raise RuntimeError(
                f"Stopping before request: projected cost ₹{cumulative_cost + estimated_cost:.4f} "
                f"would exceed budget ₹{budget_rupees:.2f}."
            )
        request_usages: list[dict[str, Any]] = []
        response, attempts = sarvam_chat(payload, api_key)
        request_count += 1
        retry_count += attempts - 1
        request_usages.append(response.get("usage") or {})
        choice = response.get("choices", [{}])[0].get("message", {})
        content = choice.get("content")
        if not isinstance(content, str):
            raise TypeError("Sarvam returned no final content")
        try:
            labels = parse_labels(content, {record["doc_id"] for record in missing})
        except (TypeError, ValueError):
            try:
                labels = parse_partial_labels(content, {record["doc_id"] for record in missing})
            except (TypeError, ValueError):
                labels = {}
            for record in missing:
                if record["doc_id"] in labels:
                    continue
                labels_for_record, single_usages, single_retries = sarvam_labels_for_record(
                    record, character_cap, api_key
                )
                request_count += len(single_usages)
                retry_count += single_retries
                request_usages.extend(single_usages)
                labels.update(labels_for_record)
        prompt_tokens = sum(int(usage.get("prompt_tokens", 0)) for usage in request_usages)
        completion_tokens = sum(int(usage.get("completion_tokens", 0)) for usage in request_usages)
        request_cost = sum(annotation_cost(usage) for usage in request_usages)
        cumulative_cost += request_cost
        per_document_cost = request_cost / len(missing)
        timestamp = datetime.now(UTC).isoformat()
        for record in missing:
            decision = labels[record["doc_id"]]
            existing[record["doc_id"]] = {
                "pipeline_version": PIPELINE_VERSION,
                "doc_id": record["doc_id"],
                "source_doc_id": record["source_doc_id"],
                "type": record["type"],
                "origin": record["origin"],
                "label": decision["label"],
                "reason_code": decision["reason_code"],
                "annotation_model": MODEL_ID,
                "annotation_timestamp_utc": timestamp,
                "annotation_characters": len(capped_text(record["text"], character_cap)),
                "request_prompt_tokens": math.ceil(prompt_tokens / len(missing)),
                "request_completion_tokens": math.ceil(completion_tokens / len(missing)),
                "request_cost_rupees": per_document_cost,
            }
        write_jsonl(output_path, [
            existing[record["doc_id"]]
            for record in source_records
            if record["doc_id"] in existing
        ])
    rows = [existing[record["doc_id"]] for record in source_records]
    return rows, {
        "requests_made_this_run": request_count,
        "retries_this_run": retry_count,
        "total_estimated_cost_rupees": cumulative_cost,
        "total_prompt_tokens": sum(int(row.get("request_prompt_tokens", 0)) for row in rows),
        "total_completion_tokens": sum(int(row.get("request_completion_tokens", 0)) for row in rows),
    }


def build_training_rows(
    source_records: list[dict[str, str]],
    annotations: list[dict[str, str]],
    synthetics_per_source: int,
) -> list[dict[str, str]]:
    """Attach weak labels and deterministic corruptions before grouped splitting."""
    annotation_by_id = {str(row["doc_id"]): row for row in annotations}
    rows: list[dict[str, str]] = []
    for source in source_records:
        annotation = annotation_by_id[source["doc_id"]]
        rows.append({
            **source,
            "label": str(annotation["label"]),
            "reason_code": str(annotation["reason_code"]),
        })
        start = stable_bucket(
            source["doc_id"], len(BAD_KINDS), salt="synthetic-corruption-kind-v2"
        )
        for offset in range(synthetics_per_source):
            rows.append(corrupt_record(source, BAD_KINDS[(start + offset) % len(BAD_KINDS)]))
    return rows


def rebalance_training_rows(rows: list[dict[str, str]], real_reject_multiplier: int) -> list[dict[str, str]]:
    """Counter synthetic-class dominance using only grouped-training real rejects."""
    real_rejects = [
        row for row in rows
        if row["origin"] in {"trusted_source", "candidate_source"} and row["label"] == "reject"
    ]
    if not real_rejects:
        raise RuntimeError("Training split has no real Sarvam reject labels")
    return rows + real_rejects * (real_reject_multiplier - 1)


def partition_by_source(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Keep each source and every derived synthetic exclusively in one split."""
    train, held_out = [], []
    for row in rows:
        bucket = stable_bucket(row["source_doc_id"], salt="source-group-split-v2")
        (held_out if bucket == 0 else train).append(row)
    if not held_out or not train:
        raise RuntimeError("Grouped train/held-out split is empty")
    return train, held_out


def fasttext_line(row: dict[str, str]) -> str:
    text = re.sub(r"\s+", " ", row["text"]).strip()
    return f"__label__{classifier_label(row)} {text}\n"


def write_fasttext_corpus(path: Path, rows: list[dict[str, str]]) -> None:
    atomic_write(path, "".join(fasttext_line(row) for row in rows))


def fit_classifier(train_rows: list[dict[str, str]], output_dir: Path) -> Any:
    import fasttext

    labels = {classifier_label(row) for row in train_rows}
    if labels != CLASSIFIER_LABELS:
        raise RuntimeError(f"Training data must contain all classifier labels; got {sorted(labels)}")
    train_path = output_dir / "fasttext_train.txt"
    write_fasttext_corpus(train_path, train_rows)
    model = fasttext.train_supervised(
        input=str(train_path), lr=0.45, wordNgrams=2, minn=2, maxn=5,
        dim=96, loss="softmax", thread=1, seed=20260804,
    )
    model.save_model(str(output_dir / "quality_classifier.bin"))
    return model


def score_rows(model: Any, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    scores = []
    for row in rows:
        labels, probabilities = model.predict(re.sub(r"\s+", " ", row["text"]).strip(), k=2)
        probability_by_label = {
            label.removeprefix("__label__"): float(probability)
            for label, probability in zip(labels, probabilities)
        }
        reject_probability = probability_by_label.get("reject", 0.0)
        scores.append({
            "doc_id": row["doc_id"], "source_doc_id": row["source_doc_id"],
            "origin": row["origin"], "type": row["type"],
            "expected_label": classifier_label(row), "expected_reason_code": row["reason_code"],
            "predicted_label": "reject" if reject_probability >= REVIEW_THRESHOLD else "retain",
            "predicted_probability": reject_probability,
        })
    return scores


def classification_summary(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Report real weak-label evidence separately from synthetic controls."""
    confusion = {
        expected: {predicted: 0 for predicted in sorted(CLASSIFIER_LABELS)}
        for expected in sorted(CLASSIFIER_LABELS)
    }
    for score in scores:
        confusion[score["expected_label"]][score["predicted_label"]] += 1

    def summary_for(subset: list[dict[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {"documents": 0, "accuracy": None, "reject_recall": None, "false_reject_rate": None}
        expected_rejects = [row for row in subset if row["expected_label"] == "reject"]
        caught_rejects = [row for row in expected_rejects if row["predicted_label"] == "reject"]
        false_rejects = [
            row for row in subset if row["expected_label"] == "retain" and row["predicted_label"] == "reject"
        ]
        return {
            "documents": len(subset),
            "accuracy": sum(row["expected_label"] == row["predicted_label"] for row in subset) / len(subset),
            "reject_recall": len(caught_rejects) / len(expected_rejects) if expected_rejects else None,
            "false_reject_rate": len(false_rejects) / len(subset),
        }

    real_origins = {"trusted_source", "candidate_source"}
    real_scores = [score for score in scores if score["origin"] in real_origins]
    bad = [score for score in scores if score["origin"] == "synthetic_bad"]
    caught = [score for score in bad if score["predicted_label"] == "reject"]
    return {
        "documents": len(scores), "all_rows": summary_for(scores),
        "real_weak_label_test": summary_for(real_scores), "confusion_matrix": confusion,
        "synthetic_control_reject_recall": len(caught) / len(bad) if bad else None,
        "bad_kind_breakdown": {
            kind: {
                "documents": sum(score["expected_reason_code"] == kind for score in bad),
                "caught_as_reject": sum(
                    score["expected_reason_code"] == kind and score["predicted_label"] == "reject"
                    for score in bad
                ),
            } for kind in BAD_KINDS
        },
        "by_origin": {
            origin: summary_for([score for score in scores if score["origin"] == origin])
            for origin in sorted({str(score["origin"]) for score in scores})
        },
        "evaluation_contract": {
            "real_weak_label_test": "Grouped held-out Sarvam weak labels; not human gold or an external generalization claim.",
            "synthetic_controls": "Deterministic counterfactual coverage only; excluded from real weak-label metrics.",
            "protected_controls": "Safety regression checks for false rejections of protected Indic forms.",
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    trusted_path, candidate_path = args.input.resolve(), args.candidate_input.resolve()
    output_dir = args.output.resolve()
    trusted_pool = load_records(trusted_path, args.trusted_documents, origin="trusted_source")
    candidate_pool = load_records(candidate_path, args.candidate_documents, origin="candidate_source")
    records = deterministic_stratified_sample(trusted_pool + candidate_pool, args.max_documents)
    annotations_path = output_dir / "sarvam_source_annotations.jsonl"
    annotation_rows, annotation_usage = annotate_source_records(
        records, annotations_path, args.api_key, args.batch_size, args.character_cap, args.credit_budget_rupees
    )
    real_rejects = sum(row["label"] == "reject" for row in annotation_rows)
    if real_rejects < args.minimum_real_rejects:
        raise RuntimeError(
            f"Only {real_rejects} Sarvam-labeled real rejects; need at least {args.minimum_real_rejects} before fitting a router."
        )
    all_rows = build_training_rows(records, annotation_rows, args.synthetics_per_source)
    train_rows, held_out_rows = partition_by_source(all_rows)
    train_rows = rebalance_training_rows(train_rows, args.real_reject_multiplier)
    train_rows.extend(protected_controls("train"))
    held_out_rows.extend(protected_controls("held_out"))
    held_out_rows.extend(held_out_bad_controls())
    synthetic_rows = [row for row in all_rows if row["origin"] == "synthetic_bad"]
    write_jsonl(output_dir / "synthetic_bad_examples.jsonl", synthetic_rows)
    write_jsonl(output_dir / "held_out_controls.jsonl", [
        row for row in held_out_rows if row["origin"] not in {"trusted_source", "candidate_source"}
    ])
    model = fit_classifier(train_rows, output_dir)
    held_out_scores = score_rows(model, held_out_rows)
    write_jsonl(output_dir / "held_out_predictions.jsonl", held_out_scores)
    summary = {
        "pipeline_version": PIPELINE_VERSION, "created_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "trusted_input_path": str(trusted_path), "candidate_input_path": str(candidate_path),
            "trusted_input_sha256": sha256_bytes(trusted_path.read_bytes()),
            "candidate_input_sha256": sha256_bytes(candidate_path.read_bytes()),
            "source_documents": len(records), "trusted_pool_documents": len(trusted_pool),
            "candidate_pool_documents": len(candidate_pool),
            "selection_rule": "deterministic stratification by origin × type × length × transparent risk band, then SHA-256 ordering",
        },
        "sarvam": {
            "model": MODEL_ID, "pricing_rupees_per_million_tokens": MODEL_PRICING_PER_MILLION_RUPEES,
            "annotation_character_cap": args.character_cap,
            "annotation_excerpt_policy": "head, middle, and tail retained when a document exceeds the cap",
            "batch_size": args.batch_size, "temperature": 0, "seed": 20260804,
            "seed_policy": "best effort only; monitor system_fingerprint for backend drift when supplied",
            "reasoning_effort": None, "response_contract": "native strict JSON Schema",
            "annotation_usage": annotation_usage, "api_key_persisted": False,
        },
        "label_policy": {
            "labels": sorted(LABELS), "classifier_labels": sorted(CLASSIFIER_LABELS),
            "automatic_deletion_policy": "none; reject predictions require a separate human or deterministic-rule decision",
            "known_bad_examples": list(BAD_KINDS),
            "protected_forms": ["Devanagari", "।", "॥", "Hinglish", "lists", "verse", "short usable text"],
        },
        "training": {
            "train_documents": len(train_rows), "held_out_documents": len(held_out_rows),
            "synthetic_bad_documents": len(synthetic_rows), "real_sarvam_rejects": real_rejects,
            "real_reject_multiplier": args.real_reject_multiplier,
            "source_group_split": "sha256('source-group-split-v2' + source_doc_id) % 5 == 0 held out",
            "model": "fastText supervised binary router: retain (Sarvam keep/review) versus reject; character subwords 2-5 plus word bigrams",
            "review_threshold": REVIEW_THRESHOLD,
        },
        "held_out_evaluation": classification_summary(held_out_scores),
        "artifacts": {
            "source_annotations": str(annotations_path), "synthetic_bad_examples": str(output_dir / "synthetic_bad_examples.jsonl"),
            "held_out_controls": str(output_dir / "held_out_controls.jsonl"),
            "held_out_predictions": str(output_dir / "held_out_predictions.jsonl"),
        },
        "limitations": [
            "Sarvam labels are model-generated weak supervision, not human gold labels.",
            "Synthetic controls measure corruption-family detection only and are excluded from real weak-label metrics.",
            "The binary local classifier routes Sarvam review labels to retain; it cannot replace a human review queue.",
            "A locked human-labeled Hindi test set and downstream training ablation are required before an external generalization or production-quality claim.",
        ],
    }
    atomic_write(output_dir / "classifier_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def plan(args: argparse.Namespace) -> dict[str, Any]:
    trusted_pool = load_records(args.input.resolve(), args.trusted_documents, origin="trusted_source")
    candidate_pool = load_records(args.candidate_input.resolve(), args.candidate_documents, origin="candidate_source")
    records = deterministic_stratified_sample(trusted_pool + candidate_pool, args.max_documents)
    return {
        "pipeline_version": PIPELINE_VERSION,
        "selection": {
            "documents": len(records),
            "by_origin": {origin: sum(row["origin"] == origin for row in records) for origin in sorted({row["origin"] for row in records})},
            "by_risk_band": {risk: sum(heuristic_risk_band(row["text"]) == risk for row in records) for risk in ("high", "medium", "low")},
            "rule": "deterministic origin × type × length × risk-band stratification, then SHA-256 ordering",
        },
        "annotation_plan": annotation_plan(records, args.batch_size, args.character_cap),
        "minimum_real_rejects": args.minimum_real_rejects,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input_path())
    parser.add_argument("--candidate-input", type=Path, default=default_candidate_path())
    parser.add_argument("--output", type=Path, default=default_output_dir())
    parser.add_argument("--max-documents", type=int, default=1_600)
    parser.add_argument("--trusted-documents", type=int, default=264)
    parser.add_argument("--candidate-documents", type=int, default=1_742)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--character-cap", type=int, default=1_500)
    parser.add_argument("--synthetics-per-source", type=int, default=2)
    parser.add_argument("--minimum-real-rejects", type=int, default=10)
    parser.add_argument("--real-reject-multiplier", type=int, default=40)
    parser.add_argument("--credit-budget-rupees", type=float, default=1_000.0)
    parser.add_argument("--api-key", default=os.environ.get("SARVAM_API_KEY"))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.plan_only and not args.api_key:
        parser.error("Set SARVAM_API_KEY or pass --api-key; do not commit a key.")
    if args.max_documents < 100 or args.trusted_documents < 1 or args.candidate_documents < 1:
        parser.error("document limits are insufficient for stratified grouped evaluation.")
    if args.batch_size < 1 or args.character_cap < 200 or args.synthetics_per_source < 1:
        parser.error("batch size, character cap, and synthetics per source must be positive.")
    if args.minimum_real_rejects < 1 or args.real_reject_multiplier < 1:
        parser.error("real reject thresholds and multipliers must be positive.")
    if not args.candidate_input.exists():
        parser.error(f"Candidate input is missing: {args.candidate_input}")
    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    try:
        if args.plan_only:
            print(json.dumps(plan(args), ensure_ascii=False, indent=2))
            raise SystemExit(0)
        summary = run(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    evaluation = summary["held_out_evaluation"]
    real = evaluation["real_weak_label_test"]
    print(
        "Completed Sarvam-assisted classifier: "
        f"{summary['source']['source_documents']} stratified source documents, "
        f"₹{summary['sarvam']['annotation_usage']['total_estimated_cost_rupees']:.4f} estimated annotation cost, "
        f"real weak-label held-out accuracy={real['accuracy'] if real['accuracy'] is not None else 'n/a'}, "
        f"synthetic-control reject recall={evaluation['synthetic_control_reject_recall'] if evaluation['synthetic_control_reject_recall'] is not None else 'n/a'}."
    )
