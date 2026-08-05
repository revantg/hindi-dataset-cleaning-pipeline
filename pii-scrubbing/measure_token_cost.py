#!/usr/bin/env python3
"""Measure the tokenizer cost of each PII replacement policy.

Run with the project venv so `tiktoken` resolves:
    .venv/bin/python pii-scrubbing/measure_token_cost.py

The point is not the absolute token count -- `cl100k_base` is not the tokenizer a
Hindi model would ship -- but the RELATIVE cost of the four replacement policies
over the same spans. A raw identifier is a long run of rare subwords; the policy
that replaces it decides how many tokens the corpus spends on content that
teaches the model nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tiktoken

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pii_techniques as pt  # noqa: E402

DATA = HERE / "data"

# Four real replacement policies from the pipelines surveyed in the notebook.
POLICIES = {
    "keep raw (no scrubbing)": None,
    "dolma  ` |||EMAIL_ADDRESS||| `": {
        "email": " |||EMAIL_ADDRESS||| ",
        "phone_in": " |||PHONE_NUMBER||| ",
        "phone_landline": " |||PHONE_NUMBER||| ",
        "phone_tollfree": " |||PHONE_NUMBER||| ",
        "ipv4": " |||IP_ADDRESS||| ",
    },
    "bigscience `PI:EMAIL`": {
        "email": "PI:EMAIL",
        "phone_in": "PI:KEY",
        "phone_landline": "PI:KEY",
        "phone_tollfree": "PI:KEY",
        "ipv4": "PI:IP_ADDRESS",
    },
    "bigcode `<EMAIL>`": {
        "email": "<EMAIL>",
        "phone_in": "<PHONE>",
        "phone_landline": "<PHONE>",
        "phone_tollfree": "<PHONE>",
        "ipv4": "172.16.31.10",
    },
}


def main() -> None:
    encoder = tiktoken.get_encoding("cl100k_base")
    records = [
        json.loads(line)
        for config in ("verified", "unverified")
        for line in (DATA / f"sangraha_hin_{config}_probe.jsonl").open(encoding="utf-8")
    ]

    # Only documents the scrubber actually changes carry a cost difference.
    touched = []
    for record in records:
        result = pt.scrub(record["text"])
        spans = [span for span in result.accepted if span.category != "handle"]
        if spans:
            touched.append((record["text"], spans))

    rows = []
    for name, mapping in POLICIES.items():
        total = 0
        for text, spans in touched:
            if mapping is None:
                rendered = text
            else:
                rendered = text
                for span in sorted(spans, key=lambda s: s.start, reverse=True):
                    token = mapping.get(span.category, pt.PLACEHOLDERS.get(span.category, "[PII]"))
                    rendered = rendered[: span.start] + token + rendered[span.end :]
            total += len(encoder.encode(rendered))
        rows.append({"policy": name, "tokens": total})

    baseline = rows[0]["tokens"]
    for row in rows:
        row["delta_vs_raw"] = row["tokens"] - baseline
        row["pct"] = round((row["tokens"] - baseline) / baseline * 100, 4)

    # Per-span cost is the number that actually generalises to corpus scale.
    span_total = sum(len(spans) for _, spans in touched)
    span_rows = []
    for name, mapping in POLICIES.items():
        if mapping is None:
            continue
        cost = sum(
            len(encoder.encode(mapping.get(category, "[PII]")))
            for _, spans in touched
            for category in (span.category for span in spans)
        )
        span_rows.append(
            {
                "policy": name,
                "tokens_spent_on_placeholders": cost,
                "tokens_per_span": round(cost / span_total, 3),
            }
        )

    raw_cost = sum(
        len(encoder.encode(text[span.start : span.end])) for text, spans in touched for span in spans
    )
    payload = {
        "tokenizer": "cl100k_base (relative comparison only, not a Hindi tokenizer)",
        "documents_touched": len(touched),
        "spans_replaced": span_total,
        "tokens_in_raw_identifiers": raw_cost,
        "tokens_per_raw_identifier": round(raw_cost / span_total, 3),
        "whole_corpus": rows,
        "placeholder_cost": span_rows,
    }
    (DATA / "token_cost.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
