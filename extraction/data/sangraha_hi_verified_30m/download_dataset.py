#!/usr/bin/env python3
"""Rebuild the local Sangraha Hindi raw JSONL cache.

Streams ``ai4bharat/sangraha`` (config ``verified``, split ``hin``) from
Hugging Face at a pinned revision and writes deterministic first-seen rows,
in stream order, to a JSONL file until a configurable estimated-token
budget is reached (default: 30,000,000 estimated tokens -- the full local
raw cache used by the cleaning pipeline in this repository).

This script is standard-library-only at import time (so ``--help`` works
without any extra dependency installed). The third-party ``datasets``
package is imported lazily, inside ``download()``, only once a run is
actually requested. Install it first (e.g. ``uv sync`` from the repo root,
or ``pip install datasets``) before running a real download.

The full raw cache produced by a 30M-token run is ~280MB and is
intentionally NOT committed to this repository (see the sibling README and
.gitignore); this script is how a collaborator reconstructs it locally.
It never reads or requires that full cache to run -- each invocation
streams fresh from the pinned Hugging Face revision.

Usage:
    python data/sangraha_hi_verified_30m/download_dataset.py
    python data/sangraha_hi_verified_30m/download_dataset.py --target-tokens 150000 \\
        --output /tmp/small_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

DATASET_ID = "ai4bharat/sangraha"
DATASET_CONFIG = "verified"
DATASET_SPLIT = "hin"
# Revision pinned by the extraction notebook (see
# extraction/data/sangraha_hi_verified_30m/manifest.json and
# extraction/stepwise_cleaning.py); streaming against a fixed revision
# keeps re-downloads reproducible.
DATASET_REVISION = "8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8"

# Same fertility assumption used throughout the extraction notebook:
# estimated_tokens = ceil(non-whitespace units * 1.35). It is a cheap
# proxy, not a real tokenizer, but it controls when streaming stops.
TOKEN_FERTILITY_ASSUMPTION = 1.35
DEFAULT_TARGET_TOKENS = 30_000_000
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "raw.jsonl"


def estimate_tokens(text: str) -> int:
    """Estimate token count as ceil(non-whitespace units * 1.35)."""
    units = len(re.findall(r"\S+", text or ""))
    return int(math.ceil(units * TOKEN_FERTILITY_ASSUMPTION)) if units else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream ai4bharat/sangraha (verified/hin) from Hugging Face and "
            "write a deterministic raw JSONL cache, stopping once the "
            "estimated-token target is reached."
        )
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=DEFAULT_TARGET_TOKENS,
        help=(
            "Estimated-token budget to stop at (default: "
            f"{DEFAULT_TARGET_TOKENS:,})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output JSONL path (default: raw.jsonl next to this script, "
            "i.e. data/sangraha_hi_verified_30m/raw.jsonl). Overwritten if "
            "it already exists."
        ),
    )
    parser.add_argument(
        "--dataset", default=DATASET_ID, help=f"Hugging Face dataset id (default: {DATASET_ID})."
    )
    parser.add_argument(
        "--config", default=DATASET_CONFIG, help=f"Dataset config (default: {DATASET_CONFIG})."
    )
    parser.add_argument(
        "--split", default=DATASET_SPLIT, help=f"Dataset split (default: {DATASET_SPLIT})."
    )
    parser.add_argument(
        "--revision",
        default=DATASET_REVISION,
        help=f"Pinned Hugging Face dataset revision (default: {DATASET_REVISION}).",
    )
    return parser.parse_args(argv)


def download(args: argparse.Namespace) -> tuple[int, int]:
    """Stream the dataset and write rows until the token target is hit.

    Returns (documents_written, estimated_tokens_written).
    """
    from datasets import load_dataset  # deferred: only needed for a real run

    stream = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        revision=args.revision,
        streaming=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    documents = 0
    token_total = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for source_row in stream:
            doc_id = str(source_row.get("doc_id") or "")
            text = source_row.get("text") or ""
            if not doc_id or not text or doc_id in seen_ids:
                continue
            tokens = estimate_tokens(text)
            record = {
                "doc_id": doc_id,
                "type": source_row.get("type", "unknown"),
                "text": text,
                "estimated_tokens": tokens,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            seen_ids.add(doc_id)
            documents += 1
            token_total += tokens
            if token_total >= args.target_tokens:
                break

    return documents, token_total


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    documents, token_total = download(args)
    print(
        f"Wrote {documents:,} documents (~{token_total:,} estimated tokens) "
        f"to {args.output} "
        f"[{args.dataset}/{args.config}/{args.split} @ {args.revision}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
