#!/usr/bin/env python3
"""Deterministic Sangraha Hindi probe cache for the PII-scrubbing workbook.

Pulls fixed stride windows from the pinned `ai4bharat/sangraha` revision via the
Hugging Face datasets-server `/rows` endpoint and writes them to JSONL. The
stride rule, window size and revision are recorded in a manifest so the cache is
rebuildable without shuffling or cherry-picking documents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATASET = "ai4bharat/sangraha"
REVISION = "8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
SIZE_ENDPOINT = "https://datasets-server.huggingface.co/size"
WINDOW = 100
HERE = Path(__file__).resolve().parent


def _get(url: str, timeout: int = 90) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "s4-pii-workbook/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        return json.load(handle)


def split_size(config: str, split: str) -> int:
    payload = _get(f"{SIZE_ENDPOINT}?dataset={urllib.parse.quote(DATASET, safe='')}")
    for entry in payload["size"]["splits"]:
        if entry["config"] == config and entry["split"] == split:
            return int(entry["num_rows"])
    raise SystemExit(f"split not found: {config}/{split}")


def fetch_window(config: str, split: str, offset: int, length: int) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            payload = _get(f"{ROWS_ENDPOINT}?{query}")
            return [row["row"] for row in payload["rows"]]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(2 * (attempt + 1))
    print(f"  ! window {config}/{split}@{offset} failed: {last_error}")
    return []


def build(config: str, split: str, windows: int, output: Path) -> dict:
    total_rows = split_size(config, split)
    stride = max(WINDOW, total_rows // windows)
    seen: set[str] = set()
    records: list[dict] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index in range(windows):
            offset = index * stride
            if offset >= total_rows:
                break
            for row in fetch_window(config, split, offset, WINDOW):
                doc_id = row.get("doc_id")
                text = row.get("text") or ""
                if not doc_id or doc_id in seen or not text.strip():
                    continue
                seen.add(doc_id)
                record = {
                    "doc_id": doc_id,
                    "source_config": config,
                    "source_split": split,
                    "source_offset": offset,
                    "type": row.get("type"),
                    "text": text,
                }
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  window {index + 1}/{windows} offset={offset} kept={len(records)}", flush=True)
    characters = sum(len(record["text"]) for record in records)
    units = sum(len(record["text"].split()) for record in records)
    return {
        "config": config,
        "split": split,
        "split_rows_total": total_rows,
        "windows": windows,
        "window_size": WINDOW,
        "stride": stride,
        "documents": len(records),
        "characters": characters,
        "whitespace_units": units,
        "estimated_tokens": math.ceil(units * 1.35),
        "path": output.name,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--outdir", type=Path, default=HERE)
    args = parser.parse_args()

    summaries = []
    for config in ("verified", "unverified"):
        print(f"== {config}/hin")
        summaries.append(
            build(config, "hin", args.windows, args.outdir / f"sangraha_hin_{config}_probe.jsonl")
        )

    manifest = {
        "dataset_id": DATASET,
        "dataset_revision": REVISION,
        "license": "CC-BY-4.0",
        "access_method": "huggingface datasets-server /rows endpoint",
        "selection_rule": (
            "For each config, take fixed stride windows across the whole split: "
            "stride = floor(split_rows / windows), window = 100 consecutive rows from each "
            "offset, preserving stream order. No shuffling, no PII-conditioned selection."
        ),
        "token_estimator": "ceil(whitespace units x 1.35)",
        "materialized_at_utc": datetime.now(timezone.utc).isoformat(),
        "splits": summaries,
    }
    (args.outdir / "probe_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
