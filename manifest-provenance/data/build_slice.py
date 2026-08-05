#!/usr/bin/env python3
"""Build the deterministic Sangraha Hindi slice for the manifest/provenance stage.

Streams `ai4bharat/sangraha` (`data_dir="verified/hin"`) at a pinned revision
and keeps rows, in stream order, until the cumulative real `o200k_base` token
count reaches the target. No shuffling and no content-conditioned selection, so
the same revision reproduces the same rows.

Two artifacts are written next to this script:

* `sangraha_hin_slice.jsonl` — `{"doc_id", "type", "text"}` in stream order.
* `slice_manifest.json` — the provenance block for that file, including the
  tokenizer name and the sha256 of the tokenizer's own BPE artifact.

Standard-library-only at import time: `datasets` and `tiktoken` are imported
inside `download()` so `--help` works without them.

Usage (from the repo root):

    uv run python manifest-provenance/data/build_slice.py
    uv run python manifest-provenance/data/build_slice.py --target-tokens 50000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

DATASET_ID = "ai4bharat/sangraha"
DATASET_DATA_DIR = "verified/hin"
DATASET_SPLIT = "train"
# Same pinned revision the extraction stage uses; streaming against a fixed
# revision is what makes "first N rows in stream order" a reproducible rule.
DATASET_REVISION = "8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8"
DATASET_LICENSE = "CC-BY-4.0"

TOKENIZER_NAME = "o200k_base"
DEFAULT_TARGET_TOKENS = 150_000

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "sangraha_hin_slice.jsonl"
MANIFEST_NAME = "slice_manifest.json"
TIKTOKEN_CACHE_DIR = SCRIPT_DIR / ".tiktoken_cache"

SELECTION_RULE = (
    "Stream the pinned revision in source order and keep every row until the "
    "cumulative real o200k_base token count reaches the target, then stop. No "
    "shuffling, no deduplication, no content-conditioned selection."
)

REBUILD_COMMAND = "uv run python manifest-provenance/data/build_slice.py"


def canonical_json(obj: object) -> bytes:
    """Sorted-key, minimally separated, UTF-8 JSON bytes.

    An approximation of RFC 8785 (JCS): it does not implement ECMAScript
    number formatting. Every numeric field written by this script is an int,
    for which the two agree.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream ai4bharat/sangraha (verified/hin) at a pinned revision and "
            "write a deterministic JSONL slice plus its provenance manifest, "
            "stopping once the real o200k_base token target is reached."
        )
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=DEFAULT_TARGET_TOKENS,
        help=f"o200k_base token budget to stop at (default: {DEFAULT_TARGET_TOKENS:,}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output JSONL path (default: sangraha_hin_slice.jsonl next to this "
            "script). The manifest is written alongside it as "
            f"{MANIFEST_NAME}. Overwritten if it already exists."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_ID,
        help=f"Hugging Face dataset id (default: {DATASET_ID}).",
    )
    parser.add_argument(
        "--data-dir",
        default=DATASET_DATA_DIR,
        help=(
            "Dataset data_dir (default: "
            f"{DATASET_DATA_DIR}). Sangraha exposes its splits as data_dir "
            "paths, not as named configs."
        ),
    )
    parser.add_argument(
        "--split",
        default=DATASET_SPLIT,
        help=f"Dataset split (default: {DATASET_SPLIT}).",
    )
    parser.add_argument(
        "--revision",
        default=DATASET_REVISION,
        help=f"Pinned Hugging Face dataset revision (default: {DATASET_REVISION}).",
    )
    return parser.parse_args(argv)


def _fail(message: str) -> None:
    """Print a loud failure with the rebuild command and exit non-zero."""
    print(f"ERROR: {message}", file=sys.stderr)
    print(f"Rebuild with: {REBUILD_COMMAND}", file=sys.stderr)
    raise SystemExit(1)


def download(args: argparse.Namespace) -> dict:
    """Stream the slice to a temp path, rename on success, return the manifest.

    Never leaves a partial slice behind: the output is written to
    `<output>.partial` and renamed only after the stream completes.
    """
    TIKTOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE_DIR))

    try:
        import tiktoken
        from datasets import load_dataset
    except ImportError as exc:  # deferred imports: only needed for a real run
        _fail(f"missing dependency ({exc}). Run `uv sync` first.")

    encoding = tiktoken.get_encoding(TOKENIZER_NAME)

    try:
        stream = load_dataset(
            args.dataset,
            data_dir=args.data_dir,
            revision=args.revision,
            split=args.split,
            streaming=True,
        )
    except Exception as exc:  # noqa: BLE001 - HF raises many unrelated types; all are fatal here
        _fail(f"could not open the Hugging Face stream: {exc!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")

    records = 0
    token_total = 0
    byte_total = 0
    try:
        with partial.open("w", encoding="utf-8") as handle:
            for source_row in stream:
                doc_id = str(source_row.get("doc_id") or "")
                text = source_row.get("text") or ""
                if not doc_id or not text:
                    continue
                record = {
                    "doc_id": doc_id,
                    "type": source_row.get("type", "unknown"),
                    "text": text,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records += 1
                token_total += len(encoding.encode(text, disallowed_special=()))
                byte_total += len(text.encode("utf-8"))
                if token_total >= args.target_tokens:
                    break
    except Exception as exc:  # noqa: BLE001 - any mid-stream failure must delete the partial
        partial.unlink(missing_ok=True)
        _fail(f"stream failed after {records} records: {exc!r}")

    if records == 0:
        partial.unlink(missing_ok=True)
        _fail("stream yielded no usable rows; refusing to write an empty slice.")

    partial.replace(args.output)

    bpe_files = sorted(TIKTOKEN_CACHE_DIR.glob("*"))
    tokenizer_bpe_sha256 = (
        sha256_file(bpe_files[0]) if len(bpe_files) == 1 else "unavailable"
    )

    return {
        "dataset_id": args.dataset,
        "revision": args.revision,
        "data_dir": args.data_dir,
        "split": args.split,
        "license": DATASET_LICENSE,
        "selection_rule": SELECTION_RULE,
        "target_tokens": args.target_tokens,
        "records": records,
        "tokenizer": TOKENIZER_NAME,
        "tokenizer_bpe_sha256": tokenizer_bpe_sha256,
        "token_count": token_total,
        "byte_count": byte_total,
        "slice_sha256": sha256_file(args.output),
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = download(args)

    manifest_path = args.output.parent / MANIFEST_NAME
    manifest_path.write_bytes(canonical_json(manifest) + b"\n")

    print(
        f"Wrote {manifest['records']:,} records "
        f"({manifest['token_count']:,} {TOKENIZER_NAME} tokens, "
        f"{manifest['byte_count']:,} text bytes) to {args.output}"
    )
    print(f"Wrote provenance manifest to {manifest_path}")
    print(f"slice sha256: {manifest['slice_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
