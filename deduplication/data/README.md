# Deduplication workbook dataset — Sangraha Hindi (`verified`/`hin`), 20,000 documents

This directory is the dataset workspace owned by
`deduplication/deduplication.py` (Session 4, Stage 5). The raw cache is
**not versioned** (20,000 documents ≈ 44 MB); only the provenance
manifest, the aggregate metrics summary, this README, and the local
`.gitignore` are tracked.

## Provenance

- **Source:** [`ai4bharat/sangraha`](https://huggingface.co/datasets/ai4bharat/sangraha)
- **Config:** `verified`
- **Split:** `hin`
- **Pinned revision:** `8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8`
- **License:** CC BY 4.0
- **Selection rule:** the first 20,000 rows of the streaming split, in
  deterministic source order, keeping rows with a non-empty `doc_id` and
  non-whitespace `text`. No shuffling, no transformation, no filtering
  beyond that emptiness check. The same revision reproduces the same rows.
- **Record schema:** `{"doc_id", "type", "text"}` JSONL, one record per line.

`verified`/`hin` was chosen over `unverified`/`hin` on measured evidence:
the same 20,000-document measurement finds **1** duplicate in
`unverified`/`hin` versus **328** exact and **28** near duplicates in
`verified`/`hin`. A demonstration on `unverified` would produce a null
result.

## What is tracked here

| Path | Tracked | Purpose |
|---|---|---|
| `README.md` | yes | This provenance record |
| `.gitignore` | yes | Excludes the raw cache and per-pair results |
| `selection_manifest.json` | yes | Source identifiers, selection rule, counts, `raw_sha256` |
| `dedup_metrics.json` | yes | Aggregate run summary: every count, all parameters, timings |
| `raw.jsonl` | **no** | 20,000-document cache, ≈44 MB — rebuild locally |
| `results/duplicate_pairs.jsonl` | **no** | One row per verified candidate pair |
| `results/clusters.json` | **no** | Near-duplicate components and their kept representative |

`raw_sha256` in the manifest is recomputed and revalidated on every
reload, so a truncated or mutated cache is rejected rather than silently
used.

## Rebuilding the raw cache

From the repository root:

```bash
uv sync
uv run marimo edit deduplication/deduplication.py
```

Then click **Materialize 20,000-document Sangraha verified/hin cache**
(≈41 s). This streams the pinned revision and writes
`deduplication/data/raw.jsonl` atomically (temporary sibling file plus
`os.replace`) together with `selection_manifest.json`.

Headless equivalent, from the repository root:

```bash
uv run python - <<'PY'
import json, os
from pathlib import Path
from datasets import load_dataset

out = Path("deduplication/data/raw.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_name(".raw.jsonl.download.tmp")
kept = 0
with tmp.open("w", encoding="utf-8") as handle:
    for row in load_dataset(
        "ai4bharat/sangraha", "verified", split="hin",
        revision="8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8", streaming=True,
    ):
        doc_id, text = str(row.get("doc_id") or ""), row.get("text") or ""
        if not doc_id or not isinstance(text, str) or not text.strip():
            continue
        handle.write(json.dumps(
            {"doc_id": doc_id, "type": str(row.get("type") or "unknown"), "text": text},
            ensure_ascii=False,
        ) + "\n")
        kept += 1
        if kept == 20_000:
            break
os.replace(tmp, out)
print(kept, "records")
PY
```

The notebook still writes the manifest on its own materialization path;
the snippet above only reconstructs `raw.jsonl`.

## Token-estimate semantics

`estimated_tokens` is **not** a tokenizer count. It is the repository's
shared proxy:

```
estimated_tokens = ceil(non_whitespace_unit_count * 1.35)
```

whitespace-delimited units scaled by a fixed 1.35 fertility assumption
chosen for Hindi and code-mixed text. It is used only to report corpus
size consistently across workbooks.

## Normalization note

Hashing normalization is whitespace collapse plus `casefold()` only. No
NFKC, no ZWJ/ZWNJ/virama stripping, no punctuation stripping — each of
those alters Devanagari. The normalized string is used solely to derive a
hash or a shingle; the retained document text is never mutated.
