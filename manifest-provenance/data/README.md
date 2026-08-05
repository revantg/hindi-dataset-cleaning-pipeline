# Sangraha Hindi (verified/hin) — manifest/provenance slice

This directory is the **self-contained** dataset workspace for
`manifest-provenance/manifest_provenance.py`. It never reads another stage
folder's data, so the workbook can be built and reviewed independently.

## What's tracked here

| Path | Size | Purpose |
|---|---|---|
| `sangraha_hin_slice.jsonl` | 225 records / ~1.2MB | Deterministic slice the workbook emits manifests over |
| `slice_manifest.json` | — | Provenance block recorded by the build script for that slice |
| `build_slice.py` | — | Standard-library-at-import script that rebuilds the slice from Hugging Face |
| `manifests/` | — | Manifests emitted by the notebook run (record/shard/corpus levels) |
| `metrics.json` | — | Executed measurements from the notebook run |
| `.tiktoken_cache/` | ~3.6MB | **Not tracked.** Local `tiktoken` BPE cache, gitignored |
| `shards/` | — | **Not tracked.** Scratch per-shard splits, gitignored |

`sangraha_hin_slice.jsonl` holds `{"doc_id", "type", "text"}` records only.

## Why this size

The stage under study is bookkeeping over bytes. Its claims — determinism,
canonical form, duplicate collapse, tamper detection, inclusion proofs — are
**exact and size-independent**: they either hold on 225 records or they are
broken. A larger slice would add committed bytes and no new evidence. The
~1.2MB size matches the 1.5MB precedent of
`extraction/data/sangraha_hi_verified_30m/sample_raw.jsonl`.

Four shards, not one: the record → shard → corpus hierarchy and a corpus root
computed over shard roots cannot be demonstrated with a single shard.

## Rebuilding

From the repo root:

```bash
uv sync
uv run python manifest-provenance/data/build_slice.py
```

With no arguments this streams `ai4bharat/sangraha` (`data_dir="verified/hin"`,
split `train`) at the pinned revision below and writes rows, in stream order,
until the cumulative **real `o200k_base`** token count reaches **150,000** (225
records land just over that threshold at 150,139 tokens). Flags
`--target-tokens`, `--output`, `--dataset`, `--data-dir`, `--split`, and
`--revision` are overridable; see `--help`.

The script is standard-library-only at import time — `--help` works without
`datasets` or `tiktoken` installed, since both imports are deferred into the
download function. It writes to a `.partial` path and renames on success, so a
failed stream never leaves a truncated slice behind. If the stream is
unavailable it fails loudly with the rebuild command rather than substituting
data; because the slice is tracked, the notebook still runs.

## Provenance

- **Source:** [`ai4bharat/sangraha`](https://huggingface.co/datasets/ai4bharat/sangraha), `data_dir="verified/hin"`, split `train`
- **License:** CC BY 4.0 (`cc-by-4.0` on the dataset card)
- **Pinned revision:** `8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8`
- **Selection rule:** stream in source order, keep rows until the cumulative
  `o200k_base` token count reaches 150,000, then stop. No shuffling, no
  deduplication, no content-conditioned selection — so the same revision
  reproduces the same rows. Row 0 is
  `doc_id=c729b49b2d13dea710df863877cb3266df6e1a9e`, which matches the first row
  of `extraction/data/sangraha_hi_verified_30m/sample_raw.jsonl`, confirming the
  stream order is deterministic across independent pulls.
- **Slice digest:** recorded as `slice_sha256` in `slice_manifest.json`.

## Token-count semantics

Counts here are **real `o200k_base` counts** from `tiktoken`, not the
`words × 1.35` proxy used by the extraction stage. The proxy is still computed
and reported in the notebook for exactly one reason: to quantify its error. On
this slice `words × 1.35` = 125,678 against a real `o200k_base` count of
150,139 — the proxy **undercounts by 16.3%**.

`o200k_base` is not the tokenizer a Hindi model would actually use. The point
is that the count is *named and reproducible*, not that it is optimal: the same
text measures 464,805 tokens under `cl100k_base`, a 3.1× spread. A token count
without a named tokenizer is meaningless.

**Tokenizer determinism.** `tiktoken` downloads its BPE file on first use and
caches it. `build_slice.py` pins `TIKTOKEN_CACHE_DIR` to
`manifest-provenance/data/.tiktoken_cache` (gitignored) and records that file's
own sha256 as `tokenizer_bpe_sha256` in the manifest, so the measuring
instrument is itself content-addressed. The `o200k_base` artifact is 3,613,922
bytes with digest `446a9538cb6c348e3516120d…`; `n_vocab` is 200,019.
