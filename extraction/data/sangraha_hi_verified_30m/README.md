# Sangraha Hindi (verified/hin) — ~30M-token sample

This directory is the dataset workspace for
`extraction/stepwise_cleaning.py`. The **full raw cache is not
versioned** (it is ~280MB); a **compact, deterministic sample** is tracked
instead so the notebook pipeline can be exercised and reviewed without
pulling the full corpus.

## What's tracked here

| Path | Size | Purpose |
|---|---|---|
| `sample_raw.jsonl` | 264 records / ~1.5MB | Deterministic sample for running/inspecting the pipeline |
| `download_dataset.py` | — | Standard-library script that streams the full source and rebuilds a raw cache locally |
| `manifest.json` | — | Provenance/metadata recorded from the pipeline run against the full local cache |
| `raw.jsonl`, `extraction_only/**/*.jsonl` | — | **Not tracked.** Full raw cache and extraction outputs, gitignored (see repo `.gitignore`) |

`sample_raw.jsonl` holds `{"doc_id", "type", "text"}` records only — the
minimal fields the pipeline needs to reproduce/inspect cleaning, dedup,
and extraction steps. It was produced deterministically by taking the
first rows, in original stream order, from a locally rebuilt full
`raw.jsonl` until the cumulative estimated-token count reached 150,000
(264 records land just over that threshold). To regenerate it from a
local `raw.jsonl` cache:

```bash
python3 - <<'PY'
import json

TARGET_TOKENS = 150_000
token_total = 0
with open("extraction/data/sangraha_hi_verified_30m/raw.jsonl", encoding="utf-8") as src, \
     open("extraction/data/sangraha_hi_verified_30m/sample_raw.jsonl", "w", encoding="utf-8") as dst:
    for line in src:
        row = json.loads(line)
        dst.write(json.dumps(
            {"doc_id": row["doc_id"], "type": row["type"], "text": row["text"]},
            ensure_ascii=False,
        ) + "\n")
        token_total += row["estimated_tokens"]
        if token_total >= TARGET_TOKENS:
            break
PY
```

## Rebuilding the full raw cache

1. **Dependency prerequisite:** the `datasets` package (already pinned in
   this repo's `pyproject.toml` / `uv.lock`). From the repo root:
   ```bash
   uv sync
   ```
   (or `pip install datasets` in whatever environment you run the script
   with).
2. **Run the script:**
   ```bash
   uv run python extraction/data/sangraha_hi_verified_30m/download_dataset.py
   ```
   With no arguments this streams `ai4bharat/sangraha` (config
   `verified`, split `hin`) at the pinned revision below and writes rows,
   in stream order, to `extraction/data/sangraha_hi_verified_30m/raw.jsonl`
   (**default output**) until the estimated-token total reaches
   **30,000,000** (**default target**) — the same target the pipeline
   notebooks use. The script does not read or require any existing local
   cache; each run streams fresh from Hugging Face.
3. **Adjust the run** with flags, e.g. a smaller local test pull:
   ```bash
   uv run python extraction/data/sangraha_hi_verified_30m/download_dataset.py \
       --target-tokens 150000 --output /tmp/small_raw.jsonl
   ```
   `--dataset`, `--config`, `--split`, and `--revision` are also
   overridable; see `--help`.

The script is standard-library-only at import time — `--help` works even
without `datasets` installed, since the `from datasets import
load_dataset` import is deferred until a download actually runs.

## Provenance

- **Source:** [`ai4bharat/sangraha`](https://huggingface.co/datasets/ai4bharat/sangraha), config `verified`, split `hin`
- **License:** CC BY 4.0
- **Pinned revision:** `8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8`
- **Selection:** deterministic first rows from the streaming split, in
  source order, until the estimated-token target is met (no shuffling —
  same run against the same revision reproduces the same rows, as
  verified while building this sample)

## Token-estimate semantics

`estimated_tokens` (present in a full `raw.jsonl` rebuild, and in the
pipeline's cache format) is **not** a real tokenizer count. It is the
same extraction-pipeline token proxy
(`extraction/stepwise_cleaning.py`):

```
estimated_tokens = ceil(non_whitespace_unit_count * 1.35)
```

i.e. whitespace-delimited "words" scaled by a fixed 1.35 fertility
assumption picked for Hindi/code-mixed text. It is only used to decide
when a download or sample has reached its target size, not as a
tokenizer-accurate count.

## Full data stays local

The full raw cache (~280MB) and full extraction outputs are
intentionally **not** committed, LFS-tracked, or pushed — only
`sample_raw.jsonl`, this README, `download_dataset.py`, and
`manifest.json` are versioned. Run `download_dataset.py` locally to
reconstruct the full cache; it is excluded via the repository's
`.gitignore`.
