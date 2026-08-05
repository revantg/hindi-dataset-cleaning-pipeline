# Session 4 · Hindi Dataset Cleaning

Hands-on data-cleaning case study for the ERA V5 Session 4 assignment. The pipeline applies the lecture's eight required strategies to a deterministic Sangraha Hindi sample and records measurable before/after results.

**Live report:** [session4-hindi-cleaning-rev.netlify.app](https://session4-hindi-cleaning-rev.netlify.app/)

## Assignment

> Work with your agent, and find in total "how many strategies" are listed in this session. Then find a 10–100M dataset like [this example](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled), and then apply these cleanups. Find interesting datasets, especially from Session 3. Finally create a widget that talks about:
>
> - how many strategies, and what they are
> - what dataset was picked
> - what was cleaned and why
> - any other strategy or concern that was cleaned up
> - final statistics
>
> Upload to Netlify and share the link.

The source assignment is preserved in [`axiom-fetched/session4_assignment.md`](axiom-fetched/session4_assignment.md). The completed interactive widget is [`assignment.py`](assignment.py).

## Dataset

- **Source:** [`ai4bharat/sangraha`](https://huggingface.co/datasets/ai4bharat/sangraha)
- **Configuration / split:** `verified` / `hin`
- **License:** CC BY 4.0
- **Pinned revision:** `8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8`
- **Selection:** first usable rows from the pinned streaming split, in source order, until about 30M estimated tokens
- **Token estimate:** `ceil(non-whitespace units × 1.35)`; this is a reproducible fertility proxy, not tokenizer output

The dataset workspace is [`extraction/data/sangraha_hi_verified_30m/`](extraction/data/sangraha_hi_verified_30m/). The compact deterministic sample is tracked; the roughly 280MB raw cache and full extraction outputs remain local and gitignored. Provenance is recorded in [`manifest.json`](extraction/data/sangraha_hi_verified_30m/manifest.json).

## Cleaning pipeline

The main pipeline runs these eight stages in order:

| # | Stage | What the pipeline does |
|---:|---|---|
| 1 | **Extract** | Removes residual HTML tags, scripts, layout artifacts, and explicit navigation/cookie boilerplate while preserving document text order. The separate [`extraction/stepwise_cleaning.py`](extraction/stepwise_cleaning.py) notebook evaluates source-aware HTML extraction methods. |
| 2 | **Normalize** | Decodes HTML entities, repairs selected mojibake, applies Unicode NFC, normalizes whitespace and line breaks, removes control/formatting artifacts, and removes inconsistent chat markers. Indic ZWJ/ZWNJ are preserved. |
| 3 | **Language ID and validation** | Checks the actual text for Hindi Devanagari, Hindi Latin-script, and Hindi–Latin code-mixed content instead of trusting the `hin` label. Other or unknown text is removed by the main case-study policy. The [`language-identification/`](language-identification/) notebook provides a broader detector comparison and routing study. |
| 4 | **Quality filtering** | Flags transparent low-value signals: too-short text, malformed symbols or repeated characters, link-heavy spam, low vocabulary diversity, and repetitive text. These are policy signals, not universal quality scores. |
| 5 | **Deduplication** | Removes exact duplicates with SHA-256, then uses a local MinHash/LSH approximation for near duplicates: 16 permutations, four-row bands, length-ratio guard, and similarity threshold. This is intentionally a bounded local approximation rather than global corpus deduplication. The separate [`deduplication/deduplication.py`](deduplication/deduplication.py) notebook surveys nine deduplication techniques and executes exact-hash plus FineWeb-parameter MinHash/LSH near-duplicate detection on a 20,000-document Sangraha sample. |
| 6 | **PII scrubbing** | Redacts email addresses, Indian phone numbers, IPv4 addresses, credential-like values, and Aadhaar-like numbers. The report exposes category counts only; removed values are never displayed. |
| 7 | **Decontamination** | Runs a limited textual proxy for named benchmark/evaluation markers such as GSM8K, MMLU, HellaSwag, ARC Challenge, and eval-harness terms. No full benchmark corpus is downloaded, so this is not a complete contamination guarantee. |
| 8 | **Manifest / provenance** | Writes source URL, dataset revision, license, split, selection rule, hashes, token-estimation rule, pipeline version, UTC run time, dedup method, decontamination scope, and final statistics. |

## Indic-specific concern

The pipeline avoids English-only assumptions. It preserves meaningful Brahmic Unicode components, keeps valid Latin-script Hindi and natural Hinglish when the language policy supports them, does not require English punctuation, and treats ghost/chat-template markers as a normalization concern rather than a ninth canonical stage.

## Repository map

- [`assignment.py`](assignment.py) — main Marimo assignment widget and eight-stage case-study pipeline
- [`extraction/stepwise_cleaning.py`](extraction/stepwise_cleaning.py) — HTML extraction technique workbook
- [`extraction/site/index.html`](extraction/site/index.html) — local extraction-method report
- [`language-identification/language_identification.py`](language-identification/language_identification.py) — bounded Indic language-ID comparison and routing study
- [`quality-filtering/quality_filtering.py`](quality-filtering/quality_filtering.py) — bounded quality-signal workbook
- [`deduplication/deduplication.py`](deduplication/deduplication.py) — nine-technique deduplication survey with one executed MinHash/LSH pipeline
- [`extraction/data/sangraha_hi_verified_30m/download_dataset.py`](extraction/data/sangraha_hi_verified_30m/download_dataset.py) — deterministic source-cache rebuild script

## Run locally

From the repository root:

```bash
uv sync
uv run marimo edit assignment.py
```

The widget lets you choose a 10–30M estimated-token target and run all eight stages. Related workbooks can be opened with:

```bash
uv run marimo edit extraction/stepwise_cleaning.py
uv run marimo edit language-identification/language_identification.py
uv run marimo edit quality-filtering/quality_filtering.py
```

To rebuild the local 30M-token raw cache:

```bash
uv run python extraction/data/sangraha_hi_verified_30m/download_dataset.py
```

Use `--target-tokens` and `--output` for a smaller test pull; see the dataset [`README.md`](extraction/data/sangraha_hi_verified_30m/README.md) for reproducibility details.

## Reproducibility boundaries

The final report records stage-level document, character, and estimated-token counts; removal reasons; language mix; exact versus near-duplicate counts; PII category counts; decontamination scope/results; and manifest fields. The token count is estimated, near-deduplication is local, and decontamination is a documented proxy. These limits are part of the result rather than hidden assumptions.
