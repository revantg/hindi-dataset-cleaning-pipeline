import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def workbook_scope(mo):
    mo.md(
        r"""
# Session 4 — Quality Filtering for Hindi/Indic Text

## Scope and decision boundary

This notebook teaches **intrinsic document-quality signals** on a bounded,
deterministic slice of `ai4bharat/sangraha` (`verified` / `hin`). It does not
claim that Sangraha is raw, that `verified` means perfect, or that 200 records
validate production thresholds. The observed distributions are measurements
of this teaching batch; the isolated controls below demonstrate mechanisms
without manufacturing corpus removals.

Quality is a **decision policy, not a universal score**. A signal can identify
material for review without proving that a document is bad. Hindi, Hinglish,
code, verse, lists, tables, OCR, and Devanagari combining marks need
language-aware review rather than English assumptions.

## The eight Session 4 stages

| # | Stage | Boundary in this workbook |
|---:|---|---|
| 1 | Extract | Separate assignment stage; HTML/layout extraction is not inferred from Sangraha's flattened `text`. |
| 2 | Normalize | NFC is used only on an in-memory measurement copy; raw cache is never overwritten. |
| 3 | Language validation | Separate stage; the `hin` label is not treated as proof of document language. |
| 4 | **Quality filtering** | **Executed here:** transparent length, Unicode, residue, link, line, and repetition signals. |
| 5 | Deduplication | Separate inter-document exact/near dedup stage; repeated lines here are intra-document review signals only. |
| 6 | PII scrubbing | Separate canonical scrubber; candidate previews use display-only masking and never expose values. |
| 7 | Decontamination | Separate benchmark/evaluation-content check; no benchmark corpus is silently substituted here. |
| 8 | Manifest / provenance | The bounded download writes a source manifest; this notebook also writes result and summary artifacts. |

AI4Bharat's [Sangraha dataset card](https://huggingface.co/datasets/ai4bharat/sangraha)
and the primary [IndicLLMSuite paper](https://arxiv.org/abs/2403.06350) describe
Setu as computing text statistics and applying line-length, safety-word, and
repetition filters while keeping fuzzy MinHash/LSH deduplication separate. That
separation is preserved here. The [Session 4 lecture transcript](deep-tutor.out.md)
is the course source for the eight-stage assignment framing.

### Primary reading

[Gopher](https://arxiv.org/abs/2112.11446) ·
[C4](https://arxiv.org/abs/1910.10683) ·
[FineWeb](https://arxiv.org/abs/2406.17557) ·
[CCNet](https://arxiv.org/abs/1911.00359) ·
[IndicLLMSuite](https://arxiv.org/abs/2403.06350) ·
[AI4Bharat Sangraha / Setu documentation](https://huggingface.co/datasets/ai4bharat/sangraha)

The source contract below intentionally uses a local 200-document batch, not
the separate 30M-token case-study cache. Estimated tokens use
`ceil(non-whitespace-units × 1.35)` as a teaching convention, never as tokenizer
output.
"""
    )
    return


@app.cell(hide_code=True)
def bounded_corpus_setup():
    import hashlib
    import json
    import math
    import os
    import re
    from datetime import datetime, timezone
    from pathlib import Path

    PIPELINE_VERSION = "s4-quality-filtering-2026-08-02-v1"
    DATASET_ID = "ai4bharat/sangraha"
    DATASET_CONFIG = "verified"
    DATASET_SPLIT = "hin"
    DATASET_REVISION = "8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8"
    DATASET_LICENSE = "CC BY 4.0"
    SOURCE_URL = "https://huggingface.co/datasets/ai4bharat/sangraha"
    SAMPLE_DOCUMENTS = 200
    TOKEN_FERTILITY_ASSUMPTION = 1.35
    DATA_ROOT = Path("data/quality_filtering_sangraha_hin_200")
    RAW_PATH = DATA_ROOT / "raw.jsonl"
    MANIFEST_PATH = DATA_ROOT / "selection_manifest.json"
    RESULTS_PATH = DATA_ROOT / "quality_signals.jsonl"
    SUMMARY_PATH = DATA_ROOT / "quality_summary.json"

    def estimate_tokens(text):
        units = len(re.findall(r"\S+", text or ""))
        return int(math.ceil(units * TOKEN_FERTILITY_ASSUMPTION)) if units else 0

    def record_stats(records):
        return {
            "documents": len(records),
            "characters": sum(len(record.get("text", "")) for record in records),
            "estimated_tokens": sum(
                estimate_tokens(record.get("text", "")) for record in records
            ),
        }

    def _sha256_path(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _atomic_write_text(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(value, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _read_raw_records():
        records = []
        with RAW_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def cache_status():
        if not RAW_PATH.exists() or not MANIFEST_PATH.exists():
            return False, None, "No matching bounded cache is present."
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            expected = {
                "dataset_id": DATASET_ID,
                "config": DATASET_CONFIG,
                "split": DATASET_SPLIT,
                "revision": DATASET_REVISION,
            }
            if any(manifest.get(key) != value for key, value in expected.items()):
                return False, manifest, "The cache manifest has different source identifiers."
            if manifest.get("retained_documents") != SAMPLE_DOCUMENTS:
                return False, manifest, "The cache manifest does not contain exactly 200 records."
            records = _read_raw_records()
            if len(records) != SAMPLE_DOCUMENTS:
                return False, manifest, "The raw cache does not contain exactly 200 records."
            if manifest.get("raw_sha256") != _sha256_path(RAW_PATH):
                return False, manifest, "The raw cache hash does not match its manifest."
            return True, manifest, "Matching bounded cache verified."
        except (OSError, ValueError, TypeError, KeyError) as error:
            return False, None, f"Cache validation failed: {type(error).__name__}: {error}"

    def materialize_cache():
        from datasets import load_dataset

        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = RAW_PATH.with_name(f".{RAW_PATH.name}.download.tmp")
        records = []
        try:
            stream = load_dataset(
                DATASET_ID,
                DATASET_CONFIG,
                split=DATASET_SPLIT,
                revision=DATASET_REVISION,
                streaming=True,
            )
            with temporary.open("w", encoding="utf-8") as handle:
                for source_row in stream:
                    doc_id = str(source_row.get("doc_id") or "")
                    text = source_row.get("text") or ""
                    if not doc_id or not isinstance(text, str) or not text.strip():
                        continue
                    record = {
                        "doc_id": doc_id,
                        "type": str(source_row.get("type") or "unknown"),
                        "text": text,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records.append(record)
                    if len(records) == SAMPLE_DOCUMENTS:
                        break
            if len(records) != SAMPLE_DOCUMENTS:
                raise RuntimeError(
                    f"The pinned stream ended after {len(records)} non-empty records; "
                    f"expected {SAMPLE_DOCUMENTS}."
                )
            os.replace(temporary, RAW_PATH)
            stats = record_stats(records)
            manifest = {
                "pipeline_version": PIPELINE_VERSION,
                "source_url": SOURCE_URL,
                "dataset_id": DATASET_ID,
                "config": DATASET_CONFIG,
                "split": DATASET_SPLIT,
                "revision": DATASET_REVISION,
                "license": DATASET_LICENSE,
                "field_schema": ["doc_id", "type", "text"],
                "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "selection_rule": (
                    "first 200 non-empty rows in deterministic streaming source order; "
                    "no shuffle or transformation"
                ),
                "requested_documents": SAMPLE_DOCUMENTS,
                "retained_documents": len(records),
                "characters": stats["characters"],
                "estimated_tokens": stats["estimated_tokens"],
                "raw_path": str(RAW_PATH),
                "raw_sha256": _sha256_path(RAW_PATH),
            }
            _atomic_write_text(
                MANIFEST_PATH,
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            )
            return records, manifest
        finally:
            if temporary.exists():
                temporary.unlink()

    def load_bounded_records():
        valid, manifest, message = cache_status()
        if not valid:
            return [], manifest, message
        return _read_raw_records(), manifest, message

    return (
        DATASET_CONFIG,
        DATASET_ID,
        DATASET_LICENSE,
        DATASET_REVISION,
        DATASET_SPLIT,
        DATA_ROOT,
        MANIFEST_PATH,
        PIPELINE_VERSION,
        RAW_PATH,
        RESULTS_PATH,
        SAMPLE_DOCUMENTS,
        SOURCE_URL,
        SUMMARY_PATH,
        cache_status,
        estimate_tokens,
        load_bounded_records,
        materialize_cache,
        record_stats,
    )


@app.cell(hide_code=True)
def quality_metrics_setup(estimate_tokens):
    import re as _re
    import unicodedata as _unicodedata
    from collections import Counter as _Counter

    _HTML_RESIDUE_RE = _re.compile(
        r"(?is)(?:<(?:script|style|noscript|iframe|template)\b[^>]*>.*?</(?:script|style|noscript|iframe|template)>|</?[A-Za-z][^>]{0,400}>|&(?:[A-Za-z]{2,32}|#\d{1,7}|#x[0-9A-Fa-f]{1,6});)"
    )
    _URL_RE = _re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>()]+")
    _BULLET_RE = _re.compile(
        r"^\s*(?:[-*•◦▪▸‣‒–—]|\d+[.)]|[A-Za-z][.)])\s+"
    )
    _CLOSERS = "\"'”’»)]}〕〉》」』】］）"
    _ASCII_TERMINALS = frozenset(".!?")
    _INDIC_TERMINALS = frozenset(".!?।॥")
    _SENTENCE_PUNCTUATION = frozenset(".!?,;:…।॥")
    _EMAIL_MASK_RE = _re.compile(
        r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    )
    _IP_MASK_RE = _re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    _PHONE_MASK_RE = _re.compile(
        r"(?<!\w)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\w)"
    )

    METRIC_NAMES = (
        "characters",
        "non_whitespace_units",
        "line_count",
        "mean_unit_characters",
        "text_bearing_fraction",
        "digit_fraction",
        "symbol_fraction",
        "residual_markup_count",
        "url_count",
        "bullet_line_fraction",
        "ellipsis_line_fraction",
        "ascii_terminal_line_fraction",
        "indic_terminal_line_fraction",
        "duplicate_line_character_fraction",
        "repeated_5gram_character_coverage",
        "max_repeated_character_run",
        "numeric_fraction",
    )
    REVIEW_CUTOFF_METRICS = (
        "duplicate_line_character_fraction",
        "repeated_5gram_character_coverage",
        "bullet_line_fraction",
        "ellipsis_line_fraction",
        "numeric_fraction",
    )
    REVIEW_REASON_LABELS = {
        "duplicate_line_character_fraction": "duplicate_line_outlier",
        "repeated_5gram_character_coverage": "repeated_5gram_outlier",
        "bullet_line_fraction": "bullet_density_outlier",
        "ellipsis_line_fraction": "ellipsis_density_outlier",
        "numeric_fraction": "numeric_density_outlier",
    }

    def normalize_nfc(text):
        return _unicodedata.normalize("NFC", text or "")

    def _comparison_text(text):
        return _re.sub(r"\s+", " ", normalize_nfc(text)).strip().casefold()

    def _nonempty_lines(text):
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _terminal_fraction(lines, endings):
        if not lines:
            return 0.0
        terminal_count = 0
        for line in lines:
            candidate = line.rstrip().rstrip(_CLOSERS).rstrip()
            if candidate and candidate[-1] in endings:
                terminal_count += 1
        return terminal_count / len(lines)

    def _line_fraction(lines, pattern):
        if not lines:
            return 0.0
        return sum(bool(pattern.search(line)) for line in lines) / len(lines)

    def _character_ratios(text, non_whitespace_characters):
        if not non_whitespace_characters:
            return 0.0, 0.0, 0.0
        text_bearing = 0
        digits = 0
        symbols = 0
        for character in non_whitespace_characters:
            category = _unicodedata.category(character)
            if category[0] in "LNM":
                text_bearing += 1
            if category == "Nd":
                digits += 1
            if (
                category[0] == "S"
                or (
                    category[0] == "P"
                    and character not in _SENTENCE_PUNCTUATION
                )
            ):
                symbols += 1
        denominator = len(non_whitespace_characters)
        return (
            text_bearing / denominator,
            digits / denominator,
            symbols / denominator,
        )

    def _duplicate_line_fraction(text):
        lines = [_comparison_text(line) for line in _nonempty_lines(text)]
        if not lines:
            return 0.0
        frequencies = _Counter(lines)
        total_characters = sum(len(line) for line in lines)
        duplicate_characters = sum(
            len(line) for line in lines if frequencies[line] > 1
        )
        return duplicate_characters / max(1, total_characters)

    def _repeated_5gram_fraction(text):
        units = _comparison_text(text).split()
        if len(units) < 5:
            return 0.0
        ngrams = _Counter(
            tuple(units[index : index + 5])
            for index in range(len(units) - 4)
        )
        marked = set()
        for index in range(len(units) - 4):
            if ngrams[tuple(units[index : index + 5])] > 1:
                marked.update(range(index, index + 5))
        total_characters = sum(len(unit) for unit in units)
        marked_characters = sum(len(units[index]) for index in marked)
        return marked_characters / max(1, total_characters)

    def _max_repeated_character_run(text):
        longest = 0
        current_character = None
        current_length = 0
        for character in text:
            if character.isspace():
                current_character = None
                current_length = 0
            elif character == current_character:
                current_length += 1
            else:
                current_character = character
                current_length = 1
            longest = max(longest, current_length)
        return longest

    def calculate_base_metrics(record):
        text = normalize_nfc(record.get("text", ""))
        lines = _nonempty_lines(text)
        units = _re.findall(r"\S+", text)
        non_whitespace_characters = [character for character in text if not character.isspace()]
        text_bearing_fraction, digit_fraction, symbol_fraction = _character_ratios(
            text, non_whitespace_characters
        )
        return {
            "doc_id": str(record.get("doc_id") or ""),
            "type": str(record.get("type") or "unknown"),
            "characters": len(text),
            "non_whitespace_units": len(units),
            "line_count": len(text.splitlines()) if text else 0,
            "mean_unit_characters": sum(len(unit) for unit in units) / max(1, len(units)),
            "text_bearing_fraction": text_bearing_fraction,
            "digit_fraction": digit_fraction,
            "symbol_fraction": symbol_fraction,
            "residual_markup_count": len(_HTML_RESIDUE_RE.findall(text)),
            "url_count": len(_URL_RE.findall(text)),
            "bullet_line_fraction": _line_fraction(lines, _BULLET_RE),
            "ellipsis_line_fraction": (
                sum("…" in line or "..." in line for line in lines) / max(1, len(lines))
            ),
            "ascii_terminal_line_fraction": _terminal_fraction(
                lines, _ASCII_TERMINALS
            ),
            "indic_terminal_line_fraction": _terminal_fraction(
                lines, _INDIC_TERMINALS
            ),
            "duplicate_line_character_fraction": _duplicate_line_fraction(text),
            "repeated_5gram_character_coverage": _repeated_5gram_fraction(text),
            "max_repeated_character_run": _max_repeated_character_run(text),
            "numeric_fraction": digit_fraction,
            "estimated_tokens": estimate_tokens(text),
        }

    def derive_review_cutoffs(metrics):
        cutoffs = {}
        for metric in REVIEW_CUTOFF_METRICS:
            values = sorted(float(item.get(metric, 0.0)) for item in metrics)
            if not values:
                cutoffs[metric] = 0.0
                continue
            position = 0.95 * (len(values) - 1)
            lower = int(position)
            upper = min(lower + 1, len(values) - 1)
            fraction = position - lower
            cutoffs[metric] = values[lower] + fraction * (values[upper] - values[lower])
        return cutoffs

    def _hard_flags(metrics):
        non_whitespace = max(1, metrics["non_whitespace_units"])
        non_text_fraction = 1.0 - metrics["text_bearing_fraction"]
        flags = []
        if metrics["characters"] < 120 or metrics["non_whitespace_units"] < 20:
            flags.append("too_short")
        if metrics["residual_markup_count"]:
            flags.append("residual_markup_or_script")
        if non_text_fraction > 0.72:
            flags.append("extreme_non_text")
        if metrics["max_repeated_character_run"] >= 10:
            flags.append("malformed_character_run")
        if (
            metrics["url_count"] >= 8
            and metrics["url_count"] / non_whitespace > 0.12
        ):
            flags.append("link_heavy_spam")
        return flags

    def evaluate_quality_records(records, cutoffs=None, control=False):
        measured = [calculate_base_metrics(record) for record in records]
        observed_cutoffs = cutoffs or derive_review_cutoffs(measured)
        results = []
        for item in measured:
            hard_flags = _hard_flags(item)
            review_reasons = []
            for metric, reason in REVIEW_REASON_LABELS.items():
                value = float(item[metric])
                threshold = float(observed_cutoffs.get(metric, 0.0))
                exceeds = value > threshold
                if control and value > 0 and value >= threshold:
                    exceeds = True
                if exceeds:
                    review_reasons.append(reason)
            action = "reject" if hard_flags else "review" if review_reasons else "keep"
            results.append(
                {
                    **item,
                    "hard_flags": hard_flags,
                    "review_reasons": review_reasons,
                    "recommended_action": action,
                }
            )
        return results, observed_cutoffs

    def masked_preview(text):
        value = _EMAIL_MASK_RE.sub("[MASKED_EMAIL]", text or "")
        value = _IP_MASK_RE.sub("[MASKED_IPV4]", value)
        value = _PHONE_MASK_RE.sub("[MASKED_PHONE]", value)
        return value[:300]

    def synthetic_control_records():
        valid_hindi = " ".join(
            [
                "यह एक मान्य हिंदी वाक्य है।",
                "भाषा और अर्थ स्पष्ट हैं।",
                "यह उदाहरण देवनागरी पाठ को सही तरह से मापता है।",
                "मात्राएँ और संयुक्त अक्षर सुरक्षित रहने चाहिए।",
                "Hinglish का स्वाभाविक मिश्रण भी अपने आप में शोर नहीं है।",
                "वाक्य के अंत में हिंदी दंड दिखाई देता है।",
                "पाठ की विविधता किसी गुणवत्ता-नियम का विकल्प नहीं है।",
                "यह नियंत्रण केवल terminal punctuation की तुलना दिखाता है।",
            ]
        )
        menu_text = "\n".join(
            [
                "- पहला विकल्प उपयोगी जानकारी देता है",
                "- दूसरा विकल्प सूची की संरचना दिखाता है",
                "- तीसरा विकल्प Hinglish शब्द रखता है",
                "- चौथा विकल्प पाठ को समाप्त करता है",
            ]
            * 2
        )
        repeated_line = "\n".join(
            ["यह पंक्ति बार बार दोहराई गई है।"] * 12
        )
        symbol_debris = ("@@@ ### $$$ %%% ^^^ &&& ||| !!! ??? " * 6).strip()
        short_text = "यह छोटा पाठ है।"
        return [
            {"doc_id": "control-valid-hindi-danda", "type": "control", "text": valid_hindi},
            {"doc_id": "control-menu-list", "type": "control", "text": menu_text},
            {"doc_id": "control-repeated-lines", "type": "control", "text": repeated_line},
            {"doc_id": "control-symbol-debris", "type": "control", "text": symbol_debris},
            {"doc_id": "control-short-text", "type": "control", "text": short_text},
        ]

    return (
        METRIC_NAMES,
        REVIEW_CUTOFF_METRICS,
        REVIEW_REASON_LABELS,
        calculate_base_metrics,
        derive_review_cutoffs,
        evaluate_quality_records,
        masked_preview,
        normalize_nfc,
        synthetic_control_records,
    )


@app.cell(hide_code=True)
def download_controls(mo):
    download_button = mo.ui.run_button(
        label="Download 200 deterministic Sangraha records"
    )
    run_button = mo.ui.run_button(label="Run quality-signal workbook")
    mo.vstack(
        [
            mo.md(
                "The first button streams only the pinned `verified/hin` split and "
                "writes a local 200-record JSONL cache. The second button never "
                "downloads data; it analyzes only that bounded cache."
            ),
            mo.hstack([download_button, run_button]),
        ]
    )
    return download_button, run_button


@app.cell(hide_code=True)
def download_records(
    DATASET_REVISION,
    MANIFEST_PATH,
    RAW_PATH,
    cache_status,
    download_button,
    load_bounded_records,
    materialize_cache,
    mo,
    record_stats,
):
    valid, manifest, validation_message = cache_status()
    download_error = None
    if download_button.value and not valid:
        try:
            records, manifest = materialize_cache()
            valid = True
            validation_message = (
                f"Downloaded and verified exactly {len(records):,} source records."
            )
        except Exception as error:
            records = []
            valid = False
            download_error = f"Download failed: {type(error).__name__}: {error}"
            validation_message = download_error
    elif valid:
        records, manifest, validation_message = load_bounded_records()
    else:
        records = []

    stats = record_stats(records) if records else None
    status = (
        "## Bounded source cache\n"
        f"{validation_message}\n\n"
        f"Raw path: `{RAW_PATH}` · manifest: `{MANIFEST_PATH}` · "
        f"revision: `{DATASET_REVISION}`"
    )
    if download_error:
        status += "\n\nNo substitute dataset or unpinned revision was used."
    mo.vstack([mo.md(status), mo.ui.table([stats] if stats else [])])
    return records, manifest, stats, valid, validation_message




@app.cell(hide_code=True)
def synthetic_control_panel(
    calculate_base_metrics,
    derive_review_cutoffs,
    evaluate_quality_records,
    mo,
    synthetic_control_records,
):
    controls = synthetic_control_records()
    control_results, control_cutoffs = evaluate_quality_records(
        controls, control=True
    )
    rows = [
        {
            "control": result["doc_id"],
            "characters": result["characters"],
            "ASCII terminal fraction": result["ascii_terminal_line_fraction"],
            "Indic terminal fraction": result["indic_terminal_line_fraction"],
            "duplicate-line fraction": result["duplicate_line_character_fraction"],
            "hard flags": ", ".join(result["hard_flags"]) or "none",
            "review signals": ", ".join(result["review_reasons"]) or "none",
            "action": result["recommended_action"],
        }
        for result in control_results
    ]
    cutoff_rows = [
        {"control calibration metric": metric, "cutoff": value}
        for metric, value in control_cutoffs.items()
    ]
    mo.vstack(
        [
            mo.md(
                "## Synthetic mechanism-only controls\n"
                "These five records are not Sangraha measurements. They validate "
                "the implementation: valid Hindi ending in `।` is invisible to "
                "the ASCII-only terminal view but visible to the Indic-safe view; "
                "repeated lines remain review material; symbol debris and short "
                "text exercise hard flags."
            ),
            mo.ui.table(rows),
            mo.md("Control-only 95th-percentile reference values:"),
            mo.ui.table(cutoff_rows),
        ]
    )


@app.cell(hide_code=True)
def quality_filter_family_teaching(
    METRIC_NAMES,
    calculate_base_metrics,
    mo,
    records,
):
    measured = [calculate_base_metrics(record) for record in records]

    def percentile(values, fraction):
        if not values:
            return 0.0
        ordered = sorted(values)
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        remainder = position - lower
        return ordered[lower] + remainder * (ordered[upper] - ordered[lower])

    distribution_rows = []
    for metric in METRIC_NAMES:
        values = [float(item[metric]) for item in measured]
        distribution_rows.append(
            {
                "metric": metric,
                "documents": len(values),
                "min": round(min(values), 4) if values else "not available",
                "median": round(percentile(values, 0.50), 4) if values else "not available",
                "p95": round(percentile(values, 0.95), 4) if values else "not available",
                "max": round(max(values), 4) if values else "not available",
            }
        )

    mo.vstack(
        [
            mo.md(
                r"""
## Deterministic quality-filter families

Each card uses the same contract: **what it measures · internal calculation ·
why/when useful · advantages · failure modes · Indic/Hinglish handling ·
execution status · primary sources**. Values in the distribution table are
computed from the bounded Sangraha cache when it is available; they are not
threshold recommendations. The final run report adds hard-rejection and
review-reason counts.

### 1. Length and information density

- **What it measures:** character count, whitespace-delimited units, and mean unit character length.
- **Internal calculation:** NFC-normalize an in-memory copy, count `len(text)` and `re.findall(r"\S+", text)`, then calculate mean non-whitespace-unit length. The only automatic length check is `characters < 120` or `units < 20`.
- **Why/when useful:** catches empty, truncated, or extremely small records before expensive scoring.
- **Advantages:** cheap, deterministic, and easy to audit.
- **Failure modes:** short verse, titles, captions, dialogue, code, and valid list entries can be useful; English-derived upper/lower bounds can erase Indic text.
- **Indic/Hinglish handling:** no minimum ASCII punctuation or Latin-script requirement; mixed Hindi/Latin units remain eligible.
- **Execution status:** **executed** as a conservative hard check; mean length is descriptive only. Gopher's English bounds are discussed, not copied.
- **Primary sources:** [Gopher](https://arxiv.org/abs/2112.11446), [C4](https://arxiv.org/abs/1910.10683).

### 2. Unicode category, symbol, digit, URL, and residue signals

- **What it measures:** text-bearing fraction, digit and symbol fractions, HTML/script residue, URLs, and repeated-character runs.
- **Internal calculation:** Unicode categories beginning with `L`, `N`, or `M` count as text-bearing; `Nd` counts as digits; symbols and non-sentence punctuation are measured separately. `।` and `॥` are sentence punctuation, never symbols. Markup is detected with bounded HTML/script/entity patterns.
- **Why/when useful:** identifies encoding/layout residue, symbol debris, link-heavy spam, and malformed runs without requiring a model.
- **Advantages:** transparent and preserves Devanagari combining marks, ZWJ, and ZWNJ.
- **Failure modes:** code, mathematical text, OCR, tables, emoji, and symbol-rich legitimate documents can look noisy; a regex can over-match markup-like prose.
- **Indic/Hinglish handling:** never uses raw `isalnum()` or an ASCII-only ratio; Latin-script Hindi is not rejected.
- **Execution status:** **executed**. Hard flags are residual markup, over 72% non-text-bearing characters, ten-character runs, and link-heavy spam (at least eight URLs and over 12% of units).
- **Primary sources:** [C4](https://arxiv.org/abs/1910.10683), [FineWeb](https://arxiv.org/abs/2406.17557), [IndicLLMSuite](https://arxiv.org/abs/2403.06350).

### 3. Line structure and terminal punctuation

- **What it measures:** bullet/list density, ellipsis density, and line-ending coverage under two views.
- **Internal calculation:** inspect non-empty lines; detect common bullet/number markers and `...`/`…`; strip trailing closing quotes/brackets before testing terminals. The English-origin view accepts `. ! ?`; the Indic-safe view accepts `. ! ? । ॥`.
- **Why/when useful:** menus, navigation fragments, transcripts, OCR blocks, and malformed line wrapping often have distinctive structure.
- **Advantages:** exposes the exact reason an ASCII heuristic disagrees with an Indic-safe one; the five-record controls are separate from corpus measurements.
- **Failure modes:** poems, legal lists, tables, subtitles, and publisher formatting can be intentionally line-structured.
- **Indic/Hinglish handling:** `।` and `॥` are valid sentence endings; mixed-script and list content is review material, never automatic rejection.
- **Execution status:** **executed** for ratios; bullet and ellipsis outliers become review-only at the observed 95th percentile. No terminal-punctuation ratio is a reject rule.
- **Primary sources:** [Gopher](https://arxiv.org/abs/2112.11446), [IndicLLMSuite](https://arxiv.org/abs/2403.06350).

### 4. Intra-document repetition

- **What it measures:** duplicate normalized-line character fraction and repeated five-unit character coverage.
- **Internal calculation:** compare NFC-normalized, case-folded, whitespace-collapsed non-empty lines; for five-unit windows, mark every unit belonging to an n-gram occurring more than once and divide marked unit characters by total unit characters.
- **Why/when useful:** repeated navigation, boilerplate, OCR loops, and templated passages waste training budget.
- **Advantages:** catches repetition that exact document equality cannot; the n-gram spans are inspectable.
- **Failure modes:** refrains, quotations, forms, tables, and deliberate repetition are valid; low repetition in a curated source is not evidence that the rule is unnecessary.
- **Indic/Hinglish handling:** comparison normalization does not strip ZWJ/ZWNJ or Indic marks; script and language are not quality proxies.
- **Execution status:** **executed**, but repetition candidates are review-only. This is intra-document analysis, not inter-document MinHash/LSH deduplication.
- **Primary sources:** [Gopher](https://arxiv.org/abs/2112.11446), [FineWeb](https://arxiv.org/abs/2406.17557), [IndicLLMSuite](https://arxiv.org/abs/2403.06350).

### 5. Source/domain and blocklist signals

- **What it measures:** in a production pipeline, domain reputation, URL blocklists, safety wordlists, NSFW classifiers, and source-specific rules.
- **Internal calculation:** this workbook reports domain reputation as **not available** because the confirmed `verified/hin` schema exposes no source URL; it does not infer a domain from document text.
- **Why/when useful:** a source-aware policy can route spam, unsafe material, or high-risk domains for review before costly scoring.
- **Advantages:** combines provenance with text evidence and can use site-specific knowledge.
- **Failure modes:** blocklists age, public domains contain valuable text, lexical lists are language-biased, and a URL inside prose is not proof of spam.
- **Indic/Hinglish handling:** no English profanity or stopword list is used as a Hindi quality criterion; safety belongs to its own stage, not this quality notebook.
- **Execution status:** **not executed** except for the transparent URL-count signal and the explicit unavailable-domain status. No fake source metadata is introduced.
- **Primary sources:** [IndicLLMSuite](https://arxiv.org/abs/2403.06350), [Sangraha dataset card](https://huggingface.co/datasets/ai4bharat/sangraha).
"""
            ),
            mo.md("### Observed bounded-corpus distributions"),
            mo.ui.table(distribution_rows),
        ]
    )


@app.cell(hide_code=True)
def character_trigram_proxy_setup(masked_preview, normalize_nfc):
    import hashlib as _hashlib
    import math as _math
    from collections import Counter as _Counter

    def _percentile(values, fraction):
        if not values:
            return None
        ordered = sorted(values)
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        remainder = position - lower
        return ordered[lower] + remainder * (ordered[upper] - ordered[lower])

    def run_character_trigram_proxy(records):
        normalized = {
            str(record.get("doc_id") or ""): normalize_nfc(record.get("text", ""))
            for record in records
        }
        held_out = []
        training = []
        for record in records:
            doc_id = str(record.get("doc_id") or "")
            bucket = int(_hashlib.sha256(doc_id.encode("utf-8")).hexdigest(), 16) % 5
            (held_out if bucket == 0 else training).append(record)

        counts = _Counter()
        total_ngrams = 0
        for record in training:
            text = normalized.get(str(record.get("doc_id") or ""), "")
            for index in range(max(0, len(text) - 2)):
                counts[text[index : index + 3]] += 1
                total_ngrams += 1
        smoothing = 0.1
        vocabulary = max(1, len(counts) + 1)
        denominator = total_ngrams + smoothing * vocabulary
        scores = {str(record.get("doc_id") or ""): None for record in records}
        for record in held_out:
            doc_id = str(record.get("doc_id") or "")
            text = normalized.get(doc_id, "")
            nll = 0.0
            observed = 0
            for index in range(max(0, len(text) - 2)):
                count = counts.get(text[index : index + 3], 0)
                probability = (count + smoothing) / denominator
                nll -= _math.log(probability)
                observed += 1
            mean_nll = nll / max(1, observed)
            scores[doc_id] = {
                "char_3gram_mean_nll": mean_nll,
                "char_3gram_perplexity_proxy": _math.exp(mean_nll),
            }

        scored = [
            value["char_3gram_perplexity_proxy"]
            for value in scores.values()
            if value is not None
        ]
        scored_rows = [
            {
                "doc_id": str(record.get("doc_id") or ""),
                **scores[str(record.get("doc_id") or "")],
                "preview": masked_preview(record.get("text", "")),
            }
            for record in held_out
        ]
        scored_rows.sort(key=lambda row: row["char_3gram_perplexity_proxy"])
        tail_examples = scored_rows[:3] + scored_rows[-3:]
        return scores, {
            "method": "Unicode character 3-gram count model",
            "label": "reference-distribution similarity, not truth or an automatic reject rule",
            "held_out_rule": "int(sha256(doc_id).hexdigest(), 16) % 5 == 0",
            "training_documents": len(training),
            "held_out_documents": len(held_out),
            "scored_documents": len(scored),
            "unscored_training_documents": len(training),
            "smoothing_alpha": smoothing,
            "observed_ngram_types": len(counts),
            "observed_training_ngrams": total_ngrams,
            "distribution": {
                "min": min(scored) if scored else None,
                "median": _percentile(scored, 0.50),
                "p95": _percentile(scored, 0.95),
                "max": max(scored) if scored else None,
            },
            "tail_examples": tail_examples,
        }

    return (run_character_trigram_proxy,)


@app.cell(hide_code=True)
def model_based_quality_cards(mo):
    mo.md(
        r"""
## Model-based quality techniques: one transparent proxy, four explicit boundaries

The general rule is layered: use cheap structural checks first, then apply a
learned score only after language-specific validation and error analysis. These
scores rank similarity to a reference distribution; they do not define truth.

### Held-out character 3-gram language-model proxy — executed

- **What it measures:** how surprising a held-out document is under the other documents' character distribution.
- **Internal calculation:** reserve `int(sha256(doc_id).hexdigest(), 16) % 5 == 0`; train Unicode character 3-gram counts on the remaining NFC text; use add-0.1 smoothing; report mean negative log likelihood and `exp(mean_nll)`.
- **Why/when useful:** a dependency-free teaching analogue of per-language n-gram perplexity filtering when a reference corpus is available.
- **Advantages:** deterministic, held-out, inspectable, and no pretrained model download.
- **Failure modes:** it rewards similarity to the reference batch, penalizes valid rare language or domains, and is not a coherence, safety, or factuality judge.
- **Indic/Hinglish handling:** Unicode code points are retained; no Latin-script or Hindi-stopword rule is applied.
- **Execution status:** **executed only when the run button is clicked**. The output label is **reference-distribution similarity, not truth or an automatic reject rule**. Production CCNet/Sangraha use per-language 5-gram KenLM-style reference LMs; `kenlm` is absent here, so this is a pedagogical proxy, not a substitute.
- **Primary sources:** [CCNet](https://arxiv.org/abs/1911.00359), [IndicLLMSuite](https://arxiv.org/abs/2403.06350).

### Reference-corpus classifier — explanatory only

- **What it measures:** learned probability that a document resembles genuinely trusted examples rather than genuinely noisy examples.
- **Internal calculation:** train fastText or logistic regression on a labelled reference corpus, then score held-out documents; calibrate by language, source, and document type.
- **Why/when useful:** combines many weak signals when labels represent the target quality policy.
- **Advantages:** cheap inference after training and adaptable to source-specific noise.
- **Failure modes:** inherited label/source bias, shortcut learning, domain shift, and probability miscalibration.
- **Indic/Hinglish handling:** labels must include valid code-mixed Hindi, Indic scripts, and relevant genres; do not use `verified/hin` versus synthetic corruption because that would score the artificial corruption.
- **Execution status:** **not executed**; this notebook has no genuinely labelled Hindi trusted/noisy reference set.
- **Primary sources:** [C4](https://arxiv.org/abs/1910.10683), [FineWeb](https://arxiv.org/abs/2406.17557).

### Multilingual neural quality scorer or LLM judge — explanatory only

- **What it measures:** a classifier or judge's learned assessment of coherence, educational value, fluency, relevance, or other task-specific quality dimensions.
- **Internal calculation:** run a validated multilingual encoder or LLM, optionally distill its reviewed scores into a cheaper model, then audit calibration and disagreements.
- **Why/when useful:** captures semantic/document-level properties that counts and n-grams miss.
- **Advantages:** can integrate context, multilingual evidence, and a richer rubric.
- **Failure modes:** expensive inference, prompt/model bias, hallucinated rationales, language imbalance, and false confidence on rare Indic forms.
- **Indic/Hinglish handling:** require Hindi/Indic validation sets and human adjudication; fluency in English is not a Hindi quality target.
- **Execution status:** **not executed**; no pretrained encoder or LLM is downloaded in this bounded workbook.
- **Primary sources:** [FineWeb](https://arxiv.org/abs/2406.17557), [IndicLLMSuite](https://arxiv.org/abs/2403.06350).

### Line-level filtering — explanatory only

- **What it measures:** quality at line or block granularity so a mixed-quality document can retain useful passages.
- **Internal calculation:** score lines with transparent or learned features, remove or route only poor lines, and retain an audit trail back to the document.
- **Why/when useful:** improves precision when boilerplate or OCR damage is localized.
- **Advantages:** avoids discarding a whole valuable document for a small bad region.
- **Failure modes:** higher annotation/inference cost, loss of context, and line boundaries that do not match semantic units.
- **Indic/Hinglish handling:** calibrate by genre and script; verse, dialogue, menus, and short Hindi lines need human examples.
- **Execution status:** **not executed**. The document-level metrics above do not claim to implement line-level filtering.
- **Primary sources:** [C4](https://arxiv.org/abs/1910.10683), [IndicLLMSuite](https://arxiv.org/abs/2403.06350).

### Human review, stratified audit, and threshold calibration — required next step

- **What it measures:** reviewed accept/review/reject decisions across source type, script mix, length, and score tails.
- **Internal calculation:** sample strata, mask PII in the review view, adjudicate disagreements, estimate false deletion/retention rates, and recalibrate thresholds.
- **Why/when useful:** converts a descriptive signal into a defensible policy without pretending that a percentile is ground truth.
- **Advantages:** catches language and genre harms that automated heuristics miss.
- **Failure modes:** reviewer fatigue, small-stratum uncertainty, inconsistent rubrics, and leakage from seeing the candidate score.
- **Indic/Hinglish handling:** reviewers must include Hindi/Indic expertise and explicitly protect valid code-mixing, ZWJ/ZWNJ, verse, and OCR forms.
- **Execution status:** **not executed here**. It is the required next step before any observed 95th-percentile review flag becomes a production threshold.
- **Primary sources:** [FineWeb](https://arxiv.org/abs/2406.17557), [IndicLLMSuite](https://arxiv.org/abs/2403.06350), [Gopher](https://arxiv.org/abs/2112.11446).
"""
    )


@app.cell(hide_code=True)
def run_quality_workbook(
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    DATASET_SPLIT,
    MANIFEST_PATH,
    PIPELINE_VERSION,
    RESULTS_PATH,
    SOURCE_URL,
    SUMMARY_PATH,
    METRIC_NAMES,
    evaluate_quality_records,
    load_bounded_records,
    masked_preview,
    mo,
    run_button,
    run_character_trigram_proxy,
):
    import hashlib as _hashlib
    import json as _json
    import os as _os
    from collections import Counter as _Counter
    from datetime import datetime as _datetime, timezone as _timezone

    def _atomic_write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            _os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _percentile(values, fraction):
        if not values:
            return None
        ordered = sorted(values)
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        remainder = position - lower
        return ordered[lower] + remainder * (ordered[upper] - ordered[lower])

    def _stats(rows):
        return {
            "documents": len(rows),
            "characters": sum(int(row.get("characters", 0)) for row in rows),
            "estimated_tokens": sum(
                int(row.get("estimated_tokens", 0)) for row in rows
            ),
        }

    def _metric_summaries(results):
        summaries = []
        for metric in METRIC_NAMES:
            values = [float(row[metric]) for row in results]
            summaries.append(
                {
                    "metric": metric,
                    "min": min(values) if values else None,
                    "median": _percentile(values, 0.50),
                    "p95": _percentile(values, 0.95),
                    "max": max(values) if values else None,
                }
            )
        return summaries

    def _load_existing():
        if not RESULTS_PATH.exists() or not SUMMARY_PATH.exists():
            return [], None
        with RESULTS_PATH.open("r", encoding="utf-8") as handle:
            results = [_json.loads(line) for line in handle if line.strip()]
        summary = _json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        return results, summary

    quality_results = []
    quality_summary = None
    if not run_button.value:
        quality_results, quality_summary = _load_existing()
        quality_status = (
            "Existing quality-signal artifacts loaded."
            if quality_summary
            else "Click `Run quality-signal workbook` after the bounded source cache is available."
        )
    else:
        _records, _manifest, _cache_message = load_bounded_records()
        if len(_records) != 200 or _manifest is None:
            quality_status = (
                "Run skipped: the matching 200-record cache is unavailable. "
                f"{_cache_message} Click the download button first."
            )
        else:
            base_results, review_cutoffs = evaluate_quality_records(_records)
            proxy_scores, proxy_summary = run_character_trigram_proxy(_records)
            source_by_id = {
                str(record.get("doc_id") or ""): record for record in _records
            }
            quality_results = []
            for result in base_results:
                doc_id = result["doc_id"]
                score = proxy_scores.get(doc_id)
                source_record = source_by_id.get(doc_id, {})
                quality_results.append(
                    {
                        **result,
                        "char_3gram_mean_nll": (
                            score["char_3gram_mean_nll"] if score else None
                        ),
                        "char_3gram_perplexity_proxy": (
                            score["char_3gram_perplexity_proxy"] if score else None
                        ),
                        "masked_preview": masked_preview(
                            source_record.get("text", "")
                        ),
                    }
                )

            after_results = [
                row for row in quality_results if row["recommended_action"] != "reject"
            ]
            hard_counts = _Counter(
                flag
                for row in quality_results
                for flag in row["hard_flags"]
            )
            review_counts = _Counter(
                reason
                for row in quality_results
                if row["recommended_action"] == "review"
                for reason in row["review_reasons"]
            )
            all_review_counts = _Counter(
                reason
                for row in quality_results
                for reason in row["review_reasons"]
            )
            source_type_mix = []
            types = sorted({row["type"] for row in quality_results})
            for source_type in types:
                source_type_mix.append(
                    {
                        "type": source_type,
                        "before_documents": sum(
                            row["type"] == source_type for row in quality_results
                        ),
                        "after_hard_filter_documents": sum(
                            row["type"] == source_type for row in after_results
                        ),
                    }
                )
            result_lines = "".join(
                _json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in quality_results
            )
            _atomic_write(RESULTS_PATH, result_lines)
            results_sha256 = _hashlib.sha256(
                RESULTS_PATH.read_bytes()
            ).hexdigest()
            quality_summary = {
                "pipeline_version": PIPELINE_VERSION,
                "run_timestamp_utc": _datetime.now(_timezone.utc).isoformat(),
                "source": {
                    "source_url": SOURCE_URL,
                    "dataset_id": DATASET_ID,
                    "config": DATASET_CONFIG,
                    "split": DATASET_SPLIT,
                    "revision": DATASET_REVISION,
                    "license": DATASET_LICENSE,
                    "selection_manifest": str(MANIFEST_PATH),
                    "raw_sha256": _manifest.get("raw_sha256"),
                },
                "selection": {
                    "rule": _manifest.get("selection_rule"),
                    "documents": len(_records),
                    "estimated_token_semantics": "ceil(non-whitespace units * 1.35), teaching proxy only",
                },
                "before": _stats(
                    [
                        {
                            "characters": len(record.get("text", "")),
                            "estimated_tokens": int(result["estimated_tokens"]),
                        }
                        for record, result in zip(_records, quality_results)
                    ]
                ),
                "after_hard_filter": _stats(after_results),
                "counts": {
                    "input_documents": len(quality_results),
                    "hard_reject_documents": sum(
                        row["recommended_action"] == "reject" for row in quality_results
                    ),
                    "review_documents": sum(
                        row["recommended_action"] == "review" for row in quality_results
                    ),
                    "keep_documents": sum(
                        row["recommended_action"] == "keep" for row in quality_results
                    ),
                    "after_hard_filter_documents": len(after_results),
                    "candidate_documents": sum(
                        row["recommended_action"] != "keep" for row in quality_results
                    ),
                },
                "source_type_mix": source_type_mix,
                "hard_reject_reason_counts": dict(sorted(hard_counts.items())),
                "review_reason_counts": dict(sorted(review_counts.items())),
                "all_review_signal_counts": dict(sorted(all_review_counts.items())),
                "review_cutoffs_95th_percentile": review_cutoffs,
                "metric_summaries": _metric_summaries(quality_results),
                "character_3gram_proxy": proxy_summary,
                "domain_reputation": "not available: verified/hin exposes no source URL field",
                "pii_display_note": "masked_preview is teaching-view masking, not the canonical PII scrubber",
                "results_path": str(RESULTS_PATH),
                "results_sha256": results_sha256,
            }
            _atomic_write(
                SUMMARY_PATH,
                _json.dumps(quality_summary, ensure_ascii=False, indent=2) + "\n",
            )
            quality_status = (
                f"Completed quality signals for exactly {len(quality_results):,} cached source documents; "
                f"{quality_summary['counts']['after_hard_filter_documents']:,} remain after hard rejects."
            )

    mo.md(quality_status)
    return quality_results, quality_status, quality_summary


@app.cell
def quality_results_report(
    MANIFEST_PATH,
    RESULTS_PATH,
    SUMMARY_PATH,
    mo,
    quality_results,
    quality_status,
    quality_summary,
):
    if quality_summary is None:
        quality_report_view = mo.md(
            "## Quality-signal results\n"
            f"{quality_status}\n\n"
            f"Artifacts will be written to `{RESULTS_PATH}` and `{SUMMARY_PATH}`."
        )
    else:
        summary = quality_summary
        stage_rows = [
            {"stage view": "before quality hard filter", **summary["before"]},
            {
                "stage view": "after hard filter (keep + review)",
                **summary["after_hard_filter"],
            },
        ]
        count_rows = [
            {"count": key, "documents": value}
            for key, value in summary["counts"].items()
        ]
        hard_rows = [
            {"hard reject reason": key, "documents": value}
            for key, value in summary["hard_reject_reason_counts"].items()
        ] or [{"hard reject reason": "none observed", "documents": 0}]
        review_rows = [
            {"review reason": key, "documents": value}
            for key, value in summary["review_reason_counts"].items()
        ] or [{"review reason": "none observed", "documents": 0}]
        manifest_rows = [
            {"provenance field": key, "value": str(value)}
            for key, value in summary["source"].items()
        ]
        candidate_rows = []
        for row in quality_results:
            if row["recommended_action"] == "keep":
                continue
            candidate_rows.append(
                {
                    "doc_id": row["doc_id"],
                    "type": row["type"],
                    "action": row["recommended_action"],
                    "reason_codes": ", ".join(
                        row["hard_flags"] + row["review_reasons"]
                    ),
                    "characters": row["characters"],
                    "preview_masked_capped_300": row["masked_preview"],
                }
            )
        proxy = summary["character_3gram_proxy"]
        proxy_rows = [
            {"proxy field": key, "value": value}
            for key, value in proxy.items()
            if key not in {"tail_examples"}
        ]
        quality_report_view = mo.vstack(
            [
                mo.md(
                    "## Quality-signal results\n"
                    f"{quality_status}\n\n"
                    "The after-hard-filter corpus includes both `keep` and `review`; "
                    "only `reject` rows lower its counts. Candidate previews are capped "
                    "at 300 characters and mask email, IPv4, and India-shaped phone "
                    "patterns for this teaching view. This is **not the canonical PII scrubber**."
                ),
                mo.md("### Before versus after counts"),
                mo.ui.table(stage_rows),
                mo.md("### Action and reason counts"),
                mo.ui.table(count_rows),
                mo.md("Hard flags are the only automatic rejection evidence."),
                mo.ui.table(hard_rows),
                mo.md("Review flags are calibrated outliers, not automatic exclusions."),
                mo.ui.table(review_rows),
                mo.md("### Source/type mix"),
                mo.ui.table(summary["source_type_mix"]),
                mo.md("### Metric distributions (all 200 source records)"),
                mo.ui.table(summary["metric_summaries"]),
                mo.md(
                    "### Review cutoffs\n"
                    "These are observed 95th-percentile descriptive cutoffs for this "
                    "batch, not production thresholds."
                ),
                mo.ui.table(
                    [
                        {"metric": key, "cutoff": value}
                        for key, value in summary["review_cutoffs_95th_percentile"].items()
                    ]
                ),
                mo.md(
                    "### Held-out character 3-gram proxy\n"
                    f"**{proxy['label']}**"
                ),
                mo.ui.table(proxy_rows),
                mo.md("Masked tail examples; these are not raw document values:"),
                mo.ui.table(proxy["tail_examples"]),
                mo.md("### Candidate records"),
                mo.ui.table(candidate_rows),
                mo.md("### Manifest / provenance fields"),
                mo.ui.table(manifest_rows),
                mo.md(
                    f"Artifacts: `{RESULTS_PATH}` · `{SUMMARY_PATH}` · source manifest `{MANIFEST_PATH}`. "
                    f"Results SHA-256: `{summary['results_sha256']}`. "
                    "Language validation, deduplication, canonical PII scrubbing, and decontamination remain separate stages."
                ),
            ]
        )
    quality_report_view


if __name__ == "__main__":
    app.run()
