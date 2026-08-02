import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def assignment_and_dataset_choice(mo):
    mo.md(r"""
    # Session 4 Assignment · Data Cleaning & Deduplication

    ## Assignment (verbatim)

    > Work with your agent, and find in total "how many strategies" are listed in this session. Then find a 10-100M dataset like [this](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled), and then apply these cleanups. Find interested datasets, specially from the Session 3 that you may have come across. Finally create a widget that talks about:
    > - how many strategies, what are they
    > - what dataset was picked
    > - what was cleaned and why
    > - any other stragtegy or concern was cleaned up?
    > - final statistics
    >
    > Upload to Netlify and then share the link.

    ## Dataset decision

    | Candidate | Decision |
    |---|---|
    | Hindi Wikipedia | In-band, but too clean for a cleaning-pipeline project. |
    | OSCAR Hindi | Good raw-web alternative, but manually gated and operationally heavier. |
    | IndicCorp Hindi | Mostly short, already-cleaned lines; weak dedup/PII/format signal. |
    | xP3 / IndicParaphrase | Instruction/paraphrase pairs, not a document-pretraining corpus. |
    | **Sangraha verified Hindi** | **Chosen.** Session 3 corpus; open, document-level Hindi web text, provenance fields, and enough residual noise for the pipeline. |

    **Locked dataset:** `ai4bharat/sangraha` · `verified` / `hin` · stream and stop at about **30M tokens**.
    """)
    return


@app.cell
def _(mo):
    fetch_preview = mo.ui.run_button(label="Fetch 5 compact Sangraha Hindi previews")
    mo.vstack(
        [
            mo.md(
                "Fetches five hosted rows only. It never loads the "
                "Hugging Face dataset library or renders full documents."
            ),
            fetch_preview,
        ]
    )
    return (fetch_preview,)


@app.cell
def _(fetch_preview):
    import json
    from urllib.parse import urlencode
    from urllib.request import urlopen

    if not fetch_preview.value:
        preview_rows = []
        preview_status = "Click the button to fetch five compact previews."
    else:
        query = urlencode(
            {
                "dataset": "ai4bharat/sangraha",
                "config": "verified",
                "split": "hin",
                "offset": 0,
                "length": 50,
            }
        )
        with urlopen(
            f"https://datasets-server.huggingface.co/rows?{query}",
            timeout=30,
        ) as response:
            payload = json.load(response)

        preview_rows = []
        for index, item in enumerate(payload["rows"]):
            row = item["row"]
            text = row.get("text", "")
            preview_rows.append(
                {
                    "index": index,
                    "doc_id": row.get("doc_id"),
                    "type": row.get("type"),
                    "char_count": len(text),
                    "preview": text[:300],
                }
            )
        preview_status = f"Loaded {len(preview_rows)} compact previews."
    return json, payload, preview_rows, preview_status


@app.cell
def _(mo, preview_rows, preview_status):
    mo.vstack(
        [
            mo.md(preview_status),
            mo.ui.table(preview_rows),
        ]
    )
    return


@app.cell
def _(json, payload):
    with open("./sample-sangraha.json", "w") as f:
        f.write(json.dumps(payload, indent=3))
    return


@app.cell
def _(payload):
    del payload
    return


@app.cell
def _(preview_rows):
    del preview_rows
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def pipeline_logic():
    import hashlib as _hashlib
    import html as _html
    import json as _json
    import math as _math
    import re as _re
    import unicodedata as _unicodedata
    from collections import Counter as _Counter
    from datetime import datetime as _datetime, timezone as _timezone
    from pathlib import Path as _Path

    from datasets import load_dataset as _load_dataset

    PIPELINE_VERSION = "s4-hindi-cleaning-2026-08-02-v1"
    DATASET_ID = "ai4bharat/sangraha"
    DATASET_CONFIG = "verified"
    DATASET_SPLIT = "hin"
    DATASET_REVISION = "8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8"
    DATASET_LICENSE = "CC BY 4.0"
    SOURCE_URL = "https://huggingface.co/datasets/ai4bharat/sangraha"
    TARGET_TOKEN_ESTIMATE = 30_000_000
    TOKEN_FERTILITY_ASSUMPTION = 1.35
    RAW_CACHE_PATH = _Path("data/sangraha_hi_verified_30m/raw.jsonl")
    MANIFEST_PATH = _Path("data/sangraha_hi_verified_30m/manifest.json")

    _BOILERPLATE_MARKERS = (
        "accept cookies",
        "accept all cookies",
        "cookie settings",
        "cookie policy",
        "privacy policy",
        "subscribe to our newsletter",
        "sign up for our newsletter",
        "skip to content",
        "enable javascript",
        "all rights reserved",
        "advertisement",
        "advertise with us",
    )
    _SCRIPT_BLOCKS = _re.compile(
        r"(?is)<(?:script|style|noscript|template|iframe)[^>]*>.*?</(?:script|style|noscript|template|iframe)>"
    )
    _BLOCK_TAGS = _re.compile(
        r"(?is)</?(?:article|aside|br|div|footer|h[1-6]|header|li|main|nav|p|section|table|tr|ul)[^>]*>"
    )
    _ANY_TAG = _re.compile(r"(?is)<[^>]{1,400}>")
    _MOJIBAKE = {
        "â€“": "–",
        "â€”": "—",
        "â€˜": "‘",
        "â€™": "’",
        "â€œ": "“",
        "â€�": "”",
        "â€¦": "…",
        "Â ": " ",
        "Ã©": "é",
    }
    _CHAT_MARKER = _re.compile(
        r"(?im)^\s*(?:\[?(?:system|user|assistant|human|bot)\]?|<\|(?:system|user|assistant|human|bot)\|>)\s*:?[ \t]*"
    )
    _DEVANAGARI = _re.compile(r"[\u0900-\u097f]")
    _LATIN = _re.compile(r"[A-Za-z]")
    _HINDI_DEVANAGARI_CUES = _re.compile(
        r"(?:है|हैं|था|थे|थी|और|का|की|के|को|में|से|यह|वह|एक|नहीं|क्या|कर|लिए|पर|तो|भी|जब|तक)"
    )
    _HINGLISH_CUES = _re.compile(
        r"\b(?:hai|hain|tha|the|thi|aur|ka|ki|ke|ko|mein|se|yeh|woh|ek|nahi|kya|kar|liye|par|to|bhi|jab|tak)\b",
        _re.IGNORECASE,
    )
    _PII_PATTERNS = (
        (
            "credential",
            _re.compile(
                r"(?i)\b(?:api[_ -]?key|access[_ -]?token|password|passwd|secret)\s*[:=]\s*[^\s,;]+"
            ),
            "[REDACTED_CREDENTIAL]",
        ),
        (
            "email",
            _re.compile(
                r"(?i)(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])"
            ),
            "[REDACTED_EMAIL]",
        ),
        (
            "phone",
            _re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)"),
            "[REDACTED_PHONE]",
        ),
        (
            "ipv4",
            _re.compile(
                r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"
            ),
            "[REDACTED_IP]",
        ),
        (
            "aadhaar_like",
            _re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)"),
            "[REDACTED_ID]",
        ),
    )
    _DECONTAMINATION_PATTERNS = (
        "gsm8k",
        "mmlu",
        "hellaswag",
        "winogrande",
        "truthfulqa",
        "bigbench",
        "arc challenge",
        "benchmark answer",
        "eval harness",
    )
    _MINHASH_PERMUTATIONS = tuple(
        (
            (0x9E3779B185EBCA87 * (index + 1)) | 1,
            0xC2B2AE3D27D4EB4F ^ (index * 0x165667B19E3779F9),
        )
        for index in range(16)
    )
    _UINT64_MASK = (1 << 64) - 1

    def _estimate_tokens(text):
        units = len(_re.findall(r"\S+", text or ""))
        return (
            int(_math.ceil(units * TOKEN_FERTILITY_ASSUMPTION)) if units else 0
        )

    def _sha256_file(path):
        digest = _hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _sha256_records(records):
        digest = _hashlib.sha256()
        for record in records:
            digest.update(str(record.get("doc_id", "")).encode("utf-8"))
            digest.update(b"\n")
            digest.update(record.get("text", "").encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def _stats(records):
        return {
            "documents": len(records),
            "characters": sum(
                len(record.get("text", "")) for record in records
            ),
            "estimated_tokens": sum(
                _estimate_tokens(record.get("text", "")) for record in records
            ),
        }

    def _language_mix(records):
        counts = _Counter(
            record.get("detected_language", "not_checked")
            for record in records
        )
        return dict(
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )

    def _extract_content(text):
        text = text or ""
        text = _SCRIPT_BLOCKS.sub("\n", text)
        text = _BLOCK_TAGS.sub("\n", text)
        text = _ANY_TAG.sub(" ", text)
        kept_lines = []
        for line in text.splitlines():
            compact = _re.sub(r"\s+", " ", line).strip()
            lowered = compact.casefold()
            if not compact:
                continue
            if any(marker in lowered for marker in _BOILERPLATE_MARKERS):
                continue
            kept_lines.append(compact)
        return "\n".join(kept_lines)

    def _normalize_content(text):
        text = _html.unescape(text or "")
        for source, replacement in _MOJIBAKE.items():
            text = text.replace(source, replacement)
        text = _unicodedata.normalize("NFC", text)
        # Preserve Devanagari ZWJ (U+200D) and ZWNJ (U+200C); remove only
        # formatting invisibles that are not meaningful to Indic orthography.
        text = (
            text.replace("\u200b", "")
            .replace("\ufeff", "")
            .replace("\u2060", "")
        )
        text = _re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = _CHAT_MARKER.sub("", text)
        text = _re.sub(r"\r\n?", "\n", text)
        text = _re.sub(r"[ \t]+", " ", text)
        text = _re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _detect_language(text):
        devanagari = len(_DEVANAGARI.findall(text))
        latin = len(_LATIN.findall(text))
        letters = devanagari + latin
        if letters < 20:
            return "unknown"
        deva_ratio = devanagari / letters
        deva_cues = len(_HINDI_DEVANAGARI_CUES.findall(text))
        latin_cues = len(_HINGLISH_CUES.findall(text))
        if devanagari >= 30 and deva_ratio >= 0.25:
            return (
                "hi-Deva+Latin" if latin and deva_ratio < 0.75 else "hi-Deva"
            )
        if latin >= 40 and latin_cues >= 2:
            return "hi-Latin"
        if deva_cues >= 2:
            return "hi-Deva"
        return "other"

    def _quality_reason(text):
        units = _re.findall(r"\S+", text)
        if len(text) < 120 or len(units) < 20:
            return "too_short"
        if _re.search(r"(.)\1{9,}", text):
            return "malformed_repetition"
        alpha_numeric = sum(character.isalnum() for character in text)
        if alpha_numeric / max(len(text), 1) < 0.28:
            return "malformed_symbols"
        link_like = len(
            _re.findall(r"(?:https?://|www\.|\.com\b)", text, flags=_re.I)
        )
        if link_like >= 8 and link_like / max(len(units), 1) > 0.12:
            return "link_heavy_spam"
        unique_ratio = len(set(unit.casefold() for unit in units)) / len(units)
        if len(units) >= 60 and unique_ratio < 0.22:
            return "low_diversity"
        shingles = [
            " ".join(units[index : index + 5])
            for index in range(max(0, len(units) - 4))
        ]
        if shingles and len(set(shingles)) / len(shingles) < 0.35:
            return "repetitive_text"
        return ""

    def _dedup_tokens(text):
        return [
            token.casefold()
            for token in _re.findall(r"\w+", text, flags=_re.UNICODE)
        ]

    def _minhash_signature(text):
        tokens = _dedup_tokens(text)
        if len(tokens) < 10:
            return None
        shingle_hashes = set()
        for index in range(len(tokens) - 4):
            shingle = "\x1f".join(tokens[index : index + 5]).encode("utf-8")
            raw = _hashlib.blake2b(shingle, digest_size=8).digest()
            shingle_hashes.add(int.from_bytes(raw, "little"))
        return tuple(
            min(((a * value + b) & _UINT64_MASK) for value in shingle_hashes)
            for a, b in _MINHASH_PERMUTATIONS
        )

    def _deduplicate(records):
        kept = []
        kept_signatures = []
        exact_hashes = set()
        lsh_buckets = {}
        exact_removed = 0
        near_removed = 0
        near_pairs = []
        for record in records:
            normalized_hash = _hashlib.sha256(
                record.get("text", "").encode("utf-8")
            ).hexdigest()
            if normalized_hash in exact_hashes:
                exact_removed += 1
                continue
            signature = _minhash_signature(record.get("text", ""))
            duplicate_of = None
            if signature is not None:
                candidates = set()
                for band_start in range(0, len(signature), 4):
                    bucket_key = (
                        band_start // 4,
                        signature[band_start : band_start + 4],
                    )
                    candidates.update(lsh_buckets.get(bucket_key, ()))
                for candidate_index in sorted(candidates):
                    candidate = kept[candidate_index]
                    candidate_signature = kept_signatures[candidate_index]
                    length_ratio = len(record["text"]) / max(
                        len(candidate["text"]), 1
                    )
                    similarity = sum(
                        left == right
                        for left, right in zip(signature, candidate_signature)
                    ) / len(signature)
                    if 0.80 <= length_ratio <= 1.25 and similarity >= 0.875:
                        duplicate_of = candidate["doc_id"]
                        break
            if duplicate_of is not None:
                near_removed += 1
                if len(near_pairs) < 8:
                    near_pairs.append(
                        {
                            "removed_doc_id": record["doc_id"],
                            "kept_doc_id": duplicate_of,
                        }
                    )
                continue
            kept_index = len(kept)
            kept.append(record)
            kept_signatures.append(signature)
            exact_hashes.add(normalized_hash)
            if signature is not None:
                for band_start in range(0, len(signature), 4):
                    bucket_key = (
                        band_start // 4,
                        signature[band_start : band_start + 4],
                    )
                    lsh_buckets.setdefault(bucket_key, []).append(kept_index)
        return kept, {
            "exact_removed": exact_removed,
            "near_removed": near_removed,
            "near_duplicate_pairs_sample": near_pairs,
            "method": "SHA-256 exact hash plus 16-permutation MinHash with four-row LSH bands; local approximation",
        }

    def _scrub_pii(text):
        counts = {}
        changed = text
        for category, pattern, replacement in _PII_PATTERNS:
            changed, count = pattern.subn(replacement, changed)
            if count:
                counts[category] = count
        return changed, counts

    def _decontamination_matches(text):
        lowered = text.casefold()
        return [term for term in _DECONTAMINATION_PATTERNS if term in lowered]

    def _load_sample(target_tokens):
        RAW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        records = []
        seen_ids = set()
        token_total = 0
        if RAW_CACHE_PATH.exists():
            with RAW_CACHE_PATH.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    if row.get("doc_id") in seen_ids or not row.get("text"):
                        continue
                    records.append(row)
                    seen_ids.add(row.get("doc_id"))
                    token_total += int(
                        row.get(
                            "estimated_tokens", _estimate_tokens(row["text"])
                        )
                    )
                    if token_total >= target_tokens:
                        return records, "cache"
        dataset = _load_dataset(
            DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT, streaming=True
        )
        mode = "a" if RAW_CACHE_PATH.exists() else "w"
        with RAW_CACHE_PATH.open(mode, encoding="utf-8") as handle:
            for source_row in dataset:
                doc_id = str(source_row.get("doc_id") or "")
                text = source_row.get("text") or ""
                if not doc_id or not text or doc_id in seen_ids:
                    continue
                estimated_tokens = _estimate_tokens(text)
                row = {
                    "doc_id": doc_id,
                    "type": source_row.get("type", "unknown"),
                    "text": text,
                    "estimated_tokens": estimated_tokens,
                }
                handle.write(_json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(row)
                seen_ids.add(doc_id)
                token_total += estimated_tokens
                if token_total >= target_tokens:
                    break
        return records, "stream"

    def run_cleaning_pipeline(target_tokens=TARGET_TOKEN_ESTIMATE):
        raw_records, selection_source = _load_sample(target_tokens)
        if not raw_records:
            raise RuntimeError(
                "The Sangraha stream returned no usable documents"
            )
        stage_stats = [{"stage": "raw sample", **_stats(raw_records)}]

        extracted = [
            {**record, "text": _extract_content(record["text"])}
            for record in raw_records
        ]
        stage_stats.append({"stage": "extract", **_stats(extracted)})
        del raw_records

        normalized = [
            {**record, "text": _normalize_content(record["text"])}
            for record in extracted
        ]
        stage_stats.append({"stage": "normalize", **_stats(normalized)})
        del extracted

        labeled = [
            {**record, "detected_language": _detect_language(record["text"])}
            for record in normalized
        ]
        del normalized

        language_before = _language_mix(labeled)
        validated = []
        language_removed = 0
        for record in labeled:
            if record["detected_language"] in {
                "hi-Deva",
                "hi-Deva+Latin",
                "hi-Latin",
            }:
                validated.append(record)
            else:
                language_removed += 1
        stage_stats.append(
            {"stage": "language validation", **_stats(validated)}
        )
        del labeled

        quality_kept = []
        quality_reasons = _Counter()
        for record in validated:
            reason = _quality_reason(record["text"])
            if reason:
                quality_reasons[reason] += 1
            else:
                quality_kept.append(record)
        stage_stats.append(
            {"stage": "quality filtering", **_stats(quality_kept)}
        )
        del validated

        deduplicated, duplicate_counts = _deduplicate(quality_kept)
        stage_stats.append({"stage": "deduplication", **_stats(deduplicated)})
        del quality_kept
        pii_counts = _Counter()
        pii_documents = 0
        pii_scrubbed = []
        for record in deduplicated:
            scrubbed_text, counts = _scrub_pii(record["text"])
            if counts:
                pii_documents += 1
                pii_counts.update(counts)
            pii_scrubbed.append({**record, "text": scrubbed_text})
        stage_stats.append({"stage": "PII scrubbing", **_stats(pii_scrubbed)})
        del deduplicated

        decontaminated = []
        decontamination_counts = _Counter()
        decontamination_documents = 0
        for record in pii_scrubbed:
            matches = _decontamination_matches(record["text"])
            if matches:
                decontamination_documents += 1
                decontamination_counts.update(matches)
            else:
                decontaminated.append(record)
        stage_stats.append(
            {"stage": "decontamination", **_stats(decontaminated)}
        )
        stage_stats.append(
            {"stage": "final manifest", **_stats(decontaminated)}
        )
        del pii_scrubbed

        language_after = _language_mix(decontaminated)
        final_hash = _sha256_records(decontaminated)
        manifest = {
            "source_url": SOURCE_URL,
            "dataset_id": DATASET_ID,
            "license": DATASET_LICENSE,
            "revision": DATASET_REVISION,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "sample_selection": "deterministic first rows from streaming split until estimated-token target; cached as JSONL",
            "target_estimated_tokens": target_tokens,
            "raw_cache_path": str(RAW_CACHE_PATH),
            "raw_cache_sha256": _sha256_file(RAW_CACHE_PATH),
            "final_records_sha256": final_hash,
            "pipeline_version": PIPELINE_VERSION,
            "run_utc": _datetime.now(_timezone.utc).isoformat(),
            "token_estimator": "ceil(non-whitespace units x 1.35); fertility assumption selected for Hindi/code-mixed text",
            "dedup_method": duplicate_counts["method"],
            "decontamination_scope": "limited textual proxy: benchmark names and evaluation markers; no full benchmark corpus downloaded",
            "selection_source": selection_source,
        }
        MANIFEST_PATH.write_text(
            _json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        del decontaminated
        return {
            "stage_stats": stage_stats,
            "language_before": language_before,
            "language_after": language_after,
            "language_removed": language_removed,
            "quality_reasons": dict(quality_reasons),
            "duplicate_counts": duplicate_counts,
            "pii_counts": dict(pii_counts),
            "pii_documents": pii_documents,
            "decontamination_counts": dict(decontamination_counts),
            "decontamination_documents": decontamination_documents,
            "manifest": manifest,
        }

    return (
        DATASET_CONFIG,
        DATASET_ID,
        DATASET_LICENSE,
        DATASET_SPLIT,
        SOURCE_URL,
        TOKEN_FERTILITY_ASSUMPTION,
        run_cleaning_pipeline,
    )


@app.cell(hide_code=True)
def pipeline_controls(mo):
    pipeline_target_millions = mo.ui.slider(
        start=10,
        stop=30,
        step=5,
        value=30,
        label="Estimated sample size (millions of tokens)",
        show_value=True,
    )
    run_pipeline = mo.ui.run_button(
        label="Download/cache sample and run all eight stages"
    )
    mo.vstack(
        [
            mo.md(
                "The run is deterministic: it streams the first rows of the locked "
                "`verified/hin` split until the selected estimated-token target, "
                "then caches raw JSONL locally. Ten million tokens is the assignment minimum."
            ),
            mo.hstack([pipeline_target_millions, run_pipeline]),
        ]
    )
    return pipeline_target_millions, run_pipeline


@app.cell(hide_code=True)
def pipeline_execution(
    mo,
    pipeline_target_millions,
    run_cleaning_pipeline,
    run_pipeline,
):
    if not run_pipeline.value:
        pipeline_result = None
        pipeline_status = (
            "Click the run button after choosing the target size."
        )
    else:
        _target_tokens = int(pipeline_target_millions.value) * 1_000_000
        pipeline_status = f"Running the cleaning pipeline for an estimated {_target_tokens:,} tokens..."
        pipeline_result = run_cleaning_pipeline(_target_tokens)
        pipeline_status = (
            f"Completed {_target_tokens:,}-token target: "
            f"{pipeline_result['stage_stats'][-1]['documents']:,} final documents."
        )
    mo.md(pipeline_status)
    return pipeline_result, pipeline_status


@app.cell(hide_code=True)
def cleaning_report(
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_SPLIT,
    SOURCE_URL,
    TOKEN_FERTILITY_ASSUMPTION,
    mo,
    pipeline_result,
    pipeline_status,
):
    def _format_number(value):
        return f"{int(value):,}"

    def _percent(value):
        return f"{value:.2f}%"

    _strategy_rows = [
        {
            "#": 1,
            "strategy": "Extract",
            "decision": "Remove HTML tags, scripts, layout lines, and explicit cookie/navigation boilerplate.",
        },
        {
            "#": 2,
            "strategy": "Normalize",
            "decision": "HTML entities, NFC, mojibake, whitespace, controls, and chat markers; preserve Devanagari ZWJ/ZWNJ.",
        },
        {
            "#": 3,
            "strategy": "Language ID and validation",
            "decision": "Heuristic Hindi Devanagari, Hindi Latin, and code-mixed checks; reject other/unknown text.",
        },
        {
            "#": 4,
            "strategy": "Quality filtering",
            "decision": "Transparent length, symbol, link-density, repetition, and vocabulary-diversity rules.",
        },
        {
            "#": 5,
            "strategy": "Deduplication",
            "decision": "Exact SHA-256 plus local MinHash/LSH near-duplicate approximation.",
        },
        {
            "#": 6,
            "strategy": "PII scrubbing",
            "decision": "Redact emails, Indian phone numbers, IPv4, credential values, and Aadhaar-like numbers.",
        },
        {
            "#": 7,
            "strategy": "Decontamination",
            "decision": "Limited proxy scan for named benchmark/evaluation markers; report scope and matches.",
        },
        {
            "#": 8,
            "strategy": "Manifest / provenance",
            "decision": "Write source, license, split, selection, hashes, version, timestamp, and methods to manifest.json.",
        },
    ]

    if pipeline_result is None:
        _report = mo.vstack(
            [
                mo.md("# Session 4 cleaning case study"),
                mo.md(
                    "## 1. How many strategies?\nThe lecture minimum is **eight**. The implementation below keeps each stage measurable."
                ),
                mo.ui.table(_strategy_rows),
                mo.md(
                    "## 2. Dataset picked\n"
                    f"**[{DATASET_ID}]({SOURCE_URL})**, `verified` / `hin`, license **{DATASET_LICENSE}**. "
                    "This is the Session 3 document-level Hindi web corpus—not the lecture's Qwen demo dataset."
                ),
                mo.md(
                    "## 3–4. What will be cleaned and the additional concern\n"
                    "The pipeline is Hindi/code-mixed safe: it validates actual text instead of trusting `hin`, "
                    "does not require English punctuation, redacts PII without displaying values, and preserves "
                    "Indic ZWJ/ZWNJ while removing inconsistent chat-template markers.\n\n"
                    f"**Status:** {pipeline_status}"
                ),
            ]
        )
    else:
        _stats = pipeline_result["stage_stats"]
        _stage_rows = []
        for _index, _stage in enumerate(_stats):
            _previous = _stats[_index - 1] if _index else _stage
            _before_docs = _previous["documents"]
            _after_docs = _stage["documents"]
            _stage_rows.append(
                {
                    "stage": _stage["stage"],
                    "documents_after": _format_number(_after_docs),
                    "documents_removed": _format_number(
                        max(0, _before_docs - _after_docs)
                    ),
                    "characters_after": _format_number(_stage["characters"]),
                    "estimated_tokens_after": _format_number(
                        _stage["estimated_tokens"]
                    ),
                    "retained_from_previous": _percent(
                        100 * _after_docs / max(_before_docs, 1)
                    ),
                }
            )
        _raw_docs = _stats[0]["documents"]
        _final_docs = _stats[-1]["documents"]
        _removal_rows = [
            {
                "reason": "language mismatch",
                "documents": _format_number(
                    pipeline_result["language_removed"]
                ),
            },
        ]
        _removal_rows.extend(
            [
                {
                    "reason": "quality: " + reason,
                    "documents": _format_number(count),
                }
                for reason, count in sorted(
                    pipeline_result["quality_reasons"].items()
                )
            ]
        )
        _removal_rows.extend(
            [
                {
                    "reason": "exact duplicate",
                    "documents": _format_number(
                        pipeline_result["duplicate_counts"]["exact_removed"]
                    ),
                },
                {
                    "reason": "near duplicate",
                    "documents": _format_number(
                        pipeline_result["duplicate_counts"]["near_removed"]
                    ),
                },
                {
                    "reason": "decontamination proxy",
                    "documents": _format_number(
                        pipeline_result["decontamination_documents"]
                    ),
                },
            ]
        )
        _pii_rows = [
            {
                "PII category": category,
                "matches redacted": _format_number(count),
            }
            for category, count in sorted(
                pipeline_result["pii_counts"].items()
            )
        ] or [{"PII category": "none detected", "matches redacted": "0"}]
        _language_before_rows = [
            {
                "detected language": language,
                "documents before": _format_number(count),
            }
            for language, count in pipeline_result["language_before"].items()
        ]
        _language_after_rows = [
            {
                "detected language": language,
                "documents after": _format_number(count),
            }
            for language, count in pipeline_result["language_after"].items()
        ]
        _manifest_rows = [
            {"field": field, "value": str(value)}
            for field, value in pipeline_result["manifest"].items()
        ]
        _report = mo.vstack(
            [
                mo.md("# Session 4 cleaning case study"),
                mo.md(
                    "## 1. How many strategies?\nThe lecture minimum is **eight**; all eight are executed below."
                ),
                mo.ui.table(_strategy_rows),
                mo.md(
                    "## 2. Dataset picked\n"
                    f"**[{DATASET_ID}]({SOURCE_URL})**, configuration `{DATASET_CONFIG}`, split `{DATASET_SPLIT}`, "
                    f"license **{DATASET_LICENSE}**. The sample is selected deterministically from the streaming split "
                    f"with a **{TOKEN_FERTILITY_ASSUMPTION} estimated-token fertility assumption**."
                ),
                mo.md(
                    "## 3. What was cleaned and why\n"
                    "Extraction removes markup/layout noise; normalization repairs representation without destroying "
                    "Indic orthography; language validation checks the text rather than the folder label; quality rules "
                    "remove low-value web noise; deduplication removes repeated documents; PII is redacted; and the "
                    "decontamination proxy removes documents with strong evaluation markers."
                ),
                mo.md(
                    "## 4. Additional strategy / concern\n"
                    "Indic Unicode handling is explicit: NFC normalization is used, Devanagari ZWJ/ZWNJ are preserved, "
                    "and only non-semantic formatting invisibles are removed. Hindi Latin-script and natural Hinglish "
                    "mixing are retained when the language heuristic supports them. Ghost/chat markers are treated as "
                    "a normalization concern, not a ninth canonical stage."
                ),
                mo.md(
                    f"## 5. Final statistics\nThe run started with **{_format_number(_raw_docs)} documents** and retained "
                    f"**{_format_number(_final_docs)} documents** ({_percent(100 * _final_docs / max(_raw_docs, 1))}). "
                    f"{pipeline_status}"
                ),
                mo.md("### Counts after every stage"),
                mo.ui.table(_stage_rows),
                mo.md("### Removal counts by reason"),
                mo.ui.table(_removal_rows),
                mo.md("### Detected language mix before validation"),
                mo.ui.table(_language_before_rows),
                mo.md("### Detected language mix after final filtering"),
                mo.ui.table(_language_after_rows),
                mo.md(
                    "### Exact versus near duplicates\n"
                    f"Exact removed: **{_format_number(pipeline_result['duplicate_counts']['exact_removed'])}**. "
                    f"Near removed: **{_format_number(pipeline_result['duplicate_counts']['near_removed'])}**. "
                    f"Method: {pipeline_result['duplicate_counts']['method']}"
                ),
                mo.md("### PII category counts (values are never displayed)"),
                mo.ui.table(_pii_rows),
                mo.md(
                    "### Decontamination result\n"
                    f"Removed **{_format_number(pipeline_result['decontamination_documents'])} documents** using a limited "
                    "textual proxy over GSM8K, MMLU, HellaSwag, Winogrande, TruthfulQA, BIG-bench, ARC Challenge, "
                    "benchmark-answer, and eval-harness markers. This is not a full benchmark-corpus comparison; that "
                    "scope limitation is recorded in the manifest."
                ),
                mo.md("### Manifest / provenance"),
                mo.ui.table(_manifest_rows),
            ]
        )
    _report
    return


if __name__ == "__main__":
    app.run()
