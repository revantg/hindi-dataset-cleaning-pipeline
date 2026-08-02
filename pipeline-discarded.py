import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def assignment(mo):
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

    ## Eight-stage checklist

    1. Extract
    2. Normalize
    3. Language ID and validation
    4. Quality filtering
    5. Deduplication
    6. PII scrubbing
    7. Decontamination
    8. Manifest / provenance

    This notebook intentionally runs those stages in separate, inspectable cells.
    """)
    return


@app.cell(hide_code=True)
def dataset_selection(mo):
    mo.md(r"""
    # Dataset selection

    | Candidate | Decision |
    |---|---|
    | Hindi Wikipedia | In-band, but too clean for a cleaning-pipeline project. |
    | OSCAR Hindi | Good raw-web alternative, but manually gated and operationally heavier. |
    | IndicCorp Hindi | Mostly short, already-cleaned lines; weak dedup/PII/format signal. |
    | xP3 / IndicParaphrase | Instruction/paraphrase pairs, not a document-pretraining corpus. |
    | **Sangraha verified Hindi** | **Chosen.** Session 3 corpus; open, document-level Hindi web text, provenance fields, and enough residual noise for the pipeline. |

    **Locked dataset:** `ai4bharat/sangraha` · `verified` / `hin` · stream and stop at about **30M estimated tokens**.

    **Why this choice:** it is a document-level Hindi web corpus with a meaningful language-validation, web-noise, duplicate, and PII signal. The lecture's Qwen reasoning-distilled demo dataset is not used.
    """)
    return


@app.cell
def preview_controls(mo):
    fetch_preview = mo.ui.run_button(label="Fetch 5 compact Sangraha Hindi previews")
    mo.vstack(
        [
            mo.md(
                "This copies the current notebook's lightweight source-preview step. "
                "It requests hosted rows only and never loads the full split."
            ),
            fetch_preview,
        ]
    )
    return (fetch_preview,)


@app.cell
def preview_fetch(fetch_preview):
    import json
    from urllib.parse import urlencode
    from urllib.request import urlopen

    preview_rows = []
    preview_payload = {"rows": []}
    if not fetch_preview.value:
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
            preview_payload = json.load(response)

        for index, item in enumerate(preview_payload["rows"][:5]):
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
    return preview_rows, preview_payload, preview_status


@app.cell
def preview_report(mo, preview_rows, preview_status):
    mo.vstack(
        [
            mo.md(preview_status),
            mo.ui.table(preview_rows),
        ]
    )
    return


@app.cell(hide_code=True)
def pipeline_setup():
    import hashlib
    import json
    import math
    from pathlib import Path

    from datasets import load_dataset

    DATASET_ID = "ai4bharat/sangraha"
    DATASET_CONFIG = "verified"
    DATASET_SPLIT = "hin"
    DATASET_LICENSE = "CC BY 4.0"
    DATASET_REVISION = "8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8"
    SOURCE_URL = "https://huggingface.co/datasets/ai4bharat/sangraha"
    PIPELINE_VERSION = "s4-stepwise-2026-08-02-v1"
    TOKEN_FERTILITY_ASSUMPTION = 1.35
    WORK_DIR = Path("data/sangraha_hi_verified_30m_stepwise")
    RAW_PATH = WORK_DIR / "00_raw.jsonl"
    STAGE_PATHS = {
        "extract": WORK_DIR / "01_extract.jsonl",
        "normalize": WORK_DIR / "02_normalize.jsonl",
        "language_validation": WORK_DIR / "03_language_validation.jsonl",
        "quality_filter": WORK_DIR / "04_quality_filter.jsonl",
        "deduplication": WORK_DIR / "05_deduplication.jsonl",
        "pii_scrubbing": WORK_DIR / "06_pii_scrubbing.jsonl",
        "decontamination": WORK_DIR / "07_decontamination.jsonl",
    }
    MANIFEST_PATH = WORK_DIR / "manifest.json"
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    def estimate_tokens(text):
        units = len(text.split())
        return int(math.ceil(units * TOKEN_FERTILITY_ASSUMPTION)) if units else 0

    def iter_jsonl(path):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def stats_for_path(path):
        if not path.exists():
            return None
        documents = 0
        characters = 0
        estimated_tokens = 0
        for record in iter_jsonl(path):
            text = record.get("text", "")
            documents += 1
            characters += len(text)
            estimated_tokens += record.get("estimated_tokens", estimate_tokens(text))
        return {
            "documents": documents,
            "characters": characters,
            "estimated_tokens": estimated_tokens,
        }

    def sha256_file(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    return (
        DATASET_CONFIG,
        DATASET_ID,
        DATASET_LICENSE,
        DATASET_REVISION,
        DATASET_SPLIT,
        MANIFEST_PATH,
        PIPELINE_VERSION,
        RAW_PATH,
        SOURCE_URL,
        STAGE_PATHS,
        TOKEN_FERTILITY_ASSUMPTION,
        WORK_DIR,
        estimate_tokens,
        iter_jsonl,
        json,
        load_dataset,
        sha256_file,
        stats_for_path,
    )


@app.cell
def download_controls(mo):
    sample_target_millions = mo.ui.slider(
        start=10,
        stop=30,
        step=5,
        value=30,
        label="Estimated sample target (millions of tokens)",
        show_value=True,
    )
    download_button = mo.ui.run_button(label="Download/cache the selected Sangraha sample")
    mo.vstack(
        [
            mo.md(
                "The download cell streams the first rows of `verified/hin` until the selected "
                "estimated-token target and writes a local JSONL cache. The default is 30M; "
                "10M is the assignment minimum."
            ),
            mo.hstack([sample_target_millions, download_button]),
        ]
    )
    return download_button, sample_target_millions


@app.cell
def download_sample(
    mo,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_SPLIT,
    RAW_PATH,
    download_button,
    estimate_tokens,
    iter_jsonl,
    load_dataset,
    sample_target_millions,
    stats_for_path,
):
    import json
    target_tokens = int(sample_target_millions.value) * 1_000_000
    cached_stats = stats_for_path(RAW_PATH)
    if not download_button.value:
        raw_stats = cached_stats
        raw_ready = raw_stats is not None and raw_stats["estimated_tokens"] >= target_tokens
        download_status = (
            f"Cached raw sample found: {raw_stats['documents']:,} documents."
            if raw_stats
            else "Click the download button to start the deterministic stream."
        )
    elif cached_stats and cached_stats["estimated_tokens"] >= target_tokens:
        raw_stats = cached_stats
        raw_ready = True
        download_status = "Using the existing raw JSONL cache; no network download was needed."
    else:
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(
            DATASET_ID,
            DATASET_CONFIG,
            split=DATASET_SPLIT,
            streaming=True,
        )
        estimated_total = 0
        with RAW_PATH.open("w", encoding="utf-8") as handle:
            for source_row in dataset:
                text = source_row.get("text") or ""
                doc_id = str(source_row.get("doc_id") or "")
                if not doc_id or not text:
                    continue
                estimated = estimate_tokens(text)
                record = {
                    "doc_id": doc_id,
                    "type": source_row.get("type", "unknown"),
                    "text": text,
                    "estimated_tokens": estimated,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                estimated_total += estimated
                if estimated_total >= target_tokens:
                    break
        raw_stats = stats_for_path(RAW_PATH)
        raw_ready = raw_stats is not None and raw_stats["estimated_tokens"] >= target_tokens
        download_status = (
            f"Downloaded {raw_stats['documents']:,} documents and "
            f"{raw_stats['estimated_tokens']:,} estimated tokens."
        )
    mo.md(download_status)
    return download_status, raw_ready, raw_stats


@app.cell
def extract_stage(mo, RAW_PATH, STAGE_PATHS, iter_jsonl, raw_ready, stats_for_path):
    import re
    import json

    extract_button = mo.ui.run_button(label="Run 1 · Extract")
    extract_path = STAGE_PATHS["extract"]
    extract_stats = stats_for_path(extract_path)
    if not raw_ready:
        extract_status = "Download the raw sample first."
        extract_ready = False
    elif extract_button.value:
        script_blocks = re.compile(
            r"(?is)<(?:script|style|noscript|template|iframe)[^>]*>.*?</(?:script|style|noscript|template|iframe)>"
        )
        block_tags = re.compile(
            r"(?is)</?(?:article|aside|br|div|footer|h[1-6]|header|li|main|nav|p|section|table|tr|ul)[^>]*>"
        )
        any_tag = re.compile(r"(?is)<[^>]{1,400}>")
        boilerplate = (
            "accept cookies", "accept all cookies", "cookie settings", "cookie policy",
            "privacy policy", "subscribe to our newsletter", "skip to content",
            "enable javascript", "all rights reserved", "advertisement",
        )
        with extract_path.open("w", encoding="utf-8") as output:
            for record in iter_jsonl(RAW_PATH):
                text = script_blocks.sub("\n", record.get("text", ""))
                text = block_tags.sub("\n", text)
                text = any_tag.sub(" ", text)
                lines = []
                for line in text.splitlines():
                    compact = re.sub(r"\s+", " ", line).strip()
                    if compact and not any(marker in compact.casefold() for marker in boilerplate):
                        lines.append(compact)
                record["text"] = "\n".join(lines)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        extract_stats = stats_for_path(extract_path)
        extract_status = f"Extract complete: {extract_stats['documents']:,} documents."
        extract_ready = True
    else:
        extract_status = "Ready. Click the stage button to remove markup and layout boilerplate."
        extract_ready = extract_stats is not None
    mo.vstack([extract_button, mo.md(extract_status), mo.ui.table([extract_stats] if extract_stats else [])])
    return extract_button, extract_ready, extract_stats, extract_status


@app.cell
def normalize_stage(mo, STAGE_PATHS, extract_ready, estimate_tokens, iter_jsonl, stats_for_path):
    import html
    import json
    import re
    import unicodedata

    normalize_button = mo.ui.run_button(label="Run 2 · Normalize")
    source_path = STAGE_PATHS["extract"]
    normalize_path = STAGE_PATHS["normalize"]
    normalize_stats = stats_for_path(normalize_path)
    if not extract_ready:
        normalize_status = "Run Extract first."
        normalize_ready = False
    elif normalize_button.value:
        mojibake = {
            "â€“": "–", "â€”": "—", "â€˜": "‘", "â€™": "’",
            "â€œ": "“", "â€�": "”", "â€¦": "…", "Â ": " ", "Ã©": "é",
        }
        chat_marker = re.compile(
            r"(?im)^\s*(?:\[?(?:system|user|assistant|human|bot)\]?|<\|(?:system|user|assistant|human|bot)\|>)\s*:?[ \t]*"
        )
        with normalize_path.open("w", encoding="utf-8") as output:
            for record in iter_jsonl(source_path):
                text = html.unescape(record.get("text", ""))
                for source, replacement in mojibake.items():
                    text = text.replace(source, replacement)
                text = unicodedata.normalize("NFC", text)
                # Preserve U+200C ZWNJ and U+200D ZWJ; remove only non-semantic invisibles.
                text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u2060", "")
                text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
                text = chat_marker.sub("", text)
                text = re.sub(r"\r\n?", "\n", text)
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
                record["text"] = text
                record["estimated_tokens"] = estimate_tokens(text)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        normalize_stats = stats_for_path(normalize_path)
        normalize_status = f"Normalize complete: {normalize_stats['documents']:,} documents."
        normalize_ready = True
    else:
        normalize_status = "Ready. NFC-normalize and clean representation markers without stripping Indic joiners."
        normalize_ready = normalize_stats is not None
    mo.vstack([normalize_button, mo.md(normalize_status), mo.ui.table([normalize_stats] if normalize_stats else [])])
    return normalize_button, normalize_ready, normalize_stats, normalize_status


@app.cell
def language_validation_stage(mo, STAGE_PATHS, normalize_ready, iter_jsonl, stats_for_path):
    import re
    import json

    language_button = mo.ui.run_button(label="Run 3 · Language ID and validation")
    source_path = STAGE_PATHS["normalize"]
    language_path = STAGE_PATHS["language_validation"]
    language_stats = stats_for_path(language_path)
    language_removed = 0
    language_mix = {}
    if not normalize_ready:
        language_status = "Run Normalize first."
        language_ready = False
    elif language_button.value:
        devanagari = re.compile(r"[\u0900-\u097f]")
        latin = re.compile(r"[A-Za-z]")
        deva_cues = re.compile(r"(?:है|हैं|था|थे|थी|और|का|की|के|को|में|से|यह|वह|एक|नहीं|क्या|कर|लिए|पर|तो|भी|जब|तक)")
        latin_cues = re.compile(r"\b(?:hai|hain|tha|the|thi|aur|ka|ki|ke|ko|mein|se|yeh|woh|ek|nahi|kya|kar|liye|par|to|bhi|jab|tak)\b", re.I)
        accepted = {"hi-Deva", "hi-Deva+Latin", "hi-Latin"}
        with language_path.open("w", encoding="utf-8") as output:
            for record in iter_jsonl(source_path):
                text = record.get("text", "")
                deva_count = len(devanagari.findall(text))
                latin_count = len(latin.findall(text))
                letters = deva_count + latin_count
                if letters < 20:
                    detected = "unknown"
                elif deva_count >= 30 and deva_count / letters >= 0.25:
                    detected = "hi-Deva+Latin" if latin_count and deva_count / letters < 0.75 else "hi-Deva"
                elif latin_count >= 40 and len(latin_cues.findall(text)) >= 2:
                    detected = "hi-Latin"
                elif len(deva_cues.findall(text)) >= 2:
                    detected = "hi-Deva"
                else:
                    detected = "other"
                language_mix[detected] = language_mix.get(detected, 0) + 1
                if detected not in accepted:
                    language_removed += 1
                    continue
                record["detected_language"] = detected
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        language_stats = stats_for_path(language_path)
        language_status = f"Language validation complete: removed {language_removed:,} mismatched/unknown documents."
        language_ready = True
    else:
        language_status = "Ready. Validate actual text and retain Hindi Devanagari, Hindi Latin, and supported code-mixed text."
        language_ready = language_stats is not None
    language_rows = [{"language": key, "documents": value} for key, value in sorted(language_mix.items())]
    mo.vstack([language_button, mo.md(language_status), mo.ui.table(language_rows), mo.ui.table([language_stats] if language_stats else [])])
    return language_button, language_mix, language_ready, language_removed, language_stats, language_status


@app.cell
def quality_filter_stage(mo, STAGE_PATHS, language_ready, iter_jsonl, stats_for_path):
    import re
    import json

    quality_button = mo.ui.run_button(label="Run 4 · Quality filtering")
    source_path = STAGE_PATHS["language_validation"]
    quality_path = STAGE_PATHS["quality_filter"]
    quality_stats = stats_for_path(quality_path)
    quality_reasons = {}
    if not language_ready:
        quality_status = "Run Language ID and validation first."
        quality_ready = False
    elif quality_button.value:
        with quality_path.open("w", encoding="utf-8") as output:
            for record in iter_jsonl(source_path):
                text = record.get("text", "")
                units = re.findall(r"\S+", text)
                reason = ""
                if len(text) < 120 or len(units) < 20:
                    reason = "too_short"
                elif re.search(r"(.)\1{9,}", text):
                    reason = "malformed_repetition"
                elif sum(character.isalnum() for character in text) / max(len(text), 1) < 0.28:
                    reason = "malformed_symbols"
                else:
                    link_like = len(re.findall(r"(?:https?://|www\.|\.com\b)", text, re.I))
                    unique_ratio = len(set(unit.casefold() for unit in units)) / max(len(units), 1)
                    shingles = [" ".join(units[index : index + 5]) for index in range(max(0, len(units) - 4))]
                    if link_like >= 8 and link_like / max(len(units), 1) > 0.12:
                        reason = "link_heavy_spam"
                    elif len(units) >= 60 and unique_ratio < 0.22:
                        reason = "low_diversity"
                    elif shingles and len(set(shingles)) / len(shingles) < 0.35:
                        reason = "repetitive_text"
                if reason:
                    quality_reasons[reason] = quality_reasons.get(reason, 0) + 1
                else:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
        quality_stats = stats_for_path(quality_path)
        quality_status = f"Quality filtering complete: removed {sum(quality_reasons.values()):,} low-value documents."
        quality_ready = True
    else:
        quality_status = "Ready. Apply transparent length, malformed-symbol, link-density, repetition, and diversity checks."
        quality_ready = quality_stats is not None
    reason_rows = [{"reason": key, "documents": value} for key, value in sorted(quality_reasons.items())]
    mo.vstack([quality_button, mo.md(quality_status), mo.ui.table(reason_rows), mo.ui.table([quality_stats] if quality_stats else [])])
    return quality_button, quality_ready, quality_reasons, quality_stats, quality_status


@app.cell
def deduplication_stage(mo, STAGE_PATHS, quality_ready, iter_jsonl, stats_for_path):
    import hashlib
    import json
    import re

    dedup_button = mo.ui.run_button(label="Run 5 · Exact + near deduplication")
    source_path = STAGE_PATHS["quality_filter"]
    dedup_path = STAGE_PATHS["deduplication"]
    dedup_stats = stats_for_path(dedup_path)
    duplicate_summary = {"exact_removed": 0, "near_removed": 0}
    if not quality_ready:
        dedup_status = "Run Quality filtering first."
        dedup_ready = False
    elif dedup_button.value:
        records = list(iter_jsonl(source_path))
        kept = []
        kept_signatures = []
        exact_hashes = set()
        lsh_buckets = {}
        exact_removed = 0
        near_removed = 0
        permutations = tuple(
            (((0x9E3779B185EBCA87 * (index + 1)) | 1), 0xC2B2AE3D27D4EB4F ^ (index * 0x165667B19E3779F9))
            for index in range(16)
        )
        mask = (1 << 64) - 1

        def signature(text):
            tokens = [token.casefold() for token in re.findall(r"\w+", text, re.UNICODE)]
            if len(tokens) < 10:
                return None
            shingle_hashes = set()
            for index in range(len(tokens) - 4):
                shingle = "\x1f".join(tokens[index : index + 5]).encode("utf-8")
                shingle_hashes.add(int.from_bytes(hashlib.blake2b(shingle, digest_size=8).digest(), "little"))
            return tuple(min((a * value + b) & mask for value in shingle_hashes) for a, b in permutations)

        for record in records:
            exact_hash = hashlib.sha256(record.get("text", "").encode("utf-8")).hexdigest()
            if exact_hash in exact_hashes:
                exact_removed += 1
                continue
            current_signature = signature(record.get("text", ""))
            duplicate = False
            if current_signature is not None:
                candidates = set()
                for band_start in range(0, len(current_signature), 4):
                    key = (band_start // 4, current_signature[band_start : band_start + 4])
                    candidates.update(lsh_buckets.get(key, ()))
                for candidate_index in sorted(candidates):
                    candidate = kept[candidate_index]
                    candidate_signature = kept_signatures[candidate_index]
                    if candidate_signature is None:
                        continue
                    length_ratio = len(record["text"]) / max(len(candidate["text"]), 1)
                    similarity = sum(left == right for left, right in zip(current_signature, candidate_signature)) / len(current_signature)
                    if 0.80 <= length_ratio <= 1.25 and similarity >= 0.875:
                        duplicate = True
                        break
            if duplicate:
                near_removed += 1
                continue
            kept_index = len(kept)
            kept.append(record)
            kept_signatures.append(current_signature)
            exact_hashes.add(exact_hash)
            if current_signature is not None:
                for band_start in range(0, len(current_signature), 4):
                    key = (band_start // 4, current_signature[band_start : band_start + 4])
                    lsh_buckets.setdefault(key, []).append(kept_index)
        with dedup_path.open("w", encoding="utf-8") as output:
            for record in kept:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        dedup_stats = stats_for_path(dedup_path)
        duplicate_summary = {"exact_removed": exact_removed, "near_removed": near_removed}
        dedup_status = f"Deduplication complete: exact {exact_removed:,}; near {near_removed:,}."
        dedup_ready = True
    else:
        dedup_status = "Ready. Exact SHA-256 and local MinHash/LSH near-duplicate detection are separate checks."
        dedup_ready = dedup_stats is not None
    duplicate_rows = [{"kind": key.replace("_", " "), "documents": value} for key, value in duplicate_summary.items()]
    mo.vstack([dedup_button, mo.md(dedup_status), mo.ui.table(duplicate_rows), mo.ui.table([dedup_stats] if dedup_stats else [])])
    return dedup_button, dedup_ready, dedup_stats, dedup_status, duplicate_summary


@app.cell
def pii_scrubbing_stage(mo, STAGE_PATHS, dedup_ready, iter_jsonl, stats_for_path):
    import re
    import json

    pii_button = mo.ui.run_button(label="Run 6 · PII scrubbing")
    source_path = STAGE_PATHS["deduplication"]
    pii_path = STAGE_PATHS["pii_scrubbing"]
    pii_stats = stats_for_path(pii_path)
    pii_counts = {}
    pii_documents = 0
    if not dedup_ready:
        pii_status = "Run Deduplication first."
        pii_ready = False
    elif pii_button.value:
        patterns = (
            ("credential", re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|password|passwd|secret)\s*[:=]\s*[^\s,;]+"), "[REDACTED_CREDENTIAL]"),
            ("email", re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])"), "[REDACTED_EMAIL]"),
            ("phone", re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)"), "[REDACTED_PHONE]"),
            ("ipv4", re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"), "[REDACTED_IP]"),
            ("aadhaar_like", re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)"), "[REDACTED_ID]"),
        )
        with pii_path.open("w", encoding="utf-8") as output:
            for record in iter_jsonl(source_path):
                text = record.get("text", "")
                changed = text
                document_had_pii = False
                for category, pattern, replacement in patterns:
                    changed, count = pattern.subn(replacement, changed)
                    if count:
                        document_had_pii = True
                        pii_counts[category] = pii_counts.get(category, 0) + count
                if document_had_pii:
                    pii_documents += 1
                record["text"] = changed
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        pii_stats = stats_for_path(pii_path)
        pii_status = f"PII scrubbing complete: redacted {sum(pii_counts.values()):,} matches across {pii_documents:,} documents."
        pii_ready = True
    else:
        pii_status = "Ready. Redact emails, Indian phones, IPv4, credentials, and Aadhaar-like identifiers; never display values."
        pii_ready = pii_stats is not None
    pii_rows = [{"category": key, "matches redacted": value} for key, value in sorted(pii_counts.items())]
    mo.vstack([pii_button, mo.md(pii_status), mo.ui.table(pii_rows), mo.ui.table([pii_stats] if pii_stats else [])])
    return pii_button, pii_counts, pii_documents, pii_ready, pii_stats, pii_status


@app.cell
def decontamination_stage(mo, STAGE_PATHS, pii_ready, iter_jsonl, stats_for_path):
    import json
    decontamination_button = mo.ui.run_button(label="Run 7 · Decontamination proxy")
    source_path = STAGE_PATHS["pii_scrubbing"]
    final_path = STAGE_PATHS["decontamination"]
    decontamination_stats = stats_for_path(final_path)
    decontamination_counts = {}
    decontamination_documents = 0
    if not pii_ready:
        decontamination_status = "Run PII scrubbing first."
        final_ready = False
    elif decontamination_button.value:
        terms = (
            "gsm8k", "mmlu", "hellaswag", "winogrande", "truthfulqa",
            "bigbench", "arc challenge", "benchmark answer", "eval harness",
        )
        with final_path.open("w", encoding="utf-8") as output:
            for record in iter_jsonl(source_path):
                lowered = record.get("text", "").casefold()
                matches = [term for term in terms if term in lowered]
                if matches:
                    decontamination_documents += 1
                    for term in matches:
                        decontamination_counts[term] = decontamination_counts.get(term, 0) + 1
                    continue
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        decontamination_stats = stats_for_path(final_path)
        decontamination_status = (
            f"Proxy complete: removed {decontamination_documents:,} documents. "
            "This is not a full benchmark-corpus comparison."
        )
        final_ready = True
    else:
        decontamination_status = "Ready. Scan named benchmark/evaluation markers and record the limited scope."
        final_ready = decontamination_stats is not None
    decontamination_rows = [{"marker": key, "matches": value} for key, value in sorted(decontamination_counts.items())]
    mo.vstack([decontamination_button, mo.md(decontamination_status), mo.ui.table(decontamination_rows), mo.ui.table([decontamination_stats] if decontamination_stats else [])])
    return decontamination_button, decontamination_counts, decontamination_documents, decontamination_status, decontamination_stats, final_path, final_ready


@app.cell
def manifest_stage(
    mo,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    DATASET_SPLIT,
    MANIFEST_PATH,
    PIPELINE_VERSION,
    RAW_PATH,
    SOURCE_URL,
    STAGE_PATHS,
    TOKEN_FERTILITY_ASSUMPTION,
    decontamination_documents,
    duplicate_summary,
    final_path,
    final_ready,
    pii_counts,
    sha256_file,
    stats_for_path,
):
    from datetime import datetime, timezone
    import json

    manifest_button = mo.ui.run_button(label="Run 8 · Write manifest / provenance")
    manifest = None
    if not final_ready:
        manifest_status = "Run Decontamination first."
    elif manifest_button.value:
        stage_stats = {"raw": stats_for_path(RAW_PATH), "extract": stats_for_path(STAGE_PATHS["extract"]), "normalize": stats_for_path(STAGE_PATHS["normalize"]), "language_validation": stats_for_path(STAGE_PATHS["language_validation"]), "quality_filter": stats_for_path(STAGE_PATHS["quality_filter"]), "deduplication": stats_for_path(STAGE_PATHS["deduplication"]), "pii_scrubbing": stats_for_path(STAGE_PATHS["pii_scrubbing"]), "decontamination": stats_for_path(final_path)}
        manifest = {
            "source_url": SOURCE_URL,
            "dataset_id": DATASET_ID,
            "license": DATASET_LICENSE,
            "revision": DATASET_REVISION,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "sample_selection": "deterministic first streaming rows until selected estimated-token target",
            "target_estimated_tokens": stage_stats["raw"]["estimated_tokens"] if stage_stats["raw"] else None,
            "raw_sha256": sha256_file(RAW_PATH),
            "final_sha256": sha256_file(final_path),
            "pipeline_version": PIPELINE_VERSION,
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "token_estimator": "ceil(non-whitespace units x 1.35)",
            "deduplication": duplicate_summary,
            "pii_category_counts": pii_counts,
            "decontamination_documents_removed": decontamination_documents,
            "decontamination_scope": "limited textual proxy over named benchmark/evaluation markers; no full benchmark corpus downloaded",
            "stage_stats": stage_stats,
        }
        MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_status = f"Manifest written to `{MANIFEST_PATH}`."
    elif MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest_status = f"Existing manifest found at `{MANIFEST_PATH}`."
    else:
        manifest_status = "Ready. Write the reproducibility manifest after all seven data stages."
    mo.vstack([manifest_button, mo.md(manifest_status), mo.ui.table([manifest] if manifest else [])])
    return manifest, manifest_button, manifest_status


@app.cell(hide_code=True)
def pipeline_overview(
    mo,
    download_status,
    extract_stats,
    normalize_stats,
    language_stats,
    quality_stats,
    dedup_stats,
    pii_stats,
    decontamination_stats,
    manifest_status,
):
    rows = [
        {"stage": "Download / raw", "status": download_status, "documents": extract_stats and "" or ""},
        {"stage": "Extract", "status": "complete" if extract_stats else "waiting", "documents": extract_stats and extract_stats["documents"] or ""},
        {"stage": "Normalize", "status": "complete" if normalize_stats else "waiting", "documents": normalize_stats and normalize_stats["documents"] or ""},
        {"stage": "Language validation", "status": "complete" if language_stats else "waiting", "documents": language_stats and language_stats["documents"] or ""},
        {"stage": "Quality filtering", "status": "complete" if quality_stats else "waiting", "documents": quality_stats and quality_stats["documents"] or ""},
        {"stage": "Deduplication", "status": "complete" if dedup_stats else "waiting", "documents": dedup_stats and dedup_stats["documents"] or ""},
        {"stage": "PII scrubbing", "status": "complete" if pii_stats else "waiting", "documents": pii_stats and pii_stats["documents"] or ""},
        {"stage": "Decontamination", "status": "complete" if decontamination_stats else "waiting", "documents": decontamination_stats and decontamination_stats["documents"] or ""},
        {"stage": "Manifest", "status": manifest_status, "documents": ""},
    ]
    mo.vstack(
        [
            mo.md("## Step-by-step run status\nClick each stage button in order. Every stage reads the previous JSONL artifact and writes its own artifact under `data/sangraha_hi_verified_30m_stepwise/`."),
            mo.ui.table(rows),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
