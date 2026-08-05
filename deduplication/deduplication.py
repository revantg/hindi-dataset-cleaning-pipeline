import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import hashlib
    import json
    import math
    import os
    import random
    import re
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    import marimo as mo
    import numpy as np

    return (
        Path,
        datetime,
        hashlib,
        json,
        math,
        mo,
        np,
        os,
        random,
        re,
        time,
        timezone,
    )


@app.cell(hide_code=True)
def workbook_scope(mo):
    mo.md(r"""
    # Session 4 — Deduplication for Hindi/Indic Text

    ## Scope and decision boundary

    This workbook is a survey of **nine deduplication techniques** with exactly
    **one executed end-to-end** on a bounded, deterministic slice of
    `ai4bharat/sangraha` (`verified` / `hin`, first 20,000 non-empty documents).
    The executed path is exact document hashing followed by MinHash/LSH
    near-duplicate detection with exact-Jaccard verification. The remaining
    eight techniques are documented with their published parameters and the
    concrete resource that blocks running them here.

    This is a **bounded local pass over 20,000 documents, not global
    cross-corpus deduplication**. Every number reported below is a measurement
    of this slice. None of them is a corpus-wide rate for Sangraha, and none
    of them is evidence about any other config, split, or language.

    Deduplication **selects documents; it never mutates them**. Normalization
    exists only to derive a hash or a shingle. The retained text is byte-identical
    to the source.

    ## The eight Session 4 stages

    | # | Stage | Boundary in this workbook |
    |---:|---|---|
    | 1 | Extract | Separate assignment stage; Sangraha's flattened `text` is taken as given and no layout recovery is attempted. |
    | 2 | Normalize | Whitespace collapse and `casefold()` are applied to a throwaway hashing copy only; the cache is never rewritten. |
    | 3 | Language validation | Separate stage; the `hin` label is accepted as the source's own and is not re-verified here. |
    | 4 | Quality filtering | Separate workbook (`quality-filtering/quality_filtering.py`); a duplicate is not a quality judgement and no document is dropped here for being low quality. |
    | 5 | **Deduplication** | **Executed here:** exact SHA-256 document hashing, then word-5-gram MinHash over 112 permutations, 14×8 LSH banding, and exact-Jaccard verification at threshold 0.75. |
    | 6 | PII scrubbing | Separate canonical scrubber; previews below are capped at 200 characters and this workbook must not be read as having scrubbed anything. |
    | 7 | Decontamination | Separate benchmark-overlap stage; near-duplicate detection against a benchmark set is a different question from intra-corpus dedup. |
    | 8 | Manifest / provenance | The bounded download writes `selection_manifest.json`; the run writes `dedup_metrics.json`. |

    ## Sangraha was already deduplicated, and that is the point

    AI4Bharat's Setu pipeline applied fuzzy MinHash/LSH deduplication before
    release — see the [Sangraha dataset card](https://huggingface.co/datasets/ai4bharat/sangraha)
    and [IndicLLMSuite](https://arxiv.org/abs/2403.06350). So this workbook is
    not cleaning a raw crawl; it is measuring **residual duplication in an
    already-cleaned corpus**, which is the more realistic engineering situation.

    The measured consequence, on identical 20,000-document samples:

    | Slice | Duplicates found |
    |---|---|
    | `unverified` / `hin` | **1** in 20,000 |
    | `verified` / `hin` | **328** exact + **28** near = 356 (1.78%) |

    `verified` / `hin` is therefore the sample. A demonstration on
    `unverified` / `hin` would have produced a null result and taught nothing
    about the mechanism.

    ## Indic normalization guard

    Hashing normalization is **whitespace collapse plus `casefold()`, and
    nothing else**:

    - **No NFKC.** It rewrites Devanagari compatibility forms and can merge
      distinct sequences.
    - **No ZWJ/ZWNJ removal.** These control conjunct rendering; deleting them
      changes the word.
    - **No virama or matra stripping.** The virama is a phonemic component of
      Devanagari orthography, not decoration.
    - **No punctuation stripping.** The danda `।` is sentence-terminating
      content.

    Two documents that differ only in Devanagari combining marks are treated as
    **different documents** here. That is deliberate: an over-aggressive
    normalizer would produce false duplicates that silently delete valid Hindi
    text, and a deletion is not recoverable downstream.

    ### Primary reading

    [Broder 1997](https://doi.org/10.1109/SEQUEN.1997.666900) ·
    [C4](https://arxiv.org/abs/1910.10683) ·
    [Lee et al. 2021](https://arxiv.org/abs/2107.06499) ·
    [FineWeb](https://arxiv.org/abs/2406.17557) ·
    [RefinedWeb](https://arxiv.org/abs/2306.01116) ·
    [Dolma](https://arxiv.org/abs/2402.00159) ·
    [SemDeDup](https://arxiv.org/abs/2303.09540) ·
    [IndicLLMSuite](https://arxiv.org/abs/2403.06350)
    """)
    return


@app.cell
def bounded_corpus_setup(
    Path,
    datetime,
    hashlib,
    json,
    math,
    os,
    re,
    time,
    timezone,
):
    # Source contract and bounded cache. Every parameter the run depends on is a
    # module constant, not a UI control: interactive retuning would invalidate the
    # persisted artifacts and the reproducibility claim.

    PIPELINE_VERSION = "s4-deduplication-2026-08-05-v1"
    DATASET_ID = "ai4bharat/sangraha"
    DATASET_CONFIG = "verified"
    DATASET_SPLIT = "hin"
    DATASET_REVISION = "8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8"
    DATASET_LICENSE = "CC BY 4.0"
    SOURCE_URL = "https://huggingface.co/datasets/ai4bharat/sangraha"
    SAMPLE_DOCUMENTS = 20_000
    TOKEN_FERTILITY_ASSUMPTION = 1.35

    # FineWeb's published near-duplicate configuration (arXiv:2406.17557 s3.4):
    # word 5-grams, 112 hash functions, 14 bands of 8 rows, ~0.75 similarity target.
    SHINGLE_K = 5
    NUM_HASHES = 112
    LSH_BANDS = 14
    LSH_ROWS = 8
    JACCARD_THRESHOLD = 0.75
    MINHASH_SEED = 20260804
    MIN_TOKENS_FOR_MINHASH = 15
    MAX_BUCKET_SIZE = 500

    DATA_ROOT = Path("deduplication/data")
    RAW_PATH = DATA_ROOT / "raw.jsonl"
    MANIFEST_PATH = DATA_ROOT / "selection_manifest.json"
    METRICS_PATH = DATA_ROOT / "dedup_metrics.json"
    RESULTS_DIR = DATA_ROOT / "results"


    def estimate_tokens(text):
        """Repository-wide token proxy: ceil(non-whitespace units x 1.35).

        Not a tokenizer count. Used only so corpus size is reported consistently
        across the Session 4 workbooks.
        """
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


    def sha256_path(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()


    def atomic_write_text(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(value, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


    def read_raw_records():
        records = []
        with RAW_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records


    def cache_status():
        """Validate the local cache against its manifest, including its SHA-256.

        A truncated, mutated, or differently-sourced cache is rejected rather than
        silently used, so a reported duplicate rate always belongs to the pinned
        revision.
        """
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
                return (
                    False,
                    manifest,
                    f"The cache manifest does not contain exactly {SAMPLE_DOCUMENTS} records.",
                )
            records = read_raw_records()
            if len(records) != SAMPLE_DOCUMENTS:
                return (
                    False,
                    manifest,
                    f"The raw cache does not contain exactly {SAMPLE_DOCUMENTS} records.",
                )
            if manifest.get("raw_sha256") != sha256_path(RAW_PATH):
                return False, manifest, "The raw cache hash does not match its manifest."
            return True, manifest, "Matching bounded cache verified."
        except (OSError, ValueError, TypeError, KeyError) as error:
            return False, None, f"Cache validation failed: {type(error).__name__}: {error}"


    def materialize_cache():
        """Stream the pinned revision in source order and write the cache atomically.

        Selection rule: keep rows with a non-empty ``doc_id`` and non-whitespace
        ``text``; stop at exactly SAMPLE_DOCUMENTS. No shuffle, no transformation.
        If the stream is unavailable the exception propagates -- no other config,
        split, or revision is ever substituted.
        """
        from datasets import load_dataset

        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = RAW_PATH.with_name(f".{RAW_PATH.name}.download.tmp")
        records = []
        started = time.perf_counter()
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
                    f"first {SAMPLE_DOCUMENTS} non-empty rows in deterministic streaming "
                    "source order; no shuffle or transformation"
                ),
                "requested_documents": SAMPLE_DOCUMENTS,
                "retained_documents": len(records),
                "characters": stats["characters"],
                "estimated_tokens": stats["estimated_tokens"],
                "token_estimate_formula": "ceil(non_whitespace_units * 1.35)",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "raw_path": str(RAW_PATH),
                "raw_sha256": sha256_path(RAW_PATH),
            }
            atomic_write_text(
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
        return read_raw_records(), manifest, message

    return (
        DATASET_CONFIG,
        DATASET_ID,
        DATASET_LICENSE,
        DATASET_REVISION,
        DATASET_SPLIT,
        JACCARD_THRESHOLD,
        LSH_BANDS,
        LSH_ROWS,
        MAX_BUCKET_SIZE,
        METRICS_PATH,
        MINHASH_SEED,
        MIN_TOKENS_FOR_MINHASH,
        NUM_HASHES,
        PIPELINE_VERSION,
        RESULTS_DIR,
        SAMPLE_DOCUMENTS,
        SHINGLE_K,
        atomic_write_text,
        cache_status,
        load_bounded_records,
        materialize_cache,
        record_stats,
        sha256_path,
    )


@app.cell(hide_code=True)
def diagram_toolkit():
    # Diagram toolkit. Hand-composed SVG, no plotting dependency: the output is
    # diffable, renders identically in the notebook and in static HTML export, and
    # says exactly what it was written to say. matplotlib/graphviz would produce
    # charts; these are explanatory diagrams.

    INK, MUTED, RULE = "#111111", "#6b7280", "#d1d5db"
    WASH, SURFACE = "#f8f9fa", "#ffffff"
    KEEP, DROP, MARK = "#0a7281", "#c2410c", "#f59e0b"
    MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"
    SANS = "Inter, system-ui, -apple-system, sans-serif"
    DEVA = "Noto Sans Devanagari, sans-serif"


    def xml_escape(text):
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )


    def arrow_defs(slug):
        """One shared arrow marker per SVG.

        Every id is slug-prefixed. All nine diagrams share one HTML document, so a
        bare id="arrow" would collide and silently kill arrowheads on all but the
        first diagram.
        """
        return (
            f'<defs><marker id="{slug}-ar" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0 0 L10 5 L0 10 z" fill="{MUTED}"/></marker></defs>'
        )


    def svg(slug, title, desc, height, body):
        return (
            f'<svg viewBox="0 0 900 {height}" width="100%" role="img" '
            f'aria-labelledby="{slug}-t {slug}-d" xmlns="http://www.w3.org/2000/svg">'
            f'<title id="{slug}-t">{xml_escape(title)}</title>'
            f'<desc id="{slug}-d">{xml_escape(desc)}</desc>'
            f"{arrow_defs(slug)}"
            f'<rect x="0" y="0" width="900" height="{height}" fill="{SURFACE}"/>'
            f"{body}"
            f"</svg>"
        )


    def box(x, y, w, h, fill=SURFACE, stroke=RULE, width=1, radius=6, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
        )


    def label(x, y, text, size=12, fill=INK, family=SANS, anchor="start", weight="normal"):
        return (
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">'
            f"{xml_escape(text)}</text>"
        )


    def arrow(slug, x1, y1, x2, y2, stroke=MUTED, dash=None, width=1.5):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}"{dash_attr} marker-end="url(#{slug}-ar)"/>'
        )


    def rule_line(x1, y1, x2, y2, stroke=RULE, dash=None, width=1):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}"{dash_attr}/>'
        )


    def doc_card(x, y, lines, tone=RULE, w=190, title=None, deva_index=None):
        """A document tile: optional title, then up to four short content lines."""
        h = 34 + 16 * len(lines) + (16 if title else 0)
        parts = [box(x, y, w, h, fill=WASH, stroke=tone, width=1.4)]
        cursor = y + 20
        if title:
            parts.append(label(x + 10, cursor, title, size=11, fill=tone, family=MONO, weight="bold"))
            cursor += 18
        for index, line in enumerate(lines):
            family = DEVA if (deva_index is not None and index in deva_index) else MONO
            parts.append(label(x + 10, cursor, line, size=11, fill=INK, family=family))
            cursor += 16
        return "".join(parts), h


    def bit_cell(x, y, value, size=22, on_fill=KEEP, off_fill=SURFACE):
        filled = str(value) == "1"
        parts = [
            box(
                x,
                y,
                size,
                size,
                fill=on_fill if filled else off_fill,
                stroke=RULE if not filled else on_fill,
                radius=3,
            ),
            label(
                x + size / 2,
                y + size - 6,
                str(value),
                size=11,
                fill=SURFACE if filled else MUTED,
                family=MONO,
                anchor="middle",
            ),
        ]
        return "".join(parts)


    def caption(y, text, height_width=900):
        return label(
            height_width / 2, y, text, size=12, fill=MUTED, family=SANS, anchor="middle"
        )


    def panel_title(x, y, text):
        return label(x, y, text, size=13, fill=INK, family=SANS, weight="bold")

    return (
        DEVA,
        DROP,
        INK,
        KEEP,
        MARK,
        MONO,
        MUTED,
        RULE,
        SANS,
        SURFACE,
        WASH,
        arrow,
        bit_cell,
        box,
        caption,
        doc_card,
        label,
        panel_title,
        rule_line,
        svg,
    )


@app.cell(hide_code=True)
def technique_diagrams(
    DEVA,
    DROP,
    INK,
    KEEP,
    MARK,
    MONO,
    MUTED,
    RULE,
    SANS,
    SURFACE,
    WASH,
    arrow,
    bit_cell,
    box,
    caption,
    doc_card,
    label,
    panel_title,
    rule_line,
    svg,
):
    # Nine hand-composed SVG diagrams, one per technique, built from the shared
    # diagram_toolkit helpers. DIAGRAMS is the single owning binding; each technique
    # cell renders mo.Html(DIAGRAMS[slug]). Regenerable by re-running this one cell.


    def _build_exact_hash():
        s = "exact-hash"
        body = "".join([
            panel_title(30, 34, "1. Normalize (whitespace collapse + casefold only)"),
            doc_card(30, 50, ["\u092f\u0939  \u092a\u0902\u0915\u094d\u0924\u093f  \u0939\u0948", "Doc A   two spaces"], tone=RULE, w=200, title="doc A", deva_index={0})[0],
            doc_card(30, 160, ["\u092f\u0939 \u092a\u0902\u0915\u094d\u0924\u093f \u0939\u0948", "DOC A   one space"], tone=RULE, w=200, title="doc B", deva_index={0})[0],
            arrow(s, 240, 90, 300, 90), arrow(s, 240, 200, 300, 200),
            box(305, 62, 150, 56, fill=WASH), label(380, 86, "normalize()", size=12, family=MONO, anchor="middle"),
            label(380, 104, "\u2192 same string", size=11, fill=MUTED, anchor="middle"),
            box(305, 172, 150, 56, fill=WASH), label(380, 196, "normalize()", size=12, family=MONO, anchor="middle"),
            label(380, 214, "\u2192 same string", size=11, fill=MUTED, anchor="middle"),
            arrow(s, 460, 90, 520, 118), arrow(s, 460, 200, 520, 172),
            box(525, 118, 175, 54, fill=SURFACE, stroke=KEEP, width=1.6),
            label(612, 140, "sha256 = 7f3a\u2026", size=11, family=MONO, anchor="middle", fill=KEEP),
            label(612, 158, "one bucket", size=11, family=SANS, anchor="middle", fill=MUTED),
            arrow(s, 705, 132, 762, 112), arrow(s, 705, 158, 762, 190),
            box(766, 86, 108, 48, fill=SURFACE, stroke=KEEP, width=1.6),
            label(820, 106, "KEEP", size=12, anchor="middle", fill=KEEP, weight="bold"),
            label(820, 124, "first seen", size=10, anchor="middle", fill=MUTED),
            box(766, 168, 108, 48, fill=SURFACE, stroke=DROP, width=1.6, dash="4 3"),
            label(820, 188, "DROP", size=12, anchor="middle", fill=DROP, weight="bold"),
            label(820, 206, "duplicate", size=10, anchor="middle", fill=MUTED),
            rule_line(30, 244, 870, 244),
            doc_card(30, 256, ["\u092f\u0939 \u0935\u093e\u0915\u094d\u092f \u0939\u0948"], tone=MARK, w=200, title="doc C: one word differs", deva_index={0})[0],
            arrow(s, 240, 282, 520, 282),
            box(525, 262, 175, 40, fill=SURFACE, stroke=MARK, width=1.6),
            label(612, 287, "sha256 = c19b\u2026", size=11, family=MONO, anchor="middle", fill=MARK),
            box(766, 262, 108, 40, fill=SURFACE, stroke=KEEP, width=1.6),
            label(820, 287, "KEEP", size=12, anchor="middle", fill=KEEP, weight="bold"),
            caption(346, "Exact hashing is all-or-nothing: one differing word is as different as a differing document."),
        ])
        return svg(s, "Exact document hashing collapses whitespace and case but nothing else",
                   "Two documents differing only in whitespace and case normalize to the same string and yield identical SHA-256 digests, so the second is dropped; a document differing by one word yields a different digest and is kept.",
                   364, body)

    def _build_bloom_filter():
        s = "bloom-filter"
        N = 24; x0 = 60; cw = 32; y = 150
        cells = []
        setbits = {2, 5, 7, 9, 12, 14, 17, 19, 21}
        for i in range(N):
            cells.append(bit_cell(x0 + i * cw, y, "1" if i in setbits else "0", size=26, on_fill=RULE, off_fill=SURFACE))

        def arc(cx1, cx2, up, color):
            top = y - 44 if up else y + 26 + 44
            mid = (cx1 + cx2) / 2
            cy = y - 6 if up else y + 26 + 6
            return (f'<path d="M{cx1} {cy} Q{mid} {top} {cx2} {cy}" fill="none" '
                    f'stroke="{color}" stroke-width="1.6" marker-end="url(#{s}-ar)"/>')

        def cx(i):
            return x0 + i * cw + cw / 2

        body = "".join([
            panel_title(30, 34, "2. Bloom filter: k hash probes per document over a shared bit array"),
            label(30, 100, "doc A \u2192 h1,h2,h3 set 3 bits", size=12, fill=KEEP, family=SANS),
            arc(cx(2), cx(5), True, KEEP), arc(cx(5), cx(9), True, KEEP), arc(cx(9), cx(14), True, KEEP),
            "".join(cells),
            label(30, y + 52, "bit array (m = 24)", size=11, fill=MUTED, family=MONO),
            label(30, y + 95, "doc B \u2192 h1,h2,h3 all already set \u2192 reported duplicate", size=12, fill=DROP, family=SANS),
            arc(cx(2), cx(7), False, DROP), arc(cx(7), cx(12), False, DROP), arc(cx(12), cx(17), False, DROP),
            rule_line(30, y + 120, 870, y + 120),
            label(30, y + 150, "FALSE POSITIVE: doc C's 3 bits were each set by different earlier docs", size=12, fill=DROP, family=SANS, weight="bold"),
            label(30, y + 170, "\u2192 a unique document is dropped, though it was never seen", size=11, fill=MUTED),
            box(600, y + 134, 274, 52, fill=WASH, stroke=DROP, width=1.4, dash="4 3"),
            label(737, y + 156, "no false negatives ever;", size=11, anchor="middle", fill=INK, family=SANS),
            label(737, y + 174, "false positives drop unique docs", size=11, anchor="middle", fill=DROP, family=SANS),
            caption(y + 205, "Membership is probabilistic in one direction only: a 'not present' answer is always true."),
        ])
        return svg(s, "A Bloom filter answers set membership with false positives but never false negatives",
                   "Each document sets k bits via k hash functions in a shared bit array; a later document whose k bits are all set is reported as a duplicate. A unique document can collide with bits set by unrelated documents, a false positive that drops it.",
                   380, body)


    def _build_shingling_jaccard():
        s = "shingling-jaccard"
        words = ["\u0917\u094d\u0930\u093e\u092e", "\u0915\u093e", "\u0928\u093e\u092e", "\u0914\u0930", "\u0915\u094d\u0930\u092e\u093e\u0902\u0915", "\u0926\u0930\u094d\u091c", "\u0939\u0948"]
        wx = 40; ww = 96; wy = 70
        wordboxes = []
        for i, w in enumerate(words):
            wordboxes.append(box(wx + i * (ww + 6), wy, ww, 34, fill=WASH))
            wordboxes.append(label(wx + i * (ww + 6) + ww / 2, wy + 22, w, size=13, family=DEVA, anchor="middle"))

        def bracket(start, yy, color, lbl):
            x1 = wx + start * (ww + 6) - 3
            x2 = wx + (start + 5) * (ww + 6) - 6 + 3
            return (f'<path d="M{x1} {yy} L{x1} {yy+10} L{x2} {yy+10} L{x2} {yy}" fill="none" '
                    f'stroke="{color}" stroke-width="1.8"/>'
                    + label((x1 + x2) / 2, yy + 26, lbl, size=10, fill=color, anchor="middle", family=MONO))

        body = "".join([
            panel_title(30, 34, "3. Word 5-gram shingling, then Jaccard resemblance of the two shingle sets"),
            "".join(wordboxes),
            bracket(0, 108, KEEP, "shingle 1"),
            bracket(1, 138, MARK, "shingle 2"),
            bracket(2, 168, DROP, "shingle 3"),
            label(40, 210, "each 5-word window is one shingle; a document becomes its set of shingles", size=11, fill=MUTED),
            rule_line(30, 228, 870, 228),
            f'<circle cx="360" cy="330" r="86" fill="{KEEP}" fill-opacity="0.12" stroke="{KEEP}" stroke-width="1.6"/>',
            f'<circle cx="470" cy="330" r="86" fill="{DROP}" fill-opacity="0.12" stroke="{DROP}" stroke-width="1.6"/>',
            label(315, 335, "|A|", size=12, anchor="middle", fill=KEEP, family=MONO),
            label(515, 335, "|B|", size=12, anchor="middle", fill=DROP, family=MONO),
            label(415, 325, "\u2229", size=16, anchor="middle", fill=INK, family=MONO),
            label(415, 345, "41", size=12, anchor="middle", fill=INK, family=MONO),
            box(600, 270, 274, 120, fill=WASH),
            label(620, 300, "|A \u2229 B| = 41", size=13, family=MONO, fill=INK),
            label(620, 326, "|A \u222a B| = 99", size=13, family=MONO, fill=INK),
            rule_line(620, 340, 854, 340),
            label(620, 366, "J(A,B) = 41 / 99 = 0.414", size=13, family=MONO, fill=KEEP, weight="bold"),
            caption(438, "Resemblance is set overlap: shingle order is irrelevant, only which windows two documents share."),
        ])
        return svg(s, "Shingling turns a document into a set of word 5-grams; Jaccard measures set overlap",
                   "A five-word window slides across a Hindi sentence producing shingles; two documents' shingle sets are compared by Jaccard resemblance, the intersection size over the union size.",
                   456, body)

    def _build_minhash():
        s = "minhash"
        rows = [("a\u2081\u00b7x+b\u2081", "0731", KEEP), ("a\u2082\u00b7x+b\u2082", "1180", KEEP),
                ("a\u2083\u00b7x+b\u2083", "0345", DROP), ("a\u2084\u00b7x+b\u2084", "0912", KEEP)]
        rowsvg = []
        ry = 120
        for i, (lbl, mn, color) in enumerate(rows):
            yy = ry + i * 34
            rowsvg.append(box(330, yy, 150, 28, fill=SURFACE))
            rowsvg.append(label(340, yy + 19, lbl, size=12, family=MONO, fill=INK))
            rowsvg.append(box(492, yy, 70, 28, fill=WASH, stroke=color, width=1.6))
            rowsvg.append(label(527, yy + 19, mn, size=12, family=MONO, anchor="middle", fill=color))
        body = "".join([
            panel_title(30, 34, "4. MinHash: the minimum hash under each permutation, compared slot by slot"),
            box(40, 90, 250, 150, fill=WASH, stroke=KEEP, width=1.4),
            label(60, 116, "shingle-hash set of doc A", size=12, family=SANS, fill=KEEP),
            label(60, 142, "h(s\u2081)=0731  h(s\u2082)=4820", size=11, family=MONO, fill=INK),
            label(60, 162, "h(s\u2083)=1180  h(s\u2084)=9107", size=11, family=MONO, fill=INK),
            label(60, 182, "\u22ee", size=11, family=MONO, fill=MUTED),
            label(60, 214, "|A| = 1,204 shingles", size=12, family=MONO, fill=INK, weight="bold"),
            arrow(s, 292, 165, 328, 165),
            label(330, 110, "permutations (min per column)", size=11, family=SANS, fill=MUTED),
            "".join(rowsvg),
            arrow(s, 566, 200, 604, 200),
            box(610, 120, 260, 130, fill=WASH, stroke=KEEP, width=1.4),
            label(630, 148, "slots where A and B agree", size=12, family=SANS, fill=KEEP),
            label(630, 178, "104 / 112", size=18, family=MONO, fill=KEEP, weight="bold"),
            label(630, 204, "J\u0302 = 0.929", size=13, family=MONO, fill=INK),
            label(630, 226, "true J = 0.941", size=13, family=MONO, fill=MUTED),
            label(405, 262, "\u22ee   112 permutations in total", size=11, family=MONO, anchor="middle", fill=MUTED),
            label(451, 280, "permutation 3 (0345) is the one row where A \u2260 B", size=11, family=SANS, anchor="middle", fill=DROP),
            box(40, 298, 830, 60, fill=SURFACE, stroke=RULE),
            label(60, 322, "P[ min(h(A)) = min(h(B)) ] = J(A, B)", size=14, family=MONO, fill=INK, weight="bold"),
            label(60, 344, "the fraction of agreeing rows is an unbiased estimate of Jaccard; measured mean |est \u2212 true| = 0.019", size=11, family=SANS, fill=MUTED),
            caption(388, "One disagreeing row is expected: the signature is a probabilistic estimator, not an exact match."),
        ])
        return svg(s, "MinHash estimates Jaccard by the fraction of permutations whose minimum agrees",
                   "For each of 112 permutations the minimum hashed shingle is recorded; the fraction of permutations where two documents' minima agree is an unbiased estimate of their Jaccard similarity.",
                   406, body)

    def _build_lsh_banding():
        s = "lsh-banding"
        pts = []
        for i in range(101):
            sv = i * 0.01
            P = 1 - (1 - sv ** 8) ** 14
            pts.append((round(520 + 340 * sv, 2), round(380 - 130 * P, 2)))
        dpath = "M" + " L".join(f"{x} {y}" for x, y in pts)
        thr = (1 / 14) ** (1 / 8)
        thr_x = round(520 + 340 * thr, 1)
        dotP = 1 - (1 - 0.75 ** 8) ** 14
        dot_x = round(520 + 340 * 0.75, 1); dot_y = round(380 - 130 * dotP, 1)
        bx = 40; bw = 30; by = 116; hot = 3
        strip = []
        for b in range(14):
            color = MARK if b == hot else RULE
            fillc = "#fdebd0" if b == hot else SURFACE
            strip.append(box(bx + b * (bw + 3), by, bw, 40, fill=fillc, stroke=color, width=1.8 if b == hot else 1))
            strip.append(label(bx + b * (bw + 3) + bw / 2, by + 56, f"b{b}", size=9, family=MONO, anchor="middle", fill=color))
        hot_cx = bx + hot * (bw + 3) + bw / 2
        body = "".join([
            panel_title(30, 34, "5. LSH banding: 112 slots split into 14 bands of 8; any one band collision \u2192 candidate"),
            label(40, 74, "signature of docs A and B, partitioned into 14 bands", size=11, family=SANS, fill=MUTED),
            label(hot_cx, 100, "\u25bc band b3 identical", size=9, family=SANS, anchor="middle", fill=MARK),
            "".join(strip),
            box(40, 202, 430, 70, fill=WASH, stroke=MARK, width=1.4),
            label(60, 228, "band b3 hashes to the same bucket for A and B", size=12, family=SANS, fill=INK),
            label(60, 252, "\u2192 (A, B) becomes a candidate pair; other bands differ", size=11, family=SANS, fill=MARK),
            label(40, 302, "P(candidate) = 1 \u2212 (1 \u2212 s\u2078)\u00b9\u2074", size=13, family=MONO, fill=INK, weight="bold"),
            label(40, 324, "threshold (1/14)^(1/8) = 0.719", size=12, family=MONO, fill=MUTED),
            label(40, 344, "at s = 0.75, P = 0.772", size=12, family=MONO, fill=KEEP),
            box(520, 250, 340, 130, fill=SURFACE, stroke=RULE),
            rule_line(520, 380, 860, 380, stroke=MUTED),
            rule_line(520, 250, 520, 380, stroke=MUTED),
            label(514, 258, "P=1", size=9, family=MONO, anchor="end", fill=MUTED),
            label(514, 382, "0", size=9, family=MONO, anchor="end", fill=MUTED),
            label(520, 394, "s=0", size=9, family=MONO, anchor="middle", fill=MUTED),
            label(860, 394, "s=1", size=9, family=MONO, anchor="middle", fill=MUTED),
            f'<path d="{dpath}" fill="none" stroke="{KEEP}" stroke-width="2"/>',
            rule_line(thr_x, 250, thr_x, 380, stroke=DROP, dash="4 3"),
            label(thr_x, 246, "0.719", size=9, family=MONO, anchor="middle", fill=DROP),
            f'<circle cx="{dot_x}" cy="{dot_y}" r="4" fill="{MARK}"/>',
            label(dot_x + 6, dot_y - 6, "(0.75, 0.772)", size=9, family=MONO, fill=MARK),
            caption(420, "The band/row split is a tunable S-curve: it sharply separates similar from dissimilar around the threshold."),
        ])
        return svg(s, "LSH banding makes near-duplicate probability an S-curve in similarity",
                   "The 112-slot signature is split into 14 bands of 8 rows; a pair is a candidate if any band matches. The probability of becoming a candidate follows 1 minus (1 minus s to the 8) to the 14, an S-curve crossing near similarity 0.72.",
                   440, body)

    def _build_simhash():
        s = "simhash"

        def bits(x0, y0, seq, color):
            out = []
            for i, ch in enumerate(seq):
                out.append(bit_cell(x0 + i * 30, y0, ch, size=26, on_fill=color, off_fill=SURFACE))
            return "".join(out)

        body = "".join([
            panel_title(30, 34, "6. SimHash: sign of each random-hyperplane projection forms the fingerprint"),
            box(40, 80, 230, 120, fill=WASH, stroke=KEEP, width=1.4),
            label(60, 106, "weighted n-gram vector", size=12, family=SANS, fill=KEEP),
            label(60, 130, "\u0917\u094d\u0930\u093e\u092e:3  \u0928\u093e\u092e:2", size=12, family=DEVA, fill=INK),
            label(60, 152, "\u0915\u094d\u0930\u092e\u093e\u0902\u0915:2  \u0926\u0930\u094d\u091c:1", size=12, family=DEVA, fill=INK),
            label(60, 180, "term weights, not sets", size=10, family=SANS, fill=MUTED),
            arrow(s, 272, 140, 320, 140),
            box(325, 80, 210, 160, fill=SURFACE, stroke=RULE),
            label(430, 100, "4 random hyperplanes", size=11, family=SANS, anchor="middle", fill=MUTED),
            rule_line(345, 200, 515, 120, stroke=MARK), rule_line(345, 140, 515, 200, stroke=MARK),
            rule_line(360, 90, 500, 230, stroke=MARK), rule_line(500, 90, 360, 230, stroke=MARK),
            f'<circle cx="430" cy="160" r="5" fill="{KEEP}"/>',
            label(430, 232, "sign per projection", size=10, family=SANS, anchor="middle", fill=MUTED),
            arrow(s, 537, 140, 585, 140),
            label(600, 104, "doc A fingerprint", size=11, family=SANS, fill=INK),
            bits(600, 116, "1011", KEEP),
            label(600, 176, "doc B fingerprint", size=11, family=SANS, fill=INK),
            bits(600, 188, "1010", DROP),
            label(730, 236, "Hamming distance = 1 \u2192 near-duplicate", size=11, family=SANS, anchor="middle", fill=DROP),
            box(40, 262, 400, 66, fill=WASH),
            label(60, 288, "MinHash \u2192 Jaccard over sets", size=12, family=MONO, fill=KEEP),
            label(60, 312, "SimHash \u2192 cosine over weighted vectors", size=12, family=MONO, fill=DROP),
            box(470, 262, 400, 66, fill=SURFACE, stroke=RULE),
            label(490, 288, "Google production (Manku 2007):", size=12, family=SANS, fill=INK),
            label(490, 312, "64-bit fingerprint, near-duplicate iff Hamming \u2264 3", size=11, family=MONO, fill=MUTED),
            caption(360, "SimHash preserves cosine similarity of weighted term vectors; close vectors yield fingerprints a few bits apart."),
        ])
        return svg(s, "SimHash encodes weighted-vector cosine similarity into a small Hamming distance",
                   "A weighted n-gram vector is projected onto random hyperplanes; the sign of each projection is one fingerprint bit. Similar documents differ in only a few bits, so near-duplicates have small Hamming distance.",
                   378, body)

    def _build_suffix_array():
        s = "suffix-array"
        suffixes = ["\u0917\u094d\u0930\u093e\u092e \u0915\u093e \u0928\u093e\u092e $\u2082 ...",
                    "\u0917\u094d\u0930\u093e\u092e \u0915\u093e \u0928\u093e\u092e $\u2081 ...",
                    "\u0915\u093e \u0928\u093e\u092e $\u2082 ...",
                    "\u0928\u093e\u092e $\u2081 ..."]
        rows = []
        ry = 120
        for i, suf in enumerate(suffixes):
            hot = i in (0, 1)
            color = MARK if hot else RULE
            rows.append(box(430, ry + i * 40, 300, 30, fill=("#fdebd0" if hot else SURFACE), stroke=color, width=1.6 if hot else 1))
            rows.append(label(444, ry + i * 40 + 20, suf, size=12, family=DEVA, fill=INK))
            rows.append(label(415, ry + i * 40 + 20, f"SA[{i}]", size=9, family=MONO, anchor="end", fill=MUTED))
        body = "".join([
            panel_title(30, 34, "7. Suffix array: sorting brings repeated substrings into adjacent rows"),
            box(40, 110, 330, 90, fill=WASH, stroke=KEEP, width=1.4),
            label(60, 134, "concatenated corpus with sentinels", size=11, family=SANS, fill=KEEP),
            label(60, 160, "\u2026\u0917\u094d\u0930\u093e\u092e \u0915\u093e \u0928\u093e\u092e $\u2081 \u0917\u094d\u0930\u093e\u092e \u0915\u093e \u0928\u093e\u092e $\u2082\u2026", size=11, family=DEVA, fill=INK),
            label(60, 186, "documents joined; $ separates them", size=10, family=SANS, fill=MUTED),
            arrow(s, 372, 155, 412, 155),
            label(430, 108, "sorted suffix array", size=11, family=SANS, fill=MUTED),
            "".join(rows),
            label(745, ry + 20, "\u2190 adjacent rows share", size=10, family=SANS, fill=MARK),
            label(745, ry + 52, "   a long common prefix", size=10, family=SANS, fill=MARK),
            box(40, 300, 400, 60, fill=WASH, stroke=DROP, width=1.4, dash="4 3"),
            label(60, 324, "cut when LCP \u2265 50 BPE tokens", size=12, family=MONO, fill=DROP),
            label(60, 346, "excised span inside a document \u2014 not the whole document", size=11, family=SANS, fill=MUTED),
            box(470, 300, 400, 60, fill=SURFACE, stroke=RULE),
            label(490, 322, "doc with a shaded excised span:", size=11, family=SANS, fill=INK),
            f'<rect x="490" y="332" width="120" height="16" fill="{WASH}" stroke="{RULE}"/>',
            f'<rect x="612" y="332" width="90" height="16" fill="#fdebd0" stroke="{MARK}"/>',
            f'<rect x="704" y="332" width="150" height="16" fill="{WASH}" stroke="{RULE}"/>',
            label(657, 344, "removed", size=9, family=MONO, anchor="middle", fill=MARK),
            caption(388, "The unit is a repeated span, not a document: sorting makes long repeats adjacent so a linear scan finds them."),
        ])
        return svg(s, "Suffix-array exact substring dedup removes long repeated spans inside documents",
                   "The corpus is concatenated with sentinels and its suffixes sorted; adjacent suffixes sharing a long common prefix mark a repeated span, excised when the shared prefix reaches the token threshold.",
                   398, body)


    def _build_semdedup():
        s = "semdedup"
        parts = [
            panel_title(30, 34, "8. SemDeDup: cluster embeddings, then collapse each dense cosine neighborhood"),
            label(30, 62, "embedding space (after k-means)", size=11, family=SANS, fill=MUTED),
        ]
        c1 = (250, 220)
        parts.append(f'<circle cx="{c1[0]}" cy="{c1[1]}" r="110" fill="{KEEP}" fill-opacity="0.07" stroke="{RULE}" stroke-dasharray="4 3"/>')
        parts.append(label(c1[0], 100, "cluster 1 (dense)", size=11, anchor="middle", fill=MUTED))
        parts.append(f'<circle cx="{c1[0]}" cy="{c1[1]}" r="4" fill="{INK}"/>')
        parts.append(label(c1[0] - 10, c1[1] + 16, "centroid", size=9, family=SANS, anchor="middle", fill=MUTED))
        parts.append(f'<circle cx="290" cy="190" r="42" fill="none" stroke="{DROP}" stroke-dasharray="3 3"/>')
        for (px, py, col) in [(300, 165, DROP), (278, 208, DROP), (315, 200, DROP), (330, 150, KEEP)]:
            parts.append(f'<circle cx="{px}" cy="{py}" r="6" fill="{col}"/>')
        parts.append(label(348, 150, "kept: farthest from centroid", size=10, family=SANS, fill=KEEP))
        parts.append(label(348, 210, "dropped (within cosine radius)", size=10, family=SANS, fill=DROP))
        parts.append(f'<circle cx="205" cy="180" r="6" fill="{MARK}"/>')
        parts.append(f'<circle cx="188" cy="196" r="6" fill="{MARK}"/>')
        parts.append(label(150, 300, "\u0917\u094d\u0930\u093e\u092e \u2248 \"the village\": no shared words,", size=10, family=DEVA, anchor="middle", fill=MARK))
        parts.append(label(150, 316, "adjacent in embedding space", size=10, family=SANS, anchor="middle", fill=MARK))
        parts.append(label(150, 332, "\u2014 MinHash cannot catch this", size=10, family=SANS, anchor="middle", fill=MARK))
        c2 = (660, 210)
        parts.append(f'<circle cx="{c2[0]}" cy="{c2[1]}" r="88" fill="none" stroke="{DROP}" stroke-width="1.6" stroke-dasharray="6 4"/>')
        parts.append(label(c2[0], 100, "cluster 2 (sparse, under-represented)", size=11, anchor="middle", fill=DROP))
        for (px, py) in [(635, 195), (680, 232), (700, 178)]:
            parts.append(f'<circle cx="{px}" cy="{py}" r="6" fill="{MARK}"/>')
        parts.append(label(c2[0], 320, "over-pruning here worsens fairness (FairDeDup)", size=10, anchor="middle", fill=DROP))
        parts.append(box(30, 344, 840, 34, fill=WASH))
        parts.append(label(50, 366, "duplicate when cos \u2265 1 \u2212 \u03b5; keep the member FARTHEST from the centroid (contrast: union-find keeps the longest text)", size=11, family=SANS, fill=INK))
        parts.append(caption(404, "Semantic dedup removes near-synonymous documents that share no surface tokens \u2014 exactly what MinHash misses."))
        return svg(s, "SemDeDup collapses embedding-space neighborhoods, keeping the least central member",
                   "Documents are embedded and k-means clustered; within a cluster, points inside a cosine radius are duplicates and all but the member farthest from the centroid are dropped. Over-pruning sparse clusters can worsen fairness.",
                   424, "".join(parts))

    def _build_dedup_scope():
        s = "dedup-scope"
        parts = [panel_title(30, 30, "9. Deduplication scope: local, global, and why global is not automatically better")]
        parts.append(label(30, 58, "Panel 1 \u2014 local (per-shard) dedup", size=11, family=SANS, fill=INK, weight="bold"))
        sx = 40
        for shard in range(4):
            x = sx + shard * 100
            parts.append(box(x, 70, 84, 118, fill=WASH, stroke=RULE))
            parts.append(label(x + 42, 86, f"shard {shard+1}", size=9, family=MONO, anchor="middle", fill=MUTED))
            for t in range(5):
                col = DROP if t == 3 else KEEP
                parts.append(f'<rect x="{x+12}" y="{96+t*16}" width="60" height="12" rx="2" fill="{col}" fill-opacity="0.18" stroke="{col}"/>')
            parts.append(label(x + 42, 202, "locally clean", size=8, family=SANS, anchor="middle", fill=KEEP))
        parts.append(label(470, 58, "Panel 2 \u2014 global merge across shards", size=11, family=SANS, fill=INK, weight="bold"))
        gx = 480
        tile_centers = []
        for shard in range(4):
            x = gx + shard * 95
            parts.append(box(x, 70, 80, 118, fill=WASH, stroke=RULE))
            for t in range(5):
                parts.append(f'<rect x="{x+10}" y="{96+t*16}" width="56" height="12" rx="2" fill="{KEEP}" fill-opacity="0.12" stroke="{RULE}"/>')
            tile_centers.append((x + 38, 108))
        for a, b in [(0, 1), (1, 3), (0, 2)]:
            (x1, y1), (x2, y2) = tile_centers[a], tile_centers[b]
            parts.append(f'<path d="M{x1} {y1} Q{(x1+x2)/2} {y1-34} {x2} {y2}" fill="none" stroke="{DROP}" stroke-width="1.4" stroke-dasharray="4 3"/>')
        parts.append(label(670, 205, "no local pass can observe these", size=9, family=SANS, anchor="middle", fill=DROP))
        parts.append(rule_line(30, 226, 870, 226))
        parts.append(label(30, 252, "Panel 3 \u2014 FineWeb: global was WORSE than per-snapshot", size=11, family=SANS, fill=INK, weight="bold"))
        parts.append(box(40, 264, 380, 108, fill=WASH, stroke=DROP, width=1.4))
        parts.append(label(60, 288, "global across 96 snapshots \u2192 ~4T tokens", size=12, family=MONO, fill=DROP))
        parts.append(label(60, 310, "ablations far below RefinedWeb", size=11, family=SANS, fill=MUTED))
        parts.append(label(60, 334, "crawl 2013-48: the 10% kept was WORSE", size=11, family=SANS, fill=DROP))
        parts.append(label(60, 354, "than the 90% removed (ads, keyword lists)", size=11, family=SANS, fill=MUTED))
        parts.append(box(460, 264, 380, 108, fill=WASH, stroke=KEEP, width=1.4))
        parts.append(label(480, 288, "per-snapshot dedup \u2192 ~20T tokens", size=12, family=MONO, fill=KEEP))
        parts.append(label(480, 310, "matches RefinedWeb quality", size=11, family=SANS, fill=MUTED))
        parts.append(label(480, 334, "delete the huge clusters in every crawl;", size=11, family=SANS, fill=INK))
        parts.append(label(480, 354, "stripping small clusters removes good text", size=11, family=SANS, fill=MUTED))
        parts.append(label(30, 392, "Memory wall: classic MinHashLSH index \u2248 23 TB at 5B docs, \u2248 46 TB at 10B \u2014 LSHBloom addresses this; FineWeb's objection was quality.", size=10, family=SANS, fill=MUTED))
        parts.append(caption(410, "Scope is a quality decision, not only a memory constraint: more global dedup is not unconditionally better."))
        return svg(s, "Deduplication scope is a quality decision, not only a systems constraint",
                   "Local dedup cleans each shard but cannot see cross-shard duplicates; global dedup can, but FineWeb found global dedup across 96 snapshots produced lower-quality survivors than per-snapshot dedup, so scope is a quality decision.",
                   418, "".join(parts))


    DIAGRAMS = {
        "exact-hash": _build_exact_hash(),
        "bloom-filter": _build_bloom_filter(),
        "shingling-jaccard": _build_shingling_jaccard(),
        "minhash": _build_minhash(),
        "lsh-banding": _build_lsh_banding(),
        "simhash": _build_simhash(),
        "suffix-array": _build_suffix_array(),
        "semdedup": _build_semdedup(),
        "dedup-scope": _build_dedup_scope(),
    }
    return (DIAGRAMS,)


@app.cell(hide_code=True)
def catalogue_intro(mo):
    mo.md(r"""
    ## The catalogue: nine techniques, one executed

    The nine cells below share one structure so the section reads as a survey:
    name and execution status, what it is, the diagram, the internal mechanics,
    **where it has actually been used** (named projects with their published
    parameters), when to use it, pros and cons, and inline sources. Every
    non-executed technique also states the concrete resource that blocks running
    it here, so the catalogue stays honest rather than aspirational.

    Two techniques are executed end-to-end on the cache: **exact document
    hashing** and the **MinHash / LSH / exact-Jaccard** near-duplicate pipeline
    (techniques 1, 3, 4, 5 form that one pipeline). The other five are documented.

    The diagrams are hand-composed SVG built by the `diagram_toolkit` helpers with
    no plotting dependency, so they are diffable in version control and render
    identically in the notebook and in static HTML export.
    """)
    return


@app.cell(hide_code=True)
def technique_exact_hash(DIAGRAMS, mo):
    technique_1 = mo.vstack([
        mo.md(
            r"""
    ### 1. Exact document hashing &mdash; Executed on this cache

    **Status:** Executed. This is the first stage of the run: normalize, hash,
    keep the first occurrence of each digest.

    **What it is.** Reduce each document to a fixed-length digest of a normalized
    copy of its text, then treat two documents as identical when their digests
    match. It catches byte-for-byte repeats and repeats that differ only in
    whitespace and letter case &mdash; nothing more.

    **How it works internally.** Build a dictionary from
    `sha256(normalize(text))` to the first `doc_id` that produced it. The first
    occurrence in source order wins; every later collision is a duplicate.
    Normalization here is whitespace collapse plus `casefold()` only. One pass,
    O(n) time, O(distinct documents) memory; a 256-bit digest makes accidental
    collision negligible.
    """
        ),
        mo.Html(DIAGRAMS["exact-hash"]),
        mo.md(
            r"""
    **Where it has been used.**

    | Project | Configuration | Source |
    |---|---|---|
    | C4 / T5 | Discarded all but one of any three-sentence span occurring more than once | [arXiv:1910.10683](https://arxiv.org/abs/1910.10683) §2.2 |
    | Dolma | Exact dedup at three granularities, Bloom-backed: exact URL removed 53.2% of docs across 25 CC snapshots; exact document text a further 14.9%; exact paragraph tagged 18.7% of the remainder | [arXiv:2402.00159](https://arxiv.org/abs/2402.00159) §3.1.4 |
    | CCNet | First 64 bits of SHA-1 over a normalized paragraph (lowercase, digits&rarr;0, strip Unicode punctuation/accents); run before language ID; reused by LLaMA and RedPajama web paths | [arXiv:1911.00359](https://arxiv.org/abs/1911.00359), [facebookresearch/cc_net](https://github.com/facebookresearch/cc_net) |
    | The Stack | Exact identity by git blob hash: 51.76B files &rarr; 5.28B unique | [arXiv:2211.15533](https://arxiv.org/abs/2211.15533) §3.1 |

    SHA-256-of-NFKC is **not** a universal default: projects differ in hash
    function, unit (URL / paragraph / document / blob), and normalization. This
    notebook's choice &mdash; SHA-256 over a whitespace-and-case-normalized document
    &mdash; is its own, chosen so no Devanagari-altering normalization runs.

    **When to use it.** Always, and first. It is the cheapest stage and removes
    the easiest duplicates before any expensive near-duplicate work, shrinking the
    input to every later stage.

    | Pros | Cons |
    |---|---|
    | Lossless for true exact repeats | Blind to a one-word or one-character change |
    | One linear pass, trivial memory | Result depends entirely on the normalization choice |
    | Deterministic and fully auditable | Cannot express "90% similar" |

    **Source:** [C4](https://arxiv.org/abs/1910.10683) &middot; [Dolma](https://arxiv.org/abs/2402.00159) &middot; [CCNet](https://arxiv.org/abs/1911.00359) &middot; [The Stack](https://arxiv.org/abs/2211.15533)
    """
        ),
    ])
    technique_1
    return


@app.cell(hide_code=True)
def technique_bloom(DIAGRAMS, mo):
    technique_2 = mo.vstack([
        mo.md(
            r"""
    ### 2. Bloom-filter approximate-exact dedup &mdash; Documented; not executed

    **Status:** Documented; not executed. **Blocked:** a Bloom filter earns its
    keep only when the exact hash set no longer fits in memory &mdash; roughly
    10&#8312; or more documents. At 20,000 an exact Python `set` is smaller,
    faster, and lossless, so introducing a false-positive risk here would be
    strictly worse.

    **What it is.** A Bloom filter is a bit array plus *k* hash functions used as
    a probabilistic set. It answers "have I seen this document before?" in
    constant time and constant memory per element, at the cost of a bounded
    false-positive rate.

    **How it works internally.** To insert, hash the item *k* ways and set those
    *k* bits. To query, check whether all *k* bits are set. A membership miss is
    always truthful (**no false negatives**), but *k* bits can be set by unrelated
    earlier insertions, so a membership hit can be wrong (**false positive**). The
    false-positive rate is approximately `(1 - e^{-kn/m})^k` for *n* items in *m*
    bits; larger *m* and tuned *k* lower it. In dedup, a false positive **drops a
    unique document** &mdash; an unrecoverable loss.
    """
        ),
        mo.Html(DIAGRAMS["bloom-filter"]),
        mo.md(
            r"""
    **Where it has been used.**

    | Project | Configuration | Source |
    |---|---|---|
    | Dolma | "Linear-time deduplication through the use of a Bloom filter" for all three web stages plus Reddit document level; Rust Bloom in the toolkit mixer. Production FPR / bit-size / hash-count are not stated in the paper; a *decontamination* config uses `desired_false_positive_rate: 1e-15` for 188,815 docs | [arXiv:2402.00159](https://arxiv.org/abs/2402.00159), [allenai/dolma](https://github.com/allenai/dolma) |
    | DataComp-LM | Compared a MinHash-in-a-suffix-array pipeline against Dolma-style near-duplicate Bloom filtering; the two were within 0.2 Core percentage points at the 7B-2x scale, and Bloom was chosen for DCLM-baseline because it "scales more easily to datasets surpassing 10TB" | [arXiv:2406.11794](https://arxiv.org/abs/2406.11794) §4.3 |

    The defining property to keep in mind: **no false negatives ever; false
    positives drop unique documents.** That asymmetry is the whole reason to
    prefer or avoid it in a given position.

    **When to use it.** At corpus scales where an exact hash set exceeds memory
    and a small, controllable rate of accidental unique-document loss is
    acceptable in exchange for constant memory.

    | Pros | Cons |
    |---|---|
    | Constant memory regardless of distinct count | False positives silently drop unique documents |
    | Constant-time insert and query | Cannot enumerate members or delete (standard variant) |
    | Never misses a true duplicate | FPR must be tuned to *n*, which must be estimated up front |

    **What would be required to run it here.** A corpus large enough that the
    exact hash set no longer fits in RAM (~10&#8312;+ documents), so the
    memory/accuracy trade actually favors the filter.

    **Source:** [Dolma](https://arxiv.org/abs/2402.00159) &middot; [DataComp-LM](https://arxiv.org/abs/2406.11794)
    """
        ),
    ])
    technique_2
    return


@app.cell(hide_code=True)
def technique_shingling(DIAGRAMS, mo):
    technique_3 = mo.vstack([
        mo.md(
            r"""
    ### 3. Shingling + exact Jaccard &mdash; Executed (verification pass)

    **Status:** Executed, as the verification pass over LSH candidates. Every
    candidate pair the MinHash/LSH stage proposes is confirmed by computing the
    **exact** Jaccard similarity of the two shingle sets; the estimator is never
    trusted on its own for the keep/drop decision.

    **What it is.** Represent a document as the set of its overlapping word
    *k*-grams ("shingles"), then measure resemblance as the Jaccard similarity of
    two such sets. Order is discarded; only which windows two documents share
    matters.

    **How it works internally.** Tokenize, lowercase, slide a *k*-word window (here
    *k* = 5) to produce a set of shingles per document. Jaccard is
    `|A &cap; B| / |A &cup; B|`. Computed directly it costs one set intersection
    per pair &mdash; fine for verifying a few hundred candidate pairs, but
    O(n&sup2;) sets if applied to all pairs, which is exactly why LSH exists to
    propose candidates first.
    """
        ),
        mo.Html(DIAGRAMS["shingling-jaccard"]),
        mo.md(
            r"""
    **Where it has been used.**

    | Project | Configuration | Source |
    |---|---|---|
    | Broder 1997 | Documents as word-shingle sets, resemblance = Jaccard, containment directional; deployed at AltaVista for syntactic near-duplicate clustering of web pages | [DOI 10.1109/SEQUEN.1997.666900](https://doi.org/10.1109/SEQUEN.1997.666900) |
    | FineWeb / RefinedWeb | Word 5-grams | [arXiv:2406.17557](https://arxiv.org/abs/2406.17557), [arXiv:2306.01116](https://arxiv.org/abs/2306.01116) |
    | Gopher / MassiveText | 13-grams | [arXiv:2112.11446](https://arxiv.org/abs/2112.11446) |
    | NeMo Curator | 24-character n-grams (default) | [NeMo Curator dedup docs](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication.md) |

    The **shingle unit** matters more than the algorithm: shorter shingles call
    more documents similar (higher recall, lower precision); longer shingles are
    stricter. This notebook uses word 5-grams to match FineWeb.

    **When to use it.** As the exact-similarity ground truth behind an approximate
    candidate generator, and whenever the resemblance definition itself must be
    inspectable rather than estimated.

    | Pros | Cons |
    |---|---|
    | Exact, interpretable similarity | O(n&sup2;) if used for candidate generation |
    | Order-independent; robust to reordering | Storing full shingle sets is memory-heavy |
    | Shingle size is a clear precision/recall dial | Choice of *k* and tokenization strongly shifts results |

    **Source:** [Broder 1997](https://doi.org/10.1109/SEQUEN.1997.666900) &middot; [FineWeb](https://arxiv.org/abs/2406.17557) &middot; [Gopher](https://arxiv.org/abs/2112.11446)
    """
        ),
    ])
    technique_3
    return


@app.cell(hide_code=True)
def technique_minhash(DIAGRAMS, mo):
    technique_4 = mo.vstack([
        mo.md(
            r"""
    ### 4. MinHash signatures &mdash; Executed on this cache

    **Status:** Executed. Each qualifying document is reduced to a 112-value
    MinHash signature; the fraction of agreeing values estimates Jaccard.

    **What it is.** A MinHash signature is a fixed-length fingerprint of a set with
    the property that, under a random permutation of the universe of shingles, the
    probability that two sets share the same minimum element **equals their
    Jaccard similarity**. Averaging over many permutations turns that identity
    into a low-variance estimator.

    **How it works internally.** Simulate 112 permutations with affine hashes
    `(a_i * x + b_i) mod p` over a Mersenne prime `p = 2^61 - 1`; for each, the
    signature slot is the minimum hash over the document's shingles. The estimate
    is the fraction of the 112 slots where two signatures agree; its standard
    error is about `1/sqrt(112) ~ 0.094` per pair, and the measured mean absolute
    error against exact Jaccard on this cache is **0.019**. The implementation is
    numpy-vectorized (~12 s for 20,000 documents) rather than a per-permutation
    Python loop (~118 s).
    """
        ),
        mo.Html(DIAGRAMS["minhash"]),
        mo.md(
            r"""
    **Where it has been used.**

    | Project | Configuration | Source |
    |---|---|---|
    | FineWeb | Word 5-grams, **112** hash functions | [arXiv:2406.17557](https://arxiv.org/abs/2406.17557) §3.4 |
    | RefinedWeb / Falcon | **9,000** hashes over 5-grams in 20 buckets of 450, explicitly more aggressive than "the 10 hashes of The Pile"; combined exact+fuzzy removed ~50% | [arXiv:2306.01116](https://arxiv.org/abs/2306.01116) |
    | Gopher / MassiveText | 13-grams, **450** hashes, duplicate when estimated Jaccard > 0.8 | [arXiv:2112.11446](https://arxiv.org/abs/2112.11446) |
    | GPT-3 | Document-level fuzzy dedup "using a locality-sensitive hashing algorithm based on MinHash" (no fingerprint width or hash count given in the paper) | [arXiv:2005.14165](https://arxiv.org/abs/2005.14165) |
    | The Stack | **256** permutations, LSH, Jaccard > 0.85, removing 38.6% of permissive files | [arXiv:2211.15533](https://arxiv.org/abs/2211.15533) |
    | `datasketch` | `MinHashLSH(threshold=0.9, num_perm=128)`; when `params` is None it grid-searches *b* and *r* to minimize weighted FP+FN, so `b*r` need not equal `num_perm` | [lsh.py](https://github.com/ekzhu/datasketch/blob/master/datasketch/lsh.py) |

    This notebook adopts FineWeb's parameters (5-grams, 112 hashes) so its results
    are directly comparable to published practice.

    **When to use it.** Whenever near-duplicate detection is needed at a scale
    where exact all-pairs Jaccard is infeasible; MinHash gives a bounded-error
    estimate at fixed per-document cost.

    | Pros | Cons |
    |---|---|
    | Fixed-size signature regardless of document length | Estimate, not exact &mdash; needs a verification pass for hard decisions |
    | Unbiased Jaccard estimator with quantifiable variance | Only captures set (Jaccard) similarity, not semantics |
    | Vectorizes cleanly; pairs cheaply via LSH | More hashes lower variance but raise time and memory |

    **Source:** [FineWeb](https://arxiv.org/abs/2406.17557) &middot; [RefinedWeb](https://arxiv.org/abs/2306.01116) &middot; [Gopher](https://arxiv.org/abs/2112.11446) &middot; [GPT-3](https://arxiv.org/abs/2005.14165) &middot; [The Stack](https://arxiv.org/abs/2211.15533)
    """
        ),
    ])
    technique_4
    return


@app.cell(hide_code=True)
def technique_lsh(DIAGRAMS, mo):
    technique_5 = mo.vstack([
        mo.md(
            r"""
    ### 5. LSH banding &mdash; Executed on this cache

    **Status:** Executed. The 112-value signatures are banded into 14 bands of 8
    rows; any band collision makes a candidate pair.

    **What it is.** Locality-sensitive hashing over MinHash signatures. It avoids
    comparing every pair by hashing signature *bands* into buckets so that only
    documents colliding in at least one band are ever compared.

    **How it works internally.** Split each 112-value signature into `b = 14`
    bands of `r = 8` rows. Two documents are a candidate pair if any band's 8
    values are identical (hashed to the same bucket). The probability a pair with
    Jaccard *s* becomes a candidate is `1 - (1 - s^r)^b`, an S-curve with its
    steep rise near `(1/b)^{1/r}` &mdash; here `(1/14)^{1/8} = 0.719`. At *s* =
    0.75 the candidate probability is 0.772. A per-bucket size cap
    (`MAX_BUCKET_SIZE`) guards against one degenerate boilerplate bucket exploding
    into O(n&sup2;) pairs.
    """
        ),
        mo.Html(DIAGRAMS["lsh-banding"]),
        mo.md(
            r"""
    **Where it has been used.**

    | Project | Configuration | Source |
    |---|---|---|
    | FineWeb | **14 buckets &times; 8 hashes** from 112, targeting documents &ge; ~75% similar; duplicates if any single band matches | [arXiv:2406.17557](https://arxiv.org/abs/2406.17557) §3.4 |
    | NVIDIA NeMo Curator | GPU fuzzy dedup defaults **num_bands = 20 &times; minhashes_per_band = 13**, `char_ngrams = 24`, seed 42, then connected components at ~80% similarity | [NeMo Curator dedup docs](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication.md) |
    | LSHBloom | Replaces the LSH posting index with one Bloom filter per band; classic MinHashLSH index &asymp; **23 TB at 5B docs** and &asymp; **46 TB at 10B**, versus &asymp; **590 GB** for LSHBloom at T = 0.8 / 128 permutations / 9 bands; ~180&times; space reduction on peS2o (ar5iv preprint; adoption beyond the paper unverified) | [arXiv:2411.04257](https://arxiv.org/abs/2411.04257) |
    | Lee et al. NearDup | 5-grams, signature 9000, **b = 20, r = 450**, then edit similarity > 0.8 | [arXiv:2107.06499](https://arxiv.org/abs/2107.06499) |

    This notebook adopts FineWeb's 14&times;8 split.

    **When to use it.** As the candidate generator in front of any exact
    verification, whenever all-pairs comparison is infeasible and the target
    similarity band is known (it sets *b* and *r*).

    | Pros | Cons |
    |---|---|
    | Turns O(n&sup2;) comparison into near-linear candidate generation | *b* and *r* must be tuned to the target similarity |
    | S-curve gives explicit control over the recall/precision knee | Degenerate buckets can still blow up without a size cap |
    | Works directly on MinHash output | Still yields candidates, not decisions &mdash; verification required |

    **Source:** [FineWeb](https://arxiv.org/abs/2406.17557) &middot; [NeMo Curator](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication.md) &middot; [LSHBloom](https://arxiv.org/abs/2411.04257) &middot; [Lee et al. 2021](https://arxiv.org/abs/2107.06499)
    """
        ),
    ])
    technique_5
    return


@app.cell(hide_code=True)
def technique_simhash(DIAGRAMS, mo):
    technique_6 = mo.vstack([
        mo.md(
            r"""
    ### 6. SimHash &mdash; Documented; not executed

    **Status:** Documented; not executed. **Blocked:** SimHash weights terms,
    which needs a corpus-wide IDF table this bounded 20,000-document sample cannot
    estimate stably, and MinHash already covers the near-duplicate regime here.

    **What it is.** A fingerprinting scheme in which cosine similarity of weighted
    term vectors is preserved as **Hamming distance** between short bit
    fingerprints. It is a random-hyperplane LSH: each bit records which side of a
    random hyperplane a document's vector falls on.

    **How it works internally.** Build a weighted term vector (typically TF-IDF).
    For each of *k* random hyperplanes, set the corresponding fingerprint bit to
    the sign of the vector's projection. Because the probability two vectors land
    on the same side of a random hyperplane is `1 - theta/pi`, close vectors yield
    fingerprints differing in few bits. Detection indexes fingerprints so that
    candidates within a small Hamming radius can be found by the pigeonhole
    principle, then verified exactly.
    """
        ),
        mo.Html(DIAGRAMS["simhash"]),
        mo.md(
            r"""
    **Where it has been used.**

    | Project | Configuration | Source |
    |---|---|---|
    | Manku et al. (Google, WWW 2007) | Production near-duplicate detection for web crawling: **64-bit** fingerprint, near-duplicate iff **Hamming distance &le; 3**, indexed as 4 chunks of 16 bits by pigeonhole then exact Hamming verification | [DOI 10.1145/1242572.1242592](https://doi.org/10.1145/1242572.1242592) |
    | Charikar 2002 | Origin of the random-hyperplane LSH for cosine that SimHash implements | random-hyperplane rounding, STOC 2002 |
    | `text-dedup` | Ships a `simhash` module (default bit-width / n-gram size not file-verified, so not stated) | [ChenghaoMou/text-dedup](https://github.com/ChenghaoMou/text-dedup) |

    **Correction worth stating explicitly:** modern LLM corpora often *assumed* to
    use SimHash actually used MinHash &mdash; GPT-3 and Gopher/MassiveText both
    did. SimHash's verified large-scale home is Google's web crawler, not LLM
    pretraining.

    **When to use it.** When similarity should follow **weighted cosine** rather
    than unweighted set overlap &mdash; e.g. when a few high-IDF terms should
    dominate the match &mdash; and a compact fingerprint with cheap Hamming lookup
    is wanted.

    | Pros | Cons |
    |---|---|
    | Very compact fingerprint (e.g. 64 bits) | Needs a stable term-weighting (IDF) model |
    | Encodes weighted cosine, not just set overlap | Hamming threshold is coarse and corpus-sensitive |
    | Fast Hamming-radius candidate lookup | Weaker than MinHash for pure set-resemblance dedup |

    **What would be required to run it here.** A corpus-wide IDF table estimated
    on a much larger sample, plus a target Hamming threshold calibrated for Hindi
    text.

    **Source:** [Manku et al. 2007](https://doi.org/10.1145/1242572.1242592) &middot; [text-dedup](https://github.com/ChenghaoMou/text-dedup)
    """
        ),
    ])
    technique_6
    return


@app.cell(hide_code=True)
def technique_suffix(DIAGRAMS, mo):
    technique_7 = mo.vstack([
        mo.md(
            r"""
    ### 7. Suffix-array exact substring dedup (ExactSubstr) &mdash; Documented; not executed

    **Status:** Documented; not executed. **Blocked:** the C4-scale reference
    configuration needs ~96 cores, >600 GB RAM, and >1 TB disk; and it removes
    spans *inside* documents, a different unit than this document-level pass.

    **What it is.** Instead of asking whether two documents are similar, it finds
    every long substring that repeats anywhere in the corpus and removes the
    repeats. The unit is the span, not the document.

    **How it works internally.** Concatenate the tokenized corpus into one
    sequence, build a suffix array (all suffixes sorted), and scan adjacent
    suffixes: any adjacent pair whose longest common prefix reaches a threshold
    (50 BPE tokens in Lee et al.) marks a repeated span to excise. Sorting is what
    makes repeats adjacent, so a single linear scan of the sorted array finds them
    all. A suffix array costs ~8 bytes per token, which is what drives the
    hardware requirement.
    """
        ),
        mo.Html(DIAGRAMS["suffix-array"]),
        mo.md(
            r"""
    **Where it has been used.**

    | Project | Configuration | Source |
    |---|---|---|
    | Lee et al. (ExactSubstr) | Concatenate tokenized corpus, build suffix array, cut at LCP &ge; **50 BPE tokens**; deduplication cut memorized emission ~10&times;; removed **7.18%** of C4 train tokens; one 61-word sentence appeared **61,036&times;** in C4 train | [arXiv:2107.06499](https://arxiv.org/abs/2107.06499) |
    | Reference implementation | Hardware note: ~96 cores / >600 GB RAM / >1 TB disk for ~300 GB C4; suffix array ~8 bytes/token | [google-research/deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets) |
    | RefinedWeb / Falcon | Exact substring via suffix arrays run alongside MinHash fuzzy dedup, ~50% total removal | [arXiv:2306.01116](https://arxiv.org/abs/2306.01116) |

    **When to use it.** When the goal is to strip repeated *passages* &mdash;
    templated boilerplate, quoted spans, memorization-inducing repeats &mdash;
    rather than to drop whole near-duplicate documents, and when the hardware to
    hold a corpus-scale suffix array is available.

    | Pros | Cons |
    |---|---|
    | Removes intra-document repeats no document-level method sees | Suffix array memory (~8 bytes/token) is the bottleneck |
    | Exact: found spans are genuinely repeated | Removes spans, which can fragment otherwise-coherent documents |
    | Directly reduces memorization of repeated text | Reference pipeline needs a large multi-core / high-RAM machine |

    **What would be required to run it here.** A machine able to hold a
    corpus-scale suffix array (tens to hundreds of GB of RAM) and a decision to
    operate at span rather than document granularity.

    **Source:** [Lee et al. 2021](https://arxiv.org/abs/2107.06499) &middot; [deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets) &middot; [RefinedWeb](https://arxiv.org/abs/2306.01116)
    """
        ),
    ])
    technique_7
    return


@app.cell(hide_code=True)
def technique_semdedup(DIAGRAMS, mo):
    technique_8 = mo.vstack([
        mo.md(
            r"""
    ### 8. SemDeDup &mdash; Documented; not executed

    **Status:** Documented; not executed. **Blocked:** it requires a GPU embedding
    pass with a Hindi-capable encoder; and per FairDeDup, semantic pruning can
    worsen fairness, which needs stratified evaluation this workbook does not have.

    **What it is.** Semantic deduplication: remove documents that are *meaning*
    duplicates even when they share no surface tokens. It operates in embedding
    space rather than on shingles or hashes, so it catches paraphrases that
    MinHash and SimHash cannot.

    **How it works internally.** Embed every document (CLIP for images, OPT for
    text in the original work), k-means cluster the embeddings, and within each
    cluster compute pairwise cosine similarity. Documents with cosine &ge; 1 &minus;
    &epsilon; are semantic duplicates; from each such group **keep the example
    farthest from the cluster centroid** (the least prototypical, most
    informative) and drop the rest. Clustering bounds the cosine comparisons to
    within-cluster pairs.
    """
        ),
        mo.Html(DIAGRAMS["semdedup"]),
        mo.md(
            r"""
    **Where it has been used.**

    | Project | Configuration | Source |
    |---|---|---|
    | SemDeDup | Embed &rarr; k-means (k = **50,000** vision, **11,000** text) &rarr; intra-cluster cosine &rarr; keep farthest from centroid. On LAION-440M ~50% of images have a duplicate at &epsilon; = 0.03; removing ~50% costs <0.5% ImageNet zero-shot and trains ~2&times; faster; on C4 ~15% efficiency gain | [arXiv:2303.09540](https://arxiv.org/abs/2303.09540), [facebookresearch/SemDeDup](https://github.com/facebookresearch/SemDeDup) |
    | D4 | SemDeDup plus SSL-prototype diversity ranking on already-MinHash-deduped CommonCrawl; at 6.7B OPT / 100B tokens, ~18&ndash;20% training-efficiency gain | [arXiv:2308.12284](https://arxiv.org/abs/2308.12284) |
    | FairDeDup | Same pipeline on LAION-400M (50k clusters, prune 50%) but keeps the member closest to under-represented sensitive-concept prototypes. Finding: SemDeDup **worsened** FACET gender disparity (mean .104&rarr;.113, max .303&rarr;.346) at essentially unchanged ImageNet accuracy | [arXiv:2404.16123](https://arxiv.org/abs/2404.16123) |

    SemDeDup / D4 / FairDeDup are research-scale results on LAION / C4 / CC;
    adoption as the default stage of a released trillion-token open corpus is
    unverified.

    **When to use it.** After exact and fuzzy dedup, when paraphrase-level
    redundancy is the remaining problem and a suitable embedding model plus GPU
    budget are available &mdash; and with a fairness/coverage evaluation attached.

    | Pros | Cons |
    |---|---|
    | Catches paraphrases no surface method detects | Requires an embedding model and GPU compute |
    | Can substantially cut training cost at fixed quality | Can worsen fairness by over-pruning sparse clusters |
    | Keeps the most informative cluster member | &epsilon; and *k* are sensitive, corpus-specific knobs |

    **What would be required to run it here.** A Hindi-capable text encoder, GPU
    compute for the embedding pass, and a stratified fairness evaluation to catch
    FairDeDup-style regressions.

    **Source:** [SemDeDup](https://arxiv.org/abs/2303.09540) &middot; [D4](https://arxiv.org/abs/2308.12284) &middot; [FairDeDup](https://arxiv.org/abs/2404.16123)
    """
        ),
    ])
    technique_8
    return


@app.cell(hide_code=True)
def technique_scope(DIAGRAMS, mo):
    technique_9 = mo.vstack([
        mo.md(
            r"""
    ### 9. Global / cross-shard deduplication &mdash; Documented; structurally impossible here

    **Status:** Documented; not executed &mdash; and structurally impossible on a
    single bounded slice. This workbook holds one 20,000-document sample; global
    dedup is defined by comparing *across* shards, sources, or snapshots, which
    there is nothing here to do.

    **What it is.** Deduplication whose scope spans the entire corpus &mdash; every
    shard, source, and crawl snapshot compared against every other &mdash; rather
    than within one shard. It is the only scope that can remove a document
    duplicated across two different shards.

    **How it works internally.** Mechanically it is the same exact/MinHash/LSH
    machinery, but the index must span the whole corpus at once, which is where the
    cost lives: a classic MinHashLSH posting index is &asymp; 23 TB at 5B documents
    and &asymp; 46 TB at 10B. The harder point is that global scope is **not
    automatically better for quality**.
    """
        ),
        mo.Html(DIAGRAMS["dedup-scope"]),
        mo.md(
            r"""
    **Where it has been used &mdash; and the finding that inverts the intuition.**

    | Project | Configuration and finding | Source |
    |---|---|---|
    | FineWeb | Ran **global** iterative MinHash across all **96** CommonCrawl snapshots (newest&rarr;oldest), cutting the corpus to ~4T tokens and stripping up to 90% of base-filtered data from the oldest snapshots. The ablation scored **far below RefinedWeb**. Inspecting crawl **2013-48**, the ~31B tokens *kept* were **lower quality** than the ~171B tokens *removed* (more ads, keyword lists, bad formatting). FineWeb therefore shipped **per-snapshot** dedup, reaching ~20T tokens and matching RefinedWeb | [arXiv:2406.17557](https://arxiv.org/abs/2406.17557) §3.4 |
    | Dolma | Bounds scope deliberately: "We perform this deduplication across files from the same subset ... but not across sources." Its CCNet stage removes ~70% of paragraphs and cuts 175.1 TB &rarr; 27.7 TB (84.2%) | [arXiv:2402.00159](https://arxiv.org/abs/2402.00159) |
    | LSHBloom | The memory wall, citably: classic MinHashLSH index &asymp; 23 TB at 5B docs, &asymp; 46 TB at 10B | [arXiv:2411.04257](https://arxiv.org/abs/2411.04257) |

    The measured cross-config overlap between `verified/hin` and `unverified/hin`
    here was **0**, which does not prove global cleanliness &mdash; it only means
    these two slices do not overlap.

    **When to use it.** When a genuine cross-shard duplication problem exists *and*
    an ablation shows global scope helps downstream quality &mdash; not by default.
    FineWeb avoided global dedup on quality grounds; LSHBloom addresses the memory
    grounds. The two arguments are separate.

    | Pros | Cons |
    |---|---|
    | The only scope that removes cross-shard duplicates | Index size grows to tens of TB at billions of docs |
    | Can enforce one global copy of any document | Can delete large useful clusters and keep worse survivors (FineWeb) |
    | Necessary for true corpus-wide uniqueness | Requires an ablation to justify; not a safe default |

    **What would be required to run it here.** The full multi-shard corpus (not one
    slice), an index able to span it, and a downstream ablation to decide whether
    global scope actually helps &mdash; because the FineWeb result shows it may not.

    **Conclusion.** **Dedup scope is a quality decision, not only a systems
    constraint.** Do not present global dedup as unconditionally correct.

    **Source:** [FineWeb](https://arxiv.org/abs/2406.17557) &middot; [Dolma](https://arxiv.org/abs/2402.00159) &middot; [LSHBloom](https://arxiv.org/abs/2411.04257)
    """
        ),
    ])
    technique_9
    return


@app.cell(hide_code=True)
def implementation_intro(mo):
    mo.md(r"""
    ## The executed pipeline: implementation

    The functions below are the exact code that runs on the cache. Nothing
    executes on definition; the run controls in the next section drive them. The
    code is shown (not hidden) because this workbook is a teaching artifact and the
    one executed mechanism is the point.

    Pipeline order: **exact hash &rarr; word 5-gram shingles &rarr; MinHash
    signatures &rarr; LSH candidate generation &rarr; exact-Jaccard verification
    &rarr; union-find clustering.** Each near-duplicate component keeps its
    **longest** document (ties broken by lowest `doc_id`); every excluded or
    skipped document is counted and reported rather than silently kept or dropped.
    """)
    return


@app.cell
def dedup_implementation(
    JACCARD_THRESHOLD,
    LSH_BANDS,
    LSH_ROWS,
    MAX_BUCKET_SIZE,
    MINHASH_SEED,
    MIN_TOKENS_FOR_MINHASH,
    NUM_HASHES,
    SHINGLE_K,
    hashlib,
    np,
    random,
    re,
    record_stats,
    time,
):
    # Executed deduplication pipeline. All functions are pure and defined here
    # only; nothing runs until a run button calls run_pipeline. Shown (not hidden)
    # on purpose: this is the mechanism the workbook teaches.


    def normalize_for_hashing(text):
        """Whitespace collapse + casefold, and nothing else.

        ZWJ/ZWNJ/virama/matra are deliberately preserved (AGENTS.md line 40): each
        is meaningful in Devanagari. This normalized string is used ONLY to derive
        a hash or shingle; the retained document text is never mutated.
        """
        return re.sub(r"\s+", " ", (text or "").strip()).casefold()


    def exact_dedup(records):
        """First-occurrence-wins exact dedup on sha256(normalize(text))."""
        seen = {}
        groups = {}
        kept = []
        for record in records:
            digest = hashlib.sha256(
                normalize_for_hashing(record.get("text", "")).encode("utf-8")
            ).hexdigest()
            if digest in seen:
                groups.setdefault(digest, [seen[digest]]).append(record["doc_id"])
            else:
                seen[digest] = record["doc_id"]
                kept.append(record)
        removed = sum(len(members) - 1 for members in groups.values())
        largest = max((len(m) for m in groups.values()), default=0)
        examples = []
        for members in groups.values():
            if len(examples) >= 5:
                break
            examples.append({"kept_doc_id": members[0], "removed_doc_id": members[1]})
        report = {
            "removed": removed,
            "groups": len(groups),
            "largest_group": largest,
            "examples": examples,
        }
        return kept, report


    def word_shingles(text, k=SHINGLE_K):
        """Set of k-word shingles. \\w with re.UNICODE matches Devanagari."""
        tokens = [t.casefold() for t in re.findall(r"\w+", text or "", re.UNICODE)]
        if len(tokens) < k:
            return set()
        return {"\x1f".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


    def build_minhash_permutations():
        """112 affine hash coefficients from the fixed seed. Odd a => valid permutation."""
        rng = random.Random(MINHASH_SEED)
        a = np.array([rng.getrandbits(31) | 1 for _ in range(NUM_HASHES)], dtype=np.uint64)
        b = np.array([rng.getrandbits(31) for _ in range(NUM_HASHES)], dtype=np.uint64)
        return a, b


    MINHASH_A, MINHASH_B = build_minhash_permutations()
    MINHASH_MOD = np.uint64((1 << 61) - 1)


    def shingle_hashes_of(shingles):
        values = np.empty(len(shingles), dtype=np.uint64)
        for i, shingle in enumerate(shingles):
            digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
            values[i] = np.uint64(int.from_bytes(digest, "little") >> 3)
        return values


    def minhash_signature(shingles):
        """Numpy-vectorized MinHash: min over ((a*x + b) mod p) per permutation.

        Returns None for documents with an empty shingle set or fewer than
        MIN_TOKENS_FOR_MINHASH tokens; such documents are excluded from near-dedup
        and counted separately.
        """
        if not shingles:
            return None
        hashes = shingle_hashes_of(shingles)
        slots = (MINHASH_A[:, None] * hashes[None, :] + MINHASH_B[:, None]) % MINHASH_MOD
        return slots.min(axis=1)


    def lsh_candidates(signatures):
        """14 bands x 8 rows. Candidate if any band collides. Returns sorted index pairs.

        Buckets larger than MAX_BUCKET_SIZE are skipped and counted, guarding a
        single degenerate boilerplate bucket from exploding into O(n^2) pairs.
        """
        buckets = {}
        for index, signature in signatures.items():
            for band in range(LSH_BANDS):
                block = signature[band * LSH_ROWS:(band + 1) * LSH_ROWS]
                key = (band, block.tobytes())
                buckets.setdefault(key, []).append(index)
        candidates = set()
        oversized = 0
        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) > MAX_BUCKET_SIZE:
                oversized += 1
                continue
            members.sort()
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    candidates.add((members[i], members[j]))
        return candidates, oversized


    def verify_pairs(candidates, shingle_sets, signatures):
        """For each candidate: estimated Jaccard (signature agreement) and true Jaccard."""
        verified = []
        for a, b in sorted(candidates):
            sig_a, sig_b = signatures[a], signatures[b]
            estimated = float(np.mean(sig_a == sig_b))
            set_a, set_b = shingle_sets[a], shingle_sets[b]
            union = len(set_a | set_b)
            true_jaccard = (len(set_a & set_b) / union) if union else 0.0
            if estimated >= JACCARD_THRESHOLD:
                verified.append({
                    "a_index": a,
                    "b_index": b,
                    "estimated_jaccard": estimated,
                    "true_jaccard": true_jaccard,
                    "confirmed": true_jaccard >= JACCARD_THRESHOLD,
                })
        return verified


    def resolve_clusters(confirmed_pairs, records, index_to_docid):
        """Union-find over confirmed pairs; keep the longest text per component.

        Tie-break: lowest doc_id. Contrast with SemDeDup, which keeps the member
        farthest from its cluster centroid.
        """
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for pair in confirmed_pairs:
            union(pair["a_index"], pair["b_index"])

        components = {}
        for node in list(parent):
            components.setdefault(find(node), set()).add(node)

        by_index = {i: record for i, record in enumerate(records)}
        removed_doc_ids = []
        clusters = []
        for members in components.values():
            if len(members) < 2:
                continue
            ordered = sorted(
                members,
                key=lambda i: (-len(by_index[i].get("text", "")), by_index[i]["doc_id"]),
            )
            kept_index = ordered[0]
            removed_indices = ordered[1:]
            removed_doc_ids.extend(by_index[i]["doc_id"] for i in removed_indices)
            clusters.append({
                "size": len(members),
                "kept_doc_id": by_index[kept_index]["doc_id"],
                "removed_doc_ids": [by_index[i]["doc_id"] for i in removed_indices],
            })
        clusters.sort(key=lambda c: c["size"], reverse=True)
        return set(removed_doc_ids), clusters


    def run_pipeline(records):
        """exact -> shingles -> signatures -> LSH -> verify -> clusters, with per-stage stats."""
        timings = {}
        started = time.perf_counter()

        exact_kept, exact_report = exact_dedup(records)
        timings["exact"] = round(time.perf_counter() - started, 3)

        t = time.perf_counter()
        shingle_sets = {}
        signatures = {}
        too_short = 0
        for index, record in enumerate(exact_kept):
            shingles = word_shingles(record.get("text", ""))
            token_count = len(re.findall(r"\w+", record.get("text", ""), re.UNICODE))
            if not shingles or token_count < MIN_TOKENS_FOR_MINHASH:
                too_short += 1
                continue
            shingle_sets[index] = shingles
            signatures[index] = minhash_signature(shingles)
        timings["signatures"] = round(time.perf_counter() - t, 3)

        t = time.perf_counter()
        candidates, oversized = lsh_candidates(signatures)
        timings["lsh"] = round(time.perf_counter() - t, 3)

        t = time.perf_counter()
        verified = verify_pairs(candidates, shingle_sets, signatures)
        timings["verify"] = round(time.perf_counter() - t, 3)

        confirmed = [pair for pair in verified if pair["confirmed"]]
        t = time.perf_counter()
        removed_doc_ids, clusters = resolve_clusters(
            confirmed, exact_kept, {i: r["doc_id"] for i, r in enumerate(exact_kept)}
        )
        timings["clusters"] = round(time.perf_counter() - t, 3)

        near_kept = [r for r in exact_kept if r["doc_id"] not in removed_doc_ids]

        index_to_docid = {i: r["doc_id"] for i, r in enumerate(exact_kept)}
        for pair in verified:
            pair["a_doc_id"] = index_to_docid[pair["a_index"]]
            pair["b_doc_id"] = index_to_docid[pair["b_index"]]

        thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
        threshold_sensitivity = {
            f"{th:.2f}": sum(1 for p in verified if p["true_jaccard"] >= th)
            for th in thresholds
        }

        errors = [abs(p["estimated_jaccard"] - p["true_jaccard"]) for p in verified]
        confirmed_count = sum(1 for p in verified if p["confirmed"])
        n = len(exact_kept)
        all_pairs = n * (n - 1) // 2

        return {
            "raw": record_stats(records),
            "after_exact": record_stats(exact_kept),
            "after_near": record_stats(near_kept),
            "exact_report": exact_report,
            "too_short_for_minhash": too_short,
            "oversized_buckets_skipped": oversized,
            "candidate_pairs": len(candidates),
            "verified_pairs": len(verified),
            "confirmed_pairs": confirmed_count,
            "confirmation_rate": (confirmed_count / len(verified)) if verified else 0.0,
            "estimator_mean_abs_error": (float(np.mean(errors)) if errors else 0.0),
            "estimator_max_abs_error": (float(np.max(errors)) if errors else 0.0),
            "clusters": clusters,
            "cluster_count": len(clusters),
            "largest_cluster": (clusters[0]["size"] if clusters else 0),
            "near_removed": len(removed_doc_ids),
            "total_removed": exact_report["removed"] + len(removed_doc_ids),
            "all_pairs_comparisons": all_pairs,
            "lsh_reduction_factor": (all_pairs / len(candidates)) if candidates else None,
            "threshold_sensitivity": threshold_sensitivity,
            "verified_detail": verified,
            "removals_by_type": removals_by_type(records, exact_report, clusters, exact_kept),
            "timings": timings,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "kept_records": near_kept,
        }


    def removals_by_type(records, exact_report, clusters, exact_kept):
        """Count removed documents by Sangraha type (web/pdf/speech)."""
        by_id = {r["doc_id"]: r.get("type", "unknown") for r in records}
        kept_ids = {r["doc_id"] for r in exact_kept}
        removed_ids = [r["doc_id"] for r in records if r["doc_id"] not in kept_ids]
        for cluster in clusters:
            removed_ids.extend(cluster["removed_doc_ids"])
        counts = {}
        for doc_id in removed_ids:
            doc_type = by_id.get(doc_id, "unknown")
            counts[doc_type] = counts.get(doc_type, 0) + 1
        return counts

    return (run_pipeline,)


@app.cell(hide_code=True)
def run_controls_intro(mo):
    mo.md(r"""
    ## Run controls, metrics, and artifacts

    Nothing heavy runs when the notebook opens. Click the first button to
    materialize the 20,000-document cache (~41 s, streamed once), then the second
    to run the pipeline (~15 s). The pipeline button refuses to run without a
    hash-valid cache and shows the validation message instead.
    """)
    return


@app.cell
def dedup_buttons(mo):
    materialize_button = mo.ui.run_button(
        label="Materialize 20,000-document Sangraha verified/hin cache"
    )
    run_button = mo.ui.run_button(label="Run deduplication pipeline")
    mo.vstack([materialize_button, run_button])
    return materialize_button, run_button


@app.cell
def materialize_cell(cache_status, materialize_button, materialize_cache, mo):
    # Cache materialization, gated on the first button. Streams the pinned revision
    # once, then reports the hash-validated cache status.
    if materialize_button.value:
        _records, _manifest = materialize_cache()
        cache_view = mo.md(
            f"""
    **Cache materialized.** {_manifest['retained_documents']:,} documents,
    {_manifest['characters']:,} characters, {_manifest['estimated_tokens']:,}
    estimated tokens in {_manifest['elapsed_seconds']} s.

    `raw_sha256 = {_manifest['raw_sha256'][:16]}...`  ·  revision
    `{_manifest['revision']}`
    """
        )
    else:
        _valid, _existing, _message = cache_status()
        if _valid:
            cache_view = mo.md(
                f"**Cache present and hash-valid:** {_message} "
                f"({_existing['retained_documents']:,} documents)."
            )
        else:
            cache_view = mo.md(
                f"**No valid cache yet.** {_message} Click "
                f"*Materialize 20,000-document Sangraha verified/hin cache* to build it."
            )
    cache_view
    return


@app.cell
def pipeline_run(
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    DATASET_SPLIT,
    JACCARD_THRESHOLD,
    LSH_BANDS,
    LSH_ROWS,
    MAX_BUCKET_SIZE,
    METRICS_PATH,
    MINHASH_SEED,
    MIN_TOKENS_FOR_MINHASH,
    NUM_HASHES,
    PIPELINE_VERSION,
    Path,
    RESULTS_DIR,
    SAMPLE_DOCUMENTS,
    SHINGLE_K,
    atomic_write_text,
    datetime,
    json,
    load_bounded_records,
    mo,
    run_button,
    run_pipeline,
    sha256_path,
    timezone,
):
    # Pipeline execution, gated on the second button. Refuses to run without a
    # hash-valid cache. On success writes the tracked metrics summary and the
    # local-only per-pair / per-cluster results, then exposes PIPELINE_RESULT.
    PIPELINE_RESULT = None
    if run_button.value:
        _records, _manifest, _message = load_bounded_records()
        if not _records:
            pipeline_status = mo.md(
                f"**Cannot run.** {_message} Materialize a hash-valid cache first."
            )
        else:
            PIPELINE_RESULT = run_pipeline(_records)
            _res = PIPELINE_RESULT
            _notebook_path = Path("deduplication/deduplication.py")
            _notebook_sha256 = (
                sha256_path(_notebook_path) if _notebook_path.exists() else None
            )
            _summary = {
                "pipeline_version": PIPELINE_VERSION,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": {
                    "id": DATASET_ID,
                    "config": DATASET_CONFIG,
                    "split": DATASET_SPLIT,
                    "revision": DATASET_REVISION,
                    "license": DATASET_LICENSE,
                    "raw_sha256": _manifest.get("raw_sha256"),
                },
                "notebook_sha256": _notebook_sha256,
                "parameters": {
                    "sample_documents": SAMPLE_DOCUMENTS,
                    "shingle_k": SHINGLE_K,
                    "num_hashes": NUM_HASHES,
                    "lsh_bands": LSH_BANDS,
                    "lsh_rows": LSH_ROWS,
                    "jaccard_threshold": JACCARD_THRESHOLD,
                    "minhash_seed": MINHASH_SEED,
                    "min_tokens_for_minhash": MIN_TOKENS_FOR_MINHASH,
                    "max_bucket_size": MAX_BUCKET_SIZE,
                },
                "funnel": {
                    "raw": _res["raw"],
                    "after_exact_dedup": _res["after_exact"],
                    "after_near_dedup": _res["after_near"],
                },
                "removal": {
                    "exact_removed": _res["exact_report"]["removed"],
                    "exact_groups": _res["exact_report"]["groups"],
                    "exact_largest_group": _res["exact_report"]["largest_group"],
                    "near_removed": _res["near_removed"],
                    "clusters": _res["cluster_count"],
                    "largest_cluster": _res["largest_cluster"],
                    "total_removed": _res["total_removed"],
                    "removed_fraction": round(_res["total_removed"] / _res["raw"]["documents"], 4),
                    "too_short_for_minhash": _res["too_short_for_minhash"],
                    "oversized_buckets_skipped": _res["oversized_buckets_skipped"],
                    "removals_by_type": _res["removals_by_type"],
                },
                "near_dedup": {
                    "candidate_pairs": _res["candidate_pairs"],
                    "verified_pairs": _res["verified_pairs"],
                    "confirmed_pairs": _res["confirmed_pairs"],
                    "confirmation_rate": round(_res["confirmation_rate"], 4),
                    "estimator_mean_abs_error": round(_res["estimator_mean_abs_error"], 4),
                    "estimator_max_abs_error": round(_res["estimator_max_abs_error"], 4),
                    "all_pairs_comparisons": _res["all_pairs_comparisons"],
                    "lsh_reduction_factor": (
                        round(_res["lsh_reduction_factor"]) if _res["lsh_reduction_factor"] else None
                    ),
                    "threshold_sensitivity": _res["threshold_sensitivity"],
                },
                "timings": _res["timings"],
                "elapsed_seconds": _res["elapsed_seconds"],
            }
            atomic_write_text(
                METRICS_PATH, json.dumps(_summary, ensure_ascii=False, indent=2) + "\n"
            )
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            _pairs_text = "".join(
                json.dumps(
                    {
                        "a_doc_id": p["a_doc_id"],
                        "b_doc_id": p["b_doc_id"],
                        "estimated_jaccard": round(p["estimated_jaccard"], 6),
                        "true_jaccard": round(p["true_jaccard"], 6),
                        "confirmed": p["confirmed"],
                    },
                    ensure_ascii=False,
                ) + "\n"
                for p in _res["verified_detail"]
            )
            atomic_write_text(RESULTS_DIR / "duplicate_pairs.jsonl", _pairs_text)
            atomic_write_text(
                RESULTS_DIR / "clusters.json",
                json.dumps(_res["clusters"], ensure_ascii=False, indent=2) + "\n",
            )
            pipeline_status = mo.md(
                f"""
    **Pipeline complete** in {_res['elapsed_seconds']} s. Removed
    {_res['total_removed']:,} of {_res['raw']['documents']:,} documents
    ({100 * _res['total_removed'] / _res['raw']['documents']:.2f}%):
    **{_res['exact_report']['removed']} exact** + **{_res['near_removed']} near**.

    Wrote `{METRICS_PATH}` (tracked) and `{RESULTS_DIR}/` (local-only).
    """
            )
    else:
        pipeline_status = mo.md(
            "*Click **Run deduplication pipeline** to execute the run and write artifacts.*"
        )
    pipeline_status
    return (PIPELINE_RESULT,)


@app.cell(hide_code=True)
def metrics_view(
    MIN_TOKENS_FOR_MINHASH,
    PIPELINE_RESULT,
    load_bounded_records,
    mo,
):
    # Metrics tables, rendered from PIPELINE_RESULT. Document text is never shown
    # beyond 200-character previews; this workbook does not scrub PII and must not
    # be read as having done so.
    if PIPELINE_RESULT is None:
        metrics_view = mo.md("*No results yet. Run the pipeline above.*")
    else:
        _r = PIPELINE_RESULT

        def _fnum(x):
            return f"{x:,}"

        _raw, _exx, _near = _r["raw"], _r["after_exact"], _r["after_near"]
        _total = _raw["documents"]
        _funnel = mo.md(
            f"""
    ### Stage funnel

    | Stage | Documents | Characters | Est. tokens | \u0394 documents | Cumulative % removed |
    |---|--:|--:|--:|--:|--:|
    | raw | {_fnum(_raw['documents'])} | {_fnum(_raw['characters'])} | {_fnum(_raw['estimated_tokens'])} | \u2014 | 0.00% |
    | after exact dedup | {_fnum(_exx['documents'])} | {_fnum(_exx['characters'])} | {_fnum(_exx['estimated_tokens'])} | \u2212{_fnum(_raw['documents'] - _exx['documents'])} | {100*(_total-_exx['documents'])/_total:.2f}% |
    | after near dedup | {_fnum(_near['documents'])} | {_fnum(_near['characters'])} | {_fnum(_near['estimated_tokens'])} | \u2212{_fnum(_exx['documents'] - _near['documents'])} | {100*(_total-_near['documents'])/_total:.2f}% |
    """
        )

        _er = _r["exact_report"]
        _breakdown = mo.md(
            f"""
    ### Removal breakdown

    | Kind | Removed | Groups / clusters | Largest |
    |---|--:|--:|--:|
    | Exact duplicates | {_fnum(_er['removed'])} | {_fnum(_er['groups'])} groups | {_fnum(_er['largest_group'])} in a group |
    | Near duplicates | {_fnum(_r['near_removed'])} | {_fnum(_r['cluster_count'])} clusters | {_fnum(_r['largest_cluster'])} in a cluster |
    | **Total** | **{_fnum(_r['total_removed'])}** | \u2014 | {100*_r['total_removed']/_total:.2f}% of corpus |

    Excluded from near-dedup (fewer than {MIN_TOKENS_FOR_MINHASH} tokens):
    **{_fnum(_r['too_short_for_minhash'])}** documents, counted not dropped.
    Oversized LSH buckets skipped: **{_r['oversized_buckets_skipped']}**.
    """
        )

        _accuracy = mo.md(
            f"""
    ### MinHash estimate versus exact Jaccard on candidate pairs

    | Measure | Value |
    |---|--:|
    | Pairs verified | {_fnum(_r['verified_pairs'])} |
    | Confirmed by exact Jaccard | {_fnum(_r['confirmed_pairs'])} ({100*_r['confirmation_rate']:.1f}%) |
    | Mean \\|estimated \u2212 true\\| Jaccard | {_r['estimator_mean_abs_error']:.4f} |
    | Max \\|estimated \u2212 true\\| Jaccard | {_r['estimator_max_abs_error']:.4f} |

    The signature agrees with exact Jaccard to within about two hundredths on
    average \u2014 the empirical demonstration that a 112-value fingerprint estimates
    set resemblance faithfully.
    """
        )

        _reduction = _r["lsh_reduction_factor"]
        _efficiency = mo.md(
            f"""
    ### LSH efficiency

    LSH generated **{_fnum(_r['candidate_pairs'])}** candidate pairs instead of the
    **{_r['all_pairs_comparisons']:.3g}** all-pairs comparisons that exact
    all-to-all Jaccard would require \u2014 a reduction of about
    **{_reduction:,.0f}\u00d7**. This single number is why LSH exists.
    """
        )

        _ts = _r["threshold_sensitivity"]
        _threshold = mo.md(
            f"""
    ### Threshold sensitivity

    Confirmed-pair count at each Jaccard threshold, computed from the *same*
    candidate set with no recomputation \u2014 the threshold is a policy dial, not a
    new run.

    | Threshold | 0.50 | 0.60 | 0.70 | 0.75 | 0.80 | 0.90 |
    |---|--:|--:|--:|--:|--:|--:|
    | Confirmed pairs | {_ts['0.50']} | {_ts['0.60']} | {_ts['0.70']} | {_ts['0.75']} | {_ts['0.80']} | {_ts['0.90']} |
    """
        )

        _all_records, _, _ = load_bounded_records()
        _preview = {rec["doc_id"]: (rec.get("text", "")[:200]) for rec in _all_records}
        _rows = []
        for _p in sorted(
            (q for q in _r["verified_detail"] if q["confirmed"]),
            key=lambda p: -p["true_jaccard"],
        )[:5]:
            _pa = _preview.get(_p["a_doc_id"], "").replace(chr(10), " ")
            _pb = _preview.get(_p["b_doc_id"], "").replace(chr(10), " ")
            _rows.append(
                f"| `{_p['a_doc_id'][:14]}` / `{_p['b_doc_id'][:14]}` | "
                f"{_p['estimated_jaccard']:.3f} | {_p['true_jaccard']:.3f} | "
                f"A: {_pa[:90]}\u2026<br>B: {_pb[:90]}\u2026 |"
            )
        _examples = mo.md(
            "### Example confirmed pairs\n\n"
            "| Pair (doc_id / doc_id) | Est. J | True J | Previews (\u2264200 chars each side) |\n"
            "|---|--:|--:|---|\n" + "\n".join(_rows)
            + "\n\nThe largest cluster is Uttar Pradesh land-record / gazette form "
            "boilerplate \u2014 genuinely redundant templated text, correctly removed."
        )

        _bt = _r["removals_by_type"]
        _bytype = mo.md(
            "### Removals by Sangraha `type`\n\n"
            "| Type | Removed |\n|---|--:|\n"
            + "\n".join(f"| {k} | {_fnum(v)} |" for k, v in sorted(_bt.items()))
            + "\n\nWhether duplication concentrates in PDF-derived forms is visible here."
        )

        metrics_view = mo.vstack([
            _funnel, _breakdown, _accuracy, _efficiency, _threshold, _examples, _bytype
        ])
    metrics_view
    return


@app.cell(hide_code=True)
def closing_boundary(mo):
    mo.md(r"""
    ## Closing boundary

    - **This is a bounded local pass over 20,000 documents.** The 1.78% removal
      rate is this sample's measurement, not Sangraha's corpus-wide duplicate rate,
      and not a claim about any other config, split, or language.
    - **The near-duplicates found are Hindi government-form boilerplate** \u2014
      Uttar Pradesh land-record and gazette templates whose fixed scaffolding
      repeats across documents. That is genuine redundancy, correctly removed; it
      is not evidence of a systemic problem in the corpus.
    - **Global cross-shard deduplication did not run here**, and per FineWeb it
      would not automatically be an improvement if it had: global dedup across 96
      CommonCrawl snapshots produced *lower*-quality survivors than per-snapshot
      dedup. Scope is a quality decision.
    - **Sangraha was already MinHash/LSH-deduplicated** by AI4Bharat's Setu
      pipeline. What this workbook measures is residual duplication in an
      already-cleaned corpus \u2014 the realistic engineering case, not a raw crawl.
    - **Eight of the nine catalogued techniques are documentation only.** Each
      states the concrete resource that blocks running it here; none of them
      contributes a number to the metrics tables. Only exact hashing and the
      MinHash/LSH/exact-Jaccard pipeline executed.
    - **Deduplication selected documents; it never mutated them.** Normalization
      touched only throwaway hashing copies. The retained text is byte-identical to
      the source, and no PII scrubbing, quality filtering, or decontamination ran
      in this workbook.
    """)
    return


if __name__ == "__main__":
    app.run()
