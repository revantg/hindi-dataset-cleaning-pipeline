import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    # Session 4 — Stage 3: Language identification and validation

    This is a bounded, executable technique study for the Session 4 cleaning
    case study. It uses a deterministic **160-document Sangraha micro-corpus**
    (20 rows from each of eight fixed source/configuration pairs), not a
    replacement for the 30M-token Hindi case-study cache. The row order in the
    pinned stream is the selection rule; rows are not shuffled or selected
    after looking at predictions.

    The central policy is **validate and route, not silently delete**. A dataset
    split is a claim that must be checked against independent script and
    language signals. Native-script, code-mixed, and romanized Indic text have
    different failure modes, so this workbook keeps those cases visible.

    > Source labels are metadata claims, not gold labels. Every agreement rate
    > below is labelled agreement with Sangraha's source split, **not model
    > accuracy**.
    """)


@app.cell(hide_code=True)
def runtime_setup():
    import hashlib as _hashlib
    import json as _json
    import shutil as _shutil
    from collections import Counter as _Counter
    from datetime import datetime as _datetime
    from pathlib import Path
    from urllib.request import urlopen as _urlopen

    import fasttext as _fasttext
    from GlotScript import sc as _script_separate
    from GlotScript import sp as _script_predict
    from lingua import Language as _LinguaLanguage
    from lingua import LanguageDetectorBuilder as _LanguageDetectorBuilder

    # These constants are deliberately kept together: they define the bounded
    # sample contract and are also copied into the provenance manifest.
    DATASET_ID = "ai4bharat/sangraha"
    DATASET_REVISION = "8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8"
    DOCUMENTS_PER_SOURCE = 20
    MAX_ANALYSIS_CHARACTERS = 4_000
    SAMPLE_PATH = Path("language-identification/data/sangraha_lid_microset.jsonl")
    SOURCE_MANIFEST_PATH = Path("language-identification/data/source_manifest.json")

    DATA_DIR = SAMPLE_PATH.parent
    MODELS_DIR = DATA_DIR / "models"
    RESULTS_DIR = DATA_DIR / "results"
    FASTTEXT_MODEL_PATH = MODELS_DIR / "lid.176.ftz"
    INDICLID_ARCHIVE_PATH = MODELS_DIR / "indiclid-ftn.zip"
    INDICLID_EXTRACT_DIR = MODELS_DIR / "indiclid-ftn"
    NOTEBOOK_PATH = Path("language-identification/language_identification.py")

    FASTTEXT_URL = (
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    )
    INDICLID_URL = (
        "https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftn.zip"
    )
    LICENSE = "CC-BY-4.0"

    SOURCE_SPECS = [
        {
            "source_config": "verified",
            "source_split": "eng",
            "claimed_language": "eng",
            "claimed_script": "Latn",
            "role": "English baseline",
        },
        {
            "source_config": "verified",
            "source_split": "hin",
            "claimed_language": "hin",
            "claimed_script": "Deva",
            "role": "Hindi native script",
        },
        {
            "source_config": "verified",
            "source_split": "ben",
            "claimed_language": "ben",
            "claimed_script": "Beng",
            "role": "Bengali native script",
        },
        {
            "source_config": "verified",
            "source_split": "tam",
            "claimed_language": "tam",
            "claimed_script": "Taml",
            "role": "Tamil native script",
        },
        {
            "source_config": "verified",
            "source_split": "tel",
            "claimed_language": "tel",
            "claimed_script": "Telu",
            "role": "Telugu native script",
        },
        {
            "source_config": "verified",
            "source_split": "urd",
            "claimed_language": "urd",
            "claimed_script": "Arab",
            "role": "Urdu native script",
        },
        {
            "source_config": "synthetic",
            "source_split": "hin_Latn",
            "claimed_language": "hin",
            "claimed_script": "Latn",
            "role": "Hindi transliteration limitation case",
        },
        {
            "source_config": "synthetic",
            "source_split": "tel_Latn",
            "claimed_language": "tel",
            "claimed_script": "Latn",
            "role": "Telugu transliteration limitation case",
        },
    ]

    EXPECTED_SCRIPTS = {
        "eng": "Latn",
        "hin": "Deva",
        "ben": "Beng",
        "tam": "Taml",
        "tel": "Telu",
        "urd": "Arab",
    }
    EXPECTED_INDICLID_LABELS = {
        "eng": "eng_Latn",
        "hin": "hin_Deva",
        "ben": "ben_Beng",
        "tam": "tam_Tamil",
        "tel": "tel_Telu",
        "urd": "urd_Arab",
    }
    FASTTEXT_LANGUAGE_LABELS = {
        "en": "eng",
        "hi": "hin",
        "bn": "ben",
        "ta": "tam",
        "te": "tel",
        "ur": "urd",
    }

    def _utc_now():
        return (
            _datetime.now(_datetime.UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def _sha256_file(path):
        digest = _hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _atomic_text_write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _download_to(path, url):
        """Download once to a sibling temporary file and return its SHA-256."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size == 0:
            temporary = path.with_name(f".{path.name}.download")
            try:
                with _urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                    _shutil.copyfileobj(response, out, length=1024 * 1024)
                if not temporary.exists() or temporary.stat().st_size == 0:
                    raise RuntimeError(f"Downloaded artifact is empty: {url}")
                temporary.replace(path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Artifact is absent or empty: {path}")
        return _sha256_file(path)

    def _sample_inventory(path):
        if not path.exists():
            return None
        expected_keys = {
            (spec["source_config"], spec["source_split"])
            for spec in SOURCE_SPECS
        }
        counts = _Counter()
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = _json.loads(line)
                    key = (row["source_config"], row["source_split"])
                    if key not in expected_keys:
                        return None
                    counts[key] += 1
        except (OSError, KeyError, TypeError, ValueError, _json.JSONDecodeError):
            return None
        return counts

    def _sample_is_complete():
        if not SAMPLE_PATH.exists() or not SOURCE_MANIFEST_PATH.exists():
            return False
        counts = _sample_inventory(SAMPLE_PATH)
        if counts is None:
            return False
        expected_keys = {
            (spec["source_config"], spec["source_split"])
            for spec in SOURCE_SPECS
        }
        if set(counts) != expected_keys or any(
            counts[key] != DOCUMENTS_PER_SOURCE for key in expected_keys
        ):
            return False
        try:
            manifest = _json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
            manifest_keys = {
                (source["source_config"], source["source_split"])
                for source in manifest["sources"]
            }
        except (OSError, KeyError, TypeError, ValueError, _json.JSONDecodeError):
            return False
        return manifest_keys == expected_keys

    def _sample_summary(path):
        rows = []
        if not path.exists():
            return rows
        grouped = {
            (spec["source_config"], spec["source_split"]): {
                "source_config": spec["source_config"],
                "source_split": spec["source_split"],
                "claimed_language": spec["claimed_language"],
                "claimed_script": spec["claimed_script"],
                "documents": 0,
                "source_characters": 0,
                "analysis_characters": 0,
                "truncated": 0,
            }
            for spec in SOURCE_SPECS
        }
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = _json.loads(line)
                    key = (record["source_config"], record["source_split"])
                    if key not in grouped:
                        continue
                    item = grouped[key]
                    item["documents"] += 1
                    item["source_characters"] += int(
                        record.get("source_character_count", len(record.get("text", "")))
                    )
                    item["analysis_characters"] += int(
                        record.get("analysis_character_count", len(record.get("text", "")))
                    )
                    item["truncated"] += int(bool(record.get("truncated", False)))
        except (OSError, KeyError, TypeError, ValueError, _json.JSONDecodeError):
            return []
        return [
            grouped[(spec["source_config"], spec["source_split"])]
            for spec in SOURCE_SPECS
        ]

    def _materialize_microset():
        """Stream each pinned source and atomically write the 160-row sample."""
        from datasets import load_dataset

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        sample_temporary = SAMPLE_PATH.with_name(f".{SAMPLE_PATH.name}.tmp")
        manifest_temporary = SOURCE_MANIFEST_PATH.with_name(
            f".{SOURCE_MANIFEST_PATH.name}.tmp"
        )
        source_manifest_rows = []
        try:
            with sample_temporary.open("w", encoding="utf-8") as output:
                for spec in SOURCE_SPECS:
                    source_key = (
                        spec["source_config"],
                        spec["source_split"],
                    )
                    retained = 0
                    source_index = 0
                    source_characters = 0
                    analysis_characters = 0
                    truncated_count = 0
                    try:
                        stream = load_dataset(
                            DATASET_ID,
                            spec["source_config"],
                            split=spec["source_split"],
                            revision=DATASET_REVISION,
                            streaming=True,
                        )
                        for source_index, source_row in enumerate(stream):
                            doc_id = str(source_row.get("doc_id") or "")
                            original_text = str(source_row.get("text") or "")
                            if not doc_id or not original_text.strip():
                                continue
                            text = original_text[:MAX_ANALYSIS_CHARACTERS]
                            was_truncated = len(original_text) > MAX_ANALYSIS_CHARACTERS
                            source_type = source_row.get("type")
                            if source_type is not None:
                                source_type = str(source_type)
                            record = {
                                "doc_id": doc_id,
                                "source_config": spec["source_config"],
                                "source_split": spec["source_split"],
                                "claimed_language": spec["claimed_language"],
                                "claimed_script": spec["claimed_script"],
                                "source_index": source_index,
                                "type": source_type,
                                "source_character_count": len(original_text),
                                "analysis_character_count": len(text),
                                "truncated": was_truncated,
                                "text": text,
                            }
                            output.write(_json.dumps(record, ensure_ascii=False) + "\n")
                            retained += 1
                            source_characters += len(original_text)
                            analysis_characters += len(text)
                            truncated_count += int(was_truncated)
                            if retained == DOCUMENTS_PER_SOURCE:
                                break
                    except Exception as exc:
                        raise RuntimeError(
                            "Unable to stream the required pinned source "
                            f"{source_key[0]}/{source_key[1]}: {exc}"
                        ) from exc
                    if retained != DOCUMENTS_PER_SOURCE:
                        raise RuntimeError(
                            "Pinned source ended before the requested 20 non-empty "
                            f"rows: {source_key[0]}/{source_key[1]} ({retained})"
                        )
                    source_manifest_rows.append(
                        {
                            "source_config": spec["source_config"],
                            "source_split": spec["source_split"],
                            "claimed_language": spec["claimed_language"],
                            "claimed_script": spec["claimed_script"],
                            "role": spec["role"],
                            "documents": retained,
                            "source_characters": source_characters,
                            "analysis_characters": analysis_characters,
                            "truncated": truncated_count,
                        }
                    )
            manifest = {
                "dataset_id": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "license": LICENSE,
                "selection_rule": (
                    "For each fixed source/configuration pair, retain the first 20 "
                    "stream rows with a non-empty doc_id and non-whitespace text; "
                    "preserve stream order; do not shuffle or inspect predictions."
                ),
                "max_analysis_characters": MAX_ANALYSIS_CHARACTERS,
                "materialized_at_utc": _utc_now(),
                "sources": source_manifest_rows,
            }
            manifest_temporary.write_text(
                _json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            sample_temporary.replace(SAMPLE_PATH)
            manifest_temporary.replace(SOURCE_MANIFEST_PATH)
        finally:
            if sample_temporary.exists():
                sample_temporary.unlink()
            if manifest_temporary.exists():
                manifest_temporary.unlink()
        return _sample_summary(SAMPLE_PATH)

    def _normalize_script_name(script):
        return "Taml" if script == "Tamil" else script

    def _safe_alpha_count(value):
        return sum(character.isalpha() for character in value or "")

    # Names exported from a Marimo cell must not begin with an underscore.
    fasttext_module = _fasttext
    script_separate = _script_separate
    script_predict = _script_predict
    LinguaLanguage = _LinguaLanguage
    LanguageDetectorBuilder = _LanguageDetectorBuilder
    utc_now = _utc_now
    sha256_file = _sha256_file
    download_to = _download_to
    sample_is_complete = _sample_is_complete
    sample_summary = _sample_summary
    materialize_microset = _materialize_microset
    normalize_script_name = _normalize_script_name
    safe_alpha_count = _safe_alpha_count
    json_module = _json
    return (
        DATASET_ID,
        DATASET_REVISION,
        DOCUMENTS_PER_SOURCE,
        MAX_ANALYSIS_CHARACTERS,
        SAMPLE_PATH,
        SOURCE_MANIFEST_PATH,
        DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        FASTTEXT_MODEL_PATH,
        INDICLID_ARCHIVE_PATH,
        INDICLID_EXTRACT_DIR,
        NOTEBOOK_PATH,
        FASTTEXT_URL,
        INDICLID_URL,
        LICENSE,
        SOURCE_SPECS,
        EXPECTED_SCRIPTS,
        EXPECTED_INDICLID_LABELS,
        FASTTEXT_LANGUAGE_LABELS,
        fasttext_module,
        script_separate,
        script_predict,
        LinguaLanguage,
        LanguageDetectorBuilder,
        utc_now,
        sha256_file,
        download_to,
        sample_is_complete,
        sample_summary,
        materialize_microset,
        normalize_script_name,
        safe_alpha_count,
        json_module,
    )


@app.cell(hide_code=True)
def dataset_and_source_table(mo, DATASET_ID, DATASET_REVISION, SOURCE_SPECS):
    source_table = [
        {
            "config": spec["source_config"],
            "split": spec["source_split"],
            "claimed_language": spec["claimed_language"],
            "claimed_script": spec["claimed_script"],
            "role": spec["role"],
            "selection": "first 20 non-empty rows in pinned stream",
        }
        for spec in SOURCE_SPECS
    ]
    mo.vstack(
        [
            mo.md(
                f"""
## 1. Dataset picked

**Dataset:** [`{DATASET_ID}`](https://huggingface.co/datasets/ai4bharat/sangraha)  
**Revision:** `{DATASET_REVISION}`  
**License recorded for this source-derived sample:** `CC-BY-4.0`  
**Scope:** 8 fixed source/configuration pairs × 20 rows = 160 bounded records.

The `verified` language split names and the `synthetic` transliteration split
names are retained as claims. They are not treated as gold labels.
                """
            ),
            mo.ui.table(source_table),
            mo.md(
                "No network request occurs until the materialization button is "
                "clicked."
            ),
        ]
    )


@app.cell(hide_code=True)
def materialization_controls(mo):
    materialize_button = mo.ui.run_button(
        label="Materialize 160-document Sangraha micro-corpus"
    )
    mo.vstack(
        [
            mo.md(
                "This is a source-derived analysis sample, not a second copy of "
                "the 30M-token Hindi corpus. Existing complete sample and manifest "
                "files are reused without a network request."
            ),
            materialize_button,
        ]
    )
    return (materialize_button,)


@app.cell(hide_code=True)
def materialize_sample(
    mo,
    materialize_button,
    SAMPLE_PATH,
    SOURCE_MANIFEST_PATH,
    DOCUMENTS_PER_SOURCE,
    sample_is_complete,
    sample_summary,
    materialize_microset,
):
    sample_ready = sample_is_complete()
    materialization_error = None
    if not sample_ready and materialize_button.value:
        try:
            materialize_microset()
            sample_ready = sample_is_complete()
        except Exception as exc:  # noqa: BLE001 - surface any materialization failure
            materialization_error = str(exc)
            sample_ready = False
    sample_stats = sample_summary(SAMPLE_PATH) if sample_ready else []
    if materialization_error:
        status = f"Materialization failed: `{materialization_error}`"
    elif sample_ready:
        status = (
            f"Using a complete cached sample: {sum(row['documents'] for row in sample_stats)} "
            f"documents across {len(sample_stats)} sources; no network request was needed."
        )
    elif materialize_button.value:
        status = "Materialization did not produce the required eight 20-row sources."
    else:
        status = "Click the button to stream the bounded source-derived sample."
    mo.vstack(
        [
            mo.md(status),
            mo.ui.table(sample_stats),
            mo.md(
                f"Sample path: `{SAMPLE_PATH}`  \n"
                f"Source manifest: `{SOURCE_MANIFEST_PATH}`  \n"
                f"Required count per source: `{DOCUMENTS_PER_SOURCE}`"
            ),
        ]
    )
    return sample_ready, sample_stats, materialization_error


@app.cell(hide_code=True)
def technique_chapters(mo):
    mo.vstack(
        [
            mo.md(
                r"""
## 2. Technique stack — script identification and histogramming

**What it is.** Writing-system identification assigns characters to Unicode
ISO 15924 scripts. It is evidence about the writing system, not a language
classifier: Devanagari can represent Hindi or Marathi, and Latin can represent
English, code, or transliterated Hindi/Telugu.

**How it works internally.** This notebook calls
[`GlotScript.sp`](https://github.com/ZurichNLP/GlotScript) for the dominant
script, share, and script-distribution details, then calls `GlotScript.sc` to
separate script spans. The implementation converts those values to JSON-safe
counts and never treats a Latin result as English evidence.

**Pros.** Fast, language-independent, interpretable, and useful as a routing
prefilter. **Cons/failure modes.** Shared scripts are ambiguous; short strings,
digits, punctuation, names, and mixed text can make the dominant histogram
misleading. **Use it when** script plausibility is a first validation signal,
not when language identity alone is needed. **This notebook uses it** on every
record and keeps the script share in the audit output.

Reference: Kargaran et al., *GlotScript: A Resource and Tool for Low Resource
System Identification* ([ACL Anthology](https://aclanthology.org/2024.lrec-main.687/)).
                """
            ),
            mo.md(
                r"""
## 3. Technique stack — document-level fastText LID

**What it is.** The official fastText language-identification model is a
compact character n-gram classifier with `lid.176.ftz` coverage for 176 labels.
It returns labels and probabilities and is designed for high-throughput text.

**How it works internally.** Subword character n-grams feed a linear classifier;
UTF-8 input is required. This comparison requests the top three labels and maps
`en/hi/bn/ta/te/ur` to ISO-639-3 names while preserving unsupported labels.

**Pros.** Small model, fast inference, broad language coverage. **Cons/failure
modes.** Short documents, code, transliteration, similar languages, and
corpus/domain shift can produce confident-looking errors. **Use it when** a
broad document-level baseline is useful and the input distribution is audited.
**This notebook uses it** as one independent signal, never as a sole keep/drop
rule.

Reference: [fastText language identification documentation](https://fasttext.cc/docs/en/language-identification.html).
                """
            ),
            mo.md(
                r"""
## 4. Technique stack — IndicLID-FTN

**What it is.** AI4Bharat's IndicLID-FTN is an Indic-aware fast linear
classifier for 22 Indian languages plus English and `Other`, with language-plus-
script labels such as `hin_Deva` and `hin_Latn`. It is the comparison detector
intended to recognize romanized Indic text.

**How it works internally.** The released fastText-compatible binary predicts
a ranked label list. The notebook retains the exact model label, parses its
language and script, and normalizes the model's `Tamil` script spelling to ISO
15924 `Taml` only for comparison. The published roughly 169 MB archive is
downloaded only on demand.

**Pros.** Indic language/script coverage and romanization awareness. **Cons/
failure modes.** It is still a statistical classifier; short, noisy, mixed, or
domain-shifted text can fail, and this 160-row comparison is not an accuracy
benchmark. **Use it when** Indic native and Latin-script coverage is in scope.
**This notebook uses it** to compare source claims and route transliteration to
a dedicated review pool rather than auto-deleting it.

Reference: Madhani et al., *Bhasha-Abhijnaanam: Native-script and romanized
Language Identification for 22 Indic languages* ([ACL Anthology](https://aclanthology.org/2023.acl-short.71/)).
                """
            ),
            mo.md(
                r"""
## 5. Technique stack — constrained Lingua LID

**What it is.** Lingua combines script/rule prefilters with statistical scoring
from character n-grams. This notebook builds one high-accuracy detector over
only `ENGLISH`, `HINDI`, `BENGALI`, `TAMIL`, `TELUGU`, and `URDU`.

**How it works internally.** Candidate languages are restricted before
1–5-gram scoring. The output label is converted to lowercase ISO-639-3 and the
full six-language confidence list is retained. A Lingua confidence is
normalized among this constrained set; it is not comparable to fastText or
IndicLID probabilities.

**Pros.** Transparent candidate policy and useful short-text comparison.
**Cons/failure modes.** Restriction creates an open-world blind spot; a language
outside the six may be forced into the closest candidate. **Use it when** the
allowed-language policy is explicit. **This notebook uses it** as a bounded
second generic detector and reports the policy beside every result.

Reference: [Lingua documentation](https://github.com/pemistahl/lingua-py).
                """
            ),
            mo.md(
                r"""
## 6. Technique stack — segment-level multilingual detection

**What it is.** A single document label hides code switching. Segment detection
returns spans with separate language predictions, which can expose a Hindi /
English or Telugu / English mixture.

**How it works internally.** The notebook calls Lingua's
`detect_multiple_languages_of`, drops spans with fewer than 30 alphabetic
characters, computes shares over the remaining span characters, and marks an
experimental mixed-language candidate only when at least two languages each
own 20% or more.

**Pros.** Makes code switching visible and supports a review route. **Cons/failure
modes.** Span boundaries and short-span decisions are model-dependent; this is
illustrative, not a production segmenter. **Use it when** multilingual behavior
is in scope and deleting a mixed record would be costly. **This notebook uses it**
before native-script keep logic and never deletes a mixed candidate.
                """
            ),
            mo.md(
                r"""
## 7. Technique stack — metadata validation, routing, and manual audit

**What it is.** A split/configuration label is metadata, not proof. Validation
compares the claim with script evidence and three LID views, then routes
ambiguous records to a review pool.

**How it works internally.** The policy preserves `hin_Latn` and `tel_Latn` as
transliteration review, routes mixed spans before native-script decisions,
quarantines short/low-share/script-mismatch records, keeps only mutually
confident native-script agreement, and records a reroute candidate or manual
review otherwise. The routing contract cell asserts all five branches.

**Pros.** Conservative, inspectable, and useful for downstream human review.
**Cons/failure modes.** Thresholds are policy choices, a six-language detector
is not open-world, and source-label agreement is not accuracy. **Use it when**
metadata and model evidence must remain distinguishable. **This notebook uses
it** for every source row and exports a bounded CSV with previews for review.

The human-audit motivation follows [IndicLLMSuite](https://github.com/AI4Bharat/IndicLLMSuite),
its [web/PDF audit resources](https://github.com/AI4Bharat/IndicLLMSuite#data-auditing),
and the production-scale caution in Caswell et al., *Language ID in the Wild*
([ACL Anthology](https://aclanthology.org/2020.coling-main.579/)).
                """
            ),
            mo.md(
                r"""
## 8. Technique catalogue — omitted alternatives

**CLD3** is a compact neural language identifier suited to fast web/document
classification, but this workspace has no pinned Python-3.12-compatible CLD3
binding/runtime, so it is documented rather than executed. **AfroLID** is
valuable for African-language coverage, but its intended coverage does not
match this selected Indic/English comparison, so it is also omitted.

The production choice should follow model coverage, training/domain fit,
script and romanization behavior, and auditability—not package popularity. The
broader curation context is discussed by [IndicLLMSuite](https://aclanthology.org/2024.acl-long.843/),
its multilingual-web audit, and [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)
([paper](https://arxiv.org/abs/2309.09400)).
                """
            ),
        ]
    )


@app.cell(hide_code=True)
def model_controls(mo):
    fasttext_download_button = mo.ui.run_button(label="Download fastText lid.176.ftz")
    indiclid_download_button = mo.ui.run_button(label="Download IndicLID-FTN")
    mo.vstack(
        [
            mo.md(
                "The model downloads are independent, explicit operations. Existing "
                "non-empty artifacts are hashed and reused."
            ),
            mo.hstack([fasttext_download_button, indiclid_download_button]),
        ]
    )
    return fasttext_download_button, indiclid_download_button


@app.cell(hide_code=True)
def download_fasttext_model(
    mo,
    fasttext_download_button,
    FASTTEXT_MODEL_PATH,
    FASTTEXT_URL,
    download_to,
    sha256_file,
):
    fasttext_ready = False
    fasttext_hash = None
    fasttext_error = None
    if fasttext_download_button.value or (
        FASTTEXT_MODEL_PATH.exists() and FASTTEXT_MODEL_PATH.stat().st_size > 0
    ):
        try:
            fasttext_hash = download_to(FASTTEXT_MODEL_PATH, FASTTEXT_URL)
            # Hashing is the integrity check recorded for reproducibility. The
            # exact value is deliberately stored in summary.json, not guessed.
            fasttext_ready = True
        except Exception as exc:  # noqa: BLE001 - surface download failures in the UI
            fasttext_error = str(exc)
            fasttext_ready = False
    if fasttext_ready:
        fasttext_status = (
            f"fastText is ready ({FASTTEXT_MODEL_PATH.stat().st_size:,} bytes; "
            f"SHA-256 `{fasttext_hash}`)."
        )
    elif fasttext_error:
        fasttext_status = f"fastText unavailable: `{fasttext_error}`"
    else:
        fasttext_status = "Click the button to download the 917 kB fastText model."
    mo.md(fasttext_status)
    return fasttext_ready, fasttext_hash, fasttext_error


@app.cell(hide_code=True)
def download_indiclid_model(
    mo,
    indiclid_download_button,
    INDICLID_ARCHIVE_PATH,
    INDICLID_EXTRACT_DIR,
    INDICLID_URL,
    download_to,
    sha256_file,
):
    import shutil
    import tempfile
    import zipfile
    from pathlib import Path as _Path

    indiclid_ready = False
    indiclid_zip_hash = None
    indiclid_bin_hash = None
    indiclid_model_path = None
    indiclid_error = None
    should_check = indiclid_download_button.value or (
        INDICLID_ARCHIVE_PATH.exists()
        and INDICLID_ARCHIVE_PATH.stat().st_size > 0
    )
    if should_check:
        try:
            indiclid_zip_hash = download_to(INDICLID_ARCHIVE_PATH, INDICLID_URL)
            existing_bins = (
                sorted(INDICLID_EXTRACT_DIR.rglob("*.bin"))
                if INDICLID_EXTRACT_DIR.exists()
                else []
            )
            if len(existing_bins) != 1:
                extraction_parent = _Path(
                    tempfile.mkdtemp(prefix="indiclid-ftn-", dir=INDICLID_EXTRACT_DIR.parent)
                )
                try:
                    with zipfile.ZipFile(INDICLID_ARCHIVE_PATH) as archive:
                        archive.extractall(extraction_parent)
                    discovered_bins = sorted(extraction_parent.rglob("*.bin"))
                    if len(discovered_bins) != 1:
                        discovered = [
                            str(path.relative_to(extraction_parent))
                            for path in sorted(extraction_parent.rglob("*"))
                            if path.is_file()
                        ]
                        raise RuntimeError(
                            "IndicLID archive must contain exactly one .bin artifact; "
                            f"discovered files: {discovered}"
                        )
                    if INDICLID_EXTRACT_DIR.exists():
                        shutil.rmtree(INDICLID_EXTRACT_DIR)
                    extraction_parent.replace(INDICLID_EXTRACT_DIR)
                    extraction_parent = None
                finally:
                    if extraction_parent is not None and extraction_parent.exists():
                        shutil.rmtree(extraction_parent)
            existing_bins = sorted(INDICLID_EXTRACT_DIR.rglob("*.bin"))
            if len(existing_bins) != 1:
                discovered = [
                    str(path.relative_to(INDICLID_EXTRACT_DIR))
                    for path in sorted(INDICLID_EXTRACT_DIR.rglob("*"))
                    if path.is_file()
                ]
                raise RuntimeError(
                    "IndicLID extracted directory must contain exactly one .bin; "
                    f"discovered files: {discovered}"
                )
            indiclid_model_path = existing_bins[0]
            indiclid_bin_hash = sha256_file(indiclid_model_path)
            indiclid_ready = True
        except Exception as exc:  # noqa: BLE001 - surface archive failures in the UI
            indiclid_error = str(exc)
            indiclid_ready = False
    if indiclid_ready:
        indiclid_status = (
            f"IndicLID-FTN is ready (`{indiclid_model_path}`; archive SHA-256 "
            f"`{indiclid_zip_hash}`; binary SHA-256 `{indiclid_bin_hash}`)."
        )
    elif indiclid_error:
        indiclid_status = f"IndicLID-FTN unavailable: `{indiclid_error}`"
    else:
        indiclid_status = "Click the button to download and inspect the IndicLID-FTN archive."
    mo.md(indiclid_status)
    return (
        indiclid_ready,
        indiclid_model_path,
        indiclid_zip_hash,
        indiclid_bin_hash,
        indiclid_error,
    )


@app.cell(hide_code=True)
def detector_stack(
    FASTTEXT_MODEL_PATH,
    EXPECTED_SCRIPTS,
    EXPECTED_INDICLID_LABELS,
    FASTTEXT_LANGUAGE_LABELS,
    fasttext_module,
    script_separate,
    script_predict,
    LinguaLanguage,
    LanguageDetectorBuilder,
    normalize_script_name,
    safe_alpha_count,
):

    constrained_languages = (
        LinguaLanguage.ENGLISH,
        LinguaLanguage.HINDI,
        LinguaLanguage.BENGALI,
        LinguaLanguage.TAMIL,
        LinguaLanguage.TELUGU,
        LinguaLanguage.URDU,
    )
    LINGUA_DETECTOR = LanguageDetectorBuilder.from_languages(
        *constrained_languages
    ).build()

    def _blank_fasttext():
        return {"label": None, "confidence": 0.0, "top3": []}

    def _blank_indiclid():
        return {
            "label": None,
            "language": None,
            "script": None,
            "confidence": 0.0,
            "top3": [],
        }

    def _strip_fasttext_label(raw_label):
        value = str(raw_label)
        return value.removeprefix("__label__")

    def analyze_script(text):
        """Return GlotScript script evidence as JSON-compatible primitives."""
        text = str(text or "")
        if safe_alpha_count(text) == 0:
            return {
                "dominant_script": "Zzzz",
                "dominant_script_share": 0.0,
                "script_details": {},
                "script_spans": [],
            }
        dominant, share, details = script_predict(text)
        script_details = {
            str(script): float(value)
            for script, value in (details.get("details") or {}).items()
        }
        separated = script_separate(text)
        script_spans = [
            {
                "script": str(script),
                "character_count": int(safe_alpha_count(span_text)),
            }
            for script, span_text in separated.items()
            if safe_alpha_count(span_text) > 0
        ]
        return {
            "dominant_script": str(dominant or "Zzzz"),
            "dominant_script_share": float(share or 0.0),
            "script_details": script_details,
            "script_spans": script_spans,
        }

    def detect_fasttext(text, model=None):
        """Predict a broad document language with fastText top-three labels."""
        text = str(text or "")
        if not text.strip():
            return _blank_fasttext()
        detector = model or fasttext_module.load_model(str(FASTTEXT_MODEL_PATH))
        model_input = text.replace("\r", " ").replace("\n", " ")
        labels, probabilities = detector.predict(model_input, k=3)
        top3 = []
        for raw_label, probability in zip(labels, probabilities):
            original = _strip_fasttext_label(raw_label)
            top3.append(
                {
                    "label": FASTTEXT_LANGUAGE_LABELS.get(original, original),
                    "confidence": float(probability),
                }
            )
        if not top3:
            return _blank_fasttext()
        return {
            "label": top3[0]["label"],
            "confidence": top3[0]["confidence"],
            "top3": top3,
        }

    def _parse_indiclid_label(raw_label):
        label = _strip_fasttext_label(raw_label)
        pieces = label.split("_", 1)
        language = pieces[0] if pieces else None
        raw_script = pieces[1] if len(pieces) == 2 else None
        return label, language, normalize_script_name(raw_script)

    def detect_indiclid(text, model=None):
        """Predict Indic language-plus-script labels from IndicLID-FTN."""
        text = str(text or "")
        if not text.strip():
            return _blank_indiclid()
        detector = model
        if detector is None:
            raise RuntimeError(
                "IndicLID model is not loaded; click Download IndicLID-FTN first."
            )
        model_input = text.replace("\r", " ").replace("\n", " ")
        labels, probabilities = detector.predict(model_input, k=3)
        top3 = []
        for raw_label, probability in zip(labels, probabilities):
            label, language, script = _parse_indiclid_label(raw_label)
            top3.append(
                {
                    "label": label,
                    "language": language,
                    "script": script,
                    "confidence": float(probability),
                }
            )
        if not top3:
            return _blank_indiclid()
        leading = top3[0]
        return {
            "label": leading["label"],
            "language": leading["language"],
            "script": leading["script"],
            "confidence": leading["confidence"],
            "top3": top3,
        }

    def _lingua_label(language):
        if language is None:
            return None
        return language.iso_code_639_3.name.lower()

    def detect_lingua(text, detector=None):
        """Return constrained Lingua prediction and all six confidence values."""
        text = str(text or "")
        if not text.strip():
            return {"label": None, "confidence": 0.0, "confidence_values": []}
        detector = detector or LINGUA_DETECTOR
        language = detector.detect_language_of(text)
        confidence_values = detector.compute_language_confidence_values(text)
        values = [
            {
                "label": _lingua_label(value.language),
                "confidence": float(value.value),
            }
            for value in confidence_values
        ]
        if language is None or not values:
            return {"label": None, "confidence": 0.0, "confidence_values": values}
        detected_label = _lingua_label(language)
        leading = next(
            (value for value in values if value["label"] == detected_label),
            values[0],
        )
        return {
            "label": detected_label,
            "confidence": float(leading["confidence"]),
            "confidence_values": values,
        }

    def detect_mixed_spans(text, detector=None):
        """Return an experimental mixed-language candidate from Lingua spans."""
        text = str(text or "")
        if not text.strip():
            return {
                "is_mixed_candidate": False,
                "language_shares": {},
                "spans": [],
            }
        detector = detector or LINGUA_DETECTOR
        try:
            detected_spans = detector.detect_multiple_languages_of(text)
        except Exception:  # noqa: BLE001 - mixed-span detection is an optional signal
            detected_spans = []
        retained = []
        for span in detected_spans:
            segment = text[span.start_index : span.end_index]
            alphabetic_characters = safe_alpha_count(segment)
            if alphabetic_characters < 30:
                continue
            character_count = len(segment)
            label = _lingua_label(span.language)
            if not label or character_count == 0:
                continue
            retained.append(
                {
                    "start_index": int(span.start_index),
                    "end_index": int(span.end_index),
                    "language": label,
                    "character_count": character_count,
                    "alphabetic_characters": alphabetic_characters,
                }
            )
        total_characters = sum(span["character_count"] for span in retained)
        shares = {}
        if total_characters:
            for span in retained:
                label = span["language"]
                shares[label] = shares.get(label, 0.0) + (
                    span["character_count"] / total_characters
                )
        is_mixed = sum(share >= 0.20 for share in shares.values()) >= 2
        return {
            "is_mixed_candidate": bool(is_mixed),
            "language_shares": {
                label: float(share) for label, share in sorted(shares.items())
            },
            "spans": retained,
        }

    def _reroute_candidate(claimed_language, script, fasttext, indiclid, lingua):
        predicted = fasttext.get("label")
        return bool(
            predicted
            and predicted != claimed_language
            and predicted == lingua.get("label")
            and fasttext.get("confidence", 0.0) >= 0.70
            and lingua.get("confidence", 0.0) >= 0.70
            and indiclid.get("language") == predicted
            and indiclid.get("confidence", 0.0) >= 0.70
            and indiclid.get("script") == EXPECTED_SCRIPTS.get(predicted)
            and script.get("dominant_script") == EXPECTED_SCRIPTS.get(predicted)
            and script.get("dominant_script_share", 0.0) >= 0.80
        )

    def validate_record(record, script, fasttext, indiclid, lingua, mixed):
        """Apply the explicit metadata-validation and routing contract."""
        claimed_language = record.get("claimed_language")
        claimed_script = record.get("claimed_script")
        source_split = record.get("source_split")
        dominant_script = script.get("dominant_script", "Zzzz")
        dominant_share = float(script.get("dominant_script_share", 0.0))
        reasons = []
        base = {
            "doc_id": record.get("doc_id"),
            "source_config": record.get("source_config"),
            "source_split": source_split,
            "claimed_language": claimed_language,
            "claimed_script": claimed_script,
            "source_index": record.get("source_index"),
            "source_character_count": record.get("source_character_count"),
            "analysis_character_count": record.get("analysis_character_count"),
            "script": script,
            "fasttext": fasttext,
            "indiclid": indiclid,
            "lingua": lingua,
            "mixed": mixed,
        }
        if source_split in {"hin_Latn", "tel_Latn"}:
            return {
                **base,
                "validation_status": "transliterated_review",
                "route": "transliteration_pool",
                "reasons": ["latin_script_is_not_language_evidence"],
            }
        if mixed.get("is_mixed_candidate", False):
            return {
                **base,
                "validation_status": "mixed_language_review",
                "route": "mixed_language_pool",
                "reasons": ["mixed_language_candidate"],
            }
        if dominant_share < 0.80:
            reasons.append("dominant_script_share_below_0_80")
        if safe_alpha_count(str(record.get("text") or "")) < 50:
            reasons.append("insufficient_alphabetic_characters")
        if dominant_script != claimed_script:
            reasons.append("dominant_script_differs_from_claim")
        if reasons:
            return {
                **base,
                "validation_status": "quarantine",
                "route": "quarantine",
                "reasons": reasons,
            }
        native_script_match = dominant_script == claimed_script and dominant_share >= 0.80
        indic_match = (
            indiclid.get("language") == claimed_language
            and indiclid.get("script") == claimed_script
            and indiclid.get("confidence", 0.0) >= 0.70
        )
        fasttext_match = (
            fasttext.get("label") == claimed_language
            and fasttext.get("confidence", 0.0) >= 0.70
        )
        lingua_match = (
            lingua.get("label") == claimed_language
            and lingua.get("confidence", 0.0) >= 0.70
        )
        if native_script_match and indic_match and fasttext_match and lingua_match:
            return {
                **base,
                "validation_status": "validated_keep",
                "route": "claimed_pool",
                "reasons": ["confident_claimed_language_and_script_agreement"],
            }
        if _reroute_candidate(claimed_language, script, fasttext, indiclid, lingua):
            predicted = fasttext["label"]
            return {
                **base,
                "validation_status": "reroute_candidate",
                "route": f"{predicted}_pool",
                "reasons": ["confident_non_claimed_language_agreement"],
            }
        return {
            **base,
            "validation_status": "review_required",
            "route": "manual_review",
            "reasons": ["detector_disagreement_or_shared_script_ambiguity"],
        }

    return (
        LINGUA_DETECTOR,
        analyze_script,
        detect_fasttext,
        detect_indiclid,
        detect_lingua,
        detect_mixed_spans,
        validate_record,
    )


@app.cell(hide_code=True)
def load_cached_models(
    mo,
    fasttext_module,
    fasttext_ready,
    FASTTEXT_MODEL_PATH,
    indiclid_ready,
    indiclid_model_path,
):
    fasttext_model = None
    indiclid_model = None
    model_load_error = None
    if fasttext_ready and indiclid_ready:
        try:
            fasttext_model = fasttext_module.load_model(str(FASTTEXT_MODEL_PATH))
            indiclid_model = fasttext_module.load_model(str(indiclid_model_path))
        except Exception as exc:  # noqa: BLE001 - surface model-load failures in the UI
            model_load_error = str(exc)
            fasttext_model = None
            indiclid_model = None
    models_ready = fasttext_model is not None and indiclid_model is not None
    if models_ready:
        status_view = mo.md("Both model artifacts loaded and are ready for the comparison.")
    elif model_load_error:
        status_view = mo.md(
            f"Model loading failed; comparison is disabled: `{model_load_error}`"
        )
    else:
        status_view = mo.md("Download both model artifacts before running the comparison.")
    status_view  # noqa: B018 - Marimo renders the final UI expression
    return fasttext_model, indiclid_model, models_ready, model_load_error


@app.cell(hide_code=True)
def diagnostic_fixtures(
    mo,
    analyze_script,
    detect_fasttext,
    detect_indiclid,
    detect_lingua,
    detect_mixed_spans,
    LINGUA_DETECTOR,
    fasttext_model,
    indiclid_model,
):
    fixtures = [
        {
            "fixture_id": "hindi_english_devanagari_latin",
            "text": (
                "यह हिंदी वाक्य है जिसमें स्थानीय समाचार, शिक्षा और सार्वजनिक सेवा की "
                "जानकारी स्पष्ट रूप से दी गई है। This English paragraph explains how a "
                "public information service works, why readers compare sources, and how "
                "a careful language audit records uncertainty."
            ),
            "expected_behavior": "multiple scripts; mixed candidate after eligible-span thresholds",
        },
        {
            "fixture_id": "telugu_english_telugu_latin",
            "text": (
                "ఇది తెలుగు వాక్యం మరియు ప్రజా సేవల గురించి వివరించే సమాచార భాగం. "
                "The English section describes a transparent data review, records the "
                "source language, and keeps code-mixed material available for human audit."
            ),
            "expected_behavior": "multiple scripts; mixed candidate after eligible-span thresholds",
        },
        {
            "fixture_id": "latin_script_hindi",
            "text": (
                "Aaj mausam bahut achha hai aur hum bazaar ja rahe hain kyunki ghar ke liye "
                "sabzi aur kitaaben kharidni hain."
            ),
            "expected_behavior": "Latin is not proof of English; IndicLID is the romanization-aware comparison",
        },
    ]
    fixture_rows = []
    for fixture in fixtures:
        _fixture_script = analyze_script(fixture["text"])
        _fixture_lingua = detect_lingua(fixture["text"], LINGUA_DETECTOR)
        _fixture_mixed = detect_mixed_spans(fixture["text"], LINGUA_DETECTOR)
        _fixture_fasttext = (
            detect_fasttext(fixture["text"], fasttext_model)
            if fasttext_model is not None
            else {"label": "not_downloaded", "confidence": 0.0, "top3": []}
        )
        _fixture_indiclid = (
            detect_indiclid(fixture["text"], indiclid_model)
            if indiclid_model is not None
            else {
                "label": "not_downloaded",
                "language": None,
                "script": None,
                "confidence": 0.0,
                "top3": [],
            }
        )
        fixture_rows.append(
            {
                "fixture_id": fixture["fixture_id"],
                "scripts": ", ".join(
                    span["script"] for span in _fixture_script["script_spans"]
                ),
                "dominant_script": _fixture_script["dominant_script"],
                "dominant_share": round(_fixture_script["dominant_script_share"], 3),
                "fastText": _fixture_fasttext["label"],
                "IndicLID": _fixture_indiclid["label"],
                "Lingua": _fixture_lingua["label"],
                "mixed_candidate": _fixture_mixed["is_mixed_candidate"],
                "expected_behavior": fixture["expected_behavior"],
            }
        )
    mo.vstack(
        [
            mo.md(
                "## Diagnostic fixtures (not Sangraha evidence)\n\n"
                "These are exactly three fixed texts. The first two are designed "
                "to expose script mixing; the Latin Hindi case demonstrates why "
                "generic LID disagreement is expected."
            ),
            mo.ui.table(fixture_rows),
        ]
    )
    return fixtures, fixture_rows


@app.cell(hide_code=True)
def routing_contract(mo, validate_record):
    def _contract_record(**overrides):
        base = {
            "doc_id": "contract",
            "source_config": "verified",
            "source_split": "eng",
            "claimed_language": "eng",
            "claimed_script": "Latn",
            "source_index": 0,
            "source_character_count": 100,
            "analysis_character_count": 100,
            "text": "A sufficiently long alphabetic document for the routing contract.",
        }
        base.update(overrides)
        return base

    blank = validate_record(
        _contract_record(text="short"),
        {"dominant_script": "Zzzz", "dominant_script_share": 0.0, "script_spans": []},
        {"label": None, "confidence": 0.0, "top3": []},
        {"label": None, "language": None, "script": None, "confidence": 0.0, "top3": []},
        {"label": None, "confidence": 0.0, "confidence_values": []},
        {"is_mixed_candidate": False, "language_shares": {}, "spans": []},
    )
    native = validate_record(
        _contract_record(
            claimed_language="hin",
            claimed_script="Deva",
            text="यह हिंदी पाठ है और यह पर्याप्त लंबा वर्णमाला वाला परीक्षण दस्तावेज है जिसमें भाषा की जांच, स्रोत का सत्यापन, लेख की समीक्षा और स्पष्ट अभिलेख शामिल हैं।",
        ),
        {"dominant_script": "Deva", "dominant_script_share": 0.96, "script_spans": [{"script": "Deva", "character_count": 60}]},
        {"label": "hin", "confidence": 0.91, "top3": []},
        {"label": "hin_Deva", "language": "hin", "script": "Deva", "confidence": 0.93, "top3": []},
        {"label": "hin", "confidence": 0.89, "confidence_values": []},
        {"is_mixed_candidate": False, "language_shares": {"hin": 1.0}, "spans": []},
    )
    reroute = validate_record(
        _contract_record(
            claimed_language="hin",
            # The synthetic branch keeps the dominant script equal to the
            # predicted native script so the documented script-mismatch
            # quarantine check does not mask rerouting precedence.
            claimed_script="Beng",
        ),
        {"dominant_script": "Beng", "dominant_script_share": 0.94, "script_spans": [{"script": "Beng", "character_count": 70}]},
        {"label": "ben", "confidence": 0.90, "top3": []},
        {"label": "ben_Beng", "language": "ben", "script": "Beng", "confidence": 0.92, "top3": []},
        {"label": "ben", "confidence": 0.88, "confidence_values": []},
        {"is_mixed_candidate": False, "language_shares": {"ben": 1.0}, "spans": []},
    )
    transliteration = validate_record(
        _contract_record(source_split="hin_Latn", claimed_language="hin", claimed_script="Latn"),
        {"dominant_script": "Latn", "dominant_script_share": 0.99, "script_spans": [{"script": "Latn", "character_count": 80}]},
        {"label": "en", "confidence": 0.80, "top3": []},
        {"label": "hin_Latn", "language": "hin", "script": "Latn", "confidence": 0.84, "top3": []},
        {"label": "en", "confidence": 0.81, "confidence_values": []},
        {"is_mixed_candidate": False, "language_shares": {"eng": 1.0}, "spans": []},
    )
    _mixed_case = validate_record(
        _contract_record(claimed_language="hin", claimed_script="Deva"),
        {"dominant_script": "Deva", "dominant_script_share": 0.80, "script_spans": [{"script": "Deva", "character_count": 50}, {"script": "Latn", "character_count": 30}]},
        {"label": "hin", "confidence": 0.80, "top3": []},
        {"label": "hin_Deva", "language": "hin", "script": "Deva", "confidence": 0.80, "top3": []},
        {"label": "hin", "confidence": 0.80, "confidence_values": []},
        {"is_mixed_candidate": True, "language_shares": {"eng": 0.45, "hin": 0.55}, "spans": []},
    )
    cases = [
        ("blank_or_short", "quarantine", blank),
        ("matching_native_script", "validated_keep", native),
        ("confident_non_claim_same_script", "reroute_candidate", reroute),
        ("latin_source_split", "transliterated_review", transliteration),
        ("mixed_spans", "mixed_language_review", _mixed_case),
    ]
    contract_rows = []
    for name, expected, _contract_result in cases:
        assert _contract_result["validation_status"] == expected, (name, _contract_result)
        contract_rows.append(
            {
                "case": name,
                "expected_status": expected,
                "actual_status": _contract_result["validation_status"],
                "route": _contract_result["route"],
                "reasons": "; ".join(_contract_result["reasons"]),
            }
        )
    mo.vstack(
        [
            mo.md(
                "## Routing contract\n\nAll five synthetic cases assert the "
                "mutually exclusive status precedence before this table renders."
            ),
            mo.ui.table(contract_rows),
        ]
    )
    return (contract_rows,)


@app.cell(hide_code=True)
def comparison_controls(mo):
    comparison_button = mo.ui.run_button(label="Run language-ID comparison")
    mo.vstack(
        [
            mo.md(
                "The comparison requires the complete cached micro-corpus and both "
                "loaded model artifacts. It writes three reproducible local files."
            ),
            comparison_button,
        ]
    )
    return (comparison_button,)


@app.cell(hide_code=True)
def run_comparison(
    mo,
    comparison_button,
    sample_ready,
    SAMPLE_PATH,
    SOURCE_MANIFEST_PATH,
    RESULTS_DIR,
    NOTEBOOK_PATH,
    fasttext_ready,
    indiclid_ready,
    models_ready,
    fasttext_hash,
    indiclid_zip_hash,
    indiclid_bin_hash,
    FASTTEXT_URL,
    INDICLID_URL,
    fasttext_model,
    indiclid_model,
    LINGUA_DETECTOR,
    analyze_script,
    detect_fasttext,
    detect_indiclid,
    detect_lingua,
    detect_mixed_spans,
    validate_record,
    json_module,
    sha256_file,
    utc_now,
):
    import csv
    import time
    from collections import Counter, defaultdict
    from importlib.metadata import PackageNotFoundError, version

    predictions = []
    aggregate_rows = []
    document_rows = []
    audit_rows = []
    comparison_error = None
    output_paths = {}
    summary = None
    if comparison_button.value:
        if not sample_ready:
            comparison_error = "Materialize the complete micro-corpus first."
        elif not fasttext_ready or not indiclid_ready or not models_ready:
            comparison_error = "Both model artifacts must be downloaded and loaded first."
        else:
            try:
                with SAMPLE_PATH.open("r", encoding="utf-8") as handle:
                    records = [
                        json_module.loads(line)
                        for line in handle
                        if line.strip()
                    ]
                timings = defaultdict(float)
                for _record in records:
                    start = time.perf_counter()
                    _script = analyze_script(_record["text"])
                    timings["script"] += time.perf_counter() - start
                    start = time.perf_counter()
                    fasttext_prediction = detect_fasttext(_record["text"], fasttext_model)
                    timings["fasttext"] += time.perf_counter() - start
                    start = time.perf_counter()
                    indiclid_prediction = detect_indiclid(_record["text"], indiclid_model)
                    timings["indiclid"] += time.perf_counter() - start
                    start = time.perf_counter()
                    lingua_prediction = detect_lingua(_record["text"], LINGUA_DETECTOR)
                    timings["lingua"] += time.perf_counter() - start
                    start = time.perf_counter()
                    mixed_prediction = detect_mixed_spans(_record["text"], LINGUA_DETECTOR)
                    timings["mixed_spans"] += time.perf_counter() - start
                    _result = validate_record(
                        _record,
                        _script,
                        fasttext_prediction,
                        indiclid_prediction,
                        lingua_prediction,
                        mixed_prediction,
                    )
                    _result["timings_seconds"] = {
                        "script": round(float(timings["script"]), 6),
                        "fasttext": round(float(timings["fasttext"]), 6),
                        "indiclid": round(float(timings["indiclid"]), 6),
                        "lingua": round(float(timings["lingua"]), 6),
                        "mixed_spans": round(float(timings["mixed_spans"]), 6),
                    }
                    predictions.append(_result)
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                predictions_path = RESULTS_DIR / "document_predictions.jsonl"
                _atomic_predictions = predictions_path.with_name(
                    f".{predictions_path.name}.tmp"
                )
                try:
                    with _atomic_predictions.open("w", encoding="utf-8") as handle:
                        for _result in predictions:
                            handle.write(json_module.dumps(_result, ensure_ascii=False) + "\n")
                    _atomic_predictions.replace(predictions_path)
                finally:
                    if _atomic_predictions.exists():
                        _atomic_predictions.unlink()

                by_source = defaultdict(list)
                for _result in predictions:
                    by_source[
                        (_result["source_config"], _result["source_split"])
                    ].append(_result)
                for (source_config, source_split), source_rows in by_source.items():
                    claimed_language = source_rows[0]["claimed_language"]
                    claimed_script = source_rows[0]["claimed_script"]
                    documents = len(source_rows)
                    native_rows = [
                        row
                        for row in source_rows
                        if row["source_split"] not in {"hin_Latn", "tel_Latn"}
                    ]
                    native_script_agreement = (
                        sum(
                            row["script"]["dominant_script"] == claimed_script
                            and row["script"]["dominant_script_share"] >= 0.80
                            for row in native_rows
                        )
                        / len(native_rows)
                        if native_rows
                        else None
                    )
                    fasttext_agreement = sum(
                        row["fasttext"]["label"] == claimed_language
                        for row in source_rows
                    )
                    indiclid_agreement = sum(
                        row["indiclid"]["label"]
                        == {"eng": "eng_Latn", "hin": "hin_Deva", "ben": "ben_Beng", "tam": "tam_Tamil", "tel": "tel_Telu", "urd": "urd_Arab"}.get(claimed_language)
                        for row in source_rows
                    )
                    lingua_agreement = sum(
                        row["lingua"]["label"] == claimed_language
                        for row in source_rows
                    )
                    generic_agreement = sum(
                        row["fasttext"]["label"] == claimed_language
                        and row["lingua"]["label"] == claimed_language
                        for row in source_rows
                    )
                    route_counts = dict(
                        sorted(Counter(row["route"] for row in source_rows).items())
                    )
                    aggregate_rows.append(
                        {
                            "source": f"{source_config}/{source_split}",
                            "claimed_language": claimed_language,
                            "claimed_script": claimed_script,
                            "documents": documents,
                            "native_script_documents": len(native_rows),
                            "mean_analysis_characters": round(
                                sum(row["analysis_character_count"] for row in source_rows)
                                / documents,
                                1,
                            ),
                            "dominant_script_agreement_rate": (
                                round(native_script_agreement, 3)
                                if native_script_agreement is not None
                                else None
                            ),
                            "fastText_claimed_agreement_rate": round(
                                fasttext_agreement / documents, 3
                            ),
                            "fastText_mean_confidence": round(
                                sum(row["fasttext"]["confidence"] for row in source_rows)
                                / documents,
                                3,
                            ),
                            "IndicLID_exact_language_script_agreement_rate": round(
                                indiclid_agreement / documents, 3
                            ),
                            "IndicLID_mean_confidence": round(
                                sum(row["indiclid"]["confidence"] for row in source_rows)
                                / documents,
                                3,
                            ),
                            "Lingua_claimed_agreement_rate": round(
                                lingua_agreement / documents, 3
                            ),
                            "Lingua_mean_confidence": round(
                                sum(row["lingua"]["confidence"] for row in source_rows)
                                / documents,
                                3,
                            ),
                            "generic_detector_agreement_rate": round(
                                generic_agreement / documents, 3
                            ),
                            "route_counts": route_counts,
                        }
                    )

                records_by_id = {_record["doc_id"]: _record for _record in records}
                selected_audit = [
                    row for row in predictions if row["validation_status"] != "validated_keep"
                ]
                for source_rows in by_source.values():
                    keep_controls = sorted(
                        (
                            row
                            for row in source_rows
                            if row["validation_status"] == "validated_keep"
                        ),
                        key=lambda row: __import__("hashlib").sha256(
                            row["doc_id"].encode("utf-8")
                        ).hexdigest(),
                    )[:5]
                    selected_audit.extend(keep_controls)
                deduplicated_audit = {}
                for row in selected_audit:
                    deduplicated_audit[row["doc_id"]] = row
                audit_fieldnames = [
                    "doc_id",
                    "source_config",
                    "source_split",
                    "claimed_language",
                    "claimed_script",
                    "validation_status",
                    "route",
                    "reasons",
                    "dominant_script",
                    "dominant_script_share",
                    "fasttext_label",
                    "fasttext_confidence",
                    "indiclid_label",
                    "indiclid_language",
                    "indiclid_script",
                    "indiclid_confidence",
                    "lingua_label",
                    "lingua_confidence",
                    "mixed_language_candidate",
                    "preview",
                    "reviewer_language",
                    "reviewer_script",
                    "reviewer_route",
                    "reviewer_notes",
                ]
                for row in sorted(
                    deduplicated_audit.values(),
                    key=lambda item: (
                        item["source_config"],
                        item["source_split"],
                        item["doc_id"],
                    ),
                ):
                    original = records_by_id[row["doc_id"]]
                    audit_rows.append(
                        {
                            "doc_id": row["doc_id"],
                            "source_config": row["source_config"],
                            "source_split": row["source_split"],
                            "claimed_language": row["claimed_language"],
                            "claimed_script": row["claimed_script"],
                            "validation_status": row["validation_status"],
                            "route": row["route"],
                            "reasons": ";".join(row["reasons"]),
                            "dominant_script": row["script"]["dominant_script"],
                            "dominant_script_share": row["script"]["dominant_script_share"],
                            "fasttext_label": row["fasttext"]["label"],
                            "fasttext_confidence": row["fasttext"]["confidence"],
                            "indiclid_label": row["indiclid"]["label"],
                            "indiclid_language": row["indiclid"]["language"],
                            "indiclid_script": row["indiclid"]["script"],
                            "indiclid_confidence": row["indiclid"]["confidence"],
                            "lingua_label": row["lingua"]["label"],
                            "lingua_confidence": row["lingua"]["confidence"],
                            "mixed_language_candidate": row["mixed"]["is_mixed_candidate"],
                            "preview": original["text"][:500],
                            "reviewer_language": "",
                            "reviewer_script": "",
                            "reviewer_route": "",
                            "reviewer_notes": "",
                        }
                    )
                audit_path = RESULTS_DIR / "manual_audit.csv"
                audit_temporary = audit_path.with_name(f".{audit_path.name}.tmp")
                try:
                    with audit_temporary.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=audit_fieldnames)
                        writer.writeheader()
                        writer.writerows(audit_rows)
                    audit_temporary.replace(audit_path)
                finally:
                    if audit_temporary.exists():
                        audit_temporary.unlink()

                source_manifest = json_module.loads(
                    SOURCE_MANIFEST_PATH.read_text(encoding="utf-8")
                )
                def package_version(distribution):
                    try:
                        return version(distribution)
                    except PackageNotFoundError:
                        return None

                statuses = dict(
                    sorted(Counter(row["validation_status"] for row in predictions).items())
                )
                routes = dict(
                    sorted(Counter(row["route"] for row in predictions).items())
                )
                claimed_mix = dict(
                    sorted(Counter(row["claimed_language"] for row in predictions).items())
                )
                kept_mix = dict(
                    sorted(
                        Counter(
                            row["claimed_language"]
                            for row in predictions
                            if row["validation_status"] == "validated_keep"
                        ).items()
                    )
                )
                summary = {
                    "dataset_id": source_manifest.get("dataset_id"),
                    "dataset_revision": source_manifest.get("dataset_revision"),
                    "source_manifest": source_manifest,
                    "sample_path": str(SAMPLE_PATH),
                    "model_artifacts": {
                        "fasttext": {"url": FASTTEXT_URL, "sha256": fasttext_hash},
                        "indiclid_archive": {"url": INDICLID_URL, "sha256": indiclid_zip_hash},
                        "indiclid_binary": {"url": INDICLID_URL, "sha256": indiclid_bin_hash},
                    },
                    "notebook_sha256": sha256_file(NOTEBOOK_PATH)
                    if NOTEBOOK_PATH.exists()
                    else None,
                    "versions": {
                        "marimo": package_version("marimo"),
                        "GlotScript": package_version("GlotScript") or package_version("glotscript"),
                        "fasttext-wheel": package_version("fasttext-wheel"),
                        "lingua-language-detector": package_version("lingua-language-detector"),
                    },
                    "run_timestamp_utc": utc_now(),
                    "timings_seconds": {
                        key: round(float(value), 6) for key, value in sorted(timings.items())
                    },
                    "aggregate_counts": {
                        "documents": len(predictions),
                        "statuses": statuses,
                        "routes": routes,
                        "claimed_language_mix_before": claimed_mix,
                        "claimed_language_mix_after_validated_keep": kept_mix,
                        "exact_duplicate_count": 0,
                        "near_duplicate_count": 0,
                        "pii_category_counts": {},
                        "decontamination_matches_removed": 0,
                    },
                    "agreement_metric_note": (
                        "All agreement rates compare predictions with Sangraha source "
                        "split metadata; they are not accuracy estimates."
                    ),
                    "outputs": {
                        "document_predictions": str(predictions_path),
                        "summary": str(RESULTS_DIR / "summary.json"),
                        "manual_audit": str(audit_path),
                    },
                }
                summary_path = RESULTS_DIR / "summary.json"
                _atomic_summary = summary_path.with_name(f".{summary_path.name}.tmp")
                try:
                    _atomic_summary.write_text(
                        json_module.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    _atomic_summary.replace(summary_path)
                finally:
                    if _atomic_summary.exists():
                        _atomic_summary.unlink()
                output_paths = {
                    "predictions": str(predictions_path),
                    "summary": str(summary_path),
                    "manual_audit": str(audit_path),
                }
            except Exception as exc:  # noqa: BLE001 - surface comparison failures in the UI
                comparison_error = str(exc)

    if comparison_error:
        comparison_view = mo.md(f"Comparison unavailable: `{comparison_error}`")
    elif not comparison_button.value:
        comparison_view = mo.md(
            "Click **Run language-ID comparison** after both model downloads are ready."
        )
    else:
        document_rows = [
            {
                "doc_id": row["doc_id"],
                "claimed": f"{row['claimed_language']}/{row['claimed_script']}",
                "dominant_script": row["script"]["dominant_script"],
                "script_share": round(row["script"]["dominant_script_share"], 3),
                "fastText": f"{row['fasttext']['label']} ({row['fasttext']['confidence']:.2f})",
                "IndicLID": f"{row['indiclid']['label']} ({row['indiclid']['confidence']:.2f})",
                "Lingua": f"{row['lingua']['label']} ({row['lingua']['confidence']:.2f})",
                "mixed": row["mixed"]["is_mixed_candidate"],
                "status": row["validation_status"],
                "route": row["route"],
                "reasons": ";".join(row["reasons"]),
            }
            for row in predictions
        ]
        comparison_view = mo.vstack(
            [
                mo.md(
                    "## Final statistics\n\n"
                    "Aggregate tables are primary. Every agreement metric means "
                    "**agreement with Sangraha's source split, not accuracy**. "
                    "The document table omits raw text; previews are limited to the "
                    "manual-audit CSV."
                ),
                mo.ui.table(aggregate_rows),
                mo.md(
                    f"Processed `{len(predictions)}` rows. Outputs: "
                    f"`{output_paths.get('predictions')}`, "
                    f"`{output_paths.get('summary')}`, "
                    f"`{output_paths.get('manual_audit')}`"
                ),
                mo.ui.table(document_rows),
            ]
        )
    comparison_view  # noqa: B018 - Marimo renders the final UI expression
    return predictions, summary, aggregate_rows, document_rows, audit_rows


if __name__ == "__main__":
    app.run()
