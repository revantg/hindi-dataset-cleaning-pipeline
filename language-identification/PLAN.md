## Context

Create a standalone Marimo workbook for Session 4 Stage 3, language identification and validation, modeled on `stepwise_cleaning.py`'s explanation-first, bounded-execution pattern. It will download only a deterministic Sangraha micro-corpus, demonstrate executable script detection, document-level LID, segment-level multilingual detection, metadata validation, routing, and audit preparation, then persist reproducible results beneath its own `language-identification/data/` directory.

Existing `assignment.py` has a narrow `_detect_language(text)` heuristic for Hindi/Latin-script Hindi (`assignment.py:337`); this workbook will not alter that assignment pipeline. This is an independently runnable, comparative technique notebook that makes mechanisms, limits, and results visible rather than claiming an unvalidated production language classifier.

## Approach

### 1. Add the notebook and pinned runtime dependencies

1. Create `language-identification/language_identification.py` as a Marimo 0.23.16 app with `app = marimo.App(width="medium")`. Follow `stepwise_cleaning.py`'s cell structure: hidden explanatory Markdown cells; explicit `mo.ui.run_button` controls before every network/materialization operation; visible bounded execution/result tables.
2. Add these direct runtime dependencies to `pyproject.toml` and regenerate `uv.lock`:
   - `GlotScript==2.0` — ISO 15924 script identification/separation.
   - `fasttext-wheel==0.9.2` — Python bindings for the official fastText model and IndicLID-FTN.
   - `lingua-language-detector==2.2.0` — constrained document and mixed-span LID.
3. Create `language-identification/data/.gitignore` with exactly `models/` and `results/`. The 917 kB fastText binary, the 169 MB IndicLID-FTN release, and every rerunnable result file remain local; the small source-derived sample and its source manifest are retained in this notebook-owned data directory. Do not add a second copy of the 30M-token Hindi corpus or modify `data/sangraha_hi_verified_30m/`.

### 2. Define and materialize the bounded, labelled Sangraha micro-corpus

1. In a single notebook setup cell, declare these constants exactly:
   - `DATASET_ID = "ai4bharat/sangraha"`
   - `DATASET_REVISION = "8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8"`
   - `DOCUMENTS_PER_SOURCE = 20`
   - `MAX_ANALYSIS_CHARACTERS = 4_000`
   - `SAMPLE_PATH = Path("language-identification/data/sangraha_lid_microset.jsonl")`
   - `SOURCE_MANIFEST_PATH = Path("language-identification/data/source_manifest.json")`
2. Stream each source in the following fixed order, retaining the first 20 rows with a non-empty `doc_id` and non-whitespace text. The row order from the pinned stream is the sole selection rule; do not shuffle, stratify, or pick rows after seeing predictions.

   | config | split | claimed language | claimed script | role |
   |---|---|---|---|---|
   | `verified` | `eng` | `eng` | `Latn` | English baseline |
   | `verified` | `hin` | `hin` | `Deva` | Hindi native script |
   | `verified` | `ben` | `ben` | `Beng` | Bengali native script |
   | `verified` | `tam` | `tam` | `Taml` | Tamil native script |
   | `verified` | `tel` | `tel` | `Telu` | Telugu native script |
   | `verified` | `urd` | `urd` | `Arab` | Urdu native script |
   | `synthetic` | `hin_Latn` | `hin` | `Latn` | Hindi transliteration limitation case |
   | `synthetic` | `tel_Latn` | `tel` | `Latn` | Telugu transliteration limitation case |

3. Write one JSONL row per retained document with the exact fields: `doc_id`, `source_config`, `source_split`, `claimed_language`, `claimed_script`, `source_index`, `type` (or `null`), `source_character_count`, `analysis_character_count`, `truncated`, and `text`. Store only the first 4,000 Unicode code points in `text`; set `truncated` when the original text is longer. This is an explicit bounded-analysis sample, not a new cleaned corpus.
4. Write `source_manifest.json` with the dataset id, revision, CC-BY-4.0 license, per-source configuration, split, claimed language/script, document count, selection rule, character cap, and UTC materialization timestamp. If both sample and manifest already exist and all eight source keys have exactly 20 rows, reuse them without a network request; otherwise rebuild both atomically via temporary sibling files and `Path.replace`.
5. Present a pre-download source table and a `Materialize 160-document Sangraha micro-corpus` button. After execution, display only per-source counts, character totals, truncation counts, and the two local paths; do not claim split labels are gold labels.

### 3. Explain the technique stack before running it

Write one Markdown chapter per item below, each with **what it is**, **how it works internally**, **pros**, **cons/failure modes**, **when to use it**, and **how this notebook uses or does not use it**. Cite the canonical links in the notebook text: GlotScript (Kargaran et al., 2024), fastText LID documentation, IndicLID (Madhani et al., 2023), Lingua documentation, Caswell et al. (2020) *Language ID in the Wild*, the Sangraha dataset card, and the named IndicLLMSuite, multilingual-web audit, and CulturaX references supplied by the user.

1. **Script identification / histogramming — executable:** Explain that scripts constrain possible languages but do not uniquely identify language. Use `GlotScript.sp` for dominant ISO 15924 script/confidence/details and `GlotScript.sc` for script-span separation. Call out the specific shared-script limitation: Hindi and Marathi can both be `Deva`; Latin cannot identify English versus transliterated Hindi or Telugu.
2. **Document-level fastText LID — executable:** Explain the character n-gram classifier, its `lid.176.ftz` model coverage, output labels/probabilities, speed, UTF-8 requirement, and its likely weakness on transliteration, short text, code, and corpus/domain shift.
3. **IndicLID-FTN — executable:** Explain that AI4Bharat’s fast linear classifier predicts language-plus-script labels for 22 Indian languages, English, and Other; it is the Indic-aware detector in the comparison and can recognize `hin_Latn` / `tel_Latn`. State that its published 169 MB release is intentionally downloaded only on demand, and that results on Sangraha source labels are an agreement check rather than independent accuracy evidence.
4. **Constrained Lingua LID — executable:** Explain its script/rule prefilter plus 1–5-gram statistical scoring; construct it only over English, Hindi, Bengali, Tamil, Telugu, and Urdu. Explain that candidate restriction is a policy choice that improves a bounded comparison but must not be passed off as open-world detection. Explain that Lingua confidence is normalized among this six-language set and is not comparable to fastText or IndicLID scores.
5. **Segment-level multilingual detection — executable:** Explain why document labels hide code switching. Use Lingua’s `detect_multiple_languages_of` spans, ignore spans with fewer than 30 alphabetic characters, calculate language shares by remaining span characters, and classify a text as a mixed-language candidate only when at least two languages each own at least 20% of valid span characters. Mark the feature experimental and illustrative.
6. **Metadata validation and routing — executable:** Explain that a dataset split name is a claim. Compare claimed language/script with independent script and LID evidence. Route rather than silently delete ambiguous or mixed text.
7. **Manual audit — executable export:** Explain why agreement with a Sangraha split is not accuracy. Generate an audit CSV for human review; do not report source-label agreement as an accuracy score.
8. **Technique catalogue, not executed:** Cover CLD3 and AfroLID with their intended use and limitations. Mark CLD3 as omitted because this workspace has no pinned, Python-3.12-compatible binding/runtime in the project; mark AfroLID as omitted because its coverage is intended for African languages, not the selected Indic/English comparison. State that model coverage—not package popularity—determines production selection.

### 4. Implement the four detectors and deterministic diagnostic fixtures

1. Add a `Download fastText lid.176.ftz` button. It downloads `https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz` into `language-identification/data/models/lid.176.ftz` only when absent; verify its SHA-256 after download and store the hash in results metadata. If download fails or the file is empty, show the exception and do not allow the comparison cell to run.
2. Add a separate `Download IndicLID-FTN` button. It downloads `https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftn.zip` to `language-identification/data/models/indiclid-ftn.zip`, verifies it is non-empty, extracts it into `language-identification/data/models/indiclid-ftn/`, and identifies the only recursive `*.bin` artifact. Fail with the discovered file list unless exactly one `.bin` file exists. Hash both the release ZIP and extracted `.bin`; record both hashes and the release URL in results metadata.
3. Implement `analyze_script(text) -> dict` using `GlotScript.sp` and `GlotScript.sc`. Return `dominant_script`, `dominant_script_share`, `script_details`, and `script_spans`; convert library output to JSON-compatible primitives. Return `dominant_script="Zzzz"` and share `0.0` for empty/non-letter input.
4. Implement `detect_fasttext(text) -> dict` using `fasttext.load_model`; request the top three labels and probabilities, map `en/hi/bn/ta/te/ur` to `eng/hin/ben/tam/tel/urd`, and retain unsupported top labels unchanged. Return `label`, `confidence`, and `top3`. Return `label=None`, `confidence=0.0`, and `top3=[]` for blank text.
5. Implement `detect_indiclid(text) -> dict` using `fasttext.load_model` on the extracted `.bin`; request its top three labels and probabilities. Normalize `__label__hin_Deva` style labels to `label`, `language`, `script`, `confidence`, and `top3`, preserving the model’s exact language-plus-script label. Convert the model’s `Tamil` script label to ISO 15924 `Taml`; retain `Arab`, `Beng`, `Deva`, `Latn`, and `Telu` unchanged. Return null fields and an empty `top3` for blank text.
6. Implement `detect_lingua(text) -> dict` using one constrained high-accuracy `LanguageDetectorBuilder` instance containing only `ENGLISH`, `HINDI`, `BENGALI`, `TAMIL`, `TELUGU`, and `URDU`. Return ISO-639-3 lowercase `label`, leading `confidence`, and the full constrained confidence list. Return the same null form when Lingua returns no language.
7. Implement `detect_mixed_spans(text) -> dict` using `detect_multiple_languages_of`. Drop spans with fewer than 30 alphabetic characters; compute retained-character language shares; return `is_mixed_candidate`, `language_shares`, and a JSON-safe list of retained spans. A single eligible language or two languages where one remains below 20% is not a mixed candidate.
8. Add a fixed, clearly labelled diagnostic fixture table separate from Sangraha. Include exactly three texts: one Hindi–English Devanagari/Latin mixture, one Telugu–English Telugu/Latin mixture, and one Latin-script Hindi sentence. The fixture table must show expected behavior, not corpus evidence: script detection should expose script mixing; the first two should be mixed-span candidates if their eligible spans meet the thresholds; IndicLID should be the only detector designed to identify Latin-script Hindi, while generic LID disagreement is an expected limitation.

### 5. Apply explicit metadata-validation and routing policy

1. Implement `validate_record(record, script, fasttext, indiclid, lingua, mixed) -> dict`. Its output must include the original claim, all detector fields, `validation_status`, `route`, and `reasons` as a list of literal reason codes.
2. Map predicted languages to expected scripts with the exact mapping `{"eng": "Latn", "hin": "Deva", "ben": "Beng", "tam": "Taml", "tel": "Telu", "urd": "Arab"}`. Define the exact IndicLID expected-label mapping as `{"eng": "eng_Latn", "hin": "hin_Deva", "ben": "ben_Beng", "tam": "tam_Tamil", "tel": "tel_Telu", "urd": "urd_Arab"}`. Never infer a transliterated language from `Latn`.
3. Apply these mutually exclusive decisions in this order:
   - **`transliterated_review` / `route="transliteration_pool"`** for source splits `hin_Latn` and `tel_Latn`; preserve their claimed language/script metadata, add `latin_script_is_not_language_evidence`, and never count them in native-script agreement metrics. Show IndicLID's language-plus-script result but do not make a keep/delete decision from this small study.
   - **`mixed_language_review` / `route="mixed_language_pool"`** when `is_mixed_candidate` is true; do not delete it.
   - **`quarantine` / `route="quarantine"`** when fewer than 50 alphabetic characters remain, GlotScript’s dominant script share is under `0.80`, or the dominant script differs from the claimed script; add the corresponding literal reason code.
   - **`validated_keep` / `route="claimed_pool"`** when the dominant script matches the claimed script at or above `0.80`, IndicLID predicts the claimed language and its normalized script at or above `0.70`, fastText predicts the claimed language at or above `0.70`, and Lingua predicts the claimed language at or above `0.70`.
   - **`reroute_candidate` / `route="<predicted_language>_pool"`** when fastText and Lingua predict the same non-claimed language at or above `0.70`, IndicLID predicts that language with its mapped native script at or above `0.70`, and GlotScript’s dominant script matches that predicted script at or above `0.80`.
   - **`review_required` / `route="manual_review"`** for every remaining result, including low confidence, detector disagreement, or shared-script ambiguity.
4. Add a visible routing-contract cell with five synthetic detector dictionaries and `assert` expected outcomes: blank/short input → `quarantine`; matching native-script evidence → `validated_keep`; matching confident non-claim evidence → `reroute_candidate`; a `hin_Latn` record → `transliterated_review`; and mixed spans → `mixed_language_review`. Render the case/status table after all assertions pass.
5. Do not average fastText, IndicLID, and Lingua confidence, use source-label agreement as model accuracy, automatically delete a document, or claim that this small comparison validates Sangraha’s release labels.

### 6. Run the bounded comparison and make outcomes inspectable

1. Provide a `Run language-ID comparison` button that requires the cached 160-document micro-corpus plus both fastText models. It processes only `record["text"]`, records elapsed seconds per detector, and writes local, reproducible outputs under `language-identification/data/results/`:
   - `document_predictions.jsonl` — one `validate_record` result per source row.
   - `summary.json` — source manifest fields, model/package versions, all three model SHA-256 values, notebook SHA-256, run timestamp, timings, and aggregate counts.
   - `manual_audit.csv` — review-ready bounded export.
2. Make aggregate outputs primary: show per claimed source a table of documents, mean analysis characters, dominant-script agreement rate, fastText claimed-label agreement rate and mean confidence, IndicLID exact language-script agreement rate and mean confidence, Lingua claimed-label agreement rate and mean confidence, generic-detector agreement rate, and final route counts. Label every agreement metric as **agreement with Sangraha’s source split, not accuracy**.
3. Show a compact deterministic document table containing `doc_id`, claimed language/script, dominant script/share, fastText prediction/score, IndicLID prediction/score, Lingua prediction/score, mixed-language flag, validation status, and route. Omit the raw document text from the results view; show excerpts only in the manual audit export.
4. Build `manual_audit.csv` from all non-`validated_keep` records plus the five lowest `sha256(doc_id)` `validated_keep` records per source split. Include `preview` limited to 500 Unicode code points and blank `reviewer_language`, `reviewer_script`, `reviewer_route`, and `reviewer_notes` columns. This gives a reviewer both likely failure cases and a deterministic control sample.

## Critical files & anchors

- `stepwise_cleaning.py` — bounded download controls, deterministic JSONL cache, result-table pattern, and Marimo conventions to copy.
- `assignment.py:293-354` — the existing Hindi-specific `_language_mix` and `_detect_language` implementation. Leave untouched; the new notebook explains and compares broader techniques instead.
- `pyproject.toml` — direct dependency declarations; update it and the already committed `uv.lock` together.
- `data/sangraha_hi_verified_30m/download_dataset.py:37-57` — established Sangraha revision, deterministic streaming practice, and token-estimate boundaries. Reuse the revision, not its 30M target/cache path.
- `language-identification/language_identification.py` — new standalone executable workbook implementing all notebook behavior and controls.

## Verification

1. From `/home/rev/timepass/dataset-cleaning-assignment`, run `uv sync`, then `uv run marimo check --strict language-identification/language_identification.py`. It must return zero and must not need `--fix`.
2. Start the notebook with `uv run marimo edit language-identification/language_identification.py`. In the browser, click **Materialize 160-document Sangraha micro-corpus**. Verify `language-identification/data/sangraha_lid_microset.jsonl` has exactly 160 nonempty JSON objects and `source_manifest.json` lists eight sources with 20 documents each.
3. Click **Download fastText lid.176.ftz** and **Download IndicLID-FTN**, then **Run language-ID comparison**. Verify that `document_predictions.jsonl` contains exactly 160 rows; every row has script, fastText, IndicLID, Lingua, mixed-span, status, route, and reason fields; and `summary.json` includes the source revision plus SHA-256 entries for all downloaded model artifacts.
4. Exercise the diagnostic fixtures. Verify that the Devanagari/Latin and Telugu/Latin fixtures show multiple scripts and are reported as mixed-language candidates when eligible span lengths clear the documented thresholds; verify that Latin-script Hindi is never presented as script-proven Hindi and that the UI distinguishes IndicLID from the generic detectors.
5. Verify the visible routing-contract table: blank/short input returns `quarantine`; matching native-script evidence returns `validated_keep`; confident, same-language non-claim evidence returns `reroute_candidate`; a Latin source split returns `transliterated_review`; and mixed-span input returns `mixed_language_review` before native-script keep logic.
6. Open `manual_audit.csv` and verify it contains every non-keep row plus five deterministic keep controls per each of the eight source splits, has previews no longer than 500 Unicode code points, and leaves all reviewer columns blank.

## Assumptions & contingencies

- The workbook is intentionally a **160-document technique study**, not the assignment’s 10–100M-token full cleaning run. Its results educate language-ID policy and expose limitations; they do not replace the main dataset-cleaning pipeline.
- The known Sangraha revision is reused. If one configured split is unavailable at that revision, fail the materialization cell with a table identifying the unavailable config/split; do not silently substitute a different language, config, or revision.
- If either model download URL becomes unavailable, leave its existing cached artifact usable; otherwise surface the failure and prevent the comparative run. Do not silently switch to a different model. If IndicLID’s archive does not contain exactly one `.bin` file, fail before inference and present the archive file list.
- If GlotScript or Lingua cannot import after `uv sync`, display a dependency-specific error and prevent results materialization. Do not replace them with the old `_detect_language` heuristic, because that would invalidate the promised comparison.
- Latin-script Sangraha synthetic splits demonstrate the transliteration gap. They are routed to review, never automatically called English or dropped based on a generic LID result.
