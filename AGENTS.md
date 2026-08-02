## Session 4 assignment — actual goal

Build a **small, hands-on data-cleaning case study**, not a production-scale corpus pipeline.

You must:

1. Identify the Session 4 cleaning strategies.
2. Pick or reuse a **10–100M-token** dataset. The lecture explicitly says this is an approximate token target; choose the fertility assumption yourself.
3. Apply the relevant cleanup strategies and explain the decisions.
4. Build a **widget/web report** showing the pipeline, choices, and outcomes.
5. Deploy it to **Netlify** and submit the link.

The instructor explicitly confirmed that this is about **downloading a dataset, applying cleaning techniques, then explaining what you did**—not reverse-engineering an already-cleaned dataset.

The linked `Qwen…Reasoning-Distilled` dataset is an example of how to discover data; the lecture says **do not use that exact demo dataset**. A Session 3 dataset is encouraged, and a subset of a larger corpus is allowed.

## Canonical count: **8 strategies**

The lecture’s pipeline is an eight-stage minimum set:

|   # | Strategy                       | What it means for the assignment                                                                                              |
| --: | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
|   1 | **Extract**                    | Keep actual document content; remove HTML/layout artifacts, nav, cookie banners, headers, and footers where present.          |
|   2 | **Normalize**                  | Decode HTML entities, repair encoding artifacts, normalize whitespace, remove irrelevant formatting and inconsistent markers. |
|   3 | **Language ID and validation** | Verify actual document language rather than trusting the dataset/config/folder label.                                         |
|   4 | **Quality filtering**          | Remove spam, SEO junk, repetition, malformed or very low-value text using transparent heuristics.                             |
|   5 | **Deduplication**              | Remove exact and near duplicates; lecture teaches MinHash/LSH-style similarity, not only exact text equality.                 |
|   6 | **PII scrubbing**              | Detect/redact emails, phone numbers, IPs, credentials, and other unnecessary personal identifiers.                            |
|   7 | **Decontamination**            | Avoid training on benchmark/evaluation content; report the check or clearly state the limited proxy used.                     |
|   8 | **Manifest / provenance**      | Record source, license, pipeline/version, timestamp, hashes, language mix, and final counts so the run is reproducible.       |

### Important classification

**Ghost tags / special-token cleanup is not a ninth canonical stage.** It is a crucial normalization concern: inconsistent `user`, `assistant`, chat-template, or markup markers should be removed or standardized before tokenization.

The lecture calls the eight stages the **minimum set**. Additional dataset-specific concerns are encouraged and belong in the widget’s “any other strategy or concern” section.

## Lecture guidance that matters for your Hindi/Indic dataset

- **Do not blindly strip zero-width characters.** ZWJ, ZWNJ, virama-related characters, and other Unicode components can be meaningful in Brahmic scripts.
- **Do not drop valid Hinglish/code-mixed text automatically.** The instructor explicitly says Latin-script Indic speech and natural language mixing are relevant training data when that user behavior is in scope.
- **Do not trust `hin` or another dataset label blindly.** Validate a sample and report the actual language distribution.
- **Do not over-filter.** Rules that are sensible for English—such as requiring particular punctuation or whitespace patterns—can destroy Indic text, Sanskrit, or code.
- **Deduplicate near duplicates, not only exact copies.** Similar news/PR copies are still redundant.
- **PII needs a boundary.** Emails, phones, IPs, credentials, and likely private identifiers are clear candidates. Do not apply a rule so broad that it erases place names, public figures, or legitimate content.
- **Production-scale global deduplication is not the assignment.** The lecture says true cross-corpus dedup requires a centralized large-memory index. For a 10–100M-token assignment sample, implement and disclose a local approximation.
- **Apply your own pipeline even to a previously cleaned source.** The instructor states that datasets cleaned by others still need to be evaluated through your project’s rules, especially for Indic-language handling.

## What “final statistics” should mean

The instructor intentionally did **not** prescribe a metric list. When asked, he said choosing meaningful cleanup statistics is itself part of the evaluation.

A defensible widget should show:

- documents, characters, and estimated tokens **before and after** each stage;
- removal counts/rates by reason: low quality, duplicates, PII, language mismatch, etc.;
- detected language mix before and after filtering;
- exact versus near-duplicate counts;
- PII category counts only—never expose the removed values;
- decontamination result: checked corpus/benchmark list, matching method, and matches removed; or explicitly state the limited scope;
- manifest/provenance fields: source URL, license, dataset revision/config/split, sample-selection method, pipeline version/hash, and run date.

## Required widget content

The assignment brief requires these five sections:

1. **How many strategies, and what are they?**
2. **What dataset was picked?**
3. **What was cleaned and why?**
4. **What additional strategy or concern was handled?**
5. **Final statistics.**

Then deploy to Netlify and share the link. The live assignment record specifies an individual submission with a **Netlify Link**, caption, and privacy confirmation. It has no server-side grading rubric.

## Source evidence

- Refetched full lecture transcript directly from the Session 4 Google Doc export: 705 lines, ~2h46m.
- Re-read `axiom-fetched/session4_assignment.md`.
- Spawned independent transcript and assignment analysts; both findings agree with the direct sources.
- The assignment’s size wording is clarified by the lecture at ~02:30: “around **10 to 100 million tokens**,” with a student-selected fertility ratio.
- The instructor’s direct confirmation at ~02:35: “downloading a data set and applying the data cleaning techniques … and then explaining it back to me. What did you do?”
