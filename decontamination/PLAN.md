# Decontamination workbook — plan

## Scope

Stage 7 of the Session 4 eight-stage pipeline, as its own bounded technique
study, matching the shape of `language-identification/`, `quality-filtering/`
and `pii-scrubbing/`: a Marimo notebook, a deterministic Sangraha slice paired
with a real held-out eval suite, a technique reference with primary-source
citations, hand-authored diagrams, and measured before/after metrics.

Decontamination is the one stage whose output is a *number about the evaluation*
rather than a cleaner corpus. Almost every document survives it. The deliverable
is therefore an audit: how much of the held-out set is reachable from the
training pool, by which mechanism, and what the leak would have been worth.

The notebook is a **technique study**, not a replacement for the proxy check in
`assignment.py`. That proxy scans for benchmark *names* (`GSM8K`, `MMLU`) — a
string match against metadata, not against evaluation content. This workbook
replaces it with a measurement against the actual held-out text and reports what
the proxy was missing.

## Dataset contract

Two sides, because contamination is a relation between two corpora and neither
side alone is measurable.

**Corpus side** — `ai4bharat/sangraha`, revision
`8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8`, config `verified`, split `hin`,
CC-BY-4.0. 120 fixed-stride windows of 100 consecutive rows across the whole
1,048,577-row split: `offset_i = i * floor(total / 120)`. Stream order preserved,
no shuffling, and critically **no contamination-conditioned selection** — the
stride is a function of split size alone, so the reported hit rate is an estimate
rather than a curated demonstration.

Stride windows rather than the existing 264-document `sample_raw.jsonl` for the
same reason the PII workbook widened its slice: contamination is rare, and a
detector study needs enough corpus surface for the rate to mean anything. The
rule is disclosed and rebuildable, so widening is not cherry-picking.

**Held-out side** — three Hindi evaluation sets, chosen because they sit at
different points on the risk surface:

| Eval set | Shape | Why it is here |
|---|---|---|
| `IndicQA` (`indicqa.hi`) | long Wikipedia-derived passages | Highest-risk surface. Eval contexts are open-web prose, so overlap with a web crawl is a genuine possibility rather than a hypothetical. |
| `MILU` (Hindi, `murthyrudra/milu-cleaned`) | short exam stems | Tests whether a 13-gram window can even span an evaluation item. Most stems are shorter than the window, which is a structural blind spot, not a tuning problem. |
| `XNLI` (`hi` test) | translated premise/hypothesis pairs | Translationese. A near-duplicate detector should score these differently from native web prose, and the difference is the point. |

Answer keys are cached in a separate field from stems, because leaking the answer
key is a strictly worse event than leaking the question, and the notebook reports
them separately.

Both caches are gitignored; `data/probe_manifest.json` carries row counts,
offsets, licences and SHA-256 digests, and `data/build_decontamination_cache.py`
rebuilds them byte-identically.

## Notebook section order

1. **Scope and boundary** — what stage 7 is, the eight-stage table with stage 7
   marked executed, and an explicit statement of what this notebook cannot
   claim (a bounded slice bounds the estimate, not the risk).
2. **What decontamination is** — the firewall definition, the train-split vs
   test-split asymmetry, why the stage exists (a contaminated benchmark does not
   merely flatter the model, it removes the only instrument for telling whether
   the model is any good), and why it is decided at sourcing rather than
   cleaning.
3. **Technique reference** — one subsection per technique. Each carries, in
   fixed order: what it is / where it was used, with citations and a one-line
   note on *how* each org used it / how it works internally / when to use it /
   pros and cons.
4. **Diagrams** — hand-authored SVG, one per mechanism worth seeing rather than
   reading.
5. **Sangraha measurement** — the ladder run against the eval suite, per-rung
   metrics, and the inflation model that converts a hit count into a benchmark
   score delta.
6. **The Indic findings** — the matra-stripping normalization trap and the
   short-stem blind spot, with executed evidence.
7. **What I would ship** — the chosen technique for the case study, the residual
   risk statement, and the honest gap versus the `assignment.py` proxy.
8. **Provenance** — manifest of the run.

## Techniques worth listing

Ordered by where they sit on the prevention → detection → verification axis.

| # | Technique | Family | Why it earns a section |
|---|---|---|---|
| 1 | Sourcing-time firewall and split discipline | prevention | The only technique that is free; everything below is cleanup for its absence. |
| 2 | Temporal cutoff / living benchmarks | prevention | LiveBench and LiveCodeBench make contamination structurally impossible rather than detectable. |
| 3 | Canary strings | prevention + verification | BIG-bench's GUID; the one mechanism that works on a model whose corpus you cannot read. |
| 4 | Exact n-gram overlap | detection | The industry default. GPT-3's 13-gram, Llama 3's 8-gram token marking, PaLM's 70%-of-n-grams rule. |
| 5 | Suffix-array exact substring | detection | Removes the window-size choice entirely; Lee et al.'s ExactSubstr with its 50-token floor. |
| 6 | MinHash/LSH near-duplicate | detection | Catches the edit-distance neighbourhood a hash set cannot; the same machinery as stage 5, aimed at a different target. |
| 7 | Embedding + LLM-judge cascade | detection | The rephrased-sample answer; the LMSYS decontaminator is the documented case. |
| 8 | Min-K% membership inference | detection | Works from logits alone when the corpus is unavailable. |
| 9 | Order-exchangeability test | detection | A statistical proof of contamination with a p-value; no corpus access, no logit heuristics. |
| 10 | Clean/dirty split re-evaluation | quantification | Turns a hit count into a score delta. This is the metric that decides whether the leak mattered. |
| 11 | Remove-and-report policy | policy | Whether you drop the document, mask the span, or keep it and annotate; the choice with the largest corpus cost. |

Deliberately excluded and stated as such: machine unlearning, inference-time
suppression, and differential-privacy training. All three are real responses to
memorization, none is a corpus-cleaning stage, and conflating them makes the
stage boundary mushy.

## Technique to implement for Sangraha

A **three-rung ladder plus a quantification step**, in this order, because each
rung is the honest baseline for the next:

1. **Indic-safe normalization** — NFC, casefold, strip punctuation *by Unicode
   category while preserving `Mn`/`Mc` combining marks*. This is the rung that
   matters most and the one that is easiest to get wrong: a naive `[^\w\s]`
   strip destroys Devanagari matras and the virama, which silently fabricates
   overlap between unrelated Hindi documents. The notebook measures both
   variants and reports the false-positive count the naive version produces.
2. **Exact 13-gram overlap** — GPT-3's parameter, hash-set membership, document
   flagged on any shared n-gram. Reported per eval set and per n-gram size
   (8/13/20) so the recall/precision trade-off is visible rather than asserted.
3. **MinHash/LSH near-duplicate against eval items** — 128 permutations,
   banded, Jaccard threshold swept, to catch the paraphrase and partial-quote
   band that rung 2 structurally cannot see.
4. **Clean/dirty re-evaluation** — partition the eval suite into the items that
   *are* reachable from the corpus and the items that are not, then report what
   a memorizing model's score would look like on each. This is GPT-3's own
   quantification method and it is what converts the audit into a decision.

Embedding + LLM-judge is documented but **not executed**, and the notebook says
why: it needs an eval-set-conditioned index and a judge model, and running it on
a bounded slice would produce a number with no error bars. Naming the gap is
more useful than filling it badly.

## Metrics to show

- Corpus surface: documents, characters, estimated tokens in the probe; eval
  items and characters per eval set.
- Per-rung flag counts and rates, per eval set, with the incremental yield of
  each rung over the one above it.
- Contamination *reach*: the fraction of held-out items that are reachable from
  the probe, which is the number that actually bounds benchmark trust — distinct
  from the fraction of corpus documents flagged.
- n-gram size sweep (8/13/20) showing flag count and manually-audited precision.
- Naive-vs-Indic-aware A/B: matra-stripping normalization versus category-aware,
  with the exact count of fabricated matches the naive rule produces.
- Stem-length distribution against the n-gram window, quantifying how many MILU
  items are structurally undetectable by 13-gram matching.
- Inflation model: reported score on the dirty subset versus the clean subset
  under a stated memorization assumption, with the assumption labelled as an
  assumption.
- Cost: corpus documents that would be removed, and the token loss, if every
  flag were enforced as a drop.
- A false-positive audit of every flagged span, read by hand, by cause.

## Diagram requirements

Hand-authored SVG, no rendering library. One diagram per mechanism:

1. The firewall — three-way data separation with the train/test asymmetry.
2. N-gram overlap internals — shingling, the hash set, and the collision.
3. The matra trap — the same Hindi string through both normalizers, character by
   character, showing where the false match is manufactured.
4. MinHash/LSH banding — why banding turns a similarity threshold into a
   collision probability.
5. The clean/dirty inflation model — how a hit count becomes a score delta.

Every diagram must be legible at notebook width, use one consistent palette,
label the mechanism rather than decorate it, and carry a caption stating what
the reader should take away.
