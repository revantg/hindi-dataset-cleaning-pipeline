# PII scrubbing workbook — plan

## Scope

Stage 6 of the Session 4 eight-stage pipeline, as its own bounded technique
study, matching the shape of `language-identification/` and `quality-filtering/`:
a Marimo notebook plus a deterministic Sangraha slice, a technique reference with
real citations, hand-drawn SVG diagrams, and measured before/after metrics.

The notebook is a **technique study**, not a replacement for the canonical
scrubber in `assignment.py`. It measures what each detector family actually does
to real Hindi text and reports the failure modes, including the ones that make
the canonical scrubber wrong.

## Dataset contract

- Source: `ai4bharat/sangraha`, revision `8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8`, CC-BY-4.0.
- Probe cache: fixed stride windows across the whole `hin` split for both the
  `verified` and `unverified` configs — 100 windows x 100 consecutive rows each,
  stream order preserved, no shuffling and no PII-conditioned selection.
  Built by `data/build_pii_probe_cache.py`, provenance in `data/probe_manifest.json`.
- Why both configs: `verified` is the split the case study trains on;
  `unverified` is the higher-risk pool. Comparing PII density across the two is
  the lecture's "judge by source" heuristic made measurable.
- Why stride windows rather than the existing 264-document `sample_raw.jsonl`:
  that sample contains **zero** regex-detectable PII. A stage whose whole point
  is detection cannot be studied on a slice with no positives. The stride rule is
  disclosed and rebuildable, so widening the sample is not cherry-picking.

## Notebook section order

1. **Scope and boundary** — what stage 6 is, what this notebook does and does
   not claim, the eight-stage table with stage 6 marked executed.
2. **What PII scrubbing is** — definition, the two reasons it exists (people and
   legal usability), where it sits in the pipeline, why it is a *policy* with a
   boundary rather than "delete anything personal".
3. **Technique reference** — one subsection per technique. Each carries: what it
   is / where it was used with citations and one-liners / how it works
   internally / when to use it / pros and cons.
4. **SVG diagrams** — the layered detector cascade, the span-conflict
   resolution mechanism, the precision-recall geometry, and the Devanagari
   digit-class trap.
5. **Sangraha measurement** — density scan, the technique ladder run on the
   probe cache, per-technique metrics.
6. **The Indic findings** — Devanagari digits and mixed-script boundaries, with
   executed evidence.
7. **What I would ship** — the chosen technique for the case study and why,
   plus the residual-risk statement.
8. **Provenance** — manifest of the run.

## Techniques worth listing

Ordered by where they sit on the precision/recall/cost surface.

| # | Technique | Family | Why it earns a section |
|---|---|---|---|
| 1 | Structured regex | rule | The near-100%-precision floor of every real pipeline; Dolma's actual method |
| 2 | Checksum / format validation | rule | Turns a shape match into an arithmetic fact; Luhn, Verhoeff for Aadhaar |
| 3 | Context windows + gazetteers | rule | Cheap recall lift on labelled identifiers; Presidio's context enhancer |
| 4 | Denylist / allowlist suppression | rule | The precision brake; StarCoder's IP allowlist is the canonical trick |
| 5 | Statistical sequence labelling (CRF) | ML | The historical baseline; explains IOB and why span decoding exists |
| 6 | Transformer token classification (NER) | ML | The only way to catch names; StarPII is the documented case |
| 7 | Cascade / two-stage routing | architecture | How to afford NER at corpus scale |
| 8 | Document-level drop vs span masking | policy | The decision Dolma actually parameterises |
| 9 | Placeholder / surrogate replacement | policy | What replaces the span, and the tokenizer cost of getting it wrong |
| 10 | LLM-as-judge | ML | The current frontier and honestly where the evidence is thin |
| 11 | Memorization auditing | verification | Scrubbing without an extraction test is unverified |

Deliberately excluded and stated as such: differential privacy training,
k-anonymity/generalisation, and format-preserving encryption — real privacy
techniques that are not corpus-scrubbing stages.

## Technique to implement for Sangraha

A **four-layer cascade**, in this order:

1. Unicode digit folding on a **detection copy only** (never on the emitted
   text) so Devanagari-digit identifiers become visible to numeric patterns.
2. Structured regex on the folded copy, with offsets mapped back to the original
   string — email, Indian phone, IPv4, credential, Aadhaar-shaped, PAN-shaped.
3. Checksum gates — Verhoeff for Aadhaar-shaped runs, Luhn for card-shaped runs
   — to convert a shape match into a decision.
4. Allowlist suppression for the false positives Hindi news text actually
   produces: dates, prices, pincodes, phone-shaped years, RBI/helpline numbers.

Names are **detected and counted, not masked**, using IndicNER as an audited
review signal. Justification is the lecture's own boundary: an Indic name model
that confuses मैसूर for a person destroys legitimate vocabulary, and Hindi news
prose is dense with public figures who are not PII under this threat model. The
notebook reports the measured false-positive rate that supports that choice
rather than asserting it.

## Metrics to show

- PII density by category, per config (`verified` vs `unverified`), per 1M tokens.
- Documents touched, spans masked, characters removed — and the fraction of the
  corpus that changes at all (expected: very small, which is itself the finding).
- Ladder table: spans caught at each cascade layer, incremental yield per layer.
- Precision/recall on a hand-labelled gold set drawn from the probe cache
  (label every candidate span the union of detectors proposes; that is a
  candidate-recall bound, disclosed as such, not corpus truth).
- The naive-vs-Indic-aware A/B: ASCII-only patterns versus the folded cascade,
  showing the exact count of identifiers the naive version misses.
- False-positive audit: every span the strict layers propose that a human read
  says is not PII, by cause.
- Token-cost delta from placeholder choice.

## SVG requirements

Hand-authored, no rendering library. Every diagram must be legible at the
notebook's width, use a single consistent palette, label the mechanism rather
than decorate it, and carry a caption stating what the reader should take away.

---

## Executed outcome

Notebook: `pii_scrubbing.py` (44 cells). Metrics regenerated by
`evaluate_cascade.py`, which reads only `data/` — every table in the notebook is
read from `data/metrics.json` rather than transcribed.

### Where the measurement contradicted this plan

1. **Dolma's weakness is precision, not recall.** The plan assumed the naive
   baseline would lose on recall. Faithfully ported, Dolma reaches the *same*
   recall as the shipped cascade (0.967) at precision 0.303 vs 0.454. Its phone
   pattern is simply too permissive for Hindi news prose.
2. **The checksum layer contributes exactly zero on this slice.** Only three
   checksum-eligible spans exist in 18,400 documents, and all three fail. Kept
   and documented rather than deleted: it is inert on news text but load-bearing
   on forms. Verhoeff correctness is property-tested in-notebook instead of
   assumed.
3. **Dolma's placeholder costs more tokens than the PII it removes** — 7.0 per
   span against 5.5 for the raw identifier, so the scrubbed corpus is *larger*.
   The plan expected a saving.
4. **Precision 0.454 is not a detector defect.** 92% of false positives are
   correctly-parsed *institutional* contact points. The residual limit is a
   semantic judgement (institutional vs personal) that no rule-based layer can
   settle — a more useful finding than the raw number.

### Findings that held

- `unverified` carries **2.2×** the PII density of `verified`, making the
  lecture's *judge by source* heuristic measurable.
- Digit folding is what catches the last identifiers; offset preservation held
  across all 18,400 documents.
- Every IPv4-shaped span in the cache is a Sanskrit scripture citation, not an
  address — the recall fix created a script-specific precision problem.
- The stage touches under 0.6% of documents: low-yield, high-stakes.

### SVG validation

Each diagram was validated three ways, and iterated until all three were clean:

1. **Geometric** — a throwaway checker parsed each SVG and measured every `<text>`
   run with real Arial/Menlo/Devanagari metrics from `fontTools`, then compared it
   against its containing rect: overflow, vertical escape, same-baseline
   collisions, rects off-canvas. Negative-tested by injecting a known overflow to
   confirm it actually fires. Final state: **0 issues across all five.**
2. **Visual** — rendered to PNG at 2000px and read back. This is what caught the
   defects geometry cannot see: garbled Devanagari from hand-computed numeric
   character references (`&#2404;` is a danda, not a digit), an off-centre Venn,
   a citation orphaning `§4` onto its own line, and a footer that read as another
   citation.
3. **Factual** — `test_diagram_claims.py` re-derives every hard-coded claim from
   `pii_techniques` and Python's `re`. This caught two diagrams asserting things
   the code does not do:
   - `\d{12}` was shown matching *space-separated* Devanagari years. It does not —
     `\d{12}` needs 12 consecutive digits. Replaced with the verified asymmetry:
     `[6-9]\d{9}` misses the mobile while `\d{10}` matches the very same digits.
   - The SUPPRESSED example was a year run that no pattern even proposes.
     Replaced with the real corpus case: `२.९.२१.१` folding to a valid dotted
     quad and being suppressed by the native-digit rule.

   The cascade figure's input document now runs through `scrub()` and produces
   exactly the three outcomes drawn beneath it, asserted span-by-span.

Structural fixes: uniform `BOTTOM_PAD` in `_frame` (no border flush to the canvas
edge), a `_wrap`/`_lines` helper measured against a character budget since SVG
does not wrap text, and citations moved into their own field so placement is
deliberate rather than a side effect of where a description happens to break.
