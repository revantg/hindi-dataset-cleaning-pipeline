import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(
        r"""
    # Session 4 — Stage 6: Removing personal information

    A bounded, executable study of **PII scrubbing**: the pipeline stage that
    removes personal identifiers from a corpus before training. It runs on a
    deterministic **18,400-document Sangraha Hindi probe cache** and reports what
    each detector family actually does to real Devanagari text.

    The stage exists for two independent reasons, and both matter:

    1. **A person is harmed by a leak.** Carlini et al. extracted verbatim
       training data — names, phone numbers, email addresses — from a deployed
       GPT-2, and later showed memorisation grows log-linearly with model
       capacity and with how often a string is duplicated. A phone number left in
       a shard is a phone number the model can be made to say.
    2. **The corpus stays legally usable.** Under India's DPDP Act 2023 and the
       GDPR, personal data in a training corpus is a processing activity that
       needs a basis. Scrubbing is what keeps a dataset shippable.

    > **What this notebook claims.** Every number below is computed in-notebook
    > from the probe cache and a hand-labelled gold set. Recall figures are a
    > **candidate-recall bound**, not corpus truth — the gold set labels spans
    > that detectors proposed, so PII no detector proposes is invisible to the
    > metric. This is stated at every table rather than buried once.

    > **What it does not claim.** It is a technique study, not a replacement for
    > the canonical scrubber in `assignment.py`. Where the measurement contradicts
    > the canonical implementation, the notebook says so and shows the evidence.
    """
    )


@app.cell(hide_code=True)
def runtime_setup():
    import json
    import sys
    from collections import Counter
    from pathlib import Path

    NOTEBOOK_DIR = Path("pii-scrubbing")
    DATA_DIR = NOTEBOOK_DIR / "data"
    if str(NOTEBOOK_DIR.resolve()) not in sys.path:
        sys.path.insert(0, str(NOTEBOOK_DIR.resolve()))

    import pii_diagrams as diagrams
    import pii_techniques as pii

    CONFIGS = ("verified", "unverified")

    def _load_jsonl(path):
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    CORPUS = {
        config: _load_jsonl(DATA_DIR / f"sangraha_hin_{config}_probe.jsonl")
        for config in CONFIGS
    }
    METRICS = json.loads((DATA_DIR / "metrics.json").read_text(encoding="utf-8"))
    MANIFEST = json.loads((DATA_DIR / "probe_manifest.json").read_text(encoding="utf-8"))
    TOKEN_COST = json.loads((DATA_DIR / "token_cost.json").read_text(encoding="utf-8"))
    GOLD = _load_jsonl(DATA_DIR / "gold_labelled.jsonl")

    DOCUMENT_INDEX = {
        (config, row["doc_id"]): row for config in CONFIGS for row in CORPUS[config]
    }
    return (
        CONFIGS,
        CORPUS,
        Counter,
        DATA_DIR,
        DOCUMENT_INDEX,
        GOLD,
        MANIFEST,
        METRICS,
        TOKEN_COST,
        diagrams,
        json,
        pii,
    )


@app.cell(hide_code=True)
def section_1_header(METRICS, mo):
    _causes = METRICS["shipped_false_positive_causes"]
    _institutional_share = 100 * sum(
        n for k, n in _causes.items() if "institutional" in k or "organisational" in k
    ) / sum(_causes.values())

    mo.md(
        f"""
    ## 1. What PII scrubbing is, and where the boundary sits

    **One-liner.** Find the spans of text that identify a real person, and replace
    or drop them before the corpus is tokenized.

    Everything hard about the stage follows from one asymmetry: **the two ways to
    be wrong are not symmetric, and neither is cheap.**

    | | What it is | What it costs |
    |---|---|---|
    | **False negative** | real PII left in the corpus | a person's identifier becomes learnable, and extractable |
    | **False positive** | non-PII masked anyway | real text destroyed; on Indic corpora this deletes place names, public figures and meaningful numerals |

    So "scrub aggressively" is not a safe default — it is a decision to damage the
    language in exchange for privacy headroom. The stage is therefore a **policy
    with a stated threat model**, not a maximisation problem.

    ### The threat model this notebook uses

    A span is PII when it **identifies a private individual**. Explicitly *not*
    PII under this model:

    - a **published institutional contact** — a bank's helpline, a ministry's
      grievance email, a newspaper's newsroom number;
    - a **public figure's name** in news prose, which is the subject of the
      reporting, not a leak;
    - a **documentation example** — `192.0.2.1`, `user@example.com`.

    This boundary is not a technicality. It turns out to explain
    **{_institutional_share:.0f}%** of what the shipped detector gets "wrong" on
    Hindi news text, as section 5.5 measures.
    """
    )


@app.cell(hide_code=True)
def diagram_pipeline(diagrams, mo):
    mo.Html(diagrams.pipeline_position())


@app.cell(hide_code=True)
def section_2_header(mo):
    mo.md(
        r"""
    ## 2. Technique reference

    Eleven techniques, ordered by where they sit on the precision / recall / cost
    surface. Each entry states what it is, where it was actually used with a
    citation, how it works internally, when to use it, and what it costs.

    Three real privacy techniques are **deliberately excluded** because they are
    not corpus-scrubbing stages: differential-privacy training (changes the
    optimiser, not the data), *k*-anonymity and generalisation (needs a tabular
    schema), and format-preserving encryption (reversible by design, which is the
    opposite of what a training corpus needs).
    """
    )


@app.cell(hide_code=True)
def technique_1(mo):
    mo.md(
        r"""
    ### 2.1 Structured regex

    > **One-liner.** If an identifier has a fixed grammar, a finite automaton
    > decides it exactly — no model, no GPU, no ambiguity.

    **What it is.** A pattern per identifier category, matched against the raw
    text. Email, phone, IP, credential, Aadhaar-shaped and PAN-shaped runs all
    have a shape that a regular language can express.

    **Where it was used.**

    - **Dolma** (AI2, 3T tokens) — the entire PII stage is regex. `FastPiiRegex`
      tags `EMAIL_ADDRESS`, `PHONE_NUMBER`, `IP_ADDRESS`; the paper states that
      model-based detection was *impractical* at their scale.
      `arXiv:2402.00159` §5.3, `allenai/dolma` → `python/dolma/taggers/pii.py`
    - **BigScience ROOTS** (1.6TB, 46 languages) — a multilingual regex pass
      replacing matches with `PI:EMAIL`-style tags, applied across 25 languages
      including Hindi. `arXiv:2303.03915` §3.3
    - **C4 / T5** — the "bad words" and boilerplate filters are the same
      machinery applied to a different target. `arXiv:1910.10683` §2.2

    **How it works internally.** The pattern compiles to a DFA/NFA; the engine
    walks the input once per pattern and reports `(start, end)` offsets. Cost is
    linear in input length and independent of corpus size, which is exactly why
    it survives at trillion-token scale.

    **When to use it.** Always, as the first layer. It is the precision floor
    every other technique is measured against.

    **Pros** — near-perfect precision on well-shaped identifiers; auditable
    (every decision traces to one pattern); microseconds per document; no
    training data.

    **Cons** — **cannot see a personal name at all**, because a name has no
    grammar. Recall is bounded by the pattern list you thought to write. And on
    non-Latin scripts the patterns silently misbehave — section 5 measures
    exactly how.
    """
    )


@app.cell(hide_code=True)
def technique_2(mo):
    mo.md(
        r"""
    ### 2.2 Checksum and format validation

    > **One-liner.** Turn a shape match into an arithmetic fact, so a 12-digit run
    > either *is* a valid Aadhaar number or provably is not.

    **What it is.** A second gate applied only to numeric categories whose issuing
    authority embeds a check digit. Aadhaar uses **Verhoeff**; payment cards use
    **Luhn** (ISO/IEC 7812-1).

    **Where it was used.**

    - **Microsoft Presidio** ships checksum validators as the `validate_result`
      hook on recognizers, e.g. Luhn for `CREDIT_CARD`, and a Verhoeff check in
      the India-specific Aadhaar recognizer.
      `microsoft/presidio` → `presidio_analyzer/predefined_recognizers/`
    - **UIDAI** specifies the Verhoeff check digit for Aadhaar, which is why the
      test is valid rather than heuristic.

    **How it works internally.** Verhoeff computes over the dihedral group
    *D*₅: digits are read right-to-left, each is permuted by a position-dependent
    permutation `p[i mod 8]`, and the results are composed through the group
    multiplication table `d`. A number is valid iff the composition is 0. The
    algebra is what buys the guarantee — **verified in-notebook below**: it
    catches *every* single-digit error and *every* adjacent transposition.

    **When to use it.** Whenever a numeric category has a published check digit,
    and only then.

    **Pros** — converts a guess into a decision; a failing checksum is a *proof*
    of non-membership, not a probability; costs microseconds.

    **Cons** — narrow applicability. And a valid checksum does **not** prove the
    number is an Aadhaar number in use — roughly 1 in 10 random 12-digit runs
    passes by chance, so the gate raises confidence without settling it.
    """
    )


@app.cell
def checksum_property_test(mo, pii):
    import random as _random

    # The claim "Verhoeff catches every single-digit error and every adjacent
    # transposition" is the reason this layer is worth its complexity, so it is
    # tested rather than asserted.
    _random.seed(0)
    _non_unique = _missed_single = _missed_swap = _single_trials = 0

    for _ in range(400):
        payload = "".join(_random.choice("0123456789") for _ in range(11))
        valid_digits = [d for d in "0123456789" if pii.verhoeff_valid(payload + d)]
        if len(valid_digits) != 1:
            _non_unique += 1
        number = payload + valid_digits[0]

        for position in range(len(number)):
            for replacement in "0123456789":
                if replacement == number[position]:
                    continue
                _single_trials += 1
                if pii.verhoeff_valid(
                    number[:position] + replacement + number[position + 1 :]
                ):
                    _missed_single += 1

        for position in range(len(number) - 1):
            if number[position] == number[position + 1]:
                continue
            swapped = (
                number[:position]
                + number[position + 1]
                + number[position]
                + number[position + 2 :]
            )
            if pii.verhoeff_valid(swapped):
                _missed_swap += 1

    checksum_property_report = mo.md(
        f"""
    | Verhoeff property | result |
    |---|---|
    | payloads with a non-unique check digit | **{_non_unique}** of 400 |
    | single-digit errors accepted | **{_missed_single}** of {_single_trials:,} |
    | adjacent transpositions accepted | **{_missed_swap}** |
    | Luhn on the ISO reference vector `79927398713` | **{pii.luhn_valid("79927398713")}** |
    | Luhn on a corrupted vector `79927398710` | **{pii.luhn_valid("79927398710")}** |

    Zero in all three error columns: the implementation has the guarantee the
    algorithm promises, so a checksum failure below can be trusted as a decision.
    """
    )
    checksum_property_report
    return (checksum_property_report,)


@app.cell(hide_code=True)
def technique_3(mo):
    mo.md(
        r"""
    ### 2.3 Context windows and gazetteers

    > **One-liner.** The words *around* a candidate are evidence about it — a
    > number labelled "मोबाइल" is far more likely a phone number than a bare run
    > of ten digits.

    **What it is.** Scan a window either side of the candidate for cue terms, and
    adjust the confidence score rather than the match itself.

    **Where it was used.**

    - **Presidio's `LemmaContextAwareEnhancer`** — adds a fixed **+0.35** when a
      supportive lemma appears in the window, then *floors* the enhanced score at
      **0.40** and caps it at 1.0.
      `microsoft/presidio` → `presidio_analyzer/context_aware_enhancers/lemma_context_aware_enhancer.py`
    - **Scrubadub** uses the same idea with detector-specific keyword lists.

    **How it works internally.** Scoring is additive and clamped, not Bayesian —
    a deliberate engineering choice, because it stays interpretable: you can
    always say *which* cue word moved a span across the threshold. This notebook
    reproduces the mechanism with Hindi cue words (`मोबाइल`, `आधार`, `ईमेल`,
    `संपर्क`) alongside English ones.

    **When to use it.** When a numeric category is too generic to act on alone.
    It is the cheapest recall available after the pattern list is exhausted.

    **Pros** — large recall gain for a keyword list; interpretable; no model.

    **Cons** — it is a **recall** instrument, so it costs precision. Measured
    below: adding this layer took precision from 0.468 to 0.373. Cue lists are
    also language-specific, so every new language re-opens the work.
    """
    )


@app.cell(hide_code=True)
def technique_4(mo):
    mo.md(
        r"""
    ### 2.4 Allowlist and denylist suppression

    > **One-liner.** The precision brake: a list of things that match the pattern
    > but are known not to be PII.

    **What it is.** After detection, suppress spans that are demonstrably benign —
    reserved IP ranges, documentation examples, published helplines, year runs.

    **Where it was used.**

    - **BigCode** exempts documentation/example IP addresses from redaction so
      tutorial text survives. `bigcode-project/bigcode-dataset` → `pii/`
    - **RFC 5737** reserves `192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`
      for documentation, and **RFC 1918** reserves the private ranges — both are
      *specifications* that a matching span is not a person's address.

    **How it works internally.** A post-filter over the candidate list. Cheap, but
    the rules must be *narrow*: suppressing the whole `phone_landline` category
    would be wrong, so the rule here fires only when a nearby cue marks the number
    as an organisation's own contact line.

    **When to use it.** As the last layer, driven by a **measured** false-positive
    audit. Every suppression rule in this notebook came from reading actual false
    positives, not from imagining them.

    **Pros** — direct precision repair; each rule is auditable and reversible;
    encodes domain knowledge a pattern cannot.

    **Cons** — an allowlist is a standing invitation to suppress real PII that
    happens to resemble a listed shape. Every rule needs a stated justification,
    which is why each one below carries its reason string.
    """
    )


@app.cell(hide_code=True)
def technique_5(mo):
    mo.md(
        r"""
    ### 2.5 Statistical sequence labelling (CRF / HMM)

    > **One-liner.** The pre-neural baseline that established *span decoding* as
    > the shape of the problem.

    **What it is.** Treat PII detection as per-token classification over a
    **BIO/IOB** tag set, with a linear-chain CRF modelling transitions so tags
    stay coherent across a span.

    **Where it was used.**

    - **CoNLL-2003** established the shared-task framing that all later NER,
      including PII NER, inherited. `doi:10.3115/1119176.1119195`
    - **Stanford NER** (Finkel et al. 2005) was the production CRF tagger for
      years. `doi:10.3115/1219840.1219885`
    - **i2b2/UTHealth 2014 de-identification challenge** — clinical PHI
      de-identification where CRF systems were the strong baseline.
      `doi:10.1016/j.jbi.2015.06.007`

    **How it works internally.** Hand-engineered features (casing, prefixes,
    suffixes, gazetteer membership, window words) feed a log-linear model;
    Viterbi decoding picks the highest-scoring tag sequence, so `B-PER I-PER` is
    reachable but `I-PER` alone is not.

    **When to use it.** Today: rarely, as a detector. It remains worth knowing
    because **BIO decoding and span-level scoring are still how transformer
    taggers work** — the representation changed, the output structure did not.

    **Pros** — trains on modest data and CPU; features are inspectable.

    **Cons** — features are language-specific and hand-built; consistently below
    transformer baselines; poor transfer to Indic scripts without redesign.
    """
    )


@app.cell(hide_code=True)
def technique_6(mo):
    mo.md(
        r"""
    ### 2.6 Transformer token classification (NER)

    > **One-liner.** The only technique that can catch a personal name — and the
    > one whose precision you must measure before trusting it.

    **What it is.** A pretrained encoder fine-tuned for token classification over
    PII entity types, so the decision uses context instead of shape.

    **Where it was used.**

    - **BigCode StarPII** — a 125M-parameter StarEncoder fine-tuned on ~12,000
      annotated files, used to redact names, emails and keys across The Stack for
      StarCoder. Reported **NAME F1 91.6**; `USERNAME` reached only **61.5** and
      was therefore **detected but deliberately not redacted**.
      `arXiv:2305.06161` §4, `bigcode/starpii`
    - **AI4Bharat IndicNER** — mBERT fine-tuned for 11 Indian languages, Hindi
      **PER F1 82.3**, built by projecting English tags across the Samanantar
      parallel corpus. `arXiv:2212.10168`, `ai4bharat/IndicNER`
    - **Presidio** ships spaCy/Stanza/transformer NLP engines for `PERSON`,
      `LOCATION`, `ORGANIZATION`.

    **How it works internally.** Subword embeddings pass through self-attention,
    a per-token linear head emits BIO logits, and contiguous tags are merged into
    spans. Because attention is contextual, the *same string* can be an entity in
    one sentence and not in the next — the source of both its power and its
    failure mode.

    **When to use it.** When names matter and you can afford GPU inference over
    the corpus, **and** you have measured its precision on your language and
    domain.

    **Pros** — the only route to names, addresses and quasi-identifiers;
    transfers across formats; recall well above any pattern list.

    **Cons** — GPU cost at corpus scale (Dolma's stated reason for skipping it);
    no auditable reason for a decision; and the precision problem is *worse* for
    Indic text. Hindi PER F1 82.3 means roughly **one in five** name predictions
    is wrong, and in news prose the wrong ones are systematically place names and
    public figures — the vocabulary you least want to delete.
    """
    )


@app.cell(hide_code=True)
def diagram_regex_ner(diagrams, mo):
    mo.Html(diagrams.regex_vs_ner())


@app.cell(hide_code=True)
def technique_7(mo):
    mo.md(
        r"""
    ### 2.7 Cascade / two-stage routing

    > **One-liner.** Run the cheap detector on everything and the expensive one
    > only where the cheap one is uncertain.

    **What it is.** Order detectors by cost. Each stage either decides or defers.
    A high-precision regex settles the easy majority; only survivors reach a
    model.

    **Where it was used.**

    - **BigCode** splits by category rather than by confidence — regex and
      `detect-secrets` for keys/IPs/emails, StarPII only for names.
      `arXiv:2305.06161` §4
    - **Presidio's `AnalyzerEngine`** is a cascade by construction: recognizers
      propose, the context enhancer adjusts, `RecognizerResult` scores are
      thresholded, then overlaps are resolved.
    - **Dolma** truncates the cascade after the cheap stage — an explicit
      cost/recall decision, not an oversight. `arXiv:2402.00159` §5.3

    **How it works internally.** Each layer emits scored candidate spans; scores
    accumulate; a threshold admits spans; **overlap resolution** picks a winner
    when two layers claim intersecting text, since masking requires
    non-overlapping edits.

    **When to use it.** At corpus scale, essentially always — it is how you afford
    any model-based layer at all.

    **Pros** — near-model recall at near-regex cost; each layer independently
    testable; the threshold gives operators one honest dial.

    **Cons** — the composition is where the bugs live. Overlap resolution and
    score arithmetic are easy to get subtly wrong, and errors compound silently
    down the stack.
    """
    )


@app.cell(hide_code=True)
def diagram_cascade(diagrams, mo):
    mo.Html(diagrams.cascade_internals())


@app.cell(hide_code=True)
def technique_8(mo):
    mo.md(
        r"""
    ### 2.8 Document-level drop versus span masking

    > **One-liner.** When a document is dense with PII, editing it is the wrong
    > move — drop it.

    **What it is.** A policy switch. Mask individual spans, or discard the whole
    document once its PII count crosses a threshold.

    **Where it was used.**

    - **Dolma** parameterises exactly this: its PII taggers emit counts and the
      mixer is configured with a **threshold of 5** PII spans per document, above
      which the document is removed rather than redacted.
      `arXiv:2402.00159` §5.3
    - **BigScience ROOTS** masks in place. `arXiv:2303.03915` §3.3

    **How it works internally.** Detection produces a per-document span count; the
    mixer compares it against the threshold and routes the document to *masked* or
    *dropped*.

    **When to use it.** Drop when density is high — a contact directory or a
    scraped user table is not worth repairing, and 40 placeholders in a row teach
    the model nothing but the placeholder.

    **Pros** — removes co-occurrence risk that per-span masking leaves behind
    (a name near a masked number is still a re-identification path); avoids
    placeholder-saturated text.

    **Cons** — throws away legitimate text with the PII, and biases the corpus
    against document *genres* that carry contact information. On this Sangraha
    slice the highest-density document is a small-business advertisement — real
    Hindi prose, but with five personal mobiles.
    """
    )


@app.cell(hide_code=True)
def technique_9(mo):
    mo.md(
        r"""
    ### 2.9 Placeholder and surrogate replacement

    > **One-liner.** What you put *in place of* the span is a tokenizer decision,
    > not a cosmetic one.

    **What it is.** The replacement policy: delete the span, insert a typed
    placeholder, or synthesise a realistic surrogate value.

    **Where it was used.**

    - **Dolma** — ` |||EMAIL_ADDRESS||| `, ` |||PHONE_NUMBER||| `,
      ` |||IP_ADDRESS||| ` (note the surrounding spaces).
      `allenai/dolma` → `python/dolma/taggers/pii.py`
    - **BigScience ROOTS** — `PI:EMAIL`-style tags. `arXiv:2303.03915` §3.3
    - **BigCode** — `<EMAIL>`, `<NAME>`, `<KEY>`, `<PASSWORD>` sentinels chosen
      to match the tokenizer's special-token conventions. `arXiv:2305.06161` §4
    - **Presidio Anonymizer** — pluggable operators including `replace`, `redact`,
      `hash`, `mask`, and `encrypt`.

    **How it works internally.** Replacement runs **back-to-front** over resolved
    spans so that earlier offsets stay valid while later text shifts.

    **When to use it.** Typed placeholders for training corpora — they preserve
    the sentence's syntax so surrounding text stays learnable. Surrogates only
    when downstream consumers need well-formed values, and never for a corpus a
    model will memorise.

    **Pros** — keeps grammatical context intact; typed tags make masked data
    countable and auditable; reversible policy (re-run from raw with a different
    tag).

    **Cons** — a verbose placeholder is **expensive**, and this notebook measures
    it: Dolma's ` |||EMAIL_ADDRESS||| ` costs **7.0 tokens** per span while the
    raw identifiers it replaced averaged only **5.5** — the masked corpus is
    *larger* than the unmasked one. BigCode's `<EMAIL>` costs 3.0. A frequent
    placeholder also becomes a token the model learns to emit.
    """
    )


@app.cell(hide_code=True)
def technique_10(mo):
    mo.md(
        r"""
    ### 2.10 LLM-as-judge

    > **One-liner.** Ask a language model whether a span is PII — flexible,
    > expensive, and the weakest-evidenced technique here.

    **What it is.** Prompt an instruction-tuned model to identify or adjudicate
    PII spans, typically as a reviewer over a cheaper detector's output.

    **Where it was used.** Honestly: **not** as the primary PII stage of any major
    published open corpus. The adjacent, documented uses are quality
    classification, not PII — **FineWeb-Edu** used Llama-3-70B to label 460k
    samples to train a small classifier (`arXiv:2406.17557`), and **Nemotron-CC**
    used ensemble LLM labelling (`arXiv:2412.02595`). The pattern worth borrowing
    is *LLM labels, small model runs*.

    **How it works internally.** A prompt frames the span-extraction or
    adjudication task; output is parsed to spans. Nothing guarantees the offsets
    are faithful, so a verification pass against the source string is mandatory.

    **When to use it.** For the genuinely ambiguous residue — "is this
    institutional or personal?" — and to label a training set for a cheap
    classifier. Not as the corpus-wide detector.

    **Pros** — handles context and ambiguity no pattern can express; adapts
    without retraining; can explain itself.

    **Cons** — cost per token makes corpus-scale use prohibitive; non-deterministic
    output breaks reproducibility, which a data pipeline needs; prompt-injectable
    by the corpus itself; and no published open-corpus precedent to lean on.
    """
    )


@app.cell(hide_code=True)
def technique_11(mo):
    mo.md(
        r"""
    ### 2.11 Memorization auditing

    > **One-liner.** Scrubbing you have not attacked is scrubbing you have not
    > verified.

    **What it is.** After training, actively try to extract PII from the model.
    Detector metrics measure the *pipeline*; extraction measures the *outcome*.

    **Where it was used.**

    - **Carlini et al. 2021**, *Extracting Training Data from Large Language
      Models* — recovered verbatim PII from GPT-2 by sampling and ranking with a
      membership-inference signal. `arXiv:2012.07805`
    - **Carlini et al. 2022**, *Quantifying Memorization* — memorisation scales
      log-linearly with model size, with duplicate count, and with prompt length.
      `arXiv:2202.07646`
    - **Lee et al. 2021**, *Deduplicating Training Data Makes Language Models
      Better* — deduplication reduces memorised output ~10×. `arXiv:2107.06499`

    **How it works internally.** Prompt with prefixes drawn from the training
    distribution, sample many continuations, and rank candidates by a calibrated
    likelihood ratio; then check hits against the known corpus.

    **When to use it.** Before any release. It is the only technique that tests
    the property you actually care about.

    **Pros** — measures real-world harm, not proxy detector metrics; catches PII
    the detectors never proposed, which is precisely the blind spot in every
    recall number in this notebook.

    **Cons** — requires a trained model, so the feedback arrives late; absence of
    extraction is weak evidence of absence of memorisation.

    **The dependency worth naming:** memorisation rises with duplicate count, so
    **stage 5 deduplication is itself a PII control.** The stages are not
    independent.
    """
    )


@app.cell(hide_code=True)
def diagram_precision_recall(diagrams, mo):
    mo.Html(diagrams.precision_recall_geometry())


@app.cell(hide_code=True)
def section_3_header(mo):
    mo.md(
        r"""
    ## 3. The Sangraha probe cache

    **Source.** `ai4bharat/sangraha`, revision `8b813c3f62d3…`, CC-BY-4.0, `hin`
    split of both the `verified` and `unverified` configs.

    **Selection rule.** Fixed stride windows across each whole split — 100
    windows × 100 consecutive rows, stream order preserved, **no shuffling and no
    PII-conditioned selection.** The rule is disclosed and rebuildable via
    `data/build_pii_probe_cache.py`, so widening the sample is not cherry-picking.

    **Why both configs.** `verified` is the pool the case study trains on;
    `unverified` is the higher-risk pool. Comparing PII density across the two
    turns the lecture's *judge by source* heuristic into a measurement.

    **Why not the existing 264-document sample.** It contains **zero**
    regex-detectable PII. A stage whose entire purpose is detection cannot be
    studied on a slice with no positives.
    """
    )


@app.cell(hide_code=True)
def probe_cache_table(MANIFEST, mo):
    _rows = "\n".join(
        f"| `{s['config']}` | {s['documents']:,} | {s['characters']:,} | "
        f"{s['estimated_tokens']:,} | {s['stride']:,} | `{s['sha256'][:16]}…` |"
        for s in MANIFEST["splits"]
    )
    mo.md(
        f"""
    | config | documents | characters | est. tokens | stride | sha256 |
    |---|---:|---:|---:|---:|---|
    {_rows}

    Token estimate: `{MANIFEST["token_estimator"]}`. Materialized
    {MANIFEST["materialized_at_utc"][:19]}Z via the
    `{MANIFEST["access_method"]}`.
    """
    )


@app.cell(hide_code=True)
def section_4_header(mo):
    mo.md(
        r"""
    ## 4. What I implement for Sangraha, and the sample that shows it

    A **four-layer cascade**, in this order:

    1. **Unicode digit folding onto a detection copy only** — never the emitted
       text — so Devanagari-digit identifiers become visible to numeric patterns.
    2. **Structured regex on the folded copy**, with offsets mapped back to the
       original string.
    3. **Checksum gates** — Verhoeff for Aadhaar-shaped runs, Luhn for
       card-shaped runs.
    4. **Allowlist suppression** for the false positives Hindi news text actually
       produces.

    **Names are detected and counted, but not masked.** This is the single most
    consequential decision in the notebook, and it follows BigCode's own
    precedent: they detected `USERNAME` at F1 61.5 and declined to redact it. With
    Hindi PER F1 at 82.3, roughly one name prediction in five is wrong, and in
    news prose the wrong ones are systematically place names and public
    figures. **A detector you do not trust enough to act on is a review signal,
    not a redaction rule.**
    """
    )


@app.cell(hide_code=True)
def showcase_document(DOCUMENT_INDEX, mo, pii):
    # Chosen because it is the clearest single-document proof that an ASCII-only
    # PII pass is not merely suboptimal on Indic text but blind: the identifier is
    # a personal mobile written in Devanagari digits, sitting next to the person's
    # own name and the word "संपर्क" (contact).
    SHOWCASE = ("verified", "ebb32e1c2f667ce777a141cd30ec010841a70f7b")
    showcase_row = DOCUMENT_INDEX[SHOWCASE]
    showcase_text = showcase_row["text"]

    folded_result = pii.scrub(showcase_text, fold=True)
    ascii_result = pii.scrub(showcase_text, fold=False)

    def _highlight(text, spans):
        out, cursor = [], 0
        for span in sorted(spans, key=lambda s: s.start):
            out.append(mo.Html(f"<span>{text[cursor:span.start]}</span>").text)
            out.append(
                f'<mark style="background:#e4f5ec;border:1px solid #1c8b5a;'
                f'border-radius:3px;padding:0 3px">{text[span.start:span.end]}'
                f'<sub style="color:#1c8b5a;font-weight:700"> {span.category}</sub></mark>'
            )
            cursor = span.end
        out.append(mo.Html(f"<span>{text[cursor:]}</span>").text)
        return "".join(out)

    _window = showcase_text[900:1180]
    _shifted = [
        pii.Span(s.start - 900, s.end - 900, s.category, s.detector, s.score)
        for s in folded_result.accepted
        if 900 <= s.start < 1180
    ]

    showcase_report = mo.md(
        f"""
    ### The document: `{SHOWCASE[1][:16]}…` (`{SHOWCASE[0]}`)

    A Hindi article about a photographer, ending with a call for volunteers. The
    contact line is a **personal mobile number written in Devanagari digits**:

    <div style="font-size:15px;line-height:2;padding:14px 16px;background:#f6f8fc;
    border:1px solid #dbe3ef;border-radius:8px">…{_highlight(_window, _shifted)}…</div>

    | | spans found | what it found |
    |---|---:|---|
    | **folded cascade** | {len(folded_result.accepted)} | `{"`, `".join(s.text_of(showcase_text) for s in folded_result.accepted) or "—"}` |
    | **ASCII-only patterns** | {len(ascii_result.accepted)} | {"`" + "`, `".join(s.text_of(showcase_text) for s in ascii_result.accepted) + "`" if ascii_result.accepted else "**nothing at all**"} |

    The document carries **{pii.native_digit_count(showcase_text)}** native-script
    digits. Every ASCII-only detector — including Dolma's, run faithfully in
    section 5.3 — reports this document as clean. The number is a person's mobile,
    published beside their name, and it survives into training untouched.
    """
    )
    showcase_report
    return SHOWCASE, ascii_result, folded_result, showcase_report, showcase_row, showcase_text


@app.cell(hide_code=True)
def diagram_digit_trap(diagrams, mo):
    mo.Html(diagrams.devanagari_digit_trap())


@app.cell(hide_code=True)
def section_5_header(mo):
    mo.md(
        r"""
    ## 5. Measurement

    ### 5.1 PII density, and the source heuristic made measurable
    """
    )


@app.cell(hide_code=True)
def density_table(METRICS, mo):
    _d = METRICS["density"]
    _ratio = (
        _d["unverified"]["spans_per_million_tokens"]
        / _d["verified"]["spans_per_million_tokens"]
    )
    _rows = "\n".join(
        f"| `{config}` | {v['documents']:,} | {v['documents_touched']} "
        f"({v['documents_touched_pct']}%) | {v['spans_total']} | "
        f"**{v['spans_per_million_tokens']}** |"
        for config, v in _d.items()
    )
    _cats = "\n".join(
        f"| `{config}` | "
        + " · ".join(f"{k} **{n}**" for k, n in v["spans_by_category"].items())
        + " |"
        for config, v in _d.items()
    )
    mo.md(
        f"""
    | config | documents | touched | spans | spans / 1M tokens |
    |---|---:|---:|---:|---:|
    {_rows}

    | config | by category |
    |---|---|
    {_cats}

    **Two findings.**

    **1. The `unverified` pool carries {_ratio:.1f}× the PII density of
    `verified`.** The lecture's *judge by source* heuristic is not a slogan — the
    gap is measurable, in the same direction the provenance label predicts, and it
    justifies routing the two pools through different thresholds.

    **2. The stage touches almost nothing.** Under 0.6% of documents change even
    in the higher-risk pool. That is itself the important result: PII scrubbing is
    a **low-yield, high-stakes** stage. It is not a filter that reshapes the
    corpus, it is a targeted repair — which is exactly why a false positive rate
    matters so much. If you damage 1% of documents to fix 0.5%, you have made the
    corpus worse.
    """
    )


@app.cell(hide_code=True)
def section_5_2_header(METRICS, mo):
    mo.md(
        f"""
    ### 5.2 The gold set

    Every span that *any* detector proposed — the cascade at all settings, plus a
    faithful port of Dolma's `FastPiiRegex` — was labelled by hand against the
    threat model in section 1. That is
    **{METRICS["gold_set"]["candidates_reviewed"]} candidate spans reviewed**.
    """
    )


@app.cell(hide_code=True)
def gold_set_summary(Counter, GOLD, METRICS, mo):
    _g = METRICS["gold_set"]
    _pii_cats = Counter(r["cat"] for r in GOLD if r["gold"])
    _not_pii = Counter(r["gold_note"] for r in GOLD if not r["gold"])
    _cat_rows = "\n".join(f"| `{k}` | {n} |" for k, n in _pii_cats.most_common())
    _top_not = "\n".join(f"| {k} | {n} |" for k, n in _not_pii.most_common(6))
    mo.md(
        f"""
    | | count |
    |---|---:|
    | candidate spans reviewed | {_g["candidates_reviewed"]} |
    | labelled **PII** | {_g["pii_spans_raw"]} |
    | after overlap dedupe (the scoring denominator) | **{_g["pii_spans_overlap_deduped"]}** |
    | labelled **not PII** | {_g["candidates_reviewed"] - _g["pii_spans_raw"]} |

    **PII spans by category** &nbsp;&nbsp; | &nbsp;&nbsp; **why the rest are not PII**

    <div style="display:flex;gap:32px;flex-wrap:wrap">
    <div>

    | category | n |
    |---|---:|
    {_cat_rows}

    </div><div>

    | reason | n |
    |---|---:|
    {_top_not}

    </div></div>

    Overlap dedupe matters: different layers propose overlapping spans for the
    same identifier, so counting both would inflate the denominator. Gold spans
    and predictions are both reduced to non-overlapping representatives before
    scoring.

    > **{_g["recall_caveat"]}**
    """
    )


@app.cell(hide_code=True)
def section_5_3_header(mo):
    mo.md(
        r"""
    ### 5.3 Detector comparison

    Four configurations against the same gold set. Variant **A** is a faithful
    port of Dolma's three PII taggers, so this is a comparison against what a
    real 3T-token pipeline actually ran — not a strawman.
    """
    )


@app.cell(hide_code=True)
def variant_table(METRICS, mo):
    _rows = "\n".join(
        f"| {name} | {r['tp']} | {r['fp']} | {r['fn']} | {r['P']:.3f} | "
        f"{r['R']:.3f} | {r['F1']:.3f} |"
        for name, r in METRICS["variants"].items()
    )
    _v = METRICS["variants"]
    _dolma = _v["A. Dolma FastPiiRegex (faithful port)"]
    _ascii = _v["B. cascade, ASCII patterns only (no digit folding)"]
    _ship = _v["D. cascade, handles excluded (shipped)"]
    mo.md(
        f"""
    | variant | TP | FP | FN | precision | recall | F1 |
    |---|---:|---:|---:|---:|---:|---:|
    {_rows}

    **Reading this table.**

    - **Dolma's real weakness on Hindi is precision, not recall.** Its port
      reaches recall {_dolma["R"]:.3f} — the same as the shipped cascade — but at
      precision {_dolma["P"]:.3f} versus {_ship["P"]:.3f}, because its permissive
      phone pattern claims {_dolma["fp"]} spans that are not personal data. On
      this slice, running Dolma's PII stage unchanged over Hindi text would mask
      **{_dolma["fp"]}** non-PII spans to catch {_dolma["tp"]} real ones.
    - **Digit folding is what buys the last identifiers.** Variant B is the same
      cascade with ASCII-only patterns: it misses {_ascii["fn"]} gold spans versus
      {_ship["fn"]}, and every one of the extra misses is a native-digit
      identifier of the kind shown in section 4.
    - **Social handles were the right thing to drop.** Variant C includes them and
      precision collapses to {_v["C. cascade, all layers, handles included"]["P"]:.3f}
      — a `@`-handle in Hindi news prose is overwhelmingly an institutional
      account, not a private individual.
    """
    )


@app.cell(hide_code=True)
def section_5_4_header(mo):
    mo.md(
        r"""
    ### 5.4 Layer yield ladder

    What each layer contributes, added one at a time. This is where a cascade
    earns or fails to earn its complexity.
    """
    )


@app.cell(hide_code=True)
def ladder_table(METRICS, mo):
    _rows = "\n".join(
        f"| {name} | {r['spans_kept']} | {r['tp']} | {r['fp']} | {r['P']:.3f} | {r['R']:.3f} |"
        for name, r in METRICS["layer_ladder"].items()
    )
    _l = METRICS["layer_ladder"]
    _ctx = _l["3. + context scoring"]
    _allow = _l["4. + allowlist suppression"]
    mo.md(
        f"""
    | cascade layers | spans kept | TP | FP | precision | recall |
    |---|---:|---:|---:|---:|---:|
    {_rows}

    **The checksum layer contributes exactly nothing on this slice, and I am
    reporting that rather than hiding it.** Only three checksum-eligible spans
    appear in 18,400 documents — two Aadhaar-shaped and one card-shaped — and all
    three fail their checksum. Inspecting them shows why the layer is still
    correct to keep: the "Aadhaar" is the PIB fact-check WhatsApp number
    `918799711259`, and the "card" is a DOI suffix, `10.1177/0022034509359125`.
    Both are exactly the false positives a checksum is supposed to reject; they
    were already below threshold, so the gate had nothing left to do. A layer that
    is inert on Hindi news text but load-bearing on scanned forms or leaked
    databases is a layer you keep and *document*, not one you delete.

    **Context scoring buys the last true positive and pays
    {_ctx["fp"] - _l["2. + checksum gates"]["fp"]} false positives for it**
    (precision {_l["2. + checksum gates"]["P"]:.3f} → {_ctx["P"]:.3f}). Allowlist
    suppression then repairs {_ctx["fp"] - _allow["fp"]} of them, returning
    precision to {_allow["P"]:.3f} while keeping recall at {_allow["R"]:.3f}. That
    push-and-repair is the cascade working as designed: a recall layer followed by
    a precision brake.
    """
    )


@app.cell(hide_code=True)
def section_6_header(mo):
    mo.md(
        r"""
    ### 5.5 False-positive audit

    Precision of 0.454 looks poor. The audit is the reason it is not the number to
    act on.
    """
    )


@app.cell(hide_code=True)
def fp_audit(METRICS, mo):
    _causes = METRICS["shipped_false_positive_causes"]
    _total = sum(_causes.values())
    _institutional = sum(
        n for k, n in _causes.items() if "institutional" in k or "organisational" in k
    )
    _rows = "\n".join(f"| {k} | {n} |" for k, n in _causes.items())
    mo.md(
        f"""
    | cause of false positive | n |
    |---|---:|
    {_rows}
    | **total** | **{_total}** |

    **{_institutional} of {_total} ({100 * _institutional / _total:.0f}%) are
    real, correctly-parsed contact details that belong to institutions** — bank
    helplines, ministry grievance lines, newspaper newsrooms. The detector read
    the string correctly; it simply cannot know whether a number belongs to a
    person or an organisation. Only **{_total - _institutional}** are genuine
    shape errors: two documentation example addresses, and four bank service
    numbers and one Russian TIN that are not phone numbers at all.

    **This reframes the precision number entirely.** The cascade's pattern layer
    is not sloppy — it is running into the fact that *institutional versus
    personal is a semantic judgement, not a lexical one.* No regex can settle it,
    and neither can a NER model reliably. It needs either a gazetteer of
    institutional numbers, or the LLM-adjudication layer from §2.10 applied to
    just these {_total} spans, or a policy decision that published contact points
    are acceptable to mask.

    That is the honest ceiling of a rule-based PII stage on news text, and it is
    worth knowing *before* choosing a threshold rather than after.
    """
    )


@app.cell(hide_code=True)
def section_5_6_header(mo):
    mo.md(r"""### 5.6 The cost of the placeholder""")


@app.cell(hide_code=True)
def token_cost_table(TOKEN_COST, mo):
    _rows = "\n".join(
        f"| {p['policy']} | {p['tokens']:,} | {p['delta_vs_raw']:+,} | {p['pct']:+.3f}% |"
        for p in TOKEN_COST["whole_corpus"]
    )
    _per = "\n".join(
        f"| {p['policy']} | {p['tokens_spent_on_placeholders']:,} | {p['tokens_per_span']} |"
        for p in TOKEN_COST["placeholder_cost"]
    )
    mo.md(
        f"""
    | replacement policy | corpus tokens | Δ vs raw | Δ % |
    |---|---:|---:|---:|
    {_rows}

    | policy | tokens spent on placeholders | per span |
    |---|---:|---:|
    {_per}

    Raw identifiers cost **{TOKEN_COST["tokens_per_raw_identifier"]}** tokens each
    on average, across {TOKEN_COST["spans_replaced"]} replaced spans.

    **Dolma's placeholder is more expensive than the PII it removes.**
    ` |||EMAIL_ADDRESS||| ` costs 7.0 tokens per span against 5.5 for the raw
    identifier, so the scrubbed corpus is *larger* than the original. BigCode's
    `<EMAIL>` costs 3.0 and shrinks it. At this slice's density the absolute
    effect is negligible (±0.12%), which is the real conclusion: **placeholder
    verbosity is a correctness and tokenizer-hygiene question, not a budget
    question** — a frequent multi-token sentinel is a string the model learns to
    emit.

    > Measured with `cl100k_base`, which is *not* a Hindi tokenizer. The
    > comparison between policies is valid; the absolute counts are not a
    > prediction for a Sarvam- or IndicBERT-class tokenizer.
    """
    )


@app.cell(hide_code=True)
def section_7_header(mo):
    mo.md(
        r"""
    ## 6. The Indic findings

    Three results that do not appear in the English-corpus literature, each with
    executed evidence.
    """
    )


@app.cell
def indic_finding_digits(CORPUS, mo, pii):
    # Finding 1: how many identifiers an ASCII-only pass cannot see, corpus-wide.
    _native_docs = _fold_only_spans = 0
    _examples = []
    for _config in ("verified", "unverified"):
        for _row in CORPUS[_config]:
            if pii.native_digit_count(_row["text"]) == 0:
                continue
            _native_docs += 1
            _with = pii.scrub(_row["text"], fold=True).accepted
            _without = pii.scrub(_row["text"], fold=False).accepted
            _gained = len(_with) - len(_without)
            if _gained > 0:
                _fold_only_spans += _gained
                _examples.append(
                    (_config, _row["doc_id"][:12], [s.text_of(_row["text"]) for s in _with])
                )

    # Finding 2: the offset-preservation invariant the whole design rests on.
    _offset_safe = all(
        pii.folding_is_offset_safe(_row["text"])
        for _config in ("verified", "unverified")
        for _row in CORPUS[_config]
    )

    indic_digit_report = mo.md(
        f"""
    ### 6.1 Devanagari digits: invisible identifiers

    | | |
    |---|---:|
    | documents containing native-script digits | **{_native_docs:,}** |
    | identifiers visible *only* after folding | **{_fold_only_spans}** |
    | documents where folding preserved every offset | **{"all 18,400" if _offset_safe else "FAILED"}** |

    Examples: {" · ".join(f"`{e[2][0]}` ({e[0]})" for e in _examples)}

    The mechanism is the asymmetry inside a single pattern. Python's `\\d` on a
    `str` matches every Unicode category-Nd digit, so `\\d{{12}}` **does** match
    Devanagari years while `[6-9]\\d{{9}}` matches **no** Devanagari mobile at all —
    the leading class is ASCII-only. One pattern, two opposite failures: silent
    misses on real identifiers, spurious hits on year runs.

    Folding fixes both by making digit handling deliberate. It is safe only because
    it maps one code point to one code point, so offsets survive — verified above
    across every document, since a violation would corrupt masking silently.
    """
    )
    indic_digit_report
    return indic_digit_report


@app.cell
def indic_finding_ipv4(CORPUS, mo, pii):
    # Finding 3: folding creates a new Indic false positive. Enumerate every
    # IPv4-shaped span in the cache and read what it actually was.
    _ipv4_hits = []
    for _config in ("verified", "unverified"):
        for _row in CORPUS[_config]:
            _text = _row["text"]
            _folded = pii.fold_digits(_text)
            for _span in pii.detect_patterns(_folded):
                if _span.category != "ipv4":
                    continue
                _ipv4_hits.append(
                    {
                        "config": _config,
                        "folded": _span.text_of(_folded),
                        "original": _span.text_of(_text),
                        "native": pii.native_digit_count(_span.text_of(_text)) > 0,
                        "context": _text[max(0, _span.start - 46) : _span.end + 10].replace(
                            "\n", " "
                        ),
                    }
                )

    _native_hits = [h for h in _ipv4_hits if h["native"]]
    _rows = "\n".join(
        f"| `{h['folded']}` | `{h['original']}` | …{h['context'][:52]}… |"
        for h in _ipv4_hits
    )
    indic_ipv4_report = mo.md(
        f"""
    ### 6.2 The cost of the fix: scripture citations become IP addresses

    | folded | as written | context |
    |---|---|---|
    {_rows}

    Every IPv4-shaped span in the cache — **{len(_native_hits)} of
    {len(_ipv4_hits)}** written in native digits — is a **Sanskrit scripture
    citation**, not a network address. Digit folding is what made them visible, so
    the recall fix created a precision problem.

    The suppression rule follows from what an IP address *is*: an ASCII artefact of
    network configuration. Nobody writes one in Devanagari numerals. So
    native-script digits are **evidence against** the IPv4 hypothesis, and layer 4
    suppresses on exactly that reasoning rather than on a hardcoded list of
    citations.

    This is the general shape of Indic PII work: **a recall fix imported from
    English creates a precision problem that only exists in the target script**,
    and it has to be found by reading the false positives.
    """
    )
    indic_ipv4_report
    return indic_ipv4_report


@app.cell
def indic_finding_boundaries(CORPUS, mo, pii):
    # Finding 3: mixed-script token boundaries. Devanagari has no word-boundary
    # relationship to \b, so an email pattern anchored on ASCII word characters
    # absorbs or truncates against adjacent Devanagari.
    _mixed = []
    for _config in ("verified", "unverified"):
        for _row in CORPUS[_config]:
            _text = _row["text"]
            for _span in pii.detect_patterns(pii.fold_digits(_text)):
                if _span.category != "email":
                    continue
                _value = _span.text_of(_text)
                if any("\u0900" <= c <= "\u097f" for c in _value):
                    _mixed.append((_config, _value, _row["doc_id"][:10]))

    _rows = "\n".join(f"| `{v}` | `{c}` |" for c, v, _ in _mixed) or "| — | — |"
    mixed_script_report = mo.md(
        f"""
    ### 6.3 Mixed-script boundaries

    | detected span | config |
    |---|---|
    {_rows}

    Devanagari characters are not ASCII word characters, so `\\b` does not fire
    between `कर` and `Roopchandrashastri@gmail.com`. Where Hindi text abuts a
    Latin-script email with no space — common in scraped news — the pattern
    **absorbs the adjacent Devanagari into the match**.

    The privacy consequence is mild: the span still covers the identifier, so
    masking still removes it. The *corpus* consequence is not — masking deletes a
    Hindi word along with the email. This is a boundary-definition bug that
    English-only corpora never surface, and it argues for script-aware boundary
    assertions rather than `\\b`.
    """
    )
    mixed_script_report
    return mixed_script_report


@app.cell(hide_code=True)
def section_8_header(METRICS, mo):
    _ship = METRICS["variants"]["D. cascade, handles excluded (shipped)"]
    _dolma = METRICS["variants"]["A. Dolma FastPiiRegex (faithful port)"]
    _density = METRICS["density"]
    _ratio = (
        _density["unverified"]["spans_per_million_tokens"]
        / _density["verified"]["spans_per_million_tokens"]
    )
    _causes = METRICS["shipped_false_positive_causes"]
    _total_fp = sum(_causes.values())
    _institutional = sum(
        n for k, n in _causes.items() if "institutional" in k or "organisational" in k
    )
    _documents = sum(v["documents"] for v in _density.values())

    mo.md(
        f"""
    ## 7. What I would ship, and what stays broken

    ### The recommendation

    **Ship the four-layer cascade with handles excluded, threshold 0.50, span
    masking with short typed placeholders, and names detected but not masked.**
    Measured: precision **{_ship["P"]:.3f}**, recall **{_ship["R"]:.3f}** against
    candidate-labelled gold.

    Route the two pools differently. `unverified` carries **{_ratio:.1f}×** the PII
    density, so it deserves the lower threshold and, if a name model is ever added,
    it is the pool to add it to first.

    ### What I would not ship

    - **Dolma's PII stage unchanged.** Faithfully ported, it reaches the same
      recall ({_dolma["R"]:.3f}) at precision **{_dolma["P"]:.3f}** on this slice,
      masking {_dolma["fp"]} non-PII spans. Its patterns are tuned for English text
      and its phone regex is far too permissive for Hindi news prose.
    - **NER-based name masking.** Hindi PER F1 82.3 means roughly one prediction
      in five is wrong, and the wrong ones are systematically place names and
      public figures. Following BigCode's `USERNAME` precedent, names are a review
      signal here, not a redaction rule.
    - **Aggressive thresholds.** Section 5.5 shows the marginal precision is being
      spent on institutional contact points, so lowering the threshold buys
      almost no real PII and destroys more legitimate text.

    ### Residual risk, stated plainly

    1. **Names are still in the corpus.** This is the largest known gap, accepted
       deliberately with the reasoning above rather than by omission.
    2. **Recall is a candidate bound.** PII that no detector proposed was never
       labelled. The only instrument that closes this gap is the memorisation
       audit of §2.11, and it needs a trained model.
    3. **Institutional-versus-personal is unresolved.**
       **{100 * _institutional / _total_fp:.0f}%** of false positives sit on this
       judgement, and no rule-based layer can settle it.
    4. **The gold set is one annotator.** No inter-annotator agreement was
       computed, so the labels carry my reading of the threat model and no
       measure of how reproducible that reading is.
    5. **{_documents:,} documents is a probe, not the corpus.** The density figures
       are stable enough to compare pools; they are not a corpus-wide guarantee.

    ### The dependency worth repeating

    Memorisation scales with duplicate count (`arXiv:2202.07646`,
    `arXiv:2107.06499`), so **stage 5 deduplication is itself a PII control**, and
    a PII span that survives scrubbing is much more dangerous in a corpus that was
    not deduplicated. These stages are not independent, and a manifest that
    records only one of them understates the risk.
    """
    )


@app.cell(hide_code=True)
def provenance(DATA_DIR, MANIFEST, METRICS, mo):
    import hashlib as _hashlib

    _files = [
        "pii_techniques.py",
        "pii_diagrams.py",
        "evaluate_cascade.py",
        "measure_token_cost.py",
        "data/build_pii_probe_cache.py",
        "data/gold_labelled.jsonl",
        "data/metrics.json",
    ]
    _rows = []
    for _name in _files:
        _path = DATA_DIR.parent / _name
        _digest = _hashlib.sha256(_path.read_bytes()).hexdigest()[:16]
        _rows.append(f"| `{_name}` | {_path.stat().st_size:,} B | `{_digest}…` |")

    mo.md(
        f"""
    ## 8. Provenance

    | artifact | size | sha256 |
    |---|---:|---|
    {chr(10).join(_rows)}

    - **Dataset** — `{MANIFEST["dataset_id"]}` @ `{MANIFEST["dataset_revision"][:12]}…`,
      {MANIFEST["license"]}, via {MANIFEST["access_method"]}.
    - **Selection rule** — {MANIFEST["selection_rule"]}
    - **Metrics regenerated** — {METRICS["generated_utc"][:19]}Z by
      `evaluate_cascade.py`, which reads only `data/` and rewrites
      `data/metrics.json`. Every table above is read from that file rather than
      transcribed, so a re-run either reproduces the notebook or contradicts it
      visibly.

    ### Reproducing

    ```bash
    # rebuild the probe cache from the pinned revision
    .venv/bin/python pii-scrubbing/data/build_pii_probe_cache.py

    # recompute every metric in this notebook
    .venv/bin/python pii-scrubbing/evaluate_cascade.py

    # re-measure placeholder token cost
    .venv/bin/python pii-scrubbing/measure_token_cost.py
    ```
    """
    )


if __name__ == "__main__":
    app.run()
