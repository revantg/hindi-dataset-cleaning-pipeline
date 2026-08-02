# Extraction techniques for Hindi / Indic training data

**Scope:** Session 4 pipeline stage 1 — **Extract**. This guide explains how to turn raw HTML, WARC, PDF, image, subtitle, audio, and video artifacts into faithful, traceable document content. It describes the internal mechanics of the major technique families, what each preserves, what each can silently lose, and how to choose and validate a route for a Hindi/Indic corpus.

**Decision in one sentence:** route by **artifact and layout**, not by a language label; preserve the source and structural evidence; use a cheap deterministic extractor first; escalate only uncertain, dynamic, layout-heavy, low-quality, or high-value material to a learned/layout-aware method.

**Evidence date:** 2026-08-02.

> **Evidence notation.** `[LOCAL]` means the claim is supported by the lecture, widget, manifest, or run artifact in this repository. `[PRIMARY]` means an official tool document, source repository, or primary paper. A benchmark number is always scoped to the benchmark named by its source; it is not a Hindi guarantee. This document explains techniques and recommends an evaluation plan. It does not claim that every listed extractor was run on the local flattened cache.

---

## 1. What “Extract” means

Extraction answers one question:

> **Which parts of the source are the document, and in what order and representation should they survive?**

It is a boundary and representation problem, not a generic “remove bad text” step. The extractor should select or transcribe source content, preserve the evidence needed to audit that decision, and avoid inventing text.

### 1.1 Input and output by artifact type

| Input artifact | Extraction output | Evidence that must survive |
|---|---|---|
| Raw HTML / WARC response | Main document blocks in reading order: title, headings, paragraphs, lists, tables, code, formulas, captions, and selected metadata | source bytes/hash, URL, response headers, DOM paths or source spans, extractor route/version |
| Text-layer PDF | Text blocks/lines/words in a defensible reading order | PDF hash, page number, block/word geometry, font or character mapping clues, extraction flags |
| Scanned PDF / image | OCR text plus page, block, bounding-box, script, and confidence evidence | image/PDF hash, page crop/render settings, OCR engine/model/version, boxes, confidence, preprocessing |
| Audio / video | Timestamped utterances, optionally speaker-labelled, assembled into a document | media hash, codec/sample rate, VAD spans, model/version, word/segment timestamps, confidence, speaker evidence |
| SRT / VTT subtitle | Cue text with timestamps and speaker/cue boundaries preserved | file hash, cue index, start/end time, raw cue text, parser warnings |
| Already-flat JSONL text | Only residual artifact extraction: remove markup that visibly remains | input hash, exact replacement rules, before/after text hashes, reason counts; no invented source boundary |

Extraction should produce a **traceable intermediate representation**, not only a string:

```text
Document {
  doc_id,
  source_uri,
  source_type,
  source_mime,
  source_sha256,
  source_revision_or_timestamp,
  extractor_name,
  extractor_version,
  route_reason,
  extraction_confidence,
  blocks: [
    {
      block_id,
      type: title | heading | paragraph | list | table | code |
            formula | image | caption | utterance | metadata,
      text,
      source_span | dom_path | page_number + bounding_box |
                   start_time + end_time,
      confidence,
      language_hint,
      warnings
    }
  ],
  discarded_blocks: [{source_span, reason}],
  warnings
}
```

The exact schema can change. The invariants cannot:

1. Every output block maps to source evidence.
2. Every discarded region has a route- or policy-level reason.
3. The original artifact is retained or content-addressed.
4. Formatting is represented before it is flattened.
5. Uncertainty is a review signal, not silently converted into deletion.

### 1.2 What extraction is not

- **Not blind HTML-tag deletion.** Tags mark structure; navigation and article text can both be inside `<div>` elements.
- **Not Unicode normalization.** Preserve source bytes and apply representation cleanup in the declared normalization stage. Do not delete Indic code points merely because they are invisible.
- **Not quality filtering.** A short heading, list, code block, table, formula, or conversational utterance can be valuable even when it looks unlike an English paragraph.
- **Not language filtering.** A Hindi page can contain English names, URLs, code, numerals, Romanized Hindi, and Hinglish. Language validation decides scope later.
- **Not deduplication.** Two correctly extracted documents may still be near duplicates; that is a later corpus-level decision.
- **Not summarization or rewriting.** A generative model must not fill gaps, paraphrase source text, or “repair” uncertain OCR without marking the transformation.

### 1.3 The extraction contract with later stages

Extraction hands later stages a document plus evidence:

```text
raw artifact
   │ preserve bytes, URI, MIME, hash
   ▼
artifact router
   │ HTML / PDF / image / subtitle / audio-video / flat text
   ▼
extract blocks + source mapping + confidence
   │
   ├─ Normalize: decode entities, repair encoding, normalize whitespace/markers
   ├─ Language ID: validate language and code-mix
   ├─ Quality: reject spam, malformed or low-value output
   ├─ Dedup: exact and near-duplicate documents
   ├─ PII: redact unnecessary identifiers
   ├─ Decontaminate: remove evaluation leakage
   └─ Manifest: record source, route, hashes, counts, and decisions
```

The same text operation can be correct in one stage and wrong in another. For example, removing a navigation block is extraction; removing a replacement character is normalization; removing a very repetitive article is quality filtering; removing a repeated copy across URLs is deduplication.

---

## 2. What the lectures establish

Session 3 defines the problem directly: collection produces raw material; extraction converts it to training text. The lecture warns that naive HTML stripping leaves navigation, cookie banners, repeated menus, hidden text, malformed Unicode, entities, boilerplate, and control characters in the corpus. The Session 3 extraction widget contrasts a naive whole-page tag strip with a body-aware extractor and records the result as illustrative, not as a corpus measurement. `[LOCAL]`

Session 4 places **Extract** first in the eight-stage pipeline and defines it as retaining the article body while dropping navigation, boilerplate, cookie banners, and footers. It explicitly treats special-token/ghost-tag cleanup as **Normalize**, not as a ninth canonical stage. `[LOCAL]`

Local evidence:

- [Session 3 lecture](../axiom-fetched/session3_lecture.md)
- [Session 3 extraction widget](../axiom-fetched/widgets/widget_17_extraction_quality.html)
- [Session 4 lecture](../axiom-fetched/session4_lecture.md)
- [Session 4 widget inventory](../axiom-fetched/session4_widget_inventory.json)

The teaching widget is useful for understanding the boundary, but its illustrative JavaScript uses a broad `U+200B–U+200D` range in one cleanup pass. That range includes ZWJ and ZWNJ. The safer Session 4 rule is the canonical one here: preserve ZWJ (`U+200D`) and ZWNJ (`U+200C`) when they are part of Indic orthography; do not copy a broad zero-width deletion range into a production extractor. `[LOCAL]`

---

## 3. The shared mechanics: bytes → artifact → blocks → text

Different tools look unrelated at the API level, but most extraction systems implement the same logical stages.

### 3.1 Preserve before parsing

Before decoding or rendering:

1. Store the raw bytes or a durable object-store reference.
2. Compute a cryptographic source hash.
3. Store URL, crawl time, response status, MIME, charset, content length, and dataset revision.
4. Record redirect and content-negotiation decisions.
5. Detect whether the payload is actually HTML, PDF, image, subtitle, audio/video, or a pre-flattened text record.
6. Store parser/model versions and route reason.

A source hash is not only provenance. It makes a later claim falsifiable: if a page changes, the same URL does not imply the same input.

### 3.2 Decode bytes without confusing decoding with cleaning

A byte-to-text decoder should:

- use an explicit charset when the protocol and document agree;
- inspect an HTML/PDF declaration when present;
- apply a declared fallback policy;
- count replacement characters and decoding failures;
- preserve the undecoded bytes for reprocessing.

Do not silently “fix” mojibake by a guess that cannot be reproduced. The decoder’s output is an input to normalization, not proof that the source was valid.

### 3.3 Parse into a structure-preserving representation

The useful unit is usually a block, not a character:

```text
Block {
  source_locator: DOM XPath | byte span | page + bbox | time interval,
  semantic_type,
  text_or_children,
  order_key,
  confidence,
  script_and_language_hints,
  kept_or_discarded,
  reason
}
```

For HTML, `source_locator` can be an XPath plus a stable content hash. For PDF/OCR, it is page geometry. For speech, it is a time range. Keeping this locator allows a reviewer to answer “where did this sentence come from?” without trusting the final plain-text file.

### 3.4 Select, then serialize

A robust design separates:

1. **Selection:** classify or choose source regions.
2. **Structural conversion:** map regions to typed blocks.
3. **Serialization:** emit Markdown, JSONL, plain text, or another training representation.

Flattening before selection destroys evidence. Serializing a table as a space-separated string destroys cell boundaries. Removing timestamps before ASR QA destroys alignment. The intermediate representation is the seam that prevents these losses.

### 3.5 Confidence is route evidence, not semantic truth

A high OCR confidence can still correspond to the wrong glyph sequence. A high ASR confidence can still hallucinate a plausible name. A high web-extractor yield can mean that navigation survived. Confidence must be combined with:

- source/output span coverage;
- independent extractor disagreement;
- script/language evidence;
- structural consistency;
- sampled human review;
- modality-specific error metrics.

---

## 4. Technique map at a glance

| Technique family | Internal signal/mechanism | Best source | Strength | Main failure mode |
|---|---|---|---|---|
| API / site selector / schema extraction | Explicit endpoint, CSS/XPath, JSON-LD, or domain template | Stable domains, publisher feeds/APIs | Highest precision when the contract is real | Template drift or incomplete selector silently omits content |
| DOM parsing and semantic tag rules | Parsed tree, block tags, attributes, source order | Static HTML | Deterministic, auditable, structure-preserving | A `<div>`/`<header>` does not guarantee document role |
| Density / link / neighboring-block heuristics | Length, link density, stopword density, tags, adjacent classes | Large heterogeneous HTML | Fast, transparent, no model serving | Language-, domain-, and genre-sensitive thresholds |
| Readability-style semantic DOM extraction | Candidate nodes scored by text, links, classes, siblings, metadata | Conventional news/blog pages | Strong mature article baseline | Article assumptions fail on forums, tables, short Indic pages |
| High-throughput generic extractor | DOM pruning, text tests, structural tag conversion, duplicate checks | WARC/Common Crawl | Practical scalable baseline | Dynamic pages and unusual structures leak or flatten |
| Layout-aware HTML conversion | Tree traversal plus display/list/table/CSS handlers | HTML with useful structure | Preserves visual/block relationships | Converts layout but does not decide main content |
| Supervised block classifier | Hand-designed DOM/text features + learned classifier | Repeated domains with labelled pages | Learns combinations of weak signals | Gold-set and domain/script shift |
| Learned semantic sequence labelling | Model labels blocks; output selects original blocks | Difficult HTML and structured pages | Semantic decisions without generative rewriting | Model cost, data requirements, transfer uncertainty |
| Browser rendering | Execute JavaScript, capture rendered DOM/network payload | Client-rendered pages | Recovers content absent from initial HTML | Timing, consent, ads, user state, security, cost |
| Native PDF extraction | Embedded text objects, spans, blocks, coordinates | Digital PDFs | Exact and cheap when text layer is sound | Font mapping, columns, tables, reading order |
| OCR + layout analysis | Image preprocessing, text detection, recognition, geometry | Scanned PDFs/images | Handles image-only content | Indic matra/conjunct errors, boxes/order, confidence limits |
| Subtitle parsing | Cue index/time/text state machine | SRT/VTT | Cheap and deterministic | Editorial/translated/abridged cues, repeated text |
| VAD + ASR | Frame speech likelihood + acoustic model + decoder | Audio/video | Converts speech unavailable as text | WER/CER, code-switching, names, overlap, chunk seams |
| Residual flat-text audit | Literal tag/marker/banner patterns | Flattened JSONL | Honest final audit when source is gone | Cannot recover original boundaries or layout |

There is no universal winner. The correct question is: **what artifact and failure mode do we have?**

---

# Part I — Web and HTML

## 5. Common web extraction anatomy

A web extractor generally performs this sequence:

```text
raw response bytes
  ├─ charset/encoding detection
  ├─ HTML parse into a tolerant DOM
  ├─ remove executable/non-content nodes
  ├─ identify candidate blocks or containers
  ├─ score, classify, or select main content
  ├─ preserve typed structure and source mapping
  ├─ serialize selected blocks
  └─ emit warnings, yield, and confidence
```

The central difficulty is that HTML is a presentation language, not a corpus annotation. The same tag can mean different things across sites:

```html
<div class="article-body">...</div>  <!-- likely content -->
<div class="article-body">...</div>  <!-- may be a recommendation widget -->
```

Class names, link density, text length, neighboring blocks, and document semantics are evidence. None is an oracle.

### 5.1 DOM parser mechanics

A tolerant parser first turns malformed tag soup into a tree. It keeps element names, attributes, child order, text nodes, and tail text. An extractor can then:

- remove `script`, `style`, `template`, and executable embeds;
- retain or discard comments by policy;
- walk block elements in document order;
- account for text that sits before a child (`element.text`) and after a child (`child.tail`);
- preserve nested lists/tables/code rather than concatenating all descendants;
- attach DOM paths and stable content hashes to output blocks.

A regex that deletes `<...>` cannot reliably distinguish markup from text containing `<`, recover malformed nesting, or know that a navigation `<li>` differs from an article `<li>`.

### 5.2 DOM-to-block pseudocode

```python
for node in dom_in_document_order:
    if node.tag in EXECUTABLE_OR_HIDDEN:
        record_discard(node, "non_document_node")
        continue

    if node.tag in STRUCTURAL_TAGS:
        block = begin_block(node.tag, dom_path(node))
        append_text_nodes(node, preserve_inline_boundaries=True)
        close_block(block)
    else:
        append_to_current_block(node)

for block in blocks:
    block.text = decode_and_trim_without_destroying_script(block.text)
    block.source_hash = sha256(canonical_source_slice(block))
```

This is only blockization. Main-content selection happens afterward.

---

## 6. API, schema, and site-specific selectors

### 6.1 How it works

When a domain has an actual contract, prefer it:

- publisher API or RSS/Atom feed;
- sitemap-linked document endpoint;
- JSON-LD `articleBody` or schema metadata;
- stable CSS/XPath selector for a known CMS template;
- domain rule that retains a known content container and drops known sidebars.

Store the endpoint/selector, version, response hash, and fields present. Keep the visible HTML as a comparison artifact when licensing and storage policy permit.

### 6.2 Why it is technically attractive

A selector is a low-entropy decision function. Instead of asking a general algorithm to infer content from every node, it maps a known template to an explicit subtree. That gives:

- high precision on the covered domain;
- predictable cost;
- easy regression tests against stored pages;
- preservation of metadata and fields generic text extractors often lose;
- straightforward alerting when the selector returns empty or implausibly small output.

### 6.3 Failure modes

- A redesign moves the article body while the selector still returns a valid but unrelated sidebar.
- A localized template changes class names.
- JSON-LD is stale, duplicated, truncated, or generated from metadata rather than visible content.
- An article rule is applied to a homepage, forum, dictionary, PDF viewer, or search page.
- The selector returns the article but omits captions, tables, comments, or code.

The dangerous failure is not an exception; it is a plausible, short, incomplete document.

### 6.4 Controls

For each domain rule, monitor:

```text
empty_rate
retained_chars / source_chars
retained_blocks by type
article_body selector match rate
output-source span coverage
schema-vs-visible-text disagreement
new-template cluster rate
```

Use a small labelled fixture set per domain. Promote a rule only after it passes a template-change test and a Hindi-literate review sample.

**Use it when:** domain concentration is high or an official feed/API exists. Treat it as a high-precision route, not a universal parser.

**Evidence:** this is a design recommendation; the local flattened cache has no source DOM on which to exercise selectors. `[LOCAL]`

---

## 7. DOM parsing and semantic tag rules

### 7.1 Mechanics

A semantic rule pass parses the page, removes clearly non-document nodes, and walks content-bearing elements. Typical evidence includes:

- `<main>` and `<article>` containers;
- heading hierarchy `h1`–`h6`;
- paragraphs and block quotes;
- ordered/unordered lists and list items;
- `pre`/`code` blocks;
- table rows/cells and captions;
- `figure`/`figcaption`;
- ARIA roles such as `main`, `navigation`, and `complementary`;
- class/id hints and link density.

Use these as features, not unconditional truth. A scholarly article’s references can be under `<footer>`, and a page’s real content can be inside a generic `<div>`.

### 7.2 What it preserves

A tree walk can retain:

- block boundaries and nesting;
- source order;
- DOM paths and source spans;
- inline emphasis and links as metadata;
- table/list/code/formula type;
- a precise discarded-node reason.

### 7.3 What it cannot decide alone

The DOM does not tell us whether:

- a sidebar is topical content or navigation;
- a `<header>` is a site masthead or the document’s actual title;
- a list is a menu or the body of an educational article;
- a comment thread is in scope;
- an advertisement contains useful text;
- client-side content is absent because the page was not rendered.

**Use it when:** structure is reasonably authored and a deterministic first pass is desired. Emit blocks and confidence instead of flattening immediately.

---

## 8. Density, link-density, and neighboring-block heuristics

JusText is a useful reference implementation because its published source makes the heuristic decision path inspectable. `[PRIMARY]`

### 8.1 Blockization

JusText first decodes HTML, parses it into a DOM, removes configured unwanted parts, and uses paragraph-like tags to emit paragraph objects. Each paragraph accumulates:

- `text`;
- character length;
- characters inside links;
- tag count and DOM path;
- stopword count/density;
- heading status.

A key detail is that paragraph formation is not simply “one `<p>` equals one unit”; many block-like tags and `<br><br>` boundaries participate. Text inside an anchor is counted separately so a link-heavy block can be recognized.

### 8.2 Context-free classification

The default source constants are:

```text
max_link_density = 0.20
length_low       = 70 characters
length_high      = 200 characters
stopwords_low    = 0.30
stopwords_high   = 0.32
```

These are algorithm defaults, not Hindi-safe values. The simplified rule is:

```text
if link_density > 0.20:
    bad
elif copyright_marker or select_path:
    bad
elif length < 70:
    short unless linked, then bad
elif stopword_density >= 0.32:
    good if length > 200 else neargood
elif stopword_density >= 0.30:
    neargood
else:
    bad
```

The actual implementation also marks headings and uses a configured stoplist. A compact abstraction is:

$$
L(b)=\frac{\text{linked characters in }b}{\text{characters in }b},\qquad
S(b)=\frac{\text{stopwords in }b}{\text{words or tokens in }b}.
$$

### 8.3 Context-sensitive revision

Short and near-good blocks are then revised using neighboring `good`/`bad` blocks:

- a short block between two good regions becomes good;
- a short block between two bad regions becomes bad;
- a heading near a good paragraph can be promoted;
- near-good blocks inherit a class from surrounding content.

This is valuable because headings and short dialogue turns are not independently “dense” but can be meaningful in context.

### 8.4 Strengths

- Very fast and low memory.
- No GPU or model serving.
- Every decision is inspectable.
- Neighbor context prevents some isolated short-block deletions.
- Easy to calibrate per source family when a gold set exists.

### 8.5 Failure modes on Hindi/Indic data

- English stoplists do not measure Hindi, Sanskrit, Telugu, Romanized Hindi, or code-mixed text correctly.
- Devanagari words may not be segmented like English words.
- Short headlines, poems, bullets, dialogue, tables, and code can be marked short or low-density.
- Link-heavy article text can be discarded.
- A grammatical navigation block can pass.
- A page with unusual whitespace or malformed structure can produce misleading paragraph boundaries.

**Hindi rule:** use an Indic stoplist only as one feature, never as a hard gate. Calibrate length, link, and density thresholds on Hindi, Romanized, code-mixed, list-heavy, and technical examples.

**Use it when:** a CPU-friendly transparent baseline is needed and the page family resembles its assumptions. Use an ensemble or a learned fallback when uncertainty is high.

**Source:** [JusText core implementation](https://raw.githubusercontent.com/miso-belica/justext/master/justext/core.py), [algorithm description](https://raw.githubusercontent.com/miso-belica/jusText/master/doc/algorithm.rst). `[PRIMARY]`

---

## 9. Readability-style semantic DOM extraction

Mozilla Readability is a representative article extractor. It is implemented as a sequence of DOM cleanup, candidate scoring, best-candidate selection, and post-cleaning passes. `[PRIMARY]`

### 9.1 Candidate preparation

The implementation configures:

- a maximum element count guard;
- number of top candidates to compare;
- minimum character threshold;
- classes to preserve;
- optional JSON-LD parsing;
- link-density modifier;
- flags for stripping unlikely nodes, weighting classes, and conditional cleaning.

It has regular expressions for likely/negative class names (`article`, `content`, `sidebar`, `footer`, `related`, and similar clues), but class names are only evidence.

### 9.2 Scoring model

The high-level algorithm is:

1. Normalize the document structure.
2. Remove scripts, styles, unlikely candidates, and irrelevant elements.
3. Score candidate containers using tag type, class/id hints, text length, comma count, punctuation, and link density.
4. Select the strongest candidate or a merged set of top candidates.
5. Include suitable siblings when their score and relationship justify it.
6. Clean conditionally: remove ads, short paragraphs, excessive links, and unrelated containers.
7. Return article HTML/text plus metadata such as title, byline, site name, direction, language, and publication clues.

The precise scoring code is intentionally heuristic. The critical operational fact is that it returns a selected content subtree rather than merely deleting tags.

### 9.3 Strengths

- Mature article/news baseline.
- Candidate comparison is more robust than one selector.
- Returns structure and metadata useful for provenance.
- Large regression corpus and browser-reader usage provide operational maturity.

### 9.4 Failure modes

- Reader-view assumptions do not equal pretraining-corpus policy.
- Article-centric scoring can drop forums, reference pages, documentation, short Hindi items, tables, and chat.
- Class/id and punctuation heuristics may encode English/news assumptions.
- Dynamic content is absent unless the DOM is rendered.
- Teasers/related links may survive; captions and structured elements may be omitted.
- It can return a successful result with a wrong boundary.

**Use it as:** a candidate generator and web baseline. Compare retained blocks to another route, preserve source mapping, and measure omission and boilerplate leakage separately.

**Source:** [Mozilla Readability source](https://raw.githubusercontent.com/mozilla/readability/main/Readability.js), [repository](https://github.com/mozilla/readability). `[PRIMARY]`

---

## 10. Trafilatura and Resiliparse: production-style generic baselines

### 10.1 Trafilatura mechanics

Trafilatura is not just `text_content()`. Its extraction path operates on an `lxml` tree and performs a configurable sequence of:

1. title and heading handling;
2. text-node processing and trimming;
3. removal of unwanted XPath-selected nodes;
4. link-density and table-link-density checks;
5. paragraph/list/quote/code/image handling;
6. duplicate or near-duplicate block checks within a document;
7. formatting-preserving conversion to an output tree;
8. serialization to text, XML, JSON, or other output formats.

The source implementation has explicit sets for formatting, quotes, tables, code, and inline-carried tags. It rewires descendants into a clean output tree rather than just concatenating strings. It protects selected attributes such as `role`, `src`, `alt`, and `title`, and performs tests such as `text_chars_test` before retaining a block.

A simplified view is:

```text
DOM
 ├─ discard configured XPath regions
 ├─ process headings and titles
 ├─ route lists/tables/quotes/code to type handlers
 ├─ clean text nodes and preserve selected inline structure
 ├─ reject empty or weak blocks
 ├─ check duplicate/teaser/link-heavy regions
 └─ serialize an output tree + metadata
```

That architecture matters for Hindi technical pages: lists, code, formulas, references, and tables can remain distinct until a later serialization choice.

### 10.2 Trafilatura strengths

- Fast static-HTML/WARC baseline.
- Configurable precision/recall behavior.
- Metadata, lists, quotes, code, links, images, and tables can be represented.
- Deterministic enough for reproducible corpus runs when versions/configuration are pinned.
- Useful as the first route in a cascade.

### 10.3 Trafilatura failure modes

- Generic rules cannot know every domain’s semantics.
- Dynamic pages can expose noisy or incomplete initial markup. IndicLLMSuite reports considerable noise from Trafilatura specifically on dynamic webpages. `[PRIMARY]`
- Structured content still needs explicit retention checks.
- A clean output tree is not proof of correct main-content selection.
- The local flat JSONL cache lacks the input DOM, so running Trafilatura there would not be a fair extraction experiment. `[LOCAL]`

### 10.4 Resiliparse mechanics

Resiliparse is a performance-oriented extraction utility. Its `HTML2Text` path parses HTML into an internal tree and converts visible text nodes into text while respecting node boundaries. The documented `main_content=True` mode applies a rule set intended to remove navigation blocks, sidebars, footers, advertisements, and invisible elements. The project also provides DOM, encoding, WARC, and process-guard utilities.

The important distinction is:

```text
HTML2Text                  -> visible/body text conversion
HTML2Text(main_content=True) -> visible text + main-content pruning rules
```

A caller should retain the raw and block-aware path when tables, code, or sidebars are important. A fast text-only conversion cannot reconstruct structure after flattening.

### 10.5 Resiliparse strengths and risks

**Strengths**

- High-throughput, CPU-friendly parsing.
- Useful for WARC/Common Crawl style workloads.
- Explicit main-content option.
- Good baseline for comparison with Trafilatura.

**Risks**

- Rule-based main-content detection can miss domain-specific semantics.
- Flattening can lose table/code/formula boundaries.
- Dynamic pages are only as complete as the input HTML.
- A benchmark result from another language or genre does not establish Hindi fidelity.

**Use them as:** paired deterministic baselines. Compare output spans, block types, retained fraction, and disagreement. Escalate only the uncertain slice rather than paying model/rendering cost for every page.

**Sources:** [Trafilatura documentation](https://trafilatura.readthedocs.io/en/latest/), [Trafilatura main extractor](https://raw.githubusercontent.com/adbar/trafilatura/master/trafilatura/main_extractor.py), [Resiliparse extraction documentation](https://resiliparse.chatnoir.eu/en/stable/man/extract.html), [AICC baseline discussion](https://arxiv.org/html/2511.16397#S3.SS2.SSS1). `[PRIMARY]`

---

## 11. Layout-aware HTML-to-text conversion

### 11.1 Why layout conversion is separate from main-content selection

A page can have the correct selected subtree and still be serialized incorrectly. Naively concatenating text nodes can turn:

```html
<table>
  <tr><td>Hindi</td><td>हिन्दी</td></tr>
</table>
```

into an ambiguous token stream. A layout-aware converter instead records row/cell boundaries or emits a table representation.

### 11.2 Inscriptis mechanics

Inscriptis is a representative layout-aware converter. Its implementation:

1. parses an `lxml` HTML tree;
2. creates a document state containing a tag stack and a canvas;
3. dispatches start/end handlers for tables, rows, cells, lists, links, line breaks, and images;
4. writes text and tail text into the current canvas location;
5. applies configured layout rules and whitespace/indentation behavior;
6. returns text or annotations.

The tag handler tables are explicit. Tables invoke table/row/cell handlers; lists invoke ordered/unordered/list-item handlers; links and images are configurable. That makes the conversion more faithful than a generic descendant-text join, but it does not tell the converter whether a sidebar is relevant.

### 11.3 Strengths

- Preserves list/table boundaries better than plain text concatenation.
- Supports selected CSS/layout behavior.
- Annotations can help create a manual reference set.
- Useful after a main-content subtree has already been selected.

### 11.4 Failure modes

- CSS and browser layout are incomplete/complex.
- Visual order can differ from semantic or source order.
- Layout tables can be mistaken for data tables.
- Main-content classification remains a separate problem.

**Use it as:** a representation-preserving serializer, not a standalone boilerplate classifier.

**Source:** [Inscriptis repository and documentation](https://github.com/weblyzard/inscriptis), [HTML engine](https://raw.githubusercontent.com/weblyzard/inscriptis/master/src/inscriptis/html_engine.py). `[PRIMARY]`

---

## 12. Supervised block extraction

### 12.1 Mechanics

A supervised extractor turns web extraction into binary or multiclass block classification:

```text
DOM → blocks → features → classifier → main/other labels → original-block selection
```

Features can include:

- block text length and punctuation;
- link density and anchor count;
- tag-name and class/id indicators;
- DOM depth and sibling position;
- neighboring block labels/features;
- character/script ratios;
- readability/JusText signals;
- repeated-template indicators.

Dragnet is a representative feature-based system. Its source has a blockifier, feature extractors, and a scikit-learn-compatible model; it can target article content and comments separately. `[PRIMARY]`

### 12.2 Why it can beat one threshold

A classifier can learn that a short heading near long paragraphs is likely content, while a long link-heavy block inside a navigation region is not. It can combine weak signals without making every rule hand-written.

### 12.3 Costs and failure modes

- Requires labelled pages and precise block-to-gold alignment.
- A model trained on English news can fail on Hindi, Romanized text, forums, documentation, or poetry.
- Template and domain shift can be severe.
- Features can encode hidden assumptions such as whitespace, punctuation, or ASCII ratio.
- A confident wrong label still needs source-span auditing.

### 12.4 When to use it

Use a supervised block model when:

- the same domains/templates recur;
- a Hindi-literate gold set exists;
- deterministic baselines disagree often;
- model inference is cheaper than manual selector maintenance.

A conservative deployment pattern is to use it only on the uncertain slice and retain a deterministic route as a fallback.

**Source:** [Dragnet repository](https://github.com/dragnet-org/dragnet), [extractor implementation](https://raw.githubusercontent.com/dragnet-org/dragnet/master/dragnet/extractor.py). `[PRIMARY]`

---

## 13. Learned semantic block labelling: MinerU-HTML / AICC

This is the most technically different web route in the guide because it treats extraction as constrained sequence labelling rather than free-form generation. `[PRIMARY]`

### 13.1 Stage 1 — Main-HTML selection

The AICC paper describes a three-part core pipeline:

```text
raw HTML
  ├─ Simplified HTML: compact model input
  └─ Mapping HTML: original structure for faithful reconstruction
        ↓
0.6B MinerU-HTML-Classifier
        ↓
constrained labels: block_1=main/other, block_2=main/other, ...
        ↓
select original blocks from Mapping HTML
        ↓
Main-HTML: content-bearing subset of the original DOM
```

The two synchronized representations solve a practical tension:

- the model should not spend tokens on layout-only markup and irrelevant attributes;
- the output must still select exact original nodes rather than regenerated text.

The classifier performs sequential block classification. A custom logits processor constrains decoding with a finite-state-machine-like format so the output remains a valid label structure. The postprocessor maps predicted labels back onto the original HTML.

### 13.2 Why selection instead of generation matters

If the model is only allowed to label blocks and the final output is copied from source nodes:

- hallucinated sentences are structurally prevented;
- source-span coverage can be checked exactly;
- formatting and attributes can survive selection;
- model output is compact compared with generating a rewritten article.

This does not eliminate errors. The model can label the wrong block, omit a valid block, or retain boilerplate. It changes the error from invented text to selection error, which is easier to audit.

### 13.3 Stage 2 — typed content list and formatting

AICC then converts Main-HTML to a structured content list with semantic types such as:

- title and paragraph;
- inline/display formula;
- inline/code block;
- table;
- list;
- image, video, and audio.

The content list is an intermediate representation. Markdown or another output format is rendered afterward. This is the right abstraction for training data because the same selected source can support different downstream policies.

### 13.4 Template-aware scaling

At web scale, the paper clusters structurally similar pages, runs the model on representative pages, maps labels back to DOM positions, synthesizes XPath/CSS retain/prune rules, and applies those rules to the cluster. This trades per-page model inference for a template-level rule after validating that the cluster is homogeneous.

### 13.5 Evidence and boundaries

On WebMainBench, the paper reports overall ROUGE-N F1 of `0.8182` for MinerU-HTML, `0.6358` for Trafilatura, and `0.6233` for Resiliparse. It reports especially large differences on tables, code, equations, and conversational pages. These are benchmark results on the paper’s multilingual annotated pages, not measurements for this local Hindi/Indic cache. `[PRIMARY]`

### 13.6 Strengths and risks

**Strengths**

- Semantic context for ambiguous pages.
- Constrained selection avoids free-form hallucination.
- Better structured-element preservation in the cited benchmark.
- Template distillation can reduce web-scale inference cost.

**Risks**

- Requires model serving, training data, and evaluation infrastructure.
- Language/template transfer must be measured, not assumed.
- Simplification can hide useful evidence if preprocessing is wrong.
- A model can be confidently wrong in a less obvious way than a rule.
- It cannot recover DOM structure from the local flattened JSONL.

**Use it as:** a fallback or teacher for difficult HTML, and only after a Hindi/Indic evaluation set demonstrates benefit over deterministic baselines.

**Sources:** [AICC paper](https://arxiv.org/html/2511.16397), [core extraction section](https://arxiv.org/html/2511.16397#S2.SS1.SSS1), [structured formatting section](https://arxiv.org/html/2511.16397#S2.SS2), [benchmark results](https://arxiv.org/html/2511.16397#S3.SS3). `[PRIMARY]`

---

## 14. Browser rendering for dynamic pages

### 14.1 Mechanics

Static extraction only sees the initial response. A controlled browser route:

1. fetches and hashes the initial HTML;
2. checks for missing/short/script-heavy output;
3. opens the page in an isolated browser context;
4. waits for a documented readiness condition or network/API response;
5. captures the rendered DOM and relevant network payload;
6. hashes the rendered artifact;
7. runs the same block extractor on the rendered DOM;
8. records timing, redirects, consent state, and failures.

The browser is not the extractor. It is an artifact acquisition step before extraction.

### 14.2 Why rendering recovers content

Client-rendered pages may deliver only:

```html
<div id="app"></div>
<script src="bundle.js"></script>
```

The article arrives later through an API or is constructed by JavaScript. Rendering can recover it when the initial response has no content-bearing blocks.

### 14.3 Risks

- Cookie banners and consent text become part of the DOM.
- Personalized recommendations and user-state content contaminate output.
- Infinite scroll can cause non-deterministic page length.
- Ads, trackers, and untrusted scripts increase security exposure.
- Timing and network state make reproducibility harder.
- Login, robots, anti-bot, or legal restrictions can block the route.

### 14.4 Trigger policy

Use rendering only when a static route produces:

- empty or near-empty output;
- a script-heavy page with missing content;
- known dynamic-domain evidence;
- high disagreement between extractors;
- suspicious output/source ratios;
- a high-value page worth the extra cost.

Save a failure reason instead of retrying forever.

**Source:** [Playwright pages documentation](https://playwright.dev/python/docs/pages). `[PRIMARY]`

---

## 15. Structured HTML elements: preserve before flattening

A prose-only extraction route is insufficient for technical and educational pages. Each type has a different fidelity invariant.

| Element | Preserve | Common destructive shortcut |
|---|---|---|
| Heading | level, order, text | concatenate into paragraph |
| List | ordered/unordered status, item boundaries, nesting | join bullets with spaces |
| Table | row/cell/span relationships, caption | linearize cells into prose |
| Code | indentation, language hint, fenced boundary | collapse whitespace |
| Formula | MathJax/KaTeX/MathML or image link, inline/display status | keep only fallback visible text |
| Figure/caption | caption, source reference, relation to figure | retain caption detached from context |
| Link | anchor text + URL metadata | turn every URL into prose or drop all links |
| Comments/forum | author/time/turn boundaries when in policy | treat each line as article prose |

The AICC content-list design and Inscriptis annotation model both illustrate the same principle: choose a typed intermediate representation before deciding how the training serialization should look.

### 15.1 Structured-element checks

For every web document, record:

```text
input_tables, output_tables, retained_table_cells
input_code_blocks, output_code_blocks, indentation_preserved
input_formulas, output_formulas, source_markup_retained
input_headings, output_headings, hierarchy_preserved
```

A document with high text recall but zero retained tables is not a successful extraction for a table-bearing corpus.

---

## 16. Web route decision matrix

| Situation | First choice | Escalate to | Avoid |
|---|---|---|---|
| Stable Hindi news domain | Site selector/API + typed DOM | Browser render or learned block model after template drift | Universal rule without domain tests |
| Static heterogeneous web crawl | Trafilatura/Resiliparse block output | JusText/Readability/Dragnet comparison on uncertain pages; semantic model on hard pages | Whole-page `get_text()` or regex tag stripping |
| JavaScript article body | Controlled browser render + generic extractor | Network/API capture or semantic block model | Treating initial HTML as complete |
| Table/code/formula-heavy page | Typed DOM/layout representation | Constrained semantic extractor | Flattening to whitespace-separated prose |
| Repeated site templates | Domain rule or template-distilled rule | Per-page model/manual review on drift | Applying a rule outside its cluster |
| Forum/Q&A/conversational page | Block-preserving route with comment policy | Learned semantic classifier | Article-only readability assumptions |
| Small high-value sample | Two extractors + human adjudication | Source-specific selector | Overengineering global model before measuring |
| Flattened JSONL | Residual literal-artifact audit | Reacquire source artifacts | Claiming DOM extraction occurred |

---

# Part II — PDF, scanned document, and image extraction

## 17. Route by text-layer quality

A PDF is a container, not a promise of readable text. A page can have selectable text and still contain:

- broken font-to-Unicode mappings;
- character order unrelated to visual order;
- columns interleaved;
- spaces missing or inserted;
- repeated headers/footers;
- table cells emitted as unrelated spans;
- rotated or hidden text.

Run a quality probe before deciding whether OCR is necessary.

### 17.1 Text-layer probe

For each page, measure:

```text
extracted_characters
replacement_character_count
whitespace_fraction
script distribution
word/line count
bbox overlap
repeated header/footer candidates
native reading-order plausibility
native-vs-rendered visual spot check
```

A simple route is:

```text
usable native text layer → native blocks/words + coordinates
poor native layer          → render + OCR + retain native for comparison
complex layout             → layout parser/document AI, even if text exists
low confidence/disagreement → quarantine or manual review
```

Do not discard the native result when OCR is used. The disagreement is diagnostic evidence.

---

## 18. Native PDF text extraction with PyMuPDF

### 18.1 PDF text model

PDF content is often positioned glyph drawing instructions rather than a semantic paragraph tree. A text extractor reconstructs characters/spans from content streams, font encodings, and coordinates. It then groups spans into lines and blocks using geometry and sorting heuristics.

That is why a PDF viewer’s visual copy order is not evidence that `page.get_text()` will return the same order.

### 18.2 PyMuPDF output levels

PyMuPDF exposes several useful granularities:

```python
page.get_text("text")   # plain text
page.get_text("blocks")  # block tuples with bbox/text and related fields
page.get_text("words")   # word tuples with bbox, block/line/word indices
page.get_text("dict")    # hierarchical structured representation
```

The exact tuple/dictionary schema is version-sensitive, so pin the version and record the output mode. The documented `sort` option can request a coordinate-based ordering, but sorting is not a guarantee of semantic reading order for every multi-column document.

A geometry-aware reconstruction commonly:

1. groups words by vertical overlap/tolerance;
2. sorts words inside a line by x-coordinate;
3. groups lines into columns/blocks;
4. identifies repeated running headers/footers using page frequency and geometry;
5. preserves page and block locators;
6. emits a table object or sends the page to a specialized layout route when grid structure is detected.

### 18.3 Strengths

- Fast and inexpensive.
- Exact when the embedded text layer and order are correct.
- Coordinates support columns, headers, footers, tables, and page-level review.
- Avoids OCR recognition errors.

### 18.4 Failure modes

- Creator-dependent order.
- Font mapping can corrupt Devanagari.
- Multi-column, footnote, rotated, and table layout requires geometry logic.
- A visually correct PDF can have a semantically unusable text layer.

**Source:** [PyMuPDF text recipes](https://pymupdf.readthedocs.io/en/latest/recipes-text.html), [Appendix text extraction](https://pymupdf.readthedocs.io/en/latest/app1.html). `[PRIMARY]`

---

## 19. OCR pipeline anatomy

For image-only input, OCR is a chain, not one function:

```text
PDF/image
  ├─ render at chosen DPI
  ├─ orientation / deskew / dewarp / denoise / contrast
  ├─ page or region layout detection
  ├─ text-line/word detection
  ├─ script/language route
  ├─ glyph/sequence recognition
  ├─ boxes + confidence + reading order
  ├─ table/formula/image handling
  └─ typed document representation
```

Every stage can introduce a different error:

- rendering loses resolution or clips margins;
- preprocessing removes thin matras or joins strokes incorrectly;
- layout detection merges columns or misses marginal text;
- recognition substitutes a visually similar conjunct;
- ordering interleaves columns;
- serialization loses boxes and page boundaries.

The original page image and preprocessing parameters are part of the evidence.

---

## 20. Tesseract 5: LSTM OCR baseline

### 20.1 Internal model path

Tesseract 4 introduced an OCR engine based on LSTM neural networks; Tesseract 5 supplies language/script model data in traineddata packages. `[PRIMARY]`

Operationally, the route is:

```text
page image
  ├─ page segmentation mode estimates regions/lines
  ├─ layout/line segmentation
  ├─ normalize image/line geometry
  ├─ LSTM sequence recognizer predicts character sequence
  ├─ language model / decoding chooses a path
  └─ output text, boxes, confidence, and optional TSV/HOCR
```

The exact layout path depends on page segmentation mode and input. Tesseract can be configured for Devanagari/Hindi data, but a language pack is not a Hindi accuracy guarantee.

### 20.2 Strengths

- Open, local, reproducible baseline.
- LSTM models available for many languages/scripts.
- Useful output formats expose word boxes and confidence.
- Easy to run as a second engine for disagreement sampling.

### 20.3 Failure modes for Indic scans

- Shirorekha and connected glyphs can be split or merged.
- Matras can be attached to the wrong base or emitted in the wrong order.
- Virama/halant, conjuncts, nukta, and legacy fonts are difficult.
- Skew, bleed-through, low DPI, compression, and multi-column order propagate errors.
- Confidence is not a Unicode correctness proof.

### 20.4 Acceptance use

Use Tesseract as:

- the local baseline for a paired OCR experiment;
- a fallback for simple, high-quality pages;
- an independent disagreement source against another OCR engine.

Review CER, script/code-point preservation, boxes, and page crops. Do not accept a page because it produced a long string.

**Sources:** [Tesseract user manual](https://tesseract-ocr.github.io/tessdoc/), [Tesseract LSTM notes](https://tesseract-ocr.github.io/tessdoc/), [training documentation](https://tesseract-ocr.github.io/tessdoc/tess5/TrainingTesseract-5.html). `[PRIMARY]`

---

## 21. PaddleOCR / PP-Structure: modular document analysis

### 21.1 Pipeline mechanics

PP-StructureV3 documents a modular pipeline containing independent or composable modules for:

- document orientation classification;
- text-image correction/dewarping;
- layout region detection;
- general OCR;
- table recognition;
- formula recognition;
- seal recognition;
- chart parsing.

The common shape is:

```text
page image
  ├─ optional orientation/correction
  ├─ layout detector → text/title/table/formula/figure/header/footer regions
  ├─ OCR text regions → boxes + strings + confidence
  ├─ table/formula/chart specialized recognizers
  ├─ reading-order reconstruction
  └─ Markdown/JSON/structured output
```

The documentation lists layout categories such as title, paragraph heading, text, page number, abstract, table, formula, figure, caption, reference, footnote, header, footer, and sidebar. This makes it more suitable than plain OCR when document structure matters.

### 21.2 Strengths

- Explicit layout and structured-element route.
- Can recover tables, formulas, figures, and reading order.
- Modules can be independently replaced or tuned.
- Supports local deployment and further training/tuning.

### 21.3 Risks and evidence boundary

- Layout labels and OCR performance depend on the selected model and its training distribution.
- Official benchmark numbers are on the project’s evaluation data, not necessarily Hindi books or low-quality Indic scans.
- More modules mean more version/model provenance to record.
- A wrong layout region can remove entire text blocks before OCR.

Treat PP-Structure as a candidate route. Validate Hindi/Devanagari, code-mixed lines, tables, and scans representative of the corpus.

**Source:** [PP-StructureV3 official documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PP-StructureV3.md). `[PRIMARY]`

---

## 22. Cloud/layout OCR and IndicLLMSuite’s page evidence

IndicLLMSuite reports using Vision OCR for document preparation and retaining hierarchical OCR information such as pages, blocks, paragraphs, words, characters, coordinates, and confidence. It also describes filters for overlapping boxes, sparsity, high-overlap pages, low script confidence, and page concatenation. `[PRIMARY]`

The lesson is not “use this cloud engine.” The lesson is that OCR text without geometry is hard to audit. A practical record should resemble:

```text
Page {
  page_number,
  image_hash,
  width, height, dpi,
  blocks: [
    {
      type: text | heading | table | formula | figure | header | footer,
      bbox,
      text,
      words: [{text, bbox, confidence}],
      script_confidence,
      reading_order,
      warnings
    }
  ]
}
```

### 22.1 Layout checks as review signals

Useful checks include:

- high pairwise box overlap;
- boxes whose order is inconsistent with reading direction;
- many tiny fragments or one giant page block;
- high horizontal/vertical sparsity;
- low script confidence;
- repeated headers/footers across pages;
- native/OCR disagreement;
- missing page regions.

These should usually quarantine or escalate a page. A sparse title page, form, poem, diagram, or table can be valid; deleting it solely because it is sparse is an extraction error.

### 22.2 Strengths and risks

**Strengths:** geometry, confidence, page-level quarantine, structured output.

**Risks:** cloud privacy/licensing, cost, network reproducibility, box heuristics, and the difference between recognition confidence and semantic correctness.

**Source:** [IndicLLMSuite Setu document preparation](https://arxiv.org/html/2403.06350#S4), [OCR/layout sections](https://arxiv.org/html/2403.06350#S4.SS1). `[PRIMARY]`

---

## 23. OCR layout reconstruction and complex documents

### 23.1 Reading order

For a two-column page, the correct order is usually column 1 top-to-bottom followed by column 2, not global y-coordinate order. A reconstruction algorithm may:

1. cluster text boxes into columns using x-overlap and gaps;
2. sort within each cluster by y then x;
3. detect headings spanning columns;
4. place captions near their figures;
5. keep footnotes and marginalia as typed blocks;
6. preserve page breaks and the original boxes.

Every heuristic needs a manual review sample because magazines, newspapers, forms, and Indic books violate simple column assumptions.

### 23.2 Tables

A table extractor should retain a grid or cell graph:

```text
Table {
  bbox,
  rows: [
    {cells: [{rowspan, colspan, bbox, text, confidence}]}
  ],
  caption,
  source_page
}
```

If cell boundaries are not reliable, do not emit a confident prose table. Quarantine or retain the table image/HTML for a specialized route.

### 23.3 Indic-specific OCR review

Review crops containing:

- shirorekha and joined glyphs;
- matra placement and orphaned marks;
- virama/halant and conjunct formation;
- nukta-bearing letters;
- legacy non-Unicode fonts;
- skew, bleed-through, low DPI;
- mixed Hindi-English lines and numerals.

A visually plausible glyph is not proof of a correct Unicode sequence. Compare OCR output to page crops and retain confidence/geometry evidence.

---

# Part III — Audio, video, subtitles, and speech

## 24. Subtitle extraction: deterministic but not necessarily faithful

### 24.1 SRT/VTT mechanics

SRT and VTT are small state machines:

```text
cue index (optional)
start --> end
cue text lines
blank separator
```

A parser should retain raw cue text and timing first, then emit:

```text
Utterance {
  cue_id,
  start_time,
  end_time,
  raw_text,
  normalized_text,
  speaker_hint,
  cue_type: speech | sound | music | metadata,
  warnings
}
```

Only after QA should adjacent cues be joined. Join by time and speaker/boundary evidence, not by arbitrary file order.

### 24.2 Cleanup decisions

Common explicit operations include:

- remove timestamps from the final training text while retaining them in the sidecar;
- remove character cues such as `[MUSIC]` only if the corpus policy says they are not document content;
- merge continuation ellipses carefully;
- reject malformed timestamps or overlapping cues for review;
- detect translated/abridged subtitles rather than treating them as verbatim speech.

IndicLLMSuite describes timestamp removal and filtering of character cues/continuation ellipses before treating a processed subtitle file as a document. `[PRIMARY]`

### 24.3 Strengths and risks

**Strengths:** cheap, deterministic, often cleaner than ASR, existing editorial boundaries.

**Risks:** subtitles can be abridged, translated, censored, speaker-ambiguous, or full of sound cues. Removing timing first makes alignment QA impossible.

---

## 25. Voice activity detection (VAD)

### 25.1 Why VAD is a separate extraction decision

Long media includes silence, music, applause, and noise. VAD proposes intervals likely to contain speech so that ASR can process bounded chunks and avoid spending compute on non-speech.

The WebRTC VAD interface accepts 16-bit mono PCM at 8, 16, 32, or 48 kHz and 10, 20, or 30 ms frames. It exposes aggressiveness modes 0–3. `[PRIMARY]`

### 25.2 Internal decision path

A conceptual frame route is:

```text
PCM frame
  ├─ validate sample rate, duration, channel, byte width
  ├─ compute frame energy/spectral features
  ├─ compare speech/noise likelihood in frequency bands
  ├─ apply aggressiveness threshold
  └─ boolean speech decision
```

The WebRTC implementation uses frame-level feature extraction and Gaussian-mixture likelihood logic with mode-dependent thresholds. It is not a transcript model and does not know Hindi.

### 25.3 Chunk assembly

A robust chunker:

- converts media to a stable mono PCM representation;
- marks speech frames;
- merges adjacent speech frames;
- adds context/overlap around boundaries;
- enforces minimum/maximum chunk duration;
- records every cut and merge;
- retains silence and rejected spans in the sidecar for audit.

Without overlap, a word at a boundary can be clipped or duplicated. Without a maximum, a long conversation can exceed model context. Both choices are measurable.

### 25.4 Failure modes

- Aggressive mode can delete low-volume Indic speech.
- Music or noise can trigger false speech.
- Overlapping speakers can produce one mixed segment.
- Very short turns can be merged incorrectly.
- Resampling or channel conversion can alter recognition quality.

**Source:** [py-webrtcvad repository](https://github.com/wiseman/py-webrtcvad), [VAD core](https://raw.githubusercontent.com/wiseman/py-webrtcvad/master/cbits/webrtc/common_audio/vad/vad_core.c). `[PRIMARY]`

---

## 26. ASR model routes

### 26.1 Common signal chain

```text
audio
  ├─ decode and resample
  ├─ VAD/segment with context
  ├─ acoustic features (often log-Mel/filterbank)
  ├─ encoder maps time-frequency frames to hidden states
  ├─ decoder/CTC/transducer predicts text tokens
  ├─ timestamps, confidence, language, optional diarization
  └─ time-ordered utterance document
```

ASR is not a lossless transcription operation. It is an inference model whose errors become training text unless measured and filtered.

### 26.2 Hindi Conformer-style route

IndicLLMSuite describes a route using WebRTC VAD, a Hindi Conformer, an Indic punctuation model, and whole-video reassembly. `[PRIMARY]`

The Conformer paper describes a block combining global self-attention with local convolution:

```text
x̃ = x + 1/2 FFN(x)
x' = x̃ + MHSA(x̃)
x'' = x' + Conv(x')
y = LayerNorm(x'' + 1/2 FFN(x''))
```

Mechanically:

- the feed-forward modules provide channel-wise nonlinear transformation;
- multi-head self-attention captures long-range dependencies across the utterance;
- the convolution module captures local time patterns efficiently;
- relative positional information helps with variable utterance length;
- the encoder is paired with a recognition decoder such as a transducer in the cited system.

The convolution module uses a pointwise projection/GLU, depthwise 1-D convolution, normalization, and Swish-like activation. `[PRIMARY]`

This architecture is attractive for Hindi speech because both local phonetic patterns and longer context matter, but architecture alone does not establish dialect, code-mix, name, or noise performance.

### 26.3 Whisper-style multilingual route

Whisper uses an encoder-decoder Transformer, 16 kHz audio, 80-channel log-Mel spectrograms, and a multilingual tokenizer. Its multitask token format can represent transcription, translation, language identification, and related controls. `[PRIMARY]`

This provides a broad fallback when a Hindi-specific model is unavailable. It does not remove the need for Hindi/Hinglish WER/CER testing; the paper itself reports that language-identification performance is not uniformly competitive across languages.

### 26.4 Punctuation restoration

Punctuation can be a useful later representation layer but is not evidence that punctuation existed in the audio. Keep:

```text
raw ASR tokens/timestamps
punctuation-restored text
punctuation model/version
```

separate. Do not allow punctuation restoration to obscure deletions, repetitions, or invented words.

### 26.5 Diarization and speaker labels

Speaker labels add structure but also error:

- speaker embeddings can confuse similar voices;
- overlap is hard;
- short turns are unstable;
- diarization boundaries may not match linguistic turns.

Use diarization only when speaker identity is part of the corpus policy. Keep it as metadata, not text inserted into every sentence unless that format is explicitly desired.

### 26.6 ASR acceptance

Require a sampled audio-to-text audit set containing:

- Hindi and other intended Indic languages;
- Hinglish and Romanized speech;
- names, numerals, URLs, and code terms;
- dialect and accent variation;
- noisy and overlapping speech;
- long and short turns;
- low-confidence and randomly sampled high-confidence spans.

Measure WER/CER, boundary duplication/omission, timestamp overlap, language-switch retention, and named-entity/numeral error.

**Sources:** [Conformer paper](https://ar5iv.labs.arxiv.org/html/2005.08100), [Whisper paper](https://ar5iv.labs.arxiv.org/html/2212.04356), [IndicLLMSuite speech extraction](https://arxiv.org/html/2403.06350#S4.SS3). `[PRIMARY]`

---

## 27. Audio/video route decision matrix

| Input | First route | Escalate to | Avoid |
|---|---|---|---|
| Clean SRT/VTT with reliable cues | Cue parser + timestamped blocks | Audio spot-check if transcript is suspect | Dropping timing before QA |
| Hindi lecture audio | VAD + Hindi-capable ASR + timestamps | Second ASR, diarization, manual review | Unbounded chunks or text-only acceptance |
| Multilingual/code-mixed speech | Multilingual ASR with language evidence | Hindi model comparison and sampled adjudication | Treating one language label as truth |
| Noisy meeting | Conservative VAD + short overlapping chunks | Diarization/second model/manual review | Aggressive VAD deletion without recall check |
| Music-heavy video | VAD + speech/non-speech sidecar | Segment-level review | Transcribing all audio blindly |
| Existing flat transcript | Residual format audit | Reacquire audio/subtitle source | Claiming ASR quality without source alignment |

---

# Part IV — Hindi / Indic safety constraints

## 28. Preserve orthographic code points

Unicode’s Indic guidance makes a key point: zero-width joiner and non-joiner can control Indic rendering and represent explicit or soft halant behavior. They are not generic garbage. `[PRIMARY]`

Preserve when valid:

- virama/halant sequences;
- nukta and combining marks;
- ZWJ (`U+200D`) and ZWNJ (`U+200C`);
- Indic digits and danda punctuation;
- valid script-specific signs and combining sequences.

Remove or flag only characters demonstrated to be artifacts for this source, such as:

- replacement characters;
- malformed byte fragments;
- BOMs in the wrong position;
- bidi overrides when they are injected controls rather than intended text;
- accidental zero-width spaces;
- other controls proven by source comparison to be corruption.

Run before/after code-point counts and keep a policy-level reason for every class removed. Do not use `category == Cf` as a universal deletion rule.

**Source:** [Unicode Indic Scripts FAQ](https://www.unicode.org/faq/indic.html). `[PRIMARY]`

## 29. Script is not language

A Hindi document may contain:

- Devanagari Hindi;
- English names and technical terms;
- URLs and email addresses;
- code and numerals;
- Romanized Hindi;
- Hinglish or other natural code-mixing;
- quotations in another language.

A Latin-script span is not automatically noise. A Devanagari page can be mislabeled or contain another Indic language. Extraction should retain the segment and provenance; language ID/validation can attach document- or span-level labels later.

## 30. Do not import English heuristics blindly

Dangerous hard gates include:

- English stopword density;
- English punctuation or whitespace requirements;
- minimum word length based on English tokenization;
- deleting short blocks, bullets, headings, or one-line dialogue;
- treating low ASCII ratio as corruption;
- deleting every zero-width/format character;
- rejecting text because it lacks a Latin full stop;
- treating Romanized Indic speech as non-language.

Every extraction rule should be tested on:

```text
Devanagari prose
Romanized Hindi
Hinglish/code-mix
Sanskrit-like text
short headlines and dialogue
lists/tables/forms
code and formulas
OCR text with combining marks
```

## 31. Extractor-specific Indic risks

| Route | Indic risk | Required check |
|---|---|---|
| Density heuristic | wrong word/stopword assumptions | script-stratified precision/recall; no English hard gate |
| Readability | article/news bias | forum, short, list, and technical samples |
| Generic web parser | dynamic/noisy markup | rendered fallback and source-span audit |
| Model HTML labeller | transfer and simplification error | Hindi-literate block gold set, no-generation source mapping |
| Native PDF | broken font mapping/order | code-point and geometry comparison |
| OCR | matra/conjunct/nukta substitution | CER by character class, page crops, confidence |
| Subtitle parser | translated/abridged cues | audio alignment sample and cue metadata |
| ASR | dialect/code-mix/name errors | WER/CER strata, entity/numeral audit, timestamps |
| Flat-text audit | irreversible lost layout | label as residual extraction only |

---

# Part V — Recommended cascade for this project

## 32. Intake and routing

```text
raw bytes / flat record
  ├─ preserve URI, MIME, timestamp, hash
  ├─ detect actual artifact type
  ├─ detect whether original structure exists
  └─ route with explicit reason
```

Rules:

1. Never send a PDF through an HTML extractor.
2. Never send flattened text through a DOM extractor.
3. Do not trust a folder or dataset label as artifact truth.
4. Produce block-level output and an extraction trace.
5. Keep low-confidence content quarantined rather than silently deleting it.

## 33. Source-level cascade when artifacts are available

### 33.1 Web

```text
raw HTML/WARC
  ├─ known domain/API/schema rule → selected typed subtree
  └─ generic static parser
       ├─ Trafilatura or Resiliparse baseline
       ├─ block + structure retention
       ├─ yield/link/boilerplate/structure checks
       ├─ low confidence/disagreement → JusText/Readability/Dragnet comparison
       ├─ dynamic page → controlled render, then repeat
       └─ difficult structure → constrained semantic block model
```

### 33.2 PDF/image

```text
PDF/image
  ├─ usable text layer → native blocks/words + coordinates
  ├─ poor text layer → render + OCR + layout reconstruction
  ├─ complex layout → document/layout model
  └─ disagreement/low confidence → second engine or manual review
```

### 33.3 Speech

```text
SRT/VTT → cue parser → timestamped utterances → boundary-aware document
Audio/video → decode → VAD → Hindi/multilingual ASR
                         ├─ timestamps/confidence/diarization
                         └─ optional punctuation restoration sidecar
```

## 34. What can honestly be done on the current local cache

The local cache contains already-flattened JSONL records with `doc_id`, `type`, `text`, and estimated-token fields. It does not contain original HTML/WARC, source URLs, PDF bytes, page images, audio, DOM/layout trees, or source spans. `[LOCAL]`

Therefore the honest local route is:

```text
flat JSONL
  ├─ detect literal residual tags/markers
  ├─ detect whole-line navigation/banner artifacts only when explicit
  ├─ preserve Indic code points
  ├─ record exact replacement counts and before/after hashes
  └─ label output residual-extraction audit
```

Do **not** claim that Trafilatura, Readability, JusText, OCR, browser rendering, or ASR was run on this flat cache. Do not claim that original article boundaries were recovered. Reacquire paired source artifacts before making a source-level extractor comparison.

### 34.1 Local residual run facts

The local extraction-only run is `residual-extract-hindi-v1`:

- source: `data/sangraha_hi_verified_30m/raw.jsonl`;
- output: `data/sangraha_hi_verified_30m/extraction_only/01_residual_extract.jsonl`;
- summary: `data/sangraha_hi_verified_30m/extraction_only/01_residual_extract_summary.json`;
- characters: `114,296,090 → 114,295,217`;
- estimated tokens: `30,000,019 → 29,999,882`;
- documents changed: `26`;
- actions: `12` block tags, `34` inline tags, `1` standalone boilerplate line, `0` script/style blocks;
- observed ZWJ/ZWNJ: `0` before and `0` after.

These are measured residual-audit facts, not proof that the original web/PDF/speech extraction was correct. `[LOCAL]`

### 34.2 Source-level experiment needed next

```text
raw_source/
  web/*.warc-or-html
  pdf/*.pdf
  image/*.png-or-jpg
  speech/*.wav-or-mp4
  subtitles/*.srt-or-vtt
paired_manifest.jsonl
  source_hash, uri, type, license, extractor, version,
  output_hash, warnings, confidence
```

Then compare at least:

- two web baselines on the same HTML/WARC sample;
- native PDF extraction against OCR/layout fallback;
- one subtitle route and one ASR route against the same media sample.

Only those paired artifacts support extraction-quality claims.

---

# Part VI — Validation: prove extraction worked

## 35. Build a Hindi-literate paired gold set

A technique is selected by measured error, not reputation or an English benchmark.

Each gold item needs:

```text
raw source artifact
human-labelled document boundaries
human reading order
typed blocks
reference transcription/text
script/language/code-mix labels
expected exclusions and reasons
adjudication notes
```

Stratify by:

- web, digital PDF, scanned PDF/image, SRT/VTT, speech;
- domain, source, template, and document genre;
- length decile;
- Devanagari-only, Romanized, and code-mixed text;
- lists, tables, code, formulas, captions, and forums;
- multi-column, sidebar, footnote, header/footer layouts;
- OCR quality, scan resolution, noise, dialect, and speaker overlap.

The small local corpus should census the speech slice and deliberately allocate web/PDF strata. A broad binary proportion estimate of roughly 385 independent items gives a worst-case 95% margin near ±5 percentage points before stratification/design effects; the correct sample size depends on the error being estimated.

Annotate both automatic flags and an unflagged control sample. Otherwise a filter can look precise while deleting an entire unflagged class.

## 36. Metrics by modality

| Modality | Primary metrics | Diagnostic/safety metrics |
|---|---|---|
| Web main-content | block precision/recall/F1; character/token precision/recall; source-span coverage | boilerplate leakage, omission, empty-output rate, retained fraction, extractor disagreement, rendered fallback rate |
| HTML structure | heading/list/table/code/formula typed retention | table cell/row alignment, code indentation, formula fidelity, DOM-to-output mapping |
| Digital PDF | block precision/recall; reading-order accuracy | header/footer leakage, page coverage, table structure, native-vs-layout disagreement |
| OCR PDF/image | CER/WER; block detection; reading order | mean/quantile confidence, script confidence, box overlap, matra/conjunct/nukta error rate, low-quality-page rate |
| SRT/ASR | WER/CER; timestamp overlap; speaker-turn accuracy | cue duplication/omission, language-switch retention, punctuation quality, low-confidence duration |
| Indic safety | code-point survival; script-span preservation | ZWJ/ZWNJ/virama/nukta loss, replacement count, unexpected normalization, Romanized-span retention |
| Operations | documents/sec; CPU/GPU time; memory; fallback rate | version/hash reproducibility, source-hash coverage, failure-reason distribution |

Useful aggregate rates:

```text
boilerplate_leakage = boilerplate_tokens_retained / output_tokens
main_content_omission = gold_main_tokens_not_retained / gold_main_tokens
extraction_yield = output_tokens / source_content_tokens
fallback_rate = escalated_documents / routed_documents
source_span_coverage = output_spans_mapped_to_source / output_spans
```

Do not interpret high yield as quality. A whole-page tag strip can have high yield because it retained the menu, legal text, and ads.

## 37. Route-specific acceptance checks

### 37.1 Web hard checks

- Every emitted block maps to the source DOM.
- Output is not empty when source has content.
- Boilerplate leakage and main-content omission are measured separately.
- Tables/code/formulas/headings have typed retention counts.
- Static and rendered outputs retain separate hashes.
- Extractor disagreement triggers review, not automatic deletion.

### 37.2 PDF/OCR hard checks

- Source page and render settings are present.
- Native/OCR disagreement is recorded.
- Reading order is reviewed on multi-column pages.
- Boxes/confidence survive serialization.
- Indic code-point classes are checked, not only total characters.
- Low-confidence pages are quarantined.

### 37.3 Speech hard checks

- Original media hash and time ranges are present.
- Every transcript segment has an audio interval.
- VAD boundary decisions are reproducible.
- WER/CER includes code-mix, names, numerals, dialect, and noise strata.
- Punctuation and diarization are sidecars, not hidden rewrites.

## 38. Escalation rules

Use thresholds only as review triggers; calibrate them on the gold set.

- **Hard fail:** output text not present in source spans for a selection-only route; invalid structured output; missing source hash; empty output for expected content.
- **Web review:** extractor disagreement, high link/boilerplate ratio, unusually low/high retained fraction, lost structured elements, script-heavy markup.
- **PDF review:** low text-layer coherence, low OCR/script confidence, high box overlap, sparse/rotated/multi-column page, native/OCR disagreement.
- **Speech review:** low ASR confidence, long silence/overlap, abrupt chunk discontinuity, high language-switch rate, name/numeral uncertainty.
- **Indic block:** unexpected loss of ZWJ/ZWNJ/virama/nukta, replacement characters, or any rule that drops a valid Romanized/code-mixed span.

A low-confidence document should be **quarantined or escalated**, not silently deleted. Record route and reason so false negatives can be recovered.

---

# Part VII — Compact implementation recommendation

For a practical Hindi/Indic extraction pipeline:

1. **Preserve first:** raw bytes, URL/identifier, MIME, timestamp, source hash, page/audio timing, and layout evidence.
2. **Route by artifact:** HTML/WARC, PDF, image, subtitle, audio/video, or flat text.
3. **Use deterministic baselines:** domain selectors for stable sites; Trafilatura/Resiliparse for static web; PyMuPDF for coherent PDF text layers; SRT parsing for subtitles.
4. **Preserve structure:** block lists, tables, code, formulas, headings, page coordinates, timestamps, boxes, and confidence.
5. **Escalate uncertainty:** browser rendering for dynamic HTML; Readability/JusText/Dragnet comparison; semantic block labelling for difficult HTML; OCR/layout models for scans; second ASR/manual review for speech.
6. **Protect Indic text:** preserve ZWJ/ZWNJ, virama, nukta, matras, danda, valid code-mix, and Romanized Indic text.
7. **Validate with Hindi examples:** measure omission and leakage separately; include structured and low-resource cases; report confidence and fallback rates.
8. **For the current flattened cache:** call the work a residual extraction audit only. Reacquire source artifacts before claiming full extraction quality.

The central engineering rule is simple:

> **If the extractor cannot show which source region produced a training span, the pipeline cannot distinguish faithful extraction from plausible-looking contamination.**

---

## Sources

### Local lecture, widgets, and run evidence

- [Session 3 lecture](../axiom-fetched/session3_lecture.md) — extraction definition and naive HTML-strip failure.
- [Session 3 extraction widget](../axiom-fetched/widgets/widget_17_extraction_quality.html) — illustrative two-way HTML example and source code.
- [Session 4 lecture](../axiom-fetched/session4_lecture.md) — Extract as stage 1 and the eight-stage boundary.
- [Session 4 widget inventory](../axiom-fetched/session4_widget_inventory.json) — verified iframe scope and canonical strategy notes.
- [Dataset manifest](../data/sangraha_hi_verified_30m/manifest.json) — source, revision, split, token estimate, and provenance.
- [Residual extraction summary](../data/sangraha_hi_verified_30m/extraction_only/01_residual_extract_summary.json) — measured local run and limitations.

### Primary papers

- [IndicLLMSuite: A Blueprint for Creating Pre-training and Fine-Tuning Datasets for Indian Languages](https://arxiv.org/html/2403.06350) — Setu routing for web/PDF/speech, OCR geometry checks, VAD + Hindi Conformer, subtitle processing, and Indic corpus concerns.
- [AICC / MinerU-HTML: Parse HTML Finer, Make Models Better](https://arxiv.org/html/2511.16397) — model-based HTML block labelling, structure preservation, template-aware scaling, and WebMainBench results.
- [CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data](https://ar5iv.labs.arxiv.org/html/1911.00359) — document-level web processing and language-aware corpus processing.
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://ar5iv.labs.arxiv.org/html/2005.08100) — self-attention plus convolution ASR architecture.
- [Robust Speech Recognition via Large-Scale Weak Supervision](https://ar5iv.labs.arxiv.org/html/2212.04356) — multilingual encoder-decoder ASR and log-Mel/token processing.

### Official tools and implementation references

- [Trafilatura documentation](https://trafilatura.readthedocs.io/en/latest/) — web discovery, main text, metadata, structure, and output formats.
- [Trafilatura main extractor](https://raw.githubusercontent.com/adbar/trafilatura/master/trafilatura/main_extractor.py) — tree processing, structural handlers, text tests, and output construction.
- [Resiliparse extraction documentation](https://resiliparse.chatnoir.eu/en/stable/man/extract.html) — DOM and main-content extraction.
- [JusText core implementation](https://raw.githubusercontent.com/miso-belica/justext/master/justext/core.py) and [algorithm description](https://raw.githubusercontent.com/miso-belica/jusText/master/doc/algorithm.rst) — paragraph formation, link/stopword density, and neighboring-block revision.
- [Mozilla Readability](https://github.com/mozilla/readability) and [source](https://raw.githubusercontent.com/mozilla/readability/main/Readability.js) — semantic DOM scoring and article cleanup.
- [Inscriptis](https://github.com/weblyzard/inscriptis) and [HTML engine](https://raw.githubusercontent.com/weblyzard/inscriptis/master/src/inscriptis/html_engine.py) — layout-aware HTML conversion and tag handlers.
- [Dragnet](https://github.com/dragnet-org/dragnet) and [extractor source](https://raw.githubusercontent.com/dragnet-org/dragnet/master/dragnet/extractor.py) — supervised feature-based block extraction.
- [Article extraction benchmark](https://github.com/scrapinghub/article-extraction-benchmark) — reproducible comparison harness; article/news-focused, not Hindi evidence.
- [Playwright pages](https://playwright.dev/python/docs/pages) — controlled browser navigation and rendered-page access.
- [PyMuPDF text recipes](https://pymupdf.readthedocs.io/en/latest/recipes-text.html) and [text extraction appendix](https://pymupdf.readthedocs.io/en/latest/app1.html) — native PDF text, blocks, words, reading order, and tables.
- [Tesseract user manual](https://tesseract-ocr.github.io/tessdoc/) — LSTM OCR engine and language/script model data.
- [PaddleOCR PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PP-StructureV3.md) — modular layout, OCR, table, formula, and document conversion pipeline.
- [py-webrtcvad](https://github.com/wiseman/py-webrtcvad) and [VAD core](https://raw.githubusercontent.com/wiseman/py-webrtcvad/master/cbits/webrtc/common_audio/vad/vad_core.c) — frame contract and speech/non-speech decision implementation.
- [Unicode Indic Scripts FAQ](https://www.unicode.org/faq/indic.html) — ZWJ/ZWNJ, virama, and Indic Unicode behavior.
- [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library) — script-aware Indic normalization, tokenization, and validation tooling.
