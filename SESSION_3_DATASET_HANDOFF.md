# Session 3 Dataset Handoff

## Scope

- **Course:** ERA V5
- **Session:** Session 3: Data Collection and Sourcing
- **Axiom session ID:** `cmr4ji9vx00xy09pl509rpl86`
- **Axiom lesson ID:** `cmr4l6d9e01ed09pl5rzee6zk`
- **Status at fetch time:** Available
- **Session opened:** 2026-07-11 03:53 UTC
- **Source:** Live Axiom Learning OS fetch plus official dataset cards/repositories
- **Download status:** No dataset files were downloaded. Only the lesson text and public metadata were reviewed.

## Training-stage map from the lecture

Session 3 defines the data unit and purpose at each stage of an LLM lifecycle:

| Training stage | Data unit | What the model learns |
|---|---|---|
| Pretraining | A document | Language, knowledge, code, and statistical structure through next-token prediction |
| Mid-training / annealing | A smaller reserve of high-quality documents | Concentrated capability in code, mathematics, reasoning, or target languages during a late training phase |
| Supervised fine-tuning (SFT) | An instruction-response pair | How to answer rather than only continue text |
| Preference optimization | A prompt with chosen and rejected responses | Which valid-looking response is preferred |
| Reinforcement learning (RL) | A prompt, generated trajectory, and reward/verifier | Measurable task outcomes and behavior |
| Evaluation / held-out data | Benchmark or proxy material kept outside training | Honest measurement; once test data trains the model, its score is contaminated |

Most named corpora in Session 3 are **pretraining** resources. The lecture also discusses smaller annealing reserves, synthetic augmentation, quality filters, and held-out evaluation data as parts of the broader data system.

## Named datasets and corpora

### 1. ERA V4 / LightningLM V0.1 corpus

**Classification:** Course-internal reference corpus; not a public download in the lesson.

**Training role:** Primarily pretraining, with separate always-on and evaluation-support tiers.

**Information in the lecture:**

- Approximately **1,118 billion tokens** across **33,353 shards**.
- **Main pretraining pools D1-D4:** move from high-quality web data toward broader web data, code, science, and mathematics.
- **Always-on data:** supplied **8% of every batch**, independently of the selector. It protected Indic data and benchmark training splits that an English-heavy selector might under-sample.
- **Golden proxy data:** benchmark test material never entered training. It supplied selection direction only.
- The lecture presents the corpus as an engineered system rather than a directory of downloaded text: collection, extraction, cleaning, filtering, deduplication, balancing, provenance, and evaluation all matter.

**Why it appears in Session 3:** It is the concrete reference for showing what a production-style training corpus looks like and why corpus partitions need different policies.

### 2. FineWeb / FineWeb-Edu

**Classification:** Large-scale filtered web-text corpus.

**Training role:** Pretraining. FineWeb-Edu is especially relevant to educational-quality filtering and web-text curation.

**Information in the lecture:**

- Session 3 uses FineWeb-Edu and DataComp as references in the quality-over-quantity discussion.
- The lecture says a smaller, well-filtered corpus can outperform a much larger raw corpus at the same training budget.
- It uses FineWeb's **per-snapshot deduplication** as an example of a gentler policy than global deduplication. The point is that removing more data is not automatically better if higher-quality older material is discarded.

**Official dataset-card metadata reviewed:**

- Main FineWeb-Edu release: approximately **1.3T tokens** of educational web pages filtered from FineWeb.
- A separate `FineWeb-Edu-score-2` release is described at approximately 5.4T tokens.
- English text-generation data.
- License: **ODC-By**.
- Parquet/text dataset with fields including:
  - `text`
  - `id`
  - `dump`
  - `url`
  - `date`
  - `file_path`
  - `language`
  - `language_score`
  - `token_count`
  - `score`
  - `int_score`
- The card exposes sample and Common Crawl snapshot configurations, including `sample-10BT`, `sample-100BT`, and `sample-350BT`.

**Official source:** [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

### 3. CulturaX

**Classification:** Large multilingual cleaned and deduplicated text corpus.

**Training role:** Multilingual pretraining.

**Information in the lecture:**

- Used to compare the amount of English data with verified human-origin Indic data.
- The lecture’s comparison shows approximately **2,847 billion English tokens** for CulturaX.
- It is used to illustrate that headline multilingual token counts can hide how little verified human-origin data exists for some languages.
- It is relevant to language identification, language-aware quality filtering, multilingual balancing, and provenance.

**Official dataset-card metadata reviewed:**

- Approximately **6.3T tokens** across **167 languages**.
- Cleaning and deduplication are core parts of the published dataset description.
- Access: **Gated**.
- Formats/tasks: Parquet text data for text generation and fill-mask use.
- The language metadata includes English and many Indic languages, including Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu, and Urdu.

**Official source:** [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)

### 4. Sangraha

**Classification:** Large Indic-language pretraining corpus.

**Training role:** Indic-language pretraining. Its verified, unverified, and synthetic partitions also make it useful for studying data provenance and synthetic-data separation.

**Information in the lecture:**

- The lecture uses Sangraha to show the difference between headline multilingual totals and verified human-origin data.
- In the verified-only view, Telugu falls from **16.3B tokens to 3.7B**, while Odia falls from **12.5B to 1.2B**.
- It illustrates why English-trained quality filters can delete good Indic text: local writing styles, code-mixed text, uncommon domains, and Brahmic-script structure may be penalized.
- This is the basis for the V4 always-on channel that protects low-resource Indic data from an English-heavy selector.

**Official dataset-card metadata reviewed:**

- Approximately **251B tokens** summed over **22 languages**.
- Sources include curated sources, existing multilingual corpora, and large-scale translations.
- License: **CC-BY-4.0**.
- Text-generation dataset with Parquet files.
- Configurations:
  - `verified`: `doc_id`, `type`, and `text`.
  - `unverified`: `doc_id` and `text`.
  - `synthetic`: `doc_id` and `text`, with script-specific language splits.
- Verified splits include languages such as Assamese, Bengali, Gujarati, Hindi, Kannada, Kashmiri, Maithili, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu, and Urdu, along with additional Indic splits.

**Official source:** [ai4bharat/sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)

### 5. DeepSeekMath corpus

**Classification:** Domain-specialized mathematical web corpus and corpus-building recipe.

**Training role:** Math-focused pretraining. The lecture also connects the approach to seed-based domain mining when no ready-made high-quality corpus exists.

**Information in the lecture and official repository:**

- Session 3 describes an iterative mining loop:
  1. Start with a trusted mathematics seed.
  2. Train a classifier/retrieval model from the seed.
  3. Find more mathematics on the web.
  4. Identify promising domains and annotate relevant URLs.
  5. Add linked pages and repeat.
- The official repository describes **OpenWebMath** as the initial seed corpus.
- It retrieves math-related pages from a deduplicated Common Crawl database.
- After four iterations, the corpus contains approximately **35.5M mathematical web pages** and **120B tokens**.
- The broader DeepSeekMath pretraining setup also combines math-related tokens with natural language and code; the repository describes 500B tokens for the overall continued-pretraining mixture, distinct from the 120B math corpus figure.

**Official source:** [deepseek-ai/DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math)

### 6. Nemotron-CC

**Classification:** Large pretraining corpus and synthetic-data augmentation reference.

**Training role:** Pretraining; the Session 3 discussion specifically focuses on real-to-synthetic augmentation rather than a separate SFT dataset.

**Information in the lecture:**

- The “Nemotron-CC pattern” is used to show that moderate synthetic augmentation can improve quality, while excessive synthetic data can cause model-collapse risk.
- Synthetic data can:
  - Rewrite weak documents
  - Generate explanations
  - Translate content
  - Produce exercises
  - Add reasoning traces
  - Create verifiable code and mathematics
- The value of synthetic data depends on the teacher model, prompt design, verification, diversity, and quality of the real seed corpus.

**Current official dataset-card metadata reviewed:**

- Described as preserving high-value math and code while adding diverse multilingual Q&A.
- Current public card: `nvidia/Nemotron-CC-v2`.
- Access: **Gated**.
- License: **Other**.
- Parquet/text dataset for text generation.

**Official source:** [nvidia/Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2)

## Other named references in the lesson

These are important to the lesson but are not all downloadable LLM datasets:

- **OPUS:** Scores candidate pretraining batches; the lecture says it retained roughly 40% of candidates. It is a scoring/selection reference, not the corpus itself. [Paper](https://arxiv.org/pdf/2602.05400)
- **DataComp:** Used as a published quality/curation comparison alongside FineWeb-Edu. It is a data-curation benchmark reference, not a text-only LLM pretraining corpus in this lecture. [Official repository](https://github.com/mlfoundations/datacomp)
- **Llama 4 and Qwen3:** Used for frontier-scale comparisons. The lecture cites more than 30T tokens for Llama 4 and 36T tokens across 119 languages for Qwen3. They are model-training scale references, not dataset choices.
- **OLMo 2:** Used as an annealing example where a small mathematics-and-code reserve is concentrated late in pretraining. The lesson does not identify a single downloadable OLMo 2 dataset.
- **OpenWebMath:** The trusted math seed in the DeepSeekMath collection recipe.
- **Common Crawl:** The raw web source from which the DeepSeekMath mining process retrieves math pages; it is an upstream web archive rather than a curated Session 3 assignment dataset.

## How these resources map to the later Session 4 assignment

The Session 4 assignment asks for a 10–100M-sized dataset and a widget explaining how the Session 4 cleaning strategies were applied. The Session 3 resources provide different cleaning angles:

| If the project focuses on... | Most relevant Session 3 resource |
|---|---|
| HTML/web extraction and educational quality | FineWeb-Edu |
| Indic language protection and provenance | Sangraha |
| Multilingual balancing and language identification | CulturaX |
| Math-domain mining and domain quality | DeepSeekMath |
| Real/synthetic mixture and verification | Nemotron-CC |
| End-to-end corpus partitioning and evaluation integrity | ERA V4 / LightningLM V0.1 reference |

The full corpora are far larger than the assignment target. This handoff intentionally does not select or download a sample.

## Axiom lesson transcript

[Session 3 transcript](https://docs.google.com/document/d/1Sez_JUnz5NaIOr7DEPxID3Y5gLtZl5NtSrXcHoWJyr4/edit?usp=sharing)

## Sources and extraction method

This handoff is intentionally self-contained. It does not depend on the prior chat, a temporary cache, a local dataset file, or a running MCP session. The identifiers, observed values, classifications, links, and collection method needed to understand the handoff are recorded here.

### Primary lecture source

- **Axiom Learning OS:** the live `sessions --course-id cmq97i5kn032208o8xu5dab4q` entity was fetched for the active **ERA V5** course.
- The Session 3 record supplied:
  - Session metadata: title, status, order, and opening time.
  - The lesson markdown used for the training-stage descriptions, dataset/corpus names, token counts, tier descriptions, and lecture claims.
  - The lesson’s embedded transcript link.
- Relevant lecture sections:
  - **Section 2:** ERA V4 / LightningLM V0.1 corpus and its tiers.
  - **Section 3:** what a dataset is for at each LLM training stage.
  - **Section 5:** capability-specific data, including code, mathematics, and multilingual data.
  - **Section 6:** quality filtering, FineWeb-Edu, and DataComp references.
  - **Section 7:** deduplication and synthetic data, including the Nemotron-CC pattern.
  - **Section 8:** Sangraha, CulturaX, verified human-origin data, and Indic filter bias.
  - **Section 10:** extraction and the mid-training/annealing reserve.
  - **Section 12:** held-out evaluation and contamination.
- The lecture source is preserved here: [Session 3 transcript](https://docs.google.com/document/d/1Sez_JUnz5NaIOr7DEPxID3Y5gLtZl5NtSrXcHoWJyr4/edit?usp=sharing)

### Official dataset and repository sources

The following canonical pages were opened directly after the lecture identified the resource names. Dataset sizes, licenses, access status, configurations, fields, and corpus-building details in this handoff come from these pages rather than from search-result summaries:

- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [CulturaX dataset card](https://huggingface.co/datasets/uonlp/CulturaX)
- [Sangraha dataset card](https://huggingface.co/datasets/ai4bharat/sangraha)
- [Nemotron-CC-v2 dataset card](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2)
- [DeepSeekMath official repository](https://github.com/deepseek-ai/DeepSeek-Math)
- [OPUS paper linked by the lecture](https://arxiv.org/pdf/2602.05400)
- [DataComp official repository](https://github.com/mlfoundations/datacomp)

### How the information was derived

1. The live Axiom Session 3 lesson was fetched first, so the lecture’s own terminology and role assignments were preserved.
2. Named resources were separated into:
   - Actual datasets/corpora.
   - Course-internal corpus partitions.
   - Data-curation or scoring references.
   - Model-scale or training-result references.
3. Training-stage labels were taken from the lecture’s lifecycle section and then applied only where the lecture or source described a role.
4. Public metadata was checked against the official dataset card or repository:
   - Dataset-card fields and configurations were transcribed from the Hugging Face cards.
   - DeepSeekMath’s seed, Common Crawl mining loop, page count, and token count were taken from its official README.
   - OPUS and DataComp were kept separate from dataset entries because the lecture uses them as selection/curation references.
5. No dataset records, shards, Parquet files, or model weights were downloaded.

### Snapshot and maintenance note

- **Handoff prepared:** 2026-08-01.
- The values are an observed snapshot. Dataset cards, access gates, sizes, and repository documentation can change independently of the Axiom lesson.
- Every external claim has a source link above or in the corresponding dataset section.
- The handoff records enough context to be read or passed to another agent without access to this conversation.
