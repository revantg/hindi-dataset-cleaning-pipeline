import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def assignment(mo):
    mo.md(r"""
    # Session 4 — Dataset Cleaning and Deduplication

    ## Case study goal

    Build a reproducible 10–100M-token cleaning case study. Each cleaning stage must retain its own decision record, measurements, and hand-off artifact.

    ## Eight required stages

    1. Extract
    2. Normalize
    3. Validate language
    4. Filter quality
    5. Deduplicate
    6. Scrub PII
    7. Decontaminate
    8. Record manifest and provenance

    ## This notebook

    This chapter isolates **Stage 1: Extraction**. It first preserves the Hindi Sangraha source-path decision, then executes comparable extractors on a separate raw-HTML evaluation corpus. No extraction result is presented as Hindi-quality evidence without Hindi-literate source labels.
    """)
    return


@app.cell(hide_code=True)
def dataset_selection(mo):
    mo.md(r"""
    # Dataset Selection

    | Candidate | Technical assessment |
    |---|---|
    | Hindi Wikipedia | Within the token target, but insufficient residual cleaning signal for this study. |
    | OSCAR Hindi | Suitable raw-web alternative; acquisition and manual review requirements are higher. |
    | IndicCorp Hindi | Predominantly short, previously cleaned lines; limited evidence for document-level extraction, deduplication, or PII analysis. |
    | xP3 / IndicParaphrase | Instruction/paraphrase records rather than a document-pretraining corpus. |
    | **Sangraha verified Hindi** | **Selected.** Open, document-level Hindi corpus with provenance metadata and measurable residual artifacts. |

    **Dataset identifier:** `ai4bharat/sangraha` · **configuration:** `verified` · **split:** `hin` · **target:** approximately **30 million estimated tokens**.

    **Selection rationale:** The corpus provides document-level Hindi text, source-type metadata, and sufficient residual formatting variation to support a conservative extraction evaluation. The lecture demonstration dataset is not used.
    """)
    return


@app.cell(hide_code=True)
def preview_controls(mo):
    fetch_preview = mo.ui.run_button(
        label="Retrieve five Sangraha reference records"
    )
    mo.vstack(
        [
            mo.md(
                "This source inspection requests a bounded hosted-row preview only. "
                "It does not materialize or transform the corpus."
            ),
            fetch_preview,
        ]
    )
    return (fetch_preview,)


@app.cell(hide_code=True)
def preview_fetch(fetch_preview):
    import json
    from urllib.parse import urlencode
    from urllib.request import urlopen

    preview_rows = []
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
            payload = json.load(response)
        for index, item in enumerate(payload["rows"][:5]):
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
    return preview_rows, preview_status


@app.cell(hide_code=True)
def preview_report(mo, preview_rows, preview_status):
    mo.vstack(
        [
            mo.md(preview_status),
            mo.ui.table(preview_rows),
        ]
    )
    return


@app.cell(hide_code=True)
def download_setup():
    import json as _download_json
    import math as _download_math
    from pathlib import Path as _DownloadPath

    DATASET_ID = "ai4bharat/sangraha"
    DATASET_CONFIG = "verified"
    DATASET_SPLIT = "hin"
    DATASET_REVISION = "8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8"
    DATASET_ROOT = (
        _DownloadPath(__file__).resolve().parent
        / "data"
        / "sangraha_hi_verified_30m"
    )
    RAW_PATH = DATASET_ROOT / "raw.jsonl"
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    def estimate_tokens(value):
        units = len(value.split())
        return int(_download_math.ceil(units * 1.35)) if units else 0

    def stats_for_path(path):
        if not path.exists():
            return None
        documents = 0
        characters = 0
        estimated_tokens = 0
        with path.open("r", encoding="utf-8") as _download_handle:
            for _download_line in _download_handle:
                if not _download_line.strip():
                    continue
                _download_record = _download_json.loads(_download_line)
                _download_value = _download_record.get("text", "")
                documents += 1
                characters += len(_download_value)
                estimated_tokens += _download_record.get(
                    "estimated_tokens", estimate_tokens(_download_value)
                )
        return {
            "documents": documents,
            "characters": characters,
            "estimated_tokens": estimated_tokens,
        }

    return (
        DATASET_CONFIG,
        DATASET_ID,
        DATASET_REVISION,
        DATASET_SPLIT,
        RAW_PATH,
        estimate_tokens,
        stats_for_path,
    )


@app.cell(hide_code=True)
def download_controls(mo):
    sample_target_millions = mo.ui.slider(
        start=10,
        stop=30,
        step=5,
        value=30,
        label="Target sample size (millions of estimated tokens)",
        show_value=True,
    )
    download_button = mo.ui.run_button(
        label="Materialize canonical Sangraha JSONL cache"
    )
    mo.vstack(
        [
            mo.md(
                "Select a target and materialize a deterministic sequential subset of "
                "`verified/hin` into the canonical JSONL cache. This operation performs "
                "no cleaning transformation."
            ),
            mo.hstack([sample_target_millions, download_button]),
        ]
    )
    return download_button, sample_target_millions


@app.cell(hide_code=True)
def download_sample(
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    RAW_PATH,
    download_button,
    estimate_tokens,
    mo,
    sample_target_millions,
    stats_for_path,
):
    from datasets import load_dataset

    target_tokens = int(sample_target_millions.value) * 1_000_000
    cached_stats = stats_for_path(RAW_PATH)
    if not download_button.value:
        raw_stats = cached_stats
        download_status = (
            f"Cached download found: {raw_stats['documents']:,} documents."
            if raw_stats
            else "Click the download button to start the stream."
        )
    elif cached_stats and cached_stats["estimated_tokens"] >= target_tokens:
        raw_stats = cached_stats
        download_status = (
            "Using the existing JSONL cache; no network download was needed."
        )
    else:
        estimated_total = 0
        with RAW_PATH.open("w", encoding="utf-8") as _download_handle:
            _download_dataset = load_dataset(
                DATASET_ID,
                DATASET_CONFIG,
                split=DATASET_SPLIT,
                streaming=True,
            )
            for _download_source_row in _download_dataset:
                _download_value = _download_source_row.get("text") or ""
                _download_doc_id = str(
                    _download_source_row.get("doc_id") or ""
                )
                if not _download_doc_id or not _download_value:
                    continue
                _download_estimated = estimate_tokens(_download_value)
                _download_handle.write(
                    _download_json.dumps(
                        {
                            "doc_id": _download_doc_id,
                            "type": _download_source_row.get(
                                "type", "unknown"
                            ),
                            "text": _download_value,
                            "estimated_tokens": _download_estimated,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                estimated_total += _download_estimated
                if estimated_total >= target_tokens:
                    break
        raw_stats = stats_for_path(RAW_PATH)
        download_status = (
            f"Downloaded {raw_stats['documents']:,} documents and "
            f"{raw_stats['estimated_tokens']:,} estimated tokens."
        )
    mo.vstack(
        [
            mo.md(download_status),
            mo.ui.table([raw_stats] if raw_stats else []),
            mo.md(
                f"Raw cache path: `{RAW_PATH}` · dataset revision: `{DATASET_REVISION}`"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def html_eval_scope(mo):
    mo.md(r"""
    # Stage 1 — HTML Extraction Technique Workbook

    ## Purpose

    The main Hindi case study uses `ai4bharat/sangraha`, whose release contains already-flattened text. A DOM extractor cannot reconstruct deleted links, element hierarchy, layout, or article boundaries from that text.

    This separate, revision-pinned English C4 subset retains raw `html`, `url`, and supplied `text`, so it is used only to **execute and compare extraction methods on real HTML**.

    ## Read this chapter in order

    1. Run the bounded 30-document evaluation below.
    2. For each technique: read what it does, how it works, where it fits, and its limits; then inspect and run the visible code cell.
    3. Compare all methods in the final unified metrics table.

    The supplied C4 text is an **agreement proxy**, not human-labelled ground truth. These results demonstrate executable behavior, not Hindi extraction quality.
    """)
    return


@app.cell(hide_code=True)
def html_eval_setup():
    import hashlib as _html_hashlib
    import importlib.util as _html_importlib
    import json as _html_json
    import math as _html_math
    import re as _html_re
    from collections import Counter as _HtmlCounter
    from html.parser import HTMLParser as _HTMLParser
    from pathlib import Path as _HtmlPath

    import extruct as _html_extruct
    import inscriptis as _html_inscriptis
    import justext as _html_justext
    import trafilatura as _html_trafilatura
    from readability import Document as _HtmlReadabilityDocument
    from resiliparse.extract.html2text import (
        extract_plain_text as _html_resiliparse_text,
    )

    HTML_EVAL_DATASET = "bs-modeling-metadata/c4-en-html-with-metadata"
    HTML_EVAL_REVISION = "d7fa66c5f9686a055f50dbdaeb0d8950209cf0f8"
    HTML_EVAL_SPLIT = "train"
    HTML_EVAL_VERSION = "html-extraction-family-evaluation-v2"
    HTML_EVAL_ROOT = (
        _HtmlPath(__file__).resolve().parent
        / "data"
        / "c4_en_html_evaluation"
    )
    HTML_EVAL_IMPL_DIR = HTML_EVAL_ROOT / "impl"
    HTML_EVAL_RAW_PATH = HTML_EVAL_ROOT / "raw_html.jsonl"
    HTML_EVAL_MANIFEST_PATH = HTML_EVAL_IMPL_DIR / "selection_manifest.json"
    HTML_EVAL_METRICS_PATH = HTML_EVAL_IMPL_DIR / "technique_metrics.json"
    HTML_EVAL_DOCUMENT_METRICS_PATH = (
        HTML_EVAL_IMPL_DIR / "document_metrics.jsonl"
    )
    _HTML_EVAL_MAX_CHARS = 500_000
    _HTML_EVAL_TOKEN = _html_re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")
    _HTML_EVAL_BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
    _HTML_EVAL_SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}
    _HTML_EVAL_LIBRARY_STATUS = {
        "Trafilatura": True,
        "jusText": True,
        "Inscriptis": True,
        "Resiliparse": True,
        "extruct": True,
        "readability-lxml": True,
    }

    def _html_normalize(value):
        return _html_re.sub(r"\s+", " ", value or " ").strip()

    def _html_tokens(value):
        return set(
            token.lower() for token in _HTML_EVAL_TOKEN.findall(value or "")
        )

    def _html_overlap(candidate, reference):
        candidate_tokens = _html_tokens(candidate)
        reference_tokens = _html_tokens(reference)
        if not candidate_tokens or not reference_tokens:
            return 0.0
        return len(candidate_tokens & reference_tokens) / len(reference_tokens)

    class _HtmlTextRenderer(_HTMLParser):
        def __init__(self, layout=False):
            super().__init__(convert_charrefs=True)
            self.parts = []
            self.layout = layout
            self.skip_depth = 0
            self.tag_count = 0

        def handle_starttag(self, tag, attrs):
            self.tag_count += 1
            if tag in _HTML_EVAL_SKIP_TAGS:
                self.skip_depth += 1
            if self.skip_depth == 0 and tag in _HTML_EVAL_BLOCK_TAGS:
                self.parts.append("\n" if self.layout else " ")

        def handle_endtag(self, tag):
            self.tag_count += 1
            if tag in _HTML_EVAL_SKIP_TAGS and self.skip_depth:
                self.skip_depth -= 1
            if self.skip_depth == 0 and tag in _HTML_EVAL_BLOCK_TAGS:
                self.parts.append("\n" if self.layout else " ")

        def handle_data(self, data):
            if self.skip_depth == 0:
                self.parts.append(data)

        def render(self):
            value = "".join(self.parts)
            return (
                _html_re.sub(r"\n{3,}", "\n\n", value).strip()
                if self.layout
                else _html_normalize(value)
            )

    class _HtmlBlockCollector(_HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.blocks = []
            self.current = None
            self.link_depth = 0
            self.skip_depth = 0

        def handle_starttag(self, tag, attrs):
            if tag in _HTML_EVAL_SKIP_TAGS:
                self.skip_depth += 1
            if self.skip_depth:
                return
            if tag == "a":
                self.link_depth += 1
            if tag in {
                "p",
                "li",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "td",
                "th",
                "pre",
            }:
                if self.current:
                    self._finish()
                self.current = {"tag": tag, "parts": [], "link_chars": 0}

        def handle_endtag(self, tag):
            if tag in _HTML_EVAL_SKIP_TAGS and self.skip_depth:
                self.skip_depth -= 1
                return
            if self.skip_depth:
                return
            if tag == "a" and self.link_depth:
                self.link_depth -= 1
            if self.current and tag == self.current["tag"]:
                self._finish()

        def handle_data(self, data):
            if self.skip_depth or not self.current:
                return
            self.current["parts"].append(data)
            if self.link_depth:
                self.current["link_chars"] += len(data)

        def close(self):
            super().close()
            if self.current:
                self._finish()

        def _finish(self):
            text = _html_normalize("".join(self.current["parts"]))
            if text:
                self.blocks.append(
                    {
                        "text": text,
                        "link_ratio": self.current["link_chars"]
                        / max(1, len(text)),
                    }
                )
            self.current = None

    class _HtmlSemanticCollector(_HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.candidates = []
            self.stack = []
            self.link_depth = 0
            self.skip_depth = 0

        def handle_starttag(self, tag, attrs):
            if tag in _HTML_EVAL_SKIP_TAGS:
                self.skip_depth += 1
            if self.skip_depth:
                return
            if tag == "a":
                self.link_depth += 1
            if tag in {"article", "main", "section", "div", "body"}:
                attr_map = dict(attrs)
                marker = " ".join(
                    (attr_map.get("id", ""), attr_map.get("class", ""))
                ).lower()
                self.stack.append(
                    {
                        "tag": tag,
                        "marker": marker,
                        "parts": [],
                        "link_chars": 0,
                    }
                )

        def handle_endtag(self, tag):
            if tag in _HTML_EVAL_SKIP_TAGS and self.skip_depth:
                self.skip_depth -= 1
                return
            if self.skip_depth:
                return
            if tag == "a" and self.link_depth:
                self.link_depth -= 1
            if self.stack and tag == self.stack[-1]["tag"]:
                candidate = self.stack.pop()
                text = _html_normalize("".join(candidate["parts"]))
                if text:
                    bonus = (
                        1200
                        if candidate["tag"] == "article"
                        else 900
                        if candidate["tag"] == "main"
                        else 0
                    )
                    if any(
                        key in candidate["marker"]
                        for key in (
                            "article",
                            "content",
                            "main",
                            "post",
                            "story",
                        )
                    ):
                        bonus += 600
                    penalty = candidate["link_chars"] * 4
                    self.candidates.append(
                        {"text": text, "score": len(text) + bonus - penalty}
                    )

        def handle_data(self, data):
            if self.skip_depth:
                return
            for candidate in self.stack:
                candidate["parts"].append(data)
                if self.link_depth:
                    candidate["link_chars"] += len(data)

    def _html_residual_extract(value):
        renderer = _HtmlTextRenderer(layout=False)
        renderer.feed(value)
        renderer.close()
        return renderer.render(), {"tag_events": renderer.tag_count}

    def _html_layout_extract(value):
        renderer = _HtmlTextRenderer(layout=True)
        renderer.feed(value)
        renderer.close()
        return renderer.render(), {"tag_events": renderer.tag_count}

    def _html_block_density_extract(value):
        parser = _HtmlBlockCollector()
        parser.feed(value)
        parser.close()
        selected = [
            block["text"]
            for block in parser.blocks
            if len(block["text"]) >= 80 and block["link_ratio"] <= 0.30
        ]
        if not selected:
            selected = [
                block["text"]
                for block in sorted(
                    parser.blocks,
                    key=lambda item: len(item["text"]),
                    reverse=True,
                )[:3]
            ]
        return "\n\n".join(selected), {
            "candidate_blocks": len(parser.blocks),
            "selected_blocks": len(selected),
        }

    def _html_semantic_extract(value):
        parser = _HtmlSemanticCollector()
        parser.feed(value)
        parser.close()
        if not parser.candidates:
            return _html_residual_extract(value)
        best = max(parser.candidates, key=lambda item: item["score"])
        return best["text"], {
            "candidate_nodes": len(parser.candidates),
            "best_score": round(best["score"], 2),
        }

    def _html_jsonld_extract(value):
        matches = _html_re.findall(
            r'(?is)<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            value,
        )
        values = []
        parsed = 0
        for payload in matches:
            try:
                item = _html_json.loads(payload)
            except _html_json.JSONDecodeError:
                continue
            parsed += 1
            nodes = (
                item
                if isinstance(item, list)
                else item.get("@graph", [item])
                if isinstance(item, dict)
                else []
            )
            for node in nodes:
                if isinstance(node, dict):
                    for key in (
                        "headline",
                        "name",
                        "description",
                        "articleBody",
                    ):
                        item_value = node.get(key)
                        if isinstance(item_value, str):
                            values.append(item_value)
        return _html_normalize(" ".join(values)), {
            "jsonld_scripts": len(matches),
            "jsonld_parsed": parsed,
        }

    def _html_trafilatura_extract(value, url=None):
        extracted = (
            _html_trafilatura.extract(
                value,
                url=url,
                include_comments=False,
                include_tables=True,
                deduplicate=False,
                output_format="txt",
            )
            or ""
        )
        return _html_normalize(extracted), {"library": "trafilatura"}

    def _html_justext_extract(value):
        paragraphs = _html_justext.justext(
            value,
            _html_justext.get_stoplist("English"),
        )
        selected = [
            paragraph.text
            for paragraph in paragraphs
            if not paragraph.is_boilerplate and paragraph.text
        ]
        return "\n\n".join(selected), {
            "paragraphs": len(paragraphs),
            "content_paragraphs": len(selected),
        }

    def _html_inscriptis_extract(value):
        return _html_inscriptis.get_text(value).strip(), {
            "library": "inscriptis"
        }

    def _html_resiliparse_extract(value):
        extracted = _html_resiliparse_text(
            value,
            preserve_formatting=True,
            main_content=True,
            comments=False,
        )
        return extracted.strip(), {"library": "resiliparse"}

    def _html_readability_extract(value):
        summary_html = _HtmlReadabilityDocument(value).summary(
            html_partial=True
        )
        extracted, signals = _html_residual_extract(summary_html)
        return extracted, {"library": "readability-lxml", **signals}

    def _html_collect_metadata_strings(value):
        values = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {
                    "headline",
                    "name",
                    "description",
                    "articleBody",
                } and isinstance(item, str):
                    values.append(item)
                else:
                    values.extend(_html_collect_metadata_strings(item))
        elif isinstance(value, list):
            for item in value:
                values.extend(_html_collect_metadata_strings(item))
        return values

    def _html_extruct_extract(value):
        extracted = _html_extruct.extract(value, syntaxes=["json-ld"])
        values = _html_collect_metadata_strings(extracted.get("json-ld", []))
        return _html_normalize(" ".join(values)), {
            "library": "extruct",
            "jsonld_records": len(extracted.get("json-ld", [])),
        }

    def _html_load_or_cache(target_documents):
        HTML_EVAL_ROOT.mkdir(parents=True, exist_ok=True)
        HTML_EVAL_IMPL_DIR.mkdir(parents=True, exist_ok=True)
        records = []
        if HTML_EVAL_RAW_PATH.exists():
            with HTML_EVAL_RAW_PATH.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        records.append(_html_json.loads(line))
            if len(records) >= target_documents:
                return records[:target_documents], "cache"
        from datasets import load_dataset

        records = []
        skipped_oversized = 0
        stream = load_dataset(
            HTML_EVAL_DATASET,
            split=HTML_EVAL_SPLIT,
            streaming=True,
            revision=HTML_EVAL_REVISION,
        )
        with HTML_EVAL_RAW_PATH.open("w", encoding="utf-8") as handle:
            for source_index, source in enumerate(stream):
                html = source.get("html") or ""
                if not html or len(html) > _HTML_EVAL_MAX_CHARS:
                    skipped_oversized += 1
                    continue
                record = {
                    "source_index": source_index,
                    "c4_shard": source.get("c4_shard"),
                    "timestamp": source.get("c4_timestamp"),
                    "url": source.get("url"),
                    "html": html,
                    "reference_text": source.get("text") or "",
                }
                handle.write(
                    _html_json.dumps(record, ensure_ascii=False) + "\n"
                )
                records.append(record)
                if len(records) >= target_documents:
                    break
        HTML_EVAL_MANIFEST_PATH.write_text(
            _html_json.dumps(
                {
                    "dataset": HTML_EVAL_DATASET,
                    "revision": HTML_EVAL_REVISION,
                    "split": HTML_EVAL_SPLIT,
                    "selection": "first sequential non-empty HTML records under 500,000 characters",
                    "requested_documents": target_documents,
                    "retained_documents": len(records),
                    "skipped_oversized_or_empty": skipped_oversized,
                    "raw_path": str(HTML_EVAL_RAW_PATH),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return records, "stream"

    def run_html_technique_sample(technique_key, target_documents=30):
        records, selection_source = _html_load_or_cache(target_documents)
        techniques = {
            "trafilatura": (
                "Trafilatura",
                "trafilatura 2.2.0",
                lambda record: _html_trafilatura_extract(
                    record["html"], record["url"]
                ),
            ),
            "justext": (
                "jusText",
                "jusText 3.0.2",
                lambda record: _html_justext_extract(record["html"]),
            ),
            "inscriptis": (
                "Inscriptis",
                "inscriptis 2.7.3",
                lambda record: _html_inscriptis_extract(record["html"]),
            ),
            "resiliparse": (
                "Resiliparse",
                "resiliparse 1.0.9",
                lambda record: _html_resiliparse_extract(record["html"]),
            ),
            "readability": (
                "readability-lxml",
                "readability-lxml 0.8.4.1",
                lambda record: _html_readability_extract(record["html"]),
            ),
            "extruct": (
                "extruct",
                "extruct 0.18.0",
                lambda record: _html_extruct_extract(record["html"]),
            ),
        }
        if technique_key not in techniques:
            raise ValueError(f"Unknown technique: {technique_key}")
        technique, implementation, extractor = techniques[technique_key]
        output_characters = 0
        nonempty = 0
        overlap_sum = 0.0
        signals = _HtmlCounter()
        for record in records:
            output, measurement = extractor(record)
            output_characters += len(output)
            nonempty += bool(output)
            overlap_sum += _html_overlap(output, record["reference_text"])
            signals.update(measurement)
        return {
            "technique": technique,
            "implementation": implementation,
            "selection_source": selection_source,
            "documents": len(records),
            "nonempty_rate": round(100 * nonempty / max(1, len(records)), 2),
            "mean_output_characters": round(
                output_characters / max(1, len(records)), 1
            ),
            "mean_reference_token_overlap_percent": round(
                100 * overlap_sum / max(1, len(records)), 2
            ),
            "aggregate_signals": dict(signals),
        }

    def run_html_technique_evaluation(target_documents):
        records, selection_source = _html_load_or_cache(target_documents)
        techniques = [
            (
                "residual_markup",
                "Residual markup removal",
                "Transparent HTMLParser control",
                _html_residual_extract,
                "main-content baseline",
            ),
            (
                "block_density",
                "Block-density selection",
                "Transparent local control",
                _html_block_density_extract,
                "main-content candidate",
            ),
            (
                "semantic_dom",
                "Semantic DOM scoring",
                "Transparent local control",
                _html_semantic_extract,
                "main-content candidate",
            ),
            (
                "layout_renderer",
                "Layout-preserving rendering",
                "Transparent HTMLParser control",
                _html_layout_extract,
                "layout-preserving text",
            ),
            (
                "jsonld_metadata",
                "Structured metadata extraction",
                "Transparent local JSON-LD control",
                _html_jsonld_extract,
                "metadata only",
            ),
            (
                "trafilatura",
                "Trafilatura",
                "trafilatura 2.2.0",
                _html_trafilatura_extract,
                "library main-content extraction",
            ),
            (
                "justext",
                "jusText",
                "jusText 3.0.2",
                _html_justext_extract,
                "library boilerplate classification",
            ),
            (
                "inscriptis",
                "Inscriptis",
                "inscriptis 2.7.3",
                _html_inscriptis_extract,
                "library layout-preserving text",
            ),
            (
                "resiliparse",
                "Resiliparse",
                "resiliparse 1.0.9",
                _html_resiliparse_extract,
                "library high-throughput text extraction",
            ),
            (
                "readability",
                "readability-lxml",
                "readability-lxml 0.8.4.1",
                _html_readability_extract,
                "library main-content extraction",
            ),
            (
                "extruct",
                "extruct",
                "extruct 0.18.0",
                _html_extruct_extract,
                "library structured metadata extraction",
            ),
        ]
        aggregate = {
            key: {
                "documents": 0,
                "nonempty": 0,
                "output_characters": 0,
                "overlap_sum": 0.0,
                "signals": _HtmlCounter(),
            }
            for key, *_ in techniques
        }
        raw_html_characters = 0
        reference_characters = 0
        document_rows = []
        with HTML_EVAL_DOCUMENT_METRICS_PATH.open(
            "w", encoding="utf-8"
        ) as output:
            for record in records:
                raw_html_characters += len(record["html"])
                reference_characters += len(record["reference_text"])
                document_metrics = {
                    "source_index": record["source_index"],
                    "url": record["url"],
                    "raw_html_characters": len(record["html"]),
                    "reference_characters": len(record["reference_text"]),
                    "techniques": {},
                }
                for key, _, _, extractor, _ in techniques:
                    extracted, signals = extractor(record["html"])
                    overlap = _html_overlap(
                        extracted, record["reference_text"]
                    )
                    stats = aggregate[key]
                    stats["documents"] += 1
                    stats["nonempty"] += bool(extracted)
                    stats["output_characters"] += len(extracted)
                    stats["overlap_sum"] += overlap
                    stats["signals"].update(signals)
                    document_metrics["techniques"][key] = {
                        "output_characters": len(extracted),
                        "reference_token_overlap": round(overlap, 6),
                        "signals": signals,
                    }
                output.write(
                    _html_json.dumps(document_metrics, ensure_ascii=False)
                    + "\n"
                )
        metrics = []
        for key, name, implementation, _, role in techniques:
            stats = aggregate[key]
            metrics.append(
                {
                    "technique": name,
                    "implementation": implementation,
                    "role": role,
                    "status": "executed",
                    "documents": stats["documents"],
                    "nonempty_rate": round(
                        100 * stats["nonempty"] / max(1, stats["documents"]), 2
                    ),
                    "mean_output_characters": round(
                        stats["output_characters"]
                        / max(1, stats["documents"]),
                        1,
                    ),
                    "output_vs_html_percent": round(
                        100
                        * stats["output_characters"]
                        / max(1, raw_html_characters),
                        3,
                    ),
                    "mean_reference_token_overlap_percent": round(
                        100
                        * stats["overlap_sum"]
                        / max(1, stats["documents"]),
                        2,
                    ),
                    "aggregate_signals": dict(stats["signals"]),
                }
            )
        for library, available in _HTML_EVAL_LIBRARY_STATUS.items():
            if not available:
                metrics.append(
                    {
                        "technique": library,
                        "implementation": "External library adapter",
                        "role": "library implementation",
                        "status": "not executed: dependency unavailable in active kernel",
                        "documents": 0,
                        "nonempty_rate": None,
                        "mean_output_characters": None,
                        "output_vs_html_percent": None,
                        "mean_reference_token_overlap_percent": None,
                        "aggregate_signals": {},
                    }
                )
        metrics.extend(
            [
                {
                    "technique": "MinerU-HTML / AICC",
                    "implementation": "Learned HTML sequence-labeling model",
                    "role": "model-based main-content extraction",
                    "status": "not executed: model/runtime unavailable; no Hindi validation claim",
                    "documents": 0,
                    "nonempty_rate": None,
                    "mean_output_characters": None,
                    "output_vs_html_percent": None,
                    "mean_reference_token_overlap_percent": None,
                    "aggregate_signals": {},
                },
                {
                    "technique": "SCRIBES",
                    "implementation": "Per-site semi-structured script generation",
                    "role": "site-specific structured extraction",
                    "status": "not executed: no runtime; requires same-site page groups",
                    "documents": 0,
                    "nonempty_rate": None,
                    "mean_output_characters": None,
                    "output_vs_html_percent": None,
                    "mean_reference_token_overlap_percent": None,
                    "aggregate_signals": {},
                },
            ]
        )
        result = {
            "dataset": HTML_EVAL_DATASET,
            "revision": HTML_EVAL_REVISION,
            "selection_source": selection_source,
            "documents": len(records),
            "raw_html_characters": raw_html_characters,
            "reference_text_characters": reference_characters,
            "library_status": _HTML_EVAL_LIBRARY_STATUS,
            "metrics": metrics,
            "manifest_path": str(HTML_EVAL_MANIFEST_PATH),
            "metrics_path": str(HTML_EVAL_METRICS_PATH),
            "document_metrics_path": str(HTML_EVAL_DOCUMENT_METRICS_PATH),
        }
        HTML_EVAL_METRICS_PATH.write_text(
            _html_json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result

    def load_html_technique_evaluation():
        if not HTML_EVAL_METRICS_PATH.exists():
            return None
        return _html_json.loads(
            HTML_EVAL_METRICS_PATH.read_text(encoding="utf-8")
        )

    return (
        HTML_EVAL_DOCUMENT_METRICS_PATH,
        load_html_technique_evaluation,
        run_html_technique_evaluation,
        run_html_technique_sample,
    )


@app.cell(hide_code=True)
def html_eval_controls(mo):
    html_eval_target_documents = mo.ui.slider(
        start=10,
        stop=50,
        step=10,
        value=30,
        label="Deterministic raw-HTML documents",
        show_value=True,
    )
    run_html_evaluation = mo.ui.run_button(
        label="Run all extraction techniques"
    )
    mo.vstack(
        [
            mo.md(
                "## Step 1 — Run the shared evaluation corpus\nThe cached corpus lives under `extraction/data/c4_en_html_evaluation/`. Running it writes aggregate and document-level measurements without rendering source-page text."
            ),
            mo.hstack([html_eval_target_documents, run_html_evaluation]),
        ]
    )
    return html_eval_target_documents, run_html_evaluation


@app.cell(hide_code=True)
def html_eval_run(
    html_eval_target_documents,
    load_html_technique_evaluation,
    mo,
    run_html_evaluation,
    run_html_technique_evaluation,
):
    if run_html_evaluation.value:
        html_eval_result = run_html_technique_evaluation(
            int(html_eval_target_documents.value)
        )
        html_eval_status = f"HTML extraction comparison completed for {html_eval_result['documents']:,} documents."
    else:
        html_eval_result = load_html_technique_evaluation()
        html_eval_status = (
            "Existing HTML extraction evaluation loaded."
            if html_eval_result
            else "Execute the comparison to materialize the bounded HTML subset."
        )
    mo.md(html_eval_status)
    return (html_eval_result,)


@app.cell(hide_code=True)
def trafilatura_explanation(mo):
    trafilatura_explanation_view = mo.md(r"""## Trafilatura — DOM main-content extraction

    ### What it does
    Trafilatura turns a web page into a prose-oriented text candidate. It is designed to keep the page’s article-like content while discarding common surrounding material such as navigation, sidebars, cookie notices, related-link clusters, and repeated template text.

    ### How it works under the hood
    1. Parses the raw HTML into a document tree.
    2. Removes or deprioritizes elements that are structurally likely to be non-content.
    3. Evaluates remaining text-bearing elements with structural, textual, and link-related signals.
    4. Assembles the selected elements into text, with optional tables, links, metadata, and comments controlled by parameters.

    ### Use it when
    Use Trafilatura as a strong general-purpose baseline for article/news/blog extraction when raw HTML and the page URL are retained. It is appropriate when output should favor readable prose over preserving every page element.

    ### Advantages
    - Mature, configurable article-extraction workflow.
    - Good default comparison point for raw web pages.
    - Can include or exclude comments, tables, links, metadata, and duplicate text deliberately.

    ### Limitations and failure modes
    - A page with several similarly dense regions can select the wrong one.
    - Non-article pages, short pages, product pages, and custom site templates need review.
    - High retained-text overlap does not prove that boilerplate was removed correctly.

    ### Notebook note
    This run uses `include_comments=False`, keeps tables, and records only aggregate measurements. For Hindi deployment, build source-page labels first; do not transfer this English-C4 result or its defaults as a Hindi quality threshold.""")
    trafilatura_explanation_view
    return


@app.cell
def trafilatura_execution(mo, run_html_technique_sample):
    trafilatura_execution_result = run_html_technique_sample("trafilatura")
    trafilatura_execution_view = mo.vstack(
        [
            mo.md("""### Executable corpus run — Trafilatura — DOM main-content extraction
    Runs the installed library on the cached 30-document raw-HTML subset. The table exposes measurements only; it does not render source-page content."""),
            mo.ui.table([trafilatura_execution_result]),
        ]
    )
    trafilatura_execution_view
    return


@app.cell(hide_code=True)
def justext_explanation(mo):
    justext_explanation_view = mo.md(r"""## jusText — block-level boilerplate classification

    ### What it does
    jusText separates page text into blocks, then classifies each block as content or boilerplate. It is aimed at removing menus, repeated navigation, link lists, and template fragments without needing a site-specific selector.

    ### How it works under the hood
    1. Parses HTML into paragraph-like text blocks.
    2. Computes block properties such as length, link density, stopword density, and paragraph structure.
    3. Marks obvious good and bad blocks.
    4. Revises ambiguous blocks by looking at neighboring blocks, which prevents isolated short content fragments from being discarded too eagerly.

    ### Use it when
    Use jusText when you want an interpretable, block-level baseline for pages with strong navigation and repeated templates. It is useful for diagnosing *why* text was retained or rejected.

    ### Advantages
    - Decisions are made at paragraph/block level rather than as opaque whole-page selection.
    - Link density makes it useful for menu-heavy pages.
    - The contextual revision step is more robust than a single length cutoff.

    ### Limitations and failure modes
    - Stopword and whitespace assumptions are language-sensitive.
    - Headings, short notices, verse, captions, and lists can resemble boilerplate.
    - It extracts prose blocks; it is not a layout-preserving converter.

    ### Notebook note
    The executable auxiliary corpus is English, so this call uses jusText’s English stoplist. That is intentional for this comparison and must not be copied to Hindi/Hinglish data without a Hindi-literate error analysis.""")
    justext_explanation_view
    return


@app.cell
def justext_execution(mo, run_html_technique_sample):
    justext_execution_result = run_html_technique_sample("justext")
    justext_execution_view = mo.vstack(
        [
            mo.md("""### Executable corpus run — jusText — block-density boilerplate classification
    Runs the installed library on the cached 30-document raw-HTML subset. The table exposes measurements only; it does not render source-page content."""),
            mo.ui.table([justext_execution_result]),
        ]
    )
    justext_execution_view
    return


@app.cell(hide_code=True)
def inscriptis_explanation(mo):
    inscriptis_explanation_view = mo.md(r"""## Inscriptis — layout-aware HTML-to-text rendering

    ### What it does
    Inscriptis converts HTML to readable text while preserving useful display boundaries. Its goal is a faithful text rendering of the page structure—not choosing one article region and deleting everything else.

    ### How it works under the hood
    1. Parses HTML and walks the document structure.
    2. Applies rendering rules for block elements, line breaks, lists, tables, and other layout-relevant tags.
    3. Omits scripts/styles and emits text with structural spacing instead of naïvely concatenating every text node.
    4. Returns a layout-aware textual representation.

    ### Use it when
    Use Inscriptis when downstream work needs paragraphs, lists, or table-related boundaries to remain understandable: archival conversion, document indexing, layout-sensitive cleaning, or pre-processing before human review.

    ### Advantages
    - Preserves readability better than raw tag stripping.
    - Useful for structurally rich pages containing lists and tables.
    - Deliberately does not pretend that layout conversion is article selection.

    ### Limitations and failure modes
    - It may retain navigation, footer text, and other boilerplate because it is not a main-content extractor.
    - CSS-heavy or JavaScript-rendered pages may not match the browser’s final visual layout.
    - Large output is not automatically a better extraction.

    ### Notebook note
    Interpret this run’s high output retention as layout preservation, not quality superiority. Pair it with a boilerplate/main-content method when the training corpus requires only central prose.""")
    inscriptis_explanation_view
    return


@app.cell
def inscriptis_execution(mo, run_html_technique_sample):
    inscriptis_execution_result = run_html_technique_sample("inscriptis")
    inscriptis_execution_view = mo.vstack(
        [
            mo.md("""### Executable corpus run — Inscriptis — layout-aware HTML rendering
    Runs the installed library on the cached 30-document raw-HTML subset. The table exposes measurements only; it does not render source-page content."""),
            mo.ui.table([inscriptis_execution_result]),
        ]
    )
    inscriptis_execution_view
    return


@app.cell(hide_code=True)
def resiliparse_explanation(mo):
    resiliparse_explanation_view = mo.md(r"""## Resiliparse — high-throughput HTML text extraction

    ### What it does
    Resiliparse provides an optimized HTML parsing and text-extraction path for crawl-scale pipelines. It is useful when raw HTML must be processed quickly while preserving a source-artifact-first workflow.

    ### How it works under the hood
    1. Parses raw HTML with an optimized DOM implementation.
    2. Traverses the DOM and removes excluded or non-textual elements.
    3. Applies its main-content extraction mode to focus the text result.
    4. Emits plain text with optional formatting, links, comments, and metadata behavior controlled by the extraction call.

    ### Use it when
    Use Resiliparse for large HTML/WARC collections where parse throughput and memory behavior matter, and where every output can still be traced to the original response.

    ### Advantages
    - Designed for high-volume web processing.
    - Fits streaming source → extract → record pipelines.
    - Lets a pipeline separate throughput benchmarking from content-quality evaluation.

    ### Limitations and failure modes
    - Fast parsing does not solve article-boundary ambiguity.
    - Main-content heuristics can still fail on unusual templates or semi-structured pages.
    - Runtime speed and retained character count are not quality metrics.

    ### Notebook note
    This notebook calls `extract_plain_text(..., main_content=True)` and preserves formatting. Before using it for a corpus decision, measure domain-stratified precision, false deletions, throughput, and memory independently.""")
    resiliparse_explanation_view
    return


@app.cell
def resiliparse_execution(mo, run_html_technique_sample):
    resiliparse_execution_result = run_html_technique_sample("resiliparse")
    resiliparse_execution_view = mo.vstack(
        [
            mo.md("""### Executable corpus run — Resiliparse — high-throughput main-content text
    Runs the installed library on the cached 30-document raw-HTML subset. The table exposes measurements only; it does not render source-page content."""),
            mo.ui.table([resiliparse_execution_result]),
        ]
    )
    resiliparse_execution_view
    return


@app.cell(hide_code=True)
def readability_explanation(mo):
    readability_explanation_view = mo.md(r"""## readability-lxml — article-candidate scoring

    ### What it does
    readability-lxml is a Python implementation of the Readability-style approach: it seeks one DOM container that most resembles the page’s primary article, then returns that candidate’s HTML/text.

    ### How it works under the hood
    1. Parses the HTML with lxml.
    2. Scores candidate containers using text volume, link-heavy content, and article-like or boilerplate-like structural markers.
    3. Selects the highest-scoring candidate as the probable main article.
    4. Produces cleaned candidate HTML; this notebook then renders that HTML into text for measurement.

    ### Use it when
    Use readability-lxml for conventional article/news/blog pages where one main content container is expected and you want a compact, readable baseline.

    ### Advantages
    - Simple single-article selection model.
    - Often effective for editorial web pages with a clear article body.
    - Useful comparator against multi-signal tools such as Trafilatura.

    ### Limitations and failure modes
    - A single winning container is a poor fit for forums, search pages, catalogs, dashboards, and multi-article pages.
    - Short content, image-led pages, and tables can be under-selected.
    - CSS class/id hints are heuristic; site conventions are not universal.

    ### Notebook note
    The code explicitly renders Readability’s selected HTML through the same residual renderer used for a control, so the measured difference reflects candidate selection rather than two unrelated text serializers.""")
    readability_explanation_view
    return


@app.cell
def readability_execution(mo, run_html_technique_sample):
    readability_execution_result = run_html_technique_sample("readability")
    readability_execution_view = mo.vstack(
        [
            mo.md("""### Executable corpus run — readability-lxml — semantic article scoring
    Runs the installed library on the cached 30-document raw-HTML subset. The table exposes measurements only; it does not render source-page content."""),
            mo.ui.table([readability_execution_result]),
        ]
    )
    readability_execution_view
    return


@app.cell(hide_code=True)
def extruct_explanation(mo):
    extruct_explanation_view = mo.md(r"""## extruct — structured metadata extraction

    ### What it does
    extruct extracts machine-readable metadata embedded in HTML. It is a provenance and routing tool, not a replacement for article-body extraction.

    ### How it works under the hood
    1. Parses HTML looking for structured-data encodings.
    2. Decodes formats such as JSON-LD, Microdata, RDFa, Open Graph, or microformats when requested.
    3. Returns structured Python records rather than a page-text candidate.
    4. A caller chooses which fields—such as `headline`, `description`, or `articleBody`—are useful for its policy.

    ### Use it when
    Use extruct when you need publisher-provided titles, dates, authors, canonical metadata, schema types, or a possible article-body field for provenance checks, routing, and validation.

    ### Advantages
    - Preserves explicit structured fields when publishers expose them.
    - Helpful for source-level provenance and downstream quality checks.
    - Complements prose extractors by providing metadata they may discard.

    ### Limitations and failure modes
    - Metadata may be missing, stale, incomplete, or inconsistent with visible text.
    - A JSON-LD `articleBody` field is publisher supplied; it is not ground truth.
    - Its output size is not comparable to a main-content extractor.

    ### Notebook note
    This executable run intentionally requests **JSON-LD only** and collects a conservative field subset. Treat low coverage as a source characteristic, not an extraction failure; never compare its token yield directly with Trafilatura, jusText, or Readability.""")
    extruct_explanation_view
    return


@app.cell
def extruct_execution(mo, run_html_technique_sample):
    extruct_execution_result = run_html_technique_sample("extruct")
    extruct_execution_view = mo.vstack(
        [
            mo.md("""### Executable corpus run — extruct — structured metadata extraction
    Runs the installed library on the cached 30-document raw-HTML subset. The table exposes measurements only; it does not render source-page content."""),
            mo.ui.table([extruct_execution_result]),
        ]
    )
    extruct_execution_view
    return


@app.cell(hide_code=True)
def html_eval_report(HTML_EVAL_DOCUMENT_METRICS_PATH, html_eval_result, mo):
    if html_eval_result is None:
        html_eval_report = mo.md(
            "## HTML Extraction Comparison\nNo evaluation output is available."
        )
    else:
        import json as _report_json
        from urllib.parse import urlparse as _report_urlparse

        _html_eval_summary_rows = [
            {"metric": "dataset", "value": html_eval_result["dataset"]},
            {"metric": "revision", "value": html_eval_result["revision"]},
            {
                "metric": "selection source",
                "value": html_eval_result["selection_source"],
            },
            {
                "metric": "documents",
                "value": f"{html_eval_result['documents']:,}",
            },
            {
                "metric": "raw HTML characters",
                "value": f"{html_eval_result['raw_html_characters']:,}",
            },
            {
                "metric": "supplied reference-text characters",
                "value": f"{html_eval_result['reference_text_characters']:,}",
            },
        ]
        _html_eval_trace_rows = []
        with HTML_EVAL_DOCUMENT_METRICS_PATH.open(
            encoding="utf-8"
        ) as _trace_handle:
            for _trace_line in _trace_handle:
                if len(_html_eval_trace_rows) == 3:
                    break
                _trace = _report_json.loads(_trace_line)
                _trace_rows = _trace["techniques"]
                _html_eval_trace_rows.append(
                    {
                        "source index": _trace["source_index"],
                        "source host": _report_urlparse(
                            _trace["url"] or ""
                        ).netloc,
                        "raw HTML chars": _trace["raw_html_characters"],
                        "reference chars": _trace["reference_characters"],
                        "residual chars": _trace_rows["residual_markup"][
                            "output_characters"
                        ],
                        "block-density chars": _trace_rows["block_density"][
                            "output_characters"
                        ],
                        "semantic-DOM chars": _trace_rows["semantic_dom"][
                            "output_characters"
                        ],
                        "layout chars": _trace_rows["layout_renderer"][
                            "output_characters"
                        ],
                        "JSON-LD chars": _trace_rows["jsonld_metadata"][
                            "output_characters"
                        ],
                    }
                )
        _html_eval_notes = (
            "### Interpretation\n"
            "The comparison executes five transparent local controls and six installed third-party extractors. "
            "Reference-token overlap is agreement with the supplied C4 text, not a human quality score. "
            "The C4 comparison corpus is English, so these results establish executable behavior—not Hindi extraction quality. "
            "`jsonld_metadata` and `extruct` extract metadata rather than article prose, so their character-retention figures are not comparable to main-content extractors."
        )
        html_eval_report = mo.vstack(
            [
                mo.md("## Final comparison — all extraction techniques"),
                mo.ui.table(_html_eval_summary_rows),
                mo.md(
                    "### Document-Level Extraction Trace\nThe trace reports structural output measurements only; raw page content is not displayed."
                ),
                mo.ui.table(_html_eval_trace_rows),
                mo.md("### Unified Technique Metrics"),
                mo.ui.table(html_eval_result["metrics"]),
                mo.md(_html_eval_notes),
                mo.md(
                    f"**Artifacts:** manifest `{html_eval_result['manifest_path']}` · aggregate metrics `{html_eval_result['metrics_path']}` · document-level metrics `{html_eval_result['document_metrics_path']}`"
                ),
            ]
        )
    html_eval_report
    return


if __name__ == "__main__":
    app.run()
