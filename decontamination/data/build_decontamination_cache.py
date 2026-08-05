#!/usr/bin/env python3
"""Deterministic caches for the Session 4 decontamination workbook.

Builds two artefacts under this directory:

1. ``sangraha_hin_stride_probe.jsonl`` -- the corpus side. Fixed-stride windows
   across the whole ``verified/hin`` split of ``ai4bharat/sangraha`` at a pinned
   revision. The stride is a function of the split size alone, so the selection
   rule is reproducible and is not conditioned on contamination signal.

2. ``eval_suite.jsonl`` -- the held-out side. Hindi evaluation items from three
   public benchmarks, each row carrying the field that a leak would actually
   reproduce (the passage, the premise, the question stem).

Neither file is committed. ``probe_manifest.json`` records row counts, offsets
and SHA-256 digests so a reader can rebuild byte-identical caches.

Usage
-----
    uv run python decontamination/data/build_decontamination_cache.py
    uv run python decontamination/data/build_decontamination_cache.py --windows 40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

CORPUS_DATASET = "ai4bharat/sangraha"
CORPUS_REVISION = "8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8"
CORPUS_CONFIG = "verified"
CORPUS_SPLIT = "hin"
CORPUS_LICENSE = "CC-BY-4.0"
WINDOW_ROWS = 100
DEFAULT_WINDOWS = 120

ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
SIZE_ENDPOINT = "https://datasets-server.huggingface.co/size"
USER_AGENT = "s4-decontamination-workbook/1.0"

# Held-out side. ``text_fields`` are the fields whose surface form a leaked
# document would reproduce; ``answer_fields`` are recorded separately because a
# leak of the answer key is a strictly worse event than a leak of the stem.
EVAL_SOURCES = [
    {
        "eval_id": "indicqa.hi",
        "dataset": "ai4bharat/IndicQA",
        "revision": "main",
        "split": "test",
        "license": "CC-BY-4.0",
        "access": "file",
        "file_url": "https://huggingface.co/datasets/ai4bharat/IndicQA/resolve/main/data/indicqa.hi.json",
        "role": "extractive QA over Wikipedia-derived Hindi passages",
        "why": "highest-risk surface: eval contexts are long prose paragraphs "
        "sourced from the same open web a crawl ingests",
    },
    {
        "eval_id": "milu.hindi",
        "dataset": "murthyrudra/milu-cleaned",
        "revision": "main",
        "config": "Hindi",
        "split": "test",
        "license": "CC-BY-4.0",
        "access": "rows",
        "cap": 15_000,
        "text_fields": ["question"],
        "answer_fields": ["option1", "option2", "option3", "option4", "target"],
        "role": "multiple-choice knowledge exam, Hindi subset",
        "why": "short stems; tests whether a 13-gram window can even span an "
        "evaluation item",
    },
    {
        "eval_id": "xnli.hi",
        "dataset": "facebook/xnli",
        "revision": "main",
        "config": "hi",
        "split": "test",
        "license": "CC-BY-NC-4.0",
        "access": "rows",
        "cap": 5_100,
        "text_fields": ["premise", "hypothesis"],
        "answer_fields": ["label"],
        "role": "natural-language inference, translated Hindi test split",
        "why": "translationese premises; a near-duplicate detector should see "
        "these differently from native web prose",
    },
]


def _get(url: str, timeout: int = 90, attempts: int = 8) -> dict:
    last: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as handle:
                return json.load(handle)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
            last = error
            # The rows endpoint rate-limits aggressively; back off geometrically
            # rather than giving up, because a partial probe is not reproducible.
            time.sleep(min(2.0 * 2**attempt, 45.0))
    raise RuntimeError(f"GET failed after {attempts} attempts: {url}") from last


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def split_rows(dataset: str, config: str, split: str) -> int:
    payload = _get(f"{SIZE_ENDPOINT}?dataset={urllib.parse.quote(dataset, safe='')}")
    for entry in payload["size"]["splits"]:
        if entry["config"] == config and entry["split"] == split:
            return int(entry["num_rows"])
    raise SystemExit(f"split not found: {dataset} {config}/{split}")


def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    revision: str | None = None,
) -> list[dict]:
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    if revision and revision != "main":
        params["revision"] = revision
    payload = _get(f"{ROWS_ENDPOINT}?{urllib.parse.urlencode(params)}")
    return [row["row"] for row in payload["rows"]]


def build_corpus_probe(windows: int, output: Path, pause: float) -> dict:
    total = split_rows(CORPUS_DATASET, CORPUS_CONFIG, CORPUS_SPLIT)
    stride = total // windows
    offsets = [index * stride for index in range(windows)]

    kept = 0
    characters = 0
    with output.open("w", encoding="utf-8") as handle:
        for index, offset in enumerate(offsets):
            rows = fetch_rows(
                CORPUS_DATASET,
                CORPUS_CONFIG,
                CORPUS_SPLIT,
                offset,
                WINDOW_ROWS,
                CORPUS_REVISION,
            )
            for position, row in enumerate(rows):
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                record = {
                    "doc_id": row.get("doc_id")
                    or hashlib.sha1(text.encode()).hexdigest(),
                    "source_type": row.get("type"),
                    "stream_offset": offset + position,
                    "window_offset": offset,
                    "text": text,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                characters += len(text)
            print(f"  window {index + 1}/{windows} @ offset {offset}: {kept} docs")
            time.sleep(pause)

    return {
        "dataset": CORPUS_DATASET,
        "revision": CORPUS_REVISION,
        "config": CORPUS_CONFIG,
        "split": CORPUS_SPLIT,
        "license": CORPUS_LICENSE,
        "split_rows": total,
        "windows": windows,
        "window_rows": WINDOW_ROWS,
        "stride": stride,
        "selection_rule": (
            f"{windows} fixed-stride windows of {WINDOW_ROWS} consecutive rows; "
            f"offset_i = i * floor({total} / {windows}); stream order preserved; "
            "no shuffling and no contamination-conditioned selection"
        ),
        "documents": kept,
        "characters": characters,
        "path": str(output.relative_to(HERE.parent.parent)),
        "sha256": _digest(output),
    }


def _eval_texts(spec: dict, row: dict) -> tuple[list[str], list[str]]:
    texts = [
        str(row[field]) for field in spec["text_fields"] if str(row.get(field, "")).strip()
    ]
    answers = [
        str(row[field])
        for field in spec.get("answer_fields", [])
        if str(row.get(field, "")).strip()
    ]
    return texts, answers


def build_eval_suite(output: Path, pause: float) -> list[dict]:
    summaries: list[dict] = []
    with output.open("w", encoding="utf-8") as handle:
        for spec in EVAL_SOURCES:
            items = 0
            characters = 0

            if spec["access"] == "file":
                request = urllib.request.Request(
                    spec["file_url"], headers={"User-Agent": USER_AGENT}
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    payload = json.load(response)
                for article in payload["data"]:
                    for paragraph in article["paragraphs"]:
                        context = (paragraph.get("context") or "").strip()
                        questions = [
                            (qa.get("question") or "").strip()
                            for qa in paragraph.get("qas", [])
                        ]
                        answers = [
                            answer["text"]
                            for qa in paragraph.get("qas", [])
                            for answer in qa.get("answers", [])
                            if answer.get("text")
                        ]
                        if not context:
                            continue
                        record = {
                            "eval_id": spec["eval_id"],
                            "item_index": items,
                            "texts": [context, *[q for q in questions if q]],
                            "answers": answers,
                            "primary_text": context,
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        items += 1
                        characters += sum(len(text) for text in record["texts"])
            else:
                offset = 0
                cap = spec["cap"]
                while offset < cap:
                    rows = fetch_rows(
                        spec["dataset"],
                        spec["config"],
                        spec["split"],
                        offset,
                        WINDOW_ROWS,
                        spec["revision"],
                    )
                    if not rows:
                        break
                    for row in rows:
                        texts, answers = _eval_texts(spec, row)
                        if not texts:
                            continue
                        record = {
                            "eval_id": spec["eval_id"],
                            "item_index": items,
                            "texts": texts,
                            "answers": answers,
                            "primary_text": texts[0],
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        items += 1
                        characters += sum(len(text) for text in texts)
                    offset += WINDOW_ROWS
                    time.sleep(pause)

            summaries.append(
                {
                    "eval_id": spec["eval_id"],
                    "dataset": spec["dataset"],
                    "revision": spec["revision"],
                    "config": spec.get("config"),
                    "split": spec["split"],
                    "license": spec["license"],
                    "role": spec["role"],
                    "why": spec["why"],
                    "items": items,
                    "characters": characters,
                }
            )
            print(f"  {spec['eval_id']}: {items} items, {characters} chars")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument(
        "--pause",
        type=float,
        default=1.2,
        help="seconds between rows-endpoint calls; the endpoint rate-limits",
    )
    parser.add_argument(
        "--corpus-output", type=Path, default=HERE / "sangraha_hin_stride_probe.jsonl"
    )
    parser.add_argument("--eval-output", type=Path, default=HERE / "eval_suite.jsonl")
    parser.add_argument("--manifest", type=Path, default=HERE / "probe_manifest.json")
    arguments = parser.parse_args()

    print("building corpus probe ...")
    corpus = build_corpus_probe(
        arguments.windows, arguments.corpus_output, arguments.pause
    )
    print(f"  {corpus['documents']} documents, {corpus['characters']} chars")

    print("building eval suite ...")
    evals = build_eval_suite(arguments.eval_output, arguments.pause)

    manifest = {
        "built_utc": datetime.now(UTC).isoformat(),
        "builder": "decontamination/data/build_decontamination_cache.py",
        "purpose": "held-out contamination measurement for Session 4 stage 7",
        "corpus": corpus,
        "eval_suite": {
            "path": str(arguments.eval_output.relative_to(HERE.parent.parent)),
            "sha256": _digest(arguments.eval_output),
            "items": sum(entry["items"] for entry in evals),
            "sources": evals,
        },
    }
    arguments.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {arguments.manifest}")


if __name__ == "__main__":
    main()
