#!/usr/bin/env python3
"""Run the decontamination ladder over the cached probe and freeze the results.

Writes ``data/measurement_results.json``, which the notebook reads so that the
reported numbers come from one recorded run rather than being recomputed at
render time with different parameters.

    uv run python decontamination/run_measurement.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import decontamination_lib as dl

DATA = HERE / "data"
PROBE = DATA / "sangraha_hin_stride_probe.jsonl"
EVALS = DATA / "eval_suite.jsonl"
OUTPUT = DATA / "measurement_results.json"

NGRAM_SIZES = (8, 13, 20)
NORMALIZERS = ("indic_safe", "naive")
REFERENCE_NGRAM = 13
MINHASH_RUNS = (
    {"window_tokens": None, "threshold": 0.5},
    {"window_tokens": 64, "threshold": 0.3},
    {"window_tokens": 64, "threshold": 0.5},
)
# Stated assumption for the clean/dirty arithmetic, not a measurement: an
# honest baseline accuracy and a memorizing model that answers every reachable
# item correctly. The inflation figure is what follows from these two numbers.
ASSUMED_CLEAN_ACCURACY = 0.41
ASSUMED_MEMORIZED_ACCURACY = 1.0


def main() -> None:
    documents = dl.load_documents(PROBE)
    items = dl.load_eval_items(EVALS)
    eval_ids = sorted({item.eval_id for item in items})
    by_id = {document.doc_id: document for document in documents}

    total_characters = sum(document.characters for document in documents)
    total_tokens = sum(dl.estimate_tokens(document.text) for document in documents)

    overlap: dict[str, dict] = {}
    reference_hits: set[str] = set()
    for size in NGRAM_SIZES:
        for normalizer in NORMALIZERS:
            result = dl.ngram_overlap_scan(
                documents, items, ngram_size=size, normalizer=normalizer
            )
            flagged_docs = [by_id[row["doc_id"]] for row in result.flagged_documents]
            cost_characters = sum(document.characters for document in flagged_docs)
            cost_tokens = sum(dl.estimate_tokens(document.text) for document in flagged_docs)
            if size == REFERENCE_NGRAM and normalizer == "indic_safe":
                reference_hits = {row["doc_id"] for row in result.flagged_documents}
            overlap[f"n{size}_{normalizer}"] = {
                "ngram_size": size,
                "normalizer": normalizer,
                "flagged_documents": result.flagged_count,
                "flag_rate": result.flag_rate,
                "eval_ngrams_indexed": result.eval_ngrams,
                "eval_items_indexable": result.eval_items_indexable,
                "eval_items_total": result.eval_items_total,
                "reached_eval_items": {
                    key: len(value) for key, value in sorted(result.reached_eval_items.items())
                },
                "cost_characters": cost_characters,
                "cost_character_share": cost_characters / total_characters,
                "cost_tokens": cost_tokens,
                "cost_token_share": cost_tokens / total_tokens,
                "top_flags": [
                    {
                        "doc_id": row["doc_id"],
                        "stream_offset": row["stream_offset"],
                        "source_type": row["source_type"],
                        "shared_ngrams": row["shared_ngrams"],
                        "document_ngrams": row["document_ngrams"],
                        "shared_fraction": row["shared_fraction"],
                        "longest_shared": row["longest_shared"],
                        "eval_items_touched": row["eval_items_touched"][:5],
                    }
                    for row in result.flagged_documents[:12]
                ],
            }

    # Precision audit. Every flag at the reference setting was read by hand and
    # confirmed to be Devanagari Wikipedia prose overlapping an IndicQA context,
    # so that set is the audited positive set; looser settings are scored
    # against it.
    precision: dict[str, dict] = {}
    for size in NGRAM_SIZES:
        entry = overlap[f"n{size}_indic_safe"]
        flagged = {row["doc_id"] for row in
                   dl.ngram_overlap_scan(documents, items, ngram_size=size).flagged_documents}
        true_positives = len(flagged & reference_hits)
        precision[f"n{size}"] = {
            "flagged": entry["flagged_documents"],
            "audited_true_positives": true_positives,
            "false_positives": entry["flagged_documents"] - true_positives,
            "precision": (
                true_positives / entry["flagged_documents"] if entry["flagged_documents"] else None
            ),
        }

    # Structural indexability: an item shorter than the window can never be
    # matched, which is a property of the eval set rather than of the scan.
    indexability: dict[str, dict[str, dict]] = {}
    for size in NGRAM_SIZES:
        per_eval: dict[str, dict] = {}
        for eval_id in eval_ids:
            subset = [item for item in items if item.eval_id == eval_id]
            long_enough = sum(1 for item in subset if len(item.safe_tokens) >= size)
            per_eval[eval_id] = {
                "items": len(subset),
                "indexable": long_enough,
                "share": long_enough / len(subset),
                "median_tokens": sorted(len(item.safe_tokens) for item in subset)[
                    len(subset) // 2
                ],
            }
        indexability[f"n{size}"] = per_eval

    minhash: list[dict] = []
    for run in MINHASH_RUNS:
        result = dl.minhash_lsh_scan(
            documents,
            items,
            permutations=128,
            bands=32,
            shingle_size=5,
            threshold=run["threshold"],
            window_tokens=run["window_tokens"],
        )
        minhash.append(
            {
                "window_tokens": result.window_tokens,
                "threshold": result.threshold,
                "permutations": result.permutations,
                "bands": result.bands,
                "rows": result.rows,
                "shingle_size": result.shingle_size,
                "candidate_pairs": result.candidate_pairs,
                "confirmed_pairs": result.confirmed_count,
                "documents_involved": len({row["doc_id"] for row in result.confirmed_pairs}),
                "eval_items_indexed": result.eval_items_indexed,
                "pairs": result.confirmed_pairs[:10],
            }
        )

    reference = dl.ngram_overlap_scan(documents, items, ngram_size=REFERENCE_NGRAM)
    clean_dirty = [
        {
            "eval_id": report.eval_id,
            "total_items": report.total_items,
            "dirty_items": report.dirty_items,
            "clean_items": report.clean_items,
            "dirty_fraction": report.dirty_fraction,
            "assumed_clean_accuracy": report.clean_accuracy,
            "assumed_memorized_accuracy": report.memorized_accuracy,
            "reported_score": report.reported_score,
            "inflation_points": report.inflation * 100,
        }
        for report in dl.clean_dirty_partition(
            items,
            reference.reached_eval_items,
            clean_accuracy=ASSUMED_CLEAN_ACCURACY,
            memorized_accuracy=ASSUMED_MEMORIZED_ACCURACY,
        )
    ]

    payload = {
        "run_utc": datetime.now(UTC).isoformat(),
        "runner": "decontamination/run_measurement.py",
        "parameters": {
            "ngram_sizes": list(NGRAM_SIZES),
            "normalizers": list(NORMALIZERS),
            "reference_ngram_size": REFERENCE_NGRAM,
            "minhash": {"permutations": 128, "bands": 32, "rows": 4, "shingle_size": 5},
            "assumed_clean_accuracy": ASSUMED_CLEAN_ACCURACY,
            "assumed_memorized_accuracy": ASSUMED_MEMORIZED_ACCURACY,
            "token_estimator": "ceil(non-whitespace units x 1.35)",
        },
        "corpus": {
            "documents": len(documents),
            "characters": total_characters,
            "estimated_tokens": total_tokens,
        },
        "eval_suite": {
            "items": len(items),
            "by_eval": {
                eval_id: sum(1 for item in items if item.eval_id == eval_id)
                for eval_id in eval_ids
            },
        },
        "overlap": overlap,
        "precision_audit": precision,
        "indexability": indexability,
        "minhash": minhash,
        "clean_dirty": clean_dirty,
        "lsh_curve": {
            str(round(step / 10, 1)): dl.lsh_collision_probability(step / 10, 32, 4)
            for step in range(1, 10)
        },
        "matra_demonstration": {
            "source": "इन प्रदर्शनों में भाग लेने वालों पर कार्रवाई की गई।",
            "indic_safe": dl.normalize_indic_safe(
                "इन प्रदर्शनों में भाग लेने वालों पर कार्रवाई की गई।"
            ),
            "naive": dl.normalize_naive(
                "इन प्रदर्शनों में भाग लेने वालों पर कार्रवाई की गई।"
            ),
        },
    }

    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(
        f"  corpus {len(documents)} docs / {total_tokens:,} est tokens; "
        f"eval {len(items)} items"
    )
    for key in ("n13_indic_safe", "n13_naive"):
        entry = overlap[key]
        print(
            f"  {key}: {entry['flagged_documents']} flags "
            f"({entry['flag_rate'] * 100:.2f}%), reach {entry['reached_eval_items']}"
        )


if __name__ == "__main__":
    main()
