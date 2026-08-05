#!/usr/bin/env python3
"""Recompute every PII-scrubbing metric the notebook reports, from disk only.

Run with the project venv, from the repository root:

    .venv/bin/python pii-scrubbing/evaluate_cascade.py

Nothing here depends on interactive state: the probe cache and the hand-labelled
gold set are read from `data/`, and every number the notebook shows is written
back to `data/metrics.json`. That is what makes the notebook's tables auditable
rather than transcribed.

Two measurement rules that the numbers depend on, stated once:

* **Overlap dedupe.** Different layers can propose overlapping spans for the same
  identifier. Counting both would inflate the denominator, so gold spans and
  predictions are both reduced to non-overlapping representatives before scoring.
* **Candidate recall, not corpus recall.** The gold set was built by labelling
  every span that the union of the cascade and a faithful Dolma port proposed.
  PII that neither detector proposes is invisible to this metric, so recall here
  is an upper bound on a detector's true recall and is reported as such.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pii_techniques as T  # noqa: E402

DATA = HERE / "data"
CONFIGS = ("verified", "unverified")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_corpus() -> dict[str, list[dict]]:
    corpus = {}
    for config in CONFIGS:
        path = DATA / f"sangraha_hin_{config}_probe.jsonl"
        corpus[config] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    return corpus


def load_gold() -> tuple[set[tuple], list[dict]]:
    rows = [
        json.loads(line)
        for line in (DATA / "gold_labelled.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = {(r["cfg"], r["doc"], r["start"], r["end"]) for r in rows if r["gold"]}
    return keys, rows


# ---------------------------------------------------------------------------
# Overlap dedupe
# ---------------------------------------------------------------------------
def dedupe(keys: set[tuple]) -> list[tuple]:
    """Keep one representative per overlapping cluster, leftmost-longest first."""
    ordered = sorted(keys, key=lambda k: (k[0], k[1], k[2], -(k[3] - k[2])))
    kept: list[tuple] = []
    for key in ordered:
        config, doc, start, end = key
        if any(
            config == k[0] and doc == k[1] and start < k[3] and k[2] < end for k in kept
        ):
            continue
        kept.append(key)
    return kept


def score(predicted: set[tuple], gold: list[tuple]) -> dict:
    """Score predictions against gold, matching on span overlap within a document."""
    gold_set = set(gold)
    matched_gold, true_positive_keys, false_positive_keys = set(), [], []
    for key in dedupe(predicted):
        config, doc, start, end = key
        hit = next(
            (
                g
                for g in gold_set
                if g[0] == config and g[1] == doc and start < g[3] and g[2] < end
            ),
            None,
        )
        if hit is None:
            false_positive_keys.append(key)
        else:
            matched_gold.add(hit)
            true_positive_keys.append(key)

    tp, fp = len(matched_gold), len(false_positive_keys)
    fn_keys = [g for g in gold_set if g not in matched_gold]
    fn = len(fn_keys)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "P": round(precision, 4),
        "R": round(recall, 4),
        "F1": round(f1, 4),
        "fp_keys": false_positive_keys,
        "fn_keys": fn_keys,
    }


# ---------------------------------------------------------------------------
# Cascade variants
# ---------------------------------------------------------------------------
def run_cascade(
    corpus: dict[str, list[dict]],
    *,
    fold: bool = True,
    checksums: bool = True,
    context: bool = True,
    allowlists: bool = True,
    threshold: float = 0.50,
    drop_handles: bool = True,
) -> set[tuple]:
    """Run the cascade with individual layers switched off, for the yield ladder."""
    predicted: set[tuple] = set()
    for config in CONFIGS:
        for row in corpus[config]:
            text = row["text"]
            folded = T.fold_digits(text) if fold else text
            spans = T.detect_patterns(folded)
            if checksums:
                spans = T.apply_checksums(spans, folded)
            if context:
                spans = T.apply_context(spans, folded)
            if allowlists:
                spans = T.apply_allowlists(spans, folded, text)
            spans = [s for s in spans if s.decision != "suppressed" and s.score >= threshold]
            for span in T.resolve_overlaps(spans):
                if drop_handles and span.category == "handle":
                    continue
                predicted.add((config, row["doc_id"], span.start, span.end))
    return predicted


# ---------------------------------------------------------------------------
# Dolma baseline
# ---------------------------------------------------------------------------
# Faithful port of Dolma's `FastPiiRegex` taggers so the comparison is against
# what Dolma actually ran, not a strawman:
#   allenai/dolma, python/dolma/taggers/pii.py
DOLMA_EMAIL = re.compile(r"[.\s@,?!;:)(]*([^\s@]+)@([^\s@,?!;:)(]+?)([.]+[^\s@,?!;:)(]+)")
DOLMA_PHONE = re.compile(r"\s*(?:[+0-9]{1,3})?[\-. (]*(?:[0-9]{3})[\-. )]*(?:[0-9]{3})[\-. ]*(?:[0-9]{4})(?: *x[0-9]+)?\s*")
DOLMA_IP = re.compile(r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)")


def dolma_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    for match in DOLMA_EMAIL.finditer(text):
        # Dolma's own offset correction for the leading punctuation class.
        offset = len(match.group(0)) - len(match.group(0).lstrip(".\t\n\r ,?!;:)("))
        spans.append((match.start() + offset, match.end(), "dolma_email_address"))
    for match in DOLMA_PHONE.finditer(text):
        raw = match.group(0)
        offset = len(raw) - len(raw.lstrip())
        spans.append((match.start() + offset, match.start() + len(raw.rstrip()), "dolma_phone_number"))
    for match in DOLMA_IP.finditer(text):
        spans.append((match.start(), match.end(), "dolma_ip_address"))
    return spans


def run_dolma(corpus: dict[str, list[dict]]) -> set[tuple]:
    predicted: set[tuple] = set()
    for config in CONFIGS:
        for row in corpus[config]:
            for start, end, _ in dolma_spans(row["text"]):
                predicted.add((config, row["doc_id"], start, end))
    return predicted


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------
def density(corpus: dict[str, list[dict]]) -> dict:
    out = {}
    for config in CONFIGS:
        characters = spans_total = touched = 0
        by_category: Counter[str] = Counter()
        docs_by_category: Counter[str] = Counter()
        for row in corpus[config]:
            characters += len(row["text"])
            result = T.scrub(row["text"])
            accepted = [s for s in result.accepted if s.category != "handle"]
            if accepted:
                touched += 1
                for category in {s.category for s in accepted}:
                    docs_by_category[category] += 1
            for span in accepted:
                by_category[span.category] += 1
                spans_total += 1
        tokens = characters / 3.79  # measured chars/token for Hindi in this cache
        out[config] = {
            "documents": len(corpus[config]),
            "characters": characters,
            "estimated_tokens": round(tokens),
            "documents_touched": touched,
            "spans_total": spans_total,
            "spans_by_category": dict(by_category.most_common()),
            "documents_by_category": dict(docs_by_category.most_common()),
            "spans_per_million_tokens": round(spans_total / (tokens / 1_000_000), 2),
            "documents_touched_pct": round(100 * touched / len(corpus[config]), 3),
        }
    return out


def main() -> None:
    corpus = load_corpus()
    gold_keys, gold_rows = load_gold()
    gold = dedupe(gold_keys)

    variants = {
        "A. Dolma FastPiiRegex (faithful port)": run_dolma(corpus),
        "B. cascade, ASCII patterns only (no digit folding)": run_cascade(corpus, fold=False),
        "C. cascade, all layers, handles included": run_cascade(corpus, drop_handles=False),
        "D. cascade, handles excluded (shipped)": run_cascade(corpus),
    }
    ladder = {
        "1. patterns only": run_cascade(corpus, checksums=False, context=False, allowlists=False),
        "2. + checksum gates": run_cascade(corpus, context=False, allowlists=False),
        "3. + context scoring": run_cascade(corpus, allowlists=False),
        "4. + allowlist suppression": run_cascade(corpus),
    }

    scored_variants = {name: score(keys, gold) for name, keys in variants.items()}
    scored_ladder = {name: score(keys, gold) for name, keys in ladder.items()}

    # Attribute every false positive of the shipped config to a labelled cause.
    reason_by_key = {
        (r["cfg"], r["doc"], r["start"], r["end"]): r.get("gold_note", "unlabelled")
        for r in gold_rows
    }
    shipped = scored_variants["D. cascade, handles excluded (shipped)"]
    fp_causes = Counter(reason_by_key.get(k, "unlabelled") for k in shipped["fp_keys"])

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gold_set": {
            "candidates_reviewed": len(gold_rows),
            "pii_spans_raw": len(gold_keys),
            "pii_spans_overlap_deduped": len(gold),
            "labelling_rule": (
                "A span is PII when it identifies a private individual. Published "
                "institutional contact points, documentation examples and non-identifier "
                "numerics are labelled not-PII."
            ),
            "recall_caveat": (
                "Recall is measured against the union of candidates proposed by the cascade "
                "and a faithful Dolma FastPiiRegex port. It is a candidate-recall bound, not "
                "corpus truth: PII that neither detector proposes is invisible to this metric."
            ),
        },
        "variants": {
            name: {k: v for k, v in result.items() if not k.endswith("_keys")}
            for name, result in scored_variants.items()
        },
        "layer_ladder": {
            name: {
                **{k: v for k, v in result.items() if not k.endswith("_keys")},
                "spans_kept": len(dedupe(ladder[name])),
            }
            for name, result in scored_ladder.items()
        },
        "shipped_false_positive_causes": dict(fp_causes.most_common()),
        "shipped_missed_spans": [
            {
                "config": k[0],
                "doc_id": k[1],
                "reason": "identifier neither layer proposes at threshold 0.50",
            }
            for k in shipped["fn_keys"]
        ],
        "density": density(corpus),
    }

    (DATA / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"gold: {len(gold_keys)} raw -> {len(gold)} overlap-deduped "
          f"(from {len(gold_rows)} reviewed candidates)\n")
    header = f"{'variant':52} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7}"
    print(header)
    print("-" * len(header))
    for name, r in scored_variants.items():
        print(f"{name:52} {r['tp']:4} {r['fp']:4} {r['fn']:4} {r['P']:7.3f} {r['R']:7.3f} {r['F1']:7.3f}")
    print()
    print(f"{'layer ladder':52} {'kept':>4} {'TP':>4} {'FP':>4} {'P':>7} {'R':>7}")
    print("-" * len(header))
    for name, r in scored_ladder.items():
        kept = len(dedupe(ladder[name]))
        print(f"{name:52} {kept:4} {r['tp']:4} {r['fp']:4} {r['P']:7.3f} {r['R']:7.3f}")
    print("\nshipped false-positive causes:")
    for cause, count in fp_causes.most_common():
        print(f"  {count:4}  {cause}")
    print("\nwrote", DATA / "metrics.json")


if __name__ == "__main__":
    main()
