"""Measurement primitives for the Session 4 decontamination workbook.

This module holds the detector implementations and the metric derivations so
that both the notebook and the offline runner execute identical code. Nothing
here reaches the network; it operates on the caches built by
``data/build_decontamination_cache.py``.

Three detector rungs plus a quantification step:

* ``normalize_indic_safe`` / ``normalize_naive`` -- the normalization rung, and
  the naive variant it exists to refute.
* ``ngram_overlap_scan`` -- exact n-gram overlap, GPT-3's mechanism.
* ``minhash_lsh_scan`` -- banded MinHash near-duplicate detection.
* ``clean_dirty_partition`` -- the clean/dirty re-evaluation that converts a hit
  count into a benchmark score delta.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

# Unicode general categories for combining marks. Devanagari dependent vowel
# signs (matras) are Mn/Mc and the virama U+094D is Mn. Any normalizer that
# treats these as punctuation destroys the orthography.
COMBINING_CATEGORIES = frozenset({"Mn", "Mc"})

_WHITESPACE = re.compile(r"\s+")
_NAIVE_STRIP = re.compile(r"[^\w\s]+")


def normalize_indic_safe(text: str) -> str:
    """NFC, casefold, and strip punctuation by Unicode category.

    Characters in categories ``Mn``/``Mc`` are preserved explicitly, so
    Devanagari matras and the virama survive. Everything that is neither
    alphanumeric, whitespace, nor a combining mark collapses to a space.
    """
    folded = unicodedata.normalize("NFC", text).casefold()
    out: list[str] = []
    for character in folded:
        if (
            character.isalnum()
            or character.isspace()
            or unicodedata.category(character) in COMBINING_CATEGORIES
        ):
            out.append(character)
        else:
            out.append(" ")
    return _WHITESPACE.sub(" ", "".join(out)).strip()


def normalize_naive(text: str) -> str:
    """The English-shaped normalizer this workbook exists to refute.

    ``[^\\w\\s]`` looks correct because Python's ``\\w`` is Unicode-aware for
    letters. It is not aware of combining marks: every Devanagari matra and the
    virama match the negated class and are deleted, so unrelated Hindi words
    collapse onto the same skeleton.
    """
    lowered = unicodedata.normalize("NFC", text).lower()
    return _WHITESPACE.sub(" ", _NAIVE_STRIP.sub(" ", lowered)).strip()


def estimate_tokens(text: str) -> int:
    """Repository-wide fertility proxy: ceil(non-whitespace units x 1.35)."""
    units = len(_WHITESPACE.sub("", text))
    return -(-units * 135 // 100)


# ---------------------------------------------------------------------------
# shingling
# ---------------------------------------------------------------------------


def word_ngrams(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    if len(tokens) < size:
        return set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _hash64(value: bytes) -> int:
    return struct.unpack("<Q", hashlib.blake2b(value, digest_size=8).digest())[0]


def hashed_ngrams(tokens: list[str], size: int) -> set[int]:
    """Hashed n-grams. A 64-bit digest keeps the eval-side set memory-bounded."""
    return {_hash64(" ".join(gram).encode()) for gram in word_ngrams(tokens, size)}


# ---------------------------------------------------------------------------
# corpus / eval records
# ---------------------------------------------------------------------------


@dataclass
class Document:
    doc_id: str
    source_type: str | None
    stream_offset: int
    text: str
    safe_tokens: list[str] = field(repr=False, default_factory=list)
    naive_tokens: list[str] = field(repr=False, default_factory=list)

    @property
    def characters(self) -> int:
        return len(self.text)


@dataclass
class EvalItem:
    eval_id: str
    item_index: int
    primary_text: str
    texts: list[str]
    answers: list[str]
    safe_tokens: list[str] = field(repr=False, default_factory=list)
    naive_tokens: list[str] = field(repr=False, default_factory=list)

    @property
    def key(self) -> tuple[str, int]:
        return (self.eval_id, self.item_index)


def load_documents(path: Path, limit: int | None = None) -> list[Document]:
    documents: list[Document] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            document = Document(
                doc_id=row["doc_id"],
                source_type=row.get("source_type"),
                stream_offset=row["stream_offset"],
                text=row["text"],
            )
            document.safe_tokens = normalize_indic_safe(document.text).split()
            document.naive_tokens = normalize_naive(document.text).split()
            documents.append(document)
            if limit and len(documents) >= limit:
                break
    return documents


def load_eval_items(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            item = EvalItem(
                eval_id=row["eval_id"],
                item_index=row["item_index"],
                primary_text=row["primary_text"],
                texts=row["texts"],
                answers=row.get("answers", []),
            )
            item.safe_tokens = normalize_indic_safe(item.primary_text).split()
            item.naive_tokens = normalize_naive(item.primary_text).split()
            items.append(item)
    return items


def _eval_tokens(item: EvalItem, normalizer: str) -> list[str]:
    return item.safe_tokens if normalizer == "indic_safe" else item.naive_tokens


# ---------------------------------------------------------------------------
# rung 2: exact n-gram overlap
# ---------------------------------------------------------------------------


@dataclass
class OverlapResult:
    ngram_size: int
    normalizer: str
    eval_ngrams: int
    eval_items_indexable: int
    eval_items_total: int
    flagged_documents: list[dict]
    documents_scanned: int
    reached_eval_items: dict[str, set[int]] = field(default_factory=dict, repr=False)

    @property
    def flagged_count(self) -> int:
        return len(self.flagged_documents)

    @property
    def flag_rate(self) -> float:
        return self.flagged_count / self.documents_scanned if self.documents_scanned else 0.0


def build_eval_index(
    items: list[EvalItem], ngram_size: int, normalizer: str = "indic_safe"
) -> tuple[dict[int, list[tuple[str, int]]], int]:
    """Map each eval n-gram hash to the eval items that contain it.

    The postings list is what turns a corpus flag into a statement about *which*
    evaluation items are reachable, which is the number that bounds benchmark
    trust. ``normalizer`` must match the one used on the corpus side: a
    contamination scan compares two normalized surfaces, so applying different
    rules to each side measures nothing.
    """
    index: dict[int, list[tuple[str, int]]] = defaultdict(list)
    indexable = 0
    for item in items:
        tokens = _eval_tokens(item, normalizer)
        grams = hashed_ngrams(tokens, ngram_size)
        if not grams:
            continue
        indexable += 1
        for gram in grams:
            index[gram].append(item.key)
    return index, indexable


def ngram_overlap_scan(
    documents: list[Document],
    items: list[EvalItem],
    ngram_size: int = 13,
    normalizer: str = "indic_safe",
) -> OverlapResult:
    index, indexable = build_eval_index(items, ngram_size, normalizer)
    flagged: list[dict] = []
    reached: dict[str, set[int]] = defaultdict(set)

    for document in documents:
        tokens = (
            document.safe_tokens if normalizer == "indic_safe" else document.naive_tokens
        )
        grams = word_ngrams(tokens, ngram_size)
        if not grams:
            continue
        hits: list[tuple[str, ...]] = []
        touched: set[tuple[str, int]] = set()
        for gram in grams:
            postings = index.get(_hash64(" ".join(gram).encode()))
            if postings:
                hits.append(gram)
                touched.update(postings)
        if hits:
            for eval_id, item_index in touched:
                reached[eval_id].add(item_index)
            flagged.append(
                {
                    "doc_id": document.doc_id,
                    "stream_offset": document.stream_offset,
                    "source_type": document.source_type,
                    "document_ngrams": len(grams),
                    "shared_ngrams": len(hits),
                    "shared_fraction": len(hits) / len(grams),
                    "eval_items_touched": sorted(touched),
                    "longest_shared": max((" ".join(gram) for gram in hits), key=len),
                }
            )

    return OverlapResult(
        ngram_size=ngram_size,
        normalizer=normalizer,
        eval_ngrams=len(index),
        eval_items_indexable=indexable,
        eval_items_total=len(items),
        flagged_documents=sorted(flagged, key=lambda row: -row["shared_ngrams"]),
        documents_scanned=len(documents),
        reached_eval_items={key: set(value) for key, value in reached.items()},
    )


# ---------------------------------------------------------------------------
# rung 3: banded MinHash / LSH
# ---------------------------------------------------------------------------

_MERSENNE = (1 << 61) - 1


def _permutations(count: int, seed: int = 0x5EED) -> list[tuple[int, int]]:
    """Deterministic (a, b) coefficient pairs for the universal hash family."""
    state = hashlib.blake2b(struct.pack("<Q", seed), digest_size=8)
    pairs: list[tuple[int, int]] = []
    for index in range(count):
        digest = hashlib.blake2b(
            state.digest() + struct.pack("<I", index), digest_size=16
        ).digest()
        multiplier = (int.from_bytes(digest[:8], "little") % (_MERSENNE - 1)) + 1
        offset = int.from_bytes(digest[8:], "little") % _MERSENNE
        pairs.append((multiplier, offset))
    return pairs


def minhash_signature(
    shingles: set[int], permutations: list[tuple[int, int]]
) -> tuple[int, ...]:
    if not shingles:
        return ()
    signature = []
    for multiplier, offset in permutations:
        signature.append(
            min(((multiplier * shingle + offset) % _MERSENNE) for shingle in shingles)
        )
    return tuple(signature)


@dataclass
class MinHashResult:
    permutations: int
    bands: int
    rows: int
    shingle_size: int
    threshold: float
    candidate_pairs: int
    confirmed_pairs: list[dict]
    documents_scanned: int
    eval_items_indexed: int
    window_tokens: int | None = None

    @property
    def confirmed_count(self) -> int:
        return len(self.confirmed_pairs)


def minhash_lsh_scan(
    documents: list[Document],
    items: list[EvalItem],
    permutations: int = 128,
    bands: int = 32,
    shingle_size: int = 5,
    threshold: float = 0.5,
    window_tokens: int | None = None,
    window_stride: int | None = None,
) -> MinHashResult:
    """Banded MinHash between corpus text and evaluation items.

    Corpus text is compared against evaluation items rather than against other
    corpus text: the target is a corpus/held-out relation, not intra-corpus
    redundancy.

    ``window_tokens`` controls the comparison unit and is the parameter that
    decides whether this technique can work at all. With ``None`` the unit is
    the whole document, which is what an off-the-shelf near-duplicate pass
    does; a 40-token evaluation item inside a 3,000-token page then has a
    Jaccard of roughly 0.013 against it, far below any usable threshold, so the
    scan reports nothing. Setting ``window_tokens`` slides a window of that
    length over each document so the unit sizes are comparable.
    """
    if permutations % bands:
        raise ValueError("permutations must be divisible by bands")
    rows = permutations // bands
    coefficients = _permutations(permutations)
    stride = window_stride or (window_tokens // 2 if window_tokens else None)

    buckets: dict[tuple[int, tuple[int, ...]], list[tuple[str, int]]] = defaultdict(list)
    eval_shingles: dict[tuple[str, int], set[int]] = {}
    indexed = 0
    for item in items:
        shingles = hashed_ngrams(item.safe_tokens, shingle_size)
        if not shingles:
            continue
        signature = minhash_signature(shingles, coefficients)
        eval_shingles[item.key] = shingles
        indexed += 1
        for band in range(bands):
            window = signature[band * rows : (band + 1) * rows]
            buckets[(band, window)].append(item.key)

    candidates = 0
    best: dict[tuple[str, str, int], dict] = {}
    for document in documents:
        if window_tokens:
            spans = [
                document.safe_tokens[start : start + window_tokens]
                for start in range(
                    0, max(1, len(document.safe_tokens) - shingle_size + 1), stride
                )
            ]
        else:
            spans = [document.safe_tokens]

        for span in spans:
            shingles = hashed_ngrams(span, shingle_size)
            if not shingles:
                continue
            signature = minhash_signature(shingles, coefficients)
            proposed: set[tuple[str, int]] = set()
            for band in range(bands):
                window = signature[band * rows : (band + 1) * rows]
                proposed.update(buckets.get((band, window), ()))
            candidates += len(proposed)
            for key in proposed:
                other = eval_shingles[key]
                union = len(shingles | other)
                if not union:
                    continue
                jaccard = len(shingles & other) / union
                if jaccard < threshold:
                    continue
                slot = (document.doc_id, key[0], key[1])
                record = {
                    "doc_id": document.doc_id,
                    "stream_offset": document.stream_offset,
                    "eval_id": key[0],
                    "item_index": key[1],
                    "jaccard": jaccard,
                    "eval_containment": len(shingles & other) / len(other),
                }
                if slot not in best or jaccard > best[slot]["jaccard"]:
                    best[slot] = record

    return MinHashResult(
        permutations=permutations,
        bands=bands,
        rows=rows,
        shingle_size=shingle_size,
        threshold=threshold,
        candidate_pairs=candidates,
        confirmed_pairs=sorted(best.values(), key=lambda row: -row["jaccard"]),
        documents_scanned=len(documents),
        eval_items_indexed=indexed,
        window_tokens=window_tokens,
    )


def lsh_collision_probability(jaccard: float, bands: int, rows: int) -> float:
    """1 - (1 - s**r)**b -- the S-curve that banding buys."""
    return 1.0 - (1.0 - jaccard**rows) ** bands


# ---------------------------------------------------------------------------
# quantification: clean / dirty re-evaluation
# ---------------------------------------------------------------------------


@dataclass
class CleanDirtyReport:
    eval_id: str
    total_items: int
    dirty_items: int
    clean_items: int
    clean_accuracy: float
    memorized_accuracy: float

    @property
    def dirty_fraction(self) -> float:
        return self.dirty_items / self.total_items if self.total_items else 0.0

    @property
    def reported_score(self) -> float:
        """Score a memorizing model would report over the whole eval set."""
        return (
            self.clean_accuracy * self.clean_items
            + self.memorized_accuracy * self.dirty_items
        ) / self.total_items if self.total_items else 0.0

    @property
    def inflation(self) -> float:
        return self.reported_score - self.clean_accuracy


def clean_dirty_partition(
    items: list[EvalItem],
    reached: dict[str, set[int]],
    clean_accuracy: float,
    memorized_accuracy: float = 1.0,
) -> list[CleanDirtyReport]:
    """Partition each eval set into reachable and unreachable items.

    ``clean_accuracy`` is a stated assumption about true ability and
    ``memorized_accuracy`` a stated assumption about what memorization buys.
    Neither is measured here; the output is the arithmetic that follows from
    them, which is exactly what GPT-3's clean/dirty analysis reports.
    """
    grouped: dict[str, int] = defaultdict(int)
    for item in items:
        grouped[item.eval_id] += 1

    reports: list[CleanDirtyReport] = []
    for eval_id, total in sorted(grouped.items()):
        dirty = len(reached.get(eval_id, ()))
        reports.append(
            CleanDirtyReport(
                eval_id=eval_id,
                total_items=total,
                dirty_items=dirty,
                clean_items=total - dirty,
                clean_accuracy=clean_accuracy,
                memorized_accuracy=memorized_accuracy,
            )
        )
    return reports
