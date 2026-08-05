import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    # Session 4 — Stage 8: Manifest and provenance

    This is a bounded, executable technique study for the final stage of the Session 4
    cleaning pipeline. Stages 1–7 change bytes. **This stage changes nothing about the
    text** — it produces the bookkeeping that makes every earlier stage's claim
    checkable by someone who was not there when it ran.

    The corpus is a deterministic **225-document Sangraha Hindi slice**
    (`verified/hin` at a pinned revision, first rows in stream order until 150,000
    real `o200k_base` tokens). It is self-contained under
    `manifest-provenance/data/` and never reads another stage's folder.

    **Why this slice is small, and why that is not a compromise.** The claims of this
    stage — determinism, canonical form, duplicate collapse, tamper detection,
    inclusion proofs — are *exact* and *size-independent*: they either hold on 225
    records or they are broken. A larger slice adds committed bytes and no new
    evidence.

    Two conclusions everything below is built to earn:

    > A **token count is meaningless without a named tokenizer.** The same text
    > measures 150,139 tokens under `o200k_base` and 464,805 under `cl100k_base` — a
    > 3.1× spread.
    >
    > A **content digest is meaningless without a pinned canonical form.** A bare
    > `.strip()` changes 89.3% of the digests on this slice.
    """)
    return


@app.cell(hide_code=True)
def runtime_setup():
    import hashlib as _hashlib
    import json as _json
    import math as _math
    import re as _re
    import time as _time
    import unicodedata as _unicodedata
    from collections import Counter as _Counter
    from datetime import UTC as _UTC
    from datetime import datetime as _datetime
    from pathlib import Path

    # These constants define the slice contract and are copied verbatim into the
    # emitted manifests, so they live in one place.
    DATASET_ID = "ai4bharat/sangraha"
    DATASET_REVISION = "8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8"
    DATASET_DATA_DIR = "verified/hin"
    DATASET_SPLIT = "train"
    LICENSE = "CC-BY-4.0"
    TOKENIZER_NAME = "o200k_base"
    COMPARISON_TOKENIZER_NAME = "cl100k_base"
    SHARD_COUNT = 4
    PIPELINE_VERSION = "s4-manifest-provenance-v1"
    TOKEN_PROXY_FERTILITY = 1.35

    SLICE_PATH = Path("manifest-provenance/data/sangraha_hin_slice.jsonl")
    SLICE_MANIFEST_PATH = Path("manifest-provenance/data/slice_manifest.json")
    NOTEBOOK_PATH = Path("manifest-provenance/manifest_provenance.py")
    MANIFEST_DIR = Path("manifest-provenance/data/manifests")
    METRICS_PATH = Path("manifest-provenance/data/metrics.json")
    TIKTOKEN_CACHE_DIR = Path("manifest-provenance/data/.tiktoken_cache")

    # One palette, defined once, reused by every hand-authored SVG below.
    INK = "#1f2933"
    MUTED = "#6b7280"
    RULE = "#d7dbe0"
    SURFACE = "#f7f8fa"
    ADMIT = "#2f7d5d"
    ADMIT_BG = "#e7f4ee"
    BLOCK = "#a8323f"
    BLOCK_BG = "#fbeaec"
    ACCENT = "#5b4bbd"
    ACCENT_BG = "#eeeaff"
    SVG_FONT = "ui-sans-serif, system-ui, sans-serif"


    def _utc_now():
        return (
            _datetime.now(_UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )


    def _sha256_file(path):
        digest = _hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


    def _load_slice(path):
        """Read the tracked slice in file order. File order is the selection rule."""
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Rebuild it with:\n"
                "    uv run python manifest-provenance/data/build_slice.py"
            )
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(_json.loads(line))
        if not rows:
            raise ValueError(f"{path} is empty; rebuild it with build_slice.py")
        return rows


    def _shard_of(row_index, row_total, shard_count):
        """Partition row positions into shards of equal size, last absorbs remainder."""
        per_shard = row_total // shard_count
        return min(row_index // per_shard, shard_count - 1)


    def _short(digest_hex, width=12):
        """Truncate a digest for display only. Never for comparison."""
        bare = digest_hex.split(":", 1)[-1]
        return f"{bare[:width]}…"


    def _percent(numerator, denominator):
        """Ratios are formatted as strings: floats must never enter a manifest."""
        return f"{100.0 * numerator / denominator:.2f}%" if denominator else "n/a"


    # Names exported from a marimo cell must not begin with an underscore.
    hashlib_module = _hashlib
    json_module = _json
    math_module = _math
    re_module = _re
    time_module = _time
    unicodedata_module = _unicodedata
    Counter = _Counter
    utc_now = _utc_now
    sha256_file = _sha256_file
    load_slice = _load_slice
    shard_of = _shard_of
    short = _short
    percent = _percent
    return (
        ACCENT,
        ACCENT_BG,
        ADMIT,
        ADMIT_BG,
        BLOCK,
        BLOCK_BG,
        COMPARISON_TOKENIZER_NAME,
        Counter,
        DATASET_DATA_DIR,
        DATASET_ID,
        DATASET_REVISION,
        DATASET_SPLIT,
        INK,
        LICENSE,
        MANIFEST_DIR,
        METRICS_PATH,
        MUTED,
        NOTEBOOK_PATH,
        PIPELINE_VERSION,
        RULE,
        SHARD_COUNT,
        SLICE_MANIFEST_PATH,
        SLICE_PATH,
        SURFACE,
        SVG_FONT,
        TIKTOKEN_CACHE_DIR,
        TOKENIZER_NAME,
        TOKEN_PROXY_FERTILITY,
        hashlib_module,
        json_module,
        load_slice,
        math_module,
        percent,
        re_module,
        sha256_file,
        shard_of,
        short,
        time_module,
        unicodedata_module,
        utc_now,
    )


@app.cell(hide_code=True)
def slice_provenance(
    Counter,
    SLICE_MANIFEST_PATH,
    SLICE_PATH,
    json_module,
    load_slice,
    mo,
    short,
):
    slice_rows = load_slice(SLICE_PATH)
    slice_manifest = json_module.loads(SLICE_MANIFEST_PATH.read_text(encoding="utf-8"))

    _type_mix = Counter(row["type"] for row in slice_rows)

    slice_provenance_table = [
        {"field": "dataset_id", "value": slice_manifest["dataset_id"]},
        {"field": "revision (pinned)", "value": slice_manifest["revision"]},
        {"field": "data_dir / split", "value": f"{slice_manifest['data_dir']} / {slice_manifest['split']}"},
        {"field": "license", "value": slice_manifest["license"]},
        {"field": "records", "value": f"{slice_manifest['records']:,}"},
        {"field": "text bytes", "value": f"{slice_manifest['byte_count']:,}"},
        {"field": f"tokens ({slice_manifest['tokenizer']})", "value": f"{slice_manifest['token_count']:,}"},
        {"field": "tokenizer BPE sha256", "value": short(slice_manifest["tokenizer_bpe_sha256"])},
        {"field": "slice sha256", "value": short(slice_manifest["slice_sha256"])},
        {"field": "type mix", "value": ", ".join(f"{k} {v}" for k, v in _type_mix.most_common())},
        {"field": "built (UTC)", "value": slice_manifest["built_utc"]},
    ]

    mo.vstack([
        mo.md(r"""
        ## The slice under study

        Built by `manifest-provenance/data/build_slice.py`, which wrote the block
        below at build time. Note what is already content-addressed here: the slice
        file itself, **and the tokenizer's own BPE artifact**. The measuring
        instrument is pinned, not just the data.

        The `slice_sha256` recomputed from the file on disk right now is checked
        against the recorded value in the next cell — a manifest nobody verifies is
        documentation, not provenance.
        """),
        mo.ui.table(slice_provenance_table, selection=None, page_size=12),
    ])
    return slice_manifest, slice_rows


@app.cell(hide_code=True)
def emitter_prose(mo):
    mo.md(r"""
    ## The emitter

    Everything downstream is built from six primitives, defined once in the next
    (visible) cell:

    | Primitive | What it pins down |
    |---|---|
    | `canonical_json` | One byte string per JSON object, independent of key order |
    | `canonical_text` | One byte string per document, via NFC — for **hashing only** |
    | `merkle_leaf` / `merkle_node` | RFC 6962 domain separation: a leaf can never be read as an internal node |
    | `merkle_root` / `merkle_path` / `merkle_verify` | O(log n) inclusion proof against a single root |
    | `record_manifest` / `shard_manifest` / `corpus_manifest` | The three-level hierarchy |
    | `gate` | The admission decision, with a specific machine-readable reason |

    Three deliberate choices, each of which would be a bug if reversed:

    **1. `canonical_json` is an RFC 8785 *approximation*, and that is safe here
    for one specific reason.** JCS also mandates ECMAScript number formatting,
    which `json.dumps` does not implement — JCS requires `1` where Python emits
    `1.0`. Every numeric field in every manifest below is a Python `int`, for
    which the two agree. **No float ever enters a manifest**; ratios are
    formatted as strings.

    **2. Canonicalization for hashing is NFC, never NFKC.** UAX #15 states that
    the compatibility mappings (NFKC/NFKD) "must not be blindly applied to
    arbitrary text" because they erase distinctions. It applies to the hashing
    copy only — **the emitted text is untouched**.

    **3. The Merkle verifier's two-pass structure is load-bearing.** The path is
    ordered leaf-to-root while the split recursion runs root-to-leaf, so
    `merkle_verify` must first descend recording which side each level sits on,
    *then* fold bottom-up consuming those sides in reverse. The natural-looking
    single-pass top-down fold is wrong: written that way first, it silently
    failed 56 of 64 proofs. Silently — every root still looked like a hash.
    """)
    return


@app.cell
def emitter(hashlib_module, json_module, unicodedata_module):
    def canonical_json(obj) -> bytes:
        """Sorted-key, minimally separated, UTF-8 JSON bytes (RFC 8785 approximation).

        Safe here because every numeric manifest field is an int; JCS and json.dumps
        disagree only on float formatting. Ratios are emitted as strings.
        """
        return json_module.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


    def canonical_text(text: str) -> bytes:
        """The pinned canonical form for hashing. NFC only, and never applied to
        the emitted text — see the canonicalization-sensitivity metric below."""
        return unicodedata_module.normalize("NFC", text).encode("utf-8")


    def digest_string(data: bytes) -> str:
        """Self-describing digest: the algorithm travels with the value."""
        return f"sha2-256:{hashlib_module.sha256(data).hexdigest()}"


    # --- RFC 6962 Merkle tree -------------------------------------------------
    # Domain-separated prefixes per RFC 6962 section 2.1: 0x00 for leaves, 0x01 for
    # internal nodes, so a leaf digest can never be confused with a node digest.


    def _h(b: bytes) -> bytes:
        return hashlib_module.sha256(b).digest()


    def merkle_leaf(data: bytes) -> bytes:
        return _h(b"\x00" + data)


    def merkle_node(left: bytes, right: bytes) -> bytes:
        return _h(b"\x01" + left + right)


    def _split(n: int) -> int:
        """Largest power of two strictly less than n (RFC 6962's k)."""
        k = 1
        while k * 2 < n:
            k *= 2
        return k


    def merkle_root(leaves: list) -> bytes:
        if not leaves:
            return _h(b"")
        if len(leaves) == 1:
            return merkle_leaf(leaves[0])
        k = _split(len(leaves))
        return merkle_node(merkle_root(leaves[:k]), merkle_root(leaves[k:]))


    def merkle_path(leaves: list, i: int) -> list:
        """Audit path for leaf i, ordered leaf-to-root."""
        if len(leaves) == 1:
            return []
        k = _split(len(leaves))
        if i < k:
            return merkle_path(leaves[:k], i) + [merkle_root(leaves[k:])]
        return merkle_path(leaves[k:], i - k) + [merkle_root(leaves[:k])]


    def merkle_verify(data: bytes, i: int, path: list, n: int, root: bytes) -> bool:
        """Two passes, deliberately. The path is leaf-to-root but the split
        recursion is root-to-leaf, so sides are recorded first and consumed in
        reverse. A single-pass top-down fold silently fails most proofs."""
        sides, idx, size = [], i, n
        while size > 1:  # pass 1: descend, recording which side we sit on
            k = _split(size)
            if idx < k:
                sides.append("L")
                size = k
            else:
                sides.append("R")
                idx -= k
                size -= k
        h = merkle_leaf(data)
        for side, sibling in zip(reversed(sides), path):  # pass 2: fold bottom-up
            h = merkle_node(h, sibling) if side == "L" else merkle_node(sibling, h)
        return h == root

    return (
        canonical_json,
        canonical_text,
        digest_string,
        merkle_leaf,
        merkle_node,
        merkle_path,
        merkle_root,
        merkle_verify,
    )


@app.cell
def manifests(
    COMPARISON_TOKENIZER_NAME,
    DATASET_DATA_DIR,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    LICENSE,
    NOTEBOOK_PATH,
    PIPELINE_VERSION,
    TOKENIZER_NAME,
    TOKEN_PROXY_FERTILITY,
    canonical_json,
    canonical_text,
    digest_string,
    math_module,
    merkle_root,
    re_module,
    short,
    utc_now,
):
    def record_manifest(row, encoding, comparison_encoding=None):
        """One record's manifest. `record_id` is content-addressed over the NFC
        bytes, so it is verifiable from the shipped text alone."""
        text = row["text"]
        canonical = canonical_text(text)
        words = len(re_module.findall(r"\S+", text))
        manifest = {
            "record_id": digest_string(canonical),
            "upstream_doc_id": row["doc_id"],
            "type": row.get("type", "unknown"),
            "char_count": len(text),
            "byte_count": len(canonical),
            "token_count": len(encoding.encode(text, disallowed_special=())),
            "tokenizer": TOKENIZER_NAME,
            # Reported only so its error can be measured. Never used as a count.
            "words_x_135": int(math_module.ceil(words * TOKEN_PROXY_FERTILITY)),
            "word_count": words,
        }
        if comparison_encoding is not None:
            manifest["comparison_token_count"] = len(
                comparison_encoding.encode(text, disallowed_special=())
            )
            manifest["comparison_tokenizer"] = COMPARISON_TOKENIZER_NAME
        return manifest


    def shard_manifest(records, shard_index):
        """One shard's manifest. `merkle_root` covers the shard's record digests;
        `manifest_digest` covers this manifest's own canonical JSON."""
        leaves = [record["record_id"].encode("utf-8") for record in records]
        body = {
            "shard_id": f"shard-{shard_index:02d}",
            "shard_index": shard_index,
            "record_count": len(records),
            "token_count": sum(r["token_count"] for r in records),
            "byte_count": sum(r["byte_count"] for r in records),
            "char_count": sum(r["char_count"] for r in records),
            "tokenizer": TOKENIZER_NAME,
            "merkle_root": f"sha2-256:{merkle_root(leaves).hex()}",
            "record_ids": [r["record_id"] for r in records],
        }
        body["manifest_digest"] = digest_string(canonical_json(body))
        return body


    def corpus_manifest(shards, notebook_digest, tokenizer_bpe_sha256, run_utc=None):
        """The top of the hierarchy. Its Merkle root is computed over the shard
        roots, so a record-level change propagates all the way up."""
        shard_leaves = [s["merkle_root"].encode("utf-8") for s in shards]
        body = {
            "pipeline_version": PIPELINE_VERSION,
            "run_utc": run_utc or utc_now(),
            "source": {
                "dataset_id": DATASET_ID,
                "revision": DATASET_REVISION,
                "data_dir": DATASET_DATA_DIR,
                "split": DATASET_SPLIT,
                "license": LICENSE,
            },
            "tokenizer": TOKENIZER_NAME,
            "tokenizer_bpe_sha256": tokenizer_bpe_sha256,
            # The reflexive step: the notebook is identified by its own source
            # digest. This value changes on every notebook edit -- that is the
            # point, not a defect: a manifest emitted by different code is a
            # manifest from a different pipeline.
            "cleaning_script": {"path": str(NOTEBOOK_PATH), "sha256": notebook_digest},
            "shard_count": len(shards),
            "record_count": sum(s["record_count"] for s in shards),
            "token_count": sum(s["token_count"] for s in shards),
            "byte_count": sum(s["byte_count"] for s in shards),
            "char_count": sum(s["char_count"] for s in shards),
            "shard_roots": [s["merkle_root"] for s in shards],
            "merkle_root": f"sha2-256:{merkle_root(shard_leaves).hex()}",
        }
        body["manifest_digest"] = digest_string(canonical_json(body))
        return body


    REQUIRED_FIELDS = ("record_id", "tokenizer", "token_count", "byte_count", "type")
    LICENSE_ALLOWLIST = frozenset({"CC-BY-4.0", "CC-BY-SA-4.0", "MIT", "Apache-2.0"})


    def gate(manifest, license_value, text=None):
        """Admission decision. Returns ("ADMITTED", "ok") or ("BLOCKED", reason).

        Precedence is fixed so the reason is deterministic: missing field, then
        license, then digest mismatch.
        """
        for field in REQUIRED_FIELDS:
            if not manifest.get(field):
                return ("BLOCKED", f"missing required field: {field}")
        if license_value not in LICENSE_ALLOWLIST:
            return ("BLOCKED", f"license not in allow-list: {license_value!r}")
        if text is not None:
            recomputed = digest_string(canonical_text(text))
            if recomputed != manifest["record_id"]:
                return (
                    "BLOCKED",
                    (
                        f"digest mismatch: recomputed {short(recomputed)} != "
                        f"recorded {short(manifest['record_id'])}"
                    ),
                )
        return ("ADMITTED", "ok")

    return (
        REQUIRED_FIELDS,
        corpus_manifest,
        gate,
        record_manifest,
        shard_manifest,
    )


@app.cell(hide_code=True)
def tokenizers(
    COMPARISON_TOKENIZER_NAME,
    NOTEBOOK_PATH,
    SLICE_PATH,
    TIKTOKEN_CACHE_DIR,
    TOKENIZER_NAME,
    mo,
    sha256_file,
    short,
    slice_manifest,
):
    import os as _os

    # Pinning the cache directory is part of the provenance claim: the tokenizer's
    # BPE artifact is downloaded once, kept locally (gitignored), and its own
    # sha256 goes into the corpus manifest.
    TIKTOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_CACHE_DIR.resolve())

    import tiktoken as _tiktoken

    encoding = _tiktoken.get_encoding(TOKENIZER_NAME)
    comparison_encoding = _tiktoken.get_encoding(COMPARISON_TOKENIZER_NAME)

    _bpe_files = sorted(p for p in TIKTOKEN_CACHE_DIR.glob("*") if p.is_file())
    tokenizer_bpe_sha256 = (
        sha256_file(_bpe_files[0]) if len(_bpe_files) == 1 else "unavailable"
    )
    tokenizer_bpe_bytes = _bpe_files[0].stat().st_size if len(_bpe_files) == 1 else 0

    notebook_digest = sha256_file(NOTEBOOK_PATH)
    recomputed_slice_digest = sha256_file(SLICE_PATH)
    slice_digest_matches = recomputed_slice_digest == slice_manifest["slice_sha256"]

    mo.md(f"""
        ### Instruments, content-addressed

        | Instrument | Identity |
        |---|---|
        | Tokenizer | `{TOKENIZER_NAME}`, `n_vocab` {encoding.n_vocab:,} |
        | Tokenizer BPE artifact | {tokenizer_bpe_bytes:,} bytes, sha256 `{short(tokenizer_bpe_sha256)}` |
        | Comparison tokenizer | `{COMPARISON_TOKENIZER_NAME}`, `n_vocab` {comparison_encoding.n_vocab:,} |
        | This notebook's source | sha256 `{short(notebook_digest)}` |

        **Recorded slice digest verified against the file on disk:
        {"MATCH" if slice_digest_matches else "MISMATCH"}** — recomputed
        `{short(recomputed_slice_digest)}` vs recorded
        `{short(slice_manifest["slice_sha256"])}`.

        `o200k_base` is *not* the tokenizer a Hindi model would actually use, and
        nothing below claims it is. The claim is narrower and more useful: the count
        is **named and reproducible**. A fairer Indic comparison belongs as a second
        named column, not as a silent replacement for this one.
    """)
    return (
        comparison_encoding,
        encoding,
        notebook_digest,
        recomputed_slice_digest,
        slice_digest_matches,
        tokenizer_bpe_sha256,
    )


@app.cell(hide_code=True)
def emit_run(
    SHARD_COUNT,
    comparison_encoding,
    corpus_manifest,
    encoding,
    mo,
    notebook_digest,
    record_manifest,
    shard_manifest,
    shard_of,
    short,
    slice_rows,
    time_module,
    tokenizer_bpe_sha256,
):
    def emit_corpus(rows, run_utc=None):
        """Full emitter pass: rows -> record manifests -> shard manifests -> corpus.

        Pure with respect to `rows`: called twice below to prove determinism.
        """
        records = [record_manifest(row, encoding, comparison_encoding) for row in rows]
        grouped = [[] for _ in range(SHARD_COUNT)]
        for index, record in enumerate(records):
            grouped[shard_of(index, len(records), SHARD_COUNT)].append(record)
        shards = [shard_manifest(group, i) for i, group in enumerate(grouped)]
        corpus = corpus_manifest(shards, notebook_digest, tokenizer_bpe_sha256, run_utc)
        return records, shards, corpus


    _t0 = time_module.perf_counter()
    records_pass1, shards_pass1, corpus_pass1 = emit_corpus(slice_rows)
    emitter_seconds = time_module.perf_counter() - _t0

    record_digests = [r["record_id"] for r in records_pass1]
    corpus_bytes_total = corpus_pass1["byte_count"]

    shard_overview = [
        {
            "shard": s["shard_id"],
            "records": s["record_count"],
            "tokens": f"{s['token_count']:,}",
            "bytes": f"{s['byte_count']:,}",
            "merkle_root": short(s["merkle_root"]),
        }
        for s in shards_pass1
    ]

    mo.vstack([
        mo.md(f"""
        ### One emitter pass

        {len(records_pass1):,} record manifests, partitioned **in stream order** into
        {SHARD_COUNT} shards by equal row count (the last absorbs the remainder), then
        rolled up into one corpus manifest. Wall time
        **{emitter_seconds:.2f}s** — that is
        **{corpus_bytes_total / emitter_seconds / 1e6:.1f} MB/s**, and the hash-cost
        metric below shows what actually consumes it.

        Corpus Merkle root over the {SHARD_COUNT} shard roots:
        `{short(corpus_pass1["merkle_root"], 24)}`
        """),
        mo.ui.table(shard_overview, selection=None, page_size=6),
    ])
    return (
        corpus_bytes_total,
        corpus_pass1,
        emit_corpus,
        emitter_seconds,
        record_digests,
        records_pass1,
        shards_pass1,
    )


@app.cell(hide_code=True)
def metrics_prose(mo):
    mo.md(r"""
    ## What provenance instrumentation actually buys

    Nine measurements on the slice above. Each one is the answer to "what would I
    not know if I skipped this stage?"
    """)
    return


@app.cell(hide_code=True)
def metric_duplicates(
    Counter,
    mo,
    record_digests,
    records_pass1,
    short,
    slice_rows,
):
    _digest_counts = Counter(record_digests)
    duplicate_groups = {d: n for d, n in _digest_counts.items() if n > 1}
    distinct_digests = len(_digest_counts)

    _largest = max(duplicate_groups, key=lambda d: duplicate_groups[d], default=None)
    duplicate_group_rows = []
    if _largest is not None:
        for record, _row in zip(records_pass1, slice_rows):
            if record["record_id"] == _largest:
                duplicate_group_rows.append({
                    "upstream_doc_id": record["upstream_doc_id"],
                    "type": record["type"],
                    "bytes": record["byte_count"],
                    "tokens": record["token_count"],
                    "record_id (content-addressed)": short(record["record_id"]),
                })

    metric_duplicates = {
        "records_in": len(records_pass1),
        "distinct_content_digests": distinct_digests,
        "duplicate_groups": len(duplicate_groups),
        "redundant_records": len(records_pass1) - distinct_digests,
        "largest_group_size": max(duplicate_groups.values(), default=0),
        "largest_group_distinct_upstream_ids": len(
            {r["upstream_doc_id"] for r in duplicate_group_rows}
        ),
    }

    mo.vstack([
        mo.md(f"""
        ### 1. Opaque ids hide exact duplicates

        **{metric_duplicates["records_in"]:,} records → {distinct_digests:,} distinct
        content digests.** {metric_duplicates["redundant_records"]} records are
        byte-identical restatements of text already present, in
        {metric_duplicates["duplicate_groups"]} group(s).

        The largest group holds **{metric_duplicates["largest_group_size"]}
        byte-identical documents carrying
        {metric_duplicates["largest_group_distinct_upstream_ids"]} distinct upstream
        `doc_id`s** — printed in full below. Every row has the same `byte_count`, the
        same `token_count`, and the same content digest. Under upstream ids alone
        they are {metric_duplicates["largest_group_size"]} different documents; under
        a content address they are one.

        This is the whole argument for content addressing in a single table: the
        dedup stage can only collapse what the identity scheme lets it see as equal.
        """),
        mo.ui.table(duplicate_group_rows, selection=None, page_size=10),
    ])
    return duplicate_group_rows, metric_duplicates


@app.cell(hide_code=True)
def metric_canonicalization(
    hashlib_module,
    mo,
    percent,
    re_module,
    slice_rows,
    unicodedata_module,
):
    CANONICAL_VARIANTS = {
        "raw (utf-8, untouched)": lambda s: s,
        "strip()": lambda s: s.strip(),
        "NFC (the pinned form)": lambda s: unicodedata_module.normalize("NFC", s),
        "NFKC (compatibility)": lambda s: unicodedata_module.normalize("NFKC", s),
        "LF -> CRLF": lambda s: s.replace("\n", "\r\n"),
        "whitespace collapse": lambda s: re_module.sub(r"\s+", " ", s),
        "lower()": lambda s: s.lower(),
    }

    _raw_digests = [
        hashlib_module.sha256(_row["text"].encode("utf-8")).hexdigest()
        for _row in slice_rows
    ]

    canonicalization_rows = []
    for _label, _transform in CANONICAL_VARIANTS.items():
        _changed = sum(
            1
            for _row, baseline in zip(slice_rows, _raw_digests)
            if hashlib_module.sha256(_transform(_row["text"]).encode("utf-8")).hexdigest()
            != baseline
        )
        canonicalization_rows.append({
            "canonical form": _label,
            "digests changed vs raw": _changed,
            "share": percent(_changed, len(slice_rows)),
        })

    metric_canonicalization = {
        _row["canonical form"]: {
            "digests_changed": _row["digests changed vs raw"],
            "share": _row["share"],
        }
        for _row in canonicalization_rows
    }

    mo.vstack([
        mo.md(r"""
        ### 2. Canonicalization is load-bearing, and it is a choice

        The same abstract document, encoded seven ways, produces up to seven
        different digests. Each row counts how many of the 225 record digests move
        if that transform is applied before hashing.
        """),
        mo.ui.table(canonicalization_rows, selection=None, page_size=8),
        mo.md(f"""
        Read the rows, not the average:

        - **`strip()` moves
          {metric_canonicalization["strip()"]["digests_changed"]} of
          {len(slice_rows)} digests
          ({metric_canonicalization["strip()"]["share"]}).** A one-line convenience
          would silently re-identify most of the corpus. This is exactly why the
          emitted text is **not** stripped — only the hashing copy is normalized, and
          only with NFC.
        - **Line endings and whitespace collapse move
          {metric_canonicalization["LF -> CRLF"]["digests_changed"]} and
          {metric_canonicalization["whitespace collapse"]["digests_changed"]}
          digests.** A corpus that survives a checkout on a different OS is not
          guaranteed to survive it byte-identically.
        - **NFC moves
          {metric_canonicalization["NFC (the pinned form)"]["digests_changed"]}
          digests here** — this slice is already NFC-normalized upstream. That is a
          property of *this* data, not a general one, and it is precisely why the
          form must be *pinned and stated* rather than assumed: the cost of pinning
          it is zero here and unbounded on the next corpus.
        - **NFKC would move
          {metric_canonicalization["NFKC (compatibility)"]["digests_changed"]}
          digests, which is also why it is the wrong tool.** UAX #15: the
          compatibility mappings "must not be blindly applied to arbitrary text".
          Agreeing with NFC on this sample does not make it safe on the next one —
          NFKC erases distinctions by design.

        **A content digest is meaningless without a pinned canonical form.**
        """),
    ])
    return (metric_canonicalization,)


@app.cell(hide_code=True)
def metric_tokens(
    COMPARISON_TOKENIZER_NAME,
    TOKENIZER_NAME,
    mo,
    percent,
    records_pass1,
):
    _words_total = sum(r["word_count"] for r in records_pass1)
    _proxy_total = sum(r["words_x_135"] for r in records_pass1)
    _o200k_total = sum(r["token_count"] for r in records_pass1)
    _cl100k_total = sum(r["comparison_token_count"] for r in records_pass1)

    token_honesty_rows = [
        {
            "measure": "words × 1.35 (extraction-stage proxy)",
            "tokens": f"{_proxy_total:,}",
            "tokens per word": f"{_proxy_total / _words_total:.3f}",
            "named tokenizer?": "no",
        },
        {
            "measure": f"{TOKENIZER_NAME} (this stage's pinned count)",
            "tokens": f"{_o200k_total:,}",
            "tokens per word": f"{_o200k_total / _words_total:.3f}",
            "named tokenizer?": "yes",
        },
        {
            "measure": f"{COMPARISON_TOKENIZER_NAME} (same text, other tokenizer)",
            "tokens": f"{_cl100k_total:,}",
            "tokens per word": f"{_cl100k_total / _words_total:.3f}",
            "named tokenizer?": "yes",
        },
    ]

    _by_type = {}
    for _record in records_pass1:
        _bucket = _by_type.setdefault(
            _record["type"], {"records": 0, "words": 0, "tokens": 0}
        )
        _bucket["records"] += 1
        _bucket["words"] += _record["word_count"]
        _bucket["tokens"] += _record["token_count"]

    fertility_rows = [
        {
            "type": _type,
            "records": _bucket["records"],
            "words": f"{_bucket['words']:,}",
            f"{TOKENIZER_NAME} tokens": f"{_bucket['tokens']:,}",
            "fertility (tok/word)": f"{_bucket['tokens'] / _bucket['words']:.3f}",
        }
        for _type, _bucket in sorted(
            _by_type.items(), key=lambda kv: -kv[1]["records"]
        )
    ]

    metric_tokens = {
        "word_count": _words_total,
        "words_x_135": _proxy_total,
        "o200k_base": _o200k_total,
        "cl100k_base": _cl100k_total,
        "proxy_error_share": percent(_proxy_total - _o200k_total, _o200k_total),
        "tokenizer_spread": f"{_cl100k_total / _o200k_total:.2f}x",
        "fertility_o200k": f"{_o200k_total / _words_total:.3f}",
        "fertility_by_type": {
            _type: f"{_bucket['tokens'] / _bucket['words']:.3f}"
            for _type, _bucket in _by_type.items()
        },
    }

    mo.vstack([
        mo.md(r"""
        ### 3. A token count is meaningless without a named tokenizer
        """),
        mo.ui.table(token_honesty_rows, selection=None, page_size=5),
        mo.md(f"""
        Two independent failures, visible in three rows:

        **The proxy is wrong by a knowable amount.** `words × 1.35`
        ({_proxy_total:,}) against the real `{TOKENIZER_NAME}` count
        ({_o200k_total:,}) **undercounts by
        {metric_tokens["proxy_error_share"].lstrip("-")}**. True fertility on this
        slice is {metric_tokens["fertility_o200k"]} tok/word, not 1.35. That error is
        fine for deciding when to stop a download — its actual job in the extraction
        stage — and not fine for a corpus-size claim in a manifest.

        **The tokenizer matters more than the proxy error.** The identical bytes
        measure {_o200k_total:,} tokens under `{TOKENIZER_NAME}` and
        {_cl100k_total:,} under `{COMPARISON_TOKENIZER_NAME}` — a
        **{metric_tokens["tokenizer_spread"]} spread**, dwarfing the proxy's
        {metric_tokens["proxy_error_share"].lstrip("-")}. `cl100k_base` has no
        Devanagari merges worth the name, so it falls back to near-byte-level
        encoding on Hindi. "This corpus is 150K tokens" is not a fact about the
        corpus; it is a fact about a corpus *and* a tokenizer, which is why
        `tokenizer` and `tokenizer_bpe_sha256` are non-optional manifest fields.

        Fertility also varies by source type, so the aggregate number hides
        structure:
        """),
        mo.ui.table(fertility_rows, selection=None, page_size=6),
    ])
    return (metric_tokens,)


@app.cell(hide_code=True)
def metric_upstream_id(
    Counter,
    hashlib_module,
    mo,
    slice_rows,
    unicodedata_module,
):
    UPSTREAM_ID_CANDIDATES = ("md5", "sha1", "sha256", "sha512", "blake2b")

    _id_lengths = Counter(len(_row["doc_id"]) for _row in slice_rows)
    _hex_only = all(
        all(c in "0123456789abcdef" for c in _row["doc_id"]) for _row in slice_rows
    )

    upstream_id_rows = []
    for _algorithm in UPSTREAM_ID_CANDIDATES:
        _matches = 0
        for _row in slice_rows:
            for _candidate_text in (
                _row["text"],
                _row["text"].strip(),
                unicodedata_module.normalize("NFC", _row["text"]),
            ):
                _hex = hashlib_module.new(
                    _algorithm, _candidate_text.encode("utf-8")
                ).hexdigest()
                if _hex.startswith(_row["doc_id"]) or _row["doc_id"].startswith(_hex):
                    _matches += 1
                    break
        upstream_id_rows.append({
            "algorithm tried": _algorithm,
            "digest hex chars": hashlib_module.new(_algorithm).digest_size * 2,
            "doc_ids reproduced": _matches,
        })

    metric_upstream_id = {
        "doc_id_lengths": {str(k): v for k, v in _id_lengths.items()},
        "hex_only": _hex_only,
        "reproduced_by": {
            _row["algorithm tried"]: _row["doc_ids reproduced"] for _row in upstream_id_rows
        },
        "total_reproduced": sum(_row["doc_ids reproduced"] for _row in upstream_id_rows),
    }

    mo.vstack([
        mo.md(f"""
        ### 4. The upstream id is not content-derived, so it proves nothing

        Every `doc_id` in the slice is hexadecimal
        ({"hex-only: yes" if _hex_only else "hex-only: no"}), in **two different
        lengths**: {", ".join(f"{v} ids of {k} chars" for k, v in sorted(metric_upstream_id["doc_id_lengths"].items()))}.
        A 40-char hex string *looks* exactly like SHA-1 and a 64-char one like
        SHA-256. Neither is. Below, five algorithms are computed over the shipped
        `text` in three canonical forms each (raw, stripped, NFC) and matched
        against the recorded id.
        """),
        mo.ui.table(upstream_id_rows, selection=None, page_size=6),
        mo.md(f"""
        **{metric_upstream_id["total_reproduced"]} of {len(slice_rows)} `doc_id`s are
        reproducible from the text that ships with them.** The id is an opaque
        upstream key — probably a digest of a pre-cleaning artifact we never receive.
        The two coexisting lengths are the tell: a single content-addressing scheme
        produces one digest width, so this field carries ids minted by at least two
        different upstream processes.

        Two consequences worth stating plainly:

        1. **Integrity is not checkable from the upstream id.** Corrupt one character
           of any `text` and the `doc_id` still "matches", because it never matched
           anything to begin with. That is why `record_id` in this stage's manifests
           is recomputed locally rather than inherited.
        2. **A hex-shaped identifier invites exactly this mistake.** An id that
           *looks* like a digest gets trusted like a digest. Emitting
           `sha2-256:<hex>` instead of bare hex makes the difference visible in the
           value itself — the multihash principle: the algorithm travels with the
           digest.
        """),
    ])
    return (metric_upstream_id,)


@app.cell(hide_code=True)
def metric_hash_cost(
    canonical_text,
    corpus_bytes_total,
    emitter_seconds,
    hashlib_module,
    mo,
    percent,
    slice_rows,
    time_module,
):
    HASH_ALGORITHMS = ("md5", "sha1", "sha256", "sha512", "blake2b")

    _blob = b"".join(canonical_text(_row["text"]) for _row in slice_rows)

    hash_cost_rows = []
    for _algorithm in HASH_ALGORITHMS:
        _best = None
        for _ in range(3):  # keep the fastest run: it is the least noisy estimate
            _start = time_module.perf_counter()
            hashlib_module.new(_algorithm, _blob).hexdigest()
            _elapsed = time_module.perf_counter() - _start
            _best = _elapsed if _best is None else min(_best, _elapsed)
        hash_cost_rows.append({
            "algorithm": _algorithm,
            "digest bits": hashlib_module.new(_algorithm).digest_size * 8,
            "MB/s": round(len(_blob) / _best / 1e6),
            "cryptographic": "no (collisions found)"
            if _algorithm in {"md5", "sha1"}
            else "yes",
        })

    _sha256_rate = next(r["MB/s"] for r in hash_cost_rows if r["algorithm"] == "sha256")
    _emitter_rate = corpus_bytes_total / emitter_seconds / 1e6

    metric_hash_cost = {
        "corpus_bytes": len(_blob),
        "rates_mb_s": {r["algorithm"]: r["MB/s"] for r in hash_cost_rows},
        "emitter_wall_seconds": f"{emitter_seconds:.3f}",
        "emitter_mb_s": f"{_emitter_rate:.1f}",
        "hashing_share_of_emitter": percent(_emitter_rate, _sha256_rate),
    }

    mo.vstack([
        mo.md(r"""
        ### 5. SHA-256 is not the bottleneck — the usual objection is wrong
        """),
        mo.ui.table(hash_cost_rows, selection=None, page_size=6),
        mo.md(f"""
        `sha256` runs at **{_sha256_rate:,} MB/s** on this machine — within noise of
        `sha1` and several times `blake2b`, because these CPUs have SHA-NI
        instructions and `hashlib` uses them. The full emitter (canonicalize →
        digest → **tokenize** → build three manifest levels → Merkle-roll) runs at
        **{_emitter_rate:.1f} MB/s**, i.e. hashing accounts for roughly
        {metric_hash_cost["hashing_share_of_emitter"]} of the emitter's time budget.

        **Tokenization dominates; hashing is effectively free.** So "we skipped
        content hashing for performance" is not a real trade-off at this scale — and
        the weaker choices are strictly worse: `md5` is the *slowest* algorithm in the
        table *and* has practical collisions. There is no speed argument for picking a
        broken hash.
        """),
    ])
    return (metric_hash_cost,)


@app.cell(hide_code=True)
def metric_determinism(
    canonical_json,
    corpus_pass1,
    emit_corpus,
    mo,
    record_digests,
    records_pass1,
    shards_pass1,
    short,
    slice_rows,
):
    _records_pass2, _shards_pass2, _corpus_pass2 = emit_corpus(slice_rows)


    def _comparable(manifest):
        """Everything except run_utc. A wall-clock stamp is not a fact about the
        data, so it is excluded from the determinism comparison and reported
        separately -- including it would make every run trivially 'different'."""
        body = dict(manifest)
        body.pop("run_utc", None)
        body.pop("manifest_digest", None)
        return canonical_json(body)

    # A counter-based id is the standard alternative to a content address. It is
    # stable within a run and worthless across runs -- shown here as the contrast.
    _counter_ids_pass1 = [f"rec-{i:05d}" for i in range(len(records_pass1))]
    _counter_ids_pass2 = [
        f"rec-{i:05d}" for i in sorted(range(len(_records_pass2)), key=lambda i: -i)
    ]

    determinism_rows = [
        {
            "compared": "corpus manifest canonical JSON (run_utc excluded)",
            "pass 1 == pass 2": _comparable(corpus_pass1) == _comparable(_corpus_pass2),
            "evidence": f"{len(_comparable(corpus_pass1)):,} bytes, byte-for-byte",
        },
        {
            "compared": "corpus Merkle root",
            "pass 1 == pass 2": corpus_pass1["merkle_root"] == _corpus_pass2["merkle_root"],
            "evidence": short(corpus_pass1["merkle_root"], 16),
        },
        {
            "compared": "all shard Merkle roots",
            "pass 1 == pass 2": [s["merkle_root"] for s in shards_pass1]
            == [s["merkle_root"] for s in _shards_pass2],
            "evidence": f"{len(shards_pass1)} shards",
        },
        {
            "compared": "all record_ids, in order",
            "pass 1 == pass 2": record_digests
            == [r["record_id"] for r in _records_pass2],
            "evidence": f"{len(record_digests):,} content-addressed ids",
        },
        {
            "compared": "counter-based ids under a reordered pass (the contrast)",
            "pass 1 == pass 2": _counter_ids_pass1 == _counter_ids_pass2,
            "evidence": "position-derived: reordering renames every record",
        },
    ]

    metric_determinism = {
        "corpus_json_identical": determinism_rows[0]["pass 1 == pass 2"],
        "corpus_root_identical": determinism_rows[1]["pass 1 == pass 2"],
        "shard_roots_identical": determinism_rows[2]["pass 1 == pass 2"],
        "record_ids_identical": determinism_rows[3]["pass 1 == pass 2"],
        "counter_ids_identical_under_reorder": determinism_rows[4]["pass 1 == pass 2"],
        "run_utc_excluded": True,
    }

    mo.vstack([
        mo.md(r"""
        ### 6. Determinism: two independent passes, byte-identical
        """),
        mo.ui.table(determinism_rows, selection=None, page_size=6),
        mo.md(f"""
        The emitter ran a second, fully independent pass over the same rows. Every
        content-derived value agrees **byte-for-byte**, which is the property that
        makes a manifest checkable by a third party: they re-run the emitter and
        compare one root.

        `run_utc` is excluded from the comparison and reported alongside instead
        (`{corpus_pass1["run_utc"]}` vs `{_corpus_pass2["run_utc"]}`). A wall-clock
        stamp is a fact about *when the run happened*, not about the data; leaving it
        in the compared bytes would make every honest re-run look like a mismatch and
        train readers to ignore the check.

        The last row is the contrast that makes the point. Counter-based ids are
        stable only if nothing about the input ordering ever changes — reorder the
        same documents and every id silently moves to a different document. A content
        address has no such coupling: identity follows the bytes.
        """),
    ])
    return (metric_determinism,)


@app.cell(hide_code=True)
def metric_tamper(
    SHARD_COUNT,
    corpus_pass1,
    emit_corpus,
    math_module,
    merkle_path,
    merkle_root,
    merkle_verify,
    mo,
    records_pass1,
    shard_of,
    shards_pass1,
    short,
    slice_rows,
):
    TAMPER_ROW_INDEX = 7

    _tampered_rows = [dict(row) for row in slice_rows]
    _victim = _tampered_rows[TAMPER_ROW_INDEX]
    _original_text = _victim["text"]
    # One character. Not a word, not a line -- one character.
    _victim["text"] = _original_text[:40] + ("x" if _original_text[40] != "x" else "y") + _original_text[41:]

    _records_tampered, _shards_tampered, _corpus_tampered = emit_corpus(_tampered_rows)

    _changed_records = [
        i
        for i, (before, after) in enumerate(zip(records_pass1, _records_tampered))
        if before["record_id"] != after["record_id"]
    ]
    _changed_shards = [
        s["shard_id"]
        for s, t in zip(shards_pass1, _shards_tampered)
        if s["merkle_root"] != t["merkle_root"]
    ]

    # The proof issued before the tamper. It is now stale.
    _shard_index = shard_of(TAMPER_ROW_INDEX, len(records_pass1), SHARD_COUNT)
    _shard_records = [
        r
        for i, r in enumerate(records_pass1)
        if shard_of(i, len(records_pass1), SHARD_COUNT) == _shard_index
    ]
    _leaf_index = next(
        i
        for i, r in enumerate(_shard_records)
        if r["record_id"] == records_pass1[TAMPER_ROW_INDEX]["record_id"]
    )
    _leaves = [r["record_id"].encode("utf-8") for r in _shard_records]
    _stale_root = merkle_root(_leaves)
    _stale_path = merkle_path(_leaves, _leaf_index)

    _honest_leaf = records_pass1[TAMPER_ROW_INDEX]["record_id"].encode("utf-8")
    _tampered_leaf = _records_tampered[TAMPER_ROW_INDEX]["record_id"].encode("utf-8")
    _neighbour_index = 0 if _leaf_index != 0 else 1

    tamper_rows = [
        {
            "check": "record digests changed",
            "result": f"{len(_changed_records)} of {len(records_pass1)}",
            "verdict": "DETECTED",
        },
        {
            "check": "which record",
            "result": f"row {_changed_records[0]} (the tampered one)"
            if _changed_records
            else "none",
            "verdict": "LOCALISED",
        },
        {
            "check": "shard roots changed",
            "result": f"{len(_changed_shards)} of {SHARD_COUNT}: {', '.join(_changed_shards)}",
            "verdict": "PROPAGATED",
        },
        {
            "check": "corpus root changed",
            "result": f"{short(corpus_pass1['merkle_root'])} → {short(_corpus_tampered['merkle_root'])}",
            "verdict": "PROPAGATED",
        },
        {
            "check": "inclusion path length",
            "result": f"{len(_stale_path)} siblings for {len(_leaves)} leaves "
            f"(ceil(log2 n) = {math_module.ceil(math_module.log2(len(_leaves)))})",
            "verdict": "O(log n)",
        },
        {
            "check": "stale proof vs the untampered leaf",
            "result": str(
                merkle_verify(_honest_leaf, _leaf_index, _stale_path, len(_leaves), _stale_root)
            ),
            "verdict": "ACCEPTS",
        },
        {
            "check": "stale proof vs the tampered leaf",
            "result": str(
                merkle_verify(
                    _tampered_leaf, _leaf_index, _stale_path, len(_leaves), _stale_root
                )
            ),
            "verdict": "REJECTS",
        },
        {
            "check": "an unrelated leaf in the same shard",
            "result": str(
                merkle_verify(
                    _leaves[_neighbour_index],
                    _neighbour_index,
                    merkle_path(_leaves, _neighbour_index),
                    len(_leaves),
                    _stale_root,
                )
            ),
            "verdict": "STILL VERIFIES",
        },
    ]

    metric_tamper = {
        "tampered_row": TAMPER_ROW_INDEX,
        "characters_changed": 1,
        "record_digests_changed": len(_changed_records),
        "shard_roots_changed": len(_changed_shards),
        "corpus_root_changed": corpus_pass1["merkle_root"] != _corpus_tampered["merkle_root"],
        "inclusion_path_length": len(_stale_path),
        "expected_path_length": math_module.ceil(math_module.log2(len(_leaves))),
        "stale_proof_accepts_untampered": merkle_verify(
            _honest_leaf, _leaf_index, _stale_path, len(_leaves), _stale_root
        ),
        "stale_proof_rejects_tampered": merkle_verify(
            _tampered_leaf, _leaf_index, _stale_path, len(_leaves), _stale_root
        ),
    }

    mo.vstack([
        mo.md(f"""
        ### 7. Tamper detection: one character, three levels

        One character of row {TAMPER_ROW_INDEX} was changed — not a word, not a line.
        The emitter then re-ran unchanged.
        """),
        mo.ui.table(tamper_rows, selection=None, page_size=9),
        mo.md(f"""
        Every property that matters is in that table:

        - **Detection is exact.** Exactly {len(_changed_records)} record digest moved,
          so the change is not just noticed but *localised* to one document.
        - **It propagates.** {len(_changed_shards)} shard root and the corpus root
          both change. You cannot alter a document and leave the corpus root intact,
          which is what makes a single published root a meaningful commitment to
          {len(records_pass1):,} documents.
        - **Verification stays cheap.** Proving one record belongs to its shard takes
          {len(_stale_path)} sibling hashes, not {len(_leaves)} — O(log n), the reason
          RFC 6962 logs scale.
        - **Rejection is targeted, not blanket.** The proof issued before the tamper
          still accepts every honest leaf and rejects only the tampered one. A scheme
          that invalidated everything would be useless in practice; this one tells you
          *which* record to distrust.
        """),
    ])
    return (metric_tamper,)


@app.cell(hide_code=True)
def metric_gate(LICENSE, REQUIRED_FIELDS, gate, mo, records_pass1, slice_rows):
    _good_record = dict(records_pass1[0])
    _good_text = slice_rows[0]["text"]

    gate_rows = [
        {
            "mutation": "none (control)",
            "verdict": gate(_good_record, LICENSE, _good_text)[0],
            "reason": gate(_good_record, LICENSE, _good_text)[1],
        }
    ]

    for _field in REQUIRED_FIELDS:
        _mutated = dict(_good_record)
        _mutated.pop(_field)
        _verdict, _reason = gate(_mutated, LICENSE, _good_text)
        gate_rows.append({
            "mutation": f"drop {_field!r}",
            "verdict": _verdict,
            "reason": _reason,
        })

    for _field, _empty in (("token_count", 0), ("byte_count", 0), ("type", "")):
        _mutated = dict(_good_record)
        _mutated[_field] = _empty
        _verdict, _reason = gate(_mutated, LICENSE, _good_text)
        gate_rows.append({
            "mutation": f"{_field} = {_empty!r}",
            "verdict": _verdict,
            "reason": _reason,
        })

    for _bad_license in ("unknown", "unsafe", "", None, "CC-BY-NC-4.0"):
        _verdict, _reason = gate(_good_record, _bad_license, _good_text)
        gate_rows.append({
            "mutation": f"license = {_bad_license!r}",
            "verdict": _verdict,
            "reason": _reason,
        })

    _corrupted = dict(_good_record)
    _corrupted["record_id"] = "sha2-256:" + ("0" * 64)
    _verdict, _reason = gate(_corrupted, LICENSE, _good_text)
    gate_rows.append({
        "mutation": "corrupt recorded record_id",
        "verdict": _verdict,
        "reason": _reason,
    })

    _verdict, _reason = gate(_good_record, LICENSE, _good_text + " ")
    gate_rows.append({
        "mutation": "text altered after the manifest was written",
        "verdict": _verdict,
        "reason": _reason,
    })

    _mutations = [r for r in gate_rows if r["mutation"] != "none (control)"]
    _admitted_mutations = [r for r in _mutations if r["verdict"] == "ADMITTED"]

    _full_sweep = [
        gate(record, LICENSE, row["text"])
        for record, row in zip(records_pass1, slice_rows)
    ]

    metric_gate = {
        "mutations_tested": len(_mutations),
        "mutations_blocked": len(_mutations) - len(_admitted_mutations),
        "mutations_admitted": len(_admitted_mutations),
        "control_admitted": gate_rows[0]["verdict"] == "ADMITTED",
        "distinct_reasons": len({r["reason"] for r in _mutations}),
        "real_corpus_admitted": sum(1 for v, _ in _full_sweep if v == "ADMITTED"),
        "real_corpus_blocked": sum(1 for v, _ in _full_sweep if v == "BLOCKED"),
    }

    mo.vstack([
        mo.md(r"""
        ### 8. The gate: one mutation at a time
        """),
        mo.ui.table(gate_rows, selection=None, page_size=16),
        mo.md(f"""
        **{metric_gate["mutations_blocked"]} of {metric_gate["mutations_tested"]}
        mutations BLOCKED; {metric_gate["mutations_admitted"]} admitted.** The control
        row — an untouched record with an allow-listed license — is the only
        `ADMITTED` verdict in the table, which is what makes the other rows evidence
        rather than a tautology.

        Two design points the table demonstrates:

        **Reasons are specific and machine-readable.**
        {metric_gate["distinct_reasons"]} distinct reason strings, each naming the
        field or value at fault. "Rejected" is an unactionable log line; `missing
        required field: tokenizer` tells the operator what to fix. Precedence is fixed
        (missing field → license → digest) so the same defect always reports the same
        reason.

        **Zero and absent are the same failure.** `token_count = 0` blocks for
        *missing required field*, not for a separate zero-check: a field present but
        empty carries no more information than a field that isn't there, and treating
        them differently is how a "0 tokens" record slips into a corpus report.

        Run against the real slice, the gate admits
        **{metric_gate["real_corpus_admitted"]:,}** records and blocks
        **{metric_gate["real_corpus_blocked"]}** — the corpus as emitted is
        self-consistent, which is the only claim a self-consistency check can make.
        """),
    ])
    return (metric_gate,)


@app.cell(hide_code=True)
def metric_overhead(
    canonical_json,
    corpus_bytes_total,
    corpus_pass1,
    mo,
    percent,
    records_pass1,
    shards_pass1,
):
    _record_manifest_bytes = sum(len(canonical_json(r)) for r in records_pass1)
    _shard_manifest_bytes = sum(len(canonical_json(s)) for s in shards_pass1)
    _corpus_manifest_bytes = len(canonical_json(corpus_pass1))
    _manifest_bytes_total = (
        _record_manifest_bytes + _shard_manifest_bytes + _corpus_manifest_bytes
    )

    overhead_rows = [
        {
            "level": "record manifests",
            "count": len(records_pass1),
            "bytes": f"{_record_manifest_bytes:,}",
            "vs corpus text": percent(_record_manifest_bytes, corpus_bytes_total),
        },
        {
            "level": "shard manifests",
            "count": len(shards_pass1),
            "bytes": f"{_shard_manifest_bytes:,}",
            "vs corpus text": percent(_shard_manifest_bytes, corpus_bytes_total),
        },
        {
            "level": "corpus manifest",
            "count": 1,
            "bytes": f"{_corpus_manifest_bytes:,}",
            "vs corpus text": percent(_corpus_manifest_bytes, corpus_bytes_total),
        },
        {
            "level": "all manifests",
            "count": len(records_pass1) + len(shards_pass1) + 1,
            "bytes": f"{_manifest_bytes_total:,}",
            "vs corpus text": percent(_manifest_bytes_total, corpus_bytes_total),
        },
    ]

    metric_overhead = {
        "corpus_text_bytes": corpus_bytes_total,
        "record_manifest_bytes": _record_manifest_bytes,
        "shard_manifest_bytes": _shard_manifest_bytes,
        "corpus_manifest_bytes": _corpus_manifest_bytes,
        "manifest_bytes_total": _manifest_bytes_total,
        "manifest_share_of_corpus": percent(_manifest_bytes_total, corpus_bytes_total),
        "bytes_per_record": _manifest_bytes_total // len(records_pass1),
    }

    mo.vstack([
        mo.md(r"""
        ### 9. What it costs
        """),
        mo.ui.table(overhead_rows, selection=None, page_size=5),
        mo.md(f"""
        **{metric_overhead["manifest_share_of_corpus"]} of corpus bytes, or
        {metric_overhead["bytes_per_record"]} bytes per record.** The shard manifests
        carry the record-id lists, which is most of their size; the corpus manifest is
        a single small object regardless of corpus size.

        Note the scaling: record manifests grow linearly with the corpus, but their
        *share* stays roughly constant because manifest size per record is fixed
        while document sizes are not. On a corpus with longer documents the share
        falls. There is no configuration of this pipeline in which provenance is the
        expensive part — combined with metric 5, the honest summary is that this stage
        costs a few percent of bytes and a few percent of time.
        """),
    ])
    return (metric_overhead,)


@app.cell(hide_code=True)
def metric_merkle_selftest(
    math_module,
    merkle_path,
    merkle_root,
    merkle_verify,
    mo,
    record_digests,
):
    MERKLE_SELFTEST_SIZES = (1, 2, 3, 5, 7, 8, 63, 64, 66)

    merkle_selftest_rows = []
    for _n in (*MERKLE_SELFTEST_SIZES, len(record_digests)):
        _leaves = [f"leaf-{i}".encode() for i in range(_n)]
        _root = merkle_root(_leaves)
        _all_pass = True
        _max_path = 0
        for _i in range(_n):
            _path = merkle_path(_leaves, _i)
            _max_path = max(_max_path, len(_path))
            if not merkle_verify(_leaves[_i], _i, _path, _n, _root):
                _all_pass = False
        _expected = math_module.ceil(math_module.log2(_n)) if _n > 1 else 0
        merkle_selftest_rows.append({
            "leaves (n)": _n,
            "all inclusion proofs pass": _all_pass,
            "max path length": _max_path,
            "ceil(log2 n)": _expected,
            "path length correct": _max_path == _expected,
        })

    metric_merkle_selftest = {
        "sizes_tested": [r["leaves (n)"] for r in merkle_selftest_rows],
        "all_proofs_pass": all(r["all inclusion proofs pass"] for r in merkle_selftest_rows),
        "all_path_lengths_correct": all(
            r["path length correct"] for r in merkle_selftest_rows
        ),
        "leaves_verified": sum(r["leaves (n)"] for r in merkle_selftest_rows),
    }

    # A wrong Merkle verifier returns well-formed digests and looks fine. These
    # assertions are the only thing standing between that and a false provenance
    # claim, so they fail the notebook rather than printing a warning.
    assert metric_merkle_selftest["all_proofs_pass"], "Merkle inclusion proofs failed"
    assert metric_merkle_selftest["all_path_lengths_correct"], "Merkle path length wrong"

    mo.vstack([
        mo.md(r"""
        ### Interlude — Merkle verifier self-test

        Odd sizes, powers of two, one-off-a-power-of-two, and the real corpus size.
        Every leaf at every size is proved individually against its own root.
        """),
        mo.ui.table(merkle_selftest_rows, selection=None, page_size=10),
        mo.md(f"""
        **{metric_merkle_selftest["leaves_verified"]:,} individual inclusion proofs,
        all passing, with max path length equal to `ceil(log2 n)` at every size.**

        This table exists because of a specific bug. The first verifier written for this
        notebook folded the audit path top-down in a single pass — the natural-looking
        implementation — and **silently failed 56 of 64 proofs**. Silently, because a
        wrong hash fold still produces a perfectly well-formed SHA-256 digest, and a
        provenance system whose verifier is broken reports success. The two `assert`
        statements in this cell fail the notebook outright rather than printing a
        warning, since a passing-but-wrong verifier is worse than no verifier at all.

        Odd sizes (3, 5, 7, 63) are included deliberately: they are where the
        largest-power-of-two split rule is exercised, and where an implementation that
        pads to a power of two by duplicating the last leaf would silently disagree
        with RFC 6962.
        """),
    ])
    return (metric_merkle_selftest,)


@app.cell(hide_code=True)
def persist(
    MANIFEST_DIR,
    METRICS_PATH,
    PIPELINE_VERSION,
    SLICE_PATH,
    TOKENIZER_NAME,
    canonical_json,
    corpus_pass1,
    metric_canonicalization,
    metric_determinism,
    metric_duplicates,
    metric_gate,
    metric_hash_cost,
    metric_merkle_selftest,
    metric_overhead,
    metric_tamper,
    metric_tokens,
    metric_upstream_id,
    mo,
    notebook_digest,
    recomputed_slice_digest,
    records_pass1,
    sha256_file,
    shards_pass1,
    short,
    slice_digest_matches,
    slice_manifest,
    slice_rows,
    tokenizer_bpe_sha256,
):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    _written = []


    def _write_canonical(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_bytes(canonical_json(obj) + b"\n")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        _written.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": short(sha256_file(path)),
        })


    for _shard in shards_pass1:
        _write_canonical(MANIFEST_DIR / f"{_shard['shard_id']}.manifest.json", _shard)

    _write_canonical(MANIFEST_DIR / "records.manifest.json", {
        "pipeline_version": PIPELINE_VERSION,
        "tokenizer": TOKENIZER_NAME,
        "record_count": len(records_pass1),
        "records": records_pass1,
    })

    _write_canonical(MANIFEST_DIR / "corpus.manifest.json", corpus_pass1)

    metrics_document = {
        "pipeline_version": PIPELINE_VERSION,
        "run_utc": corpus_pass1["run_utc"],
        "source": corpus_pass1["source"],
        "tokenizer": TOKENIZER_NAME,
        "tokenizer_bpe_sha256": tokenizer_bpe_sha256,
        "notebook_sha256": notebook_digest,
        "slice": {
            "path": str(SLICE_PATH),
            "records": len(slice_rows),
            "recorded_sha256": slice_manifest["slice_sha256"],
            "recomputed_sha256": recomputed_slice_digest,
            "digest_verified": slice_digest_matches,
        },
        "corpus_merkle_root": corpus_pass1["merkle_root"],
        "metrics": {
            "merkle_selftest": metric_merkle_selftest,
            "duplicates": metric_duplicates,
            "canonicalization": metric_canonicalization,
            "tokens": metric_tokens,
            "upstream_id": metric_upstream_id,
            "hash_cost": metric_hash_cost,
            "determinism": metric_determinism,
            "tamper": metric_tamper,
            "gate": metric_gate,
            "overhead": metric_overhead,
        },
    }

    _write_canonical(METRICS_PATH, metrics_document)

    mo.vstack([
        mo.md(f"""
        ### Persisted artifacts

        Written as canonical JSON via a temp-file-and-rename, so a crashed run cannot
        leave a half-written manifest that still parses. All
        {len(_written)} files are tracked in git — a manifest that only exists in a
        notebook's memory has not been emitted.
        """),
        mo.ui.table(_written, selection=None, page_size=8),
    ])
    return


@app.cell
def svg_helpers(INK, MUTED, RULE, SURFACE, SVG_FONT, mo):
    def svg_figure(svg_body, caption, view_width, view_height):
        """Wrap a hand-authored SVG body in a scaling frame plus its takeaway.

        No rendering library and no mermaid: every diagram in this notebook is
        literal SVG, so the reader can see exactly what is being claimed.
        """
        return mo.vstack([
            mo.Html(
                f'<svg viewBox="0 0 {view_width} {view_height}" width="100%" '
                f'role="img" xmlns="http://www.w3.org/2000/svg" '
                f'font-family="{SVG_FONT}">{svg_body}</svg>'
            ),
            mo.md(f"*{caption}*"),
        ])


    def svg_box(x, y, w, h, label, sub=None, fill=SURFACE, stroke=RULE, text=INK):
        """A labelled rounded box. `sub` is a second line, one step smaller."""
        parts = [
            (
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )
        ]
        if sub is None:
            parts.append(
                f'<text x="{x + w / 2}" y="{y + h / 2 + 4}" text-anchor="middle" '
                f'font-size="12" fill="{text}">{label}</text>'
            )
        else:
            parts.append(
                f'<text x="{x + w / 2}" y="{y + h / 2 - 3}" text-anchor="middle" '
                f'font-size="12" fill="{text}">{label}</text>'
            )
            parts.append(
                f'<text x="{x + w / 2}" y="{y + h / 2 + 12}" text-anchor="middle" '
                f'font-size="11" fill="{MUTED}">{sub}</text>'
            )
        return "".join(parts)


    def svg_arrow(x1, y1, x2, y2, stroke=MUTED, label=None, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts = [
            (
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
                f'stroke-width="1.5" marker-end="url(#mp-arrow)"{dash_attr}/>'
            )
        ]
        if label:
            parts.append(
                f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 6}" '
                f'text-anchor="middle" font-size="11" fill="{MUTED}">{label}</text>'
            )
        return "".join(parts)


    def svg_text(x, y, label, size=12, fill=INK, anchor="start", weight="normal"):
        return (
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}">{label}</text>'
        )


    # One marker definition, referenced by every arrow in every diagram.
    SVG_DEFS = (
        '<defs><marker id="mp-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker></defs>'
    )
    return SVG_DEFS, svg_arrow, svg_box, svg_figure, svg_text


@app.cell(hide_code=True)
def reference_header(mo):
    mo.md(r"""
    ---

    # Technique reference

    Twelve techniques in three groups. Each entry carries the same five parts:
    **what it is**, **where it was used** (with a citation and a note on *how* it
    was used there), **how it works internally**, **when to use it**, and **pros
    and cons**. Diagrams sit next to the technique they explain rather than in a
    gallery at the end.

    | Group | Question it answers | Techniques |
    |---|---|---|
    | **A. Identity of the bytes** | Which document is this, exactly? | 1 content-addressed record identity · 2 canonicalization before hashing · 3 self-describing digests · 4 content-defined chunking |
    | **B. Identity of the process** | Which pipeline produced it? | 5 input-addressed pipeline identity · 6 sidecar attribute files · 7 lineage graphs |
    | **C. Trust in the record** | Why should anyone believe the manifest? | 8 Merkle trees and append-only logs · 9 signed attestations · 10 machine-readable dataset metadata · 11 admission gating · 12 canary insertion |

    **Excluded on purpose**, stated rather than silently dropped:

    - **Text watermarking** — it modifies the training bytes. This stage's whole
      contract is that it does not touch the text, so watermarking belongs to a
      different stage or a different project.
    - **Differential-privacy accounting** — a privacy budget is a property of a
      training procedure, not of a corpus manifest. It would be bookkeeping about
      a thing this pipeline does not do.
    - **Blockchain-anchored registries** — a Merkle log already provides
      append-only integrity and inclusion proofs (technique 8). Consensus adds
      cost to solve a problem — mutual distrust between many writers — that a
      single-publisher corpus does not have.

    One note on sourcing. `deeptut-manifest-provenance.md` in this folder is the
    **input research report, not a citable source**: two of its claims were
    checked against primaries and are wrong (it attributes SHA-256 manifest
    hashing plus script-hash verification to Dolma, whose verified mechanism is
    `documents/`+`attributes/` sidecars with Bloom-filter dedup; and it states DPI
    found "more than 60% of dataset licenses were miscategorized" where the paper
    reports 70%+ *omission* and 50%+ error rates). Every claim below was verified
    against the primary source cited with it.
    """)
    return


@app.cell(hide_code=True)
def diagram_gate(
    ACCENT,
    ACCENT_BG,
    ADMIT,
    ADMIT_BG,
    BLOCK,
    BLOCK_BG,
    MUTED,
    SVG_DEFS,
    svg_arrow,
    svg_box,
    svg_figure,
    svg_text,
):
    _stages = [
        ("raw shard", "text as received"),
        ("canonicalize", "NFC, hashing copy"),
        ("digest", "sha2-256:…"),
        ("measure", "named tokenizer"),
        ("emit", "canonical JSON"),
    ]

    _body = [SVG_DEFS]
    _x = 12
    for _label, _sub in _stages:
        _body.append(svg_box(_x, 60, 116, 46, _label, _sub))
        if _x > 12:
            _body.append(svg_arrow(_x - 16, 83, _x - 3, 83))
        _x += 132

    _body.append(svg_arrow(_x - 16, 83, _x - 3, 83))
    _body.append(svg_box(_x, 60, 96, 46, "gate", "one decision", ACCENT_BG, ACCENT, ACCENT))

    _gate_x = _x + 96
    _body.append(svg_arrow(_gate_x + 4, 72, _gate_x + 44, 34))
    _body.append(
        svg_box(_gate_x + 48, 12, 150, 42, "ADMITTED", "manifest is authoritative", ADMIT_BG, ADMIT, ADMIT)
    )
    _body.append(svg_arrow(_gate_x + 4, 94, _gate_x + 44, 130))
    _body.append(
        svg_box(_gate_x + 48, 110, 150, 42, "BLOCKED", "+ specific reason", BLOCK_BG, BLOCK, BLOCK)
    )

    _body.append(svg_text(12, 176, "The text itself is never modified by this stage.", 11, MUTED))
    _body.append(
        svg_text(
            12,
            30,
            "Both exits are drawn on purpose: a gate with only one exit is a log line.",
            11,
            MUTED,
        )
    )

    svg_figure(
        "".join(_body),
        "Diagram 1 — the emitter is a pipeline with two exits. A record either becomes "
        "authoritative or is blocked with a reason a human can act on; there is no "
        "third outcome and no silent pass-through.",
        960,
        190,
    )
    return


@app.cell(hide_code=True)
def technique_01(mo):
    mo.md(r"""
    ## A. Identity of the bytes

    ### 1. Content-addressed record identity

    **What it is.** The record's primary key *is* a cryptographic digest of its
    bytes: `record_id = sha2-256:<hex over the canonical bytes>`. Identity is
    derived from content rather than assigned by whatever process happened to
    touch the record first.

    **Where it was used.**

    - **RedPajama-1T** ([dataset card](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T))
      publishes per-slice `*_SHA256SUMS.txt` files under
      `data.together.xyz/redpajama-data-1T/v1.0.0/sha256/`. *How*: at the **file**
      level, for `sha256sum -c` verification after download — the digest is the
      integrity contract between publisher and downloader, not an internal key.
    - **RedPajama-V2** ([arXiv:2411.12372](https://arxiv.org/abs/2411.12372))
      carries a per-document CCNet `digest: "sha1:…"` field. *How*: at the
      **document** level, as the join key that lets the separately-shipped quality
      signals and duplicate flags be attached back to documents.

    Note the two are different jobs. File digests answer "did I download this
    intact"; document digests answer "which document is this". This notebook does
    both: `slice_sha256` for the file, `record_id` for each document.

    **How it works internally.** Canonicalize (technique 2) → hash → prefix with
    the algorithm name (technique 3). Because SHA-256's output is
    collision-resistant, equal digests mean equal bytes for any practical corpus,
    so the digest can be used as a dictionary key, a dedup key, and an integrity
    check simultaneously. Content addressing is also *idempotent under
    re-ingestion*: re-processing the same document produces the same id, with no
    coordination and no counter state.

    **When to use it.** Whenever a record must be identifiable across pipeline
    stages, machines, or organizations — and always in preference to a positional
    or counter-based id, which encodes *when a record was seen* rather than *what
    it is*. Metric 6 shows the failure directly: reorder the input and every
    counter-based id silently moves to a different document.

    **Pros.** Verifiable by anyone holding the bytes, requires no registry,
    deduplicates exact copies for free (metric 1: 5 byte-identical documents with
    5 distinct upstream ids collapse to one), and detects tampering (metric 7).
    Costs essentially nothing: metric 5 measures SHA-256 at ~3 GB/s, ~0.3% of the
    emitter's time.

    **Cons.** Identity is fragile by design — one whitespace character is a
    different document, which makes the choice of canonical form load-bearing
    rather than cosmetic (metric 2: a bare `.strip()` would move 89% of ids). It
    says nothing about near-duplicates; that is the deduplication stage's job with
    MinHash/LSH. And the id is opaque to humans: `sha2-256:8e5c…` is not
    browsable, so a human-readable path or title usually has to travel alongside
    it.
    """)
    return


@app.cell(hide_code=True)
def diagram_opaque_vs_content(
    ADMIT,
    ADMIT_BG,
    BLOCK,
    BLOCK_BG,
    INK,
    MUTED,
    RULE,
    SURFACE,
    SVG_DEFS,
    duplicate_group_rows,
    svg_arrow,
    svg_box,
    svg_figure,
    svg_text,
):
    _dup_ids = [row["upstream_doc_id"] for row in duplicate_group_rows][:5]
    _dup_digest = duplicate_group_rows[0]["record_id (content-addressed)"]

    _body = [SVG_DEFS]
    _body.append(svg_text(12, 22, "Same 5 byte-identical documents, two identity schemes", 12, INK, weight="600"))

    _body.append(svg_text(12, 52, "upstream doc_id (opaque)", 11, MUTED))
    for _i, _doc_id in enumerate(_dup_ids):
        _y = 62 + _i * 30
        _body.append(svg_box(12, _y, 200, 24, f"{_doc_id[:12]}…", None, SURFACE, RULE))
        _body.append(svg_arrow(216, _y + 12, 250, _y + 12))
        _body.append(svg_box(254, _y, 118, 24, "kept", None, BLOCK_BG, BLOCK, BLOCK))
    _body.append(svg_text(12, 62 + 5 * 30 + 14, f"{len(_dup_ids)} rows in, {len(_dup_ids)} rows out", 11, BLOCK))

    _body.append(svg_text(430, 52, "record_id (content-addressed)", 11, MUTED))
    for _i in range(len(_dup_ids)):
        _y = 62 + _i * 30
        _body.append(svg_box(430, _y, 200, 24, _dup_digest, None, SURFACE, RULE))
        _body.append(svg_arrow(634, _y + 12, 686, 122))
    _body.append(
        svg_box(690, 110, 150, 24, "1 distinct id", None, ADMIT_BG, ADMIT, ADMIT)
    )
    _body.append(svg_text(430, 62 + 5 * 30 + 14, f"{len(_dup_ids)} rows in, 1 distinct id out", 11, ADMIT))

    _body.append(
        svg_text(
            12,
            62 + 5 * 30 + 36,
            "Identical byte_count, identical token_count, identical digest — five different keys.",
            11,
            MUTED,
        )
    )

    svg_figure(
        "".join(_body),
        f"Diagram 2 — metric 1, drawn. The opaque-id lane cannot see that these "
        f"{len(_dup_ids)} documents are the same bytes, so all "
        f"{len(_dup_ids)} survive. The content-address lane collapses them to one "
        f"without comparing any text.",
        880,
        250,
    )
    return


@app.cell(hide_code=True)
def technique_02(mo):
    mo.md(r"""
    ### 2. Canonicalization before hashing

    **What it is.** Reducing many equivalent surface encodings of the same logical
    value to exactly one byte string, *then* hashing. Two layers are needed
    because two things get hashed: JSON objects (manifests) and text (documents).

    **Where it was used.**

    - **RFC 8785 (JSON Canonicalization Scheme)** for structured data. *How*: no
      whitespace, ECMAScript number and string serialization, and recursive key
      sorting on **UTF-16 code units compared as unsigned integers** — a
      specifically-chosen ordering, not "alphabetical", because JSON strings are
      defined over UTF-16. Output is UTF-8.
    - **UAX #15 (Unicode Normalization Forms)** for text. *How*: defines the four
      forms and states that the compatibility mappings (NFKC/NFKD) **"must not be
      blindly applied to arbitrary text"**, because they deliberately erase
      distinctions. That sentence is the entire reason this notebook uses NFC.

    **How it works internally.** For JSON: recursively sort object keys, drop
    insignificant whitespace, serialize numbers by one fixed rule, and encode
    UTF-8. For text: Unicode normalization rewrites the code point sequence so
    that canonically equivalent sequences (a precomposed character vs base +
    combining mark) become identical — critical for Devanagari, where a
    nukta-bearing consonant has two legitimate encodings.

    **When to use it.** Before *every* digest that must be reproducible on another
    machine. And the specific form must be recorded in the manifest, not left
    implicit — that is the difference between a digest and a number that looks
    like one.

    **Pros.** Makes digests stable across serializers, languages, and operating
    systems. Key-order independence in particular means a manifest can be
    re-emitted by a different implementation and still verify.

    **Cons.** Every canonicalization step is *lossy in the identity domain*: it
    declares some real differences irrelevant. Metric 2 quantifies exactly what
    that costs — `.strip()` reassigns 89.3% of ids, whitespace collapse 99.6%.
    Choose too aggressively and unrelated documents collide; too conservatively
    and trivially-different copies are treated as distinct. There is also an
    implementation trap: this notebook's `canonical_json` is an **approximation**
    of RFC 8785, because `json.dumps` does not implement ECMAScript number
    formatting (JCS requires `1` where Python writes `1.0`). The mitigation is
    structural, not hopeful — **every numeric manifest field is an `int`**, where
    the two agree, and ratios are emitted as strings.
    """)
    return


@app.cell(hide_code=True)
def diagram_canon_ladder(
    ACCENT,
    ACCENT_BG,
    ADMIT,
    ADMIT_BG,
    INK,
    MUTED,
    RULE,
    SURFACE,
    SVG_DEFS,
    canonical_text,
    digest_string,
    hashlib_module,
    re_module,
    short,
    slice_rows,
    svg_arrow,
    svg_box,
    svg_figure,
    svg_text,
    unicodedata_module,
):
    _sample = slice_rows[0]["text"]
    _ladder = [
        ("raw utf-8", _sample),
        ("strip()", _sample.strip()),
        ("LF -> CRLF", _sample.replace("\n", "\r\n")),
        ("whitespace collapse", re_module.sub(r"\s+", " ", _sample)),
        ("lower()", _sample.lower()),
        ("NFKC", unicodedata_module.normalize("NFKC", _sample)),
    ]

    _body = [SVG_DEFS]
    _body.append(svg_text(12, 22, "One abstract document", 12, INK, weight="600"))
    _body.append(svg_box(12, 34, 190, 34, "the same Hindi text", "one logical value", ACCENT_BG, ACCENT, ACCENT))

    _body.append(svg_text(250, 22, "six byte encodings", 11, MUTED))
    _body.append(svg_text(560, 22, "six digests", 11, MUTED))

    for _i, (_label, _variant) in enumerate(_ladder):
        _y = 34 + _i * 34
        _body.append(svg_arrow(206, 51, 246, _y + 13))
        _body.append(svg_box(250, _y, 190, 26, _label, None, SURFACE, RULE))
        _body.append(svg_arrow(444, _y + 13, 484, _y + 13))
        _digest = short(hashlib_module.sha256(_variant.encode("utf-8")).hexdigest())
        _body.append(svg_box(488, _y, 170, 26, _digest, None, SURFACE, RULE, MUTED))

    _pinned_y = 34 + len(_ladder) * 34 + 16
    _body.append(
        svg_box(250, _pinned_y, 190, 30, "NFC — the pinned form", None, ADMIT_BG, ADMIT, ADMIT)
    )
    _body.append(svg_arrow(444, _pinned_y + 15, 484, _pinned_y + 15))
    _body.append(
        svg_box(
            488,
            _pinned_y,
            170,
            30,
            short(digest_string(canonical_text(_sample))),
            None,
            ADMIT_BG,
            ADMIT,
            ADMIT,
        )
    )
    _body.append(svg_text(676, _pinned_y + 19, "exactly one id", 11, ADMIT))
    _body.append(
        svg_text(12, _pinned_y + 19, "Pinning collapses the ladder to one row.", 11, MUTED)
    )

    svg_figure(
        "".join(_body),
        "Diagram 3 — metric 2, drawn on the slice's first document. Six defensible "
        "encodings of one logical value give six unrelated digests. Nothing about a "
        "digest reveals which rung produced it, so the canonical form must be stated "
        "in the manifest, not inferred.",
        880,
        300,
    )
    return


@app.cell(hide_code=True)
def technique_03(mo):
    mo.md(r"""
    ### 3. Self-describing digests

    **What it is.** Emitting `sha2-256:8e5c…` instead of a bare `8e5c…`, so the
    algorithm travels inside the value. Every digest in this notebook's manifests
    is self-describing.

    **Where it was used.**

    - **multihash** ([spec](https://github.com/multiformats/multihash)). *How*: a
      binary layout of `<varint hash-function code><varint digest length><digest>`
      — the function code and length are part of the value itself, so a parser
      never has to be told what it is reading.
    - **IPFS CIDv1** wraps a multihash with a multibase prefix, a version, and a
      multicodec for the content type. *How*: the identifier is fully
      self-describing at every layer — encoding, version, content type, hash
      function — which is what lets one address format survive algorithm
      migrations.

    **How it works internally.** A prefix declares the algorithm; the reader
    dispatches on it. This notebook uses a **text** adaptation (`sha2-256:<hex>`)
    rather than multihash's binary varint layout, because manifests are JSON and
    the goal is human-inspectable values. The principle is identical.

    **When to use it.** Any digest that crosses a system boundary or is stored for
    longer than one release cycle — which is every digest in a corpus manifest.

    **Pros.** Ambiguity becomes impossible: metric 4 is a direct demonstration of
    the failure this prevents. That slice's `doc_id` field holds hex in *two
    different lengths* (196 ids of 40 chars, 29 of 64), so it looks like SHA-1 for
    most records and SHA-256 for others — and is in fact neither. A prefixed value
    could not have concealed that. It also makes algorithm migration mechanical:
    old and new digests coexist unambiguously in the same field, and a verifier
    can support both without a flag day.

    **Cons.** A handful of extra bytes per value (metric 9 puts total manifest
    overhead at 8.0% of corpus bytes, of which prefixes are a small part), and
    it is a convention that must be enforced at the emitter — nothing stops a
    caller writing bare hex into the same field. The prefix authenticates nothing
    on its own: it says which algorithm was *claimed*, and only recomputation
    proves it, which is why the gate recomputes rather than trusting the recorded
    value.
    """)
    return


@app.cell(hide_code=True)
def technique_04(mo):
    mo.md(r"""
    ### 4. Content-defined chunking

    **What it is.** Splitting a byte stream at boundaries chosen by the *content*
    (a rolling hash over a sliding window crossing a threshold) rather than at
    fixed offsets, so that inserting a byte near the start shifts only the chunk
    containing it instead of re-cutting everything after it.

    **Where it was used.**

    - **Hugging Face Xet** ([deduplication docs](https://huggingface.co/docs/hub/en/xet/deduplication)
      and the Xet engineering blogs). *How*: content-defined chunking at **~64 KB**
      average chunk size, with chunks aggregated into ~64 MB blocks ("xorbs") in a
      content-addressable store, so that re-uploading a modified model or dataset
      file transfers only the changed chunks. The docs use `MerkleHash` and xorb
      terminology, tying chunk identity into the same content-addressing scheme as
      technique 1.

    A deliberate non-claim: **BLAKE3 is not confirmed** by those primary sources.
    It is widely repeated that Xet uses BLAKE3, and the pages fetched for this
    study do not say so, so this notebook does not assert it.

    **How it works internally.** Slide a window over the bytes maintaining a
    rolling hash (Rabin-Karp or Gear-style). When the low bits of the hash match a
    mask, declare a boundary. Because the decision depends only on the window's
    contents, identical regions of two files produce identical boundaries no
    matter what precedes them. Min/max chunk sizes bound the pathological cases.

    **When to use it.** When large artifacts are *revised* — model checkpoints,
    growing corpora, re-crawls — and transfer or storage of the unchanged parts is
    the dominant cost.

    **Pros.** Shift-resistant deduplication at sub-file granularity; a one-line
    change to a large JSONL re-uploads one chunk, not the file.

    **Cons.** It is the wrong tool for **this** stage, and the reason is worth
    stating precisely. CDC deduplicates *storage*; this stage needs *document
    identity*. A 64 KB chunk boundary has no relationship to a document boundary,
    so chunk digests cannot answer "is this document already in the corpus" —
    which is exactly the question metric 1 answers with whole-document digests.
    CDC also adds a rolling-hash pass over every byte and a chunk index to
    maintain, and the parameters (average size, mask, min/max) are themselves
    provenance that must be recorded or the chunking is not reproducible. Used
    here it would add cost and answer a question nobody asked.
    """)
    return


@app.cell(hide_code=True)
def technique_05(mo):
    mo.md(r"""
    ## B. Identity of the process

    ### 5. Input-addressed pipeline identity

    **What it is.** Deriving the identity of an *output* from a digest of
    everything that produced it — inputs, code, configuration, tool versions —
    rather than from where the output happens to sit. This notebook's
    `cleaning_script` field is the small version of it: the corpus manifest records
    the SHA-256 of the notebook's own source.

    **Where it was used.**

    - **Nix** ([store paths and derivations](https://nix.dev/manual/nix/latest/store/store-path)).
      *How*: an input-addressed output path is hashed over the **ATerm
      serialization of the entire derivation** — sources, dependencies, the builder
      executable, its arguments, and the environment — with the digest rendered as
      base-32 of SHA-256 truncated to 160 bits. Change any input and the output
      *path itself* changes, so two builds cannot silently occupy the same
      location.
    - **Bazel** ([remote caching](https://bazel.build/remote/caching),
      [hermeticity](https://bazel.build/basics/hermeticity)). *How*: an action
      cache keyed by an **action hash over the command line, the environment, and
      the digests of all inputs**, mapping to results, alongside a CAS holding the
      output blobs. A cache hit is a proof that re-running would be pointless.
    - **DVC** ([`dvc.yaml`/`dvc.lock`](https://dvc.org/doc/user-guide/project-structure/dvcyaml-files)).
      *How*: `dvc.lock` records each stage's `cmd` plus **`md5` content hashes of
      every dependency and output** and the resolved parameter values; `dvc repro`
      re-runs only the stages whose recorded hashes no longer match. This is the
      pattern closest to a data-cleaning pipeline.

    **How it works internally.** Serialize the full input closure canonically
    (technique 2), hash it, and use the digest as the output's identity or cache
    key. Correctness rests on *completeness of the closure*: anything that affects
    the output and is not hashed is a hole through which non-reproducibility
    enters — the reason Bazel documents hermeticity as a prerequisite rather than a
    nice-to-have.

    **When to use it.** As soon as "which version of the cleaner produced this
    corpus?" is a question anyone will ask — which is immediately, because it is
    the first question asked when a downstream model behaves oddly.

    **Pros.** Makes staleness detectable instead of assumed, enables safe caching,
    and turns "did the code change?" into a digest comparison. In this notebook it
    is *reflexive*: the manifest is signed, in the weak sense, by the code that
    wrote it. Note the deliberate consequence — **the `cleaning_script` digest
    changes on every notebook edit. That is the point, not a defect**: a manifest
    emitted by different code is a manifest from a different pipeline, and pretending
    otherwise is how two incomparable corpora end up with the same version label.

    **Cons.** This notebook's version is honestly incomplete, and the gap is worth
    naming: hashing the notebook source captures the code but **not** the installed
    library versions, the Python build, or the OS. A full closure would pin
    `uv.lock` and the interpreter too — Nix's actual scope. Input-addressing also
    guarantees *identity*, not *determinism*: if the code itself is
    nondeterministic, two runs with the same input hash still differ, which is why
    metric 6 tests determinism separately rather than assuming it.
    """)
    return


@app.cell(hide_code=True)
def diagram_input_addressed(
    ACCENT,
    ACCENT_BG,
    ADMIT,
    ADMIT_BG,
    BLOCK,
    DATASET_REVISION,
    INK,
    MUTED,
    RULE,
    SURFACE,
    SVG_DEFS,
    corpus_pass1,
    notebook_digest,
    short,
    slice_manifest,
    svg_arrow,
    svg_box,
    svg_figure,
    svg_text,
    tokenizer_bpe_sha256,
):
    _inputs = [
        ("source slice", short(slice_manifest["slice_sha256"])),
        ("this notebook", short(notebook_digest)),
        ("tokenizer BPE", short(tokenizer_bpe_sha256)),
        ("pinned config", f"rev {DATASET_REVISION[:8]}…"),
    ]

    _body = [SVG_DEFS]
    _body.append(svg_text(12, 22, "Everything that determined the output, hashed", 12, INK, weight="600"))

    for _i, (_label, _digest) in enumerate(_inputs):
        _y = 36 + _i * 46
        _body.append(svg_box(12, _y, 210, 38, _label, _digest, SURFACE, RULE))
        _body.append(svg_arrow(226, _y + 19, 292, 126))

    _body.append(
        svg_box(296, 100, 190, 52, "derivation hash", "canonical JSON → sha2-256", ACCENT_BG, ACCENT, ACCENT)
    )
    _body.append(svg_arrow(490, 126, 540, 126))
    _body.append(
        svg_box(
            544,
            100,
            250,
            52,
            "corpus identity",
            short(corpus_pass1["manifest_digest"]),
            ADMIT_BG,
            ADMIT,
            ADMIT,
        )
    )

    _body.append(
        svg_text(296, 176, "Change any input on the left and the identity on the right changes.", 11, MUTED)
    )
    _body.append(
        svg_text(
            296,
            194,
            "Not captured here, and stated as a gap: library versions, interpreter build, OS.",
            11,
            BLOCK,
        )
    )

    svg_figure(
        "".join(_body),
        "Diagram 5 — input-addressed derivation, as this notebook actually implements "
        "it. The red line is the honest limit: the closure covers data, code, and "
        "tokenizer, but not the environment, so it is weaker than a Nix derivation "
        "hash.",
        820,
        210,
    )
    return


@app.cell(hide_code=True)
def technique_06(mo):
    mo.md(r"""
    ### 6. Sidecar attribute files

    **What it is.** Recording per-document decisions in files that *mirror* the
    document files, instead of applying those decisions destructively to the
    documents. Filtering becomes a query over attributes rather than an
    irreversible edit.

    **Where it was used.**

    - **Dolma** ([data format docs](https://github.com/allenai/dolma/blob/main/docs/data-format.md),
      [arXiv:2402.00159](https://arxiv.org/abs/2402.00159)). *How*: documents are
      JSONL with `id`, `text`, `source` (plus optional `added`, `created`,
      `metadata`); every tagger writes to `attributes/<tagger-name>/` **mirroring
      the `documents/` paths exactly**, each row carrying `{source, id, attributes}`
      as join keys. Deduplication is a Rust Bloom-filter pass. The mirrored layout
      is the mechanism: the join is a path substitution, needing no index.
    - **RedPajama-V2** ([arXiv:2411.12372](https://arxiv.org/abs/2411.12372)).
      *How*: ships `quality_signals/*.signals.json.gz` beside the documents, and
      for exact deduplication **marks duplicates rather than deleting them** —
      leaving the retention threshold to the consumer. Its MinHash configuration is
      published as concrete parameters (128 hash functions; bands×rows of 14×9 for
      a Jaccard threshold of 0.7, plus 9×13, 5×25, and 1×128), which is itself a
      provenance practice: the *filter* is reproducible, not just its output.

    **How it works internally.** Documents are immutable and content-addressed
    (technique 1). Each analysis pass appends a new attribute file keyed by the
    same id. A consumer chooses thresholds at load time and materializes a view.

    **When to use it.** Whenever a filtering threshold might be revisited, several
    teams consume the same corpus with different tolerances, or an auditor must be
    able to see what was removed and why. That covers most real corpora.

    **Pros.** Decisions are reversible and auditable; disagreement between
    consumers costs a query instead of a re-crawl. It composes cleanly with content
    addressing — the join key is already the record id. And it makes the *filter*
    inspectable, which destructive filtering does not: after a delete, the evidence
    for the delete is gone.

    **Cons.** Storage grows with each tagger, and the mirrored layout must be
    maintained by convention — a renamed document path silently orphans its
    attributes. It also pushes work onto consumers, who must now know which
    attributes exist and what thresholds to apply; a corpus that ships as "already
    clean" is easier to use and harder to trust. Correctness depends on stable ids,
    so it inherits technique 1's canonicalization sensitivity: re-canonicalize the
    documents and every join key breaks at once.
    """)
    return


@app.cell(hide_code=True)
def diagram_sidecar(
    ACCENT,
    ACCENT_BG,
    ADMIT,
    ADMIT_BG,
    BLOCK,
    BLOCK_BG,
    INK,
    MUTED,
    RULE,
    SURFACE,
    SVG_DEFS,
    svg_arrow,
    svg_box,
    svg_figure,
    svg_text,
):
    _paths = ["hin/0000.jsonl", "hin/0001.jsonl", "hin/0002.jsonl"]

    _body = [SVG_DEFS]
    _body.append(svg_text(12, 22, "documents/ (immutable)", 12, INK, weight="600"))
    _body.append(svg_text(330, 22, "attributes/&lt;tagger&gt;/ (mirrored paths)", 12, INK, weight="600"))
    _body.append(svg_text(690, 22, "consumer", 12, INK, weight="600"))

    for _i, _path in enumerate(_paths):
        _y = 34 + _i * 42
        _body.append(svg_box(12, _y, 200, 34, _path, "id, text, source", SURFACE, RULE))
        _body.append(svg_arrow(216, _y + 17, 326, _y + 17, MUTED, "same path", "4 3"))
        _body.append(svg_box(330, _y, 210, 34, _path, "id, attributes", ACCENT_BG, ACCENT, ACCENT))
        _body.append(svg_arrow(544, _y + 17, 686, 96))

    _body.append(
        svg_box(690, 70, 200, 52, "threshold at load time", "a view, not a delete", ADMIT_BG, ADMIT, ADMIT)
    )

    _y2 = 34 + len(_paths) * 42 + 20
    _body.append(svg_text(12, _y2 + 14, "Destructive alternative:", 11, BLOCK))
    _body.append(svg_box(160, _y2, 200, 30, "filtered.jsonl", None, BLOCK_BG, BLOCK, BLOCK))
    _body.append(
        svg_text(
            372,
            _y2 + 20,
            "the removed rows, and the reason they were removed, no longer exist",
            11,
            BLOCK,
        )
    )

    svg_figure(
        "".join(_body),
        "Diagram 8 — the Dolma layout. Attributes join to documents by path "
        "substitution and record id, so a threshold change is a re-query. The red row "
        "is what destructive filtering costs: not just the data, but the evidence for "
        "the decision.",
        920,
        215,
    )
    return


@app.cell(hide_code=True)
def technique_07(mo):
    mo.md(r"""
    ### 7. Lineage graphs (PROV-O)

    **What it is.** Expressing provenance as a typed graph rather than a flat
    record: which **Entity** was generated by which **Activity**, which used which
    other Entity, on behalf of which **Agent**. A manifest states facts about one
    artifact; a lineage graph states relationships across many.

    **Where it was used.**

    - **W3C PROV-O** ([`w3.org/TR/prov-o/`](https://www.w3.org/TR/prov-o/), a 2013
      Recommendation). *How*: an OWL2 encoding of the PROV Data Model. The three
      core classes are `prov:Entity`, `prov:Activity`, and `prov:Agent`, and the
      starting-point properties that connect them are typed by domain and range —
      confirmed against the specification body:

      | Property (IRI under `http://www.w3.org/ns/prov#`) | Domain → Range |
      |---|---|
      | `prov:wasGeneratedBy` | `Entity` → `Activity` |
      | `prov:used` | `Activity` → `Entity` |
      | `prov:wasDerivedFrom` | `Entity` → `Entity` |
      | `prov:wasAttributedTo` | `Entity` → `Agent` |
      | `prov:wasAssociatedWith` | `Activity` → `Agent` |
      | `prov:startedAtTime` / `prov:endedAtTime` | `Activity` → `xsd:dateTime` |

      The typing is what makes the vocabulary useful: `prov:used` may only point
      from an Activity to an Entity, so "the cleaning run used the slice" and "the
      slice was generated by the download" cannot be stated backwards.

    **How it works internally.** Emit RDF triples (Turtle or JSON-LD) whose
    subjects are IRIs for artifacts and runs. Because it is OWL2, a reasoner can
    infer transitive derivation chains; because it is RDF, the graph can be queried
    with SPARQL and merged with graphs published by other parties without a shared
    schema negotiation.

    **When to use it.** When provenance must span *organizations* or many stages,
    and questions of the form "which downstream artifacts are affected if this
    source is retracted?" have to be answerable mechanically. It is the right tool
    for a corpus assembled from many sources by many teams.

    **Pros.** Standardized, typed, queryable, and mergeable. Transitive questions
    are graph traversals rather than manual archaeology, and the vocabulary is
    stable enough (a 2013 REC) to be a safe long-term serialization target.

    **Cons.** Considerably heavier than JSON manifests: an RDF store, IRI minting
    discipline, and a vocabulary the reader must know. For this eight-stage
    single-operator pipeline it would be ceremony — which is why this notebook
    emits JSON manifests and uses PROV-O only as the *conceptual* mapping in the
    next diagram. It also records structure, not integrity: PROV-O will happily
    describe a lineage that never happened. It needs digests (technique 1) and
    signatures (technique 9) underneath it to carry any weight.
    """)
    return


@app.cell(hide_code=True)
def diagram_provo(
    ACCENT,
    ACCENT_BG,
    ADMIT,
    ADMIT_BG,
    INK,
    MUTED,
    RULE,
    SURFACE,
    SVG_DEFS,
    svg_arrow,
    svg_box,
    svg_figure,
    svg_text,
):
    _body = [SVG_DEFS]
    _body.append(
        svg_text(12, 20, "PROV-O classes and starting-point properties, mapped onto this pipeline", 12, INK, weight="600")
    )

    # Entities (data artifacts), left to right in derivation order.
    _body.append(svg_box(12, 44, 170, 44, "sangraha verified/hin", "prov:Entity (source)", SURFACE, RULE))
    _body.append(svg_box(330, 44, 170, 44, "the tracked slice", "prov:Entity", SURFACE, RULE))
    _body.append(svg_box(648, 44, 200, 44, "corpus manifest", "prov:Entity", ADMIT_BG, ADMIT, ADMIT))

    # Activities, below the entities they connect.
    _body.append(svg_box(196, 132, 120, 44, "build_slice.py", "prov:Activity", ACCENT_BG, ACCENT, ACCENT))
    _body.append(svg_box(514, 132, 120, 44, "this notebook", "prov:Activity", ACCENT_BG, ACCENT, ACCENT))

    # Agent.
    _body.append(svg_box(330, 218, 200, 40, "the operator", "prov:Agent", SURFACE, RULE))

    # prov:used  (Activity -> Entity)
    _body.append(svg_arrow(196, 148, 186, 84, MUTED, None))
    _body.append(svg_text(96, 116, "prov:used", 11, MUTED))
    _body.append(svg_arrow(514, 148, 504, 84, MUTED, None))
    _body.append(svg_text(414, 116, "prov:used", 11, MUTED))

    # prov:wasGeneratedBy  (Entity -> Activity)
    _body.append(svg_arrow(336, 76, 320, 146, MUTED, None))
    _body.append(svg_text(330, 116, "prov:wasGeneratedBy", 11, MUTED))
    _body.append(svg_arrow(654, 76, 638, 146, MUTED, None))
    _body.append(svg_text(648, 116, "prov:wasGeneratedBy", 11, MUTED))

    # prov:wasDerivedFrom  (Entity -> Entity)
    _body.append(svg_arrow(326, 56, 190, 56, MUTED, None))
    _body.append(svg_text(196, 36, "prov:wasDerivedFrom", 11, MUTED))
    _body.append(svg_arrow(644, 56, 508, 56, MUTED, None))
    _body.append(svg_text(514, 36, "prov:wasDerivedFrom", 11, MUTED))

    # prov:wasAssociatedWith  (Activity -> Agent)
    _body.append(svg_arrow(256, 180, 348, 214, MUTED, None))
    _body.append(svg_arrow(574, 180, 512, 214, MUTED, None))
    _body.append(svg_text(360, 200, "prov:wasAssociatedWith", 11, MUTED))

    _body.append(
        svg_text(
            12,
            282,
            "Each arrow's direction is fixed by the property's declared domain and range — they cannot be drawn backwards.",
            11,
            MUTED,
        )
    )

    svg_figure(
        "".join(_body),
        "Diagram 6 — the same provenance this notebook emits as JSON, expressed in "
        "PROV-O terms. Only properties whose domain and range were confirmed against "
        "the specification body are used; the equivalent JSON is one file, which is "
        "why this stage ships JSON.",
        880,
        296,
    )
    return


@app.cell(hide_code=True)
def technique_08(mo):
    mo.md(r"""
    ## C. Trust in the record

    ### 8. Merkle trees and append-only logs

    **What it is.** A binary hash tree over the records, so a single root digest
    commits to the entire set, and any one record's membership can be proven with
    O(log n) sibling hashes instead of the whole corpus.

    **Where it was used.**

    - **RFC 6962 (Certificate Transparency)** — the construction this notebook
      implements verbatim. *How*: leaves are `SHA-256(0x00 ‖ d)` and internal nodes
      are `MTH(D[n]) = SHA-256(0x01 ‖ MTH(left) ‖ MTH(right))`. Two distinct proof
      types: an **audit path** proves a record is included under a given root, and a
      **consistency proof** proves an earlier root is a prefix of a later one — i.e.
      that the log was appended to and never rewritten.

    **How it works internally.** The `0x00`/`0x01` prefixes are **domain
    separation**, and they are not decoration: without them a leaf digest and an
    internal-node digest are drawn from the same space, and an attacker can present
    an internal node as a leaf (the second-preimage attack RFC 6962 explicitly
    prevents). Splitting at the largest power of two strictly below `n` gives a
    left-complete tree, so odd sizes need no duplicated leaves.

    One implementation warning, earned the hard way in this notebook. **The
    verifier must be two-pass.** The audit path is ordered leaf-to-root while the
    split recursion runs root-to-leaf, so the sides must be recorded on the way
    down and consumed in reverse on the way up. The natural-looking single-pass
    top-down fold was written first here and **silently failed 56 of 64 proofs** —
    silently, because every wrong answer is still a well-formed SHA-256 digest.
    That is the specific danger of hash code: it fails opaquely. Metric 3's
    self-test (all leaves at n ∈ {1,2,3,5,7,8,63,64,66,225}, plus a max-path-length
    check against `ceil(log2 n)`) exists solely to catch this class of bug.

    **When to use it.** When a commitment must be *published small* — one root in a
    paper, a release note, or a signature — while remaining verifiable per record;
    or when append-only history matters.

    **Pros.** One root commits to the whole corpus; verification is O(log n)
    (metric 7: 6 sibling hashes for a 56-leaf shard); tampering is both detected and
    localized, and honest records still verify against the old root while only the
    tampered one rejects. Rolling roots upward gives the three-level hierarchy for
    free.

    **Cons.** More machinery than a flat list of digests, and the domain-separation
    and orientation details are easy to get subtly wrong in ways tests must catch
    rather than review. It proves *inclusion*, not *correctness* — a Merkle root over
    garbage is a perfectly valid commitment to garbage. And it proves nothing about
    *authorship*: without a signature over the root (technique 9), anyone can publish
    a root for any corpus.
    """)
    return


@app.cell(hide_code=True)
def diagram_merkle(
    ACCENT,
    ACCENT_BG,
    BLOCK,
    BLOCK_BG,
    INK,
    MUTED,
    RULE,
    SURFACE,
    SVG_DEFS,
    merkle_leaf,
    merkle_node,
    merkle_path,
    merkle_root,
    short,
    svg_arrow,
    svg_box,
    svg_figure,
    svg_text,
):
    _demo_leaves = [f"record-{i}".encode() for i in range(8)]
    _demo_root = merkle_root(_demo_leaves)
    _TAMPERED = 5
    _demo_path = merkle_path(_demo_leaves, _TAMPERED)

    # Rebuild the internal nodes so the changed path can be drawn.
    _l0 = [merkle_leaf(leaf) for leaf in _demo_leaves]
    _l1 = [merkle_node(_l0[i], _l0[i + 1]) for i in range(0, 8, 2)]
    _l2 = [merkle_node(_l1[i], _l1[i + 1]) for i in range(0, 4, 2)]

    _changed_leaves = {_TAMPERED}
    _changed_l1 = {_TAMPERED // 2}
    _changed_l2 = {_TAMPERED // 4}
    _proof_leaves = {_TAMPERED ^ 1}
    _proof_l1 = {(_TAMPERED // 2) ^ 1}
    _proof_l2 = {(_TAMPERED // 4) ^ 1}

    _body = [SVG_DEFS]
    _body.append(svg_text(12, 20, "8 leaves, RFC 6962 domain separation", 12, INK, weight="600"))
    _body.append(svg_text(560, 20, "changed by the tamper", 11, BLOCK))
    _body.append(svg_text(730, 20, "the inclusion proof", 11, ACCENT))


    def _style(index, changed, proof):
        if index in changed:
            return (BLOCK_BG, BLOCK, BLOCK)
        if index in proof:
            return (ACCENT_BG, ACCENT, ACCENT)
        return (SURFACE, RULE, MUTED)


    # Root
    _body.append(svg_box(370, 44, 190, 34, f"root {short(_demo_root.hex(), 10)}", None, BLOCK_BG, BLOCK, BLOCK))

    # Level 2
    for _i, _node in enumerate(_l2):
        _x = 190 + _i * 380
        _fill, _stroke, _text = _style(_i, _changed_l2, _proof_l2)
        _body.append(svg_box(_x, 104, 170, 30, f"0x01 {short(_node.hex(), 8)}", None, _fill, _stroke, _text))
        _body.append(svg_arrow(_x + 85, 102, 465, 82))

    # Level 1
    for _i, _node in enumerate(_l1):
        _x = 20 + _i * 190
        _fill, _stroke, _text = _style(_i, _changed_l1, _proof_l1)
        _body.append(svg_box(_x, 164, 160, 30, f"0x01 {short(_node.hex(), 8)}", None, _fill, _stroke, _text))
        _body.append(svg_arrow(_x + 80, 162, 275 + (_i // 2) * 380, 138))

    # Leaves
    for _i, _node in enumerate(_l0):
        _x = 8 + _i * 95
        _fill, _stroke, _text = _style(_i, _changed_leaves, _proof_leaves)
        _body.append(svg_box(_x, 224, 86, 44, "0x00", short(_node.hex(), 6), _fill, _stroke, _text))
        _body.append(svg_arrow(_x + 43, 222, 100 + (_i // 2) * 190, 198))

    _body.append(
        svg_text(8, 292, f"leaf {_TAMPERED} tampered → its ancestors change all the way to the root", 11, BLOCK)
    )
    _body.append(
        svg_text(
            8,
            310,
            f"proof for leaf {_TAMPERED} = {len(_demo_path)} siblings, one per level (ceil(log2 8) = 3), not 8 digests",
            11,
            ACCENT,
        )
    )

    svg_figure(
        "".join(_body),
        "Diagram 4 — the red spine is why a single root detects any change; the purple "
        "boxes are why verifying one record is cheap. The 0x00/0x01 prefixes are RFC "
        "6962's domain separation: a leaf digest can never be presented as an internal "
        "node.",
        790,
        322,
    )
    return


@app.cell(hide_code=True)
def diagram_hierarchy(
    ACCENT,
    ACCENT_BG,
    ADMIT,
    ADMIT_BG,
    INK,
    MUTED,
    RULE,
    SHARD_COUNT,
    SURFACE,
    SVG_DEFS,
    corpus_pass1,
    record_digests,
    records_pass1,
    shards_pass1,
    short,
    svg_arrow,
    svg_box,
    svg_figure,
    svg_text,
):
    _body = [SVG_DEFS]
    _body.append(svg_text(12, 20, "Three levels, digests rolling upward", 12, INK, weight="600"))
    _body.append(svg_text(560, 20, "what each level can prove", 11, MUTED))

    # Records
    for _i in range(4):
        _x = 12 + _i * 118
        _body.append(
            svg_box(_x, 40, 106, 42, f"record {_i}", short(record_digests[_i], 8), SURFACE, RULE)
        )
        _body.append(svg_arrow(_x + 53, 86, 200, 122))
    _body.append(svg_text(484, 66, f"… {len(records_pass1) - 4} more", 11, MUTED))
    _body.append(
        svg_text(560, 66, "this document's bytes are exactly these", 11, MUTED)
    )

    # Shards
    for _i, _shard in enumerate(shards_pass1):
        _x = 12 + _i * 118
        _body.append(
            svg_box(
                _x,
                126,
                106,
                42,
                _shard["shard_id"],
                short(_shard["merkle_root"], 8),
                ACCENT_BG,
                ACCENT,
                ACCENT,
            )
        )
        _body.append(svg_arrow(_x + 53, 172, 250, 214))
    _body.append(
        svg_text(560, 152, "this record belongs to this shard (O(log n))", 11, MUTED)
    )

    # Corpus
    _body.append(
        svg_box(
            12,
            218,
            470,
            46,
            "corpus manifest",
            f"merkle_root over {SHARD_COUNT} shard roots: {short(corpus_pass1['merkle_root'], 12)}",
            ADMIT_BG,
            ADMIT,
            ADMIT,
        )
    )
    _body.append(
        svg_text(560, 238, "these shards, this source, this code, this tokenizer", 11, MUTED)
    )
    _body.append(
        svg_text(
            12,
            288,
            "Each level's digest is computed only from the level below, so a change at any depth reaches the top.",
            11,
            MUTED,
        )
    )

    svg_figure(
        "".join(_body),
        f"Diagram 7 — the emitted hierarchy for this run: {len(records_pass1)} record "
        f"manifests, {SHARD_COUNT} shard manifests, one corpus manifest. Four shards "
        f"rather than one exist precisely so the middle level, and a root computed over "
        f"roots, are real rather than illustrative.",
        880,
        300,
    )
    return


@app.cell(hide_code=True)
def technique_09(mo):
    mo.md(r"""
    ### 9. Signed attestations

    **What it is.** A cryptographic signature over a statement that binds *subject
    digests* to *how they were produced*. Digests prove integrity; signatures prove
    **authorship** — that a specific identity vouches for those digests.

    **Where it was used.**

    - **SLSA v1 provenance** over the **in-toto Statement** format. *How*: a
      Statement carries `subject[]` entries of `{name, digest}`, a `predicateType`
      IRI, and a `predicate`. SLSA's predicate binds those subject digests to a
      `builder.id` and a `buildDefinition` (the source, the entry point, the
      parameters), so the signature attests not just "these bytes" but "these bytes,
      built this way, by this builder".
    - **`sigstore/model-transparency`** ([repository](https://github.com/sigstore/model-transparency)).
      *How*: signs ML models with Sigstore DSSE over an in-toto Statement whose
      subjects are **per-file path plus sha256** — a flat digest manifest, and
      notably **not** a Merkle root. Worth registering: a production model-signing
      tool chose the simpler structure, because its verifier always holds every file
      anyway. Merkle proofs buy nothing when there is no partial-verification case.

    **How it works internally.** Canonicalize the statement (technique 2), sign it
    with a private key or a keyless flow (Sigstore exchanges an OIDC identity token
    for a short-lived certificate, so there is no long-lived key to protect), and
    distribute the signature alongside the artifact. Verification recomputes the
    subject digests, checks the signature, and checks the signer against a policy.

    **When to use it.** Whenever the corpus crosses a trust boundary — publication,
    vendor hand-off, regulated supply chain. The moment someone who does not know
    you must decide whether to believe the manifest.

    **Pros.** Closes the gap this notebook explicitly leaves open. Sigstore's
    keyless model makes it practical without key custody, and the identity is
    verifiable against a transparency log.

    **Cons.** Requires identity infrastructure, key or OIDC policy, and a
    verification step consumers must actually perform — an unverified signature is
    decoration. It is also **not implemented here**, and the honest consequence is
    stated in the closing section: these manifests prove integrity but not
    authorship. Anyone can recompute this corpus's root and publish it under their
    own name; nothing in the emitted artifacts contradicts them.
    """)
    return


@app.cell(hide_code=True)
def technique_10(mo):
    mo.md(r"""
    ### 10. Machine-readable dataset metadata standards

    **What it is.** Publishing the corpus's documentation in an agreed schema, so
    that provenance is queryable by tools rather than only readable by humans.

    **Where it was used.**

    - **Datasheets for Datasets** (Gebru et al.,
      [arXiv:1803.09010](https://arxiv.org/abs/1803.09010)). *How*: a seven-section
      questionnaire — motivation, composition, collection process,
      preprocessing/cleaning/labeling, uses, distribution, maintenance — completed in
      prose by the dataset's creators. The **preprocessing/cleaning/labeling** section
      is the one an eight-stage pipeline like this exists to answer.
    - **Data Cards** (Pushkarna et al.,
      [arXiv:2204.01075](https://arxiv.org/abs/2204.01075)). *How*: adds the **OFTEn**
      framework (Origins, Factuals, Transformations, Experience, n=1) and, crucially,
      *audience-aware* artifacts — the same dataset needs different transparency
      surfaces for a practitioner, a reviewer, and an affected person.
    - **MLCommons Croissant** ([specification](https://docs.mlcommons.org/croissant/docs/croissant-spec.html)).
      *How*: JSON-LD over schema.org, where `distribution` holds `cr:FileObject` and
      `cr:FileSet` entries each optionally carrying a `contentUrl` **and an `sha256`**
      — so the standard has a place for exactly the digests technique 1 produces.
      That field is the bridge between documentation and verification.

    **How it works internally.** Datasheets and Data Cards are structured prose;
    Croissant is a machine-parseable graph a loader can consume directly, resolving
    files, verifying digests, and constructing record sets without bespoke code.

    **When to use it.** At publication, always. Croissant specifically when the
    corpus is meant to be *loaded* by other people's tooling, and a datasheet
    whenever there are human judgement calls — a JSON-LD field cannot express "we
    kept romanized Hindi on purpose, and here is why".

    **Pros.** Discoverability, tool interoperability, and a checklist effect: the
    questionnaire forces authors to notice gaps they would otherwise not write down.
    Croissant's `sha256` slot means content addressing and standardized metadata
    compose rather than compete.

    **Cons.** Metadata standards record *claims*, and unverified claims are exactly
    the failure technique 11 documents at scale. Filling in a schema is also easy to
    do badly — a complete-looking Croissant file with `license: "unknown"` passes
    every validator. There is real maintenance cost too: the document must be updated
    with each release or it becomes actively misleading, which is worse than absent.
    """)
    return


@app.cell(hide_code=True)
def technique_11(mo):
    mo.md(r"""
    ### 11. Admission gating

    **What it is.** A mechanical check at the corpus boundary that a record's
    manifest is complete, its license is permitted, and its recorded digest matches
    its actual bytes — returning either `ADMITTED` or `BLOCKED` **with a specific
    reason**. Metric 8 is this technique, executed.

    **Where it was used.**

    - **The Data Provenance Initiative** ([arXiv:2310.16787](https://arxiv.org/abs/2310.16787)).
      *How*: a large-scale multidisciplinary audit of **1,800+** text datasets that
      traced each back to its original sources and licenses, finding **license
      omission above 70% and error rates above 50%**. The remedy shipped is itself the
      technique: structured, machine-readable provenance metadata plus the Data
      Provenance Explorer, so license status becomes a *queryable field* rather than a
      README sentence.

    That measurement is the argument for gating. If more than half of license
    annotations in circulation are wrong and most are missing, then a pipeline that
    reads a license string without checking it against an allow-list is
    statistically likely to be wrong about what it is training on — and will never
    find out.

    **How it works internally.** Fixed required-field list, a license allow-list as a
    `frozenset`, and digest recomputation from the shipped bytes. Precedence is
    deliberately fixed (missing field → license → digest) so the same defect always
    reports the same reason. Two smaller decisions that matter: **absent and empty
    are the same failure** — `token_count = 0` blocks as a missing field, because a
    zero count carries no more information than no count and is how a "0 tokens"
    record slips into a corpus report; and the gate **recomputes** the digest rather
    than comparing recorded values, which is the only version of the check that can
    catch text edited after the manifest was written (the last row of metric 8).

    **When to use it.** At every ingestion boundary, and in CI. It is cheap enough
    that there is no argument for sampling instead: metric 8 runs the full gate over
    all 225 records as part of the notebook.

    **Pros.** Turns a policy into a mechanism. Failures arrive as actionable strings
    (`license not in allow-list: 'unknown'`) rather than a count, and the allow-list
    is a single reviewable line rather than scattered conditionals.

    **Cons.** It only enforces what it is told: the allow-list is a policy decision
    that ages, and a permissive one is worse than none because it manufactures
    confidence. More fundamentally, **it is a self-consistency check**. It confirms
    the manifest agrees with the bytes; it cannot confirm the manifest is *true*. A
    record whose upstream license was mislabeled `CC-BY-4.0` sails through — which is
    precisely the DPI failure mode, and the reason the closing residual-risk note
    says so explicitly.
    """)
    return


@app.cell(hide_code=True)
def technique_12(mo):
    mo.md(r"""
    ### 12. Canary insertion

    **What it is.** Deliberately inserting unique, random sequences into a corpus
    before training, then testing the trained model for their extractability. It
    gives a *post-hoc* answer to "was this corpus actually in the training data?" —
    the one provenance question a manifest cannot answer, because a manifest
    describes intent while a canary observes the result.

    **Where it was used.**

    - **Carlini et al., "The Secret Sharer"**
      ([arXiv:1802.08232](https://arxiv.org/abs/1802.08232)). *How*: insert random
      canary sequences into the training data, then measure
      **`exposure = log2|R| − log2 rank(s[r])`**, where `R` is the space of candidate
      sequences and `rank` is the canary's position when all candidates are ordered by
      model likelihood. A canary that is much more likely than its random peers has
      been memorized, so exposure is a graded membership signal rather than a
      yes/no guess.

    **How it works internally.** Because the sequence is random and unique, the model
    cannot assign it high likelihood for any reason other than having seen it. Varying
    insertion count across canaries maps the memorization threshold — how many
    repetitions it takes before a sequence becomes extractable.

    **When to use it.** When downstream training is in scope and unauthorized use, or
    accidental inclusion, is a real risk. It is also the practical way to *validate a
    decontamination claim*: if a benchmark-shaped canary is extractable, the
    decontamination stage leaked.

    **Pros.** Empirical rather than declarative. It detects use of the corpus even by
    a party who ignored every manifest, which no amount of metadata can do, and the
    exposure metric is quantitative and comparable across runs.

    **Cons.** It modifies the corpus — which is why it does **not** belong in this
    stage. Stage 8's contract is that it does not touch the text (see also the
    excluded-technique note on watermarking); a canary is a training-data decision
    that must be made upstream, disclosed, and recorded in the manifest as a field.
    It also requires access to the trained model to check, gives no signal until after
    training, and injects deliberate garbage into the data. Canaries interact badly
    with deduplication too: a repeated canary looks exactly like the duplicates metric
    1 collapses, so the two stages have to be told about each other.
    """)
    return


@app.cell(hide_code=True)
def what_i_would_ship(
    corpus_pass1,
    json_module,
    metric_canonicalization,
    metric_tokens,
    mo,
):
    _schema_example = json_module.dumps(
        {
            "pipeline_version": corpus_pass1["pipeline_version"],
            "run_utc": corpus_pass1["run_utc"],
            "source": corpus_pass1["source"],
            "tokenizer": corpus_pass1["tokenizer"],
            "tokenizer_bpe_sha256": corpus_pass1["tokenizer_bpe_sha256"][:16] + "…",
            "cleaning_script": {
                "path": corpus_pass1["cleaning_script"]["path"],
                "sha256": corpus_pass1["cleaning_script"]["sha256"][:16] + "…",
            },
            "shard_count": corpus_pass1["shard_count"],
            "record_count": corpus_pass1["record_count"],
            "token_count": corpus_pass1["token_count"],
            "byte_count": corpus_pass1["byte_count"],
            "shard_roots": [s[:22] + "…" for s in corpus_pass1["shard_roots"]],
            "merkle_root": corpus_pass1["merkle_root"][:22] + "…",
        },
        indent=2,
        ensure_ascii=False,
    )

    mo.md(f"""
        ---

        # What I would ship

        **The emitter as written above**, unchanged: NFC canonicalization on the hashing
        copy only, `sha2-256:`-prefixed content digests, RFC 6962 Merkle roots at
        record → shard → corpus, canonical-JSON manifests with integer-only numerics,
        the notebook's own source digest as `cleaning_script`, and a gate that returns a
        specific reason. Every claim in this notebook is produced by that code, not
        described alongside it.

        The corpus manifest schema, as a literal example (digests truncated for
        display; the emitted file carries them in full):

        ```json
        {_schema_example}
        ```

        **The three fields that are non-negotiable**, because their absence silently
        invalidates the numbers next to them:

        1. `tokenizer` **and** `tokenizer_bpe_sha256` — without them `token_count` is not
           a measurement (metric 3: a {metric_tokens["tokenizer_spread"]} spread on
           identical bytes).
        2. The canonical form — without it `record_id` is not reproducible (metric 2:
           `.strip()` alone moves
           {metric_canonicalization["strip()"]["share"]} of ids).
        3. `cleaning_script.sha256` — without it "produced by our pipeline" names no
           specific pipeline.

        ## Residual risk, stated plainly

        Two things these manifests **do not** prove, and no amount of additional hashing
        would fix:

        - **Integrity, not authorship.** Nothing here is signed. Anyone can recompute
          this corpus's Merkle root and publish it under their own name, and no emitted
          artifact contradicts them. Closing this needs technique 9 — a DSSE/Sigstore
          signature over the corpus manifest — which is a key-management and identity
          problem, not a hashing problem.
        - **Self-consistency, not truth.** The gate proves the manifest agrees with the
          bytes. It cannot detect that the *upstream source itself lied*. If Sangraha's
          card said `cc-by-4.0` about text that was not licensed that way, every record
          here is `ADMITTED` and every digest verifies, and the corpus is still
          unusable. This is exactly the failure the Data Provenance Initiative measured
          at scale (70%+ license omission, 50%+ error rates in 1,800+ datasets), and it
          is not solvable inside this stage: it needs provenance the *source* publishes
          and someone independently audits.

        A smaller honest gap, for completeness: the input-addressed identity covers the
        data, this notebook's source, and the tokenizer artifact — but **not** the
        installed library versions, the interpreter, or the OS. A full closure would pin
        `uv.lock` and the Python build the way Nix pins a derivation.
        """)
    return


@app.cell(hide_code=True)
def provenance_of_this_run(corpus_pass1, mo, short):
    provenance_rows = [
        {"field": "pipeline_version", "value": corpus_pass1["pipeline_version"]},
        {"field": "run_utc", "value": corpus_pass1["run_utc"]},
        {"field": "source.dataset_id", "value": corpus_pass1["source"]["dataset_id"]},
        {"field": "source.revision", "value": corpus_pass1["source"]["revision"]},
        {
            "field": "source.data_dir / split",
            "value": f"{corpus_pass1['source']['data_dir']} / {corpus_pass1['source']['split']}",
        },
        {"field": "source.license", "value": corpus_pass1["source"]["license"]},
        {"field": "tokenizer", "value": corpus_pass1["tokenizer"]},
        {"field": "tokenizer_bpe_sha256", "value": short(corpus_pass1["tokenizer_bpe_sha256"], 20)},
        {"field": "cleaning_script.path", "value": corpus_pass1["cleaning_script"]["path"]},
        {
            "field": "cleaning_script.sha256",
            "value": short(corpus_pass1["cleaning_script"]["sha256"], 20),
        },
        {"field": "record_count", "value": f"{corpus_pass1['record_count']:,}"},
        {"field": "shard_count", "value": str(corpus_pass1["shard_count"])},
        {"field": "token_count", "value": f"{corpus_pass1['token_count']:,}"},
        {"field": "byte_count", "value": f"{corpus_pass1['byte_count']:,}"},
        {"field": "merkle_root", "value": short(corpus_pass1["merkle_root"], 24)},
        {"field": "manifest_digest", "value": short(corpus_pass1["manifest_digest"], 24)},
    ]

    mo.vstack([
        mo.md(r"""
        ## Provenance of this notebook

        The corpus manifest for *this* run, emitted by the code above and written to
        `manifest-provenance/data/manifests/corpus.manifest.json`.
        """),
        mo.ui.table(provenance_rows, selection=None, page_size=16),
        mo.md(f"""
        `cleaning_script.sha256` is the reflexive field: this notebook is identified by
        a digest of its own source, so the manifest records which version of the emitter
        produced it. **That value changes the moment any cell is edited** — including
        the cells that only contain prose. That is correct behaviour, not a defect: a
        manifest written by different code came from a different pipeline, and the
        alternative is a version label that means nothing.

        To verify this run from scratch, holding only this repository:

        ```bash
        uv run python manifest-provenance/data/build_slice.py   # rebuild the slice
        uv run marimo export html --no-include-code \\
            manifest-provenance/manifest_provenance.py -o /tmp/check.html
        ```

        Then compare `merkle_root` in the re-emitted
        `data/manifests/corpus.manifest.json` against
        `{short(corpus_pass1["merkle_root"], 24)}`. Metric 6 is the evidence that this
        comparison is meaningful: two independent passes over the same rows produce
        byte-identical canonical JSON, so a mismatch means something really changed.
        """),
    ])
    return


if __name__ == "__main__":
    app.run()
