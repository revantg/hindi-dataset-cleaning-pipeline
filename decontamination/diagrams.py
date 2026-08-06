"""Hand-authored SVG diagrams for the decontamination workbook.

Each function returns a complete ``<svg>`` string. No rendering library is
involved: the coordinates are written by hand so the diagrams label the
mechanism rather than decorate it.

Palette is shared across every figure so the same colour always means the same
thing: ``INK`` for structure, ``TRAIN`` for material that may be trained on,
``HELD`` for material behind the firewall, ``WARN`` for a detected leak, and
``MUTE`` for annotation.
"""

from __future__ import annotations

INK = "#1f2933"
MUTE = "#7b8794"
LINE = "#cbd2d9"
PANEL = "#f7f9fa"
TRAIN = "#2b6cb0"
TRAIN_SOFT = "#e3edf8"
HELD = "#2f855a"
HELD_SOFT = "#e2f0e8"
WARN = "#b7472a"
WARN_SOFT = "#fbe9e4"
GOLD = "#8a6d1f"
GOLD_SOFT = "#f7f0dc"

_FONT = (
    "font-family=\"ui-sans-serif,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',sans-serif\""
)


def _svg(width: int, height: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img" aria-label="{title}" {_FONT}>'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'
        f"{body}</svg>"
    )


def _text(x: float, y: float, content: str, size: int = 12, fill: str = INK,
          weight: str = "400", anchor: str = "start", mono: bool = False) -> str:
    font = ' font-family="ui-monospace,SFMono-Regular,Menlo,monospace"' if mono else ""
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}"{font}>{content}</text>'
    )


def _box(x: float, y: float, w: float, h: float, fill: str, stroke: str,
         radius: int = 6, dash: str = "", width: float = 1.2) -> str:
    dasharray = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dasharray}/>'
    )


def _arrow_defs() -> str:
    return (
        "<defs>"
        '<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTE}"/></marker>'
        '<marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{WARN}"/></marker>'
        "</defs>"
    )


def _arrow(x1: float, y1: float, x2: float, y2: float, colour: str = MUTE,
           marker: str = "ar", dash: str = "") -> str:
    dasharray = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" '
        f'stroke-width="1.4" marker-end="url(#{marker})"{dasharray}/>'
    )


# ---------------------------------------------------------------------------
# 1. the firewall
# ---------------------------------------------------------------------------


def firewall_diagram() -> str:
    """The sourcing-time three-way split and the train/test asymmetry."""
    body = [_arrow_defs()]
    body.append(_text(24, 30, "Decontamination is a sourcing decision, not a cleaning pass",
                      15, INK, "600"))
    body.append(_text(24, 50,
                      "The wall is placed once, when data is admitted. Everything downstream only enforces it.",
                      11.5, MUTE))

    # left column: trainable
    body.append(_box(24, 68, 330, 250, PANEL, LINE))
    body.append(_text(40, 92, "MAY TRAIN ON", 11, TRAIN, "700"))

    body.append(_box(40, 104, 298, 74, TRAIN_SOFT, TRAIN))
    body.append(_text(54, 126, "Pretraining pools", 12.5, INK, "600"))
    body.append(_text(54, 144, "Web crawl, code, PDFs, transcripts.", 11, INK))
    body.append(_text(54, 161, "Bulk of the corpus. Never seen by the evaluator.", 10.5, MUTE))

    body.append(_box(40, 190, 298, 106, TRAIN_SOFT, TRAIN, dash="4 3"))
    body.append(_text(54, 212, "Benchmark TRAIN splits", 12.5, INK, "600"))
    body.append(_text(54, 230, "Teaches the task <tspan font-style=\"italic\">format</tspan>: how a question is posed,", 11, INK))
    body.append(_text(54, 246, "what an answer looks like, how options are laid out.", 11, INK))
    body.append(_text(54, 268, "Fair game. Format familiarity is a capability,", 10.5, HELD))
    body.append(_text(54, 283, "not an answer key. This is the asymmetry.", 10.5, HELD))

    # the wall
    wall_x = 372
    body.append(f'<rect x="{wall_x}" y="68" width="14" height="250" rx="3" '
                f'fill="{WARN_SOFT}" stroke="{WARN}" stroke-width="1.6"/>')
    for offset in range(76, 316, 16):
        body.append(f'<line x1="{wall_x + 2}" y1="{offset}" x2="{wall_x + 12}" y2="{offset + 10}" '
                    f'stroke="{WARN}" stroke-width="1"/>')
    body.append(f'<text x="{wall_x + 7}" y="336" font-size="10.5" fill="{WARN}" '
                f'font-weight="700" text-anchor="middle">HELD-OUT FIREWALL</text>')

    # right column: never train
    body.append(_box(404, 68, 330, 250, PANEL, LINE))
    body.append(_text(420, 92, "MUST NEVER TRAIN ON", 11, HELD, "700"))

    body.append(_box(420, 104, 298, 130, HELD_SOFT, HELD))
    body.append(_text(434, 126, "Benchmark TEST / VALIDATION splits", 12.5, INK, "600"))
    body.append(_text(434, 146, "The questions and the answer key.", 11, INK))
    body.append(_text(434, 168, "This workbook holds three behind the wall:", 10.5, MUTE))
    body.append(_text(434, 186, "IndicQA hi · MILU Hindi · XNLI hi", 11, INK, "500", mono=True))
    body.append(_text(434, 210, "Used only to measure. Fingerprinted on day one", 10.5, HELD))
    body.append(_text(434, 225, "so every later shard can be checked against them.", 10.5, HELD))

    body.append(_box(420, 246, 298, 62, GOLD_SOFT, GOLD))
    body.append(_text(434, 268, "Why it matters", 11.5, INK, "600"))
    body.append(_text(434, 285, "A leak does not make the model better. It removes", 10.5, INK))
    body.append(_text(434, 300, "the only instrument that could tell you whether it is.", 10.5, INK))

    body.append(_text(24, 372, "If the wall holds:", 11, HELD, "700"))
    body.append(_text(132, 372, "reported score = true ability. The score is information.", 11, INK))
    body.append(_text(24, 394, "If the wall breaks:", 11, WARN, "700"))
    body.append(_text(138, 394,
                      "reported score = true ability + memorisation. The score is noise.",
                      11, INK))

    return _svg(760, 414, "".join(body), "Held-out firewall: what may and may not be trained on")


# ---------------------------------------------------------------------------
# 2. n-gram overlap internals
# ---------------------------------------------------------------------------


def ngram_diagram() -> str:
    """Shingling, the hash set, and what a collision does and does not prove."""
    body = [_arrow_defs()]
    body.append(_text(24, 30, "Exact n-gram overlap: how a leak is actually found", 15, INK, "600"))
    body.append(_text(24, 50,
                      "Both sides are normalised, cut into overlapping n-token windows, hashed, "
                      "and intersected.", 11.5, MUTE))

    # held-out side
    body.append(_box(24, 70, 320, 156, HELD_SOFT, HELD))
    body.append(_text(38, 92, "1 · Held-out side, built once", 11.5, INK, "600"))
    body.append(_text(38, 112, "eval item:  A B C D E F G H", 11, INK, mono=True))
    for index, label in enumerate(["A B C D", "B C D E", "C D E F", "D E F G", "E F G H"]):
        y = 130 + index * 18
        body.append(_box(38, y - 12, 118, 18, "#ffffff", HELD, radius=3, width=0.9))
        body.append(_text(48, y, label, 10, INK, mono=True))
        body.append(_text(168, y, f"→ hash h{index + 1}", 10, MUTE, mono=True))
    body.append(_text(250, 213, "n = 4 shown", 10, MUTE))

    # hash set
    body.append(_box(384, 96, 150, 96, GOLD_SOFT, GOLD))
    body.append(_text(459, 118, "hash set", 11.5, INK, "600", anchor="middle"))
    body.append(_text(459, 136, "{h1 … h5}", 11, INK, anchor="middle", mono=True))
    body.append(_text(459, 158, "O(1) membership", 10, MUTE, anchor="middle"))
    body.append(_text(459, 174, "one linear pass to build", 10, MUTE, anchor="middle"))
    body.append(_arrow(344, 144, 380, 144))

    # corpus side
    body.append(_box(24, 244, 320, 138, TRAIN_SOFT, TRAIN))
    body.append(_text(38, 266, "2 · Corpus side, streamed", 11.5, INK, "600"))
    body.append(_text(38, 286, "document:  X Y C D E F Z W", 11, INK, mono=True))
    for index, (label, hit) in enumerate(
        [("X Y C D", False), ("Y C D E", False), ("C D E F", True), ("D E F Z", False)]
    ):
        y = 304 + index * 18
        fill = WARN_SOFT if hit else "#ffffff"
        stroke = WARN if hit else TRAIN
        body.append(_box(38, y - 12, 118, 18, fill, stroke, radius=3, width=0.9))
        body.append(_text(48, y, label, 10, INK, mono=True))
        if hit:
            body.append(_text(168, y, "→ IN SET · flag", 10, WARN, "700", mono=True))
    body.append(_arrow(344, 308, 380, 200, WARN, "arw"))

    # verdict
    body.append(_box(384, 244, 350, 138, PANEL, LINE))
    body.append(_text(398, 266, "3 · What the collision means", 11.5, INK, "600"))
    body.append(_text(398, 290, "One shared n-gram ⇒ the document is flagged.", 11, INK))
    body.append(_text(398, 310, "GPT-3 uses exactly this rule at n = 13.", 11, INK))
    body.append(_text(398, 337, "It proves the surface forms coincide.", 10.5, WARN))
    body.append(_text(398, 353, "It does not prove a leak: at small n, ordinary", 10.5, WARN))
    body.append(_text(398, 369, "idiom collides too. Precision is an audit, not a default.", 10.5, WARN))

    body.append(_text(24, 414, "The parameter n is the whole trade-off.", 11.5, INK, "600"))
    body.append(_text(24, 434,
                      "Small n catches paraphrase-adjacent overlap and floods on common phrasing. "
                      "Large n is nearly", 11, INK))
    body.append(_text(24, 452,
                      "always a real leak but cannot see any evaluation item shorter than n tokens.",
                      11, INK))

    return _svg(760, 470, "".join(body), "Internals of exact n-gram overlap detection")


# ---------------------------------------------------------------------------
# 3. the matra trap
# ---------------------------------------------------------------------------


def matra_trap_diagram() -> str:
    """Why an English-shaped normalizer fabricates Hindi overlap."""
    body = [_arrow_defs()]
    body.append(_text(24, 30, "The normalisation trap that decides the whole measurement",
                      15, INK, "600"))
    body.append(_text(24, 50,
                      "Devanagari vowel signs and the virama are Unicode combining marks, "
                      "not punctuation.", 11.5, MUTE))

    # character table
    body.append(_box(24, 68, 710, 102, PANEL, LINE))
    columns = [
        ("प", "U+092A", "Lo", "letter", False),
        ("्", "U+094D", "Mn", "virama", True),
        ("र", "U+0930", "Lo", "letter", False),
        ("द", "U+0926", "Lo", "letter", False),
        ("र", "U+0930", "Lo", "letter", False),
        ("्", "U+094D", "Mn", "virama", True),
        ("श", "U+0936", "Lo", "letter", False),
        ("न", "U+0928", "Lo", "letter", False),
        ("ो", "U+094B", "Mc", "vowel sign", True),
        ("ं", "U+0902", "Mn", "anusvara", True),
    ]
    for index, (glyph, code, category, role, is_mark) in enumerate(columns):
        x = 40 + index * 69
        fill = WARN_SOFT if is_mark else "#ffffff"
        stroke = WARN if is_mark else LINE
        body.append(_box(x, 80, 60, 76, fill, stroke, radius=4))
        body.append(_text(x + 30, 104, glyph, 17, INK, anchor="middle"))
        body.append(_text(x + 30, 120, code, 8.5, MUTE, anchor="middle", mono=True))
        body.append(_text(x + 30, 134, category, 9.5,
                          WARN if is_mark else INK, "700", anchor="middle", mono=True))
        body.append(_text(x + 30, 147, role, 8, MUTE, anchor="middle"))
    body.append(_text(24, 190, "प्रदर्शनों — ten code points, four of which are combining marks (Mn/Mc).",
                      11, INK))

    # two normalizers
    body.append(_box(24, 206, 350, 148, WARN_SOFT, WARN))
    body.append(_text(38, 228, "Naive: re.sub(r\"[^\\w\\s]\", \" \", text)", 11.5, INK, "600", mono=True))
    body.append(_text(38, 250, "\\w matches letters and digits. Combining marks", 11, INK))
    body.append(_text(38, 266, "are neither, so every one of them is deleted.", 11, INK))
    body.append(_text(38, 292, "इन प्रदर्शनों में भाग लेने वालों पर", 13, MUTE))
    body.append(_arrow(190, 304, 190, 316, WARN, "arw"))
    body.append(_text(38, 336, "इन प रदर शन म भ ग ल न व ल पर", 13, WARN, "600"))

    body.append(_box(384, 206, 350, 148, HELD_SOFT, HELD))
    body.append(_text(398, 228, "Category-aware: keep Mn and Mc", 11.5, INK, "600", mono=True))
    body.append(_text(398, 250, "Strip only what is genuinely punctuation, judged", 11, INK))
    body.append(_text(398, 266, "by Unicode general category rather than by \\w.", 11, INK))
    body.append(_text(398, 292, "इन प्रदर्शनों में भाग लेने वालों पर", 13, MUTE))
    body.append(_arrow(550, 304, 550, 316, HELD))
    body.append(_text(398, 336, "इन प्रदर्शनों में भाग लेने वालों पर", 13, HELD, "600"))

    # consequence
    body.append(_box(24, 374, 710, 96, PANEL, LINE))
    body.append(_text(38, 398, "Why this is a decontamination bug and not a cosmetic one",
                      12, INK, "600"))
    body.append(_text(38, 420,
                      "Stripping marks collapses distinct words onto the same consonant skeleton. "
                      "Unrelated Hindi", 11, INK))
    body.append(_text(38, 438,
                      "documents then share long n-grams that never existed, and the scan reports "
                      "leaks that are not there.", 11, INK))
    body.append(_text(38, 460,
                      "Measured on this probe at n = 13: 822 flags under the naive rule versus 6 "
                      "under the correct one.", 11, WARN, "600"))

    return _svg(760, 484, "".join(body), "Devanagari combining marks and the normalisation trap")


# ---------------------------------------------------------------------------
# 4. MinHash / LSH banding
# ---------------------------------------------------------------------------


def minhash_diagram() -> str:
    """Banding, the S-curve, and the unit-size mismatch that defeats it here."""
    body = [_arrow_defs()]
    body.append(_text(24, 30, "MinHash and LSH banding: buying a similarity threshold",
                      15, INK, "600"))
    body.append(_text(24, 50,
                      "Signatures estimate Jaccard; banding turns a threshold into a collision "
                      "probability.", 11.5, MUTE))

    # signature
    body.append(_text(24, 82, "128-value signature, split into 32 bands of 4 rows", 11.5, INK, "600"))
    for band in range(7):
        x = 24 + band * 88
        body.append(_box(x, 92, 78, 62, TRAIN_SOFT if band != 3 else GOLD_SOFT,
                         TRAIN if band != 3 else GOLD))
        for row in range(4):
            body.append(_text(x + 39, 108 + row * 12, f"m{band * 4 + row + 1}", 9,
                              INK, anchor="middle", mono=True))
        body.append(_text(x + 39, 168, f"band {band + 1}", 9.5, MUTE, anchor="middle"))
    body.append(_box(640, 92, 94, 62, PANEL, LINE, dash="4 3"))
    body.append(_text(687, 128, "… 25 more", 10.5, MUTE, anchor="middle"))
    body.append(_text(687, 168, "bands 8–32", 9.5, MUTE, anchor="middle"))
    body.append(_text(24, 186,
                      "Two texts become candidates if all 4 rows of any one band are identical. "
                      "P(collision) = 1 − (1 − Jᵣ)ᵇ.", 11, INK))

    # s-curve
    body.append(_box(24, 204, 350, 212, PANEL, LINE))
    body.append(_text(38, 226, "The S-curve at b = 32, r = 4", 11.5, INK, "600"))
    ox, oy, w, h = 62, 356, 272, 118
    body.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + w}" y2="{oy}" stroke="{MUTE}" stroke-width="1"/>')
    body.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy - h}" stroke="{MUTE}" stroke-width="1"/>')
    points = []
    for step in range(101):
        similarity = step / 100
        probability = 1.0 - (1.0 - similarity**4) ** 32
        points.append(f"{ox + similarity * w:.1f},{oy - probability * h:.1f}")
    body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{TRAIN}" stroke-width="2"/>')
    for value in (0.0, 0.5, 1.0):
        body.append(_text(ox + value * w, oy + 15, f"{value:.1f}", 9.5, MUTE, anchor="middle"))
    body.append(_text(ox - 8, oy + 4, "0", 9.5, MUTE, anchor="end"))
    body.append(_text(ox - 8, oy - h + 4, "1", 9.5, MUTE, anchor="end"))
    body.append(_text(ox + w / 2, oy + 32, "Jaccard similarity", 10, MUTE, anchor="middle"))
    body.append(f'<line x1="{ox + 0.5 * w}" y1="{oy}" x2="{ox + 0.5 * w}" y2="{oy - h}" '
                f'stroke="{GOLD}" stroke-width="1" stroke-dasharray="3 3"/>')
    body.append(_text(ox + 0.5 * w + 6, oy - h + 16, "J = 0.5 → P = 0.87", 9.5, GOLD, "600"))
    body.append(_text(38, 402, "Sharp transition: near-duplicates collide, unrelated text does not.",
                      10, MUTE))

    # the failure mode
    body.append(_box(384, 204, 350, 212, WARN_SOFT, WARN))
    body.append(_text(398, 226, "Why it found nothing on this corpus", 11.5, INK, "600"))
    body.append(_box(398, 240, 322, 46, "#ffffff", MUTE, radius=4))
    body.append(_text(410, 258, "corpus document ≈ 2,300 tokens", 10.5, INK, mono=True))
    body.append(_box(410, 263, 32, 17, GOLD_SOFT, GOLD, radius=3))
    body.append(_text(426, 275, "15", 9, GOLD, "700", anchor="middle", mono=True))
    body.append(_text(448, 275, "← the evaluation item, to scale", 9.5, MUTE))
    body.append(_text(398, 308, "A MILU stem is 15 tokens. Document-level Jaccard", 11, INK))
    body.append(_text(398, 324, "against it is ≈ 0.006 — no threshold reaches that,", 11, INK))
    body.append(_text(398, 340, "so the scan is silent even on a verbatim leak.", 11, INK))
    body.append(_text(398, 368, "Fix: slide a comparable-length window, or switch", 10.5, HELD))
    body.append(_text(398, 384, "from Jaccard to containment. Unit size is the bug.", 10.5, HELD))

    return _svg(760, 428, "".join(body), "MinHash LSH banding and the unit-size failure mode")


# ---------------------------------------------------------------------------
# 5. clean / dirty inflation
# ---------------------------------------------------------------------------


def inflation_diagram() -> str:
    """How a reachable-item count becomes a reported-score delta."""
    body = [_arrow_defs()]
    body.append(_text(24, 30, "From a hit count to a score delta: the clean/dirty partition",
                      15, INK, "600"))
    body.append(_text(24, 50,
                      "Flagging documents is not the deliverable. Partitioning the exam is.",
                      11.5, MUTE))

    # partition bar
    body.append(_text(24, 84, "IndicQA hi · 261 held-out items", 11.5, INK, "600"))
    total_w = 700
    dirty_w = total_w * 7 / 261
    body.append(_box(24, 94, total_w, 40, HELD_SOFT, HELD))
    body.append(_box(24, 94, dirty_w, 40, WARN_SOFT, WARN))
    body.append(_text(24 + dirty_w + 14, 119, "254 clean items — unreachable from the corpus",
                      11, INK))
    body.append(_text(24, 152, "7 reachable", 10.5, WARN, "700"))
    body.append(_arrow(38, 142, 30, 136, WARN, "arw"))

    # arithmetic
    body.append(_box(24, 170, 342, 158, PANEL, LINE))
    body.append(_text(38, 194, "The arithmetic", 11.5, INK, "600"))
    body.append(_text(38, 218, "assume true ability", 11, INK))
    body.append(_text(300, 218, "41.0%", 11, INK, "600", anchor="end", mono=True))
    body.append(_text(38, 240, "assume memorised items are answered", 11, INK))
    body.append(_text(300, 240, "100%", 11, INK, "600", anchor="end", mono=True))
    body.append(f'<line x1="38" y1="252" x2="300" y2="252" stroke="{LINE}" stroke-width="1"/>')
    body.append(_text(38, 274, "reported = (254·0.41 + 7·1.00) / 261", 11, INK, mono=True))
    body.append(_text(38, 298, "reported", 11.5, INK, "600"))
    body.append(_text(300, 298, "42.6%", 12, WARN, "700", anchor="end", mono=True))
    body.append(_text(38, 318, "inflation", 11.5, INK, "600"))
    body.append(_text(300, 318, "+1.6 points", 12, WARN, "700", anchor="end", mono=True))

    # interpretation
    body.append(_box(382, 170, 342, 158, GOLD_SOFT, GOLD))
    body.append(_text(396, 194, "How to read it", 11.5, INK, "600"))
    body.append(_text(396, 218, "Both inputs are assumptions and are labelled as such.", 10.8, INK))
    body.append(_text(396, 234, "The output is the arithmetic that follows from them,", 10.8, INK))
    body.append(_text(396, 250, "which is what GPT-3 and PaLM both report.", 10.8, INK))
    body.append(_text(396, 276, "1.6 points is small — and that is the finding.", 10.8, WARN, "600"))
    body.append(_text(396, 292, "It is also larger than the margin most model", 10.8, INK))
    body.append(_text(396, 308, "comparisons are decided by, so it is not ignorable.", 10.8, INK))

    body.append(_text(24, 360,
                      "The stage's real output is a bounded statement about how far the benchmark "
                      "can be trusted,", 11, INK))
    body.append(_text(24, 378,
                      "not a smaller corpus. At n = 13 only 0.39% of tokens are even implicated.",
                      11, INK))

    return _svg(760, 396, "".join(body), "Clean/dirty partition and score inflation")


DIAGRAMS = {
    "firewall": firewall_diagram,
    "ngram": ngram_diagram,
    "matra": matra_trap_diagram,
    "minhash": minhash_diagram,
    "inflation": inflation_diagram,
}
