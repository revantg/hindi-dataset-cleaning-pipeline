"""Hand-authored SVG diagrams for the PII-scrubbing technique study.

Every diagram is a plain SVG string: no rendering library, no build step, no
runtime dependency. They are written to be read at the notebook's width, to use
one shared palette, and to label the *mechanism* rather than decorate the page.

Two implementation rules learned the hard way:

* Devanagari is written as literal UTF-8 text, never as numeric character
  references. Hand-computed `&#NNNN;` codepoints silently produce dandas and
  combining marks instead of digits.
* Body text goes through `_lines()`, which wraps on words against a measured
  character budget. SVG does not wrap text, so any unwrapped string longer than
  its panel silently overflows the border.
"""

from __future__ import annotations

from html import escape

# --- shared palette -------------------------------------------------------
INK = "#12213b"
MUTED = "#5c6b84"
LINE = "#c3cddd"
PANEL = "#f6f8fc"
EDGE = "#dbe3ef"

GREEN, GREEN_BG = "#1c8b5a", "#e4f5ec"      # correctly masked PII
AMBER, AMBER_BG = "#a8690a", "#fdf1dc"      # false positive / suppressed
RED, RED_BG = "#c02d33", "#fdeaea"          # missed PII, still exposed
BLUE, BLUE_BG = "#245bbf", "#e8effc"        # mechanism / data flow
VIOLET, VIOLET_BG = "#6c40b5", "#efe9fb"    # model-based layer

FONT = "font-family=\"ui-sans-serif,-apple-system,'Segoe UI',Roboto,'Noto Sans Devanagari',sans-serif\""
MONO = "font-family=\"ui-monospace,'SF Mono',Menlo,Consolas,monospace\""

# Approximate advance width per character, as a fraction of font-size. Used only
# to decide where to break lines, so a rough figure is adequate.
_SANS_RATIO = 0.52
_MONO_RATIO = 0.60


def _wrap(text: str, width_px: float, font_size: float, mono: bool = False) -> list[str]:
    """Greedy word wrap against an approximate pixel budget."""
    budget = max(8, int(width_px / (font_size * (_MONO_RATIO if mono else _SANS_RATIO))))
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= budget or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _lines(
    text: str,
    x: float,
    y: float,
    width_px: float,
    font_size: float = 12.0,
    fill: str = MUTED,
    leading: float = 1.45,
    mono: bool = False,
    weight: int = 400,
) -> str:
    """Emit wrapped <text> rows, returning SVG markup."""
    face = MONO if mono else FONT
    out = []
    for index, line in enumerate(_wrap(text, width_px, font_size, mono)):
        out.append(
            f'  <text x="{x}" y="{y + index * font_size * leading:.1f}" {face} '
            f'font-size="{font_size}" font-weight="{weight}" fill="{fill}">{escape(line)}</text>'
        )
    return "\n".join(out)


# Every diagram passes the y of its lowest element as `height`; the frame adds the
# bottom margin so no panel border sits flush against the canvas edge.
BOTTOM_PAD = 18


def _frame(width: int, height: int, body: str, title: str, subtitle: str) -> str:
    subtitle_rows = _lines(subtitle, 26, 55, width - 60, 13, MUTED, leading=1.35)
    canvas = height + BOTTOM_PAD
    return f"""
<svg viewBox="0 0 {width} {canvas}" width="100%" role="img"
     aria-label="{escape(title)}" xmlns="http://www.w3.org/2000/svg"
     style="max-width:{width}px;height:auto;display:block;margin:0 auto">
  <rect x="0" y="0" width="{width}" height="{canvas}" fill="#ffffff"/>
  <text x="26" y="34" {FONT} font-size="19" font-weight="700" fill="{INK}">{escape(title)}</text>
{subtitle_rows}
{body}
</svg>
""".strip()


def _arrow_right(x: float, y: float, length: float, colour: str) -> str:
    return (
        f'  <path d="M{x} {y} L{x + length - 6} {y}" stroke="{colour}" stroke-width="2"/>\n'
        f'  <path d="M{x + length - 7} {y - 5} L{x + length} {y} L{x + length - 7} {y + 5} Z" fill="{colour}"/>'
    )


def _arrow_down(x: float, y: float, length: float, colour: str, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    return (
        f'  <path d="M{x} {y} L{x} {y + length - 6}" stroke="{colour}" stroke-width="2"{dash}/>\n'
        f'  <path d="M{x - 5} {y + length - 7} L{x} {y + length} L{x + 5} {y + length - 7} Z" fill="{colour}"/>'
    )


# =========================================================================
# 1. Where the stage sits, and why it splits into two layers
# =========================================================================
def pipeline_position() -> str:
    stages = [
        ("1", "Extract"), ("2", "Normalize"), ("3", "Language ID"), ("4", "Quality"),
        ("5", "Dedup"), ("6", "PII scrub"), ("7", "Decontaminate"), ("8", "Manifest"),
    ]
    body: list[str] = []
    x, y, w, gap = 26, 84, 132, 10
    for index, (number, name) in enumerate(stages):
        left = x + index * (w + gap)
        active = number == "6"
        body.append(f"""
  <rect x="{left}" y="{y}" width="{w}" height="52" rx="8"
        fill="{BLUE_BG if active else PANEL}" stroke="{BLUE if active else EDGE}"
        stroke-width="{2.5 if active else 1}"/>
  <text x="{left + 11}" y="{y + 21}" {MONO} font-size="10.5"
        fill="{BLUE if active else MUTED}" font-weight="700">STAGE {number}</text>
  <text x="{left + 11}" y="{y + 40}" {FONT} font-size="13"
        fill="{INK}" font-weight="{700 if active else 500}">{name}</text>""")
        if index < len(stages) - 1:
            body.append(
                f'  <path d="M{left + w + 1} {y + 26} L{left + w + gap - 1} {y + 26}" '
                f'stroke="{LINE}" stroke-width="1.5"/>'
            )

    cx = 26 + 5 * (w + gap) + w / 2
    body.append(_arrow_down(cx, y + 54, 26, BLUE, dashed=True))

    panel_y = 176
    body.append(f"""
  <rect x="26" y="{panel_y}" width="1148" height="158" rx="10" fill="#ffffff" stroke="{EDGE}"/>
  <text x="46" y="{panel_y + 27}" {FONT} font-size="14" font-weight="700" fill="{INK}">The stage splits by whether the identifier has a fixed shape</text>""")

    branches = [
        (46, GREEN, GREEN_BG, "Structured \u2192 regular expressions",
         "email \u00b7 phone \u00b7 IP \u00b7 Aadhaar \u00b7 PAN \u00b7 card \u00b7 API key",
         "A fixed grammar means a deterministic decision. Precision is near-perfect, and a checksum can turn a shape match into arithmetic proof."),
        (618, VIOLET, VIOLET_BG, "Unstructured \u2192 ML / NER",
         "personal names \u00b7 addresses \u00b7 quasi-identifiers",
         "No fixed shape, so the decision needs context and therefore a model. This is where precision is spent, and Indic names spend it fastest."),
    ]
    for left, colour, fill, heading, examples, why in branches:
        body.append(f"""
  <rect x="{left}" y="{panel_y + 42}" width="536" height="100" rx="9" fill="{fill}" stroke="{colour}" stroke-width="1.6"/>
  <text x="{left + 18}" y="{panel_y + 66}" {FONT} font-size="14" font-weight="700" fill="{colour}">{heading}</text>
  <text x="{left + 18}" y="{panel_y + 88}" {FONT} font-size="12.5" fill="{INK}">{examples}</text>
{_lines(why, left + 18, panel_y + 110, 502, 12, MUTED)}""")

    body.append(f"""
  <rect x="26" y="354" width="1148" height="66" rx="8" fill="{RED_BG}" stroke="{RED}" stroke-width="1.4"/>
  <text x="46" y="376" {FONT} font-size="12.5" font-weight="700" fill="{RED}">What went wrong in V4</text>
{_lines("Dolma applied a regex PII pass to the English pool. The Indic pipeline had no PII pass at all \u2014 the one setting where a name model would struggle most, so the worst possible place to skip the stage.", 46, 396, 1108, 12.5, INK)}""")

    return _frame(
        1200, 436, "\n".join(body),
        "Stage 6 of eight, and the two layers inside it",
        "PII scrubbing runs after language validation and before decontamination. Its internal split is by identifier shape, not by importance.",
    )


# =========================================================================
# 2. The cascade: how a span is decided
# =========================================================================
def cascade_internals() -> str:
    body: list[str] = []
    # This exact document is run through pii_techniques.scrub() and produces the
    # three outcomes drawn at the bottom of the figure:
    #   ९८४५०१२३४५     -> accepted phone_in, 0.95   (visible only after folding)
    #   9999 4105 7058 -> accepted aadhaar_like, 1.0 (Verhoeff passes)
    #   २.९.२१.१       -> suppressed ipv4          (native digits => citation)
    sample = (
        "\u092e\u094b\u092c\u093e\u0907\u0932 \u096f\u096e\u096a\u096b\u0966\u0967\u0968\u0969\u096a\u096b \u092a\u0930"
        "   \u00b7   \u0906\u0927\u093e\u0930 9999 4105 7058"
        "   \u00b7   \u0927\u0930\u094d\u092e\u0938\u0942\u0924\u094d\u0930 (\u0968.\u096f.\u0968\u0967.\u0967)"
    )

    body.append(f"""
  <rect x="26" y="80" width="1148" height="66" rx="9" fill="{PANEL}" stroke="{EDGE}"/>
  <text x="42" y="103" {MONO} font-size="10.5" font-weight="700" fill="{MUTED}">INPUT DOCUMENT \u00b7 the original string, never mutated during detection</text>
  <text x="42" y="131" {MONO} font-size="14" fill="{INK}">{sample}</text>""")

    fold_y = 162
    body.append(f"""
  <rect x="26" y="{fold_y}" width="1148" height="66" rx="9" fill="{BLUE_BG}" stroke="{BLUE}" stroke-width="1.6"/>
  <text x="42" y="{fold_y + 22}" {FONT} font-size="13" font-weight="700" fill="{BLUE}">LAYER 0 \u00b7 Unicode digit folding onto a DETECTION COPY</text>
{_lines("One code point per code point, so every offset in the folded copy addresses the same character in the original. Detection reads the copy; masking writes the original.", 42, fold_y + 42, 800, 12, INK)}
  <text x="884" y="{fold_y + 44}" {MONO} font-size="13" fill="{BLUE}">\u096f\u096e\u096a\u096b\u0966\u0967\u0968\u0969\u096a\u096b \u2192 9845012345</text>""")

    layers = [
        (GREEN, GREEN_BG, "LAYER 1", "Structured patterns",
         "Every category proposes candidate spans, each with a base score.", "score := base"),
        (BLUE, BLUE_BG, "LAYER 2", "Checksum gates",
         "Verhoeff for Aadhaar, Luhn for cards: a shape becomes a fact.", "pass \u2192 0.95  \u00b7  fail \u2192 drop"),
        (VIOLET, VIOLET_BG, "LAYER 3", "Context scoring",
         "A nearby cue word lifts the score. This is Presidio's mechanism.", "score += 0.35, floor 0.40"),
        (AMBER, AMBER_BG, "LAYER 4", "Allowlist suppression",
         "Reserved IPs, helplines, year runs, native-digit citations.", "matched \u2192 suppressed"),
    ]
    y, box_w = 248, 272
    for index, (colour, fill, tag, name, why, rule) in enumerate(layers):
        left = 26 + index * (box_w + 20)
        body.append(f"""
  <rect x="{left}" y="{y}" width="{box_w}" height="132" rx="9" fill="{fill}" stroke="{colour}" stroke-width="1.6"/>
  <text x="{left + 14}" y="{y + 22}" {MONO} font-size="10.5" font-weight="700" fill="{colour}">{tag}</text>
  <text x="{left + 14}" y="{y + 44}" {FONT} font-size="13.5" font-weight="700" fill="{INK}">{name}</text>
{_lines(why, left + 14, y + 66, box_w - 30, 11.5, MUTED)}
  <rect x="{left + 14}" y="{y + 100}" width="{box_w - 28}" height="20" rx="4" fill="#ffffff" stroke="{colour}" stroke-opacity="0.45"/>
  <text x="{left + 22}" y="{y + 114}" {MONO} font-size="10" fill="{colour}">{rule}</text>""")
        if index < 3:
            body.append(_arrow_right(left + box_w + 3, y + 62, 14, LINE))

    ty = 400
    body.append(f"""
  <rect x="26" y="{ty}" width="560" height="58" rx="9" fill="{PANEL}" stroke="{EDGE}"/>
  <text x="42" y="{ty + 23}" {FONT} font-size="13" font-weight="700" fill="{INK}">Threshold \u2192 keep spans scoring \u2265 0.50</text>
{_lines("The single dial an operator turns. Lower it to buy recall, raise it to buy precision.", 42, ty + 43, 520, 11.5, MUTED)}

  <rect x="614" y="{ty}" width="560" height="58" rx="9" fill="{PANEL}" stroke="{EDGE}"/>
  <text x="630" y="{ty + 23}" {FONT} font-size="13" font-weight="700" fill="{INK}">Resolve overlaps \u2192 highest score wins</text>
{_lines("Masking needs non-overlapping edits, so cross-type overlaps have to be resolved too.", 630, ty + 43, 520, 11.5, MUTED)}""")

    oy = 478
    outcomes = [
        (GREEN, GREEN_BG, "MASKED", "layers 1+2",
         "\u092e\u094b\u092c\u093e\u0907\u0932 [PHONE] \u092a\u0930 \u00b7 \u0906\u0927\u093e\u0930 [ID]", "both scored \u2265 0.50 and were replaced"),
        (AMBER, AMBER_BG, "SUPPRESSED", "layer 4",
         "\u0927\u0930\u094d\u092e\u0938\u0942\u0924\u094d\u0930 (\u0968.\u096f.\u0968\u0967.\u0967)", "dotted quad kept: native digits \u21d2 citation"),
        (RED, RED_BG, "MISSED", "no layer proposes it",
         "a personal NAME in the prose", "the gap a regex cascade cannot close"),
    ]
    for index, (colour, fill, tag, source, text, note) in enumerate(outcomes):
        left = 26 + index * 384
        body.append(f"""
  <rect x="{left}" y="{oy}" width="372" height="78" rx="9" fill="{fill}" stroke="{colour}" stroke-width="1.6"/>
  <text x="{left + 14}" y="{oy + 21}" {MONO} font-size="10.5" font-weight="700" fill="{colour}">{tag}</text>
  <text x="{left + 358}" y="{oy + 21}" text-anchor="end" {MONO} font-size="9.5" fill="{colour}" opacity="0.75">by {source}</text>
  <text x="{left + 14}" y="{oy + 46}" {MONO} font-size="13" fill="{INK}">{text}</text>
  <text x="{left + 14}" y="{oy + 66}" {FONT} font-size="11.5" fill="{MUTED}">{note}</text>""")

    # Masked and suppressed both flow out of the threshold/overlap stage. The third
    # box does not: nothing upstream ever proposed it, so it gets a dashed arrow
    # straight from the input rather than a solid one from the threshold.
    body.append(_arrow_down(306, ty + 59, 17, GREEN))
    body.append(_arrow_down(690, ty + 59, 17, AMBER))
    body.append(
        f'  <path d="M1174 148 L1192 148 L1192 {oy - 8} L1080 {oy - 8}" fill="none" '
        f'stroke="{RED}" stroke-width="1.6" stroke-dasharray="4 3"/>\n'
        f'  <path d="M1086 {oy - 13} L1078 {oy - 8} L1086 {oy - 3} Z" fill="{RED}"/>\n'
        f'  <text x="1168" y="{oy - 16}" text-anchor="end" {MONO} font-size="9.5" '
        f'fill="{RED}" opacity="0.8">never enters the cascade</text>'
    )

    return _frame(
        1200, 574, "\n".join(body),
        "How one span is decided: the four-layer cascade",
        "A pattern only proposes; checksums, context and allowlists decide. Every layer is cheap, deterministic and independently auditable.",
    )


# =========================================================================
# 3. Precision / recall geometry
# =========================================================================
def precision_recall_geometry() -> str:
    body: list[str] = []

    # Left column runs x=26..634 (right-hand panels start at 660), so the pair of
    # circles is centred on 330 and the formula box shares the same 26px margin as
    # every other element in the figure.
    body.append(f"""
  <circle cx="270" cy="200" r="96" fill="{GREEN_BG}" fill-opacity="0.75" stroke="{GREEN}" stroke-width="2"/>
  <circle cx="390" cy="200" r="96" fill="{AMBER_BG}" fill-opacity="0.68" stroke="{AMBER}" stroke-width="2"/>
  <text x="242" y="92" text-anchor="end" {FONT} font-size="13" font-weight="700" fill="{GREEN}">truly PII</text>
  <text x="418" y="92" text-anchor="start" {FONT} font-size="13" font-weight="700" fill="{AMBER}">masked by the detector</text>
  <path d="M236 100 L212 122" stroke="{GREEN}" stroke-width="1.2"/>
  <path d="M424 100 L448 122" stroke="{AMBER}" stroke-width="1.2"/>

  <text x="220" y="196" text-anchor="middle" {MONO} font-size="13" font-weight="700" fill="{RED}">FN</text>
  <text x="220" y="216" text-anchor="middle" {FONT} font-size="11.5" fill="{RED}">missed</text>
  <text x="330" y="196" text-anchor="middle" {MONO} font-size="13" font-weight="700" fill="{GREEN}">TP</text>
  <text x="330" y="216" text-anchor="middle" {FONT} font-size="11.5" fill="{GREEN}">correct</text>
  <text x="440" y="196" text-anchor="middle" {MONO} font-size="13" font-weight="700" fill="{AMBER}">FP</text>
  <text x="440" y="216" text-anchor="middle" {FONT} font-size="11.5" fill="{AMBER}">over-masked</text>
  <text x="330" y="320" text-anchor="middle" {FONT} font-size="10.5" fill="{MUTED}">schematic \u2014 areas are not drawn to scale</text>

  <rect x="26" y="334" width="608" height="68" rx="8" fill="{PANEL}" stroke="{EDGE}"/>
  <text x="44" y="359" {MONO} font-size="12" fill="{INK}">precision = TP / (TP + FP)</text>
  <text x="256" y="359" {FONT} font-size="11.5" fill="{MUTED}">of what you masked, how much was PII</text>
  <text x="44" y="387" {MONO} font-size="12" fill="{INK}">recall    = TP / (TP + FN)</text>
  <text x="256" y="387" {FONT} font-size="11.5" fill="{MUTED}">of all the PII, how much you caught</text>""")

    # Right column mirrors the left column's vertical extent (80..402) so both
    # halves of the figure end on the same baseline.
    body.append(f"""
  <rect x="660" y="80" width="514" height="152" rx="9" fill="{RED_BG}" stroke="{RED}" stroke-width="1.5"/>
  <text x="680" y="106" {FONT} font-size="14" font-weight="700" fill="{RED}">A missed span costs a person</text>
{_lines("The identifier stays in the corpus and becomes learnable. Carlini et al. extracted verbatim PII from GPT-2, and memorisation grows log-linearly with model capacity, with how often a string is duplicated, and with prompt length.", 680, 128, 474, 12.5, INK)}
  <text x="680" y="218" {MONO} font-size="11" fill="{RED}">arXiv:2012.07805 \u00b7 arXiv:2202.07646</text>

  <rect x="660" y="250" width="514" height="152" rx="9" fill="{AMBER_BG}" stroke="{AMBER}" stroke-width="1.5"/>
  <text x="680" y="276" {FONT} font-size="14" font-weight="700" fill="{AMBER}">An over-masked span costs the language</text>
{_lines("Each false positive deletes real text. On Indic corpora the casualties are place names, public figures and meaningful numerals: \u092e\u0948\u0938\u0942\u0930 read as a person, a run of dotted numerals read as an IP address.", 680, 298, 474, 12.5, INK)}
  <text x="680" y="388" {MONO} font-size="11" fill="{AMBER}">both failure modes measured in this notebook</text>""")

    body.append(f"""
  <rect x="26" y="424" width="1148" height="68" rx="8" fill="{BLUE_BG}" stroke="{BLUE}" stroke-width="1.5"/>
  <text x="46" y="448" {FONT} font-size="13" font-weight="700" fill="{BLUE}">Why one number is never enough</text>
{_lines("The two errors are not interchangeable, so a single F1 hides the actual decision. Report both, and state which error your threshold is buying. That choice is a policy owned by the corpus maintainer, not a property of the regex.", 46, 468, 1108, 12.5, INK)}""")

    return _frame(
        1200, 508, "\n".join(body),
        "Precision and recall are two different kinds of harm",
        "The whole stage is one trade-off. Naming both error types is what makes a threshold defensible instead of arbitrary.",
    )


# =========================================================================
# 4. The Devanagari digit trap
# =========================================================================
def devanagari_digit_trap() -> str:
    body: list[str] = []

    body.append(f"""
  <rect x="26" y="80" width="1148" height="76" rx="9" fill="{PANEL}" stroke="{EDGE}"/>
  <text x="44" y="106" {FONT} font-size="13.5" font-weight="700" fill="{INK}">Python's <tspan {MONO} fill="{BLUE}">\\d</tspan> on a <tspan {MONO} fill="{BLUE}">str</tspan> pattern matches every Unicode category-Nd digit, not just <tspan {MONO} fill="{BLUE}">[0-9]</tspan>.</text>
{_lines("\u201cMatches any Unicode decimal digit \u2026 Matches [0-9] if the ASCII flag is used.\u201d \u2014 docs.python.org/3/library/re.html. Devanagari digits occupy U+0966\u2013U+096F.", 44, 130, 1108, 12.5, MUTED)}""")

    y = 178
    body.append(f'  <text x="26" y="{y}" {FONT} font-size="14" font-weight="700" fill="{INK}">The bug is the asymmetry inside one single pattern \u2014 both rows verified against Python\u2019s <tspan {MONO} fill="{BLUE}">re</tspan></text>')

    # Both rows are executed facts, not illustrations:
    #   re.search(r"[6-9]\d{9}", "\u096f\u096e\u096a\u096b\u0966\u0967\u0968\u0969\u096a\u096b") is None   -> the miss
    #   re.search(r"\d{10}",     "\u096f\u096e\u096a\u096b\u0966\u0967\u0968\u0969\u096a\u096b") matches   -> the inconsistency
    rows = [
        (r"[6-9]\d{9}", "\u096f\u096e\u096a\u096b\u0966\u0967\u0968\u0969\u096a\u096b", "no match", RED, "#ffffff",
         "a real mobile is INVISIBLE: the [6-9] class is ASCII-only"),
        (r"\d{10}", "\u096f\u096e\u096a\u096b\u0966\u0967\u0968\u0969\u096a\u096b", "MATCHES", AMBER, AMBER_BG,
         "the very same digits DO match \\d \u2014 one pattern, two behaviours"),
    ]
    ry = y + 16
    for pattern, text, verdict, colour, fill, note in rows:
        body.append(f"""
  <rect x="26" y="{ry}" width="1148" height="54" rx="8" fill="{fill}" stroke="{colour}" stroke-width="1.5"/>
  <text x="44" y="{ry + 33}" {MONO} font-size="13.5" fill="{BLUE}">{escape(pattern)}</text>
  <text x="196" y="{ry + 33}" {MONO} font-size="14" fill="{INK}">{text}</text>
  <text x="410" y="{ry + 33}" {MONO} font-size="12.5" font-weight="700" fill="{colour}">{verdict}</text>
  <text x="540" y="{ry + 33}" {FONT} font-size="12.5" fill="{MUTED}">{note}</text>""")
        ry += 62

    fy = ry + 10
    body.append(f"""
  <rect x="26" y="{fy}" width="1148" height="140" rx="9" fill="{GREEN_BG}" stroke="{GREEN}" stroke-width="1.7"/>
  <text x="44" y="{fy + 26}" {FONT} font-size="14" font-weight="700" fill="{GREEN}">The fix: fold digits onto a detection copy, and write patterns with an explicit [0-9]</text>

  <rect x="44" y="{fy + 42}" width="322" height="82" rx="7" fill="#ffffff" stroke="{EDGE}"/>
  <text x="58" y="{fy + 62}" {MONO} font-size="10" font-weight="700" fill="{MUTED}">ORIGINAL \u00b7 emitted, never mutated</text>
  <text x="58" y="{fy + 88}" {MONO} font-size="14" fill="{INK}">\u092e\u094b\u092c\u093e\u0907\u0932 \u096f\u096e\u096a\u096b\u0966\u0967\u0968\u0969\u096a\u096b</text>
  <text x="58" y="{fy + 110}" {MONO} font-size="10.5" fill="{MUTED}">span at offsets 7 \u2026 17</text>

{_arrow_right(374, fy + 84, 34, GREEN)}
  <text x="372" y="{fy + 72}" {MONO} font-size="10" fill="{GREEN}">fold</text>

  <rect x="418" y="{fy + 42}" width="322" height="82" rx="7" fill="#ffffff" stroke="{GREEN}"/>
  <text x="432" y="{fy + 62}" {MONO} font-size="10" font-weight="700" fill="{GREEN}">DETECTION COPY \u00b7 matched, never emitted</text>
  <text x="432" y="{fy + 88}" {MONO} font-size="14" fill="{INK}">\u092e\u094b\u092c\u093e\u0907\u0932 9845012345</text>
  <text x="432" y="{fy + 110}" {MONO} font-size="10.5" fill="{GREEN}">span at offsets 7 \u2026 17 \u2014 identical</text>

  <rect x="766" y="{fy + 42}" width="384" height="82" rx="7" fill="#ffffff" stroke="{EDGE}"/>
  <text x="780" y="{fy + 62}" {MONO} font-size="10" font-weight="700" fill="{MUTED}">WHY THE OFFSETS SURVIVE</text>
{_lines("Folding maps one code point to one code point, so length is preserved and a span found in the copy indexes the original exactly.", 780, fy + 82, 358, 12, INK)}""")

    cy = fy + 156
    body.append(f"""
  <rect x="26" y="{cy}" width="1148" height="94" rx="9" fill="{AMBER_BG}" stroke="{AMBER}" stroke-width="1.6"/>
  <text x="44" y="{cy + 25}" {FONT} font-size="13.5" font-weight="700" fill="{AMBER}">\u2026 and folding creates a new Indic false positive, which is exactly why layer 4 exists</text>
  <text x="44" y="{cy + 50}" {MONO} font-size="13" fill="{INK}">\u0906\u092a\u0938\u094d\u0924\u0902\u092b \u0927\u0930\u094d\u092e\u0938\u0942\u0924\u094d\u0930 (\u0968.\u096f.\u0968\u0967.\u0967)  \u2192 fold \u2192  2.9.21.1  \u2192  a valid dotted quad  \u2192  masked as an IP address</text>
{_lines("Every IPv4 \u201cmatch\u201d in the verified Hindi probe cache was in fact a Sanskrit scripture citation. An IP address is an ASCII artefact of network configuration, so native-script digits are evidence against it \u2014 the suppression rule this measurement produced.", 44, cy + 72, 1108, 12.5, MUTED)}""")

    return _frame(
        1200, cy + 118, "\n".join(body),
        "The Indic trap: one regex, two opposite failures",
        "This is the concrete reason an English-tuned PII pass cannot be lifted onto Devanagari text unchanged.",
    )


# =========================================================================
# 5. Regex vs NER: what each family can even see
# =========================================================================
def regex_vs_ner() -> str:
    body: list[str] = []

    # `ships` entries are (who, what, citation). The citation is a separate field so
    # it always lands on its own line in a consistent style, instead of depending on
    # where the description happens to wrap.
    panels = [
        (26, GREEN, GREEN_BG, "Regex \u00b7 decides from SHAPE",
         "A finite automaton over characters. No training data, no GPU, microseconds per document, and a fully auditable reason behind every decision.",
         "ananya@example.com \u00b7 98450 12345 \u00b7 203.0.113.47",
         "\u0930\u093e\u0939\u0941\u0932 \u0935\u0930\u094d\u092e\u093e \u2014 a name has no grammar to match",
         [("Dolma",
           "regex only; the paper calls model-based detection \u201cimpractical\u201d at 3T tokens.",
           "arXiv:2402.00159 \u00a75.3"),
          ("BigScience ROOTS",
           "multilingual regex emitting PI:EMAIL style tags across 25 languages including Hindi.",
           "arXiv:2303.03915 \u00a73.3")],
         "precision high \u00b7 recall bounded by the pattern list"),
        (614, VIOLET, VIOLET_BG, "NER \u00b7 decides from CONTEXT",
         "A transformer scores each token against its neighbours, so one string can be an entity in this sentence and not in the next. That is the point, and the risk.",
         "\u0930\u093e\u0939\u0941\u0932 \u0935\u0930\u094d\u092e\u093e \u00b7 Ananya Sharma \u00b7 J. Smith",
         "\u092e\u0948\u0938\u0942\u0930 / Mysuru \u2014 a city masked as a person",
         [("BigCode StarPII",
           "125M StarEncoder fine-tuned on 12,000 annotated files; NAME F1 91.6.",
           "arXiv:2305.06161 \u00a74"),
          ("AI4Bharat IndicNER",
           "Hindi PER F1 82.3, built by projecting English tags across Samanantar.",
           "arXiv:2212.10168")],
         "recall high \u00b7 precision spent on ambiguity"),
    ]
    for left, colour, fill, heading, how, can_see, cannot, ships, footer in panels:
        body.append(f"""
  <rect x="{left}" y="80" width="560" height="366" rx="10" fill="{fill}" stroke="{colour}" stroke-width="1.8"/>
  <text x="{left + 22}" y="108" {FONT} font-size="15" font-weight="700" fill="{colour}">{heading}</text>
{_lines(how, left + 22, 132, 516, 12.5, INK)}

  <rect x="{left + 22}" y="{178}" width="516" height="92" rx="7" fill="#ffffff" stroke="{EDGE}"/>
  <text x="{left + 36}" y="198" {MONO} font-size="10" font-weight="700" fill="{MUTED}">CAN SEE</text>
  <text x="{left + 36}" y="220" {MONO} font-size="12.5" fill="{GREEN}">{can_see}</text>
  <text x="{left + 36}" y="242" {MONO} font-size="10" font-weight="700" fill="{MUTED}">CANNOT SEE \u2014 OR SEES WRONGLY</text>
  <text x="{left + 36}" y="262" {MONO} font-size="12.5" fill="{AMBER if colour == VIOLET else RED}">{cannot}</text>

  <text x="{left + 22}" y="292" {FONT} font-size="12.5" font-weight="700" fill="{INK}">Who ships this</text>""")
        ty = 312
        for who, what, citation in ships:
            body.append(f'  <text x="{left + 22}" y="{ty}" {FONT} font-size="12" font-weight="700" fill="{INK}">{who}</text>')
            wrapped = _wrap(what, 516, 11.5)
            body.append(_lines(what, left + 22, ty + 17, 516, 11.5, MUTED))
            cite_y = ty + 17 + len(wrapped) * 11.5 * 1.45
            body.append(
                f'  <text x="{left + 22}" y="{cite_y:.0f}" {MONO} font-size="10" fill="{colour}" opacity="0.85">{citation}</text>'
            )
            ty += 58
        # A rule separates the per-project evidence above from the summary verdict,
        # so the footer is not mistaken for another citation line.
        body.append(
            f'  <path d="M{left + 22} 420 L{left + 538} 420" stroke="{colour}" '
            f'stroke-width="1" opacity="0.28"/>'
        )
        body.append(f'  <text x="{left + 22}" y="438" {MONO} font-size="11" fill="{colour}">{footer}</text>')

    body.append(f"""
  <rect x="26" y="468" width="1148" height="86" rx="9" fill="{PANEL}" stroke="{EDGE}"/>
  <text x="46" y="492" {FONT} font-size="13.5" font-weight="700" fill="{INK}">The industry pattern: regex for structure, a model only where structure runs out</text>
{_lines("BigCode is the clearest split \u2014 regex and detect-secrets for keys, IPs and emails, StarPII for names, and USERNAME detected but deliberately NOT redacted because its F1 was only 61.5. A detector you do not trust enough to act on is a review signal, not a redaction rule. That distinction is what this notebook applies to Hindi personal names.", 46, 514, 1108, 12.5, MUTED)}""")

    return _frame(
        1200, 570, "\n".join(body),
        "Two detector families, two different questions",
        "Regex asks whether a string has the shape of an identifier. NER asks whether the surrounding context makes a string refer to a person.",
    )


DIAGRAMS = {
    "pipeline_position": pipeline_position,
    "cascade_internals": cascade_internals,
    "precision_recall_geometry": precision_recall_geometry,
    "devanagari_digit_trap": devanagari_digit_trap,
    "regex_vs_ner": regex_vs_ner,
}
