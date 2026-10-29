"""Turkish number parsing: `parse_tr_number` and `extract_numbers`.

Turkish financial reports write numbers differently from English ones:
    1.234.567,89   -> dot = thousands separator, comma = decimal separator
    (1.234)        -> parentheses mean NEGATIVE (accounting convention)
    %12,5          -> the percent sign comes BEFORE the number
    1,2 milyar TL  -> a scale word (bin/milyon/milyar) multiplies the value
A "naive" float("1.234") would give 1.234 instead of 1234 — a 1000x error.
This module turns such strings into a `ParsedNumber` with an absolute value,
so the NumberVerifier can compare "1,2 milyar TL" with "1.200.000 bin TL".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kaprag.utils.turkish import tr_lower

# Scale words -> multiplier. Order matters in the regex (longest first).
SCALE_WORDS: dict[str, float] = {
    "trilyon": 1e12,
    "milyar": 1e9,
    "milyon": 1e6,
    "bin": 1e3,
    "billion": 1e9,
    "million": 1e6,
    "thousand": 1e3,
    "mlr": 1e9,
    "mn": 1e6,
    "mio": 1e6,
}

# Currency words/symbols -> canonical code.
CURRENCY_WORDS: dict[str, str] = {
    "tl": "TRY",
    "try": "TRY",
    "₺": "TRY",
    "türk lirası": "TRY",
    "usd": "USD",
    "abd doları": "USD",
    "dolar": "USD",
    "$": "USD",
    "eur": "EUR",
    "euro": "EUR",
    "avro": "EUR",
    "€": "EUR",
}


@dataclass(frozen=True)
class ParsedNumber:
    """A number found in text, with everything needed to compare it."""

    raw: str                 # the exact text, e.g. "1,2 milyar TL"
    value: float             # the number as written, e.g. 1.2
    scale: float = 1.0       # multiplier from a scale word, e.g. 1e9
    currency: str | None = None
    is_percent: bool = False
    decimals: int = 0        # digits after the decimal separator (precision)
    start: int = -1          # character offsets in the source text
    end: int = -1

    @property
    def absolute(self) -> float:
        """value * scale, e.g. '1,2 milyar' -> 1_200_000_000."""
        return self.value * self.scale


# ---------------------------------------------------------------------------
# 1) Converting the numeric part ("1.234,5") to a float
# ---------------------------------------------------------------------------

def _to_float(num: str) -> tuple[float, int]:
    """Convert a bare numeric string (no sign/unit) to (float, decimals).

    Rules (Turkish first, English as fallback):
      * both '.' and ','  -> the LAST one is the decimal separator
      * only ','          -> decimal comma ("1,5"), unless it looks like
                             English thousands "1,234,567" (several groups of 3)
      * only '.'          -> thousands dots if every group after a dot has 3
                             digits ("1.234.567"), else a decimal point ("3.5")
    """
    has_dot, has_comma = "." in num, "," in num
    if has_dot and has_comma:
        if num.rfind(",") > num.rfind("."):          # 1.234,56 (Turkish)
            int_part, dec_part = num.rsplit(",", 1)
            int_part = int_part.replace(".", "")
        else:                                         # 1,234.56 (English)
            int_part, dec_part = num.rsplit(".", 1)
            int_part = int_part.replace(",", "")
        return float(f"{int_part}.{dec_part}"), len(dec_part)
    if has_comma:
        groups = num.split(",")
        if len(groups) > 2 and all(len(g) == 3 for g in groups[1:]):
            return float(num.replace(",", "")), 0     # 1,234,567 (English)
        int_part, dec_part = num.rsplit(",", 1)
        return float(f"{int_part}.{dec_part}"), len(dec_part)
    if has_dot:
        groups = num.split(".")
        if all(len(g) == 3 for g in groups[1:]):
            return float(num.replace(".", "")), 0     # 1.234.567 (Turkish)
        int_part, dec_part = num.rsplit(".", 1)
        return float(f"{int_part.replace('.', '')}.{dec_part}"), len(dec_part)
    return float(num), 0


# ---------------------------------------------------------------------------
# 2) A regex that finds numbers WITH their sign, percent, scale and currency
# ---------------------------------------------------------------------------

_SCALE_RE = "|".join(sorted(SCALE_WORDS, key=len, reverse=True))
_CURR_RE = "|".join(re.escape(c) for c in sorted(CURRENCY_WORDS, key=len, reverse=True))

# Read it as: [ ( or - ] [ % or currency-symbol ] DIGITS [ ) ] [ % ] [ scale ] [ currency ]
# The \b-like lookarounds stop us from matching inside words such as "A320neo".
NUMBER_PATTERN = re.compile(
    rf"""
    (?<![\w.,])                                  # not glued to a word or another number
    (?P<open>\()?                                # accounting negative: (1.234)
    (?P<sign>[-−–])?                             # minus sign (also unicode dashes)
    (?P<pct1>%\s?)?                              # Turkish style: %12,5
    (?P<cur1>[₺$€]\s?)?                          # $12 / ₺12
    (?P<num>\d{{1,3}}(?:[.,]\d{{3}})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)
    (?P<close>\))?
    (?P<pct2>\s?%)?                              # English style: 12,5%
    (?:\s+(?P<scale>{_SCALE_RE})\b)?             # bin / milyon / milyar ...
    (?:\s*(?P<cur2>{_CURR_RE})(?![\w]))?         # TL / USD / ABD Doları ...
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _build(m: re.Match[str], offset: int = 0) -> ParsedNumber:
    """Turn one regex match into a ParsedNumber."""
    value, decimals = _to_float(m.group("num"))
    negative = bool(m.group("sign")) or (bool(m.group("open")) and bool(m.group("close")))
    if negative:
        value = -value
    scale_word = tr_lower(m.group("scale") or "")
    currency_word = tr_lower((m.group("cur2") or m.group("cur1") or "").strip())
    return ParsedNumber(
        raw=m.group(0).strip(),
        value=value,
        scale=SCALE_WORDS.get(scale_word, 1.0),
        currency=CURRENCY_WORDS.get(currency_word),
        is_percent=bool(m.group("pct1") or m.group("pct2")),
        decimals=decimals,
        start=m.start() + offset,
        end=m.end() + offset,
    )


def parse_tr_number(text: str) -> ParsedNumber | None:
    """Parse ONE number (with optional units) from a short string.

    >>> parse_tr_number("1.234.567,89").value
    1234567.89
    >>> parse_tr_number("(1.234)").value
    -1234.0
    >>> parse_tr_number("1,2 milyar TL").absolute
    1200000000.0
    """
    m = NUMBER_PATTERN.search(text.strip())
    return _build(m) if m else None


# Things that look like numbers but are not facts: dates (31.12.2025) and
# citation markers ([ASELS 2025 faaliyet_raporu s.12]).
_DATE_RE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b")
_CITATION_RE = re.compile(r"\[[^\]]*\bs\.\s*\d+[^\]]*\]")


def extract_numbers(text: str, skip_years: bool = True) -> list[ParsedNumber]:
    """Find every number in a text, skipping dates, citations and bare years."""
    # Blank out dates/citations with spaces so character offsets stay valid.
    masked = _CITATION_RE.sub(lambda m: " " * len(m.group(0)), text)
    masked = _DATE_RE.sub(lambda m: " " * len(m.group(0)), masked)
    out: list[ParsedNumber] = []
    for m in NUMBER_PATTERN.finditer(masked):
        n = _build(m)
        is_bare_int = n.decimals == 0 and n.scale == 1 and not n.is_percent and not n.currency
        if skip_years and is_bare_int and 1900 <= n.value <= 2100 and "." not in m.group("num"):
            continue  # "2025" is almost always a year, not a financial figure
        out.append(n)
    return out
