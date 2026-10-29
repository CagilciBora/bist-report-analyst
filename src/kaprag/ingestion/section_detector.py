"""Detect section headings from font size and boldness.

Idea: the most common font size in a document (weighted by characters) is the
BODY size. A short line whose font is clearly larger than the body, or a short
line that is entirely bold, is a heading such as "Riskler ve Yönetimin
Değerlendirmesi". Every paragraph/table that follows gets that title.
"""

from __future__ import annotations

import re
from collections import Counter

from kaprag.ingestion.models import Line

_LETTERS_RE = re.compile(r"[^\W\d_]")


class SectionDetector:
    def __init__(self, size_ratio: float = 1.2, max_words: int = 14) -> None:
        self.size_ratio = size_ratio
        self.max_words = max_words
        self.body_size: float = 10.0

    def fit(self, lines: list[Line]) -> float:
        """Learn the body font size of this document (character-weighted mode)."""
        sizes: Counter[float] = Counter()
        for ln in lines:
            sizes[round(ln.size, 1)] += len(ln.text)
        if sizes:
            self.body_size = sizes.most_common(1)[0][0]
        return self.body_size

    def is_heading(self, line: Line) -> bool:
        text = line.text.strip()
        words = text.split()
        letters = len(_LETTERS_RE.findall(text))
        # Guards: headings are short, mostly letters, and not sentence fragments.
        if not words or len(words) > self.max_words or letters < 4:
            return False
        if letters / max(1, len(text.replace(" ", ""))) < 0.6:
            return False                      # "%18,4" or "2.384" in a big infographic font
        if text.endswith((",", ";", ":")) or text[0].islower():
            return False
        bigger = line.size >= self.body_size * self.size_ratio
        bold_title = line.bold and line.size >= self.body_size * 0.95 and not text.endswith(".")
        # Bigger font is a heading only if it is not *tiny* text in a cover/infographic.
        return bigger or bold_title
