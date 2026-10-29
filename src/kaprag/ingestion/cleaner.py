"""Text cleaning: repeating headers/footers, page numbers, hyphenation, whitespace.

Header/footer detection is HEURISTIC and company-independent:
a line that appears at the top or bottom of many pages (after replacing
digits with '#', so "Sayfa 12" and "Sayfa 13" look the same) is boilerplate.
"""

from __future__ import annotations

import re
from collections import Counter

from kaprag.ingestion.models import Line
from kaprag.utils.turkish import tr_lower

# Lines that are just a page number: "12", "- 12 -", "Sayfa No: 12", "12 / 93"
PAGE_NUMBER_RE = re.compile(
    r"^\s*[-–]?\s*(sayfa(\s*no)?\s*:?\s*)?\d{1,4}(\s*/\s*\d{1,4})?\s*[-–]?\s*$", re.IGNORECASE
)
ZONE_LINES = 3  # how many lines at the top/bottom of a page count as header/footer zone


def _key(text: str) -> str:
    """Normalised form used to compare lines across pages."""
    return re.sub(r"\s+", " ", re.sub(r"\d", "#", tr_lower(text))).strip()


class TextCleaner:
    def __init__(self, min_ratio: float = 0.1, min_pages: int = 5) -> None:
        self.min_ratio = min_ratio
        self.min_pages = min_pages

    @staticmethod
    def _zones(lines: list[Line]) -> list[Line]:
        """The top-most and bottom-most lines of a page (by y position)."""
        ordered = sorted(lines, key=lambda ln: ln.bbox[1])
        return ordered[:ZONE_LINES] + ordered[-ZONE_LINES:]

    def find_boilerplate(self, pages: dict[int, list[Line]]) -> set[str]:
        """Return normalised keys of lines repeating in header/footer zones."""
        counts: Counter[str] = Counter()
        for lines in pages.values():
            counts.update({_key(ln.text) for ln in self._zones(lines)})  # once per page
        threshold = max(self.min_pages, self.min_ratio * len(pages))
        return {k for k, c in counts.items() if c >= threshold and k}

    def remove_boilerplate(self, pages: dict[int, list[Line]]) -> tuple[dict[int, list[Line]], set[str]]:
        """Drop repeating header/footer lines and page-number lines from the zones."""
        boiler = self.find_boilerplate(pages)
        cleaned: dict[int, list[Line]] = {}
        for page_no, lines in pages.items():
            zone_ids = {id(ln) for ln in self._zones(lines)}
            cleaned[page_no] = [
                ln
                for ln in lines
                if not (
                    id(ln) in zone_ids
                    and (_key(ln.text) in boiler or PAGE_NUMBER_RE.match(ln.text))
                )
            ]
        return cleaned, boiler

    @staticmethod
    def join_lines(lines: list[str]) -> str:
        """Join the lines of one paragraph, repairing hyphenation at line breaks.

        "faaliyet-" + "lerimiz" -> "faaliyetlerimiz", but "Ar-Ge" stays intact
        because the next line would have to start with a lowercase letter.
        """
        out = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and line[:1].islower():
                out = out[:-1] + line
            else:
                out = f"{out} {line}" if out else line
        return TextCleaner.normalize_whitespace(out)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        text = text.replace(" ", " ").replace("­", "")  # nbsp, soft hyphen
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ([,.;:!?)])", r"\1", text)
        return text.strip()
