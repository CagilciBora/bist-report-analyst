"""Plain data classes passed between the ingestion steps.

PDFParser produces a ParsedDocument (paragraphs + tables, each with page
numbers and a section title); the Chunker later turns it into Chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BBox = tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF points


@dataclass
class Line:
    """One visual line of text with the font information we need for headings."""

    text: str
    page: int            # 1-based page number
    size: float          # dominant font size of the line
    bold: bool           # True if (almost) all characters are bold
    bbox: BBox
    block_id: int        # PyMuPDF block number, lines in one block form a paragraph


@dataclass
class Paragraph:
    text: str
    page: int
    section_title: str


@dataclass
class Table:
    """A table as header + rows of strings; rendered to Markdown for the LLM."""

    page: int
    header: list[str]
    rows: list[list[str]]
    caption: str = ""
    section_title: str = ""
    engine: str = "pymupdf"   # pymupdf | layout | pdfplumber
    bbox: BBox = (0, 0, 0, 0)

    def to_markdown(self, rows: list[list[str]] | None = None) -> str:
        """Render (a subset of) rows as a Markdown table, always with the header."""
        rows = self.rows if rows is None else rows
        ncols = len(self.header)

        def fmt(cells: list[str], limit: int = 400) -> str:
            cells = (list(cells) + [""] * ncols)[:ncols]
            return "| " + " | ".join(c.replace("|", "/")[:limit] for c in cells) + " |"

        # Header cells are capped at 80 chars so a broken header can't eat the chunk budget.
        lines = [fmt(self.header, limit=80), "|" + "---|" * ncols]
        lines += [fmt(r) for r in rows]
        return "\n".join(lines)


@dataclass
class ParsedDocument:
    source_file: str
    n_pages: int
    paragraphs: list[Paragraph] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    scanned_pages: list[int] = field(default_factory=list)   # no text layer
    garbled_pages: list[int] = field(default_factory=list)   # broken font encoding
    ocr_pages: list[int] = field(default_factory=list)       # recovered with OCR
    removed_header_footer: list[str] = field(default_factory=list)

    @property
    def sections(self) -> list[str]:
        """Distinct section titles in reading order."""
        seen: dict[str, None] = {}
        for p in self.paragraphs:
            seen.setdefault(p.section_title, None)
        return [s for s in seen if s]
