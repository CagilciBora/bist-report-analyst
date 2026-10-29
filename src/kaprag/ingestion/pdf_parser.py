"""PDFParser: PDF file -> ParsedDocument (paragraphs + tables with pages & sections).

Pipeline for one document:
  1. For every page, check text quality (scanned? garbled font encoding?).
     Bad pages are logged and optionally OCR'd with Tesseract.
  2. Extract tables first (TableExtractor) and remember their bounding boxes.
  3. Extract text LINES with font size/bold info, skipping lines inside tables
     (otherwise every table number would appear twice).
  4. Remove repeating headers/footers and page numbers (TextCleaner).
  5. Walk the lines in reading order: heading lines (SectionDetector) update
     the current section; other lines are grouped into paragraphs.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

import pymupdf

from kaprag.config import ParsingConfig
from kaprag.ingestion.cleaner import TextCleaner
from kaprag.ingestion.models import Line, Paragraph, ParsedDocument, Table
from kaprag.ingestion.section_detector import SectionDetector
from kaprag.ingestion.table_extractor import TableExtractor

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[^\W\d_]{3,}")
_VOWEL_RE = re.compile(r"[aeıioöuüâîûAEIİOÖUÜÂÎÛ]")


def garbled_score(text: str) -> float:
    """0 = clean text, higher = broken font encoding.

    Main signal: share of control/private-use characters (broken fonts map
    glyphs to codes like \x00, \x01). Secondary signal: share of "words" with
    no vowel at all, above a 30% allowance (abbreviations like TSRS/GRI are
    normal, but garbage like "MBNOB ABPQRMQS" everywhere is not).
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    bad = sum(1 for c in chars if ord(c) < 32 or 0xE000 <= ord(c) <= 0xF8FF or c == "ÿ")
    words = _WORD_RE.findall(text)
    novowel = sum(1 for w in words if not _VOWEL_RE.search(w)) / max(1, len(words))
    return bad / len(chars) + max(0.0, novowel - 0.3)


class PDFParser:
    BOLD_FLAG = 16  # PyMuPDF span flag bit for bold fonts

    def __init__(self, cfg: ParsingConfig) -> None:
        self.cfg = cfg
        self.tables = TableExtractor(cfg.table_engine)
        self.sections = SectionDetector(size_ratio=cfg.heading_size_ratio)
        self.cleaner = TextCleaner(min_ratio=cfg.header_footer_min_ratio)
        self._ocr_ready = cfg.ocr_enabled and self._check_tesseract()

    # ------------------------------------------------------------------ OCR
    def _check_tesseract(self) -> bool:
        if shutil.which("tesseract") is None:
            log.warning("OCR enabled but `tesseract` not found. Install with: "
                        "sudo apt install tesseract-ocr tesseract-ocr-%s", self.cfg.ocr_lang)
            return False
        return True

    def _ocr_page(self, page: pymupdf.Page) -> str:
        import pytesseract  # optional dependency (uv sync --extra ocr)
        from PIL import Image

        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return pytesseract.image_to_string(img, lang=self.cfg.ocr_lang)

    # ---------------------------------------------------------------- lines
    def _page_lines(self, page: pymupdf.Page, page_no: int, skip: list) -> list[Line]:
        lines: list[Line] = []
        for b_id, block in enumerate(page.get_text("dict")["blocks"]):
            for ln in block.get("lines", []):
                spans = [s for s in ln["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in ln["spans"]).strip()
                x0, y0, x1, y1 = ln["bbox"]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                if any(t[0] <= cx <= t[2] and t[1] <= cy <= t[3] for t in skip):
                    continue  # this line belongs to a table
                n_chars = sum(len(s["text"]) for s in spans)
                bold_chars = sum(len(s["text"]) for s in spans if s["flags"] & self.BOLD_FLAG
                                 or "bold" in s["font"].lower())
                size = max(spans, key=lambda s: len(s["text"]))["size"]
                lines.append(Line(text, page_no, round(size, 1), bold_chars >= 0.8 * n_chars,
                                  (x0, y0, x1, y1), b_id))
        return lines

    # ---------------------------------------------------------------- parse
    def parse(self, path: Path | str) -> ParsedDocument:
        path = Path(path)
        doc = pymupdf.open(path)
        result = ParsedDocument(source_file=str(path), n_pages=len(doc))
        page_lines: dict[int, list[Line]] = {}
        page_tables: dict[int, list[Table]] = {}

        for idx, page in enumerate(doc):
            page_no = idx + 1
            text = page.get_text()
            n_chars = len(text.strip())
            if n_chars < self.cfg.scanned_min_chars:
                kind = "scanned"
            elif garbled_score(text) > self.cfg.garbled_ratio_threshold:
                kind = "garbled"
            else:
                kind = "ok"

            if kind != "ok":
                (result.scanned_pages if kind == "scanned" else result.garbled_pages).append(page_no)
                if self._ocr_ready:
                    ocr_text = self._ocr_page(page)
                    if len(ocr_text.strip()) >= self.cfg.scanned_min_chars:
                        result.ocr_pages.append(page_no)
                        # OCR gives no font info: each text line becomes a body line.
                        page_lines[page_no] = [
                            Line(t, page_no, 0.0, False, (0, i, 0, i), i)
                            for i, t in enumerate(ocr_text.splitlines()) if t.strip()
                        ]
                continue

            tables = self.tables.extract(page, page_no, str(path))
            page_tables[page_no] = tables
            page_lines[page_no] = self._page_lines(page, page_no, [t.bbox for t in tables])

        if result.scanned_pages:
            log.warning("%s: %d scanned pages (no text layer): %s", path.name,
                        len(result.scanned_pages), result.scanned_pages)
        if result.garbled_pages:
            log.warning("%s: %d garbled pages (broken font encoding): %s%s", path.name,
                        len(result.garbled_pages), result.garbled_pages[:15],
                        " ..." if len(result.garbled_pages) > 15 else "")

        page_lines, boiler = self.cleaner.remove_boilerplate(page_lines)
        result.removed_header_footer = sorted(boiler)
        self.sections.fit([ln for lines in page_lines.values() for ln in lines])
        self._assemble(result, page_lines, page_tables)
        doc.close()
        return result

    def _assemble(self, result: ParsedDocument, page_lines: dict[int, list[Line]],
                  page_tables: dict[int, list[Table]]) -> None:
        """Walk pages in order, track the current section, build paragraphs."""
        section = ""
        for page_no in sorted(set(page_lines) | set(page_tables)):
            lines = page_lines.get(page_no, [])
            tables = sorted(page_tables.get(page_no, []), key=lambda t: t.bbox[1])
            buf: list[str] = []
            buf_block = None
            heading_parts: list[str] = []
            pending_tables = list(tables)

            for ln in lines:
                # Tables located above this line get the section that is current now.
                while pending_tables and pending_tables[0].bbox[1] < ln.bbox[1] and ln.size:
                    t = pending_tables.pop(0)
                    t.section_title = section
                    result.tables.append(t)
                if ln.size and self.sections.is_heading(ln):
                    self._flush(result, buf, page_no, section)
                    heading_parts.append(ln.text.strip())
                    continue
                if heading_parts:  # a heading may span several lines -> join them
                    section = self.cleaner.normalize_whitespace(" ".join(heading_parts))[:150]
                    heading_parts = []
                if ln.block_id != buf_block:
                    self._flush(result, buf, page_no, section)
                    buf_block = ln.block_id
                buf.append(ln.text)
            self._flush(result, buf, page_no, section)
            if heading_parts:
                section = self.cleaner.normalize_whitespace(" ".join(heading_parts))[:150]
            for t in pending_tables:
                t.section_title = section
                result.tables.append(t)

    def _flush(self, result: ParsedDocument, buf: list[str], page_no: int, section: str) -> None:
        """Turn the buffered lines into one Paragraph and empty the buffer."""
        text = self.cleaner.join_lines(buf)
        if len(text) >= 3:
            result.paragraphs.append(Paragraph(text, page_no, section))
        buf.clear()
