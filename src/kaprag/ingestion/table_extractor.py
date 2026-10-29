"""Table extraction: ruled tables with PyMuPDF + borderless numeric tables by layout.

Two kinds of tables appear in KAP documents:
1. RULED tables (lines/borders): PyMuPDF `page.find_tables()` handles them well.
2. BORDERLESS financial statements (balance sheet, income statement): there are
   no lines, so find_tables() collapses everything into a single cell. For these
   we use word coordinates: group words into rows by y, split rows into cells at
   big x gaps, and align numbers into columns by their RIGHT edge (numbers in
   statements are right-aligned).
pdfplumber was compared on the same pages and produced character-level garbage
on borderless pages and nothing better on ruled ones, so it is only kept as an
optional engine (config `parsing.table_engine: pdfplumber`). See docs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pymupdf

from kaprag.ingestion.models import BBox, Table

log = logging.getLogger(__name__)

# A cell that is a number in a statement: "86.035", "(1.234)", "%12,5", "-"
NUMERIC_CELL_RE = re.compile(r"^[(\-−–]?%?\s?\d[\d.,]*\)?%?$|^[-–—]$")


def _clean_cell(cell: str | None) -> str:
    return re.sub(r"\s+", " ", (cell or "").replace("\n", " ")).strip()


def _overlaps(a: BBox, b: BBox) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _caption_above(page: pymupdf.Page, bbox: BBox, max_gap: float = 60) -> str:
    """Text of the closest 1–2 lines directly above a table (used as caption)."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for ln in block.get("lines", []):
            x0, y0, x1, y1 = ln["bbox"]
            text = "".join(s["text"] for s in ln["spans"]).strip()
            horizontally_related = x1 > bbox[0] and x0 < bbox[2]
            if text and horizontally_related and bbox[1] - max_gap <= y1 <= bbox[1] + 1:
                lines.append((y1, text))
    lines.sort()
    return " ".join(t for _, t in lines[-2:])


@dataclass
class _Cell:
    text: str
    x0: float
    x1: float

    @property
    def numeric(self) -> bool:
        return bool(NUMERIC_CELL_RE.match(self.text))


class TableExtractor:
    def __init__(self, engine: str = "pymupdf") -> None:
        self.engine = engine

    # ------------------------------------------------------------------ public
    def extract(self, page: pymupdf.Page, page_no: int, pdf_path: str | None = None) -> list[Table]:
        if self.engine == "pdfplumber" and pdf_path:
            tables = self._pdfplumber_tables(pdf_path, page_no)
        else:
            tables = self._pymupdf_tables(page, page_no)
        tables += self._layout_tables(page, page_no, exclude=[t.bbox for t in tables])
        return tables

    # ------------------------------------------------------- 1) ruled tables
    def _pymupdf_tables(self, page: pymupdf.Page, page_no: int) -> list[Table]:
        out: list[Table] = []
        try:
            found = page.find_tables().tables
        except Exception as exc:  # PyMuPDF can fail on odd vector graphics
            log.debug("find_tables failed on page %s: %s", page_no, exc)
            return out
        for t in found:
            raw_rows = t.extract()
            if self._is_collapsed(raw_rows):
                continue  # e.g. a borderless statement squeezed into one cell
            rows = self._clean_rows(raw_rows)
            if len(rows) < 2 or len(rows[0]) < 2:
                continue
            bbox = tuple(t.bbox)
            out.append(
                Table(page=page_no, header=rows[0], rows=rows[1:], engine="pymupdf",
                      caption=_caption_above(page, bbox), bbox=bbox)  # type: ignore[arg-type]
            )
        return out

    @staticmethod
    def _is_collapsed(rows: list[list[str | None]]) -> bool:
        """True if find_tables() merged many visual rows into single cells.

        Happens on borderless statements: one cell holds a whole text block, or
        a cell holds several numbers stacked on separate lines ("86.035\n95.992").
        Such tables are dropped so the layout-based extractor handles the page.
        """
        cells = [c for r in rows for c in r if c]
        if not cells:
            return True
        avg_breaks = sum(c.count("\n") for c in cells) / len(cells)
        stacked = sum(
            1 for c in cells
            if sum(bool(NUMERIC_CELL_RE.match(x.strip())) for x in c.split("\n")) >= 2
        )
        return (avg_breaks > 4 or max(c.count("\n") for c in cells) > 12
                or stacked / len(cells) > 0.15)

    @staticmethod
    def _clean_rows(raw_rows: list[list[str | None]]) -> list[list[str]]:
        """Normalise cells, blank out cells repeated by merged spans, drop empty rows/cols."""
        rows: list[list[str]] = []
        for raw in raw_rows:
            row, prev = [], None
            for cell in raw:
                c = _clean_cell(cell)
                row.append("" if c and c == prev else c)  # merged cell copied into neighbours
                prev = c or prev
            if any(row):
                rows.append(row)
        if not rows:
            return rows
        ncols = max(len(r) for r in rows)
        rows = [r + [""] * (ncols - len(r)) for r in rows]
        keep = [j for j in range(ncols) if any(r[j] for r in rows)]
        return [[r[j] for j in keep] for r in rows]

    # ---------------------------------------------------- optional pdfplumber
    @staticmethod
    def _pdfplumber_tables(pdf_path: str, page_no: int) -> list[Table]:
        import pdfplumber  # imported lazily; only used when configured

        out = []
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_no - 1]
            for t in page.find_tables():
                rows = TableExtractor._clean_rows(t.extract())
                if len(rows) >= 2 and len(rows[0]) >= 2:
                    out.append(Table(page=page_no, header=rows[0], rows=rows[1:],
                                     engine="pdfplumber", bbox=tuple(t.bbox)))  # type: ignore[arg-type]
        return out

    # ------------------------------------------ 2) borderless numeric tables
    def _layout_tables(self, page: pymupdf.Page, page_no: int, exclude: list[BBox]) -> list[Table]:
        words = page.get_text("words")
        if len(words) < 20:
            return []
        rows = self._group_rows(words)
        tables: list[Table] = []
        i = 0
        while i < len(rows):
            start, end = self._find_region(rows, i)
            if start is None:
                break
            region = rows[start:end]
            bbox = self._rows_bbox(region)
            if not any(_overlaps(bbox, ex) for ex in exclude):
                table = self._build_table(rows, start, end, page_no)
                if table:
                    tables.append(table)
            i = end
        return tables

    @staticmethod
    def _group_rows(words: list[tuple]) -> list[tuple[float, float, list[_Cell]]]:
        """Group words into visual rows (same y) and rows into cells (big x gaps)."""
        words = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
        raw_rows: list[list[tuple]] = []
        for w in words:
            yc = (w[1] + w[3]) / 2
            if raw_rows and abs(yc - (raw_rows[-1][0][1] + raw_rows[-1][0][3]) / 2) <= 3:
                raw_rows[-1].append(w)
            else:
                raw_rows.append([w])
        out = []
        for ws in raw_rows:
            ws.sort(key=lambda w: w[0])
            cells: list[_Cell] = []
            for w in ws:
                height = w[3] - w[1]
                if cells and w[0] - cells[-1].x1 < max(4.0, height * 0.6):
                    cells[-1].text += " " + w[4]
                    cells[-1].x1 = w[2]
                else:
                    cells.append(_Cell(w[4], w[0], w[2]))
            out.append((min(w[1] for w in ws), max(w[3] for w in ws), cells))
        return out

    @staticmethod
    def _is_numeric_row(cells: list[_Cell]) -> bool:
        nums = [c for c in cells if c.numeric]
        labels = [c for c in cells if not c.numeric]
        return len(nums) >= 2 or (len(nums) >= 1 and len(labels) >= 1 and len(cells) >= 2)

    def _find_region(self, rows, start_at: int) -> tuple[int | None, int]:
        """Find the next run of rows that is dominated by numeric rows."""
        i = start_at
        while i < len(rows):
            if self._is_numeric_row(rows[i][2]):
                start, end, gap, n_numeric = i, i + 1, 0, 1
                j = i + 1
                while j < len(rows) and gap < 3:
                    if self._is_numeric_row(rows[j][2]):
                        n_numeric, gap, end = n_numeric + 1, 0, j + 1
                    else:
                        gap += 1  # a label-only row like "Dönen Varlıklar" is allowed
                    j += 1
                if n_numeric >= 4:
                    return start, end
                i = end
            else:
                i += 1
        return None, len(rows)

    @staticmethod
    def _rows_bbox(rows) -> BBox:
        x0 = min(c.x0 for _, _, cells in rows for c in cells)
        x1 = max(c.x1 for _, _, cells in rows for c in cells)
        return (x0, rows[0][0], x1, rows[-1][1])

    def _build_table(self, rows, start: int, end: int, page_no: int) -> Table | None:
        region = rows[start:end]
        # Column anchors = clusters of right edges of numeric cells.
        edges = sorted(c.x1 for _, _, cells in region for c in cells if c.numeric)
        anchors: list[list[float]] = []
        for e in edges:
            if anchors and e - anchors[-1][-1] <= 15:
                anchors[-1].append(e)
            else:
                anchors.append([e])
        # Keep columns used by at least 2 rows (removes stray numbers).
        anchors = [a for a in anchors if len(a) >= 2]
        if not anchors:
            return None
        cols = [sum(a) / len(a) for a in anchors]
        first_col_left = min(c.x0 for _, _, cells in region for c in cells
                             if c.numeric and abs(c.x1 - cols[0]) <= 20)

        def place(cells: list[_Cell], allow_text: bool) -> tuple[str, list[str]] | None:
            label_parts, values = [], [""] * len(cols)
            for c in cells:
                near = min(range(len(cols)), key=lambda k: abs(c.x1 - cols[k]))
                fits_column = abs(c.x1 - cols[near]) <= (40 if allow_text else 20)
                if (c.numeric or allow_text) and fits_column and c.x0 >= first_col_left - 60:
                    text = re.sub(r"\(\s+", "(", c.text)  # "( 799.912)" -> "(799.912)"
                    values[near] = f"{values[near]} {text}".strip()
                else:
                    label_parts.append(c.text)
            return " ".join(label_parts), values

        body = []
        for _, _, cells in region:
            label, values = place(cells, allow_text=False)
            body.append([label, *values])

        # Header: up to 4 non-numeric rows directly above, aligned with the columns.
        header_vals = [""] * len(cols)
        header_label = ""
        top = start
        for k in range(start - 1, max(-1, start - 8), -1):
            y0, y1, cells = rows[k]
            if self._is_numeric_row(cells) or rows[top][0] - y1 > 30:
                break
            label, values = place(cells, allow_text=True)
            if not any(values):
                if not any(header_vals) and len(label) < 60:
                    body.insert(0, [label, *[""] * len(cols)])  # sub-heading like "Dönen Varlıklar"
                    top = k
                    continue
                break  # a title line spanning the page, not a column header
            header_vals = [f"{v} {h}".strip() for v, h in zip(values, header_vals, strict=True)]
            header_label = f"{label} {header_label}".strip()
            top = k
        # Caption = up to 3 lines above the header block (title + "Milyon TL" unit line).
        caption_rows = rows[max(0, top - 3):top]
        caption = " ".join(" ".join(c.text for c in cells) for _, _, cells in caption_rows)
        header = [header_label or "Kalem", *[h or f"Sütun {i + 1}" for i, h in enumerate(header_vals)]]
        bbox = self._rows_bbox(rows[top:end])
        return Table(page=page_no, header=header, rows=body, caption=caption.strip(),
                     engine="layout", bbox=bbox)
