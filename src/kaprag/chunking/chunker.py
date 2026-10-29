"""Structure-aware chunking.

Basic RAG cuts text every N characters, which splits sentences, mixes two
sections in one chunk, and cuts tables in half (rows without their header are
meaningless: "| 86.035 | 95.992 |" — of what? which year?). Here:

* TEXT: paragraphs are grouped by section; inside a section we pack whole
  sentences until ~target_tokens, then start a new chunk that repeats the last
  ~overlap_tokens of sentences. We NEVER merge text from two sections.
* TABLES: each table is its own chunk. A big table is split by rows and EVERY
  part repeats the caption and the header row, so each part stands alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kaprag.chunking.models import Chunk
from kaprag.chunking.token_counter import TokenCounter
from kaprag.config import ChunkingConfig
from kaprag.ingestion.metadata import DocMeta
from kaprag.ingestion.models import ParsedDocument, Table

# Sentence boundary: . ! ? followed by space and an uppercase/digit start.
# "A.Ş." or "31.12.2025" are not boundaries because no space+capital follows the dot inside.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9\"'(“])")


@dataclass
class _Unit:
    """A sentence with its page; the atom we pack into chunks."""

    text: str
    page: int
    tokens: int
    para_start: bool  # first sentence of a paragraph -> join with a blank line


class Chunker:
    def __init__(self, cfg: ChunkingConfig, counter: TokenCounter) -> None:
        self.cfg = cfg
        self.counter = counter

    # ------------------------------------------------------------- public
    def chunk(self, doc: ParsedDocument, meta: DocMeta) -> list[Chunk]:
        chunks: list[Chunk] = []

        def make(text: str, p0: int, p1: int, section: str, kind: str) -> None:
            idx = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{meta.ticker}_{meta.year}_{meta.doc_type}_{meta.period}_{idx:05d}",
                ticker=meta.ticker, company_name=meta.company_name, year=meta.year,
                period=meta.period, doc_type=meta.doc_type, source_file=meta.source_file,
                page_start=p0, page_end=p1, section_title=section, chunk_type=kind, text=text,
            ))

        for section, units in self._sections(doc):
            for text, p0, p1 in self._pack(units):
                make(text, p0, p1, section, "text")
        for table in doc.tables:
            for text in self.split_table(table):
                make(text, table.page, table.page, table.section_title, "table")
        return chunks

    # --------------------------------------------------------------- text
    def _sentences(self, text: str) -> list[str]:
        parts = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
        out: list[str] = []
        for s in parts:  # hard-split monster "sentences" (lists without punctuation)
            if self.counter.count(s) <= self.cfg.max_tokens:
                out.append(s)
                continue
            words, cur = s.split(), []
            for w in words:
                cur.append(w)
                if self.counter.count(" ".join(cur)) >= self.cfg.target_tokens:
                    out.append(" ".join(cur))
                    cur = []
            if cur:
                out.append(" ".join(cur))
        return out

    def _sections(self, doc: ParsedDocument) -> list[tuple[str, list[_Unit]]]:
        """Consecutive paragraphs with the same section title form one group."""
        groups: list[tuple[str, list[_Unit]]] = []
        for para in doc.paragraphs:
            if not groups or groups[-1][0] != para.section_title:
                groups.append((para.section_title, []))
            for i, sent in enumerate(self._sentences(para.text)):
                groups[-1][1].append(_Unit(sent, para.page, self.counter.count(sent), i == 0))
        return groups

    @staticmethod
    def _join(units: list[_Unit]) -> str:
        out = ""
        for u in units:
            sep = "\n\n" if u.para_start else " "
            out = f"{out}{sep}{u.text}" if out else u.text
        return out

    def _pack(self, units: list[_Unit]) -> list[tuple[str, int, int]]:
        """Greedy packing of sentences into chunks with sentence-level overlap."""
        chunks: list[list[_Unit]] = []
        cur: list[_Unit] = []
        cur_tokens = 0
        for u in units:
            if cur and cur_tokens + u.tokens > self.cfg.target_tokens:
                chunks.append(cur)
                # Overlap: carry the last sentences (up to overlap_tokens) forward.
                carry: list[_Unit] = []
                carry_tokens = 0
                for prev in reversed(cur):
                    if carry_tokens + prev.tokens > self.cfg.overlap_tokens:
                        break
                    carry.insert(0, prev)
                    carry_tokens += prev.tokens
                if carry_tokens + u.tokens > self.cfg.max_tokens:
                    carry, carry_tokens = [], 0
                cur, cur_tokens = carry, carry_tokens
            cur.append(u)
            cur_tokens += u.tokens
        if cur:
            # A tiny tail is merged into the previous chunk of the SAME section.
            if chunks and cur_tokens < self.cfg.min_tokens:
                new_units = [u for u in cur if all(u is not v for v in chunks[-1])]
                chunks[-1].extend(new_units)
            else:
                chunks.append(cur)
        out = []
        for c in chunks:
            text = self._join(c)
            if self.counter.count(text) >= 8:  # drop crumbs like a lone "Ekler"
                out.append((text, min(u.page for u in c), max(u.page for u in c)))
        return out

    # -------------------------------------------------------------- tables
    def split_table(self, table: Table) -> list[str]:
        """Render a table as one or more Markdown chunks; each repeats caption+header."""
        caption = f"Tablo: {table.caption}".strip() if table.caption else "Tablo"
        full = f"{caption}\n{table.to_markdown()}"
        if self.counter.count(full) <= self.cfg.table_max_tokens or len(table.rows) <= 1:
            return [full]
        base = self.counter.count(f"{caption} (parça 99/99)\n{table.to_markdown([])}")
        budget = max(1, self.cfg.table_max_tokens - base)
        parts: list[list[list[str]]] = []
        cur: list[list[str]] = []
        cur_tokens = 0
        for row in table.rows:
            row_tokens = self.counter.count("| " + " | ".join(row) + " |")
            if cur and cur_tokens + row_tokens > budget:
                parts.append(cur)
                cur, cur_tokens = [], 0
            cur.append(row)
            cur_tokens += row_tokens
        if cur:
            parts.append(cur)
        n = len(parts)
        return [f"{caption} (parça {i + 1}/{n})\n{table.to_markdown(rows)}"
                for i, rows in enumerate(parts)]
