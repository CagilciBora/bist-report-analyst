"""The Chunk: the unit we index, retrieve and cite."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Chunk:
    chunk_id: str            # readable id, e.g. "THYAO_2025_finansal_rapor_00042"
    ticker: str
    company_name: str
    year: int
    period: str              # FY, Q1..Q4 or a disclosure date
    doc_type: str
    source_file: str
    page_start: int
    page_end: int
    section_title: str
    chunk_type: str          # "text" or "table"
    text: str

    @property
    def point_id(self) -> str:
        """Qdrant needs UUIDs (or ints) as ids; derive a stable UUID from chunk_id."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, self.chunk_id))

    @property
    def index_text(self) -> str:
        """Text that is embedded and BM25-indexed: a short context header + content.

        Adding company/year/section to every chunk ("contextual chunk header")
        lets a chunk that only says "Hasılat 955.472" still match a question
        about "THY'nin 2025 hasılatı".
        """
        header = f"{self.company_name} ({self.ticker}) {self.year} {self.doc_type}"
        if self.section_title:
            header += f" | {self.section_title}"
        return f"{header}\n{self.text}"

    @property
    def citation(self) -> str:
        """The citation tag the LLM must use: [TICKER YEAR doc_type s.PAGE]."""
        return f"[{self.ticker} {self.year} {self.doc_type} s.{self.page_start}]"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})
