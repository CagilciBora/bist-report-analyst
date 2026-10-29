"""SearchFilter: ONE metadata filter object used by BOTH retrievers.

Why filter INSIDE the search instead of after it? If we take the dense top-30
over all companies and then drop non-ASELS chunks, we may be left with 2
chunks (or 0) — the other 28 slots were wasted on THYAO/BIMAS. Filtering
inside the search returns the best 30 *ASELS* chunks.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from qdrant_client import models as qm


class SearchFilter(BaseModel):
    tickers: list[str] | None = None
    years: list[int] | None = None
    periods: list[str] | None = None
    doc_types: list[str] | None = None
    chunk_types: list[str] | None = None

    def is_empty(self) -> bool:
        return not (self.tickers or self.years or self.periods or self.doc_types or self.chunk_types)

    def matches(self, payload: dict[str, Any]) -> bool:
        """Python-side check (used by BM25 to pick the subset of chunks to score)."""
        return (
            (not self.tickers or payload["ticker"] in self.tickers)
            and (not self.years or payload["year"] in self.years)
            and (not self.periods or payload["period"] in self.periods)
            and (not self.doc_types or payload["doc_type"] in self.doc_types)
            and (not self.chunk_types or payload["chunk_type"] in self.chunk_types)
        )

    def to_qdrant(self) -> qm.Filter | None:
        """The same conditions in Qdrant's filter language (applied during search)."""
        must: list[qm.FieldCondition] = []
        if self.tickers:
            must.append(qm.FieldCondition(key="ticker", match=qm.MatchAny(any=self.tickers)))
        if self.years:
            must.append(qm.FieldCondition(key="year", match=qm.MatchAny(any=self.years)))
        if self.periods:
            must.append(qm.FieldCondition(key="period", match=qm.MatchAny(any=self.periods)))
        if self.doc_types:
            must.append(qm.FieldCondition(key="doc_type", match=qm.MatchAny(any=self.doc_types)))
        if self.chunk_types:
            must.append(qm.FieldCondition(key="chunk_type", match=qm.MatchAny(any=self.chunk_types)))
        return qm.Filter(must=must) if must else None
