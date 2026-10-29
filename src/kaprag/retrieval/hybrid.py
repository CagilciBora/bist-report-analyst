"""HybridRetriever: dense + BM25 -> RRF -> optional cross-encoder rerank.

retrieval_mode:
  dense          only vector search
  bm25           only keyword search
  hybrid         both, fused with Reciprocal Rank Fusion
  hybrid_rerank  hybrid, then the cross-encoder re-orders the fused top-N
Every returned chunk keeps the score/rank from EACH stage, so the UI can show
why a chunk was retrieved ("BM25 rank 1, dense rank 14, fused rank 2").
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from dataclasses import dataclass, field

from kaprag.chunking.models import Chunk
from kaprag.config import RetrievalConfig
from kaprag.indexing.bm25_index import BM25Index
from kaprag.indexing.chunk_store import ChunkStore
from kaprag.indexing.dense_index import DenseIndex
from kaprag.indexing.embedder import Embedder
from kaprag.indexing.filters import SearchFilter
from kaprag.retrieval.fusion import reciprocal_rank_fusion
from kaprag.retrieval.query_parser import ParsedQuery
from kaprag.retrieval.reranker import Reranker

log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    fused_score: float | None = None
    fused_rank: int | None = None
    rerank_score: float | None = None
    final_score: float = 0.0
    group: str = ""            # "THYAO 2025" for comparison questions


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    timings: dict[str, float] = field(default_factory=dict)
    filters_used: list[SearchFilter] = field(default_factory=list)


class HybridRetriever:
    def __init__(self, cfg: RetrievalConfig, store: ChunkStore, dense: DenseIndex, bm25: BM25Index,
                 embedder: Embedder | None, reranker_factory=None) -> None:
        self.cfg = cfg
        self.store = store
        self.dense = dense
        self.bm25 = bm25
        self.embedder = embedder
        self._reranker: Reranker | None = None
        self._reranker_factory = reranker_factory or (lambda: Reranker(cfg.reranker_model))

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:  # loaded on first use (2 GB model)
            self._reranker = self._reranker_factory()
        return self._reranker

    # ------------------------------------------------------------- single
    def retrieve(self, question: str, flt: SearchFilter, mode: str | None = None,
                 boost_tables: bool = False, top_k: int | None = None,
                 timings: dict[str, float] | None = None) -> list[RetrievedChunk]:
        mode = mode or self.cfg.mode
        top_k = top_k or self.cfg.final_top_k
        timings = timings if timings is not None else {}
        boost = self.cfg.table_boost if (boost_tables and self.cfg.table_boost_enabled) else 1.0
        found: dict[str, RetrievedChunk] = {}

        def entry(cid: str) -> RetrievedChunk:
            if cid not in found:
                found[cid] = RetrievedChunk(self.store.get(cid))
            return found[cid]

        dense_ids: list[str] = []
        bm25_ids: list[str] = []
        if mode in ("dense", "hybrid", "hybrid_rerank"):
            t = time.perf_counter()
            vec = self.embedder.embed_query(question)
            timings["embed_query"] = timings.get("embed_query", 0) + time.perf_counter() - t
            t = time.perf_counter()
            for rank, (cid, score) in enumerate(self.dense.search(vec, flt, self.cfg.dense_top_k), 1):
                e = entry(cid)
                e.dense_score, e.dense_rank = score, rank
                dense_ids.append(cid)
            timings["dense_search"] = timings.get("dense_search", 0) + time.perf_counter() - t
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            t = time.perf_counter()
            for rank, (cid, score) in enumerate(self.bm25.search(question, flt, self.cfg.bm25_top_k), 1):
                e = entry(cid)
                e.bm25_score, e.bm25_rank = score, rank
                bm25_ids.append(cid)
            timings["bm25_search"] = timings.get("bm25_search", 0) + time.perf_counter() - t

        def boosted(e: RetrievedChunk, score: float) -> float:
            return score * boost if e.chunk.chunk_type == "table" else score

        if mode == "dense":
            ranked = sorted((entry(c) for c in dense_ids), key=lambda e: -boosted(e, e.dense_score))
            for e in ranked:
                e.final_score = boosted(e, e.dense_score)
            return ranked[:top_k]
        if mode == "bm25":
            ranked = sorted((entry(c) for c in bm25_ids), key=lambda e: -boosted(e, e.bm25_score))
            for e in ranked:
                e.final_score = boosted(e, e.bm25_score)
            return ranked[:top_k]

        t = time.perf_counter()
        fused = reciprocal_rank_fusion([dense_ids, bm25_ids], k=self.cfg.rrf_k)
        for rank, (cid, score) in enumerate(fused, 1):
            e = entry(cid)
            e.fused_score, e.fused_rank = score, rank
            e.final_score = boosted(e, score)
        ranked = sorted((entry(cid) for cid, _ in fused), key=lambda e: -e.final_score)
        timings["fusion"] = timings.get("fusion", 0) + time.perf_counter() - t
        if mode == "hybrid":
            return ranked[:top_k]

        # hybrid_rerank: cross-encoder over the fused top-N
        t = time.perf_counter()
        candidates = ranked[: self.cfg.rerank_top_n]
        scores = self.reranker.score(question, [e.chunk.index_text for e in candidates])
        for e, s in zip(candidates, scores, strict=True):
            e.rerank_score = s
            e.final_score = boosted(e, s)
        candidates.sort(key=lambda e: -e.final_score)
        timings["rerank"] = timings.get("rerank", 0) + time.perf_counter() - t
        return candidates[:top_k]

    # ---------------------------------------------------------- top level
    def search(self, parsed: ParsedQuery, mode: str | None = None, boost_tables: bool | None = None,
               base_filter: SearchFilter | None = None) -> RetrievalResult:
        """Plain or comparison retrieval depending on the parsed question."""
        timings: dict[str, float] = {}
        boost = parsed.is_financial if boost_tables is None else boost_tables
        base = base_filter or SearchFilter(
            tickers=parsed.tickers or None, years=parsed.years or None,
            periods=parsed.periods or None, doc_types=parsed.doc_types or None,
        )
        pairs = list(itertools.product(base.tickers or [None], base.years or [None]))
        if not parsed.is_comparison or len(pairs) < 2:
            chunks = self.retrieve(parsed.question, base, mode, boost, timings=timings)
            return RetrievalResult(chunks, timings, [base])

        # Comparison: one sub-query per (company, year) with an EQUAL budget, so a
        # company with more/longer documents cannot crowd out the other one.
        per_pair = max(2, math.ceil(self.cfg.final_top_k / len(pairs)))
        out: list[RetrievedChunk] = []
        filters: list[SearchFilter] = []
        for ticker, year in pairs:
            flt = base.model_copy(update={"tickers": [ticker] if ticker else base.tickers,
                                          "years": [year] if year else base.years})
            filters.append(flt)
            label = " ".join(str(x) for x in (ticker, year) if x)
            prefix = self.store_company_name(ticker)
            sub_q = f"{prefix} {year or ''} {parsed.question}".strip()
            for e in self.retrieve(sub_q, flt, mode, boost, top_k=per_pair, timings=timings):
                e.group = label
                out.append(e)
        return RetrievalResult(out, timings, filters)

    def store_company_name(self, ticker: str | None) -> str:
        if not ticker:
            return ""
        for c in self.store.chunks.values():
            if c.ticker == ticker:
                return c.company_name
        return ticker
