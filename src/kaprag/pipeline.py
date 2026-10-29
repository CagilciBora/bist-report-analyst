"""RAGPipeline: question -> parsed filters -> retrieval -> prompt -> LLM -> number check.

This is the one object the CLI, the API and the evaluation all use, so every
entry point behaves identically. `ask()` returns a structured AskResponse with
the answer AND everything needed to audit it (chunks with per-retriever
scores, parsed filters, verified/unverified numbers, timings per stage).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from pydantic import BaseModel, Field

from kaprag.config import Settings, get_settings
from kaprag.generation.llm import LLMUnavailableError, make_llm
from kaprag.generation.prompt import PromptBuilder, disclaimer_for
from kaprag.generation.verifier import NumberVerifier
from kaprag.indexing.bm25_index import BM25Index
from kaprag.indexing.chunk_store import ChunkStore
from kaprag.indexing.dense_index import DenseIndex
from kaprag.indexing.filters import SearchFilter
from kaprag.ingestion.metadata import CompanyRegistry
from kaprag.retrieval.hybrid import HybridRetriever, RetrievedChunk
from kaprag.retrieval.query_parser import ParsedQuery, QueryParser

log = logging.getLogger(__name__)
_CITATION_RE = re.compile(r"\[([A-ZÇĞİÖŞÜ]{3,6}) (\d{4}) ([a-z_]+) s\.\s?(\d+)[^\]]*\]")


# ------------------------------------------------------------------ schemas
class ExplicitFilters(BaseModel):
    """Filters chosen in the UI/API; when set they OVERRIDE the parsed ones."""

    tickers: list[str] | None = None
    year_from: int | None = None
    year_to: int | None = None
    doc_types: list[str] | None = None


class ChunkOut(BaseModel):
    chunk_id: str
    ticker: str
    company_name: str
    year: int
    period: str
    doc_type: str
    source_file: str
    page_start: int
    page_end: int
    section_title: str
    chunk_type: str
    text: str
    citation: str
    group: str = ""
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    fused_score: float | None = None
    fused_rank: int | None = None
    rerank_score: float | None = None
    final_score: float = 0.0


class NumberOut(BaseModel):
    raw: str
    value: float
    status: str
    start: int
    end: int
    source_chunk: str | None = None
    note: str = ""


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    unknown_citations: list[str] = Field(default_factory=list)  # cited but not retrieved
    chunks: list[ChunkOut] = Field(default_factory=list)
    parsed_query: dict[str, Any] = Field(default_factory=dict)
    filters_used: list[dict[str, Any]] = Field(default_factory=list)
    searched_all_documents: bool = False
    retrieval_mode: str
    numbers: list[NumberOut] = Field(default_factory=list)
    unverified_numbers: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    llm_backend: str = ""
    llm_model: str = ""
    prompt_version: str = ""
    disclaimer: str = ""
    error: str | None = None


def _chunk_out(rc: RetrievedChunk) -> ChunkOut:
    c = rc.chunk
    return ChunkOut(**c.to_dict(), citation=c.citation, group=rc.group,
                    dense_score=rc.dense_score, dense_rank=rc.dense_rank, bm25_score=rc.bm25_score,
                    bm25_rank=rc.bm25_rank, fused_score=rc.fused_score, fused_rank=rc.fused_rank,
                    rerank_score=rc.rerank_score, final_score=rc.final_score)


# ----------------------------------------------------------------- pipeline
class RAGPipeline:
    def __init__(self, settings: Settings | None = None, dense: DenseIndex | None = None,
                 load_models: bool = True) -> None:
        self.settings = settings or get_settings()
        s = self.settings
        self.store = ChunkStore(s.paths.index_dir / "chunks.jsonl")
        self.dense = dense or DenseIndex(s.paths.index_dir / "qdrant")
        self.bm25 = BM25Index(self.store, s.bm25, s.paths.index_dir / "bm25")
        self.registry = CompanyRegistry(s.paths.companies_csv)
        years = [c.year for c in self.store.chunks.values()]
        self.parser = QueryParser(self.registry.alias_index(), set(self.registry.companies),
                                  latest_year=max(years) if years else None)
        embedder = None
        if load_models:
            from kaprag.indexing.embedder import Embedder

            embedder = Embedder(s.embedding)
        self.retriever = HybridRetriever(s.retrieval, self.store, self.dense, self.bm25, embedder)
        self.prompts = PromptBuilder(s.paths.prompts_dir / "answer_tr.md")
        self.llm = make_llm(s.llm)
        self.verifier = NumberVerifier()

    # ------------------------------------------------------------ helpers
    def parse(self, question: str) -> ParsedQuery:
        parsed = self.parser.parse(question)
        if self.settings.llm.use_llm_query_parser:
            from kaprag.retrieval.llm_query_parser import LLMQueryParser

            parsed = LLMQueryParser(self.llm, set(self.registry.companies)).refine(parsed)
        return parsed

    @staticmethod
    def apply_overrides(parsed: ParsedQuery, explicit: ExplicitFilters | None) -> ParsedQuery:
        """UI/API filters win over what the parser inferred (field by field)."""
        if not explicit:
            return parsed
        p = parsed.model_copy(deep=True)
        if explicit.tickers:
            p.tickers = explicit.tickers
        if explicit.year_from or explicit.year_to:
            lo = explicit.year_from or explicit.year_to
            hi = explicit.year_to or explicit.year_from
            p.years = list(range(lo, hi + 1))
        if explicit.doc_types:
            p.doc_types = explicit.doc_types
        p.is_comparison = p.is_comparison or len(p.tickers) > 1 or len(p.years) > 1
        p.notes = [n for n in p.notes if "ALL documents" not in n]
        if explicit.tickers or explicit.year_from or explicit.year_to or explicit.doc_types:
            p.notes.append("Explicit filters from the request override parsed filters.")
        return p

    def retrieve(self, question: str, mode: str | None = None, explicit: ExplicitFilters | None = None,
                 boost_tables: bool | None = None) -> tuple[ParsedQuery, list[RetrievedChunk], dict]:
        t = time.perf_counter()
        parsed = self.apply_overrides(self.parse(question), explicit)
        timings = {"parse_query": time.perf_counter() - t}
        result = self.retriever.search(parsed, mode=mode, boost_tables=boost_tables)
        timings.update(result.timings)
        return parsed, result.chunks, {"timings": timings, "filters": result.filters_used}

    # ---------------------------------------------------------------- ask
    def ask(self, question: str, mode: str | None = None, explicit: ExplicitFilters | None = None,
            generate: bool = True, boost_tables: bool | None = None) -> AskResponse:
        mode = mode or self.settings.retrieval.mode
        t_total = time.perf_counter()
        parsed, chunks, extra = self.retrieve(question, mode, explicit, boost_tables)
        timings = extra["timings"]
        filters: list[SearchFilter] = extra["filters"]
        resp = AskResponse(
            question=question, answer="", retrieval_mode=mode,
            chunks=[_chunk_out(rc) for rc in chunks], parsed_query=parsed.model_dump(),
            filters_used=[f.model_dump(exclude_none=True) for f in filters],
            searched_all_documents=all(f.is_empty() for f in filters),
            llm_backend=self.llm.name, llm_model=self.llm.model,
            prompt_version=self.prompts.version, disclaimer=disclaimer_for(question),
        )
        if generate:
            self._generate(resp, parsed, chunks, timings)
        timings["total"] = time.perf_counter() - t_total
        resp.timings_ms = {k: round(v * 1000, 1) for k, v in timings.items()}
        return resp

    def _generate(self, resp: AskResponse, parsed: ParsedQuery, chunks: list[RetrievedChunk],
                  timings: dict[str, float]) -> None:
        if not chunks:
            # Nothing retrieved (e.g. the filtered company/year is not indexed): no LLM call needed.
            target = ", ".join(parsed.tickers) or "istenen şirket"
            years = ", ".join(map(str, parsed.years)) or "istenen dönem"
            resp.answer = (f"Bu bilgi sağlanan belgelerde yer almıyor: {target} / {years} için "
                           f"indekslenmiş belge parçası bulunamadı.\n\n{resp.disclaimer}")
            return
        t = time.perf_counter()
        prompt = self.prompts.build(parsed.question, chunks, parsed.is_comparison)
        timings["build_prompt"] = time.perf_counter() - t
        try:
            self.llm.check()
            t = time.perf_counter()
            answer = self.llm.generate(prompt.system, prompt.user).strip()
            timings["generate"] = time.perf_counter() - t
        except LLMUnavailableError as exc:
            resp.error = str(exc)
            resp.answer = f"LLM unavailable: {exc}"
            return
        if resp.disclaimer not in answer:  # guarantee the disclaimer, whatever the model did
            answer = f"{answer}\n\n{resp.disclaimer}"
        resp.answer = answer

        # citations: which tags did the model use, and do they point to retrieved chunks?
        retrieved = {(c.chunk.ticker, c.chunk.year, c.chunk.doc_type) for c in chunks}
        pages = {(c.chunk.ticker, c.chunk.year, c.chunk.doc_type, p)
                 for c in chunks for p in range(c.chunk.page_start, c.chunk.page_end + 1)}
        for m in _CITATION_RE.finditer(answer):
            tag = m.group(0)
            key = (m.group(1), int(m.group(2)), m.group(3), int(m.group(4)))
            if tag not in resp.citations:
                resp.citations.append(tag)
            if key not in pages and key[:3] not in retrieved and tag not in resp.unknown_citations:
                resp.unknown_citations.append(tag)

        t = time.perf_counter()
        checks = self.verifier.verify(answer, {c.chunk.chunk_id: c.chunk.text for c in chunks})
        timings["verify_numbers"] = time.perf_counter() - t
        resp.numbers = [NumberOut(**c.to_dict()) for c in checks]
        resp.unverified_numbers = [c.raw for c in checks if c.status == "unverified"]
