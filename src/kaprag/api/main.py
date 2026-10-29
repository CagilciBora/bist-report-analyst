"""FastAPI backend.

Run:  uv run uvicorn kaprag.api.main:app --port 8000
Docs: http://localhost:8000/docs (Swagger UI generated from the Pydantic models)

One RAGPipeline is created at startup (loading models takes ~20 s) and shared
by all requests. Embedded Qdrant allows one process per index folder, so the
API owns the index while running: ingest through POST /ingest, not the CLI.
"""

from __future__ import annotations

import logging
import shutil
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from kaprag.config import DOC_TYPES, get_settings
from kaprag.generation.llm import LLMUnavailableError
from kaprag.indexing.indexer import Indexer
from kaprag.logging_setup import setup_logging
from kaprag.pipeline import AskResponse, ExplicitFilters, RAGPipeline

log = logging.getLogger(__name__)
state: dict = {}
_lock = threading.Lock()  # ingestion and querying share the index; serialize writes


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    settings = get_settings()
    pipeline = RAGPipeline(settings)
    state["pipeline"] = pipeline
    state["indexer"] = Indexer(settings, dense=pipeline.dense)  # share the Qdrant client
    yield
    pipeline.dense.close()


app = FastAPI(title="BIST Report Analyst API", version="0.1.0", lifespan=lifespan,
              description="Hybrid-search RAG over KAP disclosures. Analysis only, not investment advice.")


# ------------------------------------------------------------------ models
class AskRequest(BaseModel):
    question: str
    filters: ExplicitFilters | None = None
    retrieval_mode: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] | None = None
    generate: bool = True


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    indexed_documents: int
    llm_backend: str
    llm_model: str
    llm_ready: bool
    llm_message: str = ""


class DocumentOut(BaseModel):
    doc_key: str
    ticker: str
    company_name: str
    year: int
    doc_type: str
    period: str
    n_chunks: int
    n_pages: int
    source_file: str
    indexed_at: str


class CompanyOut(BaseModel):
    ticker: str
    name: str
    aliases: list[str]
    sector: str
    indexed: bool


class IngestResponse(BaseModel):
    status: str
    doc_key: str
    n_chunks: int
    message: str = ""


# --------------------------------------------------------------- endpoints
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    p: RAGPipeline = state["pipeline"]
    ready, msg = True, ""
    try:
        p.llm.check()
    except LLMUnavailableError as exc:
        ready, msg = False, str(exc)
    return HealthResponse(status="ok", indexed_chunks=len(p.store), indexed_documents=len(state["indexer"].manifest),
                          llm_backend=p.llm.name, llm_model=p.llm.model, llm_ready=ready, llm_message=msg)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(400, "question is empty")
    return state["pipeline"].ask(req.question, mode=req.retrieval_mode, explicit=req.filters,
                                 generate=req.generate)


@app.get("/documents", response_model=list[DocumentOut])
def documents() -> list[DocumentOut]:
    return [DocumentOut(**d) for d in state["indexer"].list_documents()]


@app.get("/companies", response_model=list[CompanyOut])
def companies() -> list[CompanyOut]:
    indexed = {d["ticker"] for d in state["indexer"].list_documents()}
    return [CompanyOut(ticker=c.ticker, name=c.name, aliases=c.aliases, sector=c.sector,
                       indexed=c.ticker in indexed)
            for c in state["pipeline"].registry.companies.values()]


@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...), ticker: str = Form(...), year: int = Form(...),
           doc_type: str = Form("faaliyet_raporu"), period: str = Form("FY")) -> IngestResponse:
    if doc_type not in DOC_TYPES:
        raise HTTPException(400, f"doc_type must be one of {DOC_TYPES}")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF files are accepted")
    settings = get_settings()
    ticker = ticker.upper()
    name = f"{year}_{doc_type}.pdf" if period == "FY" else f"{period}_{doc_type}.pdf"
    target = settings.paths.raw_dir / ticker / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    with _lock:
        indexer: Indexer = state["indexer"]
        result = indexer.ingest([target], ticker, year, doc_type, period)[0]
        p: RAGPipeline = state["pipeline"]
        p.store.chunks, p.store.doc_keys = indexer.store.chunks, indexer.store.doc_keys
        p.bm25.build()  # new chunks become searchable by keyword immediately
    return IngestResponse(status=result.status, doc_key=result.doc_key, n_chunks=result.n_chunks,
                          message=result.message)
