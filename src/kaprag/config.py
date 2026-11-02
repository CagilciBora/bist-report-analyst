"""Load config.yaml into typed Pydantic models.

Why Pydantic? Each section below is a class with typed fields and defaults.
`Settings.load()` reads the YAML and Pydantic checks every value, so wrong
types or unknown retrieval modes raise an error immediately.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# Project root = two folders above this file (src/kaprag/config.py -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
DocType = Literal["faaliyet_raporu", "finansal_rapor", "ozel_durum", "other"]
DOC_TYPES: tuple[str, ...] = ("faaliyet_raporu", "finansal_rapor", "ozel_durum", "other")


class PathsConfig(BaseModel):
    raw_dir: Path = Path("data/raw")
    index_dir: Path = Path("data/index")
    companies_csv: Path = Path("data/companies.csv")
    prompts_dir: Path = Path("prompts")

    def resolve(self) -> PathsConfig:
        """Turn relative paths into absolute ones under the project root."""
        def fix(p: Path) -> Path:
            return p if p.is_absolute() else PROJECT_ROOT / p

        return PathsConfig(
            raw_dir=fix(self.raw_dir),
            index_dir=fix(self.index_dir),
            companies_csv=fix(self.companies_csv),
            prompts_dir=fix(self.prompts_dir),
        )


class ParsingConfig(BaseModel):
    scanned_min_chars: int = 40
    garbled_ratio_threshold: float = 0.08
    ocr_enabled: bool = False
    ocr_lang: str = "tur"
    header_footer_min_ratio: float = 0.1
    heading_size_ratio: float = 1.2
    table_engine: Literal["pymupdf", "pdfplumber"] = "pymupdf"


class ChunkingConfig(BaseModel):
    target_tokens: int = 400
    max_tokens: int = 450
    min_tokens: int = 40
    overlap_tokens: int = 60
    table_max_tokens: int = 450


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-m3"
    device: str = "auto"
    batch_size: int = 8
    max_seq_length: int = 512
    fp16: bool = True


class BM25Config(BaseModel):
    stemming: bool = False
    k1: float = 1.5
    b: float = 0.75


class RetrievalConfig(BaseModel):
    mode: RetrievalMode = "hybrid_rerank"
    dense_top_k: int = 30
    bm25_top_k: int = 30
    rrf_k: int = 60
    rerank_top_n: int = 20
    final_top_k: int = 6
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    table_boost_enabled: bool = False
    table_boost: float = Field(1.15, ge=1.0)


class LLMConfig(BaseModel):
    backend: Literal["ollama", "claude_code", "anthropic", "openai"] = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b-instruct"
    anthropic_model: str = "claude-opus-5-5"
    openai_model: str = "gpt-4o-mini"
    claude_code_model: str = "opus"      # alias understood by the Claude Code CLI
    claude_code_path: str = ""           # empty = auto-detect the `claude` binary
    judge_backend: str = "claude_code"   # LLM used by eval/judge.py
    judge_workers: int = 3               # parallel judge calls
    temperature: float = 0.1
    max_tokens: int = 900
    timeout_s: int = 600
    use_llm_query_parser: bool = False


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class Settings(BaseModel):
    """The whole config.yaml as one object."""

    paths: PathsConfig = PathsConfig()
    parsing: ParsingConfig = ParsingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    bm25: BM25Config = BM25Config()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    api: APIConfig = APIConfig()

    @classmethod
    def load(cls, path: Path | str | None = None) -> Settings:
        """Read YAML (default: $KAP_CONFIG or ./config.yaml) and validate it."""
        path = Path(path or os.environ.get("KAP_CONFIG", PROJECT_ROOT / "config.yaml"))
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        settings = cls.model_validate(data or {})
        settings.paths = settings.paths.resolve()
        # Environment overrides (used by docker-compose: Ollama runs on the host).
        if url := os.environ.get("KAP_OLLAMA_URL"):
            settings.llm.ollama_url = url
        if backend := os.environ.get("KAP_LLM_BACKEND"):
            settings.llm = settings.llm.model_copy(update={"backend": backend})
        return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached global settings (loaded once per process)."""
    return Settings.load()
