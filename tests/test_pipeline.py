"""Tests for explicit-filter overrides and the prompt builder (no models needed)."""

from pathlib import Path

from kaprag.chunking.models import Chunk
from kaprag.generation.prompt import DISCLAIMER_EN, DISCLAIMER_TR, PromptBuilder, disclaimer_for
from kaprag.pipeline import ExplicitFilters, RAGPipeline
from kaprag.retrieval.hybrid import RetrievedChunk
from kaprag.retrieval.query_parser import ParsedQuery

ROOT = Path(__file__).resolve().parents[1]


def test_explicit_filters_override_parsed() -> None:
    parsed = ParsedQuery(question="q", tickers=["THYAO"], years=[2025],
                         notes=["No company/year/doc filter inferred: searching across ALL documents."])
    p = RAGPipeline.apply_overrides(parsed, ExplicitFilters(tickers=["ASELS", "BIMAS"], year_from=2024, year_to=2025))
    assert p.tickers == ["ASELS", "BIMAS"]
    assert p.years == [2024, 2025]
    assert p.is_comparison
    assert not any("ALL documents" in n for n in p.notes)


def test_no_override_keeps_parsed() -> None:
    parsed = ParsedQuery(question="q", tickers=["THYAO"], years=[2025])
    assert RAGPipeline.apply_overrides(parsed, None) is parsed
    assert RAGPipeline.apply_overrides(parsed, ExplicitFilters()).tickers == ["THYAO"]


def test_prompt_contains_citation_tags_and_disclaimer() -> None:
    pb = PromptBuilder(ROOT / "prompts" / "answer_tr.md")
    chunk = Chunk("id1", "THYAO", "Türk Hava Yolları A.O.", 2025, "FY", "finansal_rapor", "x.pdf",
                  11, 11, "Gelir Tablosu", "table", "| Hasılat | 955.472 |")
    built = pb.build("THY'nin 2025 hasılatı nedir?", [RetrievedChunk(chunk)], is_comparison=False)
    assert "[THYAO 2025 finansal_rapor s.11]" in built.user
    assert DISCLAIMER_TR in built.system
    assert pb.version != "unversioned"


def test_disclaimer_language() -> None:
    assert disclaimer_for("THY'nin hasılatı ne kadar?") == DISCLAIMER_TR
    assert disclaimer_for("What was revenue in 2025?") == DISCLAIMER_EN
