"""Tests for the query parser, RRF fusion and metadata filtering."""

from pathlib import Path

import pytest

from kaprag.chunking.models import Chunk
from kaprag.config import BM25Config
from kaprag.indexing.bm25_index import BM25Index
from kaprag.indexing.chunk_store import ChunkStore
from kaprag.indexing.filters import SearchFilter
from kaprag.ingestion.metadata import CompanyRegistry
from kaprag.retrieval.fusion import reciprocal_rank_fusion
from kaprag.retrieval.query_parser import QueryParser

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def parser() -> QueryParser:
    reg = CompanyRegistry(ROOT / "data" / "companies.csv")
    return QueryParser(reg.alias_index(), set(reg.companies), latest_year=2025)


# ------------------------------------------------------------ query parser
@pytest.mark.parametrize(
    ("question", "tickers"),
    [
        ("ASELS 2025 hasılatı nedir?", ["ASELS"]),
        ("asels'in borcu", ["ASELS"]),
        ("Aselsan'ın ihracatı ne kadar?", ["ASELS"]),
        ("Türk Hava Yolları yolcu sayısı", ["THYAO"]),
        ("turk hava yollari filo", ["THYAO"]),          # typed without Turkish letters
        ("THY'nin net kârı", ["THYAO"]),
        ("BİM mağaza sayısı", ["BIMAS"]),
        ("ASELSAN ile THY'yi karşılaştır", ["ASELS", "THYAO"]),
        ("Garanti BBVA kredi büyümesi", ["GARAN"]),
    ],
)
def test_tickers_and_aliases(parser: QueryParser, question: str, tickers: list[str]) -> None:
    assert parser.parse(question).tickers == tickers


def test_ambiguous_alias_needs_capital(parser: QueryParser) -> None:
    # "garanti" = guarantee; must not trigger GARAN
    assert parser.parse("verilen garanti tutarı nedir").tickers == []


def test_years_explicit(parser: QueryParser) -> None:
    q = parser.parse("ASELS 2024 yılı ve 2025 hasılat")
    assert q.years == [2024, 2025]
    assert q.is_comparison  # two years -> comparison


def test_relative_years(parser: QueryParser) -> None:
    assert parser.parse("BİM'in son iki yıldaki satışları").years == [2024, 2025]
    assert parser.parse("son 3 yıl").years == [2023, 2024, 2025]
    assert parser.parse("THY geçen yıl kaç yolcu taşıdı").years == [2024]


def test_period_and_doc_type(parser: QueryParser) -> None:
    q = parser.parse("ASELS ilk çeyrek faaliyet raporu")
    assert q.periods == ["Q1"]
    assert q.doc_types == ["faaliyet_raporu"]
    assert parser.parse("Q3 sonuçları").periods == ["Q3"]
    assert parser.parse("yıllık hasılat").periods == ["FY"]
    # statement words are only a soft hint, not a filter
    q = parser.parse("THY bilanço toplam varlıklar")
    assert q.doc_types == [] and q.doc_type_hints == ["finansal_rapor"]


def test_financial_and_comparison_flags(parser: QueryParser) -> None:
    q = parser.parse("Hangisi daha karlı, ASELSAN mı BİM mi?")
    assert q.is_comparison and q.is_financial
    assert not parser.parse("ASELSAN'ın sürdürülebilirlik politikası").is_financial


def test_no_filter_note(parser: QueryParser) -> None:
    q = parser.parse("Karbon emisyonu hedefleri nelerdir?")
    assert not q.tickers and not q.years
    assert any("ALL documents" in n for n in q.notes)


# ------------------------------------------------------------------ RRF
def test_rrf_toy_example() -> None:
    dense = ["A", "B", "C"]
    bm25 = ["C", "A", "D"]
    fused = dict(reciprocal_rank_fusion([dense, bm25], k=60))
    assert fused["A"] == pytest.approx(1 / 61 + 1 / 62)   # rank 1 + rank 2
    assert fused["C"] == pytest.approx(1 / 63 + 1 / 61)   # rank 3 + rank 1
    assert fused["B"] == pytest.approx(1 / 62)
    order = [d for d, _ in reciprocal_rank_fusion([dense, bm25], k=60)]
    assert order == ["A", "C", "B", "D"]   # found by both > found by one


def test_rrf_small_k_rewards_top_rank_more() -> None:
    gap_small_k = 1 / (1 + 1) - 1 / (1 + 2)
    gap_big_k = 1 / (60 + 1) - 1 / (60 + 2)
    assert gap_small_k > gap_big_k


# ----------------------------------------------------- metadata filtering
def _chunk(cid: str, ticker: str, year: int, text: str, kind: str = "text") -> Chunk:
    return Chunk(cid, ticker, ticker, year, "FY", "faaliyet_raporu", "x.pdf", 1, 1, "", kind, text)


def test_filter_matches() -> None:
    payload = {"ticker": "ASELS", "year": 2025, "period": "FY", "doc_type": "faaliyet_raporu",
               "chunk_type": "table"}
    assert SearchFilter(tickers=["ASELS"], years=[2025]).matches(payload)
    assert not SearchFilter(tickers=["THYAO"]).matches(payload)
    assert not SearchFilter(years=[2024]).matches(payload)
    assert SearchFilter().matches(payload) and SearchFilter().is_empty()
    assert SearchFilter(tickers=["ASELS"]).to_qdrant() is not None
    assert SearchFilter().to_qdrant() is None


def test_bm25_scores_only_filtered_subset(tmp_path: Path) -> None:
    store = ChunkStore(tmp_path / "chunks.jsonl")
    store.replace_doc("a", [_chunk("a1", "ASELS", 2025, "hasılat ihracat savunma"),
                            _chunk("a2", "ASELS", 2024, "hasılat geçen yıl")])
    store.replace_doc("t", [_chunk("t1", "THYAO", 2025, "hasılat hasılat hasılat yolcu")])
    # filler chunks: BM25Okapi gives a NEGATIVE idf to terms found in over half of the corpus
    store.replace_doc("f", [_chunk(f"f{i}", "BIMAS", 2025, f"mağaza çalışan {i}") for i in range(6)])
    bm25 = BM25Index(store, BM25Config(), tmp_path)
    # Without a filter THYAO's chunk (3x "hasılat") wins ...
    assert bm25.search("hasılat", SearchFilter(), 5)[0][0] == "t1"
    # ... with a filter, only ASELS 2025 chunks are even scored.
    hits = bm25.search("hasılat", SearchFilter(tickers=["ASELS"], years=[2025]), 5)
    assert [cid for cid, _ in hits] == ["a1"]
