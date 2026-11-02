"""Tests for the judge's evidence picker (eval/judge.py).

Both false alarms in the first LLM dataset review came from the judge not
being SHOWN the right chunk. These tests pin the fixes.
"""

import importlib.util
from pathlib import Path

from kaprag.chunking.models import Chunk
from kaprag.indexing.chunk_store import ChunkStore

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("judge", ROOT / "eval" / "judge.py")
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def _chunk(cid: str, page: int, text: str) -> Chunk:
    return Chunk(cid, "BIMAS", "BİM", 2024, "FY", "faaliyet_raporu", "x.pdf", page, page, "", "text", text)


def _store(tmp_path: Path, chunks: list[Chunk]) -> ChunkStore:
    store = ChunkStore(tmp_path / "c.jsonl")
    store.replace_doc("d", chunks)
    return store


def test_every_expected_page_is_represented(tmp_path: Path) -> None:
    filler = [_chunk(f"f{i}", 9, f"ilgisiz metin {i}") for i in range(10)]
    store = _store(tmp_path, [*filler, _chunk("p18", 18, "başka bir sayfa")])
    item = {"question": "q", "expected_numbers": [],
            "expected_pages": [{"ticker": "BIMAS", "year": 2024, "doc_type": "faaliyet_raporu", "page": 9},
                               {"ticker": "BIMAS", "year": 2024, "doc_type": "faaliyet_raporu", "page": 18}]}
    ids = [c.chunk_id for c in judge.gold_chunks(store, item)]
    assert "p18" in ids  # page 18 must not be crowded out by page 9 chunks


def test_number_with_footnote_mark_still_ranks_first(tmp_path: Path) -> None:
    store = _store(tmp_path, [_chunk("a", 18, "Extel ödülleri"), _chunk("b", 18, "Ödüller"),
                              _chunk("c", 18, "Ödül listesi"), _chunk("d", 18, "Ek bilgi"),
                              _chunk("hit", 18, "2024 yılında 18,1* milyar TL yatırım gerçekleştirmiştir.")])
    item = {"question": "yatırım ne kadar?", "expected_numbers": ["18,1 milyar TL"],
            "expected_pages": [{"ticker": "BIMAS", "year": 2024, "doc_type": "faaliyet_raporu", "page": 18}]}
    assert judge.gold_chunks(store, item)[0].chunk_id == "hit"


def test_evidence_quote_wins_for_text_questions(tmp_path: Path) -> None:
    store = _store(tmp_path, [_chunk(f"x{i}", 5, f"yönetim kurulu toplantı {i}") for i in range(6)]
                   + [_chunk("hit", 5, "Yönetim Kurulu Başkanı Prof. Dr. Ahmet Bolat")])
    item = {"question": "Yönetim kurulu başkanı kim?", "expected_numbers": [],
            "evidence": "Yönetim Kurulu Başkanı Prof. Dr. Ahmet Bolat",
            "expected_pages": [{"ticker": "BIMAS", "year": 2024, "doc_type": "faaliyet_raporu", "page": 5}]}
    assert judge.gold_chunks(store, item)[0].chunk_id == "hit"
