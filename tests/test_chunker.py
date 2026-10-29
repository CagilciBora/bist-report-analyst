"""Tests for structure-aware chunking and table splitting (src/kaprag/chunking/chunker.py)."""

from kaprag.chunking.chunker import Chunker
from kaprag.config import ChunkingConfig
from kaprag.ingestion.metadata import DocMeta
from kaprag.ingestion.models import Paragraph, ParsedDocument, Table


class WordCounter:
    """Fake token counter: 1 word = 1 token (keeps tests fast and offline)."""

    def count(self, text: str) -> int:
        return len(text.split())


def make_chunker(**kw) -> Chunker:
    cfg = ChunkingConfig(target_tokens=20, max_tokens=25, min_tokens=3, overlap_tokens=6,
                         table_max_tokens=40, **kw)
    return Chunker(cfg, WordCounter())  # type: ignore[arg-type]


META = DocMeta(ticker="ASELS", year=2025, doc_type="faaliyet_raporu", period="FY",
               source_file="data/raw/ASELS/2025_faaliyet_raporu.pdf", company_name="ASELSAN")


def test_table_split_repeats_header_and_caption() -> None:
    table = Table(page=7, header=["Kalem", "2025", "2024"],
                  rows=[[f"Kalem {i}", f"{i}.000", f"{i}.500"] for i in range(30)],
                  caption="Gelir Tablosu (milyon TL)")
    parts = make_chunker().split_table(table)
    assert len(parts) > 1
    for i, part in enumerate(parts, 1):
        lines = part.splitlines()
        assert lines[0].startswith("Tablo: Gelir Tablosu (milyon TL)")
        assert f"(parça {i}/{len(parts)})" in lines[0]
        assert lines[1] == "| Kalem | 2025 | 2024 |"   # header repeated in EVERY part
        assert lines[2].startswith("|---")
    # every data row appears exactly once across all parts
    body = [ln for p in parts for ln in p.splitlines()[3:]]
    assert len(body) == 30


def test_small_table_is_one_chunk() -> None:
    table = Table(page=1, header=["A", "B"], rows=[["x", "1"], ["y", "2"]])
    assert len(make_chunker().split_table(table)) == 1


def test_text_chunks_never_cross_sections() -> None:
    long = " ".join(f"Cümle {i} burada bitiyor." for i in range(12))
    doc = ParsedDocument(source_file="x.pdf", n_pages=3, paragraphs=[
        Paragraph(long, 1, "Riskler"),
        Paragraph(long, 2, "Gelir Tablosu"),
    ])
    chunks = make_chunker().chunk(doc, META)
    assert {c.section_title for c in chunks} == {"Riskler", "Gelir Tablosu"}
    for c in chunks:
        # a chunk from "Riskler" only contains page-1 text and vice versa
        assert (c.page_start, c.page_end) == ((1, 1) if c.section_title == "Riskler" else (2, 2))


def test_text_chunks_respect_size_and_overlap() -> None:
    sentences = [f"Bu {i}. cümle dört kelime." for i in range(20)]
    doc = ParsedDocument(source_file="x.pdf", n_pages=1,
                         paragraphs=[Paragraph(" ".join(sentences), 1, "S")])
    chunks = make_chunker().chunk(doc, META)
    assert len(chunks) > 2
    assert all(len(c.text.split()) <= 25 for c in chunks)
    # overlap: the last sentence of chunk k is the first sentence of chunk k+1
    for a, b in zip(chunks, chunks[1:], strict=False):
        last_sentence = a.text.split(". ")[-1].rstrip(".")
        assert last_sentence in b.text


def test_chunk_metadata_fields() -> None:
    doc = ParsedDocument(source_file="x.pdf", n_pages=1,
                         paragraphs=[Paragraph("Kısa ama yeterince uzun bir paragraf metni burada.", 4, "Giriş")],
                         tables=[Table(page=5, header=["a", "b"], rows=[["1", "2"]], section_title="Ekler")])
    chunks = make_chunker().chunk(doc, META)
    text, table = chunks
    assert text.chunk_type == "text" and table.chunk_type == "table"
    assert text.ticker == "ASELS" and text.year == 2025 and text.period == "FY"
    assert table.page_start == 5 and table.section_title == "Ekler"
    assert text.citation == "[ASELS 2025 faaliyet_raporu s.4]"
    assert text.chunk_id != table.chunk_id
