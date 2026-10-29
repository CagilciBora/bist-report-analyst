"""Tests for path-based metadata parsing (src/kaprag/ingestion/metadata.py)."""

import pytest

from kaprag.ingestion.metadata import CompanyRegistry, parse_path


def test_annual_report_path() -> None:
    m = parse_path("data/raw/ASELS/2025_faaliyet_raporu.pdf")
    assert (m.ticker, m.year, m.doc_type, m.period) == ("ASELS", 2025, "faaliyet_raporu", "FY")


def test_disclosure_with_date() -> None:
    m = parse_path("data/raw/BIMAS/2025-03-14_ozel_durum.pdf")
    assert (m.ticker, m.year, m.doc_type, m.period) == ("BIMAS", 2025, "ozel_durum", "2025-03-14")


def test_unknown_doc_type_becomes_other() -> None:
    assert parse_path("data/raw/THYAO/2024_sunum.pdf").doc_type == "other"


def test_explicit_flags_override_path() -> None:
    m = parse_path("data/raw/THYAO/2024_finansal_rapor.pdf", ticker="PGSUS", year=2023,
                   doc_type="faaliyet_raporu", period="Q3")
    assert (m.ticker, m.year, m.doc_type, m.period) == ("PGSUS", 2023, "faaliyet_raporu", "Q3")


def test_missing_info_raises() -> None:
    with pytest.raises(ValueError):
        parse_path("random.pdf")


def test_company_registry_aliases(tmp_path) -> None:
    csv = tmp_path / "c.csv"
    csv.write_text("# review me\nticker,name,aliases,sector\n"
                   "THYAO,Türk Hava Yolları A.O.,THY;Türk Hava Yolları,Ulaştırma\n", encoding="utf-8")
    reg = CompanyRegistry(csv)
    idx = reg.alias_index()
    assert idx["turk hava yollari"] == "THYAO"
    assert idx["thy"] == "THYAO"
    assert reg.name_of("THYAO") == "Türk Hava Yolları A.O."
