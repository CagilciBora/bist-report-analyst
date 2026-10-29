"""Tests for numeric verification (src/kaprag/generation/verifier.py)."""

from kaprag.generation.verifier import NumberVerifier

CTX = {
    "c1": "Tablo: Gelir Tablosu (milyon TL)\n| Hasılat | 955.472 | 745.430 |\n| Net kâr | 118.208 | 113.378 |",
    "c2": "Grup'un ihracatı 1.200.000 bin TL olmuştur. FAVÖK marjı %24,3 seviyesindedir.",
}


def statuses(answer: str) -> dict[str, str]:
    return {c.raw: c.status for c in NumberVerifier().verify(answer, CTX)}


def test_exact_copy_is_verified() -> None:
    s = statuses("2025 hasılatı 955.472 milyon TL oldu [THYAO 2025 finansal_rapor s.11].")
    assert s == {"955.472 milyon TL": "verified"}


def test_unit_scale_conversion() -> None:
    # 1,2 milyar TL == 1.200.000 bin TL
    assert statuses("İhracat 1,2 milyar TL.") == {"1,2 milyar TL": "verified"}
    # rounded: 955.472 milyon -> 955,5 milyar
    assert statuses("Hasılat yaklaşık 955,5 milyar TL.") == {"955,5 milyar TL": "verified"}


def test_wrong_scale_is_caught() -> None:
    # "bin TL" instead of "milyon TL" (the table caption says milyon TL) -> 1000x off -> flagged
    assert statuses("Hasılat 955.472 bin TL.") == {"955.472 bin TL": "unverified"}
    # a number that does not exist anywhere
    assert statuses("Hasılat 812.000 milyon TL.") == {"812.000 milyon TL": "unverified"}


def test_percent_verified() -> None:
    assert statuses("FAVÖK marjı %24,3 oldu.") == {"%24,3": "verified"}


def test_invented_number_flagged() -> None:
    s = statuses("Net kâr 999.999 milyon TL oldu.")
    assert s["999.999 milyon TL"] == "unverified"


def test_shown_calculation_result_accepted() -> None:
    s = statuses("Büyüme: (955.472 - 745.430) / 745.430 = %28,2")
    assert s["955.472"] == "verified" and s["745.430"] == "verified"
    assert s["%28,2"] == "calculated"


def test_wrong_calculation_flagged() -> None:
    checks = NumberVerifier().verify("Büyüme: (955.472 - 745.430) / 745.430 = %35,0", CTX)
    bad = [c for c in checks if c.raw == "%35,0"][0]
    assert bad.status == "unverified" and "calculation" in bad.note


def test_years_citations_and_list_markers_ignored() -> None:
    answer = "1. 2025 yılında [THYAO 2025 finansal_rapor s.11] hasılat arttı.\n2. Net kâr 118.208 milyon TL."
    assert statuses(answer) == {"118.208 milyon TL": "verified"}


def test_spans_point_at_the_number() -> None:
    answer = "Net kâr 999.999 milyon TL."
    c = NumberVerifier().verify(answer, CTX)[0]
    assert answer[c.start:c.end] == "999.999 milyon TL"
