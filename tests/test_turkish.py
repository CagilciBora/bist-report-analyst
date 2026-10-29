"""Tests for Turkish case handling (src/kaprag/utils/turkish.py)."""

from kaprag.utils.turkish import normalize_tr, tr_ascii_fold, tr_lower, tr_upper


def test_python_lower_is_wrong_for_turkish() -> None:
    # This documents WHY tr_lower exists: Python adds a combining dot.
    assert "İ".lower() != "i"
    assert "I".lower() == "i"  # Turkish expects "ı"


def test_tr_lower_dotted_and_dotless() -> None:
    assert tr_lower("İSTANBUL") == "istanbul"
    assert tr_lower("ISPARTA") == "ısparta"
    assert tr_lower("İHRACAT") == "ihracat"
    assert tr_lower("KIRILIM") == "kırılım"


def test_tr_upper() -> None:
    assert tr_upper("istanbul") == "İSTANBUL"
    assert tr_upper("ılık") == "ILIK"


def test_roundtrip() -> None:
    word = "Gelir Tablosu İçin Iğdır"
    assert tr_lower(tr_upper(tr_lower(word))) == tr_lower(word)


def test_normalize_handles_decomposed_dot() -> None:
    # "i" + COMBINING DOT ABOVE, as some PDFs store it
    assert normalize_tr("Bi̇lanço") == "bilanço"


def test_ascii_fold() -> None:
    assert tr_ascii_fold("Türk Hava Yolları") == "turk hava yollari"
    assert tr_ascii_fold("BİM") == "bim"
