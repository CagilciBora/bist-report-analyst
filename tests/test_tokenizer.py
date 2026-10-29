"""Tests for the Turkish BM25 tokenizer (src/kaprag/indexing/tokenizer.py)."""

from kaprag.indexing.tokenizer import TurkishTokenizer


def test_turkish_lowercasing_in_tokens() -> None:
    tok = TurkishTokenizer()
    assert tok.tokenize("İHRACAT GELİRLERİ") == tok.tokenize("ihracat gelirleri")
    assert "ısı" in tok.tokenize("ISI")


def test_apostrophe_suffix_dropped() -> None:
    tok = TurkishTokenizer()
    assert tok.tokenize("ASELSAN'ın THY’nin") == ["aselsan", "thy"]


def test_numbers_and_percentages_are_single_tokens() -> None:
    tok = TurkishTokenizer()
    toks = tok.tokenize("Hasılat 955.472 milyon TL, marj %12,5 ve 3,5% oldu")
    assert "955472" in toks           # thousands dots removed, one token
    assert "%12,5" in toks
    assert "%3,5" in toks             # suffix percent normalised to prefix form
    assert "12" not in toks and "5" not in toks


def test_number_typed_without_dots_matches() -> None:
    tok = TurkishTokenizer()
    assert tok.tokenize("86035") == tok.tokenize("86.035")


def test_ticker_kept() -> None:
    assert "asels" in TurkishTokenizer().tokenize("ASELS 2025 faaliyet raporu")


def test_stopwords_removed() -> None:
    assert TurkishTokenizer().tokenize("ve bu bir için") == []


def test_stemming_merges_inflections() -> None:
    tok = TurkishTokenizer(stemming=True)
    a = tok.tokenize("gelirleri")
    b = tok.tokenize("gelirlerimiz")
    assert a == b
    assert tok.tokenize("86.035") == ["86035"]  # numbers are never stemmed
