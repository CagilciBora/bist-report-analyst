"""Tests for the Turkish number parser (src/kaprag/utils/numbers.py)."""

import pytest

from kaprag.utils.numbers import extract_numbers, parse_tr_number


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("1.234.567,89", 1234567.89),   # Turkish thousands + decimal comma
        ("1.234", 1234.0),              # dot with a group of 3 = thousands
        ("12,5", 12.5),                 # decimal comma
        ("0,75", 0.75),
        ("1,234,567", 1234567.0),       # English thousands
        ("1,234.56", 1234.56),          # English decimal
        ("3.5", 3.5),                   # English decimal point (not group of 3)
        ("42", 42.0),
        ("(1.234)", -1234.0),           # accounting negative
        ("(12,5)", -12.5),
        ("-1.234,5", -1234.5),
        ("−7", -7.0),                   # unicode minus
    ],
)
def test_plain_values(text: str, value: float) -> None:
    n = parse_tr_number(text)
    assert n is not None
    assert n.value == pytest.approx(value)


def test_percent_prefix_and_suffix() -> None:
    a = parse_tr_number("%12,5")
    b = parse_tr_number("12,5%")
    assert a and b
    assert a.is_percent and b.is_percent
    assert a.value == b.value == pytest.approx(12.5)


@pytest.mark.parametrize(
    ("text", "absolute", "currency"),
    [
        ("1,2 milyar TL", 1.2e9, "TRY"),
        ("350 milyon TL", 350e6, "TRY"),
        ("1.200.000 bin TL", 1.2e9, "TRY"),
        ("5 bin TL", 5e3, "TRY"),
        ("2,5 milyar USD", 2.5e9, "USD"),
        ("100 milyon ABD Doları", 100e6, "USD"),
        ("10 TL", 10.0, "TRY"),
        ("₺10", 10.0, "TRY"),
        ("$3,5", 3.5, "USD"),
        ("7 trilyon", 7e12, None),
    ],
)
def test_scale_and_currency(text: str, absolute: float, currency: str | None) -> None:
    n = parse_tr_number(text)
    assert n is not None
    assert n.absolute == pytest.approx(absolute)
    assert n.currency == currency


def test_scale_words_are_case_insensitive_and_turkish() -> None:
    n = parse_tr_number("3 Milyar TL")
    assert n and n.absolute == pytest.approx(3e9)


def test_decimals_are_counted() -> None:
    n = parse_tr_number("1,25 milyar")
    assert n and n.decimals == 2


def test_no_number() -> None:
    assert parse_tr_number("hiç sayı yok") is None


def test_extract_numbers_skips_years_dates_and_citations() -> None:
    text = (
        "2025 yılında hasılat 1,2 milyar TL oldu (31.12.2025 itibarıyla), "
        "marj %12,5 [ASELS 2025 faaliyet_raporu s.14]."
    )
    nums = extract_numbers(text)
    raws = [n.raw for n in nums]
    assert raws == ["1,2 milyar TL", "%12,5"]


def test_extract_numbers_inside_words_ignored() -> None:
    nums = extract_numbers("Filoda A320neo ve 737 MAX uçakları var.")
    assert [n.value for n in nums] == [737.0]


def test_extract_negative_table_value() -> None:
    nums = extract_numbers("| Finansman giderleri | (12.345) | (9.876) |")
    assert [n.value for n in nums] == [-12345.0, -9876.0]
