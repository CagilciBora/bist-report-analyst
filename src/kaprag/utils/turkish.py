"""Turkish-aware case conversion.

The Turkish alphabet has FOUR letters i/ı/İ/I where English has two:
    dotted:   İ (upper)  <->  i (lower)
    dotless:  I (upper)  <->  ı (lower)
Python's str.lower() follows English rules, so:
    "ISTANBUL".lower()  -> "istanbul"   (should be "ıstanbul" in Turkish)
    "İSTANBUL".lower()  -> "i̇stanbul"   (an 'i' + a combining dot U+0307!)
Both break keyword search: the query "ihracat" would never match "İHRACAT".
We fix it by mapping the two special capitals BEFORE calling .lower().
"""

from __future__ import annotations

import unicodedata

# str.maketrans builds a fast character-to-character lookup table.
_LOWER_MAP = str.maketrans({"I": "ı", "İ": "i"})
_UPPER_MAP = str.maketrans({"i": "İ", "ı": "I"})


def tr_lower(text: str) -> str:
    """Lowercase with Turkish rules: I -> ı, İ -> i."""
    return text.translate(_LOWER_MAP).lower()


def tr_upper(text: str) -> str:
    """Uppercase with Turkish rules: i -> İ, ı -> I."""
    return text.translate(_UPPER_MAP).upper()


def normalize_tr(text: str) -> str:
    """NFC-normalise (merges decomposed letters like 'i' + U+0307) and lowercase."""
    text = unicodedata.normalize("NFC", text)
    # A leftover combining dot above (from PDFs that store İ as I + dot) is removed.
    text = text.replace("i̇", "i").replace("̇", "")
    return tr_lower(text)


# Map Turkish letters to ASCII. Used for "fuzzy" alias matching so that a user
# typing "turk hava yollari" (no Turkish keyboard) still finds THYAO.
_ASCII_MAP = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def tr_ascii_fold(text: str) -> str:
    """Lowercase (Turkish rules) and strip Turkish diacritics: 'Türk Hava' -> 'turk hava'."""
    return normalize_tr(text).translate(_ASCII_MAP)
