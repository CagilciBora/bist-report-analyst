"""TurkishTokenizer: turns text into BM25 tokens.

BM25 matches EXACT tokens, so the tokenizer decides what "the same word" means:
* Turkish lowercasing (İ->i, I->ı) so "İHRACAT" == "ihracat".
* Apostrophe suffixes are cut: "ASELSAN'ın" -> "aselsan", "THY'nin" -> "thy".
* Numbers stay ONE token, written canonically without thousands dots:
  "86.035" -> "86035", "%12,5" -> "%12,5". A user typing "86035" or "86.035"
  matches the same token.
* Optional Snowball stemming: "gelirleri", "gelirlerimiz" -> "gelir".
"""

from __future__ import annotations

import re

import snowballstemmer

from kaprag.utils.turkish import normalize_tr

# Order matters: percentages first, then numbers, then words.
_TOKEN_RE = re.compile(
    r"""
      (?P<pct>%\s?\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*\s?%)      # %12,5  or 12,5%
    | (?P<num>\d+(?:[.,]\d+)*)                             # 1.234.567,89
    | (?P<word>[^\W\d_]+)(?:['’][^\W\d_]+)?                # word, suffix after ' dropped
    """,
    re.VERBOSE,
)

# Very frequent words with no search value. Kept short on purpose: BM25's IDF
# already down-weights common words, this list just saves index space.
STOPWORDS = frozenset("""
ve ile bu bir da de için olarak olan gibi daha çok en ise ya veya ancak her
şu o ki mi mu mı mü ne kadar sonra önce göre ait üzere yer ilgili
the and of to in for on is are with by as at an
""".split())


def _canonical_number(num: str) -> str:
    """'1.234.567,89' -> '1234567,89'; '86.035' -> '86035'; '3.5' stays '3.5'."""
    if "," in num:                              # Turkish decimal comma present
        int_part, dec = num.rsplit(",", 1)
        return int_part.replace(".", "") + "," + dec
    groups = num.split(".")
    if len(groups) > 1 and all(len(g) == 3 for g in groups[1:]):
        return num.replace(".", "")             # thousands dots
    return num


class TurkishTokenizer:
    def __init__(self, stemming: bool = False) -> None:
        self.stemming = stemming
        self._stemmer = snowballstemmer.stemmer("turkish") if stemming else None

    def tokenize(self, text: str) -> list[str]:
        text = normalize_tr(text)
        tokens: list[str] = []
        for m in _TOKEN_RE.finditer(text):
            if m.group("pct"):
                tokens.append("%" + _canonical_number(re.sub(r"[%\s]", "", m.group("pct"))))
            elif m.group("num"):
                tokens.append(_canonical_number(m.group("num")))
            else:
                word = m.group("word")
                if word in STOPWORDS or len(word) < 2:
                    continue
                if self._stemmer is not None and len(word) > 3:
                    word = self._stemmer.stemWord(word)
                tokens.append(word)
        return tokens
