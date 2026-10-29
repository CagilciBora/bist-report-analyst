"""QueryParser: rule-based extraction of filters from a Turkish/English question.

"ASELSAN ile THY'nin son iki yıldaki hasılatını karşılaştır"
  -> tickers [ASELS, THYAO], years [2024, 2025] (relative to the latest indexed
     year), is_comparison=True, is_financial=True
Rules first because they are explainable and testable; an optional LLM parser
(behind `llm.use_llm_query_parser`) can fill in what the rules miss.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from kaprag.utils.turkish import tr_ascii_fold

# Single-word aliases that are also ordinary Turkish words ("garanti" = guarantee,
# "koç" = ram). They only count as company names when written Capitalized.
AMBIGUOUS_ALIASES = {"garanti", "koc", "koza", "tav", "sasa", "enka", "astor", "migros", "pegasus"}

_NUMBER_WORDS = {"bir": 1, "iki": 2, "uc": 3, "dort": 4, "bes": 5, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}

# (regex on ascii-folded lowercase text, period)
PERIOD_PATTERNS = [
    (r"\b(ilk|birinci|1\.?)\s*ceyrek\b|\bq1\b|\bilk uc ay", "Q1"),
    (r"\b(ikinci|2\.?)\s*ceyrek\b|\bq2\b|\bilk yari|\balti aylik", "Q2"),
    (r"\b(ucuncu|3\.?)\s*ceyrek\b|\bq3\b|\bdokuz aylik|\b9 aylik", "Q3"),
    (r"\b(dorduncu|son|4\.?)\s*ceyrek\b|\bq4\b", "Q4"),
    (r"\byillik\b|\byil sonu|\btum yil|\bfull year|\bannual\b", "FY"),
]

# Explicit document mentions -> hard filter.
DOC_TYPE_PATTERNS = [
    (r"faaliyet raporu|entegre (faaliyet )?rapor|annual report", "faaliyet_raporu"),
    (r"finansal rapor|finansal tablo|bagimsiz denetim|financial statement", "finansal_rapor"),
    (r"ozel durum|kap aciklamasi|material event", "ozel_durum"),
]
# Words that suggest statement data -> soft hint only (they also live in annual reports).
STATEMENT_HINTS = r"bilanco|gelir tablosu|nakit akis|dipnot"

FINANCIAL_WORDS = re.compile(
    r"gelir|hasilat|ciro|satis|net kar|net donem kar|donem kar|\bkar\b|kari|karli|favok|ebitda|borc|nakit|ihracat"
    r"|oran|marj|ozkaynak|varlik|bilanco|tutar|yatirim|harcama|maliyet|gider|temettu|revenue|profit"
)
COMPARISON_WORDS = re.compile(
    r"karsilastir|kiyasla|\bfark|hangisi daha|hangisinin|gore daha|\bvs\b|versus|compare|arasinda"
)


class ParsedQuery(BaseModel):
    question: str
    tickers: list[str] = Field(default_factory=list)
    years: list[int] = Field(default_factory=list)
    periods: list[str] = Field(default_factory=list)
    doc_types: list[str] = Field(default_factory=list)      # hard filter
    doc_type_hints: list[str] = Field(default_factory=list)  # soft hint, not filtered
    is_comparison: bool = False
    is_financial: bool = False
    notes: list[str] = Field(default_factory=list)


class QueryParser:
    def __init__(self, alias_index: dict[str, str], known_tickers: set[str],
                 latest_year: int | None = None) -> None:
        self.known_tickers = known_tickers
        self.latest_year = latest_year
        # Longest alias first so "türk hava yolları" wins over "thy" overlaps.
        self.aliases = sorted(alias_index.items(), key=lambda kv: -len(kv[0]))

    # ------------------------------------------------------------ tickers
    def _tickers(self, question: str) -> list[str]:
        folded = tr_ascii_fold(question)
        found: list[tuple[int, str]] = []
        taken = [False] * len(folded)
        for alias, ticker in self.aliases:
            # whole word, optionally followed by an apostrophe suffix ("aselsan'in")
            for m in re.finditer(rf"(?<![\w]){re.escape(alias)}(?![\w])", folded):
                if any(taken[m.start():m.end()]):
                    continue
                if alias in AMBIGUOUS_ALIASES and not question[m.start():m.start() + 1].isupper():
                    continue  # "garanti" (guarantee) vs "Garanti" (the bank)
                for i in range(m.start(), m.end()):
                    taken[i] = True
                found.append((m.start(), ticker))
        ordered: list[str] = []
        for _, t in sorted(found):
            if t not in ordered:
                ordered.append(t)
        return ordered

    # -------------------------------------------------------------- years
    def _years(self, folded: str) -> tuple[list[int], list[str]]:
        notes: list[str] = []
        years = {int(y) for y in re.findall(r"(?<!\d)(20[0-4]\d|19[89]\d)(?!\d)", folded)}
        if self.latest_year:
            m = re.search(r"son (\w+) (yil|sene)", folded)
            if m and m.group(1) in _NUMBER_WORDS:
                n = _NUMBER_WORDS[m.group(1)]
                years |= set(range(self.latest_year - n + 1, self.latest_year + 1))
                notes.append(f"'{m.group(0)}' -> {self.latest_year - n + 1}-{self.latest_year} "
                             f"(latest indexed year {self.latest_year})")
            elif re.search(r"\b(son yil|bu yil|en son|guncel)\b", folded):
                years.add(self.latest_year)
                notes.append(f"'latest' -> {self.latest_year}")
            if re.search(r"\b(gecen yil|onceki yil|bir onceki yil)\b", folded):
                years.add(self.latest_year - 1)
        return sorted(years), notes

    # -------------------------------------------------------------- parse
    def parse(self, question: str) -> ParsedQuery:
        folded = tr_ascii_fold(question)
        q = ParsedQuery(question=question)
        q.tickers = self._tickers(question)
        q.years, notes = self._years(folded)
        q.notes.extend(notes)
        q.periods = sorted({p for pat, p in PERIOD_PATTERNS if re.search(pat, folded)})
        q.doc_types = sorted({d for pat, d in DOC_TYPE_PATTERNS if re.search(pat, folded)})
        if re.search(STATEMENT_HINTS, folded):
            q.doc_type_hints = ["finansal_rapor"]
        q.is_financial = bool(FINANCIAL_WORDS.search(folded))
        q.is_comparison = (len(q.tickers) > 1 or len(q.years) > 1
                           or bool(COMPARISON_WORDS.search(folded)))
        if not q.tickers and not q.years and not q.doc_types:
            q.notes.append("No company/year/doc filter inferred: searching across ALL documents.")
        return q
