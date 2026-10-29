"""Optional LLM-based query parser (flag: llm.use_llm_query_parser).

The rule-based parser runs first; the LLM may ADD tickers/years the rules
missed (e.g. "Türkiye'nin bayrak taşıyıcı havayolu" -> THYAO). Its output is
JSON validated by Pydantic and restricted to known tickers, so a
hallucinated ticker can never reach the filters. On any error we silently
keep the rule-based result.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, ValidationError

from kaprag.generation.llm import LLMClient
from kaprag.retrieval.query_parser import ParsedQuery

log = logging.getLogger(__name__)

SYSTEM = ("Extract search filters from a question about Borsa Istanbul companies. "
          'Reply with JSON only: {"tickers": [..], "years": [..], "is_comparison": bool}. '
          "Use BIST tickers (e.g. THYAO, ASELS, BIMAS). Empty lists if not mentioned.")


class _LLMFilters(BaseModel):
    tickers: list[str] = []
    years: list[int] = []
    is_comparison: bool = False


class LLMQueryParser:
    def __init__(self, llm: LLMClient, known_tickers: set[str]) -> None:
        self.llm = llm
        self.known = known_tickers

    def refine(self, parsed: ParsedQuery) -> ParsedQuery:
        try:
            raw = self.llm.generate(SYSTEM, parsed.question)
            data = _LLMFilters.model_validate(json.loads(re.search(r"\{.*\}", raw, re.S).group(0)))
        except (ValidationError, json.JSONDecodeError, AttributeError, Exception) as exc:
            log.info("LLM query parser failed (%s); using rule-based filters", exc)
            return parsed
        p = parsed.model_copy(deep=True)
        for t in data.tickers:
            if t.upper() in self.known and t.upper() not in p.tickers:
                p.tickers.append(t.upper())
        for y in data.years:
            if 1990 <= y <= 2100 and y not in p.years:
                p.years.append(y)
        p.years.sort()
        p.is_comparison = p.is_comparison or data.is_comparison or len(p.tickers) > 1
        p.notes.append("LLM query parser refined the rule-based filters.")
        return p
