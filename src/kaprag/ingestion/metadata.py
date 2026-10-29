"""Document metadata from the file path + the company registry (companies.csv).

Convention: data/raw/<TICKER>/<YEAR>_<doc_type>.pdf
            data/raw/<TICKER>/<YYYY-MM-DD>_<doc_type>.pdf   (KAP disclosures)
Explicit CLI flags (--ticker/--year/--doc-type) override what the path says.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from kaprag.config import DOC_TYPES
from kaprag.utils.turkish import tr_ascii_fold, tr_upper

_NAME_RE = re.compile(r"^(?P<date>\d{4}(?:-\d{2}-\d{2})?)_(?P<doc_type>[a-z_]+)$")


@dataclass
class DocMeta:
    ticker: str
    year: int
    doc_type: str
    period: str          # "FY", "Q1".."Q4" or an ISO date for disclosures
    source_file: str
    company_name: str = ""


def parse_path(path: Path | str, ticker: str | None = None, year: int | None = None,
               doc_type: str | None = None, period: str | None = None) -> DocMeta:
    """Read ticker/year/doc_type from the path; explicit arguments win."""
    path = Path(path)
    m = _NAME_RE.match(path.stem)
    path_ticker = tr_upper(path.parent.name) if path.parent.name else None
    path_date = m.group("date") if m else None
    path_type = m.group("doc_type") if m else None

    ticker = ticker or path_ticker
    date = path_date
    year = year or (int(date[:4]) if date else None)
    doc_type = doc_type or path_type or "other"
    if doc_type not in DOC_TYPES:
        doc_type = "other"
    if not ticker or not year:
        raise ValueError(f"Cannot infer ticker/year from {path}. Use data/raw/<TICKER>/<YEAR>_<doc_type>.pdf "
                         "or pass --ticker/--year.")
    if period is None:
        # A full date in the name means a one-off disclosure; else a full-year report.
        period = date if date and len(date) == 10 else "FY"
    return DocMeta(ticker=ticker, year=int(year), doc_type=doc_type, period=period,
                   source_file=str(path))


@dataclass
class Company:
    ticker: str
    name: str
    aliases: list[str] = field(default_factory=list)
    sector: str = ""


class CompanyRegistry:
    """Loads data/companies.csv; maps names/aliases to tickers."""

    def __init__(self, csv_path: Path) -> None:
        self.companies: dict[str, Company] = {}
        if csv_path.exists():
            with csv_path.open(encoding="utf-8") as fh:
                # Lines starting with '#' are comments (e.g. the review note).
                lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
                for row in csv.DictReader(lines):
                    aliases = [a.strip() for a in row.get("aliases", "").split(";") if a.strip()]
                    c = Company(row["ticker"].strip(), row["name"].strip(), aliases,
                                row.get("sector", "").strip())
                    self.companies[c.ticker] = c

    def name_of(self, ticker: str) -> str:
        c = self.companies.get(ticker)
        return c.name if c else ticker

    def alias_index(self) -> dict[str, str]:
        """ascii-folded alias -> ticker, e.g. 'turk hava yollari' -> 'THYAO'."""
        index: dict[str, str] = {}
        for c in self.companies.values():
            for alias in [c.ticker, c.name, *c.aliases]:
                index[tr_ascii_fold(alias)] = c.ticker
        return index
