"""DocumentLoader: path -> (metadata, parsed document, chunks), with a parse cache.

Parsing a 260-page report takes minutes (table detection is the slow part),
so the ParsedDocument is cached on disk under data/cache/parsed/, keyed by
the file's content hash + the parsing settings. Change a parsing setting in
config.yaml and the cache key changes, so the file is parsed again.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from kaprag.chunking.chunker import Chunker
from kaprag.chunking.models import Chunk
from kaprag.chunking.token_counter import TokenCounter
from kaprag.config import PROJECT_ROOT, Settings
from kaprag.ingestion.metadata import CompanyRegistry, DocMeta, parse_path
from kaprag.ingestion.models import ParsedDocument
from kaprag.ingestion.pdf_parser import PDFParser

log = logging.getLogger(__name__)
PARSER_VERSION = "3"  # bump when parsing code changes to invalidate caches


@lru_cache(maxsize=256)
def file_hash(path: str) -> str:
    """SHA-256 of the file content (same file renamed -> same hash)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def settings_hash(obj: object) -> str:
    """Short hash of a Pydantic settings object (used in cache keys/manifest)."""
    data = json.dumps(obj.model_dump(mode="json"), sort_keys=True)  # type: ignore[attr-defined]
    return hashlib.sha256(data.encode()).hexdigest()[:10]


@dataclass
class LoadedDocument:
    meta: DocMeta
    parsed: ParsedDocument
    chunks: list[Chunk]
    file_hash: str


class DocumentLoader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.parser = PDFParser(settings.parsing)
        self.registry = CompanyRegistry(settings.paths.companies_csv)
        self.counter = TokenCounter(settings.embedding.model_name)
        self.chunker = Chunker(settings.chunking, self.counter)
        self.cache_dir = PROJECT_ROOT / "data" / "cache" / "parsed"

    def parse_cached(self, path: Path) -> tuple[ParsedDocument, str]:
        fh = file_hash(str(path))
        key = f"{fh[:16]}_{settings_hash(self.settings.parsing)}_v{PARSER_VERSION}.pkl"
        cache_file = self.cache_dir / key
        if cache_file.exists():
            with cache_file.open("rb") as f:
                parsed = pickle.load(f)
            parsed.source_file = str(path)
            return parsed, fh
        log.info("Parsing %s ...", path)
        parsed = self.parser.parse(path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with cache_file.open("wb") as f:
            pickle.dump(parsed, f)
        return parsed, fh

    def load(self, path: Path | str, ticker: str | None = None, year: int | None = None,
             doc_type: str | None = None, period: str | None = None) -> LoadedDocument:
        path = Path(path)
        meta = parse_path(path, ticker, year, doc_type, period)
        meta.company_name = self.registry.name_of(meta.ticker)
        try:  # store paths relative to the project so the index is portable
            meta.source_file = str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            meta.source_file = str(path)
        parsed, fh = self.parse_cached(path)
        chunks = self.chunker.chunk(parsed, meta)
        return LoadedDocument(meta, parsed, chunks, fh)

    def discover(self, root: Path | None = None) -> list[Path]:
        """All PDFs under data/raw (or under the given folder / single file)."""
        root = root or self.settings.paths.raw_dir
        if root.is_file():
            return [root]
        return sorted(p for p in root.rglob("*") if p.suffix.lower() == ".pdf")
