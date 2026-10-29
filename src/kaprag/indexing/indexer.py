"""Indexer: ingest / list / remove documents, incrementally.

A manifest (data/index/manifest.json) remembers for every document:
its file hash and a hash of the settings that shaped its chunks/vectors
(parsing + chunking + embedding model). On `kap ingest`:
  * same file + same settings  -> skipped (seconds instead of minutes)
  * changed file or settings   -> old points deleted, document re-indexed
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from kaprag.config import Settings
from kaprag.indexing.chunk_store import ChunkStore
from kaprag.indexing.dense_index import DenseIndex
from kaprag.ingestion.loader import PARSER_VERSION, DocumentLoader, file_hash

log = logging.getLogger(__name__)


def doc_key(ticker: str, year: int, doc_type: str, period: str) -> str:
    return f"{ticker}|{year}|{doc_type}|{period}"


@dataclass
class IngestResult:
    source_file: str
    doc_key: str
    status: str          # indexed | skipped | failed | empty
    n_chunks: int = 0
    seconds: float = 0.0
    message: str = ""


class Indexer:
    def __init__(self, settings: Settings, dense: DenseIndex | None = None) -> None:
        self.settings = settings
        self.index_dir = settings.paths.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.index_dir / "manifest.json"
        self.manifest: dict[str, dict] = (
            json.loads(self.manifest_path.read_text(encoding="utf-8")) if self.manifest_path.exists() else {}
        )
        self.store = ChunkStore(self.index_dir / "chunks.jsonl")
        self.dense = dense or DenseIndex(self.index_dir / "qdrant")
        self._loader: DocumentLoader | None = None
        self._embedder = None

    # Heavy objects are created only when a document actually needs indexing.
    @property
    def loader(self) -> DocumentLoader:
        if self._loader is None:
            self._loader = DocumentLoader(self.settings)
        return self._loader

    @property
    def embedder(self):
        if self._embedder is None:
            from kaprag.indexing.embedder import Embedder

            self._embedder = Embedder(self.settings.embedding)
        return self._embedder

    def settings_fingerprint(self) -> str:
        """Hash of everything that changes chunks or vectors."""
        s = self.settings
        relevant = {
            "parsing": s.parsing.model_dump(mode="json"),
            "chunking": s.chunking.model_dump(mode="json"),
            "embedding": {"model": s.embedding.model_name, "max_seq": s.embedding.max_seq_length},
            "parser_version": PARSER_VERSION,
        }
        return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]

    # ------------------------------------------------------------- ingest
    def ingest(self, paths: list[Path], ticker: str | None = None, year: int | None = None,
               doc_type: str | None = None, period: str | None = None,
               force: bool = False) -> list[IngestResult]:
        results: list[IngestResult] = []
        fingerprint = self.settings_fingerprint()
        changed = False
        for path in paths:
            t0 = time.perf_counter()
            try:
                from kaprag.ingestion.metadata import parse_path

                meta = parse_path(path, ticker, year, doc_type, period)
                key = doc_key(meta.ticker, meta.year, meta.doc_type, meta.period)
                fh = file_hash(str(path))
                entry = self.manifest.get(key)
                if (not force and entry and entry["file_hash"] == fh
                        and entry["settings"] == fingerprint):
                    results.append(IngestResult(str(path), key, "skipped", entry["n_chunks"],
                                                message="already indexed with same file + settings"))
                    continue
                doc = self.loader.load(path, ticker, year, doc_type, period)
                if not doc.chunks:
                    results.append(IngestResult(str(path), key, "empty", 0, time.perf_counter() - t0,
                                                f"no extractable text (scanned {len(doc.parsed.scanned_pages)}, "
                                                f"garbled {len(doc.parsed.garbled_pages)} pages); enable OCR"))
                    continue
                log.info("Embedding %d chunks of %s", len(doc.chunks), path.name)
                vectors = self.embedder.embed_passages([c.index_text for c in doc.chunks])
                self.dense.delete_doc(key)          # remove the old version first
                self.dense.upsert(doc.chunks, vectors, key)
                self.store.replace_doc(key, doc.chunks)
                self.manifest[key] = {
                    "source_file": doc.meta.source_file, "ticker": meta.ticker, "year": meta.year,
                    "doc_type": meta.doc_type, "period": meta.period,
                    "company_name": doc.meta.company_name, "file_hash": fh, "settings": fingerprint,
                    "n_chunks": len(doc.chunks), "n_pages": doc.parsed.n_pages,
                    "n_tables": len(doc.parsed.tables), "scanned_pages": doc.parsed.scanned_pages,
                    "garbled_pages": len(doc.parsed.garbled_pages),
                    "embedding_model": self.settings.embedding.model_name,
                    "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                changed = True
                self._save()  # save after every document: a crash keeps finished work
                results.append(IngestResult(str(path), key, "indexed", len(doc.chunks),
                                            time.perf_counter() - t0))
            except Exception as exc:
                log.exception("Failed to ingest %s", path)
                results.append(IngestResult(str(path), "", "failed", message=str(exc)))
        if changed:
            self._save()
        return results

    # -------------------------------------------------------- list/remove
    def list_documents(self) -> list[dict]:
        return [{"doc_key": k, **v} for k, v in sorted(self.manifest.items())]

    def remove(self, ticker: str, year: int, doc_type: str) -> list[str]:
        keys = [k for k, v in self.manifest.items()
                if v["ticker"] == ticker and v["year"] == year and v["doc_type"] == doc_type]
        for key in keys:
            self.dense.delete_doc(key)
            self.store.remove_doc(key)
            del self.manifest[key]
        if keys:
            self._save()
        return keys

    def _save(self) -> None:
        self.store.save()
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
