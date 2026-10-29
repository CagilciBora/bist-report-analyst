"""BM25Index: keyword search with rank_bm25 + our TurkishTokenizer.

BM25 score of document d for query q:
    score(d, q) = Σ_t  IDF(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 − b + b·|d|/avgdl) )
  * tf   = how often term t occurs in d (more is better, with diminishing returns via k1)
  * IDF  = log-ish(N / n_t): rare terms ("FAVÖK", "ASELS", "955472") weigh more
  * |d|/avgdl = length normalisation (b): long chunks don't win just by being long
The tokenized corpus is cached to disk, keyed by the tokenizer settings, so a
restart does not re-tokenize everything.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from kaprag.config import BM25Config
from kaprag.indexing.chunk_store import ChunkStore
from kaprag.indexing.filters import SearchFilter
from kaprag.indexing.tokenizer import TurkishTokenizer

log = logging.getLogger(__name__)


class BM25Index:
    def __init__(self, store: ChunkStore, cfg: BM25Config, cache_dir: Path) -> None:
        self.store = store
        self.cfg = cfg
        self.tokenizer = TurkishTokenizer(stemming=cfg.stemming)
        self.cache_dir = cache_dir
        self.ids: list[str] = []
        self.payloads: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self.build()

    def _cache_file(self) -> Path:
        # Key = which chunks are indexed + tokenizer settings.
        h = hashlib.sha256()
        h.update(f"stem={self.cfg.stemming}".encode())
        for cid in sorted(self.store.chunks):
            h.update(cid.encode())
        return self.cache_dir / f"bm25_tokens_{h.hexdigest()[:16]}.pkl"

    def build(self) -> None:
        """(Re)build BM25 from the chunk store, loading cached tokens if possible."""
        self.ids = list(self.store.chunks)
        self.payloads = [
            {"ticker": c.ticker, "year": c.year, "period": c.period, "doc_type": c.doc_type,
             "chunk_type": c.chunk_type}
            for c in self.store.chunks.values()
        ]
        if not self.ids:
            self.bm25 = None
            return
        cache = self._cache_file()
        if cache.exists():
            with cache.open("rb") as fh:
                corpus = pickle.load(fh)
        else:
            corpus = [self.tokenizer.tokenize(c.index_text) for c in self.store.chunks.values()]
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for old in self.cache_dir.glob("bm25_tokens_*.pkl"):
                old.unlink()  # keep only the current cache
            with cache.open("wb") as fh:
                pickle.dump(corpus, fh)
        self.bm25 = BM25Okapi(corpus, k1=self.cfg.k1, b=self.cfg.b)
        log.debug("BM25 built over %d chunks (stemming=%s)", len(self.ids), self.cfg.stemming)

    def search(self, query: str, flt: SearchFilter, top_k: int) -> list[tuple[str, float]]:
        """Score ONLY the chunks that pass the filter; return [(chunk_id, score)]."""
        if self.bm25 is None:
            return []
        q = self.tokenizer.tokenize(query)
        if not q:
            return []
        subset = [i for i, p in enumerate(self.payloads) if flt.matches(p)]
        if not subset:
            return []
        scores = np.asarray(self.bm25.get_batch_scores(q, subset))
        order = np.argsort(-scores)[:top_k]
        return [(self.ids[subset[i]], float(scores[i])) for i in order if scores[i] > 0]
