"""Reranker: a cross-encoder that re-scores (question, chunk) pairs.

Bi-encoder (our embedder): question and chunk are embedded SEPARATELY, then
compared with cosine. Fast (chunk vectors are precomputed) but the two texts
never "see" each other.
Cross-encoder: question and chunk go through the transformer TOGETHER, so
attention can match "2025 hasılatı" to the exact "Hasılat | 955.472" row.
Much more accurate, but must run for every pair at query time, so we only
apply it to the top ~20 fused candidates.
"""

from __future__ import annotations

import logging

from kaprag.indexing.embedder import resolve_device

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str, device: str = "auto", max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder

        self.device = resolve_device(device)
        log.info("Loading reranker %s on %s", model_name, self.device)
        self.model = CrossEncoder(model_name, device=self.device, max_length=max_length)
        if self.device == "cuda":
            self.model.model.half()

    def score(self, question: str, passages: list[str]) -> list[float]:
        """Relevance in [0, 1] for each passage (sigmoid of the model logit)."""
        if not passages:
            return []
        scores = self.model.predict([(question, p) for p in passages], batch_size=8,
                                    show_progress_bar=False)
        return [float(s) for s in scores]
