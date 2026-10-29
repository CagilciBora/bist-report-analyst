"""Embedder: text -> dense vector with a multilingual sentence-transformers model.

Dense embeddings map a text to a point in a ~1024-dim space where texts with
similar MEANING are close (cosine similarity), even with different words:
"net kâr" and "dönem karı" end up near each other. Vectors are L2-normalised,
so cosine similarity == dot product.
"""

from __future__ import annotations

import logging

import numpy as np

from kaprag.config import EmbeddingConfig

log = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    """'auto' -> 'cuda' if an NVIDIA GPU is usable, else 'cpu'."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(self, cfg: EmbeddingConfig) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, done lazily

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        log.info("Loading embedding model %s on %s", cfg.model_name, self.device)
        self.model = SentenceTransformer(cfg.model_name, device=self.device)
        self.model.max_seq_length = cfg.max_seq_length
        if self.device == "cuda" and cfg.fp16:
            self.model.half()  # fp16 halves VRAM: bge-m3 fits in ~1.2 GB
        # e5 models were trained with these prefixes and perform worse without them.
        self._is_e5 = "e5" in cfg.model_name.lower()

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_passages(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        if self._is_e5:
            texts = [f"passage: {t}" for t in texts]
        return self.model.encode(texts, batch_size=self.cfg.batch_size, normalize_embeddings=True,
                                 show_progress_bar=show_progress, convert_to_numpy=True)

    def embed_query(self, text: str) -> np.ndarray:
        if self._is_e5:
            text = f"query: {text}"
        return self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
