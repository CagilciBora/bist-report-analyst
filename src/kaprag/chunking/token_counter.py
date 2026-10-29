"""Count tokens with the SAME tokenizer the embedding model uses.

"400 tokens" only means something relative to a tokenizer. Turkish words are
split into more sub-word tokens than English ones, so counting words would
under-estimate chunk sizes. We load only the tokenizer files (a few MB), not
the model weights. If loading fails (offline), we fall back to ~4 chars/token.
"""

from __future__ import annotations

import logging
from functools import lru_cache

log = logging.getLogger(__name__)


class TokenCounter:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        try:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(model_name)
        except Exception as exc:  # pragma: no cover - network/offline issues
            log.warning("Tokenizer for %s unavailable (%s); using 4 chars/token estimate",
                        model_name, exc)
            self._tok = None

    def count(self, text: str) -> int:
        return _count_cached(self, text)

    def _count(self, text: str) -> int:
        if self._tok is None:
            return max(1, len(text) // 4)
        return len(self._tok.encode(text, add_special_tokens=False))


@lru_cache(maxsize=200_000)
def _count_cached(counter: TokenCounter, text: str) -> int:
    # Sentences are counted many times while packing chunks; cache the result.
    return counter._count(text)
