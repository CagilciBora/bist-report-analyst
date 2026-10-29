"""Reciprocal Rank Fusion (RRF).

    RRF(d) = Σ_i 1 / (k + rank_i(d))        rank starts at 1, k = 60

Why ranks and not scores? A BM25 score (e.g. 23.7) and a cosine similarity
(e.g. 0.81) live on different, query-dependent scales; adding them needs
fragile normalisation. RRF only uses positions, so it is scale-free. A
document ranked well by BOTH retrievers beats one ranked first by only one.
k dampens the advantage of the very top ranks (1/61 vs 1/62 is a tiny gap).
"""

from __future__ import annotations


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse several best-first id lists into one [(id, rrf_score)] list, best first."""
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # sort by score desc; ties broken by id for deterministic output
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
