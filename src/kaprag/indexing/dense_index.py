"""DenseIndex: chunk vectors in Qdrant (embedded/local mode, no Docker).

Qdrant stores each chunk as a "point" = id + vector + payload (our metadata).
`QdrantClient(path=...)` runs the database inside this Python process and
saves it to a folder. The SAME code works against a Qdrant server by passing
a URL instead, which is how you would scale it later.
Note: embedded mode allows only one process to open the folder at a time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from kaprag.chunking.models import Chunk
from kaprag.indexing.filters import SearchFilter

log = logging.getLogger(__name__)
COLLECTION = "chunks"


class DenseIndex:
    def __init__(self, path: Path, dim: int | None = None) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))
        self.dim = dim

    def ensure_collection(self, dim: int) -> None:
        if self.client.collection_exists(COLLECTION):
            return
        self.client.create_collection(
            COLLECTION, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE)
        )
        # Payload indexes make filtered search fast on a Qdrant server. (Embedded
        # mode accepts them and filters correctly either way.)
        for field, schema in (("ticker", qm.PayloadSchemaType.KEYWORD),
                              ("year", qm.PayloadSchemaType.INTEGER),
                              ("period", qm.PayloadSchemaType.KEYWORD),
                              ("doc_type", qm.PayloadSchemaType.KEYWORD),
                              ("chunk_type", qm.PayloadSchemaType.KEYWORD),
                              ("doc_key", qm.PayloadSchemaType.KEYWORD)):
            self.client.create_payload_index(COLLECTION, field, field_schema=schema)

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, doc_key: str) -> None:
        self.ensure_collection(vectors.shape[1])
        points = [
            qm.PointStruct(id=c.point_id, vector=v.tolist(), payload={**c.to_dict(), "doc_key": doc_key})
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), 256):  # batches keep memory flat
            self.client.upsert(COLLECTION, points=points[i:i + 256])

    def delete_doc(self, doc_key: str) -> None:
        if not self.client.collection_exists(COLLECTION):
            return
        self.client.delete(COLLECTION, points_selector=qm.FilterSelector(filter=qm.Filter(
            must=[qm.FieldCondition(key="doc_key", match=qm.MatchValue(value=doc_key))])))

    def search(self, vector: np.ndarray, flt: SearchFilter, top_k: int) -> list[tuple[str, float]]:
        """Return [(chunk_id, cosine_score)] best first, filter applied inside the search."""
        if not self.client.collection_exists(COLLECTION):
            return []
        res = self.client.query_points(COLLECTION, query=vector.tolist(), query_filter=flt.to_qdrant(),
                                       limit=top_k, with_payload=["chunk_id"])
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points]

    def count(self) -> int:
        if not self.client.collection_exists(COLLECTION):
            return 0
        return self.client.count(COLLECTION).count

    def close(self) -> None:
        self.client.close()
