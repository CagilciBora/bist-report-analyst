"""ChunkStore: every indexed chunk (text + metadata) in one JSONL file.

It is the source of truth for BM25 (which needs all texts) and lets us look
up a chunk by id after any retriever returns ids. ~4k chunks = a few MB.
"""

from __future__ import annotations

import json
from pathlib import Path

from kaprag.chunking.models import Chunk


class ChunkStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.chunks: dict[str, Chunk] = {}
        self.doc_keys: dict[str, str] = {}  # chunk_id -> doc_key
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    d = json.loads(line)
                    self.doc_keys[d["chunk_id"]] = d.pop("doc_key")
                    self.chunks[d["chunk_id"]] = Chunk.from_dict(d)

    def replace_doc(self, doc_key: str, chunks: list[Chunk]) -> None:
        self.remove_doc(doc_key)
        for c in chunks:
            self.chunks[c.chunk_id] = c
            self.doc_keys[c.chunk_id] = doc_key

    def remove_doc(self, doc_key: str) -> int:
        ids = [cid for cid, k in self.doc_keys.items() if k == doc_key]
        for cid in ids:
            del self.chunks[cid]
            del self.doc_keys[cid]
        return len(ids)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for cid, c in self.chunks.items():
                fh.write(json.dumps({**c.to_dict(), "doc_key": self.doc_keys[cid]}, ensure_ascii=False) + "\n")
        tmp.replace(self.path)  # atomic: never leaves a half-written file

    def get(self, chunk_id: str) -> Chunk:
        return self.chunks[chunk_id]

    def __len__(self) -> int:
        return len(self.chunks)
