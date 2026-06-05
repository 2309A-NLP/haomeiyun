from __future__ import annotations

from ..models.schemas import DocumentChunk
from .embedding import cosine_similarity


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._rows: list[tuple[DocumentChunk, list[float]]] = []

    def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk count and vector count must match")
        self._rows = list(zip(chunks, vectors, strict=False))

    def search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        del query_text
        scored = [
            (chunk, cosine_similarity(query_vector, vector))
            for chunk, vector in self._rows
            if not document_ids or chunk.document_id in document_ids
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def ready(self) -> bool:
        return bool(self._rows)

    def available(self) -> bool:
        return True
