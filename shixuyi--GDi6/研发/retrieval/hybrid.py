from __future__ import annotations

from difflib import SequenceMatcher
import re

from ..core.config import settings
from ..core.logging import logger
from ..models.schemas import DocumentChunk, RetrievalHit
from .bm25 import SimpleBM25Index
from .embedding import BGEEmbeddingProvider
from .inmemory_vector_store import InMemoryVectorStore
from .vector_store import MilvusVectorStore


class HybridRetriever:
    def __init__(self) -> None:
        self.embedder = BGEEmbeddingProvider()
        self.vector_backend = (settings.vector_backend or "inmemory").lower()
        self.vector_store = self._build_vector_store()
        self.bm25_index: SimpleBM25Index | None = None

    def prepare(self, chunks: list[DocumentChunk]) -> None:
        self.bm25_index = SimpleBM25Index(chunks) if chunks else None

    def build(self, chunks: list[DocumentChunk]) -> None:
        self.prepare(chunks)
        vectors = self.embedder.encode([chunk.text for chunk in chunks])
        try:
            self.vector_store.upsert(chunks, vectors)
        except Exception as exc:
            if isinstance(self.vector_store, MilvusVectorStore):
                logger.warning("Milvus upsert failed, fallback to in-memory store: %s", exc)
                self.vector_store = InMemoryVectorStore()
                self.vector_store.upsert(chunks, vectors)
                return
            raise

    def ready(self) -> bool:
        return self.bm25_index is not None or self.vector_store.ready()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievalHit]:
        if not self.ready():
            return []

        requested_top_k = max(1, top_k or settings.vector_top_k)
        candidate_k = max(requested_top_k * 8, 32)
        merged: dict[str, RetrievalHit] = {}
        dense_ranks: dict[str, int] = {}
        bm25_ranks: dict[str, int] = {}

        if self.vector_store.ready():
            dense_hits = self.vector_store.search(
                query_text=query,
                query_vector=self.embedder.encode_query(query),
                top_k=candidate_k,
                document_ids=document_ids,
            )
            for rank, (chunk, score) in enumerate(dense_hits, start=1):
                dense_ranks[chunk.chunk_id] = rank
                merged[chunk.chunk_id] = RetrievalHit(
                    chunk=chunk,
                    dense_score=score,
                    bm25_score=0.0,
                    rrf_score=0.0,
                    rerank_score=0.0,
                    final_score=0.0,
                )

        if self.bm25_index is not None:
            for rank, (chunk, score) in enumerate(self.bm25_index.search(query, candidate_k), start=1):
                if document_ids and chunk.document_id not in document_ids:
                    continue
                bm25_ranks[chunk.chunk_id] = rank
                hit = merged.get(chunk.chunk_id)
                if hit is None:
                    merged[chunk.chunk_id] = RetrievalHit(
                        chunk=chunk,
                        dense_score=0.0,
                        bm25_score=score,
                        rrf_score=0.0,
                        rerank_score=0.0,
                        final_score=0.0,
                    )
                else:
                    hit.bm25_score = score

        results = list(merged.values())
        if not results:
            return []

        self._apply_rrf_fusion(results, dense_ranks, bm25_ranks)
        deduped = self._deduplicate_results(sorted(results, key=lambda item: item.final_score, reverse=True))
        return deduped[:requested_top_k]

    def _apply_rrf_fusion(
        self,
        results: list[RetrievalHit],
        dense_ranks: dict[str, int],
        bm25_ranks: dict[str, int],
    ) -> None:
        rrf_k = max(1, settings.retrieval_rrf_k)
        for item in results:
            score = 0.0
            dense_rank = dense_ranks.get(item.chunk.chunk_id)
            bm25_rank = bm25_ranks.get(item.chunk.chunk_id)
            if dense_rank is not None:
                score += 1.0 / (rrf_k + dense_rank)
            if bm25_rank is not None:
                score += 1.0 / (rrf_k + bm25_rank)
            item.rrf_score = score
            item.final_score = score

    def _deduplicate_results(self, results: list[RetrievalHit]) -> list[RetrievalHit]:
        deduped: list[RetrievalHit] = []
        seen_texts: list[str] = []
        for item in results:
            normalized = self._normalize_text(item.chunk.text)
            if not normalized:
                continue
            if any(self._is_near_duplicate(normalized, existing) for existing in seen_texts):
                continue
            deduped.append(item)
            seen_texts.append(normalized)
        return deduped

    def _normalize_text(self, text: str) -> str:
        compact = re.sub(r"\s+", "", text or "")
        compact = re.sub(r"1-1-\d+", "", compact)
        return compact

    def _is_near_duplicate(self, left: str, right: str) -> bool:
        if left == right:
            return True
        shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
        if shorter and shorter in longer and len(shorter) / max(len(longer), 1) >= 0.7:
            return True
        return SequenceMatcher(None, left, right).ratio() >= 0.88

    def _build_vector_store(self):
        if self.vector_backend != "milvus":
            logger.info("Using in-memory vector store backend")
            return InMemoryVectorStore()

        store = MilvusVectorStore()
        if store.available():
            logger.info("Using Milvus vector store backend")
            return store

        logger.warning("Milvus unavailable, fallback to in-memory vector store")
        return InMemoryVectorStore()
