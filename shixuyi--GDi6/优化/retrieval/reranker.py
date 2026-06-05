from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..core.config import settings
from ..core.logging import logger
from ..models.schemas import DocumentChunk


class BGECrossEncoderReranker:
    def __init__(self) -> None:
        self.provider = (settings.rerank_provider or "bge").lower()
        self.model_name = settings.rerank_model_name or "BAAI/bge-reranker-base"
        self.model_path = settings.rerank_model_path.strip()

    def rerank(self, query: str, chunks: list[DocumentChunk], top_k: int) -> list[tuple[DocumentChunk, float]]:
        if not chunks:
            return []

        if self.provider == "keyword":
            return self._keyword_rerank(query, chunks, top_k)

        model_bundle = self._load_cross_encoder()
        if model_bundle is None:
            logger.warning("BGE reranker unavailable, fallback to keyword reranking")
            return self._keyword_rerank(query, chunks, top_k)

        tokenizer, model, torch = model_bundle
        pairs = [(query, chunk.text) for chunk in chunks]
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(**inputs, return_dict=True).logits.view(-1)
        scores = logits.detach().cpu().tolist()

        ranked = sorted(zip(chunks, scores, strict=False), key=lambda item: item[1], reverse=True)
        normalized = self._min_max_normalize([score for _, score in ranked])
        return [(chunk, score) for (chunk, _), score in zip(ranked, normalized, strict=False)][:top_k]

    @lru_cache(maxsize=1)
    def _load_cross_encoder(self) -> tuple[Any, Any, Any] | None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            logger.warning("Missing reranker dependency: %s", exc)
            return None

        model_source = self.model_path or self.model_name
        local_files_only = bool(self.model_path)
        if self.model_path and not Path(self.model_path).exists():
            logger.warning("Configured reranker path does not exist: %s", self.model_path)
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_files_only)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                local_files_only=local_files_only,
            )
            model.eval()
        except Exception as exc:  # pragma: no cover - depends on local model files
            logger.warning("Failed to load reranker model from %s: %s", model_source, exc)
            return None

        logger.info("Loaded reranker model from %s", model_source)
        return tokenizer, model, torch

    def _keyword_rerank(self, query: str, chunks: list[DocumentChunk], top_k: int) -> list[tuple[DocumentChunk, float]]:
        query_text = query or ""
        query_tokens = set(self._tokenize(query_text))
        scored: list[tuple[DocumentChunk, float]] = []
        for chunk in chunks:
            text = chunk.text or ""
            chunk_tokens = set(self._tokenize(text))
            keyword_tokens = set(chunk.keywords or [])
            overlap = len(query_tokens & chunk_tokens)
            keyword_overlap = len(query_tokens & keyword_tokens)
            compact_query = re.sub(r"\s+", "", query_text)
            compact_text = re.sub(r"\s+", "", text)
            phrase_bonus = 1.0 if compact_query and compact_query in compact_text else 0.0
            short_penalty = 0.15 if len(compact_text) <= 8 and phrase_bonus == 0.0 else 0.0
            score = (
                overlap / max(len(query_tokens), 1)
                + keyword_overlap / max(len(query_tokens), 1) * 0.8
                + phrase_bonus
                - short_penalty
            )
            scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []

        for part in re.findall(r"[A-Za-z0-9]{1,20}", text or ""):
            tokens.append(part)

        for part in re.findall(r"[\u4e00-\u9fa5]{2,}", text or ""):
            tokens.append(part)
            for size in range(2, min(5, len(part) + 1)):
                for start in range(0, len(part) - size + 1):
                    tokens.append(part[start : start + size])

        return tokens

    def _min_max_normalize(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        max_score = max(scores)
        min_score = min(scores)
        if max_score == min_score:
            return [1.0 for _ in scores]
        return [(score - min_score) / (max_score - min_score) for score in scores]
