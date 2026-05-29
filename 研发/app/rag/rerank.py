from __future__ import annotations

from typing import Dict, List

try:
    from FlagEmbedding import FlagReranker
    _FLAG_RERANK_IMPORT_ERROR = None
except ImportError as exc:
    FlagReranker = None
    _FLAG_RERANK_IMPORT_ERROR = exc

from ..core.config import settings
from ..utils.logger import logger


class BGEReranker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.model_path = settings.RERANKER_MODEL_PATH
        self._load_model()
        self._initialized = True

    def _load_model(self):
        if FlagReranker is None:
            raise RuntimeError(
                "FlagEmbedding is not installed. Install the model dependencies before using rerank features."
            ) from _FLAG_RERANK_IMPORT_ERROR

        logger.info("Loading BGE-Reranker from %s", self.model_path)
        self.model = FlagReranker(self.model_path, use_fp16=True)
        if self._check_cuda():
            self.model.model = self.model.model.cuda()
        logger.info("BGE-Reranker loaded successfully")

    def _check_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Dict]:
        if not documents:
            return []

        try:
            pairs = [[query, doc] for doc in documents]
            scores = self.model.compute_score(pairs)
            results = [
                {"id": i, "content": doc, "score": float(score), "rank": 0}
                for i, (doc, score) in enumerate(zip(documents, scores))
            ]
            results.sort(key=lambda x: x["score"], reverse=True)
            for i, item in enumerate(results, start=1):
                item["rank"] = i
            return results[:top_k] if top_k else results
        except Exception as exc:
            logger.error("Reranking failed: %s", exc)
            return [{"id": i, "content": d, "score": 0.5, "rank": i + 1} for i, d in enumerate(documents)]
