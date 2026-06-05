from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from ..core.config import settings


class BGEEmbeddingProvider:
    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension or settings.embedding_dimension
        self.model_name = settings.embedding_model_name or "BAAI/bge-large-zh-v1.5"
        self.model_path = settings.embedding_model_path.strip()

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        tokenizer, model, torch, functional, device = self._load_model_bundle()
        vectors: list[list[float]] = []

        for start in range(0, len(texts), 16):
            batch = [text or "" for text in texts[start : start + 16]]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {name: value.to(device) for name, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, return_dict=True)
                embeddings = outputs.last_hidden_state[:, 0]
                embeddings = functional.normalize(embeddings, p=2, dim=1)

            vectors.extend(embeddings.detach().cpu().tolist())

        return vectors

    def encode_query(self, text: str) -> list[float]:
        vectors = self.encode([text])
        return vectors[0] if vectors else [0.0] * self.dimension

    @lru_cache(maxsize=1)
    def _load_model_bundle(self) -> tuple[Any, Any, Any, Any, Any]:
        try:
            import torch
            import torch.nn.functional as functional
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("BGE embedding dependencies are missing") from exc

        model_source = self.model_path or self.model_name
        local_files_only = bool(self.model_path)
        if self.model_path and not Path(self.model_path).exists():
            raise RuntimeError(f"Configured embedding path does not exist: {self.model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_files_only)
        model = AutoModel.from_pretrained(model_source, local_files_only=local_files_only)
        model.eval()

        hidden_size = int(getattr(model.config, "hidden_size", self.dimension))
        if hidden_size != self.dimension:
            raise RuntimeError(
                f"Embedding dimension mismatch: configured={self.dimension}, model={hidden_size}, source={model_source}"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model, torch, functional, device


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))
