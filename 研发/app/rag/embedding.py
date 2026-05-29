from __future__ import annotations

from typing import Dict, List, Union

import numpy as np

try:
    from FlagEmbedding import BGEM3FlagModel
    _FLAG_EMBEDDING_IMPORT_ERROR = None
except ImportError as exc:
    BGEM3FlagModel = None
    _FLAG_EMBEDDING_IMPORT_ERROR = exc

from ..core.config import settings
from ..utils.logger import logger


class BGEEmbedder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.model_path = settings.EMBEDDING_MODEL_PATH
        self._load_model()
        self._initialized = True

    def _load_model(self):
        if BGEM3FlagModel is None:
            raise RuntimeError(
                "FlagEmbedding is not installed. Install the model dependencies before using embedding features."
            ) from _FLAG_EMBEDDING_IMPORT_ERROR

        device = "cuda" if self._check_cuda() else "cpu"
        use_fp16 = device == "cuda"
        logger.info("Loading BGE-M3 model from %s", self.model_path)
        self.model = BGEM3FlagModel(self.model_path, use_fp16=use_fp16, device=device)
        logger.info("BGE-M3 model loaded successfully on %s, fp16=%s", device, use_fp16)

    def _check_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 8,
        max_length: int = 8192,
        return_sparse: bool = True,
    ) -> Dict[str, np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]

        return self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )

    # 查询文本向量化的入口方法，它将用户输入的查询文本转换为向量表示
    def encode_query(self, text: str, return_sparse: bool = False) -> Dict[str, np.ndarray]:
        return self.encode(
            text,
            batch_size=1,  # 批量大小：只处理1条文本
            max_length=settings.QUERY_MAX_LENGTH,  # 最大长度限制
            return_sparse=return_sparse,  # 是否返回稀疏向量
        )

    def encode_dense(self, text: str) -> List[float]:
        result = self.encode(text, return_sparse=False)
        return result["dense_vecs"][0].tolist()
