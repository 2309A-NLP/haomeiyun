from typing import Dict, Iterable, List, Optional

from ..rag.bm25 import BM25IndexManager


class BM25Store:
    """Thin wrapper around the shared BM25 index."""  # 类文档字符串

    def __init__(self):                      # 构造函数
        self._manager = BM25IndexManager()   # 持有管理器实例
        # 为什么用 _manager 而不是 manager？
            # 下划线约定：表示这是内部实现细节
            # 外部代码不应直接访问 _manager

    def upsert_documents(self, documents: Iterable[Dict]) -> int:
        # documents: Iterable[Dict] - 可迭代的文档字典集合
        # Iterable 比 List 更通用（可以是列表、生成器等）,支持流式处理大文档集
        return self._manager.upsert_documents(documents)  # 委托调用,不做任何额外处理

    # 执行BM25关键词检索
    def search(
        self,
        query: str,          # 查询字符串
        top_k: int = 10,     # 返回结果数量（默认10条）
        legal_field: Optional[str] = None,   # 法律领域过滤（可选）
        knowledge_type: Optional[str] = None,  # 知识类型过滤（可选）
    ) -> List[Dict]:   # 返回字典列表
        return self._manager.search(   # 委托给管理器执行
            query=query,
            top_k=top_k,
            legal_field=legal_field,    # 法律领域筛选
            knowledge_type=knowledge_type,   # 按知识类型筛选
        )

    def delete_by_document_title(self, document_title: str) -> int:
        return self._manager.delete_by_document_title(document_title)

    def clear(self) -> None:
        self._manager.clear()

    def stats(self) -> Dict:
        return self._manager.stats()
