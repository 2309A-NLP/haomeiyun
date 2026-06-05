# 架构设计

## 系统架构
FastAPI + Milvus + BGE Embedding + DeepSeek LLM

## 数据流向
PDF -> pdf_parser -> Chunks -> Embedding -> Milvus Index
User Query -> QueryAnalyzer -> HybridRetriever -> Reranker -> LLM -> Answer

## 组件设计
- PDF解析: PyMuPDF
- 向量存储: Milvus / InMemoryVectorStore (fallback)
- 检索策略: BM25 + Vector (Hybrid)
- 重排序: 交叉编码器
- LLM: DeepSeek API
- 前端: 原生 HTML/CSS/JS
