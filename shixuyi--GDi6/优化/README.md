# 优化 (Optimization)

本项目中的文件归类及说明：

## retrieval/ (检索优化)
项目路径: `app/retrieval/`
- bm25.py - BM25稀疏检索实现
- embedding.py - BGE embedding向量化
- hybrid.py - 混合检索（BM25 + 向量）
- reranker.py - 重排序模型
- vector_store.py - Milvus向量存储
- inmemory_vector_store.py - 内存向量存储

## query_analysis/ (查询分析)
项目路径: `app/query/`
- analyzer.py - 查询分析器

## evaluation_results/ (评估结果)
项目路径: `query/`
- change_query.csv - 查询变更记录
- comparison_result.csv - 对比结果
- LLM.csv - LLM评估数据
- text-llm.csv - 文本LLM评估结果
- text-llm_precision_change.csv - 精度变化
- text-llm_precision_comparison.csv - 精度对比
