# 测试计划

## 基准测试集
- 来源：兴图新科人工标注 QA
- 数量：10 问

## 测试指标
- 精确率 (Precision)
- 召回率 (Recall)
- F1 分数
- MRR (Mean Reciprocal Rank)

## 测试流程
1. 导入 PDF 文档
2. 执行文档解析与分块
3. 向量化与索引构建
4. 执行查询评估
5. 对比分析结果

## 对比维度
- 不同 retrieval 策略 (BM25 vs Vector vs Hybrid)
- 不同 chunk 参数
- 不同重排序策略
