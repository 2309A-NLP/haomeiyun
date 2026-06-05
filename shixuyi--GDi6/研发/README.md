# 招股说明书 RAG 系统

面向招股说明书场景的 LLM 问答项目，支持多份 PDF 同时入库，覆盖：

- Query 理解：意图识别、实体提取、子查询拆分、检索词扩展
- 混合检索：PDF 解析保留页码、向量检索、BM25、重排
- 答案生成：页码引用、拒答、`PDF回答 vs 纯LLM回答` 对比
- 多文档管理：按公司/标签聚合索引到 Milvus
- 长期对话：基于 Redis 的会话上下文补全
- 基准评测：内置 10 问 benchmark

## 项目结构

```text
app/
  api/                # FastAPI 路由
  core/               # 配置与日志
  ingest/             # PDF 解析与切块
  llm/                # OpenAI/Qwen 兼容调用
  models/             # Pydantic 数据模型
  query/              # Query 理解
  retrieval/          # BM25、向量检索、混合检索、重排
  services/           # RAG 主流程、文档服务、评测服务
data/
  benchmarks/         # 10 问基准
  seed/               # 无 PDF 时的演示知识
scripts/              # ingestion / benchmark CLI
```

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 复制配置

```bash
copy .env.example .env
```

3. 启动 Redis（用于长期对话）

```bash
redis-server
```

4. 准备 PDF

可以通过前端连续上传多份 PDF，系统会自动抽取公司名并作为标签入库。

如果仍使用本地单文件方式，可把原始 PDF 放到：

`data/raw/武汉兴图新科电子股份有限公司招股说明书.pdf`

5. 解析并建库

```bash
python scripts/ingest_pdf.py
```

6. 启动服务

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

7. 跑基准

```bash
python scripts/run_benchmark.py
```

## 关键说明

- 默认 `VECTOR_BACKEND=inmemory`，方便离线演示。
- 使用 `VECTOR_BACKEND=milvus` 时，系统会把多份招股书以 `document_label/company_name` 标签写入同一个 Milvus collection。
- Redis 用于保存 `session_id` 对应的最近一轮问答上下文，支持“那武汉力源信息技术股份有限公司呢？”这类追问补全。
- 默认 `EMBEDDING_PROVIDER=hash`，这是离线兜底模式。
- 文本切分已改为按 PDF 段落分块，不再按字符长度切片。
- 默认重排器为 `BAAI/bge-reranker-base`；如果已下载到本地，可通过 `RERANK_MODEL_PATH` 指向本地目录。
- 首次启用 BGE 重排前需要安装 `torch` 和 `transformers`，并重新执行 `python scripts/ingest_pdf.py` 重建段落索引。
- 未接入真实 LLM 时，系统会走“抽取式回答 + 基准知识兜底”。

## 输出格式

```text
【答案】...
【来源】招股说明书第X页
【相关片段】...
```

## 当前交付边界

- 已实现完整项目骨架和可运行主链路。
- 已内置 10 问 benchmark 数据与演示知识。
- 如果仓库里还没有原始 PDF，页码会先显示为“待PDF校准”。
- 一旦接入原始 PDF 并执行 ingestion，系统会优先返回真实页码证据。
