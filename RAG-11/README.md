# RAG-11: Embedding 模型微调项目

## 项目概述

这是一个面向**招股说明书专业领域**的 Embedding 模型微调项目。

### 核心问题

通用中文 Embedding 模型（如 BAAI/bge-base-zh-v1.5）对**法律/金融专业术语存在语义混淆**。例如：

| 术语 | 法律/金融含义 | 通用模型可能理解为 |
|------|---------------|-------------------|
| consideration | 对价（合同交易金额） | 考虑 |
| complaint | 起诉状 | 抱怨/投诉 |
| discovery | 证据开示 | 发现/探索 |
| underwriter | 承销商 | 保险承保人 |
| offering | 证券发行 | 提供/赠予 |

微调后，模型能区分"complaint"在诉讼场景 vs 日常场景的语义差异，从而让 RAG 检索更精准。

### 微调效果（已实测）

| 指标 | 基础模型 | 微调后 | 提升 |
|------|----------|--------|------|
| accuracy | 0.5200 | **1.0000** | +0.48 |
| recall@1 | 0.1400 | **0.2600** | +0.12 |
| recall@5 | 0.5200 | **0.9400** | +0.42 |
| recall@10 | 0.9200 | **1.0000** | +0.08 |

微调后 avg_neg_sim（不相关文本相似度）从 0.64 降至 -0.36，模型学会了有效推开不相关内容。

---

## 项目结构

```
RAG-11/
├── run.py                          # 一键入口
├── .env                            # 环境变量（DeepSeek API Key 等）
├── requirements.txt                # Python 依赖
├── finetune/                       # 微调模块
│   ├── configs/
│   │   └── config.yaml             # 训练配置
│   ├── scripts/
│   │   ├── generate_dataset.py     # 数据集生成（合成模式，无需 PDF）
│   │   ├── train.py                # 微调训练（4 种损失函数）
│   │   └── evaluate.py             # 微调前后对比评估
│   ├── data/
│   │   └── train_qa.jsonl          # 300 条训练数据（三元组格式）
│   ├── outputs/                    # 微调后模型保存位置
│   └── checkpoints/                # 训练中间检查点
└── rag/                            # RAG 问答服务
    ├── app/
    │   ├── main.py                 # FastAPI 入口（端口 8001）
    │   ├── api/chat.py             # /ask GET/POST + /search 接口
    │   ├── core/config.py          # 配置管理（.env 读取）
    │   ├── rag/
    │   │   ├── embedding.py        # 嵌入模型（base/finetuned 切换）
    │   │   └── pipeline.py         # PDF 加载 → 分块 → 检索管线
    │   ├── services/rag_service.py # RAG 问答（检索 + DeepSeek LLM）
    │   └── vector_store/
    │       └── memory_store.py     # 内存向量存储（免 Milvus）
    └── query/
        ├── test_qa.py             # 测试脚本（5 个招股书样本问题）
        └── sample_questions.txt    # 示例查询列表
```

---

## 使用流程

### 前置条件

1. **本地模型文件**存在 `D:\八维\zg3\bge-base-zh-v1.5\`（~1.3GB）
   - 微调和 RAG 服务均从本地加载，不联网
   - 如需其他模型，修改 `finetune/configs/config.yaml` 的 `local_path` 和 `.env` 的 `EMBEDDING_MODEL_PATH`

2. **DeepSeek API Key** 配置在 `.env` 中
   - RAG 问答服务依赖 DeepSeek 生成回答
   - 纯评估（Recall@k 对比）不需要 API Key

### 快速开始

```powershell
cd "D:\八维\BW - RAG工单\RAG-11"
.venv\Scripts\python run.py generate    # 1. 生成 300 条训练数据
.venv\Scripts\python run.py train       # 2. 微调嵌入模型
.venv\Scripts\python run.py eval        # 3. 微调前后对比评估
.venv\Scripts\python run.py rag         # 4. 启动 RAG 问答服务（终端 1）
```

```powershell
# 终端 2 — 测试问答
.venv\Scripts\python run.py test        # 发 5 个样本问题测试
```

或全流程一步完成：
```powershell
.venv\Scripts\python run.py all         # generate → train → eval
```

### 可用命令

| 命令 | 说明 |
|------|------|
| `generate` / `gen` | 合成生成 300 条三元组训练数据 |
| `train` | 加载本地模型 → 微调 → 保存到 outputs/ |
| `eval` / `evaluate` | base 模型 vs fine-tuned 模型 Recall@k 对比 |
| `all` | generate → train → eval 全流程 |
| `rag` | 启动 FastAPI 问答服务（localhost:8001） |

`train` 支持覆盖参数：
```powershell
.venv\Scripts\python run.py train --loss=contrastive --epochs=3
```

`eval` 会自动读 `finetune/configs/config.yaml` 中的 `local_path`，不需要传参。

---

## 训练数据

已生成 **300 条**三元组（`finetune/data/train_qa.jsonl`），格式：

```json
{"anchor": "XXX在法律语境中的含义是什么？", "positive": "XXX的专业法律定义...", "negative": "XXX的日常错误理解..."}
```

### 数据类别覆盖

| 类别 | 术语数量 | 示例 |
|------|---------|------|
| 诉讼程序术语 | 18 个 | complaint, discovery, motion, injunction, verdict |
| 招股书/证券术语 | 15 个 | underwriter, prospectus, shelf registration, green shoe |
| 公司法/并购术语 | 16 个 | merger, earn-out, escrow, indemnification |
| 会计/报表术语 | 8 个 | amortization, depreciation, contingent liability |
| 金融术语 | 6 个 | yield, spread, leverage, derivative |
| 金融概念补充 | 30 条 | IPO, VIE架构, poison pill, EBITDA, tag-along |
| 法律对比对 | 15 条 | material vs 重大, best efforts vs 最大努力 |

核心设计：negative 不是随机无关文本，而是**有语义关联但含义错误的日常解释**，迫使模型区分法律/金融专业含义。

---

## 损失函数

支持 4 种，通过 `config.yaml` 的 `loss.loss_type` 或命令行 `--loss=` 切换：

| 类型 | key | 适用场景 |
|------|-----|----------|
| **TripletLoss** | `triplet` | 三元组数据的标准选择（推荐） |
| ContrastiveLoss | `contrastive` | 正负例对数据 |
| CosineSimilarityLoss | `cosine_similarity` | 相似度打分数据 |
| MatryoshkaLoss | `matryoshka` | 多维度输出（256/384/512/768） |

---

## 配置文件说明

### `finetune/configs/config.yaml`

```yaml
model:
  base_model_name: "BAAI/bge-base-zh-v1.5"
  local_path: "D:/八维/zg3/bge-base-zh-v1.5"    # 本地路径，不走联网
  device: "cpu"

training:
  batch_size: 16
  num_epochs: 1
  learning_rate: 2e-5
  output_dir: "./outputs"

data:
  train_data_path: "finetune/data/train_qa.jsonl"
  val_ratio: 0.1

loss:
  loss_type: "triplet"
  margin: 0.5
```

### `.env`

```ini
DEEPSEEK_API_KEY=你的key
DEEPSEEK_BASE_URL=https://api.deepseek.com
EMBEDDING_MODEL_NAME=BAAI/bge-base-zh-v1.5
EMBEDDING_DIMENSION=768
EMBEDDING_DEVICE=cpu
EMBEDDING_MODEL_PATH=D:/八维/zg3/bge-base-zh-v1.5
PORT=8001
```

---

## RAG 问答 API

启动 `run.py rag` 后，FastAPI 服务运行在 `localhost:8001`。

### 接口

**问答** — `GET /ask`

```powershell
curl "http://localhost:8001/ask?question=公司主营业务是什么&use_finetuned=false"
```

参数：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `question` | str | 必填 | 问题 |
| `top_k` | int | 5 | 检索文档数 |
| `use_finetuned` | bool | false | `true` = 用微调后的模型检索 |

也支持 `POST /ask`（JSON body: `{"question": "...", "top_k": 5, "use_finetuned": false}`）

**仅检索** — `GET /search`

```powershell
curl "http://localhost:8001/search?question=主营业务&top_k=5"
```

返回相关文档片段及其相似度分数。

**健康检查** — `GET /health`

---

## 注意事项

### 1. 终端中文路径问题
WSL 终端进入中文目录（`八维`、`BW - RAG工单`）后，shell session 的 cwd 会受损，后续命令报 `cd: y: No such file or directory`。强制用 `workdir=/tmp` 恢复，或通过符号链接 `/tmp/GAG11` 操作。

如果直接在 Windows PowerShell 上运行则无此问题。

### 2. 本地模型不走联网
所有 `SentenceTransformer` 加载点都加了 `local_files_only=True`：
- `train.py` 第 159 行
- `evaluate.py` 第 172 行
- `embedding.py` 第 25 行

如果 `local_path` 路径不对或文件缺失，会直接报错而非联网下载。请确保 `D:\八维\zg3\bge-base-zh-v1.5\` 完整。

### 3. Windows 假激活问题
PowerShell 中 `(.venv)` 提示符不保证 `python` 指向 venv 内的解释器。系统级 PATH 中的 Python 3.14 可能优先于 venv。最佳做法是用绝对路径调用：

```powershell
.venv\Scripts\python run.py <命令>
```

而不是：
```powershell
python run.py <命令>
```

### 4. 微调后模型路径
`train.py` 将模型保存到 `finetune/outputs/`，`evaluate.py` 默认从 `finetune/outputs/` 加载微调模型做对比。如果改过 `config.yaml` 的 `training.output_dir`，evaluate.py 不会自动感知，需要手动传 `--finetuned_model`。

### 5. RAG 服务需要 DeepSeek API Key
纯评估（`run.py eval`）不需要 API Key。RAG 问答服务（`run.py rag`）需要 `.env` 中有 `DEEPSEEK_API_KEY`，否则启动时会报 `Missing credentials` 错误。

### 6. 包依赖
所有包装在项目自带的 `.venv\` 中（Python 3.12），依赖见 `requirements.txt`。核心包是 `sentence-transformers`（自动装 `transformers` + `torch`）+ `openai` + `fastapi` + `uvicorn` + `pymupdf`。

---

## 与其他工单的关系

| 工单 | 内容 | 关系 |
|------|------|------|
| 工单-04 | 武汉兴图新科招股书 RAG | PDF 解析、分块、检索管线原型 |
| 工单-06 | 通用招股书 RAG 系统 | FastAPI 服务、Milvus 向量库 |
| **RAG-11** | Embedding 模型微调 | **微调 BGE 模型提升检索精度** |

RAG-11 的微调模型可直接接入工单-06 的 RAG 系统，替换其 `embedding_model_name` 为 `finetune/outputs/` 路径即可。

---

## 扩展：如何在已有 RAG 系统接入微调模型

如果已有工单-04/06 的 RAG 系统想使用这个微调模型：

1. 把 `finetune/outputs/` 整个目录复制到工单-04/06 项目下
2. 在该项目 `.env` 中设置 `EMBEDDING_MODEL_PATH=outputs的绝对路径`
3. 或直接修改其 embedding 模块，传入微调模型路径

RAG-11 自带的 RAG 服务已内置此切换逻辑（`?use_finetuned=true` 参数）。

---

## 许可

本项目数据为招股说明书领域专用，代码部分可自由复用。