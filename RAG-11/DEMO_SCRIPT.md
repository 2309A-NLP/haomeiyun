# RAG-11 演示录制脚本

> 用途：录制项目功能演示视频
> 预计时长：5-8 分钟
> 录制工具建议：OBS Studio / Windows 自带录屏 (Win+G)

---

## 准备步骤（录制前做，不录进去）

```powershell
# 1. 确认本地模型存在
dir "D:\八维\zg3\bge-base-zh-v1.5" | select Name, Length

# 2. 确认 .env 已有 DeepSeek API Key
type .env

# 3. 确认 venv 完整
.venv\Scripts\python --version
.venv\Scripts\pip list | findstr "sentence-transformers openai pymupdf"
```

---

## 录制内容

### 第 1 段：项目概览（30 秒）

**画面：** VS Code 打开 RAG-11 目录，展示目录树

**旁白脚本：**
> "这是一个 Embedding 模型微调项目。通用模型把法律术语当日常词理解——'complaint'不等于'抱怨'，'consideration'不等于'考虑'。我们微调了 BGE 模型，让它在招股说明书这个专业领域上检索更准。"

**键盘操作：**
1. 打开 RAG-11 目录
2. 展开 `finetune/` 目录树
3. 简单浏览 `finetune/scripts/` 下三个文件

**预期画面：**
```
RAG-11/
├── run.py
├── .env
├── README.md
├── PROCESS_LOG.md
├── DEMO_SCRIPT.md
├── finetune/
│   ├── configs/config.yaml
│   ├── data/train_qa.jsonl
│   ├── outputs/
│   ├── scripts/
│   │   ├── generate_dataset.py
│   │   ├── train.py
│   │   └── evaluate.py
└── rag/
    └── app/
```

---

### 第 2 段：训练数据（1 分钟）

**画面：** 打开 `finetune/data/train_qa.jsonl`，展示前几条数据

**旁白脚本：**
> "我们先看数据。每条是三元组——anchor（问题）、positive（法律专业含义的段落）、negative（日常错误理解的段落）。模型通过对比学习，学会把同一术语的专业含义拉到一起，把日常含义推开。"

**键盘操作：**
1. 打开 `finetune/data/train_qa.jsonl`
2. 滚动展示前 3-5 条
3. 放大展示一条典型的三元组

**重点展示的数据（建议手动标记）：**
```json
{
  "query": "法律术语complaint在诉讼程序中的含义是什么",
  "positive": "在法律诉讼程序中，complaint指原告向法院提交的起诉状...",
  "negative": "在日常用语中，complaint指顾客对服务或产品的抱怨投诉"
}
```

**旁白：**
> "注意 negative 不是随便选的无关文本——它是同一个词在日常语境下的含义。微调的核心就是让模型学会区分这两种语义。"

---

### 第 3 段：生成数据（1 分钟）

**画面：** PowerShell 执行 `run.py generate`

**旁白脚本：**
> "如果数据需要重新生成，执行 `run.py generate`。内置了 60 多个法律和金融术语模板，自动生成训练数据。不需要外部 API——完全本地生成。"

**键盘操作 + 预期输出：**
```
PS D:\八维\BW - RAG工单\RAG-11>
.venv\Scripts\python run.py generate

[信息] 生成合成训练数据...
[信息] 共生成 300 条三元组
[信息] 保存到 finetune/data/train_qa.jsonl
```

---

### 第 4 段：微调训练（1.5 分钟）

**画面：** PowerShell 执行 `run.py train`，展示训练过程输出

**旁白脚本：**
> "微调用的是本地模型——D盘下的 bge-base-zh-v1.5，完全不联网。训练使用 TripletLoss，5 个 epoch。输出会显示每个 epoch 的 loss 下降趋势。"

**键盘操作 + 预期输出：**
```
PS D:\八维\BW - RAG工单\RAG-11>
.venv\Scripts\python run.py train

[信息] 从本地加载模型: D:/八维/zg3/bge-base-zh-v1.5
[信息] 使用损失函数: TripletLoss
[信息] 训练数据: finetune/data/train_qa.jsonl (300 条)
Epoch 1/5: Loss: 0.8234
Epoch 2/5: Loss: 0.4512
Epoch 3/5: Loss: 0.2876
Epoch 4/5: Loss: 0.1943
Epoch 5/5: Loss: 0.1421
[信息] 微调模型保存到: finetune/outputs/
```

**旁白（训练等待期间）：**
> "训练大概需要几分钟，取决于 CPU 还是 GPU。Loss 从 0.82 降到 0.14，说明模型在学习语义区分。"

---

### 第 5 段：评估对比（1.5 分钟）——核心段落

**画面：** PowerShell 执行 `run.py eval`，展示两列对比输出

**旁白脚本：**
> "评估是最关键的部分。微调前后的模型做同样的测试——300 条法律术语问题上，看谁检索更准。左边是基础模型，右边是微调后模型。"

**键盘操作 + 预期输出：**
```
PS D:\八维\BW - RAG工单\RAG-11>
.venv\Scripts\python run.py eval

========== 评估基础模型 ==========
  accuracy:     0.5200
  recall@1:     0.1400
  recall@5:     0.5200
  recall@10:    0.9200
  avg_pos_sim:  0.76
  avg_neg_sim:  0.64

========== 评估微调模型 ==========
  accuracy:     1.0000
  recall@1:     0.2600
  recall@5:     0.9400
  recall@10:    1.0000
  avg_pos_sim:  0.85
  avg_neg_sim:  -0.36
```

**旁白（全屏放大对比）：**
> "准确率从 0.52 翻到 1.00，召回率全面提升。最关键的指标是 avg_neg_sim——基础模型对不相关文本的平均相似度是 0.64（几乎把不相关当相关），微调后降到 -0.36（模型学会了推开不相关内容）。这是 Embedding 微调最核心的改进：能说 '不'。"

---

### 第 6 段：RAG 问答演示（1.5 分钟）

**画面：** 两个终端——终端1启动 RAG 服务，终端2发送测试查询

**旁白脚本：**
> "最后在实际 RAG 系统中验证。我们启动 FastAPI 服务，然后通过测试查询对比基础模型和微调模型的实际问答效果。"

**键盘操作 1（终端1）：**
```
PS D:\八维\BW - RAG工单\RAG-11>
.venv\Scripts\python run.py rag

INFO:     Started server process [xxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

**键盘操作 2（终端2）：**
```
# 基础模型
curl "http://localhost:8001/ask?question=什么是complaint&use_finetuned=false"
curl "http://localhost:8001/ask?question=underwriter在招股书中指什么&use_finetuned=false"

# 微调模型
curl "http://localhost:8001/ask?question=什么是complaint&use_finetuned=true"
curl "http://localhost:8001/ask?question=underwriter在招股书中指什么&use_finetuned=true"
```

**旁白：**
> "注意对比两种模型的回答。基础模型可能给出日常含义，微调后的模型会给出正确的法律/金融专业定义。这说明微调不仅提升了检索指标，也直接改善了问答质量。"

---

### 第 7 段：总结（30 秒）

**画面：** 回到 README.md 的评估对照表

**旁白脚本：**
> "项目完成了 Embedding 模型微调的完整流程。从生成高质量领域数据，到用 TripletLoss 微调模型，再到多维度评估验证。微调后 accuracy 翻倍、recall@5 接近 1.0、不相关文本相似度从 0.64 降到 -0.36。全部指标证明：领域微调能有效消除通用模型的语义鸿沟。"

**键盘操作：**
1. 打开 README.md
2. 滚动到评估对照表

---

## 录屏注意事项

1. **终端字体调大** —— 建议 16-18px，确保对比表格观众能看清
2. **PowerShell 背景色** —— 深色主题视觉效果更好
3. **先清空历史** —— `cls` 清屏后再执行命令
4. **慢速操作** —— 每个命令之后停留 2-3 秒再执行下一个
5. **不要录训练等待** —— 训练过程可以快进或者剪掉等待时间
6. **对比表格全屏展示** —— 评估输出那一段是全视频核心，建议放大或暂停几秒

## 如果不想开两个终端（跳过第 6 段）

- 删掉第 6 段，直接在第 5 段评估结束后用 README 的评估表收尾
- 时长缩短到 4-5 分钟，但仍然有完整的「准备数据→微调→评估对比」主线

## 录制后的剪辑建议

| 时间点 | 内容 | 建议操作 |
|---|---|---|
| 0:00-0:30 | 项目概览 | 保留 |
| 0:30-1:30 | 训练数据展示 | 保留（观众需要理解数据格式） |
| 1:30-2:30 | 生成数据 | 可加速 2x 或剪掉等待 |
| 2:30-4:00 | 微调训练 | 只保留前 5 秒 + 最后输出，中间剪掉 |
| 4:00-5:30 | 评估对比 | **保留全速**，这是核心内容 |
| 5:30-7:00 | RAG 问答 | 可选，取决于是否需要端到端演示 |
| 7:00-7:30 | 总结 | 保留 |