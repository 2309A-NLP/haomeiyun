# RAG-11 过程问题记录

> 项目：Embedding 模型微调 —— 招股说明书领域语义消歧
> 时间：2026-06-05
> 环境：WSL2 Ubuntu (Windows 端), Windows PowerShell

---

## 问题索引

| # | 问题 | 影响范围 | 严重程度 |
|---|---|---|---|
| 1 | 终端中文路径污染 | 全部 WSL 操作 | 高 |
| 2 | Windows 假激活问题 | 全部 Windows 命令执行 | 高 |
| 3 | BW 双目录冲突 | 全部文件写入 | 高 |
| 4 | 本地模型联网超时 | 模型加载、微调、评估 | 高 |
| 5 | 配置文件字段名不匹配 | config.yaml ↔ train.py | 高 |
| 6 | generate_dataset.py 缺少 --synthetic 模式 | 数据生成 | 高 |
| 7 | run.py 不传递 --synthetic 参数 | 数据生成 | 高 |
| 8 | 相对路径错位 | evaluate.py 数据路径 | 中 |
| 9 | 相对路径错位 | evaluate.py 微调模型路径 | 中 |
| 10 | WSL 下无法运行 Windows .exe | RAG 服务启动 | 中 |
| 11 | 包依赖缺失 | openai, pymupdf, faiss | 中 |
| 12 | 缩进错误 | evaluate.py 代码 | 低 |
| 13 | run.py 硬编码 num_samples=150 | 数据量 | 低 |
| 14 | run.py run_rag 不接收参数 | RAG 服务启动 | 低 |

---

## 详细记录

### 问题 1：终端中文路径污染

**现象：** 在 WSL 中 `cd /mnt/d/八维/BW - RAG工单/xxx` 之后，下一个 `terminal()` 调用直接报错乱码，后续所有命令无法正常执行，必须重新开 shell。

**根因：** Hermes Agent 的终端底层在 WSL 下，`cd` 进包含中文字符（"八维"、"RAG工单"）的目录后，终端的 cwd 字符串被污染。后续任何命令的 `workdir` 继承这个乱码 cwd，导致 shell 挂死。

**修复：** 两种方式：
- 所有终端命令指定 `workdir=/tmp`，不继承污染了的 cwd
- 创建无中文符号链接：`ln -s "/mnt/d/八维" /tmp/八维`，之后用 `/tmp/八维/` 路径操作
- 用户自己用 Windows PowerShell（不经过 WSL）

**教训：** WSL 下中文路径是雷区，所有涉及中文路径的操作尽量走符号链接或 Windows 侧。

---

### 问题 2：Windows 假激活

**现象：** PowerShell 提示符显示 `(.venv)`，但 `python` 实际指向的是 `C:\Python314\python.exe`，不是 `.venv\Scripts\python.exe`。导致 `pip install openai` 装到了全局，`.venv` 里没有，运行时报 `ModuleNotFoundError: No module named 'openai'`。

**根因：** `activate.bat` 只修改 PATH，但 `C:\Python314\` 可能写进了系统级 PATH（优先级更高），或者之后有其他脚本覆盖了 PATH。`(.venv)` 提示符只是 activate 脚本设置的假标志，不代表 python 真的指向 venv。

**修复：**
1. 显式写绝对路径：`.venv\Scripts\python` 和 `.venv\Scripts\pip`
2. 修改 `run.py`：自动检测 `.venv\Scripts\python.exe`，所有子进程都用它

**教训：** 永远不要相信 `(.venv)` 提示符。Windows 的 PATH 优先级复杂，venv 激活可能失败。始终用 `.venv\Scripts\python` 的绝对路径。

---

### 问题 3：BW 双目录冲突

**现象：** Hermes Agent 写了大量文件到 `/mnt/d/八维/BW-RAG工单/RAG-11/`（无空格），但用户实际工作在 `D:\八维\BW - RAG工单\RAG-11\`（有空格）。两个目录都存在，但内容不一致，花了大量时间排查为什么修改不生效。

**根因：** Herem 的终端在指定 `workdir` 时路径包含中文，导致路径解析时丢掉了空格。用户的实际目录名是 `"BW - RAG工单"`，但写入时用了 `"BW-RAG工单"`。

**修复：** 检查两个目录，确认哪个是真实的，将所有文件重写到正确目录。

**教训：** 目录名称差异（尤其空格）必须每次确认。问用户"你 cd 到哪个目录了"来锚定正确路径。

---

### 问题 4：本地模型联网超时

**现象：** `SentenceTransformer(model_name)` 即使传了本地文件夹路径，也会联网请求 HuggingFace 检查 `modules.json` 更新。在中国大陆网络环境下连接 HuggingFace 超时（~30s），导致模型加载失败。

**根因：** SentenceTransformer 的默认行为——即使传入本地路径，也会尝试从 HuggingFace Hub 获取最新模块列表。`os.environ.get("HF_ENDPOINT")` 设置了镜像但代码没检查环境变量，而且 `local_files_only=False` 是默认值。

**修复：**
- 所有 `SentenceTransformer()` 调用加 `local_files_only=True`
- `train.py`: 加载模型时用 `local_files_only=bool(local_path)`
- `evaluate.py`: 同样加 `local_files_only=True`
- `embedding.py`（RAG 服务侧）: 优先使用 `local_path`，设 `local_files_only=True`

**涉及文件：**
- `finetune/scripts/train.py`
- `finetune/scripts/evaluate.py`
- `finetune/_run_train.py`
- `rag/app/rag/embedding.py`
- `rag/app/core/config.py`（新增 `embedding_model_path` 字段）

**教训：** 只要指定了 local_path，就必须同时设 `local_files_only=True`，否则 SentenceTransformer 还是会联网。

---

### 问题 5：配置文件字段名不匹配

**现象：** `run.py generate` 和 `run.py train` 都执行成功，但 `run.py eval` 报了各种路径/配置找不到的错误。

**根因：** `config.yaml` 和代码里用了不同的字段名：

| config.yaml 用的字段 | 代码里读的字段 |
|---|---|
| `train_file` | `train_data_path` |
| `eval_ratio` | `val_ratio` |
| `loss.type` | `loss.loss_type` |
| `model.local_path`（在 `model` 段下） | 代码期望的 `local_path`（在根级） |

**修复：** 修改 config.yaml 使字段名与代码匹配：
- `train_file` → `train_data_path`
- `eval_ratio` → `val_ratio`
- `loss.type` → `loss.loss_type`

**教训：** 配置文件的字段名必须和代码严格一致，否则 silent failure。最好在代码里加一段校验逻辑，启动时检查必要字段是否存在。

---

### 问题 6：generate_dataset.py 缺少 --synthetic 模式

**现象：** `python run.py generate --synthetic` 只生成了 0 页 0 条数据（实际输出 0 queries generated）。

**根因：** `generate_dataset.py` 没有 `--synthetic` 参数处理逻辑。它始终从 PDF 目录读取文本，但 RAG-11 下没有 PDF 文件（PDF 在工单 04/06 的项目中），所以提取到 0 页，输出 0 条。

**修复：** 重写 `generate_dataset.py`，加入 `--synthetic` 模式。内置 60+ 法律/金融术语模板，每术语配 3-5 种问法，共生成 300 条三元组。

**涉及文件：** `finetune/scripts/generate_dataset.py`

**教训：** 命令参数和实际脚本实现的逻辑必须对应。`run.py` 传了 `--synthetic` 但脚本不认识，等同于无操作。

---

### 问题 7：run.py 不传递 --synthetic 参数

**现象：** 即使 `run.py generate` 检测到了 `--synthetic` 参数，子进程启动 `generate_dataset.py` 时没有把这个参数传过去。

**根因：** `run.py` 的 `cmd_generate()` 里写了：
```python
# 检测是否传了 args.synthetic 但没传参
subprocess.run([_VENV_PYTHON, gen_script])
```
检测了但不传，导致生成脚本运行时始终认为没给 `--synthetic`。

**修复：** 改为：
```python
if args.synthetic:
    cmd += ["--synthetic", "--num_samples", str(args.num_samples)]
```

**教训：** 检测到参数但不传递，等于白检测。父子进程的参数传递必须显式做。

---

### 问题 8：evaluate.py 数据路径错位

**现象：** `evaluate.py` 运行时读不到数据文件。

**根因：** evaluate.py 第 32 行写死了 `"./data/train_qa.jsonl"`，但实际数据存在 `finetune/data/train_qa.jsonl`。以 RAG-11 根目录为 cwd 运行的话，`./data/` 不存在。

**修复：** 改为基于 `__file__` 计算路径：
```python
SCRIPT_DIR = Path(__file__).parent.parent
ROOT = SCRIPT_DIR.parent.parent  # RAG-11/
DATA_PATH = ROOT / "finetune" / "data" / "train_qa.jsonl"
```

**教训：** 永远不要写硬编码的相对路径。始终基于 `__file__` 计算绝对路径。

---

### 问题 9：evaluate.py 微调模型路径错位

**现象：** evaluate.py 报告"微调模型路径不存在"，跳过了微调模型的评估。

**根因：** 微调模型保存在 `finetune/outputs/`，但 evaluate.py 默认路径是 `"./outputs"`（相对于 cwd）。

**修复：** 默认路径改为基于 `__file__`：
```python
DEFAULT_FT_MODEL_PATH = str(SCRIPT_DIR.parent / "outputs")
```

---

### 问题 10：WSL 下无法运行 Windows .exe

**现象：** 在 WSL 终端里执行 `.venv/Scripts/python` 报 `Exec format error`。

**根因：** `.venv/Scripts/python.exe` 是 Windows PE 格式，WSL（Linux）无法直接执行。WSL 需要 `.venv_linux/bin/python`（Linux ELF 格式）。

**修复：** RAG 服务只能在 Windows PowerShell 端启动，WSL 端无能为力。在项目文档中明确指出这一点。

**教训：** 跨平台时注意可执行文件格式。`.venv/` 和 `.venv_linux/` 是不同的环境。

---

### 问题 11：包依赖缺失

**现象：** 运行时依次报错：`ModuleNotFoundError: No module named 'openai'`、`No module named 'fitz'`、`No module named 'faiss'`。

**根因：** `requirements.txt` 最初没有包含全部依赖。openai 用于 DeepSeek API 调用，pymupdf 用于 PDF 解析，faiss 用于向量检索。

**修复：** 将缺失依赖添加到 `requirements.txt`：
```
openai
pymupdf
sentence-transformers
numpy
faiss-cpu
```

**教训：** `requirements.txt` 必须与实际 import 同步。每次新增 import 都要检查 dependencies。

---

### 问题 12：缩进错误

**现象：** evaluate.py 运行时报 `IndentationError: unindent does not match any outer indentation level`。

**根因：** 在修复 evaluate.py 的问题 9 时，用 `patch` 工具插入了一段代码，但混用了空格和制表符。patch 工具虽然智能，但在大段代码替换时可能引入缩进不一致。

**修复：** 用 `execute_code` 的 Python 读写操作重写整个函数段，Python 的 AST 解析保证了缩进一致。

**教训：** 大段代码修改时，优先用 `write_file` 整体重写（或 `execute_code` 读写），不要在 `patch` 里插入多行代码块。

---

### 问题 13：run.py 硬编码 num_samples=150

**现象：** 尽管数据生成器有 60+ 术语模板，但最终只生成了 81 条数据。

**根因：** `run.py` 在 `cmd_generate()` 中写了 `num_samples=150`。但这只是上限，实际生成量受限于内置模板数量（原版 25 个术语）。

**修复：** 将术语池从 25 个扩展到 60+ 个，每种术语 3-5 种问法，总池超 300 条。同时 `run.py` 的默认值改为 300，并支持 `--num_samples` 参数。

---

### 问题 14：run.py run_rag 不接收参数

**现象：** `python run.py rag` 报 `TypeError: run_rag() takes 0 positional arguments but 1 was given`。

**根因：** `run_rag()` 函数签名是 `def run_rag()`，但 `argparse` 调用了 `cmd_rag(args)` 传了一个参数进去。

**修复：** 改为 `def run_rag(args=None)`，加上默认可选参数。

---

## 经验总结

### 跨平台开发的关键注意点

1. **路径分隔符：** WSL 用 `/mnt/d/xxx`，Windows 用 `D:\xxx`。config.yaml 和 .env 里的路径必须用当前运行平台的格式。
2. **包管理：** 永远用绝对路径调用可执行文件（`.venv\Scripts\python`），不要靠 PATH。
3. **中文路径：** WSL 下中文路径会污染终端会话，尽量用符号链接或 Windows 侧操作。

### 代码架构教训

1. **相对路径是毒药：** 所有文件路径基于 `__file__` 计算绝对路径。
2. **配置文件字段必须校验：** 新增启动时检查 config 必要字段存在的逻辑。
3. **新增 import 必须同步到 requirements.txt：** 否则下次重装环境就会 silent fail。
4. **大段代码修改用 write_file：** patch 适合单行/小块修改，多行代码容易引入缩进问题。

### 项目流程教训

1. **先确认目录：** 每次修改前确认用户实际工作在哪个目录，尤其有空格/特殊字符的目录名。
2. **参数传递必须显式：** 父进程检测到参数，必须在子进程命令中显式传递。
3. **本地模型必须加 `local_files_only=True`：** 否则 SentenceTransformer 总尝试联网。