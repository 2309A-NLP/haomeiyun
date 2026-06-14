#!/usr/bin/env python3
"""
RAG-11 嵌入模型微调工具

命令:
  generate   - 生成训练数据（合成生成，无需 PDF）
  train      - 微调嵌入模型
  eval       - 评估微调效果
  all        - 全流程: generate → train → eval
  rag        - 启动 RAG 问答服务
"""

import sys
import os
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
_VENV_PYTHON = sys.executable


def cmd_generate(args=None):
    """生成训练数据"""
    print("=" * 60)
    print("生成合成训练数据（法律领域术语消歧）...")
    print("=" * 60)
    subprocess.run([
        _VENV_PYTHON,
        os.path.join(ROOT, "finetune", "scripts", "generate_dataset.py"),
        "--synthetic",
        "--output", os.path.join(ROOT, "finetune", "data", "train_qa.jsonl"),
        "--num_samples", "300",
    ])


def cmd_train(args=None):
    """微调嵌入模型"""
    print("=" * 60)
    print("微调嵌入模型...")
    print("=" * 60)
    cmd = [
        _VENV_PYTHON,
        os.path.join(ROOT, "finetune", "scripts", "train.py"),
        "--config", os.path.join(ROOT, "finetune", "configs", "config.yaml"),
    ]
    if args:
        for a in args:
            if a.startswith("--loss="):
                cmd.extend(["--loss", a.split("=")[1]])
            elif a.startswith("--epochs="):
                cmd.extend(["--epochs", a.split("=")[1]])
    subprocess.run(cmd)


def cmd_eval(args=None):
    """评估微调效果"""
    print("=" * 60)
    print("评估微调效果...")
    print("=" * 60)
    subprocess.run([
        _VENV_PYTHON,
        os.path.join(ROOT, "finetune", "scripts", "evaluate.py"),
    ])


def cmd_rag(args=None):
    """启动 RAG 问答服务"""
    sys.path.insert(0, os.path.join(ROOT, "rag"))
    import uvicorn
    from app.main import app
    print("启动 RAG 问答服务: http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)


def cmd_all(args=None):
    cmd_generate()
    print()
    cmd_train()
    print()
    cmd_eval()


if __name__ == "__main__":
    cmds = {
        "generate": cmd_generate,
        "gen": cmd_generate,
        "train": cmd_train,
        "eval": cmd_eval,
        "evaluate": cmd_eval,
        "rag": cmd_rag,
        "all": cmd_all,
    }
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    fn = cmds.get(sys.argv[1])
    if fn:
        fn(sys.argv[2:])
    else:
        print(f"未知命令: {sys.argv[1]}")
        print("可用: generate, train, eval, rag, all")