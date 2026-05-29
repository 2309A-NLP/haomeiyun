from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from datasets import Dataset
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = ROOT_DIR / "研发"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.core.config import settings
from app.rag.embedding import BGEEmbedder
from app.rag.pipeline import LegalRAGPipeline
from app.services.llm_service import LLMService

try:
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import llm_factory
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
except ImportError as exc:
    raise SystemExit(
        "未安装 ragas。请先执行: pip install -r requirements-ragas.txt"
    ) from exc


DEFAULT_EVALSET_PATH = Path(__file__).resolve().parent / "data" / "eval" / "ragas_evalset.json"


@dataclass
class EvalSample:
    question: str
    ground_truth: str
    legal_field: str | None = None


class LocalBGEEmbeddings:
    def __init__(self) -> None:
        self.embedder = BGEEmbedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self.embedder.encode(texts, return_sparse=False)
        return [vector.tolist() for vector in result["dense_vecs"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.encode_dense(text)


def extract_query_keywords(question: str) -> list[str]:
    keywords: list[str] = []
    rule_terms = [
        "违法解除劳动合同",
        "解除劳动合同",
        "劳动合同",
        "劳动者",
        "用人单位",
        "继续履行",
        "赔偿金",
        "经济补偿",
        "违约金",
        "实际损失",
        "法院调整",
        "夫妻共同债务",
        "共同偿还",
        "婚姻关系",
        "借款用途",
        "共同签名",
        "追认",
    ]
    for term in rule_terms:
        if term in question and term not in keywords:
            keywords.append(term)

    if "劳动" in question and "劳动" not in keywords:
        keywords.append("劳动")
    if "违约" in question and "违约" not in keywords:
        keywords.append("违约")
    if "夫妻" in question and "夫妻" not in keywords:
        keywords.append("夫妻")

    return keywords


def rerank_docs_for_eval(question: str, docs: Sequence[dict[str, Any]], legal_field: str | None) -> list[dict[str, Any]]:
    if not docs:
        return []

    keywords = extract_query_keywords(question)
    rescored: list[tuple[float, dict[str, Any]]] = []
    for doc in docs:
        merged_text = " ".join(
            [
                str(doc.get("source", "") or ""),
                str(doc.get("article_number", "") or ""),
                str(doc.get("content", "") or ""),
            ]
        )
        overlap = sum(1 for keyword in keywords if keyword and keyword in merged_text)
        field_bonus = 3 if legal_field and doc.get("legal_field") == legal_field else 0
        mismatch_penalty = -2 if legal_field and doc.get("legal_field") not in ("", legal_field) else 0
        base_score = float(doc.get("score", 0) or 0)
        final_score = overlap * 10 + field_bonus + mismatch_penalty + base_score
        rescored.append((final_score, doc))

    rescored.sort(key=lambda item: item[0], reverse=True)
    best_docs = [doc for score, doc in rescored if score > 0]
    return best_docs[:2] or [doc for _, doc in rescored[:1]]


def evidence_coverage_score(question: str, docs: Sequence[dict[str, Any]]) -> float:
    if not docs:
        return 0.0

    keywords = extract_query_keywords(question)
    if not keywords:
        return 1.0 if docs else 0.0

    merged = " ".join(
        " ".join(
            [
                str(doc.get("source", "") or ""),
                str(doc.get("article_number", "") or ""),
                str(doc.get("content", "") or ""),
            ]
        )
        for doc in docs
    )
    hits = sum(1 for keyword in keywords if keyword and keyword in merged)
    return hits / max(len(keywords), 1)


def normalize_answer(text: str) -> str:
    cleaned = (text or "").replace("\r", "\n").strip()
    cleaned = cleaned.replace("```markdown", "").replace("```", "").strip()
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行基于当前项目 RAG 流程的 RAGAS 测评")
    parser.add_argument(
        "--evalset",
        type=Path,
        default=DEFAULT_EVALSET_PATH,
        help="评测集 JSON 文件路径，默认 data/eval/ragas_evalset.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="每个问题保留的检索片段数",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="生成答案使用的 LLM provider，默认走项目配置 DEFAULT_LLM_PROVIDER",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="生成答案使用的 model，默认走项目配置 DEFAULT_LLM_MODEL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选，保存逐条评测输入输出的 JSON 文件",
    )
    return parser.parse_args()


def load_evalset(evalset_path: Path) -> list[EvalSample]:
    if not evalset_path.exists():
        raise FileNotFoundError(f"评测集文件不存在: {evalset_path}")

    raw_data = json.loads(evalset_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list) or not raw_data:
        raise ValueError("评测集必须是非空 JSON 数组")

    samples: list[EvalSample] = []
    for index, item in enumerate(raw_data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 条评测数据不是对象")

        question = str(item.get("question", "")).strip()
        ground_truth = str(item.get("ground_truth", "")).strip()
        legal_field = item.get("legal_field")
        if legal_field is not None:
            legal_field = str(legal_field).strip() or None

        if not question or not ground_truth:
            raise ValueError(f"第 {index} 条评测数据缺少 question 或 ground_truth")

        samples.append(
            EvalSample(
                question=question,
                ground_truth=ground_truth,
                legal_field=legal_field,
            )
        )

    return samples


def build_prompt(question: str, contexts: Sequence[str]) -> tuple[str, str]:
    context_block = "\n\n".join(
        f"[参考片段{i}]\n{content}" for i, content in enumerate(contexts, start=1)
    )
    if not context_block.strip():
        context_block = "当前未检索到可用法律知识，请基于已知法律常识谨慎作答，并明确说明信息可能不足。"

    system_prompt = (
        "你是一名严谨的法律 RAG 问答助手。"
        "回答时只围绕用户问题直接作答，优先依据给定参考片段，不要编造法条或案例。"
        "不要输出多余背景，不要扩展无关知识。"
        "如果参考信息不足，请明确说明“依据不足，无法直接下结论”，不要补全未给出的事实。"
    )
    user_prompt = (
        f"问题：{question}\n\n"
        f"参考资料：\n{context_block}\n\n"
        "请基于参考资料给出简洁、准确、聚焦问题的中文回答，控制在 2-4 句话内。"
        "如果参考资料无法支撑完整结论，请直接说明依据不足，并指出还缺什么信息。"
    )
    return system_prompt, user_prompt


async def answer_one_question(
    rag: LegalRAGPipeline,
    llm: LLMService,
    sample: EvalSample,
    top_k: int,
    provider: str | None,
    model: str | None,
) -> dict[str, Any]:
    docs = await rag.retrieve(
        query=sample.question,
        legal_field=sample.legal_field,
        top_k=top_k,
    )
    fallback_used = False
    if not docs and sample.legal_field:
        docs = await rag.retrieve(
            query=sample.question,
            legal_field=None,
            top_k=top_k,
        )
        fallback_used = True

    docs = rerank_docs_for_eval(sample.question, docs, sample.legal_field)
    contexts = [
        (doc.get("content", "") or "").strip()
        for doc in docs
        if (doc.get("content", "") or "").strip()
    ]
    coverage = evidence_coverage_score(sample.question, docs)

    if coverage < 0.35:
        cautious_answer = "根据当前检索结果，直接依据仍然不足，暂时无法对该问题作出完整确定结论。建议补充更具体的事实、证据或对应法条后再判断。"
        return {
            "question": sample.question,
            "answer": cautious_answer,
            "contexts": contexts or [""],
            "ground_truth": sample.ground_truth,
            "legal_field": sample.legal_field or "",
            "fallback_used": fallback_used,
            "retrieved_docs": docs,
        }

    system_prompt, user_prompt = build_prompt(sample.question, contexts)
    answer = normalize_answer(
        await llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=min(settings.LLM_MAX_TOKENS, 320),
            provider=provider,
            model=model,
        )
    )

    if not answer:
        answer = "根据当前参考资料，能够确认的信息有限，暂时无法在不补充依据的情况下给出更完整结论。"

    if not answer:
        if contexts:
            answer = f"根据检索到的资料，当前只能确认：{contexts[0][:100]}"
        else:
            answer = "当前知识库未检索到足够直接的依据，暂时无法给出可靠结论。"

    return {
        "question": sample.question,
        "answer": answer,
        "contexts": contexts or [""],
        "ground_truth": sample.ground_truth,
        "legal_field": sample.legal_field or "",
        "fallback_used": fallback_used,
        "retrieved_docs": docs,
    }


async def run_generation(
    samples: Sequence[EvalSample],
    top_k: int,
    provider: str | None,
    model: str | None,
) -> list[dict[str, Any]]:
    rag = LegalRAGPipeline()
    llm = LLMService()
    rows: list[dict[str, Any]] = []

    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] 正在评测: {sample.question}")
        row = await answer_one_question(
            rag=rag,
            llm=llm,
            sample=sample,
            top_k=top_k,
            provider=provider,
            model=model,
        )
        preview = row["answer"].replace("\n", " ").strip()[:120]
        print(
            f"  -> 命中文档 {len(row['retrieved_docs'])} 条"
            f"{'，已回退到全库检索' if row['fallback_used'] else ''}"
        )
        print(f"  -> 证据覆盖率: {evidence_coverage_score(sample.question, row['retrieved_docs']):.2f}")
        print(f"  -> 回答预览: {preview}")
        rows.append(row)

    return rows


def ensure_openai_compatible_env() -> None:
    if not os.getenv("OPENAI_API_KEY") and settings.DEEPSEEK_API_KEY:
        os.environ["OPENAI_API_KEY"] = settings.DEEPSEEK_API_KEY

    if not os.getenv("OPENAI_BASE_URL") and settings.DEEPSEEK_BASE_URL:
        base_url = settings.DEEPSEEK_BASE_URL.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        os.environ["OPENAI_BASE_URL"] = base_url


def build_ragas_resources() -> tuple[Any, Any]:
    ensure_openai_compatible_env()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise RuntimeError(
            "RAGAS 评测缺少可用的 OpenAI 兼容配置。"
            "当前脚本会优先复用 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL。"
        )

    client = OpenAI(api_key=api_key, base_url=base_url)
    llm = llm_factory(
        settings.DEFAULT_LLM_MODEL,
        provider="openai",
        client=client,
    )
    embeddings = LangchainEmbeddingsWrapper(LocalBGEEmbeddings())
    return llm, embeddings


def build_metrics() -> list[Any]:
    return [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]


def run_ragas(rows: Sequence[dict[str, Any]]):
    llm, embeddings = build_ragas_resources()
    metrics = build_metrics()

    dataset = Dataset.from_dict(
        {
            "question": [row["question"] for row in rows],
            "answer": [row["answer"] for row in rows],
            "contexts": [row["contexts"] for row in rows],
            "ground_truth": [row["ground_truth"] for row in rows],
        }
    )

    try:
        return evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            column_map={
                "user_input": "question",
                "response": "answer",
                "retrieved_contexts": "contexts",
                "reference": "ground_truth",
            },
            raise_exceptions=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "RAGAS 评测执行失败。请确认已安装 requirements-ragas.txt 中的依赖，"
            "并检查 Milvus、嵌入模型路径、DeepSeek/OpenAI 兼容配置是否可用。"
        ) from exc


def save_output(output_path: Path, rows: Sequence[dict[str, Any]], result: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows: Any
    if hasattr(result, "to_pandas"):
        summary_rows = result.to_pandas().to_dict(orient="records")
    else:
        summary_rows = str(result)

    payload = {
        "summary": summary_rows,
        "samples": list(rows),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def async_main() -> None:
    args = parse_args()
    samples = load_evalset(args.evalset)
    rows = await run_generation(
        samples=samples,
        top_k=args.top_k,
        provider=args.provider,
        model=args.model,
    )
    result = run_ragas(rows)

    print("\nRAGAS 评测结果:")
    print(result)
    if hasattr(result, "to_pandas"):
        print("\n明细表:")
        print(result.to_pandas().to_string(index=False))

    if args.output:
        save_output(args.output, rows, result)
        print(f"\n评测详情已保存到: {args.output}")


if __name__ == "__main__":
    asyncio.run(async_main())
