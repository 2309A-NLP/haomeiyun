from __future__ import annotations

import argparse
import csv
import re
from collections import OrderedDict
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [{key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(file)]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\r", " ").replace("\n", " ")).strip()


def normalize_text(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def bigrams(text: str) -> set[str]:
    if not text:
        return set()
    if len(text) == 1:
        return {text}
    return {text[i : i + 2] for i in range(len(text) - 1)}


def similarity_score(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm and not right_norm:
        return 100.0
    if not left_norm or not right_norm:
        return 0.0

    seq_ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    left_bigrams = bigrams(left_norm)
    right_bigrams = bigrams(right_norm)
    union = left_bigrams | right_bigrams
    jaccard = len(left_bigrams & right_bigrams) / len(union) if union else 1.0
    return round((seq_ratio * 0.65 + jaccard * 0.35) * 100, 2)


def extract_numbers(text: str) -> list[str]:
    values = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?%?", clean_text(text))
    return list(OrderedDict.fromkeys(values))


def normalize_number(num: str) -> str:
    return num.replace(",", "").strip()


def numbers_match(left_numbers: list[str], right_numbers: list[str]) -> tuple[int, int, int]:
    left_normalized = [normalize_number(num) for num in left_numbers]
    right_normalized = [normalize_number(num) for num in right_numbers]
    shared = sum(1 for num in left_normalized if num in right_normalized)
    return len(left_numbers), len(right_numbers), shared


def classify_hallucination(
    question: str,
    text_answer: str,
    llm_answer: str,
    score: float,
) -> tuple[str, str]:
    text_clean = clean_text(text_answer)
    llm_clean = clean_text(llm_answer)
    text_numbers = extract_numbers(text_clean)
    llm_numbers = extract_numbers(llm_clean)
    left_num_count, right_num_count, shared_num_count = numbers_match(text_numbers, llm_numbers)

    reasons: list[str] = []
    hallucination = "否"

    if not llm_clean:
        return "未知", "LLM.csv 中未找到对应答案"

    if text_clean == "证据不足。":
        if llm_clean and llm_clean != "证据不足。":
            hallucination = "疑似是"
            reasons.append("检索答案为证据不足，但大模型给出了明确结论")
        return hallucination, "；".join(reasons) if reasons else "基准答案为证据不足，无法进一步核验"

    if score < 35:
        hallucination = "疑似是"
        reasons.append("文本相似度很低")
    elif score < 60:
        hallucination = "可能是"
        reasons.append("文本相似度偏低")

    if left_num_count and not right_num_count:
        hallucination = "疑似是"
        reasons.append("基准答案包含关键数值，但大模型未给出对应数值")
    elif left_num_count and right_num_count:
        if shared_num_count == 0:
            hallucination = "疑似是"
            reasons.append("关键数值全部不一致")
        elif shared_num_count < left_num_count:
            hallucination = "可能是" if hallucination == "否" else hallucination
            reasons.append("关键数值仅部分一致")

    generic_patterns = [
        "根据公开信息",
        "通常",
        "需查阅",
        "最新财报",
        "未明确列出",
        "某重大国防工程",
    ]
    if any(pattern in llm_clean for pattern in generic_patterns) and score < 80:
        hallucination = "可能是" if hallucination == "否" else hallucination
        reasons.append("回答存在泛化或模糊表述")

    if question.find("法定代表人") >= 0 and "程家明" in text_clean and "程家明" in llm_clean:
        hallucination = "否"
        reasons = ["核心事实一致"]

    if not reasons:
        if score >= 85:
            reasons.append("核心内容高度一致")
        else:
            reasons.append("未发现明显幻觉特征")

    return hallucination, "；".join(OrderedDict.fromkeys(reasons))


def build_analysis(
    text_answer: str,
    llm_answer: str,
    score: float,
    hallucination: str,
    hallucination_reason: str,
) -> str:
    text_numbers = extract_numbers(text_answer)
    llm_numbers = extract_numbers(llm_answer)
    left_num_count, right_num_count, shared_num_count = numbers_match(text_numbers, llm_numbers)

    comments: list[str] = []
    if score >= 85:
        comments.append("两份答案整体高度一致")
    elif score >= 60:
        comments.append("两份答案主题接近，但细节存在差异")
    else:
        comments.append("两份答案差异较大")

    if left_num_count or right_num_count:
        comments.append(
            f"数值对比：text-llm={left_num_count}个，LLM={right_num_count}个，共享={shared_num_count}个"
        )

    comments.append(f"幻觉判断：{hallucination}")
    comments.append(f"判断依据：{hallucination_reason}")
    return "；".join(comments)


def compare_files(left_path: Path, right_path: Path, output_path: Path) -> tuple[int, int, int]:
    left_rows = load_rows(left_path)
    right_rows = load_rows(right_path)
    right_map = {row.get("question", ""): row for row in right_rows if row.get("question", "")}

    matched = 0
    output_rows: list[dict[str, str]] = []
    for left_row in left_rows:
        question = left_row.get("question", "")
        text_answer = clean_text(left_row.get("answer", ""))
        right_row = right_map.get(question)
        llm_answer = clean_text(right_row.get("answer", "")) if right_row else ""
        if right_row:
            matched += 1

        score = similarity_score(text_answer, llm_answer)
        hallucination, hallucination_reason = classify_hallucination(question, text_answer, llm_answer, score)
        analysis = build_analysis(text_answer, llm_answer, score, hallucination, hallucination_reason)

        output_rows.append(
            {
                "id": left_row.get("id", ""),
                "question": question,
                "text_llm_answer": text_answer,
                "llm_answer": llm_answer,
                "similarity_score": f"{score:.2f}",
                "hallucination": hallucination,
                "hallucination_reason": hallucination_reason,
                "analysis": analysis,
                "text_llm_source": left_row.get("source", ""),
                "llm_source": right_row.get("source", "") if right_row else "",
            }
        )

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "question",
                "text_llm_answer",
                "llm_answer",
                "similarity_score",
                "hallucination",
                "hallucination_reason",
                "analysis",
                "text_llm_source",
                "llm_source",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    return len(left_rows), len(right_rows), matched


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare answers for the same questions in two CSV files.")
    parser.add_argument("--left", default="text-llm.csv", help="Left CSV file, default is text-llm.csv")
    parser.add_argument("--right", default="LLM.csv", help="Right CSV file, default is LLM.csv")
    parser.add_argument(
        "--output",
        default="answer_comparison.csv",
        help="Output CSV file, default is answer_comparison.csv",
    )
    args = parser.parse_args()

    left_path = ROOT / args.left
    right_path = ROOT / args.right
    output_path = ROOT / args.output

    left_total, right_total, matched = compare_files(left_path, right_path, output_path)
    print(f"left_total={left_total}")
    print(f"right_total={right_total}")
    print(f"matched={matched}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
