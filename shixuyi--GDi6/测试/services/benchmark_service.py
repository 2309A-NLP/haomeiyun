from __future__ import annotations

import json
from functools import cached_property

from ..core.config import settings
from ..models.schemas import AskRequest, BenchmarkCase, BenchmarkResult, BenchmarkSummary
from .rag_service import ProspectusRAGService


class BenchmarkService:
    def __init__(self) -> None:
        pass

    @cached_property
    def rag(self) -> ProspectusRAGService:
        return ProspectusRAGService()

    def run(self) -> BenchmarkSummary:
        cases = self._load_cases()
        results: list[BenchmarkResult] = []
        total_latency = 0
        correct = 0

        for case in cases:
            response = self.rag.ask(AskRequest(question=case.question, compare_plain_llm=False))
            matched = self._judge(case.expected_answer, response.answer)
            if matched:
                correct += 1
            total_latency += response.latency_ms
            results.append(
                BenchmarkResult(
                    id=case.id,
                    question=case.question,
                    expected_answer=case.expected_answer,
                    predicted_answer=response.answer,
                    matched=matched,
                    source=response.source,
                )
            )

        total = len(results)
        return BenchmarkSummary(
            total=total,
            correct=correct,
            avg_latency_ms=(total_latency / total) if total else 0.0,
            results=results,
        )

    def _load_cases(self) -> list[BenchmarkCase]:
        raw = json.loads((settings.data_dir / "benchmarks" / "xingtu_10_questions.json").read_text(encoding="utf-8"))
        return [BenchmarkCase.model_validate(item) for item in raw]

    def _judge(self, expected: str, predicted: str) -> bool:
        normalized_expected = expected.replace(" ", "").replace("（", "(").replace("）", ")")
        normalized_predicted = predicted.replace(" ", "").replace("（", "(").replace("）", ")")
        return normalized_expected in normalized_predicted or normalized_predicted in normalized_expected
