from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.benchmark_service import BenchmarkService


def main() -> None:
    summary = BenchmarkService().run()
    print(f"total={summary.total}")
    print(f"correct={summary.correct}")
    print(f"avg_latency_ms={summary.avg_latency_ms:.1f}")
    for result in summary.results:
        print(
            f"[{result.id}] matched={result.matched} "
            f"question={result.question} predicted={result.predicted_answer}"
        )


if __name__ == "__main__":
    main()
