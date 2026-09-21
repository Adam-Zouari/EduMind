"""Isolated worker for one retrieval/reranking candidate."""

from __future__ import annotations

import sys
import time
from pathlib import Path

WORKER_STARTED_AT = time.perf_counter()

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.benchmarks.common.process import json_worker_main  # noqa: E402
from experiments.benchmarks.rag.retrieval.benchmark import execute_payload  # noqa: E402


def execute_with_start(payload: dict[str, object]) -> dict[str, object]:
    payload["worker_started_at"] = WORKER_STARTED_AT
    return execute_payload(payload)


if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute_with_start))
