"""Isolated worker for one retrieval/reranking candidate."""

from __future__ import annotations

from experiments.benchmarks.common.process import json_worker_main
from experiments.benchmarks.rag.retrieval_reranking.benchmark import (
    execute_payload,
)

if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute_payload))
