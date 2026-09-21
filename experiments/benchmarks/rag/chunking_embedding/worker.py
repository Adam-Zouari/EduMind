"""Isolated worker for one chunker/embedding candidate pair."""

from __future__ import annotations

from experiments.benchmarks.common.process import json_worker_main
from experiments.benchmarks.rag.chunking_embedding.benchmark import execute_payload


if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute_payload))
