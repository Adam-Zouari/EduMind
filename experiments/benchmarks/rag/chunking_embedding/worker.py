"""Isolated worker for one chunker/embedding candidate pair."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.benchmarks.common.process import json_worker_main
from experiments.benchmarks.rag.chunking_embedding.benchmark import execute_payload


if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute_payload))
