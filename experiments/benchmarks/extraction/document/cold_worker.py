"""Measure one document profile from a fresh Python process."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from edumind.extraction import ExtractionPipeline
from experiments.benchmarks.extraction.common import _extract_once
from experiments.benchmarks.extraction.registry import build_experiment_registry


def main(payload_path: Path) -> int:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    pipeline = ExtractionPipeline(registry=build_experiment_registry())
    _extract_once(
        "document",
        payload["candidate"],
        payload["item"],
        payload["model_lock"],
        payload["component_options"],
        pipeline,
    )
    print("EDUMIND_FIRST_ITEM_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
