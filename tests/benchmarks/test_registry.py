from __future__ import annotations

from edumind.benchmarks.registry import CANDIDATES, candidates_for, validate_unique_registry
from edumind.extraction.pipeline import build_default_registry


def test_every_selectable_extractor_has_a_benchmark_entry() -> None:
    registered = build_default_registry().names()
    benchmarked = {item.name for item in CANDIDATES if item.suite == "extraction"}
    assert set(registered) <= benchmarked
    validate_unique_registry()
    assert len(candidates_for("rag", "generation")) == 10
