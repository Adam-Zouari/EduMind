"""Immutable benchmark plan, sample, and result contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from edumind.common.artifacts import stable_hash


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    version: str
    task: str
    split: str
    source: str
    license: str
    revision: str
    checksum: str
    preprocessing_version: str
    split_seed: int
    samples: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class BenchmarkPlan:
    suite: str
    stage: str
    profile: str
    dataset: str
    candidates: tuple[str, ...]
    seed: int = 42
    repetitions: int = 1
    bootstrap_resamples: int = 10_000
    cold_measurements: bool = True
    warmups: int = 2

    @property
    def fingerprint(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class SampleResult:
    sample_id: str
    metrics: Mapping[str, float]
    latency_seconds: float
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateResult:
    candidate: str
    status: str
    fingerprint: str
    metrics: Mapping[str, float]
    intervals: Mapping[str, Mapping[str, float]]
    samples: tuple[SampleResult, ...]
    operational: Mapping[str, float]
    error: str | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    run_id: str
    plan: BenchmarkPlan
    provenance: Mapping[str, object]
    candidates: tuple[CandidateResult, ...]
    pareto_candidates: tuple[str, ...]
    authoritative: bool
    artifact_directory: Path
