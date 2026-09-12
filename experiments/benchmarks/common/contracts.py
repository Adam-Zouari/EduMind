"""Small value objects shared by direct benchmark scripts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from edumind.common.artifacts import stable_hash


class CandidateExecutionError(RuntimeError):
    """Unexpected measured failure that still carries partial audit evidence."""

    def __init__(
        self,
        message: str,
        *,
        parameters: Mapping[str, object] | None = None,
        artifacts: Mapping[str, object] | None = None,
        samples: tuple[SampleResult, ...] = (),
        metrics: Mapping[str, float | None] | None = None,
        intervals: Mapping[str, Mapping[str, float]] | None = None,
        operational: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(message)
        self.parameters = dict(parameters or {})
        self.artifacts = dict(artifacts or {})
        self.samples = samples
        self.metrics = dict(metrics or {})
        self.intervals = dict(intervals or {})
        self.operational = dict(operational or {})


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

    @property
    def fingerprint(self) -> str:
        """Hash content plus provenance fields, not samples alone."""
        return stable_hash(asdict(self))


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
    warmups: int = 2
    settings: Mapping[str, object] = field(default_factory=dict)


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
    metrics: Mapping[str, float | None]
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
    complete: bool
    completion_problems: tuple[str, ...]
    artifact_directory: Path
