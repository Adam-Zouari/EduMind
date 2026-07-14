"""Reproducible execution, cache, provenance, and atomic result layout."""

from __future__ import annotations

import json
import random
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from edumind.common.artifacts import (
    atomic_write_json,
    git_provenance,
    hardware_summary,
    sha256_file,
    stable_hash,
)
from edumind.common.paths import PROJECT_ROOT

from .contracts import BenchmarkPlan, BenchmarkResult, CandidateResult, SampleResult
from .resources import ResourceMonitor
from .statistics import aggregate_samples, select_pareto
from .tracking import Tracker, build_tracker

CandidateEvaluator = Callable[[str], tuple[list[SampleResult], Mapping[str, float]]]


class BenchmarkHarness:
    def __init__(self, artifact_root: Path, *, tracking_uri: str | None = None) -> None:
        self.artifact_root = artifact_root
        self.tracker: Tracker = build_tracker(tracking_uri)

    def run(
        self,
        plan: BenchmarkPlan,
        evaluator: CandidateEvaluator,
        *,
        dataset_checksum: str,
        directions: Mapping[str, str],
        model_revisions: Mapping[str, str] | None = None,
        hard_gates: Mapping[str, tuple[str, float]] | None = None,
    ) -> BenchmarkResult:
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        run_directory = self.artifact_root / plan.suite / plan.stage / run_id
        provenance = self._provenance(plan, dataset_checksum, model_revisions or {})
        atomic_write_json(run_directory / "plan.json", asdict(plan))
        atomic_write_json(run_directory / "provenance.json", provenance)
        candidate_results: list[CandidateResult] = []
        with self.tracker.run(f"{plan.suite}-{plan.stage}-{run_id}"):
            self.tracker.log_parameters(
                {"plan_fingerprint": plan.fingerprint, "profile": plan.profile}
            )
            candidate_order = list(plan.candidates)
            random.Random(plan.seed).shuffle(candidate_order)
            for candidate in candidate_order:
                candidate_results.append(
                    self._run_candidate(
                        plan, candidate, evaluator, dataset_checksum, provenance, run_directory
                    )
                )
        successful = [result for result in candidate_results if result.status == "success"]
        selection_rows = {
            result.candidate: {
                **result.metrics,
                **{f"operational.{key}": value for key, value in result.operational.items()},
            }
            for result in successful
        }
        gates = hard_gates or {}
        gate_failures = {
            candidate: _failed_gates(row, gates)
            for candidate, row in selection_rows.items()
            if _failed_gates(row, gates)
        }
        eligible_rows = {
            candidate: row
            for candidate, row in selection_rows.items()
            if candidate not in gate_failures
        }
        missing_direction_metrics = [
            name for name in directions if any(name not in row for row in eligible_rows.values())
        ]
        if missing_direction_metrics:
            raise ValueError(
                f"Pareto metrics missing from candidate results: {missing_direction_metrics}"
            )
        pareto = (
            select_pareto(eligible_rows, directions)
            if eligible_rows and plan.profile in {"standard", "full"}
            else []
        )
        summary = {
            "run_id": run_id,
            "plan": asdict(plan),
            "candidates": [self._candidate_payload(result) for result in candidate_results],
            "pareto_candidates": pareto,
            "gate_failures": gate_failures,
            "authoritative": plan.profile in {"standard", "full"} and bool(eligible_rows),
        }
        atomic_write_json(run_directory / "summary.json", summary)
        atomic_write_json(run_directory / "_SUCCESS.json", {"run_id": run_id, "complete": True})
        return BenchmarkResult(
            run_id,
            plan,
            provenance,
            tuple(candidate_results),
            tuple(pareto),
            bool(summary["authoritative"]),
            run_directory,
        )

    def _run_candidate(
        self,
        plan: BenchmarkPlan,
        candidate: str,
        evaluator: CandidateEvaluator,
        dataset_checksum: str,
        provenance: Mapping[str, object],
        run_directory: Path,
    ) -> CandidateResult:
        fingerprint = stable_hash(
            {
                "plan": asdict(plan),
                "candidate": candidate,
                "dataset_checksum": dataset_checksum,
                "provenance": provenance,
            }
        )
        cache_path = self.artifact_root / "cache" / f"{fingerprint}.json"
        cached = self._load_cache(cache_path)
        if cached is not None:
            result = self._candidate_from_payload(cached)
            atomic_write_json(
                run_directory / "candidates" / f"{_safe_name(candidate)}.json", cached
            )
            return result
        try:
            with self.tracker.run(candidate, nested=True):
                with ResourceMonitor() as resources:
                    samples, operational = evaluator(candidate)
                operational = {**operational, **resources.metrics()}
                if not samples:
                    raise RuntimeError("Candidate produced no sample results")
                metrics, intervals = aggregate_samples(
                    samples, resamples=plan.bootstrap_resamples, seed=plan.seed
                )
                result = CandidateResult(
                    candidate,
                    "success",
                    fingerprint,
                    metrics,
                    intervals,
                    tuple(samples),
                    dict(operational),
                )
                self.tracker.log_metrics(
                    {
                        **metrics,
                        **{f"operational_{key}": value for key, value in operational.items()},
                    }
                )
        except Exception as exc:
            result = CandidateResult(
                candidate, "failed", fingerprint, {}, {}, (), {}, f"{type(exc).__name__}: {exc}"
            )
        payload = self._candidate_payload(result)
        candidate_path = run_directory / "candidates" / f"{_safe_name(candidate)}.json"
        atomic_write_json(candidate_path, payload)
        if result.status == "success":
            atomic_write_json(cache_path, payload)
        return result

    @staticmethod
    def _load_cache(path: Path) -> dict[str, object] | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if payload.get("status") == "success" and payload.get("samples") else None

    @staticmethod
    def _candidate_payload(result: CandidateResult) -> dict[str, object]:
        return {
            **asdict(result),
            "samples": [asdict(sample) for sample in result.samples],
        }

    @staticmethod
    def _candidate_from_payload(payload: Mapping[str, object]) -> CandidateResult:
        raw_samples = payload.get("samples", [])
        if not isinstance(raw_samples, list):
            raise ValueError("Cached candidate samples must be a list")
        samples = tuple(SampleResult(**cast(Any, item)) for item in raw_samples)
        return CandidateResult(
            str(payload["candidate"]),
            str(payload["status"]),
            str(payload["fingerprint"]),
            dict(cast(Mapping[str, float], payload.get("metrics", {}))),
            dict(cast(Mapping[str, Mapping[str, float]], payload.get("intervals", {}))),
            samples,
            dict(cast(Mapping[str, float], payload.get("operational", {}))),
            str(payload["error"]) if payload.get("error") else None,
        )

    @staticmethod
    def _provenance(
        plan: BenchmarkPlan, dataset_checksum: str, model_revisions: Mapping[str, str]
    ) -> dict[str, object]:
        lock = PROJECT_ROOT / "requirements.lock"
        return {
            "plan_fingerprint": plan.fingerprint,
            "dataset_checksum": dataset_checksum,
            "git": git_provenance(PROJECT_ROOT),
            "dependency_lock_checksum": sha256_file(lock) if lock.is_file() else None,
            "model_revisions": dict(model_revisions),
            "hardware": hardware_summary(),
            "seed": plan.seed,
        }


def _safe_name(name: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_" for character in name
    )


def _failed_gates(row: Mapping[str, float], gates: Mapping[str, tuple[str, float]]) -> list[str]:
    failures: list[str] = []
    for metric, (direction, threshold) in gates.items():
        if metric not in row:
            failures.append(f"{metric}:missing")
        elif direction == "max" and row[metric] < threshold:
            failures.append(f"{metric}<{threshold}")
        elif direction == "min" and row[metric] > threshold:
            failures.append(f"{metric}>{threshold}")
        elif direction not in {"min", "max"}:
            raise ValueError(f"Unknown gate direction for {metric}: {direction}")
    return failures
