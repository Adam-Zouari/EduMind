"""Reusable CUDA qualification and model-placement evidence."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from edumind.common.paths import PROJECT_ROOT

from .contracts import DatasetManifest
from .process import WorkerMeasurementError, WorkerResourceLimitError
from .protocol import ProtocolMetadata
from .provenance import hardware_summary
from .tracking import DEFAULT_TRACKING_URI, EXPERIMENT_NAMES, tracker


def model_lock_fingerprints(
    model_lock: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    """Fingerprint immutable model identity without binding it to a local path."""

    return {
        name: stable_hash(
            {
                key: value
                for key, value in entry.items()
                if key in {"revision", "selection_revision"} or key.endswith("_sha256")
            }
        )
        for name, entry in model_lock.items()
    }


def rag_stress_manifest(manifest: DatasetManifest) -> DatasetManifest:
    """Keep one demanding reviewed document and one of its answerable questions."""

    documents = {
        str(row.get("id")): row
        for row in manifest.samples
        if row.get("kind") == "document"
    }
    questions = [
        row
        for row in manifest.samples
        if row.get("kind") == "question"
        and row.get("answerable")
        and str(row.get("document_id")) in documents
    ]
    if not questions:
        raise ValueError("GPU preflight requires an answerable reviewed RAG question")
    question = max(
        questions,
        key=lambda row: (
            len(str(documents[str(row["document_id"])].get("text", ""))),
            len(str(row.get("question", ""))),
            str(row.get("id", "")),
        ),
    )
    document = documents[str(question["document_id"])]
    return replace(
        manifest,
        name=f"{manifest.name}-preflight",
        samples=(document, question),
    )


def current_qualification_fingerprint(
    *,
    benchmark: str,
    candidates: Sequence[str],
    protocols: Mapping[str, ProtocolMetadata],
    revisions: Mapping[str, str],
    execution: Mapping[str, object],
    input_envelope: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    hardware = dict(hardware_summary())
    locks = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in (
            PROJECT_ROOT / "requirements/app.lock",
            PROJECT_ROOT / "requirements/benchmarks.lock",
        )
        if path.is_file()
    }
    fingerprint = qualification_fingerprint(
        benchmark=benchmark,
        candidates=candidates,
        protocols=protocols,
        revisions=revisions,
        hardware=hardware,
        execution=execution,
        input_envelope=input_envelope,
        dependency_locks=locks,
    )
    return fingerprint, {
        "hardware": hardware,
        "dependency_locks": locks,
        "execution": dict(execution),
        "input_envelope": dict(input_envelope),
        "protocols": {
            name: {
                "schema_version": value.schema_version,
                "version": value.version,
                "checksum": value.checksum,
                "source_path": str(value.source_path),
                "resolved": value.resolved,
            }
            for name, value in protocols.items()
        },
        "revisions": dict(revisions),
    }


@dataclass(frozen=True)
class PreflightResult:
    run_id: str
    mlflow_run_id: str
    fingerprint: str
    qualified_candidates: tuple[str, ...]
    excluded_candidates: tuple[str, ...]
    blocked_candidates: tuple[str, ...]
    artifact_directory: Path

    @property
    def ready_for_development(self) -> bool:
        return bool(self.qualified_candidates) and not self.blocked_candidates


def eligible_candidates(
    requested: Sequence[str],
    report: Mapping[str, object],
    *,
    profile: str,
    label: str,
) -> tuple[str, ...]:
    """Apply qualification without treating definitive exclusions as run failures."""

    qualified = {str(value) for value in report["qualified_candidates"]}
    if profile == "development":
        eligible = tuple(candidate for candidate in requested if candidate in qualified)
        if not eligible:
            raise ValueError(f"No {label} candidates qualified for development")
        return eligible
    rejected = sorted(set(requested) - qualified)
    if rejected:
        raise ValueError(
            f"{label} candidates did not pass the matching GPU preflight: "
            + ", ".join(rejected)
        )
    return tuple(requested)


def qualification_fingerprint(
    *,
    benchmark: str,
    candidates: Sequence[str],
    protocols: Mapping[str, ProtocolMetadata],
    revisions: Mapping[str, str],
    hardware: Mapping[str, object],
    execution: Mapping[str, object],
    input_envelope: Mapping[str, object],
    dependency_locks: Mapping[str, str],
) -> str:
    return stable_hash(
        {
            "benchmark": benchmark,
            "candidates": list(candidates),
            "protocols": {
                name: {"version": value.version, "checksum": value.checksum}
                for name, value in protocols.items()
            },
            "revisions": dict(revisions),
            "hardware": dict(hardware),
            "execution": dict(execution),
            "input_envelope": dict(input_envelope),
            "dependency_locks": dict(dependency_locks),
        }
    )


def run_preflight(
    *,
    benchmark: str,
    candidates: Sequence[str],
    fingerprint: str,
    context: Mapping[str, object],
    probe: Callable[[str], Mapping[str, object]],
    decision_files: Mapping[str, Path] | None = None,
    no_mlflow: bool = False,
    artifact_root: Path = PROJECT_ROOT / "artifacts/benchmarks",
) -> PreflightResult:
    """Qualify candidates without turning hardware eligibility into quality evidence."""

    if benchmark not in EXPERIMENT_NAMES:
        raise ValueError(f"Unknown preflight benchmark: {benchmark}")
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Preflight candidates must be non-empty and unique")
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    directory = artifact_root / benchmark / "preflight" / run_id
    tracking = tracker(disabled=no_mlflow, experiment=EXPERIMENT_NAMES[benchmark])
    rows: list[dict[str, object]] = []
    raw_protocol_context = context.get("protocols", {})
    protocol_context = (
        raw_protocol_context if isinstance(raw_protocol_context, Mapping) else {}
    )
    protocol_tags = {
        f"protocol.{name}.checksum": value.get("checksum", "")
        for name, value in protocol_context.items()
        if isinstance(value, Mapping)
    }
    protocol_artifacts: list[tuple[str, Path, Path]] = []
    for name, value in protocol_context.items():
        if not isinstance(value, Mapping):
            continue
        source = Path(str(value.get("source_path", "")))
        resolved_path = directory / f"{name}_protocol.json"
        atomic_write_json(
            resolved_path,
            {
                "schema_version": value.get("schema_version"),
                "protocol_version": value.get("version"),
                "checksum": value.get("checksum"),
                "resolved": value.get("resolved"),
            },
        )
        protocol_artifacts.append((str(name), source, resolved_path))
    decision_provenance = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in (decision_files or {}).items()
    }
    decision_fingerprint = (
        stable_hash(
            {name: value["sha256"] for name, value in decision_provenance.items()}
        )
        if decision_provenance
        else ""
    )
    with tracking.run(
        f"{benchmark}-preflight-{time.strftime('%Y%m%d-%H%M%S')}"
    ) as parent_id:
        tracking.parameters(
            {
                "qualification_fingerprint": fingerprint,
                "candidates": json.dumps(list(candidates)),
                "context": json.dumps(context, sort_keys=True),
                "engineer_decisions": json.dumps(decision_provenance, sort_keys=True),
            }
        )
        tracking.tags(
            {
                "benchmark": benchmark,
                "profile": "preflight",
                "phase": "preflight",
                "run_type": "preflight",
                "device": "cuda",
                "dataset_checksum": context.get("stress_manifest_checksum", ""),
                "qualification.fingerprint": fingerprint,
                "decision.fingerprint": decision_fingerprint,
                **protocol_tags,
            }
        )
        for name, source, resolved_path in protocol_artifacts:
            if source.is_file():
                tracking.artifact(source, f"inputs/protocols/{name}")
            tracking.artifact(resolved_path)
        for name, path in (decision_files or {}).items():
            tracking.artifact(path, f"engineer-decisions/{name}")
        for candidate in candidates:
            with tracking.run(candidate, nested=True):
                tracking.parameters(
                    {
                        "candidate": candidate,
                        "qualification_fingerprint": fingerprint,
                        "execution": json.dumps(
                            context.get("execution", {}), sort_keys=True
                        ),
                        "input_envelope": json.dumps(
                            context.get("input_envelope", {}), sort_keys=True
                        ),
                        **{
                            f"protocol.{name}.version": value.get("version", "")
                            for name, value in protocol_context.items()
                            if isinstance(value, Mapping)
                        },
                        **{
                            f"protocol.{name}.checksum": value.get("checksum", "")
                            for name, value in protocol_context.items()
                            if isinstance(value, Mapping)
                        },
                    }
                )
                tracking.tags(
                    {
                        "benchmark": benchmark,
                        "profile": "preflight",
                        "phase": "preflight",
                        "run_type": "preflight",
                        "device": "cuda",
                        "candidate": candidate,
                        "dataset_checksum": context.get("stress_manifest_checksum", ""),
                        "qualification.fingerprint": fingerprint,
                        "decision.fingerprint": decision_fingerprint,
                        **protocol_tags,
                    }
                )
                row = _probe_candidate(candidate, probe)
                rows.append(row)
                numeric = {
                    key: float(value)
                    for key, value in row.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                if numeric:
                    tracking.metrics(numeric)
                tracking.parameters(
                    {
                        "qualification_status": row["status"],
                        "reason_code": row["reason_code"],
                    }
                )
                tracking.tags(
                    {
                        "qualification.status": row["status"],
                        "qualification.reason": row["reason_code"],
                        "qualification.outcome": row["outcome"],
                    }
                )
        qualified = tuple(
            str(row["candidate"]) for row in rows if row["status"] == "qualified"
        )
        excluded = tuple(
            str(row["candidate"]) for row in rows if row["status"] == "excluded"
        )
        blocked = tuple(
            str(row["candidate"]) for row in rows if row["status"] == "blocked"
        )
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "mlflow_run_id": parent_id,
            "benchmark": benchmark,
            "qualification_fingerprint": fingerprint,
            "context": dict(context),
            "engineer_decisions": decision_provenance,
            "candidates": rows,
            "qualified_candidates": list(qualified),
            "excluded_candidates": list(excluded),
            "blocked_candidates": list(blocked),
            "ready_for_development": bool(qualified) and not blocked,
        }
        report = directory / "preflight_report.json"
        atomic_write_json(report, payload)
        tracking.artifact(report)
        tracking.metrics(
            {
                "qualified_candidates": float(len(qualified)),
                "excluded_candidates": float(len(excluded)),
                "blocked_candidates": float(len(blocked)),
                "ready_for_development": float(bool(qualified) and not blocked),
            }
        )
        tracking.tags(
            {
                "qualification.fingerprint": fingerprint,
                "qualification.ready": str(bool(qualified) and not blocked).lower(),
            }
        )
    return PreflightResult(
        run_id,
        parent_id,
        fingerprint,
        qualified,
        excluded,
        blocked,
        directory,
    )


def load_preflight_report(
    path: Path,
    *,
    fingerprint: str,
    declared_candidates: Sequence[str],
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path} must use preflight schema_version 1")
    if payload.get("qualification_fingerprint") != fingerprint:
        raise ValueError(f"{path} does not match the current qualification fingerprint")
    rows = payload.get("candidates")
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no candidate qualification rows")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{path} contains a malformed candidate row")
    observed = [str(row.get("candidate", "")) for row in rows]
    if set(observed) != set(declared_candidates) or len(observed) != len(set(observed)):
        raise ValueError(
            f"{path} does not cover the complete declared candidate roster"
        )
    by_status = {
        status: [str(row["candidate"]) for row in rows if row.get("status") == status]
        for status in ("qualified", "excluded", "blocked")
    }
    if sum(len(values) for values in by_status.values()) != len(rows):
        raise ValueError(f"{path} contains an unknown qualification status")
    allowed_outcomes = {
        "qualified": {"qualified"},
        "excluded": {"vram_limit_exceeded", "gpu_oom", "offload_detected"},
        "blocked": {
            "measurement_unavailable",
            "placement_unverifiable",
            "wrong_device",
            "execution_error",
        },
    }
    if any(
        row.get("outcome") not in allowed_outcomes[str(row["status"])]
        or row.get("reason_code") != row.get("outcome")
        for row in rows
    ):
        raise ValueError(f"{path} contains an invalid qualification outcome")
    for status, key in (
        ("qualified", "qualified_candidates"),
        ("excluded", "excluded_candidates"),
        ("blocked", "blocked_candidates"),
    ):
        if payload.get(key) != by_status[status]:
            raise ValueError(f"{path} has inconsistent {key}")
    ready = bool(by_status["qualified"]) and not by_status["blocked"]
    if payload.get("ready_for_development") is not ready:
        raise ValueError(f"{path} has an inconsistent readiness state")
    if not ready:
        raise ValueError(f"{path} contains unresolved preflight failures")
    return payload


def resolve_preflight_report(
    *,
    benchmark: str,
    fingerprint: str,
    candidates: Sequence[str],
    explicit: Path | None = None,
    run_id: str | None = None,
    artifact_root: Path = PROJECT_ROOT / "artifacts/benchmarks",
) -> tuple[Path, dict[str, object]]:
    """Resolve one exact qualification, never an unrelated latest run."""

    if explicit is not None and run_id is not None:
        raise ValueError(
            "Use either --preflight-report or --preflight-run-id, not both"
        )
    if explicit is not None:
        path = explicit.resolve()
    else:
        path = (
            _find_report_by_run_id(artifact_root, benchmark, run_id) if run_id else None
        )
        if path is None:
            path = _find_mlflow_preflight_report(
                benchmark, fingerprint, artifact_root, run_id=run_id
            )
        if path is None and run_id is not None:
            raise FileNotFoundError(
                f"No {benchmark} preflight report has run ID {run_id!r}"
            )
        if path is None:
            path = find_local_preflight_report(
                artifact_root, benchmark, fingerprint, candidates
            )
    payload = load_preflight_report(
        path,
        fingerprint=fingerprint,
        declared_candidates=candidates,
    )
    if payload.get("benchmark") != benchmark:
        raise ValueError(f"{path} belongs to a different benchmark")
    return path, payload


def find_local_preflight_report(
    artifact_root: Path,
    benchmark: str,
    fingerprint: str,
    candidates: Sequence[str],
) -> Path:
    matches: list[Path] = []
    for path in (artifact_root / benchmark / "preflight").glob(
        "*/preflight_report.json"
    ):
        try:
            load_preflight_report(
                path,
                fingerprint=fingerprint,
                declared_candidates=candidates,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"No valid {benchmark} preflight matches the current hardware and protocol; "
            "run --profile preflight or pass --preflight-report PATH"
        )
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _find_report_by_run_id(
    artifact_root: Path, benchmark: str, run_id: str | None
) -> Path | None:
    if run_id is None:
        return None
    for path in (artifact_root / benchmark / "preflight").glob(
        "*/preflight_report.json"
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if run_id in {payload.get("run_id"), payload.get("mlflow_run_id")}:
            return path
    return None


def _find_mlflow_preflight_report(
    benchmark: str,
    fingerprint: str,
    artifact_root: Path,
    *,
    run_id: str | None = None,
) -> Path | None:
    try:
        from mlflow import MlflowClient
    except ImportError:
        return None
    try:
        client = MlflowClient(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
        )
        if run_id is not None:
            run = client.get_run(run_id)
            tags = run.data.tags
            runs = (
                [run]
                if tags.get("qualification.fingerprint") == fingerprint
                and tags.get("qualification.ready") == "true"
                else []
            )
        else:
            experiment = client.get_experiment_by_name(EXPERIMENT_NAMES[benchmark])
            if experiment is None:
                return None
            escaped = fingerprint.replace("'", "\\'")
            runs = client.search_runs(
                [experiment.experiment_id],
                filter_string=(
                    f"tags.`qualification.fingerprint` = '{escaped}' and "
                    "tags.`qualification.ready` = 'true'"
                ),
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
        if not runs:
            return None
        destination = artifact_root / benchmark / "preflight" / "mlflow-cache"
        downloaded = client.download_artifacts(
            runs[0].info.run_id,
            "preflight_report.json",
            str(destination),
        )
        return Path(downloaded)
    except Exception:  # noqa: BLE001 - local exact-match fallback remains available
        return None


def _probe_candidate(candidate: str, probe) -> dict[str, object]:
    try:
        evidence = dict(probe(candidate))
    except WorkerResourceLimitError as exc:
        return {
            "candidate": candidate,
            "status": "excluded",
            "reason_code": "vram_limit_exceeded",
            "outcome": "vram_limit_exceeded",
            "peak_vram_mb": exc.peak_vram_mb,
            "vram_limit_mb": exc.limit_mb,
            "resource_samples": list(exc.samples),
        }
    except WorkerMeasurementError as exc:
        return {
            "candidate": candidate,
            "status": "blocked",
            "reason_code": "measurement_unavailable",
            "outcome": "measurement_unavailable",
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - classify and preserve probe failures
        message = f"{type(exc).__name__}: {exc}"
        lowered = message.casefold()
        oom = "out of memory" in lowered or "cuda oom" in lowered
        return {
            "candidate": candidate,
            "status": "excluded" if oom else "blocked",
            "reason_code": "gpu_oom" if oom else "execution_error",
            "outcome": "gpu_oom" if oom else "execution_error",
            "error": message,
        }
    placement = evidence.get("placement")
    placement_status = (
        str(placement.get("status", "placement_unverifiable"))
        if isinstance(placement, Mapping)
        else "placement_unverifiable"
    )
    if placement_status == "offload_detected":
        status, reason = "excluded", "offload_detected"
    elif placement_status in {"placement_unverifiable", "wrong_device"}:
        status, reason = "blocked", placement_status
    elif evidence.get("vram_measurement_method") in {None, "unavailable"}:
        status, reason = "blocked", "measurement_unavailable"
    else:
        status, reason = "qualified", "qualified"
    return {
        **evidence,
        "candidate": candidate,
        "status": status,
        "reason_code": reason,
        "outcome": reason,
    }
