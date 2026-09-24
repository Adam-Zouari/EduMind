"""Reusable CUDA qualification and model-placement evidence."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from edumind.common.paths import PROJECT_ROOT

from .contracts import DatasetManifest
from .preflight_reports import qualification_groups
from .process import WorkerMeasurementError, WorkerResourceLimitError
from .protocol import ProtocolMetadata
from .provenance import hardware_summary
from .tracking import EXPERIMENT_NAMES, tracker


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


def stress_input_identity(
    manifest: DatasetManifest, samples: Sequence[Mapping[str, object]]
) -> dict[str, str]:
    """Bind qualification to the exact reviewed manifest and selected stress rows."""

    return {
        "stress_manifest_checksum": manifest.fingerprint,
        "stress_input_fingerprint": stable_hash([dict(row) for row in samples]),
    }


def current_qualification_fingerprint(
    *,
    benchmark: str,
    candidates: Sequence[str],
    protocols: Mapping[str, ProtocolMetadata],
    revisions: Mapping[str, str],
    execution: Mapping[str, object],
    input_envelope: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    required_input_identity = {"stress_manifest_checksum", "stress_input_fingerprint"}
    missing = sorted(required_input_identity - set(input_envelope))
    if missing:
        raise ValueError(
            "Qualification input envelope is missing: " + ", ".join(missing)
        )
    hardware = dict(hardware_summary())
    locks = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in (
            PROJECT_ROOT / "requirements/app.lock",
            PROJECT_ROOT / "requirements/benchmarks.lock",
        )
        if path.is_file()
    }
    source = source_provenance()
    fingerprint = qualification_fingerprint(
        benchmark=benchmark,
        candidates=candidates,
        protocols=protocols,
        revisions=revisions,
        hardware=hardware,
        execution=execution,
        input_envelope=input_envelope,
        dependency_locks=locks,
        source_provenance=source,
    )
    return fingerprint, {
        "hardware": hardware,
        "dependency_locks": locks,
        "source_provenance": source,
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
    readiness: bool
    group_readiness: Mapping[str, bool]
    artifact_directory: Path

    @property
    def ready_for_development(self) -> bool:
        return self.readiness


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
    source_provenance: Mapping[str, object],
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
            "source_provenance": dict(source_provenance),
        }
    )


@lru_cache(maxsize=1)
def source_provenance() -> dict[str, object]:
    """Identify the executable source tree, including uncommitted Python edits."""

    roots = (
        PROJECT_ROOT / "experiments/benchmarks",
        PROJECT_ROOT / "src/edumind",
    )
    files = sorted(
        (
            path
            for root in roots
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
        ),
        key=lambda path: path.as_posix(),
    )
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if pyproject.is_file():
        files.append(pyproject)
    checksums = {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path) for path in files
    }
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return {
        "schema": "benchmark-source-tree-v1",
        "git_commit": commit,
        "source_tree_sha256": stable_hash(checksums),
        "source_file_count": len(checksums),
    }


def run_preflight(
    *,
    benchmark: str,
    candidates: Sequence[str],
    fingerprint: str,
    context: Mapping[str, object],
    probe: Callable[[str], Mapping[str, object]],
    required_groups: Mapping[str, Sequence[str]] | None = None,
    decision_files: Mapping[str, Path] | None = None,
    no_mlflow: bool = False,
    artifact_root: Path = PROJECT_ROOT / "artifacts/benchmarks",
) -> PreflightResult:
    """Qualify candidates without turning hardware eligibility into quality evidence."""

    if benchmark not in EXPERIMENT_NAMES:
        raise ValueError(f"Unknown preflight benchmark: {benchmark}")
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Preflight candidates must be non-empty and unique")
    groups = qualification_groups(candidates, required_groups)
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
                candidate_artifact = directory / "candidates" / _safe_name(candidate)
                candidate_artifact = candidate_artifact.with_suffix(".json")
                atomic_write_json(
                    candidate_artifact,
                    {
                        "schema_version": 1,
                        "benchmark": benchmark,
                        "qualification_fingerprint": fingerprint,
                        **row,
                    },
                )
                tracking.artifact(candidate_artifact, "preflight-candidates")
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
        group_readiness = {
            name: any(candidate in qualified for candidate in members)
            for name, members in groups.items()
        }
        ready = all(group_readiness.values()) and not blocked
        payload = {
            "schema_version": 2,
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
            "required_groups": {name: list(values) for name, values in groups.items()},
            "group_readiness": group_readiness,
            "ready_for_development": ready,
        }
        report = directory / "preflight_report.json"
        atomic_write_json(report, payload)
        tracking.artifact(report)
        tracking.metrics(
            {
                "qualified_candidates": float(len(qualified)),
                "excluded_candidates": float(len(excluded)),
                "blocked_candidates": float(len(blocked)),
                "ready_for_development": float(ready),
            }
        )
        tracking.tags(
            {
                "qualification.fingerprint": fingerprint,
                "qualification.ready": str(ready).lower(),
            }
        )
    return PreflightResult(
        run_id,
        parent_id,
        fingerprint,
        qualified,
        excluded,
        blocked,
        ready,
        group_readiness,
        directory,
    )


def _safe_name(value: str) -> str:
    slug = "".join(character if character.isalnum() else "-" for character in value)
    return f"{slug[:80]}-{stable_hash(value)[:12]}"


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
