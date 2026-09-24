"""Validation and exact discovery of persisted GPU qualification reports."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from edumind.common.paths import PROJECT_ROOT

from .tracking import DEFAULT_TRACKING_URI, EXPERIMENT_NAMES


def qualification_groups(
    candidates: Sequence[str],
    required_groups: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    groups = required_groups or {"candidates": tuple(candidates)}
    normalized: dict[str, tuple[str, ...]] = {}
    for name, members in groups.items():
        if isinstance(members, (str, bytes)):
            raise ValueError("Preflight qualification-group members must be lists")
        normalized[str(name)] = tuple(str(candidate) for candidate in members)
    if not normalized or any(
        not name or not members for name, members in normalized.items()
    ):
        raise ValueError("Preflight qualification groups must be non-empty")
    declared = set(candidates)
    unknown = sorted(
        candidate
        for members in normalized.values()
        for candidate in members
        if candidate not in declared
    )
    if unknown:
        raise ValueError(
            "Preflight qualification groups contain undeclared candidates: "
            + ", ".join(sorted(set(unknown)))
        )
    return normalized


def load_preflight_report(
    path: Path,
    *,
    fingerprint: str,
    declared_candidates: Sequence[str],
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError(f"{path} must use preflight schema_version 2")
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
    groups = qualification_groups(
        declared_candidates,
        payload.get("required_groups")
        if isinstance(payload.get("required_groups"), Mapping)
        else None,
    )
    group_readiness = {
        name: any(candidate in by_status["qualified"] for candidate in members)
        for name, members in groups.items()
    }
    if payload.get("group_readiness") != group_readiness:
        raise ValueError(f"{path} has inconsistent qualification-group readiness")
    ready = all(group_readiness.values()) and not by_status["blocked"]
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
