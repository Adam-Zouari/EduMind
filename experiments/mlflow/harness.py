"""Shared staged-experiment execution helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from experiments.mlflow.mlflow_config import MLFLOW_DIR

RESULTS_ROOT = MLFLOW_DIR / "staged_results"


@dataclass(frozen=True)
class StageCandidateResult:
    """Cached result for one stage candidate."""

    stage: str
    dataset_name: str
    dataset_version: str
    split: str
    candidate_name: str
    candidate_config: dict[str, object] = field(default_factory=dict)
    status: str = "completed"
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly result payload."""
        return asdict(self)


def collect_hardware_info() -> dict[str, object]:
    """Return lightweight hardware/runtime info for experiment logging."""
    info: dict[str, object] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import torch
    except ImportError:
        info["cuda_available"] = False
    else:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_count"] = int(torch.cuda.device_count())
            info["cuda_device_name"] = str(torch.cuda.get_device_name(0))
    return info


def build_candidate_hash(
    *,
    stage: str,
    dataset_name: str,
    dataset_version: str,
    split: str,
    candidate_config: dict[str, object],
) -> str:
    """Build a stable hash for candidate resume support."""
    payload = {
        "stage": stage,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "split": split,
        "candidate_config": candidate_config,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def build_stage_directory(dataset_name: str, split: str, stage: str) -> Path:
    """Return the cache directory for one dataset split and stage."""
    path = RESULTS_ROOT / dataset_name / split / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_cached_candidate_result(
    *,
    stage: str,
    dataset_name: str,
    dataset_version: str,
    split: str,
    candidate_config: dict[str, object],
) -> StageCandidateResult | None:
    """Load a cached candidate result if present."""
    candidate_hash = build_candidate_hash(
        stage=stage,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        split=split,
        candidate_config=candidate_config,
    )
    path = build_stage_directory(dataset_name, split, stage) / f"{candidate_hash}.json"
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return StageCandidateResult(
        stage=str(payload.get("stage", stage)),
        dataset_name=str(payload.get("dataset_name", dataset_name)),
        dataset_version=str(payload.get("dataset_version", dataset_version)),
        split=str(payload.get("split", split)),
        candidate_name=str(payload.get("candidate_name", "")),
        candidate_config=_coerce_dict(payload.get("candidate_config")),
        status=str(payload.get("status", "completed")),
        metrics=_coerce_float_dict(payload.get("metrics")),
        artifacts=_coerce_dict(payload.get("artifacts")),
        notes=[str(note) for note in payload.get("notes", []) if isinstance(note, str)],
        skip_reason=_coerce_optional_string(payload.get("skip_reason")),
    )


def save_cached_candidate_result(result: StageCandidateResult) -> Path:
    """Persist one candidate result for resume support."""
    candidate_hash = build_candidate_hash(
        stage=result.stage,
        dataset_name=result.dataset_name,
        dataset_version=result.dataset_version,
        split=result.split,
        candidate_config=result.candidate_config,
    )
    path = build_stage_directory(result.dataset_name, result.split, result.stage) / (
        f"{candidate_hash}.json"
    )
    path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def save_stage_outputs(
    *,
    stage: str,
    dataset_name: str,
    split: str,
    leaderboard: list[dict[str, object]],
    best_candidates: dict[str, object],
    summary_markdown: str,
) -> dict[str, Path]:
    """Persist standard stage artifacts used by later stages."""
    stage_dir = build_stage_directory(dataset_name, split, stage)
    leaderboard_json_path = stage_dir / "leaderboard.json"
    leaderboard_csv_path = stage_dir / "leaderboard.csv"
    best_candidates_path = stage_dir / "best_candidates.json"
    summary_path = stage_dir / "stage_summary.md"

    leaderboard_json_path.write_text(
        json.dumps(leaderboard, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    _write_leaderboard_csv(leaderboard_csv_path, leaderboard)
    best_candidates_path.write_text(
        json.dumps(best_candidates, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    summary_path.write_text(summary_markdown, encoding="utf-8")
    return {
        "leaderboard_json": leaderboard_json_path,
        "leaderboard_csv": leaderboard_csv_path,
        "best_candidates_json": best_candidates_path,
        "stage_summary_md": summary_path,
    }


def load_stage_best_candidates(
    *,
    stage: str,
    dataset_name: str,
    split: str,
) -> dict[str, object] | None:
    """Load previously persisted best-candidate metadata for one stage."""
    path = build_stage_directory(dataset_name, split, stage) / "best_candidates.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def load_stage_leaderboard(
    *,
    stage: str,
    dataset_name: str,
    split: str,
) -> list[dict[str, object]]:
    """Load a saved stage leaderboard if present."""
    path = build_stage_directory(dataset_name, split, stage) / "leaderboard.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def normalized_higher_is_better(values: list[float]) -> dict[float, float]:
    """Normalize a metric where higher values are better."""
    if not values:
        return {}
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return {value: 1.0 for value in values}
    return {value: (value - minimum) / (maximum - minimum) for value in values}


def normalized_lower_is_better(values: list[float]) -> dict[float, float]:
    """Normalize a metric where lower values are better."""
    if not values:
        return {}
    higher = normalized_higher_is_better(values)
    return {value: 1.0 - higher[value] for value in values}


def build_stage_summary(
    *,
    stage: str,
    dataset_name: str,
    split: str,
    leaderboard: list[dict[str, object]],
    notes: list[str] | None = None,
) -> str:
    """Render a compact stage summary Markdown artifact."""
    lines = [
        f"# {stage.title()} Stage Summary",
        "",
        f"- Dataset: `{dataset_name}` / `{split}`",
        f"- Candidates evaluated: `{len(leaderboard)}`",
    ]
    if notes:
        lines.append("- Notes:")
        for note in notes:
            lines.append(f"  - {note}")

    if leaderboard:
        top_row = leaderboard[0]
        lines.extend(
            [
                "",
                "## Top Candidate",
                "",
                f"- Name: `{top_row.get('candidate_name', '<unknown>')}`",
                f"- Score: `{top_row.get('stage_score', 0.0):.4f}`",
            ]
        )
    return "\n".join(lines) + "\n"


def top_candidate_names(
    leaderboard: list[dict[str, object]],
    *,
    count: int,
    status: str = "completed",
) -> list[str]:
    """Return the top candidate names from a stage leaderboard."""
    selected: list[str] = []
    for row in leaderboard:
        if str(row.get("status", "")) != status:
            continue
        name = row.get("candidate_name")
        if isinstance(name, str):
            selected.append(name)
        if len(selected) >= count:
            break
    return selected


def candidate_names_from_rows(
    rows: list[dict[str, object]],
    *,
    key: str,
    count: int,
) -> list[str]:
    """Return unique values from ranked rows for one key."""
    selected: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or value in seen:
            continue
        selected.append(value)
        seen.add(value)
        if len(selected) >= count:
            break
    return selected


def append_stage_score(
    rows: list[dict[str, object]],
    *,
    score_key: str,
    sort_descending: bool = True,
) -> list[dict[str, object]]:
    """Return rows sorted by their stage score."""
    ranked_rows = list(rows)
    ranked_rows.sort(key=lambda row: float(row.get(score_key, 0.0)), reverse=sort_descending)
    return ranked_rows


def stage_results_to_artifact_payload(results: list[StageCandidateResult]) -> list[dict[str, object]]:
    """Convert cached stage results into JSON artifacts."""
    return [result.to_dict() for result in results]


def _write_leaderboard_csv(path: Path, leaderboard: list[dict[str, object]]) -> None:
    if not leaderboard:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in leaderboard:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in leaderboard:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _coerce_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): value for key, value in value.items()}
    return {}


def _coerce_float_dict(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    metrics: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, (int, float)):
            metrics[str(key)] = float(item)
    return metrics


def _coerce_optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
