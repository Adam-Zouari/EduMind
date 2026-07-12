"""Shared MLflow configuration for maintained experiments."""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow

from edumind.common.paths import ARTIFACTS_DIR, DATA_DIR

logger = logging.getLogger(__name__)

MLFLOW_DIR = ARTIFACTS_DIR / "experiments" / "mlflow"
DB_PATH = MLFLOW_DIR / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = MLFLOW_DIR / "mlartifacts"
TRACKING_URI = f"sqlite:///{DB_PATH.as_posix()}"
EVALUATION_DIR = DATA_DIR / "evaluation"


def configure_mlflow(verbose: bool = True) -> str:
    """Create the maintained MLflow directories and set the tracking URI."""
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if verbose:
        logger.info("MLflow tracking URI: %s", TRACKING_URI)
        logger.info("MLflow artifact root: %s", MLFLOW_ARTIFACTS_DIR)
    return TRACKING_URI


def ensure_experiment(experiment_name: str) -> str:
    """Create the experiment with a stable local artifact root when needed."""
    configure_mlflow(verbose=False)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id

    artifact_dir = get_experiment_artifact_dir(experiment_name)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return mlflow.create_experiment(
        experiment_name,
        artifact_location=artifact_dir.resolve().as_uri(),
    )


def get_tracking_uri() -> str:
    """Return the configured SQLite tracking URI."""
    return TRACKING_URI


def get_artifacts_dir() -> Path:
    """Return the maintained MLflow artifact root."""
    return MLFLOW_ARTIFACTS_DIR


def get_experiment_artifact_dir(experiment_name: str) -> Path:
    """Return the artifact directory for one experiment name."""
    return MLFLOW_ARTIFACTS_DIR / _slugify(experiment_name)


def _slugify(value: str) -> str:
    """Build a filesystem-safe name for experiment artifacts."""
    slug = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return slug.strip("-") or "experiment"
