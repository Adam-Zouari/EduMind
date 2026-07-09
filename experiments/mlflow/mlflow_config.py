"""Shared MLflow configuration for EduMind experiments."""

from __future__ import annotations

import mlflow

from edumind.common.paths import ARTIFACTS_DIR, DATA_DIR

MLFLOW_DIR = ARTIFACTS_DIR / "mlflow"
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = MLFLOW_DIR / "mlflow.db"
ARTIFACTS_DIR_MLFLOW = MLFLOW_DIR / "mlartifacts"
ARTIFACTS_DIR_MLFLOW.mkdir(parents=True, exist_ok=True)

DB_URI = f"sqlite:///{DB_PATH.as_posix()}"
EVALUATION_DIR = DATA_DIR / "evaluation"


def configure_mlflow(verbose: bool = True) -> None:
    mlflow.set_tracking_uri(DB_URI)
    if verbose:
        print("[OK] MLflow configured")
        print(f"  Database: {DB_PATH}")
        print(f"  URI: {DB_URI}")


def get_tracking_uri() -> str:
    return DB_URI


def get_artifacts_dir() -> Path:
    return ARTIFACTS_DIR_MLFLOW


configure_mlflow(verbose=False)
