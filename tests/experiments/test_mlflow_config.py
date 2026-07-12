from __future__ import annotations

import pytest

pytest.importorskip("mlflow")

from edumind.common.paths import ARTIFACTS_DIR, PROJECT_ROOT
from experiments.mlflow import mlflow_config


def test_mlflow_paths_live_under_artifacts() -> None:
    assert mlflow_config.DB_PATH.is_relative_to(ARTIFACTS_DIR)
    assert mlflow_config.get_artifacts_dir().is_relative_to(ARTIFACTS_DIR)
    assert not mlflow_config.DB_PATH.is_relative_to(PROJECT_ROOT / "experiments" / "mlflow")


def test_configure_mlflow_returns_sqlite_tracking_uri() -> None:
    tracking_uri = mlflow_config.configure_mlflow(verbose=False)

    assert tracking_uri.startswith("sqlite:///")
    assert "artifacts/experiments/mlflow/mlflow.db" in tracking_uri
