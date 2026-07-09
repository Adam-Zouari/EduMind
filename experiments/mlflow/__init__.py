"""MLflow experiment package."""

from __future__ import annotations

__all__ = ["configure_mlflow", "get_artifacts_dir", "get_tracking_uri"]


def __getattr__(name: str):
    if name in {"configure_mlflow", "get_artifacts_dir", "get_tracking_uri"}:
        from .mlflow_config import configure_mlflow, get_artifacts_dir, get_tracking_uri

        return {
            "configure_mlflow": configure_mlflow,
            "get_artifacts_dir": get_artifacts_dir,
            "get_tracking_uri": get_tracking_uri,
        }[name]
    raise AttributeError(name)
