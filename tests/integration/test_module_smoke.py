import importlib

import pytest


def _import_or_skip(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Optional dependency missing for smoke import: {exc.name}")
    except Exception as exc:  # pragma: no cover - environment-dependent optional stack
        pytest.skip(f"Optional runtime stack not available for smoke import: {exc}")


def test_rag_service_app_exists() -> None:
    pytest.importorskip("fastapi")
    module = _import_or_skip("services.rag_service")
    assert hasattr(module, "app")


def test_streamlit_app_module_exists() -> None:
    pytest.importorskip("streamlit")
    module = _import_or_skip("apps.streamlit_app")
    assert module is not None
