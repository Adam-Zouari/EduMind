"""Application configuration has one source of defaults and strict keys."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from edumind.common.config import ConfigurationError, load_settings
from edumind.rag.errors import RAGConfigurationError
from edumind.rag.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config/base.yaml"


def _configured(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_application_defaults_come_from_yaml(tmp_path: Path) -> None:
    payload = deepcopy(yaml.safe_load(BASE.read_text(encoding="utf-8")))
    payload["extraction"]["video"]["scene_threshold"] = 0.6
    payload["generation"]["maximum_answer_tokens"] = 128
    settings = load_settings(_configured(tmp_path, payload))
    assert settings.extraction.video.scene_threshold == 0.6
    assert settings.generation.maximum_answer_tokens == 128
    assert not hasattr(settings.embedding, "dimension")


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("embedding", "model_name"),
        ("chunking", "chunk_size"),
        ("generation", "maximum_answer_tokens"),
    ],
)
def test_application_rejects_missing_settings(
    tmp_path: Path, section: str, key: str
) -> None:
    payload = deepcopy(yaml.safe_load(BASE.read_text(encoding="utf-8")))
    del payload[section][key]
    with pytest.raises(ConfigurationError, match="missing"):
        load_settings(_configured(tmp_path, payload))


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("extraction", "model_lock_path"),
        ("embedding", "model_path"),
        ("vector", "embedded"),
        ("chunking", "unused_option"),
    ],
)
def test_application_rejects_unknown_and_retired_settings(
    tmp_path: Path, section: str, key: str
) -> None:
    payload = deepcopy(yaml.safe_load(BASE.read_text(encoding="utf-8")))
    payload[section][key] = True
    with pytest.raises(ConfigurationError, match="unknown"):
        load_settings(_configured(tmp_path, payload))


def test_application_rejects_nonfinite_and_coerced_values(tmp_path: Path) -> None:
    payload = deepcopy(yaml.safe_load(BASE.read_text(encoding="utf-8")))
    payload["generation"]["temperature"] = float("nan")
    with pytest.raises(ConfigurationError, match="temperature"):
        load_settings(_configured(tmp_path, payload))
    payload["generation"]["temperature"] = 0.0
    payload["extraction"]["cache_enabled"] = "false"
    with pytest.raises(ConfigurationError, match="cache_enabled"):
        load_settings(_configured(tmp_path, payload))


def test_application_environment_override_still_applies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EDUMIND_VECTOR_ENDPOINT", "http://127.0.0.1:8123")
    settings = load_settings(BASE)
    assert settings.vector.endpoint == "http://127.0.0.1:8123"


def test_application_rejects_invalid_vector_port() -> None:
    with pytest.raises(ConfigurationError, match="valid HTTP URL"):
        load_settings(BASE, overrides={"vector": {"endpoint": "http://localhost:abc"}})


def test_vector_distance_must_match_selected_embedding_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings(BASE, overrides={"vector": {"distance_metric": "dot"}})
    monkeypatch.setattr("edumind.rag.pipeline.load_model_lock", lambda _path: {})
    monkeypatch.setattr(
        "edumind.rag.pipeline.require_model",
        lambda _lock, _name: SimpleNamespace(revision="pinned", path=tmp_path),
    )
    with pytest.raises(RAGConfigurationError, match="Vector distance metric"):
        RAGPipeline(settings=settings, vector_store=object())
