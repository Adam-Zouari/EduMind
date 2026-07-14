from __future__ import annotations

import pytest

from edumind.common.config import ConfigurationError, default_config_path, load_settings


def test_packaged_defaults_load_outside_repository(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert default_config_path().is_file()
    assert settings.service.host == "127.0.0.1"
    assert settings.retrieval.context_token_budget == 2048


def test_invalid_cross_field_configuration_fails() -> None:
    with pytest.raises(ConfigurationError, match="chunk_overlap"):
        load_settings(overrides={"chunking": {"chunk_size": 10, "chunk_overlap": 10}})
    with pytest.raises(ConfigurationError, match="candidate_k"):
        load_settings(overrides={"retrieval": {"top_k": 10, "candidate_k": 5}})
    with pytest.raises(ConfigurationError, match="loopback"):
        load_settings(overrides={"service": {"host": "0.0.0.0"}})
    with pytest.raises(ConfigurationError, match="Unknown keys"):
        load_settings(overrides={"embedding": {"device": "cuda"}})
    with pytest.raises(ConfigurationError, match="boolean"):
        load_settings(overrides={"extraction": {"cache_enabled": "false"}})
    with pytest.raises(ConfigurationError, match="must not exceed"):
        load_settings(overrides={"chunking": {"chunk_size": 300}})
    with pytest.raises(ConfigurationError, match="must remain 'chroma'"):
        load_settings(overrides={"vector": {"backend": "qdrant-local"}})
    with pytest.raises(ConfigurationError, match="loopback HTTP"):
        load_settings(overrides={"generation": {"base_url": "https://example.com"}})
    with pytest.raises(ConfigurationError, match="keep_alive"):
        load_settings(overrides={"generation": {"keep_alive": "forever"}})
    with pytest.raises(ConfigurationError, match="immutable revision"):
        load_settings(overrides={"retrieval": {"strategy": "rrf-minilm-reranker"}})
    with pytest.raises(ConfigurationError, match="only valid"):
        load_settings(overrides={"retrieval": {"reranker_revision": "abc123"}})

    reranked = load_settings(
        overrides={
            "retrieval": {
                "strategy": "rrf-minilm-reranker",
                "reranker_revision": "abc123",
            }
        }
    )
    assert reranked.retrieval.reranker_device == "cpu"


def test_environment_override_is_typed(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EDUMIND_ARTIFACTS", str(tmp_path))
    assert load_settings().benchmark.artifact_directory == tmp_path.resolve()
