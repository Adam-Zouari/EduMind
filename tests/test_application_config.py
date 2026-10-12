"""Application configuration has one source of defaults and strict keys."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from edumind.common.config import ConfigurationError, load_settings
from edumind.common.models import ModelPreparationError, load_model_lock, require_model
from edumind.extraction import ExtractionProfile
from edumind.extraction.pipeline import ExtractionPipeline
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


@pytest.mark.parametrize("payload", [[], None, {}, {"models": []}])
def test_model_lock_rejects_malformed_root(tmp_path, payload):
    import json

    path = tmp_path / "selected.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelPreparationError, match="Malformed model lock"):
        load_model_lock(path)


@pytest.mark.parametrize(
    "entry",
    [
        {"revision": "pinned"},
        {"revision": "pinned", "model_path": ""},
        {"revision": None},
    ],
)
def test_missing_snapshot_path_cannot_resolve_to_working_directory(entry):
    with pytest.raises(ModelPreparationError, match="incomplete"):
        require_model({"model": entry}, "model")


def test_video_defaults_and_overrides_drive_extraction_cache_identity(tmp_path):
    settings = load_settings(
        BASE, overrides={"models": {"lock_path": str(tmp_path / "missing.json")}}
    )
    profile = ExtractionProfile("video", "video-hybrid", "ffmpeg-system")
    original, _ = ExtractionPipeline(settings)._prepare_profile(profile)
    changed_settings = load_settings(
        BASE,
        overrides={
            "models": {"lock_path": str(tmp_path / "missing.json")},
            "extraction": {"video": {"scene_threshold": 0.6}},
        },
    )
    changed, _ = ExtractionPipeline(changed_settings)._prepare_profile(profile)
    assert original.options["scene_threshold"] == 0.35
    assert changed.options["scene_threshold"] == 0.6
    assert original.fingerprint != changed.fingerprint
    override, _ = ExtractionPipeline(changed_settings)._prepare_profile(
        ExtractionProfile(
            "video", "video-hybrid", "ffmpeg-system", options={"scene_threshold": 0.7}
        )
    )
    assert override.options["scene_threshold"] == 0.7


@pytest.mark.parametrize("top_k", [None, 3])
def test_application_and_controller_preserve_configured_retrieval_default(top_k):
    from edumind.application import EduMindPipeline
    from edumind.ui.controller import AppController

    selected = []

    def query(_text, *, top_k, filter_metadata):
        selected.append(top_k)
        return []

    pipeline = EduMindPipeline(extraction=object(), rag=SimpleNamespace(query=query))
    pipeline.query("question", top_k=top_k, generate_answer=False)
    assert selected == [top_k]
    calls = []
    controller = AppController(
        SimpleNamespace(query=lambda *args, **kwargs: calls.append(kwargs))
    )
    controller.query("question", top_k=top_k)
    assert calls[0]["top_k"] == top_k


def test_query_view_uses_configured_retrieval_default(monkeypatch):
    from edumind.ui import streamlit_app

    calls = []
    monkeypatch.setattr(streamlit_app.st, "text_area", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        streamlit_app.st,
        "select_slider",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(streamlit_app.st, "button", lambda *args, **kwargs: False)
    controller = SimpleNamespace(pipeline=SimpleNamespace(rag=SimpleNamespace(top_k=7)))
    streamlit_app._render_query(controller)
    assert calls == [{"options": [1, 3, 5, 7, 10], "value": 7}]


@pytest.mark.parametrize("top_k", [None, 3, 0, -1, True, 1.5])
def test_rag_query_resolves_and_validates_limit_before_embedding(top_k):
    calls = []
    pipeline = object.__new__(RAGPipeline)
    pipeline.top_k = 7
    pipeline.settings = SimpleNamespace(
        retrieval=SimpleNamespace(candidate_k=20, context_token_budget=100)
    )
    pipeline.embedder = SimpleNamespace(
        embed_query=lambda _text: SimpleNamespace(tolist=lambda: calls.append("embed"))
    )
    pipeline.vector_store = SimpleNamespace(query_dense=lambda *args, **kwargs: [])
    pipeline._pack_hits = lambda _hits, limit, _budget: limit
    if top_k is None or top_k == 3:
        assert pipeline.query("question", top_k=top_k) == (7 if top_k is None else 3)
        assert calls == ["embed"]
    else:
        with pytest.raises(ValueError, match="positive integer"):
            pipeline.query("question", top_k=top_k)
        assert calls == []


def test_context_packing_counts_hits_without_stored_token_counts():
    from edumind.rag.types import RetrievalHit

    pipeline = object.__new__(RAGPipeline)
    pipeline.tokenizer = SimpleNamespace(
        count=lambda text: len(text.split()),
        truncate=lambda text, count: " ".join(text.split()[:count]),
    )
    candidates = [RetrievalHit(str(i), "one two three", {}, 1.0) for i in range(3)]
    packed = pipeline._pack_hits(candidates, limit=3, token_budget=10)
    assert [hit.token_count for hit in packed] == [3, 1]
    assert packed[1].metadata["context_truncated"] is True
