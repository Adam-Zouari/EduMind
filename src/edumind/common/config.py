"""Small, typed runtime configuration loaded from ``config/base.yaml``."""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when the single runtime configuration is invalid."""


@dataclass(frozen=True)
class ExtractionSettings:
    cache_enabled: bool = True
    cache_directory: Path = Path("artifacts/extraction/cache")
    maximum_upload_bytes: int = 100 * 1024 * 1024


@dataclass(frozen=True)
class ModelSettings:
    lock_path: Path = Path("data/benchmarks/models/selected.json")


@dataclass(frozen=True)
class EmbeddingSettings:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    indexing_device: str = "cpu"
    query_device: str = "cpu"
    query_prefix: str = ""
    document_prefix: str = ""
    normalize: bool = True
    similarity: str = "cosine"
    maximum_length: int = 256
    batch_size: int = 32


@dataclass(frozen=True)
class ChunkingSettings:
    strategy: str = "token"
    chunk_size: int = 256
    chunk_overlap: int = 32
    tokenizer: str = "embedding"


@dataclass(frozen=True)
class VectorSettings:
    backend: str = "chroma-server"
    endpoint: str = "http://127.0.0.1:8001"
    collection_name: str = "edumind"
    distance_metric: str = "cosine"


@dataclass(frozen=True)
class RetrievalSettings:
    strategy: str = "dense"
    top_k: int = 5
    candidate_k: int = 20
    context_token_budget: int = 2048


@dataclass(frozen=True)
class GenerationSettings:
    model_name: str = "Qwen/Qwen3-1.7B"
    device: str = "cpu"
    dtype: str = "auto"
    reasoning: bool = False
    temperature: float = 0.0
    seed: int = 42
    context_tokens: int = 8192
    maximum_answer_tokens: int = 256


@dataclass(frozen=True)
class Settings:
    models: ModelSettings = field(default_factory=ModelSettings)
    extraction: ExtractionSettings = field(default_factory=ExtractionSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    vector: VectorSettings = field(default_factory=VectorSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)


def default_config_path() -> Path:
    """Return the repository's only runtime configuration file."""
    return Path(__file__).resolve().parents[3] / "config" / "base.yaml"


def load_yaml_config(config_path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv(override=False)
    path = Path(config_path or os.getenv("EDUMIND_CONFIG") or default_config_path()).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError("Configuration root must be a mapping")
    return dict(payload)


def load_settings(
    config_path: str | Path | None = None,
    *,
    overrides: Mapping[str, object] | None = None,
) -> Settings:
    raw = load_yaml_config(config_path)
    if overrides:
        _merge(raw, overrides)
    _apply_environment(raw)
    return _build(raw)


def _merge(target: dict[str, Any], update: Mapping[str, object]) -> None:
    for key, value in update.items():
        if isinstance(target.get(key), dict) and isinstance(value, Mapping):
            _merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def _apply_environment(raw: dict[str, Any]) -> None:
    values = {
        "EDUMIND_VECTOR_ENDPOINT": ("vector", "endpoint"),
        "EDUMIND_MODEL_LOCK": ("models", "lock_path"),
        "EDUMIND_GENERATION_DEVICE": ("generation", "device"),
    }
    for variable, (section, key) in values.items():
        value = os.getenv(variable)
        if value is not None:
            raw.setdefault(section, {})[key] = value


def _section(raw: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"'{name}' must be a mapping")
    return value


def _integer(section: Mapping[str, object], key: str, default: int, minimum: int = 1) -> int:
    value = section.get(key, default)
    if isinstance(value, bool):
        raise ConfigurationError(f"'{key}' must be an integer")
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"'{key}' must be an integer") from exc
    if result < minimum:
        raise ConfigurationError(f"'{key}' must be >= {minimum}")
    return result


def _path(value: object, default: Path) -> Path:
    path = Path(value if isinstance(value, (str, Path)) else default).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _build(raw: Mapping[str, object]) -> Settings:
    models = _section(raw, "models")
    extraction = _section(raw, "extraction")
    embedding = _section(raw, "embedding")
    chunking = _section(raw, "chunking")
    vector = _section(raw, "vector")
    retrieval = _section(raw, "retrieval")
    generation = _section(raw, "generation")

    unknown_models = sorted(set(models) - {"lock_path"})
    if unknown_models:
        raise ConfigurationError("Unknown model settings: " + ", ".join(unknown_models))
    if "model_lock_path" in extraction:
        raise ConfigurationError(
            "extraction.model_lock_path was replaced by models.lock_path"
        )
    obsolete_embedding = sorted({"revision", "model_path"} & embedding.keys())
    if obsolete_embedding:
        raise ConfigurationError(
            "Embedding snapshots are resolved through models.lock_path; remove: "
            + ", ".join(obsolete_embedding)
        )

    allowed_generation_keys = {
        "model_name",
        "device",
        "dtype",
        "reasoning",
        "temperature",
        "seed",
        "context_tokens",
        "maximum_answer_tokens",
    }
    unknown_generation = sorted(set(generation) - allowed_generation_keys)
    if unknown_generation:
        raise ConfigurationError(
            "Unknown direct Hugging Face generation settings: "
            + ", ".join(unknown_generation)
        )

    obsolete_vector_keys = {
        "persist_directory",
        "persistence_path",
        "deployment_mode",
        "embedded",
        "local_path",
    }
    found_obsolete = sorted(obsolete_vector_keys & vector.keys())
    if found_obsolete:
        raise ConfigurationError(
            "Embedded vector configuration was removed. Delete these keys and configure the "
            f"Chroma HTTP endpoint instead: {', '.join(found_obsolete)}"
        )

    chunk_size = _integer(chunking, "chunk_size", 256)
    chunk_overlap = _integer(chunking, "chunk_overlap", 32, 0)
    if chunk_overlap >= chunk_size:
        raise ConfigurationError("chunk_overlap must be smaller than chunk_size")
    chunking_strategy = str(chunking.get("strategy", "token"))
    if chunking_strategy != "token":
        raise ConfigurationError("The provisional application supports only token chunking")
    top_k = _integer(retrieval, "top_k", 5)
    candidate_k = _integer(retrieval, "candidate_k", 20)
    if candidate_k < top_k:
        raise ConfigurationError("candidate_k must be >= top_k")

    backend = str(vector.get("backend", "chroma-server"))
    if backend != "chroma-server":
        raise ConfigurationError("The provisional application supports only 'chroma-server'")
    endpoint = str(vector.get("endpoint", "http://127.0.0.1:8001"))
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.port:
        raise ConfigurationError("vector.endpoint must be an HTTP URL with an explicit port")
    strategy = str(retrieval.get("strategy", "dense"))
    if strategy != "dense":
        raise ConfigurationError("The provisional application supports only dense retrieval")
    similarity = str(embedding.get("similarity", "cosine"))
    distance = str(vector.get("distance_metric", "cosine"))
    if similarity != distance or similarity not in {"cosine", "dot"}:
        raise ConfigurationError("Embedding similarity and vector distance must match")
    generation_device = str(generation.get("device", "cpu"))
    if generation_device not in {"cpu", "cuda"}:
        raise ConfigurationError("generation.device must be 'cpu' or 'cuda'")
    generation_dtype = str(generation.get("dtype", "auto"))
    if generation_dtype != "auto":
        raise ConfigurationError("generation.dtype must be 'auto' for native checkpoints")

    return Settings(
        models=ModelSettings(
            lock_path=_path(
                models.get("lock_path"), Path("data/benchmarks/models/selected.json")
            )
        ),
        extraction=ExtractionSettings(
            cache_enabled=bool(extraction.get("cache_enabled", True)),
            cache_directory=_path(
                extraction.get("cache_directory"), Path("artifacts/extraction/cache")
            ),
            maximum_upload_bytes=_integer(
                extraction, "maximum_upload_bytes", 100 * 1024 * 1024
            ),
        ),
        embedding=EmbeddingSettings(
            model_name=str(embedding.get("model_name", EmbeddingSettings.model_name)),
            dimension=_integer(embedding, "dimension", 384),
            indexing_device=str(embedding.get("indexing_device", "cpu")),
            query_device=str(embedding.get("query_device", "cpu")),
            query_prefix=str(embedding.get("query_prefix", "")),
            document_prefix=str(embedding.get("document_prefix", "")),
            normalize=bool(embedding.get("normalize", True)),
            similarity=similarity,
            maximum_length=_integer(embedding, "maximum_length", 256),
            batch_size=_integer(embedding, "batch_size", 32),
        ),
        chunking=ChunkingSettings(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=str(chunking.get("tokenizer", "embedding")),
        ),
        vector=VectorSettings(
            backend=backend,
            endpoint=endpoint,
            collection_name=str(vector.get("collection_name", "edumind")),
            distance_metric=distance,
        ),
        retrieval=RetrievalSettings(
            strategy=strategy,
            top_k=top_k,
            candidate_k=candidate_k,
            context_token_budget=_integer(retrieval, "context_token_budget", 2048),
        ),
        generation=GenerationSettings(
            model_name=str(generation.get("model_name", GenerationSettings.model_name)),
            device=generation_device,
            dtype=generation_dtype,
            reasoning=bool(generation.get("reasoning", False)),
            temperature=float(generation.get("temperature", 0.0)),
            seed=_integer(generation, "seed", 42, 0),
            context_tokens=_integer(generation, "context_tokens", 8192),
            maximum_answer_tokens=_integer(generation, "maximum_answer_tokens", 256),
        ),
    )
