"""Small, typed runtime configuration loaded from ``config/base.yaml``."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when the single runtime configuration is invalid."""


@dataclass(frozen=True)
class VideoSettings:
    keyframe_strategy: str
    fixed_interval_seconds: float
    scene_threshold: float
    maximum_hybrid_gap_seconds: float


@dataclass(frozen=True)
class ExtractionSettings:
    cache_enabled: bool
    cache_directory: Path
    maximum_upload_bytes: int
    video: VideoSettings


@dataclass(frozen=True)
class ModelSettings:
    lock_path: Path


@dataclass(frozen=True)
class EmbeddingSettings:
    model_name: str
    indexing_device: str
    query_device: str
    batch_size: int


@dataclass(frozen=True)
class ChunkingSettings:
    strategy: str
    chunk_size: int
    chunk_overlap: int
    tokenizer: str


@dataclass(frozen=True)
class VectorSettings:
    backend: str
    endpoint: str
    collection_name: str
    distance_metric: str


@dataclass(frozen=True)
class RetrievalSettings:
    strategy: str
    top_k: int
    candidate_k: int
    context_token_budget: int


@dataclass(frozen=True)
class GenerationSettings:
    model_name: str
    device: str
    dtype: str
    reasoning: bool
    temperature: float
    do_sample: bool
    seed: int
    context_tokens: int
    maximum_answer_tokens: int
    streamer_timeout_seconds: float


@dataclass(frozen=True)
class Settings:
    models: ModelSettings
    extraction: ExtractionSettings
    embedding: EmbeddingSettings
    chunking: ChunkingSettings
    vector: VectorSettings
    retrieval: RetrievalSettings
    generation: GenerationSettings


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


def _section(
    raw: Mapping[str, object], name: str, fields: set[str]
) -> Mapping[str, object]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"'{name}' must be a mapping")
    missing = fields - value.keys()
    unknown = value.keys() - fields
    if missing or unknown:
        raise ConfigurationError(
            f"Invalid '{name}' settings: missing {sorted(missing)}, "
            f"unknown {sorted(unknown, key=str)}"
        )
    return value


def _integer(section: Mapping[str, object], key: str, minimum: int = 1) -> int:
    value = section[key]
    if type(value) is not int:
        raise ConfigurationError(f"'{key}' must be an integer")
    if value < minimum:
        raise ConfigurationError(f"'{key}' must be >= {minimum}")
    return value


def _path(value: object) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigurationError("Configured path must be a non-empty string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _number(
    section: Mapping[str, object], key: str, *, minimum: float = 0.0
) -> float:
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"'{key}' must be numeric")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ConfigurationError(f"'{key}' must be finite") from exc
    if not math.isfinite(result) or result < minimum:
        raise ConfigurationError(f"'{key}' must be >= {minimum}")
    return result


def _boolean(section: Mapping[str, object], key: str) -> bool:
    value = section[key]
    if not isinstance(value, bool):
        raise ConfigurationError(f"'{key}' must be a boolean")
    return value


def _string(section: Mapping[str, object], key: str) -> str:
    value = section[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"'{key}' must be a non-empty string")
    return value


def _build(raw: Mapping[str, object]) -> Settings:
    sections = {
        "models", "extraction", "embedding", "chunking", "vector", "retrieval",
        "generation",
    }
    missing = sections - raw.keys()
    unknown = raw.keys() - sections
    if missing or unknown:
        raise ConfigurationError(
            f"Invalid configuration sections: missing {sorted(missing)}, "
            f"unknown {sorted(unknown, key=str)}"
        )
    models = _section(raw, "models", {"lock_path"})
    extraction = _section(
        raw, "extraction",
        {"cache_enabled", "cache_directory", "maximum_upload_bytes", "video"},
    )
    video = _section(
        extraction, "video",
        {"keyframe_strategy", "fixed_interval_seconds", "scene_threshold", "maximum_hybrid_gap_seconds"},
    )
    embedding = _section(
        raw, "embedding", {"model_name", "indexing_device", "query_device", "batch_size"},
    )
    chunking = _section(
        raw, "chunking", {"strategy", "chunk_size", "chunk_overlap", "tokenizer"},
    )
    vector = _section(
        raw, "vector", {"backend", "endpoint", "collection_name", "distance_metric"},
    )
    retrieval = _section(
        raw, "retrieval", {"strategy", "top_k", "candidate_k", "context_token_budget"},
    )
    generation = _section(
        raw, "generation",
        {
            "model_name", "device", "dtype", "reasoning", "temperature", "do_sample",
            "seed", "context_tokens", "maximum_answer_tokens", "streamer_timeout_seconds",
        },
    )

    video_strategy = _string(video, "keyframe_strategy")
    if video_strategy not in {"fixed", "scene", "hybrid"}:
        raise ConfigurationError(
            "extraction.video.keyframe_strategy must be fixed, scene, or hybrid"
        )
    scene_threshold = _number(video, "scene_threshold")
    if scene_threshold > 1:
        raise ConfigurationError("extraction.video.scene_threshold must be <= 1")

    chunk_size = _integer(chunking, "chunk_size")
    chunk_overlap = _integer(chunking, "chunk_overlap", minimum=0)
    if chunk_overlap >= chunk_size:
        raise ConfigurationError("chunk_overlap must be smaller than chunk_size")
    chunking_strategy = _string(chunking, "strategy")
    if chunking_strategy != "token":
        raise ConfigurationError("The provisional application supports only token chunking")
    chunking_tokenizer = _string(chunking, "tokenizer")
    if chunking_tokenizer != "cl100k_base":
        raise ConfigurationError(
            "The provisional application supports only the frozen 'cl100k_base' "
            "chunking tokenizer"
        )
    top_k = _integer(retrieval, "top_k")
    candidate_k = _integer(retrieval, "candidate_k")
    if candidate_k < top_k:
        raise ConfigurationError("candidate_k must be >= top_k")

    backend = _string(vector, "backend")
    if backend != "chroma-server":
        raise ConfigurationError("The provisional application supports only 'chroma-server'")
    endpoint = _string(vector, "endpoint")
    try:
        parsed = urlparse(endpoint)
        valid_port = parsed.port is not None
    except ValueError as exc:
        raise ConfigurationError("vector.endpoint is not a valid HTTP URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not valid_port:
        raise ConfigurationError("vector.endpoint must be an HTTP URL with an explicit port")
    strategy = _string(retrieval, "strategy")
    if strategy != "dense":
        raise ConfigurationError("The provisional application supports only dense retrieval")
    distance = _string(vector, "distance_metric")
    if distance not in {"cosine", "dot"}:
        raise ConfigurationError("Unsupported vector distance metric")
    generation_device = _string(generation, "device")
    if generation_device not in {"cpu", "cuda"}:
        raise ConfigurationError("generation.device must be 'cpu' or 'cuda'")
    generation_dtype = _string(generation, "dtype")
    if generation_dtype != "auto":
        raise ConfigurationError("generation.dtype must be 'auto' for native checkpoints")

    return Settings(
        models=ModelSettings(
            lock_path=_path(models["lock_path"])
        ),
        extraction=ExtractionSettings(
            cache_enabled=_boolean(extraction, "cache_enabled"),
            cache_directory=_path(extraction["cache_directory"]),
            maximum_upload_bytes=_integer(extraction, "maximum_upload_bytes"),
            video=VideoSettings(
                keyframe_strategy=video_strategy,
                fixed_interval_seconds=_number(
                    video, "fixed_interval_seconds", minimum=1e-9
                ),
                scene_threshold=scene_threshold,
                maximum_hybrid_gap_seconds=_number(
                    video, "maximum_hybrid_gap_seconds", minimum=1e-9
                ),
            ),
        ),
        embedding=EmbeddingSettings(
            model_name=_string(embedding, "model_name"),
            indexing_device=_string(embedding, "indexing_device"),
            query_device=_string(embedding, "query_device"),
            batch_size=_integer(embedding, "batch_size"),
        ),
        chunking=ChunkingSettings(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=chunking_tokenizer,
        ),
        vector=VectorSettings(
            backend=backend,
            endpoint=endpoint,
            collection_name=_string(vector, "collection_name"),
            distance_metric=distance,
        ),
        retrieval=RetrievalSettings(
            strategy=strategy,
            top_k=top_k,
            candidate_k=candidate_k,
            context_token_budget=_integer(retrieval, "context_token_budget"),
        ),
        generation=GenerationSettings(
            model_name=_string(generation, "model_name"),
            device=generation_device,
            dtype=generation_dtype,
            reasoning=_boolean(generation, "reasoning"),
            temperature=_number(generation, "temperature"),
            do_sample=_boolean(generation, "do_sample"),
            seed=_integer(generation, "seed", minimum=0),
            context_tokens=_integer(generation, "context_tokens"),
            maximum_answer_tokens=_integer(generation, "maximum_answer_tokens"),
            streamer_timeout_seconds=_number(
                generation,
                "streamer_timeout_seconds",
                minimum=1e-9,
            ),
        ),
    )
