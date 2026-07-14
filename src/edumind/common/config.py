"""Typed, packaged configuration for EduMind."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when configuration is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class ExtractionSettings:
    default_profile: str = "balanced"
    normalization_profile: str = "conservative"
    cache_enabled: bool = True
    cache_directory: Path = Path("artifacts/extraction/cache")
    maximum_upload_bytes: int = 100 * 1024 * 1024
    model_lock_path: Path = Path("data/benchmarks/models/extraction.json")


@dataclass(frozen=True)
class EmbeddingSettings:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
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
    backend: str = "chroma"
    collection_name: str = "edumind_documents"
    persist_directory: Path = Path("artifacts/rag/vector_store")
    distance_metric: str = "cosine"


@dataclass(frozen=True)
class RetrievalSettings:
    strategy: str = "rrf"
    reranker_revision: str | None = None
    reranker_device: str = "cpu"
    top_k: int = 5
    candidate_k: int = 20
    context_token_budget: int = 2048
    rrf_k: int = 60


@dataclass(frozen=True)
class GenerationSettings:
    model_name: str = "qwen3:1.7b"
    digest: str = "unpinned"
    base_url: str = "http://127.0.0.1:11434"
    thinking: str = "off"
    temperature: float = 0.0
    seed: int = 42
    context_tokens: int = 8192
    maximum_answer_tokens: int = 256
    timeout_seconds: int = 120
    keep_alive: str | int = "5m"
    model_lock_path: Path = Path("data/benchmarks/models/ollama.json")


@dataclass(frozen=True)
class BenchmarkSettings:
    profile: str = "smoke"
    seed: int = 42
    bootstrap_resamples: int = 10_000
    tracking_uri: str | None = None
    artifact_directory: Path = Path("artifacts/benchmarks")


@dataclass(frozen=True)
class ServiceSettings:
    host: str = "127.0.0.1"
    extraction_port: int = 8000
    rag_port: int = 8001


@dataclass(frozen=True)
class Settings:
    extraction: ExtractionSettings = field(default_factory=ExtractionSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    vector: VectorSettings = field(default_factory=VectorSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    service: ServiceSettings = field(default_factory=ServiceSettings)
    logging_level: str = "INFO"


def default_config_path() -> Path:
    """Return the packaged configuration resource as a filesystem path."""
    return Path(str(files("edumind").joinpath("defaults.yaml")))


def load_yaml_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load packaged defaults and merge one optional local YAML file."""
    load_dotenv(override=False)
    defaults = _read_yaml(default_config_path())
    selected = config_path or os.getenv("EDUMIND_CONFIG")
    if selected:
        path = Path(selected).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            raise ConfigurationError(f"Configuration file does not exist: {path}")
        _deep_merge(defaults, _read_yaml(path))
    return defaults


def load_settings(
    config_path: str | Path | None = None,
    *,
    overrides: Mapping[str, object] | None = None,
) -> Settings:
    """Load, merge, validate, and return the complete runtime settings."""
    raw = load_yaml_config(config_path)
    if overrides:
        _deep_merge(raw, deepcopy(dict(overrides)))
    _apply_environment(raw)
    return _build_settings(raw)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {path}")
    return dict(payload)


def _deep_merge(target: dict[str, Any], update: Mapping[str, object]) -> None:
    for key, value in update.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _deep_merge(current, value)
        else:
            target[key] = deepcopy(value)


def _apply_environment(raw: dict[str, Any]) -> None:
    mapping = {
        "EDUMIND_ARTIFACTS": ("benchmark", "artifact_directory", str),
        "EDUMIND_EXTRACTION_CACHE": ("extraction", "cache_directory", str),
        "EDUMIND_EXTRACTION_MODEL_LOCK": ("extraction", "model_lock_path", str),
        "EDUMIND_OLLAMA_URL": ("generation", "base_url", str),
        "EDUMIND_OLLAMA_MODEL": ("generation", "model_name", str),
        "EDUMIND_OLLAMA_MODEL_LOCK": ("generation", "model_lock_path", str),
        "EDUMIND_MLFLOW_TRACKING_URI": ("benchmark", "tracking_uri", str),
        "EDUMIND_LOG_LEVEL": ("logging", "level", str),
    }
    for env_name, (section, key, converter) in mapping.items():
        value = os.getenv(env_name)
        if value is not None:
            raw.setdefault(section, {})[key] = converter(value)


def _section(raw: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Configuration section '{name}' must be a mapping")
    return value


def _reject_unknown(section: Mapping[str, object], name: str, allowed: set[str]) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ConfigurationError(
            f"Unknown keys in configuration section '{name}': {', '.join(sorted(unknown))}"
        )


def _required_int(section: Mapping[str, object], key: str, *, minimum: int = 1) -> int:
    value = section.get(key)
    if isinstance(value, bool):
        raise ConfigurationError(f"'{key}' must be an integer, not a boolean")
    if not isinstance(value, (str, int, float)):
        raise ConfigurationError(f"'{key}' must be an integer (received {value!r})")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"'{key}' must be an integer (received {value!r})") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ConfigurationError(f"'{key}' must be an integer (received {value!r})")
    if result < minimum:
        raise ConfigurationError(f"'{key}' must be >= {minimum} (received {result})")
    return result


def _required_float(section: Mapping[str, object], key: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ConfigurationError(f"'{key}' must be numeric (received {value!r})")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"'{key}' must be numeric (received {value!r})") from exc
    if not math.isfinite(result):
        raise ConfigurationError(f"'{key}' must be finite (received {value!r})")
    return result


def _required_bool(section: Mapping[str, object], key: str) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise ConfigurationError(f"'{key}' must be a boolean (received {value!r})")
    return value


def _path(value: object, *, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigurationError(f"'{name}' must be a non-empty path")
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (Path.cwd() / candidate).resolve()


def _build_settings(raw: Mapping[str, object]) -> Settings:
    _reject_unknown(
        raw,
        "root",
        {
            "extraction",
            "embedding",
            "chunking",
            "vector",
            "retrieval",
            "generation",
            "benchmark",
            "service",
            "logging",
        },
    )
    extraction = _section(raw, "extraction")
    embedding = _section(raw, "embedding")
    chunking = _section(raw, "chunking")
    vector = _section(raw, "vector")
    retrieval = _section(raw, "retrieval")
    generation = _section(raw, "generation")
    benchmark = _section(raw, "benchmark")
    service = _section(raw, "service")
    logging = _section(raw, "logging")

    _reject_unknown(
        extraction,
        "extraction",
        {
            "default_profile",
            "normalization_profile",
            "cache_enabled",
            "cache_directory",
            "maximum_upload_bytes",
            "model_lock_path",
        },
    )
    _reject_unknown(
        embedding,
        "embedding",
        {
            "model_name",
            "revision",
            "dimension",
            "indexing_device",
            "query_device",
            "query_prefix",
            "document_prefix",
            "normalize",
            "similarity",
            "maximum_length",
            "batch_size",
        },
    )
    _reject_unknown(
        chunking,
        "chunking",
        {"strategy", "chunk_size", "chunk_overlap", "tokenizer"},
    )
    _reject_unknown(
        vector,
        "vector",
        {"backend", "collection_name", "persist_directory", "distance_metric"},
    )
    _reject_unknown(
        retrieval,
        "retrieval",
        {
            "strategy",
            "reranker_revision",
            "reranker_device",
            "top_k",
            "candidate_k",
            "context_token_budget",
            "rrf_k",
        },
    )
    _reject_unknown(
        generation,
        "generation",
        {
            "model_name",
            "digest",
            "base_url",
            "thinking",
            "temperature",
            "seed",
            "context_tokens",
            "maximum_answer_tokens",
            "timeout_seconds",
            "keep_alive",
            "model_lock_path",
        },
    )
    _reject_unknown(
        benchmark,
        "benchmark",
        {"profile", "seed", "bootstrap_resamples", "tracking_uri", "artifact_directory"},
    )
    _reject_unknown(service, "service", {"host", "extraction_port", "rag_port"})
    _reject_unknown(logging, "logging", {"level"})

    chunk_size = _required_int(chunking, "chunk_size")
    chunk_overlap = _required_int(chunking, "chunk_overlap", minimum=0)
    embedding_maximum_length = _required_int(embedding, "maximum_length")
    if chunk_overlap >= chunk_size:
        raise ConfigurationError("chunk_overlap must be smaller than chunk_size")
    if chunk_size > embedding_maximum_length:
        raise ConfigurationError("chunking.chunk_size must not exceed embedding.maximum_length")
    top_k = _required_int(retrieval, "top_k")
    candidate_k = _required_int(retrieval, "candidate_k")
    if candidate_k < top_k:
        raise ConfigurationError("retrieval.candidate_k must be >= retrieval.top_k")
    similarity = str(embedding.get("similarity", "cosine"))
    if similarity not in {"cosine", "dot"}:
        raise ConfigurationError("embedding.similarity must be 'cosine' or 'dot'")
    distance_metric = str(vector.get("distance_metric", "cosine"))
    if distance_metric != similarity:
        raise ConfigurationError("vector.distance_metric must match embedding.similarity")
    indexing_device = str(embedding.get("indexing_device", "cpu"))
    query_device = str(embedding.get("query_device", "cpu"))
    if indexing_device not in {"cpu", "cuda"} or query_device not in {"cpu", "cuda"}:
        raise ConfigurationError("embedding devices must be 'cpu' or 'cuda'")
    host = str(service.get("host", "127.0.0.1"))
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigurationError("service.host must be loopback for local single-user deployment")
    normalization = str(extraction.get("normalization_profile", "conservative"))
    if normalization not in {"minimal", "conservative", "aggressive"}:
        raise ConfigurationError("extraction.normalization_profile is unsupported")
    chunking_strategy = str(chunking.get("strategy", "token"))
    if chunking_strategy not in {
        "token",
        "token-256-32",
        "token-384-64",
        "recursive-character",
        "sentence-8-2",
        "semantic",
    }:
        raise ConfigurationError("chunking.strategy is unsupported for production")
    tokenizer_name = str(chunking.get("tokenizer", "embedding"))
    if tokenizer_name != "embedding":
        raise ConfigurationError(
            "chunking.tokenizer must be 'embedding' so runtime and benchmarks share offsets"
        )
    vector_backend = str(vector.get("backend", "chroma"))
    if vector_backend != "chroma":
        raise ConfigurationError(
            "vector.backend must remain 'chroma' until a benchmark recommendation promotes another"
        )
    retrieval_strategy = str(retrieval.get("strategy", "rrf"))
    reranking_strategies = {"rrf-minilm-reranker", "rrf-qwen3-reranker"}
    if retrieval_strategy not in {"dense", "bm25", "rrf", *reranking_strategies}:
        raise ConfigurationError("retrieval.strategy is unsupported for production")
    reranker_revision_value = retrieval.get("reranker_revision")
    reranker_revision = (
        str(reranker_revision_value).strip() if reranker_revision_value is not None else None
    )
    if retrieval_strategy in reranking_strategies and reranker_revision in {
        None,
        "",
        "main",
        "unpinned",
    }:
        raise ConfigurationError(
            "retrieval.reranker_revision must be an immutable revision for a reranker stack"
        )
    if retrieval_strategy not in reranking_strategies and reranker_revision:
        raise ConfigurationError(
            "retrieval.reranker_revision is only valid for a reranker retrieval strategy"
        )
    reranker_device = str(retrieval.get("reranker_device", "cpu"))
    if reranker_device not in {"cpu", "cuda"}:
        raise ConfigurationError("retrieval.reranker_device must be 'cpu' or 'cuda'")
    thinking = str(generation.get("thinking", "off"))
    if thinking not in {"off", "on", "low", "medium"}:
        raise ConfigurationError("generation.thinking must be off, on, low, or medium")
    temperature = _required_float(generation, "temperature")
    if temperature < 0 or temperature > 2:
        raise ConfigurationError("generation.temperature must be in [0, 2]")
    base_url = str(generation.get("base_url", ""))
    parsed_ollama = urlparse(base_url)
    if parsed_ollama.scheme not in {"http", "https"} or parsed_ollama.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ConfigurationError("generation.base_url must be a loopback HTTP(S) URL")
    context_tokens = _required_int(generation, "context_tokens")
    maximum_answer_tokens = _required_int(generation, "maximum_answer_tokens")
    context_token_budget = _required_int(retrieval, "context_token_budget")
    if context_token_budget + maximum_answer_tokens >= context_tokens:
        raise ConfigurationError(
            "retrieval context budget plus maximum answer tokens must be below model context"
        )
    extraction_port = _required_int(service, "extraction_port")
    rag_port = _required_int(service, "rag_port")
    if max(extraction_port, rag_port) > 65535 or extraction_port == rag_port:
        raise ConfigurationError("service ports must be distinct values in [1, 65535]")
    benchmark_profile = str(benchmark.get("profile", "smoke"))
    if benchmark_profile not in {"smoke", "standard", "full"}:
        raise ConfigurationError("benchmark.profile must be smoke, standard, or full")
    keep_alive = generation.get("keep_alive", "5m")
    valid_duration = isinstance(keep_alive, str) and re.fullmatch(r"\d+(?:ms|s|m|h)", keep_alive)
    if isinstance(keep_alive, bool) or not (
        isinstance(keep_alive, int) and keep_alive >= 0 or valid_duration
    ):
        raise ConfigurationError(
            "generation.keep_alive must be a non-negative integer or duration such as '5m'"
        )

    return Settings(
        extraction=ExtractionSettings(
            default_profile=str(extraction.get("default_profile", "balanced")),
            normalization_profile=normalization,
            cache_enabled=_required_bool(extraction, "cache_enabled"),
            cache_directory=_path(extraction.get("cache_directory"), name="cache_directory"),
            maximum_upload_bytes=_required_int(extraction, "maximum_upload_bytes"),
            model_lock_path=_path(extraction.get("model_lock_path"), name="model_lock_path"),
        ),
        embedding=EmbeddingSettings(
            model_name=str(embedding.get("model_name")),
            revision=str(embedding.get("revision", "main")),
            dimension=_required_int(embedding, "dimension"),
            indexing_device=indexing_device,
            query_device=query_device,
            query_prefix=str(embedding.get("query_prefix", "")),
            document_prefix=str(embedding.get("document_prefix", "")),
            normalize=_required_bool(embedding, "normalize"),
            similarity=similarity,
            maximum_length=embedding_maximum_length,
            batch_size=_required_int(embedding, "batch_size"),
        ),
        chunking=ChunkingSettings(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer_name,
        ),
        vector=VectorSettings(
            backend=vector_backend,
            collection_name=str(vector.get("collection_name", "edumind_documents")),
            persist_directory=_path(vector.get("persist_directory"), name="persist_directory"),
            distance_metric=distance_metric,
        ),
        retrieval=RetrievalSettings(
            strategy=retrieval_strategy,
            reranker_revision=reranker_revision,
            reranker_device=reranker_device,
            top_k=top_k,
            candidate_k=candidate_k,
            context_token_budget=context_token_budget,
            rrf_k=_required_int(retrieval, "rrf_k"),
        ),
        generation=GenerationSettings(
            model_name=str(generation.get("model_name")),
            digest=str(generation.get("digest", "unpinned")),
            base_url=base_url,
            thinking=thinking,
            temperature=temperature,
            seed=_required_int(generation, "seed", minimum=0),
            context_tokens=context_tokens,
            maximum_answer_tokens=maximum_answer_tokens,
            timeout_seconds=_required_int(generation, "timeout_seconds"),
            keep_alive=cast(str | int, keep_alive),
            model_lock_path=_path(generation.get("model_lock_path"), name="model_lock_path"),
        ),
        benchmark=BenchmarkSettings(
            profile=benchmark_profile,
            seed=_required_int(benchmark, "seed", minimum=0),
            bootstrap_resamples=_required_int(benchmark, "bootstrap_resamples"),
            tracking_uri=(
                str(benchmark["tracking_uri"]) if benchmark.get("tracking_uri") else None
            ),
            artifact_directory=_path(
                benchmark.get("artifact_directory"), name="artifact_directory"
            ),
        ),
        service=ServiceSettings(
            host=host,
            extraction_port=extraction_port,
            rag_port=rag_port,
        ),
        logging_level=str(logging.get("level", "INFO")).upper(),
    )
