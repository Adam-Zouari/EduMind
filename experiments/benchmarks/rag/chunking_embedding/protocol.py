"""Strict protocol for chunking/embedding pair selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from edumind.rag.contracts import EMBEDDING_SPECS
from experiments.benchmarks.common.protocol import (
    PreflightSettings,
    ProtocolMetadata,
    boolean,
    choice,
    execution_profiles,
    increasing_integers,
    integer,
    load_yaml,
    mapping,
    metadata,
    number,
    preflight_settings,
    sequence,
    strict_object,
    string,
)
from experiments.benchmarks.rag.chunking_embedding.profiles import (
    EXPERIMENTAL_EMBEDDING_SPECS,
)

DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("protocol.yaml")
_STRATEGY_KINDS = {
    "recursive-character": "recursive-character",
    "token-256-32": "token",
    "token-384-64": "token",
    "token-512-64": "token",
    "sentence-8-2": "sentence",
    "semantic": "semantic",
    "section-aware-512-64": "section-aware",
    "structure-aware-512-64": "structure-aware",
}


@dataclass(frozen=True)
class ChunkingEmbeddingProtocol:
    meta: ProtocolMetadata
    preflight: PreflightSettings
    tokenizer: str
    strategies: Mapping[str, Mapping[str, object]]
    embedding_models: tuple[str, ...]
    smoke_pair: str
    embedding_batch_size: int
    similarity: str
    stored_vector_dtype: str
    reject_truncation: bool
    model_config: Mapping[str, Mapping[str, bool]]
    quality_cutoffs: tuple[int, ...]
    audit_depth: int
    alpha_ndcg_alpha: float
    repeatability_depth: int
    minimum_latency_ci_documents: int
    minimum_quality_ci_documents: int | None
    minimum_repeatability_ci_documents: int | None
    data_requirements: Mapping[str, object]
    confidence_level: float
    maximum_finalists: int
    authoritative_peak_vram_mb: float

    def profile(self, name: str):
        return self.meta.profile(name)

    @property
    def unfrozen_ci_policies(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, minimum in (
                ("quality", self.minimum_quality_ci_documents),
                ("ranking_repeatability", self.minimum_repeatability_ci_documents),
            )
            if minimum is None
        )

    @property
    def primary_quality_metrics(self) -> tuple[str, ...]:
        return tuple(
            f"{name}_at_{cutoff}"
            for name in ("ndcg", "evidence_unit_recall", "evidence_token_precision")
            for cutoff in self.quality_cutoffs
        )

    @property
    def alpha_ndcg_metrics(self) -> tuple[str, ...]:
        return tuple(f"alpha_ndcg_at_{cutoff}" for cutoff in self.quality_cutoffs)

    @property
    def quality_metrics(self) -> tuple[str, ...]:
        return (*self.primary_quality_metrics, *self.alpha_ndcg_metrics)

    def strategy(self, alias: str) -> Mapping[str, object]:
        try:
            return self.strategies[alias]
        except KeyError as exc:
            raise ValueError(f"Unknown chunking strategy: {alias}") from exc

    def model_config_options(self, name: str) -> tuple[tuple[str, bool], ...]:
        return tuple(sorted(self.model_config.get(name, {}).items()))

    @property
    def development_candidates(self) -> tuple[str, ...]:
        return tuple(
            f"{strategy}|{model}"
            for strategy, model in product(self.strategies, self.embedding_models)
        )


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> ChunkingEmbeddingProtocol:
    return protocol_from_mapping(
        load_yaml(path, "chunking-embedding"), source_path=path
    )


def protocol_from_mapping(
    value: object, *, source_path: Path = DEFAULT_PROTOCOL_PATH
) -> ChunkingEmbeddingProtocol:
    root = strict_object(
        value,
        "chunking-embedding protocol root",
        {
            "schema_version",
            "protocol_version",
            "seed",
            "tokenizer",
            "strategies",
            "embedding_models",
            "smoke_pair",
            "index",
            "quality",
            "repeatability",
            "statistics",
            "data",
            "selection",
            "resources",
            "preflight",
            "profiles",
        },
    )
    tokenizer = choice(root["tokenizer"], "tokenizer", {"tiktoken:cl100k_base"})
    strategies = {
        alias: _strategy(alias, value)
        for alias, value in mapping(root["strategies"], "strategies").items()
    }
    if set(strategies) != set(_STRATEGY_KINDS):
        raise ValueError("strategies must declare all eight supported chunkers")
    models = tuple(
        string(item, "embedding_models item")
        for item in sequence(root["embedding_models"], "embedding_models")
    )
    if not models or len(set(models)) != len(models):
        raise ValueError("embedding_models must be nonempty and unique")
    if set(models) != set(EMBEDDING_SPECS) | set(EXPERIMENTAL_EMBEDDING_SPECS):
        raise ValueError("embedding_models must match supported embedding contracts")
    smoke_pair = string(root["smoke_pair"], "smoke_pair")
    pair_parts = smoke_pair.split("|", 1)
    if (
        len(pair_parts) != 2
        or pair_parts[0] not in strategies
        or pair_parts[1] not in models
    ):
        raise ValueError("smoke_pair must use a declared strategy and embedding model")
    index = strict_object(
        root["index"],
        "index",
        {
            "embedding_batch_size",
            "similarity",
            "normalize_vectors",
            "stored_vector_dtype",
            "reject_truncation",
            "model_config",
        },
    )
    batch_size = integer(
        index["embedding_batch_size"], "index.embedding_batch_size", minimum=1
    )
    similarity = choice(index["similarity"], "index.similarity", {"exact-cosine"})
    normalized = boolean(index["normalize_vectors"], "index.normalize_vectors")
    vector_dtype = choice(
        index["stored_vector_dtype"], "index.stored_vector_dtype", {"float32"}
    )
    reject_truncation = boolean(index["reject_truncation"], "index.reject_truncation")
    model_config = {}
    for name, configuration in mapping(
        index["model_config"], "index.model_config"
    ).items():
        if name != "Snowflake/snowflake-arctic-embed-m-v2.0":
            raise ValueError(f"No audited configuration overrides for {name}")
        options = strict_object(
            configuration,
            f"index.model_config.{name}",
            {"use_memory_efficient_attention", "unpad_inputs"},
        )
        model_config[name] = {
            key: boolean(value, f"index.model_config.{name}.{key}")
            for key, value in options.items()
        }
        if (
            model_config[name]["unpad_inputs"]
            and not model_config[name]["use_memory_efficient_attention"]
        ):
            raise ValueError("Snowflake unpadding requires memory-efficient attention")
    if not model_config:
        raise ValueError("Snowflake attention and padding settings must be explicit")
    if not normalized or not reject_truncation:
        raise ValueError(
            "Chunking/embedding requires normalized vectors and truncation rejection"
        )
    quality = strict_object(
        root["quality"], "quality", {"cutoffs", "audit_depth", "alpha_ndcg_alpha"}
    )
    cutoffs = increasing_integers(quality["cutoffs"], "quality.cutoffs")
    audit_depth = integer(
        quality["audit_depth"], "quality.audit_depth", minimum=max(cutoffs)
    )
    alpha = number(
        quality["alpha_ndcg_alpha"],
        "quality.alpha_ndcg_alpha",
        minimum=0,
        maximum=1,
        maximum_exclusive=True,
    )
    statistics = strict_object(
        root["statistics"],
        "statistics",
        {
            "minimum_latency_ci_documents",
            "minimum_quality_ci_documents",
            "minimum_repeatability_ci_documents",
            "confidence_level",
        },
    )
    minimum_documents = integer(
        statistics["minimum_latency_ci_documents"],
        "statistics.minimum_latency_ci_documents",
        minimum=2,
    )
    quality_support = statistics["minimum_quality_ci_documents"]
    if quality_support is not None:
        quality_support = integer(
            quality_support, "statistics.minimum_quality_ci_documents", minimum=2
        )
    repeatability = strict_object(root["repeatability"], "repeatability", {"depth"})
    repeatability_depth = integer(
        repeatability["depth"], "repeatability.depth", minimum=1
    )
    if repeatability_depth > audit_depth:
        raise ValueError("repeatability.depth must not exceed quality.audit_depth")
    repeatability_support = statistics["minimum_repeatability_ci_documents"]
    if repeatability_support is not None:
        repeatability_support = integer(
            repeatability_support,
            "statistics.minimum_repeatability_ci_documents",
            minimum=2,
        )
    data = strict_object(
        root["data"], "data", {"qasper_papers", "minimum_structured_questions"}
    )
    counts = strict_object(
        data["qasper_papers"],
        "data.qasper_papers",
        {"development", "validation", "locked"},
    )
    slices = strict_object(
        data["minimum_structured_questions"],
        "data.minimum_structured_questions",
        {"table", "formula", "mixed"},
    )
    data = {
        "qasper_papers": {
            name: integer(count, f"data.qasper_papers.{name}", minimum=1)
            for name, count in counts.items()
        },
        "minimum_structured_questions": {
            name: integer(count, f"data.minimum_structured_questions.{name}", minimum=1)
            for name, count in slices.items()
        },
    }
    confidence = number(
        statistics["confidence_level"],
        "statistics.confidence_level",
        minimum=0,
        maximum=1,
        minimum_exclusive=True,
        maximum_exclusive=True,
    )
    selection = strict_object(root["selection"], "selection", {"maximum_finalists"})
    maximum_finalists = integer(
        selection["maximum_finalists"], "selection.maximum_finalists", minimum=1
    )
    resources = strict_object(
        root["resources"], "resources", {"authoritative_peak_vram_mb"}
    )
    peak = number(
        resources["authoritative_peak_vram_mb"],
        "resources.authoritative_peak_vram_mb",
        minimum=0,
        minimum_exclusive=True,
    )
    profiles = execution_profiles(
        root["profiles"], names=("smoke", "development", "validation", "locked")
    )
    preflight = preflight_settings(root["preflight"])
    if {profile.batch_size for profile in profiles.values()} != {batch_size}:
        raise ValueError(
            "Chunking/embedding profile and embedding batch sizes must agree"
        )
    return ChunkingEmbeddingProtocol(
        metadata("chunking_embedding", source_path, root, profiles=profiles),
        preflight,
        tokenizer,
        strategies,
        models,
        smoke_pair,
        batch_size,
        similarity,
        vector_dtype,
        reject_truncation,
        model_config,
        cutoffs,
        audit_depth,
        alpha,
        repeatability_depth,
        minimum_documents,
        quality_support,
        repeatability_support,
        data,
        confidence,
        maximum_finalists,
        peak,
    )


def protocol_from_settings(settings: Mapping[str, object]) -> ChunkingEmbeddingProtocol:
    payload = mapping(
        settings.get("chunking_embedding_protocol"), "chunking_embedding_protocol"
    )
    protocol = protocol_from_mapping(payload.get("resolved"))
    protocol.meta.validate_worker_payload(payload)
    return protocol


def _strategy(alias: str, value: object) -> dict[str, object]:
    raw = mapping(value, f"strategies.{alias}")
    kind = choice(
        raw.get("kind"),
        f"strategies.{alias}.kind",
        {
            "token",
            "sentence",
            "recursive-character",
            "semantic",
            "section-aware",
            "structure-aware",
        },
    )
    if _STRATEGY_KINDS.get(alias) != kind:
        raise ValueError(f"strategies.{alias}.kind contradicts its candidate alias")
    expected = {
        "token": {"kind", "size", "overlap"},
        "sentence": {"kind", "sentences", "overlap"},
        "recursive-character": {
            "kind",
            "size",
            "overlap",
            "separators",
            "minimum_boundary_ratio",
        },
        "semantic": {
            "kind",
            "maximum_tokens",
            "boundary_percentile",
            "boundary_embedding",
        },
        "section-aware": {"kind", "size", "overlap", "parser"},
        "structure-aware": {"kind", "size", "overlap", "parser"},
    }[kind]
    row = strict_object(raw, f"strategies.{alias}", expected)
    if "size" in row:
        size = integer(row["size"], f"strategies.{alias}.size", minimum=1)
        overlap = integer(row["overlap"], f"strategies.{alias}.overlap", minimum=0)
        if overlap >= size:
            raise ValueError(f"strategies.{alias}.overlap must be smaller than size")
    if "sentences" in row:
        sentences = integer(
            row["sentences"], f"strategies.{alias}.sentences", minimum=1
        )
        if (
            integer(row["overlap"], f"strategies.{alias}.overlap", minimum=0)
            >= sentences
        ):
            raise ValueError(
                f"strategies.{alias}.overlap must be smaller than sentences"
            )
    if "maximum_tokens" in row:
        integer(row["maximum_tokens"], f"strategies.{alias}.maximum_tokens", minimum=1)
        number(
            row["boundary_percentile"],
            f"strategies.{alias}.boundary_percentile",
            minimum=0,
            maximum=1,
        )
        choice(
            row["boundary_embedding"],
            f"strategies.{alias}.boundary_embedding",
            {"candidate-embedding"},
        )
    if "separators" in row:
        separators = sequence(row["separators"], f"strategies.{alias}.separators")
        if not separators or not all(
            isinstance(item, str) and item for item in separators
        ):
            raise ValueError(f"strategies.{alias}.separators must contain strings")
        number(
            row["minimum_boundary_ratio"],
            f"strategies.{alias}.minimum_boundary_ratio",
            minimum=0,
            maximum=1,
        )
    if "parser" in row:
        expected_parser = {
            "section-aware": "markdown-heading-v1",
            "structure-aware": "markdown-table-formula-v1",
        }[kind]
        choice(row["parser"], f"strategies.{alias}.parser", {expected_parser})
    return row
