"""Strict, versioned protocol for retrieval/reranking experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from edumind.common.artifacts import stable_hash
from experiments.benchmarks.common.protocol import (
    ExecutionProfile,
    ProtocolMetadata,
    boolean,
    choice,
    execution_profile,
    integer,
    mapping,
    number,
    plain,
    sequence,
    strict_object,
    string,
    validate_execution as validate_protocol_execution,
)
from .profiles import RERANKER_MODELS


DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("protocol.yaml")


@dataclass(frozen=True)
class RetrievalProtocol:
    schema_version: int
    checksum: str
    version: str
    resolved: Mapping[str, object]
    seed: int
    smoke_expected_chunk_count: int
    pool_size: int
    evaluation_tokenizer: str
    evidence_coverage_rule: str
    bm25_variant: str
    bm25_tokenizer: str
    bm25_k1: float
    bm25_b: float
    bm25_epsilon: float
    dense_similarity: str
    dense_normalized: bool
    embedding_batch_size: int
    rrf_source_depth: int
    rrf_constant: int
    rrf_weights: tuple[float, float]
    reranker_batch_size: int
    reject_truncation: bool
    reranker_input_template: str
    reranker_maximum_tokens: Mapping[str, int]
    quality_cutoffs: tuple[int, ...]
    alpha_ndcg_alpha: float
    minimum_latency_ci_documents: int
    confidence_level: float
    maximum_finalists: int
    profiles: Mapping[str, ExecutionProfile]
    authoritative_peak_vram_mb: float

    def metadata(self, source_path: Path = DEFAULT_PROTOCOL_PATH) -> ProtocolMetadata:
        return ProtocolMetadata(
            name="retrieval",
            source_path=source_path.resolve(),
            schema_version=self.schema_version,
            version=self.version,
            checksum=self.checksum,
            resolved=self.resolved,
            seed=self.seed,
            profiles=self.profiles,
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

    @property
    def pool_recall_metric(self) -> str:
        return f"diagnostic.pool_evidence_unit_recall_at_{self.pool_size}"

    def profile(self, name: str) -> ExecutionProfile:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise ValueError(f"Retrieval protocol has no profile {name!r}") from exc

    def maximum_tokens(self, reranker_alias: str) -> int:
        try:
            return self.reranker_maximum_tokens[reranker_alias]
        except KeyError as exc:
            raise ValueError(
                f"Retrieval protocol has no token limit for {reranker_alias!r}"
            ) from exc

    def validate_execution(
        self,
        profile_name: str,
        *,
        seed: int,
        warmups: int,
        repetitions: int,
        bootstrap_resamples: int,
        device: str,
        dtype: str,
    ) -> None:
        profile = self.profile(profile_name)
        validate_protocol_execution(
            self.metadata(),
            profile_name,
            seed=seed,
            warmups=warmups,
            repetitions=repetitions,
            bootstrap_resamples=bootstrap_resamples,
            device=device,
            dtype=dtype,
            batch_size=profile.batch_size,
        )


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> RetrievalProtocol:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return protocol_from_mapping(payload)


def protocol_from_settings(settings: Mapping[str, object]) -> RetrievalProtocol:
    payload = _mapping(settings.get("retrieval_protocol"), "retrieval_protocol")
    protocol = protocol_from_mapping(payload.get("resolved"))
    protocol.metadata().validate_worker_payload(payload)
    return protocol


def protocol_from_mapping(value: object) -> RetrievalProtocol:
    root = _object(
        value,
        "root",
        {
            "schema_version",
            "protocol_version",
            "seed",
            "smoke_fixture",
            "retrieval",
            "reranking",
            "quality",
            "statistics",
            "selection",
            "profiles",
            "resources",
        },
    )
    schema_version = _integer(root["schema_version"], "schema_version")
    if schema_version != 1:
        raise ValueError("Retrieval protocol must use schema_version 1")
    version = _string(root["protocol_version"], "protocol_version")
    seed = _integer(root["seed"], "seed", minimum=0)
    smoke = _object(
        root["smoke_fixture"],
        "smoke_fixture",
        {"expected_chunk_count"},
    )
    smoke_chunk_count = _integer(
        smoke["expected_chunk_count"],
        "smoke_fixture.expected_chunk_count",
        minimum=1,
    )

    retrieval = _object(
        root["retrieval"],
        "retrieval",
        {
            "pool_size",
            "evaluation_tokenizer",
            "evidence_coverage_rule",
            "bm25",
            "dense",
            "rrf",
        },
    )
    pool_size = _integer(retrieval["pool_size"], "retrieval.pool_size", minimum=1)
    evaluation_tokenizer = _choice(
        retrieval["evaluation_tokenizer"],
        "retrieval.evaluation_tokenizer",
        {"tiktoken:cl100k_base"},
    )
    coverage_rule = _choice(
        retrieval["evidence_coverage_rule"],
        "retrieval.evidence_coverage_rule",
        {"single-chunk-complete-unit-v1"},
    )

    bm25 = _object(
        retrieval["bm25"],
        "retrieval.bm25",
        {"variant", "tokenizer", "k1", "b", "epsilon"},
    )
    bm25_variant = _choice(
        bm25["variant"], "retrieval.bm25.variant", {"rank_bm25.BM25Okapi"}
    )
    bm25_tokenizer = _choice(
        bm25["tokenizer"],
        "retrieval.bm25.tokenizer",
        {"unicode-word-casefold-v1"},
    )
    bm25_k1 = _number(bm25["k1"], "retrieval.bm25.k1", minimum=0, exclusive=True)
    bm25_b = _number(bm25["b"], "retrieval.bm25.b", minimum=0, maximum=1)
    bm25_epsilon = _number(
        bm25["epsilon"], "retrieval.bm25.epsilon", minimum=0
    )

    dense = _object(
        retrieval["dense"],
        "retrieval.dense",
        {"similarity", "normalized", "embedding_batch_size"},
    )
    dense_similarity = _choice(
        dense["similarity"], "retrieval.dense.similarity", {"exact-cosine"}
    )
    dense_normalized = _boolean(dense["normalized"], "retrieval.dense.normalized")
    if not dense_normalized:
        raise ValueError("Retrieval dense vectors must be normalized")
    embedding_batch_size = _integer(
        dense["embedding_batch_size"],
        "retrieval.dense.embedding_batch_size",
        minimum=1,
    )

    rrf = _object(
        retrieval["rrf"],
        "retrieval.rrf",
        {"source_depth", "constant", "weights"},
    )
    rrf_source_depth = _integer(
        rrf["source_depth"], "retrieval.rrf.source_depth", minimum=1
    )
    if rrf_source_depth != pool_size:
        raise ValueError("RRF source_depth must equal retrieval.pool_size")
    rrf_constant = _integer(rrf["constant"], "retrieval.rrf.constant", minimum=0)
    raw_weights = _sequence(rrf["weights"], "retrieval.rrf.weights")
    if len(raw_weights) != 2:
        raise ValueError("RRF requires exactly dense and BM25 weights")
    rrf_weights = tuple(
        _number(weight, f"retrieval.rrf.weights[{index}]", minimum=0, exclusive=True)
        for index, weight in enumerate(raw_weights)
    )

    reranking = _object(
        root["reranking"],
        "reranking",
        {
            "batch_size",
            "reject_truncation",
            "input_template",
            "maximum_input_tokens",
        },
    )
    reranker_batch_size = _integer(
        reranking["batch_size"], "reranking.batch_size", minimum=1
    )
    reject_truncation = _boolean(
        reranking["reject_truncation"], "reranking.reject_truncation"
    )
    if not reject_truncation:
        raise ValueError("Retrieval benchmark protocol must reject truncation")
    input_template = _choice(
        reranking["input_template"],
        "reranking.input_template",
        {"tokenizer(query, passage)"},
    )
    raw_limits = _mapping(
        reranking["maximum_input_tokens"], "reranking.maximum_input_tokens"
    )
    expected_aliases = set(RERANKER_MODELS)
    if set(raw_limits) != expected_aliases:
        raise ValueError(
            "reranking.maximum_input_tokens must define exactly: "
            + ", ".join(sorted(expected_aliases))
        )
    maximum_tokens = {
        alias: _integer(
            raw_limits[alias],
            f"reranking.maximum_input_tokens.{alias}",
            minimum=1,
        )
        for alias in sorted(expected_aliases)
    }

    quality = _object(
        root["quality"], "quality", {"cutoffs", "alpha_ndcg_alpha"}
    )
    cutoffs = tuple(
        _integer(value, f"quality.cutoffs[{index}]", minimum=1)
        for index, value in enumerate(_sequence(quality["cutoffs"], "quality.cutoffs"))
    )
    if not cutoffs or tuple(sorted(set(cutoffs))) != cutoffs:
        raise ValueError("quality.cutoffs must be non-empty, unique, and increasing")
    if max(cutoffs) > pool_size:
        raise ValueError("quality.cutoffs cannot exceed retrieval.pool_size")
    alpha = _number(
        quality["alpha_ndcg_alpha"],
        "quality.alpha_ndcg_alpha",
        minimum=0,
        maximum=1,
        maximum_exclusive=True,
    )

    statistics = _object(
        root["statistics"],
        "statistics",
        {"minimum_latency_ci_documents", "confidence_level"},
    )
    minimum_documents = _integer(
        statistics["minimum_latency_ci_documents"],
        "statistics.minimum_latency_ci_documents",
        minimum=1,
    )
    confidence = _number(
        statistics["confidence_level"],
        "statistics.confidence_level",
        minimum=0,
        maximum=1,
        exclusive=True,
        maximum_exclusive=True,
    )

    selection = _object(
        root["selection"], "selection", {"maximum_finalists"}
    )
    maximum_finalists = _integer(
        selection["maximum_finalists"],
        "selection.maximum_finalists",
        minimum=1,
    )

    raw_profiles = _object(
        root["profiles"], "profiles", {"smoke", "development", "validation"}
    )
    profiles = {
        name: _execution_profile(name, raw_profiles[name])
        for name in ("smoke", "development", "validation")
    }
    if {profile.batch_size for profile in profiles.values()} != {
        embedding_batch_size
    } or embedding_batch_size != reranker_batch_size:
        raise ValueError(
            "Retrieval profile, embedding, and reranker batch sizes must agree"
        )
    resources = _object(
        root["resources"], "resources", {"authoritative_peak_vram_mb"}
    )
    peak_vram = _number(
        resources["authoritative_peak_vram_mb"],
        "resources.authoritative_peak_vram_mb",
        minimum=0,
        exclusive=True,
    )
    resolved = _plain(root)
    return RetrievalProtocol(
        schema_version=schema_version,
        checksum=stable_hash(resolved),
        version=version,
        resolved=resolved,
        seed=seed,
        smoke_expected_chunk_count=smoke_chunk_count,
        pool_size=pool_size,
        evaluation_tokenizer=evaluation_tokenizer,
        evidence_coverage_rule=coverage_rule,
        bm25_variant=bm25_variant,
        bm25_tokenizer=bm25_tokenizer,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        bm25_epsilon=bm25_epsilon,
        dense_similarity=dense_similarity,
        dense_normalized=dense_normalized,
        embedding_batch_size=embedding_batch_size,
        rrf_source_depth=rrf_source_depth,
        rrf_constant=rrf_constant,
        rrf_weights=(rrf_weights[0], rrf_weights[1]),
        reranker_batch_size=reranker_batch_size,
        reject_truncation=reject_truncation,
        reranker_input_template=input_template,
        reranker_maximum_tokens=maximum_tokens,
        quality_cutoffs=cutoffs,
        alpha_ndcg_alpha=alpha,
        minimum_latency_ci_documents=minimum_documents,
        confidence_level=confidence,
        maximum_finalists=maximum_finalists,
        profiles=profiles,
        authoritative_peak_vram_mb=peak_vram,
    )


def _execution_profile(name: str, value: object) -> ExecutionProfile:
    return execution_profile(value, f"profiles.{name}")


def _object(value: object, label: str, fields: set[str]) -> dict[str, object]:
    return strict_object(value, f"Retrieval protocol {label}", fields)


def _mapping(value: object, label: str) -> dict[str, object]:
    return mapping(value, f"Retrieval protocol {label}")


def _sequence(value: object, label: str) -> list[object]:
    return sequence(value, f"Retrieval protocol {label}")


def _string(value: object, label: str) -> str:
    return string(value, f"Retrieval protocol {label}")


def _choice(value: object, label: str, choices: set[str]) -> str:
    return choice(value, f"Retrieval protocol {label}", choices)


def _boolean(value: object, label: str) -> bool:
    return boolean(value, f"Retrieval protocol {label}")


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    return integer(value, f"Retrieval protocol {label}", minimum=minimum)


def _number(
    value: object,
    label: str,
    *,
    minimum: float,
    maximum: float | None = None,
    exclusive: bool = False,
    maximum_exclusive: bool = False,
) -> float:
    return number(
        value,
        f"Retrieval protocol {label}",
        minimum=minimum,
        maximum=maximum,
        minimum_exclusive=exclusive,
        maximum_exclusive=maximum_exclusive,
    )


def _plain(value: object) -> object:
    return plain(value)
