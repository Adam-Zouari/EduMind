"""Strict, versioned protocol for retrieval/reranking experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from experiments.benchmarks.common.protocol import (
    ExecutionProfile,
    ProtocolMetadata,
    boolean,
    choice,
    execution_profiles,
    integer,
    load_yaml,
    mapping,
    metadata,
    number,
    sequence,
    strict_object,
)
from experiments.benchmarks.common.protocol import (
    validate_execution as validate_protocol_execution,
)

from .profiles import RERANKER_MODELS

DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("protocol.yaml")


@dataclass(frozen=True)
class RetrievalProtocol:
    meta: ProtocolMetadata
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
    authoritative_peak_vram_mb: float

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
        return self.meta.profile(name)

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
            self.meta,
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
    return protocol_from_mapping(load_yaml(path, "retrieval"), source_path=path)


def protocol_from_settings(settings: Mapping[str, object]) -> RetrievalProtocol:
    payload = mapping(settings.get("retrieval_protocol"), "retrieval_protocol")
    protocol = protocol_from_mapping(payload.get("resolved"))
    protocol.meta.validate_worker_payload(payload)
    return protocol


def protocol_from_mapping(
    value: object, *, source_path: Path = DEFAULT_PROTOCOL_PATH
) -> RetrievalProtocol:
    root = strict_object(
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
    smoke = strict_object(
        root["smoke_fixture"],
        "smoke_fixture",
        {"expected_chunk_count"},
    )
    smoke_chunk_count = integer(
        smoke["expected_chunk_count"],
        "smoke_fixture.expected_chunk_count",
        minimum=1,
    )

    retrieval = strict_object(
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
    pool_size = integer(retrieval["pool_size"], "retrieval.pool_size", minimum=1)
    evaluation_tokenizer = choice(
        retrieval["evaluation_tokenizer"],
        "retrieval.evaluation_tokenizer",
        {"tiktoken:cl100k_base"},
    )
    coverage_rule = choice(
        retrieval["evidence_coverage_rule"],
        "retrieval.evidence_coverage_rule",
        {"single-chunk-complete-unit-v1"},
    )

    bm25 = strict_object(
        retrieval["bm25"],
        "retrieval.bm25",
        {"variant", "tokenizer", "k1", "b", "epsilon"},
    )
    bm25_variant = choice(
        bm25["variant"], "retrieval.bm25.variant", {"rank_bm25.BM25Okapi"}
    )
    bm25_tokenizer = choice(
        bm25["tokenizer"],
        "retrieval.bm25.tokenizer",
        {"unicode-word-casefold-v1"},
    )
    bm25_k1 = number(bm25["k1"], "retrieval.bm25.k1", minimum=0, minimum_exclusive=True)
    bm25_b = number(bm25["b"], "retrieval.bm25.b", minimum=0, maximum=1)
    bm25_epsilon = number(bm25["epsilon"], "retrieval.bm25.epsilon", minimum=0)

    dense = strict_object(
        retrieval["dense"],
        "retrieval.dense",
        {"similarity", "normalized", "embedding_batch_size"},
    )
    dense_similarity = choice(
        dense["similarity"], "retrieval.dense.similarity", {"exact-cosine"}
    )
    dense_normalized = boolean(dense["normalized"], "retrieval.dense.normalized")
    if not dense_normalized:
        raise ValueError("Retrieval dense vectors must be normalized")
    embedding_batch_size = integer(
        dense["embedding_batch_size"],
        "retrieval.dense.embedding_batch_size",
        minimum=1,
    )

    rrf = strict_object(
        retrieval["rrf"],
        "retrieval.rrf",
        {"source_depth", "constant", "weights"},
    )
    rrf_source_depth = integer(
        rrf["source_depth"], "retrieval.rrf.source_depth", minimum=1
    )
    if rrf_source_depth != pool_size:
        raise ValueError("RRF source_depth must equal retrieval.pool_size")
    rrf_constant = integer(rrf["constant"], "retrieval.rrf.constant", minimum=0)
    raw_weights = sequence(rrf["weights"], "retrieval.rrf.weights")
    if len(raw_weights) != 2:
        raise ValueError("RRF requires exactly dense and BM25 weights")
    rrf_weights = tuple(
        number(
            weight, f"retrieval.rrf.weights[{index}]", minimum=0, minimum_exclusive=True
        )
        for index, weight in enumerate(raw_weights)
    )

    reranking = strict_object(
        root["reranking"],
        "reranking",
        {
            "batch_size",
            "reject_truncation",
            "input_template",
            "maximum_input_tokens",
        },
    )
    reranker_batch_size = integer(
        reranking["batch_size"], "reranking.batch_size", minimum=1
    )
    reject_truncation = boolean(
        reranking["reject_truncation"], "reranking.reject_truncation"
    )
    if not reject_truncation:
        raise ValueError("Retrieval benchmark protocol must reject truncation")
    input_template = choice(
        reranking["input_template"],
        "reranking.input_template",
        {"tokenizer(query, passage)"},
    )
    raw_limits = mapping(
        reranking["maximum_input_tokens"], "reranking.maximum_input_tokens"
    )
    expected_aliases = set(RERANKER_MODELS)
    if set(raw_limits) != expected_aliases:
        raise ValueError(
            "reranking.maximum_input_tokens must define exactly: "
            + ", ".join(sorted(expected_aliases))
        )
    maximum_tokens = {
        alias: integer(
            raw_limits[alias],
            f"reranking.maximum_input_tokens.{alias}",
            minimum=1,
        )
        for alias in sorted(expected_aliases)
    }

    quality = strict_object(root["quality"], "quality", {"cutoffs", "alpha_ndcg_alpha"})
    cutoffs = tuple(
        integer(value, f"quality.cutoffs[{index}]", minimum=1)
        for index, value in enumerate(sequence(quality["cutoffs"], "quality.cutoffs"))
    )
    if not cutoffs or tuple(sorted(set(cutoffs))) != cutoffs:
        raise ValueError("quality.cutoffs must be non-empty, unique, and increasing")
    if max(cutoffs) > pool_size:
        raise ValueError("quality.cutoffs cannot exceed retrieval.pool_size")
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
        {"minimum_latency_ci_documents", "confidence_level"},
    )
    minimum_documents = integer(
        statistics["minimum_latency_ci_documents"],
        "statistics.minimum_latency_ci_documents",
        minimum=1,
    )
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
        selection["maximum_finalists"],
        "selection.maximum_finalists",
        minimum=1,
    )

    profiles = execution_profiles(
        root["profiles"], names=("smoke", "development", "validation", "locked")
    )
    if {profile.batch_size for profile in profiles.values()} != {
        embedding_batch_size
    } or embedding_batch_size != reranker_batch_size:
        raise ValueError(
            "Retrieval profile, embedding, and reranker batch sizes must agree"
        )
    resources = strict_object(
        root["resources"], "resources", {"authoritative_peak_vram_mb"}
    )
    peak_vram = number(
        resources["authoritative_peak_vram_mb"],
        "resources.authoritative_peak_vram_mb",
        minimum=0,
        minimum_exclusive=True,
    )
    return RetrievalProtocol(
        meta=metadata("retrieval", source_path, root, profiles=profiles),
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
        authoritative_peak_vram_mb=peak_vram,
    )
