"""Strict, versioned protocol for retrieval/reranking experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

import yaml

from edumind.common.artifacts import stable_hash

from .profiles import RERANKER_MODELS


DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("protocol.yaml")


@dataclass(frozen=True)
class ExecutionProfile:
    warmups: int
    repetitions: int
    bootstrap_resamples: int
    device: str
    dtype: str
    hardware_required: bool


@dataclass(frozen=True)
class RetrievalProtocol:
    checksum: str
    version: str
    resolved: Mapping[str, object]
    seed: int
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
        observed = (seed, warmups, repetitions, bootstrap_resamples)
        expected = (
            self.seed,
            profile.warmups,
            profile.repetitions,
            profile.bootstrap_resamples,
        )
        if observed != expected:
            raise ValueError("Benchmark plan does not match the retrieval protocol profile")
        if profile.hardware_required and (device, dtype) != (
            profile.device,
            profile.dtype,
        ):
            raise ValueError(
                "Benchmark hardware does not match the retrieval protocol profile"
            )


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> RetrievalProtocol:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return protocol_from_mapping(payload)


@lru_cache(maxsize=1)
def default_protocol() -> RetrievalProtocol:
    return load_protocol(DEFAULT_PROTOCOL_PATH)


def protocol_from_settings(settings: Mapping[str, object]) -> RetrievalProtocol:
    payload = _mapping(settings.get("retrieval_protocol"), "retrieval_protocol")
    protocol = protocol_from_mapping(payload)
    expected = str(settings.get("retrieval_protocol_checksum", ""))
    if not expected or expected != protocol.checksum:
        raise ValueError("Retrieval protocol checksum does not match the resolved plan")
    if settings.get("retrieval_protocol_version") != protocol.version:
        raise ValueError("Retrieval protocol version does not match the resolved plan")
    return protocol


def protocol_from_mapping(value: object) -> RetrievalProtocol:
    root = _object(
        value,
        "root",
        {
            "schema_version",
            "protocol_version",
            "seed",
            "retrieval",
            "reranking",
            "quality",
            "statistics",
            "selection",
            "profiles",
            "resources",
        },
    )
    if _integer(root["schema_version"], "schema_version") != 1:
        raise ValueError("Retrieval protocol must use schema_version 1")
    version = _string(root["protocol_version"], "protocol_version")
    seed = _integer(root["seed"], "seed", minimum=0)

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
        checksum=stable_hash(resolved),
        version=version,
        resolved=resolved,
        seed=seed,
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
    device_key = "default_device" if name == "smoke" else "required_device"
    dtype_key = "default_dtype" if name == "smoke" else "required_dtype"
    payload = _object(
        value,
        f"profiles.{name}",
        {"warmups", "repetitions", "bootstrap_resamples", device_key, dtype_key},
    )
    return ExecutionProfile(
        _integer(payload["warmups"], f"profiles.{name}.warmups", minimum=0),
        _integer(payload["repetitions"], f"profiles.{name}.repetitions", minimum=1),
        _integer(
            payload["bootstrap_resamples"],
            f"profiles.{name}.bootstrap_resamples",
            minimum=0,
        ),
        _choice(payload[device_key], f"profiles.{name}.{device_key}", {"cpu", "cuda"}),
        _choice(
            payload[dtype_key],
            f"profiles.{name}.{dtype_key}",
            {"float32", "float16", "bfloat16"},
        ),
        name != "smoke",
    )


def _object(value: object, label: str, fields: set[str]) -> dict[str, object]:
    payload = _mapping(value, label)
    missing = sorted(fields - set(payload))
    unknown = sorted(set(payload) - fields)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"Retrieval protocol {label}: {'; '.join(details)}")
    return payload


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Retrieval protocol {label} must be an object")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Retrieval protocol {label} must be a list")
    return list(value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Retrieval protocol {label} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"Retrieval protocol {label} must be a non-empty string")
    return result


def _choice(value: object, label: str, choices: set[str]) -> str:
    result = _string(value, label)
    if result not in choices:
        raise ValueError(
            f"Retrieval protocol {label} must be one of: {', '.join(sorted(choices))}"
        )
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Retrieval protocol {label} must be boolean")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(
            f"Retrieval protocol {label} must be an integer >= {minimum}"
        )
    return value


def _number(
    value: object,
    label: str,
    *,
    minimum: float,
    maximum: float | None = None,
    exclusive: bool = False,
    maximum_exclusive: bool = False,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Retrieval protocol {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Retrieval protocol {label} must be finite")
    if result < minimum or (exclusive and result == minimum):
        raise ValueError(f"Retrieval protocol {label} is below its valid range")
    if maximum is not None and (
        result > maximum or (maximum_exclusive and result == maximum)
    ):
        raise ValueError(f"Retrieval protocol {label} exceeds its valid range")
    return result


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value
