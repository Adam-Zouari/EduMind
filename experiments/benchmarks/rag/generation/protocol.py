"""Strict protocol for answer-generation benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from experiments.benchmarks.common.protocol import (
    ProtocolMetadata,
    boolean,
    choice,
    execution_profiles,
    integer,
    load_yaml,
    mapping,
    metadata,
    number,
    strict_object,
    string,
)

DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("protocol.yaml")
GENERATOR_LOADERS = {
    "falcon-h1-tiny-r-90m-control": "causal-lm",
    "qwen3-0.6b-reasoning": "causal-lm",
    "qwen3.5-0.8b-reasoning": "multimodal-lm",
    "minicpm5-1b-reasoning": "causal-lm",
}


@dataclass(frozen=True)
class GenerationProtocol:
    meta: ProtocolMetadata
    models: Mapping[str, str]
    temperature: float
    do_sample: bool
    context_tokens: int
    maximum_answer_tokens: int
    streamer_timeout_seconds: float
    question_count: int
    context_packing_tokens: int
    context_tokenizer: str
    faithfulness_model: str
    faithfulness_trust_remote_code: bool
    confidence_level: float
    maximum_finalists: int
    authoritative_peak_vram_mb: float

    def profile(self, name: str):
        return self.meta.profile(name)

    def model_id(self, alias: str) -> str:
        try:
            return self.models[alias]
        except KeyError as exc:
            raise ValueError(f"Unknown generation candidate: {alias}") from exc


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> GenerationProtocol:
    root = strict_object(
        load_yaml(path, "generation"),
        "generation protocol root",
        {
            "schema_version",
            "protocol_version",
            "seed",
            "models",
            "generation",
            "development_screen",
            "context",
            "faithfulness",
            "statistics",
            "selection",
            "resources",
            "profiles",
        },
    )
    models = {
        alias: string(model, f"models.{alias}")
        for alias, model in mapping(root["models"], "models").items()
    }
    if set(models) != set(GENERATOR_LOADERS) or len(set(models.values())) != len(
        models
    ):
        raise ValueError(
            "Generation models must define every supported generator exactly once with unique model IDs"
        )
    generation = strict_object(
        root["generation"],
        "generation",
        {
            "temperature",
            "do_sample",
            "context_tokens",
            "maximum_answer_tokens",
            "streamer_timeout_seconds",
        },
    )
    temperature = number(generation["temperature"], "generation.temperature", minimum=0)
    do_sample = boolean(generation["do_sample"], "generation.do_sample")
    if do_sample or temperature != 0:
        raise ValueError("Generation benchmark must remain deterministic")
    context_tokens = integer(
        generation["context_tokens"], "generation.context_tokens", minimum=1
    )
    answer_tokens = integer(
        generation["maximum_answer_tokens"],
        "generation.maximum_answer_tokens",
        minimum=1,
    )
    streamer_timeout = number(
        generation["streamer_timeout_seconds"],
        "generation.streamer_timeout_seconds",
        minimum=0,
        minimum_exclusive=True,
    )
    screen = strict_object(
        root["development_screen"],
        "development_screen",
        {"question_count", "selection"},
    )
    question_count = integer(
        screen["question_count"], "development_screen.question_count", minimum=1
    )
    choice(
        screen["selection"],
        "development_screen.selection",
        {"deterministic-stratified-v1"},
    )
    context = strict_object(
        root["context"], "context", {"frozen_packing_limit_tokens", "tokenizer"}
    )
    packing = integer(
        context["frozen_packing_limit_tokens"],
        "context.frozen_packing_limit_tokens",
        minimum=1,
    )
    if packing >= context_tokens:
        raise ValueError("Frozen context packing must leave room for generation")
    tokenizer = choice(
        context["tokenizer"], "context.tokenizer", {"tiktoken:cl100k_base"}
    )
    faithfulness = strict_object(
        root["faithfulness"],
        "faithfulness",
        {"model_id", "behavior", "trust_remote_code"},
    )
    model = string(faithfulness["model_id"], "faithfulness.model_id")
    choice(
        faithfulness["behavior"],
        "faithfulness.behavior",
        {"local-sequence-classification-predict-v1"},
    )
    trust = boolean(faithfulness["trust_remote_code"], "faithfulness.trust_remote_code")
    statistics = strict_object(root["statistics"], "statistics", {"confidence_level"})
    confidence = number(
        statistics["confidence_level"],
        "statistics.confidence_level",
        minimum=0,
        maximum=1,
        minimum_exclusive=True,
        maximum_exclusive=True,
    )
    selection_limits = strict_object(
        root["selection"], "selection", {"maximum_finalists"}
    )
    maximum_finalists = integer(
        selection_limits["maximum_finalists"], "selection.maximum_finalists", minimum=1
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
        root["profiles"], names=("smoke", "development", "validation")
    )
    if {profile.batch_size for profile in profiles.values()} != {1}:
        raise ValueError("Generation profiles must use batch size one")
    return GenerationProtocol(
        metadata("generation", path, root, profiles=profiles),
        models,
        temperature,
        do_sample,
        context_tokens,
        answer_tokens,
        streamer_timeout,
        question_count,
        packing,
        tokenizer,
        model,
        trust,
        confidence,
        maximum_finalists,
        peak,
    )
