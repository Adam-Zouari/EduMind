"""Strict, local-only seven model-mode generation protocol."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from experiments.benchmarks.common.protocol import (
    PreflightSettings,
    ProtocolMetadata,
    boolean,
    execution_profiles,
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

DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("protocol.yaml")
GENERATOR_LOADERS = {
    "falcon-h1-tiny-r-90m-control": "causal-lm",
    **{
        f"{model}-{mode}": loader
        for model, loader in (
            ("qwen3-0.6b", "causal-lm"),
            ("qwen3.5-0.8b", "multimodal-lm"),
            ("minicpm5-1b", "causal-lm"),
        )
        for mode in ("direct", "reasoning")
    },
}


@dataclass(frozen=True)
class GenerationProtocol:
    meta: ProtocolMetadata
    models: Mapping[str, str]
    decoding: Mapping[str, Mapping[str, object]]
    context_tokens: int
    measured_seeds: tuple[int, ...]
    warmup_seed: int
    streamer_timeout_seconds: float
    judge_contract_path: Path | None
    confidence_level: float
    minimum_ci_sources: Mapping[str, int]
    data_requirements: Mapping[str, object]
    maximum_finalists: int
    authoritative_peak_vram_mb: float
    preflight: PreflightSettings

    def profile(self, name):
        return self.meta.profile(name)

    def model_id(self, alias):
        if alias not in self.models:
            raise ValueError(f"Unknown generation candidate: {alias}")
        return self.models[alias]

    def mode(self, alias):
        self.model_id(alias)
        return "direct" if alias.endswith("-direct") else "reasoning"

    def decoder(self, alias):
        self.model_id(alias)
        return dict(self.decoding[alias])


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> GenerationProtocol:
    path = Path(path)
    root = strict_object(
        load_yaml(path, "generation"),
        "generation protocol",
        {
            "schema_version",
            "protocol_version",
            "seed",
            "models",
            "decoding",
            "generation",
            "judge",
            "statistics",
            "data",
            "selection",
            "resources",
            "profiles",
            "preflight",
        },
    )
    models = {
        alias: string(value, f"models.{alias}")
        for alias, value in mapping(root["models"], "models").items()
    }
    if set(models) != set(GENERATOR_LOADERS):
        raise ValueError(
            "Models must define every supported generator model-mode exactly once"
        )
    for prefix in ("qwen3-0.6b", "qwen3.5-0.8b", "minicpm5-1b"):
        if models[prefix + "-direct"] != models[prefix + "-reasoning"]:
            raise ValueError("Modes of a checkpoint must use the same model ID")
    if len(set(models.values())) != 4:
        raise ValueError("Generation requires four independent checkpoints")
    decoders = strict_object(root["decoding"], "decoding", set(models))
    for alias, value in decoders.items():
        if alias.startswith("falcon-"):
            if value != {"source": "pinned-generation-config"}:
                raise ValueError(
                    "Falcon decoder must resolve its pinned official generation config"
                )
            continue
        fields = {"do_sample", "temperature", "top_p"}
        if alias.startswith("qwen"):
            fields |= {"top_k", "min_p"}
        if alias.startswith("qwen3.5"):
            fields |= {"presence_penalty", "repetition_penalty"}
        decoder = strict_object(value, f"decoding.{alias}", fields)
        if not boolean(decoder["do_sample"], "do_sample"):
            raise ValueError("Generation requires sampled decoding")
        for name, item in decoder.items():
            if name == "do_sample":
                continue
            if name == "top_k":
                integer(item, name, minimum=1)
            else:
                number(
                    item,
                    name,
                    minimum=0,
                    maximum=1 if name in {"top_p", "min_p"} else None,
                    minimum_exclusive=name
                    in {"temperature", "top_p", "repetition_penalty"},
                )
    generation = strict_object(
        root["generation"],
        "generation",
        {"context_tokens", "measured_seeds", "warmup_seed", "streamer_timeout_seconds"},
    )
    seeds = tuple(
        integer(value, "measured seed", minimum=0)
        for value in sequence(generation["measured_seeds"], "measured_seeds")
    )
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("Measured seeds must be unique and support a pair task")
    context = integer(generation["context_tokens"], "context_tokens", minimum=1)
    warmup_seed = integer(generation["warmup_seed"], "warmup_seed", minimum=0)
    timeout = number(
        generation["streamer_timeout_seconds"],
        "streamer_timeout_seconds",
        minimum=0,
        minimum_exclusive=True,
    )
    judge = strict_object(root["judge"], "judge", {"contract_path"})
    judge_path = (
        None
        if judge["contract_path"] is None
        else Path(string(judge["contract_path"], "judge.contract_path"))
    )
    if judge_path is not None and not judge_path.is_absolute():
        judge_path = (path.parent / judge_path).resolve()
    stats = strict_object(
        root["statistics"], "statistics", {"confidence_level", "minimum_ci_sources"}
    )
    confidence = number(
        stats["confidence_level"],
        "confidence_level",
        minimum=0,
        maximum=1,
        minimum_exclusive=True,
        maximum_exclusive=True,
    )
    minima = {
        key: integer(value, f"minimum_ci_sources.{key}", minimum=2)
        for key, value in mapping(
            stats["minimum_ci_sources"], "minimum_ci_sources"
        ).items()
    }
    data = strict_object(
        root["data"], "data", {"qasper_papers", "minimum_structured_questions"}
    )
    for name, fields in (
        ("qasper_papers", {"development", "validation", "locked"}),
        ("minimum_structured_questions", {"table", "formula", "mixed"}),
    ):
        for key, value in strict_object(data[name], name, fields).items():
            integer(value, f"{name}.{key}", minimum=1)
    selection = strict_object(root["selection"], "selection", {"maximum_finalists"})
    maximum = integer(
        selection["maximum_finalists"], "maximum_finalists", minimum=1, maximum=3
    )
    resources = strict_object(
        root["resources"], "resources", {"authoritative_peak_vram_mb"}
    )
    peak = number(
        resources["authoritative_peak_vram_mb"],
        "authoritative_peak_vram_mb",
        minimum=0,
        minimum_exclusive=True,
    )
    preflight = preflight_settings(root["preflight"])
    if (preflight.warmups, preflight.repetitions, preflight.bootstrap_resamples) != (
        0,
        1,
        0,
    ):
        raise ValueError("Preflight requires zero warmups and one stress inference")
    profiles = execution_profiles(
        root["profiles"], names=("smoke", "development", "validation", "locked")
    )
    for name, profile in profiles.items():
        if (
            profile.warmups != 1
            or profile.batch_size != 1
            or profile.repetitions != (1 if name == "smoke" else len(seeds))
        ):
            raise ValueError(
                "Generation profiles require one warmup, batch one, and repetitions matching the aligned seed list"
            )
        if name == "smoke" and profile.bootstrap_resamples:
            raise ValueError("Smoke has no bootstrap intervals")
        if name == "smoke":
            if (
                profile.devices != ("cpu", "cuda")
                or profile.dtype_for("cpu") != "float32"
                or profile.dtype_for("cuda") != "float16"
                or profile.hardware_required
            ):
                raise ValueError(
                    "Smoke requires independent CPU/float32 and CUDA/float16 paths"
                )
        elif (profile.device, profile.dtype, profile.hardware_required) != (
            "cuda",
            "float16",
            True,
        ):
            raise ValueError("Authoritative generation requires CUDA float16")
    return GenerationProtocol(
        metadata("generation", path, root, profiles=profiles),
        models,
        decoders,
        context,
        seeds,
        warmup_seed,
        timeout,
        judge_path,
        confidence,
        minima,
        data,
        maximum,
        peak,
        preflight,
    )
