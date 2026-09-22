"""Strict protocol for ASR extraction benchmarks."""

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
    sequence,
    strict_object,
    string,
)

DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("protocol.yaml")
_BACKEND_BY_ALIAS = {
    "whisper-small-en-control": "transformers",
    "canary-180m": "nemo",
    "parakeet-tdt-0.6b-v2": "nemo",
    "moss-transcribe-diarize": "moss",
}


@dataclass(frozen=True)
class AudioCandidate:
    alias: str
    model_id: str
    backend: str


@dataclass(frozen=True)
class AudioProtocol:
    meta: ProtocolMetadata
    audio: Mapping[str, object]
    decoding: Mapping[str, Mapping[str, object]]
    candidates: Mapping[str, AudioCandidate]
    alignment_threshold: float
    speech_counts: Mapping[str, int]
    required_conditions: frozenset[str]
    exclusive_condition_groups: tuple[frozenset[str], ...]
    reliability_categories: frozenset[str]
    smoke_reliability_categories: frozenset[str]
    confidence_level: float
    maximum_finalists: int
    authoritative_peak_vram_mb: float

    def profile(self, name: str):
        return self.meta.profile(name)

    @property
    def batch_size(self) -> int:
        return next(iter(self.meta.profiles.values())).batch_size

    def candidate(self, alias: str) -> AudioCandidate:
        try:
            return self.candidates[alias]
        except KeyError as exc:
            raise ValueError(f"Unknown ASR candidate: {alias}") from exc

    def decoder(self, alias: str) -> Mapping[str, object]:
        self.candidate(alias)
        return self.decoding[alias]

    def dtype(self, alias: str, device: str) -> str:
        return str(self.decoder(alias)[f"{device}_dtype"])


def load_protocol(
    path: Path = DEFAULT_PROTOCOL_PATH,
) -> AudioProtocol:
    return protocol_from_mapping(
        load_yaml(path, "audio"),
        source_path=path,
    )


def protocol_from_mapping(
    value: object,
    *,
    source_path: Path = DEFAULT_PROTOCOL_PATH,
) -> AudioProtocol:
    root = strict_object(
        value,
        "audio protocol root",
        {
            "schema_version",
            "protocol_version",
            "seed",
            "audio",
            "decoding",
            "alignment",
            "datasets",
            "statistics",
            "selection",
            "resources",
            "profiles",
        },
    )
    audio = strict_object(
        root["audio"],
        "audio",
        {
            "sample_rate_hz",
            "channels",
            "sample_width_bytes",
            "encoding",
            "maximum_duration_seconds",
            "manifest_duration_tolerance_seconds",
        },
    )
    if integer(audio["sample_rate_hz"], "audio.sample_rate_hz", minimum=1) != 16_000:
        raise ValueError("Canonical ASR audio must use 16 kHz")
    if integer(audio["channels"], "audio.channels", minimum=1) != 1:
        raise ValueError("Canonical ASR audio must be mono")
    if integer(audio["sample_width_bytes"], "audio.sample_width_bytes", minimum=1) != 2:
        raise ValueError("Canonical ASR audio must use signed 16-bit samples")
    choice(
        audio["encoding"],
        "audio.encoding",
        {"PCM signed 16-bit little-endian"},
    )
    number(
        audio["maximum_duration_seconds"],
        "audio.maximum_duration_seconds",
        minimum=0,
        minimum_exclusive=True,
        maximum=30,
    )
    number(
        audio["manifest_duration_tolerance_seconds"],
        "audio.manifest_duration_tolerance_seconds",
        minimum=0,
    )
    raw_decoding = mapping(root["decoding"], "decoding")
    if set(raw_decoding) != set(_BACKEND_BY_ALIAS):
        raise ValueError(
            "Audio decoding must define every supported ASR adapter exactly once"
        )
    decoding = {
        alias: _decoder(alias, _BACKEND_BY_ALIAS[alias], value)
        for alias, value in raw_decoding.items()
    }
    candidates = {
        alias: AudioCandidate(alias, str(value["model_id"]), _BACKEND_BY_ALIAS[alias])
        for alias, value in decoding.items()
    }
    alignment = strict_object(root["alignment"], "alignment", {"content_f1_threshold"})
    alignment_threshold = number(
        alignment["content_f1_threshold"],
        "alignment.content_f1_threshold",
        minimum=0,
        maximum=1,
    )
    datasets = strict_object(
        root["datasets"],
        "datasets",
        {
            "speech_counts",
            "required_speech_conditions",
            "exclusive_condition_groups",
            "reliability_categories",
            "smoke_reliability_categories",
        },
    )
    counts = strict_object(
        datasets["speech_counts"],
        "datasets.speech_counts",
        {"smoke", "development", "validation", "locked"},
    )
    speech_counts = {
        name: integer(value, f"datasets.speech_counts.{name}", minimum=1)
        for name, value in counts.items()
    }
    conditions = frozenset(
        _strings(
            datasets["required_speech_conditions"],
            "datasets.required_speech_conditions",
        )
    )
    exclusive_groups = tuple(
        frozenset(_strings(group, f"datasets.exclusive_condition_groups[{index}]"))
        for index, group in enumerate(
            sequence(
                datasets["exclusive_condition_groups"],
                "datasets.exclusive_condition_groups",
            )
        )
    )
    if not exclusive_groups or any(
        not group <= conditions for group in exclusive_groups
    ):
        raise ValueError(
            "Exclusive ASR condition groups must be non-empty subsets of required conditions"
        )
    reliability = frozenset(
        _strings(datasets["reliability_categories"], "datasets.reliability_categories")
    )
    smoke_reliability = frozenset(
        _strings(
            datasets["smoke_reliability_categories"],
            "datasets.smoke_reliability_categories",
        )
    )
    if not smoke_reliability <= reliability:
        raise ValueError(
            "Smoke reliability categories must be a subset of authoritative categories"
        )
    statistics = strict_object(root["statistics"], "statistics", {"confidence_level"})
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
        root["resources"],
        "resources",
        {
            "authoritative_peak_vram_mb",
            "allow_fallback",
            "allow_offload",
            "allow_quantization",
            "allow_device_splitting",
        },
    )
    peak = number(
        resources["authoritative_peak_vram_mb"],
        "resources.authoritative_peak_vram_mb",
        minimum=0,
        minimum_exclusive=True,
    )
    policy = {
        name: boolean(resources[name], f"resources.{name}")
        for name in resources
        if name.startswith("allow_")
    }
    if any(policy.values()):
        raise ValueError("Authoritative ASR resource fallbacks must remain disabled")
    profiles = execution_profiles(
        root["profiles"], names=("smoke", "development", "validation", "locked")
    )
    if {profile.batch_size for profile in profiles.values()} != {1}:
        raise ValueError("Every ASR execution profile must use batch size one")
    return AudioProtocol(
        metadata("audio", source_path, root, profiles=profiles),
        audio,
        decoding,
        candidates,
        alignment_threshold,
        speech_counts,
        conditions,
        exclusive_groups,
        reliability,
        smoke_reliability,
        confidence,
        maximum_finalists,
        peak,
    )


def protocol_from_worker(value: object) -> AudioProtocol:
    payload = mapping(value, "audio worker protocol")
    root = mapping(payload.get("resolved"), "audio resolved protocol")
    protocol = protocol_from_mapping(root)
    protocol.meta.validate_worker_payload(payload)
    return protocol


def _decoder(alias: str, backend: str, value: object) -> dict[str, object]:
    common = {
        "model_id",
        "language",
        "decoder",
        "timestamp_method",
        "cpu_dtype",
        "cuda_dtype",
    }
    extras = {
        "transformers": {"return_timestamps", "do_sample"},
        "nemo": (
            {"timestamps", "beam_size", "punctuation_and_capitalization"}
            if alias == "canary-180m"
            else {"timestamps", "decoding_strategy", "timestamp_level"}
        ),
        "moss": {"max_new_tokens", "do_sample"},
    }[backend]
    row = strict_object(value, f"decoding.{alias}", common | extras)
    string(row["model_id"], f"decoding.{alias}.model_id")
    choice(row["language"], f"decoding.{alias}.language", {"English"})
    expected_descriptors = {
        "whisper-small-en-control": ("greedy", "native-word"),
        "canary-180m": ("beam-1-pnc", "native-segment"),
        "parakeet-tdt-0.6b-v2": ("greedy", "native-segment"),
        "moss-transcribe-diarize": ("deterministic", "native-segment"),
    }
    decoder, timestamp_method = expected_descriptors[alias]
    choice(row["decoder"], f"decoding.{alias}.decoder", {decoder})
    choice(
        row["timestamp_method"],
        f"decoding.{alias}.timestamp_method",
        {timestamp_method},
    )
    choice(row["cpu_dtype"], f"decoding.{alias}.cpu_dtype", {"float32"})
    choice(row["cuda_dtype"], f"decoding.{alias}.cuda_dtype", {"float16", "bfloat16"})
    for name in extras:
        if name in {"do_sample", "timestamps", "punctuation_and_capitalization"}:
            boolean(row[name], f"decoding.{alias}.{name}")
        elif name in {"beam_size", "max_new_tokens"}:
            integer(row[name], f"decoding.{alias}.{name}", minimum=1)
        elif name == "return_timestamps":
            choice(row[name], f"decoding.{alias}.{name}", {"word"})
        elif name == "decoding_strategy":
            choice(row[name], f"decoding.{alias}.{name}", {"greedy_batch"})
        elif name == "timestamp_level":
            choice(row[name], f"decoding.{alias}.{name}", {"segment_then_word"})
        else:
            string(row[name], f"decoding.{alias}.{name}")
    return row


def _strings(value: object, label: str) -> tuple[str, ...]:
    values = tuple(
        string(item, f"{label}[{index}]")
        for index, item in enumerate(sequence(value, label))
    )
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must contain unique values")
    return values
