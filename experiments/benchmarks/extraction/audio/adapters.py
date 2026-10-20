"""The four executable ASR profiles used by the audio benchmark."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from edumind.common.gpu import release_runtime_memory
from edumind.common.model_placement import inspect_model_placement
from edumind.extraction.errors import MissingDependencyError
from edumind.extraction.extractors.audio import load_whisper_runtime
from experiments.benchmarks.common.provenance import package_versions
from experiments.benchmarks.extraction.audio.protocol import AudioProtocol


@dataclass(frozen=True)
class ASRProfile:
    candidate: str
    model: str
    backend: str
    decoder: str
    timestamp_method: str
    decoding: Mapping[str, object]


@dataclass(frozen=True)
class Transcript:
    text: str
    segments: tuple[Mapping[str, object], ...]
    warnings: tuple[str, ...] = ()


class ASRPlacementError(RuntimeError):
    """A rejected placement whose evidence must survive model cleanup."""

    def __init__(self, message, model, expected, locations):
        super().__init__(message)
        self.placement_report = inspect_model_placement(model, expected_device=expected)
        self.placement_report["observed_device_locations"] = sorted(locations)
        if self.placement_report["status"] == "qualified":
            self.placement_report["status"] = (
                "device_splitting_detected"
                if len(locations) > 1
                else "placement_unverifiable"
            )


def profiles(protocol: AudioProtocol) -> dict[str, ASRProfile]:
    return {
        alias: ASRProfile(
            alias,
            candidate.model_id,
            candidate.backend,
            str(protocol.decoder(alias)["decoder"]),
            str(protocol.decoder(alias)["timestamp_method"]),
            protocol.decoder(alias),
        )
        for alias, candidate in protocol.candidates.items()
    }


def build_runtime(
    candidate: str,
    model_lock: Mapping[str, Mapping[str, object]],
    device: str,
    protocol: AudioProtocol,
):
    configured = profiles(protocol)
    try:
        profile = configured[candidate]
    except KeyError as exc:
        raise ValueError(f"Unknown ASR candidate: {candidate}") from exc
    entry = model_lock.get(profile.model)
    if not isinstance(entry, Mapping) or not entry:
        raise RuntimeError(f"The model lock has no entry for {profile.model}")
    if not isinstance(entry.get("revision"), str) or not entry["revision"].strip():
        raise RuntimeError(
            f"The model lock lacks a pinned revision for {profile.model}"
        )
    if entry.get("model", profile.model) != profile.model:
        raise RuntimeError(
            f"The model lock identifies a different model than {profile.model}"
        )
    raw_path = entry.get("model_path")
    model_path = Path(str(raw_path)) if raw_path else None
    if model_path is None or not model_path.is_dir():
        raise FileNotFoundError(f"Pinned ASR snapshot is missing: {model_path}")
    runtime_type = {
        "transformers": WhisperRuntime,
        "nemo": NemoRuntime,
        "moss": MossRuntime,
    }[profile.backend]
    runtime = runtime_type(profile, model_path.resolve(), device, protocol)
    runtime.model_lock_entry = dict(entry)
    return runtime


class BaseRuntime:
    def __init__(
        self,
        profile: ASRProfile,
        model_path: Path,
        device: str,
        protocol: AudioProtocol,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("ASR device must be cpu or cuda")
        self.profile = profile
        self.model_path = model_path
        self.device = device
        self.protocol = protocol
        self._runtime: Any | None = None
        self.model_lock_entry: Mapping[str, object] = {}
        self.dtype = protocol.dtype(profile.candidate, device)
        self.raw_prediction = None

    def load(self) -> None:
        raise NotImplementedError

    def transcribe(self, source: Path) -> Transcript:
        raise NotImplementedError

    def close(self) -> None:
        self._runtime = None
        release_runtime_memory()

    def placement_models(self) -> tuple[object, ...]:
        if self._runtime is None:
            return ()
        return (self._runtime,)

    def parameters(self) -> dict[str, object]:
        return {
            "candidate": self.profile.candidate,
            "model": self.profile.model,
            "model_path": str(self.model_path),
            "model_revision": self.model_lock_entry.get("revision", ""),
            "selection_revision": self.model_lock_entry.get("selection_revision", ""),
            "model_cache_manifest_sha256": self.model_lock_entry.get(
                "model_cache_manifest_sha256", ""
            ),
            "backend": self.profile.backend,
            "runtime_version": _runtime_version(self.profile.backend),
            "device": self.device,
            "dtype": self.dtype,
            "language": self.profile.decoding["language"],
            "decoder": self.profile.decoder,
            "timestamp_method": self.profile.timestamp_method,
            "sample_rate_hz": self.protocol.audio["sample_rate_hz"],
            "batch_size": self.protocol.batch_size,
            "package_versions": package_versions(
                (
                    "torch",
                    "torchaudio",
                    "transformers",
                    "nemo_toolkit",
                    "moss-transcribe-diarize",
                    "soundfile",
                    "jiwer",
                    "rapidfuzz",
                )
            ),
        }


class WhisperRuntime(BaseRuntime):
    def placement_models(self) -> tuple[object, ...]:
        return (self._runtime.model,) if self._runtime is not None else ()

    def parameters(self) -> dict[str, object]:
        return {
            **super().parameters(),
            "return_timestamps": self.profile.decoding["return_timestamps"],
            "do_sample": self.profile.decoding["do_sample"],
            "num_beams": 1,
        }

    def load(self) -> None:
        expected_dtype = self.dtype
        self._runtime, self.dtype = load_whisper_runtime(
            self.model_path,
            self.device,
            dtype=expected_dtype,
        )
        if self.dtype != expected_dtype:
            raise RuntimeError(
                f"Whisper loaded {self.dtype} instead of protocol dtype {expected_dtype}"
            )
        _assert_device(self._runtime.model, self.device, dtype=expected_dtype)

    def transcribe(self, source: Path) -> Transcript:
        result = self._runtime(
            str(source),
            batch_size=self.protocol.batch_size,
            return_timestamps=str(self.profile.decoding["return_timestamps"]),
            generate_kwargs={
                "do_sample": self.profile.decoding["do_sample"],
                "num_beams": 1,
            },
        )
        self.raw_prediction = result
        if not isinstance(result, Mapping):
            raise ValueError("Whisper output must be an object")
        text = _text(result.get("text"))
        segments = []
        for chunk in _sequence(result.get("chunks", []), "Whisper chunks"):
            if not isinstance(chunk, Mapping):
                raise ValueError("Whisper chunks must contain objects")
            stamp = _sequence(chunk.get("timestamp"), "Whisper timestamp")
            if len(stamp) != 2:
                raise ValueError("Whisper timestamp must contain two boundaries")
            # Retain incomplete lexical chunks: validation must fail the attempt,
            # not score only the successfully timestamped portion of its text.
            segments.append(
                {
                    "text": _text(chunk.get("text")),
                    "start": None if stamp[0] is None else _boundary(stamp[0]),
                    "end": None if stamp[1] is None else _boundary(stamp[1]),
                }
            )
        missing = sum(item["start"] is None or item["end"] is None for item in segments)
        warnings = (
            (f"{missing} Whisper chunk(s) lacked complete timestamp boundaries",)
            if missing
            else ()
        )
        return Transcript(text, tuple(segments), warnings)


class NemoRuntime(BaseRuntime):
    def parameters(self) -> dict[str, object]:
        values = {
            **super().parameters(),
            "timestamps": self.profile.decoding["timestamps"],
        }
        if self.profile.candidate == "canary-180m":
            values.update(
                {
                    "decoding_strategy": "beam",
                    "beam_size": self.profile.decoding["beam_size"],
                    "punctuation_and_capitalization": self.profile.decoding[
                        "punctuation_and_capitalization"
                    ],
                }
            )
        else:
            values.update(
                {
                    "decoding_strategy": self.profile.decoding["decoding_strategy"],
                    "timestamp_level": self.profile.decoding["timestamp_level"],
                }
            )
        return values

    def load(self) -> None:
        try:
            import nemo.collections.asr as nemo_asr
            import torch
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("NVIDIA NeMo ASR is required") from exc
        checkpoints = sorted(self.model_path.glob("*.nemo"))
        if len(checkpoints) != 1:
            raise FileNotFoundError(
                f"Expected exactly one pinned .nemo checkpoint under {self.model_path}, "
                f"found {len(checkpoints)}"
            )
        self._runtime = nemo_asr.models.ASRModel.restore_from(
            restore_path=str(checkpoints[0]), map_location=self.device
        )
        if getattr(self._runtime, "timestamps_asr_model", None) is not None:
            raise RuntimeError(
                "The native-timestamp ASR profile cannot use an external aligner"
            )
        expected_dtype = self.dtype
        self._runtime = self._runtime.to(
            device=self.device,
            dtype=getattr(torch, expected_dtype),
        ).eval()
        from omegaconf import open_dict

        decoding = self._runtime.cfg.decoding
        with open_dict(decoding):
            if self.profile.candidate == "canary-180m":
                decoding.strategy = "beam"
                decoding.beam.beam_size = int(self.profile.decoding["beam_size"])
                decoding.beam.return_best_hypothesis = True
            else:
                decoding.strategy = str(self.profile.decoding["decoding_strategy"])
                decoding.compute_timestamps = self.profile.decoding["timestamps"]
                # NeMo's "all" computes segment and word timestamps before first use.
                decoding.rnnt_timestamp_type = {"segment_then_word": "all"}[
                    str(self.profile.decoding["timestamp_level"])
                ]
        self._runtime.change_decoding_strategy(decoding)
        _assert_device(self._runtime, self.device, dtype=expected_dtype)

    def transcribe(self, source: Path) -> Transcript:
        arguments: dict[str, object] = {
            "batch_size": self.protocol.batch_size,
            "timestamps": self.profile.decoding["timestamps"],
            "return_hypotheses": True,
        }
        if self.profile.candidate == "canary-180m":
            language = {"English": "en"}[str(self.profile.decoding["language"])]
            arguments.update(source_lang=language, target_lang=language)
            arguments["pnc"] = (
                "True"
                if self.profile.decoding["punctuation_and_capitalization"]
                else "False"
            )
        outputs = _sequence(
            self._runtime.transcribe([str(source)], **arguments), "NeMo outputs"
        )
        if len(outputs) != 1:
            raise ValueError(
                "NeMo must return exactly one hypothesis for batch size one"
            )
        output = outputs[0]
        self.raw_prediction = {
            "text": output
            if isinstance(output, str)
            else getattr(output, "text", None),
            "timestamps": getattr(output, "timestamp", None),
        }
        text = _text(
            output if isinstance(output, str) else getattr(output, "text", None)
        )
        if self.profile.candidate == "canary-180m" and hasattr(output, "y_sequence"):
            token_ids = output.y_sequence
            if not isinstance(token_ids, list):
                token_ids = token_ids.tolist()
            raw_text = _text(self._runtime.decoding.decode_ids_to_str(token_ids))
            self.raw_prediction["decoded_text"] = raw_text
            # NeMo leaves a lone timestamp token in hyp.text when silence produces
            # no complete timestamp pair. It is metadata, not lexical speech.
            text = re.sub(r"<\|[0-9]+\|>", " ", text).strip()
            # NeMo's AED timestamp postprocessor retains only paired timestamp
            # spans and can discard lexical prefixes/tails. Audit the original
            # decoded hypothesis as well, without repairing its transcription.
            if re.sub(r"<\|[0-9]+\|>", " ", raw_text).split() != text.split():
                raise ValueError("Canary timestamp processing discarded lexical text")
        timestamp_payload = getattr(output, "timestamp", {})
        # NeMo's empty Hypothesis defaults its timestamp field to an empty list.
        if isinstance(timestamp_payload, list) and not timestamp_payload:
            timestamp_payload = {}
        if not isinstance(timestamp_payload, Mapping):
            raise ValueError("NeMo timestamps must be an object")
        segments = _sequence(
            timestamp_payload.get("segment", []), "NeMo segment timestamps"
        )
        if not segments:
            segments = _sequence(
                timestamp_payload.get("word", []), "NeMo word timestamps"
            )
        return Transcript(
            text,
            tuple(_nemo_segment(item) for item in segments),
        )


class MossRuntime(BaseRuntime):
    def placement_models(self) -> tuple[object, ...]:
        return (self._runtime[0],) if self._runtime is not None else ()

    def load(self) -> None:
        try:
            import torch
            from moss_transcribe_diarize.inference_utils import (
                build_transcription_messages,
                generate_transcription,
            )
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ModuleNotFoundError as exc:
            raise MissingDependencyError(
                "The pinned MOSS-Transcribe-Diarize runtime is required"
            ) from exc
        dtype = getattr(torch, self.protocol.dtype(self.profile.candidate, self.device))
        processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=True,
            fix_mistral_regex=True,
        )
        device = torch.device(self.device)
        # One explicit backend and one load: no retries, auto dtype, or remote fetch.
        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        config = model.config
        attention = {
            name: getattr(item, "_attn_implementation", None)
            for name, item in (
                ("model", config),
                ("text", getattr(config, "text_config", None)),
                ("audio", getattr(config, "audio_config", None)),
            )
        }
        if "sdpa" not in attention.values() or any(
            value not in (None, "sdpa") for value in attention.values()
        ):
            raise RuntimeError(
                f"MOSS changed the requested SDPA attention backend: {attention}"
            )
        attention_report = {
            "requested": "sdpa",
            "selected": "sdpa",
            "config": attention,
        }
        model.generation_config.num_beams = 1
        self._runtime = (
            model.to(device=device, dtype=dtype).eval(),
            processor,
            attention_report,
            device,
            dtype,
        )
        _assert_device(self._runtime[0], self.device, dtype=self.dtype)
        self.dtype = str(dtype).removeprefix("torch.")
        self.attention_report = attention_report
        self._build_messages = build_transcription_messages
        self._generate = generate_transcription

    def parameters(self) -> dict[str, object]:
        return {
            **super().parameters(),
            "attention": str(getattr(self, "attention_report", {})),
            "max_new_tokens": self.profile.decoding["max_new_tokens"],
            "do_sample": self.profile.decoding["do_sample"],
            "num_beams": 1,
        }

    def transcribe(self, source: Path) -> Transcript:
        model, processor, attention_report, device, dtype = self._runtime
        result = self._generate(
            model,
            processor,
            self._build_messages(str(source)),
            max_new_tokens=int(self.profile.decoding["max_new_tokens"]),
            do_sample=bool(self.profile.decoding["do_sample"]),
            device=device,
            dtype=dtype,
            attention_report=attention_report,
        )
        self.raw_prediction = result
        if not isinstance(result, Mapping):
            raise ValueError("MOSS generation output must be an object")
        return _moss_transcript(_text(result.get("text")))


def _nemo_segment(item: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(item, Mapping):
        raise ValueError("NeMo timestamp units must be objects")
    return {
        "text": _text(item.get("segment", item.get("word", item.get("text")))),
        # Native transcribe() already converts frame offsets to seconds.
        "start": _boundary(item.get("start")),
        "end": _boundary(item.get("end")),
    }


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("ASR lexical text must be a string")
    return value.strip()


def _sequence(value: object, label: str) -> Sequence:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return value


def _boundary(value: object) -> float:
    if isinstance(value, (bool, str, bytes)) or value is None:
        raise ValueError("ASR timestamp boundaries must be finite numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("ASR timestamp boundaries must be finite numbers")
    return result


_MOSS_SEGMENT = re.compile(
    r"\[([0-9.]{1,32})\]\s*\[S[0-9]{1,15}\](.*?)\[([0-9.]{1,32})\](?=\s*(?:\[|$))",
    re.DOTALL,
)


def _moss_transcript(raw: str) -> Transcript:
    # The producer's streaming parser intentionally ignores malformed prefixes
    # and unfinished tails. Benchmark conversion must consume the entire output.
    segments = []
    position = 0
    while position < len(raw):
        match = _MOSS_SEGMENT.match(raw, position)
        if match is None:
            raise ValueError("MOSS returned malformed or incomplete timestamp output")
        start, text, end = match.groups()
        segment = {
            "text": text.strip(),
            "start": _boundary(float(start)),
            "end": _boundary(float(end)),
        }
        if segment["start"] < 0 or segment["end"] <= segment["start"]:
            raise ValueError("MOSS returned invalid timestamp boundaries")
        if segment["text"]:
            segments.append(segment)
        position = match.end()
        while position < len(raw) and raw[position].isspace():
            position += 1
    return Transcript(" ".join(item["text"] for item in segments), tuple(segments))


def _assert_device(runtime: object, expected: str, *, dtype: str | None = None) -> None:
    model = runtime
    if any(
        getattr(model, name, False)
        for name in ("is_quantized", "is_loaded_in_4bit", "is_loaded_in_8bit")
    ):
        raise RuntimeError("ASR quantization is forbidden")
    device_map = getattr(model, "hf_device_map", None)
    try:
        parameters = list(model.parameters())
        tensors = [*parameters, *getattr(model, "buffers", lambda: ())()]
        observed = {_device_name(item.device) for item in tensors}
        if isinstance(device_map, Mapping):
            observed.update(_device_name(value) for value in device_map.values())
        for _name, module in getattr(model, "named_modules", lambda: ())():
            hook = getattr(module, "_hf_hook", None)
            if hook is not None and (
                getattr(hook, "offload", False)
                or getattr(hook, "weights_map", None) is not None
            ):
                raise ASRPlacementError(
                    "ASR model offload hooks are forbidden", model, expected, observed
                )
    except (AttributeError, StopIteration, TypeError) as exc:
        raise RuntimeError("ASR runtime does not expose its model device") from exc
    if (
        not observed
        or len(observed) != 1
        or {value.split(":", 1)[0] for value in observed} != {expected}
    ):
        raise ASRPlacementError(
            f"ASR model used {sorted(observed)} instead of one requested {expected} device",
            model,
            expected,
            observed,
        )
    if dtype is not None and any(
        str(item.dtype).removeprefix("torch.") != dtype for item in parameters
    ):
        raise RuntimeError(f"ASR model weights do not all use protocol dtype {dtype}")


def _device_name(value: object) -> str:
    name = str(value)
    if name.isdigit():
        return f"cuda:{name}"
    if name == "cuda":
        return "cuda:0"
    if name.startswith("cpu:"):
        return "cpu"
    return name


def _runtime_version(backend: str) -> str:
    distribution = {
        "transformers": "transformers",
        "nemo": "nemo_toolkit",
        "moss": "moss-transcribe-diarize",
    }[backend]
    try:
        return version(distribution)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"Cannot identify installed ASR runtime {distribution}"
        ) from exc
