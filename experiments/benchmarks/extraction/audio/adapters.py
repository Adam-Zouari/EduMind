"""The four executable ASR profiles used by the audio benchmark."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from edumind.extraction.errors import MissingDependencyError
from edumind.extraction.extractors.audio import load_whisper_runtime, transcribe_whisper
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
    if not entry:
        raise RuntimeError(f"The model lock has no entry for {profile.model}")
    model_path = Path(str(entry.get("model_path", "")))
    if not model_path.is_dir():
        raise FileNotFoundError(f"Pinned ASR snapshot is missing: {model_path}")
    runtime_type = {
        "transformers": WhisperRuntime,
        "nemo": NemoRuntime,
        "moss": MossRuntime,
    }[profile.backend]
    return runtime_type(profile, model_path, device, protocol)


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
        self.dtype = protocol.dtype(profile.candidate, device)

    def load(self) -> None:
        raise NotImplementedError

    def transcribe(self, source: Path) -> Transcript:
        raise NotImplementedError

    def close(self) -> None:
        self._runtime = None
        _release_memory()

    def parameters(self) -> dict[str, object]:
        return {
            "candidate": self.profile.candidate,
            "model": self.profile.model,
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
                )
            ),
        }


class WhisperRuntime(BaseRuntime):
    def parameters(self) -> dict[str, object]:
        return {
            **super().parameters(),
            "return_timestamps": self.profile.decoding["return_timestamps"],
            "do_sample": self.profile.decoding["do_sample"],
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

    def transcribe(self, source: Path) -> Transcript:
        result = transcribe_whisper(
            self._runtime,
            source,
            return_timestamps=str(self.profile.decoding["return_timestamps"]),
            do_sample=bool(self.profile.decoding["do_sample"]),
        )
        complete = tuple(
            segment
            for segment in result.segments
            if segment.get("start") is not None and segment.get("end") is not None
        )
        missing = len(result.segments) - len(complete)
        warnings = (
            (f"{missing} Whisper chunk(s) lacked complete timestamp boundaries",)
            if missing
            else ()
        )
        return Transcript(result.text, complete, warnings)


class NemoRuntime(BaseRuntime):
    def parameters(self) -> dict[str, object]:
        values = {
            **super().parameters(),
            "timestamps": self.profile.decoding["timestamps"],
        }
        if self.profile.candidate == "canary-180m":
            values.update(
                {
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
        if not checkpoints:
            raise FileNotFoundError(
                f"No pinned .nemo checkpoint exists under {self.model_path}"
            )
        self._runtime = nemo_asr.models.ASRModel.restore_from(
            restore_path=str(checkpoints[0]), map_location=self.device
        )
        expected_dtype = self.protocol.dtype(self.profile.candidate, self.device)
        self._runtime = self._runtime.to(
            device=self.device,
            dtype=getattr(torch, expected_dtype),
        ).eval()
        _assert_device(self._runtime, self.device)
        decoding = self._runtime.cfg.decoding
        if self.profile.candidate == "canary-180m":
            decoding.beam.beam_size = int(self.profile.decoding["beam_size"])
        else:
            decoding.strategy = str(self.profile.decoding["decoding_strategy"])
        self._runtime.change_decoding_strategy(decoding)
        self.dtype = str(next(self._runtime.parameters()).dtype).removeprefix("torch.")
        if self.dtype != expected_dtype:
            raise RuntimeError(
                f"NeMo loaded {self.dtype} instead of protocol dtype {expected_dtype}"
            )

    def transcribe(self, source: Path) -> Transcript:
        arguments: dict[str, object] = {
            "batch_size": self.protocol.batch_size,
            "timestamps": self.profile.decoding["timestamps"],
        }
        if self.profile.candidate == "canary-180m":
            arguments["pnc"] = (
                "True"
                if self.profile.decoding["punctuation_and_capitalization"]
                else "False"
            )
        output = self._runtime.transcribe([str(source)], **arguments)[0]
        timestamp_payload = getattr(output, "timestamp", {}) or {}
        segments = timestamp_payload.get("segment", [])
        if not segments:
            segments = timestamp_payload.get("word", [])
        return Transcript(
            str(getattr(output, "text", output)).strip(),
            tuple(_nemo_segment(item) for item in segments),
        )


class MossRuntime(BaseRuntime):
    def load(self) -> None:
        try:
            import torch
            from moss_transcribe_diarize.attention import (
                load_model_with_attention_fallback,
            )
            from transformers import AutoProcessor
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
        model, attention_report = load_model_with_attention_fallback(
            str(self.model_path), device=device, dtype=dtype
        )
        self._runtime = (
            model.to(device=device, dtype=dtype).eval(),
            processor,
            attention_report,
            device,
            dtype,
        )
        _assert_device(self._runtime[0], self.device)
        self.dtype = str(dtype).removeprefix("torch.")
        self.attention_report = attention_report

    def parameters(self) -> dict[str, object]:
        return {
            **super().parameters(),
            "attention": str(self.attention_report),
            "max_new_tokens": self.profile.decoding["max_new_tokens"],
            "do_sample": self.profile.decoding["do_sample"],
        }

    def transcribe(self, source: Path) -> Transcript:
        try:
            from moss_transcribe_diarize import parse_transcript
            from moss_transcribe_diarize.inference_utils import (
                build_transcription_messages,
                generate_transcription,
            )
        except ModuleNotFoundError as exc:
            raise MissingDependencyError(
                "The pinned MOSS-Transcribe-Diarize runtime is required"
            ) from exc
        model, processor, attention_report, device, dtype = self._runtime
        result = generate_transcription(
            model,
            processor,
            build_transcription_messages(str(source)),
            max_new_tokens=int(self.profile.decoding["max_new_tokens"]),
            do_sample=bool(self.profile.decoding["do_sample"]),
            device=device,
            dtype=dtype,
            attention_report=attention_report,
        )
        raw = (
            str(result.get("text", "")) if isinstance(result, Mapping) else str(result)
        )
        parsed = parse_transcript(raw)
        segments = tuple(
            {
                "text": str(getattr(segment, "text", "")).strip(),
                "start": float(getattr(segment, "start", -1)),
                "end": float(getattr(segment, "end", -1)),
            }
            for segment in parsed
            if str(getattr(segment, "text", "")).strip()
        )
        return Transcript(
            " ".join(str(item["text"]) for item in segments) or raw.strip(), segments
        )


def _nemo_segment(item: Mapping[str, object]) -> dict[str, object]:
    return {
        "text": str(
            item.get("segment", item.get("word", item.get("text", "")))
        ).strip(),
        "start": float(item.get("start", -1)),
        "end": float(item.get("end", -1)),
    }


def _release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass


def _assert_device(runtime: object, expected: str) -> None:
    model = getattr(runtime, "model", runtime)
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        observed = {str(value).split(":", 1)[0] for value in device_map.values()}
        if observed != {expected}:
            raise RuntimeError(
                f"ASR model used {sorted(observed)} instead of the requested {expected} device"
            )
        return
    try:
        observed = str(next(model.parameters()).device).split(":", 1)[0]
    except (AttributeError, StopIteration, TypeError) as exc:
        raise RuntimeError("ASR runtime does not expose its model device") from exc
    if observed != expected:
        raise RuntimeError(
            f"ASR model used {observed} instead of the requested {expected} device"
        )


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
