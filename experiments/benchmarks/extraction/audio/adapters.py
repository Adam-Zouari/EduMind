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


@dataclass(frozen=True)
class ASRProfile:
    candidate: str
    model: str
    backend: str
    decoder: str
    timestamp_method: str


@dataclass(frozen=True)
class Transcript:
    text: str
    segments: tuple[Mapping[str, object], ...]
    warnings: tuple[str, ...] = ()


ASR_PROFILES = {
    profile.candidate: profile
    for profile in (
        ASRProfile(
            "whisper-small-en-control",
            "openai/whisper-small.en",
            "transformers",
            "greedy",
            "native-word",
        ),
        ASRProfile(
            "canary-180m",
            "nvidia/canary-180m-flash",
            "nemo",
            "beam-1-pnc",
            "native-segment",
        ),
        ASRProfile(
            "parakeet-tdt-0.6b-v2",
            "nvidia/parakeet-tdt-0.6b-v2",
            "nemo",
            "greedy",
            "native-segment",
        ),
        ASRProfile(
            "moss-transcribe-diarize",
            "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            "moss",
            "deterministic",
            "native-segment",
        ),
    )
}


def build_runtime(
    candidate: str, model_lock: Mapping[str, Mapping[str, object]], device: str
):
    try:
        profile = ASR_PROFILES[candidate]
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
    return runtime_type(profile, model_path, device)


class BaseRuntime:
    def __init__(
        self,
        profile: ASRProfile,
        model_path: Path,
        device: str,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("ASR device must be cpu or cuda")
        self.profile = profile
        self.model_path = model_path
        self.device = device
        self._runtime: Any | None = None
        self.dtype = "float32" if device == "cpu" else "float16"

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
            "language": "English",
            "decoder": self.profile.decoder,
            "timestamp_method": self.profile.timestamp_method,
            "sample_rate_hz": 16_000,
            "batch_size": 1,
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
            "return_timestamps": "word",
            "do_sample": False,
        }

    def load(self) -> None:
        self._runtime, self.dtype = load_whisper_runtime(self.model_path, self.device)

    def transcribe(self, source: Path) -> Transcript:
        result = transcribe_whisper(self._runtime, source)
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
            "timestamps": True,
        }
        if self.profile.candidate == "canary-180m":
            values.update({"beam_size": 1, "punctuation_and_capitalization": True})
        else:
            values.update(
                {"decoding_strategy": "greedy_batch", "timestamp_level": "segment_then_word"}
            )
        return values

    def load(self) -> None:
        try:
            import nemo.collections.asr as nemo_asr
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("NVIDIA NeMo ASR is required") from exc
        checkpoints = sorted(self.model_path.glob("*.nemo"))
        if not checkpoints:
            raise FileNotFoundError(f"No pinned .nemo checkpoint exists under {self.model_path}")
        self._runtime = nemo_asr.models.ASRModel.restore_from(
            restore_path=str(checkpoints[0]), map_location=self.device
        )
        self._runtime = self._runtime.to(self.device).eval()
        _assert_device(self._runtime, self.device)
        decoding = self._runtime.cfg.decoding
        if self.profile.candidate == "canary-180m":
            decoding.beam.beam_size = 1
        else:
            decoding.strategy = "greedy_batch"
        self._runtime.change_decoding_strategy(decoding)
        self.dtype = str(next(self._runtime.parameters()).dtype).removeprefix("torch.")

    def transcribe(self, source: Path) -> Transcript:
        arguments: dict[str, object] = {"batch_size": 1, "timestamps": True}
        if self.profile.candidate == "canary-180m":
            arguments["pnc"] = "True"
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
            from moss_transcribe_diarize.attention import load_model_with_attention_fallback
            from transformers import AutoProcessor
        except ModuleNotFoundError as exc:
            raise MissingDependencyError(
                "The pinned MOSS-Transcribe-Diarize runtime is required"
            ) from exc
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
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
            "max_new_tokens": 2048,
            "do_sample": False,
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
            max_new_tokens=2048,
            do_sample=False,
            device=device,
            dtype=dtype,
            attention_report=attention_report,
        )
        raw = str(result.get("text", "")) if isinstance(result, Mapping) else str(result)
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
        return Transcript(" ".join(str(item["text"]) for item in segments) or raw.strip(), segments)


def _nemo_segment(item: Mapping[str, object]) -> dict[str, object]:
    return {
        "text": str(item.get("segment", item.get("word", item.get("text", "")))).strip(),
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
        raise RuntimeError(f"Cannot identify installed ASR runtime {distribution}") from exc
