"""The five executable ASR profiles used by the audio benchmark."""

from __future__ import annotations

import gc
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from edumind.extraction.contracts import (
    ExtractedDocument,
    ExtractionRequest,
    ExtractionWarning,
    SourceKind,
)
from edumind.extraction.errors import ExtractionBackendError, MissingDependencyError
from edumind.extraction.extractors.audio import load_whisper_runtime, transcribe_whisper
from edumind.extraction.extractors.base import build_document


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
        ASRProfile(
            "qwen3-asr-1.7b-aligned",
            "Qwen/Qwen3-ASR-1.7B-hf",
            "qwen-transformers",
            "deterministic-english",
            "qwen-forced-aligner-0.6b",
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
    aligner_path = None
    if candidate == "qwen3-asr-1.7b-aligned":
        aligner_path = _forced_aligner_path(entry)
    runtime_type = {
        "transformers": WhisperRuntime,
        "nemo": NemoRuntime,
        "moss": MossRuntime,
        "qwen-transformers": QwenRuntime,
    }[profile.backend]
    return runtime_type(profile, model_path, device, aligner_path)


class BaseRuntime:
    def __init__(
        self,
        profile: ASRProfile,
        model_path: Path,
        device: str,
        aligner_path: Path | None = None,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("ASR device must be cpu or cuda")
        self.profile = profile
        self.model_path = model_path
        self.device = device
        self.aligner_path = aligner_path
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
        }


class WhisperRuntime(BaseRuntime):
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
        return {**super().parameters(), "attention": str(self.attention_report)}

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


class QwenRuntime(BaseRuntime):
    def load(self) -> None:
        self._load_asr()
        self._runtime = None
        _release_memory()
        aligner = self._load_aligner()
        del aligner
        _release_memory()

    def _load_asr(self) -> None:
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("Transformers 5.13 or newer is required") from exc
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.dtype = str(dtype).removeprefix("torch.")
        processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            str(self.model_path), dtype=dtype, local_files_only=True
        ).to(self.device).eval()
        _assert_device(model, self.device)
        self._runtime = (model, processor)

    def transcribe(self, source: Path) -> Transcript:
        if self._runtime is None:
            self._load_asr()
        import torch

        model, processor = self._runtime
        inputs = processor.apply_transcription_request(
            audio=str(source), language="English"
        ).to(model.device, model.dtype)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        parsed = processor.decode(generated, return_format="parsed")[0]
        text = str(parsed.get("transcription", "")).strip()
        self._runtime = None
        del model, processor
        _release_memory()
        if not text:
            return Transcript("", ())
        aligner_model, aligner_processor = self._load_aligner()
        try:
            aligner_inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
                audio=str(source), transcript=text, language="English"
            )
            aligner_inputs = aligner_inputs.to(aligner_model.device, aligner_model.dtype)
            with torch.inference_mode():
                outputs = aligner_model(**aligner_inputs)
            aligned = aligner_processor.decode_forced_alignment(
                logits=outputs.logits,
                input_ids=aligner_inputs["input_ids"],
                word_lists=word_lists,
                timestamp_token_id=aligner_model.config.timestamp_token_id,
            )[0]
        finally:
            del aligner_model, aligner_processor
            _release_memory()
        if not isinstance(aligned, Sequence) or isinstance(aligned, (str, bytes)) or not aligned:
            raise RuntimeError("Qwen forced aligner returned no timestamped items")
        segments = tuple(
            {
                "text": str(_item_value(item, "text", _item_value(item, "word", ""))).strip(),
                "start": float(_item_value(item, "start_time", _item_value(item, "start", -1))),
                "end": float(_item_value(item, "end_time", _item_value(item, "end", -1))),
            }
            for item in aligned
            if str(_item_value(item, "text", _item_value(item, "word", ""))).strip()
        )
        return Transcript(text, segments)

    def _load_aligner(self):
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoProcessor
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("Transformers 5.13 or newer is required") from exc
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(
            str(self.aligner_path), local_files_only=True
        )
        model = AutoModelForTokenClassification.from_pretrained(
            str(self.aligner_path), dtype=dtype, local_files_only=True
        ).to(self.device).eval()
        _assert_device(model, self.device)
        return model, processor


class ExperimentalAudioExtractor:
    """Pipeline adapter retained for video runs with a frozen ASR profile."""

    supported_kinds = frozenset({SourceKind.AUDIO})

    def __init__(self, candidate: str, model: str, _revision: str) -> None:
        profile = ASR_PROFILES.get(candidate)
        if profile is None or profile.model != model or candidate == "whisper-small-en-control":
            raise ValueError(f"Unknown experimental ASR candidate: {candidate}")
        self.candidate = self.engine = self.name = candidate
        self.model = model
        self._runtime: BaseRuntime | None = None

    def extract(self, request: ExtractionRequest, kind: SourceKind) -> ExtractedDocument:
        if request.profile is None:
            raise ValueError("Resolved extraction profile is required")
        started = time.perf_counter()
        try:
            if self._runtime is None:
                lock_entry: dict[str, object] = {
                    "model_path": request.options.get("model_path", "")
                }
                if request.options.get("aligner_model_path"):
                    lock_entry["submodels"] = [
                        {
                            "role": "forced-aligner",
                            "model_path": request.options["aligner_model_path"],
                        }
                    ]
                self._runtime = build_runtime(
                    self.candidate, {self.model: lock_entry}, request.profile.device
                )
                self._runtime.load()
            transcript = self._runtime.transcribe(request.source_path)
        except MissingDependencyError:
            raise
        except Exception as exc:
            raise ExtractionBackendError(
                f"Audio extraction failed with {self.name}", detail=str(exc)
            ) from exc
        return build_document(
            request,
            kind,
            request.profile,
            [str(segment["text"]) for segment in transcript.segments] or [transcript.text],
            timestamps=[
                (float(segment["start"]), float(segment["end"]))
                for segment in transcript.segments
            ],
            separators=" ",
            metadata={
                "candidate": self.candidate,
                "model": self.model,
                "engine_revision": request.profile.engine_revision,
                "alignment_execution": (
                    "sequential" if self.candidate == "qwen3-asr-1.7b-aligned" else "native"
                ),
            },
            warnings=[
                ExtractionWarning("asr_runtime_warning", warning)
                for warning in transcript.warnings
            ],
            seconds=time.perf_counter() - started,
        )


def _nemo_segment(item: Mapping[str, object]) -> dict[str, object]:
    return {
        "text": str(item.get("segment", item.get("word", item.get("text", "")))).strip(),
        "start": float(item.get("start", -1)),
        "end": float(item.get("end", -1)),
    }


def _item_value(item: object, name: str, default: object) -> object:
    return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)


def _forced_aligner_path(entry: Mapping[str, object]) -> Path:
    submodels = entry.get("submodels", [])
    if isinstance(submodels, Sequence) and not isinstance(submodels, (str, bytes)):
        for submodel in submodels:
            if isinstance(submodel, Mapping) and submodel.get("role") == "forced-aligner":
                path = Path(str(submodel.get("model_path", "")))
                if path.is_dir():
                    return path
    raise FileNotFoundError("The pinned Qwen forced-aligner snapshot is missing")


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
        "qwen-transformers": "transformers",
    }[backend]
    try:
        return version(distribution)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"Cannot identify installed ASR runtime {distribution}") from exc
