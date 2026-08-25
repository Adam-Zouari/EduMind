"""ASR integrations that remain experimental until a benchmark promotion."""

from __future__ import annotations

import gc
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from edumind.extraction.contracts import ExtractedDocument, ExtractionRequest, SourceKind
from edumind.extraction.errors import ExtractionBackendError, MissingDependencyError
from edumind.extraction.extractors.audio import model_directory
from edumind.extraction.extractors.base import build_document


class ExperimentalAudioExtractor:
    supported_kinds = frozenset({SourceKind.AUDIO})

    def __init__(self, candidate: str, model: str, revision: str) -> None:
        if candidate not in {
            "canary-180m",
            "parakeet-tdt-0.6b-v2",
            "moss-transcribe-diarize",
            "qwen3-asr-1.7b-aligned",
        }:
            raise ValueError(f"Unknown experimental ASR candidate: {candidate}")
        self.candidate = candidate
        self.engine = candidate
        self.model = model
        self.name = candidate
        self.revision = revision
        self._runtime: Any | None = None

    def extract(self, request: ExtractionRequest, kind: SourceKind) -> ExtractedDocument:
        if request.profile is None:
            raise ValueError("Resolved extraction profile is required")
        started = time.perf_counter()
        try:
            if self.candidate in {"canary-180m", "parakeet-tdt-0.6b-v2"}:
                texts, timestamps = self._nemo(request)
            elif self.candidate == "moss-transcribe-diarize":
                texts, timestamps = self._moss(request)
            else:
                texts, timestamps = self._qwen_aligned(request)
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
            texts,
            timestamps=timestamps,
            metadata={
                "candidate": self.candidate,
                "model": self.model,
                "engine_revision": request.profile.engine_revision,
                "alignment_execution": (
                    "sequential" if self.candidate == "qwen3-asr-1.7b-aligned" else "native"
                ),
            },
            seconds=time.perf_counter() - started,
        )

    def _nemo(self, request: ExtractionRequest):
        try:
            import nemo.collections.asr as nemo_asr
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("NVIDIA NeMo ASR is required") from exc
        profile = request.profile
        assert profile is not None
        path = model_directory(request)
        checkpoints = sorted(path.glob("*.nemo"))
        if not checkpoints:
            raise FileNotFoundError(f"No pinned .nemo checkpoint exists under {path}")
        if self._runtime is None:
            self._runtime = nemo_asr.models.ASRModel.restore_from(
                restore_path=str(checkpoints[0]), map_location=profile.device
            )
        output = self._runtime.transcribe([str(request.source_path)], timestamps=True)[0]
        timestamp_payload = getattr(output, "timestamp", {}) or {}
        segments = timestamp_payload.get("segment", [])
        if segments:
            return (
                [str(item.get("segment", item.get("text", ""))).strip() for item in segments],
                [(float(item["start"]), float(item["end"])) for item in segments],
            )
        return [str(getattr(output, "text", output)).strip()], [(0.0, 0.0)]

    def _moss(self, request: ExtractionRequest):
        try:
            import torch
            from moss_transcribe_diarize import parse_transcript
            from moss_transcribe_diarize.attention import load_model_with_attention_fallback
            from moss_transcribe_diarize.inference_utils import (
                build_transcription_messages,
                generate_transcription,
            )
            from transformers import AutoProcessor
        except ModuleNotFoundError as exc:
            raise MissingDependencyError(
                "The pinned MOSS-Transcribe-Diarize runtime is required"
            ) from exc
        profile = request.profile
        assert profile is not None
        path = model_directory(request)
        if self._runtime is None:
            dtype = torch.bfloat16 if profile.device == "cuda" else torch.float32
            processor = AutoProcessor.from_pretrained(
                str(path), trust_remote_code=True, local_files_only=True
            )
            device = torch.device(profile.device)
            model, attention_report = load_model_with_attention_fallback(
                str(path), device=device, dtype=dtype
            )
            model = model.to(device=device, dtype=dtype).eval()
            self._runtime = (model, processor, attention_report, device, dtype)
        model, processor, attention_report, device, dtype = self._runtime
        messages = build_transcription_messages(str(request.source_path))
        result = generate_transcription(
            model,
            processor,
            messages,
            max_new_tokens=2048,
            do_sample=False,
            device=device,
            dtype=dtype,
            attention_report=attention_report,
        )
        raw = str(result.get("text", "")) if isinstance(result, Mapping) else str(result)
        segments = parse_transcript(raw)
        texts: list[str] = []
        timestamps: list[tuple[float, float]] = []
        for segment in segments:
            text = str(getattr(segment, "text", "")).strip()
            if text:
                texts.append(text)
                timestamps.append(
                    (float(getattr(segment, "start", 0.0)), float(getattr(segment, "end", 0.0)))
                )
        return (texts, timestamps) if texts else ([raw.strip()], [(0.0, 0.0)])

    def _qwen_aligned(self, request: ExtractionRequest):
        try:
            import torch
            from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("The official qwen-asr runtime is required") from exc
        profile = request.profile
        assert profile is not None
        path = model_directory(request)
        aligner_path = Path(str(request.options.get("aligner_model_path", "")))
        if not aligner_path.is_dir():
            raise FileNotFoundError("The pinned Qwen forced-aligner snapshot is missing")
        dtype = torch.bfloat16 if profile.device == "cuda" else torch.float32
        asr = Qwen3ASRModel.from_pretrained(
            str(path),
            dtype=dtype,
            device_map=profile.device,
            max_inference_batch_size=1,
            max_new_tokens=2048,
        )
        transcription = asr.transcribe(audio=str(request.source_path), language="English")
        text = str(getattr(transcription[0], "text", transcription[0])).strip()
        del asr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        aligner = Qwen3ForcedAligner.from_pretrained(
            str(aligner_path), dtype=dtype, device_map=profile.device
        )
        aligned = aligner.align(audio=str(request.source_path), text=text, language="English")
        del aligner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        words = aligned[0] if isinstance(aligned, Sequence) and aligned else aligned
        items = words if isinstance(words, Sequence) and not isinstance(words, str) else []
        texts: list[str] = []
        timestamps: list[tuple[float, float]] = []
        for item in items:
            word = getattr(item, "text", getattr(item, "word", ""))
            start = getattr(item, "start", getattr(item, "start_time", 0.0))
            end = getattr(item, "end", getattr(item, "end_time", start))
            if str(word).strip():
                texts.append(str(word).strip())
                timestamps.append((float(start), float(end)))
        return (texts, timestamps) if texts else ([text], [(0.0, 0.0)])

