"""Lazy local speech-to-text implementations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..contracts import ExtractedDocument, ExtractionRequest, SourceKind
from ..errors import ExtractionBackendError, MissingDependencyError
from .base import build_document


class AudioExtractor:
    supported_kinds = frozenset({SourceKind.AUDIO})

    def __init__(self, engine: str, model: str, compute_type: str = "int8") -> None:
        self.engine = engine
        self.model = model
        self.compute_type = compute_type
        self.name = f"{engine}-{model}-{compute_type}"
        self.revision = "unpinned"
        self._runtime: Any | None = None

    def extract(self, request: ExtractionRequest, kind: SourceKind) -> ExtractedDocument:
        if request.profile is None:
            raise ValueError("Resolved extraction profile is required")
        started = time.perf_counter()
        try:
            texts, timestamps = self._transcribe(request)
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
                "engine": self.engine,
                "model": self.model,
                "compute_type": self.compute_type,
                "engine_revision": request.profile.engine_revision,
            },
            seconds=time.perf_counter() - started,
        )

    def _transcribe(
        self, request: ExtractionRequest
    ) -> tuple[list[str], list[tuple[float, float]]]:
        profile = request.profile
        if profile is None:
            raise ValueError("Resolved extraction profile is required")
        if self.engine == "faster-whisper":
            try:
                from faster_whisper import WhisperModel
            except ModuleNotFoundError as exc:
                raise MissingDependencyError(
                    "faster-whisper is required; install requirements/app.lock"
                ) from exc
            configured_path = request.options.get("model_path")
            model_path = Path(str(configured_path)).expanduser() if configured_path else None
            if model_path is None or not model_path.is_dir():
                raise FileNotFoundError(
                    "faster-whisper weights are not prepared locally; run `python "
                    "experiments/benchmarks/prepare.py extraction-models`"
                )
            if self._runtime is None:
                self._runtime = WhisperModel(
                    str(model_path),
                    device=profile.device,
                    compute_type=self.compute_type,
                    local_files_only=True,
                )
            segments, _ = self._runtime.transcribe(
                str(request.source_path),
                language="en",
                vad_filter=bool(request.options.get("vad", False)),
            )
            resolved = list(segments)
            return [segment.text.strip() for segment in resolved], [
                (float(segment.start), float(segment.end)) for segment in resolved
            ]
        if self.engine == "openai-whisper":
            try:
                import whisper
            except ModuleNotFoundError as exc:
                raise MissingDependencyError(
                    "openai-whisper is required; install requirements/benchmarks.lock"
                ) from exc
            configured_path = request.options.get("model_path")
            model_path = Path(str(configured_path)).expanduser() if configured_path else None
            if model_path is None or not model_path.is_file():
                raise FileNotFoundError(
                    "OpenAI Whisper weights are not prepared locally; set model_path from the "
                    "model lock created by the preparation command"
                )
            if self._runtime is None:
                self._runtime = whisper.load_model(str(model_path), device=profile.device)
            result = self._runtime.transcribe(str(request.source_path), language="en")
            raw_segments = result.get("segments", [])
            return [str(item.get("text", "")).strip() for item in raw_segments], [
                (float(item.get("start", 0.0)), float(item.get("end", 0.0)))
                for item in raw_segments
            ]
        if self.engine == "transformers-asr":
            return self._transformers_asr(request)
        if self.engine == "nemo-canary":
            return self._nemo_canary(request)
        raise ValueError(f"Unknown ASR engine: {self.engine}")

    def _transformers_asr(
        self, request: ExtractionRequest
    ) -> tuple[list[str], list[tuple[float, float]]]:
        try:
            import torch
            from transformers import pipeline
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("Transformers is required for this ASR candidate") from exc
        profile = request.profile
        assert profile is not None
        model_path = Path(str(request.options.get("model_path", ""))).expanduser()
        if not model_path.is_dir():
            raise FileNotFoundError(
                "ASR weights are not prepared; run `python "
                "experiments/benchmarks/prepare.py extraction-models`"
            )
        if self._runtime is None:
            dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }.get(self.compute_type, torch.float32)
            self._runtime = pipeline(
                "automatic-speech-recognition",
                model=str(model_path),
                device=0 if profile.device == "cuda" else -1,
                dtype=dtype,
                model_kwargs={"local_files_only": True},
            )
        result = self._runtime(str(request.source_path), return_timestamps=True)
        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        texts: list[str] = []
        timestamps: list[tuple[float, float]] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            timestamp = chunk.get("timestamp")
            if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                continue
            start = float(timestamp[0] or 0.0)
            end = float(timestamp[1] if timestamp[1] is not None else start)
            texts.append(str(chunk.get("text", "")).strip())
            timestamps.append((start, end))
        if texts:
            return texts, timestamps
        text = str(result.get("text", "")).strip() if isinstance(result, dict) else str(result)
        return [text], [(0.0, 0.0)]

    def _nemo_canary(
        self, request: ExtractionRequest
    ) -> tuple[list[str], list[tuple[float, float]]]:
        try:
            from nemo.collections.speechlm2.models import SALM
        except ModuleNotFoundError as exc:
            raise MissingDependencyError(
                "Canary-Qwen requires NVIDIA NeMo; install the candidate exactly as described "
                "in guide.md"
            ) from exc
        model_directory = Path(str(request.options.get("model_path", ""))).expanduser()
        if not model_directory.is_dir() or not (model_directory / "model.safetensors").is_file():
            raise FileNotFoundError(
                "The prepared Canary Hugging Face snapshot is missing model.safetensors"
            )
        if self._runtime is None:
            # Canary-Qwen is published as a Hugging Face safetensors repository,
            # not a legacy .nemo archive. Loading the pinned local directory keeps
            # inference offline and prevents `main` from moving after preparation.
            self._runtime = SALM.from_pretrained(str(model_directory))
            if hasattr(self._runtime, "to"):
                self._runtime = self._runtime.to(request.profile.device)
        answer_ids = self._runtime.generate(
            prompts=[
                [
                    {
                        "role": "user",
                        "content": f"Transcribe the following: {self._runtime.audio_locator_tag}",
                        "audio": [str(request.source_path)],
                    }
                ]
            ],
            max_new_tokens=256,
        )
        text = str(self._runtime.tokenizer.ids_to_text(answer_ids[0].cpu())).strip()
        return [text], [(0.0, 0.0)]
