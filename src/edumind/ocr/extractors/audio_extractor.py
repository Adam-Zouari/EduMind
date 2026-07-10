"""Audio extraction using Whisper."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

from ..config import WHISPER_LANGUAGE, WHISPER_MODEL
from ..core.base_extractor import BaseExtractor, ExtractionResult
from ._media_runtime import get_whisper_device, load_whisper_model


class AudioExtractor(BaseExtractor):
    """Extract text from audio files using Whisper."""

    def __init__(self, model_name: str = WHISPER_MODEL):
        super().__init__()
        self.model_name = model_name
        self.device = get_whisper_device()
        self.model, self.runtime_error = load_whisper_model(model_name)
        if self.model is None and self.runtime_error:
            self.logger.error(self.runtime_error)

    def extract(self, file_path: Path, **kwargs: object) -> ExtractionResult:
        start_time = time.time()
        self.logger.info(f"Transcribing audio: {file_path}")

        if self.model is None:
            return self._create_error_result(
                file_path,
                (
                    self.runtime_error
                    or "Whisper model not available. Install optional audio dependencies."
                ),
            )

        try:
            transcribe_kwargs: dict[str, object] = {"verbose": False}
            language = kwargs.get("language")
            raw_languages = kwargs.get("languages")
            if language is None and isinstance(raw_languages, list) and raw_languages:
                language = raw_languages[0]
            if language is None:
                language = WHISPER_LANGUAGE
            if isinstance(language, str) and language:
                transcribe_kwargs["language"] = language

            result = cast(Any, self.model).transcribe(str(file_path), **transcribe_kwargs)
            segments = [
                {"start": segment["start"], "end": segment["end"], "text": segment["text"]}
                for segment in result.get("segments", [])
            ]

            return ExtractionResult(
                text=result["text"],
                metadata={
                    "language": result.get("language", "unknown"),
                    "duration": result.get("duration", 0),
                    "num_segments": len(segments),
                    "segments": segments[:10],
                    "model": self.model_name,
                    "device": self.device,
                    "extractor": "whisper",
                },
                format_type="audio",
                file_path=str(file_path),
                extraction_time=time.time() - start_time,
                success=True,
            )
        except Exception as exc:
            self.logger.error(f"Audio extraction failed: {exc}")
            return self._create_error_result(file_path, str(exc))
