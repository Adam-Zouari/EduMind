"""Audio extraction using Whisper."""

from __future__ import annotations

import os
import time
from pathlib import Path

from ..config import FFMPEG_PATH, WHISPER_LANGUAGE, WHISPER_MODEL
from ..core.base_extractor import BaseExtractor, ExtractionResult

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WHISPER_AVAILABLE = False
try:
    import torch
    import whisper

    torch.set_num_threads(1)
    WHISPER_AVAILABLE = True
except Exception:
    whisper = None

if FFMPEG_PATH != "ffmpeg" and os.path.exists(FFMPEG_PATH):
    ffmpeg_dir = str(Path(FFMPEG_PATH).parent)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


class AudioExtractor(BaseExtractor):
    """Extract text from audio files using Whisper with model caching."""

    _whisper_models: dict[str, object] = {}

    def __init__(self, model_name: str = WHISPER_MODEL):
        super().__init__()
        self.model_name = model_name
        self.model = None

        if not WHISPER_AVAILABLE or whisper is None:
            self.logger.error("Whisper is not available. Install optional audio dependencies.")
            return

        if model_name not in AudioExtractor._whisper_models:
            self.logger.info(f"Loading Whisper model: {model_name} on CPU")
            try:
                AudioExtractor._whisper_models[model_name] = whisper.load_model(model_name, device="cpu")
            except Exception as exc:
                self.logger.error(f"Failed to load Whisper model: {exc}")
                return

        self.model = AudioExtractor._whisper_models.get(model_name)

    def extract(self, file_path: Path, **kwargs) -> ExtractionResult:
        start_time = time.time()
        self.logger.info(f"Transcribing audio: {file_path}")

        if self.model is None:
            return self._create_error_result(
                file_path,
                "Whisper model not available. Install optional audio dependencies.",
            )

        try:
            result = self.model.transcribe(str(file_path), language=WHISPER_LANGUAGE, verbose=False)
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
