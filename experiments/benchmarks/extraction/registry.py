"""Experiment-only extractor registration over production contracts."""

from __future__ import annotations

from edumind.extraction import SourceKind
from edumind.extraction.extractors.audio import WhisperExtractor
from edumind.extraction.extractors.document import DoclingExtractor
from edumind.extraction.extractors.video import VideoExtractor
from edumind.extraction.registry import ExtractorRegistration, ExtractorRegistry

from experiments.benchmarks.extraction.audio.adapters import ExperimentalAudioExtractor
from experiments.benchmarks.extraction.document.adapters import ExperimentalDocumentExtractor

ASR_MODELS = {
    "canary-180m": "nvidia/canary-180m-flash",
    "parakeet-tdt-0.6b-v2": "nvidia/parakeet-tdt-0.6b-v2",
    "moss-transcribe-diarize": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
    "qwen3-asr-1.7b-aligned": "Qwen/Qwen3-ASR-1.7B-hf",
}


def build_experiment_registry() -> ExtractorRegistry:
    registry = ExtractorRegistry()
    registrations = [
        ExtractorRegistration(
            "docling-standard",
            frozenset({SourceKind.IMAGE, SourceKind.PDF, SourceKind.DOCX}),
            lambda: DoclingExtractor("from-lock"),
        ),
        ExtractorRegistration(
            "docling-vlm-granite-258m",
            frozenset({SourceKind.IMAGE, SourceKind.PDF}),
            lambda: ExperimentalDocumentExtractor("docling-vlm-granite-258m", "from-lock"),
        ),
        ExtractorRegistration(
            "paddleocr-vl-1.6",
            frozenset({SourceKind.IMAGE, SourceKind.PDF}),
            lambda: ExperimentalDocumentExtractor("paddleocr-vl-1.6", "from-lock"),
        ),
        ExtractorRegistration(
            "whisper-small-en-control",
            frozenset({SourceKind.AUDIO}),
            lambda: WhisperExtractor("from-lock"),
        ),
    ]
    registrations.extend(
        ExtractorRegistration(
            engine,
            frozenset({SourceKind.AUDIO}),
            lambda engine=engine, model=model: ExperimentalAudioExtractor(
                engine, model, "from-lock"
            ),
        )
        for engine, model in ASR_MODELS.items()
    )
    registrations.extend(
        ExtractorRegistration(
            f"video-{routing}",
            frozenset({SourceKind.VIDEO}),
            lambda routing=routing: VideoExtractor(
                routing,
                audio_factory=_audio_factory,
                image_factory=_image_factory,
            ),
        )
        for routing in ("fixed", "scene", "hybrid")
    )
    for registration in registrations:
        registry.register(registration)
    return registry


def _audio_factory(engine: str, revision: str):
    if engine == "whisper-small-en-control":
        return WhisperExtractor(revision)
    try:
        model = ASR_MODELS[engine]
    except KeyError as exc:
        raise ValueError(f"Unknown experimental ASR engine: {engine}") from exc
    return ExperimentalAudioExtractor(engine, model, revision)


def _image_factory(engine: str, revision: str):
    if engine == "docling-standard":
        return DoclingExtractor(revision)
    return ExperimentalDocumentExtractor(engine, revision)
