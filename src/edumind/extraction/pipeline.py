"""Production extraction classifier, router, cache, and normalization pipeline."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from edumind.common.config import Settings, load_settings

from .cache import ExtractionCache
from .contracts import ExtractedDocument, ExtractionProfile, ExtractionRequest, SourceKind
from .detection import classify_source
from .normalization import normalize_document
from .registry import ExtractorRegistration, ExtractorRegistry

DEFAULT_PROFILE_BY_KIND = {
    SourceKind.IMAGE: ExtractionProfile("image-baseline", "tesseract-5", "5", "document"),
    SourceKind.PDF: ExtractionProfile(
        "pdf-hybrid-baseline", "hybrid-pdf", "1", "document", routing="page-hybrid"
    ),
    SourceKind.DOCX: ExtractionProfile(
        "docx-baseline", "python-docx", "1", normalization="conservative"
    ),
    SourceKind.AUDIO: ExtractionProfile(
        "audio-baseline",
        "faster-whisper-base-int8",
        "from-lock",
        device="cpu",
    ),
    SourceKind.VIDEO: ExtractionProfile(
        "video-baseline", "video-hybrid", "from-lock", device="cpu", routing="hybrid"
    ),
}


def build_default_registry() -> ExtractorRegistry:
    from .extractors.audio import AudioExtractor
    from .extractors.docx import DOCXExtractor
    from .extractors.image import ImageExtractor
    from .extractors.pdf import PDFExtractor
    from .extractors.video import VideoExtractor

    registry = ExtractorRegistry()
    registrations = [
        ExtractorRegistration(
            "tesseract-5",
            frozenset({SourceKind.IMAGE}),
            lambda: ImageExtractor("tesseract-5", "5"),
        ),
        ExtractorRegistration(
            "paddleocr-v5-mobile",
            frozenset({SourceKind.IMAGE}),
            lambda: ImageExtractor("paddleocr-v5-mobile", "v5"),
        ),
        ExtractorRegistration(
            "paddleocr-v5-server",
            frozenset({SourceKind.IMAGE}),
            lambda: ImageExtractor("paddleocr-v5-server", "v5"),
        ),
        ExtractorRegistration(
            "doctr-fast-parseq",
            frozenset({SourceKind.IMAGE}),
            lambda: ImageExtractor("doctr-fast-parseq", "unpinned"),
        ),
        ExtractorRegistration(
            "pypdf", frozenset({SourceKind.PDF}), lambda: PDFExtractor("pypdf")
        ),
        ExtractorRegistration(
            "pdfplumber",
            frozenset({SourceKind.PDF}),
            lambda: PDFExtractor("pdfplumber"),
        ),
        ExtractorRegistration(
            "docling-pdf",
            frozenset({SourceKind.PDF}),
            lambda: PDFExtractor("docling-pdf"),
        ),
        ExtractorRegistration(
            "hybrid-pdf",
            frozenset({SourceKind.PDF}),
            lambda: PDFExtractor("hybrid-pdf", "1"),
        ),
        ExtractorRegistration(
            "ocr-pdf",
            frozenset({SourceKind.PDF}),
            lambda: PDFExtractor("ocr-pdf", "1"),
        ),
        ExtractorRegistration(
            "python-docx",
            frozenset({SourceKind.DOCX}),
            lambda: DOCXExtractor("python-docx", "1"),
        ),
        ExtractorRegistration(
            "mammoth",
            frozenset({SourceKind.DOCX}),
            lambda: DOCXExtractor("mammoth"),
        ),
        ExtractorRegistration(
            "docling-docx",
            frozenset({SourceKind.DOCX}),
            lambda: DOCXExtractor("docling-docx"),
        ),
        ExtractorRegistration(
            "unstructured-docx",
            frozenset({SourceKind.DOCX}),
            lambda: DOCXExtractor("unstructured-docx"),
        ),
        ExtractorRegistration(
            "openai-whisper-small-en",
            frozenset({SourceKind.AUDIO}),
            lambda: AudioExtractor("openai-whisper", "small.en", "float16"),
        ),
        ExtractorRegistration(
            "faster-whisper-tiny-int8",
            frozenset({SourceKind.AUDIO}),
            lambda: AudioExtractor("faster-whisper", "tiny.en", "int8"),
        ),
        ExtractorRegistration(
            "faster-whisper-base-int8",
            frozenset({SourceKind.AUDIO}),
            lambda: AudioExtractor("faster-whisper", "base.en", "int8"),
        ),
        ExtractorRegistration(
            "faster-whisper-small-int8",
            frozenset({SourceKind.AUDIO}),
            lambda: AudioExtractor("faster-whisper", "small.en", "int8"),
        ),
        ExtractorRegistration(
            "faster-whisper-small-float16",
            frozenset({SourceKind.AUDIO}),
            lambda: AudioExtractor("faster-whisper", "small.en", "float16"),
        ),
        ExtractorRegistration(
            "faster-whisper-turbo-int8",
            frozenset({SourceKind.AUDIO}),
            lambda: AudioExtractor("faster-whisper", "turbo", "int8"),
        ),
        ExtractorRegistration(
            "video-fixed",
            frozenset({SourceKind.VIDEO}),
            lambda: VideoExtractor("fixed"),
        ),
        ExtractorRegistration(
            "video-scene",
            frozenset({SourceKind.VIDEO}),
            lambda: VideoExtractor("scene"),
        ),
        ExtractorRegistration(
            "video-hybrid",
            frozenset({SourceKind.VIDEO}),
            lambda: VideoExtractor("hybrid"),
        ),
    ]
    for registration in registrations:
        registry.register(registration)
    return registry


class ExtractionPipeline:
    """Classify, route, extract, normalize, and cache one source."""

    def __init__(
        self, settings: Settings | None = None, registry: ExtractorRegistry | None = None
    ) -> None:
        self.settings = settings or load_settings()
        self.registry = registry or build_default_registry()
        self.cache = ExtractionCache(self.settings.extraction.cache_directory)

    def extract(
        self,
        source: str | Path | ExtractionRequest,
        *,
        profile: ExtractionProfile | None = None,
        mime_type: str | None = None,
        source_kind: SourceKind | None = None,
        options: Mapping[str, object] | None = None,
        use_cache: bool | None = None,
    ) -> ExtractedDocument:
        if isinstance(source, ExtractionRequest):
            request = source
            kind, resolved_mime = classify_source(request.source_path, request.mime_type)
            kind = request.source_kind or kind
            resolved_profile = profile or request.profile or DEFAULT_PROFILE_BY_KIND[kind]
            resolved_profile, prepared_options = self._prepare_profile(resolved_profile)
            request = replace(
                request,
                source_kind=kind,
                mime_type=resolved_mime,
                profile=resolved_profile,
                options={
                    **prepared_options,
                    **resolved_profile.options,
                    **request.options,
                },
            )
        else:
            path = Path(source)
            kind, resolved_mime = classify_source(path, mime_type)
            kind = source_kind or kind
            resolved_profile = profile or DEFAULT_PROFILE_BY_KIND[kind]
            resolved_profile, prepared_options = self._prepare_profile(resolved_profile)
            request = ExtractionRequest.from_path(
                path,
                mime_type=resolved_mime,
                source_kind=kind,
                profile=resolved_profile,
                options={
                    **prepared_options,
                    **resolved_profile.options,
                    **dict(options or {}),
                },
            )
        should_cache = self.settings.extraction.cache_enabled if use_cache is None else use_cache
        if should_cache and (cached := self.cache.get(request)) is not None:
            return cached
        started = time.perf_counter()
        resolved_request_profile = request.profile
        if resolved_request_profile is None:
            raise RuntimeError("Extraction routing failed to resolve a profile")
        extractor = self.registry.create(resolved_request_profile.engine, kind)
        document = normalize_document(
            extractor.extract(request, kind), resolved_request_profile.normalization
        )
        document = replace(document, extraction_seconds=time.perf_counter() - started)
        if should_cache:
            self.cache.put(request, document)
        return document

    def supported_sources(self) -> dict[str, list[str]]:
        return {kind.value: self.registry.names(kind) for kind in SourceKind}

    def _prepare_profile(
        self, profile: ExtractionProfile
    ) -> tuple[ExtractionProfile, dict[str, str]]:
        lock_path = self.settings.extraction.model_lock_path
        if not lock_path.is_file():
            if profile.engine_revision == "from-lock":
                raise RuntimeError(
                    f"Missing extraction model lock {lock_path}; run `python "
                    "experiments/benchmarks/prepare.py extraction-models`"
                )
            return profile, {}
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read extraction model lock {lock_path}: {exc}") from exc
        models = payload.get("models", {})
        if not isinstance(models, Mapping):
            raise RuntimeError(f"Malformed extraction model lock: {lock_path}")
        candidate = (
            "faster-whisper-base-int8" if profile.engine.startswith("video-") else profile.engine
        )
        entry = models.get(candidate, {})
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"Malformed model lock entry for {candidate}")
        locked_revision = entry.get("revision")
        if locked_revision and profile.engine_revision == "from-lock":
            profile = replace(profile, engine_revision=str(locked_revision))
        if (
            locked_revision
            and not profile.engine.startswith("video-")
            and str(locked_revision) != profile.engine_revision
        ):
            raise RuntimeError(
                f"Extraction profile revision mismatch for {candidate}: profile has "
                f"{profile.engine_revision}, model lock has {locked_revision}. Rebuild the "
                "profile or prepare the expected model revision."
            )
        return profile, {
            str(key): str(value)
            for key, value in entry.items()
            if str(key).endswith("_path") or str(key).endswith("_dir")
        }
