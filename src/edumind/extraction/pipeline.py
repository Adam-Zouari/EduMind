"""Production extraction classifier, router, cache, and normalization pipeline."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from edumind.common.config import Settings, load_settings
from edumind.common.models import load_model_lock, require_model

from .cache import ExtractionCache
from .contracts import ExtractedDocument, ExtractionProfile, ExtractionRequest, SourceKind
from .detection import classify_source
from .normalization import normalize_document
from .registry import ExtractorRegistration, ExtractorRegistry

DEFAULT_PROFILE_BY_KIND = {
    SourceKind.IMAGE: ExtractionProfile(
        "document-control-image",
        "docling-standard",
        "from-lock",
        options={
            "ocr_engine": "rapidocr",
            "ocr_mode": "full_page",
            "table_mode": "fast",
            "formula_enrichment": False,
        },
    ),
    SourceKind.PDF: ExtractionProfile(
        "document-control-pdf",
        "docling-standard",
        "from-lock",
        options={
            "ocr_engine": "rapidocr",
            "ocr_mode": "pdf_aware_layout_regions",
            "table_mode": "fast",
            "formula_enrichment": False,
        },
    ),
    SourceKind.DOCX: ExtractionProfile(
        "document-control-docx",
        "docling-standard",
        "from-lock",
        normalization="conservative",
    ),
    SourceKind.AUDIO: ExtractionProfile(
        "audio-control",
        "whisper-small-en-control",
        "from-lock",
        device="cpu",
    ),
    SourceKind.VIDEO: ExtractionProfile(
        "video-control",
        "video-hybrid",
        "ffmpeg-system",
        device="cpu",
        routing="hybrid",
        options={
            "audio_candidate": "whisper-small-en-control",
            "image_engine": "docling-standard",
            "ocr_engine": "rapidocr",
            "ocr_mode": "full_page",
            "table_mode": "fast",
            "formula_enrichment": False,
        },
    ),
}

LOCK_CANDIDATE_BY_ENGINE = {
    "docling-standard": "docling-standard",
    "whisper-small-en-control": "openai/whisper-small.en",
}


def build_default_registry() -> ExtractorRegistry:
    from .extractors.audio import WhisperExtractor
    from .extractors.document import DoclingExtractor
    from .extractors.video import VideoExtractor

    registry = ExtractorRegistry()
    registrations: list[ExtractorRegistration] = [
        ExtractorRegistration(
            "docling-standard",
            frozenset({SourceKind.IMAGE, SourceKind.PDF, SourceKind.DOCX}),
            lambda: DoclingExtractor("from-lock"),
        ),
        ExtractorRegistration(
            "whisper-small-en-control",
            frozenset({SourceKind.AUDIO}),
            lambda: WhisperExtractor("from-lock"),
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

    def readiness(self) -> dict[str, object]:
        errors: dict[str, str] = {}
        for kind, profile in DEFAULT_PROFILE_BY_KIND.items():
            try:
                self._prepare_profile(profile)
            except Exception as exc:
                errors[kind.value] = str(exc)
        return {"ready": not errors, "errors": errors}

    def _prepare_profile(
        self, profile: ExtractionProfile
    ) -> tuple[ExtractionProfile, dict[str, object]]:
        lock_path = self.settings.models.lock_path
        if not lock_path.is_file():
            if profile.engine_revision == "from-lock":
                raise RuntimeError(
                    f"Missing extraction model lock {lock_path}; run `python "
                    "experiments/benchmarks/prepare.py app-models`"
                )
            return profile, {}
        models = load_model_lock(lock_path)
        if profile.engine.startswith("video-"):
            if profile.options.get("audio_model_path") and profile.options.get(
                "image_model_path"
            ):
                return profile, {}
            audio_engine = str(profile.options.get("audio_candidate", "whisper-small-en-control"))
            image_engine = str(profile.options.get("image_engine", "docling-standard"))
            audio = require_model(models, LOCK_CANDIDATE_BY_ENGINE[audio_engine])
            image = require_model(models, LOCK_CANDIDATE_BY_ENGINE[image_engine])
            return profile, {
                "audio_revision": audio.revision,
                "image_revision": image.revision,
                **{
                    f"audio_{key}": value
                    for key, value in _lock_options(audio.values).items()
                },
                **{
                    f"image_{key}": value
                    for key, value in _lock_options(image.values).items()
                },
            }
        candidate = LOCK_CANDIDATE_BY_ENGINE.get(profile.engine, profile.engine)
        snapshot = require_model(models, candidate)
        if profile.engine_revision == "from-lock":
            profile = replace(profile, engine_revision=snapshot.revision)
        if (
            not profile.engine.startswith("video-")
            and snapshot.revision != profile.engine_revision
        ):
            raise RuntimeError(
                f"Extraction profile revision mismatch for {candidate}: profile has "
                f"{profile.engine_revision}, model lock has {snapshot.revision}. Rebuild the "
                "profile or prepare the expected model revision."
            )
        return profile, _lock_options(snapshot.values)


def _lock_options(entry: Mapping[str, object]) -> dict[str, object]:
    options = {
        str(key): value
        for key, value in entry.items()
        if str(key).endswith("_path") or str(key).endswith("_dir")
    }
    submodels = entry.get("submodels", [])
    if isinstance(submodels, list):
        for submodel in submodels:
            if not isinstance(submodel, Mapping):
                continue
            if submodel.get("role") == "forced-aligner":
                options["aligner_model_path"] = str(submodel.get("model_path", ""))
    return options
