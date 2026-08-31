"""Public contracts shared by extraction runtime and benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, TypeAlias, runtime_checkable

from edumind.common.artifacts import sha256_file, stable_hash

BoundingBox: TypeAlias = tuple[float, float, float, float]


class SourceKind(str, Enum):
    IMAGE = "image"
    PDF = "pdf"
    DOCX = "docx"
    AUDIO = "audio"
    VIDEO = "video"


class SegmentKind(str, Enum):
    TEXT = "text"
    TITLE = "title"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FORMULA = "formula"
    CAPTION = "caption"
    FIGURE = "figure"
    CODE = "code"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    AUDIO = "audio"
    VISUAL_TEXT = "visual_text"


@dataclass(frozen=True)
class ExtractionProfile:
    """Everything that can affect an extraction result."""

    name: str
    engine: str
    engine_revision: str
    preprocessing: str = "raw"
    device: str = "cpu"
    routing: str = "direct"
    normalization: str = "conservative"
    options: Mapping[str, object] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class ExtractionRequest:
    """One immutable local extraction request."""

    source_path: Path
    checksum: str
    mime_type: str | None = None
    source_kind: SourceKind | None = None
    profile: ExtractionProfile | None = None
    options: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_path(
        cls,
        source_path: str | Path,
        *,
        mime_type: str | None = None,
        source_kind: SourceKind | None = None,
        profile: ExtractionProfile | None = None,
        options: Mapping[str, object] | None = None,
    ) -> ExtractionRequest:
        path = Path(source_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Extraction source does not exist: {path}")
        return cls(
            source_path=path,
            checksum=sha256_file(path),
            mime_type=mime_type,
            source_kind=source_kind,
            profile=profile,
            options=dict(options or {}),
        )


@dataclass(frozen=True)
class ExtractionWarning:
    code: str
    message: str
    recoverable: bool = True
    segment_index: int | None = None


@dataclass(frozen=True)
class ExtractedSegment:
    text: str
    start: int
    end: int
    element_id: str | None = None
    parent_id: str | None = None
    order: int | None = None
    page_number: int | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    bounding_box: BoundingBox | None = None
    kind: SegmentKind = SegmentKind.TEXT
    structured_content: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("Segment offsets must satisfy 0 <= start <= end")
        if self.end - self.start != len(self.text):
            raise ValueError("Segment offsets must be half-open offsets for segment text")
        if self.timestamp_start is not None and self.timestamp_end is not None:
            if self.timestamp_end < self.timestamp_start:
                raise ValueError("Segment timestamps are reversed")
        if self.kind is SegmentKind.TABLE and self.structured_content:
            rows = self.structured_content.get("rows")
            if not isinstance(rows, (list, tuple)):
                raise ValueError("Structured table segments require a rows sequence")
        if self.kind is SegmentKind.FORMULA and self.structured_content:
            if not isinstance(self.structured_content.get("latex"), str):
                raise ValueError("Structured formula segments require a LaTeX string")


@dataclass(frozen=True)
class ExtractedDocument:
    source_name: str
    source_path: str
    source_kind: SourceKind
    source_checksum: str
    mime_type: str | None
    text: str
    segments: tuple[ExtractedSegment, ...]
    profile: ExtractionProfile
    metadata: Mapping[str, object] = field(default_factory=dict)
    warnings: tuple[ExtractionWarning, ...] = ()
    extraction_seconds: float = 0.0
    cache_hit: bool = False

    def __post_init__(self) -> None:
        for segment in self.segments:
            if segment.end > len(self.text):
                raise ValueError("Segment offset exceeds document text")
            if self.text[segment.start : segment.end] != segment.text:
                raise ValueError("Segment text does not match document offsets")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source_kind"] = self.source_kind.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExtractedDocument:
        raw_profile = payload.get("profile")
        if not isinstance(raw_profile, Mapping):
            raise ValueError("Cached extraction profile is malformed")
        profile = ExtractionProfile(**dict(raw_profile))
        raw_segments = payload.get("segments", [])
        raw_warnings = payload.get("warnings", [])
        if not isinstance(raw_segments, (list, tuple)) or not isinstance(
            raw_warnings, (list, tuple)
        ):
            raise ValueError("Cached extraction sequence is malformed")
        raw_metadata = payload.get("metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        raw_seconds = payload.get("extraction_seconds", 0.0)
        seconds = float(raw_seconds) if isinstance(raw_seconds, (str, int, float)) else 0.0
        return cls(
            source_name=str(payload.get("source_name", "")),
            source_path=str(payload.get("source_path", "")),
            source_kind=SourceKind(str(payload.get("source_kind"))),
            source_checksum=str(payload.get("source_checksum", "")),
            mime_type=str(payload["mime_type"]) if payload.get("mime_type") else None,
            text=str(payload.get("text", "")),
            segments=tuple(
                _segment_from_dict(item) for item in raw_segments if isinstance(item, Mapping)
            ),
            profile=profile,
            metadata=metadata,
            warnings=tuple(
                ExtractionWarning(**dict(item))
                for item in raw_warnings
                if isinstance(item, Mapping)
            ),
            extraction_seconds=seconds,
            cache_hit=bool(payload.get("cache_hit", False)),
        )


@runtime_checkable
class Extractor(Protocol):
    """Production extraction interface; benchmark candidates implement this exact protocol."""

    name: str
    revision: str
    supported_kinds: frozenset[SourceKind]

    def extract(self, request: ExtractionRequest, kind: SourceKind) -> ExtractedDocument:
        """Extract a document or raise a structured extraction error."""


def _segment_from_dict(payload: Mapping[str, object]) -> ExtractedSegment:
    values = dict(payload)
    values["kind"] = SegmentKind(str(values.get("kind", SegmentKind.TEXT.value)))
    structured = values.get("structured_content", {})
    values["structured_content"] = dict(structured) if isinstance(structured, Mapping) else {}
    metadata = values.get("metadata", {})
    values["metadata"] = dict(metadata) if isinstance(metadata, Mapping) else {}
    bounding_box = values.get("bounding_box")
    if isinstance(bounding_box, (list, tuple)):
        values["bounding_box"] = tuple(float(value) for value in bounding_box)
    return ExtractedSegment(**values)
