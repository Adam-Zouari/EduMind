"""Shared benchmark contracts and native dataset loading for experiments."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from edumind.common.paths import DATA_DIR
from edumind.rag.types import IngestDocument, MetadataScalar, sanitize_filter_metadata

EVALUATION_ROOT = DATA_DIR / "evaluation"
_WHITESPACE_RE = re.compile(r"\s+")
_QUESTION_PREFIXES = (
    ("how", "explanation"),
    ("why", "reasoning"),
    ("when", "timeline"),
    ("where", "location"),
    ("who", "entity"),
    ("what", "definition"),
)


@dataclass(frozen=True)
class BenchmarkAsset:
    """One benchmark asset tracked by the staged experiment system."""

    asset_id: str
    source_path: str
    modality: str
    title: str
    subject_domain: str
    tags: tuple[str, ...] = ()
    language: str = "en"


@dataclass(frozen=True)
class BenchmarkSnapshot:
    """Frozen OCR-normalized content used by RAG experiments."""

    asset_id: str
    source_id: str
    source: str
    format_type: str | None
    file_path: str | None
    normalized_text: str
    metadata: dict[str, object] = field(default_factory=dict)
    page_segments: tuple[dict[str, object], ...] = ()
    ocr_metadata: dict[str, object] = field(default_factory=dict)
    snapshot_version: str = "0.1.0"
    language: str = "en"

    def to_ingest_document(self) -> IngestDocument:
        """Convert the snapshot into the shared RAG ingest contract."""
        return IngestDocument(
            text=self.normalized_text,
            source_id=self.source_id,
            source=self.source,
            format_type=self.format_type,
            file_path=self.file_path,
            metadata=dict(self.metadata),
            filter_metadata=sanitize_filter_metadata(
                self.metadata,
                source=self.source,
                format_type=self.format_type,
                file_path=self.file_path,
            ),
        )


@dataclass(frozen=True)
class BenchmarkQuestion:
    """One benchmark query with source-level and excerpt-level supervision."""

    question_id: str
    query_text: str
    difficulty: str
    question_type: str
    relevant_source_ids: tuple[str, ...]
    support_excerpts: tuple[str, ...]
    gold_answer: str
    filter_metadata: dict[str, MetadataScalar] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    language: str = "en"


@dataclass(frozen=True)
class BenchmarkDataset:
    """Loaded benchmark dataset for one named split."""

    name: str
    version: str
    split: str
    assets: tuple[BenchmarkAsset, ...]
    snapshots: tuple[BenchmarkSnapshot, ...]
    questions: tuple[BenchmarkQuestion, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_ingest_documents(self) -> list[IngestDocument]:
        """Convert all snapshots into ingest documents."""
        return [snapshot.to_ingest_document() for snapshot in self.snapshots]

    def snapshot_map(self) -> dict[str, BenchmarkSnapshot]:
        """Return snapshots keyed by source id for retrieval evaluation."""
        return {snapshot.source_id: snapshot for snapshot in self.snapshots}

    def asset_map(self) -> dict[str, BenchmarkAsset]:
        """Return assets keyed by asset id."""
        return {asset.asset_id: asset for asset in self.assets}


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Runtime config logged for one benchmark experiment candidate."""

    stage: str
    candidate_name: str
    dataset_name: str
    dataset_version: str
    split: str
    seed: int
    top_k: int
    candidate_config: dict[str, object] = field(default_factory=dict)
    hardware: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation of the run config."""
        return asdict(self)


def list_benchmark_datasets() -> list[str]:
    """Return benchmark datasets defined under the evaluation root."""
    if not EVALUATION_ROOT.exists():
        return []
    return sorted(
        path.name
        for path in EVALUATION_ROOT.iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    )


def load_benchmark_dataset(dataset_name: str, split: str | None = None) -> BenchmarkDataset:
    """Load one native benchmark dataset and split."""
    dataset_dir = EVALUATION_ROOT / dataset_name
    manifest_path = dataset_dir / "manifest.json"
    manifest = _load_json_object(manifest_path)

    dataset_version = str(manifest.get("version", "0.1.0"))
    resolved_split = split or str(manifest.get("default_split", "default"))
    split_configs = manifest.get("splits", {})
    if not isinstance(split_configs, Mapping):
        raise ValueError(f"Invalid split config in {manifest_path}")

    split_config = split_configs.get(resolved_split)
    if not isinstance(split_config, Mapping):
        available = ", ".join(sorted(str(key) for key in split_configs)) or "<none>"
        raise ValueError(
            f"Unknown split '{resolved_split}' for dataset '{dataset_name}'. "
            f"Available splits: {available}"
        )

    split_filename = str(split_config.get("file", f"{resolved_split}.json"))
    split_payload = _load_json_object(dataset_dir / split_filename)

    assets = _load_assets(split_payload.get("assets", []))
    snapshots = _load_snapshots(split_payload.get("snapshots", []), dataset_version)
    questions = _load_questions(split_payload.get("questions", []))
    metadata = {
        "description": str(manifest.get("description", "")),
        "language": str(manifest.get("language", "en")),
    }

    split_metadata = split_payload.get("metadata", {})
    if isinstance(split_metadata, Mapping):
        metadata.update({str(key): value for key, value in split_metadata.items()})
    metadata.setdefault("split_file", split_filename)

    return BenchmarkDataset(
        name=str(split_payload.get("name", dataset_name)),
        version=str(split_payload.get("version", dataset_version)),
        split=str(split_payload.get("split", resolved_split)),
        assets=tuple(assets),
        snapshots=tuple(snapshots),
        questions=tuple(questions),
        metadata=metadata,
    )


def prepare_benchmark_dataset(dataset_name: str, split: str | None = None) -> dict[str, object]:
    """Validate and summarize one benchmark dataset for preparation checks."""
    dataset = load_benchmark_dataset(dataset_name, split=split)
    question_languages = sorted({question.language for question in dataset.questions})
    modalities = sorted({asset.modality for asset in dataset.assets})
    subjects = sorted({asset.subject_domain for asset in dataset.assets})
    issues: list[str] = []

    if not dataset.questions:
        issues.append("Dataset has no benchmark questions.")
    if not dataset.snapshots:
        issues.append("Dataset has no snapshots.")
    if any(not question.gold_answer.strip() for question in dataset.questions):
        issues.append("Some questions are missing gold answers.")

    return {
        "dataset_name": dataset.name,
        "dataset_version": dataset.version,
        "split": dataset.split,
        "num_assets": len(dataset.assets),
        "num_snapshots": len(dataset.snapshots),
        "num_questions": len(dataset.questions),
        "modalities": modalities,
        "languages": question_languages,
        "subjects": subjects,
        "issues": issues,
    }


def build_chunk_relevance_map(
    questions: Sequence[BenchmarkQuestion],
    chunk_records: Sequence[object],
) -> dict[str, set[str]]:
    """Resolve relevant chunk ids from source labels plus support excerpts."""
    relevance_map: dict[str, set[str]] = {}
    for question in questions:
        relevant_chunks: set[str] = set()
        normalized_excerpts = [
            _normalize_for_match(excerpt)
            for excerpt in question.support_excerpts
            if excerpt.strip()
        ]
        for chunk in chunk_records:
            chunk_id = getattr(chunk, "id", None)
            source_id = getattr(chunk, "source_id", None)
            text = getattr(chunk, "text", "")
            if not isinstance(chunk_id, str) or not isinstance(source_id, str):
                continue
            if source_id not in question.relevant_source_ids:
                continue

            normalized_text = _normalize_for_match(str(text))
            if normalized_excerpts:
                if any(excerpt in normalized_text for excerpt in normalized_excerpts):
                    relevant_chunks.add(chunk_id)
            else:
                relevant_chunks.add(chunk_id)

        if not relevant_chunks:
            for chunk in chunk_records:
                chunk_id = getattr(chunk, "id", None)
                source_id = getattr(chunk, "source_id", None)
                if isinstance(chunk_id, str) and source_id in question.relevant_source_ids:
                    relevant_chunks.add(chunk_id)
        relevance_map[question.question_id] = relevant_chunks
    return relevance_map


def build_source_relevance_map(
    questions: Sequence[BenchmarkQuestion],
) -> dict[str, set[str]]:
    """Return source-level relevance ids keyed by question id."""
    return {
        question.question_id: set(question.relevant_source_ids)
        for question in questions
    }


def _load_assets(raw_assets: object) -> list[BenchmarkAsset]:
    if not isinstance(raw_assets, list):
        return []
    assets: list[BenchmarkAsset] = []
    for raw_asset in raw_assets:
        if not isinstance(raw_asset, Mapping):
            continue
        raw_tags = raw_asset.get("tags", [])
        tags = tuple(str(tag) for tag in raw_tags) if isinstance(raw_tags, list) else ()
        assets.append(
            BenchmarkAsset(
                asset_id=str(raw_asset.get("asset_id", "")),
                source_path=str(raw_asset.get("source_path", "")),
                modality=str(raw_asset.get("modality", "text")),
                title=str(raw_asset.get("title", "")),
                subject_domain=str(raw_asset.get("subject_domain", "General")),
                tags=tags,
                language=str(raw_asset.get("language", "en")),
            )
        )
    return assets


def _load_snapshots(
    raw_snapshots: object,
    dataset_version: str,
) -> list[BenchmarkSnapshot]:
    if not isinstance(raw_snapshots, list):
        return []
    snapshots: list[BenchmarkSnapshot] = []
    for raw_snapshot in raw_snapshots:
        if not isinstance(raw_snapshot, Mapping):
            continue
        raw_page_segments = raw_snapshot.get("page_segments", [])
        page_segments = tuple(
            dict(segment)
            for segment in raw_page_segments
            if isinstance(segment, Mapping)
        )
        snapshots.append(
            BenchmarkSnapshot(
                asset_id=str(raw_snapshot.get("asset_id", "")),
                source_id=str(raw_snapshot.get("source_id", "")),
                source=str(raw_snapshot.get("source", "")),
                format_type=_coerce_optional_string(raw_snapshot.get("format_type")),
                file_path=_coerce_optional_string(raw_snapshot.get("file_path")),
                normalized_text=str(raw_snapshot.get("normalized_text", "")),
                metadata=_coerce_object_dict(raw_snapshot.get("metadata")),
                page_segments=page_segments,
                ocr_metadata=_coerce_object_dict(raw_snapshot.get("ocr_metadata")),
                snapshot_version=str(raw_snapshot.get("snapshot_version", dataset_version)),
                language=str(raw_snapshot.get("language", "en")),
            )
        )
    return snapshots


def _load_questions(raw_questions: object) -> list[BenchmarkQuestion]:
    if not isinstance(raw_questions, list):
        return []
    questions: list[BenchmarkQuestion] = []
    for raw_question in raw_questions:
        if not isinstance(raw_question, Mapping):
            continue
        relevant_source_ids = raw_question.get("relevant_source_ids", [])
        support_excerpts = raw_question.get("support_excerpts", [])
        question_text = str(raw_question.get("query_text", "")).strip()
        questions.append(
            BenchmarkQuestion(
                question_id=str(raw_question.get("question_id", "")),
                query_text=question_text,
                difficulty=str(raw_question.get("difficulty", "normal")),
                question_type=str(
                    raw_question.get("question_type", _infer_question_type(question_text))
                ),
                relevant_source_ids=tuple(
                    str(source_id) for source_id in relevant_source_ids
                )
                if isinstance(relevant_source_ids, list)
                else (),
                support_excerpts=tuple(str(excerpt) for excerpt in support_excerpts)
                if isinstance(support_excerpts, list)
                else (),
                gold_answer=str(raw_question.get("gold_answer", "")),
                filter_metadata=_coerce_filter_metadata(raw_question.get("filter_metadata")),
                metadata=_coerce_object_dict(raw_question.get("metadata")),
                language=str(raw_question.get("language", "en")),
            )
        )
    return questions


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark file must contain a JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


def _coerce_filter_metadata(raw_value: object) -> dict[str, MetadataScalar]:
    if not isinstance(raw_value, Mapping):
        return {}
    filter_metadata: dict[str, MetadataScalar] = {}
    for key, value in raw_value.items():
        if isinstance(value, (str, int, float, bool)):
            filter_metadata[str(key)] = value
    return filter_metadata


def _coerce_object_dict(raw_value: object) -> dict[str, object]:
    if not isinstance(raw_value, Mapping):
        return {}
    return {str(key): value for key, value in raw_value.items()}


def _coerce_optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _infer_question_type(query_text: str) -> str:
    lowered = query_text.strip().lower()
    for prefix, label in _QUESTION_PREFIXES:
        if lowered.startswith(prefix):
            return label
    return "general"


def _normalize_for_match(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()
