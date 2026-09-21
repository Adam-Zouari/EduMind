from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from experiments.benchmarks.extraction.audio.protocol import (
    load_protocol as load_audio_protocol,
)
from experiments.benchmarks.extraction.document.protocol import (
    load_protocol as load_document_protocol,
)
from experiments.benchmarks.extraction.video.candidates import fixed_candidates
from experiments.benchmarks.extraction.video.protocol import (
    load_protocol as load_video_protocol,
)
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    load_protocol as load_chunking_protocol,
)
from experiments.benchmarks.rag.chunking_embedding.strategies import (
    build_chunking_strategy,
)
from experiments.benchmarks.rag.final.protocol import (
    load_protocol as load_final_protocol,
)
from experiments.benchmarks.rag.generation.protocol import (
    load_protocol as load_generation_protocol,
)
from experiments.benchmarks.rag.generation.evaluate import _questions
from experiments.benchmarks.rag.retrieval.protocol import (
    load_protocol as load_retrieval_protocol,
)
from experiments.benchmarks.vectordb.protocol import (
    load_protocol as load_vector_protocol,
)
from experiments.benchmarks.common.runner import _protocol_confidence_level


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_CASES = (
    (
        "document",
        ROOT / "experiments/benchmarks/extraction/document/protocol.yaml",
        load_document_protocol,
        ("parser", "image_scale"),
    ),
    (
        "audio",
        ROOT / "experiments/benchmarks/extraction/audio/protocol.yaml",
        load_audio_protocol,
        ("alignment", "content_f1_threshold"),
    ),
    (
        "video",
        ROOT / "experiments/benchmarks/extraction/video/protocol.yaml",
        load_video_protocol,
        ("visual_text", "occurrence_content_f1_threshold"),
    ),
    (
        "chunking_embedding",
        ROOT / "experiments/benchmarks/rag/chunking_embedding/protocol.yaml",
        load_chunking_protocol,
        ("quality", "alpha_ndcg_alpha"),
    ),
    (
        "retrieval",
        ROOT / "experiments/benchmarks/rag/retrieval/protocol.yaml",
        load_retrieval_protocol,
        ("retrieval", "bm25", "k1"),
    ),
    (
        "generation",
        ROOT / "experiments/benchmarks/rag/generation/protocol.yaml",
        load_generation_protocol,
        ("generation", "temperature"),
    ),
    (
        "final_rag",
        ROOT / "experiments/benchmarks/rag/final/protocol.yaml",
        load_final_protocol,
        ("selection", "top_k", 0),
    ),
    (
        "vector_database",
        ROOT / "experiments/benchmarks/vectordb/protocol.yaml",
        load_vector_protocol,
        ("synthetic", "cluster_noise"),
    ),
)


def _metadata(protocol, source: Path):
    return protocol.metadata(source) if hasattr(protocol, "metadata") else protocol.meta


def _write_protocol(tmp_path: Path, name: str, payload: dict[str, object]) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _set_nested(payload: object, path: tuple[object, ...], value: object) -> None:
    current = payload
    for key in path[:-1]:
        current = current[key]  # type: ignore[index]
    current[path[-1]] = value  # type: ignore[index]


@pytest.mark.parametrize(("name", "source", "loader", "_numeric"), PROTOCOL_CASES)
def test_every_protocol_has_stable_identity_and_rejects_worker_drift(
    name, source, loader, _numeric
) -> None:
    first = loader(source)
    second = loader(source)
    first_meta = _metadata(first, source)
    second_meta = _metadata(second, source)
    assert first_meta.name == name
    assert first_meta.schema_version == 1
    assert first_meta.version
    assert first_meta.checksum == second_meta.checksum
    first_meta.validate_worker_payload(first_meta.worker_payload())
    mismatched = {**first_meta.worker_payload(), "checksum": "wrong"}
    with pytest.raises(ValueError, match="checksum"):
        first_meta.validate_worker_payload(mismatched)


@pytest.mark.parametrize(("name", "source", "loader", "numeric"), PROTOCOL_CASES)
def test_every_protocol_rejects_missing_unknown_and_nonfinite_fields(
    tmp_path, name, source, loader, numeric
) -> None:
    original = yaml.safe_load(source.read_text(encoding="utf-8"))

    missing = deepcopy(original)
    missing.pop("seed")
    with pytest.raises(ValueError):
        loader(_write_protocol(tmp_path, f"{name}-missing", missing))

    unknown = deepcopy(original)
    unknown["unexpected"] = True
    with pytest.raises(ValueError):
        loader(_write_protocol(tmp_path, f"{name}-unknown", unknown))

    nonfinite = deepcopy(original)
    _set_nested(nonfinite, numeric, float("nan"))
    with pytest.raises(ValueError):
        loader(_write_protocol(tmp_path, f"{name}-nonfinite", nonfinite))

    invalid_profile = deepcopy(original)
    first_profile = next(iter(invalid_profile["profiles"]))
    invalid_profile["profiles"][first_profile]["device"] = "tpu"
    with pytest.raises(ValueError):
        loader(_write_protocol(tmp_path, f"{name}-profile", invalid_profile))


def test_document_audio_and_video_behavior_comes_from_protocol(tmp_path) -> None:
    document_root = deepcopy(load_document_protocol().meta.resolved)
    document_root["candidate_factors"]["smoke_configuration"]["ocr_engine"] = "tesseract"
    document_root["parser"]["image_scale"] = 2.5
    document = load_document_protocol(
        _write_protocol(tmp_path, "document-behavior", document_root)
    )
    assert "ocr=tesseract" in document.configuration_candidates("smoke")[0]
    standard_options = document.parser_options("docling-standard")
    assert standard_options["image_scale"] == 2.5
    assert standard_options["ocr_engine"] == "tesseract"

    audio_root = deepcopy(load_audio_protocol().meta.resolved)
    audio_root["decoding"]["moss-transcribe-diarize"]["max_new_tokens"] = 1024
    audio_root["alignment"]["content_f1_threshold"] = 0.6
    audio = load_audio_protocol(_write_protocol(tmp_path, "audio-behavior", audio_root))
    assert audio.decoder("moss-transcribe-diarize")["max_new_tokens"] == 1024
    assert audio.alignment_threshold == 0.6

    video_root = deepcopy(load_video_protocol().meta.resolved)
    video_root["keyframes"]["fixed_intervals_seconds"] = [4, 8, 12]
    video = load_video_protocol(_write_protocol(tmp_path, "video-behavior", video_root))
    assert fixed_candidates(video) == (
        "video-fixed-4s",
        "video-fixed-8s",
        "video-fixed-12s",
    )


def test_chunking_strategy_kind_cannot_contradict_candidate_alias(
    tmp_path: Path,
) -> None:
    root = deepcopy(load_chunking_protocol().meta.resolved)
    root["strategies"]["token-256-32"] = {
        "kind": "sentence",
        "sentences": 8,
        "overlap": 2,
    }

    with pytest.raises(ValueError, match="contradicts its candidate alias"):
        load_chunking_protocol(_write_protocol(tmp_path, "chunk-kind", root))


def test_rag_and_vector_behavior_comes_from_protocol(tmp_path) -> None:
    chunk_root = deepcopy(load_chunking_protocol().meta.resolved)
    chunk_root["strategies"]["token-256-32"].update({"size": 8, "overlap": 2})
    chunk_root["quality"].update({"cutoffs": [2, 4], "audit_depth": 7})
    chunking = load_chunking_protocol(
        _write_protocol(tmp_path, "chunking-behavior", chunk_root)
    )
    chunks = build_chunking_strategy(
        "token-256-32", protocol=chunking
    ).split("one two three four five six seven eight nine ten eleven twelve")
    assert chunking.quality_cutoffs == (2, 4)
    assert chunking.audit_depth == 7
    assert len(chunks) > 1 and chunks[0][2] <= 8

    retrieval_root = deepcopy(load_retrieval_protocol().resolved)
    retrieval_root["retrieval"]["pool_size"] = 12
    retrieval_root["retrieval"]["rrf"]["source_depth"] = 12
    retrieval = load_retrieval_protocol(
        _write_protocol(tmp_path, "retrieval-behavior", retrieval_root)
    )
    assert retrieval.pool_size == retrieval.rrf_source_depth == 12

    generation_root = deepcopy(load_generation_protocol().meta.resolved)
    generation_root["development_screen"]["question_count"] = 12
    generation_root["context"]["frozen_packing_limit_tokens"] = 3000
    generation_root["generation"]["maximum_answer_tokens"] = 128
    generation_root["generation"]["streamer_timeout_seconds"] = 90
    generation = load_generation_protocol(
        _write_protocol(tmp_path, "generation-behavior", generation_root)
    )
    assert (
        generation.question_count,
        generation.context_packing_tokens,
        generation.maximum_answer_tokens,
        generation.streamer_timeout_seconds,
    ) == (12, 3000, 128, 90)

    final_root = deepcopy(load_final_protocol().meta.resolved)
    final_root["selection"]["top_k"] = [2, 4]
    final = load_final_protocol(_write_protocol(tmp_path, "final-behavior", final_root))
    assert final.top_k == (2, 4)

    vector_root = deepcopy(load_vector_protocol().meta.resolved)
    vector_root["workloads"]["smoke"][0]["size"] = 321
    vector_root["workloads"]["development"][1]["concurrency"] = [1, 4]
    vector_root["adapters"]["upsert_batch_sizes"]["chroma"] = 17
    vector_root["adapters"]["qdrant_indexing_threshold"] = 2
    vector_root["statistics"]["confidence_level"] = 0.9
    vector = load_vector_protocol(_write_protocol(tmp_path, "vector-behavior", vector_root))
    assert vector.workloads["smoke"][0]["size"] == 321
    assert vector.concurrency_levels("development", "development-100k-d1024") == (
        1,
        4,
    )
    assert vector.concurrency_levels("development") == (1, 4, 8, 32)
    assert vector.upsert_batch_sizes["chroma"] == 17
    assert vector.qdrant_indexing_threshold == 2
    assert vector.confidence_level == 0.9
    assert (
        vector.tie_breaking
        == "latency-then-m-then-ef-search-then-ef-construction"
    )
    assert _protocol_confidence_level({"vector_database": vector.meta}) == 0.9


def test_generation_screen_limit_and_final_profiles_are_explicit() -> None:
    manifest = SimpleNamespace(
        samples=tuple(
            {
                "id": f"q-{index}",
                "kind": "question",
                "answerable": True,
                "evidence": [{"unit_id": f"e-{index}"}],
                "evidence_type": "text",
            }
            for index in range(30)
        )
    )
    assert len(_questions(manifest, 24, 42)) == 24
    assert len(_questions(manifest, None, 42)) == 30

    final = load_final_protocol()
    assert final.profile("development") == final.meta.profile("development")
    assert final.profile("validation") == final.meta.profile("validation")
    assert final.profile("locked") == final.meta.profile("locked")
    with pytest.raises(ValueError, match="no profile 'standard'"):
        final.profile("standard")


def test_retrieval_rejects_an_undeclared_smoke_input(tmp_path) -> None:
    root = deepcopy(load_retrieval_protocol().resolved)
    root["smoke_fixture"]["chunking_embedding_candidate"] = "unknown|model"
    with pytest.raises(ValueError, match="candidates.yaml"):
        load_retrieval_protocol(_write_protocol(tmp_path, "retrieval-alias", root))
