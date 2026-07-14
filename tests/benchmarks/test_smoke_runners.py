from __future__ import annotations

from edumind.benchmarks.extraction import run_extraction_stage
from edumind.benchmarks.rag import (
    run_chunking_embedding,
    run_final,
    run_generation,
    run_retrieval,
)
from edumind.benchmarks.vectordb import run_vectordb


def test_network_and_ollama_free_smoke_paths(tmp_path) -> None:
    extraction_results = [
        run_extraction_stage(stage, "smoke", artifact_root=tmp_path)
        for stage in ("image", "pdf", "docx", "audio", "video", "normalization", "routing")
    ]
    rag = run_chunking_embedding("smoke", artifact_root=tmp_path)
    retrieval = run_retrieval("smoke", artifact_root=tmp_path)
    generation = run_generation("smoke", artifact_root=tmp_path)
    final = run_final("smoke", artifact_root=tmp_path)
    vectors = run_vectordb("smoke", artifact_root=tmp_path)
    results = (*extraction_results, rag, retrieval, generation, final, vectors)
    assert all(result.candidates for result in results)
    assert not any(result.authoritative for result in results)
    assert all((result.artifact_directory / "_SUCCESS.json").is_file() for result in results)
