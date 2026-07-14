"""Single registry for every selectable runtime and benchmark candidate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCandidate:
    name: str
    suite: str
    stage: str
    dependency_extra: str
    deployment_eligible: bool = True


CANDIDATES = (
    # Extraction
    BenchmarkCandidate("tesseract-5", "extraction", "image", "extraction"),
    BenchmarkCandidate("paddleocr-v5-mobile", "extraction", "image", "extraction"),
    BenchmarkCandidate("paddleocr-v5-server", "extraction", "image", "extraction"),
    BenchmarkCandidate("doctr-fast-parseq", "extraction", "image", "benchmarks"),
    BenchmarkCandidate("pypdf", "extraction", "pdf", "extraction"),
    BenchmarkCandidate("pdfplumber", "extraction", "pdf", "extraction"),
    BenchmarkCandidate("docling-pdf", "extraction", "pdf", "benchmarks"),
    BenchmarkCandidate("hybrid-pdf", "extraction", "pdf", "extraction"),
    BenchmarkCandidate("python-docx", "extraction", "docx", "extraction"),
    BenchmarkCandidate("mammoth", "extraction", "docx", "benchmarks"),
    BenchmarkCandidate("docling-docx", "extraction", "docx", "benchmarks"),
    BenchmarkCandidate("unstructured-docx", "extraction", "docx", "benchmarks"),
    BenchmarkCandidate("openai-whisper-small-en", "extraction", "audio", "asr"),
    BenchmarkCandidate("faster-whisper-tiny-int8", "extraction", "audio", "asr"),
    BenchmarkCandidate("faster-whisper-base-int8", "extraction", "audio", "asr"),
    BenchmarkCandidate("faster-whisper-small-int8", "extraction", "audio", "asr"),
    BenchmarkCandidate("faster-whisper-small-float16", "extraction", "audio", "asr"),
    BenchmarkCandidate("faster-whisper-turbo-int8", "extraction", "audio", "asr"),
    BenchmarkCandidate("video-fixed", "extraction", "video", "asr,extraction"),
    BenchmarkCandidate("video-scene", "extraction", "video", "asr,extraction"),
    BenchmarkCandidate("video-hybrid", "extraction", "video", "asr,extraction"),
    BenchmarkCandidate("minimal", "extraction", "normalization", ""),
    BenchmarkCandidate("conservative", "extraction", "normalization", ""),
    BenchmarkCandidate("aggressive", "extraction", "normalization", ""),
    BenchmarkCandidate("always-native", "extraction", "routing", "extraction"),
    BenchmarkCandidate("always-ocr", "extraction", "routing", "extraction"),
    BenchmarkCandidate("document-router", "extraction", "routing", "extraction"),
    BenchmarkCandidate("page-hybrid-router", "extraction", "routing", "extraction"),
    # RAG
    *(
        BenchmarkCandidate(name, "rag", "chunking-embedding", "rag")
        for name in (
            "recursive-character",
            "token-256-32",
            "token-384-64",
            "sentence-8-2",
            "semantic",
        )
    ),
    *(
        BenchmarkCandidate(name, "rag", "chunking-embedding", "rag")
        for name in (
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-base-en-v1.5",
            "nomic-ai/nomic-embed-text-v1.5",
            "Qwen/Qwen3-Embedding-0.6B",
        )
    ),
    *(
        BenchmarkCandidate(name, "rag", "retrieval", "rag")
        for name in ("dense", "bm25", "rrf", "rrf-minilm-reranker", "rrf-qwen3-reranker")
    ),
    *(
        BenchmarkCandidate(name, "rag", "generation", "rag")
        for name in (
            "qwen3:1.7b",
            "qwen3.5:4b-direct",
            "qwen3.5:4b-thinking",
            "qwen3.5:9b-direct",
            "qwen3.5:9b-thinking",
            "gemma3:4b",
            "gemma3:12b",
            "ministral-3:8b-instruct-2512-q4_K_M",
            "gpt-oss:20b-low",
            "gpt-oss:20b-medium",
        )
    ),
    # Vector systems
    BenchmarkCandidate("numpy-exact-smoke", "systems", "vectordb", "", deployment_eligible=False),
    BenchmarkCandidate("chroma", "systems", "vectordb", "benchmarks"),
    BenchmarkCandidate("qdrant-local", "systems", "vectordb", "benchmarks"),
    BenchmarkCandidate("lancedb-local", "systems", "vectordb", "benchmarks"),
)


def candidates_for(suite: str, stage: str) -> tuple[str, ...]:
    return tuple(item.name for item in CANDIDATES if item.suite == suite and item.stage == stage)


def validate_unique_registry() -> None:
    keys = [(item.suite, item.stage, item.name) for item in CANDIDATES]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate benchmark registry entry")
