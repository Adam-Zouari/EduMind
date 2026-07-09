"""RAG package with lazy exports."""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "OCRProcessor",
    "TextChunker",
    "Embedder",
    "VectorStore",
    "RAGPipeline",
]


def __getattr__(name: str):
    if name == "OCRProcessor":
        from .ocr_processor import OCRProcessor

        return OCRProcessor
    if name == "TextChunker":
        from .text_chunker import TextChunker

        return TextChunker
    if name == "Embedder":
        from .embedder import Embedder

        return Embedder
    if name == "VectorStore":
        from .vector_store import VectorStore

        return VectorStore
    if name == "RAGPipeline":
        from .rag_pipeline import RAGPipeline

        return RAGPipeline
    raise AttributeError(name)
