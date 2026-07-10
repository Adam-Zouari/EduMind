from __future__ import annotations

import numpy as np

from edumind.rag.text_chunker import TextChunker
from edumind.rag.types import ChunkingSettings, IngestDocument


class FakeEmbedder:
    def embed_texts(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        del show_progress
        return np.array(
            [[float(index + 1), 1.0] for index, _ in enumerate(texts)],
            dtype=float,
        )


def test_chunk_document_is_deterministic_and_preserves_filter_metadata() -> None:
    chunker = TextChunker(
        settings=ChunkingSettings(
            chunk_size=30,
            chunk_overlap=10,
            separators=("\n\n", "\n", " ", ""),
        ),
        embedder=FakeEmbedder(),
    )
    document = IngestDocument(
        text=(
            "Alpha beta gamma delta.\n\n"
            "Epsilon zeta eta theta.\n\n"
            "Iota kappa lambda mu.\n\n"
            "Nu xi omicron pi."
        ),
        source_id="study-doc",
        source="notes.pdf",
        format_type="pdf",
        file_path="notes.pdf",
        metadata={"page": 1},
        filter_metadata={"course": "biology", "page": 1},
    )

    first = chunker.chunk_document(document)
    second = chunker.chunk_document(document)

    assert len(first) >= 2
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert all(chunk.filter_metadata["course"] == "biology" for chunk in first)
    assert all(chunk.metadata["source_id"] == "study-doc" for chunk in first)
    assert all(len(chunk.text) <= 55 for chunk in first)


def test_chunk_text_uses_metadata_source_information() -> None:
    chunker = TextChunker(
        settings=ChunkingSettings(
            chunk_size=60,
            chunk_overlap=0,
            separators=("\n\n", "\n", " ", ""),
        ),
        embedder=FakeEmbedder(),
    )

    chunks = chunker.chunk_text(
        "Sentence one. Sentence two. Sentence three.",
        metadata={
            "source": "lecture-notes.txt",
            "format_type": "txt",
            "file_path": "lecture-notes.txt",
            "page": 3,
        },
    )

    assert chunks
    assert chunks[0].metadata["source"] == "lecture-notes.txt"
    assert chunks[0].filter_metadata["page"] == 3
