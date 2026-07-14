from __future__ import annotations

from edumind.rag.text_chunker import (
    RecursiveCharacterChunkingStrategy,
    SentenceChunkingStrategy,
    TextChunker,
    TokenChunkingStrategy,
)
from edumind.rag.tokenizers import RegexOffsetTokenizer
from edumind.rag.types import IngestDocument


def test_token_chunking_preserves_exact_offsets_and_overlap() -> None:
    text = "one two three four five six seven eight"
    strategy = TokenChunkingStrategy(RegexOffsetTokenizer(), 4, 1)
    spans = strategy.split(text)
    assert len(spans) == 3
    assert [text[start:end] for start, end, _ in spans] == [
        "one two three four",
        "four five six seven",
        "seven eight",
    ]


def test_chunk_records_are_deterministic_and_source_exact() -> None:
    document = IngestDocument("First sentence. Second sentence. Third sentence.", "source", "notes")
    chunker = TextChunker(strategy=SentenceChunkingStrategy(RegexOffsetTokenizer(), 2, 1))
    first = chunker.chunk_document(document)
    second = chunker.chunk_document(document)
    assert [item.id for item in first] == [item.id for item in second]
    assert all(document.text[item.start : item.end] == item.text for item in first)
    assert all(item.token_count > 0 for item in first)


def test_recursive_character_strategy_makes_progress() -> None:
    text = "paragraph one. " * 100
    spans = RecursiveCharacterChunkingStrategy(RegexOffsetTokenizer(), 80, 20).split(text)
    assert len(spans) > 2
    assert spans[-1][1] == len(text)
