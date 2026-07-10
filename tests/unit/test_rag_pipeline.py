from __future__ import annotations

from edumind.rag.rag_pipeline import RAGPipeline
from edumind.rag.types import AnswerResult, ChunkRecord, IngestDocument, RetrievalHit


class FakeOCRProcessor:
    def normalize_document(self, document):
        del document
        return IngestDocument(
            text="Normalized study content",
            source_id="doc-1",
            source="notes.pdf",
            format_type="pdf",
            file_path="notes.pdf",
            metadata={"page": 1, "source": "notes.pdf"},
            filter_metadata={"page": 1},
        )


class FakeTextChunker:
    def chunk_document(self, document: IngestDocument) -> list[ChunkRecord]:
        return [
            ChunkRecord(
                id="doc-1:0",
                source_id=document.source_id,
                text="chunk one",
                chunk_index=0,
                total_chunks=2,
                metadata=dict(document.metadata),
                filter_metadata=dict(document.filter_metadata),
            ),
            ChunkRecord(
                id="doc-1:1",
                source_id=document.source_id,
                text="chunk two",
                chunk_index=1,
                total_chunks=2,
                metadata=dict(document.metadata),
                filter_metadata=dict(document.filter_metadata),
            ),
        ]


class FakeEmbedder:
    def embed_chunks(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        return [
            ChunkRecord(
                id=chunk.id,
                source_id=chunk.source_id,
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks,
                metadata=dict(chunk.metadata),
                filter_metadata=dict(chunk.filter_metadata),
                embedding=[0.1, 0.2],
            )
            for chunk in chunks
        ]


class FakeVectorStore:
    def __init__(self) -> None:
        self.upserted: list[ChunkRecord] = []

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> int:
        self.upserted = chunks
        return len(chunks)

    def get_collection_count(self) -> int:
        return len(self.upserted)


class FakeLLMGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[RetrievalHit], str | None, bool]] = []

    def generate_with_results(
        self,
        query: str,
        results: list[RetrievalHit],
        *,
        system_prompt: str | None = None,
        stream: bool = False,
    ) -> str:
        self.calls.append((query, results, system_prompt, stream))
        return "Grounded answer"


def test_ingest_document_returns_report_and_upserts_chunks() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.ocr_processor = FakeOCRProcessor()
    pipeline.text_chunker = FakeTextChunker()
    pipeline.embedder = FakeEmbedder()
    pipeline.vector_store = FakeVectorStore()
    pipeline._log_active_mlflow_ingest = lambda document, chunk_count: None

    report = RAGPipeline.ingest_document(pipeline, {"text": "raw payload"})

    assert report.source_id == "doc-1"
    assert report.chunks_created == 2
    assert len(pipeline.vector_store.upserted) == 2


def test_generate_answer_queries_once_and_reuses_results() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    llm_generator = FakeLLMGenerator()
    results = [
        RetrievalHit(
            id="chunk-1",
            document="Study this section first",
            metadata={"source": "lesson.pdf", "page": 3},
            score=0.93,
        )
    ]
    query_calls: list[tuple[str, int | None, object]] = []

    def fake_query(
        query_text: str,
        top_k: int | None = None,
        filter_metadata=None,
    ) -> list[RetrievalHit]:
        query_calls.append((query_text, top_k, filter_metadata))
        return results

    pipeline.query = fake_query
    pipeline.llm_generator = llm_generator
    pipeline._log_active_mlflow_query = lambda query, results, answer: None

    answer = RAGPipeline.generate_answer(
        pipeline,
        query="What should I review?",
        top_k=4,
        filter_metadata={"page": 3},
    )

    assert isinstance(answer, AnswerResult)
    assert answer.answer == "Grounded answer"
    assert answer.sources == results
    assert len(query_calls) == 1
    assert llm_generator.calls[0][1] == results
    assert "Study this section first" in answer.context
