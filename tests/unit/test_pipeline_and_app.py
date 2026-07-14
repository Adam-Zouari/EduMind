from __future__ import annotations

from edumind.app import AppController
from edumind.extraction import ExtractedDocument, ExtractedSegment, ExtractionProfile, SourceKind
from edumind.pipeline import EduMindPipeline
from edumind.rag.types import IngestReport


class FakeExtraction:
    def extract(self, path, profile=None):
        return ExtractedDocument(
            "upload.pdf",
            str(path),
            SourceKind.PDF,
            "checksum",
            "application/pdf",
            "alpha",
            (ExtractedSegment("alpha", 0, 5, page_number=1),),
            ExtractionProfile("fake", "fake", "1"),
        )

    def supported_sources(self):
        return {"pdf": ["fake"]}


class FakeRAG:
    llm_generator = None

    def __init__(self):
        self.ingests = 0
        self.resets = 0

    def ingest_document(self, document):
        self.ingests += 1
        return IngestReport("source", "upload.pdf", 1)

    def get_stats(self):
        return {"ready": True}

    def reset(self):
        self.resets += 1


def test_app_controller_prevents_duplicate_ingestion_and_cleans_temp(tmp_path) -> None:
    rag = FakeRAG()
    pipeline = EduMindPipeline(extraction=FakeExtraction(), rag=rag, use_llm=False)
    controller = AppController(pipeline)
    records = {}
    first, processed = controller.process_upload("notes.pdf", b"same", records)
    second, processed_again = controller.process_upload("notes.pdf", b"same", records)
    assert processed is True and processed_again is False
    assert first == second
    assert first.filename == "notes.pdf"
    assert rag.ingests == 1
    controller.reset(records)
    assert not records and rag.resets == 1
