"""Testable application actions independent of Streamlit."""

from __future__ import annotations

import hashlib
import logging
import tempfile
from collections.abc import Callable, MutableMapping
from pathlib import Path

from edumind.application import (
    DocumentProcessResult,
    EduMindPipeline,
    PipelineQueryResult,
    ProgressEvent,
)

from .state import DocumentRecord, DocumentStatus

LOGGER = logging.getLogger(__name__)


class AppController:
    def __init__(self, pipeline: EduMindPipeline) -> None:
        self.pipeline = pipeline

    def process_upload(
        self,
        filename: str,
        content: bytes,
        records: MutableMapping[str, dict[str, object]],
        *,
        progress: Callable[[ProgressEvent], None] | None = None,
    ) -> tuple[DocumentRecord, bool]:
        checksum = hashlib.sha256(content).hexdigest()
        if checksum in records:
            return DocumentRecord.from_dict(records[checksum]), False
        safe_name = Path(filename).name or "upload.bin"
        maximum = self.pipeline.extraction.settings.extraction.maximum_upload_bytes
        if len(content) > maximum:
            record = DocumentRecord(
                checksum,
                safe_name,
                DocumentStatus.FAILED,
                error=f"Upload exceeds the configured {maximum / (1024 ** 2):.0f} MiB limit.",
            )
            records[checksum] = record.to_dict()
            return record, True
        suffix = Path(safe_name).suffix[:16]
        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, prefix="edumind-ui-"
            ) as handle:
                handle.write(content)
                path = Path(handle.name)
            result = self.pipeline.process_file(
                path,
                ingest=True,
                source_name=safe_name,
                progress=progress,
            )
            record = self._record(checksum, safe_name, result)
        except Exception as exc:
            LOGGER.exception("Failed to process upload %s", safe_name)
            record = DocumentRecord(
                checksum, safe_name, DocumentStatus.FAILED, error=safe_error(exc)
            )
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
        if record.status == DocumentStatus.READY:
            stale = [
                key
                for key, value in records.items()
                if key != checksum and str(value.get("filename")) == safe_name
            ]
            for key in stale:
                del records[key]
        records[checksum] = record.to_dict()
        return record, True

    def query(self, text: str, *, top_k: int = 5) -> PipelineQueryResult:
        return self.pipeline.query(text, top_k=top_k, generate_answer=True)

    def reset(self, records: MutableMapping[str, dict[str, object]]) -> None:
        self.pipeline.reset_index()
        records.clear()

    def readiness(self) -> dict[str, object]:
        return self.pipeline.readiness()

    @staticmethod
    def _record(checksum: str, filename: str, result: DocumentProcessResult) -> DocumentRecord:
        ingest = result.ingest
        return DocumentRecord(
            checksum,
            filename,
            DocumentStatus.READY,
            source_kind=result.extraction.source_kind.value,
            characters=len(result.extraction.text),
            chunks=ingest.chunks_created if ingest else 0,
            timings=result.timings,
            warnings=result.warnings,
        )


def safe_error(exc: Exception) -> str:
    name = type(exc).__name__
    if name in {"MissingDependencyError", "RAGConfigurationError"}:
        return str(exc)
    return "Processing failed. Check the local application logs for details."
