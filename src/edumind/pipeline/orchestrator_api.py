"""API-first orchestrator for the microservices deployment mode."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


class APIOrchestrator:
    """Orchestrate OCR and RAG through their HTTP services."""

    def __init__(
        self,
        ocr_url: str = "http://localhost:8000",
        rag_url: str = "http://localhost:8001",
    ) -> None:
        self.ocr_url = ocr_url.rstrip("/")
        self.rag_url = rag_url.rstrip("/")
        self._check_services()

    def _check_services(self) -> None:
        try:
            requests.get(f"{self.ocr_url}/health", timeout=5).raise_for_status()
        except Exception as exc:
            raise ConnectionError(f"OCR service not available at {self.ocr_url}") from exc

        try:
            requests.get(f"{self.rag_url}/health", timeout=5).raise_for_status()
        except Exception as exc:
            raise ConnectionError(f"RAG service not available at {self.rag_url}") from exc

    def process_file(self, file_path: str | Path, ingest_to_rag: bool = True) -> dict[str, Any]:
        path = Path(file_path)
        with path.open("rb") as handle:
            response = requests.post(
                f"{self.ocr_url}/extract",
                files={"file": (path.name, handle, "application/octet-stream")},
                timeout=120,
            )
        response.raise_for_status()
        ocr_result = response.json()

        rag_chunks = 0
        rag_source_id: str | None = None
        if ingest_to_rag and ocr_result["success"]:
            ingest_response = requests.post(
                f"{self.rag_url}/ingest",
                json={
                    "document": {
                        "text": ocr_result["text"],
                        "source": path.name,
                        "format_type": ocr_result.get("format_type"),
                        "file_path": str(path),
                        "metadata": ocr_result["metadata"],
                    }
                },
                timeout=120,
            )
            ingest_response.raise_for_status()
            ingest_payload = ingest_response.json()
            rag_chunks = ingest_payload.get("chunks", 0)
            rag_source_id = ingest_payload.get("source_id")

        return {
            "ocr_success": ocr_result["success"],
            "ocr_error": None if ocr_result["success"] else ocr_result.get("error"),
            "text": ocr_result["text"],
            "metadata": ocr_result["metadata"],
            "format_type": ocr_result["format_type"],
            "extraction_time": ocr_result["extraction_time"],
            "file_path": str(path),
            "rag_ingested": ingest_to_rag and rag_chunks > 0,
            "rag_chunks": rag_chunks,
            "rag_source_id": rag_source_id,
        }

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        generate_answer: bool = True,
        filter_metadata: dict[str, str | int | float | bool] | None = None,
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.rag_url}/query",
            json={
                "query": query_text,
                "top_k": top_k,
                "generate_answer": generate_answer,
                "filter_metadata": filter_metadata or {},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def get_stats(self) -> dict[str, Any]:
        rag_stats = requests.get(f"{self.rag_url}/stats", timeout=30).json()
        ocr_formats = requests.get(f"{self.ocr_url}/formats", timeout=30).json()
        return {"rag": rag_stats, "ocr_formats": ocr_formats["formats"]}

    def reset_database(self) -> dict[str, Any]:
        response = requests.delete(f"{self.rag_url}/reset", timeout=30)
        response.raise_for_status()
        return response.json()
