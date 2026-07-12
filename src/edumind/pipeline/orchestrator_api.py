"""API-first orchestrator for the microservices deployment mode."""

from __future__ import annotations

import logging
from collections.abc import Mapping
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
        *,
        session: requests.Session | None = None,
        verify_on_init: bool = True,
        health_timeout: int = 5,
        request_timeout: int = 120,
        stats_timeout: int = 30,
    ) -> None:
        self.ocr_url = ocr_url.rstrip("/")
        self.rag_url = rag_url.rstrip("/")
        self.session = session or requests.Session()
        self.health_timeout = health_timeout
        self.request_timeout = request_timeout
        self.stats_timeout = stats_timeout
        if verify_on_init:
            self.check_services()

    def check_services(self) -> None:
        """Verify that both OCR and RAG services are reachable."""
        self._request_json(
            "GET",
            f"{self.ocr_url}/health",
            timeout=self.health_timeout,
            error_message=f"OCR service not available at {self.ocr_url}",
        )
        self._request_json(
            "GET",
            f"{self.rag_url}/health",
            timeout=self.health_timeout,
            error_message=f"RAG service not available at {self.rag_url}",
        )

    def process_file(
        self,
        file_path: str | Path,
        ingest_to_rag: bool = True,
    ) -> dict[str, Any]:
        """Process one file through the OCR and optional RAG HTTP boundaries."""
        path = Path(file_path)
        with path.open("rb") as handle:
            ocr_result = self._request_json(
                "POST",
                f"{self.ocr_url}/extract",
                timeout=self.request_timeout,
                files={"file": (path.name, handle, "application/octet-stream")},
            )

        ocr_success = bool(ocr_result.get("success", False))
        ocr_text = str(ocr_result.get("text", ""))
        metadata = _coerce_metadata(ocr_result.get("metadata"))
        format_type = str(ocr_result.get("format_type", ""))
        extraction_time = _coerce_float(ocr_result.get("extraction_time"))
        rag_ingested = False
        rag_chunks = 0
        rag_source_id: str | None = None
        rag_error: str | None = None

        if ingest_to_rag and ocr_success and ocr_text.strip():
            try:
                ingest_payload = self._request_json(
                    "POST",
                    f"{self.rag_url}/ingest",
                    timeout=self.request_timeout,
                    json=self._build_ingest_payload(
                        file_path=path,
                        text=ocr_text,
                        metadata=metadata,
                        format_type=format_type,
                    ),
                )
                rag_ingested = bool(ingest_payload.get("success", True))
                rag_chunks = int(ingest_payload.get("chunks", 0) or 0)
                rag_source_id = _optional_string(ingest_payload.get("source_id"))
            except (ConnectionError, RuntimeError) as exc:
                logger.error("Remote RAG ingestion failed for %s: %s", path, exc)
                rag_error = str(exc)

        return {
            "ocr_success": ocr_success,
            "ocr_error": None if ocr_success else _optional_string(ocr_result.get("error")),
            "text": ocr_text,
            "metadata": metadata,
            "format_type": format_type,
            "extraction_time": extraction_time,
            "file_path": str(path),
            "rag_ingested": rag_ingested,
            "rag_chunks": rag_chunks,
            "rag_source_id": rag_source_id,
            "rag_error": rag_error,
        }

    def process_batch(
        self,
        file_paths: list[str | Path],
        ingest_to_rag: bool = True,
    ) -> list[dict[str, Any]]:
        """Process many files through the OCR and RAG HTTP boundaries."""
        return [
            self.process_file(file_path, ingest_to_rag=ingest_to_rag)
            for file_path in file_paths
        ]

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        generate_answer: bool = True,
        filter_metadata: dict[str, str | int | float | bool] | None = None,
    ) -> dict[str, Any]:
        """Query the remote RAG service and return its JSON payload."""
        return self._request_json(
            "POST",
            f"{self.rag_url}/query",
            timeout=self.request_timeout,
            json={
                "query": query_text,
                "top_k": top_k,
                "generate_answer": generate_answer,
                "filter_metadata": filter_metadata or {},
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Return combined remote OCR and RAG runtime stats."""
        rag_stats = self._request_json(
            "GET",
            f"{self.rag_url}/stats",
            timeout=self.stats_timeout,
        )
        ocr_formats_payload = self._request_json(
            "GET",
            f"{self.ocr_url}/formats",
            timeout=self.stats_timeout,
        )
        raw_formats = ocr_formats_payload.get("formats", [])
        ocr_formats = [str(value) for value in raw_formats] if isinstance(raw_formats, list) else []
        return {
            "rag": rag_stats,
            "ocr_formats": ocr_formats,
            "ocr_extractors": ocr_formats,
        }

    def reset_rag(self) -> dict[str, Any]:
        """Reset the remote RAG index."""
        return self._request_json(
            "DELETE",
            f"{self.rag_url}/reset",
            timeout=self.stats_timeout,
        )

    def reset_database(self) -> dict[str, Any]:
        """Backward-compatible alias for resetting the remote RAG index."""
        return self.reset_rag()

    def _build_ingest_payload(
        self,
        *,
        file_path: Path,
        text: str,
        metadata: dict[str, object],
        format_type: str,
    ) -> dict[str, object]:
        """Build the nested ingest contract expected by the RAG service."""
        return {
            "document": {
                "text": text,
                "source": file_path.name,
                "format_type": format_type,
                "file_path": str(file_path),
                "metadata": metadata,
            }
        }

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        timeout: int,
        error_message: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Perform one HTTP request and require a JSON object response."""
        try:
            response = self.session.request(method=method, url=url, timeout=timeout, **kwargs)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ConnectionError(error_message or f"Request failed: {method} {url}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON response from {url}") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Expected JSON object response from {url}")
        return dict(payload)


def _coerce_metadata(value: object) -> dict[str, object]:
    """Normalize metadata payloads returned by remote services."""
    return dict(value) if isinstance(value, dict) else {}


def _optional_string(value: object) -> str | None:
    """Normalize optional string-like values."""
    return value if isinstance(value, str) else None


def _coerce_float(value: object) -> float:
    """Normalize JSON numeric values into floats."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
