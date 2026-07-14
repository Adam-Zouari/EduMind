"""Typed client for optional extraction and RAG service deployment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import requests


class ServiceClient:
    def __init__(
        self,
        extraction_url: str = "http://127.0.0.1:8000",
        rag_url: str = "http://127.0.0.1:8001",
        *,
        session: requests.Session | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.extraction_url = extraction_url.rstrip("/")
        self.rag_url = rag_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def readiness(self) -> dict[str, object]:
        return {
            "extraction": self._request("GET", f"{self.extraction_url}/health/ready"),
            "rag": self._request("GET", f"{self.rag_url}/health/ready"),
        }

    def process_file(self, file_path: str | Path, *, ingest: bool = True) -> dict[str, object]:
        path = Path(file_path)
        with path.open("rb") as handle:
            extraction = self._request(
                "POST",
                f"{self.extraction_url}/extract",
                files={"file": (path.name, handle, "application/octet-stream")},
            )
        document = extraction.get("document")
        if not isinstance(document, Mapping):
            raise RuntimeError("Extraction service returned no document")
        ingest_result = None
        if ingest:
            ingest_result = self._request(
                "POST",
                f"{self.rag_url}/ingest",
                json={
                    "document": {
                        "text": document.get("text", ""),
                        "source": document.get("source_name", path.name),
                        "format_type": document.get("source_kind"),
                        "file_path": path.name,
                        "metadata": document.get("metadata", {}),
                    }
                },
            )
        return {"extraction": dict(document), "ingest": ingest_result}

    def query(
        self, query: str, *, top_k: int = 5, generate_answer: bool = True
    ) -> dict[str, object]:
        return self._request(
            "POST",
            f"{self.rag_url}/query",
            json={
                "query": query,
                "top_k": top_k,
                "generate_answer": generate_answer,
                "filter_metadata": {},
            },
        )

    def reset_index(self) -> dict[str, object]:
        return self._request("DELETE", f"{self.rag_url}/reset")

    def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, object]:
        try:
            response = self.session.request(method, url, timeout=self.timeout_seconds, **kwargs)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise ConnectionError(f"Service request failed: {method} {url}") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Service returned a non-object response: {url}")
        return dict(payload)
