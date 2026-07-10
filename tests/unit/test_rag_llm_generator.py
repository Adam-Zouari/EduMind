from __future__ import annotations

import json

import pytest
import requests

from edumind.rag.errors import OllamaConnectionError
from edumind.rag.llm_generator import OllamaGenerator
from edumind.rag.types import RetrievalHit


class FakeResponse:
    def __init__(
        self,
        *,
        json_data: dict[str, object] | None = None,
        status_code: int = 200,
        text: str = "",
        lines: list[bytes] | None = None,
    ) -> None:
        self._json_data = json_data or {}
        self.status_code = status_code
        self.text = text
        self._lines = lines or []

    def json(self) -> dict[str, object]:
        return self._json_data

    def iter_lines(self):
        return iter(self._lines)


class FakeSession:
    def __init__(
        self,
        responses: list[FakeResponse] | None = None,
        exc: Exception | None = None,
    ) -> None:
        self.responses = responses or []
        self.exc = exc
        self.calls: list[dict[str, object]] = []

    def request(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.responses.pop(0)


def test_generate_builds_request_and_returns_response_text() -> None:
    session = FakeSession(
        responses=[FakeResponse(json_data={"response": "Use active recall."})]
    )
    generator = OllamaGenerator(session=session, model_name="test-model")

    answer = generator.generate("How do I study?", "Context block")

    assert answer == "Use active recall."
    assert session.calls[0]["url"] == "http://localhost:11434/api/generate"
    assert session.calls[0]["json"]["model"] == "test-model"


def test_generate_with_results_reuses_result_context() -> None:
    session = FakeSession(
        responses=[FakeResponse(json_data={"response": "Answer from retrieved sources."})]
    )
    generator = OllamaGenerator(session=session)
    results = [
        RetrievalHit(
            id="chunk-1",
            document="Important biology note",
            metadata={"source": "biology.pdf", "page": 7},
            score=0.91,
        )
    ]

    answer = generator.generate_with_results("What matters here?", results)

    assert answer == "Answer from retrieved sources."
    assert "Important biology note" in session.calls[0]["json"]["prompt"]


def test_stream_chat_yields_incremental_text() -> None:
    lines = [
        json.dumps({"message": {"content": "Hello "}}).encode("utf-8"),
        json.dumps({"message": {"content": "world"}, "done": True}).encode("utf-8"),
    ]
    session = FakeSession(responses=[FakeResponse(lines=lines)])
    generator = OllamaGenerator(session=session)

    chunks = list(generator.stream_chat([{"role": "user", "content": "hi"}]))

    assert chunks == ["Hello ", "world"]


def test_connection_errors_raise_typed_error() -> None:
    session = FakeSession(exc=requests.ConnectionError("offline"))
    generator = OllamaGenerator(session=session)

    assert generator.health_check() is False
    with pytest.raises(OllamaConnectionError):
        generator.list_models()
