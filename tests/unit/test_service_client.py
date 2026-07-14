from __future__ import annotations

import pytest
import requests

from edumind.pipeline.service_client import ServiceClient


class FakeResponse:
    def __init__(self, payload, error: Exception | None = None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def test_service_client_process_query_readiness_and_reset(tmp_path) -> None:
    source = tmp_path / "notes.pdf"
    source.write_bytes(b"pdf")
    session = FakeSession(
        [
            FakeResponse({"status": "ready"}),
            FakeResponse({"status": "ready"}),
            FakeResponse(
                {
                    "document": {
                        "text": "alpha",
                        "source_name": "notes.pdf",
                        "source_kind": "pdf",
                        "metadata": {"course": "ml"},
                    }
                }
            ),
            FakeResponse({"success": True, "chunks": 1}),
            FakeResponse({"answer": "alpha [1]"}),
            FakeResponse({"success": True}),
        ]
    )
    client = ServiceClient(session=session, timeout_seconds=3)
    assert client.readiness()["rag"] == {"status": "ready"}
    processed = client.process_file(source)
    assert processed["ingest"] == {"success": True, "chunks": 1}
    assert client.query("alpha", top_k=3)["answer"] == "alpha [1]"
    assert client.reset_index()["success"] is True
    assert all(call[2]["timeout"] == 3 for call in session.calls)


def test_service_client_rejects_bad_payloads_and_wraps_transport(tmp_path) -> None:
    source = tmp_path / "x.pdf"
    source.write_bytes(b"x")
    client = ServiceClient(session=FakeSession([FakeResponse({"missing": True})]))
    with pytest.raises(RuntimeError, match="no document"):
        client.process_file(source, ingest=False)

    failure = requests.ConnectionError("offline")
    client = ServiceClient(session=FakeSession([FakeResponse({}, failure)]))
    with pytest.raises(ConnectionError, match="Service request failed"):
        client.reset_index()

    client = ServiceClient(session=FakeSession([FakeResponse(["not", "an", "object"])]))
    with pytest.raises(RuntimeError, match="non-object"):
        client.reset_index()
