from __future__ import annotations

import json

import pytest

from edumind.rag.contracts import GenerationProfile
from edumind.rag.errors import OllamaRequestError
from edumind.rag.llm_generator import OllamaGenerator
from edumind.rag.types import RetrievalHit


def test_ollama_payload_pins_seed_limits_and_thinking() -> None:
    generator = OllamaGenerator(GenerationProfile("model", "digest", "medium", 0.0, 42, 8192, 256))
    payload = generator._payload("prompt", stream=False)
    assert payload["think"] == "medium"
    assert payload["options"] == {
        "temperature": 0.0,
        "seed": 42,
        "num_ctx": 8192,
        "num_predict": 256,
    }
    assert payload["keep_alive"] == "5m"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, digest):
        self.digest = digest
        self.calls = 0

    def request(self, method, url, **kwargs):
        self.calls += 1
        return FakeResponse({"models": [{"name": "model", "digest": self.digest}]})


def test_ollama_validates_pinned_digest_once() -> None:
    session = FakeSession("sha256:correct")
    generator = OllamaGenerator(
        GenerationProfile("model", "sha256:correct", "off", 0.0, 42, 8192, 256),
        session=session,
    )
    generator._validate_digest()
    generator._validate_digest()
    assert session.calls == 1

    mismatched = OllamaGenerator(
        GenerationProfile("model", "sha256:expected", "off", 0.0, 42, 8192, 256),
        session=FakeSession("sha256:other"),
    )
    with pytest.raises(OllamaRequestError, match="digest mismatch"):
        mismatched._validate_digest()


class StreamingResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        rows = [
            {"thinking": "brief private reasoning"},
            {"response": "Supported [1]."},
            {
                "done": True,
                "load_duration": 2_000_000_000,
                "prompt_eval_duration": 500_000_000,
                "eval_duration": 1_000_000_000,
                "prompt_eval_count": 20,
                "eval_count": 10,
            },
        ]
        return [json.dumps(row).encode() for row in rows]


class StreamingSession:
    def post(self, *args, **kwargs):
        return StreamingResponse()


def test_ollama_measured_generation_keeps_only_reasoning_count() -> None:
    generator = OllamaGenerator(
        GenerationProfile("model", "unpinned", "off", 0.0, 42, 8192, 256),
        session=StreamingSession(),
    )
    hit = RetrievalHit("1", "evidence", {"source": "notes"}, 1.0, 1, "dense", 1)
    measurement = generator.generate_measured_with_results("question", [hit])
    assert measurement.answer == "Supported [1]."
    assert measurement.load_seconds == 2.0
    assert measurement.prompt_tokens == 20
    assert measurement.answer_tokens == 10
    assert measurement.reasoning_tokens_estimate == 3
    assert measurement.tokens_per_second == 10.0
