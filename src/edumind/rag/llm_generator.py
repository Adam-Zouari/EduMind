"""Pinned-profile Ollama generation with citation-grounded prompts."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import requests

from .contracts import GenerationProfile
from .errors import OllamaConnectionError, OllamaRequestError
from .types import RetrievalHit

DEFAULT_SYSTEM_PROMPT = """You are a careful study assistant.
Answer only from the numbered evidence.
Every factual claim must cite one or more evidence blocks using [1], [2], and so on.
If the evidence cannot answer the question, say exactly: I don't have enough evidence to answer.
Do not invent sources, page numbers, facts, or citations. Keep the answer concise."""


@dataclass(frozen=True)
class GenerationMeasurement:
    answer: str
    total_seconds: float
    time_to_first_token_seconds: float
    load_seconds: float
    prompt_evaluation_seconds: float
    generation_seconds: float
    prompt_tokens: int
    answer_tokens: int
    reasoning_words_estimate: int
    tokens_per_second: float


class OllamaGenerator:
    def __init__(
        self,
        profile: GenerationProfile,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: int = 120,
        session: requests.Session | None = None,
    ) -> None:
        self.profile = profile
        self.model_name = profile.model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self._digest_validated = profile.digest == "unpinned"

    def health_check(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.ok
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        payload = self._request("GET", "/api/tags")
        models = payload.get("models", [])
        if not isinstance(models, Sequence):
            return []
        return [str(item.get("name", "")) for item in models if isinstance(item, Mapping)]

    def generate(
        self,
        query: str,
        context: str,
        *,
        system_prompt: str | None = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        self._validate_digest()
        prompt = self._prompt(query, context, system_prompt)
        payload = self._payload(prompt, stream=stream)
        if stream:
            return self._stream(payload)
        response = self._request("POST", "/api/generate", json=payload)
        return str(response.get("response", "")).strip()

    def generate_with_results(
        self,
        query: str,
        results: Sequence[RetrievalHit],
        system_prompt: str | None = None,
        stream: bool = False,
    ) -> str:
        context = self.build_context(results)
        output = self.generate(query, context, system_prompt=system_prompt, stream=stream)
        return "".join(output) if not isinstance(output, str) else output

    def generate_measured_with_results(
        self,
        query: str,
        results: Sequence[RetrievalHit],
        system_prompt: str | None = None,
    ) -> GenerationMeasurement:
        """Stream one answer while retaining timings/counts, never a reasoning trace."""
        self._validate_digest()
        prompt = self._prompt(query, self.build_context(results), system_prompt)
        payload = self._payload(prompt, stream=True)
        started = time.perf_counter()
        first_token_at: float | None = None
        answer_parts: list[str] = []
        reasoning_words_estimate = 0
        final: Mapping[str, object] = {}
        try:
            with self.session.post(
                f"{self.base_url}/api/generate",
                json=dict(payload),  # type: ignore[arg-type]
                stream=True,
                timeout=self.timeout_seconds,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    item = json.loads(line)
                    if not isinstance(item, Mapping):
                        continue
                    response_text = str(item.get("response", ""))
                    thinking_text = str(item.get("thinking", ""))
                    if response_text and first_token_at is None:
                        first_token_at = time.perf_counter()
                    if response_text:
                        answer_parts.append(response_text)
                    if thinking_text:
                        reasoning_words_estimate += len(thinking_text.split())
                    final = item
        except requests.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Ollama is unavailable at {self.base_url}. Start Ollama and install "
                f"{self.model_name}."
            ) from exc
        except requests.RequestException as exc:
            raise OllamaRequestError("Ollama streaming request failed") from exc
        finished = time.perf_counter()
        generation_seconds = _duration_seconds(final.get("eval_duration"))
        answer_tokens = _integer(final.get("eval_count"))
        return GenerationMeasurement(
            "".join(answer_parts).strip(),
            finished - started,
            (first_token_at or finished) - started,
            _duration_seconds(final.get("load_duration")),
            _duration_seconds(final.get("prompt_eval_duration")),
            generation_seconds,
            _integer(final.get("prompt_eval_count")),
            answer_tokens,
            reasoning_words_estimate,
            answer_tokens / generation_seconds if generation_seconds > 0 else 0.0,
        )

    def unload(self) -> None:
        """Explicitly unload the selected model between benchmark candidates."""
        self._request(
            "POST",
            "/api/generate",
            json={"model": self.profile.model_name, "keep_alive": 0, "stream": False},
        )

    def runtime_memory(self) -> dict[str, float]:
        """Return Ollama-reported loaded-model allocation; omit unavailable fields."""
        payload = self._request("GET", "/api/ps")
        rows = payload.get("models", [])
        if not isinstance(rows, Sequence):
            return {}
        row = next(
            (
                value
                for value in rows
                if isinstance(value, Mapping)
                and str(value.get("name", value.get("model", ""))) == self.profile.model_name
            ),
            None,
        )
        if not isinstance(row, Mapping):
            return {}
        result = {}
        size = _integer(row.get("size"))
        size_vram = _integer(row.get("size_vram"))
        if size:
            result["ollama_model_memory_gb"] = size / (1024**3)
        if size_vram:
            result["ollama_model_vram_mb"] = size_vram / (1024**2)
        return result

    @staticmethod
    def build_context(results: Sequence[RetrievalHit]) -> str:
        return "\n\n".join(
            f"[{index}] source={hit.source}; page={hit.page}\n{hit.document}"
            for index, hit in enumerate(results, start=1)
        )

    def _payload(self, prompt: str, *, stream: bool) -> dict[str, object]:
        options: dict[str, object] = {
            "temperature": self.profile.temperature,
            "seed": self.profile.seed,
            "num_ctx": self.profile.context_tokens,
            "num_predict": self.profile.maximum_answer_tokens,
        }
        payload: dict[str, object] = {
            "model": self.profile.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": options,
            "keep_alive": self.profile.keep_alive,
        }
        if self.profile.thinking != "off":
            payload["think"] = self.profile.thinking
        return payload

    @staticmethod
    def _prompt(query: str, context: str, system_prompt: str | None) -> str:
        return (
            f"{system_prompt or DEFAULT_SYSTEM_PROMPT}\n\n"
            f"Evidence:\n{context}\n\nQuestion: {query}\nAnswer with citations:"
        )

    def _validate_digest(self) -> None:
        if self._digest_validated:
            return
        payload = self._request("GET", "/api/tags")
        rows = payload.get("models", [])
        installed = (
            {
                str(item.get("name")): str(item.get("digest"))
                for item in rows
                if isinstance(item, Mapping)
            }
            if isinstance(rows, Sequence)
            else {}
        )
        actual = installed.get(self.profile.model_name)
        if actual is None:
            raise OllamaRequestError(
                f"Pinned Ollama model is missing: {self.profile.model_name}. Run model preparation."
            )
        if actual != self.profile.digest:
            raise OllamaRequestError(
                f"Ollama digest mismatch for {self.profile.model_name}; expected "
                f"{self.profile.digest}, received {actual}. Re-run preparation and benchmarks."
            )
        self._digest_validated = True

    def _request(self, method: str, path: str, **kwargs: Any) -> Mapping[str, object]:
        try:
            response = self.session.request(
                method, f"{self.base_url}{path}", timeout=self.timeout_seconds, **kwargs
            )
            response.raise_for_status()
        except requests.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Ollama is unavailable at {self.base_url}. Start Ollama and install "
                f"{self.model_name}."
            ) from exc
        except requests.RequestException as exc:
            raise OllamaRequestError(f"Ollama request failed for model {self.model_name}") from exc
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise OllamaRequestError("Ollama returned a non-object response")
        return payload

    def _stream(self, payload: Mapping[str, object]) -> Iterator[str]:
        try:
            with self.session.post(
                f"{self.base_url}/api/generate",
                json=dict(payload),  # type: ignore[arg-type]
                stream=True,
                timeout=self.timeout_seconds,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    item = json.loads(line)
                    text = item.get("response", "")
                    if text:
                        yield str(text)
        except requests.RequestException as exc:
            raise OllamaRequestError("Ollama streaming request failed") from exc


def _duration_seconds(value: object) -> float:
    return _integer(value) / 1_000_000_000


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
