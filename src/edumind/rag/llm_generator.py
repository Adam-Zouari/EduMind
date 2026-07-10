"""Library-safe Ollama client used by the RAG pipeline."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import requests

from .errors import OllamaConnectionError, OllamaRequestError
from .types import DEFAULT_OLLAMA_TIMEOUT, LLMSettings, RetrievalHit

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer the user's question using the provided context. "
    "If the context is insufficient, say so plainly. Be concise and accurate."
)


class OllamaGenerator:
    """Generate answers with Ollama without doing network work at construction time."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        model_name: str = "qwen3:1.7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        request_timeout: int = DEFAULT_OLLAMA_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        resolved_settings = settings or LLMSettings(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )
        self.model_name = resolved_settings.model_name
        self.base_url = resolved_settings.base_url.rstrip("/")
        self.temperature = resolved_settings.temperature
        self.max_tokens = resolved_settings.max_tokens
        self.request_timeout = resolved_settings.request_timeout
        self.session = session or requests.Session()

    def health_check(self) -> bool:
        """Return whether the Ollama service is reachable."""
        try:
            self.list_models()
        except (OllamaConnectionError, OllamaRequestError):
            return False
        return True

    def list_models(self) -> list[str]:
        """Return all model names advertised by Ollama."""
        response = self._request("GET", "/api/tags", timeout=5)
        payload = response.json()
        models = payload.get("models", [])
        return [str(model.get("name", "")) for model in models if isinstance(model, dict)]

    def generate(
        self,
        query: str,
        context: str,
        *,
        system_prompt: str | None = None,
        stream: bool = False,
    ) -> str:
        """Generate an answer from question plus retrieved context."""
        payload = self._build_generate_payload(
            query=query,
            context=context,
            system_prompt=system_prompt,
            stream=stream,
        )

        if stream:
            return "".join(self.stream_generate(query, context, system_prompt=system_prompt))

        response = self._request(
            "POST",
            "/api/generate",
            json=payload,
            timeout=self.request_timeout,
        )
        return _parse_generate_response(response.json())

    def stream_generate(
        self,
        query: str,
        context: str,
        *,
        system_prompt: str | None = None,
    ) -> Iterator[str]:
        """Yield generated text chunks from Ollama."""
        payload = self._build_generate_payload(
            query=query,
            context=context,
            system_prompt=system_prompt,
            stream=True,
        )
        response = self._request(
            "POST",
            "/api/generate",
            json=payload,
            timeout=self.request_timeout,
            stream=True,
        )
        yield from _iter_stream_text(response, extractor=_extract_generate_chunk_text)

    def generate_with_results(
        self,
        query: str,
        results: Sequence[RetrievalHit],
        *,
        system_prompt: str | None = None,
        stream: bool = False,
    ) -> str:
        """Generate an answer using already retrieved results."""
        context = self.build_context(results)
        return self.generate(query, context, system_prompt=system_prompt, stream=stream)

    def build_context(self, results: Sequence[RetrievalHit]) -> str:
        """Build a textual context block from retrieval hits."""
        context_parts = [
            f"[Document {index}]\n{hit.document}\nSource: {hit.source}, Page: {hit.page}"
            for index, hit in enumerate(results, start=1)
        ]
        return "\n\n".join(context_parts)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
    ) -> str:
        """Send chat-style messages to Ollama."""
        payload = self._build_chat_payload(messages, stream=stream)
        if stream:
            return "".join(self.stream_chat(messages))

        response = self._request(
            "POST",
            "/api/chat",
            json=payload,
            timeout=self.request_timeout,
        )
        return _parse_chat_response(response.json())

    def stream_chat(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield chat chunks from Ollama."""
        payload = self._build_chat_payload(messages, stream=True)
        response = self._request(
            "POST",
            "/api/chat",
            json=payload,
            timeout=self.request_timeout,
            stream=True,
        )
        yield from _iter_stream_text(response, extractor=_extract_chat_chunk_text)

    def _build_generate_payload(
        self,
        *,
        query: str,
        context: str,
        system_prompt: str | None,
        stream: bool,
    ) -> dict[str, object]:
        """Build the Ollama generate payload."""
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        return {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt or DEFAULT_SYSTEM_PROMPT,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

    def _build_chat_payload(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool,
    ) -> dict[str, object]:
        """Build the Ollama chat payload."""
        return {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: int,
        stream: bool = False,
        **kwargs: Any,
    ) -> requests.Response:
        """Perform an Ollama request with consistent error handling."""
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=timeout,
                stream=stream,
                **kwargs,
            )
        except requests.ConnectionError as exc:
            raise OllamaConnectionError(f"Could not connect to Ollama at {self.base_url}") from exc
        except requests.RequestException as exc:
            raise OllamaRequestError(f"Ollama request failed: {exc}") from exc

        if response.status_code >= 400:
            raise OllamaRequestError(
                f"Ollama API request failed ({response.status_code}): {response.text}"
            )
        return response


def _parse_generate_response(payload: Mapping[str, object]) -> str:
    """Decode a non-streaming generate response body."""
    return str(payload.get("response", "")).strip()


def _parse_chat_response(payload: Mapping[str, object]) -> str:
    """Decode a non-streaming chat response body."""
    message = payload.get("message", {})
    if not isinstance(message, Mapping):
        return ""
    return str(message.get("content", "")).strip()


def _iter_stream_text(
    response: requests.Response,
    *,
    extractor: Callable[[Mapping[str, object]], str],
) -> Iterator[str]:
    """Yield decoded text chunks from Ollama's JSON-lines stream."""
    for payload in _iter_stream_payloads(response):
        text = extractor(payload)
        if text:
            yield text
        if payload.get("done", False):
            break


def _iter_stream_payloads(response: requests.Response) -> Iterator[Mapping[str, object]]:
    """Yield decoded JSON objects from an Ollama streaming response."""
    for line in response.iter_lines():
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            yield payload


def _extract_generate_chunk_text(payload: Mapping[str, object]) -> str:
    """Extract text from one generate stream payload."""
    return str(payload.get("response", ""))


def _extract_chat_chunk_text(payload: Mapping[str, object]) -> str:
    """Extract text from one chat stream payload."""
    message = payload.get("message", {})
    if not isinstance(message, Mapping):
        return ""
    return str(message.get("content", ""))

