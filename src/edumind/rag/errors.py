"""RAG-specific exceptions."""

from __future__ import annotations


class RAGConfigurationError(ValueError):
    """Raised when the RAG configuration is invalid."""


class MetadataFilterError(ValueError):
    """Raised when metadata filters are unsupported or malformed."""


class OllamaConnectionError(ConnectionError):
    """Raised when the Ollama service cannot be reached."""


class OllamaRequestError(RuntimeError):
    """Raised when an Ollama request fails."""
