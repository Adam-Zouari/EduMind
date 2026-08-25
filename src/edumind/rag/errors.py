"""RAG-specific exceptions."""

from __future__ import annotations


class RAGConfigurationError(ValueError):
    """Raised when the RAG configuration is invalid."""


class IndexCompatibilityError(RAGConfigurationError):
    """Raised when persisted vectors do not match the active contracts."""


class MetadataFilterError(ValueError):
    """Raised when metadata filters are unsupported or malformed."""


class ModelLoadError(RuntimeError):
    """Raised when a pinned local model cannot be loaded exactly as configured."""


class GenerationError(RuntimeError):
    """Raised when local Hugging Face generation fails."""
