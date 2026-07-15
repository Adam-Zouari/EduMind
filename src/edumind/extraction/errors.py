"""Stable extraction failures suitable for the local application."""

from __future__ import annotations


class ExtractionError(RuntimeError):
    code = "extraction_error"
    recoverable = False

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.public_message = message
        self.detail = detail


class UnsupportedSourceError(ExtractionError):
    code = "unsupported_source"


class MissingDependencyError(ExtractionError):
    code = "missing_dependency"
    recoverable = True


class ExtractionBackendError(ExtractionError):
    code = "backend_failed"
    recoverable = True
