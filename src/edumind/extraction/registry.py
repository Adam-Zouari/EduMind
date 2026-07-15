"""Extractor registry and candidate metadata."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .contracts import Extractor, SourceKind


@dataclass(frozen=True)
class ExtractorRegistration:
    name: str
    kinds: frozenset[SourceKind]
    factory: Callable[[], Extractor]


class ExtractorRegistry:
    def __init__(self) -> None:
        self._registrations: dict[str, ExtractorRegistration] = {}
        self._instances: dict[str, Extractor] = {}

    def register(self, registration: ExtractorRegistration) -> None:
        if registration.name in self._registrations:
            raise ValueError(f"Extractor already registered: {registration.name}")
        self._registrations[registration.name] = registration

    def create(self, name: str, kind: SourceKind) -> Extractor:
        try:
            registration = self._registrations[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown extractor '{name}'. Available: {', '.join(self.names(kind))}"
            ) from exc
        if kind not in registration.kinds:
            raise ValueError(f"Extractor '{name}' does not support {kind.value}")
        if name not in self._instances:
            self._instances[name] = registration.factory()
        return self._instances[name]

    def names(self, kind: SourceKind | None = None) -> list[str]:
        return sorted(
            name for name, item in self._registrations.items() if kind is None or kind in item.kinds
        )
