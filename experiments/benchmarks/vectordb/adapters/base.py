"""Small common contract for benchmark-only vector server adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


class InvalidIndexState(RuntimeError):
    """The server index cannot be trusted for the requested benchmark trial."""


@dataclass(frozen=True)
class Config:
    dimension: int
    m: int = 16
    ef_construction: int = 100
    ef_search: int = 64
    collection: str = "edumind_benchmark"


@dataclass(frozen=True)
class Record:
    identifier: str
    vector: Sequence[float]
    text: str
    metadata: Mapping[str, str | int | float | bool]


@dataclass(frozen=True)
class Hit:
    identifier: str
    score: float
    metadata: Mapping[str, object]


class Adapter(Protocol):
    config: Config

    def health(self) -> bool: ...
    def reset(self) -> None: ...
    def upsert(self, records: Sequence[Record]) -> None: ...
    def search(
        self, vector: Sequence[float], limit: int, filters: Mapping[str, object] | None = None
    ) -> list[Hit]: ...
    def delete(self, identifiers: Sequence[str]) -> None: ...
    def delete_document(self, source_id: str) -> int: ...
    def count(self) -> int: ...
    def index_info(self) -> Mapping[str, object]: ...
    def close(self) -> None: ...


def ensure_dimension(config: Config, records: Sequence[Record]) -> None:
    for record in records:
        if len(record.vector) != config.dimension:
            raise ValueError(
                f"Vector {record.identifier} has dimension {len(record.vector)}; "
                f"expected {config.dimension}"
            )
