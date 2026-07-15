"""Streamlit session-state value objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum


class DocumentStatus(str, Enum):
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class DocumentRecord:
    checksum: str
    filename: str
    status: DocumentStatus
    source_kind: str | None = None
    characters: int = 0
    chunks: int = 0
    timings: Mapping[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> DocumentRecord:
        raw_timings = payload.get("timings", {})
        timings = (
            {
                str(key): float(value)
                for key, value in raw_timings.items()
                if isinstance(value, (int, float))
            }
            if isinstance(raw_timings, Mapping)
            else {}
        )
        raw_warnings = payload.get("warnings", [])
        warnings = raw_warnings if isinstance(raw_warnings, (list, tuple)) else []
        return cls(
            checksum=str(payload.get("checksum", "")),
            filename=str(payload.get("filename", "")),
            status=DocumentStatus(str(payload.get("status", "failed"))),
            source_kind=str(payload["source_kind"]) if payload.get("source_kind") else None,
            characters=_int_value(payload.get("characters")),
            chunks=_int_value(payload.get("chunks")),
            timings=timings,
            warnings=tuple(str(item) for item in warnings),
            error=str(payload["error"]) if payload.get("error") else None,
        )


def _int_value(value: object) -> int:
    return int(value) if isinstance(value, (str, int, float)) else 0
