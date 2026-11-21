"""Factory for the four benchmark-only vector server adapters."""

from .base import Adapter, Config, Hit, InvalidIndexState, Record
from .chroma import Chroma
from .pgvector import PgVector
from .qdrant import Qdrant
from .weaviate import Weaviate

ADAPTERS = {
    "chroma": Chroma,
    "qdrant": Qdrant,
    "weaviate": Weaviate,
    "pgvector": PgVector,
}


def create(name: str, config: Config) -> Adapter:
    try:
        return ADAPTERS[name](config)
    except KeyError as exc:
        raise ValueError(f"Unknown vector server candidate: {name}") from exc


__all__ = ["Adapter", "Config", "Hit", "InvalidIndexState", "Record", "create"]
