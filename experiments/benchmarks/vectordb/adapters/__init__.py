"""Factory for the four benchmark-only vector server adapters."""

from .base import Adapter, Config, Hit, Record
from .chroma import Chroma
from .pgvector import PgVector
from .qdrant import Qdrant
from .weaviate import Weaviate


def create(name: str, config: Config) -> Adapter:
    adapters = {
        "chroma": Chroma,
        "qdrant": Qdrant,
        "weaviate": Weaviate,
        "pgvector": PgVector,
    }
    try:
        return adapters[name](config)
    except KeyError as exc:
        raise ValueError(f"Unknown vector server candidate: {name}") from exc


__all__ = ["Adapter", "Config", "Hit", "Record", "create"]
