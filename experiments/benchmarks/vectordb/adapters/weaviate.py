"""Weaviate gRPC-backed Python client adapter."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from .base import Config, Hit, InvalidIndexState, Record, ensure_dimension


class Weaviate:
    def __init__(self, config: Config) -> None:
        import weaviate
        from weaviate.config import AdditionalConfig, ConnectionConfig, Timeout

        self.config = config
        # The pinned client uses one gRPC channel, not an HTTP-style connection
        # pool for near_vector. Bound concurrent RPCs explicitly and record that
        # distinction; HTTP connection settings alone do not limit gRPC search.
        self._search_slots = threading.BoundedSemaphore(config.connection_pool_limit)
        self.client = weaviate.connect_to_custom(
            http_host="127.0.0.1",
            http_port=8080,
            http_secure=False,
            grpc_host="127.0.0.1",
            grpc_port=50051,
            grpc_secure=False,
            additional_config=AdditionalConfig(
                connection=ConnectionConfig(
                    session_pool_connections=self.config.connection_pool_limit,
                    session_pool_maxsize=self.config.connection_pool_limit,
                    session_pool_max_retries=0,
                ),
                timeout=Timeout(
                    init=self.config.request_timeout_seconds,
                    query=self.config.request_timeout_seconds,
                    insert=self.config.request_timeout_seconds,
                ),
            ),
        )
        self.collection: Any | None = None

    def health(self) -> bool:
        return bool(self.client.is_ready())

    def reset(self) -> None:
        from weaviate.classes.config import (
            Configure,
            DataType,
            Property,
            Tokenization,
            VectorDistances,
        )

        if self.client.collections.exists(self.config.collection):
            self.client.collections.delete(self.config.collection)
        self.client.collections.create(
            self.config.collection,
            properties=[
                Property(
                    name="edumind_id",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                ),
                Property(name="text", data_type=DataType.TEXT),
                Property(
                    name="source_id",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                ),
                Property(name="scope_50", data_type=DataType.TEXT),
                Property(name="scope_10", data_type=DataType.TEXT),
                Property(name="scope_1", data_type=DataType.TEXT),
                Property(name="scope_01", data_type=DataType.TEXT),
                Property(
                    name="document_id",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                ),
                Property(name="start", data_type=DataType.INT),
                Property(name="end", data_type=DataType.INT),
                Property(name="token_count", data_type=DataType.INT),
                Property(name="chunking_fingerprint", data_type=DataType.TEXT),
                Property(name="embedding_fingerprint", data_type=DataType.TEXT),
            ],
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                    ef_construction=self.config.ef_construction,
                    max_connections=self.config.m,
                    ef=self.config.ef_search,
                ),
            ),
        )
        self.collection = self.client.collections.get(self.config.collection)

    def upsert(self, records: Sequence[Record]) -> None:
        ensure_dimension(self.config, records)
        collection = self._collection()
        with collection.batch.fixed_size(
            batch_size=self.config.upsert_batch_size
        ) as batch:
            for row in records:
                batch.add_object(
                    uuid=_uuid(row.identifier),
                    properties={
                        "edumind_id": row.identifier,
                        "text": row.text,
                        **dict(row.metadata),
                    },
                    vector=list(row.vector),
                )
        if collection.batch.failed_objects:
            raise RuntimeError(
                f"Weaviate rejected {len(collection.batch.failed_objects)} records"
            )

    def search(self, vector, limit, filters=None) -> list[Hit]:
        from weaviate.classes.query import MetadataQuery

        slots = getattr(self, "_search_slots", None)
        from contextlib import nullcontext

        with slots if slots is not None else nullcontext():
            result = self._collection().query.near_vector(
                near_vector=list(vector),
                limit=limit,
                filters=_filter(filters or {}),
                return_metadata=MetadataQuery(distance=True),
            )
        hits = []
        for row in result.objects:
            properties = _present_properties(row.properties or {})
            identifier = str(properties.pop("edumind_id", row.uuid))
            properties.pop("text", None)
            if row.metadata.distance is None:
                raise ValueError("Weaviate response is missing required distance")
            hits.append(Hit(identifier, 1.0 - float(row.metadata.distance), properties))
        return hits

    def fetch(self, identifiers):
        result = []
        for identifier in identifiers:
            row = self._collection().query.fetch_object_by_id(
                _uuid(identifier), include_vector=True
            )
            if row is not None:
                properties = _present_properties(row.properties)
                stored_id = properties.pop("edumind_id")
                text = properties.pop("text")
                vector = row.vector["default"]
                result.append(Record(stored_id, vector, text, properties))
        return result

    def delete(self, identifiers: Sequence[str]) -> None:
        for identifier in identifiers:
            self._collection().data.delete_by_id(_uuid(identifier))

    def delete_document(self, source_id: str) -> int:
        from weaviate.classes.query import Filter

        collection = self._collection()
        where = Filter.by_property("source_id").equal(source_id)
        expected = collection.aggregate.over_all(
            total_count=True, filters=where
        ).total_count
        if type(expected) is not int or expected < 0:
            raise ValueError("Weaviate document count is unavailable or invalid")
        deleted = 0
        # Each delete_many call is bounded by the server's maximum batch size.
        # The benchmark owns this collection and performs mutations serially.
        while deleted < expected:
            result = collection.data.delete_many(where=where)
            successful = result.successful
            if (
                result.failed
                or type(successful) is not int
                or not 0 < successful <= expected - deleted
            ):
                raise RuntimeError(
                    "Weaviate document deletion failed or made no valid progress: "
                    f"{deleted}/{expected} removed, response={result}"
                )
            deleted += successful
        return deleted

    def count(self) -> int:
        result = self._collection().aggregate.over_all(total_count=True)
        return int(result.total_count or 0)

    def index_info(self) -> Mapping[str, object]:
        config = self._collection().config.get()
        if not config.vector_config or set(config.vector_config) != {"default"}:
            raise InvalidIndexState("Weaviate did not report the default vector index")
        index = config.vector_config["default"].vector_index_config
        if index.vector_index_type() != "hnsw" or index.skip:
            raise InvalidIndexState("Weaviate did not report an enabled HNSW index")
        actual = {
            "space": index.distance_metric.value,
            "max_neighbors": index.max_connections,
            "ef_construction": index.ef_construction,
            "ef_search": index.ef,
        }
        expected = {
            "space": "cosine",
            "max_neighbors": self.config.m,
            "ef_construction": self.config.ef_construction,
            "ef_search": self.config.ef_search,
        }
        if actual != expected:
            raise InvalidIndexState(f"Weaviate HNSW configuration mismatch: {actual}")
        return {"type": "hnsw", "configuration": actual}

    def close(self) -> None:
        self.client.close()

    def _collection(self):
        if self.collection is None:
            self.collection = self.client.collections.get(self.config.collection)
        return self.collection


def _filter(filters: Mapping[str, object]):
    if not filters:
        return None
    from weaviate.classes.query import Filter

    clauses = [
        Filter.by_property(key).equal(value) for key, value in sorted(filters.items())
    ]
    result = clauses[0]
    for clause in clauses[1:]:
        result = result & clause
    return result


def _uuid(value: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"edumind:{value}")


def _present_properties(properties):
    # Weaviate returns null for declared but unset optional fields. They are
    # absent metadata, not added source facts; retain every non-null value.
    return {key: value for key, value in properties.items() if value is not None}
