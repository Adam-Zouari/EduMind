"""Weaviate gRPC-backed Python client adapter."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from .base import Config, Hit, Record, ensure_dimension


class Weaviate:
    def __init__(self, config: Config) -> None:
        import weaviate

        self.config = config
        self.client = weaviate.connect_to_custom(
            http_host="127.0.0.1",
            http_port=8080,
            http_secure=False,
            grpc_host="127.0.0.1",
            grpc_port=50051,
            grpc_secure=False,
        )
        self.collection: Any | None = None

    def health(self) -> bool:
        return bool(self.client.is_ready())

    def reset(self) -> None:
        from weaviate.classes.config import Configure, DataType, Property, VectorDistances

        if self.client.collections.exists(self.config.collection):
            self.client.collections.delete(self.config.collection)
        self.client.collections.create(
            self.config.collection,
            properties=[
                Property(name="edumind_id", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source_id", data_type=DataType.TEXT),
                Property(name="scope_50", data_type=DataType.TEXT),
                Property(name="scope_10", data_type=DataType.TEXT),
                Property(name="scope_1", data_type=DataType.TEXT),
                Property(name="scope_01", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.TEXT),
                Property(name="start", data_type=DataType.INT),
                Property(name="end", data_type=DataType.INT),
                Property(name="token_count", data_type=DataType.INT),
                Property(name="chunking_fingerprint", data_type=DataType.TEXT),
                Property(name="embedding_fingerprint", data_type=DataType.TEXT),
            ],
            vector_config=Configure.Vectors.self_provided(
                distance_metric=VectorDistances.COSINE,
                vector_index_config=Configure.VectorIndex.hnsw(
                    ef_construction=self.config.ef_construction,
                    max_connections=self.config.m,
                    dynamic_ef_min=self.config.ef_search,
                    dynamic_ef_max=self.config.ef_search,
                ),
            ),
        )
        self.collection = self.client.collections.get(self.config.collection)

    def upsert(self, records: Sequence[Record]) -> None:
        ensure_dimension(self.config, records)
        collection = self._collection()
        with collection.batch.fixed_size(batch_size=200) as batch:
            for row in records:
                batch.add_object(
                    uuid=_uuid(row.identifier),
                    properties={"edumind_id": row.identifier, "text": row.text, **dict(row.metadata)},
                    vector=list(row.vector),
                )
        if collection.batch.failed_objects:
            raise RuntimeError(f"Weaviate rejected {len(collection.batch.failed_objects)} records")

    def search(self, vector, limit, filters=None) -> list[Hit]:
        from weaviate.classes.query import MetadataQuery

        result = self._collection().query.near_vector(
            near_vector=list(vector),
            limit=limit,
            filters=_filter(filters or {}),
            return_metadata=MetadataQuery(distance=True),
        )
        hits = []
        for row in result.objects:
            properties = dict(row.properties or {})
            identifier = str(properties.pop("edumind_id", row.uuid))
            properties.pop("text", None)
            hits.append(Hit(identifier, 1.0 - float(row.metadata.distance or 0.0), properties))
        return hits

    def delete(self, identifiers: Sequence[str]) -> None:
        for identifier in identifiers:
            self._collection().data.delete_by_id(_uuid(identifier))

    def delete_document(self, source_id: str) -> int:
        from weaviate.classes.query import Filter

        result = self._collection().data.delete_many(
            where=Filter.by_property("source_id").equal(source_id)
        )
        return int(getattr(result, "successful", 0))

    def count(self) -> int:
        result = self._collection().aggregate.over_all(total_count=True)
        return int(result.total_count or 0)

    def index_info(self) -> Mapping[str, object]:
        config = self._collection().config.get()
        value = str(config.vector_index_config)
        if "hnsw" not in value.casefold():
            raise RuntimeError("Weaviate did not report HNSW")
        return {"type": "hnsw", "configuration": value}

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

    clauses = [Filter.by_property(key).equal(value) for key, value in sorted(filters.items())]
    result = clauses[0]
    for clause in clauses[1:]:
        result = result & clause
    return result


def _uuid(value: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"edumind:{value}")
