"""Qdrant server adapter using the REST client path."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence

from .base import Config, Hit, Record, ensure_dimension


class Qdrant:
    def __init__(self, config: Config) -> None:
        from qdrant_client import QdrantClient, models

        self.config = config
        self.models = models
        self.client = QdrantClient(url="http://127.0.0.1:6333", timeout=120)

    def health(self) -> bool:
        return self.client.get_collections() is not None

    def reset(self) -> None:
        if self.client.collection_exists(self.config.collection):
            self.client.delete_collection(self.config.collection)
        self.client.create_collection(
            self.config.collection,
            vectors_config=self.models.VectorParams(
                size=self.config.dimension,
                distance=self.models.Distance.COSINE,
                hnsw_config=self.models.HnswConfigDiff(
                    m=self.config.m,
                    ef_construct=self.config.ef_construction,
                    full_scan_threshold=0,
                ),
            ),
            optimizers_config=self.models.OptimizersConfigDiff(indexing_threshold=1),
        )
        for field in ("source_id", "scope_50", "scope_10", "scope_1", "scope_01"):
            self.client.create_payload_index(
                self.config.collection, field, self.models.PayloadSchemaType.KEYWORD, wait=True
            )

    def upsert(self, records: Sequence[Record]) -> None:
        ensure_dimension(self.config, records)
        for start in range(0, len(records), 2_000):
            rows = records[start : start + 2_000]
            self.client.upsert(
                self.config.collection,
                points=[
                    self.models.PointStruct(
                        id=str(_uuid(row.identifier)),
                        vector=list(row.vector),
                        payload={"edumind_id": row.identifier, "text": row.text, **dict(row.metadata)},
                    )
                    for row in rows
                ],
                wait=True,
            )

    def search(self, vector, limit, filters=None) -> list[Hit]:
        query_filter = None
        if filters:
            query_filter = self.models.Filter(
                must=[
                    self.models.FieldCondition(key=key, match=self.models.MatchValue(value=value))
                    for key, value in sorted(filters.items())
                ]
            )
        result = self.client.query_points(
            self.config.collection,
            query=list(vector),
            query_filter=query_filter,
            limit=limit,
            search_params=self.models.SearchParams(hnsw_ef=self.config.ef_search, exact=False),
            with_payload=True,
        )
        hits = []
        for point in result.points:
            payload = dict(point.payload or {})
            identifier = str(payload.pop("edumind_id", point.id))
            payload.pop("text", None)
            hits.append(Hit(identifier, float(point.score), payload))
        return hits

    def delete(self, identifiers: Sequence[str]) -> None:
        if identifiers:
            self.client.delete(
                self.config.collection,
                self.models.PointIdsList(points=[str(_uuid(value)) for value in identifiers]),
                wait=True,
            )

    def delete_document(self, source_id: str) -> int:
        condition = self.models.Filter(
            must=[self.models.FieldCondition(key="source_id", match=self.models.MatchValue(value=source_id))]
        )
        count = int(self.client.count(self.config.collection, count_filter=condition, exact=True).count)
        self.client.delete(
            self.config.collection,
            self.models.FilterSelector(filter=condition),
            wait=True,
        )
        return count

    def count(self) -> int:
        return int(self.client.count(self.config.collection, exact=True).count)

    def index_info(self) -> Mapping[str, object]:
        deadline = time.monotonic() + 120
        while True:
            info = self.client.get_collection(self.config.collection)
            status = str(info.status).casefold()
            points = int(info.points_count or 0)
            indexed = int(info.indexed_vectors_count or 0)
            if "green" in status and (not points or indexed >= points):
                return {"type": "hnsw", "status": status, "indexed": indexed, "points": points}
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Qdrant HNSW was not ready: {indexed}/{points}, {status}")
            time.sleep(0.25)

    def close(self) -> None:
        self.client.close()


def _uuid(value: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"edumind:{value}")
