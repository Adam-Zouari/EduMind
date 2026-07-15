"""PostgreSQL 17 + pgvector HNSW adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from .base import Config, Hit, Record, ensure_dimension


class PgVector:
    def __init__(self, config: Config) -> None:
        from psycopg_pool import ConnectionPool

        self.config = config
        self.pool = ConnectionPool(
            "postgresql://edumind:edumind@127.0.0.1:5433/edumind",
            min_size=1,
            max_size=64,
            kwargs={"autocommit": True},
            open=True,
        )

    def health(self) -> bool:
        with self.pool.connection() as connection:
            return connection.execute("SELECT 1").fetchone() == (1,)

    def reset(self) -> None:
        with self.pool.connection() as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute("DROP TABLE IF EXISTS edumind_benchmark")
            connection.execute(
                "CREATE TABLE edumind_benchmark ("
                "id text PRIMARY KEY, "
                f"embedding vector({self.config.dimension}) NOT NULL, "
                "text text NOT NULL, metadata jsonb NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX edumind_benchmark_metadata ON edumind_benchmark USING gin (metadata)"
            )

    def finish_index(self) -> None:
        with self.pool.connection() as connection:
            connection.execute("DROP INDEX IF EXISTS edumind_benchmark_hnsw")
            connection.execute(
                "CREATE INDEX edumind_benchmark_hnsw ON edumind_benchmark "
                "USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m={self.config.m}, ef_construction={self.config.ef_construction})"
            )

    def upsert(self, records: Sequence[Record]) -> None:
        ensure_dimension(self.config, records)
        rows = [
            (row.identifier, _vector(row.vector), row.text, json.dumps(row.metadata))
            for row in records
        ]
        with self.pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO edumind_benchmark (id, embedding, text, metadata) "
                    "VALUES (%s, %s::vector, %s, %s::jsonb) "
                    "ON CONFLICT (id) DO UPDATE SET embedding=excluded.embedding, "
                    "text=excluded.text, metadata=excluded.metadata",
                    rows,
                )

    def search(self, vector, limit, filters=None) -> list[Hit]:
        clauses = []
        parameters: list[object] = [_vector(vector)]
        for key, value in sorted((filters or {}).items()):
            clauses.append("metadata @> %s::jsonb")
            parameters.append(json.dumps({key: value}))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        parameters.extend([_vector(vector), limit])
        with self.pool.connection() as connection:
            with connection.transaction():
                connection.execute(f"SET LOCAL hnsw.ef_search = {self.config.ef_search}")
                if filters:
                    connection.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
                connection.execute("SET LOCAL enable_seqscan = off")
                rows = connection.execute(
                    "SELECT id, metadata, 1 - (embedding <=> %s::vector) AS score "
                    f"FROM edumind_benchmark{where} ORDER BY embedding <=> %s::vector LIMIT %s",
                    tuple(parameters),
                ).fetchall()
        return [Hit(str(row[0]), float(row[2]), row[1]) for row in rows]

    def delete(self, identifiers: Sequence[str]) -> None:
        if identifiers:
            with self.pool.connection() as connection:
                connection.execute(
                    "DELETE FROM edumind_benchmark WHERE id = ANY(%s)", (list(identifiers),)
                )

    def delete_document(self, source_id: str) -> int:
        with self.pool.connection() as connection:
            result = connection.execute(
                "DELETE FROM edumind_benchmark WHERE metadata @> %s::jsonb",
                (json.dumps({"source_id": source_id}),),
            )
        return int(result.rowcount or 0)

    def count(self) -> int:
        with self.pool.connection() as connection:
            return int(connection.execute("SELECT count(*) FROM edumind_benchmark").fetchone()[0])

    def index_info(self) -> Mapping[str, object]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                "SELECT indexname, indexdef FROM pg_indexes WHERE tablename='edumind_benchmark'"
            ).fetchall()
        result = {str(name): str(definition) for name, definition in rows}
        if not any("USING hnsw" in definition for definition in result.values()):
            raise RuntimeError("pgvector did not report HNSW")
        with self.pool.connection() as connection:
            with connection.transaction():
                connection.execute("SET LOCAL enable_seqscan=off")
                explain = connection.execute(
                    "EXPLAIN SELECT id FROM edumind_benchmark "
                    "ORDER BY embedding <=> %s::vector LIMIT 10",
                    ("[" + ",".join("0" for _ in range(self.config.dimension)) + "]",),
                ).fetchall()
        if "Index Scan" not in " ".join(str(row[0]) for row in explain):
            raise RuntimeError("PostgreSQL planner did not use the HNSW index")
        return result

    def close(self) -> None:
        self.pool.close()


def _vector(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in values) + "]"
