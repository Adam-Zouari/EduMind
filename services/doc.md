# Services

The FastAPI adapters expose the same production extraction and RAG implementations used by the local pipeline.

- `extraction_service.py` streams uploads to a bounded temporary file, enforces the configurable 100 MiB default, validates the source, cleans up temporary data, and returns `ExtractedDocument` data through a stable error schema.
- `rag_service.py` exposes ingestion, retrieval/generation, statistics, and serialized reset operations.

Both APIs bind to `127.0.0.1` through the public CLI. `/health/live` reports process liveness; `/health/ready` checks whether required runtime components are ready. Public errors contain stable codes and safe messages, while detailed exceptions remain in internal logs. These services are local single-user adapters, not hardened public internet APIs.
