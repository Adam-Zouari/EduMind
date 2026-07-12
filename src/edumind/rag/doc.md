# RAG Package Guide

This document describes the final RAG architecture under `src/edumind/rag` after the runtime-boundary redesign. It focuses on the live code path, typed contracts, retrieval behavior, service startup model, and the self-test flow you can use locally.

## Current status

The RAG subsystem is now organized around one package-first flow:

1. OCR or another upstream producer yields an extracted document payload
2. `OCRProcessor` normalizes it into a typed `IngestDocument`
3. `TextChunker` splits the text into deterministic `ChunkRecord` objects
4. `Embedder` attaches embeddings using one shared sentence-transformers runtime
5. `VectorStore` upserts those chunks into dense and lexical retrieval state
6. `RAGPipeline` queries once and optionally passes the retrieved hits to `OllamaGenerator`

The main supported product boundaries are now:

- `apps/streamlit_app.py` as the primary UI
- `services/rag_service.py` as the main RAG API
- `src/edumind/pipeline/orchestrator.py` as the local OCR-to-RAG coordinator

## Package layout

```text
src/edumind/rag/
|-- __init__.py
|-- doc.md
|-- embedder.py
|-- errors.py
|-- llm_generator.py
|-- ocr_processor.py
|-- rag_pipeline.py
|-- serializers.py
|-- text_chunker.py
|-- types.py
`-- vector_store.py
```

## File roles

## `types.py`

This is the typed contract layer for the subsystem.

It defines:

- `RAGConfig` and the per-subsystem settings dataclasses
- `IngestDocument`
- `ChunkRecord`
- `RetrievalHit`
- `IngestReport`
- `AnswerResult`
- helper utilities for deterministic source ids and chunk ids
- scalar metadata filtering rules

This file is the reason the rest of the package no longer needs to pass raw `dict[str, Any]` objects internally for normal ingest and retrieval work.

## `errors.py`

This file holds package-local exceptions such as:

- `RAGConfigurationError`
- `MetadataFilterError`
- `OllamaConnectionError`
- `OllamaRequestError`

The goal is to keep third-party failures translated into RAG-specific failure types instead of leaking arbitrary library exceptions through the whole stack.

## `ocr_processor.py`

This module is the OCR-to-RAG normalization boundary.

Its job is not OCR itself. Instead, it takes OCR output or OCR-like payloads and converts them into `IngestDocument`.

Key behaviors:

- requires non-empty text
- merges nested `metadata` with top-level extra fields
- derives `source`, `format_type`, and `file_path`
- builds a deterministic `source_id`
- extracts scalar-only `filter_metadata` for queryable retrieval filters

This is the main contract alignment point between the OCR package and the RAG package.

## `text_chunker.py`

This module owns text splitting.

Key behaviors:

- uses the shared `Embedder` instead of loading its own model
- honors configured `chunk_overlap`
- honors configured `separators`
- emits deterministic chunk ids
- preserves document-level filter metadata on every chunk

The chunker still uses semantic similarity between adjacent units, but it now does that through the same embedding backend used by the rest of the pipeline rather than maintaining a duplicate model runtime.

## `embedder.py`

This module is the only owner of the sentence-transformers model.

Key behaviors:

- lazy model construction
- blank-input zero-vector handling
- batch embedding
- embedding attachment for chunk records
- actual embedding-dimension validation against config

This fixes the old duplicate model-loading problem. Importing the module is light, and the model is only instantiated when the first real embedding request happens.

## `vector_store.py`

This module owns retrieval persistence and hybrid search.

It manages:

- the Chroma collection
- the JSON lexical manifest
- BM25 rebuild from that manifest
- deterministic upsert behavior
- hybrid score merging
- scalar metadata filtering
- reset and collection statistics

Important retrieval rules:

- only cosine distance is supported in this pass
- repeated ingest of the same chunk id updates existing state instead of creating duplicates
- scalar metadata is preserved in native types for filtering
- non-scalar metadata is serialized into the stored raw metadata payload

The dense store and lexical store now have a clearer relationship:

- Chroma is the source of dense retrieval
- the lexical manifest is the canonical persisted source for BM25 rebuild
- reset clears both layers together
- `lexical_index.json` is the only supported lexical persistence format

## `llm_generator.py`

This module is now a library-safe Ollama client.

It provides:

- `health_check()`
- `list_models()`
- `generate()`
- `stream_generate()`
- `chat()`
- `stream_chat()`
- `generate_with_results()`

Important cleanup changes:

- no `logging.basicConfig(...)`
- no `print(...)`
- no `__main__` demo code
- no connection test during construction
- typed exceptions instead of `"Error: ..."` strings

## `serializers.py`

This file converts typed RAG dataclasses into shared API/UI payload shapes.

It keeps the response contract consistent across:

- FastAPI responses
- the local orchestrator
- Streamlit display logic

## `rag_pipeline.py`

This is the main facade and coordinator for the RAG subsystem.

It is responsible for:

- loading config
- constructing the shared RAG runtime
- ingesting documents
- querying the vector store
- generating answers
- exposing stats
- resetting retrieval state
- optional MLflow logging when a run is already active

Important behavior changes:

- answer generation reuses one retrieval result set for answer, sources, and context
- MLflow setup no longer creates a noisy initialization run
- heavy runtime work is deferred until the first real embedding, vector-store, or Ollama action

## `__init__.py`

The package root exports the public RAG surface lazily:

- `OCRProcessor`
- `TextChunker`
- `Embedder`
- `VectorStore`
- `OllamaGenerator`
- `RAGPipeline`
- `IngestDocument`
- `RetrievalHit`
- `AnswerResult`

That keeps imports lighter and makes the package surface complete.

## Typed contracts

## OCR-to-RAG ingest contract

The expected normalized ingest payload is:

```python
{
    "text": "...",
    "source": "lesson.pdf",
    "format_type": "pdf",
    "file_path": "/abs/or/temp/path/lesson.pdf",
    "metadata": {
        "page": 3,
        "file_hash": "...",
        ...
    },
}
```

The RAG package converts that into `IngestDocument`.

Rules:

- `text` must be non-empty
- `metadata` can contain any JSON-like values
- only top-level scalar values become `filter_metadata`
- `source_id` is deterministic and derived from `file_hash`, path, source, format, or text fallback

## Chunk contract

`TextChunker` emits `ChunkRecord`.

Important fields:

- `id`
- `source_id`
- `text`
- `chunk_index`
- `total_chunks`
- `metadata`
- `filter_metadata`
- `embedding`

Chunk ids are deterministic, so repeated ingest of the same source no longer creates duplicate chunk rows by default.

## Retrieval contract

`VectorStore.query_by_text(...)` and `RAGPipeline.query(...)` return `RetrievalHit`.

Important fields:

- `id`
- `document`
- `metadata`
- `score`

Derived convenience properties:

- `source`
- `page`

Score contract:

- one normalized `score`
- range is intended to be `0.0` to `1.0`
- used consistently across package code, service payloads, and Streamlit UI

## Metadata filters

`filter_metadata` is now real, but intentionally narrow in scope.

Supported:

- top-level scalar equality filters only
- strings
- integers
- floats
- booleans

Not supported in this pass:

- nested filter objects
- lists
- range queries
- partial matches
- compound query operators

Unsupported filter shapes raise `MetadataFilterError` clearly instead of being silently ignored.

## Runtime boundaries

## Local orchestrator

`src/edumind/pipeline/orchestrator.py` is the main local integration boundary between OCR and RAG.

It now:

- keeps OCR extraction as the upstream source
- converts OCR output into the new nested ingest contract
- receives typed RAG results
- serializes those results for UI consumption

## RAG FastAPI service

`services/rag_service.py` is now import-safe.

Important startup behavior:

- importing the module does not instantiate `RAGPipeline`
- `/health` does not require Ollama or model loading
- the heavy pipeline is created lazily through a cached factory on first real ingest, query, stats, or reset use

That makes service import and startup much more predictable.

## Streamlit boundary

`apps/streamlit_app.py` is now the primary supported UI.

It is intentionally thinner than before:

- no duplicate retrieval logic
- no parsing sources back out of a context string
- no separate legacy answer-shape assumptions
- no eager pipeline construction at import time

The app initializes the orchestrator only when the user explicitly starts the workspace from the sidebar.

## Self-test flow

Recommended local validation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev,ocr,rag,ui,api]
ruff check src/edumind/rag src/edumind/pipeline services/rag_service.py apps/streamlit_app.py
mypy src/edumind/rag
pytest -q
```

Primary manual smoke tests:

1. Run `edumind-ui`
2. Initialize the workspace
3. Upload at least one OCR-supported file
4. Confirm the file is ingested and chunk counts increase
5. Ask a grounded question and verify sources appear with normalized scores
6. Run `edumind-rag-api` and call `/health` before any ingest to confirm lightweight startup

## Final assessment

The RAG package is now in a much stronger state than the earlier review version.

Main improvements:

- one shared embedding runtime
- deterministic chunk ids and upsert semantics
- real scalar metadata filtering
- typed internal contracts
- cleaned Ollama client
- lazy service startup
- one primary UI path

There is still room for future work, especially around retrieval metrics, richer filtering, and deeper benchmark coverage, but the package is now substantially cleaner, safer, and easier to extend.
