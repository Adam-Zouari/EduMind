# Technical Documentation

## System overview

EduMind-AI is an OCR-to-RAG system with a package-first architecture.

The current design has four active layers:

1. `src/edumind/ocr` for extraction and normalization
2. `src/edumind/rag` for chunking, embeddings, retrieval, and generation
3. `src/edumind/pipeline` for orchestration between OCR and RAG
4. thin runtime entrypoints in `apps/` and `services/`

The direct Streamlit application is the default product path. Local FastAPI services are optional and mostly useful for API testing or deployment experiments.

## Package layout

```text
src/edumind/
|-- common/
|   |-- config.py      YAML config loader
|   |-- paths.py       project and artifact path helpers
|   `-- schemas.py     shared request and health models
|-- ocr/
|   |-- core/          pipeline, format detection, base result model
|   |-- extractors/    PDF, DOCX, image, audio, video, and web extractors
|   |-- processors/    cleaning and math-aware post-processing
|   `-- utils/         logging and file helpers
|-- pipeline/
|   |-- orchestrator.py      direct package orchestration
|   `-- orchestrator_api.py  API-backed orchestration adapter
`-- rag/
    |-- rag_pipeline.py  end-to-end retrieval pipeline
    |-- vector_store.py  ChromaDB adapter
    |-- embedder.py      embedding model adapter
    |-- llm_generator.py Ollama answer generation
    `-- text_chunker.py  chunking strategy
```

## End-to-end flow

### Ingestion flow

1. a file enters through Streamlit, FastAPI, or a script
2. `DataIngestionPipeline` validates the file and selects an extractor
3. extractors return an `ExtractionResult`
4. OCR post-processing cleans text and preserves math where possible
5. `ExtractionResult.to_dict()` normalizes the payload for RAG ingestion
6. `RAGPipeline` chunks text, creates embeddings, and stores vectors in ChromaDB

### Query flow

1. the user submits a question
2. `RAGPipeline` embeds the query
3. the vector store returns the closest chunks
4. the pipeline optionally filters by score threshold
5. if LLM mode is enabled, Ollama generates an answer from retrieved context
6. the response includes source metadata for the retrieved chunks

## Runtime entrypoints

### Apps

- `apps/streamlit_app.py` - direct package UI
- `apps/streamlit_microservices.py` - Streamlit UI that talks to local APIs
- `apps/rag_standalone.py` - RAG-only interface

### Services

- `services/ocr_service.py` - OCR FastAPI app
- `services/rag_service.py` - RAG FastAPI app

### CLI

`src/edumind/cli.py` exposes the supported runtime targets:

- `ui`
- `ui-microservices`
- `rag-ui`
- `ocr-api`
- `rag-api`
- `experiments`

## Configuration and state

### Configuration sources

- `config/base.yaml` holds shared runtime defaults
- `.env` can override external tool locations and MLflow settings
- `pyproject.toml` is the dependency and tool-config source of truth

### Persistent local state

Generated state is intentionally routed outside the source tree:

- vector store: `artifacts/rag/vector_store/`
- experiment MLflow database and artifacts: `artifacts/mlflow/`
- uploads and temporary outputs: `artifacts/`

This keeps the Git repository clean and makes runtime data easy to reset.

## Shared contracts

### OCR normalized payload

`ExtractionResult.to_dict()` produces the ingestion shape used across OCR and RAG:

```json
{
  "text": "Extracted content",
  "source": "document.pdf",
  "format_type": "pdf",
  "success": true
}
```

Format-specific metadata such as `num_pages`, `confidence`, or `author` is flattened onto the same payload.

### Service request models

`src/edumind/common/schemas.py` defines the core API models:

- `IngestRequest`
- `QueryRequest`
- `OCRExtractResponse`
- `ServiceHealth`

These schemas keep the Streamlit and FastAPI layers aligned around shared contracts.

## Testing and CI

The current quality loop is intentionally lightweight and reproducible:

- unit tests under `tests/unit/`
- integration smoke tests under `tests/integration/`
- experiment-focused tests under `tests/experiments/`
- GitHub Actions runs `ruff`, `mypy`, and `pytest`

Heavy local dependencies such as Ollama, Tesseract, Whisper, and PaddleOCR are not required for every CI run.

## Production posture

This codebase is structured well enough for portfolio presentation, team onboarding, and maintainable local development. It is not yet a full production deployment blueprint.

For a production rollout, the next gaps are mostly operational:

- deployment manifests and infrastructure
- secrets handling
- auth around APIs
- deeper observability
- stronger contract and load testing
