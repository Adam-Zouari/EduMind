# Pipeline Package Guide

## Purpose

`src/edumind/pipeline` is the orchestration layer between OCR and RAG.

It does not implement extraction or retrieval itself. Its job is to:

- take a file path or a query from the app/runtime boundary
- call the OCR package when a document must be processed
- hand normalized OCR output to the RAG package
- expose one simpler interface for the UI or HTTP-facing code

This folder is small by design. The heavy logic should stay in:

- `edumind.ocr` for extraction
- `edumind.rag` for ingest, retrieval, and answer generation

## Files

### `__init__.py`

Role:

- exports the two orchestration entrypoints:
  - `OCRRAGOrchestrator`
  - `APIOrchestrator`

Assessment:

- good as-is
- minimal and clear

### `orchestrator.py`

Role:

- local in-process orchestrator
- used when Streamlit or another caller imports OCR and RAG directly in the same Python process
- owns the simplest product flow:
  1. run OCR on a file
  2. optionally ingest OCR output into RAG
  3. query RAG and optionally generate an answer

How it works:

- creates one `DataIngestionPipeline`
- creates one `RAGPipeline`
- `process_file(...)` delegates to OCR and converts the result into a stable JSON-friendly payload
- `process_batch(...)` uses the OCR package batch path first, then performs optional RAG ingestion per result
- `query(...)` calls the RAG pipeline and serializes either retrieval hits or an answer payload
- `get_stats()` returns combined OCR/RAG runtime information
- `reset_rag()` clears only the RAG index

What was wrong before:

- batch processing was not using the OCR batch path; it looped over `process_file(...)` instead
- result payloads were not fully stable because `rag_error` was only present on failure
- stats shape differed from the API orchestrator
- there was no backward-compatible `reset_database()` alias

What is better now:

- batch work now uses `DataIngestionPipeline.process_batch(...)`
- every processed-document payload always includes the same RAG keys
- stats now expose both `ocr_extractors` and `ocr_formats`
- local reset now has the same alias shape as the API orchestrator

Current quality:

- good
- responsibilities are clear
- the file stays orchestration-focused and does not duplicate OCR/RAG internals

Remaining limits:

- it still catches broad exceptions at the RAG-ingest boundary
- that is acceptable here because this class is intentionally the failure-isolation boundary between subsystems

### `orchestrator_api.py`

Role:

- HTTP orchestrator for the microservices mode
- used when OCR and RAG run as separate FastAPI services

How it works:

- stores OCR and RAG base URLs
- reuses one `requests.Session`
- optionally verifies service health during initialization
- uploads files to OCR `/extract`
- sends normalized nested ingest payloads to RAG `/ingest`
- sends user queries to RAG `/query`
- reads combined runtime stats from OCR `/formats` and RAG `/stats`

What was wrong before:

- service verification was eager and tightly coupled to construction
- raw `requests.get/post/delete` calls were repeated instead of using one request helper
- there was no `process_batch(...)`
- stats shape did not match the local orchestrator
- remote RAG ingest failure could make the whole pipeline feel less predictable
- ingest payload assembly was duplicated inline

What is better now:

- one session is reused across requests
- health verification is optional with `verify_on_init`
- `_request_json(...)` centralizes HTTP + JSON handling
- `process_batch(...)` now exists for parity with the local orchestrator
- remote RAG ingest failures are isolated into `rag_error` while preserving the OCR result
- nested ingest payload building is centralized in one helper
- stats now use the same key names as the local orchestrator

Current quality:

- good for a thin boundary layer
- much less brittle than before
- still intentionally simple because the service layer already owns the heavy work

Remaining limits:

- there is no true remote bulk endpoint, so `process_batch(...)` is still a sequential wrapper over `process_file(...)`
- OCR service failures still raise instead of returning a fake local payload, which is the correct behavior for a transport boundary

## Processed Document Payload

Both orchestrators now return the same logical shape from file-processing calls:

- `ocr_success`
- `ocr_error`
- `text`
- `metadata`
- `file_path`
- `format_type`
- `extraction_time`
- `rag_ingested`
- `rag_chunks`
- `rag_source_id`
- `rag_error`

Important detail:

- OCR success and RAG ingest success are tracked separately
- this means a file can succeed in OCR and still surface a non-fatal RAG ingest problem

## Why This Folder Should Stay Small

This package should coordinate, not absorb subsystem logic.

Bad direction:

- moving chunking logic here
- moving OCR extraction rules here
- moving retrieval score logic here
- building app-specific UI state here

Good direction:

- keep normalized payload assembly here
- keep failure isolation here
- keep local-vs-API runtime choice here

## Tests Added For This Pass

The pipeline layer now has direct unit coverage for:

- local orchestrator single-file processing
- local orchestrator batch processing
- local orchestrator stats/reset consistency
- API orchestrator stable payload handling
- API orchestrator remote-ingest failure isolation
- API orchestrator health/stats/reset behavior

## Final Assessment

The pipeline folder is now in a good state from a coding point of view:

- small
- readable
- consistent across local and API modes
- better tested
- not over-engineered

It is not the place for deeper product logic. If more complexity appears here later, that is usually a sign that the change belongs in OCR, RAG, or the app/service boundary instead.
