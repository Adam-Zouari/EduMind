# Apps Folder Guide

## Purpose

`apps/` contains the human-facing runtime entrypoints for the project.

After the cleanup, this folder is intentionally minimal:

- one maintained Streamlit app
- no parallel UI products
- no app-side business logic that belongs in OCR, RAG, or pipeline

## Files

### `streamlit_app.py`

This is the only maintained Streamlit product entrypoint.

Its responsibilities are:

- initialize the local `OCRRAGOrchestrator` on demand
- accept uploads and send them through the batch orchestration path
- render OCR results, ingest warnings, and processed-file history
- execute query flow with answer-generation fallback to retrieval-only mode
- keep lightweight typed session state for processed files and chat history

Important design choices:

- the app does not mutate backend payloads directly
- UI-only fields such as `filename` and `timestamp` are added in normalized UI records
- the app is a consumer of the orchestration layer, not a second orchestration layer

### `__init__.py`

This file stays minimal and only marks `apps/` as a package for imports and tests.

## What changed in this cleanup

- removed legacy `rag_standalone` and `streamlit_microservices` entrypoints
- removed matching CLI targets
- switched upload processing to `orchestrator.process_batch(...)`
- added typed UI records for processed files, sources, query display, and chat history
- added retrieval-only fallback when answer generation fails
- sourced upload types from orchestrator stats instead of duplicating a hardcoded list

## Current assessment

The folder is now in a good state:

- one real app
- clearer typing
- less duplicated logic
- less drift between UI and backend contracts
- better testability

If future complexity grows here again, it should be treated as a signal to improve the package layers underneath rather than grow more app entrypoints.
