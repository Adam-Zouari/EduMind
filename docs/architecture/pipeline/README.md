# Orchestration Layer

The orchestration layer lives in `src/edumind/pipeline` and coordinates OCR extraction with RAG ingestion and querying.

## Main class

`OCRRAGOrchestrator` is the primary package-level integration point.

```python
from edumind.pipeline.orchestrator import OCRRAGOrchestrator

orchestrator = OCRRAGOrchestrator(use_llm=True)

result = orchestrator.process_file("document.pdf", ingest_to_rag=True)
answer = orchestrator.query("What is this document about?")
```

## Responsibilities

- call the OCR pipeline
- normalize OCR output for RAG ingestion
- optionally ingest extracted text into the vector store
- expose query and answer-generation helpers
- provide simple stats for UI and service layers

## Output from `process_file`

The orchestrator returns a dictionary that includes:

- `ocr_success`
- `ocr_error`
- `text`
- `metadata`
- `file_path`
- `format_type`
- `extraction_time`
- `rag_ingested`
- `rag_chunks`

## Where it is used

- `apps/streamlit_app.py`
- `apps/rag_standalone.py`
- future scripts or notebooks that need an end-to-end local package API

## API-backed variant

`orchestrator_api.py` exists for service-based workflows and keeps the UI layer from needing to know service request details.
