# OCR to RAG Integration Guide

The OCR layer and the RAG layer are designed to connect through a shared normalized payload.

## Recommended integration: orchestrator

For most product code, use the orchestration layer instead of wiring OCR and RAG manually.

```python
from edumind.pipeline.orchestrator import OCRRAGOrchestrator

orchestrator = OCRRAGOrchestrator(use_llm=True)
result = orchestrator.process_file("document.pdf", ingest_to_rag=True)

print(result["ocr_success"])
print(result["rag_chunks"])
```

## Manual integration

If you need lower-level control, bridge the layers with `ExtractionResult.to_dict()`:

```python
from edumind.ocr.core.pipeline import DataIngestionPipeline
from edumind.rag.rag_pipeline import RAGPipeline

ocr = DataIngestionPipeline()
rag = RAGPipeline(use_llm=True)

ocr_result = ocr.process_file("document.pdf")
chunks = rag.ingest_document(ocr_result.to_dict())
print(chunks)
```

## JSON fixture flow

You can also ingest from saved JSON fixtures:

```python
from edumind.rag.rag_pipeline import RAGPipeline

rag = RAGPipeline(use_llm=False)
total_chunks = rag.ingest_from_json("ocr_output.json")
print(total_chunks)
```

## Important contract details

- OCR payloads must contain `text`
- `source` is used for attribution
- format-specific metadata is kept alongside the text payload
- the RAG layer does not need the original OCR object, only the normalized dictionary
