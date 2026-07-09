# OCR Output Format

This file documents the current normalized OCR payload used by the RAG layer.

## Current shape

`ExtractionResult.to_dict()` emits a flat, ingestion-friendly document:

```json
{
  "text": "Extracted content...",
  "source": "document.pdf",
  "format_type": "pdf",
  "num_pages": 5,
  "author": "Jane Doe",
  "success": true
}
```

## Rules

The serializer follows these rules:

1. always include `text`
2. flatten metadata onto the top level
3. include `format_type` if metadata does not already provide it
4. map `file_path` to `source` using the filename
5. include `success`

## Why this shape works well

- easy to ingest into the RAG layer
- easy to store as JSON fixtures
- easy to use through APIs
- easy to filter on metadata without deep nesting

## Legacy note

Older repository notes may still describe nested OCR metadata from the pre-package layout. The active code path is the flat structure above, implemented in `src/edumind/ocr/core/base_extractor.py`.
