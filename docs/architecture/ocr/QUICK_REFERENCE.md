# OCR Quick Reference

## Basic usage

```python
from edumind.ocr.core.pipeline import DataIngestionPipeline

pipeline = DataIngestionPipeline()
result = pipeline.process_file("document.pdf")
payload = result.to_dict()
```

## Batch usage

```python
results = pipeline.process_batch(["a.pdf", "b.docx"], parallel=True)
```

## Core result fields

- `result.text`
- `result.metadata`
- `result.format_type`
- `result.file_path`
- `result.extraction_time`
- `result.success`
- `result.error`

## Normalized payload fields

`result.to_dict()` produces the RAG-friendly shape:

- `text`
- `source`
- `format_type`
- `success`
- flattened metadata fields

## Common metadata examples

- PDF: `num_pages`, `title`, `author`, `extractor`
- DOCX: `num_paragraphs`, `num_tables`, `author`
- image: `confidence`, `ocr_engine`, `languages`
- audio or video: `language`, `duration`, `num_segments`

## Related files

- pipeline: `src/edumind/ocr/core/pipeline.py`
- base result model: `src/edumind/ocr/core/base_extractor.py`
- OCR image extractor: `src/edumind/ocr/extractors/ocr_extractor.py`
