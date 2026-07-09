# OCR Package Guide

The OCR subsystem lives in `src/edumind/ocr` and is responsible for extracting text and metadata from supported file types.

## Main modules

```text
src/edumind/ocr/
|-- core/
|   |-- base_extractor.py
|   |-- format_detector.py
|   `-- pipeline.py
|-- extractors/
|   |-- pdf_extractor.py
|   |-- docx_extractor.py
|   |-- ocr_extractor.py
|   |-- audio_extractor.py
|   |-- video_extractor.py
|   `-- web_extractor.py
|-- processors/
|   |-- text_cleaner.py
|   `-- math_extractor.py
`-- utils/
```

## Supported inputs

- PDF
- DOCX
- images such as PNG and JPG
- HTML and article-like web content
- audio through Whisper
- video through FFmpeg plus Whisper

## Execution model

`DataIngestionPipeline` is the main entrypoint:

```python
from edumind.ocr.core.pipeline import DataIngestionPipeline

pipeline = DataIngestionPipeline()
result = pipeline.process_file("sample.pdf")

print(result.success)
print(result.format_type)
print(result.metadata)
```

## Output contract

The OCR layer returns an `ExtractionResult` object. For downstream RAG ingestion, call `to_dict()`:

```python
payload = result.to_dict()
```

That normalized payload always includes:

- `text`
- `success`
- `source` when a file path is available
- `format_type` when it is not already part of metadata

Format-specific metadata such as `num_pages`, `confidence`, or `author` is flattened into the same payload.

## Dependency behavior

- Tesseract is used for image OCR and acts as a reliable fallback
- PaddleOCR is preferred when installed and enabled
- audio and video extractors are lazy-loaded because they are heavy
- FFmpeg and Whisper are required for media workflows

## Where OCR fits in the system

- direct UI mode calls OCR through `OCRRAGOrchestrator`
- service mode exposes OCR through `services/ocr_service.py`
- experiments usually consume OCR outputs or curated evaluation inputs rather than running full extraction every time
