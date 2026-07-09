# OCR Improvements

This note summarizes the current OCR subsystem shape and the most valuable next improvements.

## Current strengths

- multi-format ingestion through one `DataIngestionPipeline`
- PDF, DOCX, image, audio, video, and web support
- normalized `ExtractionResult` output for downstream RAG ingestion
- text cleaning with optional math preservation
- file validation and metadata enrichment before ingestion
- lazy loading for the heavier audio and video extractors

## Current package layout

```text
src/edumind/ocr/
|-- core/
|-- extractors/
|-- processors/
`-- utils/
```

## Immediate improvement priorities

### 1. Reduce eager optional imports

Some extractor modules still import heavy dependencies at module load time. Converting more of those imports to lazy or guarded imports would make partial installs more reliable.

### 2. Add typed OCR DTOs

The OCR layer already has a strong normalized result object. Adding a small typed DTO layer for downstream consumers would make API contracts clearer and simplify validation.

### 3. Expand fixture coverage

The next useful tests are small fixture-based checks for:

- PDF metadata extraction
- DOCX table extraction
- image OCR fallback behavior
- audio and video lazy-load failure paths

### 4. Improve observability

The OCR pipeline would benefit from structured timing metrics per extractor so local bottlenecks are easier to measure and compare.

### 5. Separate extractor capability reporting

Today runtime capability depends on local tools being installed. A capability-reporting helper would let apps and APIs present clearer messages when Tesseract, FFmpeg, or PaddleOCR are unavailable.

## Practical outcome

The OCR layer is already suitable for a strong portfolio demo. The next improvements are about predictability, diagnostics, and install flexibility rather than a fundamental redesign.
