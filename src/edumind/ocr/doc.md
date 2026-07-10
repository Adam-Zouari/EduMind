# OCR Package Guide

This document explains the current OCR package under `src/edumind/ocr`, what each file is responsible for, how the live extraction path works, and where the package is now stable versus still intentionally lightweight.

## Current status

The OCR package is now package-first and self-contained enough to install with:

```bash
pip install -e .[dev,ocr]
```

Key stabilization changes completed:

- `mlflow` is optional inside image OCR, so OCR imports no longer fail when experiment dependencies are absent
- `ExtractionResult` now has a dedicated cache serialization contract separate from the RAG ingest payload
- Whisper runtime setup is centralized in `extractors/_media_runtime.py`
- video extraction no longer uses `tempfile.mktemp()` or `eval()`
- web fetching now honors `WEB_TIMEOUT` and `USER_AGENT`
- `layout_parser.py` was removed because it was empty
- the OCR test flow no longer depends on `tests/conftest.py` path injection
- scanned PDFs can now use page-level OCR fallback with cache reuse
- the pipeline now supports opt-in profiling, structured outputs, language overrides, and stricter format detection
- a repeatable benchmark runner now lives at `scripts/ocr_benchmark.py`
- OCR-focused unit and integration coverage now exercises the package directly

## Public package surface

These are the OCR entrypoints that should stay stable:

- `edumind.ocr`
- `edumind.ocr.DataIngestionPipeline`
- `edumind.ocr.ExtractionResult`
- `edumind.ocr.__version__`
- the current extractor class names under `edumind.ocr.extractors`
- `ExtractionResult.to_dict()` for downstream ingest payloads

## Package layout

```text
src/edumind/ocr/
|-- __init__.py
|-- config.py
|-- doc.md
|-- core/
|   |-- base_extractor.py
|   |-- format_detector.py
|   `-- pipeline.py
|-- extractors/
|   |-- _media_runtime.py
|   |-- audio_extractor.py
|   |-- docx_extractor.py
|   |-- ocr_extractor.py
|   |-- pdf_extractor.py
|   |-- video_extractor.py
|   `-- web_extractor.py
|-- processors/
|   |-- form_recognizer.py
|   |-- layout_analyzer.py
|   |-- math_extractor.py
|   `-- text_cleaner.py
`-- utils/
    |-- file_handler.py
    `-- logger.py
```

## Live execution flow

### 1. Import surface

`__init__.py` lazily exposes `DataIngestionPipeline` and `ExtractionResult`, so importing `edumind.ocr` does not immediately pull in the whole runtime stack.

The package also exposes `__version__`, which is safe to read without importing the heavy OCR runtime.

### 2. Pipeline entry

`core/pipeline.py` is the orchestrator. It:

- validates the file
- detects the format
- chooses the matching extractor
- runs text cleanup
- preserves and restores LaTeX-like math
- appends metadata such as format info, file size, and file hash
- supports opt-in flags for scanned-PDF OCR, layout/form metadata, profiling, and language overrides

### 3. Format-specific extraction

The live extractors are:

- `PDFExtractor`
- `DOCXExtractor`
- `OCRExtractor`
- `WebExtractor`
- `AudioExtractor`
- `VideoExtractor`

Audio and video stay lazy-loaded because Whisper is the heaviest runtime dependency in the package.

### 4. Result contract

Every extractor returns `ExtractionResult`.

There are now two intentionally different serialization paths:

- `to_dict()` for downstream ingest into RAG
- `to_cache_dict()` / `from_cache_dict()` for full OCR cache round-trips

That separation is important. The ingest payload is intentionally flattened and compact, while the cache payload preserves operational fields such as `metadata`, `file_path`, `extraction_time`, and `timestamp`.

### 5. OCR-to-RAG handoff contract

The OCR package is now the upstream producer for the redesigned RAG package.

When OCR output is ingested into RAG through the local orchestrator or the RAG service, the handoff is treated as a nested document payload:

```python
{
    "text": result.text,
    "source": "lesson.pdf",
    "format_type": result.format_type,
    "file_path": result.file_path,
    "metadata": result.metadata,
}
```

Important detail:

- `ExtractionResult.to_dict()` still stays stable for downstream compatibility
- the live OCR-to-RAG boundary no longer depends on flattening OCR metadata into the top level
- RAG now receives a clearer document contract and derives scalar filter metadata from the nested `metadata` payload

## File-by-file roles

## `config.py`

Owns OCR-specific runtime configuration and artifact locations.

Important live settings:

- `TEMP_DIR`
- `OCR_CACHE_DIR`
- `OCR_QUALITY_THRESHOLD`
- `WHISPER_MODEL`
- `WHISPER_DEVICE`
- `WEB_TIMEOUT`
- `USER_AGENT`

The file still creates local artifact directories at import time. That side effect is acceptable for this repo because all paths are routed into `artifacts/ocr/`.

## `core/base_extractor.py`

Defines:

- `ExtractionResult`
- `BaseExtractor`

This is the shared contract layer. It is simple, stable, and now safe for both ingest serialization and cache serialization.

## `core/format_detector.py`

Detects file type using:

- extension mapping first
- optional `python-magic`
- optional `tika`

The module now logs optional dependency fallback behavior through the package logger instead of `print()`.

If `python-magic` is not installed or `libmagic` is unavailable, the warning is expected and the package falls back to extension-based detection.

Known extensions now short-circuit MIME probing by default. MIME detection only runs for unknown extensions or when `strict_format_detection=True`.

## `core/pipeline.py`

Coordinates the OCR flow end to end.

Strengths:

- clean control plane
- lazy loading for Whisper-based extractors
- batch mode with preserved result ordering
- additive metadata for profiling, page-level PDF fallback, and optional structured OCR outputs

Limitations:

- batch concurrency is still thread-based, which is good enough for local use but not a full job-scheduling system
- layout analysis is heuristic and best-effort, not a guaranteed document-structure reconstruction

## `extractors/_media_runtime.py`

Internal helper shared by audio and video extraction.

Responsibilities:

- configure environment defaults for Whisper
- expose safe device resolution through `WHISPER_DEVICE`
- cache Whisper models by `(model_name, device)`
- add a custom FFmpeg directory to `PATH` when configured

This file exists to keep the audio and video extractors small and consistent.

## `extractors/ocr_extractor.py`

Main image OCR engine.

What it does:

- chooses PaddleOCR or Tesseract
- supports both file-based and in-memory image extraction
- scores image quality
- preprocesses images with rotation, perspective, denoising, and thresholding
- retries with an inverted image when confidence is low
- validates extraction quality
- caches successful OCR results
- supports custom cache keys for page-level PDF OCR reuse
- logs to MLflow only when the dependency is installed and an active run exists

Current optimization status:

- good enough for portfolio-grade local use
- significantly more robust than a plain `pytesseract` call
- still the most complex and highest-maintenance file in the package

Remaining tradeoffs:

- quality scoring is still heuristic
- Paddle result parsing is necessarily defensive because upstream result shapes vary
- OCR confidence gating can drop low-confidence but still useful tokens

## `extractors/pdf_extractor.py`

Uses PyMuPDF for native extraction first, then optionally runs page-level OCR fallback.

Current live behavior:

- `pdf_ocr_mode="off"` keeps native extraction only
- `pdf_ocr_mode="auto"` OCRs low-text pages
- `pdf_ocr_mode="force"` OCRs every page
- per-page metadata is stored under `metadata["pages"]`
- rendered page OCR reuses the shared OCR cache through a PDF-page-specific cache key

This is now a real scanned-PDF support path, although it is still local-first and heuristic rather than a fully specialized document OCR engine.

## `extractors/docx_extractor.py`

Extracts paragraphs, tables, and core document metadata from DOCX files.

It is intentionally simple and reliable. It does not preserve advanced Word layout semantics.

## `extractors/web_extractor.py`

Handles:

- local HTML files
- remote URLs

The remote path now uses `requests` with the configured timeout and user-agent before passing HTML into Trafilatura and then `newspaper3k`.

## `extractors/audio_extractor.py`

Runs Whisper speech-to-text for audio files.

The extractor now:

- uses shared runtime setup
- honors `WHISPER_DEVICE`
- keeps failure behavior explicit when Whisper is unavailable

## `extractors/video_extractor.py`

Extracts audio with FFmpeg and transcribes it with Whisper.

Safety improvements now in place:

- safe temp-file creation
- guaranteed temp cleanup in `finally`
- explicit frame-rate parsing without `eval()`

## `processors/text_cleaner.py`

Applies lightweight cleanup after extraction:

- fix encoding glitches with `ftfy`
- normalize whitespace
- remove common header/footer lines
- apply regex-based OCR correction heuristics

This remains heuristic by design. It is fast and useful for study material, but not a language-aware normalization engine.

## `processors/math_extractor.py`

Preserves math expressions during cleanup and restores them afterward.

This is regex-based and intentionally lightweight. It is good for common inline and display math, but not a full LaTeX parser.

The inline extractor now avoids double-counting `$$...$$` display expressions as inline math.

## `processors/layout_analyzer.py`

Experimental utility for reconstructing layout structure from OCR token boxes.

It is now available through the live pipeline only when `include_layout=True`. The output is attached as metadata and does not rewrite the main extracted text.

## `processors/form_recognizer.py`

Experimental regex-based structured field extraction for form-like documents.

It is now available through the live pipeline only when `include_form_fields=True`. The current implementation is useful for prototypes and demos, not for high-precision document intelligence.

When a generic `Label: Value` match overlaps with a more specific typed field such as `email`, the structured dict now keeps the higher-confidence result instead of overwriting it with the generic one.

## `utils/file_handler.py`

Provides:

- file validation
- file size lookup
- file hashing
- directory creation
- temp directory cleanup

Low complexity and stable.

## `utils/logger.py`

Sets up OCR logging with Loguru and writes logs into `artifacts/ocr/logs/`.

The package still configures logging at import time, which is convenient for this repo but would usually be pushed up to the app boundary in a larger production system.

## What is optimized well

- package-level lazy exports
- lazy loading of Whisper-based extractors
- shared Whisper model caching
- thread-safe shared Paddle initialization
- scanned-PDF OCR fallback with page-level cache reuse
- separate ingest and cache serialization contracts
- extension short-circuiting in format detection
- auto batch routing that avoids parallel audio/video processing by default
- optional profiling and hash skipping
- artifact routing outside source folders
- controlled web fetch behavior

## What is intentionally still lightweight

- form extraction heuristics
- layout preservation heuristics
- math parsing depth
- batch parallelism strategy
- scanned PDF page rendering and OCR heuristics

## Automated coverage now in place

The OCR-focused test suite now covers:

- `ExtractionResult` ingest and cache contracts
- format detection fallback behavior
- text cleaning heuristics
- math preservation and restoration
- layout and form utilities
- OCR cache hit and miss behavior
- page-level scanned PDF OCR fallback and cache reuse
- `strict_format_detection` short-circuit behavior
- `include_file_hash=False` hash skipping
- `profile=True` performance metadata
- `batch_strategy="auto"` routing rules
- language propagation through the pipeline
- optional `mlflow` handling
- audio/video missing-dependency paths
- video temp cleanup behavior
- integration flow for generated PDF, DOCX, and image fixtures
- scanned PDF repeated-run cache behavior
- packaging smoke import from outside the repo root
- OCR service import and health smoke behavior

The official test path is now an editable install of the package itself. The suite is no longer designed around injecting `src/` into `sys.path`.

## Official self-test flow

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev,ocr,api]
ruff check .
mypy src
pytest -q
python -c "from edumind.ocr import DataIngestionPipeline; print(DataIngestionPipeline)"
```

Windows activation:

```powershell
.\.venv\Scripts\activate
```

Then run a few manual checks:

- process a small PDF
- process a scanned PDF with `pdf_ocr_mode="off"` and again with `pdf_ocr_mode="force"`
- process a small DOCX
- process the same image twice and confirm the second run uses `artifacts/ocr/cache/`
- from outside the repo root, rerun `python -c "from edumind.ocr import DataIngestionPipeline; print(DataIngestionPipeline)"` to confirm package imports do not rely on local path hacks
- optionally test short audio and video files if Whisper and FFmpeg are available locally

For performance measurements:

```bash
python scripts/ocr_benchmark.py
```

This writes corpus files plus benchmark JSON and CSV outputs into `artifacts/ocr/benchmarks/`.

## Final assessment

The OCR package is now in a much better state for a production-style repository:

- installable as a real package
- less coupled to experiments
- safer in its media paths
- clearer in its serialization contracts
- materially better covered by tests
- more capable for scanned PDFs and structured OCR workflows
- measurable through a local benchmark runner
- documented in a way that matches the actual package instead of the pre-cleanup tree

It is still a local-first OCR subsystem rather than a fully hardened enterprise document-processing platform, but it is now clean, reproducible, and credible as a standalone package.

## How the OCR package works

The OCR package is a unified extraction pipeline, not only an image-to-text utility. It handles `pdf`, `docx`, `image`, `web`, `audio`, and `video` inputs through one shared entrypoint: `DataIngestionPipeline` in `src/edumind/ocr/core/pipeline.py`.

### Main flow

The usual entrypoint is `DataIngestionPipeline.process_file(...)`.

At a high level, the flow is:

1. validate the input file
2. detect the file format
3. choose the matching extractor
4. run extraction and normalize the output into `ExtractionResult`
5. optionally clean text, preserve math, and attach extra metadata
6. return the normalized result or flatten it with `ExtractionResult.to_dict()` for downstream ingest

`ExtractionResult` is the package-wide output contract. It always carries:

- extracted `text`
- `metadata`
- `format_type`
- `file_path`
- timing and success/error fields

`ExtractionResult.to_dict()` keeps the downstream ingest payload simple by flattening metadata into one dictionary with the extracted text and success flag.

### Format detection

Format detection is handled by `FormatDetector`.

Default behavior:

- if the file extension clearly maps to a supported format, use that immediately
- skip optional MIME detection work for speed

Strict behavior:

- if `strict_format_detection=True`, also try optional MIME detectors such as `python-magic` and `tika`

This gives the package a fast normal path and a stricter path when an extension may be misleading.

### Pipeline orchestration

Once a format is known, `DataIngestionPipeline` routes the file to the correct extractor:

- `PDFExtractor` for PDFs
- `DOCXExtractor` for Word files
- `OCRExtractor` for images
- `WebExtractor` for HTML or fetched web content
- `AudioExtractor` for audio transcription
- `VideoExtractor` for video transcription through extracted audio

The pipeline then performs shared post-processing:

- text cleaning through `TextCleaner`
- optional math preservation and restoration through `MathExtractor`
- optional layout metadata through `LayoutAnalyzer`
- optional structured form metadata through `FormRecognizer`
- file size and optional file hash attachment
- optional performance timing metadata

### Image OCR path

Image OCR is implemented in `OCRExtractor`.

The extractor does the following:

1. load the image with OpenCV
2. check whether a cached OCR result already exists
3. assess image quality
4. preprocess the image
5. run OCR with PaddleOCR if available, otherwise Tesseract
6. retry once with an inverted image if confidence is too low
7. validate the extracted text
8. cache successful results for future runs

Preprocessing can include:

- rotation correction
- perspective correction
- grayscale conversion
- denoising
- thresholding
- light morphology for weak images

The extractor also tracks metadata such as:

- OCR engine used
- confidence
- languages
- quality score
- preprocessing steps
- retry attempts
- validation outcome
- cache hit/miss state

When layout metadata is requested, the extractor can also return token-level OCR box data that is later consumed by the layout analyzer.

### PDF path

PDF extraction is handled by `PDFExtractor` and follows a two-stage strategy:

1. extract native text with PyMuPDF
2. decide whether page-level OCR fallback is needed

The `pdf_ocr_mode` flag controls this:

- `off`: never OCR pages
- `auto`: OCR only low-text pages
- `force`: OCR every page

In `auto` mode, OCR fallback is triggered when:

- a page has fewer than 40 alphanumeric characters
- or the whole document has fewer than 150 alphanumeric characters

If fallback is needed, the page is rendered to an image and passed to `OCRExtractor.extract_image(...)`. The final PDF text is rebuilt in the original page order.

Per-page metadata is stored under `metadata["pages"]`, including:

- page index
- whether the page came from native extraction or OCR
- OCR confidence when available
- extraction time
- fallback reason
- cache information for OCR-rendered pages

This makes PDFs much smarter than a one-size-fits-all OCR pass. Native digital PDFs stay fast and clean, while scanned PDFs can still be recovered.

### DOCX and web path

`DOCXExtractor` uses `python-docx` to extract:

- paragraph text
- table text
- document core metadata

`WebExtractor` works with either local HTML files or fetched remote pages. It:

- fetches remote pages with the configured timeout and user-agent
- tries `trafilatura` first
- falls back to `newspaper3k` if the first extraction is weak

### Audio and video path

`AudioExtractor` and `VideoExtractor` are built around Whisper.

Audio flow:

- load Whisper lazily
- transcribe the audio file
- return text plus segment metadata

Video flow:

- extract audio from the video using FFmpeg
- transcribe that audio with Whisper
- collect lightweight video metadata such as size, codec, FPS, and duration
- clean up temporary files safely afterward

Shared runtime setup for audio and video lives in `_media_runtime.py`, which centralizes:

- FFmpeg path setup
- Whisper import handling
- device selection
- Whisper model caching

### Text cleanup and math preservation

After extraction, the pipeline can refine the text before returning it.

`TextCleaner` is responsible for:

- fixing encoding artifacts with `ftfy`
- removing likely headers and footers
- normalizing whitespace
- applying lightweight OCR error corrections

`MathExtractor` protects LaTeX-style expressions during cleanup so math content is not accidentally damaged. The flow is:

1. replace math spans with placeholders
2. clean the surrounding text
3. restore the original math
4. store extracted math expressions in metadata

This is especially useful for study material, lecture notes, and technical PDFs.

### Optional structured metadata

The pipeline can add extra metadata without rewriting the final text.

`LayoutAnalyzer`:

- reads OCR token box data
- classifies simple block types such as title, paragraph, list, or caption
- reconstructs a lightweight reading order

`FormRecognizer`:

- looks for likely structured fields such as name, email, phone, date, ID, amount, and checkbox values
- also detects generic `Label: Value` pairs

These outputs are attached to metadata only:

- `metadata["layout_blocks"]`
- `metadata["structured_fields"]`

The default `result.text` stays backward compatible.

### Caching

Caching is one of the main performance improvements in the refined package.

Image OCR caching:

- keyed by stable file identity information
- stored as JSON under `artifacts/ocr/cache`

Scanned PDF page caching:

- keyed by document identity, page index, OCR engine choice, languages, and OCR threshold inputs
- reused on repeated OCR fallback runs

The cache uses `ExtractionResult.to_cache_dict()` and `ExtractionResult.from_cache_dict()` so cache round-tripping preserves the full OCR result structure instead of only the ingest payload.

### Batch processing and performance behavior

`process_batch(...)` supports three strategies:

- `sequential`
- `threads`
- `auto`

In `auto` mode:

- `pdf`, `docx`, `image`, and `web` can run concurrently
- `audio` and `video` stay sequential
- image OCR concurrency is limited on CPU to reduce oversubscription
- result ordering is preserved

If `profile=True`, the pipeline records timing metadata for:

- format detection
- extraction
- cleaning
- hashing
- total processing

### Configuration

The main OCR settings live in `config.py`. They define:

- artifact, temp, log, output, and cache directories
- OCR engine settings
- GPU and angle-classifier settings
- Tesseract command
- OCR quality and confidence thresholds
- Whisper model and device
- FFmpeg path
- web timeout and user-agent

### In one sentence

The OCR package validates a file, detects its format, routes it to a specialized extractor, normalizes the output into `ExtractionResult`, optionally cleans and enriches the content, and caches expensive OCR work so repeated local runs are faster and more predictable.
