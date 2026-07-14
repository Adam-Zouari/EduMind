# Extraction subsystem

`edumind.extraction` is the only production boundary for turning local source files into normalized, provenance-bearing text. The runtime and benchmark runners use the same extractor registry and contracts.

## Flow

```text
path or stream -> source detection -> extraction router -> lazy extractor
               -> normalization -> ExtractedDocument -> revisioned cache
```

The supported `SourceKind` values are image, PDF, DOCX, audio, and video. Web pages, structured tables, formulas, and form parsing are intentionally outside the public contract. A PDF or DOCX extractor may retain flattened text from these elements, but callers must not infer structural fidelity.

## Contracts

- `ExtractionRequest` contains the source, checksum, MIME type, profile, and complete options.
- `ExtractionProfile` pins the engine/model revision, preprocessing, device, routing, and normalization mode.
- `ExtractedDocument` contains normalized text, ordered segments, source identity, warnings, and profile provenance.
- `ExtractedSegment` uses half-open character offsets and may include a page, timestamp, or bounding box.
- `Extractor` is the protocol implemented by every production adapter and exercised by benchmarks.

Optional libraries and models load only when their engine is selected. Preparation and preflight commands own downloads; importing this package must not download anything or mutate the filesystem. Temporary media files are always cleaned up. Cache identity includes source checksum, extractor/model revision, and all behavior-changing options.

`edumind benchmark prepare extraction-models` downloads PaddleOCR, docTR, OpenAI Whisper, and faster-whisper weights explicitly and records their immutable revisions and local paths. Optional extractors require those prepared paths and fail with installation guidance instead of downloading during a request.

## Runtime defaults and selection

Defaults are provisional until a standard benchmark and human-required downstream validation promote them. Extraction candidates are compared per modality; there is no universal winner. PDF selection is a routing policy and may choose native extraction for one page and OCR for another.

Run the smoke contract path with:

```powershell
edumind benchmark --profile smoke extraction all
```

The benchmark protocols, metrics, promotion rules, and limitations are documented under `experiments/benchmarks/extraction/`.
