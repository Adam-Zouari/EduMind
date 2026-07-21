# Extraction subsystem

`edumind.extraction` is the only production boundary for turning local source files into normalized, provenance-bearing text. The runtime and benchmark runners use the same extractor registry and contracts.

## Flow

```text
local path -> source detection -> extraction router -> lazy extractor
           -> normalization -> ExtractedDocument -> revisioned cache
```

The supported `SourceKind` values are image, PDF, DOCX, audio, and video. Tables and formulas are typed elements inside image/PDF/DOCX results rather than separate source kinds or services. Web pages and dedicated form parsing remain outside the public contract.

## Contracts

- `ExtractionRequest` contains the source, checksum, MIME type, profile, and complete options.
- `ExtractionProfile` pins the engine/model revision, preprocessing, device, routing, and normalization mode.
- `ExtractedDocument` contains normalized text, ordered segments, source identity, warnings, and profile provenance.
- `ExtractedSegment` uses half-open character offsets, has a semantic kind, may include a page/timestamp/bounding box, and may carry table rows or normalized LaTeX.
- `Extractor` is the protocol implemented by every production adapter and exercised by benchmarks.

Optional libraries and models load only when their engine is selected. Explicit preparation commands own downloads; importing this package must not download anything or mutate the filesystem. Temporary media files are always cleaned up. Cache identity includes source checksum, extractor/model revision, and all behavior-changing options.

`python experiments/benchmarks/prepare.py extraction-models --candidate NAME` downloads one selected OCR, complete-document, or ASR model and records its revision and local path immediately. The option can be repeated, partial Hugging Face downloads resume, and optional extractors fail with installation guidance instead of downloading during a request.

## Runtime defaults and selection

Defaults are provisional until a standard benchmark and human-required downstream validation promote them. Extraction candidates are compared per modality; there is no universal winner. PDF selection is a routing policy and may choose native extraction for one page and OCR for another.

Run the smoke contract path with:

```powershell
python experiments/benchmarks/extraction/image/run.py --profile smoke
```

The benchmark protocols, metrics, promotion rules, and limitations are documented under `experiments/benchmarks/extraction/`.
