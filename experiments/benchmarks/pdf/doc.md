# PDF-to-text experiment

## Question, candidates, and controls

Which extractor and page-routing policy best handles digital, scanned, mixed, and broken PDFs locally? Compare pypdf (native control), pdfplumber, Docling, and page-level hybrid native plus the selected image OCR. Routing is separately tested in the routing experiment; the PDF winner may be a policy, not one engine. Marker is excluded from the standard matrix by target VRAM and licensing gates.

## Dataset, splits, and procedure

The licensed 60-document corpus is split 36/12/12 by document and balanced across digital, scanned, mixed, broken-encoding, slides, and academic layouts. Its manifest pins licenses, revisions, IDs, file/page checksums, page text/order/structure, preprocessing, and seed. Each production extractor processes the same seeded order with cold load, warmups, repetitions, temporary cleanup, and page-level observations.

## Metrics and rationale

CER/WER are edit distance divided by reference characters/words (0 upward, lower). Page Coverage is expected pages with recovered content/expected pages; Reading Order Accuracy is correct annotated pair order/all annotated pairs; Heading/List Structure F1 are matched structural units; Page Attribution Accuracy is text assigned to the correct page/all attributed text (0–1, higher). Missing/Duplicate Page Rate is affected pages/expected pages (0–1, lower). p50/p95 page/document latency, pages/minute, RAM/VRAM, and temporary disk are operational. No structured table/formula score exists.

## Statistics, promotion, and artifacts

Correctness, empty/missing-page, determinism, resource, and license gates precede per-document/page bootstrap intervals and Pareto selection. Interval ties use p95, memory, then storage. The run saves plan/provenance, all per-page results, aggregates/intervals, Pareto candidates, `_SUCCESS.json`, and optional MLflow parent/child runs.

## Commands, example, and limitations

```powershell
edumind benchmark --profile smoke extraction pdf
edumind benchmark --profile standard extraction pdf
edumind benchmark --profile full extraction pdf
```

Example: low CER on digital files cannot justify always-native when scanned-page coverage is zero. Results do not establish structural table/formula/form extraction, handwriting support, or quality outside the frozen English layouts.
