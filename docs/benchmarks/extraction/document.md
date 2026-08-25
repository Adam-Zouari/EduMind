# Document extraction benchmark

[Benchmark overview](../overview.md) · [Benchmark manual](../methodology.md) ·
[Candidate selection](../model-selection.md) · [Preparation guide](../../setup/installation.md)

This experiment evaluates complete image, PDF, and DOCX parsing. It does not rank
standalone OCR engines outside the document pipeline.

The configuration phase runs the 24 Docling Standard combinations formed by three
OCR engines, two OCR modes, two TableFormer modes, and formula enrichment off/on.
It preserves text, page order, tables, formulas, bounding boxes where available,
and parser provenance in the common extracted-document contract.

| Varied factor | Values | Reason |
|---|---|---|
| OCR engine | RapidOCR, Tesseract, EasyOCR | Compares classical and neural recognition inside the same layout pipeline. |
| OCR mode | PDF-aware regions, full page | Tests native-text preservation against complete raster recognition. |
| TableFormer | fast, accurate | Measures table-structure quality versus cost. |
| Formula enrichment | off, on | Measures whether CodeFormula recovery justifies loading the extra model. |

The full `3 × 2 × 2 × 2` matrix retains interactions between recognition, page
mode, table reconstruction, and formula recovery. Docling 2.117.0, English, OCR
scale 3.0, table-cell matching, canonical output, code enrichment off, and native
DOCX ingestion stay fixed. These values define the common English experiment and
required output; varying them would introduce a separate language, rendering,
code-extraction, or DOCX-conversion question.

```powershell
python experiments/benchmarks/extraction/document/run.py --profile smoke
python experiments/benchmarks/extraction/document/run.py --profile standard --phase configuration
python experiments/benchmarks/extraction/document/run.py --profile standard --phase architecture --shortlist SUMMARY_JSON
```

The architecture phase compares the non-dominated Standard configurations with
Granite Docling 258M and PaddleOCR-VL-1.6. Primary metrics are CER, WER, reading
order, page attribution, block/table/formula quality, and content F1. Operational
metrics include p50/p95 latency, throughput, RAM, and VRAM. Standard/full retain
per-document results and paired bootstrap confidence intervals.

The architecture comparison uses the common image/PDF subset because Granite and
Paddle are visual parsers. DOCX remains native Docling input and is measured in the
Standard configuration/control path; it is never rasterized merely to make a VLM
appear applicable.

Full details and evidence for these choices are in the
[model-selection document](../model-selection.md).
