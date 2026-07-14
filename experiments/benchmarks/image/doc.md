# Image-to-text experiment

## Hypothesis and candidates

Compare Tesseract 5, PP-OCRv5 English mobile/server, and docTR fast_base + PARSeq under raw, document, and photo preprocessing. The hypothesis is that preprocessing interacts with engine choice, so the standard run uses the feasible engine × preprocessing factorial rather than selecting them independently.

## Dataset, splits, and procedure

The authoritative manifest contains 120 English pages split by source document 72/24/24 across clean scans, noisy/skewed scans, phone photos, low resolution, and multi-column pages. It pins license, source revision, IDs, checksums, and ground truth. Candidate order is seeded and randomized; each engine is cold-loaded, warmed, then measured. The benchmark invokes the registered production `Extractor`.

## Metrics and rationale

Character Error Rate is character edit distance/reference characters; Word Error Rate is word edit distance/reference words. Both range from 0 upward and lower is better. Reading Order Accuracy measures correctly ordered annotated pairs. Text Block precision/recall/F1 measure matched annotated blocks. Page Coverage is pages with expected text recovered/pages. Diagnostics are Word Accuracy, missing and hallucinated text rates, empty and duplicate output rates, and determinism. Operational metrics are cold load, p50/p95 page latency, pages/minute, RAM/VRAM, temporary disk, and cache speedup.

Correctness gates reject crashes, empty-output regressions, nondeterminism, and incompatible licenses. Quality then uses CER/WER, reading order, block F1, and page coverage as separate Pareto objectives. Precision, recall, F1, coverage, word accuracy, and determinism range 0–1 and higher is better; error/empty/duplicate rates range 0–1 and lower is better.

## Statistics, promotion, and artifacts

Per-page measurements are stored before 95% fixed-seed bootstrap intervals. Declared multiple comparisons use Holm correction. Hard gates precede Pareto selection; overlapping quality intervals prefer p95, memory, then storage. The run directory contains plan/provenance/model revisions, all sample results, aggregate intervals, Pareto candidates, `_SUCCESS.json`, and optional MLflow parent/child runs.

## Commands, worked example, and limitations

```powershell
edumind benchmark --profile smoke extraction image
edumind benchmark --profile standard extraction image
edumind benchmark --profile full extraction image
```

Example: lower CER with overlapping confidence intervals does not justify a slower engine; choose the lower-p95 non-dominated candidate. The experiment does not validate handwriting, non-English OCR, tables, formulas, or forms.
