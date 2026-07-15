# Image-to-text benchmark

## Question and candidates

Which OCR engine and preprocessing pair recovers English educational text most accurately? Standard compares Tesseract 5, PaddleOCR PP-OCRv5 mobile/server, and docTR fast_base + PARSeq with raw, document, and photo preprocessing. The production extraction registry executes every candidate.

## Data and procedure

The target corpus is 120 pages split by source document 72/24/24 across clean scans, noisy or skewed scans, phone photos, low resolution, and multi-column pages. Manifests pin source, license, revision, asset checksum, preprocessing version, and split seed. Candidate order is randomized with seed 42. The first invocation measures lazy load, two invocations warm the engine, and standard/full measure every page three times. Quality is computed once per page; repeated output equality measures determinism, avoiding false statistical inflation.

## Metrics

- CER = character Levenshtein distance / reference characters; WER is the word equivalent. Range 0 upward, lower is better.
- Content precision/recall/F1 use normalized token multisets. Missing and hallucinated text rates are unmatched reference/predicted token fractions. Range 0-1.
- Reading Order Accuracy is the fraction of pairwise orders preserved among tokens occurring once in both texts. Range 0-1, higher is better.
- Block precision/recall/F1 use exact normalized non-empty lines as blocks. Range 0-1, higher is better.
- Page Coverage = pages containing extracted text / annotated `reference_pages`. Range 0-1, higher is better.
- Word Accuracy is `max(0, 1 - min(1, WER))`; empty output, repeated-line rate, and determinism are diagnostics.
- Operations: cold invocation, p50/p95 page latency, pages/minute, process RAM, and VRAM when measurable.

Standard/full retain per-page values and compute 95% intervals with 10,000 bootstrap resamples. Pareto selection keeps quality objectives separate; no weighted score is created. Smoke only proves the real path executes.

## Commands and limits

```powershell
python experiments/benchmarks/extraction/image/run.py --profile smoke
python experiments/benchmarks/extraction/image/run.py --profile standard
python experiments/benchmarks/extraction/image/run.py --profile full --shortlist SUMMARY_JSON
```

Artifacts are `plan.json`, `provenance.json`, candidate JSON, `samples/*.parquet`, and `summary.json`, also logged to local MLflow. This benchmark does not establish handwriting, non-English, table, formula, or form extraction quality.

Example: if two CER intervals overlap, a small point-estimate difference is not a quality win; compare p95 latency, then RAM/storage, while confirming neither candidate fails determinism.
