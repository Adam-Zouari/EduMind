# PDF-to-text benchmark

## Question and candidates

Which extractor best recovers digital, scanned, mixed, and broken-encoding PDFs? Standard compares pypdf, pdfplumber, Docling, and the production page-level hybrid. Routing policies are evaluated separately.

## Data and procedure

The target corpus is 60 documents split 36/12/12 by document across digital, scanned, mixed, broken-encoding, slide, and academic layouts. Manifests pin licenses, revisions, source checksums, normalized references, `reference_pages`, preprocessing, and seed. Runs use seed 42, a cold invocation, two warmups, and three measured repetitions in standard/full. Temporary page images are created by the production hybrid extractor inside an automatically cleaned temporary directory.

## Metrics

CER, WER, normalized-token content precision/recall/F1, pairwise Reading Order Accuracy, exact normalized-line block precision/recall/F1, missing/hallucinated text, empty/repeated output, determinism, and Page Coverage are computed exactly as in the image benchmark. Lower error is better; precision, recall, order, structure, coverage, and determinism range 0-1 and are higher-better. Operations are cold invocation, p50/p95 document latency, documents/minute, process RAM, and VRAM when available.

The current common extraction contract preserves page numbers but not heading/list semantic labels. Therefore the benchmark does not pretend to report Heading F1, List F1, or page-attribution accuracy until the manifest and extractor output expose those annotations. This is deliberate: an unavailable metric is omitted, never filled with zero.

Per-document 95% bootstrap intervals and separate Pareto objectives are produced for standard/full. Smoke is non-authoritative.

## Commands and limits

```powershell
python experiments/benchmarks/extraction/pdf/run.py --profile smoke
python experiments/benchmarks/extraction/pdf/run.py --profile standard --image-summary IMAGE_SUMMARY_JSON
python experiments/benchmarks/extraction/pdf/run.py --profile full --shortlist SUMMARY_JSON --image-summary IMAGE_SUMMARY_JSON
```

Artifacts are plan/provenance JSON, per-candidate Parquet and JSON, and `summary.json`, mirrored in local MLflow. No structured table, formula, or form claim is made.

Example: low CER on digital PDFs cannot justify always-native extraction when its scanned-page coverage is zero; these remain separate Pareto objectives.
