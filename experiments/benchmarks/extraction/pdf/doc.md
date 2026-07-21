# PDF-to-text benchmark

## Question and candidates

Which extractor best recovers text, reading order, tables, formulas, and page provenance from digital, scanned, mixed, and broken-encoding PDFs? Standard compares pypdf, pdfplumber, Docling, PP-StructureV3, PaddleOCR-VL-1.6, GLM-OCR, MinerU 2.5 Pro, olmOCR 2, and the production page-level native/OCR hybrid. Routing policies are evaluated separately.

## Data and procedure

The target corpus is 60 documents split 36/12/12 by document across digital, scanned, mixed, broken-encoding, slide, and academic layouts. Manifests pin licenses, revisions, source checksums, normalized references, `reference_pages`, page-ordered `reference_page_texts`, preprocessing, and seed. Runs use seed 42, a cold invocation, two warmups, and three measured repetitions in standard/full. Temporary page images are created by the production hybrid extractor inside an automatically cleaned temporary directory.

## Metrics

CER, WER, normalized-token content precision/recall/F1, pairwise Reading Order Accuracy, exact normalized-line block precision/recall/F1, missing/hallucinated text, empty/repeated output, determinism, and Page Coverage are computed exactly as in the image benchmark. Same-page Content F1 scores the content assigned to each page. Page Attribution Accuracy assigns each predicted page to its highest-content-F1 reference page and checks whether the page number agrees. Missing and duplicate page rates expose collapsed or repeated page output. Lower error is better; precision, recall, order, structure, coverage, and determinism range 0-1 and are higher-better. Operations are cold invocation, p50/p95 document latency, documents/minute, process RAM, and VRAM when available.

The typed extraction contract distinguishes text, headings, lists, captions, tables, and formulas. Table detection/content/row-column relation F1 and formula detection/normalized-LaTeX similarity/exact match are reported only for annotated samples. OmniDocBench's official TEDS and CDM evaluators remain separate authoritative tracks; EduMind does not label its simpler transparent metrics as TEDS or CDM. Bounding-box metrics are reported only when both references and candidate output expose them.

Per-document 95% bootstrap intervals and separate Pareto objectives are produced for standard/full. Smoke is non-authoritative.

## Commands and limits

```powershell
python experiments/benchmarks/extraction/pdf/run.py --profile smoke
python experiments/benchmarks/extraction/pdf/run.py --profile standard --image-summary IMAGE_SUMMARY_JSON
python experiments/benchmarks/extraction/pdf/run.py --profile full --shortlist SUMMARY_JSON --image-summary IMAGE_SUMMARY_JSON
```

Artifacts are plan/provenance JSON, per-candidate Parquet and JSON, and `summary.json`, mirrored in local MLflow. Complete extraction claims require separate text, table, formula, provenance, and operational gates; one combined score is forbidden.

Example: low CER on digital PDFs cannot justify always-native extraction when its scanned-page coverage is zero; these remain separate Pareto objectives.
