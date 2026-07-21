# PDF extraction-routing benchmark

## Question and candidates

Can a routing policy approach the annotated extractor oracle? Compare always-native pypdf, always-OCR, a document-layout router, and page-level native/OCR hybrid. These are distinct real paths: always-OCR rasterizes every page; hybrid OCRs only pages whose native text is unusable.

## Data and procedure

Use frozen PDF manifests with digital/scanned/mixed layout labels, `oracle_engine`, `oracle_quality`, normalized reference text, page count, asset checksum, and provenance. Seed 42 fixes candidate/document order. Standard/full use a cold invocation, two warmups, and three measured repetitions through the production pipeline.

## Metrics

Router Selection Accuracy is exact agreement between the selected engine and `oracle_engine`. Quality Regret is `max(0, oracle_quality - measured_content_f1)`; lower is better and cannot become negative. Fallback Success Rate is one when the routed path returns non-empty content. CER, WER, token content metrics, reading order, block scores, page coverage, empty/repeated output, determinism, cold invocation, p50/p95 latency, documents/minute, RAM, and optional VRAM provide quality and operational context.

The oracle is only an annotated upper bound. Standard/full retain per-document results and bootstrap intervals, then apply gates and Pareto selection. Cache behavior is not mixed into this routing experiment; it requires a separate hypothesis and dataset if added later.

## Commands and limits

```powershell
python experiments/benchmarks/extraction/routing/run.py --profile smoke
python experiments/benchmarks/extraction/routing/run.py --profile standard --image-summary IMAGE_SUMMARY_JSON
python experiments/benchmarks/extraction/routing/run.py --profile full --shortlist SUMMARY_JSON --image-summary IMAGE_SUMMARY_JSON
```

Results do not generalize beyond represented PDF layouts. Text-only routing quality cannot establish table/formula quality; structural routing conclusions require annotated table/formula pages. Dedicated forms and web extraction remain out of scope.

Artifacts are plan/provenance JSON, candidate JSON, per-document Parquet, summary intervals/comparisons, and local MLflow runs. Example: high route-label accuracy is insufficient if quality regret remains worse than page-hybrid routing.
