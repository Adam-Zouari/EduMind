# Extraction routing and cache experiment

## Question, candidates, and control

Can a deployable router approach the per-page extraction oracle without stale cache behavior? Compare always-native (control), always-OCR, document-level digital/scanned/mixed routing, and page-level hybrid routing through the production registry.

## Dataset, splits, and procedure

Use the frozen PDF development/validation/test documents with page-level route labels and every extractor's paired page quality. Run each policy in seeded order. The oracle chooses the measured best eligible extractor per page and is only an upper bound. Cache cases independently mutate source bytes, options, engine revision, preprocessing, routing, and normalization; concurrency cases race identical requests and validate the resulting atomic artifact.

## Metrics and rationale

Router Selection Accuracy is correct annotated choices/all choices (0–1, higher). Quality Regret is `oracle page quality - selected page quality` (0 upward, lower); negative floating error is clamped to zero. Fallback Success Rate is successful recoveries/attempted fallbacks (0–1, higher). Cache Correctness is outputs equal to a fresh run, invalidation is changed cases correctly missed/all changed cases, and hit rate is hits/eligible requests. Correctness/invalidation/determinism are hard gates; hit rate and speedup are diagnostics.

## Statistics, promotion, and artifacts

Paired per-page bootstrap intervals compare routing policies; Holm correction applies to declared multiple route comparisons. Gates precede Pareto selection and interval ties use p95, memory, then storage. The run saves samples, route decisions, cache outcomes, plan/provenance, summary, success marker, and optional MLflow parent/children.

## Commands, example, and limitations

```powershell
edumind benchmark --profile smoke extraction routing
edumind benchmark --profile standard extraction routing
edumind benchmark --profile full extraction routing
```

Example: a 10x cache speedup is invalid if an engine-revision change returns old output. Oracle quality is not deployable performance. Results do not prove quality on unrepresented PDF layouts or on unsupported web/table/formula/form inputs.
