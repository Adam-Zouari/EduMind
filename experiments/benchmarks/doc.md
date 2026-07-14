# EduMind benchmark program

## Question

Which production extraction and RAG profiles give the best evidence-grounded quality while satisfying correctness, memory, and local latency gates on the target computer?

## Experimental contract

Every selectable strategy is registered in `edumind.benchmarks.registry` and the benchmark imports its production implementation. Smoke-only fakes are explicitly marked non-deployment-eligible and cannot create an authoritative recommendation. A run freezes the dataset checksum, full plan, Git/dirty hash, dependency lock, candidate revisions, seed, and hardware before evaluation. Development chooses candidates, validation promotes them, and the locked test is evaluated once after human review.

Each run writes `plan.json`, `provenance.json`, one complete per-candidate file containing per-sample observations, `summary.json`, and `_SUCCESS.json`. Only successful candidates with samples are cacheable. MLflow is an optional tracking adapter with a parent plan run and nested candidate runs; benchmark correctness never depends on MLflow.

## Statistics and promotion

Metrics are aggregated only after per-sample persistence. A 95% percentile confidence interval is calculated from 10,000 paired bootstrap resamples with seed 42 (`smoke` uses 500 solely for speed). Formal families of p-values use Holm correction. Correctness and resource gates run before Pareto selection. If quality intervals overlap, the tie-break order is p95 latency, peak memory, then storage. No normalized or weighted overall score is produced.

## Profiles and commands

```powershell
edumind benchmark --profile smoke preflight
edumind benchmark --profile smoke all
edumind benchmark prepare extraction-models
edumind benchmark --profile standard extraction image
edumind benchmark --profile standard rag chunking-embedding
edumind benchmark --profile standard systems vectordb
edumind benchmark report artifacts/benchmarks/<suite>/<stage>/<run>/summary.json
```

`smoke` is deterministic, offline, and non-authoritative. Its fake model reads committed modality fixtures through the real classifier, request, router, normalization, and result contracts. `standard` is the local decision run. `full` expands repetitions/candidates for overnight qualification. Preparation downloads optional extraction weights sequentially and writes an immutable local lock; missing datasets, engines, packages, model locks, or Ollama models fail preflight explicitly.

## Invalid conclusions

A smoke winner is not a quality winner. Component results do not prove end-to-end superiority. Automated generation metrics do not replace the 60 blinded judgments. Validation is not the locked test. Results from changed data, code, dependencies, model digests, or hardware are a different run fingerprint.
