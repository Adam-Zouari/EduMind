# Retrieval and reranking experiment

## Question, candidates, and control

For at most three promoted chunking/embedding pairs, which production retrieval stack best ranks evidence? Compare dense (control), BM25, dense+BM25 reciprocal-rank fusion, RRF+MiniLM cross-encoder, and RRF+Qwen3 Reranker 0.6B. Dense and BM25 retrieve 20 before fusion/reranking.

## Dataset, splits, and procedure

Use the same frozen paper-level QASPER development/validation manifests and exact offsets as chunking/embedding. Each stack sees identical chunks, vectors, query order, filters, top-20 pool, seed, warmups, and repetitions. RRF uses `sum(1/(60+rank))`, combining ranks rather than incompatible raw scores. Production `BM25Ranker`, `reciprocal_rank_fusion`, and `CrossEncoderReranker` are called directly.

## Metrics and rationale

Primary metrics are nDCG@3/5, Context Precision@3/5, Context Recall@3/5, and Context Recall@2,048 tokens, each 0–1 and higher, with formulas defined in the chunking/embedding document. Diagnostics are Precision/Recall/Hit Rate@1/3/5/10, MAP@3/5/10, MRR, nDCG@10, and Context Precision/Recall@1/10. Reranker latency, p50/p95 total retrieval, tokens, CPU/GPU/RAM/VRAM, and storage remain operational objectives.

## Statistics, promotion, and artifacts

Paired per-query bootstrap intervals and Holm-corrected declared comparisons follow hard correctness/resource gates. Pareto selection advances at most three complete stacks; interval ties prefer p95, memory, then storage. Artifacts include the plan/provenance, model revisions, per-query ranks/metrics, aggregate intervals, Pareto set, success marker, and optional nested MLflow runs.

## Commands, example, and limitations

```powershell
edumind benchmark --profile smoke rag retrieval
edumind benchmark --profile standard rag retrieval
edumind benchmark --profile full rag retrieval
```

Example: a reranker with overlapping recall but 4x p95 is not preferred. MAP is diagnostic, not a second vote in an overall score. Import success does not prove ranking quality, and this component run cannot establish generation faithfulness.
