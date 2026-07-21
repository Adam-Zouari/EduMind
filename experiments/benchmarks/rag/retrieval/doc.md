# Retrieval and reranking benchmark

## Question and candidates

For up to three explicitly approved chunker/embedding pairs, which ranking strategy retrieves the best evidence? Compare dense, BM25, reciprocal-rank fusion of dense and BM25, and RRF followed separately by MiniLM, BGE v2-m3, Qwen3 0.6B, or Qwen3 4B reranking. BM25, RRF, and rerankers live only in experiments.

## Data and procedure

Pass the chunking/embedding `summary.json` with `--embedding-summary`. The runner rejects more than three unresolved Pareto pairs, crosses each approved pair with every strategy, builds its exact index, retrieves 20 before fusion/reranking, and evaluates the combined text/table/formula/mixed manifest at top 10. RRF combines ranks as `sum(1 / (60 + rank))`; incompatible raw dense/BM25 scores are never mixed. Standard/full repeat queries three times and report determinism without treating repeats as extra questions. Overall metrics select candidates; evidence-type aggregates remain mandatory diagnostics so a structural failure cannot be hidden by the larger text stratum.

## Metrics

Primary metrics are nDCG@3/@5, Context Precision@3/@5, Context Recall@3/@5, and Context Recall under 2,048 tokens. Diagnostics are Precision/Recall/Hit Rate@1/3/5/10, MAP@3/5/10, MRR, nDCG@10, Context Precision/Recall@1/@10, retrieved tokens, and determinism. Operational output includes total ranking p50/p95, indexing duration, chunk distribution, vector storage, RAM, and optional VRAM; reranker cost is included in total ranking latency rather than reported as a fabricated separate timer.

Standard/full retain per-question observations, individual 95% intervals, and paired candidate-difference intervals. Interval-aware Pareto selection is used; explicitly approve at most three complete retrieval stacks for final RAG.

## Commands and limits

```powershell
python experiments/benchmarks/rag/retrieval/run.py --profile smoke --no-mlflow
python experiments/benchmarks/rag/retrieval/run.py --profile standard --embedding-summary EMBEDDING_SUMMARY_JSON
python experiments/benchmarks/rag/retrieval/run.py --profile full --shortlist RETRIEVAL_SUMMARY_JSON
```

Import or smoke success is not evidence of ranking quality. This stage cannot establish generation faithfulness or install a winner into production.

Artifacts are plan/provenance, candidate JSON, per-question Parquet, paired intervals, Pareto set, and local MLflow runs. Example: a reranker with overlapping recall but four-times p95 is not preferred under the tie rule.
