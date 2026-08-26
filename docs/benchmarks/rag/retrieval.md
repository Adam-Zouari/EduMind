# Retrieval and reranking benchmark

[Benchmark overview](../overview.md) · [Benchmark manual](../methodology.md) ·
[Chunking/embedding stage](chunking-embedding.md) · [Candidate selection](../model-selection.md)

## Question and candidates

For up to three explicitly approved chunker/embedding pairs, which ranking strategy retrieves the best evidence? Compare dense, BM25, reciprocal-rank fusion of dense and BM25, and RRF followed separately by MiniLM, Ettin 150M, Ettin 400M, Ettin 1B, or Qwen3 4B reranking. BM25, RRF, and rerankers live only in experiments.

## Data and procedure

Pass an engineer decision from the complete chunking/embedding run with `--embedding-selection`. The runner validates at most three selected pairs, crosses each pair with every strategy, builds its exact index, retrieves 20 before fusion/reranking, and evaluates the combined text/table/formula/mixed manifest at top 10. RRF combines ranks as `sum(1 / (60 + rank))`; incompatible raw dense/BM25 scores are never mixed. Standard/full repeat queries three times and report determinism without treating repeats as extra questions. Evidence-type aggregates remain mandatory diagnostics so a structural failure cannot be hidden by the larger text stratum.

## Metrics

Primary metrics are nDCG@3/@5, Context Precision@3/@5, Context Recall@3/@5, and Context Recall under 2,048 tokens. Diagnostics are Precision/Recall/Hit Rate@1/3/5/10, MAP@3/5/10, MRR, nDCG@10, Context Precision/Recall@1/@10, retrieved tokens, and determinism. Operational output includes total ranking p50/p95, indexing duration, chunk distribution, vector storage, RAM, and optional VRAM; reranker cost is included in total ranking latency rather than reported as a fabricated separate timer.

Standard/full retain per-question observations, individual 95% intervals, and paired candidate-difference intervals. The main metric is nDCG@5, but it does not select a winner. An engineer reviews all metrics in MLflow and may record at most three complete retrieval stacks for final RAG.

## Commands and limits

```powershell
python experiments/benchmarks/rag/retrieval/run.py --profile smoke --no-mlflow
python experiments/benchmarks/rag/retrieval/run.py --profile standard --embedding-selection EMBEDDING_DECISION_JSON
python experiments/benchmarks/rag/retrieval/run.py --profile full --shortlist RETRIEVAL_DECISION_JSON
```

Import or smoke success is not evidence of ranking quality. This stage cannot establish generation faithfulness or install a winner into production.

Artifacts are plan/provenance, candidate JSON, per-question Parquet, paired intervals, a completeness report, and local MLflow runs. The runner presents the evidence and does not interpret the trade-off for the engineer.
