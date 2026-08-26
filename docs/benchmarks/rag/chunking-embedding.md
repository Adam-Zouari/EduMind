# Chunking and embedding benchmark

[Benchmark overview](../overview.md) · [Benchmark manual](../methodology.md) ·
[Candidate selection](../model-selection.md) · [Preparation guide](../../setup/installation.md)

## Question

Which deployable chunker–embedding pair retrieves verified educational evidence most effectively? Exact NumPy search isolates this choice from vector-server ANN error.

The standard experiment crosses eight chunkers with eight embeddings for exactly 64 pairs. The chunkers are recursive character, token 256/32, token 384/64, token 512/64, sentence 8/2, semantic, section-aware 512/64, and structure-aware 512/64. The embeddings are MiniLM, Snowflake Arctic Embed M v2, F2LLM v2 0.6B, Octen 0.6B/4B, Qwen3 Embedding 0.6B/4B, and Nemotron Embed 1B.

All candidates use exact pinned local snapshots and explicit contracts for query/document interfaces or prefixes, tokenizer limit, checkpoint pooling behavior, normalized vectors, dimensionality, and cosine similarity. The same model supplies semantic boundaries and retrieval vectors for a semantic pair; that result selects a complete deployable pair and is not interpreted as an isolated embedding effect.

## Data and procedure

Standard uses the development selection manifest; full uses explicit finalists on validation. The paper/document is the split unit. Chunks retain exact half-open offsets `[start, end)`. Evidence overlap is:

```python
max(0, min(chunk_end, evidence_end) - max(chunk_start, evidence_start))
```

Retrieved intervals are merged before evidence recall so overlapping chunks cannot receive duplicate credit. Candidate order and query order use seed 42; standard/full run three measured repetitions.

## Metrics

The main metric is nDCG@5. The other required quality metrics are nDCG@3, Context Recall@3/@5, rank-aware Context Precision@3/@5, and Context Recall under 2,048 retrieved tokens. Diagnostics are Precision, Recall, Hit Rate, and Context Precision/Recall at 1/3/5/10; MAP@3/@5/@10; MRR; and nDCG@10. Operational output includes chunk count, token distribution, indexing time, embedding storage, and p50/p95 query latency.

The runner reports paired intervals and no ranking. After reviewing the complete MLflow run, an engineer may record at most three pairs in a decision file for the retrieval experiment.

```powershell
python experiments/benchmarks/rag/chunking_embedding/run.py --profile smoke
python experiments/benchmarks/rag/chunking_embedding/run.py --profile standard
python experiments/benchmarks/rag/chunking_embedding/run.py --profile full --shortlist DECISION_JSON
```
