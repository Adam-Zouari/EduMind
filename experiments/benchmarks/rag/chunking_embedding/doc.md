# Chunking and embedding benchmark

## Question and candidates

Which chunker/embedding pair puts verified educational evidence near the top of exact retrieval? Standard crosses recursive-character, token 256/32, token 384/64, token 512/64, sentence 8/2, semantic, section-aware 512/64, and structure-aware 512/64 chunking with MiniLM, EmbeddingGemma 300M, Jasper Token Compression 600M, Qwen3 Embedding 0.6B/4B, and Nemotron 3 Embed 1B/8B: all 56 combinations. Exact NumPy search isolates this decision from database ANN error.

## Data and procedure

`prepare.py qasper` creates paper-isolated text-only source manifests. `prepare.py rag-selection` combines each split with an independently verified structured manifest containing table, formula, and mixed-evidence questions. The resulting `rag-selection-dev`, `rag-selection-validation`, and `rag-selection-locked-test` manifests are the actual standard/full inputs. Component tuning never reads locked test. Every candidate uses the production chunker, tokenizer, query/document interface, prompts or prefixes, normalization, dimension, similarity, and pinned model revision. Section-aware respects headings and prose sections. Structure-aware additionally protects Markdown/HTML tables and displayed formulas and splits oversized Markdown tables at row boundaries. Both fall back deterministically when structure is absent. Standard/full execute each query three times for latency/determinism while computing one quality observation per question. Results are reported overall and by text/table/formula/mixed evidence type.

## Metrics

Chunk/evidence overlap is `max(0, min(chunk_end, evidence_end) - max(chunk_start, evidence_start))`. Intervals are merged before recall, preventing negative or double-counted overlap. Primary metrics are nDCG@3/@5, Context Recall@3/@5, rank-aware Context Precision@3/@5, and Context Recall after ranked packing to 2,048 tokens. Each ranges 0-1 and higher is better.

Diagnostics are Precision, Recall, and Hit Rate at 1/3/5/10; MAP@3/5/10; MRR; nDCG@10; Context Precision/Recall@1/@10; retrieved tokens; and determinism. Operational output is one-time indexing duration, chunk count, mean/p95 chunk tokens, repeated-query p50/p95 latency, vector bytes, process RAM, and VRAM when available. It does not claim separate chunking and embedding throughput because those timers are not independently instrumented.

Per-question values are retained. Standard/full calculate 10,000-resample 95% intervals and pairwise paired-bootstrap difference intervals. Interval-aware Pareto selection treats overlapping quality intervals as ties, then uses p95 latency, RAM, and storage. Explicitly approve at most three pairs for retrieval testing.

## Commands and limits

```powershell
python experiments/benchmarks/rag/chunking_embedding/run.py --profile smoke --no-mlflow
python experiments/benchmarks/rag/chunking_embedding/run.py --profile standard
python experiments/benchmarks/rag/chunking_embedding/run.py --profile full --shortlist SUMMARY_JSON
```

Artifacts are plan/provenance, candidate JSON, per-question Parquet, summary intervals/comparisons/Pareto set, and local MLflow runs. This component result does not prove answer quality, ANN performance, or extraction robustness.

Example: token-384/Qwen may dominate token-256/Qwen, but that says nothing about structure-aware/EmbeddingGemma; conclusions belong to complete pairs. Running all 56 pairs measures those interactions instead of assuming them away.
