# Chunking x embedding experiment

## Question, candidates, and control

Which chunker/embedding pair preserves QASPER evidence near the top of exact search? Run all 20 combinations: recursive character, token 256/32, token 384/64, sentence 8/2, and semantic chunking with MiniLM, BGE base, Nomic v1.5, and Qwen3 Embedding 0.6B. Exact NumPy cosine/dot search is the control that removes ANN backend error.

## Dataset, splits, and procedure

QASPER is frozen by paper: 100 development, 40 validation, and 40 locked-test papers, with answerability, accepted answers, evidence IDs, normalized half-open offsets, license/revision/checksum, preprocessing, and seed. The locked test is not used here during tuning. Every pair uses the production chunker and `EmbeddingSpec`, encodes the same corpus, retrieves 10 once per query, and reports all cutoffs. Candidate/query order is seeded with cold measurements, warmups, and repetitions.

## Metrics and formulas

Chunk/evidence overlap is `max(0, min(chunk_end,evidence_end) - max(chunk_start,evidence_start))`; retrieved intervals are unioned before recall so overlap cannot be negative or double-counted. nDCG@K is `DCG@K / ideal_DCG@K` with graded overlap relevance (0–1, higher). Context Recall@K is unioned evidence characters recovered/total evidence characters; rank-aware Context Precision@K discounts irrelevant or weakly relevant context at earlier ranks (0–1, higher). Context Recall@2,048 tokens applies the same recall after ranked token-budget packing. These at K=3/5 are primary.

Diagnostics are Precision, Recall, Hit Rate at 1/3/5/10; MAP@3/5/10; MRR; nDCG@10; and Context Precision/Recall@1/10. Operational data includes chunk count/token distribution, chunking/embedding throughput, retrieved tokens, p50/p95 latency, RAM/VRAM, and storage.

## Statistics, promotion, and artifacts

Per-query paired bootstrap intervals are computed with seed 42; declared multi-comparison claims use Holm correction. Gates precede Pareto selection, and at most three non-dominated pairs advance. Artifacts contain plan/provenance/model revisions, per-query results, intervals/Pareto set, success marker, and optional MLflow parent/children.

## Commands, example, and limitations

```powershell
edumind benchmark --profile smoke rag chunking-embedding
edumind benchmark --profile standard rag chunking-embedding
edumind benchmark --profile full rag chunking-embedding
```

Example: token-384 plus BGE may dominate token-256 plus BGE but says nothing about token-384 with Qwen; conclusions belong to pairs. Retrieval evidence quality does not prove answer quality, ANN performance, or extraction robustness.
