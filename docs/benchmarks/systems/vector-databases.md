# Vector database server experiment

[Benchmark overview](../overview.md) · [Benchmark manual](../methodology.md) ·
[Candidate selection](../model-selection.md) · [Preparation guide](../../setup/installation.md)

## Question and candidates

Which self-hosted server preserves exact-neighbor and metadata-filter correctness while providing the best latency, throughput, memory, and storage trade-off? The first comparison is Chroma 1.5.9, Qdrant 1.17.0, Weaviate 1.38.2, and PostgreSQL 17 with pgvector 0.8.2. Chroma is the provisional application default, not an assumed winner.

The databases receive identical precomputed normalized float32 vectors, IDs, text, and metadata. They never create embeddings. NumPy computes the exact cosine neighbors only as the oracle; it is not a database candidate.

## Procedure

Prepare clients and immutable Docker digests, then start the four loopback-only servers explicitly:

```powershell
python experiments/benchmarks/prepare.py vectordb
docker compose -f infrastructure/chroma.yml down
docker compose --env-file experiments/benchmarks/vectordb/.env -f experiments/benchmarks/vectordb/compose.yml down -v
docker compose --env-file experiments/benchmarks/vectordb/.env -f experiments/benchmarks/vectordb/compose.yml up -d
python experiments/benchmarks/vectordb/run.py --profile smoke
python experiments/benchmarks/vectordb/run.py --profile standard
```

The production Chroma Compose file and benchmark Chroma use the same loopback port, so they must not run together. Removing the benchmark volumes before a measured run prevents old indexes from contaminating storage and ingestion measurements. The command targets only this named benchmark Compose project.

Smoke uses 1,000 vectors, 50 queries, and concurrency 1. Standard uses 100,000 vectors at dimensions 384 and 1,024, 500 queries, three repetitions, and concurrency 1/8/32. Full requires `--embedding-selection` with exactly one engineer-selected chunking/embedding pair; it rebuilds that pair's real combined text/table/formula/mixed RAG vectors and also uses a one-million-vector clustered corpus at the selected dimension with concurrency 1/8/32/64. Smoke proves only that the real path works.

For standard/full, each server tries `m {16,32}`, construction breadth `{100,200}`, and search breadth `{64,128}` on a 10,000-vector validation slice. Every supported trial's Recall@10 and p95 latency is logged in the server's MLflow child, and unsupported settings are counted. The lowest-latency setting reaching the documented Recall@10 reference target of 0.99 is measured; if none reaches it, the highest-recall supported setting is measured instead and the low recall remains visible evidence rather than failing the candidate. This repeatable configuration rule chooses the configuration used to compare servers; it does not select a database winner. Every run records Git state, hardware, client versions, server image digests, selected settings, and per-query observations.

Before timing, the runner verifies each live container against its prepared immutable image digest. Conformance then checks health, cosine behavior, vector-dimension rejection, compound and empty filters, duplicate-ID replacement, whole-document replacement without stale records, deletion, HNSW existence/use, and persistence across a real Docker restart.

## Metrics

`ANN Recall@K = |approximate top-K ∩ exact top-K| / |exact top-K|`; it ranges from 0 to 1 and higher is better. It is reported at K=1/3/5/10. Filtered ANN Recall uses the exact top-K after applying the identical filter. Filter correctness is the fraction of returned rows satisfying every requested predicate. A conformance flag of 1 means the behavior was correct; 0 records an observed correctness failure without turning the measured run itself into a runner failure.

Latency is measured around the complete client call, so serialization and loopback transport are included. Reports contain p50/p95/p99 latency, queries/second, error rate at each concurrency, build throughput, client and sampled peak server memory, persistent storage, and restart readiness. Resource values that cannot be measured are omitted, never replaced by zero.

## Engineering decision

ANN Recall@10 is the main metric. Filtered recall, conformance, latency, throughput, memory, and storage remain equally visible required evidence. The runner does not label a server eligible, apply a threshold, calculate a Pareto set, or promote anything. The engineer records the chosen trade-off only after reviewing a complete standard/full run. A smoke result, a lower latency obtained by silent flat search, or results from another unrecorded environment cannot support that decision.

The application remains on Chroma until a standard and full run plus the complete retrieval experiment justify an explicit change. This benchmark does not test clusters, cloud services, backups, rolling restarts, or multi-region behavior.

After explicitly approving exactly one chunking/embedding pair and one retrieval strategy, run the complete query-path comparison on the two dense finalists plus Chroma:

```powershell
python experiments/benchmarks/vectordb/retrieval_run.py `
  --database-selection VDB_DECISION_JSON `
  --embedding-selection EMBEDDING_DECISION_JSON `
  --retrieval-selection RETRIEVAL_DECISION_JSON
```

This stage uses the same experiment-only BM25/RRF/reranker implementation for every server and changes only the server-provided dense ranking. It reports the RAG retrieval metrics rather than treating ANN recall as application quality. It does not modify `config/base.yaml`.

Artifacts are plan/provenance, per-query Parquet, candidate JSON, conformance flags, selected/unsupported HNSW settings, paired intervals, the completeness report, and local MLflow runs. For example, MLflow will show both a server's lower p95 and its filtered Recall@10 of 0.97; deciding whether that trade-off is acceptable belongs to the engineer.
