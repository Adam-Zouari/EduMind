# Retrieval-augmented generation subsystem

[Project overview](../../README.md) ·
[Architecture](overview.md) ·
[Pipeline](application.md) · [RAG benchmark methodology](../benchmarks/methodology.md#5-chunking-and-embedding)

## Role

`edumind.rag` turns normalized extracted text into indexed chunks, retrieves a
bounded evidence set for a question, and optionally generates a cited answer.

```text
ExtractedDocument -> exact-offset chunks -> embeddings -> Chroma HTTP
question -> query embedding -> ranked hits -> token-budget context -> answer
```

## Current production profile

| Component | Provisional default |
|---|---|
| Chunker | Token 256/32 with exact source offsets |
| Embedding | Pinned MiniLM, 384 dimensions, normalized cosine vectors |
| Vector store | Chroma HTTP server |
| Retrieval | Dense top-5 from 20 candidates |
| Context packing | Ranked evidence capped at 2,048 tokens |
| Generator | Pinned Hugging Face Qwen3 1.7B, CPU, thinking disabled |

Numbered contexts are passed to a citation-constrained prompt. The generator uses
the checkpoint chat template, native checkpoint dtype, one whole-model device,
temperature 0, seed 42, and a fixed output limit. It never falls back to a hosted
service or downloads a missing checkpoint.

## Contracts and compatibility

Each embedding profile records its exact revision, tokenizer/prompt interface,
pooling behavior, dimension, normalization, similarity, maximum length, and
index/query devices. Chunking preserves half-open offsets. Their fingerprints are
stored in the index manifest so an incompatible index is rejected with a rebuild
instruction.

The vector database receives precomputed embeddings; it never chooses or creates
an embedding model. Re-ingesting a logical document replaces all of its chunks so
stale content is not left searchable.

## Experiment boundary

Alternative chunkers/embeddings share this production contract. BM25, reciprocal
rank fusion, rerankers, and alternative vector servers remain experiment-only
until an explicit benchmark-backed production change.

- [Chunking and embedding experiment](../benchmarks/methodology.md#5-chunking-and-embedding)
- [Retrieval and reranking experiment](../benchmarks/methodology.md#6-retrieval-and-reranking)
- [Generation experiment](../benchmarks/methodology.md#8-generation)
- [Final RAG experiment](../benchmarks/methodology.md#9-final-rag-and-human-review)
- [Vector database experiment](../benchmarks/methodology.md#7-vector-database-servers)
