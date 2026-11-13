# Retrieval-augmented generation subsystem

[Project overview](../../README.md) ·
[Architecture](overview.md) ·
[Pipeline](application.md) · [RAG benchmark methodology](../benchmarks/rag/chunking_embedding/methodology.md)

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
| Chunker | Token 256/32 with exact source offsets measured by `cl100k_base` |
| Embedding | Pinned GTE ModernBERT base, 768 dimensions, normalized cosine vectors |
| Vector store | Chroma HTTP server |
| Retrieval | Dense top-5 from 20 candidates |
| Context packing | Ranked evidence capped at 2,048 tokens |
| Generator | Pinned Hugging Face Qwen3 1.7B, CPU, thinking disabled |

The `2,048`-token context cap describes the provisional application only. The
chunking/embedding and retrieval/reranking benchmarks do not apply a context
budget when comparing candidates.

Numbered contexts are passed to a citation-constrained prompt. The generator uses
the checkpoint chat template, native checkpoint dtype, one whole-model device,
greedy decoding (`do_sample=false`), seed 42, and a fixed output limit. The
configured temperature is zero but is not passed to greedy generation.
It never falls back to a hosted service or downloads a missing checkpoint.

## Contracts and compatibility

Each embedding profile records its exact revision, tokenizer/prompt interface,
pooling behavior, dimension, normalization, similarity, maximum length, and
index/query devices. Chunking preserves half-open offsets. Their fingerprints are
stored in the index manifest so an incompatible index is rejected with a rebuild
instruction.

Ingestion retains extraction segment spans. Each chunk stores every overlapping
page/time/structure location, with exact source-text offsets, including chunks
crossing pages or segments. Context truncation clips those spans and removes
excluded locations; timestamps still describe the original segment, not an
invented word-level interpolation. Context headers and the UI display these
locations. Existing indexed chunks without this provenance need re-ingestion to
gain it; the application neither reconstructs missing locations nor resets an
old index automatically.

The vector database receives precomputed embeddings; it never chooses or creates
an embedding model. Re-ingesting a logical document replaces all of its chunks so
stale content is not left searchable.

Replacement snapshots the old IDs, text, metadata, and vectors. Writes and
compensating rollback use bounded batches respecting the server's supported
limit, including NumPy-backed restored vectors. A failed replacement is reported
even after successful restoration; if restoration also fails, both causes remain
in an exception group for local diagnostics. This is compensating rollback, not
a multi-client server transaction or a guarantee after restoration failure.

The generator owns its model through stream consumption and worker cleanup.
Stream failure requests cooperative decoder cancellation and joins the worker
before another request or unload can acquire ownership. A native call that does
not return cannot be forcibly killed by this thread-based adapter: ownership
remains held while cleanup waits. Backend exceptions retain their original cause.
Visible-answer parsing honors the provisional Qwen template's open/closed
thinking boundaries and hides unfinished reasoning. Reasoning-only output leaves
the visible answer empty; it is never restored as an answer fallback. These are
application behaviors, not benchmark generation or quality qualifications.

## Experiment boundary

Alternative chunkers/embeddings share this production contract. BM25, reciprocal
rank fusion, rerankers, and alternative vector servers remain experiment-only
until an explicit benchmark-backed production change.

- [Chunking and embedding experiment](../benchmarks/rag/chunking_embedding/methodology.md)
- [Retrieval and reranking experiment](../benchmarks/rag/retrieval_reranking/methodology.md)
- [Generation experiment](../benchmarks/rag/generation/methodology.md)
- [Final RAG experiment](../benchmarks/rag/final/methodology.md)
- [Vector database experiment](../benchmarks/vectordb/methodology.md)
