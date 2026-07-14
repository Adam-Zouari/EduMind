# RAG subsystem

The RAG package indexes `ExtractedDocument` values and answers questions from ranked, cited evidence. Benchmarks import these production contracts rather than maintaining experiment-only copies.

## Indexing

1. An offset-preserving `ChunkingStrategy` creates `ChunkRecord` values.
2. `EmbeddingSpec` applies the model revision, tokenizer, prefixes, normalization, dimension, similarity, maximum length, and indexing/query devices.
3. `IndexManifest` records content, embedding, chunking, and backend contracts.
4. `VectorStore.replace_document` replaces every chunk for one logical source under a local lock, updates the lexical index atomically, and refreshes the content checksum.

An index with an incompatible contract raises `IndexCompatibilityError` with a rebuild instruction. Query filters are allowed only for fields declared in the manifest. Chroma multi-field filters use explicit `$and` clauses.

## Retrieval and generation

Production retrieval supports dense ranking, BM25, reciprocal-rank fusion, and lazy cross-encoder reranking. RRF combines ranks rather than incompatible dense and lexical scores. Contexts are packed in rank order under the configured token budget; there is no global similarity threshold.

Selectable stacks are `dense`, `bm25`, `rrf`, `rrf-minilm-reranker`, and `rrf-qwen3-reranker`. A reranker stack requires an immutable revision and defaults to CPU. Ollama generation resolves an unpinned configured model through the prepared digest lock and refuses to start if the lock is missing or incompatible.

Prompts number each source and require bracket citations. `GenerationProfile` pins the Ollama model/digest contract, thinking mode, seed, sampling, context limit, and output limit. Runtime logs exclude document text, questions, generated answers, and reasoning traces by default.

Indexing and query devices are configured independently. The local default embeds and reranks queries on CPU so the 4 GB GPU remains available to Ollama.

## Promotion boundary

Packaged defaults point to `recommendations/default.json`. That manifest remains non-authoritative until standard retrieval/generation benchmarks, 60 blinded human judgments, and the one-time locked test are complete. Smoke success only proves that contracts execute.

See `experiments/benchmarks/rag/` for exact formulas, candidates, and promotion gates.
