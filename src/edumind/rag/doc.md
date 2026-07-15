# Production RAG

The current production path is intentionally one provisional stack: exact-offset token chunks (256 tokens, 32 overlap), the audited `all-MiniLM-L6-v2` embedding contract, dense search through Chroma HTTP, ranked context packing under 2,048 tokens, and cited Ollama generation with `qwen3:1.7b`.

`EmbeddingSpec` fixes the model revision, tokenizer, prefixes, normalization, dimension, similarity, maximum length, and indexing/query devices. `IndexManifest` rejects an existing Chroma collection when the embedding or chunking contract is incompatible. Whole-document replacement removes every old chunk before adding the new version and restores the previous rows if insertion fails. Compound Chroma filters use `$and`.

There is no global similarity threshold. Contexts are packed in rank order and truncated only at tokenizer boundaries. Prompts number evidence blocks and require bracket citations; an empty ranking produces an evidence-based refusal.

Production does not contain BM25, RRF, rerankers, alternative database clients, an embedded store, or recommendation-manifest machinery. Those candidates are evaluated under `experiments/benchmarks` and may enter production only through a later explicit code/config change.
