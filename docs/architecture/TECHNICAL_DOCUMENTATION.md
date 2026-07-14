# Technical architecture

EduMind is a package-first local extraction and cited-RAG system.

```text
source -> edumind.extraction -> ExtractedDocument
       -> edumind.pipeline -> edumind.rag -> cited answer
                              |-- chunking + embeddings
                              |-- Chroma + BM25 + RRF/reranker
                              `-- token-budget packing + Ollama

production contracts <-> edumind.benchmarks -> artifacts/recommendations
```

Configuration defaults ship inside the wheel and may be overridden by `.env`, `EDUMIND_CONFIG`, or typed programmatic overrides. Common utilities centralize hashing, atomic writes, locks, provenance, and artifact paths. No optional model downloads occur at import time.

The Streamlit and FastAPI files are adapters. Business behavior is testable below them, APIs bind locally, and errors are logged internally while clients receive stable safe schemas. Benchmark tracking is adapter based; MLflow is not a normal RAG dependency.

For precise contracts see the `doc.md` beside extraction, RAG, pipeline, apps, and services. For experimental methods see [`experiments/benchmarks/doc.md`](../../experiments/benchmarks/doc.md).
