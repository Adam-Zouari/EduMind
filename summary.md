# EduMind implementation summary

EduMind is now organized around benchmarked production contracts rather than OCR-specific scripts or MLflow-specific experiments.

## Active architecture

- `edumind.extraction` classifies image, PDF, DOCX, audio, and video sources; routes them to lazy extractors; normalizes text; preserves offsets/pages/timestamps/provenance; and caches by source plus complete engine contract.
- `edumind.rag` provides exact-offset chunking, model-specific embeddings, dense/BM25/RRF retrieval, lazy reranking, token-budget context packing, cited Ollama generation, manifest compatibility, and atomic logical-document replacement.
- `edumind.pipeline` returns typed results, timings, warnings, readiness, and progress events.
- `edumind.app` owns testable UI behavior. `apps/streamlit_app.py` is only rendering and wiring.
- `services` contains bounded, local-only FastAPI adapters with stable errors and separate liveness/readiness.
- `edumind.benchmarks` and `experiments/benchmarks` contain the technology-neutral benchmark engine and documentation. MLflow is an optional tracking adapter.

The former `edumind.ocr`, `experiments/mlflow`, templated RAG evaluation data, and experiment-only strategy copies are removed. Web, structured table, formula, and dedicated form extraction are not supported.

## Benchmark status

Committed smoke fixtures exercise every extraction modality through the real pipeline with a deterministic fake model, all 20 chunking/embedding combinations, retrieval/generation contracts, and exact vector conformance without network or Ollama. Smoke results are explicitly non-authoritative.

Authoritative recommendation remains pending. It requires preparation of licensed manifest-driven corpora and QASPER splits; pinned optional models/backends; standard/full runs; three validation finalists; 60 blinded human judgments; and exactly one locked-test evaluation. Until then the packaged recommendation manifest declares `baseline_pending_benchmarks`, and Chroma remains the temporary backend.

## Validated engineering gates

The current implementation passes 69 unit/integration/benchmark tests, Ruff, MyPy, dependency validation, the complete 12-stage offline smoke run, and the 75% coverage gate (76.83% measured). A fresh wheel builds and loads packaged defaults and recommendations from outside the repository. Optional real-engine conformance still requires the explicit standard-model/data preparation; preflight reports those missing local tools, datasets, and model locks rather than silently skipping candidates.

See `README.md` for commands and `experiments/benchmarks/doc.md` for benchmark design.
