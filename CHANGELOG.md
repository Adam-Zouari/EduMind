# Changelog

## Unreleased

- Reworked the README and documentation hierarchy around project purpose, reader
  tasks, architecture, setup, experiment order, and authoritative sources.
- Aligned production with Docling Standard, Whisper `small.en`, MiniLM, Chroma HTTP,
  and direct pinned Hugging Face Qwen3 1.7B controls.
- Kept experiment implementation, candidates, metrics, and MLflow logging under
  `experiments/benchmarks`, with all reader documentation centralized in `docs/`.
- Moved Streamlit and end-to-end orchestration into the installed `edumind`
  package and separated experiment-only model and strategy implementations.
- Removed the API/service layer, packaged benchmark framework, embedded vector databases, duplicate application modules, recommendation machinery, and broad build/lint/coverage workflows.
- Split application and benchmark dependencies into separate lock files.
- Added direct extraction, RAG, and four-server vector-database benchmarks with per-sample artifacts and confidence intervals.
- Made the approved selection evidence authoritative for model preparation and removed
  retired candidate registries, model locks, and serving-specific generation code.
