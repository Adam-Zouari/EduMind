# Changelog

## Unreleased

- Simplified production to Streamlit -> extraction/RAG pipeline -> Chroma HTTP -> Ollama.
- Moved all experiment implementation, candidates, metrics, MLflow logging, and documentation to `experiments/benchmarks`.
- Removed the API/service layer, packaged benchmark framework, embedded vector databases, duplicate application modules, recommendation machinery, and broad build/lint/coverage workflows.
- Split application and benchmark dependencies into separate lock files.
- Added direct extraction, RAG, and four-server vector-database benchmarks with per-sample artifacts and confidence intervals.
