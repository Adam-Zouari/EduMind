# Changelog

## 0.1.0 - 2026-07-09

- Reorganized the repo around `src/edumind`, `apps`, `services`, `experiments`, `docs`, `config`, and `data`.
- Added a root `pyproject.toml` with optional dependency groups and CLI entrypoints.
- Moved runtime state to `artifacts/` and evaluation fixtures to `data/evaluation/`.
- Replaced path-hack imports in the main OCR, RAG, pipeline, and service layers.
- Added CI scaffolding, contributor docs, environment examples, and pytest-based tests.
