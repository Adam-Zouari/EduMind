# System Status

This document describes the current repository state after the structural modernization.

## Repository status

The active codebase is organized around a single installable package and thin runtime entrypoints.

- package code: `src/edumind/`
- apps: `apps/`
- services: `services/`
- experiments: `experiments/mlflow/`
- tests: `tests/`
- current docs: `docs/`
- local runtime state: `artifacts/`

## Runtime status model

The repo supports three first-class operating modes:

1. Direct Streamlit UI through the internal package
2. Optional OCR and RAG microservices through FastAPI
3. Local MLflow experiment execution

The direct UI is the default product path. Microservices remain available for API testing and deployment experiments.

## Dependency model

The project now uses one root `pyproject.toml` with optional extras:

- `ocr`
- `rag`
- `ui`
- `api`
- `experiments`
- `dev`

The preferred local environment is a single root `.venv`.

## Storage and artifact policy

Tracked in Git:

- source code
- curated test fixtures
- documentation
- configuration templates

Kept out of Git and routed into `artifacts/`:

- vector store state
- experiment MLflow database and run artifacts
- temporary uploads
- caches and local outputs

## Quality gates

Current automated checks:

- `ruff check .`
- `mypy src`
- `pytest`
- GitHub Actions CI on push and pull request

Local heavyweight capabilities such as Ollama, Tesseract, PaddleOCR, and Whisper remain opt-in runtime dependencies rather than CI requirements.

## Remaining production-minded gaps

The repo now presents well for a portfolio and team handoff, but it is still a local-first system rather than a fully hardened production deployment.

Areas that would still need work for a true production rollout:

- secrets management beyond local `.env` usage
- deployment manifests and infrastructure automation
- persistent hosted observability
- API authentication and authorization
- stronger contract and load testing
- containerized runtime parity across OCR and RAG stacks

## Source-of-truth files

- dependency and tool configuration: `pyproject.toml`
- shared runtime config: `config/base.yaml`
- environment variables: `.env.example`
- CLI entrypoints: `src/edumind/cli.py`
- CI workflow: `.github/workflows/ci.yml`
