# Contributing

## Workflow

1. Create a feature branch from `main`.
2. Install the project in editable mode with the extras you need.
3. Run `ruff check .`, `mypy src`, and `pytest` before opening a PR.
4. Keep generated outputs in `artifacts/`, not in Git.

## Structure expectations

- Reusable Python code belongs in `src/edumind/`.
- User-facing apps belong in `apps/`.
- API surfaces belong in `services/`.
- Benchmark implementation belongs in `src/edumind/benchmarks/`; experiment protocols belong in `experiments/benchmarks/`.
- Long-form notes belong in `docs/`.

## Pull requests

- Prefer focused PRs with clear intent.
- Update docs when behavior, structure, or setup changes.
- Add or adjust tests when logic changes.
