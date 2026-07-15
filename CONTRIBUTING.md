# Contributing

## Workflow

1. Create a feature branch from `main`.
2. Install the app and benchmark lock files in a fresh virtual environment.
3. Install the source once with `python -m pip install -e . --no-deps`.
4. Run the two small metric/dataset checks and the smoke script for the area you changed.
5. Keep generated outputs in `artifacts/`, not in Git.

## Structure expectations

- Reusable Python code belongs in `src/edumind/`.
- User-facing apps belong in `apps/`.
- Benchmark code, candidates, metrics, and procedures belong in `experiments/benchmarks/`.
- Do not add an API/service layer until deployment requirements justify it.
- Long-form notes belong in `docs/`.

## Pull requests

- Prefer focused PRs with clear intent.
- Update docs when behavior, structure, or setup changes.
- Add a focused metric/dataset example when experimental validity logic changes.
