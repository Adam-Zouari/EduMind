# Contributing

[Project overview](README.md) · [Documentation map](docs/README.md) ·
[Architecture](docs/architecture/overview.md) ·
[Benchmark methodology](docs/benchmarks/methodology.md) ·
[Benchmark runbook](docs/benchmarks/running.md)

## Workflow

1. Create a feature branch from `main`.
2. Install the app and benchmark lock files in a fresh virtual environment.
3. Install the source once with `python -m pip install -e . --no-deps`.
4. Run the focused validity checks and the smoke script for the area you changed.
5. Keep generated outputs in `artifacts/`, not in Git.

The repository intentionally has no Ruff, MyPy, coverage, or wheel-build gate.
For changes to benchmark selection, datasets, or metrics, run:

```powershell
python -m pytest tests/test_benchmark_metrics.py tests/test_benchmark_datasets.py tests/test_selection_alignment.py tests/test_documentation_links.py -q
```

## Structure expectations

- Reusable Python code belongs in `src/edumind/`.
- Deployable UI code belongs in `src/edumind/ui/`.
- Benchmark code, candidates, metrics, and procedures belong in `experiments/benchmarks/`.
- Do not add an API/service layer until deployment requirements justify it.
- Long-form notes belong in `docs/`.

## Documentation expectations

- Keep [README.md](README.md) focused on purpose, current status, first use, and
  navigation.
- Put complete installation/download instructions in the
  [installation guide](docs/setup/installation.md).
- Put experiment order, candidates, datasets, procedures, metric rationale, and
  limitations in the [benchmark methodology](docs/benchmarks/methodology.md).
- Put exact formulas and edge cases in the
  [metric reference](docs/benchmarks/metrics.md).
- Put benchmark commands and operational troubleshooting in the
  [benchmark runbook](docs/benchmarks/running.md).
- Put public candidate-screening evidence only in the
  [model-selection rationale](docs/benchmarks/model-selection.md).
- Do not create a stage page that repeats these authorities. Add a page only
  when it owns genuinely separate information.
- Link new pages from [docs/README.md](docs/README.md) so readers do not have to
  discover them from the directory tree.

## Pull requests

- Prefer focused PRs with clear intent.
- Update docs when behavior, structure, or setup changes.
- Add a focused metric/dataset example when experimental validity logic changes.
