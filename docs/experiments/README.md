# Experiments

All maintained experiment code is technology-neutral under `edumind.benchmarks`, with entry wrappers and method documents under [`experiments/benchmarks`](../../experiments/benchmarks/doc.md). MLflow is an optional tracking adapter, not the experiment architecture or a normal runtime dependency.

Each experiment document defines its hypothesis, controls, manifests/splits, exact procedure, formulas/directions, promotion gates, artifacts, commands, worked interpretation, and invalid conclusions. Generated runs live under `artifacts/benchmarks/`; only completed successful candidate results are reusable.
