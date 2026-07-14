# Quick reference

| Purpose | Location or command |
|---|---|
| Packaged defaults | `src/edumind/defaults.yaml` |
| Recommendation manifest | `src/edumind/recommendations/default.json` |
| Extraction API | `edumind extraction-api` on `127.0.0.1:8000` |
| RAG API | `edumind rag-api` on `127.0.0.1:8001` |
| UI | `edumind ui` |
| Smoke all | `edumind benchmark all` |
| Standard all | `edumind benchmark --profile standard all` |
| Benchmark artifacts | `artifacts/benchmarks/` |
| Runtime vector data | configured under `artifacts/` |

Verification: `pytest`, `ruff check src apps services experiments tests`, `mypy src apps services experiments`, `python -m pip check`, then smoke preflight/all.
