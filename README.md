# EduMind-AI

EduMind-AI is a package-first OCR + RAG monorepo for document ingestion, semantic retrieval, and MLflow-backed experimentation.

The repository is organized for portfolio-quality maintainability:

- reusable Python code lives under `src/edumind/`
- apps and APIs are thin entrypoints
- experiments are isolated from product code
- generated state is routed into gitignored `artifacts/`
- setup, architecture, and historical notes are separated under `docs/`

## Repository map

```text
.
|-- apps/                  Streamlit entrypoints
|-- artifacts/             Local runtime outputs, caches, vector stores, MLflow state
|-- config/                Shared YAML configuration
|-- data/                  Curated fixtures and evaluation inputs
|-- docs/                  Current documentation plus archived legacy notes
|-- experiments/mlflow/    Experiment runners and analysis utilities
|-- scripts/               Cross-platform helper scripts
|-- services/              FastAPI service entrypoints
|-- src/edumind/
|   |-- common/            Shared paths, config loaders, schemas
|   |-- ocr/               Extraction pipeline and format-specific extractors
|   |-- pipeline/          OCR-to-RAG orchestration layer
|   `-- rag/               Chunking, embeddings, retrieval, and answer generation
`-- tests/                 Unit, integration, and experiment tests
```

## Quick start

EduMind-AI targets Python 3.10. The full local product workflow usually needs:

- Python 3.10
- Ollama for local LLM inference
- Tesseract for image OCR
- FFmpeg for audio and video extraction

Create one environment at the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev,ui,api,rag,experiments,ocr]
```

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -e .[dev,ui,api,rag,experiments,ocr]
```

For lighter workflows you can install only the extras you need:

- `.[ui,rag]` for RAG-only UI work
- `.[api,rag]` for service work without OCR
- `.[experiments]` for experiment analysis
- `.[ocr]` to add the heavier extraction stack

Copy environment defaults before running locally:

```bash
cp .env.example .env
```

Windows:

```powershell
Copy-Item .env.example .env
```

## Run the product

The default demo path is the direct package UI. It calls the internal package layer without requiring separate services.

```bash
edumind-ui
```

Equivalent command:

```bash
python -m edumind.cli ui
```

Optional API mode:

```bash
edumind-ocr-api
edumind-rag-api
```

Equivalent module commands:

```bash
python -m edumind.cli ocr-api
python -m edumind.cli rag-api
```

Legacy note:

- `python -m edumind.cli ui-microservices` now opens a deprecation notice page
- `python -m edumind.cli rag-ui` now opens a deprecation notice page

Windows convenience launcher:

```powershell
scripts\windows\start_all_services.bat
```

## Experiments

Run the bundled experiment suite:

```bash
python -m edumind.cli experiments
```

For advanced experiment flags:

```bash
python experiments/mlflow/run_all_experiments.py --full --ui
```

Curated evaluation inputs live in `data/evaluation/`. The experiment suite writes local MLflow state into `artifacts/mlflow/`.

## Configuration

- `config/base.yaml` is the shared runtime config.
- `.env.example` shows supported environment overrides.
- vector store data persists in `artifacts/rag/vector_store/`
- `EDUMIND_MLFLOW_TRACKING_URI` can point runtime logging at the shared MLflow store in `artifacts/mlflow/`

The repo keeps source code and curated fixtures in Git, while local runtime state stays outside the tracked tree.

## Development workflow

Quality checks:

```bash
ruff check .
mypy src
pytest
```

## OCR self-test

For OCR-only stabilization work, the official validation path is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev,ocr,api]
ruff check .
mypy src
pytest -q
```

From outside the repo root, confirm the editable install works:

```bash
python -c "from edumind.ocr import DataIngestionPipeline; print(DataIngestionPipeline)"
```

Manual OCR smoke checks:

- run a small PDF through `DataIngestionPipeline`
- run a scanned PDF with `pdf_ocr_mode="force"` and compare it to `pdf_ocr_mode="off"`
- run a small DOCX through `DataIngestionPipeline`
- run an image twice and confirm `artifacts/ocr/cache/` is populated and the second pass uses cache
- optionally test short audio and video files if Whisper and FFmpeg are installed locally

To benchmark the refined OCR path locally:

```bash
python scripts/ocr_benchmark.py
```

The benchmark runner writes JSON and CSV output into `artifacts/ocr/benchmarks/`.

CI runs the same core checks on pull requests with GitHub Actions.

## Documentation

- `docs/README.md` - documentation index
- `docs/setup/` - onboarding, commands, and environment guidance
- `docs/architecture/` - system design and package/module docs
- `docs/experiments/` - experiment workflow notes
- `docs/archive/` - legacy reports, screenshots, and historical notes

## License

MIT
