# Quick Reference

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,ui,api,rag,experiments,ocr]
```

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .[dev,ui,api,rag,experiments,ocr]
```

## Run

Default UI:

```bash
python -m edumind.cli ui
```

Microservices:

```bash
python -m edumind.cli ocr-api
python -m edumind.cli rag-api
python -m edumind.cli ui-microservices
```

Experiments:

```bash
python -m edumind.cli experiments
```

## Important paths

- package root: `src/edumind/`
- apps: `apps/`
- services: `services/`
- experiments: `experiments/mlflow/`
- shared config: `config/base.yaml`
- environment defaults: `.env.example`
- vector store: `artifacts/rag/vector_store/`
- experiment MLflow state: `artifacts/mlflow/`
- curated fixtures: `data/evaluation/`

## Ports

- `8501` - Streamlit UI
- `8000` - OCR API
- `8001` - RAG API
- `11434` - Ollama

## Quality checks

```bash
ruff check .
mypy src
pytest
```

## Health checks

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:11434/api/tags
```

## Current defaults

- LLM model: `qwen3:1.7b`
- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- chunk size: `1000`
- chunk overlap: `200`
- vector collection: `ocr_documents`

## If something breaks

- reinstall editable dependencies with `pip install -e .[...]`
- verify Ollama is running before answer-generation tests
- verify Tesseract and FFmpeg are installed for OCR media workflows
