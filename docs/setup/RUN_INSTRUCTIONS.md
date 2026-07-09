# Run Instructions

This guide covers the supported runtime modes after the repo modernization.

## Recommended mode: direct package UI

This is the main product path and the best default for demos and local development.

```bash
python -m edumind.cli ui
```

What it does:

- starts Streamlit on port `8501`
- imports `edumind.pipeline.orchestrator.OCRRAGOrchestrator` directly
- avoids the extra process overhead of local service mode

## Optional mode: microservices

Use this when you want isolated HTTP services or you are testing API boundaries.

Terminal 1:

```bash
python -m edumind.cli ocr-api
```

Terminal 2:

```bash
python -m edumind.cli rag-api
```

Terminal 3:

```bash
python -m edumind.cli ui-microservices
```

Windows helper:

```powershell
scripts\windows\start_all_services.bat
```

Ports:

- Streamlit UI: `http://localhost:8501`
- OCR API docs: `http://localhost:8000/docs`
- RAG API docs: `http://localhost:8001/docs`
- Ollama API: `http://localhost:11434`

## Standalone RAG UI

```bash
python -m edumind.cli rag-ui
```

Use this for retrieval and answer-generation work that does not need the OCR ingestion interface.

## Experiment runner

Quick pass:

```bash
python -m edumind.cli experiments
```

Full run plus MLflow UI:

```bash
python experiments/mlflow/run_all_experiments.py --full --ui
```

Skip LLM experiments when Ollama is unavailable:

```bash
python experiments/mlflow/run_all_experiments.py --skip-llm
```

## Health checks

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:11434/api/tags
```

Expected:

- OCR and RAG services return `{"status":"healthy"}`
- Ollama returns the locally available model list

## Common setup notes

- install from the repo root into `.venv`
- use `pip install -e .[dev,ui,api,rag,experiments,ocr]` for the full local stack
- copy `.env.example` to `.env` before local overrides
- `config/base.yaml` is the shared runtime config
- generated state should appear under `artifacts/`, not inside source folders

## Troubleshooting

- `ModuleNotFoundError`: make sure the environment was installed with `pip install -e ...`
- empty answers: confirm Ollama is running and `qwen3:1.7b` is available
- OCR failures on images: verify Tesseract is installed and reachable through `TESSERACT_CMD`
- audio or video extraction failures: verify `FFMPEG_PATH` points to a working FFmpeg binary
