# Microservices Guide

Microservices are an optional advanced mode in the current repository. They are useful when you want explicit HTTP boundaries, separate process lifecycles, or deployment experiments.

## When to use this mode

- testing FastAPI endpoints directly
- isolating OCR and RAG runtime processes
- simulating a service-oriented deployment

If you just want the product running locally, use `python -m edumind.cli ui` instead.

## Start the services

OCR API:

```bash
python -m edumind.cli ocr-api
```

RAG API:

```bash
python -m edumind.cli rag-api
```

Windows launcher:

```powershell
scripts\windows\start_all_services.bat
```

The microservices helper no longer launches a dedicated Streamlit UI. Use the service APIs directly for boundary testing, and use `python -m edumind.cli ui` for the maintained local product UI.

## Endpoints

### OCR service

- `GET /`
- `GET /health`
- `POST /extract`
- `GET /formats`

### RAG service

- `GET /`
- `GET /health`
- `POST /ingest`
- `POST /query`
- `GET /stats`
- `DELETE /reset`

## Environment guidance

The default repo guidance is one `.venv` at the root. If you later want isolated environments per service for deployment experiments, treat that as a deliberate local variation rather than the documented default.
