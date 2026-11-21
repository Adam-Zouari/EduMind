# Run the current EduMind application

[Project overview](../../README.md) · [Documentation map](../README.md) ·
[Model preparation](model-preparation.md) ·
[Application internals](../architecture/ui.md)

Complete the one-time [environment setup](installation.md) and
[app-model preparation](model-preparation.md#preparation-commands) first.

## 1. Activate the environment

Run from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
```

If `.venv` is missing, return to [environment setup](installation.md).
If `data/benchmarks/models/selected.json` is missing, follow
[model preparation](model-preparation.md#preparation-commands).

## 2. Start Chroma

Docker Desktop must already be running.

```powershell
docker compose -f infrastructure/chroma.yml up -d
docker compose -f infrastructure/chroma.yml ps
```

Chroma binds to `127.0.0.1:8001`. Do not start the four-server vector benchmark
Compose project at the same time because its Chroma service uses the same port.

## 3. Start Streamlit

```powershell
streamlit run src/edumind/ui/streamlit_app.py
```

Upload a supported image, PDF, DOCX, audio, or video file, wait for extraction and
indexing, then ask a question. Answers should cite the numbered evidence shown by
the interface.

Check the interface's extraction, vector-index, and generator readiness messages
before uploading. They verify preparation/connectivity, not benchmark quality.
Model operations share a process-wide lock; use one application process and
avoid unrelated GPU workloads when choosing an explicit CUDA profile.

## 4. Stop the application dependencies

Stop Streamlit with `Ctrl+C`, then stop Chroma:

```powershell
docker compose -f infrastructure/chroma.yml down
```

This preserves the named Chroma volume. Add `-v` only when you intentionally want
to delete the stored index.

## Migrating an existing MiniLM index

GTE produces 768-dimensional vectors, so an existing 384-dimensional MiniLM
collection cannot be reused. EduMind keeps the `edumind` collection name and
checks its embedding/chunking fingerprint at startup; it rejects the old index
instead of deleting or silently mixing vectors.

Only if you need to migrate this old index, confirm that you can re-ingest the
original documents and that you intend to delete the app's existing Chroma
data. Stop the app, remove its dedicated volume, then recreate the service:

```powershell
docker compose -f infrastructure/chroma.yml down -v
docker compose -f infrastructure/chroma.yml up -d
```

Re-upload the source documents to
rebuild the index with GTE. No application or benchmark command performs this
destructive migration automatically.

## Common failures

| Symptom | What to check |
|---|---|
| Missing `selected.json` or model directory | Rerun `python -m experiments.benchmarks.prepare app-models` in the activated environment. |
| Chroma unavailable | Start Docker Desktop, run the Compose `up -d` command, then inspect `docker compose -f infrastructure/chroma.yml logs`. |
| Tesseract benchmark error | Tesseract is experiment-only; run `tesseract --list-langs` and confirm `eng` is present before the document configuration benchmark. |
| FFmpeg not found | Install FFmpeg and reopen the terminal so `ffmpeg` is on `PATH`. |
| Python dependency conflicts | Confirm the prompt shows `(.venv)`, then use `python -m pip check`; do not repair a global environment in place. |
| CUDA benchmark failure | The application does not require CUDA. For benchmarks, install the matching PyTorch/CUDA environment and use one explicit common device. |

For benchmark setup beyond the application controls, see the
[benchmark runbook](../benchmarks/running.md). For internal behavior, read the
[architecture overview](../architecture/overview.md).
