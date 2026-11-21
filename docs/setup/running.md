# Run the current EduMind application

[Project overview](../../README.md) · [Documentation map](../README.md) ·
[Model preparation](model-preparation.md) ·
[Application internals](../architecture/ui.md)

## Start with the Windows launcher

Install the [system requirements](installation.md#1-system-requirements), start
Docker Desktop, and run this from the extracted release folder in PowerShell:

```powershell
.\start.ps1
```

The launcher sets up `.venv` if needed and asks before downloading missing
models. Declining or a failed preparation leaves Chroma and Streamlit
unstarted. The launcher waits up to 60 seconds for Chroma to answer before it
starts Streamlit. On later launches it verifies the local models and reuses the
environment without reinstalling unchanged requirements. Use the individual
commands below if you prefer to prepare or start each part yourself.

## Start manually

Complete the one-time [environment setup](installation.md) and
[app-model preparation](model-preparation.md#preparation-commands) first.

## 1. Activate the environment

Run from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
```

If `.venv` is missing, return to [environment setup](installation.md).
If `data/models/app.json` is missing, follow
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

## Migrating an existing index

GTE produces 768-dimensional vectors, so an existing 384-dimensional MiniLM
collection cannot be reused. EduMind keeps the `edumind` collection name and
checks its embedding/chunking fingerprint at startup; it rejects the old index
instead of deleting or silently mixing vectors.

An index built with the same GTE revision but a different snapshot path
also has a different fingerprint. If you need to keep the old index, set
a new `vector.collection_name` in a copy of `config/base.yaml` and point
`EDUMIND_CONFIG` at that file. Re-upload the source documents into the new
collection.

Only if you intend to delete the old data and can re-ingest the original
documents, stop the app, remove its dedicated volume, and recreate the service:

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
| Missing `app.json` or model directory | Run `python -m edumind.prepare_models` in the activated environment. Check the reported path before replacing any damaged files. |
| Chroma unavailable | Start Docker Desktop, run the Compose `up -d` command, then inspect `docker compose -f infrastructure/chroma.yml logs`. |
| FFmpeg not found | Install FFmpeg and reopen the terminal so `ffmpeg` is on `PATH`. |
| Python dependency conflicts | Confirm the prompt shows `(.venv)`, then use `python -m pip check`; do not repair a global environment in place. |

For benchmark setup beyond the application controls, see the
[benchmark runbook](../benchmarks/running.md). For internal behavior, read the
[architecture overview](../architecture/overview.md).
