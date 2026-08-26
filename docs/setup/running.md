# Run the current EduMind application

[Project overview](../../README.md) · [Documentation map](../README.md) ·
[Complete preparation guide](installation.md) ·
[Application internals](../architecture/ui.md)

This page is the application operations guide. Complete the one-time
[installation and model preparation](installation.md) first.

## 1. Activate the environment

Run from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
```

If `.venv` or `data/benchmarks/models/selected.json` is missing, return to the
[installation guide](installation.md); this page does not duplicate environment
creation or download instructions.

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

## 4. Stop the application dependencies

Stop Streamlit with `Ctrl+C`, then stop Chroma:

```powershell
docker compose -f infrastructure/chroma.yml down
```

This preserves the named Chroma volume. Add `-v` only when you intentionally want
to delete the stored index.

## Common failures

| Symptom | What to check |
|---|---|
| Missing `selected.json` or model directory | Rerun `python experiments/benchmarks/prepare.py app-models` in the activated environment. |
| Chroma unavailable | Start Docker Desktop, run the Compose `up -d` command, then inspect `docker compose -f infrastructure/chroma.yml logs`. |
| Tesseract benchmark error | Tesseract is experiment-only; run `tesseract --list-langs` and confirm `eng` is present before the document configuration benchmark. |
| FFmpeg not found | Install FFmpeg and reopen the terminal so `ffmpeg` is on `PATH`. |
| Python dependency conflicts | Confirm the prompt shows `(.venv)`, then use `python -m pip check`; do not repair a global environment in place. |
| CUDA benchmark failure | The application does not require CUDA. For benchmarks, install the matching PyTorch/CUDA environment and use one explicit common device. |

For model and dataset preparation beyond the application controls, continue with
the [complete guide](installation.md). For internal behavior, read the
[architecture overview](../architecture/overview.md).
