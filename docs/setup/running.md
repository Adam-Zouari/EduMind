# Run the current EduMind application

[Project overview](../../README.md) · [Documentation map](../README.md) ·
[Complete preparation guide](installation.md) ·
[Application internals](../architecture/ui.md)

This page is the short application-only path. Use the complete preparation guide
when installing system tools, all benchmark candidates, datasets, or vector-server
images.

## 1. Prerequisites

The complete application needs Python 3.11, Git, Docker Desktop with Compose,
FFmpeg for audio/video inputs, and enough disk space for the pinned application
models. The provisional generator runs on CPU, so CUDA is optional. Tesseract is
needed by one document-benchmark configuration, not by the current RapidOCR
application profile.

Verify the required external programs:

```powershell
py -3.11 --version
git --version
docker version
docker compose version
ffmpeg -version
```

If any command is missing, follow the
[installation guide](installation.md#1-system-requirements).

## 2. Create the environment

Run these commands from the repository root. Do not install the project into a
global Python environment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements/app.lock
python -m pip install -e . --no-deps
python -m pip check
```

The editable installation makes source changes immediately visible; no wheel
rebuild is needed.

## 3. Prepare application models

Preview the exact downloads:

```powershell
python experiments/benchmarks/prepare.py app-models --dry-run
```

Then download Docling Standard artifacts, Whisper `small.en`, MiniLM, and Hugging
Face Qwen3 1.7B:

```powershell
python experiments/benchmarks/prepare.py app-models
```

Models are stored under `data/benchmarks/downloads/`; the command generates
`data/benchmarks/models/selected.json`. Interrupted Hugging Face downloads can be
rerun and resume through their local transfer metadata.

## 4. Start Chroma

Docker Desktop must already be running.

```powershell
docker compose -f infrastructure/chroma.yml up -d
docker compose -f infrastructure/chroma.yml ps
```

Chroma binds to `127.0.0.1:8001`. Do not start the four-server vector benchmark
Compose project at the same time because its Chroma service uses the same port.

## 5. Start Streamlit

```powershell
streamlit run src/edumind/ui/streamlit_app.py
```

Upload a supported image, PDF, DOCX, audio, or video file, wait for extraction and
indexing, then ask a question. Answers should cite the numbered evidence shown by
the interface.

## 6. Stop the application dependencies

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
