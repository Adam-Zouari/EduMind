# Start Here

This is the fastest path to a working local EduMind-AI setup.

## 1. Create one root environment

EduMind-AI now uses a single project environment named `.venv`.

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

## 2. Copy local environment defaults

```bash
cp .env.example .env
```

Windows:

```powershell
Copy-Item .env.example .env
```

## 3. Prepare optional local services

You only need these if your workflow uses them:

- Ollama for answer generation
- Tesseract for image OCR
- FFmpeg for audio and video extraction

Start Ollama and pull the default model:

```bash
ollama serve
ollama pull qwen3:1.7b
```

## 4. Run the default product path

The main demo mode is the direct Streamlit app:

```bash
python -m edumind.cli ui
```

Open `http://localhost:8501`.

## 5. Use optional service mode only when needed

Microservices are still available, but they are no longer the default setup.

```bash
python -m edumind.cli ocr-api
python -m edumind.cli rag-api
python -m edumind.cli ui-microservices
```

Windows helper:

```powershell
scripts\windows\start_all_services.bat
```

## What to read next

- `docs/setup/RUN_INSTRUCTIONS.md` for the full command matrix
- `docs/setup/QUICK_REFERENCE.md` for a compact cheat sheet
- `docs/architecture/TECHNICAL_DOCUMENTATION.md` for the package layout and system flow
