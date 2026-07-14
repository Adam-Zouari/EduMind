# Start here

Use Python 3.10 or newer on the target Windows machine; Python 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ui,api,rag,extraction,asr,benchmarks]"
edumind benchmark preflight
pytest
edumind benchmark all
```

The last command is a non-authoritative, network-free smoke suite. Standard runs require explicit dataset/model preparation and local tools such as Tesseract, FFmpeg, and Ollama. Preflight reports missing requirements as failures; it does not silently remove candidates.

Configuration ships in the package. Put local secrets/settings in `.env` (never commit it), set `EDUMIND_CONFIG` to a YAML override, or pass typed overrides in Python.
