@echo off
setlocal

cd /d %~dp0\..\..

echo ========================================
echo EduMind-AI - Complete Installation
echo ========================================
echo.

if not exist .venv (
    python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e .[dev,ui,api,rag,experiments]

echo.
echo Optional OCR stack:
echo   pip install -e .[ocr]
echo.
echo If you use Ollama:
echo   ollama serve
echo   ollama pull qwen3:1.7b
echo.
echo Installation complete.
