@echo off
setlocal

cd /d %~dp0\..\..
call .venv\Scripts\activate.bat

echo Starting OCR API...
start "EduMind OCR API" cmd /k "cd /d %cd% && call .venv\Scripts\activate.bat && python -m edumind.cli ocr-api"
timeout /t 2 >nul

echo Starting RAG API...
start "EduMind RAG API" cmd /k "cd /d %cd% && call .venv\Scripts\activate.bat && python -m edumind.cli rag-api"

echo Services launched:
echo   OCR API: http://localhost:8000/docs
echo   RAG API: http://localhost:8001/docs
echo   Primary UI: run `edumind-ui` separately when needed
