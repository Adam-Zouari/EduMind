@echo off
setlocal

cd /d %~dp0\..\..
call .venv\Scripts\activate.bat

echo Starting Extraction API...
start "EduMind Extraction API" cmd /k "cd /d %cd% && call .venv\Scripts\activate.bat && python -m edumind.cli extraction-api"
timeout /t 2 >nul

echo Starting RAG API...
start "EduMind RAG API" cmd /k "cd /d %cd% && call .venv\Scripts\activate.bat && python -m edumind.cli rag-api"

echo Services launched:
echo   Extraction API: http://127.0.0.1:8000/docs
echo   RAG API: http://127.0.0.1:8001/docs
echo   Primary UI: run `edumind-ui` separately when needed
