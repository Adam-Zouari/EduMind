@echo off
setlocal

cd /d %~dp0\..\..
call .venv\Scripts\activate.bat

python -m compileall src apps services experiments tests
if errorlevel 1 (
    echo Compile smoke test failed.
    exit /b 1
)

echo Compile smoke test passed.
