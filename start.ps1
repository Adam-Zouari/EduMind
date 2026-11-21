#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

function Invoke-Step {
    param([string]$Program, [string[]]$CommandArgs)
    & $Program @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Program $($CommandArgs -join ' ') failed (exit $LASTEXITCODE)."
    }
}

Push-Location -LiteralPath $PSScriptRoot
try {
    $python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
            throw 'Install Python 3.12 with the Windows py launcher, then run start.ps1 again.'
        }
        & py -3.12 --version
        if ($LASTEXITCODE -ne 0) {
            throw 'Python 3.12 is not available. Install it, then run start.ps1 again.'
        }
    }
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        throw 'Install FFmpeg and put it on PATH, then run start.ps1 again.'
    }
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Install Docker Desktop with Compose, then run start.ps1 again.'
    }
    & docker info --format '{{.ServerVersion}}' | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw 'Start Docker Desktop and wait for its engine to be ready.'
    }
    & docker compose version | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw 'Docker Compose is not available. Check the Docker Desktop installation.'
    }

    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        Invoke-Step 'py' @('-3.12', '-m', 'venv', '.venv')
    }
    & $python -B -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'
    if ($LASTEXITCODE -ne 0) {
        throw 'The existing .venv does not run Python 3.12. Use a Python 3.12 environment.'
    }
    $installStamp = '.venv/.edumind-app-installed'
    $requirementsHash = (Get-FileHash -LiteralPath 'requirements/app.lock').Hash
    if (-not (Test-Path -LiteralPath $installStamp -PathType Leaf) -or
        (Get-Content -LiteralPath $installStamp -Raw).Trim() -ne $requirementsHash) {
        Invoke-Step $python @('-m', 'pip', 'install', '-r', 'requirements/app.lock')
        Invoke-Step $python @('-m', 'pip', 'install', '-e', '.', '--no-deps')
        Invoke-Step $python @('-m', 'pip', 'check')
        Set-Content -LiteralPath $installStamp -Value $requirementsHash -Encoding ASCII
    }

    & $python -B -m edumind.prepare_models --check
    switch ($LASTEXITCODE) {
        0 { }
        2 {
            $answer = Read-Host 'Download the required models now? [y/N]'
            if ($answer -notmatch '^y(es)?$') {
                Write-Host 'Models were not downloaded. Chroma and the app were not started.'
                exit 0
            }
            Invoke-Step $python @('-m', 'edumind.prepare_models')
        }
        default { throw 'App models could not be verified. See the error above.' }
    }

    Invoke-Step 'docker' @('compose', '-f', 'infrastructure/chroma.yml', 'up', '-d')
    Invoke-Step $python @('-c', 'from edumind.rag.vector_store import wait_for_chroma; wait_for_chroma()')
    Invoke-Step $python @('-m', 'streamlit', 'run', 'src/edumind/ui/streamlit_app.py')
}
finally {
    Pop-Location
}
