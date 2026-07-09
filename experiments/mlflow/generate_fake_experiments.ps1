Set-Location (Join-Path $PSScriptRoot "..\\..")
& ".venv\\Scripts\\Activate.ps1"
python -m experiments.mlflow.generate_fake_experiments $args
