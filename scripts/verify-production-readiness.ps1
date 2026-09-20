# Thin wrapper: see scripts/verify_production_readiness.py --help
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py (Join-Path $root "scripts\verify_production_readiness.py") @args
exit $LASTEXITCODE
