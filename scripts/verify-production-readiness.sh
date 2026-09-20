#!/usr/bin/env sh
# Thin wrapper: see scripts/verify_production_readiness.py --help
cd "$(dirname "$0")/.." || exit 1
PY="backend/.venv/bin/python"; [ -x "$PY" ] || PY="python"
exec "$PY" scripts/verify_production_readiness.py "$@"
