#!/usr/bin/env sh
cd "$(dirname "$0")/.." || exit 1
PY="backend/.venv/bin/python"; [ -x "$PY" ] || PY="python"
exec "$PY" scripts/check_no_demo_data.py "$@"
