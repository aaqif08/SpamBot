"""Shared bootstrap for CLI scripts: puts ``backend/`` on sys.path and initialises the DB."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

warnings.filterwarnings("ignore")

# Windows consoles default to a legacy code page; force UTF-8 so labels print.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def init() -> None:
    from app.core.config import get_settings
    from app.core.logging import configure_logging
    from app.db.database import init_db

    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()
    init_db()
