"""Celery worker entry point.

    celery -A app.worker worker --loglevel=info --concurrency=1

Requires ``BOTSHIELD_JOB_BACKEND=celery`` and ``BOTSHIELD_REDIS_URL``.
Each task executes one persisted job through :func:`app.services.jobs.execute_job`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from celery import Celery

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402

settings = get_settings()
configure_logging(settings.log_level, settings.log_json or False)

celery_app = Celery("botshield", broker=settings.redis_url or "memory://", backend=None)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_time_limit=settings.celery_task_time_limit_seconds,
    task_soft_time_limit=max(60, settings.celery_task_time_limit_seconds - 60),
    broker_connection_retry_on_startup=True,
    worker_hijack_root_logger=False,
)


@celery_app.task(name="botshield.run_job", bind=True, max_retries=0)
def run_job_task(self, job_id: str) -> None:  # noqa: ANN001
    # Handlers are registered on import.
    from app.services import handlers  # noqa: F401
    from app.services.jobs import execute_job

    execute_job(job_id)
