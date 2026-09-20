"""Persisted background jobs.

Every long-running operation (training, batch prediction, benchmark import) is a
``jobs`` row. Execution backend is selected by ``BOTSHIELD_JOB_BACKEND``:

* ``celery`` — tasks are queued on Redis and executed by ``celery -A app.worker``
  worker processes (production).
* ``thread`` — a single in-process worker thread (development, tests, tiny
  single-node deployments).

Handlers receive a :class:`JobContext` and report progress through it; the row
is the single source of truth for the UI (``GET /api/v1/jobs/{id}``).
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import job_id_var, job_log
from app.db.database import SessionLocal
from app.db.models import Job, JobStatus, JobType, User
from ml.utils import json_safe

log = logging.getLogger(__name__)

JobHandler = Callable[["JobContext"], dict[str, Any]]
_HANDLERS: dict[JobType, JobHandler] = {}


def register_handler(job_type: JobType) -> Callable[[JobHandler], JobHandler]:
    def deco(fn: JobHandler) -> JobHandler:
        _HANDLERS[job_type] = fn
        return fn

    return deco


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobContext:
    """Handle passed to job handlers for progress reporting."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self._db = SessionLocal()
        self.job: Job = self._db.get(Job, job_id)  # type: ignore[assignment]
        if self.job is None:
            raise KeyError(job_id)
        self._log: list[str] = json.loads(self.job.log_json or "[]")

    @property
    def db(self) -> Session:
        return self._db

    @property
    def params(self) -> dict[str, Any]:
        return json.loads(self.job.params_json or "{}")

    @property
    def organization_id(self) -> str:
        return self.job.organization_id

    @property
    def user(self) -> User | None:
        return self._db.get(User, self.job.created_by) if self.job.created_by else None

    def progress(self, stage: str, pct: float, message: str) -> None:
        self.job.stage = stage[:60]
        self.job.progress = float(max(0.0, min(1.0, pct)))
        self.job.message = message[:500]
        self._log.append(f"{_now().strftime('%H:%M:%S')} [{stage}] {message}")
        self.job.log_json = json.dumps(self._log[-200:])
        self._db.commit()
        job_log.info(message, extra={"stage": stage, "progress": round(pct, 3)})

    def close(self) -> None:
        self._db.close()


def create_job(db: Session, *, organization_id: str, created_by: str | None, job_type: JobType, params: dict[str, Any], target_type: str = "", target_id: str | None = None) -> Job:
    job = Job(
        organization_id=organization_id,
        created_by=created_by,
        job_type=job_type,
        status=JobStatus.QUEUED,
        stage="queued",
        message="Queued",
        params_json=json.dumps(json_safe(params)),
        target_type=target_type,
        target_id=target_id,
    )
    db.add(job)
    db.commit()
    return job


def execute_job(job_id: str) -> None:
    """Run a job to completion in the current process (called by both backends)."""
    token = job_id_var.set(job_id)
    ctx: JobContext | None = None
    try:
        ctx = JobContext(job_id)
        if ctx.job.status in (JobStatus.COMPLETED, JobStatus.CANCELLED):
            return
        handler = _HANDLERS.get(ctx.job.job_type)
        if handler is None:
            raise RuntimeError(f"No handler registered for job type {ctx.job.job_type}")
        ctx.job.status = JobStatus.PROCESSING
        ctx.job.started_at = _now()
        ctx.job.worker = socket.gethostname()[:120]
        ctx.db.commit()
        result = handler(ctx)
        ctx.job.result_json = json.dumps(json_safe(result))
        ctx.job.status = JobStatus.COMPLETED
        ctx.job.stage = "completed"
        ctx.job.progress = 1.0
        ctx.job.completed_at = _now()
        ctx.db.commit()
        job_log.info("job completed", extra={"job_type": ctx.job.job_type.value})
    except Exception as exc:  # noqa: BLE001 - job boundary
        job_log.error("job failed: %s", exc, extra={"trace": traceback.format_exc()[-2000:]})
        if ctx is not None:
            try:
                ctx.db.rollback()
                ctx.job.status = JobStatus.FAILED
                ctx.job.stage = "failed"
                ctx.job.error = f"{type(exc).__name__}: {exc}"[:2000]
                ctx.job.message = "Failed"
                ctx.job.completed_at = _now()
                ctx.db.commit()
            except Exception:  # noqa: BLE001
                log.exception("could not persist job failure")
    finally:
        if ctx is not None:
            ctx.close()
        job_id_var.reset(token)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _thread_executor() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jobs")
        return _executor


def dispatch(job: Job) -> None:
    s = get_settings()
    if s.job_backend == "celery":
        from app.worker import run_job_task

        run_job_task.delay(job.id)
    else:
        _thread_executor().submit(execute_job, job.id)


def job_public(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "job_type": job.job_type.value,
        "status": job.status.value,
        "stage": job.stage,
        "progress": job.progress,
        "message": job.message,
        "log": json.loads(job.log_json or "[]")[-80:],
        "error": job.error,
        "result": json.loads(job.result_json) if job.result_json else None,
        "target_type": job.target_type,
        "target_id": job.target_id,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }


def mark_interrupted_jobs(db: Session) -> int:
    """On API/worker start-up: jobs left PROCESSING by a crashed process are failed explicitly."""
    n = 0
    for job in db.query(Job).filter(Job.status == JobStatus.PROCESSING).all():
        if get_settings().job_backend == "thread":
            job.status = JobStatus.FAILED
            job.stage = "failed"
            job.error = "Interrupted: the process executing this job stopped before completion"
            job.completed_at = _now()
            n += 1
    if n:
        db.commit()
    return n
