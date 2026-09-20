"""Data-retention enforcement.

Runs when ``BOTSHIELD_RETENTION_PREDICTIONS_DAYS`` / ``BOTSHIELD_RETENTION_AUDIT_DAYS`` are set:
once at API start-up and then every 24 h in the background (``schedule_retention``), or on demand
via ``python -m app.cli apply-retention``. Deletions are recorded in the audit log per organization
(actor = system) so purges remain traceable.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import app_log
from app.db.models import AuditLog, Explanation, Organization, Prediction
from app.services import audit

RETENTION_INTERVAL_SECONDS = 24 * 3600


def apply_retention(db: Session) -> dict[str, int]:
    """Delete rows older than the configured retention windows. Returns counts per category."""
    s = get_settings()
    now = datetime.now(timezone.utc)
    deleted = {"predictions": 0, "audit": 0}

    if s.retention_predictions_days:
        cutoff = now - timedelta(days=s.retention_predictions_days)
        org_rows = db.query(Prediction.organization_id, func.count(Prediction.id)).filter(Prediction.created_at < cutoff).group_by(Prediction.organization_id).all()
        for org_id, n in org_rows:
            ids = [r[0] for r in db.query(Prediction.id).filter(Prediction.organization_id == org_id, Prediction.created_at < cutoff).all()]
            if not ids:
                continue
            db.query(Explanation).filter(Explanation.prediction_id.in_(ids)).delete(synchronize_session=False)
            db.query(Prediction).filter(Prediction.id.in_(ids)).delete(synchronize_session=False)
            db.commit()
            deleted["predictions"] += int(n)
            audit.record(db, "retention.predictions_purged", organization_id=org_id, target_type="prediction", outcome="success", details={"older_than_days": s.retention_predictions_days, "deleted": int(n), "actor": "system"})

    if s.retention_audit_days:
        cutoff = now - timedelta(days=s.retention_audit_days)
        n = db.query(AuditLog).filter(AuditLog.created_at < cutoff).delete(synchronize_session=False)
        db.commit()
        deleted["audit"] = int(n or 0)

    db.expire_all()
    if any(deleted.values()):
        app_log.info("retention applied", extra=deleted)
    return deleted


def retention_enabled() -> bool:
    s = get_settings()
    return bool(s.retention_predictions_days or s.retention_audit_days)


async def schedule_retention(session_factory, stop: asyncio.Event) -> None:  # noqa: ANN001
    """Background loop for the API process: apply retention now, then every 24 h until ``stop`` is set."""
    while not stop.is_set():
        db = session_factory()
        try:
            await asyncio.to_thread(apply_retention, db)
        except Exception:  # noqa: BLE001 - never crash the API because of housekeeping
            app_log.exception("retention run failed")
        finally:
            db.close()
        try:
            await asyncio.wait_for(stop.wait(), timeout=RETENTION_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            continue


__all__ = ["apply_retention", "retention_enabled", "schedule_retention", "Organization"]
