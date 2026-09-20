"""Audit logging — one row per security-relevant or business-relevant action.

Details are scrubbed of secrets before storage and the same event is emitted on
the ``security`` logger so log shipping picks it up.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import Request
from sqlalchemy.orm import Session

from app.core.logging import request_id_var, security_log
from app.core.security import client_ip
from app.db.models import AuditLog, User

_SENSITIVE = ("password", "token", "secret", "authorization", "cookie")


def _scrub(details: dict[str, Any] | None) -> dict[str, Any]:
    if not details:
        return {}
    return {k: ("***" if any(s in k.lower() for s in _SENSITIVE) else v) for k, v in details.items()}


def record(
    db: Session,
    action: str,
    *,
    actor: User | None = None,
    actor_email: str = "",
    organization_id: str | None = None,
    target_type: str = "",
    target_id: str = "",
    outcome: str = "success",
    details: dict[str, Any] | None = None,
    request: Request | None = None,
    commit: bool = True,
) -> AuditLog:
    row = AuditLog(
        organization_id=organization_id or (actor.organization_id if actor else None),
        actor_user_id=actor.id if actor else None,
        actor_email=(actor.email if actor else actor_email)[:320],
        action=action[:80],
        target_type=target_type[:60],
        target_id=str(target_id)[:120],
        outcome=outcome[:20],
        ip_address=(client_ip(request) if request else "")[:64],
        request_id=request_id_var.get()[:64],
        details_json=json.dumps(_scrub(details), default=str)[:4000],
    )
    db.add(row)
    if commit:
        db.commit()
    security_log.info(
        "audit",
        extra={
            "action": action,
            "outcome": outcome,
            "actor": row.actor_email,
            "organization_id": row.organization_id,
            "target_type": target_type,
            "target_id": target_id,
            "ip": row.ip_address,
        },
    )
    return row
