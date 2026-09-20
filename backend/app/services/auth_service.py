"""Authentication, sessions and user administration."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException, Request, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import security_log
from app.core.security import (
    SlidingWindowLimiter,
    client_ip,
    create_access_token,
    hash_password,
    hash_token,
    new_opaque_token,
    validate_password_strength,
    verify_password,
)
from app.db.models import Organization, PasswordResetToken, RefreshSession, Role, User, UserStatus
from app.services import audit

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_login_limiter = SlidingWindowLimiter(get_settings().login_rate_limit_per_minute, 60.0)


class AuthError(HTTPException):
    def __init__(self, detail: str = "Invalid email or password", status_code: int = status.HTTP_401_UNAUTHORIZED) -> None:
        super().__init__(status_code=status_code, detail=detail)


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return slug[:80] or "org"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Bootstrap / organisations
# --------------------------------------------------------------------------- #


def create_organization_with_admin(db: Session, org_name: str, email: str, password: str, full_name: str = "") -> tuple[Organization, User]:
    """Explicit initialisation step (``create-admin`` CLI or first-run setup)."""
    email = email.strip().lower()
    problems = validate_password_strength(password)
    if problems:
        raise ValueError("Password must contain " + ", ".join(problems))
    if db.query(User).filter(func.lower(User.email) == email).first():
        raise ValueError("A user with this email already exists")
    slug = slugify(org_name)
    base = slug
    n = 1
    while db.query(Organization).filter(Organization.slug == slug).first():
        n += 1
        slug = f"{base}-{n}"
    org = Organization(name=org_name.strip()[:200], slug=slug)
    db.add(org)
    db.flush()
    user = User(
        organization_id=org.id,
        email=email,
        full_name=full_name.strip()[:200],
        password_hash=hash_password(password),
        role=Role.ADMIN,
        status=UserStatus.ACTIVE,
        password_changed_at=_now(),
    )
    db.add(user)
    db.commit()
    audit.record(db, "organization.created", actor=user, target_type="organization", target_id=org.id, details={"slug": slug})
    audit.record(db, "user.created", actor=user, target_type="user", target_id=user.id, details={"role": "ADMIN", "bootstrap": True})
    return org, user


def system_has_users(db: Session) -> bool:
    return db.query(User.id).first() is not None


# --------------------------------------------------------------------------- #
# Login / sessions
# --------------------------------------------------------------------------- #


def authenticate(db: Session, email: str, password: str, request: Request | None = None) -> User:
    s = get_settings()
    email = email.strip().lower()
    ip = client_ip(request) if request else "unknown"
    allowed, retry = _login_limiter.hit(f"login:{ip}")
    if not allowed:
        security_log.warning("login rate limited", extra={"ip": ip})
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many login attempts; try again later", headers={"Retry-After": str(retry)})

    user = db.query(User).filter(func.lower(User.email) == email).first()
    if user is None:
        # constant-ish time: still hash to avoid trivial user enumeration by timing
        verify_password(password, "$2b$12$C6UzMDM.H6dfI/f/IKcEeO5t3G9rXv8ns3Ff7ObRbbc0DeH/xFj0C")
        audit.record(db, "auth.login_failed", actor_email=email, outcome="failure", details={"reason": "unknown_user"}, request=request)
        raise AuthError()
    if user.locked_until and user.locked_until > _now():
        audit.record(db, "auth.login_failed", actor=user, outcome="failure", details={"reason": "locked"}, request=request)
        raise AuthError("Account temporarily locked after repeated failed logins", status.HTTP_423_LOCKED)
    if not verify_password(password, user.password_hash):
        user.failed_login_count += 1
        if user.failed_login_count >= s.login_lockout_threshold:
            user.locked_until = _now() + timedelta(minutes=s.login_lockout_minutes)
            user.failed_login_count = 0
            security_log.warning("account locked", extra={"user_id": user.id})
        db.commit()
        audit.record(db, "auth.login_failed", actor=user, outcome="failure", details={"reason": "bad_password"}, request=request)
        raise AuthError()
    if user.status != UserStatus.ACTIVE:
        audit.record(db, "auth.login_failed", actor=user, outcome="failure", details={"reason": "disabled"}, request=request)
        raise AuthError("Account is disabled", status.HTTP_403_FORBIDDEN)
    if not user.organization.is_active:
        raise AuthError("Organization is inactive", status.HTTP_403_FORBIDDEN)
    user.failed_login_count = 0
    user.locked_until = None
    user.last_login_at = _now()
    db.commit()
    _login_limiter.reset(f"login:{ip}")
    audit.record(db, "auth.login", actor=user, request=request)
    return user


def issue_tokens(db: Session, user: User, request: Request | None = None) -> tuple[str, str, datetime]:
    """Return (access_token, refresh_token, refresh_expires_at)."""
    s = get_settings()
    access = create_access_token(user.id, user.organization_id, user.role.value)
    refresh = new_opaque_token()
    expires = _now() + timedelta(days=s.refresh_token_days)
    db.add(
        RefreshSession(
            user_id=user.id,
            token_hash=hash_token(refresh),
            user_agent=(request.headers.get("user-agent", "") if request else "")[:300],
            ip_address=(client_ip(request) if request else "")[:64],
            expires_at=expires,
        )
    )
    db.commit()
    return access, refresh, expires


def rotate_refresh(db: Session, refresh_token: str, request: Request | None = None) -> tuple[User, str, str, datetime]:
    """Validate a refresh token, revoke it and issue a new pair (rotation)."""
    row = db.query(RefreshSession).filter(RefreshSession.token_hash == hash_token(refresh_token)).first()
    if row is None or row.revoked_at is not None or row.expires_at <= _now():
        raise AuthError("Session expired")
    user = db.get(User, row.user_id)
    if user is None or user.status != UserStatus.ACTIVE:
        raise AuthError("Session expired")
    row.revoked_at = _now()
    db.commit()
    access, refresh, expires = issue_tokens(db, user, request)
    return user, access, refresh, expires


def revoke_refresh(db: Session, refresh_token: str | None, user: User | None = None, request: Request | None = None) -> None:
    if refresh_token:
        row = db.query(RefreshSession).filter(RefreshSession.token_hash == hash_token(refresh_token)).first()
        if row and row.revoked_at is None:
            row.revoked_at = _now()
            db.commit()
    if user is not None:
        audit.record(db, "auth.logout", actor=user, request=request)


def revoke_all_sessions(db: Session, user_id: str) -> int:
    n = 0
    for row in db.query(RefreshSession).filter(RefreshSession.user_id == user_id, RefreshSession.revoked_at.is_(None)).all():
        row.revoked_at = _now()
        n += 1
    db.commit()
    return n


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #


def change_password(db: Session, user: User, current_password: str, new_password: str, request: Request | None = None) -> None:
    if not verify_password(current_password, user.password_hash):
        audit.record(db, "auth.password_change_failed", actor=user, outcome="failure", request=request)
        raise AuthError("Current password is incorrect", status.HTTP_400_BAD_REQUEST)
    problems = validate_password_strength(new_password)
    if problems:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Password must contain " + ", ".join(problems))
    user.password_hash = hash_password(new_password)
    user.password_changed_at = _now()
    db.commit()
    revoke_all_sessions(db, user.id)
    audit.record(db, "auth.password_changed", actor=user, request=request)


def create_password_reset(db: Session, email: str, request: Request | None = None) -> str | None:
    """Create a one-time reset token (returned to the caller for delivery by an
    out-of-band channel such as an admin or an e-mail integration). Returns
    ``None`` for unknown users without revealing that."""
    user = db.query(User).filter(func.lower(User.email) == email.strip().lower()).first()
    if user is None:
        return None
    token = new_opaque_token()
    db.add(PasswordResetToken(user_id=user.id, token_hash=hash_token(token), expires_at=_now() + timedelta(hours=2)))
    db.commit()
    audit.record(db, "auth.password_reset_requested", actor=user, request=request)
    return token


def consume_password_reset(db: Session, token: str, new_password: str, request: Request | None = None) -> User:
    row = db.query(PasswordResetToken).filter(PasswordResetToken.token_hash == hash_token(token)).first()
    if row is None or row.used_at is not None or row.expires_at <= _now():
        raise AuthError("Reset token is invalid or expired", status.HTTP_400_BAD_REQUEST)
    problems = validate_password_strength(new_password)
    if problems:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Password must contain " + ", ".join(problems))
    user = db.get(User, row.user_id)
    if user is None:
        raise AuthError("Reset token is invalid or expired", status.HTTP_400_BAD_REQUEST)
    user.password_hash = hash_password(new_password)
    user.password_changed_at = _now()
    user.failed_login_count = 0
    user.locked_until = None
    row.used_at = _now()
    db.commit()
    revoke_all_sessions(db, user.id)
    audit.record(db, "auth.password_reset_completed", actor=user, request=request)
    return user


# --------------------------------------------------------------------------- #
# User administration (ADMIN)
# --------------------------------------------------------------------------- #


def list_users(db: Session, organization_id: str) -> list[User]:
    return db.query(User).filter(User.organization_id == organization_id).order_by(User.created_at.asc()).all()


def create_user(db: Session, admin: User, email: str, password: str, role: Role, full_name: str = "", request: Request | None = None) -> User:
    email = email.strip().lower()
    problems = validate_password_strength(password)
    if problems:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Password must contain " + ", ".join(problems))
    if db.query(User).filter(func.lower(User.email) == email).first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A user with this email already exists")
    user = User(
        organization_id=admin.organization_id,
        email=email,
        full_name=full_name.strip()[:200],
        password_hash=hash_password(password),
        role=role,
        status=UserStatus.ACTIVE,
        password_changed_at=_now(),
    )
    db.add(user)
    db.commit()
    audit.record(db, "user.created", actor=admin, target_type="user", target_id=user.id, details={"role": role.value, "email": email}, request=request)
    return user


def update_user(db: Session, admin: User, user_id: str, *, role: Role | None = None, status_value: UserStatus | None = None, full_name: str | None = None, request: Request | None = None) -> User:
    user = db.query(User).filter(User.id == user_id, User.organization_id == admin.organization_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    changes: dict[str, Any] = {}
    if role is not None and role != user.role:
        if user.id == admin.id and role != Role.ADMIN:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot remove your own administrator role")
        user.role = role
        changes["role"] = role.value
    if status_value is not None and status_value != user.status:
        if user.id == admin.id and status_value != UserStatus.ACTIVE:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot disable your own account")
        user.status = status_value
        changes["status"] = status_value.value
        if status_value == UserStatus.DISABLED:
            revoke_all_sessions(db, user.id)
    if full_name is not None:
        user.full_name = full_name.strip()[:200]
        changes["full_name"] = user.full_name
    if role is not None or status_value is not None:
        admins_left = db.query(User).filter(User.organization_id == admin.organization_id, User.role == Role.ADMIN, User.status == UserStatus.ACTIVE).count()
        if admins_left == 0:
            db.rollback()
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="An organization must keep at least one active administrator")
    db.commit()
    audit.record(db, "user.updated", actor=admin, target_type="user", target_id=user.id, details=changes, request=request)
    return user


def admin_reset_password(db: Session, admin: User, user_id: str, new_password: str, request: Request | None = None) -> User:
    user = db.query(User).filter(User.id == user_id, User.organization_id == admin.organization_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    problems = validate_password_strength(new_password)
    if problems:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Password must contain " + ", ".join(problems))
    user.password_hash = hash_password(new_password)
    user.password_changed_at = _now()
    user.failed_login_count = 0
    user.locked_until = None
    db.commit()
    revoke_all_sessions(db, user.id)
    audit.record(db, "user.password_reset_by_admin", actor=admin, target_type="user", target_id=user.id, request=request)
    return user


def update_organization(db: Session, admin: User, name: str, request: Request | None = None) -> Organization:
    org = db.get(Organization, admin.organization_id)
    if org is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    org.name = name.strip()[:200]
    db.commit()
    audit.record(db, "organization.updated", actor=admin, target_type="organization", target_id=org.id, details={"name": org.name}, request=request)
    return org
