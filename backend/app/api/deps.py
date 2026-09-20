"""Shared FastAPI dependencies: authentication, RBAC, rate limits."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import user_id_var
from app.core.security import SlidingWindowLimiter, client_ip, decode_access_token, raise_rate_limited
from app.db.database import get_db
from app.db.models import Role, User, UserStatus

_bearer = HTTPBearer(auto_error=False)
_settings = get_settings()
_api_limiter = SlidingWindowLimiter(_settings.rate_limit_per_minute, 60.0)
_ml_limiter = SlidingWindowLimiter(_settings.ml_rate_limit_per_minute, 60.0)


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required", headers={"WWW-Authenticate": "Bearer"})
    payload = decode_access_token(credentials.credentials)
    user = db.get(User, payload["sub"])
    if user is None or user.status != UserStatus.ACTIVE or user.organization_id != payload.get("org"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})
    if user.password_changed_at and payload.get("iat") and int(user.password_changed_at.timestamp()) > int(payload["iat"]) + 1:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired", headers={"WWW-Authenticate": "Bearer"})
    user_id_var.set(user.id)
    request.state.user = user
    return user


def require_roles(*roles: Role) -> Callable[..., User]:
    allowed = set(roles)

    def _dep(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not have permission to perform this action")
        return user

    return _dep


CurrentUser = Depends(get_current_user)
AdminUser = Depends(require_roles(Role.ADMIN))
AnalystUser = Depends(require_roles(Role.ADMIN, Role.ANALYST))
AnyUser = Depends(require_roles(Role.ADMIN, Role.ANALYST, Role.VIEWER))


def api_rate_limit(request: Request) -> None:
    ok, retry = _api_limiter.hit(f"api:{client_ip(request)}")
    if not ok:
        raise_rate_limited(retry)


def ml_rate_limit(request: Request, user: User = Depends(get_current_user)) -> None:
    """Expensive endpoints (predict, explain, train, batch): per-user limit."""
    ok, retry = _ml_limiter.hit(f"ml:{user.id}")
    if not ok:
        raise_rate_limited(retry)


ApiRateLimited = Depends(api_rate_limit)
MlRateLimited = Depends(ml_rate_limit)
