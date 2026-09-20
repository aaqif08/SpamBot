"""Authentication, session, users and organisation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.api.deps import AdminUser, ApiRateLimited, CurrentUser
from app.core.config import get_settings
from app.db.database import get_db
from app.db.models import Organization, Role, User, UserStatus
from app.schemas.api import (
    AdminResetPasswordRequest,
    ChangePasswordRequest,
    CreateUserRequest,
    LoginRequest,
    OrganizationPublic,
    PasswordResetConfirm,
    PasswordResetRequest,
    SetupStatus,
    TokenResponse,
    UpdateOrganizationRequest,
    UpdateUserRequest,
    UserPublic,
)
from app.services import auth_service

router = APIRouter(tags=["auth"])


def user_public(u: User) -> UserPublic:
    return UserPublic(
        id=u.id, email=u.email, full_name=u.full_name, role=u.role.value, status=u.status.value,
        organization_id=u.organization_id, organization_name=u.organization.name, last_login_at=u.last_login_at, created_at=u.created_at,
    )


def _set_refresh_cookie(response: Response, token: str, expires_at) -> None:  # noqa: ANN001
    s = get_settings()
    response.set_cookie(
        key=s.refresh_cookie_name,
        value=token,
        httponly=True,
        secure=bool(s.cookie_secure),
        samesite=s.cookie_samesite,
        path="/api/v1/auth",
        domain=s.cookie_domain,
        expires=expires_at,
    )


def _clear_refresh_cookie(response: Response) -> None:
    s = get_settings()
    response.delete_cookie(key=s.refresh_cookie_name, path="/api/v1/auth", domain=s.cookie_domain, secure=bool(s.cookie_secure), samesite=s.cookie_samesite, httponly=True)


@router.get("/auth/setup-status", response_model=SetupStatus)
def setup_status(db: Session = Depends(get_db)) -> SetupStatus:
    return SetupStatus(initialized=auth_service.system_has_users(db), self_signup_enabled=get_settings().allow_self_signup)


@router.post("/auth/login", response_model=TokenResponse, dependencies=[ApiRateLimited])
def login(body: LoginRequest, request: Request, response: Response, db: Session = Depends(get_db)) -> TokenResponse:
    user = auth_service.authenticate(db, body.email, body.password, request)
    access, refresh, expires = auth_service.issue_tokens(db, user, request)
    _set_refresh_cookie(response, refresh, expires)
    return TokenResponse(access_token=access, expires_in=get_settings().access_token_minutes * 60, user=user_public(user))


@router.post("/auth/refresh", response_model=TokenResponse, dependencies=[ApiRateLimited])
def refresh(request: Request, response: Response, db: Session = Depends(get_db)) -> TokenResponse:
    token = request.cookies.get(get_settings().refresh_cookie_name)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No session")
    user, access, new_refresh, expires = auth_service.rotate_refresh(db, token, request)
    _set_refresh_cookie(response, new_refresh, expires)
    return TokenResponse(access_token=access, expires_in=get_settings().access_token_minutes * 60, user=user_public(user))


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response, db: Session = Depends(get_db), user: User = CurrentUser) -> None:
    auth_service.revoke_refresh(db, request.cookies.get(get_settings().refresh_cookie_name), user, request)
    _clear_refresh_cookie(response)


@router.get("/auth/me", response_model=UserPublic)
def me(user: User = CurrentUser) -> UserPublic:
    return user_public(user)


@router.post("/auth/change-password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(body: ChangePasswordRequest, request: Request, response: Response, db: Session = Depends(get_db), user: User = CurrentUser) -> None:
    auth_service.change_password(db, user, body.current_password, body.new_password, request)
    _clear_refresh_cookie(response)


@router.post("/auth/password-reset/request", status_code=status.HTTP_202_ACCEPTED, dependencies=[ApiRateLimited])
def password_reset_request(body: PasswordResetRequest, request: Request, db: Session = Depends(get_db)) -> dict:
    """Creates a one-time reset token. Delivery is out of band: an administrator retrieves it
    from the security log / audit trail (no e-mail integration is bundled). The response is
    identical whether or not the address exists."""
    token = auth_service.create_password_reset(db, body.email, request)
    if token:
        from app.core.logging import security_log

        security_log.info("password reset token issued", extra={"email": body.email, "reset_token": token})
    return {"detail": "If the account exists, a reset token has been issued to the security log."}


@router.post("/auth/password-reset/confirm", status_code=status.HTTP_204_NO_CONTENT, dependencies=[ApiRateLimited])
def password_reset_confirm(body: PasswordResetConfirm, request: Request, db: Session = Depends(get_db)) -> None:
    auth_service.consume_password_reset(db, body.token, body.new_password, request)


# ---- users (ADMIN) ---------------------------------------------------------- #


@router.get("/users", response_model=list[UserPublic])
def list_users(db: Session = Depends(get_db), admin: User = AdminUser) -> list[UserPublic]:
    return [user_public(u) for u in auth_service.list_users(db, admin.organization_id)]


@router.post("/users", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
def create_user(body: CreateUserRequest, request: Request, db: Session = Depends(get_db), admin: User = AdminUser) -> UserPublic:
    return user_public(auth_service.create_user(db, admin, body.email, body.password, Role(body.role), body.full_name, request))


@router.patch("/users/{user_id}", response_model=UserPublic)
def update_user(user_id: str, body: UpdateUserRequest, request: Request, db: Session = Depends(get_db), admin: User = AdminUser) -> UserPublic:
    return user_public(
        auth_service.update_user(
            db, admin, user_id,
            role=Role(body.role) if body.role else None,
            status_value=UserStatus(body.status) if body.status else None,
            full_name=body.full_name,
            request=request,
        )
    )


@router.post("/users/{user_id}/reset-password", status_code=status.HTTP_204_NO_CONTENT)
def admin_reset_password(user_id: str, body: AdminResetPasswordRequest, request: Request, db: Session = Depends(get_db), admin: User = AdminUser) -> None:
    auth_service.admin_reset_password(db, admin, user_id, body.new_password, request)


# ---- organisation ------------------------------------------------------------- #


@router.get("/organization", response_model=OrganizationPublic)
def get_organization(db: Session = Depends(get_db), user: User = CurrentUser) -> OrganizationPublic:
    org = db.get(Organization, user.organization_id)
    return OrganizationPublic(id=org.id, name=org.name, slug=org.slug, created_at=org.created_at)


@router.patch("/organization", response_model=OrganizationPublic)
def update_organization(body: UpdateOrganizationRequest, request: Request, db: Session = Depends(get_db), admin: User = AdminUser) -> OrganizationPublic:
    org = auth_service.update_organization(db, admin, body.name, request)
    return OrganizationPublic(id=org.id, name=org.name, slug=org.slug, created_at=org.created_at)
