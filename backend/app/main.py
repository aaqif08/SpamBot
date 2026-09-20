"""BotShield AI — FastAPI application factory (production)."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app import __version__  # noqa: E402
from app.api.v1 import analyses, auth, datasets, models, system  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import app_log, configure_logging, request_id_var, user_id_var  # noqa: E402
from app.db.database import SessionLocal, check_connection, current_migration_revision  # noqa: E402

log = logging.getLogger("app.http")

API_DESCRIPTION = """
**BotShield AI** — ML-based social bot and fake-follower detection with explainable predictions (SHAP, LIME).

Methodology follows *Javed et al., IEEE Access 2025 (DOI 10.1109/ACCESS.2025.3551993)*.

* All endpoints under `/api/v1` except `/api/v1/health*`, `/api/v1/auth/login`, `/api/v1/auth/refresh`,
  `/api/v1/auth/setup-status`, `/api/v1/auth/password-reset/*` and `/api/v1/research` require a bearer access token.
* Roles: **ADMIN** (users, models, settings), **ANALYST** (analyses, datasets, training), **VIEWER** (read-only).
* Every record is scoped to the caller's organisation.
* Predictions are model outputs (estimated probabilities), not verified facts about an account.
"""


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Request id + access log with latency; security headers; body-size guard."""

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        s = get_settings()
        rid = request.headers.get("x-request-id", "")[:64] or uuid.uuid4().hex
        token = request_id_var.set(rid)
        user_id_var.set("-")
        start = time.perf_counter()
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > s.max_request_body_mb * 1024 * 1024:
            return JSONResponse(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, content={"detail": "Request body too large", "code": "http_413"}, headers={"X-Request-ID": rid})
        try:
            response = await call_next(request)
            latency_ms = (time.perf_counter() - start) * 1000
            if not request.url.path.startswith("/api/v1/health"):
                log.info("request", extra={"method": request.method, "path": request.url.path, "status": response.status_code, "latency_ms": round(latency_ms, 1), "user_id": getattr(getattr(request.state, "user", None), "id", None)})
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if s.cookie_secure:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    configure_logging(s.log_level, bool(s.log_json))
    s.ensure_dirs()
    if not check_connection():
        app_log.error("database is not reachable (BOTSHIELD_DATABASE_URL)")
    else:
        rev = current_migration_revision()
        if rev is None:
            msg = "database schema is not migrated - run: alembic upgrade head"
            if s.is_production:
                raise RuntimeError(msg)
            app_log.warning(msg)
        else:
            app_log.info("database ready", extra={"revision": rev})
        from app.services import handlers  # noqa: F401  (register job handlers)
        from app.services.jobs import mark_interrupted_jobs

        db = SessionLocal()
        try:
            if rev is not None:
                n = mark_interrupted_jobs(db)
                if n:
                    app_log.warning("marked %d interrupted job(s) as failed", n)
        finally:
            db.close()
    app_log.info("%s v%s started", s.app_name, __version__, extra={"environment": s.environment, "job_backend": s.job_backend, "storage": s.storage_backend})
    stop = asyncio.Event()
    retention_task = None
    from app.services.retention import retention_enabled, schedule_retention

    if retention_enabled() and check_connection():
        retention_task = asyncio.create_task(schedule_retention(SessionLocal, stop))
    try:
        yield
    finally:
        stop.set()
        if retention_task is not None:
            retention_task.cancel()


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title=s.app_name,
        version=__version__,
        description=API_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs" if s.docs_enabled else None,
        redoc_url="/redoc" if s.docs_enabled else None,
        openapi_url="/openapi.json" if s.docs_enabled else None,
        swagger_ui_parameters={"persistAuthorization": True},
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    if s.trusted_proxy_list:
        from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

        app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*" if "*" in s.trusted_proxy_list else s.trusted_proxy_list)

    prefix = "/api/v1"
    for r in (system.router, auth.router, analyses.router, datasets.router, models.router):
        app.include_router(r, prefix=prefix)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, (str, dict, list)) else str(exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"detail": detail, "code": f"http_{exc.status_code}"}, headers=getattr(exc, "headers", None) or None)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": [str(p) for p in e.get("loc", [])], "msg": e.get("msg", ""), "type": e.get("type", "")} for e in exc.errors()]
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": "Request validation failed", "code": "validation_error", "errors": errors})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error: %s", exc)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "Internal server error", "code": "internal_error", "request_id": request_id_var.get()})

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {"app": s.app_name, "version": __version__, "api": "/api/v1/health"}

    @app.get("/health", include_in_schema=False)
    async def root_health() -> dict:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
