"""Application settings (environment variables via pydantic-settings).

All variables are prefixed ``BOTSHIELD_``. Secrets are read from the
environment or ``.env`` on the server only and are never exposed through the
API or bundled into the frontend.
"""

from __future__ import annotations

import re
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent

Environment = Literal["development", "staging", "production", "test"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="BOTSHIELD_",
        extra="ignore",
    )

    # ---- runtime ------------------------------------------------------------ #
    app_name: str = "BotShield AI"
    environment: Environment = "development"
    log_level: str = "INFO"
    log_json: bool | None = None  # default: JSON in staging/production, text otherwise

    # ---- database ------------------------------------------------------------ #
    database_url: str = f"sqlite:///{(BACKEND_ROOT / 'data' / 'botshield.db').as_posix()}"
    # Optional PostgreSQL schema (e.g. "botshield") so the app can share a database with other
    # applications without table-name clashes. Created by "app.cli migrate" if missing.
    db_schema: str | None = None
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # ---- auth ------------------------------------------------------------------ #
    secret_key: str = ""  # REQUIRED in staging/production (JWT signing); auto-generated in dev/test
    access_token_minutes: int = 30
    refresh_token_days: int = 7
    refresh_cookie_name: str = "botshield_refresh"
    cookie_secure: bool | None = None  # default: True in staging/production
    cookie_domain: str | None = None
    # "strict" when the SPA and API share a site; "none" (requires cookie_secure) when they are served
    # from different domains, e.g. app.example.com + api.example.com or two Render services.
    cookie_samesite: Literal["strict", "lax", "none"] = "strict"
    password_min_length: int = 10
    login_rate_limit_per_minute: int = 10
    login_lockout_threshold: int = 8  # failed attempts per email before temporary lock
    login_lockout_minutes: int = 15
    allow_self_signup: bool = False

    # ---- http ------------------------------------------------------------------- #
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    trusted_proxies: str = ""  # comma-separated; enables X-Forwarded-* handling when set (or "*")
    max_request_body_mb: int = 60
    max_upload_mb: int = 50
    max_batch_rows: int = 50_000
    rate_limit_per_minute: int = 120  # general API limit per client
    ml_rate_limit_per_minute: int = 30  # predict / explain / train / batch per user
    docs_enabled: bool | None = None  # default: enabled outside production

    # ---- storage --------------------------------------------------------------- #
    storage_backend: Literal["local", "s3"] = "local"
    storage_local_root: Path = BACKEND_ROOT / "data" / "storage"
    storage_cache_dir: Path = BACKEND_ROOT / "data" / "cache"
    s3_bucket: str | None = None
    s3_endpoint_url: str | None = None
    s3_region: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_prefix: str = "botshield"

    # ---- jobs ------------------------------------------------------------------- #
    job_backend: Literal["thread", "celery"] = "thread"
    redis_url: str | None = None  # required for celery
    celery_task_time_limit_seconds: int = 3600

    # ---- ML ---------------------------------------------------------------------- #
    max_tweets_per_user: int = 100
    training_search_iterations_max: int = 30
    explanation_lime_samples: int = 3000

    # ---- X (Twitter) API v2 adapter -------------------------------------------- #
    x_bearer_token: str | None = None
    x_api_base_url: str = "https://api.x.com/2"
    x_max_tweets: int = 100
    x_timeout_seconds: float = 15.0
    x_cache_ttl_seconds: int = 600

    # ---- retention ---------------------------------------------------------------- #
    retention_predictions_days: int | None = None  # None = keep until deleted by an admin
    retention_audit_days: int | None = None

    # ---- derived -------------------------------------------------------------------- #

    @field_validator("environment", mode="before")
    @classmethod
    def _env_lower(cls, v: object) -> object:
        return v.lower() if isinstance(v, str) else v

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalise_db_url(cls, v: object) -> object:
        # Managed providers (Render, Heroku, Railway) hand out postgres:// or postgresql:// URLs;
        # SQLAlchemy 2 + psycopg3 need the explicit driver.
        if isinstance(v, str):
            for prefix in ("postgres://", "postgresql://"):
                if v.startswith(prefix):
                    return "postgresql+psycopg://" + v[len(prefix):]
        return v

    @field_validator("db_schema", mode="before")
    @classmethod
    def _schema_ident(cls, v: object) -> object:
        if isinstance(v, str):
            v = v.strip() or None
            if v is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v):
                raise ValueError("BOTSHIELD_DB_SCHEMA must be a plain identifier (letters, digits, underscore)")
        return v

    @model_validator(mode="after")
    def _finalise(self) -> "Settings":
        if self.log_json is None:
            self.log_json = self.environment in ("staging", "production")
        if self.cookie_secure is None:
            self.cookie_secure = self.environment in ("staging", "production")
        if self.cookie_samesite == "none" and not self.cookie_secure:
            raise ValueError("BOTSHIELD_COOKIE_SAMESITE=none requires BOTSHIELD_COOKIE_SECURE=true")
        if self.docs_enabled is None:
            self.docs_enabled = self.environment != "production"
        if not self.secret_key:
            if self.environment in ("staging", "production"):
                raise ValueError("BOTSHIELD_SECRET_KEY must be set in staging/production (use: python -c \"import secrets; print(secrets.token_urlsafe(48))\")")
            self.secret_key = secrets.token_urlsafe(48)
        if self.environment in ("staging", "production") and self.database_url.startswith("sqlite"):
            raise ValueError("SQLite is not supported in staging/production; set BOTSHIELD_DATABASE_URL to a PostgreSQL URL")
        if self.job_backend == "celery" and not self.redis_url:
            raise ValueError("BOTSHIELD_REDIS_URL is required when BOTSHIELD_JOB_BACKEND=celery")
        if self.storage_backend == "s3" and not self.s3_bucket:
            raise ValueError("BOTSHIELD_S3_BUCKET is required when BOTSHIELD_STORAGE_BACKEND=s3")
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def trusted_proxy_list(self) -> list[str]:
        return [o.strip() for o in self.trusted_proxies.split(",") if o.strip()]

    @property
    def data_dir(self) -> Path:
        return BACKEND_ROOT / "data"

    @property
    def datasets_dir(self) -> Path:
        """Local benchmark datasets (Cresci) placed by the operator; read-only inputs."""
        return self.data_dir / "datasets"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.storage_local_root, self.storage_cache_dir, self.datasets_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
