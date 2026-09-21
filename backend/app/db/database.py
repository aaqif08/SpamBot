"""SQLAlchemy engine / session.

* PostgreSQL in staging/production, SQLite for local development and tests
  (enforced by ``Settings``).
* Schema is managed by Alembic migrations (``alembic upgrade head``). The only
  place that calls ``create_all`` is the test-suite helper ``create_schema``.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


def build_engine(url: str | None = None) -> Engine:
    settings = get_settings()
    url = url or settings.database_url
    if url.startswith("sqlite"):
        eng = create_engine(url, connect_args={"check_same_thread": False}, pool_pre_ping=True)

        @event.listens_for(eng, "connect")
        def _sqlite_pragmas(dbapi_connection, _record) -> None:  # pragma: no cover - driver hook
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return eng
    connect_args: dict = {}
    if settings.db_schema and url.startswith("postgresql"):
        # Every connection works inside the dedicated schema (tables, enums, alembic_version).
        connect_args["options"] = f"-csearch_path={settings.db_schema}"
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        connect_args=connect_args,
    )


def ensure_schema(eng: Engine | None = None) -> None:
    """Create the configured PostgreSQL schema if it does not exist (no-op for SQLite / unset)."""
    settings = get_settings()
    if not settings.db_schema or not settings.database_url.startswith("postgresql"):
        return
    from sqlalchemy import text

    base = create_engine(settings.database_url, pool_pre_ping=True)
    with base.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{settings.db_schema}"'))
    base.dispose()


engine: Engine = build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_schema(target: Engine | None = None) -> None:
    """Create all tables directly (tests and local tooling only — production uses Alembic)."""
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=target or engine)


def check_connection(target: Engine | None = None) -> bool:
    try:
        with (target or engine).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


def current_migration_revision(target: Engine | None = None) -> str | None:
    try:
        with (target or engine).connect() as conn:
            row = conn.execute(text("SELECT version_num FROM alembic_version")).fetchone()
            return row[0] if row else None
    except Exception:  # noqa: BLE001
        return None
