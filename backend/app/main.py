"""BotShield AI — FastAPI application factory."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Make ``ml`` importable when running ``uvicorn app.main:app`` from backend/
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app import __version__  # noqa: E402
from app.api import datasets, evaluation, explain, health, history, misc, models, predict, train  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.db.database import SessionLocal, init_db  # noqa: E402

log = logging.getLogger("botshield")

API_DESCRIPTION = """
**BotShield AI** — Interpretable AI-Based Social Bot and Fake Follower Detection System.

Implements the methodology of *Javed et al., "Identification of Spambots and Fake Followers on
Social Network via Interpretable AI-Based Machine Learning", IEEE Access 2025
(DOI 10.1109/ACCESS.2025.3551993)*: a 31-feature compact feature set, nine classifiers with
stratified cross-validation, and SHAP + LIME explanations.

* **Risk Score** is an application-level representation of the model's estimated bot probability —
  not a definitive statement that an account is malicious.
* Results labelled *"Reported in base paper"* come from the paper; results labelled
  *"Experiment reproduced by this implementation"* were measured here. They are never mixed.
* Demo datasets/models are synthetic (**DEMO DATA — NOT REAL SOCIAL MEDIA DATA**).
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()
    init_db()
    # Keep the SQLite ``models`` table in sync with the joblib registry on boot.
    from app.services.training_service import sync_model_records

    db = SessionLocal()
    try:
        sync_model_records(db)
    finally:
        db.close()
    # Warm-up: load the active model (pipeline + explainer artefacts) so the
    # first prediction request does not pay the joblib/SHAP start-up cost.
    from app.services.registry import get_predictor
    from ml.predict import ModelNotAvailableError

    try:
        loaded = get_predictor().get()
        log.info("Active model loaded: %s (%s, %d features, demo=%s)", loaded.entry.name, loaded.entry.id, len(loaded.feature_names), loaded.entry.is_demo)
    except ModelNotAvailableError:
        log.warning("No trained model available - run scripts/fetch_datasets.py and scripts/train_model.py")
    except Exception as exc:  # noqa: BLE001 - never block start-up on a bad artefact
        log.exception("Active model could not be loaded: %s", exc)
    log.info("%s v%s started (env=%s, demo_mode=%s)", settings.app_name, __version__, settings.environment, settings.demo_mode_enabled)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=API_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = "/api"
    app.include_router(health.router, prefix=prefix)
    app.include_router(models.router, prefix=prefix)
    app.include_router(predict.router, prefix=prefix)
    app.include_router(datasets.router, prefix=prefix)
    app.include_router(train.router, prefix=prefix)
    app.include_router(evaluation.router, prefix=prefix)
    app.include_router(explain.router, prefix=prefix)
    app.include_router(history.router, prefix=prefix)
    app.include_router(misc.router, prefix=prefix)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail), "code": f"http_{exc.status_code}"})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": [str(p) for p in e.get("loc", [])], "msg": e.get("msg", ""), "type": e.get("type", "")}
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Request validation failed", "code": "validation_error", "errors": errors},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "code": "internal_error"},
        )

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {"app": settings.app_name, "version": __version__, "docs": "/docs", "api": "/api/health"}

    return app


app = create_app()
