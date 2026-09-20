"""Health/readiness, runtime info, dashboard, jobs, providers, audit, research."""

from __future__ import annotations

import json
import platform
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as pkg_version

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app import __version__
from app.api.deps import AdminUser, AnyUser, MlRateLimited
from app.core.config import get_settings
from app.core.storage import get_storage
from app.db.database import check_connection, current_migration_revision, get_db
from app.db.models import AuditLog, Job, User
from app.schemas.api import AuditResponse, DashboardResponse, HealthResponse, JobResponse, ProviderFetchRequest, ProviderInfo, ReadinessResponse, ResearchResponse, RuntimeInfo
from app.services import audit
from app.services.dashboard_service import analytics
from app.services.jobs import job_public
from app.services.model_service import ModelService
from app.services.providers import PROVIDERS, XApiError, list_providers
from ml import FEATURE_VERSION
from ml.features import FEATURE_DESCRIPTIONS, FEATURE_GROUPS, FEATURE_NAMES
from ml.paper_results import PAPER_CITATION, PAPER_REPORTED_RESULTS
from ml.train import MODEL_ZOO

router = APIRouter(tags=["system"])


def _v(name: str) -> str:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return "missing"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: the process is up. Does not touch the database."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/health/ready", response_model=ReadinessResponse)
def ready() -> ReadinessResponse:
    """Readiness: database reachable + migrated, storage reachable, job backend configured."""
    s = get_settings()
    checks: dict[str, object] = {}
    db_ok = check_connection()
    checks["database"] = {"ok": db_ok, "revision": current_migration_revision() if db_ok else None}
    try:
        storage = get_storage()
        checks["storage"] = {"ok": True, **storage.describe()}
    except Exception as exc:  # noqa: BLE001
        checks["storage"] = {"ok": False, "error": type(exc).__name__}
    queue_ok = True
    if s.job_backend == "celery":
        try:
            import redis

            redis.Redis.from_url(s.redis_url or "").ping()
        except Exception as exc:  # noqa: BLE001
            queue_ok = False
            checks["queue"] = {"ok": False, "backend": "celery", "error": type(exc).__name__}
    if queue_ok:
        checks["queue"] = {"ok": True, "backend": s.job_backend}
    overall = db_ok and bool(checks["storage"]["ok"]) and queue_ok and checks["database"]["revision"] is not None  # type: ignore[index]
    if not overall:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"status": "not ready", "checks": checks})
    return ReadinessResponse(status="ready", version=__version__, environment=s.environment, checks=checks)


@router.get("/runtime", response_model=RuntimeInfo)
def runtime(db: Session = Depends(get_db), user: User = AnyUser) -> RuntimeInfo:
    s = get_settings()
    prod = ModelService(db).production(user.organization_id)
    return RuntimeInfo(
        version=__version__, environment=s.environment, feature_version=FEATURE_VERSION, n_features=len(FEATURE_NAMES),
        job_backend=s.job_backend, storage_backend=s.storage_backend, python=platform.python_version(),
        libraries={k: _v(k) for k in ("scikit-learn", "shap", "lime", "xgboost", "lightgbm", "pandas", "numpy", "fastapi")},
        production_model={"id": prod.id, "name": prod.name, "version": prod.version, "algorithm": prod.algorithm, "n_features": prod.n_features} if prod else None,
    )


@router.get("/dashboard", response_model=DashboardResponse)
def dashboard(db: Session = Depends(get_db), user: User = AnyUser) -> DashboardResponse:
    return DashboardResponse(**analytics(db, user.organization_id))


# ---- jobs ------------------------------------------------------------------ #


@router.get("/jobs", response_model=list[JobResponse])
def list_jobs(db: Session = Depends(get_db), user: User = AnyUser, job_type: str | None = None, limit: int = Query(default=50, ge=1, le=200)) -> list[JobResponse]:
    q = db.query(Job).filter(Job.organization_id == user.organization_id)
    if job_type:
        q = q.filter(Job.job_type == job_type)
    return [JobResponse(**job_public(j)) for j in q.order_by(Job.created_at.desc()).limit(limit).all()]


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, db: Session = Depends(get_db), user: User = AnyUser) -> JobResponse:
    j = db.query(Job).filter(Job.id == job_id, Job.organization_id == user.organization_id).first()
    if j is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return JobResponse(**job_public(j))


# ---- providers -------------------------------------------------------------- #


@router.get("/providers", response_model=list[ProviderInfo])
def providers(user: User = AnyUser) -> list[ProviderInfo]:
    return [ProviderInfo(**p) for p in list_providers()]


@router.post("/providers/{name}/fetch", dependencies=[MlRateLimited])
def provider_fetch(name: str, body: ProviderFetchRequest, request: Request, db: Session = Depends(get_db), user: User = AnyUser) -> dict:
    provider = PROVIDERS.get(name)
    if provider is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown provider")
    if not provider.configured():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="External data integration is not configured")
    try:
        payload = provider.fetch_account(body.identifier)
    except XApiError as exc:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        raise HTTPException(status_code=exc.status, detail=str(exc), headers=headers)
    except NotImplementedError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    audit.record(db, "provider.fetch", actor=user, target_type="provider", target_id=name, details={"identifier": body.identifier}, request=request)
    return payload


# ---- audit ------------------------------------------------------------------- #


@router.get("/audit", response_model=AuditResponse)
def audit_log(
    db: Session = Depends(get_db),
    admin: User = AdminUser,
    action: str | None = Query(default=None, max_length=80),
    actor: str | None = Query(default=None, max_length=320),
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
) -> AuditResponse:
    q = db.query(AuditLog).filter(AuditLog.organization_id == admin.organization_id)
    if action:
        q = q.filter(AuditLog.action.like(f"{action}%"))
    if actor:
        q = q.filter(AuditLog.actor_email.ilike(f"%{actor}%"))
    if date_from:
        q = q.filter(AuditLog.created_at >= date_from)
    if date_to:
        q = q.filter(AuditLog.created_at <= date_to)
    total = q.count()
    rows = q.order_by(AuditLog.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return AuditResponse(
        items=[
            {"id": r.id, "action": r.action, "actor_email": r.actor_email, "target_type": r.target_type, "target_id": r.target_id, "outcome": r.outcome, "ip_address": r.ip_address, "details": json.loads(r.details_json or "{}"), "created_at": r.created_at}
            for r in rows
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


# ---- research metadata ---------------------------------------------------------- #


@router.get("/research", response_model=ResearchResponse)
def research() -> ResearchResponse:
    return ResearchResponse(
        citation=PAPER_CITATION,
        paper_reported=PAPER_REPORTED_RESULTS,
        feature_groups=FEATURE_GROUPS,
        feature_descriptions=FEATURE_DESCRIPTIONS,
        classifiers=[{"key": k, "name": v.display_name, "family": v.family} for k, v in MODEL_ZOO.items()],
        pipeline=["Dataset", "Preprocessing", "Feature Engineering", "Feature Selection (SHAP)", "Sentiment Analysis", "Feature Vector", "Train / Val / Test Split", "ML Classifier", "Prediction (Human / Bot)", "SHAP + LIME Explanation"],
        engineering_adaptations=[
            "Derived-feature formulas are not published in the paper; standard definitions are used and documented in docs/methodology.md.",
            "Hyperparameter grids are not published; compact randomised searches (F1-scored, stratified CV) are used.",
            "TextBlob provides polarity/subjectivity (the paper reports avg_polarity / avg_subjectivity, TextBlob's output names).",
            "Features that are constant in the training data are dropped and recorded on the model card (e.g. tweet-derived features when only user-level Cresci files are available).",
            "XGBoost SHAP contributions are reported in log-odds when the installed shap/xgboost combination cannot produce probability-scale values; the scale is always reported.",
        ],
    )


@router.get("/features")
def features(user: User = AnyUser) -> dict:
    return {"feature_version": FEATURE_VERSION, "n_features": len(FEATURE_NAMES), "groups": FEATURE_GROUPS, "descriptions": FEATURE_DESCRIPTIONS, "order": FEATURE_NAMES}
