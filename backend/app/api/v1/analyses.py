"""Single-account analyses, explanations, history, batches."""

from __future__ import annotations

import io
import json
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import AdminUser, AnalystUser, AnyUser, MlRateLimited
from app.core.config import get_settings
from app.core.security import read_validated_csv, sanitize_filename
from app.core.storage import get_storage
from app.db.database import get_db
from app.db.models import Batch, JobStatus, JobType, MLModel, User
from app.schemas.api import AnalysisResponse, AnalyzeRequest, BatchResponse, ExplanationResponse, HistoryResponse, ModelRef, RetentionRequest
from app.services import audit
from app.services.dataset_service import DatasetService
from app.services.jobs import create_job, dispatch
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService
from ml.predict import ModelNotAvailableError

router = APIRouter(tags=["analyses"])


@router.post("/analyses", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED, dependencies=[MlRateLimited])
def analyze(body: AnalyzeRequest, request: Request, db: Session = Depends(get_db), user: User = AnalystUser) -> AnalysisResponse:
    result = PredictionService(db).analyze(user, body.account.to_account_dict(), model_id=body.model_id, source=body.source, explain=body.explain, request=request)
    return AnalysisResponse(**result)


@router.get("/analyses", response_model=HistoryResponse)
def history(
    db: Session = Depends(get_db),
    user: User = AnyUser,
    prediction: str | None = Query(default=None, pattern="^(BOT|HUMAN)$"),
    min_risk: int | None = Query(default=None, ge=0, le=100),
    model_id: str | None = None,
    source: str | None = Query(default=None, pattern="^(manual|batch|x_api|csv)$"),
    batch_id: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    search: str | None = Query(default=None, max_length=100),
    sort: str = Query(default="created_at:desc", pattern="^(created_at|risk_score|bot_probability|account):(asc|desc)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
) -> HistoryResponse:
    return HistoryResponse(**PredictionService(db).history(user.organization_id, prediction=prediction, min_risk=min_risk, model_id=model_id, source=source, batch_id=batch_id, date_from=date_from, date_to=date_to, search=search, sort=sort, page=page, page_size=page_size))


@router.get("/analyses/{prediction_id}", response_model=AnalysisResponse)
def analysis_detail(prediction_id: str, db: Session = Depends(get_db), user: User = AnyUser) -> AnalysisResponse:
    return AnalysisResponse(**PredictionService(db).detail(user.organization_id, prediction_id))


@router.delete("/analyses/{prediction_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(prediction_id: str, request: Request, db: Session = Depends(get_db), user: User = AnalystUser) -> None:
    PredictionService(db).delete(user, prediction_id, request)


@router.post("/analyses/retention/purge")
def purge_analyses(body: RetentionRequest, request: Request, db: Session = Depends(get_db), admin: User = AdminUser) -> dict:
    """Data retention: delete analyses (and their explanations) older than N days."""
    deleted = PredictionService(db).purge_older_than(admin, body.older_than_days, request)
    return {"deleted": deleted, "older_than_days": body.older_than_days}


@router.get("/analyses/{prediction_id}/explanations/{method}", response_model=ExplanationResponse, dependencies=[MlRateLimited])
def explanation(prediction_id: str, method: str, db: Session = Depends(get_db), user: User = AnyUser) -> ExplanationResponse:
    if method not in ("shap", "lime"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown explanation method")
    return ExplanationResponse(**PredictionService(db).explanation(user, prediction_id, method))


# ---- batches ---------------------------------------------------------------- #


def _batch_public(db: Session, b: Batch) -> BatchResponse:
    model = db.get(MLModel, b.model_id) if b.model_id else None
    return BatchResponse(
        id=b.id, name=b.name, status=b.status.value, job_id=b.job_id,
        model=ModelRef(id=model.id, name=model.name, version=model.version) if model else None,
        dataset_version_id=b.dataset_version_id, total_rows=b.total_rows, processed_rows=b.processed_rows, failed_rows=b.failed_rows,
        n_bots=b.n_bots, n_humans=b.n_humans, avg_bot_probability=b.avg_bot_probability, high_risk=b.high_risk,
        summary=json.loads(b.summary_json or "{}"), error=b.error, has_output=bool(b.output_storage_key), created_at=b.created_at, completed_at=b.completed_at,
    )


@router.post("/batches", response_model=BatchResponse, status_code=status.HTTP_202_ACCEPTED, dependencies=[MlRateLimited])
async def create_batch(
    request: Request,
    file: UploadFile | None = File(default=None),
    dataset_id: str | None = Form(default=None),
    model_id: str | None = Form(default=None),
    name: str | None = Form(default=None),
    db: Session = Depends(get_db),
    user: User = AnalystUser,
) -> BatchResponse:
    settings = get_settings()
    storage = get_storage()
    models = ModelService(db)
    try:
        model_row = models.resolve(user.organization_id, model_id)
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    dataset_version_id: str | None = None
    if file is not None:
        data = await read_validated_csv(file, settings.max_upload_mb)
        try:
            head = pd.read_csv(io.BytesIO(data), nrows=5, encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Could not parse CSV: {exc}") from exc
        if head.shape[1] == 0:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="CSV has no columns")
        batch_name = name or sanitize_filename(file.filename or "batch.csv")
    elif dataset_id:
        ds_service = DatasetService(db)
        ds = ds_service.get(user.organization_id, dataset_id)
        version = ds_service.current_version(ds)
        data = storage.get_bytes(version.storage_key)
        dataset_version_id = version.id
        batch_name = name or f"{ds.name} v{version.version}"
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide a CSV file or a dataset_id")

    batch = Batch(organization_id=user.organization_id, created_by=user.id, model_id=model_row.id, dataset_version_id=dataset_version_id, name=batch_name[:200], status=JobStatus.QUEUED)
    db.add(batch)
    db.flush()
    key = f"org/{user.organization_id}/batches/{batch.id}/input.csv"
    storage.put_bytes(key, data, content_type="text/csv")
    batch.input_storage_key = key
    job = create_job(db, organization_id=user.organization_id, created_by=user.id, job_type=JobType.BATCH_PREDICTION, params={"batch_id": batch.id, "model_id": model_row.id}, target_type="batch", target_id=batch.id)
    batch.job_id = job.id
    db.commit()
    audit.record(db, "batch.submitted", actor=user, target_type="batch", target_id=batch.id, details={"model": model_row.id, "dataset_version": dataset_version_id}, request=request)
    dispatch(job)
    return _batch_public(db, batch)


@router.get("/batches", response_model=list[BatchResponse])
def list_batches(db: Session = Depends(get_db), user: User = AnyUser, limit: int = Query(default=50, ge=1, le=200)) -> list[BatchResponse]:
    rows = db.query(Batch).filter(Batch.organization_id == user.organization_id).order_by(Batch.created_at.desc()).limit(limit).all()
    return [_batch_public(db, b) for b in rows]


@router.get("/batches/{batch_id}", response_model=BatchResponse)
def get_batch(batch_id: str, db: Session = Depends(get_db), user: User = AnyUser) -> BatchResponse:
    b = db.query(Batch).filter(Batch.id == batch_id, Batch.organization_id == user.organization_id).first()
    if b is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found")
    return _batch_public(db, b)


@router.get("/batches/{batch_id}/download")
def download_batch(batch_id: str, db: Session = Depends(get_db), user: User = AnyUser) -> StreamingResponse:
    b = db.query(Batch).filter(Batch.id == batch_id, Batch.organization_id == user.organization_id).first()
    if b is None or not b.output_storage_key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch output not available")
    data = get_storage().get_bytes(b.output_storage_key)
    return StreamingResponse(io.BytesIO(data), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="predictions-{batch_id[:8]}.csv"'})


@router.delete("/batches/{batch_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_batch(batch_id: str, request: Request, db: Session = Depends(get_db), user: User = AnalystUser) -> None:
    b = db.query(Batch).filter(Batch.id == batch_id, Batch.organization_id == user.organization_id).first()
    if b is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found")
    if b.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Batch is still running")
    storage = get_storage()
    for key in (b.input_storage_key, b.output_storage_key):
        if key:
            try:
                storage.delete(key)
            except Exception:  # noqa: BLE001
                pass
    db.delete(b)
    db.commit()
    audit.record(db, "batch.deleted", actor=user, target_type="batch", target_id=batch_id, request=request)
