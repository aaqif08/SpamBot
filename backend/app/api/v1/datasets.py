"""Dataset endpoints: upload, versions, inspection, evaluation, benchmark import."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import AdminUser, AnalystUser, AnyUser, MlRateLimited
from app.core.config import get_settings
from app.core.security import read_validated_csv
from app.db.database import get_db
from app.db.models import EvaluationRun, JobType, User
from app.schemas.api import BenchmarkStatus, DatasetEvaluateRequest, DatasetPublic, DatasetVersionPublic, EvaluationResult, JobResponse, ModelRef
from app.services import audit
from app.services.dataset_service import DatasetService
from app.services.jobs import create_job, dispatch, job_public
from app.services.model_service import ModelService
from ml.datasets import CRESCI_SUBSETS, labelled_frame
from ml.evaluation import full_evaluation
from ml.predict import InferenceEngine, ModelNotAvailableError
from ml.utils import json_safe

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetPublic])
def list_datasets(db: Session = Depends(get_db), user: User = AnyUser) -> list[DatasetPublic]:
    return [DatasetPublic(**d) for d in DatasetService(db).list(user.organization_id)]


@router.post("", response_model=DatasetPublic, status_code=status.HTTP_201_CREATED, dependencies=[MlRateLimited])
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    name: str | None = Form(default=None, max_length=200),
    description: str = Form(default="", max_length=2000),
    dataset_id: str | None = Form(default=None),
    db: Session = Depends(get_db),
    user: User = AnalystUser,
) -> DatasetPublic:
    """Upload a CSV as a new dataset, or as a new version of ``dataset_id``."""
    data = await read_validated_csv(file, get_settings().max_upload_mb)
    service = DatasetService(db)
    ds = service.upload(user, data, file.filename or "upload.csv", name=name, description=description, dataset_id=dataset_id, request=request)
    return DatasetPublic(**service.dataset_public(ds, with_summary=True))


@router.get("/benchmarks", response_model=list[BenchmarkStatus])
def benchmarks(db: Session = Depends(get_db), user: User = AnyUser) -> list[BenchmarkStatus]:
    """Availability of the Cresci benchmark files placed on the server by the operator."""
    return [BenchmarkStatus(**b) for b in DatasetService(db).benchmark_status()]


@router.post("/benchmarks/{kind}/import", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def import_benchmark(kind: str, request: Request, use_cache: bool = True, db: Session = Depends(get_db), admin: User = AdminUser) -> JobResponse:
    if kind not in CRESCI_SUBSETS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown benchmark")
    status_info = next(b for b in DatasetService(db).benchmark_status() if b["kind"] == kind)
    if not status_info["available"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{kind} files not found. Expected {status_info['install_path_hint']}")
    job = create_job(db, organization_id=admin.organization_id, created_by=admin.id, job_type=JobType.DATASET_IMPORT, params={"kind": kind, "use_cache": use_cache}, target_type="dataset")
    audit.record(db, "dataset.benchmark_import_started", actor=admin, target_type="job", target_id=job.id, details={"kind": kind}, request=request)
    dispatch(job)
    return JobResponse(**job_public(job))


@router.post("/benchmarks/combined", response_model=DatasetPublic, status_code=status.HTTP_201_CREATED)
def combined_benchmark(request: Request, db: Session = Depends(get_db), admin: User = AdminUser) -> DatasetPublic:
    service = DatasetService(db)
    ds = service.combined_benchmark(admin, request)
    return DatasetPublic(**service.dataset_public(ds, with_summary=True))


@router.get("/{dataset_id}", response_model=DatasetPublic)
def get_dataset(dataset_id: str, db: Session = Depends(get_db), user: User = AnyUser) -> DatasetPublic:
    service = DatasetService(db)
    return DatasetPublic(**service.dataset_public(service.get(user.organization_id, dataset_id), with_summary=True))


@router.get("/{dataset_id}/versions/{version_id}", response_model=DatasetVersionPublic)
def get_version(dataset_id: str, version_id: str, db: Session = Depends(get_db), user: User = AnyUser) -> DatasetVersionPublic:
    service = DatasetService(db)
    v = service.get_version(user.organization_id, version_id)
    if v.dataset_id != dataset_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset version not found")
    return DatasetVersionPublic(**service.version_public(v, with_summary=True))


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: str, request: Request, db: Session = Depends(get_db), user: User = AnalystUser) -> None:
    DatasetService(db).delete(user, dataset_id, request)


@router.post("/{dataset_id}/evaluate", response_model=EvaluationResult, dependencies=[MlRateLimited])
def evaluate_dataset(dataset_id: str, body: DatasetEvaluateRequest | None, request: Request, db: Session = Depends(get_db), user: User = AnalystUser) -> EvaluationResult:
    """Evaluate a model on a labelled dataset version (stored as an evaluation run)."""
    body = body or DatasetEvaluateRequest()
    service = DatasetService(db)
    ds = service.get(user.organization_id, dataset_id)
    version = service.current_version(ds)
    if not version.has_label:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Dataset has no label column; evaluation requires labels")
    models = ModelService(db)
    try:
        model_row = models.resolve(user.organization_id, body.model_id)
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    loaded = models.load(model_row)
    df = service.load_frame(version)
    try:
        X, y = labelled_frame(df)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    if len(set(y.tolist())) < 2:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Evaluation requires both HUMAN and BOT labels")
    proba = InferenceEngine.predict_vectors(loaded, X)
    ev = full_evaluation(y, (proba >= 0.5).astype(int), proba)
    run = EvaluationRun(
        organization_id=user.organization_id, created_by=user.id, model_id=model_row.id, dataset_version_id=version.id, kind="dataset",
        n_samples=int(ev["n_samples"]), metrics_json=json.dumps(json_safe(ev["metrics"])), details_json=json.dumps(json_safe({"holdout": ev, "dataset_id": ds.id, "dataset_name": ds.name})),
    )
    db.add(run)
    db.commit()
    audit.record(db, "model.evaluated", actor=user, target_type="model", target_id=model_row.id, details={"dataset": ds.id, "f1": ev["metrics"]["f1"]}, request=request)
    return EvaluationResult(model=ModelRef(id=model_row.id, name=model_row.name, version=model_row.version), dataset_version_id=version.id, n_samples=int(ev["n_samples"]), evaluation=ev, evaluation_run_id=run.id)
