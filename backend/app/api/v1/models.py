"""Model lifecycle, training jobs, evaluation artefacts, global explanations."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.deps import AdminUser, AnalystUser, AnyUser, MlRateLimited
from app.db.database import get_db
from app.db.models import JobType, ModelStatus, User
from app.schemas.api import GlobalExplanationResponse, ModelEvaluationResponse, ModelListResponse, ModelPublic, TrainRequest, TrainSubmitted
from app.services import audit
from app.services.dataset_service import DatasetService
from app.services.jobs import create_job, dispatch
from app.services.model_service import ModelService, model_public
from ml.features import FEATURE_DESCRIPTIONS, FEATURE_GROUPS
from ml.train import MODEL_ZOO, model_choices

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
def list_models(db: Session = Depends(get_db), user: User = AnyUser) -> ModelListResponse:
    service = ModelService(db)
    prod = service.production(user.organization_id)
    return ModelListResponse(production_model_id=prod.id if prod else None, models=[ModelPublic(**model_public(m)) for m in service.list(user.organization_id)], supported_algorithms=model_choices())


@router.post("/train", response_model=TrainSubmitted, status_code=status.HTTP_202_ACCEPTED, dependencies=[MlRateLimited])
def train(body: TrainRequest, request: Request, db: Session = Depends(get_db), user: User = AnalystUser) -> TrainSubmitted:
    if body.algorithm not in MODEL_ZOO:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Unknown algorithm '{body.algorithm}'")
    ds_service = DatasetService(db)
    ds = ds_service.get(user.organization_id, body.dataset_id)
    version = ds_service.get_version(user.organization_id, body.dataset_version_id) if body.dataset_version_id else ds_service.current_version(ds)
    if version.dataset_id != ds.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset version not found")
    if not version.has_label:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Training requires a labelled dataset (label / is_bot / class column)")
    if version.status.value == "INVALID":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Dataset version failed validation")
    models = ModelService(db)
    row = models.new_training_row(user.organization_id, user.id, body.algorithm, None)
    params = body.model_dump() | {"model_row_id": row.id, "dataset_version_id": version.id}
    job = create_job(db, organization_id=user.organization_id, created_by=user.id, job_type=JobType.TRAINING, params=params, target_type="model", target_id=row.id)
    row.job_id = job.id
    db.commit()
    audit.record(db, "training.started", actor=user, target_type="model", target_id=row.id, details={"algorithm": body.algorithm, "dataset_version": version.id, "job": job.id}, request=request)
    dispatch(job)
    return TrainSubmitted(job_id=job.id, model_id=row.id, status=job.status.value)


@router.get("/{model_id}", response_model=ModelPublic)
def get_model(model_id: str, db: Session = Depends(get_db), user: User = AnyUser) -> ModelPublic:
    return ModelPublic(**model_public(ModelService(db).get(user.organization_id, model_id)))


@router.post("/{model_id}/activate", response_model=ModelPublic)
def activate(model_id: str, request: Request, db: Session = Depends(get_db), admin: User = AdminUser) -> ModelPublic:
    return ModelPublic(**model_public(ModelService(db).activate(admin, model_id, request)))


@router.post("/{model_id}/deprecate", response_model=ModelPublic)
def deprecate(model_id: str, request: Request, db: Session = Depends(get_db), admin: User = AdminUser) -> ModelPublic:
    return ModelPublic(**model_public(ModelService(db).deprecate(admin, model_id, request)))


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(model_id: str, request: Request, db: Session = Depends(get_db), admin: User = AdminUser) -> None:
    ModelService(db).delete(admin, model_id, request)


@router.get("/{model_id}/evaluation", response_model=ModelEvaluationResponse)
def evaluation(model_id: str, db: Session = Depends(get_db), user: User = AnyUser) -> ModelEvaluationResponse:
    service = ModelService(db)
    m = service.get(user.organization_id, model_id)
    if m.status not in (ModelStatus.READY, ModelStatus.PRODUCTION, ModelStatus.DEPRECATED):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"No evaluation available: model is {m.status.value.lower()}")
    loaded = service.load(m)
    evals = [
        {"id": e.id, "kind": e.kind, "dataset_version_id": e.dataset_version_id, "n_samples": e.n_samples, "metrics": json.loads(e.metrics_json or "{}"), "details": json.loads(e.details_json or "{}"), "created_at": e.created_at}
        for e in service.evaluations(user.organization_id, m.id)
    ]
    return ModelEvaluationResponse(model=ModelPublic(**model_public(m)), feature_metadata=loaded.feature_metadata, evaluations=evals, feature_importance=service.feature_importance(m.id))


@router.get("/{model_id}/explanation", response_model=GlobalExplanationResponse)
def global_explanation(model_id: str, db: Session = Depends(get_db), user: User = AnyUser) -> GlobalExplanationResponse:
    service = ModelService(db)
    m = service.get(user.organization_id, model_id)
    if m.status not in (ModelStatus.READY, ModelStatus.PRODUCTION, ModelStatus.DEPRECATED):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Model is not trained")
    loaded = service.load(m)
    if not loaded.shap_global:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SHAP explanation unavailable for this model")
    return GlobalExplanationResponse(model=ModelPublic(**model_public(m)), shap_global=loaded.shap_global, feature_descriptions=FEATURE_DESCRIPTIONS, feature_groups={g: [f for f in fs if f in loaded.feature_names] for g, fs in FEATURE_GROUPS.items()})
