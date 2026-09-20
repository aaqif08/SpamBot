"""Model lifecycle: persist trained artefacts, load the production model
(with checksum verification), activate / deprecate / delete, evaluations."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.storage import get_storage
from app.db.models import EvaluationRun, FeatureImportance, MLModel, ModelStatus, User
from app.services import audit
from ml.features import FEATURE_GROUP_OF
from ml.model_registry import ArtifactIntegrityError
from ml.predict import LoadedModel, ModelNotAvailableError, load_model_dir
from ml.train import MODEL_ZOO, TrainResult
from ml.utils import json_safe

log = logging.getLogger(__name__)

_cache: dict[str, LoadedModel] = {}
_cache_lock = threading.Lock()


def invalidate_cache(model_id: str | None = None) -> None:
    with _cache_lock:
        if model_id:
            _cache.pop(model_id, None)
        else:
            _cache.clear()


def model_public(m: MLModel, *, include_metrics: bool = True) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": m.id,
        "name": m.name,
        "version": m.version,
        "algorithm": m.algorithm,
        "status": m.status.value,
        "is_production": m.status == ModelStatus.PRODUCTION,
        "dataset_id": m.dataset_id,
        "dataset_version_id": m.dataset_version_id,
        "dataset_name": m.dataset_name,
        "feature_version": m.feature_version,
        "feature_names": json.loads(m.feature_names_json or "[]"),
        "n_features": m.n_features,
        "training_seconds": m.training_seconds,
        "trained_at": m.trained_at,
        "created_at": m.created_at,
        "created_by": m.created_by,
        "job_id": m.job_id,
        "notes": m.notes,
        "artifact_checksum": (json.loads(m.artifact_checksums_json or "{}") or {}).get("pipeline.joblib", ""),
    }
    if include_metrics:
        d["test_metrics"] = json.loads(m.test_metrics_json or "{}")
        d["validation_metrics"] = json.loads(m.validation_metrics_json or "{}")
        d["params"] = json.loads(m.params_json or "{}")
    return d


class ModelService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.storage = get_storage()
        self.settings = get_settings()

    # ---- queries -------------------------------------------------------------- #

    def list(self, organization_id: str) -> list[MLModel]:
        return self.db.query(MLModel).filter(MLModel.organization_id == organization_id).order_by(MLModel.created_at.desc()).all()

    def get(self, organization_id: str, model_id: str) -> MLModel:
        m = self.db.query(MLModel).filter(MLModel.id == model_id, MLModel.organization_id == organization_id).first()
        if m is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
        return m

    def production(self, organization_id: str) -> MLModel | None:
        return self.db.query(MLModel).filter(MLModel.organization_id == organization_id, MLModel.status == ModelStatus.PRODUCTION).first()

    def resolve(self, organization_id: str, model_id: str | None) -> MLModel:
        if model_id:
            m = self.get(organization_id, model_id)
            if m.status not in (ModelStatus.READY, ModelStatus.PRODUCTION, ModelStatus.DEPRECATED):
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Model is {m.status.value.lower()} and cannot be used for inference")
            return m
        m = self.production(organization_id)
        if m is None:
            raise ModelNotAvailableError("No production model is configured. Train a model and activate it.")
        return m

    # ---- loading ---------------------------------------------------------------- #

    def load(self, m: MLModel) -> LoadedModel:
        with _cache_lock:
            cached = _cache.get(m.id)
        if cached is not None:
            return cached
        try:
            artifact_dir = self.storage.local_dir(m.artifact_prefix)
            loaded = load_model_dir(artifact_dir, m.id, m.name, m.version, m.algorithm, json.loads(m.artifact_checksums_json or "{}"))
        except ArtifactIntegrityError as exc:
            log.error("artifact integrity failure for model %s: %s", m.id, exc)
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Model artefacts failed integrity verification; the model cannot be loaded") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Model artefacts are missing from storage") from exc
        with _cache_lock:
            _cache[m.id] = loaded
        return loaded

    # ---- persistence after training --------------------------------------------- #

    def persist_training_result(self, *, organization_id: str, created_by: str | None, result: TrainResult, dataset_id: str | None, dataset_version_id: str | None, dataset_name: str, job_id: str | None, model_row: MLModel | None = None) -> MLModel:
        card = result.card
        if model_row is None:
            version = (self.db.query(MLModel).filter(MLModel.organization_id == organization_id, MLModel.name == card.name).count()) + 1
            model_row = MLModel(organization_id=organization_id, created_by=created_by, name=card.name, version=version, algorithm=card.algorithm, status=ModelStatus.TRAINING)
            self.db.add(model_row)
            self.db.flush()
        prefix = f"org/{organization_id}/models/{model_row.id}"
        self.storage.upload_dir(result.artifact_dir, prefix)
        model_row.status = ModelStatus.READY
        model_row.dataset_id = dataset_id
        model_row.dataset_version_id = dataset_version_id
        model_row.dataset_name = dataset_name[:200]
        model_row.feature_version = card.feature_version
        model_row.feature_names_json = json.dumps(card.feature_names)
        model_row.n_features = len(card.feature_names)
        model_row.validation_metrics_json = json.dumps(json_safe({"mean": card.cv_metrics, "std": card.cv_std, "folds": result.metrics["cross_validation"]["n_folds"]}))
        model_row.test_metrics_json = json.dumps(json_safe(card.holdout_metrics))
        model_row.params_json = json.dumps(json_safe(card.params))
        model_row.artifact_prefix = prefix
        model_row.artifact_checksums_json = json.dumps(result.checksums)
        model_row.training_seconds = card.training_seconds
        model_row.trained_at = datetime.now(timezone.utc)
        model_row.job_id = job_id
        model_row.notes = card.notes[:2000]

        self.db.add(
            EvaluationRun(
                organization_id=organization_id,
                created_by=created_by,
                model_id=model_row.id,
                dataset_version_id=dataset_version_id,
                kind="holdout",
                n_samples=int(result.metrics["holdout"]["n_samples"]),
                metrics_json=json.dumps(json_safe(result.metrics["holdout"]["metrics"])),
                details_json=json.dumps(json_safe({k: v for k, v in result.metrics.items() if k != "holdout"} | {"holdout": result.metrics["holdout"]})),
            )
        )
        for row in result.shap_global.get("importance", []):
            self.db.add(FeatureImportance(model_id=model_row.id, feature=row["feature"], feature_group=row.get("group") or FEATURE_GROUP_OF.get(row["feature"], ""), mean_abs_shap=float(row["mean_abs_shap"]), mean_shap=float(row["mean_shap"]), rank=int(row["rank"])))
        self.db.commit()
        # Artefacts were copied to storage; the temporary training directory is no longer needed.
        shutil.rmtree(result.artifact_dir, ignore_errors=True)
        return model_row

    def new_training_row(self, organization_id: str, created_by: str | None, algorithm: str, job_id: str | None) -> MLModel:
        name = MODEL_ZOO[algorithm].display_name
        version = self.db.query(MLModel).filter(MLModel.organization_id == organization_id, MLModel.name == name).count() + 1
        row = MLModel(organization_id=organization_id, created_by=created_by, name=name, version=version, algorithm=algorithm, status=ModelStatus.TRAINING, job_id=job_id)
        self.db.add(row)
        self.db.commit()
        return row

    @staticmethod
    def temp_artifact_dir() -> Path:
        return Path(tempfile.mkdtemp(prefix="botshield-train-"))

    # ---- lifecycle ------------------------------------------------------------------ #

    def activate(self, user: User, model_id: str, request: Request | None = None) -> MLModel:
        m = self.get(user.organization_id, model_id)
        if m.status not in (ModelStatus.READY, ModelStatus.DEPRECATED, ModelStatus.PRODUCTION):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"A model in status {m.status.value} cannot be activated")
        # verify artefacts before promoting
        self.load(m)
        previous = self.production(user.organization_id)
        if previous and previous.id != m.id:
            previous.status = ModelStatus.READY
        m.status = ModelStatus.PRODUCTION
        self.db.commit()
        audit.record(self.db, "model.activated", actor=user, target_type="model", target_id=m.id, details={"previous": previous.id if previous else None, "name": m.name, "version": m.version}, request=request)
        return m

    def deprecate(self, user: User, model_id: str, request: Request | None = None) -> MLModel:
        m = self.get(user.organization_id, model_id)
        m.status = ModelStatus.DEPRECATED
        self.db.commit()
        invalidate_cache(m.id)
        audit.record(self.db, "model.deprecated", actor=user, target_type="model", target_id=m.id, request=request)
        return m

    def delete(self, user: User, model_id: str, request: Request | None = None) -> None:
        m = self.get(user.organization_id, model_id)
        if m.status == ModelStatus.PRODUCTION:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Deactivate the production model before deleting it")
        if m.artifact_prefix:
            try:
                self.storage.delete_prefix(m.artifact_prefix)
            except Exception:  # noqa: BLE001
                log.warning("could not delete artefacts for %s", m.id)
        self.db.delete(m)
        self.db.commit()
        invalidate_cache(model_id)
        audit.record(self.db, "model.deleted", actor=user, target_type="model", target_id=model_id, request=request)

    # ---- evaluation ------------------------------------------------------------------ #

    def evaluations(self, organization_id: str, model_id: str) -> list[EvaluationRun]:
        return self.db.query(EvaluationRun).filter(EvaluationRun.organization_id == organization_id, EvaluationRun.model_id == model_id).order_by(EvaluationRun.created_at.desc()).all()

    def feature_importance(self, model_id: str) -> list[dict[str, Any]]:
        rows = self.db.query(FeatureImportance).filter(FeatureImportance.model_id == model_id).order_by(FeatureImportance.rank.asc()).all()
        return [{"feature": r.feature, "group": r.feature_group, "mean_abs_shap": r.mean_abs_shap, "mean_shap": r.mean_shap, "rank": r.rank} for r in rows]
