"""Job handlers executed by the job runner (thread executor or Celery worker).

Each handler receives a :class:`JobContext`, does the work with its own DB
session, reports progress, and returns a JSON-safe result stored on the job.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.core.storage import get_storage
from app.db.models import Batch, Dataset, JobStatus, JobType, MLModel, ModelStatus, Prediction
from app.services import audit
from app.services.dataset_service import DatasetService
from app.services.jobs import JobContext, register_handler
from app.services.model_service import ModelService, model_public
from ml.datasets import labelled_frame, normalise_columns
from ml.evaluation import full_evaluation, probability_histogram
from ml.features import coerce_label, detect_label_column
from ml.predict import InferenceEngine
from ml.train import MODEL_ZOO, TrainConfig, train_model
from ml.utils import json_safe

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@register_handler(JobType.TRAINING)
def training_job(ctx: JobContext) -> dict[str, Any]:
    p = ctx.params
    db = ctx.db
    ds_service = DatasetService(db)
    models = ModelService(db)
    model_row = db.get(MLModel, p["model_row_id"])
    if model_row is None:
        raise RuntimeError("Model row missing")
    try:
        ctx.progress("preprocessing", 0.02, "Loading dataset")
        version = ds_service.get_version(ctx.organization_id, p["dataset_version_id"])
        dataset = db.get(Dataset, version.dataset_id)
        df = ds_service.load_frame(version)
        X, y = labelled_frame(df)
        ctx.progress("preprocessing", 0.04, f"Featurised {len(X)} labelled accounts")
        config = TrainConfig(
            algorithm=p["algorithm"],
            test_size=float(p.get("test_size", 0.25)),
            cv_folds=int(p.get("cv_folds", 5)),
            seed=int(p.get("seed", 42)),
            hyperparameter_search=bool(p.get("hyperparameter_search", True)),
            search_iterations=int(p.get("search_iterations", 6)),
            feature_selection=bool(p.get("feature_selection", False)),
            feature_selection_top_k=int(p.get("feature_selection_top_k", 20)),
            notes=str(p.get("notes", "")),
        )
        artifact_dir = models.temp_artifact_dir()
        try:
            result = train_model(X, y, config, artifact_dir, progress=ctx.progress)
            ctx.progress("saving", 0.95, "Registering model version")
            row = models.persist_training_result(
                organization_id=ctx.organization_id,
                created_by=ctx.job.created_by,
                result=result,
                dataset_id=version.dataset_id,
                dataset_version_id=version.id,
                dataset_name=dataset.name if dataset else "",
                job_id=ctx.job.id,
                model_row=model_row,
            )
        finally:
            shutil.rmtree(artifact_dir, ignore_errors=True)
        activated = False
        if p.get("activate") and models.production(ctx.organization_id) is None:
            row.status = ModelStatus.PRODUCTION
            db.commit()
            activated = True
        elif p.get("activate"):
            previous = models.production(ctx.organization_id)
            if previous and previous.id != row.id:
                previous.status = ModelStatus.READY
            row.status = ModelStatus.PRODUCTION
            db.commit()
            activated = True
        user = ctx.user
        audit.record(db, "training.completed", actor=user, organization_id=ctx.organization_id, target_type="model", target_id=row.id, details={"algorithm": config.algorithm, "activated": activated, "f1": result.card.holdout_metrics.get("f1")})
        return {
            "model_id": row.id,
            "model": model_public(row),
            "metrics": result.metrics,
            "feature_importance": result.shap_global.get("importance", [])[:20],
            "explainer": result.shap_global.get("explainer"),
            "output_scale": result.shap_global.get("output_scale"),
            "activated": activated,
        }
    except Exception as exc:
        model_row.status = ModelStatus.FAILED
        model_row.notes = (f"{type(exc).__name__}: {exc}")[:2000]
        db.commit()
        audit.record(db, "training.failed", actor=ctx.user, organization_id=ctx.organization_id, target_type="model", target_id=model_row.id, outcome="failure", details={"error": str(exc)[:300]})
        raise


# --------------------------------------------------------------------------- #
# Batch prediction
# --------------------------------------------------------------------------- #


@register_handler(JobType.BATCH_PREDICTION)
def batch_prediction_job(ctx: JobContext) -> dict[str, Any]:
    p = ctx.params
    db = ctx.db
    storage = get_storage()
    models = ModelService(db)
    batch = db.get(Batch, p["batch_id"])
    if batch is None:
        raise RuntimeError("Batch row missing")
    try:
        batch.status = JobStatus.PROCESSING
        db.commit()
        ctx.progress("preprocessing", 0.05, "Loading input file")
        data = storage.get_bytes(batch.input_storage_key)
        df = pd.read_csv(io.BytesIO(data), encoding="utf-8", encoding_errors="replace", on_bad_lines="skip", low_memory=False)
        if len(df) == 0:
            raise ValueError("The input file contains no rows")
        model_row = models.get(ctx.organization_id, p["model_id"])
        loaded = models.load(model_row)
        norm = normalise_columns(df)
        id_col = next((c for c in ("id", "screen_name", "account_id", "username", "user_id") if c in norm.columns), None)
        label_col = detect_label_column(norm.columns)
        batch.total_rows = int(len(norm))
        db.commit()

        ctx.progress("feature_engineering", 0.15, f"Featurising {len(norm):,} accounts")
        chunk = 2000
        outputs: list[pd.DataFrame] = []
        failed = 0
        for start in range(0, len(norm), chunk):
            part = norm.iloc[start : start + chunk]
            try:
                res = InferenceEngine.predict_frame(loaded, part)
            except Exception as exc:  # noqa: BLE001 - keep going, count failures
                log.warning("batch chunk failed: %s", exc)
                failed += len(part)
                continue
            res.insert(0, "account_id", part[id_col].astype(str).values if id_col else [f"row_{i + 1}" for i in range(start, start + len(part))])
            if label_col is not None:
                res.insert(1, "label", part[label_col].map(coerce_label).map({0: "HUMAN", 1: "BOT"}).fillna("").values)
            outputs.append(res)
            batch.processed_rows = int(sum(len(o) for o in outputs))
            batch.failed_rows = failed
            db.commit()
            ctx.progress("processing", 0.15 + 0.7 * (start + len(part)) / len(norm), f"Scored {batch.processed_rows:,}/{len(norm):,} accounts")
        if not outputs:
            raise ValueError("No rows could be scored")
        results = pd.concat(outputs, ignore_index=True)

        evaluation: dict[str, Any] | None = None
        if label_col is not None and "label" in results.columns:
            mask = results["label"].isin(["HUMAN", "BOT"]).values
            if mask.sum() >= 2 and results.loc[mask, "label"].nunique() == 2:
                y_true = (results.loc[mask, "label"] == "BOT").astype(int).values
                y_proba = results.loc[mask, "bot_probability"].values
                evaluation = full_evaluation(y_true, (y_proba >= 0.5).astype(int), y_proba)

        ctx.progress("saving", 0.9, "Writing results")
        p_bot = results["bot_probability"].values
        top_feats = [r["feature"] for r in models.feature_importance(model_row.id)[:10]] or loaded.feature_names[:10]
        export_cols = ["account_id"] + (["label"] if "label" in results.columns else []) + ["prediction", "bot_probability", "human_probability", "risk_score", "risk_band"] + top_feats
        buf = io.StringIO()
        results[export_cols].to_csv(buf, index=False)
        out_key = f"org/{ctx.organization_id}/batches/{batch.id}/predictions.csv"
        storage.put_bytes(out_key, buf.getvalue().encode("utf-8"), content_type="text/csv")

        now = datetime.now(timezone.utc)
        rows = []
        for _, r in results.iterrows():
            rows.append(
                Prediction(
                    organization_id=ctx.organization_id,
                    created_by=ctx.job.created_by,
                    account_identifier=str(r["account_id"])[:200],
                    batch_id=batch.id,
                    model_id=model_row.id,
                    model_name=model_row.name,
                    model_version=model_row.version,
                    source="batch",
                    status="COMPLETED",
                    prediction=str(r["prediction"]),
                    bot_probability=float(r["bot_probability"]),
                    human_probability=float(r["human_probability"]),
                    risk_score=int(r["risk_score"]),
                    risk_band=str(r["risk_band"]),
                    features_json=json.dumps({f: float(r[f]) for f in loaded.feature_names}),
                    label_true=(str(r["label"]) if "label" in results.columns and r["label"] in ("HUMAN", "BOT") else None),
                    created_at=now,
                    updated_at=now,
                )
            )
            if len(rows) >= 1000:
                db.bulk_save_objects(rows)
                db.commit()
                rows = []
        if rows:
            db.bulk_save_objects(rows)
            db.commit()

        batch.output_storage_key = out_key
        batch.n_bots = int((results["prediction"] == "BOT").sum())
        batch.n_humans = int(len(results) - batch.n_bots)
        batch.avg_bot_probability = float(np.mean(p_bot))
        batch.high_risk = int((results["risk_score"] >= 60).sum())
        batch.processed_rows = int(len(results))
        batch.failed_rows = failed
        batch.summary_json = json.dumps(json_safe({
            "risk_band_distribution": {k: int(v) for k, v in results["risk_band"].value_counts().to_dict().items()},
            "probability_histogram": probability_histogram(p_bot),
            "top_features_exported": top_feats,
            "label_column": label_col,
            "evaluation": evaluation,
        }))
        batch.status = JobStatus.COMPLETED
        batch.completed_at = now
        db.commit()
        audit.record(db, "batch.completed", actor=ctx.user, organization_id=ctx.organization_id, target_type="batch", target_id=batch.id, details={"rows": int(len(results)), "bots": batch.n_bots, "failed": failed})
        return {"batch_id": batch.id, "total_rows": batch.total_rows, "processed_rows": batch.processed_rows, "failed_rows": failed, "n_bots": batch.n_bots, "n_humans": batch.n_humans}
    except Exception as exc:
        batch.status = JobStatus.FAILED
        batch.error = f"{type(exc).__name__}: {exc}"[:2000]
        batch.completed_at = datetime.now(timezone.utc)
        db.commit()
        audit.record(db, "batch.failed", actor=ctx.user, organization_id=ctx.organization_id, target_type="batch", target_id=batch.id, outcome="failure", details={"error": str(exc)[:300]})
        raise


# --------------------------------------------------------------------------- #
# Benchmark import
# --------------------------------------------------------------------------- #


@register_handler(JobType.DATASET_IMPORT)
def dataset_import_job(ctx: JobContext) -> dict[str, Any]:
    p = ctx.params
    service = DatasetService(ctx.db)
    ctx.progress("preprocessing", 0.05, f"Importing {p['kind']}")
    ds = service.import_benchmark(ctx.job.created_by, ctx.organization_id, p["kind"], progress=lambda m: ctx.progress("preprocessing", 0.5, m), use_cache=bool(p.get("use_cache", True)))
    ctx.job.target_type = "dataset"
    ctx.job.target_id = ds.id
    audit.record(ctx.db, "dataset.benchmark_imported", actor=ctx.user, organization_id=ctx.organization_id, target_type="dataset", target_id=ds.id, details={"kind": p["kind"]})
    return {"dataset_id": ds.id, "dataset": service.dataset_public(ds)}


def algorithm_display(key: str) -> str:
    return MODEL_ZOO[key].display_name if key in MODEL_ZOO else key
