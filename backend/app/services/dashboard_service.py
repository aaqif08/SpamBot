"""Organisation-scoped analytics for the dashboard — every value is a database
query or comes from the production model's stored evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import Batch, Dataset, DatasetStatus, EvaluationRun, Job, JobStatus, MLModel, ModelStatus, Prediction
from app.services.model_service import ModelService, model_public
from ml.evaluation import probability_histogram
from ml.utils import json_safe


def analytics(db: Session, organization_id: str) -> dict[str, Any]:
    org = organization_id
    total = db.query(func.count(Prediction.id)).filter(Prediction.organization_id == org).scalar() or 0
    bots = db.query(func.count(Prediction.id)).filter(Prediction.organization_id == org, Prediction.prediction == "BOT").scalar() or 0
    humans = int(total - bots)
    avg_p = db.query(func.avg(Prediction.bot_probability)).filter(Prediction.organization_id == org).scalar()
    high_risk = db.query(func.count(Prediction.id)).filter(Prediction.organization_id == org, Prediction.risk_score >= 60).scalar() or 0
    accounts = db.query(func.count(func.distinct(Prediction.account_identifier))).filter(Prediction.organization_id == org).scalar() or 0
    latest = db.query(Prediction).filter(Prediction.organization_id == org).order_by(Prediction.created_at.desc()).first()
    datasets = db.query(func.count(Dataset.id)).filter(Dataset.organization_id == org, Dataset.status != DatasetStatus.ARCHIVED).scalar() or 0
    models_total = db.query(func.count(MLModel.id)).filter(MLModel.organization_id == org, MLModel.status.in_([ModelStatus.READY, ModelStatus.PRODUCTION, ModelStatus.DEPRECATED])).scalar() or 0
    jobs_total = db.query(func.count(Job.id)).filter(Job.organization_id == org).scalar() or 0
    jobs_running = db.query(func.count(Job.id)).filter(Job.organization_id == org, Job.status.in_([JobStatus.QUEUED, JobStatus.PROCESSING])).scalar() or 0
    batches = db.query(func.count(Batch.id)).filter(Batch.organization_id == org).scalar() or 0

    probs = [p[0] for p in db.query(Prediction.bot_probability).filter(Prediction.organization_id == org).order_by(Prediction.created_at.desc()).limit(5000).all()]
    hist = probability_histogram(np.asarray(probs)) if probs else []

    since = datetime.now(timezone.utc) - timedelta(days=30)
    daily_rows = (
        db.query(func.date(Prediction.created_at), Prediction.prediction, func.count(Prediction.id))
        .filter(Prediction.organization_id == org, Prediction.created_at >= since)
        .group_by(func.date(Prediction.created_at), Prediction.prediction)
        .order_by(func.date(Prediction.created_at))
        .all()
    )
    daily: dict[str, dict[str, Any]] = {}
    for day, label, n in daily_rows:
        d = daily.setdefault(str(day), {"date": str(day), "BOT": 0, "HUMAN": 0})
        d[label] = int(n)

    risk_rows = db.query(Prediction.risk_band, func.count(Prediction.id)).filter(Prediction.organization_id == org).group_by(Prediction.risk_band).all()
    model_usage = (
        db.query(Prediction.model_name, Prediction.model_version, func.count(Prediction.id))
        .filter(Prediction.organization_id == org)
        .group_by(Prediction.model_name, Prediction.model_version)
        .order_by(func.count(Prediction.id).desc())
        .limit(10)
        .all()
    )

    ms = ModelService(db)
    prod = ms.production(org)
    prod_info: dict[str, Any] | None = None
    feature_importance: list[dict[str, Any]] = []
    holdout: dict[str, Any] | None = None
    if prod is not None:
        prod_info = model_public(prod)
        feature_importance = ms.feature_importance(prod.id)[:15]
        ev = db.query(EvaluationRun).filter(EvaluationRun.model_id == prod.id, EvaluationRun.kind == "holdout").order_by(EvaluationRun.created_at.desc()).first()
        if ev is not None:
            details = json.loads(ev.details_json or "{}")
            holdout = {
                "metrics": json.loads(ev.metrics_json or "{}"),
                "confusion_matrix": (details.get("holdout") or {}).get("confusion_matrix"),
                "roc_curve": (details.get("holdout") or {}).get("roc_curve"),
                "n_samples": ev.n_samples,
                "class_distribution": details.get("class_distribution"),
            }

    comparison = [
        {
            "id": m.id, "name": m.name, "version": m.version, "algorithm": m.algorithm, "status": m.status.value, "dataset": m.dataset_name,
            **{k: json.loads(m.test_metrics_json or "{}").get(k) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
            "training_seconds": m.training_seconds,
        }
        for m in ms.list(org)
        if m.status in (ModelStatus.READY, ModelStatus.PRODUCTION, ModelStatus.DEPRECATED)
    ]

    return json_safe(
        {
            "cards": {
                "accounts_analyzed": int(accounts),
                "predictions_total": int(total),
                "bots_classified": int(bots),
                "humans_classified": humans,
                "bot_rate": (bots / total) if total else None,
                "average_bot_probability": float(avg_p) if avg_p is not None else None,
                "high_risk": int(high_risk),
                "datasets": int(datasets),
                "models": int(models_total),
                "jobs_total": int(jobs_total),
                "jobs_running": int(jobs_running),
                "batches": int(batches),
                "latest_analysis_at": latest.created_at if latest else None,
                "production_model": {"id": prod.id, "name": prod.name, "version": prod.version, "algorithm": prod.algorithm} if prod else None,
            },
            "charts": {
                "bot_vs_human": [{"name": "BOT", "value": int(bots)}, {"name": "HUMAN", "value": humans}] if total else [],
                "probability_histogram": hist,
                "timeline": list(daily.values()),
                "risk_distribution": [{"band": b or "unknown", "count": int(n)} for b, n in risk_rows],
                "model_usage": [{"model": f"{n} v{v}", "count": int(c)} for n, v, c in model_usage],
                "feature_importance": feature_importance,
                "model_comparison": comparison,
                "production_holdout": holdout,
            },
            "production_model": prod_info,
            "has_model": prod is not None,
            "has_predictions": total > 0,
        }
    )
