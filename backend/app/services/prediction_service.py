"""Single-account prediction, explanation persistence, history and retention."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException, Request, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import AnalyzedAccount, Batch, Explanation, Prediction, User
from app.services import audit
from app.services.model_service import ModelService
from ml.predict import InferenceEngine, ModelNotAvailableError
from ml.utils import json_safe

log = logging.getLogger(__name__)

RISK_SCORE_NOTE = (
    "Risk Score is an application-level representation of the model's estimated bot probability "
    "(risk_score = round(100 × P(bot))). It is a model output, not a verified fact about the account."
)

_FEATURE_PHRASES: dict[str, tuple[str, str]] = {
    "hashtag_count": ("hashtag-heavy content", "moderate hashtag usage"),
    "avg_hashtag": ("high hashtags per tweet", "few hashtags per tweet"),
    "url_count": ("frequent link sharing", "limited link sharing"),
    "avg_url": ("high URL rate per tweet", "low URL rate per tweet"),
    "ffratio": ("low follower/following ratio", "healthy follower/following ratio"),
    "friends_count": ("following pattern typical of automated accounts", "following pattern typical of humans"),
    "followers_count": ("follower count typical of automated accounts", "follower count typical of humans"),
    "favorites_count": ("very few likes given", "active liking behaviour"),
    "reply_count": ("reply pattern typical of automated engagement", "reply pattern typical of conversation"),
    "avg_replies": ("replies per tweet typical of bots", "replies per tweet typical of humans"),
    "retweet_count": ("retweet pattern typical of amplification", "retweet pattern typical of organic sharing"),
    "avg_retweets": ("retweets per tweet typical of bots", "retweets per tweet typical of humans"),
    "mentions_count": ("mention pattern typical of bots", "mention pattern typical of conversation"),
    "avg_mentions": ("mentions per tweet typical of bots", "mentions per tweet typical of humans"),
    "avg_user_engagement": ("low engagement received", "healthy engagement received"),
    "statuses_count": ("posting volume typical of automation", "posting volume typical of humans"),
    "listed_count": ("rarely listed by others", "listed by other users"),
    "verified": ("not verified", "verified account"),
    "default_profile": ("default profile theme", "customised profile"),
    "default_profile_image": ("default profile picture", "custom profile picture"),
    "geo_enabled": ("geo-tagging disabled", "geo-tagging enabled"),
    "profile_background_tile": ("tiled background", "non-tiled background"),
    "profile_completeness": ("incomplete profile", "complete profile"),
    "description_binary": ("missing description", "description present"),
    "unique_word_count": ("small vocabulary", "rich vocabulary"),
    "unique_word_use": ("repetitive wording", "diverse wording"),
    "punctuation_count": ("punctuation pattern typical of bots", "punctuation pattern typical of humans"),
    "punctuation_density": ("punctuation density typical of bots", "punctuation density typical of humans"),
    "avg_sentence_length": ("sentence length typical of bots", "sentence length typical of humans"),
    "avg_polarity": ("promotional/uniform tone", "natural sentiment variation"),
    "avg_subjectivity": ("subjectivity level typical of bots", "subjectivity level typical of humans"),
}

_INPUT_SUMMARY_KEYS = (
    "account_id", "screen_name", "name", "verified", "friends_count", "followers_count", "listed_count", "favorites_count",
    "statuses_count", "location", "url", "default_profile", "default_profile_image", "geo_enabled", "profile_background_tile",
    "has_profile_banner", "hashtag_count", "mentions_count", "retweet_count", "reply_count", "url_count", "favorite_count_received", "tweets_observed",
)


def build_interpretation(prediction: str, p_bot: float, risk_score: int, risk_band: str, shap_exp: dict[str, Any] | None) -> dict[str, Any]:
    contribs = (shap_exp or {}).get("contributions", [])
    bot_side = [c for c in contribs if c["direction"] == "BOT"][:4]
    human_side = [c for c in contribs if c["direction"] == "HUMAN"][:4]

    def phrase(c: dict[str, Any]) -> str:
        pair = _FEATURE_PHRASES.get(c["feature"])
        return (pair[0] if c["direction"] == "BOT" else pair[1]) if pair else c["feature"].replace("_", " ")

    summary = f"Model classification: {prediction}. Estimated bot probability {p_bot:.1%} (risk score {risk_score}/100, {risk_band} band)."
    if bot_side:
        summary += " Indicators pushing toward BOT: " + ", ".join(phrase(c) for c in bot_side) + "."
    if human_side:
        summary += " Indicators pushing toward HUMAN: " + ", ".join(phrase(c) for c in human_side) + "."
    if not contribs:
        summary += " Feature-level explanation unavailable for this prediction."
    if risk_band in ("critical", "high"):
        recommendation = "High estimated bot probability. Recommend analyst review before any action; the classification is a model output."
    elif risk_band == "medium":
        recommendation = "Mixed signals. Consider collecting more activity data (tweets, engagement) before drawing conclusions."
    else:
        recommendation = "Low estimated bot probability. No action suggested; continue routine monitoring."
    return {
        "summary": summary,
        "recommendation": recommendation,
        "bot_indicators": [{"feature": c["feature"], "phrase": phrase(c), "impact": c["shap"], "value": c["value"]} for c in bot_side],
        "human_indicators": [{"feature": c["feature"], "phrase": phrase(c), "impact": c["shap"], "value": c["value"]} for c in human_side],
        "disclaimer": RISK_SCORE_NOTE,
    }


class PredictionService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()
        self.models = ModelService(db)

    # ---- single account ------------------------------------------------------ #

    def analyze(self, user: User, account: dict[str, Any], *, model_id: str | None, source: str, explain: bool, request: Request | None = None) -> dict[str, Any]:
        try:
            model_row = self.models.resolve(user.organization_id, model_id)
        except ModelNotAvailableError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        loaded = self.models.load(model_row)
        result = InferenceEngine.predict_account(loaded, account, explain=explain, lime_samples=self.settings.explanation_lime_samples)

        identifier = str(account.get("account_id") or account.get("screen_name") or account.get("name") or "").strip()[:200]
        acct = None
        if identifier:
            acct = self.db.query(AnalyzedAccount).filter(AnalyzedAccount.organization_id == user.organization_id, AnalyzedAccount.identifier == identifier).first()
            if acct is None:
                acct = AnalyzedAccount(organization_id=user.organization_id, identifier=identifier, display_name=str(account.get("name") or "")[:200], source=source)
                self.db.add(acct)
                self.db.flush()
            acct.last_analyzed_at = datetime.now(timezone.utc)
            acct.analysis_count += 1
            acct.source = source
        else:
            identifier = "unnamed account"

        row = Prediction(
            organization_id=user.organization_id,
            created_by=user.id,
            account_id=acct.id if acct else None,
            account_identifier=identifier,
            model_id=model_row.id,
            model_name=model_row.name,
            model_version=model_row.version,
            source=source,
            status="COMPLETED",
            prediction=result["prediction"],
            bot_probability=result["bot_probability"],
            human_probability=result["human_probability"],
            risk_score=result["risk_score"],
            risk_band=result["risk_band"],
            features_json=json.dumps(json_safe(result["features"])),
            auxiliary_json=json.dumps(json_safe(result["auxiliary"])),
            input_summary_json=json.dumps(json_safe({k: account.get(k) for k in _INPUT_SUMMARY_KEYS if k in account} | {"n_tweets_supplied": len(account.get("tweets") or [])})),
            inference_ms=float(result["inference_ms"]),
        )
        self.db.add(row)
        self.db.flush()
        for method in ("shap", "lime"):
            payload = result.get(f"{method}_explanation")
            err = result["explanation_errors"].get(method)
            if payload is None and not err and not explain:
                continue
            self.db.add(
                Explanation(
                    organization_id=user.organization_id,
                    prediction_id=row.id,
                    model_id=model_row.id,
                    method=method,
                    status="COMPLETED" if payload else "FAILED",
                    explainer=(payload or {}).get("explainer", ""),
                    output_scale=(payload or {}).get("output_scale", ""),
                    payload_json=json.dumps(json_safe(payload)) if payload else "{}",
                    compute_ms=float(result["explanation_ms"].get(method, 0.0)),
                    error=err,
                )
            )
        self.db.commit()
        audit.record(self.db, "prediction.created", actor=user, target_type="prediction", target_id=row.id, details={"source": source, "model": model_row.id, "prediction": row.prediction}, request=request)
        return self.detail(user.organization_id, row.id)

    # ---- retrieval --------------------------------------------------------------- #

    def _row(self, organization_id: str, prediction_id: str) -> Prediction:
        row = self.db.query(Prediction).filter(Prediction.id == prediction_id, Prediction.organization_id == organization_id).first()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
        return row

    def detail(self, organization_id: str, prediction_id: str) -> dict[str, Any]:
        row = self._row(organization_id, prediction_id)
        exps = {e.method: e for e in row.explanations}
        shap_payload = json.loads(exps["shap"].payload_json) if "shap" in exps and exps["shap"].status == "COMPLETED" else None
        lime_payload = json.loads(exps["lime"].payload_json) if "lime" in exps and exps["lime"].status == "COMPLETED" else None
        errors = {m: e.error for m, e in exps.items() if e.status == "FAILED" and e.error}
        features = json.loads(row.features_json or "{}")
        top = [
            {"feature": c["feature"], "group": c["group"], "description": c.get("description", ""), "value": c["value"], "impact": c["shap"], "direction": c["direction"]}
            for c in (shap_payload or {}).get("contributions", [])[:8]
        ]
        from ml.features import features_by_group

        return json_safe(
            {
                "prediction_id": row.id,
                "account_identifier": row.account_identifier,
                "account_ref": row.account_id,
                "prediction": row.prediction,
                "bot_probability": row.bot_probability,
                "human_probability": row.human_probability,
                "confidence": max(row.bot_probability, row.human_probability),
                "risk_score": row.risk_score,
                "risk_band": row.risk_band,
                "risk_score_note": RISK_SCORE_NOTE,
                "model": {"id": row.model_id, "name": row.model_name, "version": row.model_version},
                "features": features,
                "feature_groups": features_by_group(features),
                "auxiliary": json.loads(row.auxiliary_json or "{}"),
                "input_summary": json.loads(row.input_summary_json or "{}"),
                "top_features": top,
                "shap_explanation": shap_payload,
                "lime_explanation": lime_payload,
                "explanation_errors": errors,
                "explanation_status": {m: e.status for m, e in exps.items()},
                "interpretation": build_interpretation(row.prediction, row.bot_probability, row.risk_score, row.risk_band, shap_payload),
                "source": row.source,
                "status": row.status,
                "batch_id": row.batch_id,
                "label_true": row.label_true,
                "inference_ms": row.inference_ms,
                "created_at": row.created_at,
                "created_by": row.created_by,
            }
        )

    def explanation(self, user: User, prediction_id: str, method: str) -> dict[str, Any]:
        """Stored explanation; computed on demand (and persisted) for batch rows."""
        row = self._row(user.organization_id, prediction_id)
        existing = next((e for e in row.explanations if e.method == method), None)
        if existing and existing.status == "COMPLETED":
            return {"prediction_id": row.id, "method": method, "explanation": json.loads(existing.payload_json), "computed_now": False}
        if row.model_id is None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The model used for this analysis no longer exists")
        model_row = self.models.get(user.organization_id, row.model_id)
        loaded = self.models.load(model_row)
        features = json.loads(row.features_json or "{}")
        if any(f not in features for f in loaded.feature_names):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Stored feature vector is incompatible with the model")
        t0 = time.perf_counter()
        try:
            payload = InferenceEngine.explain_vector(loaded, features, method, lime_samples=self.settings.explanation_lime_samples)
            err = None
        except Exception as exc:  # noqa: BLE001
            payload, err = None, f"{type(exc).__name__}: {exc}"
        ms = (time.perf_counter() - t0) * 1000
        if existing is None:
            existing = Explanation(organization_id=user.organization_id, prediction_id=row.id, model_id=row.model_id, method=method)
            self.db.add(existing)
        existing.status = "COMPLETED" if payload else "FAILED"
        existing.explainer = (payload or {}).get("explainer", "")
        existing.output_scale = (payload or {}).get("output_scale", "")
        existing.payload_json = json.dumps(json_safe(payload)) if payload else "{}"
        existing.compute_ms = ms
        existing.error = err
        self.db.commit()
        if payload is None:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"{method.upper()} explanation could not be computed for this model")
        return {"prediction_id": row.id, "method": method, "explanation": payload, "computed_now": True}

    # ---- history ------------------------------------------------------------------- #

    def history(self, organization_id: str, *, prediction: str | None, min_risk: int | None, model_id: str | None, source: str | None, batch_id: str | None, date_from: datetime | None, date_to: datetime | None, search: str | None, sort: str, page: int, page_size: int) -> dict[str, Any]:
        q = self.db.query(Prediction).filter(Prediction.organization_id == organization_id)
        if prediction:
            q = q.filter(Prediction.prediction == prediction)
        if min_risk is not None:
            q = q.filter(Prediction.risk_score >= min_risk)
        if model_id:
            q = q.filter(Prediction.model_id == model_id)
        if source:
            q = q.filter(Prediction.source == source)
        if batch_id:
            q = q.filter(Prediction.batch_id == batch_id)
        if date_from:
            q = q.filter(Prediction.created_at >= date_from)
        if date_to:
            q = q.filter(Prediction.created_at <= date_to)
        if search:
            q = q.filter(Prediction.account_identifier.ilike(f"%{search[:100]}%"))
        total = q.count()
        col, _, direction = sort.partition(":")
        order_col = {"created_at": Prediction.created_at, "risk_score": Prediction.risk_score, "bot_probability": Prediction.bot_probability, "account": Prediction.account_identifier}.get(col, Prediction.created_at)
        q = q.order_by(order_col.asc() if direction == "asc" else order_col.desc())
        rows = q.offset((page - 1) * page_size).limit(page_size).all()
        return {
            "items": [
                {
                    "id": r.id, "account_identifier": r.account_identifier, "prediction": r.prediction, "bot_probability": r.bot_probability,
                    "risk_score": r.risk_score, "risk_band": r.risk_band, "model_id": r.model_id, "model_name": r.model_name, "model_version": r.model_version,
                    "source": r.source, "status": r.status, "batch_id": r.batch_id, "label_true": r.label_true, "created_at": r.created_at,
                }
                for r in rows
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def delete(self, user: User, prediction_id: str, request: Request | None = None) -> None:
        row = self._row(user.organization_id, prediction_id)
        self.db.delete(row)
        self.db.commit()
        audit.record(self.db, "prediction.deleted", actor=user, target_type="prediction", target_id=prediction_id, request=request)

    # ---- retention ------------------------------------------------------------------- #

    def purge_older_than(self, user: User, days: int, request: Request | None = None) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        q = self.db.query(Prediction).filter(Prediction.organization_id == user.organization_id, Prediction.created_at < cutoff)
        n = q.count()
        for row in q.all():
            self.db.delete(row)
        self.db.commit()
        audit.record(self.db, "retention.predictions_purged", actor=user, details={"older_than_days": days, "deleted": n}, request=request)
        return n

    def counts(self, organization_id: str) -> dict[str, int]:
        total = self.db.query(func.count(Prediction.id)).filter(Prediction.organization_id == organization_id).scalar() or 0
        bots = self.db.query(func.count(Prediction.id)).filter(Prediction.organization_id == organization_id, Prediction.prediction == "BOT").scalar() or 0
        batches = self.db.query(func.count(Batch.id)).filter(Batch.organization_id == organization_id).scalar() or 0
        return {"total": int(total), "bots": int(bots), "batches": int(batches)}
