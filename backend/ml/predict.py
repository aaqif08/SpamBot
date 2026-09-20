"""Prediction engine: loads a trained artefact directory (after checksum
verification) and turns account data into predictions, risk scores and
explanations.

Risk score
----------
``risk_score = round(100 × P(bot))``. It is an application-level presentation
of the model's estimated bot probability — **not** a definitive statement that
an account is malicious.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .explain import LimeExplainer, ShapExplainer
from .features import FEATURE_DESCRIPTIONS, FEATURE_GROUP_OF, FeatureExtractor, features_by_group
from .model_registry import verify_checksums
from .utils import json_safe

log = logging.getLogger(__name__)

RISK_BANDS: list[tuple[int, str]] = [(80, "critical"), (60, "high"), (40, "medium"), (20, "low"), (0, "minimal")]


def risk_band(score: int) -> str:
    for threshold, name in RISK_BANDS:
        if score >= threshold:
            return name
    return "minimal"


def risk_score_from_probability(p_bot: float) -> int:
    return int(round(max(0.0, min(1.0, float(p_bot))) * 100))


class ModelNotAvailableError(RuntimeError):
    """Raised when no production model is configured."""


@dataclass
class LoadedModel:
    model_id: str
    name: str
    version: int
    algorithm: str
    pipeline: Any
    feature_names: list[str]
    feature_metadata: dict[str, Any]
    background: np.ndarray
    lime_sample: np.ndarray
    shap_global: dict[str, Any]
    metrics: dict[str, Any]
    artifact_dir: Path
    _shap: ShapExplainer | None = None
    _lime: LimeExplainer | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def feature_version(self) -> str:
        return str(self.feature_metadata.get("feature_version", ""))

    def shap(self) -> ShapExplainer:
        with self._lock:
            if self._shap is None:
                self._shap = ShapExplainer(self.pipeline, self.feature_names, self.algorithm, self.background)
            return self._shap

    def lime(self, num_samples: int = 3000) -> LimeExplainer:
        with self._lock:
            if self._lime is None:
                self._lime = LimeExplainer(self.pipeline, self.feature_names, self.lime_sample, num_samples=num_samples)
            return self._lime

    def info(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "name": self.name,
            "version": self.version,
            "algorithm": self.algorithm,
            "feature_version": self.feature_version,
            "n_features": len(self.feature_names),
        }


def load_model_dir(
    artifact_dir: Path,
    model_id: str,
    name: str,
    version: int,
    algorithm: str,
    checksums: Mapping[str, str],
) -> LoadedModel:
    """Load artefacts from a directory; refuses to load if checksums do not match."""
    artifact_dir = Path(artifact_dir)
    verify_checksums(artifact_dir, dict(checksums))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = joblib.load(artifact_dir / "pipeline.joblib")
    feature_metadata = json.loads((artifact_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    shap_path = artifact_dir / "shap_global.json"
    shap_global = json.loads(shap_path.read_text(encoding="utf-8")) if shap_path.exists() else {}
    return LoadedModel(
        model_id=model_id,
        name=name,
        version=version,
        algorithm=algorithm,
        pipeline=pipeline,
        feature_names=list(feature_metadata["feature_names"]),
        feature_metadata=feature_metadata,
        background=np.load(artifact_dir / "background.npy"),
        lime_sample=np.load(artifact_dir / "lime_sample.npy"),
        shap_global=shap_global,
        metrics=metrics,
        artifact_dir=artifact_dir,
    )


class InferenceEngine:
    """Stateless inference helpers over a :class:`LoadedModel`."""

    @staticmethod
    def predict_account(
        model: LoadedModel,
        account: Mapping[str, Any],
        explain: bool = True,
        top_n: int = 8,
        lime_samples: int = 3000,
    ) -> dict[str, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return InferenceEngine._predict_impl(model, account, explain, top_n, lime_samples)

    @staticmethod
    def _predict_impl(model: LoadedModel, account: Mapping[str, Any], explain: bool, top_n: int, lime_samples: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        extractor = FeatureExtractor(model.feature_names)
        result = extractor.transform(account)
        x = result.vector(model.feature_names)
        proba = model.pipeline.predict_proba(x.reshape(1, -1))[0]
        p_bot = float(proba[1])
        p_human = float(proba[0])
        score = risk_score_from_probability(p_bot)
        inference_ms = (time.perf_counter() - t0) * 1000

        out: dict[str, Any] = {
            "prediction": "BOT" if p_bot >= 0.5 else "HUMAN",
            "bot_probability": p_bot,
            "human_probability": p_human,
            "confidence": max(p_bot, p_human),
            "risk_score": score,
            "risk_band": risk_band(score),
            "model": model.info(),
            "features": result.features,
            "feature_groups": features_by_group(result.features),
            "auxiliary": result.auxiliary,
            "inference_ms": inference_ms,
            "shap_explanation": None,
            "lime_explanation": None,
            "top_features": [],
            "explanation_errors": {},
            "explanation_ms": {},
        }

        if explain:
            t1 = time.perf_counter()
            try:
                shap_local = model.shap().local(x)
                out["shap_explanation"] = shap_local
                out["top_features"] = [
                    {
                        "feature": c["feature"],
                        "group": c["group"],
                        "description": FEATURE_DESCRIPTIONS.get(c["feature"], ""),
                        "value": c["value"],
                        "impact": c["shap"],
                        "direction": c["direction"],
                    }
                    for c in shap_local["contributions"][:top_n]
                ]
            except Exception as exc:  # noqa: BLE001 - explanation failure must not fail prediction
                log.exception("SHAP explanation failed")
                out["explanation_errors"]["shap"] = f"{type(exc).__name__}: {exc}"
            out["explanation_ms"]["shap"] = (time.perf_counter() - t1) * 1000
            t2 = time.perf_counter()
            try:
                out["lime_explanation"] = model.lime(lime_samples).explain(x)
            except Exception as exc:  # noqa: BLE001
                log.exception("LIME explanation failed")
                out["explanation_errors"]["lime"] = f"{type(exc).__name__}: {exc}"
            out["explanation_ms"]["lime"] = (time.perf_counter() - t2) * 1000

        if not out["top_features"]:
            ranked = model.shap_global.get("importance", [])[:top_n]
            out["top_features"] = [
                {
                    "feature": r["feature"],
                    "group": r.get("group", FEATURE_GROUP_OF.get(r["feature"], "")),
                    "description": FEATURE_DESCRIPTIONS.get(r["feature"], ""),
                    "value": result.features.get(r["feature"]),
                    "impact": None,
                    "direction": "GLOBAL",
                }
                for r in ranked
            ]
        return json_safe(out)

    @staticmethod
    def predict_frame(model: LoadedModel, df: pd.DataFrame) -> pd.DataFrame:
        extractor = FeatureExtractor(model.feature_names)
        feats = extractor.transform_frame(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = model.pipeline.predict_proba(feats[model.feature_names].values)
        p_bot = proba[:, 1]
        out = pd.DataFrame(index=df.index)
        out["prediction"] = np.where(p_bot >= 0.5, "BOT", "HUMAN")
        out["bot_probability"] = p_bot
        out["human_probability"] = proba[:, 0]
        out["risk_score"] = [risk_score_from_probability(p) for p in p_bot]
        out["risk_band"] = [risk_band(s) for s in out["risk_score"]]
        for f in model.feature_names:
            out[f] = feats[f].values
        return out

    @staticmethod
    def predict_vectors(model: LoadedModel, X: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.pipeline.predict_proba(X[model.feature_names].values)[:, 1]

    @staticmethod
    def explain_vector(model: LoadedModel, features: Mapping[str, float], method: str, lime_samples: int = 3000) -> dict[str, Any]:
        x: Sequence[float] = [float(features[f]) for f in model.feature_names]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.shap().local(x) if method == "shap" else model.lime(lime_samples).explain(x)
