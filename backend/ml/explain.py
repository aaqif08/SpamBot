"""Explainability: SHAP (global + local) and LIME (local).

Paper alignment:
  * SHAP (§III-E): TreeExplainer for tree ensembles; beeswarm summary of the
    top-20 features ranked by mean |SHAP|; local contributions per account.
  * LIME (§III-D): ``LimeTabularExplainer`` producing Human/Bot prediction
    probabilities and a two-sided list of feature-range rules with weights.

All values returned here are computed by the ``shap`` and ``lime`` libraries on
the actual fitted model — nothing is synthesised.

SHAP output scale: tree models from scikit-learn natively explain class
probability. XGBoost/LightGBM explain log-odds by default; we request
``model_output="probability"`` with an interventional background so that every
model reports contributions on the same probability scale. Non-tree models use
``KernelExplainer`` on ``predict_proba``. The scale actually used is reported in
every payload as ``output_scale``.
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .features import FEATURE_DESCRIPTIONS, FEATURE_GROUP_OF
from .utils import json_safe

log = logging.getLogger(__name__)

TREE_ALGORITHMS: frozenset[str] = frozenset({"random_forest", "extra_trees", "decision_tree", "xgboost", "lightgbm"})
CLASS_NAMES = ["HUMAN", "BOT"]
TOP_N_GLOBAL = 20


def _pre(pipeline: Pipeline) -> Pipeline:
    return Pipeline(steps=pipeline.steps[:-1])


def _clf(pipeline: Pipeline):
    return pipeline.steps[-1][1]


def _positive_class_values(values: Any) -> np.ndarray:
    """Normalise the many shapes ``shap`` returns into an (n, f) array for class 1."""
    if isinstance(values, list):
        arr = np.asarray(values[1] if len(values) > 1 else values[0])
    else:
        arr = np.asarray(values)
        if arr.ndim == 3:
            arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype(float)


def _positive_base_value(expected: Any) -> float:
    if isinstance(expected, (list, tuple, np.ndarray)):
        arr = np.asarray(expected).ravel()
        return float(arr[1] if arr.size > 1 else arr[0])
    return float(expected)


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #


class ShapExplainer:
    """SHAP explanations for a fitted pipeline (imputer → scaler → clf)."""

    def __init__(
        self,
        pipeline: Pipeline,
        feature_names: Sequence[str],
        algorithm: str,
        background_scaled: np.ndarray | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.feature_names = list(feature_names)
        self.algorithm = algorithm
        self.pre = _pre(pipeline)
        self.clf = _clf(pipeline)
        self.background = background_scaled
        self._explainer: Any = None
        self.output_scale = "probability"
        self.explainer_type = ""

    # ---- construction ------------------------------------------------------ #

    def _ensure_explainer(self, background_scaled: np.ndarray | None = None) -> Any:
        if self._explainer is not None:
            return self._explainer
        import shap

        if background_scaled is not None:
            self.background = background_scaled
        if self.background is None:
            raise RuntimeError("A background sample is required to build the SHAP explainer")
        bg = np.asarray(self.background, dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.algorithm in TREE_ALGORITHMS:
                if self.algorithm in ("xgboost", "lightgbm"):
                    try:
                        candidate = shap.TreeExplainer(
                            self.clf, data=bg, feature_perturbation="interventional", model_output="probability"
                        )
                        # Some shap/xgboost combinations only fail at call time; verify now.
                        candidate.shap_values(bg[:1], check_additivity=False)
                        self._explainer = candidate
                        self.output_scale = "probability"
                    except Exception as exc:  # depends on shap / booster versions
                        log.warning("Probability-scale TreeExplainer unavailable for %s (%s); using raw log-odds", self.algorithm, exc)
                        self._explainer = shap.TreeExplainer(self.clf)
                        self.output_scale = "log_odds"
                else:
                    self._explainer = shap.TreeExplainer(self.clf)
                    self.output_scale = "probability"
                self.explainer_type = "TreeExplainer"
            else:
                summary = shap.kmeans(bg, min(10, len(bg))) if len(bg) > 10 else bg
                self._explainer = shap.KernelExplainer(lambda x: self.clf.predict_proba(x)[:, 1], summary)
                self.output_scale = "probability"
                self.explainer_type = "KernelExplainer"
        return self._explainer

    # ---- core --------------------------------------------------------------- #

    def shap_values(self, X_scaled: np.ndarray, nsamples: int | str = "auto") -> tuple[np.ndarray, float]:
        explainer = self._ensure_explainer()
        X_scaled = np.asarray(X_scaled, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.explainer_type == "KernelExplainer":
                values = explainer.shap_values(X_scaled, nsamples=nsamples, silent=True)
            else:
                values = explainer.shap_values(X_scaled, check_additivity=False)
        sv = _positive_class_values(values)
        base = _positive_base_value(explainer.expected_value)
        return sv, base

    def scale(self, X_raw: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X_raw, pd.DataFrame):
            X_raw = X_raw[self.feature_names]
        return np.asarray(self.pre.transform(X_raw), dtype=float)

    # ---- global -------------------------------------------------------------- #

    def global_importance(self, X_raw: pd.DataFrame, sample_size: int = 300, seed: int = 42) -> dict[str, Any]:
        """Mean |SHAP| ranking plus a beeswarm sample for the top features."""
        X_raw = X_raw[self.feature_names]
        n = min(sample_size, len(X_raw))
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_raw), size=n, replace=False)
        sample_raw = X_raw.iloc[idx]
        sample_scaled = self.scale(sample_raw)

        if self.background is None:
            bg_idx = rng.choice(len(X_raw), size=min(100, len(X_raw)), replace=False)
            self.background = self.scale(X_raw.iloc[bg_idx])
        self._ensure_explainer()

        if self.explainer_type == "KernelExplainer":
            # Kernel SHAP is expensive: cap the explained rows and coalitions.
            cap = min(n, 120)
            sample_raw, sample_scaled = sample_raw.iloc[:cap], sample_scaled[:cap]
            sv, base = self.shap_values(sample_scaled, nsamples=200)
        else:
            sv, base = self.shap_values(sample_scaled)

        mean_abs = np.abs(sv).mean(axis=0)
        mean_signed = sv.mean(axis=0)
        order = np.argsort(-mean_abs)
        importance = [
            {
                "feature": self.feature_names[i],
                "group": FEATURE_GROUP_OF.get(self.feature_names[i], ""),
                "mean_abs_shap": float(mean_abs[i]),
                "mean_shap": float(mean_signed[i]),
                "rank": r + 1,
            }
            for r, i in enumerate(order)
        ]
        top = [self.feature_names[i] for i in order[:TOP_N_GLOBAL]]
        max_points = min(len(sv), 150)
        beeswarm = []
        for i in order[:TOP_N_GLOBAL]:
            name = self.feature_names[i]
            beeswarm.append(
                {
                    "feature": name,
                    "points": [
                        {
                            "shap": float(sv[j, i]),
                            "value_scaled": float(sample_scaled[j, i]),
                            "value": float(sample_raw.iloc[j][name]),
                        }
                        for j in range(max_points)
                    ],
                }
            )
        group_importance: dict[str, float] = {}
        for i, name in enumerate(self.feature_names):
            g = FEATURE_GROUP_OF.get(name, "other")
            group_importance[g] = group_importance.get(g, 0.0) + float(mean_abs[i])
        return json_safe(
            {
                "explainer": self.explainer_type,
                "output_scale": self.output_scale,
                "base_value": base,
                "n_samples": int(len(sv)),
                "importance": importance,
                "top_features": top,
                "group_importance": group_importance,
                "beeswarm": beeswarm,
            }
        )

    # ---- local --------------------------------------------------------------- #

    def local(self, x_raw: Sequence[float] | np.ndarray) -> dict[str, Any]:
        """Local explanation for one account (raw feature vector in canonical order)."""
        x_raw = np.asarray(x_raw, dtype=float).reshape(1, -1)
        x_scaled = self.scale(x_raw)
        sv, base = self.shap_values(x_scaled)
        contrib = sv[0]
        order = np.argsort(-np.abs(contrib))
        contributions = []
        cumulative = base
        for i in order:
            cumulative += float(contrib[i])
            contributions.append(
                {
                    "feature": self.feature_names[i],
                    "group": FEATURE_GROUP_OF.get(self.feature_names[i], ""),
                    "description": FEATURE_DESCRIPTIONS.get(self.feature_names[i], ""),
                    "value": float(x_raw[0, i]),
                    "value_scaled": float(x_scaled[0, i]),
                    "shap": float(contrib[i]),
                    "direction": "BOT" if contrib[i] > 0 else ("HUMAN" if contrib[i] < 0 else "NEUTRAL"),
                    "cumulative": float(cumulative),
                }
            )
        model_output = float(base + contrib.sum())
        return json_safe(
            {
                "explainer": self.explainer_type,
                "output_scale": self.output_scale,
                "base_value": float(base),
                "model_output": model_output,
                "sum_positive": float(contrib[contrib > 0].sum()),
                "sum_negative": float(contrib[contrib < 0].sum()),
                "contributions": contributions,
            }
        )


# --------------------------------------------------------------------------- #
# LIME
# --------------------------------------------------------------------------- #

_RULE_FEATURE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _feature_from_rule(rule: str, names: Sequence[str]) -> str:
    """Extract the feature name from a LIME rule such as ``0.00 < ffratio <= 0.01``."""
    candidates = _RULE_FEATURE_RE.findall(rule)
    for c in candidates:
        if c in names:
            return c
    return candidates[0] if candidates else rule


class LimeExplainer:
    """LIME tabular explanations on the scaled feature space (as in the paper's figures)."""

    def __init__(
        self,
        pipeline: Pipeline,
        feature_names: Sequence[str],
        training_scaled: np.ndarray,
        seed: int = 42,
        num_samples: int = 3000,
    ) -> None:
        from lime.lime_tabular import LimeTabularExplainer

        self.pipeline = pipeline
        self.feature_names = list(feature_names)
        self.pre = _pre(pipeline)
        self.clf = _clf(pipeline)
        self.num_samples = num_samples
        self.seed = seed
        self._explainer = LimeTabularExplainer(
            training_data=np.asarray(training_scaled, dtype=float),
            feature_names=self.feature_names,
            class_names=CLASS_NAMES,
            mode="classification",
            discretize_continuous=True,
            random_state=seed,
        )

    def explain(self, x_raw: Sequence[float] | np.ndarray, num_features: int | None = None) -> dict[str, Any]:
        x_raw = np.asarray(x_raw, dtype=float).reshape(1, -1)
        x_scaled = np.asarray(self.pre.transform(x_raw), dtype=float)[0]
        k = num_features or len(self.feature_names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = self._explainer.explain_instance(
                x_scaled,
                self.clf.predict_proba,
                labels=(1,),
                num_features=k,
                num_samples=self.num_samples,
            )
        proba = [float(p) for p in exp.predict_proba]
        items = []
        for rule, weight in exp.as_list(label=1):
            name = _feature_from_rule(rule, self.feature_names)
            idx = self.feature_names.index(name) if name in self.feature_names else -1
            items.append(
                {
                    "feature": name,
                    "group": FEATURE_GROUP_OF.get(name, ""),
                    "description": FEATURE_DESCRIPTIONS.get(name, ""),
                    "rule": rule,
                    "weight": float(weight),
                    "direction": "BOT" if weight > 0 else ("HUMAN" if weight < 0 else "NEUTRAL"),
                    "value": float(x_raw[0, idx]) if idx >= 0 else None,
                    "value_scaled": float(x_scaled[idx]) if idx >= 0 else None,
                }
            )
        items.sort(key=lambda d: -abs(d["weight"]))
        bot_items = [i for i in items if i["weight"] > 0]
        human_items = [i for i in items if i["weight"] < 0]
        local_pred = exp.local_pred
        return json_safe(
            {
                "class_names": CLASS_NAMES,
                "prediction_probabilities": {"HUMAN": proba[0], "BOT": proba[1]},
                "intercept": float(exp.intercept[1]) if isinstance(exp.intercept, dict) else None,
                "local_prediction": float(np.asarray(local_pred).ravel()[0]) if local_pred is not None else None,
                "surrogate_r2": float(exp.score) if exp.score is not None else None,
                "num_samples": self.num_samples,
                "bot_contribution": float(sum(i["weight"] for i in bot_items)),
                "human_contribution": float(-sum(i["weight"] for i in human_items)),
                "bot_indicators": bot_items,
                "human_indicators": human_items,
                "items": items,
            }
        )
