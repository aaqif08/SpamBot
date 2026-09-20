"""Training pipeline: model zoo, stratified split, k-fold CV, randomised
hyperparameter search, evaluation, SHAP analysis and artefact persistence.

Paper alignment (§III, §IV):
  * nine classifiers (Table 5/6)
  * stratified train/test split + stratified k-fold cross-validation
  * hyperparameters "optimised through cross-validation" — grids are not
    published, so compact randomised searches are used
  * SHAP-based feature selection (rank by mean |SHAP|, keep top-k) — optional
  * features min–max scaled (values in the paper's LIME figures lie in [0, 1])

Everything is wrapped in a single sklearn ``Pipeline`` so the same artefact is
used for prediction, SHAP and LIME.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import FEATURE_VERSION
from .evaluation import classification_metrics, full_evaluation
from .explain import ShapExplainer, TREE_ALGORITHMS
from .features import FEATURE_GROUP_OF, FEATURE_NAMES
from .model_registry import ModelCard, write_checksums
from .utils import DEFAULT_SEED, json_safe, set_seed, timer, utc_now_iso

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float, str], None]

# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    family: str
    build: Callable[[int], BaseEstimator]
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    supports_tree_shap: bool = False


def _xgb(seed: int) -> BaseEstimator:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def _lgbm(seed: int) -> BaseEstimator:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )


MODEL_ZOO: dict[str, ModelSpec] = {
    "random_forest": ModelSpec(
        key="random_forest",
        display_name="Random Forest",
        family="ensemble-bagging",
        build=lambda s: RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1),
        search_space={
            "clf__n_estimators": [200, 300, 500],
            "clf__max_depth": [None, 10, 20, 40],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", 0.5],
        },
        supports_tree_shap=True,
    ),
    "svm": ModelSpec(
        key="svm",
        display_name="SVM",
        family="kernel",
        build=lambda s: SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=s),
        search_space={"clf__C": [0.3, 1, 3, 10, 30], "clf__gamma": ["scale", 0.1, 0.01]},
    ),
    "decision_tree": ModelSpec(
        key="decision_tree",
        display_name="Decision Tree",
        family="tree",
        build=lambda s: DecisionTreeClassifier(random_state=s),
        search_space={
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_leaf": [1, 2, 5, 10],
            "clf__criterion": ["gini", "entropy"],
        },
        supports_tree_shap=True,
    ),
    "xgboost": ModelSpec(
        key="xgboost",
        display_name="XGBoost",
        family="ensemble-boosting",
        build=_xgb,
        search_space={
            "clf__n_estimators": [200, 300, 500],
            "clf__max_depth": [3, 4, 6, 8],
            "clf__learning_rate": [0.03, 0.1, 0.2],
            "clf__subsample": [0.8, 1.0],
        },
        supports_tree_shap=True,
    ),
    "lightgbm": ModelSpec(
        key="lightgbm",
        display_name="LightGBM",
        family="ensemble-boosting",
        build=_lgbm,
        search_space={
            "clf__n_estimators": [200, 300, 500],
            "clf__num_leaves": [15, 31, 63],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__min_child_samples": [10, 20, 40],
        },
        supports_tree_shap=True,
    ),
    "logistic_regression": ModelSpec(
        key="logistic_regression",
        display_name="Logistic Regression",
        family="linear",
        build=lambda s: LogisticRegression(max_iter=2000, random_state=s),
        search_space={"clf__C": [0.03, 0.1, 0.3, 1, 3, 10], "clf__penalty": ["l2"], "clf__solver": ["lbfgs"]},
    ),
    "extra_trees": ModelSpec(
        key="extra_trees",
        display_name="Extra Trees",
        family="ensemble-bagging",
        build=lambda s: ExtraTreesClassifier(n_estimators=300, random_state=s, n_jobs=-1),
        search_space={
            "clf__n_estimators": [200, 300, 500],
            "clf__max_depth": [None, 10, 20, 40],
            "clf__min_samples_leaf": [1, 2, 4],
        },
        supports_tree_shap=True,
    ),
    "naive_bayes": ModelSpec(
        key="naive_bayes",
        display_name="Naive Bayes",
        family="probabilistic",
        build=lambda s: GaussianNB(),
        search_space={"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]},
    ),
    "adaboost": ModelSpec(
        key="adaboost",
        display_name="AdaBoost",
        family="ensemble-boosting",
        build=lambda s: AdaBoostClassifier(n_estimators=200, random_state=s),
        search_space={"clf__n_estimators": [100, 200, 400], "clf__learning_rate": [0.3, 0.5, 1.0]},
    ),
}

assert set(MODEL_ZOO) == set(TREE_ALGORITHMS) | {"svm", "logistic_regression", "naive_bayes", "adaboost"}


def model_choices() -> list[dict[str, Any]]:
    out = []
    for spec in MODEL_ZOO.values():
        available = True
        reason = ""
        if spec.key in ("xgboost", "lightgbm"):
            try:
                spec.build(0)
            except Exception as exc:  # pragma: no cover - depends on env
                available = False
                reason = f"{type(exc).__name__}: {exc}"
        out.append(
            {
                "key": spec.key,
                "display_name": spec.display_name,
                "family": spec.family,
                "available": available,
                "unavailable_reason": reason,
                "supports_tree_shap": spec.supports_tree_shap,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def build_pipeline(algorithm: str, seed: int = DEFAULT_SEED) -> Pipeline:
    spec = MODEL_ZOO[algorithm]
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
            ("clf", spec.build(seed)),
        ]
    )


def preprocessor_of(pipeline: Pipeline) -> Pipeline:
    """The imputer → scaler part of a fitted pipeline."""
    return Pipeline(steps=pipeline.steps[:-1])


def classifier_of(pipeline: Pipeline) -> ClassifierMixin:
    return pipeline.steps[-1][1]


# --------------------------------------------------------------------------- #
# Training configuration / result
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    algorithm: str = "lightgbm"
    test_size: float = 0.25
    cv_folds: int = 5
    seed: int = DEFAULT_SEED
    hyperparameter_search: bool = True
    search_iterations: int = 8
    feature_selection: bool = False
    feature_selection_top_k: int = 31
    shap_sample_size: int = 300
    lime_sample_size: int = 2000
    background_size: int = 100
    notes: str = ""

    def validate(self) -> None:
        if self.algorithm not in MODEL_ZOO:
            raise ValueError(f"Unknown algorithm '{self.algorithm}'. Choose from {sorted(MODEL_ZOO)}")
        if not 0.05 <= self.test_size <= 0.5:
            raise ValueError("test_size must be between 0.05 and 0.5")
        if not 2 <= self.cv_folds <= 10:
            raise ValueError("cv_folds must be between 2 and 10")
        if not 1 <= self.feature_selection_top_k <= len(FEATURE_NAMES):
            raise ValueError("feature_selection_top_k out of range")


@dataclass
class TrainResult:
    card: ModelCard
    metrics: dict[str, Any]
    shap_global: dict[str, Any]
    artifact_dir: Path
    checksums: dict[str, str]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def _noop_progress(stage: str, progress: float, message: str) -> None:  # pragma: no cover
    log.info("[%s %.0f%%] %s", stage, progress * 100, message)


def _cross_validate(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, folds: int, seed: int) -> dict[str, Any]:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    res = cross_validate(clone(pipeline), X, y, cv=skf, scoring=scoring, n_jobs=1, return_train_score=False)
    folds_out = []
    for i in range(folds):
        folds_out.append({k: float(res[f"test_{k}"][i]) for k in scoring})
    summary = {k: {"mean": float(np.mean(res[f"test_{k}"])), "std": float(np.std(res[f"test_{k}"]))} for k in scoring}
    return {"folds": folds_out, "summary": summary, "n_folds": folds, "fit_time_mean": float(np.mean(res["fit_time"]))}


def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    config: TrainConfig,
    artifact_dir: Path,
    progress: ProgressFn | None = None,
) -> TrainResult:
    """Full training run. ``X`` must contain the canonical feature columns.

    Artefacts are written into ``artifact_dir`` (a fresh local directory); the
    caller is responsible for persisting them (storage layer + database).
    """
    progress = progress or _noop_progress
    config.validate()
    set_seed(config.seed)
    with warnings.catch_warnings():
        # LightGBM / sklearn emit many benign feature-name warnings during CV;
        # printing thousands of them dominates runtime.
        warnings.simplefilter("ignore")
        return _train_model_impl(X, y, config, Path(artifact_dir), progress)


def _train_model_impl(
    X: pd.DataFrame,
    y: np.ndarray,
    config: TrainConfig,
    artifact_dir: Path,
    progress: ProgressFn,
) -> TrainResult:

    feature_names = [f for f in FEATURE_NAMES if f in X.columns]
    missing = [f for f in FEATURE_NAMES if f not in X.columns]
    if missing:
        raise ValueError(f"Training data is missing features: {missing}")
    X = X[feature_names].astype(float)
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Training data must contain both HUMAN (0) and BOT (1) labels")
    if len(y) < 20:
        raise ValueError("At least 20 labelled accounts are required to train")

    # Features that are constant in the training data carry no information (e.g. the
    # tweet-derived features when only user-level Cresci files are available). They are
    # dropped and recorded so the model card states exactly which features were used.
    constant_features = [f for f in feature_names if X[f].nunique(dropna=False) <= 1]
    if constant_features and len(constant_features) < len(feature_names):
        feature_names = [f for f in feature_names if f not in constant_features]
        X = X[feature_names]
        progress("preprocessing", 0.03, f"Dropped {len(constant_features)} constant feature(s): {', '.join(constant_features)}")

    progress("preprocessing", 0.05, f"{len(y)} accounts, {len(feature_names)} features; stratified split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.seed, shuffle=True
    )

    spec = MODEL_ZOO[config.algorithm]
    pipeline = build_pipeline(config.algorithm, config.seed)

    # ---- optional SHAP-based feature selection (paper §III-C) ------------- #
    selection_info: dict[str, Any] = {"enabled": False}
    if config.feature_selection and config.feature_selection_top_k < len(feature_names):
        progress("feature_engineering", 0.12, "SHAP feature selection: fitting screening model")
        screen = clone(pipeline).fit(X_train, y_train)
        ranking = ShapExplainer(screen, feature_names, config.algorithm).global_importance(
            X_train, sample_size=min(config.shap_sample_size, len(X_train)), seed=config.seed
        )
        ranked = [r["feature"] for r in ranking["importance"]]
        selected = ranked[: config.feature_selection_top_k]
        dropped = [f for f in feature_names if f not in selected]
        feature_names = [f for f in FEATURE_NAMES if f in selected]  # keep canonical order
        X_train, X_test = X_train[feature_names], X_test[feature_names]
        selection_info = {"enabled": True, "top_k": config.feature_selection_top_k, "selected": feature_names, "dropped": dropped}
        progress("feature_engineering", 0.18, f"Selected {len(feature_names)} features by mean |SHAP|")
    else:
        progress("feature_engineering", 0.15, "Using the full 31-feature set")

    # ---- hyperparameter search --------------------------------------------- #
    best_params: dict[str, Any] = {}
    skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    with timer() as t_train:
        if config.hyperparameter_search and spec.search_space:
            n_iter = min(config.search_iterations, int(np.prod([len(v) for v in spec.search_space.values()])))
            progress("training", 0.25, f"Randomised search ({n_iter} candidates × {config.cv_folds} folds)")
            search = RandomizedSearchCV(
                pipeline,
                spec.search_space,
                n_iter=n_iter,
                cv=skf,
                scoring="f1",
                random_state=config.seed,
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            pipeline = search.best_estimator_
            best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        else:
            progress("training", 0.25, f"Fitting {spec.display_name}")
            pipeline.fit(X_train, y_train)

    progress("cross_validation", 0.55, f"{config.cv_folds}-fold stratified cross-validation")
    cv = _cross_validate(pipeline, X_train, y_train, config.cv_folds, config.seed)

    progress("cross_validation", 0.7, "Hold-out evaluation")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    holdout = full_evaluation(y_test, y_pred, y_proba)
    train_metrics = classification_metrics(y_train, pipeline.predict(X_train), pipeline.predict_proba(X_train)[:, 1])

    # ---- SHAP analysis ------------------------------------------------------ #
    progress("shap_analysis", 0.78, "Computing global SHAP importance")
    explainer = ShapExplainer(pipeline, feature_names, config.algorithm)
    shap_global = explainer.global_importance(
        X_train, sample_size=min(config.shap_sample_size, len(X_train)), seed=config.seed
    )

    # ---- persist ------------------------------------------------------------ #
    progress("saving", 0.9, "Saving artefacts")
    model_dir = artifact_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    pre = preprocessor_of(pipeline)
    X_train_scaled = pre.transform(X_train)
    rng = np.random.RandomState(config.seed)
    bg_idx = rng.choice(len(X_train_scaled), size=min(config.background_size, len(X_train_scaled)), replace=False)
    lime_idx = rng.choice(len(X_train_scaled), size=min(config.lime_sample_size, len(X_train_scaled)), replace=False)
    np.save(model_dir / "background.npy", X_train_scaled[bg_idx])
    np.save(model_dir / "lime_sample.npy", X_train_scaled[lime_idx])
    np.save(model_dir / "lime_sample_raw.npy", X_train.values[lime_idx])

    joblib.dump(pipeline, model_dir / "pipeline.joblib")
    joblib.dump(pre, model_dir / "scaler.joblib")

    feature_metadata = {
        "feature_version": FEATURE_VERSION,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "groups": {g: [f for f in fs if f in feature_names] for g, fs in _groups()},
        "feature_selection": selection_info,
        "dropped_constant_features": constant_features,
        "scaling": "min-max (fit on training split)",
        "imputation": "median (fit on training split)",
        "raw_feature_ranges": {
            f: {"min": float(X_train[f].min()), "max": float(X_train[f].max()), "median": float(X_train[f].median())}
            for f in feature_names
        },
    }
    (model_dir / "feature_metadata.json").write_text(json.dumps(json_safe(feature_metadata), indent=2), encoding="utf-8")

    metrics = {
        "holdout": holdout,
        "train": train_metrics,
        "cross_validation": cv,
        "split": {
            "test_size": config.test_size,
            "train_size": int(len(y_train)),
            "test_size_n": int(len(y_test)),
            "cv_folds": config.cv_folds,
            "stratified": True,
            "seed": config.seed,
        },
        "class_distribution": {
            "train": {"human": int((y_train == 0).sum()), "bot": int((y_train == 1).sum())},
            "test": {"human": int((y_test == 0).sum()), "bot": int((y_test == 1).sum())},
        },
        "training_seconds": t_train["seconds"],
        "best_params": best_params,
        "hyperparameter_search": bool(config.hyperparameter_search and spec.search_space),
    }
    (model_dir / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2), encoding="utf-8")
    (model_dir / "shap_global.json").write_text(json.dumps(json_safe(shap_global), indent=2), encoding="utf-8")

    checksums = write_checksums(model_dir)
    card = ModelCard(
        name=spec.display_name,
        algorithm=config.algorithm,
        feature_version=FEATURE_VERSION,
        feature_names=feature_names,
        holdout_metrics=holdout["metrics"],
        cv_metrics={k: v["mean"] for k, v in cv["summary"].items()},
        cv_std={k: v["std"] for k, v in cv["summary"].items()},
        params={"config": {k: v for k, v in config.__dict__.items()}, "best_params": best_params},
        training_seconds=t_train["seconds"],
        trained_at=utc_now_iso(),
        notes=config.notes,
    )
    progress("completed", 1.0, f"{spec.display_name} trained ({len(feature_names)} features)")
    return TrainResult(card=card, metrics=metrics, shap_global=shap_global, artifact_dir=model_dir, checksums=checksums)


def _groups():
    from .features import FEATURE_GROUPS

    return FEATURE_GROUPS.items()


def train_all_models(
    X: pd.DataFrame,
    y: np.ndarray,
    base: TrainConfig,
    artifact_root: Path,
    algorithms: list[str] | None = None,
    progress: ProgressFn | None = None,
) -> list[TrainResult]:
    """Train every classifier in the zoo; each into ``artifact_root/<algorithm>``."""
    results: list[TrainResult] = []
    algos = algorithms or list(MODEL_ZOO)
    for i, algo in enumerate(algos):
        cfg = TrainConfig(**{**base.__dict__, "algorithm": algo})
        if progress:
            progress("training", i / len(algos), f"Training {MODEL_ZOO[algo].display_name}")
        try:
            results.append(train_model(X, y, cfg, Path(artifact_root) / algo, progress=None))
        except Exception as exc:
            log.exception("Training %s failed: %s", algo, exc)
    return results
