"""Training, prediction, SHAP and LIME tests (ML package, no HTTP)."""

from __future__ import annotations

import numpy as np
import pytest

from ml.explain import LimeExplainer, ShapExplainer
from ml.features import FEATURE_NAMES
from ml.predict import ModelNotAvailableError, Predictor, risk_band, risk_score_from_probability
from ml.model_registry import ModelRegistry
from ml.train import MODEL_ZOO, TrainConfig, model_choices, train_model


def test_model_zoo_has_paper_classifiers():
    assert set(MODEL_ZOO) == {
        "random_forest", "svm", "decision_tree", "xgboost", "lightgbm",
        "logistic_regression", "extra_trees", "naive_bayes", "adaboost",
    }
    assert all(c["available"] for c in model_choices())


def test_train_config_validation():
    with pytest.raises(ValueError):
        TrainConfig(algorithm="nope").validate()
    with pytest.raises(ValueError):
        TrainConfig(test_size=0.9).validate()
    with pytest.raises(ValueError):
        TrainConfig(cv_folds=1).validate()


def test_training_produces_registry_entry_and_artefacts(trained_model, registry):
    entry = trained_model.entry
    assert registry.get(entry.id) is not None
    assert registry.active_id() == entry.id
    d = trained_model.model_dir
    for name in ("pipeline.joblib", "scaler.joblib", "feature_metadata.json", "metrics.json", "shap_global.json", "background.npy", "lime_sample.npy"):
        assert (d / name).exists(), name
    assert (registry.root / "best_model.joblib").exists()
    m = entry.metrics["holdout"]
    assert 0.5 <= m["accuracy"] <= 1.0
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    cv = trained_model.metrics["cross_validation"]
    assert cv["n_folds"] == 3 and len(cv["folds"]) == 3
    assert entry.n_features == 31 and entry.feature_names == FEATURE_NAMES
    assert entry.is_demo is True
    assert "path" not in entry.public()


def test_training_rejects_single_class(demo_frame, registry):
    X = demo_frame[FEATURE_NAMES]
    y = np.zeros(len(X), dtype=int)
    with pytest.raises(ValueError):
        train_model(X, y, TrainConfig(algorithm="decision_tree", hyperparameter_search=False), registry)


def test_shap_feature_selection_reduces_features(demo_frame, tmp_path):
    reg = ModelRegistry(tmp_path / "models")
    cfg = TrainConfig(
        algorithm="decision_tree",
        hyperparameter_search=False,
        cv_folds=2,
        feature_selection=True,
        feature_selection_top_k=12,
        shap_sample_size=40,
        activate=True,
    )
    res = train_model(demo_frame[FEATURE_NAMES], demo_frame["label"].values, cfg, reg)
    assert res.entry.n_features == 12
    meta = res.metrics
    assert meta["split"]["cv_folds"] == 2
    pred = Predictor(reg)
    out = pred.predict_account({"followers_count": 5, "friends_count": 900, "statuses_count": 3, "default_profile": True})
    assert out["model"]["n_features"] == 12
    assert len(out["features"]) == 12


def test_global_shap_structure(trained_model):
    g = trained_model.shap_global
    assert g["explainer"] == "TreeExplainer"
    assert g["output_scale"] == "probability"
    assert len(g["importance"]) == 31
    assert g["importance"][0]["rank"] == 1
    assert g["importance"][0]["mean_abs_shap"] >= g["importance"][-1]["mean_abs_shap"]
    assert len(g["top_features"]) == 20
    assert len(g["beeswarm"]) == 20 and len(g["beeswarm"][0]["points"]) > 0
    assert set(g["group_importance"]) == {"user_profile", "content", "engagement", "linguistic", "profile_attributes", "sentiment"}


def test_local_shap_is_additive(trained_model):
    from ml.predict import load_model

    loaded = load_model(ModelRegistry(trained_model.model_dir.parent), trained_model.model_id)
    x = np.zeros(31)
    x[FEATURE_NAMES.index("friends_count")] = 800
    x[FEATURE_NAMES.index("statuses_count")] = 5
    local = loaded.shap().local(x)
    total = local["base_value"] + sum(c["shap"] for c in local["contributions"])
    assert total == pytest.approx(local["model_output"], abs=1e-6)
    p = loaded.pipeline.predict_proba(x.reshape(1, -1))[0, 1]
    # Tree SHAP on a scikit-learn forest is exact on the probability scale.
    assert local["model_output"] == pytest.approx(p, abs=1e-4)
    assert {c["direction"] for c in local["contributions"]} <= {"BOT", "HUMAN", "NEUTRAL"}


def test_kernel_shap_for_non_tree_model(demo_frame, tmp_path):
    reg = ModelRegistry(tmp_path / "models")
    cfg = TrainConfig(algorithm="logistic_regression", hyperparameter_search=False, cv_folds=2, shap_sample_size=20, background_size=20)
    res = train_model(demo_frame[FEATURE_NAMES], demo_frame["label"].values, cfg, reg)
    assert res.shap_global["explainer"] == "KernelExplainer"
    pred = Predictor(reg)
    out = pred.predict_account({"followers_count": 5, "friends_count": 900, "statuses_count": 3})
    assert out["shap_explanation"]["explainer"] == "KernelExplainer"
    assert out["lime_explanation"] is not None


def test_lime_explanation_structure(trained_model):
    from ml.predict import load_model

    loaded = load_model(ModelRegistry(trained_model.model_dir.parent), trained_model.model_id)
    lime = LimeExplainer(loaded.pipeline, loaded.feature_names, loaded.lime_sample, num_samples=500)
    x = np.zeros(31)
    x[FEATURE_NAMES.index("hashtag_count")] = 500
    out = lime.explain(x)
    probs = out["prediction_probabilities"]
    assert probs["HUMAN"] + probs["BOT"] == pytest.approx(1.0, abs=1e-6)
    assert len(out["items"]) == 31
    assert all(i["feature"] in FEATURE_NAMES for i in out["items"])
    assert all(i["weight"] > 0 for i in out["bot_indicators"])
    assert all(i["weight"] < 0 for i in out["human_indicators"])
    assert out["bot_contribution"] >= 0 and out["human_contribution"] >= 0


def test_predictor_and_risk_score(trained_model, registry):
    pred = Predictor(registry)
    out = pred.predict_account({"account_id": "x", "followers_count": 3, "friends_count": 1500, "statuses_count": 8, "default_profile": True, "default_profile_image": True})
    assert out["prediction"] in ("BOT", "HUMAN")
    assert out["bot_probability"] + out["human_probability"] == pytest.approx(1.0, abs=1e-6)
    assert out["risk_score"] == round(out["bot_probability"] * 100)
    assert out["risk_band"] == risk_band(out["risk_score"])
    assert out["shap_explanation"] is not None and out["lime_explanation"] is not None
    assert out["explanation_errors"] == {}
    assert len(out["top_features"]) == 8
    assert risk_score_from_probability(1.2) == 100 and risk_score_from_probability(-1) == 0
    assert risk_band(85) == "critical" and risk_band(5) == "minimal"


def test_predict_frame(trained_model, registry, demo_frame):
    pred = Predictor(registry)
    out = pred.predict_frame(demo_frame.head(20))
    assert len(out) == 20
    assert set(out["prediction"]) <= {"BOT", "HUMAN"}
    assert ((out["bot_probability"] >= 0) & (out["bot_probability"] <= 1)).all()


def test_predictor_without_model(tmp_path):
    pred = Predictor(ModelRegistry(tmp_path / "empty"))
    assert not pred.is_available()
    with pytest.raises(ModelNotAvailableError):
        pred.predict_account({})


def test_shap_explainer_requires_background(trained_model):
    exp = ShapExplainer(trained_model.entry and __import__("joblib").load(trained_model.model_dir / "pipeline.joblib"), FEATURE_NAMES, "random_forest")
    with pytest.raises(RuntimeError):
        exp.local(np.zeros(31))


def test_constant_features_are_dropped_and_recorded(demo_frame, tmp_path):
    """User-level Cresci data has no tweets: tweet-derived features are constant and must not be used."""
    import json

    reg = ModelRegistry(tmp_path / "models")
    X = demo_frame[FEATURE_NAMES].copy()
    for f in ("hashtag_count", "avg_hashtag", "url_count"):
        X[f] = 0.0
    cfg = TrainConfig(algorithm="decision_tree", hyperparameter_search=False, cv_folds=2, shap_sample_size=30)
    res = train_model(X, demo_frame["label"].values, cfg, reg)
    assert res.entry.n_features == 28
    assert "hashtag_count" not in res.entry.feature_names
    meta = json.loads((res.model_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    assert sorted(meta["dropped_constant_features"]) == ["avg_hashtag", "hashtag_count", "url_count"]
    # prediction still accepts the full account schema and ignores the dropped features
    out = Predictor(reg).predict_account({"followers_count": 10, "friends_count": 50, "hashtag_count": 999, "statuses_count": 100})
    assert set(out["features"]) == set(res.entry.feature_names)
