"""Training, artefact integrity, prediction, SHAP and LIME tests (ML package, no HTTP)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ml.explain import LimeExplainer
from ml.features import FEATURE_NAMES
from ml.model_registry import ArtifactIntegrityError, verify_checksums
from ml.predict import InferenceEngine, load_model_dir, risk_band, risk_score_from_probability
from ml.train import MODEL_ZOO, TrainConfig, model_choices, train_model


def _load(result):  # noqa: ANN001
    return load_model_dir(result.artifact_dir, "m", result.card.name, 1, result.card.algorithm, result.checksums)


def test_model_zoo_has_paper_classifiers():
    assert set(MODEL_ZOO) == {"random_forest", "svm", "decision_tree", "xgboost", "lightgbm", "logistic_regression", "extra_trees", "naive_bayes", "adaboost"}
    assert all(c["available"] for c in model_choices())


def test_train_config_validation():
    with pytest.raises(ValueError):
        TrainConfig(algorithm="nope").validate()
    with pytest.raises(ValueError):
        TrainConfig(test_size=0.9).validate()
    with pytest.raises(ValueError):
        TrainConfig(cv_folds=1).validate()


def test_training_artifacts_and_card(trained_model):
    d = trained_model.artifact_dir
    for name in ("pipeline.joblib", "scaler.joblib", "feature_metadata.json", "metrics.json", "shap_global.json", "background.npy", "lime_sample.npy", "CHECKSUMS.json"):
        assert (d / name).exists(), name
    card = trained_model.card
    assert card.feature_names == FEATURE_NAMES and set(card.holdout_metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert 0.5 <= card.holdout_metrics["accuracy"] <= 1.0
    assert trained_model.metrics["cross_validation"]["n_folds"] == 3
    assert json.loads((d / "CHECKSUMS.json").read_text()) == trained_model.checksums


def test_checksum_verification_blocks_tampering(trained_model, tmp_path):
    verify_checksums(trained_model.artifact_dir, trained_model.checksums)
    with pytest.raises(ArtifactIntegrityError):
        verify_checksums(trained_model.artifact_dir, {**trained_model.checksums, "pipeline.joblib": "0" * 64})
    with pytest.raises(ArtifactIntegrityError):
        verify_checksums(trained_model.artifact_dir, {})
    with pytest.raises(ArtifactIntegrityError):
        load_model_dir(trained_model.artifact_dir, "m", "x", 1, "random_forest", {"pipeline.joblib": "bad"})


def test_training_rejects_single_class(synthetic_frame, tmp_path):
    with pytest.raises(ValueError):
        train_model(synthetic_frame[FEATURE_NAMES], np.zeros(len(synthetic_frame), dtype=int), TrainConfig(algorithm="decision_tree", hyperparameter_search=False), tmp_path / "x")


def test_shap_feature_selection_reduces_features(synthetic_frame, tmp_path):
    cfg = TrainConfig(algorithm="decision_tree", hyperparameter_search=False, cv_folds=2, feature_selection=True, feature_selection_top_k=12, shap_sample_size=40)
    res = train_model(synthetic_frame[FEATURE_NAMES], synthetic_frame["label"].values, cfg, tmp_path / "sel")
    assert len(res.card.feature_names) == 12
    out = InferenceEngine.predict_account(_load(res), {"followers_count": 5, "friends_count": 900, "statuses_count": 3})
    assert out["model"]["n_features"] == 12 and len(out["features"]) == 12


def test_constant_features_are_dropped_and_recorded(synthetic_frame, tmp_path):
    X = synthetic_frame[FEATURE_NAMES].copy()
    for f in ("hashtag_count", "avg_hashtag", "url_count"):
        X[f] = 0.0
    res = train_model(X, synthetic_frame["label"].values, TrainConfig(algorithm="decision_tree", hyperparameter_search=False, cv_folds=2, shap_sample_size=30), tmp_path / "const")
    assert len(res.card.feature_names) == 28
    meta = json.loads((res.artifact_dir / "feature_metadata.json").read_text())
    assert sorted(meta["dropped_constant_features"]) == ["avg_hashtag", "hashtag_count", "url_count"]


def test_global_shap_structure(trained_model):
    g = trained_model.shap_global
    assert g["explainer"] == "TreeExplainer" and g["output_scale"] == "probability"
    assert len(g["importance"]) == 31 and g["importance"][0]["rank"] == 1
    assert g["importance"][0]["mean_abs_shap"] >= g["importance"][-1]["mean_abs_shap"]
    assert len(g["top_features"]) == 20 and len(g["beeswarm"]) == 20
    assert set(g["group_importance"]) == {"user_profile", "content", "engagement", "linguistic", "profile_attributes", "sentiment"}


def test_local_shap_is_additive(trained_model):
    loaded = _load(trained_model)
    x = np.zeros(31)
    x[FEATURE_NAMES.index("friends_count")] = 800
    x[FEATURE_NAMES.index("statuses_count")] = 5
    local = loaded.shap().local(x)
    assert local["base_value"] + sum(c["shap"] for c in local["contributions"]) == pytest.approx(local["model_output"], abs=1e-6)
    assert local["model_output"] == pytest.approx(loaded.pipeline.predict_proba(x.reshape(1, -1))[0, 1], abs=1e-4)


def test_kernel_shap_for_non_tree_model(synthetic_frame, tmp_path):
    cfg = TrainConfig(algorithm="logistic_regression", hyperparameter_search=False, cv_folds=2, shap_sample_size=20, background_size=20)
    res = train_model(synthetic_frame[FEATURE_NAMES], synthetic_frame["label"].values, cfg, tmp_path / "lr")
    assert res.shap_global["explainer"] == "KernelExplainer"
    out = InferenceEngine.predict_account(_load(res), {"followers_count": 5, "friends_count": 900, "statuses_count": 3}, lime_samples=500)
    assert out["shap_explanation"]["explainer"] == "KernelExplainer" and out["lime_explanation"] is not None


def test_lime_explanation_structure(trained_model):
    loaded = _load(trained_model)
    lime = LimeExplainer(loaded.pipeline, loaded.feature_names, loaded.lime_sample, num_samples=500)
    x = np.zeros(31)
    x[FEATURE_NAMES.index("hashtag_count")] = 500
    out = lime.explain(x)
    p = out["prediction_probabilities"]
    assert p["HUMAN"] + p["BOT"] == pytest.approx(1.0, abs=1e-6) and len(out["items"]) == 31
    assert all(i["weight"] > 0 for i in out["bot_indicators"]) and all(i["weight"] < 0 for i in out["human_indicators"])


def test_inference_engine_and_risk_score(trained_model):
    loaded = _load(trained_model)
    out = InferenceEngine.predict_account(loaded, {"account_id": "x", "followers_count": 3, "friends_count": 1500, "statuses_count": 8, "default_profile": True, "default_profile_image": True}, lime_samples=500)
    assert out["prediction"] in ("BOT", "HUMAN") and out["bot_probability"] + out["human_probability"] == pytest.approx(1.0, abs=1e-6)
    assert out["risk_score"] == round(out["bot_probability"] * 100) and out["risk_band"] == risk_band(out["risk_score"])
    assert out["explanation_errors"] == {} and len(out["top_features"]) == 8 and out["inference_ms"] >= 0
    assert risk_score_from_probability(1.2) == 100 and risk_score_from_probability(-1) == 0
    assert risk_band(85) == "critical" and risk_band(5) == "minimal"


def test_predict_frame(trained_model, synthetic_frame):
    out = InferenceEngine.predict_frame(_load(trained_model), synthetic_frame.head(20))
    assert len(out) == 20 and set(out["prediction"]) <= {"BOT", "HUMAN"}
    assert ((out["bot_probability"] >= 0) & (out["bot_probability"] <= 1)).all()
