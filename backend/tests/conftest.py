"""Pytest configuration: isolated temp data/model dirs and a small demo model.

Environment variables must be set before ``app`` is imported because settings
are cached and the SQLite engine is created at import time.
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

_TMP = Path(tempfile.mkdtemp(prefix="botshield-tests-"))
os.environ["BOTSHIELD_DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["BOTSHIELD_MODELS_DIR"] = str(_TMP / "models")
os.environ["BOTSHIELD_DATA_DIR"] = str(_TMP / "data")
os.environ["BOTSHIELD_RATE_LIMIT_PER_MINUTE"] = "10000"
os.environ["BOTSHIELD_DEMO_MODE_ENABLED"] = "true"  # tests use the synthetic generator regardless of .env

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


@pytest.fixture(scope="session")
def demo_frame():
    from ml.demo_data import generate_demo_accounts

    return generate_demo_accounts(n=240, seed=3)


@pytest.fixture(scope="session")
def registry():
    from ml.model_registry import ModelRegistry

    return ModelRegistry(_TMP / "models")


@pytest.fixture(scope="session")
def trained_model(demo_frame, registry):
    """A quick Random Forest trained on demo data (no hyperparameter search)."""
    from ml.features import FEATURE_NAMES
    from ml.train import TrainConfig, train_model

    cfg = TrainConfig(
        algorithm="random_forest",
        hyperparameter_search=False,
        cv_folds=3,
        shap_sample_size=40,
        background_size=30,
        lime_sample_size=200,
        dataset_id="test-demo",
        dataset_name="test demo",
        is_demo=True,
    )
    return train_model(demo_frame[FEATURE_NAMES], demo_frame["label"].values, cfg, registry)


@pytest.fixture(scope="session")
def client(trained_model):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
