"""Pytest configuration.

* Isolated temp SQLite database (migrated with Alembic), temp local storage,
  in-process job backend.
* Two organisations with users of each role so tenancy and RBAC can be tested.
* A small synthetic labelled dataset (tests/fixtures — never production data).

Environment variables must be set before ``app`` is imported.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

_TMP = Path(tempfile.mkdtemp(prefix="botshield-tests-"))
os.environ.update(
    {
        "BOTSHIELD_ENVIRONMENT": "test",
        "BOTSHIELD_DATABASE_URL": f"sqlite:///{(_TMP / 'test.db').as_posix()}",
        "BOTSHIELD_STORAGE_BACKEND": "local",
        "BOTSHIELD_STORAGE_LOCAL_ROOT": str(_TMP / "storage"),
        "BOTSHIELD_STORAGE_CACHE_DIR": str(_TMP / "cache"),
        "BOTSHIELD_JOB_BACKEND": "thread",
        "BOTSHIELD_RATE_LIMIT_PER_MINUTE": "100000",
        "BOTSHIELD_ML_RATE_LIMIT_PER_MINUTE": "100000",
        "BOTSHIELD_LOGIN_RATE_LIMIT_PER_MINUTE": "100000",
        "BOTSHIELD_SECRET_KEY": "test-secret-key-not-for-production",
        "BOTSHIELD_LOG_LEVEL": "WARNING",
    }
)

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

PASSWORD = "Str0ngPassw0rd!"


@pytest.fixture(scope="session", autouse=True)
def _migrated_db():
    from app.cli import main as cli

    assert cli(["migrate"]) == 0
    yield


@pytest.fixture(scope="session")
def synthetic_frame():
    from tests.fixtures.synthetic import generate_synthetic_accounts

    return generate_synthetic_accounts(n=240, seed=3)


@pytest.fixture(scope="session")
def orgs():
    """Two organisations: A (admin, analyst, viewer) and B (admin)."""
    from app.db.database import SessionLocal
    from app.db.models import Role
    from app.services.auth_service import create_organization_with_admin, create_user

    db = SessionLocal()
    try:
        org_a, admin_a = create_organization_with_admin(db, "Org A", "admin-a@example.com", PASSWORD, "Admin A")
        analyst_a = create_user(db, admin_a, "analyst-a@example.com", PASSWORD, Role.ANALYST, "Analyst A")
        viewer_a = create_user(db, admin_a, "viewer-a@example.com", PASSWORD, Role.VIEWER, "Viewer A")
        org_b, admin_b = create_organization_with_admin(db, "Org B", "admin-b@example.com", PASSWORD, "Admin B")
        return {"org_a": org_a.id, "admin_a": admin_a.id, "analyst_a": analyst_a.id, "viewer_a": viewer_a.id, "org_b": org_b.id, "admin_b": admin_b.id}
    finally:
        db.close()


@pytest.fixture(scope="session")
def client(orgs):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


def _login(client, email: str) -> dict[str, str]:
    r = client.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="session")
def admin_a(client, orgs):
    return _login(client, "admin-a@example.com")


@pytest.fixture(scope="session")
def analyst_a(client, orgs):
    return _login(client, "analyst-a@example.com")


@pytest.fixture(scope="session")
def viewer_a(client, orgs):
    return _login(client, "viewer-a@example.com")


@pytest.fixture(scope="session")
def admin_b(client, orgs):
    return _login(client, "admin-b@example.com")


def csv_bytes(df) -> bytes:  # noqa: ANN001
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def wait_job(client, headers: dict[str, str], job_id: str, timeout: float = 240.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
        if s["status"] in ("COMPLETED", "FAILED"):
            return s
        time.sleep(0.3)
    raise AssertionError("job did not finish in time")


@pytest.fixture(scope="session")
def dataset_a(client, analyst_a, synthetic_frame):
    r = client.post("/api/v1/datasets", headers=analyst_a, files={"file": ("synthetic.csv", csv_bytes(synthetic_frame), "text/csv")}, data={"name": "Synthetic test set"})
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture(scope="session")
def production_model_a(client, analyst_a, dataset_a):
    """A quick Decision Tree trained and activated in org A."""
    r = client.post("/api/v1/models/train", headers=analyst_a, json={"dataset_id": dataset_a["id"], "algorithm": "decision_tree", "cv_folds": 2, "hyperparameter_search": False, "activate": True})
    assert r.status_code == 202, r.text
    s = wait_job(client, analyst_a, r.json()["job_id"])
    assert s["status"] == "COMPLETED", s
    return s["result"]["model"]


@pytest.fixture(scope="session")
def trained_model(synthetic_frame, tmp_path_factory):
    """Direct ML-level training result (no HTTP) for explainability tests."""
    from ml.features import FEATURE_NAMES
    from ml.train import TrainConfig, train_model

    cfg = TrainConfig(algorithm="random_forest", hyperparameter_search=False, cv_folds=3, shap_sample_size=40, background_size=30, lime_sample_size=200)
    return train_model(synthetic_frame[FEATURE_NAMES], synthetic_frame["label"].values, cfg, tmp_path_factory.mktemp("rf"))
