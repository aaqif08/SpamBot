"""Workflow tests: datasets → training job → analyses → explanations → batches → dashboard."""

from __future__ import annotations

import pandas as pd

from tests.conftest import csv_bytes, wait_job


def test_empty_state_before_any_data(client, admin_b):
    d = client.get("/api/v1/dashboard", headers=admin_b).json()
    assert d["has_model"] is False and d["has_predictions"] is False
    assert d["cards"]["predictions_total"] == 0 and d["charts"]["bot_vs_human"] == []
    assert client.get("/api/v1/models", headers=admin_b).json() == {"production_model_id": None, "models": [], "supported_algorithms": client.get("/api/v1/models", headers=admin_b).json()["supported_algorithms"]}
    assert client.get("/api/v1/analyses", headers=admin_b).json()["total"] == 0
    r = client.post("/api/v1/analyses", headers=admin_b, json={"account": {"followers_count": 3}})
    assert r.status_code == 409 and "production model" in r.json()["detail"].lower()
    assert client.post("/api/v1/batches", headers=admin_b, data={"dataset_id": "x"}).status_code == 409


def test_upload_validation(client, analyst_a):
    assert client.post("/api/v1/datasets", headers=analyst_a, files={"file": ("x.txt", b"a,b\n1,2", "text/plain")}).status_code == 415
    assert client.post("/api/v1/datasets", headers=analyst_a, files={"file": ("x.csv", b"   ", "text/csv")}).status_code == 400
    assert client.post("/api/v1/datasets", headers=analyst_a, files={"file": ("x.csv", b"\x00\x01bin", "text/csv")}).status_code == 415
    assert client.post("/api/v1/datasets", headers=analyst_a, files={"file": ("x.csv", b"PK\x03\x04zip", "text/csv")}).status_code == 415
    # unrecognised columns → stored but INVALID (no account columns)
    r = client.post("/api/v1/datasets", headers=analyst_a, files={"file": ("bad.csv", b"foo,bar\n1,2\n", "text/csv")})
    assert r.status_code == 201 and r.json()["status"] == "INVALID" and r.json()["current_version"]["validation_errors"]
    assert client.post("/api/v1/models/train", headers=analyst_a, json={"dataset_id": r.json()["id"], "algorithm": "decision_tree"}).status_code == 422
    assert client.delete(f"/api/v1/datasets/{r.json()['id']}", headers=analyst_a).status_code == 204


def test_dataset_inspection_and_versions(client, analyst_a, dataset_a, synthetic_frame):
    d = client.get(f"/api/v1/datasets/{dataset_a['id']}", headers=analyst_a).json()
    assert d["n_rows"] == 240 and d["has_label"] and d["status"] == "VALIDATED"
    assert d["summary"]["feature_availability"]["coverage"] == 1.0
    assert d["summary"]["class_distribution"]["BOT"] + d["summary"]["class_distribution"]["HUMAN"] == 240
    # new version of the same dataset
    r = client.post("/api/v1/datasets", headers=analyst_a, files={"file": ("v2.csv", csv_bytes(synthetic_frame.head(100)), "text/csv")}, data={"dataset_id": dataset_a["id"]})
    assert r.status_code == 201 and r.json()["n_versions"] == 2 and r.json()["current_version"]["version"] == 2
    vid = r.json()["current_version"]["id"]
    assert client.get(f"/api/v1/datasets/{dataset_a['id']}/versions/{vid}", headers=analyst_a).json()["n_rows"] == 100


def test_training_job_and_model_lifecycle(client, admin_a, analyst_a, viewer_a, production_model_a):
    m = client.get("/api/v1/models", headers=viewer_a).json()
    assert m["production_model_id"] == production_model_a["id"]
    prod = next(x for x in m["models"] if x["id"] == production_model_a["id"])
    assert prod["status"] == "PRODUCTION" and prod["artifact_checksum"] and 0.5 <= prod["test_metrics"]["accuracy"] <= 1.0
    assert prod["validation_metrics"]["folds"] == 2
    ev = client.get(f"/api/v1/models/{prod['id']}/evaluation", headers=viewer_a).json()
    assert ev["evaluations"][0]["kind"] == "holdout" and ev["feature_importance"]
    assert client.get(f"/api/v1/models/{prod['id']}/explanation", headers=viewer_a).json()["shap_global"]["importance"]
    # unknown algorithm / bad params
    assert client.post("/api/v1/models/train", headers=analyst_a, json={"dataset_id": "nope", "algorithm": "decision_tree"}).status_code == 404
    assert client.post("/api/v1/models/train", headers=analyst_a, json={"dataset_id": production_model_a["dataset_id"], "algorithm": "unknown"}).status_code == 422
    # jobs listing
    jobs = client.get("/api/v1/jobs", headers=viewer_a).json()
    assert any(j["job_type"] == "TRAINING" and j["status"] == "COMPLETED" for j in jobs)
    # lifecycle: cannot delete production model; viewer cannot activate
    assert client.delete(f"/api/v1/models/{prod['id']}", headers=admin_a).status_code == 409
    assert client.post(f"/api/v1/models/{prod['id']}/activate", headers=viewer_a).status_code == 403


def test_second_model_activation_switch(client, admin_a, analyst_a, dataset_a, production_model_a):
    r = client.post("/api/v1/models/train", headers=analyst_a, json={"dataset_id": dataset_a["id"], "algorithm": "logistic_regression", "cv_folds": 2, "hyperparameter_search": False, "activate": False})
    s = wait_job(client, analyst_a, r.json()["job_id"])
    assert s["status"] == "COMPLETED"
    new_id = s["result"]["model_id"]
    assert client.get(f"/api/v1/models/{new_id}", headers=analyst_a).json()["status"] == "READY"
    assert client.post(f"/api/v1/models/{new_id}/activate", headers=admin_a).status_code == 200
    m = client.get("/api/v1/models", headers=admin_a).json()
    assert m["production_model_id"] == new_id
    assert next(x for x in m["models"] if x["id"] == production_model_a["id"])["status"] == "READY"
    # switch back
    assert client.post(f"/api/v1/models/{production_model_a['id']}/activate", headers=admin_a).status_code == 200
    assert client.post(f"/api/v1/models/{new_id}/deprecate", headers=admin_a).json()["status"] == "DEPRECATED"
    assert client.delete(f"/api/v1/models/{new_id}", headers=admin_a).status_code == 204
    assert client.get(f"/api/v1/models/{new_id}", headers=admin_a).status_code == 404


def test_analysis_with_explanations_and_history(client, analyst_a, viewer_a, admin_b, production_model_a):
    account = {"account_id": "acct_x", "name": "Acct X", "followers_count": 7, "friends_count": 1200, "statuses_count": 9, "default_profile": True, "default_profile_image": True, "description": "", "tweets": ["hello", "http://bit.ly/x #followback"]}
    r = client.post("/api/v1/analyses", headers=analyst_a, json={"account": account})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["prediction"] in ("BOT", "HUMAN")
    assert body["risk_score"] == round(body["bot_probability"] * 100)
    assert body["shap_explanation"]["contributions"] and body["lime_explanation"]["items"]
    assert body["explanation_status"] == {"shap": "COMPLETED", "lime": "COMPLETED"}
    assert "model output" in body["risk_score_note"].lower() or "not" in body["risk_score_note"].lower()
    assert "tweets" not in body["input_summary"]  # raw tweet text is not stored
    pid = body["prediction_id"]

    # detail + stored explanations, visible to viewer in the same org, invisible to org B
    assert client.get(f"/api/v1/analyses/{pid}", headers=viewer_a).json()["prediction_id"] == pid
    assert client.get(f"/api/v1/analyses/{pid}/explanations/shap", headers=viewer_a).json()["computed_now"] is False
    assert client.get(f"/api/v1/analyses/{pid}", headers=admin_b).status_code == 404
    assert client.get(f"/api/v1/analyses/{pid}/explanations/foo", headers=viewer_a).status_code == 404

    # history filters / sort / pagination
    h = client.get("/api/v1/analyses", headers=viewer_a, params={"search": "acct_x", "sort": "risk_score:desc", "page_size": 5}).json()
    assert h["total"] >= 1 and h["items"][0]["account_identifier"] == "acct_x"
    assert client.get("/api/v1/analyses", headers=viewer_a, params={"prediction": "MAYBE"}).status_code == 422
    assert client.get("/api/v1/analyses", headers=viewer_a, params={"sort": "drop table"}).status_code == 422

    # validation
    assert client.post("/api/v1/analyses", headers=analyst_a, json={"account": {"followers_count": -1}}).status_code == 422
    assert client.post("/api/v1/analyses", headers=analyst_a, json={}).status_code == 422
    # analysing again updates the account record
    client.post("/api/v1/analyses", headers=analyst_a, json={"account": account})
    d = client.get("/api/v1/dashboard", headers=viewer_a).json()
    assert d["has_predictions"] and d["cards"]["accounts_analyzed"] >= 1 and d["cards"]["predictions_total"] >= 2
    assert d["charts"]["bot_vs_human"] and d["charts"]["probability_histogram"]
    # viewer cannot delete; analyst can
    assert client.delete(f"/api/v1/analyses/{pid}", headers=viewer_a).status_code == 403
    assert client.delete(f"/api/v1/analyses/{pid}", headers=analyst_a).status_code == 204
    assert client.get(f"/api/v1/analyses/{pid}", headers=analyst_a).status_code == 404


def test_batch_job(client, analyst_a, viewer_a, dataset_a, production_model_a, synthetic_frame):
    unl = synthetic_frame.drop(columns=["label", "bot_type"]).head(50)
    r = client.post("/api/v1/batches", headers=analyst_a, files={"file": ("u.csv", csv_bytes(unl), "text/csv")}, data={"name": "unlabelled batch"})
    assert r.status_code == 202 and r.json()["status"] == "QUEUED"
    b = r.json()
    s = wait_job(client, analyst_a, b["job_id"])
    assert s["status"] == "COMPLETED", s
    done = client.get(f"/api/v1/batches/{b['id']}", headers=viewer_a).json()
    assert done["status"] == "COMPLETED" and done["processed_rows"] == 50 and done["failed_rows"] == 0 and done["has_output"]
    assert done["summary"]["evaluation"] is None
    dl = client.get(f"/api/v1/batches/{b['id']}/download", headers=viewer_a)
    assert dl.status_code == 200 and dl.text.splitlines()[0].startswith("account_id,prediction,bot_probability")
    # labelled batch from a registered dataset → evaluation
    r = client.post("/api/v1/batches", headers=analyst_a, data={"dataset_id": dataset_a["id"]})
    s = wait_job(client, analyst_a, r.json()["job_id"])
    assert s["status"] == "COMPLETED"
    done = client.get(f"/api/v1/batches/{r.json()['id']}", headers=viewer_a).json()
    assert done["summary"]["evaluation"]["metrics"]["accuracy"] <= 1.0
    # batch rows are in history with lazy explanations
    h = client.get("/api/v1/analyses", headers=viewer_a, params={"source": "batch", "page_size": 1}).json()
    assert h["total"] >= 50
    pid = h["items"][0]["id"]
    assert client.get(f"/api/v1/analyses/{pid}/explanations/lime", headers=viewer_a).json()["computed_now"] is True
    assert client.get(f"/api/v1/analyses/{pid}/explanations/lime", headers=viewer_a).json()["computed_now"] is False
    assert client.delete(f"/api/v1/batches/{b['id']}", headers=analyst_a).status_code == 204
    assert client.get("/api/v1/batches", headers=viewer_a).status_code == 200


def test_dataset_evaluation_endpoint(client, analyst_a, dataset_a, production_model_a):
    r = client.post(f"/api/v1/datasets/{dataset_a['id']}/evaluate", headers=analyst_a, json={})
    assert r.status_code == 200 and 0 <= r.json()["evaluation"]["metrics"]["f1"] <= 1
    ev = client.get(f"/api/v1/models/{production_model_a['id']}/evaluation", headers=analyst_a).json()
    assert any(e["kind"] == "dataset" for e in ev["evaluations"])


def test_retention_and_audit(client, admin_a, analyst_a):
    assert client.post("/api/v1/analyses/retention/purge", headers=analyst_a, json={"older_than_days": 30}).status_code == 403
    r = client.post("/api/v1/analyses/retention/purge", headers=admin_a, json={"older_than_days": 3650})
    assert r.status_code == 200 and r.json()["deleted"] == 0
    a = client.get("/api/v1/audit", headers=admin_a, params={"action": "training"}).json()
    assert a["total"] >= 1 and all(i["action"].startswith("training") for i in a["items"])


def test_providers(client, analyst_a):
    ps = {p["name"]: p for p in client.get("/api/v1/providers", headers=analyst_a).json()}
    assert set(ps) == {"manual", "csv", "x_api"}
    assert ps["x_api"]["configured"] is False and ps["x_api"]["configuration_hint"]
    assert client.post("/api/v1/providers/x_api/fetch", headers=analyst_a, json={"identifier": "nasa"}).status_code == 409
    assert client.post("/api/v1/providers/nope/fetch", headers=analyst_a, json={"identifier": "nasa"}).status_code == 404


def test_benchmark_status(client, viewer_a):
    r = client.get("/api/v1/datasets/benchmarks", headers=viewer_a)
    assert r.status_code == 200 and {b["kind"] for b in r.json()} == {"cresci-15", "cresci-17"}


def test_openapi_has_security_scheme(client):
    spec = client.get("/openapi.json").json()
    assert "HTTPBearer" in spec["components"]["securitySchemes"]
    assert all(p.startswith("/api/v1") or p in ("/",) for p in spec["paths"])


def test_no_synthetic_generator_in_production_package():
    import importlib

    assert importlib.util.find_spec("ml.demo_data") is None
    import pkgutil

    import app.services as services

    names = {m.name for m in pkgutil.iter_modules(services.__path__)}
    assert not {"sample", "demo", "seed"} & {n.split("_")[0] for n in names}


def test_automatic_retention_purges_old_predictions(client, admin_a, production_model_a, orgs, monkeypatch):
    """apply_retention deletes predictions (and cascaded explanations) older than the configured window, per organization."""
    from datetime import datetime, timedelta, timezone

    from app.core.config import get_settings
    from app.db.database import SessionLocal
    from app.db.models import AuditLog, Prediction
    from app.services.retention import apply_retention

    payload = {"account": {"account_id": "retention_probe", "verified": False, "friends_count": 10, "followers_count": 5, "listed_count": 0, "favorites_count": 1, "statuses_count": 20, "default_profile": True, "default_profile_image": False, "geo_enabled": False, "profile_background_tile": False, "has_profile_banner": False, "tweets": []}, "explain": False}
    pid = client.post("/api/v1/analyses", headers=admin_a, json=payload).json()["prediction_id"]

    db = SessionLocal()
    try:
        row = db.get(Prediction, pid)
        org_id = row.organization_id
        row.created_at = datetime.now(timezone.utc) - timedelta(days=400)
        db.commit()
        monkeypatch.setattr(get_settings(), "retention_predictions_days", 365)
        deleted = apply_retention(db)
        assert deleted["predictions"] >= 1
        assert db.get(Prediction, pid) is None
        assert db.query(AuditLog).filter(AuditLog.action == "retention.predictions_purged", AuditLog.organization_id == org_id).count() >= 1
    finally:
        monkeypatch.setattr(get_settings(), "retention_predictions_days", None)
        db.close()
    assert client.get(f"/api/v1/analyses/{pid}", headers=admin_a).status_code == 404
