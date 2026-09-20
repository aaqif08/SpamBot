"""API tests (FastAPI TestClient) — schema validation, workflows, error handling."""

from __future__ import annotations

import io
import time

import pandas as pd
import pytest

from ml.demo_data import generate_demo_accounts, sample_accounts


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_features"] == 31
    assert body["model_available"] is True
    assert "shap" in body["libraries"]


def test_openapi_docs_available(client):
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_models_listing_separates_paper_results(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    assert body["active_model_id"]
    assert len(body["supported_algorithms"]) == 9
    paper = body["paper_reported"]
    assert "NOT measured" in paper["source"]
    assert paper["citation"]["doi"] == "10.1109/ACCESS.2025.3551993"
    assert len(paper["datasets"]["cresci-15"]["results"]) == 9
    for m in body["models"]:
        assert "path" not in m
        assert m["source"] == "Experiment reproduced by this implementation"


def test_predict_validation_errors(client):
    r = client.post("/api/predict", json={"account": {"followers_count": -1}})
    assert r.status_code == 422
    assert r.json()["code"] == "validation_error"
    r = client.post("/api/predict", json={})
    assert r.status_code == 422


def test_predict_sample_and_explanations(client):
    sample = sample_accounts()[1]["account"]
    r = client.post("/api/predict", json={"account": sample, "source": "sample"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in ("BOT", "HUMAN")
    assert body["is_demo"] is True
    assert body["risk_score"] == round(body["bot_probability"] * 100)
    assert "NOT a definitive statement" in body["risk_score_note"]
    assert body["shap_explanation"]["contributions"]
    assert body["lime_explanation"]["items"]
    assert body["interpretation"]["summary"].startswith("Model prediction:")
    pid = body["prediction_id"]
    assert pid

    shap_r = client.get(f"/api/explain/shap/{pid}")
    assert shap_r.status_code == 200
    assert shap_r.json()["explanation"]["contributions"][0]["feature"]
    lime_r = client.get(f"/api/explain/lime/{pid}")
    assert lime_r.status_code == 200
    assert lime_r.json()["explanation"]["prediction_probabilities"]["BOT"] >= 0

    detail = client.get(f"/api/history/{pid}")
    assert detail.status_code == 200
    assert detail.json()["account_identifier"] == sample["account_id"]
    assert client.get("/api/history/does-not-exist").status_code == 404
    assert client.get("/api/explain/shap/does-not-exist").status_code == 404


def test_predict_with_unknown_model(client):
    r = client.post("/api/predict", json={"account": {"followers_count": 1}, "model_id": "nope"})
    assert r.status_code == 404


def test_history_filters(client):
    r = client.get("/api/history", params={"prediction": "BOT", "page_size": 5})
    assert r.status_code == 200
    body = r.json()
    assert all(i["prediction"] == "BOT" for i in body["items"])
    assert client.get("/api/history", params={"prediction": "MAYBE"}).status_code == 422


def test_dataset_upload_inspect_batch_and_download(client):
    df = generate_demo_accounts(n=60, seed=11)
    files = {"file": ("mini.csv", _csv_bytes(df), "text/csv")}
    r = client.post("/api/datasets/upload", files=files)
    assert r.status_code == 201, r.text
    ds = r.json()
    assert ds["n_rows"] == 60 and ds["has_label"] is True
    assert ds["summary"]["class_distribution"]["BOT"] + ds["summary"]["class_distribution"]["HUMAN"] == 60
    assert ds["summary"]["feature_availability"]["coverage"] == 1.0

    listed = client.get("/api/datasets").json()
    assert any(d["id"] == ds["id"] for d in listed)
    assert client.get(f"/api/datasets/{ds['id']}").status_code == 200

    ev = client.post(f"/api/datasets/{ds['id']}/evaluate", json={})
    assert ev.status_code == 200
    assert 0 <= ev.json()["evaluation"]["metrics"]["accuracy"] <= 1

    b = client.post("/api/predict/batch", data={"dataset_id": ds["id"]})
    assert b.status_code == 200, b.text
    batch = b.json()
    assert batch["total_accounts"] == 60
    assert batch["predicted_bots"] + batch["predicted_humans"] == 60
    assert batch["evaluation"]["confusion_matrix"]["tp"] >= 0
    dl = client.get(batch["download_url"])
    assert dl.status_code == 200
    header = dl.text.splitlines()[0]
    for col in ("account_id", "prediction", "bot_probability", "human_probability", "risk_score"):
        assert col in header
    assert client.get(f"/api/predict/batch/{batch['batch_id']}").status_code == 200
    assert any(x["batch_id"] == batch["batch_id"] for x in client.get("/api/predict/batches").json())

    # batch rows get on-demand explanations through the explain endpoints
    hist = client.get("/api/history", params={"source": "batch", "page_size": 1}).json()
    pid = hist["items"][0]["id"]
    assert client.get(f"/api/explain/shap/{pid}").status_code == 200
    assert client.get(f"/api/explain/lime/{pid}").status_code == 200

    assert client.delete(f"/api/datasets/{ds['id']}").status_code == 204
    assert client.get(f"/api/datasets/{ds['id']}").status_code == 404


def test_batch_upload_file_unlabelled(client):
    df = generate_demo_accounts(n=20, seed=5).drop(columns=["label", "bot_type"])
    r = client.post("/api/predict/batch", files={"file": ("u.csv", _csv_bytes(df), "text/csv")})
    assert r.status_code == 200
    assert r.json()["evaluation"] is None


def test_upload_rejections(client):
    assert client.post("/api/datasets/upload", files={"file": ("x.txt", b"a,b\n1,2", "text/plain")}).status_code == 415
    assert client.post("/api/datasets/upload", files={"file": ("x.csv", b"   ", "text/csv")}).status_code == 400
    assert client.post("/api/datasets/upload", files={"file": ("x.csv", b"\x00\x01binary", "text/csv")}).status_code == 415
    assert client.post("/api/predict/batch", data={}).status_code == 400
    assert client.post("/api/predict/batch", data={"dataset_id": "missing"}).status_code == 404


def test_demo_dataset_and_async_training(client):
    r = client.post("/api/datasets/demo", params={"n": 200, "seed": 2})
    assert r.status_code == 201
    ds = r.json()
    assert ds["is_demo"] is True and "DEMO DATA" in ds["name"]

    bad = client.post("/api/train", json={"dataset_id": ds["id"], "algorithm": "unknown"})
    assert bad.status_code == 422
    assert client.post("/api/train", json={"dataset_id": "missing", "algorithm": "decision_tree"}).status_code == 404

    r = client.post("/api/train", json={"dataset_id": ds["id"], "algorithm": "decision_tree", "cv_folds": 2, "hyperparameter_search": False, "activate": False})
    assert r.status_code == 202
    job = r.json()["job_id"]
    status = None
    for _ in range(120):
        status = client.get(f"/api/train/status/{job}").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    assert status is not None and status["status"] == "completed", status
    result = status["result"]
    assert result["is_demo"] is True
    assert result["metrics"]["holdout"]["metrics"]["accuracy"] <= 1
    assert client.get("/api/train/stages").json()[-1] == "completed"
    runs = client.get("/api/train/runs").json()
    assert runs[0]["status"] == "completed"
    assert client.get("/api/train/status/nope").status_code == 404

    mid = result["model_id"]
    assert client.get(f"/api/evaluation/{mid}").status_code == 200
    assert client.get(f"/api/explain/global/{mid}").status_code == 200
    assert client.post(f"/api/models/{mid}/activate").status_code == 200
    assert client.get("/api/models").json()["active_model_id"] == mid
    assert client.delete(f"/api/models/{mid}").status_code == 204
    assert client.get(f"/api/evaluation/{mid}").status_code == 404
    assert client.get("/api/models").json()["active_model_id"] != mid


def test_evaluation_and_global_explanation_active(client):
    ev = client.get("/api/evaluation/active")
    assert ev.status_code == 200
    body = ev.json()
    assert body["source"] == "Experiment reproduced by this implementation"
    assert body["metrics"]["holdout"]["roc_curve"]["auc"] is not None
    g = client.get("/api/explain/global/active").json()
    assert len(g["shap_global"]["importance"]) == 31
    assert g["feature_groups"]["sentiment"] == ["avg_polarity", "avg_subjectivity"]


def test_dashboard_and_misc(client):
    d = client.get("/api/dashboard").json()
    assert d["has_model"] is True
    assert d["cards"]["total_accounts_analyzed"] >= 1
    assert d["charts"]["confusion_matrix"] is not None
    assert client.get("/api/research").json()["citation"]["year"] == 2025
    assert client.get("/api/features").json()["n_features"] == 31
    adapters = client.get("/api/adapters").json()
    assert {a["name"] for a in adapters} == {"sample", "x_api"}
    assert client.post("/api/adapters/sample/fetch", json={"identifier": "demo-human-1"}).status_code == 200
    assert client.post("/api/adapters/sample/fetch", json={"identifier": "nobody"}).status_code == 404
    assert client.post("/api/adapters/x_api/fetch", json={"identifier": "x"}).status_code == 409
    assert client.post("/api/adapters/none/fetch", json={"identifier": "x"}).status_code == 404
    cresci = client.get("/api/datasets/cresci/status").json()
    assert {c["kind"] for c in cresci} == {"cresci-15", "cresci-17"}
    assert client.post("/api/datasets/cresci/cresci-15/import").status_code == 404


def test_security_helpers():
    from pathlib import Path

    from fastapi import HTTPException

    from app.core.security import safe_join, sanitize_filename

    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("my data (1).csv") == "my_data_1_.csv"
    root = Path.cwd()
    with pytest.raises(HTTPException):
        safe_join(root, "..", "..", "x")
