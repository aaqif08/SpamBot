"""Authentication, sessions, RBAC, tenancy isolation, audit."""

from __future__ import annotations

import pytest

from tests.conftest import PASSWORD


def test_public_and_protected_endpoints(client):
    assert client.get("/api/v1/health").json()["status"] == "ok"
    assert client.get("/api/v1/health/ready").json()["status"] == "ready"
    assert client.get("/api/v1/auth/setup-status").json()["initialized"] is True
    assert client.get("/api/v1/research").status_code == 200
    for path in ("/api/v1/dashboard", "/api/v1/analyses", "/api/v1/datasets", "/api/v1/models", "/api/v1/audit", "/api/v1/users"):
        r = client.get(path)
        assert r.status_code == 401, path
        assert r.headers.get("www-authenticate") == "Bearer"
    assert client.get("/api/v1/dashboard", headers={"Authorization": "Bearer not-a-token"}).status_code == 401


def test_login_failure_and_lockout(client, orgs):
    for _ in range(3):
        r = client.post("/api/v1/auth/login", json={"email": "viewer-a@example.com", "password": "wrong-password"})
        assert r.status_code == 401
    assert client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": "x"}).status_code == 401
    assert client.post("/api/v1/auth/login", json={"email": "not-an-email", "password": "x"}).status_code == 422
    # correct password resets the failure counter
    assert client.post("/api/v1/auth/login", json={"email": "viewer-a@example.com", "password": PASSWORD}).status_code == 200


def test_lockout_after_threshold(client, orgs):
    from app.core.config import get_settings

    for _ in range(get_settings().login_lockout_threshold):
        client.post("/api/v1/auth/login", json={"email": "admin-b@example.com", "password": "wrong"})
    r = client.post("/api/v1/auth/login", json={"email": "admin-b@example.com", "password": PASSWORD})
    assert r.status_code == 423
    # unlock for the rest of the suite
    from app.db.database import SessionLocal
    from app.db.models import User

    db = SessionLocal()
    try:
        u = db.query(User).filter(User.email == "admin-b@example.com").one()
        u.locked_until = None
        u.failed_login_count = 0
        db.commit()
    finally:
        db.close()


def test_refresh_rotation_and_logout(client, orgs):
    r = client.post("/api/v1/auth/login", json={"email": "viewer-a@example.com", "password": PASSWORD})
    cookie = r.cookies.get("botshield_refresh")
    assert cookie
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    r2 = client.post("/api/v1/auth/refresh", cookies={"botshield_refresh": cookie})
    assert r2.status_code == 200 and r2.json()["access_token"]
    # old refresh token is revoked after rotation
    assert client.post("/api/v1/auth/refresh", cookies={"botshield_refresh": cookie}).status_code == 401
    new_cookie = r2.cookies.get("botshield_refresh")
    assert client.post("/api/v1/auth/logout", headers=headers, cookies={"botshield_refresh": new_cookie}).status_code == 204
    assert client.post("/api/v1/auth/refresh", cookies={"botshield_refresh": new_cookie}).status_code == 401


def test_rbac(client, admin_a, analyst_a, viewer_a):
    assert client.get("/api/v1/users", headers=viewer_a).status_code == 403
    assert client.get("/api/v1/users", headers=analyst_a).status_code == 403
    assert client.get("/api/v1/users", headers=admin_a).status_code == 200
    assert client.get("/api/v1/audit", headers=analyst_a).status_code == 403
    assert client.post("/api/v1/datasets/benchmarks/cresci-15/import", headers=analyst_a).status_code == 403
    assert client.get("/api/v1/analyses", headers=viewer_a).status_code == 200
    assert client.post("/api/v1/analyses", headers=viewer_a, json={"account": {"followers_count": 1}}).status_code == 403


def test_user_management(client, admin_a, orgs):
    r = client.post("/api/v1/users", headers=admin_a, json={"email": "temp-a@example.com", "password": "weak", "role": "VIEWER"})
    assert r.status_code == 422  # password policy
    r = client.post("/api/v1/users", headers=admin_a, json={"email": "temp-a@example.com", "password": PASSWORD, "role": "ANALYST", "full_name": "Temp"})
    assert r.status_code == 201
    uid = r.json()["id"]
    assert client.post("/api/v1/users", headers=admin_a, json={"email": "temp-a@example.com", "password": PASSWORD, "role": "VIEWER"}).status_code == 409
    r = client.patch(f"/api/v1/users/{uid}", headers=admin_a, json={"role": "VIEWER", "status": "DISABLED"})
    assert r.status_code == 200 and r.json()["status"] == "DISABLED"
    assert client.post("/api/v1/auth/login", json={"email": "temp-a@example.com", "password": PASSWORD}).status_code == 403
    # cannot remove own admin role
    assert client.patch(f"/api/v1/users/{orgs['admin_a']}", headers=admin_a, json={"role": "VIEWER"}).status_code == 400
    assert client.post(f"/api/v1/users/{uid}/reset-password", headers=admin_a, json={"new_password": PASSWORD + "2"}).status_code == 204


def test_change_password_invalidates_old_sessions(client, admin_a):
    assert client.post("/api/v1/users", headers=admin_a, json={"email": "pw-a@example.com", "password": PASSWORD, "role": "VIEWER"}).status_code == 201
    r = client.post("/api/v1/auth/login", json={"email": "pw-a@example.com", "password": PASSWORD})
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    import time

    time.sleep(1.2)  # tokens issued in the same second as the change stay valid (1s tolerance)
    assert client.post("/api/v1/auth/change-password", headers=headers, json={"current_password": "wrong", "new_password": PASSWORD + "X"}).status_code == 400
    assert client.post("/api/v1/auth/change-password", headers=headers, json={"current_password": PASSWORD, "new_password": PASSWORD + "X"}).status_code == 204
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 401  # old access token rejected
    assert client.post("/api/v1/auth/login", json={"email": "pw-a@example.com", "password": PASSWORD}).status_code == 401
    assert client.post("/api/v1/auth/login", json={"email": "pw-a@example.com", "password": PASSWORD + "X"}).status_code == 200


def test_password_reset_flow(client, orgs, admin_a):
    from app.db.database import SessionLocal
    from app.services.auth_service import consume_password_reset, create_password_reset

    assert client.post("/api/v1/auth/password-reset/request", json={"email": "ghost@example.com"}).status_code == 202
    client.post("/api/v1/users", headers=admin_a, json={"email": "reset-a@example.com", "password": PASSWORD, "role": "VIEWER"})
    db = SessionLocal()
    try:
        token = create_password_reset(db, "reset-a@example.com")
        assert token
        with pytest.raises(Exception):
            consume_password_reset(db, "bogus", PASSWORD)
        consume_password_reset(db, token, PASSWORD + "R")
    finally:
        db.close()
    assert client.post("/api/v1/auth/login", json={"email": "reset-a@example.com", "password": PASSWORD + "R"}).status_code == 200
    assert client.post("/api/v1/auth/password-reset/confirm", json={"token": "x" * 20, "new_password": PASSWORD}).status_code == 400


def test_tenancy_isolation(client, admin_a, admin_b, dataset_a):
    # Org B cannot see or touch Org A's dataset
    assert client.get(f"/api/v1/datasets/{dataset_a['id']}", headers=admin_b).status_code == 404
    assert client.delete(f"/api/v1/datasets/{dataset_a['id']}", headers=admin_b).status_code == 404
    assert all(d["id"] != dataset_a["id"] for d in client.get("/api/v1/datasets", headers=admin_b).json())
    assert client.get("/api/v1/dashboard", headers=admin_b).json()["cards"]["datasets"] == 0
    # audit is per organisation
    actions_b = {a["actor_email"] for a in client.get("/api/v1/audit", headers=admin_b).json()["items"]}
    assert "admin-a@example.com" not in actions_b


def test_organization_endpoints(client, admin_a, viewer_a):
    assert client.get("/api/v1/organization", headers=viewer_a).json()["name"] == "Org A"
    assert client.patch("/api/v1/organization", headers=viewer_a, json={"name": "X"}).status_code == 403
    r = client.patch("/api/v1/organization", headers=admin_a, json={"name": "Org A Ltd"})
    assert r.status_code == 200 and r.json()["name"] == "Org A Ltd"


def test_security_headers_and_request_id(client):
    r = client.get("/api/v1/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
