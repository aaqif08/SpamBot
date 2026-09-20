"""Storage backends (local + S3 via moto) and the X API provider (mocked HTTP)."""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest

from app.core.config import get_settings
from app.core.storage import LocalStorage, S3Storage, StorageError
from app.services.x_api_adapter import XApiAdapter, XApiError


def _exercise(storage) -> None:  # noqa: ANN001
    storage.put_bytes("org/a/x/one.txt", b"hello")
    storage.put_bytes("org/a/x/two.bin", b"\x00\x01")
    assert storage.exists("org/a/x/one.txt") and not storage.exists("org/a/x/none")
    assert storage.get_bytes("org/a/x/one.txt") == b"hello"
    assert storage.size("org/a/x/two.bin") == 2
    assert storage.list("org/a/x") == ["org/a/x/one.txt", "org/a/x/two.bin"]
    d = storage.local_dir("org/a/x")
    assert (Path(d) / "one.txt").read_bytes() == b"hello"
    assert storage.delete_prefix("org/a/x") == 2
    assert storage.list("org/a/x") == []
    with pytest.raises(StorageError):
        storage.get_bytes("org/a/x/one.txt")


def test_local_storage(tmp_path):
    st = LocalStorage(tmp_path / "root")
    _exercise(st)
    with pytest.raises(StorageError):
        st.put_bytes("../escape.txt", b"x")
    with pytest.raises(StorageError):
        st.get_bytes("org/../../etc/passwd")
    src = tmp_path / "artifacts"
    src.mkdir()
    (src / "a.json").write_text("{}")
    (src / "sub").mkdir()
    (src / "sub" / "b.npy").write_bytes(b"npy")
    keys = st.upload_dir(src, "org/a/models/m1")
    assert keys == ["org/a/models/m1/a.json", "org/a/models/m1/sub/b.npy"]


def test_s3_storage_with_moto(tmp_path, monkeypatch):
    moto = pytest.importorskip("moto")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with moto.mock_aws():
        import boto3

        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="botshield-test")
        st = S3Storage(bucket="botshield-test", prefix="tenant", region="us-east-1", cache_dir=tmp_path / "cache")
        _exercise(st)
        assert st.describe()["backend"] == "s3"


# ---- X API provider --------------------------------------------------------- #

USER = {"data": {"id": "12", "name": "Example", "username": "example_acct", "description": "Photographer", "profile_image_url": "https://pbs.twimg.com/profile_images/1/abc_normal.jpg", "protected": False, "verified": False, "public_metrics": {"followers_count": 540, "following_count": 410, "tweet_count": 4120, "listed_count": 7, "like_count": 2300}}}
TWEETS = {"data": [{"id": "1", "text": "Sunset run #running http://example.org/a @friend", "public_metrics": {"retweet_count": 2, "reply_count": 1, "like_count": 9}, "entities": {"hashtags": [{"tag": "running"}], "urls": [{"url": "http://example.org/a"}], "mentions": [{"username": "friend"}]}}]}


@pytest.fixture
def token(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "x_bearer_token", "test-token")
    monkeypatch.setattr(s, "x_cache_ttl_seconds", 0)


def _adapter(handler) -> XApiAdapter:  # noqa: ANN001
    return XApiAdapter(transport=httpx.MockTransport(handler))


def test_x_not_configured(monkeypatch):
    monkeypatch.setattr(get_settings(), "x_bearer_token", None)
    with pytest.raises(XApiError) as exc:
        XApiAdapter().fetch_account("nasa")
    assert exc.value.status == 409


def test_x_fetch_maps_profile_and_tweets(token):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-token"
        return httpx.Response(200, json=USER if "by/username" in request.url.path else TWEETS)

    out = _adapter(handler).fetch_account("@example_acct")
    acc = out["account"]
    assert out["source"] == "x_api" and out["tweets_fetched"] == 1
    assert acc["followers_count"] == 540 and acc["friends_count"] == 410 and acc["favorites_count"] == 2300
    assert acc["tweets"][0]["num_hashtags"] == 1 and acc["default_profile_image"] is False
    from app.schemas.api import AccountInput
    from ml.features import FeatureExtractor

    feats = FeatureExtractor().transform(AccountInput(**acc).to_account_dict()).features
    assert feats["hashtag_count"] == 1 and feats["ffratio"] == pytest.approx(540 / 411)


def test_x_error_mapping(token):
    with pytest.raises(XApiError) as e401:
        _adapter(lambda r: httpx.Response(401, json={"title": "Unauthorized"})).fetch_account("example_acct")
    assert e401.value.status == 401
    reset = str(int(time.time()) + 30)
    with pytest.raises(XApiError) as e429:
        _adapter(lambda r: httpx.Response(429, headers={"x-rate-limit-reset": reset}, json={})).fetch_account("example_acct")
    assert e429.value.status == 429 and e429.value.retry_after is not None
    with pytest.raises(XApiError) as e422:
        _adapter(lambda r: httpx.Response(200, json=USER)).fetch_account("bad handle!")
    assert e422.value.status == 422

    def free_tier(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=USER) if "by/username" in request.url.path else httpx.Response(403, json={"detail": "not authorized"})

    out = _adapter(free_tier).fetch_account("example_acct")
    assert out["tweets_fetched"] == 0 and "Basic tier" in out["tweets_error"]


def test_x_provider_endpoint_configured(client, analyst_a, monkeypatch):
    monkeypatch.setattr(get_settings(), "x_bearer_token", "test-token")
    ps = {p["name"]: p for p in client.get("/api/v1/providers", headers=analyst_a).json()}
    assert ps["x_api"]["configured"] is True
    r = client.post("/api/v1/providers/x_api/fetch", headers=analyst_a, json={"identifier": "bad handle"})
    assert r.status_code == 422
