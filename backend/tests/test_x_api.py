"""X API v2 adapter tests with a mocked HTTP transport (no network)."""

from __future__ import annotations

import json
import time

import httpx
import pytest

from app.core.config import get_settings
from app.services.x_api_adapter import XApiAdapter, XApiError

USER = {
    "data": {
        "id": "12",
        "name": "Example Account",
        "username": "example_acct",
        "description": "Photographer & coffee nerd. Views my own.",
        "location": "Berlin",
        "profile_image_url": "https://pbs.twimg.com/profile_images/1/abc_normal.jpg",
        "protected": False,
        "verified": False,
        "verified_type": "none",
        "created_at": "2015-03-01T10:00:00.000Z",
        "entities": {"url": {"urls": [{"expanded_url": "https://example.org"}]}},
        "public_metrics": {"followers_count": 540, "following_count": 410, "tweet_count": 4120, "listed_count": 7, "like_count": 2300},
    }
}
TWEETS = {
    "data": [
        {
            "id": "1",
            "text": "Sunset run today #running http://example.org/a @friend",
            "public_metrics": {"retweet_count": 2, "reply_count": 1, "like_count": 9},
            "entities": {"hashtags": [{"tag": "running"}], "urls": [{"url": "http://example.org/a"}], "mentions": [{"username": "friend"}]},
        },
        {"id": "2", "text": "Coffee, rain, and a good book.", "public_metrics": {"retweet_count": 0, "reply_count": 0, "like_count": 3}},
    ]
}


def _adapter(handler) -> XApiAdapter:
    return XApiAdapter(transport=httpx.MockTransport(handler))


@pytest.fixture
def token(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "x_bearer_token", "test-token")
    monkeypatch.setattr(s, "x_cache_ttl_seconds", 0)
    yield s


def test_not_configured(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "x_bearer_token", None)
    a = XApiAdapter()
    assert a.configured() is False
    with pytest.raises(XApiError) as exc:
        a.fetch_account("nasa")
    assert exc.value.status == 409


def test_placeholder_token_is_not_configured(monkeypatch):
    monkeypatch.setattr(get_settings(), "x_bearer_token", "YOUR_TOKEN_HERE")
    assert XApiAdapter().configured() is False


def test_invalid_username(token):
    with pytest.raises(XApiError) as exc:
        _adapter(lambda r: httpx.Response(200, json=USER)).fetch_account("not a handle!")
    assert exc.value.status == 422


def test_fetch_maps_profile_and_tweets(token):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("authorization")))
        if request.url.path.endswith("/users/by/username/example_acct"):
            return httpx.Response(200, json=USER)
        if request.url.path.endswith("/users/12/tweets"):
            assert request.url.params["max_results"] == "100"
            return httpx.Response(200, json=TWEETS)
        return httpx.Response(404, json={"errors": [{"detail": "nope"}]})

    out = _adapter(handler).fetch_account("@example_acct")
    assert seen[0][1] == "Bearer test-token"
    acc = out["account"]
    assert out["source"] == "x_api" and out["tweets_fetched"] == 2 and out["tweets_error"] is None
    assert acc["followers_count"] == 540 and acc["friends_count"] == 410 and acc["statuses_count"] == 4120
    assert acc["favorites_count"] == 2300 and acc["listed_count"] == 7
    assert acc["url"] == "https://example.org" and acc["default_profile_image"] is False
    assert acc["tweets"][0] == {"text": "Sunset run today #running http://example.org/a @friend", "retweet_count": 2, "reply_count": 1, "favorite_count": 9, "num_hashtags": 1, "num_mentions": 1, "num_urls": 1}
    assert "default_profile" in out["unavailable_fields"]

    # The mapped payload must be a valid AccountInput for the feature extractor.
    from app.schemas.api import AccountInput
    from ml.features import FeatureExtractor

    feats = FeatureExtractor().transform(AccountInput(**acc).to_account_dict()).features
    assert feats["hashtag_count"] == 1 and feats["url_count"] == 1 and feats["ffratio"] == pytest.approx(540 / 411)


def test_free_tier_403_on_tweets_keeps_profile(token):
    def handler(request: httpx.Request) -> httpx.Response:
        if "by/username" in request.url.path:
            return httpx.Response(200, json=USER)
        return httpx.Response(403, json={"detail": "Your account is not authorized for this endpoint", "title": "Forbidden"})

    out = _adapter(handler).fetch_account("example_acct")
    assert out["tweets_fetched"] == 0
    assert "Basic tier" in out["tweets_error"]


def test_bad_token_401(token):
    with pytest.raises(XApiError) as exc:
        _adapter(lambda r: httpx.Response(401, json={"title": "Unauthorized"})).fetch_account("example_acct")
    assert exc.value.status == 401


def test_rate_limit_429(token):
    reset = str(int(time.time()) + 30)
    with pytest.raises(XApiError) as exc:
        _adapter(lambda r: httpx.Response(429, headers={"x-rate-limit-reset": reset}, json={})).fetch_account("example_acct")
    assert exc.value.status == 429 and exc.value.retry_after is not None


def test_user_not_found(token):
    body = {"errors": [{"detail": "Could not find user with username: [ghost]", "title": "Not Found Error"}]}
    with pytest.raises(XApiError) as exc:
        _adapter(lambda r: httpx.Response(200, json=body)).fetch_account("ghost")
    assert exc.value.status == 404


def test_cache_hit(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "x_bearer_token", "test-token")
    monkeypatch.setattr(s, "x_cache_ttl_seconds", 60)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=USER if "by/username" in request.url.path else TWEETS)

    a = _adapter(handler)
    a.fetch_account("example_acct")
    out = a.fetch_account("example_acct")
    assert calls["n"] == 2 and out.get("cached") is True


def test_adapter_endpoint_reports_status(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "x_bearer_token", None)
    listing = {a["name"]: a for a in client.get("/api/adapters").json()}
    assert listing["x_api"]["configured"] is False
    assert client.post("/api/adapters/x_api/fetch", json={"identifier": "nasa"}).status_code == 409

    monkeypatch.setattr(get_settings(), "x_bearer_token", "test-token")
    assert {a["name"]: a for a in client.get("/api/adapters").json()}["x_api"]["configured"] is True
    r = client.post("/api/adapters/x_api/fetch", json={"identifier": "bad handle"})
    assert r.status_code == 422
    assert "username" in r.json()["detail"]
    json.dumps(listing)  # serialisable
