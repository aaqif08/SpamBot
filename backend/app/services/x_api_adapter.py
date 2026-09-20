"""X (Twitter) API v2 adapter.

Fetches an account profile and its recent tweets with an app-only **bearer
token** and maps them onto the ``AccountInput`` schema consumed by the feature
extractor. Nothing here is fabricated: if the token is missing, invalid, lacks
the required access level, or the account is protected/suspended, the adapter
raises an :class:`XApiError` with the API's own reason.

Endpoints used (read-only):

* ``GET /2/users/by/username/{username}`` — profile + ``public_metrics``
* ``GET /2/users/{id}/tweets``            — up to 100 most recent tweets

Access level: reading other users' tweets requires at least the *Basic*
tier of the X developer platform; the *Free* tier cannot call the tweets
endpoint (HTTP 403). The profile-only path still works there, and the adapter
then reports which fields could not be fetched.

Field mapping notes (v2 does not expose every v1.1 profile flag):

* ``default_profile_image`` — inferred from the profile image URL
  (``default_profile_images`` path).
* ``default_profile``, ``geo_enabled``, ``profile_background_tile``,
  ``has_profile_banner`` — not available in v2; sent as ``False`` and listed
  in ``unavailable_fields`` so the UI can say so.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import get_settings

log = logging.getLogger(__name__)

USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{1,15}$")
USER_FIELDS = "created_at,description,entities,location,profile_image_url,protected,public_metrics,url,verified,verified_type"
TWEET_FIELDS = "created_at,entities,public_metrics,referenced_tweets,text,lang"


class XApiError(RuntimeError):
    """Raised for any X API failure; ``status`` carries the upstream HTTP status."""

    def __init__(self, message: str, status: int = 502, *, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


@dataclass
class _CacheEntry:
    expires_at: float
    payload: dict[str, Any]


class XApiAdapter:
    name = "x_api"
    kind = "x-api-v2"

    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self._transport = transport
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    # ---- contract --------------------------------------------------------- #

    @property
    def description(self) -> str:
        s = get_settings()
        return (
            f"X (Twitter) API v2 adapter: profile via /2/users/by/username, up to {s.x_max_tweets} recent tweets via /2/users/:id/tweets. "
            "Requires BOTSHIELD_X_BEARER_TOKEN (Basic tier or higher to read tweets)."
        )

    def configured(self) -> bool:
        token = get_settings().x_bearer_token
        return bool(token and token.strip() and not token.strip().lower().startswith("your"))

    def fetch_account(self, identifier: str) -> dict[str, Any]:
        settings = get_settings()
        if not self.configured():
            raise XApiError("X API adapter is not configured: set BOTSHIELD_X_BEARER_TOKEN in .env and restart the backend.", 409)
        username = identifier.strip().lstrip("@")
        if not USERNAME_RE.match(username):
            raise XApiError("Identifier must be an X username (1-15 letters, digits or underscores), e.g. @nasa.", 422)

        cached = self._get_cached(username.lower())
        if cached is not None:
            return {**cached, "cached": True}

        headers = {"Authorization": f"Bearer {settings.x_bearer_token.strip()}", "User-Agent": "BotShield-AI/1.0"}
        with httpx.Client(base_url=settings.x_api_base_url, headers=headers, timeout=settings.x_timeout_seconds, transport=self._transport) as client:
            user = self._get(client, f"/users/by/username/{username}", {"user.fields": USER_FIELDS})
            data = user.get("data")
            if not data:
                err = (user.get("errors") or [{}])[0]
                raise XApiError(f"X API: {err.get('detail') or err.get('title') or 'user not found'}", 404)

            tweets: list[dict[str, Any]] = []
            tweets_error: str | None = None
            if data.get("protected"):
                tweets_error = "Account is protected; tweets are not readable with an app-only token."
            else:
                try:
                    tw = self._get(
                        client,
                        f"/users/{data['id']}/tweets",
                        {"max_results": str(max(5, min(100, settings.x_max_tweets))), "tweet.fields": TWEET_FIELDS},
                    )
                    tweets = tw.get("data") or []
                except XApiError as exc:
                    # Free-tier tokens cannot read tweets (403); keep the profile and say so.
                    if exc.status in (401, 403, 429):
                        tweets_error = str(exc)
                    else:
                        raise

        payload = self._to_payload(data, tweets, tweets_error)
        self._set_cached(username.lower(), payload)
        return payload

    # ---- helpers ----------------------------------------------------------- #

    def _get(self, client: httpx.Client, path: str, params: dict[str, str]) -> dict[str, Any]:
        try:
            resp = client.get(path, params=params)
        except httpx.TimeoutException as exc:
            raise XApiError(f"X API timeout after {get_settings().x_timeout_seconds:.0f}s", 504) from exc
        except httpx.HTTPError as exc:
            raise XApiError(f"X API connection error: {exc}", 502) from exc
        if resp.status_code == 429:
            reset = resp.headers.get("x-rate-limit-reset")
            wait = max(0, int(reset) - int(time.time())) if reset and reset.isdigit() else None
            raise XApiError(f"X API rate limit reached; retry in {wait or 'a few'} seconds.", 429, retry_after=wait)
        if resp.status_code == 401:
            raise XApiError("X API rejected the bearer token (401 Unauthorized). Check BOTSHIELD_X_BEARER_TOKEN.", 401)
        if resp.status_code == 403:
            detail = self._detail(resp)
            raise XApiError(f"X API access denied (403): {detail} — reading tweets requires the Basic tier or higher.", 403)
        if resp.status_code == 404:
            raise XApiError("X API: user not found (404).", 404)
        if resp.status_code >= 400:
            raise XApiError(f"X API error {resp.status_code}: {self._detail(resp)}", 502)
        try:
            return resp.json()
        except ValueError as exc:
            raise XApiError("X API returned a non-JSON response", 502) from exc

    @staticmethod
    def _detail(resp: httpx.Response) -> str:
        try:
            body = resp.json()
        except ValueError:
            return resp.text[:200]
        if isinstance(body, dict):
            errs = body.get("errors")
            if errs and isinstance(errs, list):
                return str(errs[0].get("detail") or errs[0].get("message") or errs[0].get("title") or body)
            return str(body.get("detail") or body.get("title") or body.get("reason") or body)[:300]
        return str(body)[:300]

    @staticmethod
    def _to_payload(u: dict[str, Any], tweets: list[dict[str, Any]], tweets_error: str | None) -> dict[str, Any]:
        pm = u.get("public_metrics") or {}
        image = u.get("profile_image_url") or ""
        ent = u.get("entities") or {}
        expanded_url = ""
        try:
            expanded_url = (ent.get("url", {}).get("urls") or [{}])[0].get("expanded_url") or u.get("url") or ""
        except (AttributeError, IndexError, TypeError):
            expanded_url = u.get("url") or ""

        mapped_tweets = []
        for t in tweets:
            te = t.get("entities") or {}
            tm = t.get("public_metrics") or {}
            mapped_tweets.append(
                {
                    "text": t.get("text", ""),
                    "retweet_count": int(tm.get("retweet_count", 0) or 0),
                    "reply_count": int(tm.get("reply_count", 0) or 0),
                    "favorite_count": int(tm.get("like_count", 0) or 0),
                    "num_hashtags": len(te.get("hashtags") or []),
                    "num_mentions": len(te.get("mentions") or []),
                    "num_urls": len(te.get("urls") or []),
                }
            )

        account = {
            "account_id": u.get("username"),
            "screen_name": u.get("username"),
            "name": u.get("name"),
            "verified": bool(u.get("verified")) or (u.get("verified_type") not in (None, "", "none")),
            "friends_count": int(pm.get("following_count", 0) or 0),
            "followers_count": int(pm.get("followers_count", 0) or 0),
            "listed_count": int(pm.get("listed_count", 0) or 0),
            "favorites_count": int(pm.get("like_count", 0) or 0),
            "statuses_count": int(pm.get("tweet_count", 0) or 0),
            "description": u.get("description") or "",
            "location": u.get("location") or "",
            "url": expanded_url,
            "default_profile": False,
            "default_profile_image": "default_profile_images" in image or not image,
            "geo_enabled": False,
            "profile_background_tile": False,
            "has_profile_banner": False,
            "tweets": mapped_tweets,
        }
        unavailable = ["default_profile", "geo_enabled", "profile_background_tile", "has_profile_banner"]
        if "like_count" not in pm:
            unavailable.append("favorites_count")
        return {
            "account": account,
            "source": "x_api",
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user_id": u.get("id"),
            "created_at": u.get("created_at"),
            "protected": bool(u.get("protected")),
            "tweets_fetched": len(mapped_tweets),
            "tweets_error": tweets_error,
            "unavailable_fields": unavailable,
            "notice": "Live data from the X API v2 (app-only bearer token). Fields not exposed by v2 are sent as False and listed in unavailable_fields.",
        }

    # ---- cache ---------------------------------------------------------------- #

    def _get_cached(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry and entry.expires_at > time.monotonic():
                return entry.payload
            self._cache.pop(key, None)
            return None

    def _set_cached(self, key: str, payload: dict[str, Any]) -> None:
        ttl = get_settings().x_cache_ttl_seconds
        if ttl <= 0:
            return
        with self._lock:
            self._cache[key] = _CacheEntry(expires_at=time.monotonic() + ttl, payload=payload)
