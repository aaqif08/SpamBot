"""Social-data providers — the ingestion side, separated from ML inference.

A provider turns an external identifier into an ``AccountInput`` dict. Providers
report whether they are configured; nothing is ever fabricated when they are not.

* ``manual``  — data entered in the UI / API body (always available)
* ``csv``     — rows of an uploaded CSV (batch analysis)
* ``x_api``   — X (Twitter) API v2 via ``BOTSHIELD_X_BEARER_TOKEN``
"""

from __future__ import annotations

from typing import Any, Protocol

from app.services.x_api_adapter import XApiAdapter, XApiError  # noqa: F401  (re-exported)


class SocialDataProvider(Protocol):
    name: str
    kind: str
    description: str

    def configured(self) -> bool: ...

    def fetch_account(self, identifier: str) -> dict[str, Any]: ...


class ManualProvider:
    name = "manual"
    kind = "form"
    description = "Account fields entered manually or posted to the API"

    def configured(self) -> bool:
        return True

    def fetch_account(self, identifier: str) -> dict[str, Any]:
        raise NotImplementedError("The manual provider does not fetch; submit the account fields directly")


class CsvProvider:
    name = "csv"
    kind = "file"
    description = "Accounts supplied as rows of an uploaded CSV (batch analysis)"

    def configured(self) -> bool:
        return True

    def fetch_account(self, identifier: str) -> dict[str, Any]:
        raise NotImplementedError("The CSV provider is used through batch analysis")


PROVIDERS: dict[str, SocialDataProvider] = {p.name: p for p in (ManualProvider(), CsvProvider(), XApiAdapter())}


def list_providers() -> list[dict[str, Any]]:
    return [
        {
            "name": p.name,
            "kind": p.kind,
            "configured": p.configured(),
            "description": p.description,
            "fetchable": p.name == "x_api",
            "configuration_hint": None if p.configured() else "Set BOTSHIELD_X_BEARER_TOKEN on the backend and restart the API (see docs/deployment.md).",
        }
        for p in PROVIDERS.values()
    ]
