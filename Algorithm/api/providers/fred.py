from __future__ import annotations

import httpx

from api.providers import ProviderHealth
from api.providers.base import BaseProvider, ProviderError, get_env, has_env


class FREDProvider(BaseProvider):
    name = "fred"

    def enabled(self) -> bool:
        return has_env("FRED_API_KEY")

    def health_check(self) -> ProviderHealth:
        if not self.enabled():
            return ProviderHealth(
                name=self.name,
                enabled=False,
                status="down",
                message="FRED_API_KEY not set",
            )

        api_key = get_env("FRED_API_KEY", "")
        url = "https://api.stlouisfed.org/fred/series"
        params = {"series_id": "GDP", "api_key": api_key, "file_type": "json"}

        try:
            response = httpx.get(url, params=params, timeout=5.0)
            data = response.json() if response.status_code == 200 else None
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc

        if response.status_code != 200:
            return ProviderHealth(
                name=self.name,
                enabled=True,
                status="down",
                message="FRED responded with non-200 status",
            )

        if not isinstance(data, dict) or "seriess" not in data:
            return ProviderHealth(
                name=self.name,
                enabled=True,
                status="down",
                message="FRED payload missing 'seriess'",
            )

        return ProviderHealth(name=self.name, enabled=True, status="ok", message="")
