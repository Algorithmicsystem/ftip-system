from __future__ import annotations

import httpx

from api.providers import ProviderHealth
from api.providers.base import BaseProvider, ProviderError, get_env, has_env


class FinnhubProvider(BaseProvider):
    name = "finnhub"

    def enabled(self) -> bool:
        return has_env("FINNHUB_API_KEY")

    def health_check(self) -> ProviderHealth:
        if not self.enabled():
            return ProviderHealth(
                name=self.name,
                enabled=False,
                status="down",
                message="FINNHUB_API_KEY not set",
            )

        api_key = get_env("FINNHUB_API_KEY", "")
        url = "https://finnhub.io/api/v1/stock/symbol"
        params = {"exchange": "US", "token": api_key}

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
                message="Finnhub responded with non-200 status",
            )

        if not isinstance(data, list) or len(data) == 0:
            return ProviderHealth(
                name=self.name,
                enabled=True,
                status="down",
                message="Finnhub symbol list was empty",
            )

        return ProviderHealth(name=self.name, enabled=True, status="ok", message="")
