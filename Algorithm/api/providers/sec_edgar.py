from __future__ import annotations

import httpx

from api.providers import ProviderHealth
from api.providers.base import BaseProvider, ProviderError, get_env


DEFAULT_SEC_USER_AGENT = "CFOTwin/0.1 (no-email)"


class SecEdgarProvider(BaseProvider):
    name = "sec_edgar"

    def enabled(self) -> bool:
        return True

    def health_check(self) -> ProviderHealth:
        configured_user_agent = get_env("SEC_USER_AGENT")
        degraded = configured_user_agent is None
        user_agent = configured_user_agent or DEFAULT_SEC_USER_AGENT

        headers = {
            "User-Agent": user_agent,
            "Accept": "application/json",
            "Host": "data.sec.gov",
        }

        try:
            response = httpx.get(
                "https://data.sec.gov/submissions/CIK0000320193.json",
                headers=headers,
                timeout=5.0,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc

        if response.status_code != 200:
            return ProviderHealth(
                name=self.name,
                enabled=True,
                status="down",
                message="SEC EDGAR responded with non-200 status",
            )

        if degraded:
            return ProviderHealth(
                name=self.name,
                enabled=True,
                status="degraded",
                message="SEC_USER_AGENT not set; configure a real user agent",
            )

        return ProviderHealth(name=self.name, enabled=True, status="ok", message="")
