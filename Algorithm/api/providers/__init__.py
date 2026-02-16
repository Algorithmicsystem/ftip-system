from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel

from api.providers.base import ProviderError


class ProviderHealth(BaseModel):
    name: str
    enabled: bool
    status: Literal["ok", "degraded", "down"]
    message: str = ""


class ProvidersHealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    providers: List[ProviderHealth]


def _build_providers():
    from api.providers.finnhub import FinnhubProvider
    from api.providers.fred import FREDProvider
    from api.providers.sec_edgar import SecEdgarProvider

    return [FinnhubProvider(), FREDProvider(), SecEdgarProvider()]


def get_providers_health() -> ProvidersHealthResponse:
    providers = _build_providers()
    provider_health = []

    for provider in providers:
        try:
            provider_health.append(provider.health_check())
        except ProviderError as exc:
            provider_health.append(
                ProviderHealth(
                    name=provider.name,
                    enabled=provider.enabled(),
                    status="down",
                    message=str(exc),
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            provider_health.append(
                ProviderHealth(
                    name=provider.name,
                    enabled=provider.enabled(),
                    status="down",
                    message="Unexpected error: %s" % (exc,),
                )
            )

    enabled_results = [result for result in provider_health if result.enabled]
    has_down = any(result.status == "down" for result in enabled_results)
    has_degraded = any(result.status == "degraded" for result in provider_health)

    if has_down:
        overall_status = "down"
    elif has_degraded:
        overall_status = "degraded"
    else:
        overall_status = "ok"

    return ProvidersHealthResponse(status=overall_status, providers=provider_health)


__all__ = ["ProviderHealth", "get_providers_health"]
