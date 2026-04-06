from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from api import config

from .errors import ProviderUnavailable, SymbolNoData

BASE_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"


def fetch_indicator(
    *,
    country: str,
    indicator: str,
    per_page: int = 24,
) -> Dict[str, Any]:
    if not config.world_bank_enabled():
        raise ProviderUnavailable("PROVIDER_UNAVAILABLE", "World Bank is disabled")
    response = requests.get(
        BASE_URL.format(country=country, indicator=indicator),
        params={
            "format": "json",
            "per_page": max(1, min(per_page, 60)),
            "mrv": max(1, min(per_page, 60)),
        },
        timeout=config.data_fabric_timeout_seconds(),
    )
    if response.status_code != 200:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE", f"World Bank HTTP {response.status_code}"
        )
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise SymbolNoData("NO_DATA", "World Bank returned no series payload")
    meta = payload[0] or {}
    rows = payload[1] or []
    if not isinstance(rows, list) or not rows:
        raise SymbolNoData("NO_DATA", "World Bank returned no observations")
    observations: List[Dict[str, Any]] = []
    for row in rows:
        value = _float_or_none(row.get("value"))
        if value is None:
            continue
        observations.append(
            {
                "date": row.get("date"),
                "value": value,
                "country": ((row.get("country") or {}).get("value")),
                "indicator": ((row.get("indicator") or {}).get("value")),
            }
        )
    if not observations:
        raise SymbolNoData("NO_DATA", "World Bank returned no numeric observations")
    return {
        "country": country,
        "indicator": indicator,
        "page": meta,
        "observations": observations,
        "source": "world_bank",
    }


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
