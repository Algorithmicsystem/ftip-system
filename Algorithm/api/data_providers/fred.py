from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import requests

from api import config

from .errors import ProviderUnavailable, SymbolNoData

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_series(
    series_id: str,
    *,
    start_date: Optional[dt.date] = None,
    end_date: Optional[dt.date] = None,
    limit: int = 24,
) -> Dict[str, Any]:
    api_key = config.fred_api_key()
    if not api_key:
        raise ProviderUnavailable("PROVIDER_UNAVAILABLE", "FRED_API_KEY not set")
    params: Dict[str, Any] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": max(1, min(limit, 120)),
    }
    if start_date is not None:
        params["observation_start"] = start_date.isoformat()
    if end_date is not None:
        params["observation_end"] = end_date.isoformat()
    response = requests.get(
        BASE_URL,
        params=params,
        timeout=config.data_fabric_timeout_seconds(),
    )
    if response.status_code != 200:
        raise ProviderUnavailable("PROVIDER_UNAVAILABLE", f"FRED HTTP {response.status_code}")
    payload = response.json()
    observations = payload.get("observations") or []
    if not isinstance(observations, list) or not observations:
        raise SymbolNoData("NO_DATA", f"FRED returned no observations for {series_id}")
    parsed: List[Dict[str, Any]] = []
    for row in observations:
        date_text = row.get("date")
        if not date_text:
            continue
        parsed.append(
            {
                "date": date_text,
                "value": _float_or_none(row.get("value")),
            }
        )
    parsed = [row for row in parsed if row.get("value") is not None]
    if not parsed:
        raise SymbolNoData("NO_DATA", f"FRED returned no numeric values for {series_id}")
    return {
        "series_id": series_id,
        "title": payload.get("seriess", [{}])[0].get("title")
        if isinstance(payload.get("seriess"), list)
        else None,
        "observations": parsed,
        "source": "fred",
    }


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, "", ".", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
