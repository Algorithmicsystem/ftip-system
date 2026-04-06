from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import requests

from api import config

from .errors import ProviderUnavailable, SymbolNoData
from .symbols import canonical_symbol

BASE_URL = "https://finnhub.io/api/v1"


def _request(path: str, params: Dict[str, Any]) -> Any:
    api_key = config.finnhub_api_key()
    if not api_key:
        raise ProviderUnavailable("PROVIDER_UNAVAILABLE", "FINNHUB_API_KEY not set")
    response = requests.get(
        f"{BASE_URL}{path}",
        params={**params, "token": api_key},
        timeout=config.data_fabric_timeout_seconds(),
    )
    if response.status_code != 200:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE", f"Finnhub HTTP {response.status_code}"
        )
    payload = response.json()
    if payload in ({}, [], None):
        raise SymbolNoData("NO_DATA", "Finnhub returned no data")
    return payload


def fetch_company_news(
    symbol: str,
    from_date: dt.date,
    to_date: dt.date,
) -> List[Dict[str, object]]:
    payload = _request(
        "/company-news",
        {
            "symbol": canonical_symbol(symbol),
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
        },
    )
    if not isinstance(payload, list):
        raise SymbolNoData("NO_DATA", "Finnhub company news payload is not a list")

    items: List[Dict[str, object]] = []
    for row in payload:
        timestamp = row.get("datetime")
        if timestamp is None:
            continue
        published_at = dt.datetime.fromtimestamp(
            int(timestamp), tz=dt.timezone.utc
        )
        items.append(
            {
                "symbol": canonical_symbol(symbol),
                "published_at": published_at,
                "source": "finnhub_news",
                "title": row.get("headline") or "",
                "url": row.get("url") or "",
                "content_snippet": row.get("summary"),
                "category": row.get("category"),
                "image": row.get("image"),
                "related": row.get("related"),
            }
        )
    return [item for item in items if item.get("title") and item.get("url")]


def fetch_basic_financials(symbol: str) -> Dict[str, Any]:
    payload = _request(
        "/stock/metric",
        {"symbol": canonical_symbol(symbol), "metric": "all"},
    )
    metric = payload.get("metric") or {}
    if not isinstance(metric, dict) or not metric:
        raise SymbolNoData("NO_DATA", "Finnhub metrics were empty")
    return {
        "symbol": canonical_symbol(symbol),
        "52_week_high": _float_or_none(metric.get("52WeekHigh")),
        "52_week_low": _float_or_none(metric.get("52WeekLow")),
        "market_cap": _float_or_none(metric.get("marketCapitalization")),
        "beta": _float_or_none(metric.get("beta")),
        "net_margin": _float_or_none(metric.get("netMargin")),
        "operating_margin_ttm": _float_or_none(metric.get("operatingMarginTTM")),
        "revenue_growth_ttm_yoy": _float_or_none(metric.get("revenueGrowthTTMYoy")),
        "eps_growth_5y": _float_or_none(metric.get("epsGrowth5Y")),
        "roa_ttm": _float_or_none(metric.get("roaTTM")),
        "roe_ttm": _float_or_none(metric.get("roeTTM")),
        "quick_ratio_quarterly": _float_or_none(metric.get("quickRatioQuarterly")),
        "current_ratio_quarterly": _float_or_none(metric.get("currentRatioQuarterly")),
        "total_debt_to_equity_quarterly": _float_or_none(
            metric.get("totalDebt/totalEquityQuarterly")
        ),
        "source": "finnhub_basic_financials",
    }


def fetch_company_profile(symbol: str) -> Dict[str, Any]:
    payload = _request("/stock/profile2", {"symbol": canonical_symbol(symbol)})
    if not isinstance(payload, dict) or not payload.get("ticker"):
        raise SymbolNoData("NO_DATA", "Finnhub company profile was empty")
    return {
        "symbol": canonical_symbol(symbol),
        "ticker": payload.get("ticker"),
        "name": payload.get("name"),
        "country": payload.get("country"),
        "currency": payload.get("currency"),
        "exchange": payload.get("exchange"),
        "finnhub_industry": payload.get("finnhubIndustry"),
        "ipo": payload.get("ipo"),
        "market_capitalization": _float_or_none(payload.get("marketCapitalization")),
        "share_outstanding": _float_or_none(payload.get("shareOutstanding")),
        "source": "finnhub_profile",
    }


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, "", "None", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
