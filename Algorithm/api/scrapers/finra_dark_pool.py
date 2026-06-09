"""FINRA OTC/Dark pool transparency data scraper.

FINRA publishes weekly off-exchange (dark pool) trading volume for all
securities. This is institutional "hidden" trading activity — one of the
most powerful free alternative data sources available.

Data published at: https://www.finra.org/finra-data/fintech/otc-transparency
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def fetch_finra_otc_data(symbol: str) -> Dict[str, Any]:
    """Fetch FINRA OTC transparency data for a symbol.

    Uses FINRA's public API to get weekly off-exchange volume.
    Returns dark pool volume metrics and institutional pressure score.
    """
    try:
        import httpx

        url = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
        payload = {
            "compareFilters": [
                {
                    "fieldName": "issueSymbolIdentifier",
                    "fieldValue": symbol,
                    "compareType": "EQUAL",
                }
            ],
            "sortFields": ["-weekStartDate"],
            "limit": 52,
            "offset": 0,
        }
        headers = {"Accept": "application/json", "Content-Type": "application/json"}

        resp = httpx.post(url, json=payload, headers=headers, timeout=15)
        if resp.status_code != 200:
            logger.warning("finra_otc_failed symbol=%s status=%d", symbol, resp.status_code)
            return _default_dark_pool_result(symbol)

        data = resp.json()
        records = data if isinstance(data, list) else data.get("data", [])
        if not records:
            return _default_dark_pool_result(symbol)

        total_otc_volume = sum(r.get("totalWeeklyShareQuantity", 0) or 0 for r in records[:8])
        avg_volume = total_otc_volume / min(8, len(records)) if records else 0

        recent = records[0] if records else {}
        latest_otc = recent.get("totalWeeklyShareQuantity", 0) or 0
        recent_vs_avg = (latest_otc / avg_volume) if avg_volume > 0 else 1.0

        if recent_vs_avg > 1.5:
            dark_pool_score = 75.0
            signal = "elevated_institutional"
        elif recent_vs_avg > 1.2:
            dark_pool_score = 65.0
            signal = "above_average"
        elif recent_vs_avg < 0.6:
            dark_pool_score = 35.0
            signal = "below_average"
        else:
            dark_pool_score = 50.0
            signal = "normal"

        return {
            "symbol": symbol,
            "dark_pool_score": dark_pool_score,
            "signal": signal,
            "latest_otc_volume": latest_otc,
            "avg_weekly_otc_volume": round(avg_volume, 0),
            "recent_vs_avg_ratio": round(recent_vs_avg, 2),
            "weeks_analyzed": min(8, len(records)),
        }

    except Exception as exc:
        logger.warning("finra_dark_pool_failed symbol=%s err=%s", symbol, exc)
        return _default_dark_pool_result(symbol)


def _default_dark_pool_result(symbol: str) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "dark_pool_score": 50.0,
        "signal": "no_data",
        "latest_otc_volume": 0,
        "avg_weekly_otc_volume": 0,
        "recent_vs_avg_ratio": 1.0,
        "weeks_analyzed": 0,
    }
