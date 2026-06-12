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
        # sortFields causes 400 unless all partition keys (weekStartDate, tierIdentifier)
        # are specified as EQUAL filters — omit it and sort client-side.
        payload = {
            "compareFilters": [
                {
                    "fieldName": "issueSymbolIdentifier",
                    "fieldValue": symbol,
                    "compareType": "EQUAL",
                }
            ],
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

        # Sort descending by weekStartDate since we can't use sortFields in the request
        records = sorted(
            records,
            key=lambda r: r.get("weekStartDate") or "",
            reverse=True,
        )

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


def fetch_dark_pool_data(symbols: list) -> dict:
    """Fetch FINRA dark pool data for all symbols in parallel.

    Runs up to 10 concurrent calls with a small delay.
    Returns: {symbol: dark_pool_result_dict}
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict = {}

    def _fetch_one(sym: str) -> tuple:
        _time.sleep(0.05)
        return sym, fetch_finra_otc_data(sym)

    n_workers = min(10, len(symbols)) if symbols else 1
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_fetch_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym, data = future.result()
            results[sym] = data

    logger.info("dark_pool_bulk_fetch symbols=%d", len(results))
    return results


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
