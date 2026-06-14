"""EPA ECHO (Enforcement and Compliance History Online) scraper.

Uses the free public EPA ECHO API to fetch environmental violations and
enforcement actions for public companies.

API docs: https://echo.epa.gov/tools/web-services/
No API key required. Rate limit: ~1,000 requests/day per IP.

Results are cached for 7 days — EPA data changes slowly.
Stores results in the epa_violations table (migration 115).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)

_ECHO_FACILITY_URL = "https://echo.epa.gov/tools/web-services/facility-search/facilities"
_CACHE_DAYS = 7


def _fetch_echo_facilities(
    company_name: str,
    timeout: int = 20,
) -> Optional[List[Dict[str, Any]]]:
    """Search EPA ECHO for active facilities matching company_name."""
    try:
        import urllib.parse
        import urllib.request

        params = {
            "output": "JSON",
            "p_nm": company_name,
            "p_act": "Y",
            "p_med": "AIR,WAT,HAZ",
            "responseset": "50",
        }
        url = _ECHO_FACILITY_URL + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())

        results = data.get("Results", {})
        return results.get("Facilities", []) or []
    except Exception as exc:
        logger.debug("epa_echo.facility_search_failed company=%s err=%s", company_name, exc)
        return None


def _is_cached(ticker: str, cache_days: int = _CACHE_DAYS) -> bool:
    """Return True if we have a fresh EPA record for this ticker."""
    from api import db
    if not db.db_read_enabled():
        return False
    try:
        row = db.safe_fetchone(
            """
            SELECT 1 FROM epa_violations
            WHERE symbol = %s
              AND updated_at >= now() - (%s || ' days')::interval
            LIMIT 1
            """,
            (ticker, cache_days),
        )
        return row is not None
    except Exception:
        return False


def fetch_epa_violations(ticker: str, company_name: str) -> Dict[str, Any]:
    """Fetch EPA ECHO violation data for a company.

    Returns a dict with:
        violation_count_3yr, total_penalties_usd, esg_risk_score,
        facilities_count, source, as_of_date.

    All values default to 0 when the API is unavailable or returns no data.
    """
    today = dt.date.today()
    empty: Dict[str, Any] = {
        "ticker": ticker,
        "violation_count_3yr": 0,
        "total_penalties_usd": 0.0,
        "esg_risk_score": 0.0,
        "facilities_count": 0,
        "source": "EPA_ECHO",
        "as_of_date": today.isoformat(),
    }

    facilities = _fetch_echo_facilities(company_name)
    if facilities is None or len(facilities) == 0:
        return empty

    total_violations = 0
    total_penalties = 0.0

    for fac in facilities[:20]:
        try:
            # Quarters in violation over last 3 years (each medium)
            air_viols = int(fac.get("CAA3YrQtrsInViol") or 0)
            water_viols = int(fac.get("CWA3YrQtrsInViol") or 0)
            haz_viols = int(fac.get("RCRA3YrQtrsInViol") or 0)
            total_violations += air_viols + water_viols + haz_viols

            # Penalty scores (EPA ECHO provides aggregate compliance scores)
            air_score = float(fac.get("CAAScore") or 0)
            water_score = float(fac.get("CWAScore") or 0)
            total_penalties += air_score + water_score
        except (TypeError, ValueError):
            continue

    # ESG risk: violations drive base score; penalties compound logarithmically
    esg_risk = min(100.0, total_violations * 5.0 + math.log1p(total_penalties) * 10.0)

    result: Dict[str, Any] = {
        "ticker": ticker,
        "violation_count_3yr": total_violations,
        "total_penalties_usd": round(total_penalties, 2),
        "esg_risk_score": round(esg_risk, 2),
        "facilities_count": len(facilities),
        "source": "EPA_ECHO",
        "as_of_date": today.isoformat(),
    }
    logger.debug(
        "epa_echo.fetched ticker=%s violations=%d esg_risk=%.1f",
        ticker, total_violations, esg_risk,
    )
    return result


def fetch_bulk_epa_violations(
    tickers: List[str],
    *,
    skip_cached: bool = True,
    delay_seconds: float = 1.5,
) -> List[Dict[str, Any]]:
    """Fetch EPA violations for a list of tickers.

    Skips tickers that have a fresh DB record (within _CACHE_DAYS) when
    skip_cached=True. Adds delay_seconds between requests.
    """
    import time
    from api.scrapers.entity_resolver import get_company_name

    results = []
    for ticker in tickers:
        if skip_cached and _is_cached(ticker):
            logger.debug("epa_echo.cache_hit ticker=%s", ticker)
            continue
        company_name = get_company_name(ticker)
        rec = fetch_epa_violations(ticker, company_name)
        results.append(rec)
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return results


def store_epa_violations(records: List[Dict[str, Any]]) -> int:
    """Persist EPA violation records to the epa_violations table."""
    from api import db

    if not db.db_write_enabled():
        return 0

    written = 0
    for rec in records:
        try:
            db.safe_execute(
                """
                INSERT INTO epa_violations (
                    symbol, as_of_date, violation_count_3yr, total_penalties_usd,
                    esg_risk_score, facilities_count, source, raw
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (symbol, as_of_date) DO UPDATE SET
                    violation_count_3yr = EXCLUDED.violation_count_3yr,
                    total_penalties_usd = EXCLUDED.total_penalties_usd,
                    esg_risk_score      = EXCLUDED.esg_risk_score,
                    facilities_count    = EXCLUDED.facilities_count,
                    source              = EXCLUDED.source,
                    raw                 = EXCLUDED.raw,
                    updated_at          = now()
                """,
                (
                    rec["ticker"],
                    rec["as_of_date"],
                    rec["violation_count_3yr"],
                    rec["total_penalties_usd"],
                    rec["esg_risk_score"],
                    rec["facilities_count"],
                    rec["source"],
                    json.dumps(rec),
                ),
            )
            written += 1
        except Exception as exc:
            logger.debug("epa_violations.store_failed ticker=%s err=%s", rec.get("ticker"), exc)

    logger.info("epa_violations.stored written=%d total=%d", written, len(records))
    return written
