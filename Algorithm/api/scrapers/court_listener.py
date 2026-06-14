"""CourtListener litigation risk scraper.

Uses the free public CourtListener API to fetch federal court dockets
mentioning a company, classifies cases by type, and computes a litigation
risk score.

API: https://www.courtlistener.com/api/rest/v4/
Rate limit: 5,000 requests/day (no auth required for basic queries).

Stores results in the litigation_risk table (migration 116).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any, Dict, List

from api import db

logger = logging.getLogger(__name__)

_CL_BASE = "https://www.courtlistener.com/api/rest/v4"

# Keywords for case type classification (checked against case name + nature of suit)
_SECURITIES_KW = {"securities", "sec ", "fraud", "insider trading", "10b-5", "exchange act", "class action"}
_EMPLOYMENT_KW = {"employment", "discrimination", "labor", "erisa", "wage", "wrongful termination", "retaliation"}
_ANTITRUST_KW = {"antitrust", "monopoly", "competition", "sherman", "ftc", "price fixing"}
_IP_KW = {"patent", "trademark", "copyright", "trade secret", "intellectual property", "infringement"}

# Securities fraud cases are weighted 3× in the risk score (most material to investors)
_CASE_WEIGHTS = {"securities": 3, "employment": 1, "antitrust": 2, "ip": 1, "other": 1}


def _classify_case(case_name: str, description: str) -> str:
    combined = (case_name + " " + description).lower()
    if any(kw in combined for kw in _SECURITIES_KW):
        return "securities"
    if any(kw in combined for kw in _EMPLOYMENT_KW):
        return "employment"
    if any(kw in combined for kw in _ANTITRUST_KW):
        return "antitrust"
    if any(kw in combined for kw in _IP_KW):
        return "ip"
    return "other"


def fetch_litigation_risk(
    ticker: str,
    company_name: str,
    lookback_days: int = 365,
) -> Dict[str, Any]:
    """Fetch active federal litigation for a company from CourtListener.

    Returns case counts by type and a total_litigation_score (0-100, higher = more risk).
    Securities fraud cases are weighted 3× due to their direct investor impact.
    """
    today = dt.date.today()
    filed_after = (today - dt.timedelta(days=lookback_days)).isoformat()

    empty: Dict[str, Any] = {
        "ticker": ticker,
        "active_cases_1yr": 0,
        "securities_fraud_cases": 0,
        "employment_cases": 0,
        "antitrust_cases": 0,
        "ip_cases": 0,
        "other_cases": 0,
        "total_litigation_score": 0.0,
        "source": "CourtListener",
        "as_of_date": today.isoformat(),
    }

    try:
        import urllib.parse
        import urllib.request

        params = {
            "q": f'"{company_name}"',
            "filed_after": filed_after,
            "order_by": "-date_filed",
            "format": "json",
        }
        url = f"{_CL_BASE}/dockets/?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "FTIP-Research/1.0 (research platform; contact: ops@ftip.ai)",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())

        counts: Dict[str, int] = {"securities": 0, "employment": 0, "antitrust": 0, "ip": 0, "other": 0}
        for docket in data.get("results", []):
            case_name = docket.get("case_name") or ""
            description = (
                (docket.get("docket_text") or "")
                + " "
                + (docket.get("nature_of_suit_str") or "")
            )
            case_type = _classify_case(case_name, description)
            counts[case_type] += 1

        total = sum(counts.values())
        weighted = sum(counts[t] * _CASE_WEIGHTS[t] for t in counts)
        score = min(100.0, weighted * 10.0)

        result: Dict[str, Any] = {
            "ticker": ticker,
            "active_cases_1yr": total,
            "securities_fraud_cases": counts["securities"],
            "employment_cases": counts["employment"],
            "antitrust_cases": counts["antitrust"],
            "ip_cases": counts["ip"],
            "other_cases": counts["other"],
            "total_litigation_score": round(score, 2),
            "source": "CourtListener",
            "as_of_date": today.isoformat(),
        }
        logger.debug(
            "court_listener.fetched ticker=%s cases=%d score=%.1f",
            ticker, total, score,
        )
        return result

    except Exception as exc:
        logger.debug(
            "court_listener.fetch_failed ticker=%s company=%s err=%s",
            ticker, company_name, exc,
        )
        return empty


def fetch_bulk_litigation_risk(
    tickers: List[str],
    *,
    delay_seconds: float = 1.0,
) -> List[Dict[str, Any]]:
    """Fetch litigation risk for a list of tickers.

    Respects CourtListener's rate limit via delay_seconds between requests.
    """
    import time
    from api.scrapers.entity_resolver import get_company_name

    results = []
    for ticker in tickers:
        company_name = get_company_name(ticker)
        rec = fetch_litigation_risk(ticker, company_name)
        results.append(rec)
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    return results


def store_litigation_risk(records: List[Dict[str, Any]]) -> int:
    """Persist litigation risk records to the litigation_risk table."""
    from api import db

    if not db.db_write_enabled():
        return 0

    written = 0
    for rec in records:
        try:
            db.safe_execute(
                """
                INSERT INTO litigation_risk (
                    symbol, as_of_date, active_cases_1yr, securities_fraud_cases,
                    employment_cases, antitrust_cases, ip_cases, other_cases,
                    total_litigation_score, source, raw
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (symbol, as_of_date) DO UPDATE SET
                    active_cases_1yr       = EXCLUDED.active_cases_1yr,
                    securities_fraud_cases = EXCLUDED.securities_fraud_cases,
                    employment_cases       = EXCLUDED.employment_cases,
                    antitrust_cases        = EXCLUDED.antitrust_cases,
                    ip_cases               = EXCLUDED.ip_cases,
                    other_cases            = EXCLUDED.other_cases,
                    total_litigation_score = EXCLUDED.total_litigation_score,
                    source                 = EXCLUDED.source,
                    raw                    = EXCLUDED.raw,
                    updated_at             = now()
                """,
                (
                    rec["ticker"],
                    rec["as_of_date"],
                    rec["active_cases_1yr"],
                    rec["securities_fraud_cases"],
                    rec["employment_cases"],
                    rec.get("antitrust_cases", 0),
                    rec.get("ip_cases", 0),
                    rec.get("other_cases", 0),
                    rec["total_litigation_score"],
                    rec["source"],
                    json.dumps(rec),
                ),
            )
            written += 1
        except Exception as exc:
            logger.debug("litigation_risk.store_failed ticker=%s err=%s", rec.get("ticker"), exc)

    logger.info("litigation_risk.stored written=%d total=%d", written, len(records))
    return written
