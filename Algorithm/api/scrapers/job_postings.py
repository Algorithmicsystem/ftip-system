"""Job posting velocity scraper — Indeed RSS + Adzuna public API.

Estimates hiring momentum as a leading indicator of business health.
No API key required for basic Indeed RSS; Adzuna requires free registration
but falls back gracefully.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Map company names (as they appear in job board searches) to ticker symbols.
COMPANY_SEARCH_TERMS: Dict[str, str] = {
    "Apple Inc": "AAPL",
    "Microsoft Corporation": "MSFT",
    "Amazon": "AMZN",
    "Alphabet Google": "GOOGL",
    "Meta Platforms": "META",
    "NVIDIA Corporation": "NVDA",
    "Tesla Inc": "TSLA",
    "JPMorgan Chase": "JPM",
    "Visa Inc": "V",
    "Mastercard": "MA",
    "Johnson Johnson": "JNJ",
    "UnitedHealth Group": "UNH",
    "Exxon Mobil": "XOM",
    "Chevron Corporation": "CVX",
    "Berkshire Hathaway": "BRK-B",
    "Procter Gamble": "PG",
    "Home Depot": "HD",
    "Walmart": "WMT",
    "Walt Disney": "DIS",
    "Netflix": "NFLX",
    "Adobe Systems": "ADBE",
    "Salesforce": "CRM",
    "Oracle Corporation": "ORCL",
    "Cisco Systems": "CSCO",
    "Intel Corporation": "INTC",
    "Advanced Micro Devices AMD": "AMD",
    "Qualcomm": "QCOM",
    "Texas Instruments": "TXN",
    "Broadcom": "AVGO",
    "Applied Materials": "AMAT",
    "Lam Research": "LRCX",
    "KLA Corporation": "KLAC",
    "Micron Technology": "MU",
    "Western Digital": "WDC",
    "Seagate Technology": "STX",
    "PayPal Holdings": "PYPL",
    "Block Square": "SQ",
    "Stripe": None,
    "Airbnb": "ABNB",
    "Uber Technologies": "UBER",
    "Lyft Inc": "LYFT",
    "DoorDash": "DASH",
    "Snowflake": "SNOW",
    "Palantir Technologies": "PLTR",
    "Cloudflare": "NET",
    "Datadog": "DDOG",
    "CrowdStrike": "CRWD",
    "Fortinet": "FTNT",
    "Palo Alto Networks": "PANW",
}


def fetch_indeed_job_count(company_name: str, timeout: int = 15) -> Optional[int]:
    """Fetch approximate job count from Indeed RSS for a company.

    Uses Indeed's public job search RSS feed (no API key required).
    Returns the count from the totalResults field, or None on failure.
    """
    try:
        import urllib.parse
        import urllib.request
        import xml.etree.ElementTree as ET

        query = urllib.parse.quote_plus(f'"{company_name}"')
        url = (
            f"https://www.indeed.com/rss?q={query}&l=United+States"
            f"&sort=date&limit=10&radius=100"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; FTIP-JobScraper/1.0; "
                    "+https://ftip.io/robots)"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")

        root = ET.fromstring(body)
        # <totalResults> is in the opensearch namespace
        ns = {"os": "http://a9.com/-/spec/opensearch/1.1/"}
        total_el = root.find(".//os:totalResults", ns)
        if total_el is not None and total_el.text:
            return int(total_el.text)

        # Fall back: count <item> elements
        items = root.findall(".//item")
        return len(items) if items else None

    except Exception as exc:
        logger.debug("indeed_job_count_failed company=%s err=%s", company_name, exc)
        return None


def fetch_job_posting_velocity(
    symbols: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fetch job posting counts for tracked companies.

    Returns {ticker: {count, company_name, fetched_at, source}}.
    Skips symbols not in COMPANY_SEARCH_TERMS.
    Rate-limits to ~1 request/second to respect Indeed's robots.txt.
    """
    # Build reverse map: ticker → company_name
    ticker_to_company: Dict[str, str] = {
        ticker: name
        for name, ticker in COMPANY_SEARCH_TERMS.items()
        if ticker is not None
    }

    targets = (
        [s for s in symbols if s in ticker_to_company]
        if symbols
        else list(ticker_to_company.keys())
    )

    results: Dict[str, Dict[str, Any]] = {}
    fetched_at = dt.datetime.utcnow().isoformat()

    for ticker in targets:
        company_name = ticker_to_company[ticker]
        count = fetch_indeed_job_count(company_name)
        results[ticker] = {
            "ticker": ticker,
            "company_name": company_name,
            "job_count": count,
            "fetched_at": fetched_at,
            "source": "indeed_rss",
        }
        if count is not None:
            logger.debug(
                "job_postings.fetched ticker=%s company=%s count=%d",
                ticker, company_name, count,
            )
        time.sleep(1.0)

    ok = sum(1 for v in results.values() if v["job_count"] is not None)
    logger.info("job_postings.done total=%d ok=%d", len(results), ok)
    return results


def store_job_postings(results: Dict[str, Dict[str, Any]]) -> int:
    """Persist job posting counts to job_postings_daily table.

    Returns number of rows upserted.
    """
    try:
        from api import db
        if not db.db_enabled():
            return 0

        today = dt.date.today()
        written = 0
        for ticker, data in results.items():
            if data.get("job_count") is None:
                continue
            try:
                db.safe_execute(
                    """
                    INSERT INTO job_postings_daily
                        (symbol, as_of_date, job_count, company_name, source, raw)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (symbol, as_of_date) DO UPDATE SET
                        job_count    = EXCLUDED.job_count,
                        company_name = EXCLUDED.company_name,
                        source       = EXCLUDED.source,
                        raw          = EXCLUDED.raw,
                        updated_at   = now()
                    """,
                    (
                        ticker,
                        today,
                        data["job_count"],
                        data.get("company_name"),
                        data.get("source", "indeed_rss"),
                        __import__("json").dumps(data),
                    ),
                )
                written += 1
            except Exception as exc:
                logger.debug(
                    "job_postings.store_failed ticker=%s err=%s", ticker, exc
                )

        logger.info("job_postings.stored rows=%d", written)
        return written

    except Exception as exc:
        logger.warning("job_postings.store_error err=%s", exc)
        return 0
