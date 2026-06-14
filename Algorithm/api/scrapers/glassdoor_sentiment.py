"""Employee sentiment scraper.

Attempts to retrieve employee sentiment scores for public companies.

Data sources (in priority order):
1. Google Custom Search API — requires GOOGLE_API_KEY + GOOGLE_CX env vars.
   Searches for Glassdoor ratings and parses rating from search snippets.
2. Comparably.com — public company ratings page (no auth required).
3. Neutral fallback — returns 50.0 for all scores when no data available.

Stores results in the employee_sentiment table (migration 114).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)


def _fetch_google_cse(company_name: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """Use Google Custom Search API to find a Glassdoor rating snippet."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    cx = os.environ.get("GOOGLE_CX", "")
    if not api_key or not cx:
        return None
    try:
        import urllib.parse
        import urllib.request

        params = {
            "key": api_key,
            "cx": cx,
            "q": f"{company_name} glassdoor rating site:glassdoor.com",
            "num": "3",
        }
        url = "https://www.googleapis.com/customsearch/v1?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())

        for item in data.get("items", []):
            snippet = (item.get("snippet") or "") + " " + (item.get("title") or "")
            # Glassdoor shows "X.X/5" rating
            m = re.search(r"(\d\.\d)\s*/\s*5", snippet)
            if m:
                raw = float(m.group(1))
                overall = round(min(100.0, (raw / 5.0) * 100.0), 1)
                return {
                    "overall_rating": overall,
                    "ceo_approval": overall,
                    "culture_score": overall,
                    "source": "google_cse_glassdoor",
                }
    except Exception as exc:
        logger.debug("glassdoor_sentiment.google_cse_failed company=%s err=%s", company_name, exc)
    return None


def _fetch_comparably(company_name: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """Scrape Comparably.com public rating for a company."""
    try:
        import urllib.parse
        import urllib.request

        slug = re.sub(r"[^\w\s-]", "", company_name.lower())
        slug = re.sub(r"[\s_]+", "-", slug.strip())
        slug = slug.strip("-")
        url = f"https://www.comparably.com/companies/{slug}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.5",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Look for JSON-LD rating or meta rating value
        overall_m = re.search(r'"ratingValue"\s*:\s*"?([\d.]+)"?', html)
        if overall_m:
            raw = float(overall_m.group(1))
            # Comparably often uses a 0-100 scale directly or 0-5
            overall = min(100.0, (raw / 5.0) * 100.0) if raw <= 5.0 else min(100.0, raw)
            culture_m = re.search(r'culture["\s:]+(\d+(?:\.\d+)?)', html, re.IGNORECASE)
            ceo_m = re.search(r'ceo[^}]{0,30}?(\d{1,3})%', html, re.IGNORECASE)
            return {
                "overall_rating": round(overall, 1),
                "ceo_approval": round(float(ceo_m.group(1)), 1) if ceo_m else round(overall, 1),
                "culture_score": round(float(culture_m.group(1)), 1) if culture_m else round(overall, 1),
                "source": "comparably",
            }
    except Exception as exc:
        logger.debug("glassdoor_sentiment.comparably_failed company=%s err=%s", company_name, exc)
    return None


def fetch_employee_sentiment(ticker: str, company_name: str) -> Dict[str, Any]:
    """Fetch employee sentiment data for a company.

    Returns a dict with overall_rating, ceo_approval, culture_score (0-100 scale).
    Falls back to neutral 50.0 if no external data is available.
    """
    today = dt.date.today().isoformat()
    neutral = {
        "ticker": ticker,
        "overall_rating": 50.0,
        "ceo_approval": 50.0,
        "culture_score": 50.0,
        "source": "neutral_fallback",
        "as_of_date": today,
    }

    # Google CSE first (highest quality)
    result = _fetch_google_cse(company_name)
    if result is None:
        result = _fetch_comparably(company_name)

    if result:
        logger.debug(
            "glassdoor_sentiment.fetched ticker=%s source=%s rating=%.1f",
            ticker, result["source"], result["overall_rating"],
        )
        return {
            "ticker": ticker,
            "overall_rating": result["overall_rating"],
            "ceo_approval": result["ceo_approval"],
            "culture_score": result["culture_score"],
            "source": result["source"],
            "as_of_date": today,
        }

    return neutral


def fetch_bulk_employee_sentiment(
    tickers: List[str],
    *,
    delay_seconds: float = 1.0,
) -> List[Dict[str, Any]]:
    """Fetch employee sentiment for a list of tickers.

    Uses KNOWN_NAMES from entity_resolver for ticker→company_name lookup.
    Adds delay_seconds between requests to avoid rate limiting.
    """
    import time
    from api.scrapers.entity_resolver import get_company_name

    results = []
    for ticker in tickers:
        company_name = get_company_name(ticker)
        rec = fetch_employee_sentiment(ticker, company_name)
        results.append(rec)
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    return results


def store_employee_sentiment(records: List[Dict[str, Any]]) -> int:
    """Persist employee sentiment records to the employee_sentiment table."""
    from api import db

    if not db.db_write_enabled():
        return 0

    written = 0
    for rec in records:
        try:
            db.safe_execute(
                """
                INSERT INTO employee_sentiment (
                    symbol, as_of_date, overall_rating, ceo_approval, culture_score, source, raw
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (symbol, as_of_date) DO UPDATE SET
                    overall_rating = EXCLUDED.overall_rating,
                    ceo_approval   = EXCLUDED.ceo_approval,
                    culture_score  = EXCLUDED.culture_score,
                    source         = EXCLUDED.source,
                    raw            = EXCLUDED.raw,
                    updated_at     = now()
                """,
                (
                    rec["ticker"],
                    rec["as_of_date"],
                    rec["overall_rating"],
                    rec["ceo_approval"],
                    rec["culture_score"],
                    rec["source"],
                    json.dumps(rec),
                ),
            )
            written += 1
        except Exception as exc:
            logger.debug("employee_sentiment.store_failed ticker=%s err=%s", rec.get("ticker"), exc)

    logger.info("employee_sentiment.stored written=%d total=%d", written, len(records))
    return written
