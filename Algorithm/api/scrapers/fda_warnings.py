"""FDA warning letter and enforcement action scraper.

Uses the FDA's public openFDA API (no key required, rate-limited to
~240 requests/minute) to fetch recent warning letters and enforcement
reports, then maps company names to public ticker symbols.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_FDA_ENFORCEMENT_URL = "https://api.fda.gov/drug/enforcement.json"
_FDA_DEVICE_URL = "https://api.fda.gov/device/enforcement.json"
_FDA_WARNINGS_URL = "https://api.fda.gov/drug/event.json"

# Map company name substrings (lowercase) → ticker symbols.
_FDA_COMPANY_TICKERS: Dict[str, Optional[str]] = {
    "pfizer": "PFE",
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "abbvie": "ABBV",
    "merck": "MRK",
    "bristol-myers": "BMY",
    "bristol myers": "BMY",
    "eli lilly": "LLY",
    "amgen": "AMGN",
    "gilead": "GILD",
    "biogen": "BIIB",
    "regeneron": "REGN",
    "moderna": "MRNA",
    "astrazeneca": None,
    "novartis": None,
    "roche": None,
    "sanofi": None,
    "bayer": None,
    "genentech": None,
    "medtronic": "MDT",
    "boston scientific": "BSX",
    "abbott": "ABT",
    "baxter": "BAX",
    "becton dickinson": "BDX",
    "stryker": "SYK",
    "zimmer biomet": "ZBH",
    "edwards lifesciences": "EW",
    "intuitive surgical": "ISRG",
    "cardinal health": "CAH",
    "mckesson": "MCK",
    "amerisourcebergen": "ABC",
    "cvs health": "CVS",
    "walgreens": "WBA",
}


def _match_fda_company_to_ticker(company_name: str) -> Optional[str]:
    """Match a company name from FDA records to a known ticker."""
    lower = company_name.lower()
    for keyword, ticker in _FDA_COMPANY_TICKERS.items():
        if keyword in lower:
            return ticker
    return None


def _fetch_fda_endpoint(
    url: str,
    search_query: str,
    limit: int = 100,
    timeout: int = 15,
) -> List[Dict[str, Any]]:
    """Generic openFDA fetch."""
    try:
        import urllib.parse
        import urllib.request
        import json

        params = {"search": search_query, "limit": str(limit)}
        full_url = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            full_url,
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        return data.get("results", [])
    except Exception as exc:
        logger.debug("fda_fetch_failed url=%s err=%s", url, exc)
        return []


def fetch_recent_fda_warnings(days_back: int = 30) -> List[Dict[str, Any]]:
    """Fetch recent FDA drug and device enforcement actions.

    Returns a list of enforcement records with keys:
      recall_number, product_description, recalling_firm, reason_for_recall,
      recall_initiation_date, classification, ticker, category
    """
    today = dt.date.today()
    from_date = (today - dt.timedelta(days=days_back)).strftime("%Y%m%d")
    to_date = today.strftime("%Y%m%d")

    date_search = f"recall_initiation_date:[{from_date}+TO+{to_date}]"

    warnings: List[Dict[str, Any]] = []

    for category, url in [("drug", _FDA_ENFORCEMENT_URL), ("device", _FDA_DEVICE_URL)]:
        records = _fetch_fda_endpoint(url, date_search, limit=100)
        for rec in records:
            firm = rec.get("recalling_firm", "") or ""
            ticker = _match_fda_company_to_ticker(firm)
            warnings.append({
                "recall_number": rec.get("recall_number"),
                "product_description": rec.get("product_description", "")[:500],
                "recalling_firm": firm,
                "reason_for_recall": (rec.get("reason_for_recall") or "")[:500],
                "recall_initiation_date": rec.get("recall_initiation_date"),
                "classification": rec.get("classification"),
                "voluntary_mandated": rec.get("voluntary_mandated"),
                "ticker": ticker,
                "category": category,
                "source": "openfda",
            })

    matched = sum(1 for w in warnings if w["ticker"])
    logger.info(
        "fda_warnings.fetched total=%d matched_tickers=%d",
        len(warnings), matched,
    )
    return warnings
