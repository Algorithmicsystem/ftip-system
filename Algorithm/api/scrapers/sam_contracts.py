"""SAM.gov federal contract award scraper.

Uses the public SAM.gov Opportunities API (no key required for basic access)
to find recent federal contract awards and map them to public company tickers.

Endpoint: https://api.sam.gov/opportunities/v2/search
  - Requires SAM_GOV_API_KEY env var for full access.
  - Falls back to public beta endpoint (rate-limited, subset of data).
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Map SAM.gov awardee name patterns → ticker symbols.
# Keys are lowercase substrings to match in awardee names.
_SAM_COMPANY_TICKERS: Dict[str, Optional[str]] = {
    "lockheed martin": "LMT",
    "boeing": "BA",
    "raytheon": "RTX",
    "general dynamics": "GD",
    "northrop grumman": "NOC",
    "l3harris": "LHX",
    "leidos": "LDOS",
    "booz allen": "BAH",
    "science applications": "SAIC",
    "caci international": "CACI",
    "mantech": "MANT",
    "palantir": "PLTR",
    "ibm": "IBM",
    "microsoft": "MSFT",
    "amazon web services": "AMZN",
    "amazon": "AMZN",
    "google": "GOOGL",
    "oracle": "ORCL",
    "accenture": "ACN",
    "deloitte": None,
    "kpmg": None,
    "mckesson": "MCK",
    "cardinal health": "CAH",
    "unitedhealth": "UNH",
    "humana": "HUM",
    "general electric": "GE",
    "honeywell": "HON",
    "bae systems": None,
    "textron": "TXT",
    "huntington ingalls": "HII",
}

_SAM_BASE_URL = "https://api.sam.gov/opportunities/v2/search"
_SAM_BETA_URL = "https://api.sam.gov/opportunities/v1/search"


def _match_company_to_ticker(award_text: str) -> Optional[str]:
    """Match an awardee name string to a known ticker."""
    lower = award_text.lower()
    for keyword, ticker in _SAM_COMPANY_TICKERS.items():
        if keyword in lower:
            return ticker
    return None


def fetch_recent_contract_awards(days_back: int = 7) -> List[Dict[str, Any]]:
    """Fetch recent federal contract awards from SAM.gov.

    Returns a list of award dicts with keys:
      notice_id, title, awardee, award_amount, award_date, ticker, naics
    """
    api_key = os.environ.get("SAM_GOV_API_KEY", "")
    try:
        import urllib.parse
        import urllib.request
        import json

        today = dt.date.today()
        from_date = (today - dt.timedelta(days=days_back)).strftime("%m/%d/%Y")
        to_date = today.strftime("%m/%d/%Y")

        params = {
            "api_key": api_key,
            "postedFrom": from_date,
            "postedTo": to_date,
            "ptype": "a",   # awards only
            "limit": "100",
            "offset": "0",
        }
        url = _SAM_BASE_URL + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())

        awards = []
        for opp in data.get("opportunitiesData", []):
            awardee = (
                (opp.get("award") or {}).get("awardee", {}).get("name", "")
                or opp.get("awardee", "")
                or ""
            )
            amount_raw = (opp.get("award") or {}).get("amount")
            try:
                amount = float(amount_raw) if amount_raw else None
            except (ValueError, TypeError):
                amount = None

            ticker = _match_company_to_ticker(awardee)
            awards.append({
                "notice_id": opp.get("noticeId") or opp.get("id"),
                "title": opp.get("title", ""),
                "awardee": awardee,
                "award_amount": amount,
                "award_date": opp.get("modifiedDate") or opp.get("postedDate"),
                "ticker": ticker,
                "naics": opp.get("naicsCode"),
                "source": "sam_gov",
            })

        logger.info(
            "sam_contracts.fetched total=%d matched=%d",
            len(awards),
            sum(1 for a in awards if a["ticker"]),
        )
        return awards

    except Exception as exc:
        logger.warning("sam_contracts.fetch_failed err=%s", exc)
        return []
