from __future__ import annotations

import datetime as dt
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests

from api import config

from .errors import ProviderUnavailable, SymbolNoData
from .symbols import canonical_symbol

BASE_URL = "https://data.sec.gov"
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


def fetch_company_filing_profile(symbol: str) -> Dict[str, Any]:
    cik = resolve_cik(symbol)
    submissions = fetch_submissions(cik)
    filings = ((submissions.get("filings") or {}).get("recent") or {})
    accession_numbers = filings.get("accessionNumber") or []
    filing_dates = filings.get("filingDate") or []
    forms = filings.get("form") or []
    primary_documents = filings.get("primaryDocument") or []

    latest_filing_date = filing_dates[0] if filing_dates else None
    latest_form = forms[0] if forms else None
    filing_recency_days = None
    if latest_filing_date:
        try:
            filing_recency_days = (
                dt.datetime.now(dt.timezone.utc).date()
                - dt.date.fromisoformat(str(latest_filing_date))
            ).days
        except Exception:
            filing_recency_days = None

    companyfacts = fetch_companyfacts(cik)
    facts = (companyfacts.get("facts") or {}).get("us-gaap") or {}
    coverage_flags = {
        "revenue": any(key in facts for key in ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues")),
        "gross_profit": "GrossProfit" in facts,
        "operating_income": "OperatingIncomeLoss" in facts,
        "net_income": "NetIncomeLoss" in facts,
        "cash_from_operations": "NetCashProvidedByUsedInOperatingActivities" in facts,
        "capital_expenditure": "PaymentsToAcquirePropertyPlantAndEquipment" in facts,
        "assets": "Assets" in facts,
        "liabilities": "Liabilities" in facts,
    }

    recent_filings: List[Dict[str, Any]] = []
    for accession, filed_at, form, document in zip(
        accession_numbers[:8],
        filing_dates[:8],
        forms[:8],
        primary_documents[:8],
    ):
        recent_filings.append(
            {
                "accession_number": accession,
                "filing_date": filed_at,
                "form": form,
                "primary_document": document,
            }
        )

    return {
        "symbol": canonical_symbol(symbol),
        "cik": cik,
        "name": submissions.get("name"),
        "sic": submissions.get("sic"),
        "sic_description": submissions.get("sicDescription"),
        "fiscal_year_end": submissions.get("fiscalYearEnd"),
        "latest_filing_date": latest_filing_date,
        "latest_form": latest_form,
        "filing_recency_days": filing_recency_days,
        "coverage_flags": coverage_flags,
        "recent_filings": recent_filings,
        "source": "sec_edgar",
    }


@lru_cache(maxsize=1)
def _ticker_map() -> Dict[str, str]:
    response = requests.get(
        _TICKER_MAP_URL,
        headers=_headers(),
        timeout=config.data_fabric_timeout_seconds(),
    )
    if response.status_code != 200:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            f"SEC ticker map HTTP {response.status_code}",
        )
    payload = response.json()
    mapping: Dict[str, str] = {}
    if isinstance(payload, dict):
        for row in payload.values():
            ticker = str(row.get("ticker") or "").strip().upper()
            cik = str(row.get("cik_str") or "").strip()
            if ticker and cik:
                mapping[ticker] = cik.zfill(10)
    if not mapping:
        raise SymbolNoData("NO_DATA", "SEC ticker map was empty")
    return mapping


def resolve_cik(symbol: str) -> str:
    ticker = canonical_symbol(symbol).split(".")[0]
    cik = _ticker_map().get(ticker)
    if not cik:
        raise SymbolNoData("NO_DATA", f"SEC ticker map missing {ticker}")
    return cik


def fetch_submissions(cik: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_URL}/submissions/CIK{str(cik).zfill(10)}.json",
        headers=_headers(),
        timeout=config.data_fabric_timeout_seconds(),
    )
    if response.status_code != 200:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            f"SEC submissions HTTP {response.status_code}",
        )
    payload = response.json()
    if not isinstance(payload, dict) or not payload:
        raise SymbolNoData("NO_DATA", "SEC submissions payload was empty")
    return payload


def fetch_companyfacts(cik: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_URL}/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json",
        headers=_headers(),
        timeout=config.data_fabric_timeout_seconds(),
    )
    if response.status_code != 200:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            f"SEC companyfacts HTTP {response.status_code}",
        )
    payload = response.json()
    if not isinstance(payload, dict) or not payload:
        raise SymbolNoData("NO_DATA", "SEC companyfacts payload was empty")
    return payload


def _headers() -> Dict[str, str]:
    return {
        "User-Agent": config.sec_user_agent(),
        "Accept": "application/json",
        "Host": "data.sec.gov",
    }
