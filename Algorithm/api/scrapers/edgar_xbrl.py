"""EDGAR XBRL Harvester — fetches company fundamentals from SEC EDGAR XBRL facts API.

Rate limit: 10 requests/second. No authentication required.
API: https://data.sec.gov/api/xbrl/companyfacts/CIK{N:010d}.json
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_EDGAR_TICKERS_URL = "https://data.sec.gov/files/company_tickers.json"
_HEADERS = {
    "User-Agent": "AXIOM Research contact@axiom.ai",
    "Accept": "application/json",
}
_RATE_LIMIT_SLEEP = 0.12  # 100ms + buffer to stay under 10 req/s

XBRL_CONCEPTS: Dict[str, List[str]] = {
    "revenue": [
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityAttributableToParent",
    ],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssued",
    ],
    "research_and_development": ["ResearchAndDevelopmentExpense"],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
    ],
}

# In-process CIK lookup cache: {symbol_upper: cik_str_zero_padded}
_cik_cache: Dict[str, Optional[str]] = {}
# Cached ticker map from EDGAR
_ticker_map: Optional[Dict[str, str]] = None


def _load_ticker_map() -> Dict[str, str]:
    """Fetch SEC EDGAR company_tickers.json and return {ticker_upper: cik_padded}."""
    global _ticker_map
    if _ticker_map is not None:
        return _ticker_map
    try:
        import httpx
        resp = httpx.get(_EDGAR_TICKERS_URL, headers=_HEADERS, timeout=15)
        if resp.status_code != 200:
            _ticker_map = {}
            return {}
        data = resp.json()
        mapping: Dict[str, str] = {}
        for entry in data.values():
            ticker = str(entry.get("ticker") or "").strip().upper()
            cik = entry.get("cik_str")
            if ticker and cik:
                mapping[ticker] = f"{int(cik):010d}"
        _ticker_map = mapping
        logger.info("edgar_ticker_map_loaded tickers=%d", len(mapping))
        return mapping
    except Exception as exc:
        logger.debug("edgar_ticker_map_error err=%s", exc)
        _ticker_map = {}
        return {}


def _get_cik_for_symbol(symbol: str) -> Optional[str]:
    """Return zero-padded 10-digit CIK for a symbol, or None if not found."""
    sym = symbol.upper()
    if sym in _cik_cache:
        return _cik_cache[sym]
    mapping = _load_ticker_map()
    cik = mapping.get(sym)
    _cik_cache[sym] = cik
    return cik


def _fetch_company_facts(cik: str) -> Optional[Dict[str, Any]]:
    """Fetch raw XBRL facts JSON for a CIK. Returns None on failure."""
    try:
        import httpx
        url = _EDGAR_FACTS_URL.format(cik=cik)
        resp = httpx.get(url, headers=_HEADERS, timeout=30)
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            logger.debug("edgar_facts_failed cik=%s status=%d", cik, resp.status_code)
            return None
        return resp.json()
    except Exception as exc:
        logger.debug("edgar_facts_error cik=%s err=%s", cik, exc)
        return None


def _extract_quarterly_values(
    facts: Dict[str, Any],
    concept: str,
    n_quarters: int = 8,
) -> List[Dict[str, Any]]:
    """Extract the most recent N quarterly (10-Q/10-K) values for a GAAP concept.

    Returns list of {period, value, form} sorted newest-first.
    """
    us_gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    concept_data = us_gaap.get(concept) or {}
    units = concept_data.get("units") or {}

    # Prefer USD, fall back to shares or pure numbers
    values_list = units.get("USD") or units.get("shares") or units.get("pure") or []

    quarterly: List[Dict[str, Any]] = []
    seen_periods: set = set()
    for entry in values_list:
        form = entry.get("form", "")
        if form not in ("10-Q", "10-K"):
            continue
        end = entry.get("end") or ""
        if not end or end in seen_periods:
            continue
        # Only include entries with a fiscal period label or a valid end date
        val = entry.get("val")
        if val is None:
            continue
        seen_periods.add(end)
        quarterly.append({"period": end, "value": val, "form": form})

    # Sort newest first
    quarterly.sort(key=lambda x: x["period"], reverse=True)
    return quarterly[:n_quarters]


def fetch_xbrl_fundamentals(symbol: str, n_quarters: int = 8) -> Optional[Dict[str, Any]]:
    """Fetch XBRL fundamentals for a symbol.

    Returns dict with:
      - cik, symbol
      - Per-concept latest values (revenue, gross_profit, operating_income, etc.)
      - Derived ratios: gross_margin, op_margin, fcf_margin
      - quarterly_data: {period: {concept: value, ...}}
    """
    cik = _get_cik_for_symbol(symbol)
    if not cik:
        logger.debug("xbrl_no_cik symbol=%s", symbol)
        return None

    facts = _fetch_company_facts(cik)
    if not facts:
        return None

    # Extract quarterly data for each concept
    quarterly_by_concept: Dict[str, List[Dict[str, Any]]] = {}
    for metric, concepts in XBRL_CONCEPTS.items():
        for concept in concepts:
            rows = _extract_quarterly_values(facts, concept, n_quarters)
            if rows:
                quarterly_by_concept[metric] = rows
                break  # Use first matching concept

    if not quarterly_by_concept:
        return None

    # Build period-indexed quarterly snapshot
    periods: Dict[str, Dict[str, Any]] = {}
    for metric, rows in quarterly_by_concept.items():
        for row in rows:
            period = row["period"]
            periods.setdefault(period, {})[metric] = row["value"]

    # Latest-value convenience fields
    def _latest(metric: str) -> Optional[float]:
        rows = quarterly_by_concept.get(metric)
        if rows:
            val = rows[0]["value"]
            return float(val) if val is not None else None
        return None

    rev = _latest("revenue")
    gross = _latest("gross_profit")
    op_inc = _latest("operating_income")
    ocf = _latest("operating_cash_flow")
    capex_val = _latest("capex")
    fcf = (ocf - abs(capex_val)) if (ocf is not None and capex_val is not None) else None

    result: Dict[str, Any] = {
        "symbol": symbol,
        "cik": cik,
        "revenue": rev,
        "gross_profit": gross,
        "operating_income": op_inc,
        "net_income": _latest("net_income"),
        "total_assets": _latest("total_assets"),
        "total_liabilities": _latest("total_liabilities"),
        "stockholders_equity": _latest("stockholders_equity"),
        "operating_cash_flow": ocf,
        "capex": capex_val,
        "free_cash_flow": fcf,
        "cash_and_equivalents": _latest("cash_and_equivalents"),
        "long_term_debt": _latest("long_term_debt"),
        "eps_diluted": _latest("eps_diluted"),
        "shares_outstanding": _latest("shares_outstanding"),
        "research_and_development": _latest("research_and_development"),
        "gross_margin": round(gross / rev, 4) if (gross is not None and rev and rev > 0) else None,
        "op_margin": round(op_inc / rev, 4) if (op_inc is not None and rev and rev > 0) else None,
        "fcf_margin": round(fcf / rev, 4) if (fcf is not None and rev and rev > 0) else None,
        "quarterly_data": periods,
        "fetched_at": dt.datetime.utcnow().isoformat(),
    }
    logger.info("xbrl_fetched symbol=%s periods=%d", symbol, len(periods))
    return result


def fetch_xbrl_fundamentals_bulk(
    symbols: List[str],
    n_quarters: int = 4,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Fetch XBRL fundamentals for multiple symbols sequentially with rate limiting.

    Returns {symbol: fundamentals_dict_or_None}.
    """
    # Pre-load ticker map once (avoids repeated fetches)
    _load_ticker_map()

    results: Dict[str, Optional[Dict[str, Any]]] = {}
    for i, symbol in enumerate(symbols):
        try:
            data = fetch_xbrl_fundamentals(symbol, n_quarters=n_quarters)
            results[symbol] = data
        except Exception as exc:
            logger.debug("xbrl_bulk_error symbol=%s err=%s", symbol, exc)
            results[symbol] = None
        if i < len(symbols) - 1:
            time.sleep(_RATE_LIMIT_SLEEP)

    ok = sum(1 for v in results.values() if v is not None)
    logger.info("xbrl_bulk_complete symbols=%d ok=%d", len(symbols), ok)
    return results
