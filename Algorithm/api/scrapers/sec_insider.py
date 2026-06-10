"""SEC EDGAR Form 4 insider transaction scraper.

Fetches recent Form 4 filings via SEC EDGAR APIs and computes an
Insider Confidence Score (ICS) 0-100.

ICS weights:
  net_buy_ratio    × 0.40   (ratio of buy volume to total traded volume)
  open_market_prem × 0.30   (premium for open-market vs option exercises)
  recency          × 0.20   (bias toward transactions in the last 30 days)
  officer_rank     × 0.10   (bias toward named executives vs directors)
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dashes}/{primary_doc}"
_SEC_HEADERS = {
    "User-Agent": "FTIP-System contact@ftip.ai",
    "Accept-Encoding": "gzip, deflate",
}

# Transaction codes: P = open-market purchase, S = open-market sale, M = exercise
_OPEN_MARKET_BUY = {"P"}
_OPEN_MARKET_SELL = {"S"}
_NON_OPEN_MARKET = {"M", "A", "D", "F", "G", "J", "K", "L", "U", "W", "X", "Z"}

_OFFICER_KEYWORDS = {"ceo", "cfo", "coo", "cto", "president", "chief", "founder"}


def _get(url: str, timeout: int = 15) -> Optional[Any]:
    try:
        import httpx
        resp = httpx.get(url, headers=_SEC_HEADERS, timeout=timeout, follow_redirects=True)
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logger.debug("sec_insider.get_failed url=%s error=%s", url, exc)
    return None


def _lookup_cik(symbol: str) -> Optional[str]:
    data = _get(_TICKERS_URL, timeout=20)
    if not data:
        return None
    sym_upper = symbol.upper()
    for entry in data.values():
        if isinstance(entry, dict) and entry.get("ticker", "").upper() == sym_upper:
            cik = str(entry["cik_str"]).zfill(10)
            return cik
    return None


def fetch_insider_transactions(symbol: str, days_back: int = 90) -> List[Dict[str, Any]]:
    """Return recent Form 4 transactions for *symbol* from SEC EDGAR.

    Each transaction dict has:
      transaction_date, shares, price_per_share, transaction_code,
      is_open_market_buy, is_open_market_sell, reporter_name,
      reporter_title, accession_number
    """
    cik = _lookup_cik(symbol)
    if not cik:
        logger.debug("sec_insider.cik_not_found symbol=%s", symbol)
        return []

    submissions = _get(_SUBMISSIONS_URL.format(cik=cik), timeout=20)
    if not submissions:
        return []

    cutoff = date.today() - timedelta(days=days_back)
    filings = (submissions.get("filings") or {}).get("recent") or {}
    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])

    transactions: List[Dict[str, Any]] = []

    for i, form in enumerate(forms):
        if form != "4":
            continue
        try:
            filing_date = date.fromisoformat(dates[i])
        except Exception:
            continue
        if filing_date < cutoff:
            continue

        acc_raw = accessions[i]
        acc_no_dashes = acc_raw.replace("-", "")
        primary_doc = primary_docs[i] if i < len(primary_docs) else ""

        xml_data = _parse_form4_xml(cik, acc_no_dashes, primary_doc)
        if xml_data:
            for txn in xml_data:
                txn["accession_number"] = acc_raw
                transactions.append(txn)

        time.sleep(0.1)  # be gentle with EDGAR rate limits

    return transactions


def _parse_form4_xml(cik: str, acc_no_dashes: str, primary_doc: str) -> List[Dict[str, Any]]:
    """Fetch and parse Form 4 XML, extracting non-derivative transactions."""
    url = _EDGAR_ARCHIVE.format(cik=cik, acc_no_dashes=acc_no_dashes, primary_doc=primary_doc)
    try:
        import httpx
        resp = httpx.get(url, headers=_SEC_HEADERS, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            return []
        text = resp.text
    except Exception:
        return []

    # Reporter info
    reporter_name = _xml_value(text, "rptOwnerName") or ""
    reporter_title = _xml_value(text, "officerTitle") or _xml_value(text, "relationship") or ""

    transactions = []
    # Extract non-derivative transactions (direct open-market buys/sells)
    for block in re.findall(r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>", text, re.S):
        txn_code = _xml_value(block, "transactionCode") or ""
        shares_str = _xml_value(block, "transactionShares") or "0"
        price_str = _xml_value(block, "transactionPricePerShare") or "0"
        txn_date_str = _xml_value(block, "transactionDate") or ""

        try:
            shares = float(shares_str)
            price = float(price_str)
            txn_date = date.fromisoformat(txn_date_str[:10]) if txn_date_str else date.today()
        except Exception:
            continue

        transactions.append({
            "transaction_date": txn_date.isoformat(),
            "shares": shares,
            "price_per_share": price,
            "transaction_code": txn_code,
            "is_open_market_buy": txn_code in _OPEN_MARKET_BUY,
            "is_open_market_sell": txn_code in _OPEN_MARKET_SELL,
            "reporter_name": reporter_name,
            "reporter_title": reporter_title,
        })

    return transactions


def _xml_value(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.S)
    return m.group(1).strip() if m else None


def compute_insider_score(symbol: str, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Insider Confidence Score (ICS) 0-100 from Form 4 transactions.

    Returns:
      symbol, ics_score, net_signal ("BUY"|"SELL"|"NEUTRAL"),
      buy_count, sell_count, total_buy_shares, total_sell_shares,
      open_market_buy_count, officer_buy_count, recent_transactions
    """
    if not transactions:
        return _default_insider_result(symbol)

    today = date.today()
    buy_shares = 0.0
    sell_shares = 0.0
    open_mkt_buy_shares = 0.0
    total_buy_count = 0
    total_sell_count = 0
    open_mkt_buy_count = 0
    officer_buy_count = 0
    recent_buys = 0  # within 30 days

    for txn in transactions:
        code = txn.get("transaction_code", "")
        shares = float(txn.get("shares") or 0)

        if txn.get("is_open_market_buy") or code in _OPEN_MARKET_BUY:
            buy_shares += shares
            total_buy_count += 1
            open_mkt_buy_shares += shares
            open_mkt_buy_count += 1
            title = (txn.get("reporter_title") or "").lower()
            if any(kw in title for kw in _OFFICER_KEYWORDS):
                officer_buy_count += 1
            try:
                txn_date = date.fromisoformat(str(txn.get("transaction_date", ""))[:10])
                if (today - txn_date).days <= 30:
                    recent_buys += 1
            except Exception:
                pass
        elif txn.get("is_open_market_sell") or code in _OPEN_MARKET_SELL:
            sell_shares += shares
            total_sell_count += 1
        # Non-open-market (exercises, gifts, etc.) excluded from ratio

    total_traded = buy_shares + sell_shares
    net_buy_ratio = (buy_shares / total_traded) if total_traded > 0 else 0.5
    open_market_prem = (open_mkt_buy_shares / buy_shares) if buy_shares > 0 else 0.5
    recency_score = min(recent_buys / max(total_buy_count, 1), 1.0)
    officer_rank = (officer_buy_count / max(total_buy_count, 1)) if total_buy_count > 0 else 0.5

    raw = (
        net_buy_ratio    * 0.40
        + open_market_prem * 0.30
        + recency_score    * 0.20
        + officer_rank     * 0.10
    )
    ics = round(min(max(raw * 100.0, 0.0), 100.0), 2)

    if ics >= 60:
        net_signal = "BUY"
    elif ics <= 40:
        net_signal = "SELL"
    else:
        net_signal = "NEUTRAL"

    return {
        "symbol": symbol,
        "ics_score": ics,
        "net_signal": net_signal,
        "buy_count": total_buy_count,
        "sell_count": total_sell_count,
        "total_buy_shares": round(buy_shares, 0),
        "total_sell_shares": round(sell_shares, 0),
        "open_market_buy_count": open_mkt_buy_count,
        "officer_buy_count": officer_buy_count,
        "recent_transactions": transactions[:10],
    }


def _default_insider_result(symbol: str) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "ics_score": 50.0,
        "net_signal": "NEUTRAL",
        "buy_count": 0,
        "sell_count": 0,
        "total_buy_shares": 0.0,
        "total_sell_shares": 0.0,
        "open_market_buy_count": 0,
        "officer_buy_count": 0,
        "recent_transactions": [],
    }
