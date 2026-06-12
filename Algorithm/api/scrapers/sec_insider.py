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

_EFTS_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22&forms=4&dateRange=custom&startdt={start}&enddt={end}&hits.hits.total.value=0&_source=accession_no,entity_id,period_of_report,file_date"
_EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dashes}/{primary_doc}"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_SEC_HEADERS = {
    "User-Agent": "AXIOM Financial Intelligence axiom@axiom.ai",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
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


def _search_efts(symbol: str, start: str, end: str) -> List[Dict[str, Any]]:
    """Query EFTS for Form 4 filings containing the ticker symbol."""
    try:
        import httpx
        url = (
            f"https://efts.sec.gov/LATEST/search-index"
            f"?q=%22{symbol.upper()}%22"
            f"&forms=4"
            f"&dateRange=custom&startdt={start}&enddt={end}"
        )
        resp = httpx.get(url, headers=_SEC_HEADERS, timeout=20, follow_redirects=True)
        if resp.status_code != 200:
            logger.debug("sec_insider.efts_failed status=%d symbol=%s", resp.status_code, symbol)
            return []
        data = resp.json()
        hits = (data.get("hits") or {}).get("hits") or []
        results = []
        for hit in hits:
            src = hit.get("_source") or {}
            acc = src.get("accession_no") or ""
            cik = str(src.get("entity_id") or "").zfill(10)
            period = src.get("period_of_report") or src.get("file_date") or ""
            if acc and cik:
                results.append({"accession_no": acc, "cik": cik, "period": period})
        return results
    except Exception as exc:
        logger.debug("sec_insider.efts_error symbol=%s err=%s", symbol, exc)
        return []


def _get_primary_doc(cik: str, acc_raw: str) -> str:
    """Get the primary document filename for a filing from EDGAR submissions."""
    try:
        acc_no_dashes = acc_raw.replace("-", "")
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_no_dashes}/{acc_no_dashes}-index.json"
        data = _get(index_url, timeout=10)
        if data:
            for item in (data.get("directory", {}).get("item") or []):
                name = item.get("name", "")
                if name.endswith(".xml") and not name.startswith("R"):
                    return name
    except Exception:
        pass
    # Fallback: guess the primary doc name from accession
    acc_no_dashes = acc_raw.replace("-", "")
    return f"{acc_no_dashes}.xml"


def fetch_insider_transactions(symbol: str, days_back: int = 90) -> List[Dict[str, Any]]:
    """Return recent Form 4 transactions for *symbol* from SEC EDGAR via EFTS search.

    Each transaction dict has:
      transaction_date, shares, price_per_share, transaction_code,
      is_open_market_buy, is_open_market_sell, reporter_name,
      reporter_title, accession_number
    """
    today = date.today()
    start = (today - timedelta(days=days_back)).isoformat()
    end = today.isoformat()

    filings = _search_efts(symbol, start, end)
    if not filings:
        logger.debug("sec_insider.no_filings_found symbol=%s days_back=%d", symbol, days_back)
        return []

    transactions: List[Dict[str, Any]] = []

    for filing in filings[:30]:  # cap at 30 filings to avoid rate-limit pileup
        acc_raw = filing["accession_no"]
        cik = filing["cik"]
        acc_no_dashes = acc_raw.replace("-", "")
        primary_doc = _get_primary_doc(cik, acc_raw)

        xml_data = _parse_form4_xml(cik, acc_no_dashes, primary_doc)
        if xml_data:
            for txn in xml_data:
                txn["accession_number"] = acc_raw
                transactions.append(txn)

        time.sleep(0.1)  # be gentle with EDGAR rate limits

    logger.debug("sec_insider.fetched symbol=%s txns=%d", symbol, len(transactions))
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


def fetch_insider_transactions_bulk(
    symbols: List[str],
    days_back: int = 90,
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch insider transactions for all symbols in parallel.

    Instead of sequential per-symbol fetching, runs up to 10 concurrent
    EDGAR queries with a 0.1s delay per worker (~10 req/s total).

    Returns: {symbol: [transaction, ...]}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: Dict[str, List] = {}

    def _fetch_one(sym: str) -> tuple:
        time.sleep(0.1)
        try:
            txns = fetch_insider_transactions(sym, days_back=days_back)
            return sym, txns
        except Exception:
            return sym, []

    n_workers = min(10, len(symbols))
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_fetch_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym, txns = future.result()
            results[sym] = txns

    logger.info(
        "insider_bulk_fetch symbols=%d symbols_with_data=%d",
        len(symbols),
        sum(1 for v in results.values() if v),
    )
    return results


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
