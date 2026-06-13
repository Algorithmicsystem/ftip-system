"""Congressional trading scraper — public data via STOCK Act disclosures.

Sources (tried in order):
  1. Capitol Trades — public web scraper, no auth required
  2. QuiverQuant free API — cloud-accessible, 24h delay
  3. SEC EDGAR full-text search — Form 4 filings by members of Congress
  4. Original S3 house/senate watchers — blocked on Railway but kept as last resort
  5. Neutral default (CIS=50) if all sources fail
"""
from __future__ import annotations

import datetime as dt
import json as _json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AXIOM/1.0; +https://axiom.ai)",
    "Accept": "application/json",
}

_QUIVER_URL = "https://api.quiverquant.com/beta/live/congresstrading"
_EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_HOUSE_S3 = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
_SENATE_S3 = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"
_CAPITOLTRADES_URL = "https://capitoltrades.com/trades"

# Maps committee keyword → sectors that get 1.5x weight multiplier
COMMITTEE_SECTOR_MAP: Dict[str, List[str]] = {
    "financial services": ["Financials"],
    "banking": ["Financials"],
    "energy": ["Energy"],
    "natural resources": ["Energy", "Materials"],
    "technology": ["Information Technology"],
    "commerce": ["Information Technology", "Consumer Discretionary"],
    "communications": ["Communication Services"],
    "intelligence": ["Information Technology"],
    "armed services": ["Industrials"],
    "defense": ["Industrials"],
    "homeland security": ["Industrials"],
    "health": ["Health Care"],
    "agriculture": ["Consumer Staples"],
    "transportation": ["Industrials"],
    "environment": ["Utilities", "Energy"],
    "housing": ["Real Estate"],
}

_committee_cache: Optional[Dict[str, List[str]]] = None


def _parse_amount_range(amount_str: str) -> tuple[Optional[int], Optional[int]]:
    """Parse '$1,001 - $15,000' style strings into (min, max) ints."""
    if not amount_str:
        return None, None
    try:
        clean = amount_str.replace("$", "").replace(",", "").strip()
        if " - " in clean:
            parts = clean.split(" - ")
            return int(float(parts[0])), int(float(parts[1]))
        if "+" in clean:
            return int(float(clean.replace("+", ""))), None
        val = int(float(clean))
        return val, val
    except Exception:
        return None, None


def _fetch_quiverquant(cutoff: dt.date) -> List[Dict[str, Any]]:
    """Fetch from QuiverQuant free congressional trading API."""
    try:
        import httpx
        from api import config
        api_key = config.env("QUIVERQUANT_API_KEY", "")
        headers = dict(_HEADERS)
        if api_key:
            headers["Authorization"] = f"Token {api_key}"

        resp = httpx.get(_QUIVER_URL, headers=headers, timeout=20, follow_redirects=True)
        if resp.status_code == 401:
            logger.info("congress_quiverquant_auth_required — no API key, trying SEC EDGAR")
            return []
        if resp.status_code != 200:
            logger.info("congress_quiverquant_unavailable status=%d", resp.status_code)
            return []

        trades = []
        for item in resp.json():
            try:
                date_str = item.get("Date") or item.get("transaction_date") or ""
                if not date_str:
                    continue
                tx_date = dt.date.fromisoformat(date_str[:10])
                if tx_date < cutoff:
                    continue
                ticker = (item.get("Ticker") or item.get("ticker") or "").strip().upper()
                if not ticker or ticker in ("N/A", "--", ""):
                    continue
                tx_type = (item.get("Transaction") or item.get("type") or "").lower()
                if "purchase" in tx_type or "buy" in tx_type:
                    direction = "buy"
                elif "sale" in tx_type or "sell" in tx_type:
                    direction = "sell"
                else:
                    continue
                amount_str = str(item.get("Range") or item.get("amount") or "")
                amin, amax = _parse_amount_range(amount_str)
                trades.append({
                    "symbol": ticker,
                    "transaction_date": tx_date.isoformat(),
                    "politician": item.get("Representative") or item.get("politician") or "",
                    "party": item.get("Party") or item.get("party") or "",
                    "chamber": (item.get("Chamber") or "house").lower(),
                    "transaction_type": direction,
                    "amount_range": amount_str,
                    "amount_min": amin,
                    "amount_max": amax,
                    "disclosure_date": item.get("ReportDate") or item.get("disclosure_date") or "",
                    "source": "quiverquant",
                })
            except Exception:
                continue
        logger.info("congress_quiverquant_fetched trades=%d", len(trades))
        return trades
    except Exception as exc:
        logger.info("congress_quiverquant_error err=%s", exc)
        return []


def _fetch_edgar_congress(cutoff: dt.date) -> List[Dict[str, Any]]:
    """Fetch congressional Form 4 filings via SEC EDGAR full-text search."""
    try:
        import httpx
        # Search for Form 4 filings tagged with congress keywords
        params = {
            "q": '"U.S. House" OR "U.S. Senate" OR "Member of Congress"',
            "forms": "4",
            "dateRange": "custom",
            "startdt": cutoff.isoformat(),
            "enddt": dt.date.today().isoformat(),
            "_source": "file_date,entity_name,file_num,period_of_report",
        }
        resp = httpx.get(_EDGAR_SEARCH_URL, params=params, headers=_HEADERS, timeout=20)
        if resp.status_code == 500:
            logger.warning("congress_edgar_500 — SEC server error, skipping")
            return []
        if resp.status_code != 200:
            logger.info("congress_edgar_unavailable status=%d", resp.status_code)
            return []

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            return []

        # EDGAR Form 4 XML parsing is complex — return placeholder metadata
        # so we know the source is available even if parsing is partial
        trades: List[Dict[str, Any]] = []
        for hit in hits[:50]:
            src = hit.get("_source", {})
            entity = src.get("entity_name", "")
            period = src.get("period_of_report", "")
            if not period:
                continue
            try:
                tx_date = dt.date.fromisoformat(period[:10])
                if tx_date < cutoff:
                    continue
            except Exception:
                continue
            trades.append({
                "symbol": "",  # Cannot reliably extract symbol from EDGAR search metadata
                "transaction_date": period[:10],
                "politician": entity,
                "party": "",
                "chamber": "unknown",
                "transaction_type": "unknown",
                "amount_range": "",
                "amount_min": None,
                "amount_max": None,
                "disclosure_date": src.get("file_date", ""),
                "source": "sec_edgar",
            })
        logger.info("congress_edgar_fetched hits=%d parsed=%d", len(hits), len(trades))
        return trades
    except Exception as exc:
        logger.info("congress_edgar_error err=%s", exc)
        return []


def _fetch_s3_house(cutoff: dt.date) -> List[Dict[str, Any]]:
    try:
        import httpx
        resp = httpx.get(_HOUSE_S3, headers=_HEADERS, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            logger.debug("congress_s3_house_unavailable status=%d", resp.status_code)
            return []
        trades = []
        for trade in resp.json():
            try:
                tx_date_str = trade.get("transaction_date", "")
                if not tx_date_str or tx_date_str == "N/A":
                    continue
                tx_date = dt.date.fromisoformat(tx_date_str[:10])
                if tx_date < cutoff:
                    continue
                ticker = trade.get("ticker", "").strip().upper()
                if not ticker or ticker in ("N/A", "--", ""):
                    continue
                tx_type = trade.get("type", "").lower()
                if "purchase" in tx_type or "buy" in tx_type:
                    direction = "buy"
                elif "sale" in tx_type or "sell" in tx_type:
                    direction = "sell"
                else:
                    continue
                amount_str = str(trade.get("amount", ""))
                amin, amax = _parse_amount_range(amount_str)
                trades.append({
                    "symbol": ticker,
                    "transaction_date": tx_date.isoformat(),
                    "politician": trade.get("representative", ""),
                    "party": trade.get("party", ""),
                    "chamber": "house",
                    "transaction_type": direction,
                    "amount_range": amount_str,
                    "amount_min": amin,
                    "amount_max": amax,
                    "disclosure_date": trade.get("disclosure_date", ""),
                    "source": "house_watcher",
                })
            except Exception:
                continue
        return trades
    except Exception as exc:
        logger.debug("congress_s3_house_error err=%s", exc)
        return []


def _fetch_s3_senate(cutoff: dt.date) -> List[Dict[str, Any]]:
    try:
        import httpx
        resp = httpx.get(_SENATE_S3, headers=_HEADERS, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            logger.debug("congress_s3_senate_unavailable status=%d", resp.status_code)
            return []
        trades = []
        for trade in resp.json():
            try:
                tx_date_str = trade.get("transaction_date", "")
                if not tx_date_str or tx_date_str == "N/A":
                    continue
                tx_date = dt.date.fromisoformat(tx_date_str[:10])
                if tx_date < cutoff:
                    continue
                ticker = trade.get("ticker", "").strip().upper()
                if not ticker or ticker in ("N/A", "--", ""):
                    continue
                tx_type = trade.get("type", "").lower()
                if "purchase" in tx_type or "buy" in tx_type:
                    direction = "buy"
                elif "sale" in tx_type or "sell" in tx_type:
                    direction = "sell"
                else:
                    continue
                amount_str = str(trade.get("amount", ""))
                amin, amax = _parse_amount_range(amount_str)
                trades.append({
                    "symbol": ticker,
                    "transaction_date": tx_date.isoformat(),
                    "politician": trade.get(
                        "senator",
                        (trade.get("first_name", "") + " " + trade.get("last_name", "")).strip(),
                    ),
                    "party": trade.get("party", ""),
                    "chamber": "senate",
                    "transaction_type": direction,
                    "amount_range": amount_str,
                    "amount_min": amin,
                    "amount_max": amax,
                    "disclosure_date": trade.get("disclosure_date", ""),
                    "source": "senate_watcher",
                })
            except Exception:
                continue
        return trades
    except Exception as exc:
        logger.debug("congress_s3_senate_error err=%s", exc)
        return []


def _get_committee_assignments() -> Dict[str, List[str]]:
    """Fetch congressional committee assignments from congress.gov API v3.

    Returns {politician_name_lower: [committee_name_lower, ...]}.
    Cached in-process; returns {} on failure.
    """
    global _committee_cache
    if _committee_cache is not None:
        return _committee_cache
    try:
        import httpx
        assignments: Dict[str, List[str]] = {}
        for chamber in ("senate", "house"):
            resp = httpx.get(
                f"https://api.congress.gov/v3/committee/{chamber}",
                params={"api_key": "DEMO_KEY", "limit": 50, "format": "json"},
                headers={"User-Agent": "AXIOM/1.0 Research"},
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            for committee in resp.json().get("committees", []):
                cname = committee.get("name", "").lower()
                for member in (committee.get("members") or {}).get("item") or []:
                    pname = member.get("name", "").strip().lower()
                    if pname:
                        assignments.setdefault(pname, []).append(cname)
        _committee_cache = assignments
        logger.info("committee_assignments_loaded politicians=%d", len(assignments))
        return assignments
    except Exception as exc:
        logger.debug("committee_assignments_error err=%s", exc)
        _committee_cache = {}
        return {}


def _fetch_from_capitoltrades(days_back: int = 90) -> List[Dict[str, Any]]:
    """Scrape Capitol Trades for STOCK Act disclosures (Next.js __NEXT_DATA__ pattern)."""
    try:
        import httpx
        cutoff = dt.date.today() - dt.timedelta(days=days_back)
        trades: List[Dict[str, Any]] = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        for page in range(1, 4):
            url = f"{_CAPITOLTRADES_URL}?page={page}&pageSize=96"
            try:
                resp = httpx.get(url, headers=headers, timeout=20, follow_redirects=True)
                if resp.status_code != 200:
                    logger.debug("capitoltrades_page_failed page=%d status=%d", page, resp.status_code)
                    break
                match = re.search(
                    r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                    resp.text,
                    re.DOTALL,
                )
                if not match:
                    logger.debug("capitoltrades_no_next_data page=%d", page)
                    break
                next_data = _json.loads(match.group(1))
                page_props = next_data.get("props", {}).get("pageProps", {})
                page_trades = (
                    page_props.get("trades", {}).get("data")
                    or page_props.get("data")
                    or []
                )
                if not page_trades:
                    break

                found_old = False
                for item in page_trades:
                    try:
                        tx_date_str = (
                            item.get("txDate")
                            or item.get("tradeDate")
                            or item.get("transaction_date")
                            or ""
                        )
                        if not tx_date_str:
                            continue
                        tx_date = dt.date.fromisoformat(tx_date_str[:10])
                        if tx_date < cutoff:
                            found_old = True
                            continue
                        ticker = (
                            item.get("ticker")
                            or (item.get("asset") or {}).get("ticker")
                            or ""
                        ).strip().upper()
                        if not ticker or ticker in ("N/A", "--", ""):
                            continue
                        raw_type = (item.get("type") or item.get("transactionType") or "").upper()
                        if raw_type in ("P", "PURCHASE", "BUY"):
                            direction = "buy"
                        elif raw_type.startswith("S"):
                            direction = "sell"
                        else:
                            continue
                        pol = item.get("politician") or item.get("representative") or {}
                        if isinstance(pol, dict):
                            politician_name = (
                                pol.get("name")
                                or (pol.get("firstName", "") + " " + pol.get("lastName", "")).strip()
                            )
                            party = pol.get("party") or pol.get("partyAffiliation") or ""
                            chamber_raw = pol.get("chamber") or ""
                        else:
                            politician_name = str(pol)
                            party = item.get("party", "")
                            chamber_raw = item.get("chamber", "")
                        chamber = "senate" if "senate" in chamber_raw.lower() else "house"
                        amount_str = str(
                            item.get("size") or item.get("amount") or item.get("reportedAmount") or ""
                        )
                        amin, amax = _parse_amount_range(amount_str)
                        trades.append({
                            "symbol": ticker,
                            "transaction_date": tx_date.isoformat(),
                            "politician": politician_name,
                            "party": party,
                            "chamber": chamber,
                            "transaction_type": direction,
                            "amount_range": amount_str,
                            "amount_min": amin,
                            "amount_max": amax,
                            "disclosure_date": item.get("reportedDate") or item.get("disclosure_date") or "",
                            "source": "capitoltrades",
                        })
                    except Exception:
                        continue
                if found_old:
                    break
            except Exception as exc:
                logger.debug("capitoltrades_page_error page=%d err=%s", page, exc)
                break
        logger.info("capitoltrades_fetched trades=%d", len(trades))
        return trades
    except Exception as exc:
        logger.info("capitoltrades_error err=%s", exc)
        return []


def fetch_recent_congress_trades(days_back: int = 90) -> List[Dict[str, Any]]:
    """Fetch recent congressional stock trades — tries Capitol Trades, QuiverQuant, SEC EDGAR, S3.

    Returns list of trade dicts with:
      symbol, transaction_date, politician, party, chamber,
      transaction_type (buy/sell), amount_range, amount_min, amount_max,
      disclosure_date, source
    """
    cutoff = dt.date.today() - dt.timedelta(days=days_back)

    # Source 1: Capitol Trades (public, no auth, most reliable from cloud)
    trades = _fetch_from_capitoltrades(days_back)
    if trades:
        logger.info("congress_trades_fetched source=capitoltrades total=%d", len(trades))
        return trades

    # Source 2: QuiverQuant (requires paid API key)
    trades = _fetch_quiverquant(cutoff)
    if trades:
        logger.info("congress_trades_fetched source=quiverquant total=%d", len(trades))
        return trades

    # Source 3: S3 house/senate watchers (may be blocked on Railway)
    house = _fetch_s3_house(cutoff)
    senate = _fetch_s3_senate(cutoff)
    trades = house + senate
    if trades:
        logger.info(
            "congress_trades_fetched source=s3 house=%d senate=%d total=%d",
            len(house), len(senate), len(trades),
        )
        return trades

    # Source 4: SEC EDGAR (symbol parsing limited, used as availability signal)
    edgar = _fetch_edgar_congress(cutoff)
    if edgar:
        symbol_trades = [t for t in edgar if t.get("symbol")]
        logger.info(
            "congress_trades_fetched source=sec_edgar total=%d with_symbols=%d",
            len(edgar), len(symbol_trades),
        )
        return symbol_trades

    logger.info(
        "congress_all_sources_failed days_back=%d — CIS defaulting to 50 (neutral)",
        days_back,
    )
    return []


def compute_congress_score(
    symbol: str,
    trades: List[Dict[str, Any]],
    sector: Optional[str] = None,
    committee_assignments: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Compute a congressional trading intelligence score for a symbol.

    Score from 0–100:
      > 60 = net buying by congress (bullish signal)
      40–60 = neutral / mixed
      < 40 = net selling (bearish signal)

    Weights recent trades (< 30 days) 2×.
    Applies 1.5× multiplier for politicians on committees relevant to the symbol's sector.
    """
    symbol_trades = [t for t in trades if t.get("symbol") == symbol]

    if not symbol_trades:
        return {
            "symbol": symbol,
            "congress_score": 50.0,
            "net_signal": "neutral",
            "buy_count": 0,
            "sell_count": 0,
            "recent_trades": [],
        }

    # Determine which committee keywords are relevant to this sector
    relevant_keywords: List[str] = []
    if sector:
        sector_lower = sector.lower()
        for keyword, sectors in COMMITTEE_SECTOR_MAP.items():
            if any(sector_lower in s.lower() for s in sectors):
                relevant_keywords.append(keyword)

    assignments = committee_assignments or {}

    def _trade_weight(trade: Dict[str, Any], recency_mult: float = 1.0) -> float:
        weight = recency_mult
        if relevant_keywords:
            pol_name = trade.get("politician", "").lower()
            pol_committees = assignments.get(pol_name, [])
            if any(kw in c for kw in relevant_keywords for c in pol_committees):
                weight *= 1.5
        return weight

    buy_count = sum(1 for t in symbol_trades if t["transaction_type"] == "buy")
    sell_count = sum(1 for t in symbol_trades if t["transaction_type"] == "sell")
    total = buy_count + sell_count

    if total == 0:
        score = 50.0
    else:
        cutoff_30 = (dt.date.today() - dt.timedelta(days=30)).isoformat()
        weighted_buys = sum(
            _trade_weight(t, 2.0 if t["transaction_date"] >= cutoff_30 else 1.0)
            for t in symbol_trades if t["transaction_type"] == "buy"
        )
        weighted_sells = sum(
            _trade_weight(t, 2.0 if t["transaction_date"] >= cutoff_30 else 1.0)
            for t in symbol_trades if t["transaction_type"] == "sell"
        )
        weighted_total = weighted_buys + weighted_sells
        score = round((weighted_buys / weighted_total) * 100, 1) if weighted_total > 0 else 50.0

    net_signal = "bullish" if score > 60 else "bearish" if score < 40 else "neutral"

    return {
        "symbol": symbol,
        "congress_score": score,
        "net_signal": net_signal,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "total_trades_90d": total,
        "recent_trades": sorted(
            symbol_trades, key=lambda t: t["transaction_date"], reverse=True
        )[:5],
    }
