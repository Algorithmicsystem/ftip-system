"""Congressional trading scraper — public data via STOCK Act disclosures.

Scrapes housestockwatcher.com for recent congressional trades.
This data is public by law and freely available.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_CONGRESS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AXIOM/1.0; +https://axiom.ai)",
    "Accept": "application/json",
}

_HOUSE_URL = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
_SENATE_URL = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"


def _fetch_house_trades(cutoff: dt.date) -> List[Dict[str, Any]]:
    try:
        import httpx
        resp = httpx.get(_HOUSE_URL, headers=_CONGRESS_HEADERS, timeout=30, follow_redirects=True)
        if resp.status_code != 200:
            logger.warning("congress_house_fetch_failed status=%d", resp.status_code)
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
                trades.append({
                    "symbol": ticker,
                    "transaction_date": tx_date.isoformat(),
                    "politician": trade.get("representative", ""),
                    "party": trade.get("party", ""),
                    "chamber": "house",
                    "transaction_type": direction,
                    "amount_range": trade.get("amount", ""),
                    "disclosure_date": trade.get("disclosure_date", ""),
                })
            except Exception:
                continue
        return trades
    except Exception as exc:
        logger.warning("congress_house_failed err=%s", exc)
        return []


def _fetch_senate_trades(cutoff: dt.date) -> List[Dict[str, Any]]:
    try:
        import httpx
        resp = httpx.get(_SENATE_URL, headers=_CONGRESS_HEADERS, timeout=30, follow_redirects=True)
        if resp.status_code != 200:
            logger.warning("congress_senate_fetch_failed status=%d", resp.status_code)
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
                trades.append({
                    "symbol": ticker,
                    "transaction_date": tx_date.isoformat(),
                    "politician": trade.get("senator", trade.get("first_name", "") + " " + trade.get("last_name", "")),
                    "party": trade.get("party", ""),
                    "chamber": "senate",
                    "transaction_type": direction,
                    "amount_range": trade.get("amount", ""),
                    "disclosure_date": trade.get("disclosure_date", ""),
                })
            except Exception:
                continue
        return trades
    except Exception as exc:
        logger.warning("congress_senate_failed err=%s", exc)
        return []


def fetch_recent_congress_trades(days_back: int = 90) -> List[Dict[str, Any]]:
    """Fetch recent congressional stock trades from both house and senate watchers.

    Returns list of trade dicts with:
      symbol, transaction_date, politician, party, chamber,
      transaction_type (buy/sell), amount_range, disclosure_date
    """
    cutoff = dt.date.today() - dt.timedelta(days=days_back)
    house = _fetch_house_trades(cutoff)
    senate = _fetch_senate_trades(cutoff)
    trades = house + senate
    logger.info(
        "congress_trades_fetched house=%d senate=%d total=%d days_back=%d",
        len(house), len(senate), len(trades), days_back,
    )
    return trades


def compute_congress_score(symbol: str, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute a congressional trading intelligence score for a symbol.

    Score from 0–100:
      > 60 = net buying by congress (bullish signal)
      40–60 = neutral / mixed
      < 40 = net selling (bearish signal)

    Weights recent trades (< 30 days) double.
    """
    symbol_trades = [t for t in trades if t["symbol"] == symbol]

    if not symbol_trades:
        return {
            "symbol": symbol,
            "congress_score": 50.0,
            "net_signal": "neutral",
            "buy_count": 0,
            "sell_count": 0,
            "recent_trades": [],
        }

    buy_count = sum(1 for t in symbol_trades if t["transaction_type"] == "buy")
    sell_count = sum(1 for t in symbol_trades if t["transaction_type"] == "sell")
    total = buy_count + sell_count

    if total == 0:
        score = 50.0
    else:
        cutoff_30 = (dt.date.today() - dt.timedelta(days=30)).isoformat()
        recent_buys = sum(
            1 for t in symbol_trades
            if t["transaction_type"] == "buy" and t["transaction_date"] >= cutoff_30
        )
        recent_sells = sum(
            1 for t in symbol_trades
            if t["transaction_type"] == "sell" and t["transaction_date"] >= cutoff_30
        )
        # Recent trades count 2x
        weighted_buys = (buy_count - recent_buys) + (recent_buys * 2)
        weighted_sells = (sell_count - recent_sells) + (recent_sells * 2)
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
