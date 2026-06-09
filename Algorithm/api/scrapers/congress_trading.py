"""Congressional trading scraper — public data via STOCK Act disclosures.

Scrapes housestockwatcher.com for recent congressional trades.
This data is public by law and freely available.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def fetch_recent_congress_trades(days_back: int = 90) -> List[Dict[str, Any]]:
    """Fetch recent congressional stock trades from housestockwatcher.com.

    Returns list of trade dicts with:
      symbol, transaction_date, politician, party, chamber,
      transaction_type (buy/sell), amount_range, disclosure_date
    """
    try:
        import httpx

        url = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        if resp.status_code != 200:
            logger.warning("congress_trades_fetch_failed status=%d", resp.status_code)
            return []

        all_trades = resp.json()
        cutoff = dt.date.today() - dt.timedelta(days=days_back)

        trades = []
        for trade in all_trades:
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

        logger.info("congress_trades_fetched count=%d days_back=%d", len(trades), days_back)
        return trades

    except Exception as exc:
        logger.warning("congress_trades_failed err=%s", exc)
        return []


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
