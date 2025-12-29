from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from api import db


def fetch_bars(symbol: str, from_date: dt.date, to_date: dt.date) -> List[Dict[str, Any]]:
    rows = db.safe_fetchall(
        """
        SELECT date, open, high, low, close, adj_close, volume, source
        FROM prosperity_daily_bars
        WHERE symbol=%s AND date BETWEEN %s AND %s
        ORDER BY date ASC
        """,
        (symbol, from_date, to_date),
    )
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "date": r[0].isoformat(),
                "open": r[1],
                "high": r[2],
                "low": r[3],
                "close": r[4],
                "adj_close": r[5],
                "volume": r[6],
                "source": r[7],
            }
        )
    return out


def latest_signal(symbol: str, lookback: int) -> Optional[Dict[str, Any]]:
    row = db.safe_fetchone(
        """
        SELECT symbol, as_of, lookback, score, signal, regime, thresholds, confidence, notes, features, meta
        FROM prosperity_signals_daily
        WHERE symbol=%s AND lookback=%s
        ORDER BY as_of DESC
        LIMIT 1
        """,
        (symbol, lookback),
    )
    if not row:
        return None
    return {
        "symbol": row[0],
        "as_of": row[1].isoformat(),
        "lookback": row[2],
        "score": row[3],
        "signal": row[4],
        "regime": row[5],
        "thresholds": row[6],
        "confidence": row[7],
        "notes": row[8],
        "features": row[9],
        "meta": row[10],
    }


def latest_features(symbol: str, lookback: int) -> Optional[Dict[str, Any]]:
    row = db.safe_fetchone(
        """
        SELECT symbol, as_of, lookback, features, meta
        FROM prosperity_features_daily
        WHERE symbol=%s AND lookback=%s
        ORDER BY as_of DESC
        LIMIT 1
        """,
        (symbol, lookback),
    )
    if not row:
        return None
    return {
        "symbol": row[0],
        "as_of": row[1].isoformat(),
        "lookback": row[2],
        "features": row[3],
        "meta": row[4],
    }
