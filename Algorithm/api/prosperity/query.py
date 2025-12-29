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
        SELECT symbol, as_of_date, lookback, score_mode, score, signal, regime, thresholds, confidence
        FROM prosperity_signals_daily
        WHERE symbol=%s AND lookback=%s
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (symbol, lookback),
    )
    if not row:
        return None
    return {
        "symbol": row[0],
        "as_of_date": row[1].isoformat(),
        "lookback": row[2],
        "score_mode": row[3],
        "score": row[4],
        "signal": row[5],
        "regime": row[6],
        "thresholds": row[7],
        "confidence": row[8],
    }


def latest_features(symbol: str, lookback: int) -> Optional[Dict[str, Any]]:
    row = db.safe_fetchone(
        """
        SELECT symbol, as_of_date, lookback, mom_5, mom_21, mom_63, trend_sma20_50, volatility_ann, rsi14, volume_z20, last_close, regime
        FROM prosperity_features_daily
        WHERE symbol=%s AND lookback=%s
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (symbol, lookback),
    )
    if not row:
        return None
    return {
        "symbol": row[0],
        "as_of_date": row[1].isoformat(),
        "lookback": row[2],
        "mom_5": row[3],
        "mom_21": row[4],
        "mom_63": row[5],
        "trend_sma20_50": row[6],
        "volatility_ann": row[7],
        "rsi14": row[8],
        "volume_z20": row[9],
        "last_close": row[10],
        "regime": row[11],
    }
