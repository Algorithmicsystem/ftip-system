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
        SELECT symbol, as_of, lookback, score_mode, score, base_score, stacked_score, signal, regime, thresholds, confidence, notes, features, calibration_meta, meta, signal_hash
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
        "score_mode": row[3],
        "score": row[4],
        "base_score": row[5],
        "stacked_score": row[6],
        "signal": row[7],
        "regime": row[8],
        "thresholds": row[9],
        "confidence": row[10],
        "notes": row[11],
        "features": row[12],
        "calibration_meta": row[13],
        "meta": row[14],
        "signal_hash": row[15],
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
