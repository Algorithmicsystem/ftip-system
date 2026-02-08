from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

from api import db


def fetch_bars(
    symbol: str, from_date: dt.date, to_date: dt.date
) -> List[Dict[str, Any]]:
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


def fetch_bars_with_latest(
    symbol: str, from_date: dt.date, to_date: dt.date
) -> Tuple[List[Dict[str, Any]], Optional[dt.date]]:
    bars = fetch_bars(symbol, from_date, to_date)
    latest_date = None
    for bar in bars:
        bar_date = dt.date.fromisoformat(bar["date"])
        if latest_date is None or bar_date > latest_date:
            latest_date = bar_date
    return bars, latest_date


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


def signal_as_of(
    symbol: str, lookback: int, as_of: dt.date
) -> Optional[Dict[str, Any]]:
    row = db.safe_fetchone(
        """
        SELECT symbol, as_of, lookback, score_mode, score, base_score, stacked_score, signal, regime, thresholds, confidence,
               notes, features, calibration_meta, meta, signal_hash
        FROM prosperity_signals_daily
        WHERE symbol=%s AND lookback=%s AND as_of<=%s
        ORDER BY as_of DESC
        LIMIT 1
        """,
        (symbol, lookback, as_of),
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


def signal_history(
    symbol: str, lookback: int, as_of: dt.date, window_days: int
) -> List[Dict[str, Any]]:
    cutoff = as_of - dt.timedelta(days=max(0, window_days))
    rows = db.safe_fetchall(
        """
        SELECT as_of, score_mode, score, signal, regime, thresholds, confidence
        FROM prosperity_signals_daily
        WHERE symbol=%s AND lookback=%s AND as_of BETWEEN %s AND %s
        ORDER BY as_of DESC
        """,
        (symbol, lookback, cutoff, as_of),
    )
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "as_of": row[0].isoformat(),
                "score_mode": row[1],
                "score": row[2],
                "signal": row[3],
                "regime": row[4],
                "thresholds": row[5],
                "confidence": row[6],
            }
        )
    return out


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


def features_as_of(
    symbol: str, lookback: int, as_of: dt.date
) -> Optional[Dict[str, Any]]:
    row = db.safe_fetchone(
        """
        SELECT symbol, as_of, lookback, features, meta
        FROM prosperity_features_daily
        WHERE symbol=%s AND lookback=%s AND as_of<=%s
        ORDER BY as_of DESC
        LIMIT 1
        """,
        (symbol, lookback, as_of),
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
