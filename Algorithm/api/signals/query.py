"""Unified signal read path for the FTIP platform.

Single source of truth for fetching the latest signal for a symbol.  Reads
``prosperity_signals_daily`` first (canonical Phase 9 path) and falls back to
the legacy ``signals_daily`` table.  Always returns a normalised dict so
callers never need to know which table the row came from or worry about the
``signal``/``action`` column-name discrepancy.

Normalised return shape (all fields optional/nullable)::

    {
        "signal": "BUY" | "SELL" | "HOLD",   # canonical name
        "action": "BUY" | "SELL" | "HOLD",   # alias for backward compat
        "score": float,
        "confidence": float,
        "regime": str | None,
        "thresholds": dict,
        "score_mode": str | None,
        "base_score": float | None,
        "stacked_score": float | None,
        "entry_low": float | None,
        "entry_high": float | None,
        "stop_loss": float | None,
        "take_profit_1": float | None,
        "take_profit_2": float | None,
        "reason_codes": list,
        "reason_details": dict,
        "signal_version": str | None,      # always TEXT even when read from legacy INT col
        "feature_version": str | None,
        "meta": dict,
        "as_of": str,                       # ISO date
        "source_table": "prosperity_signals_daily" | "signals_daily",
    }

Returns ``None`` when no signal is found in either table.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

from api import db


def get_unified_signal(
    symbol: str,
    as_of_date: dt.date,
) -> Optional[Dict[str, Any]]:
    """Return the most recent signal for *symbol* on or before *as_of_date*.

    Reads ``prosperity_signals_daily`` first.  Falls back to ``signals_daily``
    when the canonical table has no row.
    """
    row = _read_prosperity(symbol, as_of_date)
    if row is not None:
        return row
    return _read_legacy(symbol, as_of_date)


# ---------------------------------------------------------------------------
# Internal readers
# ---------------------------------------------------------------------------

def _read_prosperity(symbol: str, as_of_date: dt.date) -> Optional[Dict[str, Any]]:
    row = db.safe_fetchone(
        """
        SELECT signal, score, confidence, regime, thresholds, score_mode,
               base_score, stacked_score, signal_version, feature_version, meta, as_of
        FROM prosperity_signals_daily
        WHERE symbol = %s AND as_of = %s
        ORDER BY updated_at DESC NULLS LAST
        LIMIT 1
        """,
        (symbol, as_of_date),
    )
    if not row:
        # Nearest earlier date fallback
        row = db.safe_fetchone(
            """
            SELECT signal, score, confidence, regime, thresholds, score_mode,
                   base_score, stacked_score, signal_version, feature_version, meta, as_of
            FROM prosperity_signals_daily
            WHERE symbol = %s AND as_of <= %s
            ORDER BY as_of DESC, updated_at DESC NULLS LAST
            LIMIT 1
            """,
            (symbol, as_of_date),
        )
    if not row:
        return None
    signal_str = str(row[0] or "HOLD").upper()
    meta = dict(row[10] or {})
    return {
        "signal": signal_str,
        "action": signal_str,
        "score": float(row[1] or 0.0),
        "confidence": float(row[2] or 0.0),
        "regime": row[3],
        "thresholds": dict(row[4] or {}),
        "score_mode": row[5],
        "base_score": float(row[6]) if row[6] is not None else None,
        "stacked_score": float(row[7]) if row[7] is not None else None,
        "entry_low": None,
        "entry_high": None,
        "stop_loss": None,
        "take_profit_1": None,
        "take_profit_2": None,
        "reason_codes": meta.get("reason_codes") or [],
        "reason_details": meta.get("reason_details") or {},
        "signal_version": str(row[8]) if row[8] is not None else None,
        "feature_version": str(row[9]) if row[9] is not None else None,
        "meta": meta,
        "as_of": row[11].isoformat() if hasattr(row[11], "isoformat") else str(row[11] or as_of_date),
        "source_table": "prosperity_signals_daily",
    }


def _read_legacy(symbol: str, as_of_date: dt.date) -> Optional[Dict[str, Any]]:
    row = db.safe_fetchone(
        """
        SELECT action, score, confidence, regime, thresholds, score_mode,
               base_score, stacked_score, signal_version, entry_low, entry_high,
               stop_loss, take_profit_1, take_profit_2, reason_codes, reason_details,
               signal_meta, as_of_date
        FROM signals_daily
        WHERE symbol = %s AND as_of_date = %s
        """,
        (symbol, as_of_date),
    )
    if not row:
        row = db.safe_fetchone(
            """
            SELECT action, score, confidence, regime, thresholds, score_mode,
                   base_score, stacked_score, signal_version, entry_low, entry_high,
                   stop_loss, take_profit_1, take_profit_2, reason_codes, reason_details,
                   signal_meta, as_of_date
            FROM signals_daily
            WHERE symbol = %s AND as_of_date <= %s
            ORDER BY as_of_date DESC
            LIMIT 1
            """,
            (symbol, as_of_date),
        )
    if not row:
        return None
    action_str = str(row[0] or "HOLD").upper()
    meta = dict(row[16] or {})
    # signal_version is stored as INT in signals_daily — normalise to TEXT
    sv_raw = row[8]
    signal_version = str(sv_raw) if sv_raw is not None else None
    as_of_raw = row[17]
    return {
        "signal": action_str,
        "action": action_str,
        "score": float(row[1] or 0.0),
        "confidence": float(row[2] or 0.0),
        "regime": row[3],
        "thresholds": dict(row[4] or {}),
        "score_mode": row[5],
        "base_score": float(row[6]) if row[6] is not None else None,
        "stacked_score": float(row[7]) if row[7] is not None else None,
        "entry_low": float(row[9]) if row[9] is not None else None,
        "entry_high": float(row[10]) if row[10] is not None else None,
        "stop_loss": float(row[11]) if row[11] is not None else None,
        "take_profit_1": float(row[12]) if row[12] is not None else None,
        "take_profit_2": float(row[13]) if row[13] is not None else None,
        "reason_codes": list(row[14] or []),
        "reason_details": dict(row[15] or {}),
        "signal_version": signal_version,
        "feature_version": meta.get("feature_version"),
        "meta": meta,
        "as_of": as_of_raw.isoformat() if hasattr(as_of_raw, "isoformat") else str(as_of_raw or as_of_date),
        "source_table": "signals_daily",
    }


__all__ = ["get_unified_signal"]
