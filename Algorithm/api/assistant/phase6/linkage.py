from __future__ import annotations

import datetime as dt
from typing import Any, Callable, Dict, List, Optional

from api import db

from .common import as_date, hold_band, horizon_days, iso_date, safe_float


def default_bar_fetcher(symbol: str, as_of_date: dt.date, limit: int) -> List[Dict[str, Any]]:
    if not db.db_enabled() or not db.db_read_enabled():
        return []
    rows = db.safe_fetchall(
        """
        SELECT as_of_date, close
        FROM market_bars_daily
        WHERE symbol = %s
          AND as_of_date >= %s
        ORDER BY as_of_date ASC
        LIMIT %s
        """,
        (symbol, as_of_date, limit),
    )
    return [
        {
            "as_of_date": iso_date(row[0]),
            "close": safe_float(row[1]),
        }
        for row in rows
        if safe_float(row[1]) is not None
    ]


def _directional_correct(signal_label: str, forward_return: float, threshold: float) -> bool:
    action = str(signal_label or "HOLD").upper()
    if action == "BUY":
        return forward_return > threshold
    if action == "SELL":
        return forward_return < -threshold
    return abs(forward_return) <= threshold


def _outcome_label(forward_return: float, threshold: float) -> str:
    if forward_return > threshold:
        return "positive"
    if forward_return < -threshold:
        return "negative"
    return "neutral"


def _aligned_return(signal_label: str, forward_return: float) -> float:
    action = str(signal_label or "HOLD").upper()
    if action == "SELL":
        return -forward_return
    if action == "HOLD":
        return -abs(forward_return)
    return forward_return


def link_realized_outcome(
    prediction: Dict[str, Any],
    *,
    bar_fetcher: Optional[Callable[[str, dt.date, int], List[Dict[str, Any]]]] = None,
    evaluation_as_of_date: Optional[dt.date] = None,
) -> Dict[str, Any]:
    symbol = str(prediction.get("symbol") or "")
    prediction_date = as_date(prediction.get("as_of_date"))
    if not symbol or prediction_date is None:
        return {
            "outcome_status": "invalid_prediction",
            "matured": False,
        }

    horizon = horizon_days(prediction)
    if bar_fetcher is None:
        bar_fetcher = default_bar_fetcher
    rows = bar_fetcher(symbol, prediction_date, horizon + 1)
    if not rows:
        return {
            "outcome_status": "missing_bars",
            "matured": False,
            "horizon_days": horizon,
        }

    entry_price = safe_float(rows[0].get("close"))
    if entry_price in (None, 0):
        return {
            "outcome_status": "invalid_entry_price",
            "matured": False,
            "horizon_days": horizon,
        }
    if len(rows) <= horizon:
        latest_available = rows[-1].get("as_of_date")
        return {
            "outcome_status": "pending",
            "matured": False,
            "horizon_days": horizon,
            "entry_price": entry_price,
            "latest_available_date": latest_available,
        }

    exit_row = rows[horizon]
    exit_price = safe_float(exit_row.get("close"))
    if exit_price in (None, 0):
        return {
            "outcome_status": "invalid_exit_price",
            "matured": False,
            "horizon_days": horizon,
            "entry_price": entry_price,
        }

    path_prices = [safe_float(row.get("close")) for row in rows[1 : horizon + 1]]
    clean_path = [price for price in path_prices if price is not None]
    forward_return = exit_price / entry_price - 1.0
    favorable_excursion = max(clean_path) / entry_price - 1.0 if clean_path else forward_return
    adverse_excursion = min(clean_path) / entry_price - 1.0 if clean_path else forward_return
    threshold = hold_band(horizon)
    raw_signal = str(prediction.get("signal_action") or prediction.get("raw_signal") or "HOLD")
    final_signal = str(prediction.get("final_signal") or prediction.get("signal") or raw_signal)

    return {
        "outcome_status": "matured",
        "matured": True,
        "evaluation_as_of_date": iso_date(evaluation_as_of_date or dt.date.today()),
        "horizon_days": horizon,
        "entry_price": round(entry_price, 6),
        "exit_price": round(exit_price, 6),
        "forward_return": round(forward_return, 6),
        "absolute_forward_return": round(abs(forward_return), 6),
        "favorable_excursion": round(favorable_excursion, 6),
        "adverse_excursion": round(adverse_excursion, 6),
        "downside_tail_proxy": round(min(0.0, adverse_excursion), 6),
        "threshold": round(threshold, 6),
        "raw_signal_correct": _directional_correct(raw_signal, forward_return, threshold),
        "final_signal_correct": _directional_correct(final_signal, forward_return, threshold),
        "raw_signal_edge_return": round(_aligned_return(raw_signal, forward_return), 6),
        "final_signal_edge_return": round(_aligned_return(final_signal, forward_return), 6),
        "outcome_label": _outcome_label(forward_return, threshold),
        "entry_date": rows[0].get("as_of_date"),
        "exit_date": exit_row.get("as_of_date"),
    }
