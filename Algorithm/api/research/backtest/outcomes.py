from __future__ import annotations

import datetime as dt
from typing import Any, Callable, Dict, List, Optional

from api import db

from .common import as_date, clamp, hold_band, horizon_days, iso_date, safe_float
from .costs import estimate_trade_cost


def default_ohlc_bar_fetcher(symbol: str, as_of_date: dt.date, limit: int) -> List[Dict[str, Any]]:
    if not db.db_enabled() or not db.db_read_enabled():
        return []
    rows = db.safe_fetchall(
        """
        SELECT as_of_date, open, high, low, close, volume
        FROM market_bars_daily
        WHERE symbol = %s
          AND as_of_date >= %s
        ORDER BY as_of_date ASC
        LIMIT %s
        """,
        (symbol, as_of_date, limit),
    )
    output: List[Dict[str, Any]] = []
    for row in rows:
        close = safe_float(row[4])
        if close is None:
            continue
        output.append(
            {
                "as_of_date": iso_date(row[0]),
                "open": safe_float(row[1]) or close,
                "high": safe_float(row[2]) or close,
                "low": safe_float(row[3]) or close,
                "close": close,
                "volume": safe_float(row[5]),
            }
        )
    return output


def _direction_multiplier(signal_label: str) -> int:
    action = str(signal_label or "HOLD").upper()
    if action == "BUY":
        return 1
    if action == "SELL":
        return -1
    return 0


def _aligned_return(signal_label: str, forward_return: float) -> float:
    direction = _direction_multiplier(signal_label)
    if direction == -1:
        return -forward_return
    if direction == 0:
        return -abs(forward_return)
    return forward_return


def _directionally_correct(signal_label: str, forward_return: float, threshold: float) -> bool:
    action = str(signal_label or "HOLD").upper()
    if action == "BUY":
        return forward_return > threshold
    if action == "SELL":
        return forward_return < -threshold
    return abs(forward_return) <= threshold


def _invalidation_threshold(record: Dict[str, Any], horizon_value: int) -> float:
    feature_vector = dict(record.get("feature_vector") or {})
    proprietary_scores = dict(record.get("proprietary_scores") or {})
    atr_pct = safe_float(feature_vector.get("atr_pct")) or 0.025
    implementation_fragility = safe_float(feature_vector.get("implementation_fragility_score"))
    if implementation_fragility is None:
        implementation_fragility = safe_float(proprietary_scores.get("Signal Fragility Index")) or 45.0
    event_overhang = safe_float(feature_vector.get("event_overhang_score")) or 0.0
    threshold = atr_pct * 1.85 + implementation_fragility / 2600.0 + event_overhang / 4200.0
    threshold *= 1.0 + min(max(horizon_value - 5, 0), 63) / 220.0
    return clamp(threshold, 0.0125, 0.12)


def evaluate_prediction_outcome(
    record: Dict[str, Any],
    *,
    bar_fetcher: Optional[Callable[[str, dt.date, int], List[Dict[str, Any]]]] = None,
    cost_model: Optional[Dict[str, Any]] = None,
    evaluation_as_of_date: Optional[dt.date] = None,
) -> Dict[str, Any]:
    symbol = str(record.get("symbol") or "")
    prediction_date = as_date(record.get("as_of_date"))
    if not symbol or prediction_date is None:
        return {"outcome_status": "invalid_prediction", "matured": False}

    horizon_value = safe_float(record.get("horizon_days"))
    if horizon_value is None:
        horizon_value = horizon_days(record.get("horizon"))
    horizon_value = int(horizon_value)
    cost_model = dict(cost_model or {})
    if bar_fetcher is None:
        bar_fetcher = default_ohlc_bar_fetcher

    rows = bar_fetcher(symbol, prediction_date, horizon_value + 1)
    if not rows:
        return {
            "outcome_status": "missing_bars",
            "matured": False,
            "horizon_days": horizon_value,
        }
    if len(rows) <= horizon_value:
        return {
            "outcome_status": "pending",
            "matured": False,
            "horizon_days": horizon_value,
            "latest_available_date": rows[-1].get("as_of_date"),
        }

    entry_row = rows[0]
    exit_row = rows[horizon_value]
    entry_price = safe_float(entry_row.get("close"))
    exit_price = safe_float(exit_row.get("close"))
    if entry_price in (None, 0) or exit_price is None:
        return {"outcome_status": "invalid_prices", "matured": False, "horizon_days": horizon_value}

    action = str(record.get("final_signal") or record.get("signal_action") or "HOLD").upper()
    direction = _direction_multiplier(action)
    raw_forward_return = exit_price / entry_price - 1.0
    gross_trade_return = raw_forward_return * direction if direction != 0 else 0.0
    gross_edge_return = _aligned_return(action, raw_forward_return)
    threshold = hold_band(horizon_value)

    highs = [safe_float(row.get("high")) or safe_float(row.get("close")) for row in rows[1 : horizon_value + 1]]
    lows = [safe_float(row.get("low")) or safe_float(row.get("close")) for row in rows[1 : horizon_value + 1]]
    closes = [safe_float(row.get("close")) for row in rows[1 : horizon_value + 1]]
    clean_highs = [value for value in highs if value is not None]
    clean_lows = [value for value in lows if value is not None]
    clean_closes = [value for value in closes if value is not None]

    if direction >= 0:
        favorable_excursion = max(clean_highs) / entry_price - 1.0 if clean_highs else raw_forward_return
        adverse_excursion = min(clean_lows) / entry_price - 1.0 if clean_lows else raw_forward_return
    else:
        favorable_excursion = -(min(clean_lows) / entry_price - 1.0) if clean_lows else -raw_forward_return
        adverse_excursion = -(max(clean_highs) / entry_price - 1.0) if clean_highs else -raw_forward_return

    threshold_invalidation = _invalidation_threshold(record, horizon_value)
    invalidation_triggered = (adverse_excursion or 0.0) <= -threshold_invalidation

    directional_path: List[float] = []
    for close in clean_closes:
        step_return = close / entry_price - 1.0
        directional_path.append(_aligned_return(action, step_return))
    max_path = max(directional_path, default=gross_edge_return)
    peak_index = directional_path.index(max_path) if directional_path else 0
    half_life_days = None
    if directional_path:
        decay_target = max_path * 0.5
        for idx, value in enumerate(directional_path[peak_index:], start=peak_index + 1):
            if value <= decay_target:
                half_life_days = idx
                break
        if half_life_days is None:
            half_life_days = len(directional_path)

    continuation_decay_score = 0.0
    if max_path and max_path > 0:
        continuation_decay_score = clamp((max_path - gross_edge_return) / max_path * 100.0, 0.0, 100.0)

    friction = estimate_trade_cost(record, cost_model=cost_model)
    net_trade_return = gross_trade_return - friction["cost_rate"]
    net_edge_return = gross_edge_return - friction["cost_rate"]

    return {
        "outcome_status": "matured",
        "matured": True,
        "evaluation_as_of_date": iso_date(evaluation_as_of_date or dt.date.today()),
        "horizon_days": horizon_value,
        "entry_date": entry_row.get("as_of_date"),
        "exit_date": exit_row.get("as_of_date"),
        "entry_price": round(entry_price, 6),
        "exit_price": round(exit_price, 6),
        "forward_return": round(raw_forward_return, 6),
        "absolute_forward_return": round(abs(raw_forward_return), 6),
        "raw_signal_correct": _directionally_correct(
            str(record.get("signal_action") or action),
            raw_forward_return,
            threshold,
        ),
        "final_signal_correct": _directionally_correct(action, raw_forward_return, threshold),
        "raw_signal_edge_return": round(
            _aligned_return(str(record.get("signal_action") or action), raw_forward_return),
            6,
        ),
        "final_signal_edge_return": round(gross_edge_return, 6),
        "gross_edge_return": round(gross_edge_return, 6),
        "net_edge_return": round(net_edge_return, 6),
        "gross_trade_return": round(gross_trade_return, 6),
        "net_trade_return": round(net_trade_return, 6),
        "estimated_cost_bps": friction["total_bps"],
        "friction_cost_summary": friction,
        "favorable_excursion": round(favorable_excursion, 6),
        "adverse_excursion": round(adverse_excursion, 6),
        "mae": round(abs(min(0.0, adverse_excursion or 0.0)), 6),
        "mfe": round(max(0.0, favorable_excursion or 0.0), 6),
        "downside_tail_proxy": round(min(0.0, adverse_excursion or 0.0), 6),
        "invalidation_threshold": round(threshold_invalidation, 6),
        "invalidation_triggered": bool(invalidation_triggered),
        "signal_half_life_days": int(half_life_days or horizon_value),
        "continuation_decay_score": round(continuation_decay_score, 4),
        "threshold": round(threshold, 6),
    }

