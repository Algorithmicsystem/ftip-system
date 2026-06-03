"""Phase 11.4: Drawdown Intelligence Engine.

Computes full drawdown history, expected recovery time, and max pain index.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, List, Optional

from api import db
from api.assistant.phase3.common import clamp

RECOVERY_RATES = {
    "TRENDING": 0.25,
    "CHOPPY": 0.12,
    "HIGH_VOL": 0.08,
    "RECOVERY": 0.30,
}

PAIN_THRESHOLDS = {
    "conservative": -0.10,
    "moderate": -0.20,
    "aggressive": -0.35,
}


def compute_drawdown_series(returns: List[float]) -> Dict:
    """Compute full drawdown history from return series.

    Returns drawdown statistics including max drawdown, duration, and Calmar ratio.
    """
    if not returns:
        return {
            "current_drawdown_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_start": 0,
            "max_drawdown_end": 0,
            "current_drawdown_duration_days": 0,
            "recovery_pct": 100.0,
            "calmar_ratio": 0.0,
        }

    # Build equity curve starting at 1.0
    equity: List[float] = []
    ev = 1.0
    for r in returns:
        ev *= (1.0 + r)
        equity.append(ev)

    n = len(equity)

    # Running max and drawdown series
    running_max: List[float] = []
    cur_max = equity[0]
    for v in equity:
        cur_max = max(cur_max, v)
        running_max.append(cur_max)

    drawdown = [
        (equity[i] - running_max[i]) / running_max[i] for i in range(n)
    ]

    max_dd = min(drawdown)
    max_dd_end = drawdown.index(max_dd)

    # Peak before trough
    peak_val = max(equity[: max_dd_end + 1])
    max_dd_start = next(
        (i for i, v in enumerate(equity[: max_dd_end + 1]) if v == peak_val),
        0,
    )

    current_dd = drawdown[-1]

    # Drawdown duration: days since last time we were at a peak (drawdown ~ 0)
    last_peak_idx = 0
    for i in range(n):
        if abs(drawdown[i]) < 1e-10:
            last_peak_idx = i
    current_dd_duration = 0 if abs(current_dd) < 1e-10 else (n - 1 - last_peak_idx)

    # Recovery pct: how far from trough back toward peak
    if max_dd < -1e-10:
        recovery_pct = clamp((current_dd - max_dd) / abs(max_dd) * 100.0, 0.0, 100.0)
    else:
        recovery_pct = 100.0

    # CAGR and Calmar ratio
    if equity[-1] > 0:
        cagr = (equity[-1] ** (252.0 / n)) - 1.0
    else:
        cagr = 0.0
    calmar = cagr / abs(max_dd) if max_dd < -1e-10 else 0.0

    return {
        "current_drawdown_pct": round(current_dd, 6),
        "max_drawdown_pct": round(max_dd, 6),
        "max_drawdown_start": max_dd_start,
        "max_drawdown_end": max_dd_end,
        "current_drawdown_duration_days": current_dd_duration,
        "recovery_pct": round(recovery_pct, 2),
        "calmar_ratio": round(calmar, 4),
    }


def compute_expected_recovery_time(
    current_drawdown_pct: float,
    historical_cagr: float,
    regime_label: str,
) -> Dict:
    """Estimate recovery time based on regime-specific recovery rates."""
    recovery_rate = RECOVERY_RATES.get(regime_label.upper(), 0.15)
    # days_to_recover = abs(drawdown) / (annual_rate / 252 days/year)
    days_to_recover = int(round(abs(current_drawdown_pct) / (recovery_rate / 252.0)))

    recovery_date = (
        dt.date.today() + dt.timedelta(days=days_to_recover)
    ).isoformat()

    analog_periods = _find_similar_drawdown_analogs(current_drawdown_pct, regime_label)

    return {
        "estimated_recovery_days": days_to_recover,
        "estimated_recovery_date": recovery_date,
        "recovery_rate_assumption": recovery_rate,
        "regime_label": regime_label,
        "analog_periods": analog_periods,
    }


def _find_similar_drawdown_analogs(
    drawdown_pct: float,
    regime_label: str,
) -> List[Dict[str, Any]]:
    if not db.db_read_enabled():
        return []
    try:
        rows = db.safe_fetchall(
            """
            SELECT analog_id, reference_date, description
              FROM regime_analog_library
             WHERE regime_label = %s
             LIMIT 3
            """,
            (regime_label.lower(),),
        )
        return [
            {
                "analog_id": str(r[0]),
                "reference_date": str(r[1]),
                "description": str(r[2] or ""),
            }
            for r in (rows or [])
        ]
    except Exception:
        return []


def compute_max_pain_index(
    current_drawdown_pct: float,
    position_size_pct: float,
    investor_risk_tolerance: str = "moderate",
) -> Dict:
    """At what drawdown level would a rational investor exit?

    distance_to_max_pain > 0 = room remaining before pain threshold
    distance_to_max_pain < 0 = threshold has been breached
    """
    threshold = PAIN_THRESHOLDS.get(investor_risk_tolerance, -0.20)
    distance_to_pain = current_drawdown_pct - threshold  # positive=room, negative=breached
    pain_triggered = distance_to_pain < 0

    if pain_triggered:
        recommendation = "exit"
    elif abs(distance_to_pain) < 0.05:
        recommendation = "reduce"
    else:
        recommendation = "hold"

    return {
        "max_pain_threshold": threshold,
        "current_drawdown": current_drawdown_pct,
        "distance_to_max_pain": round(distance_to_pain, 4),
        "pain_triggered": pain_triggered,
        "recommendation": recommendation,
    }
