"""Phase 11.5: Systemic Risk Index (full implementation).

7-component weighted composite for daily systemic crisis probability.
Scores 0–100; higher = higher systemic risk.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any, Dict, List, Optional

from api import db
from api.assistant.phase3.common import clamp

logger = logging.getLogger(__name__)

_SRI_COMPONENTS = {
    "avg_fragility_score": 0.20,
    "scps_composite": 0.18,
    "bfs_composite": 0.15,
    "correlation_spike_score": 0.15,
    "ic_degradation_score": 0.12,
    "vix_z_score": 0.10,
    "breadth_deterioration": 0.10,
}

_IC_DEG_MAP = {
    "STRONG": 0.0,
    "MODERATE": 25.0,
    "INSUFFICIENT": 50.0,
    "WEAK": 75.0,
    "DEGRADED": 100.0,
}

_BREADTH_MAP = {
    "BULLISH": 10.0,
    "NEUTRAL": 40.0,
    "BEARISH": 80.0,
    "UNKNOWN": 50.0,
}


def _sri_label(sri: float) -> str:
    if sri >= 85.0:
        return "critical"
    if sri >= 70.0:
        return "high_alert"
    if sri >= 50.0:
        return "warning"
    if sri >= 25.0:
        return "elevated"
    return "stable"


def _sri_recommendation(label: str) -> str:
    return {
        "stable": "Maintain current positioning. Systemic risk is within normal parameters.",
        "elevated": "Monitor closely. Consider reducing leverage.",
        "warning": "Reduce risk exposure. Increase cash or hedges.",
        "high_alert": "Significant risk reduction recommended. Scale back positions.",
        "critical": "Maximum defensive positioning. Consider exiting risk assets.",
    }.get(label, "Monitor market conditions.")


def compute_sri(as_of_date: dt.date) -> Dict:
    """Compute the 7-component Systemic Risk Index for a given date.

    Returns 50.0 (neutral) with stable label when DB is unavailable.
    """
    if not db.db_read_enabled():
        components = {k: {"value": 50.0, "weight": w} for k, w in _SRI_COMPONENTS.items()}
        return {
            "sri": 50.0,
            "sri_label": _sri_label(50.0),
            "components": components,
            "primary_driver": "avg_fragility_score",
            "trend": "stable",
            "recommendation": _sri_recommendation(_sri_label(50.0)),
            "as_of_date": as_of_date.isoformat(),
        }

    component_values: Dict[str, float] = {}

    # 1. avg_fragility_score — mean critical_fragility engine score
    component_values["avg_fragility_score"] = _fetch_avg_fragility(as_of_date)

    # 2. scps_composite — mean Sornette crash score
    component_values["scps_composite"] = _fetch_avg_scps(as_of_date)

    # 3. bfs_composite — mean bubble fragility score
    component_values["bfs_composite"] = _fetch_avg_bfs(as_of_date)

    # 4. correlation_spike_score — via signal_pnl_daily volatility proxy
    component_values["correlation_spike_score"] = _fetch_correlation_score(as_of_date)

    # 5. ic_degradation_score — inverted IC quality
    component_values["ic_degradation_score"] = _fetch_ic_degradation(as_of_date)

    # 6. vix_z_score — volatility proxy (normalized to 0–100)
    component_values["vix_z_score"] = _fetch_vix_proxy(as_of_date)

    # 7. breadth_deterioration — inverted breadth score
    component_values["breadth_deterioration"] = _fetch_breadth_deterioration(as_of_date)

    sri_raw = sum(
        component_values[k] * w for k, w in _SRI_COMPONENTS.items()
    )
    sri = round(clamp(sri_raw, 0.0, 100.0), 2)

    # Optional cross-asset divergence adjustment
    try:
        from api.macro.cross_asset_engine import compute_cross_asset_snapshot
        ca = compute_cross_asset_snapshot({}, "UNKNOWN")
        ca_conf = ca.cross_asset_confirmation_score
        if ca_conf < 30:
            sri = round(clamp(sri + 7.5, 0.0, 100.0), 2)
        elif ca_conf > 70:
            sri = round(clamp(sri - 2.5, 0.0, 100.0), 2)
    except Exception:
        pass

    label = _sri_label(sri)

    # Primary driver: component with highest weighted contribution
    weighted = {k: component_values[k] * _SRI_COMPONENTS[k] for k in _SRI_COMPONENTS}
    primary_driver = max(weighted, key=weighted.__getitem__)

    # Trend vs 5 days ago
    trend = _compute_trend(as_of_date, sri)

    # Persist to market_breadth_daily.sri
    _store_sri(as_of_date, sri)

    components = {
        k: {"value": round(component_values[k], 2), "weight": w}
        for k, w in _SRI_COMPONENTS.items()
    }

    return {
        "sri": sri,
        "sri_label": label,
        "components": components,
        "primary_driver": primary_driver,
        "trend": trend,
        "recommendation": _sri_recommendation(label),
        "as_of_date": as_of_date.isoformat(),
    }


def get_sri_history(lookback_days: int = 63) -> List[Dict]:
    """Return daily SRI values for chart visualization."""
    if not db.db_read_enabled():
        return []
    try:
        since = dt.date.today() - dt.timedelta(days=lookback_days)
        rows = db.safe_fetchall(
            """
            SELECT as_of_date, sri
              FROM market_breadth_daily
             WHERE sri IS NOT NULL
               AND as_of_date >= %s
             ORDER BY as_of_date ASC
            """,
            (since,),
        )
        return [
            {"as_of_date": str(r[0]), "sri": float(r[1])}
            for r in (rows or [])
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Component fetchers
# ---------------------------------------------------------------------------

def _fetch_avg_fragility(as_of_date: dt.date) -> float:
    try:
        rows = db.safe_fetchall(
            """
            SELECT (payload->'engine_scores'->'critical_fragility'->>'score')::numeric
              FROM axiom_scores_daily
             WHERE as_of_date = %s
               AND payload IS NOT NULL
             LIMIT 30
            """,
            (as_of_date,),
        )
        scores = [float(r[0]) for r in (rows or []) if r[0] is not None]
        return sum(scores) / len(scores) if scores else 50.0
    except Exception:
        return 50.0


def _fetch_avg_scps(as_of_date: dt.date) -> float:
    try:
        rows = db.safe_fetchall(
            """
            SELECT (payload->'engine_scores'->'critical_fragility'->'components'->>'scps_component')::numeric
              FROM axiom_scores_daily
             WHERE as_of_date = %s
               AND payload IS NOT NULL
             LIMIT 30
            """,
            (as_of_date,),
        )
        scores = [float(r[0]) for r in (rows or []) if r[0] is not None]
        return sum(scores) / len(scores) if scores else 50.0
    except Exception:
        return 50.0


def _fetch_avg_bfs(as_of_date: dt.date) -> float:
    try:
        rows = db.safe_fetchall(
            """
            SELECT (payload->'engine_scores'->'critical_fragility'->'components'->>'bfs_component')::numeric
              FROM axiom_scores_daily
             WHERE as_of_date = %s
               AND payload IS NOT NULL
             LIMIT 30
            """,
            (as_of_date,),
        )
        scores = [float(r[0]) for r in (rows or []) if r[0] is not None]
        return sum(scores) / len(scores) if scores else 50.0
    except Exception:
        return 50.0


def _fetch_correlation_score(as_of_date: dt.date) -> float:
    """Correlation proxy from signal_pnl_daily return dispersion."""
    try:
        since = as_of_date - dt.timedelta(days=21)
        rows = db.safe_fetchall(
            """
            SELECT return_pct
              FROM signal_pnl_daily
             WHERE signal_date >= %s AND signal_date <= %s
               AND return_pct IS NOT NULL AND horizon_days = 5
            """,
            (since, as_of_date),
        )
        if not rows or len(rows) < 10:
            return 50.0
        import statistics
        rets = [float(r[0]) for r in rows]
        std = statistics.stdev(rets) if len(rets) > 1 else 1.0
        # Low dispersion = high correlation = higher systemic risk score
        return clamp(1.0 - std * 10.0, 0.0, 1.0) * 100.0
    except Exception:
        return 50.0


def _fetch_ic_degradation(as_of_date: dt.date) -> float:
    try:
        row = db.safe_fetchone(
            """
            SELECT ic_state
              FROM signal_ic_daily
             WHERE as_of_date <= %s
             ORDER BY as_of_date DESC LIMIT 1
            """,
            (as_of_date,),
        )
        if row:
            return _IC_DEG_MAP.get(str(row[0] or "INSUFFICIENT").upper(), 50.0)
        return 50.0
    except Exception:
        return 50.0


def _fetch_vix_proxy(as_of_date: dt.date) -> float:
    """Volatility proxy from market_breadth_daily or return dispersion."""
    try:
        row = db.safe_fetchone(
            """
            SELECT (meta->>'vix_proxy')::numeric
              FROM market_breadth_daily
             WHERE as_of_date <= %s
               AND meta IS NOT NULL
             ORDER BY as_of_date DESC LIMIT 1
            """,
            (as_of_date,),
        )
        if row and row[0] is not None:
            return clamp(float(row[0]), 0.0, 100.0)
        return 50.0
    except Exception:
        return 50.0


def _fetch_breadth_deterioration(as_of_date: dt.date) -> float:
    """Inverted breadth score: high deterioration = high risk."""
    try:
        row = db.safe_fetchone(
            "SELECT breadth_state FROM market_breadth_daily WHERE as_of_date = %s",
            (as_of_date,),
        )
        if row:
            return _BREADTH_MAP.get(str(row[0] or "UNKNOWN").upper(), 50.0)
        return 50.0
    except Exception:
        return 50.0


def _compute_trend(as_of_date: dt.date, current_sri: float) -> str:
    """Compare current SRI with SRI from 5 days ago."""
    try:
        since = as_of_date - dt.timedelta(days=7)
        row = db.safe_fetchone(
            """
            SELECT sri
              FROM market_breadth_daily
             WHERE as_of_date >= %s AND as_of_date < %s
               AND sri IS NOT NULL
             ORDER BY as_of_date DESC LIMIT 1
            """,
            (since, as_of_date),
        )
        if row and row[0] is not None:
            prior = float(row[0])
            if current_sri > prior + 2.0:
                return "rising"
            if current_sri < prior - 2.0:
                return "falling"
        return "stable"
    except Exception:
        return "stable"


def _store_sri(as_of_date: dt.date, sri: float) -> None:
    try:
        db.safe_execute(
            """
            UPDATE market_breadth_daily
               SET sri = %s
             WHERE as_of_date = %s
            """,
            (sri, as_of_date),
        )
    except Exception as exc:
        logger.debug("sri_store_failed date=%s err=%s", as_of_date, exc)
