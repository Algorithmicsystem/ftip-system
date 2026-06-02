"""Phase 3: PE and Corporate Intelligence Module.

Provides:
  store_entity_financials  — upsert periodic financials for a portfolio company
  compute_entity_health    — health score (0–100) with components
  compute_exit_timing      — exit readiness and optimal window analysis
  get_portfolio_overview   — all entities for an org with health scores
  get_portfolio_stress_alerts — distressed entities needing attention
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)

# Thresholds
_LEVERAGE_SAFE = 3.0   # debt/EBITDA below this is healthy
_LEVERAGE_HIGH = 6.0   # above this triggers stress alert
_HEALTH_ALERT  = 40.0  # entity health score below this → alert


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def store_entity_financials(
    entity_id: str,
    period_end: dt.date,
    financials: Dict[str, Any],
    period_type: str = "quarterly",
) -> bool:
    """Upsert one period of financials for a portfolio company."""
    if not db.db_write_enabled():
        return False
    try:
        db.safe_execute(
            """
            INSERT INTO private_entity_financials
                (entity_id, period_end, period_type,
                 revenue, ebitda, net_income, total_debt, cash,
                 capex, free_cash_flow, headcount, arr)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (entity_id, period_end, period_type)
            DO UPDATE SET
                revenue        = EXCLUDED.revenue,
                ebitda         = EXCLUDED.ebitda,
                net_income     = EXCLUDED.net_income,
                total_debt     = EXCLUDED.total_debt,
                cash           = EXCLUDED.cash,
                capex          = EXCLUDED.capex,
                free_cash_flow = EXCLUDED.free_cash_flow,
                headcount      = EXCLUDED.headcount,
                arr            = EXCLUDED.arr,
                reported_at    = now()
            """,
            (
                entity_id,
                period_end,
                period_type,
                financials.get("revenue"),
                financials.get("ebitda"),
                financials.get("net_income"),
                financials.get("total_debt"),
                financials.get("cash"),
                financials.get("capex"),
                financials.get("free_cash_flow"),
                financials.get("headcount"),
                financials.get("arr"),
            ),
        )
        return True
    except Exception as exc:
        logger.warning("pe.store_financials_failed entity=%s error=%s", entity_id, exc)
        return False


# ---------------------------------------------------------------------------
# Health Score
# ---------------------------------------------------------------------------

def _load_recent_periods(entity_id: str, n: int = 5) -> List[Dict[str, Any]]:
    """Load the n most recent financial periods for an entity."""
    if not db.db_read_enabled():
        return []
    try:
        rows = db.safe_fetchall(
            """
            SELECT period_end, revenue, ebitda, net_income,
                   total_debt, cash, free_cash_flow
            FROM private_entity_financials
            WHERE entity_id = %s
            ORDER BY period_end DESC
            LIMIT %s
            """,
            (entity_id, n),
        )
    except Exception as exc:
        logger.warning("pe.load_periods_failed entity=%s error=%s", entity_id, exc)
        return []

    return [
        {
            "period_end": r[0],
            "revenue": float(r[1]) if r[1] is not None else None,
            "ebitda": float(r[2]) if r[2] is not None else None,
            "net_income": float(r[3]) if r[3] is not None else None,
            "total_debt": float(r[4]) if r[4] is not None else None,
            "cash": float(r[5]) if r[5] is not None else None,
            "free_cash_flow": float(r[6]) if r[6] is not None else None,
        }
        for r in rows
    ]


def compute_entity_health(entity_id: str) -> Dict[str, Any]:
    """Compute a 0–100 health score for a PE portfolio company.

    Components:
    - ebitda_margin_score  (0–100): EBITDA / revenue, mapped 0→50%+ margin
    - revenue_growth_score (0–100): YoY revenue growth, mapped -20%→+30%
    - leverage_score       (0–100): debt/EBITDA, lower is better
    - fcf_conversion_score (0–100): FCF / EBITDA, higher is better
    - liquidity_score      (0–100): cash / (|EBITDA| × 0.25), i.e. 1yr coverage
    """
    periods = _load_recent_periods(entity_id, n=8)

    if not periods:
        return {
            "status": "no_data",
            "entity_id": entity_id,
            "health_score": None,
            "components": {},
        }

    latest = periods[0]
    components: Dict[str, Optional[float]] = {}

    # EBITDA margin
    ebitda_margin_score: Optional[float] = None
    if latest["revenue"] and latest["revenue"] > 0 and latest["ebitda"] is not None:
        margin = latest["ebitda"] / latest["revenue"]
        ebitda_margin_score = _clamp((margin / 0.30) * 100, 0, 100)
    components["ebitda_margin_score"] = ebitda_margin_score

    # Revenue growth YoY (compare most recent to 4 periods ago if quarterly)
    revenue_growth_score: Optional[float] = None
    if len(periods) >= 4 and latest["revenue"] and periods[3]["revenue"] and periods[3]["revenue"] > 0:
        yoy_growth = (latest["revenue"] - periods[3]["revenue"]) / periods[3]["revenue"]
        # Map: -0.20 → 0, 0.0 → 40, +0.30 → 100
        revenue_growth_score = _clamp(((yoy_growth + 0.20) / 0.50) * 100, 0, 100)
    components["revenue_growth_score"] = revenue_growth_score

    # Leverage: debt / EBITDA
    leverage_score: Optional[float] = None
    if latest["total_debt"] is not None and latest["ebitda"] and latest["ebitda"] > 0:
        leverage = latest["total_debt"] / latest["ebitda"]
        # Map: 0x → 100, 3x → 60, 6x → 0
        leverage_score = _clamp(100.0 - (leverage / _LEVERAGE_HIGH) * 100, 0, 100)
    components["leverage_score"] = leverage_score

    # FCF conversion: FCF / EBITDA
    fcf_score: Optional[float] = None
    if latest["free_cash_flow"] is not None and latest["ebitda"] and latest["ebitda"] > 0:
        fcf_conversion = latest["free_cash_flow"] / latest["ebitda"]
        fcf_score = _clamp(fcf_conversion * 100, 0, 100)
    components["fcf_conversion_score"] = fcf_score

    # Liquidity: cash / (EBITDA × 0.25) = quarters of EBITDA coverage
    liquidity_score: Optional[float] = None
    if latest["cash"] is not None and latest["ebitda"] and abs(latest["ebitda"]) > 0:
        quarterly_burn = abs(latest["ebitda"]) * 0.25
        coverage = latest["cash"] / quarterly_burn
        liquidity_score = _clamp((coverage / 4.0) * 100, 0, 100)
    components["liquidity_score"] = liquidity_score

    # Weighted average of available components
    weights = {
        "ebitda_margin_score": 0.30,
        "revenue_growth_score": 0.25,
        "leverage_score": 0.25,
        "fcf_conversion_score": 0.12,
        "liquidity_score": 0.08,
    }
    total_w = 0.0
    total_score = 0.0
    for k, w in weights.items():
        v = components.get(k)
        if v is not None:
            total_w += w
            total_score += v * w

    health_score = round(total_score / total_w, 1) if total_w > 0 else None

    return {
        "status": "ok",
        "entity_id": entity_id,
        "health_score": health_score,
        "latest_period": latest["period_end"].isoformat() if hasattr(latest["period_end"], "isoformat") else str(latest["period_end"]),
        "components": {k: round(v, 1) if v is not None else None for k, v in components.items()},
        "alert": health_score is not None and health_score < _HEALTH_ALERT,
    }


# ---------------------------------------------------------------------------
# Exit Timing
# ---------------------------------------------------------------------------

def compute_exit_timing(entity_id: str) -> Dict[str, Any]:
    """Assess exit readiness and optimal exit window for a portfolio company.

    Factors:
    - Months held vs target hold period
    - EBITDA trajectory (improving / stable / declining)
    - Current implied multiple vs target multiple
    - Leverage headroom
    """
    if not db.db_read_enabled():
        return {"status": "db_disabled", "entity_id": entity_id}

    # Load entity metadata
    try:
        meta_row = db.safe_fetchone(
            """
            SELECT entry_date, entry_ev, target_exit_date, target_exit_multiple,
                   entity_name, sector
            FROM private_entities
            WHERE entity_id = %s AND is_active = TRUE
            """,
            (entity_id,),
        )
    except Exception as exc:
        logger.warning("pe.exit_timing_meta_failed entity=%s error=%s", entity_id, exc)
        meta_row = None

    if not meta_row:
        return {"status": "not_found", "entity_id": entity_id}

    entry_date, entry_ev, target_exit_date, target_multiple, name, sector = meta_row
    today = dt.date.today()

    months_held: Optional[float] = None
    if entry_date:
        months_held = round((today - entry_date).days / 30.44, 1)

    months_to_target: Optional[float] = None
    if target_exit_date:
        months_to_target = round((target_exit_date - today).days / 30.44, 1)

    # Load financials for EBITDA trajectory
    periods = _load_recent_periods(entity_id, n=6)
    ebitda_trend: Optional[str] = None
    current_ebitda: Optional[float] = None

    if len(periods) >= 2:
        current_ebitda = periods[0]["ebitda"]
        prev_ebitda = periods[1]["ebitda"]
        if current_ebitda is not None and prev_ebitda is not None and prev_ebitda != 0:
            delta = (current_ebitda - prev_ebitda) / abs(prev_ebitda)
            if delta > 0.05:
                ebitda_trend = "improving"
            elif delta < -0.05:
                ebitda_trend = "declining"
            else:
                ebitda_trend = "stable"

    # Exit readiness score (0–100)
    readiness_components: Dict[str, Optional[float]] = {}

    # Holding period: prefer 24–48 months as sweet spot
    if months_held is not None:
        if months_held < 12:
            hp_score = months_held / 12 * 30  # too early
        elif months_held <= 48:
            hp_score = 60.0 + (months_held - 12) / 36 * 40
        else:
            hp_score = max(60.0, 100.0 - (months_held - 48) * 1.5)
        readiness_components["holding_period_score"] = _clamp(hp_score, 0, 100)
    else:
        readiness_components["holding_period_score"] = None

    # EBITDA trend
    trend_scores = {"improving": 80.0, "stable": 55.0, "declining": 25.0}
    readiness_components["ebitda_trend_score"] = trend_scores.get(ebitda_trend) if ebitda_trend else None

    # Leverage headroom (same as health leverage)
    lev_score = None
    if periods and periods[0]["total_debt"] is not None and periods[0]["ebitda"] and periods[0]["ebitda"] > 0:
        leverage = periods[0]["total_debt"] / periods[0]["ebitda"]
        lev_score = _clamp(100.0 - (leverage / _LEVERAGE_HIGH) * 100, 0, 100)
    readiness_components["leverage_headroom_score"] = lev_score

    total_w = 0.0
    total_s = 0.0
    w_map = {"holding_period_score": 0.40, "ebitda_trend_score": 0.35, "leverage_headroom_score": 0.25}
    for k, w in w_map.items():
        v = readiness_components.get(k)
        if v is not None:
            total_w += w
            total_s += v * w

    exit_readiness_score = round(total_s / total_w, 1) if total_w > 0 else None

    recommendation: str
    if exit_readiness_score is None:
        recommendation = "insufficient_data"
    elif exit_readiness_score >= 70:
        recommendation = "exit_ready"
    elif exit_readiness_score >= 50:
        recommendation = "monitor_and_prepare"
    else:
        recommendation = "hold_and_improve"

    return {
        "status": "ok",
        "entity_id": entity_id,
        "entity_name": name,
        "sector": sector,
        "months_held": months_held,
        "months_to_target": months_to_target,
        "ebitda_trend": ebitda_trend,
        "current_ebitda": current_ebitda,
        "exit_readiness_score": exit_readiness_score,
        "recommendation": recommendation,
        "components": {k: round(v, 1) if v is not None else None for k, v in readiness_components.items()},
    }


# ---------------------------------------------------------------------------
# Portfolio Views
# ---------------------------------------------------------------------------

def get_portfolio_overview(org_id: str) -> Dict[str, Any]:
    """Return all active entities for an org with their health scores."""
    if not db.db_read_enabled():
        return {"status": "db_disabled", "org_id": org_id, "entities": []}

    try:
        rows = db.safe_fetchall(
            """
            SELECT entity_id, entity_name, sector, entry_date,
                   target_exit_date, public_peer_symbol
            FROM private_entities
            WHERE org_id = %s AND is_active = TRUE
            ORDER BY entry_date DESC NULLS LAST
            """,
            (org_id,),
        )
    except Exception as exc:
        logger.warning("pe.portfolio_overview_failed org=%s error=%s", org_id, exc)
        return {"status": "error", "org_id": org_id, "entities": []}

    entities = []
    for row in rows:
        entity_id, name, sector, entry_date, exit_date, peer = row
        health = compute_entity_health(entity_id)
        entities.append({
            "entity_id": entity_id,
            "entity_name": name,
            "sector": sector,
            "entry_date": entry_date.isoformat() if entry_date else None,
            "target_exit_date": exit_date.isoformat() if exit_date else None,
            "public_peer_symbol": peer,
            "health_score": health.get("health_score"),
            "alert": health.get("alert", False),
        })

    healthy = [e for e in entities if e["health_score"] is not None]
    avg_health = (
        round(sum(e["health_score"] for e in healthy) / len(healthy), 1)
        if healthy else None
    )

    return {
        "status": "ok",
        "org_id": org_id,
        "entity_count": len(entities),
        "avg_health_score": avg_health,
        "alert_count": sum(1 for e in entities if e["alert"]),
        "entities": entities,
    }


def get_portfolio_stress_alerts(org_id: str) -> Dict[str, Any]:
    """Return entities in distress: health < 40 or leverage > 6x."""
    overview = get_portfolio_overview(org_id)
    alerts = [e for e in overview.get("entities", []) if e.get("alert")]
    return {
        "status": overview["status"],
        "org_id": org_id,
        "alert_count": len(alerts),
        "alerts": alerts,
    }
