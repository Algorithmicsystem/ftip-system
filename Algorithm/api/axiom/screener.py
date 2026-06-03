"""Session 16: Universe screener — conviction-ranked daily opportunity scan.

Single bulk query over axiom_scores_daily × prosperity_signals_daily.
Per-symbol conviction scoring and Kelly sizing happen in Python so the DB
does one round-trip regardless of universe size.

No DB writes — pure read-only operation over existing tables.
"""
from __future__ import annotations

import json
import math
import datetime as dt
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from api import db

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _engine_score(payload: Dict, engine: str, default: float = 50.0) -> float:
    try:
        eng = (payload.get("engine_scores") or {}).get(engine) or {}
        v = eng.get("score")
        if v is not None:
            return _safe_float(v, default)
    except Exception:
        pass
    return default


def _load_ic_state_bulk(as_of_date: dt.date) -> str:
    if not db.db_read_enabled():
        return "INSUFFICIENT"
    try:
        row = db.safe_fetchone(
            """
            SELECT ic_state FROM signal_ic_daily
            WHERE score_field = 'composite' AND horizon_label = '21d'
              AND as_of_date <= %s
            ORDER BY as_of_date DESC LIMIT 1
            """,
            (as_of_date,),
        )
        return str(row[0]) if row and row[0] else "INSUFFICIENT"
    except Exception:
        return "INSUFFICIENT"


def _load_breadth_state_bulk(as_of_date: dt.date) -> str:
    if not db.db_read_enabled():
        return "NEUTRAL"
    try:
        row = db.safe_fetchone(
            "SELECT breadth_state FROM market_breadth_daily WHERE as_of_date = %s",
            (as_of_date,),
        )
        return str(row[0]) if row and row[0] else "NEUTRAL"
    except Exception:
        return "NEUTRAL"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScreenerResult:
    rank: int
    symbol: str
    as_of_date: str
    signal_label: str
    dau: float
    conviction_score: float
    conviction_tier: str
    suggested_weight: float
    suggested_weight_pct: str
    size_band: str
    deployability_tier: str
    regime_label: str
    breadth_state: str
    ic_state: str
    downside_flags: List[str] = field(default_factory=list)
    active_constraint: str = "kelly"
    factor_composite_score: float = 50.0


# ---------------------------------------------------------------------------
# Core screen function
# ---------------------------------------------------------------------------

def screen_universe(
    as_of_date: dt.date,
    *,
    min_dau: float = 0.0,
    signal_filter: Optional[List[str]] = None,
    min_conviction: float = 0.0,
    max_weight: float = 0.10,
    fractional_kelly: float = 0.5,
    limit: int = 50,
) -> Dict[str, Any]:
    """Return conviction-ranked opportunities for the given date.

    One DB round-trip (bulk join); conviction scoring and Kelly sizing
    run entirely in Python.
    """
    if not db.db_read_enabled():
        return {"status": "db_disabled", "results": [], "total_screened": 0,
                "count": 0, "ic_state": "INSUFFICIENT", "breadth_state": "NEUTRAL",
                "as_of_date": as_of_date.isoformat()}

    ic_state = _load_ic_state_bulk(as_of_date)
    breadth_state = _load_breadth_state_bulk(as_of_date)

    # Build WHERE clause dynamically
    conditions = ["a.as_of_date = %s"]
    params: List[Any] = [as_of_date]

    if min_dau > 0:
        conditions.append("(a.payload->>'deployable_alpha_utility')::numeric >= %s")
        params.append(min_dau)

    where_clause = " AND ".join(conditions)

    # Safety cap at 500 rows before Python filtering
    params.append(500)

    rows = db.safe_fetchall(
        f"""
        SELECT
            a.symbol,
            a.payload,
            COALESCE(p.signal, 'HOLD') AS signal_label
        FROM axiom_scores_daily a
        LEFT JOIN prosperity_signals_daily p
            ON p.symbol = a.symbol
            AND p.as_of = a.as_of_date
            AND p.lookback = 252
        WHERE {where_clause}
        ORDER BY (a.payload->>'deployable_alpha_utility')::numeric DESC NULLS LAST
        LIMIT %s
        """,
        tuple(params),
    )

    total_screened = len(rows) if rows else 0

    # Late imports — no circular dependency risk
    from api.jobs.alerts import compute_conviction_score
    from api.axiom.sizer import compute_kelly_size
    from api.axiom.memo import conviction_tier as get_tier

    signal_set = {s.upper() for s in signal_filter} if signal_filter else None
    candidates: List[ScreenerResult] = []

    for row in (rows or []):
        symbol = str(row[0])
        payload_raw = row[1]
        signal_label = str(row[2] or "HOLD")

        if signal_set and signal_label not in signal_set:
            continue

        # Parse payload
        if isinstance(payload_raw, str):
            try:
                payload: Dict = json.loads(payload_raw)
            except Exception:
                continue
        else:
            payload = payload_raw or {}

        dau = _safe_float(payload.get("deployable_alpha_utility"), 0.0)
        if dau < min_dau:
            continue
        fcs = _safe_float(payload.get("factor_composite_score"), 50.0)

        fragility_score   = _engine_score(payload, "critical_fragility",  50.0)
        liquidity_score   = _engine_score(payload, "liquidity_convexity",  50.0)
        research_score    = _engine_score(payload, "research_integrity",   50.0)
        overall_confidence = _safe_float(payload.get("overall_confidence"), 50.0)
        deployability_tier = str(payload.get("deployability_tier") or "monitor_only")
        regime_label       = str(payload.get("regime_label") or "unknown")

        conviction = compute_conviction_score(
            dau=dau,
            signal_label=signal_label,
            regime_label=regime_label,
            breadth_state=breadth_state,
            ic_state=ic_state,
        )

        if conviction < min_conviction:
            continue

        sizing = compute_kelly_size(
            symbol=symbol,
            as_of_date=as_of_date.isoformat(),
            dau=dau,
            fragility_score=fragility_score,
            liquidity_score=liquidity_score,
            research_score=research_score,
            overall_confidence=overall_confidence,
            deployability_tier=deployability_tier,
            ic_state=ic_state,
            fractional_kelly=fractional_kelly,
            max_weight=max_weight,
        )

        candidates.append(ScreenerResult(
            rank=0,  # assigned after sorting
            symbol=symbol,
            as_of_date=as_of_date.isoformat(),
            signal_label=signal_label,
            dau=round(dau, 2),
            conviction_score=round(conviction, 2),
            conviction_tier=get_tier(conviction),
            suggested_weight=sizing.suggested_weight,
            suggested_weight_pct=f"{sizing.suggested_weight * 100:.2f}%",
            size_band=sizing.size_band,
            deployability_tier=deployability_tier,
            regime_label=regime_label,
            breadth_state=breadth_state,
            ic_state=ic_state,
            downside_flags=sizing.downside_flags,
            active_constraint=sizing.active_constraint,
            factor_composite_score=round(fcs, 2),
        ))

    candidates.sort(key=lambda r: r.conviction_score, reverse=True)
    top = candidates[:limit]
    for i, r in enumerate(top, start=1):
        r.rank = i

    return {
        "status": "ok",
        "as_of_date": as_of_date.isoformat(),
        "total_screened": total_screened,
        "count": len(top),
        "ic_state": ic_state,
        "breadth_state": breadth_state,
        "results": [asdict(r) for r in top],
    }
