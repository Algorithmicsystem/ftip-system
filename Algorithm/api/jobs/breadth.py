"""Daily market breadth computation job.

Aggregates per-symbol signals from prosperity_signals_daily into
market-wide breadth metrics and stores them in market_breadth_daily.
The stored row is later injected into the depth layer as a fallback
when per-symbol breadth features are absent.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import statistics
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from api import db, security

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)

logger = logging.getLogger(__name__)

_BREADTH_OVERLAY_KEYS = (
    "breadth_confirmation_score",
    "participation_breadth_score",
    "breadth_thrust_proxy",
    "cross_sectional_dispersion_proxy",
    "leadership_concentration_score",
    "internal_market_divergence_score",
    "leader_strength_score",
    "laggard_pressure_score",
    "narrow_leadership_warning",
    "broad_participation_confirmation",
    "breadth_state",
)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_market_breadth(as_of_date: dt.date, lookback: int = 252) -> Dict[str, Any]:
    rows = db.safe_fetchall(
        """
        SELECT score, signal
        FROM prosperity_signals_daily
        WHERE as_of = %s AND lookback = %s
        """,
        (as_of_date, lookback),
    )
    scores = [float(r[0]) for r in rows if r[0] is not None]
    signals = [str(r[1]) for r in rows if r[1] is not None]
    n = len(scores)
    if n == 0:
        return {}

    buy_count = sum(1 for s in signals if s == "BUY")
    sell_count = sum(1 for s in signals if s == "SELL")
    positive_count = sum(1 for s in scores if s > 0)

    participation_breadth_score = round(positive_count / n * 100, 2)
    breadth_confirmation_score = round(buy_count / n * 100, 2)
    # net thrust: 0=all-sell, 50=neutral, 100=all-buy
    breadth_thrust_proxy = round(((buy_count - sell_count) / n * 0.5 + 0.5) * 100, 2)
    cross_sectional_dispersion_proxy = round(statistics.stdev(scores) * 100, 2) if n > 1 else 0.0

    sorted_scores = sorted(scores, reverse=True)
    q_size = max(1, n // 4)
    top_q = sorted_scores[:q_size]
    bottom_q = sorted_scores[-q_size:]
    top_mean = sum(top_q) / len(top_q)
    bottom_mean = sum(bottom_q) / len(bottom_q)

    leader_strength_score = round(max(0.0, min(top_mean, 1.0)) * 100, 2)
    laggard_pressure_score = round((1.0 - max(0.0, min(bottom_mean, 1.0))) * 100, 2)
    leadership_concentration_score = round((top_mean - bottom_mean) * 50 + 50, 2)
    internal_market_divergence_score = round((top_mean - bottom_mean) * 100, 2)

    narrow_leadership_warning = (
        leadership_concentration_score > 75 and participation_breadth_score < 35
    )
    broad_participation_confirmation = participation_breadth_score > 65

    if breadth_confirmation_score >= 65 and participation_breadth_score >= 60:
        breadth_state = "EXPANDING"
    elif breadth_confirmation_score <= 30 or participation_breadth_score <= 30:
        breadth_state = "CONTRACTING"
    elif cross_sectional_dispersion_proxy > 60:
        breadth_state = "STRESSED"
    else:
        breadth_state = "NEUTRAL"

    return {
        "breadth_confirmation_score": breadth_confirmation_score,
        "participation_breadth_score": participation_breadth_score,
        "breadth_thrust_proxy": breadth_thrust_proxy,
        "cross_sectional_dispersion_proxy": cross_sectional_dispersion_proxy,
        "leadership_concentration_score": leadership_concentration_score,
        "internal_market_divergence_score": internal_market_divergence_score,
        "leader_strength_score": leader_strength_score,
        "laggard_pressure_score": laggard_pressure_score,
        "narrow_leadership_warning": narrow_leadership_warning,
        "broad_participation_confirmation": broad_participation_confirmation,
        "breadth_state": breadth_state,
        "universe_size": n,
    }


# ---------------------------------------------------------------------------
# DB store / load
# ---------------------------------------------------------------------------

def store_market_breadth(as_of_date: dt.date, payload: Dict[str, Any]) -> bool:
    if not payload:
        return False
    try:
        db.safe_execute(
            """
            INSERT INTO market_breadth_daily (
                as_of_date,
                breadth_confirmation_score, participation_breadth_score,
                breadth_thrust_proxy, cross_sectional_dispersion_proxy,
                leadership_concentration_score, internal_market_divergence_score,
                leader_strength_score, laggard_pressure_score,
                narrow_leadership_warning, broad_participation_confirmation,
                breadth_state, universe_size, meta
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
            ON CONFLICT (as_of_date) DO UPDATE SET
                breadth_confirmation_score       = EXCLUDED.breadth_confirmation_score,
                participation_breadth_score      = EXCLUDED.participation_breadth_score,
                breadth_thrust_proxy             = EXCLUDED.breadth_thrust_proxy,
                cross_sectional_dispersion_proxy = EXCLUDED.cross_sectional_dispersion_proxy,
                leadership_concentration_score   = EXCLUDED.leadership_concentration_score,
                internal_market_divergence_score = EXCLUDED.internal_market_divergence_score,
                leader_strength_score            = EXCLUDED.leader_strength_score,
                laggard_pressure_score           = EXCLUDED.laggard_pressure_score,
                narrow_leadership_warning        = EXCLUDED.narrow_leadership_warning,
                broad_participation_confirmation = EXCLUDED.broad_participation_confirmation,
                breadth_state                    = EXCLUDED.breadth_state,
                universe_size                    = EXCLUDED.universe_size,
                meta                             = EXCLUDED.meta,
                updated_at                       = now()
            """,
            (
                as_of_date,
                payload.get("breadth_confirmation_score"),
                payload.get("participation_breadth_score"),
                payload.get("breadth_thrust_proxy"),
                payload.get("cross_sectional_dispersion_proxy"),
                payload.get("leadership_concentration_score"),
                payload.get("internal_market_divergence_score"),
                payload.get("leader_strength_score"),
                payload.get("laggard_pressure_score"),
                payload.get("narrow_leadership_warning"),
                payload.get("broad_participation_confirmation"),
                payload.get("breadth_state"),
                payload.get("universe_size"),
                json.dumps({}),
            ),
        )
        return True
    except Exception:
        logger.warning("breadth_job.store_failed", extra={"as_of_date": str(as_of_date)})
        return False


def load_market_breadth(as_of_date: Any) -> Optional[Dict[str, Any]]:
    """Return stored market breadth for a date, or None if unavailable."""
    if not db.db_read_enabled():
        return None
    try:
        date = (
            as_of_date
            if isinstance(as_of_date, dt.date)
            else dt.date.fromisoformat(str(as_of_date)[:10])
        )
        row = db.safe_fetchone(
            """
            SELECT breadth_confirmation_score, participation_breadth_score,
                   breadth_thrust_proxy, cross_sectional_dispersion_proxy,
                   leadership_concentration_score, internal_market_divergence_score,
                   leader_strength_score, laggard_pressure_score,
                   narrow_leadership_warning, broad_participation_confirmation,
                   breadth_state, universe_size
            FROM market_breadth_daily
            WHERE as_of_date = %s
            """,
            (date,),
        )
        if not row:
            return None
        keys = (
            "breadth_confirmation_score", "participation_breadth_score",
            "breadth_thrust_proxy", "cross_sectional_dispersion_proxy",
            "leadership_concentration_score", "internal_market_divergence_score",
            "leader_strength_score", "laggard_pressure_score",
            "narrow_leadership_warning", "broad_participation_confirmation",
            "breadth_state", "universe_size",
        )
        return {k: v for k, v in zip(keys, row) if v is not None}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

class BreadthSnapshotRequest(BaseModel):
    as_of_date: Optional[str] = None
    lookback: int = 252


@router.post("/breadth/daily-snapshot")
async def breadth_daily_snapshot(req: BreadthSnapshotRequest, request: Request):
    if not db.db_read_enabled():
        return {"status": "skipped", "reason": "db_read_disabled"}

    as_of = (
        dt.date.fromisoformat(req.as_of_date)
        if req.as_of_date
        else dt.date.today() - dt.timedelta(days=1)
    )
    payload = compute_market_breadth(as_of, req.lookback)
    stored = False
    if payload and db.db_write_enabled():
        stored = store_market_breadth(as_of, payload)

    return {
        "status": "ok" if payload else "no_data",
        "as_of_date": as_of.isoformat(),
        "lookback": req.lookback,
        "stored": stored,
        "result": payload,
    }
