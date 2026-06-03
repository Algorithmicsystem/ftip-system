"""Phase 10.4: Intraday IC Tracking.

Tracks IC at the intraday level to find optimal signal timing throughout the day.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

from api import db, security

router = APIRouter(
    prefix="/jobs/ic",
    tags=["ic"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)
logger = logging.getLogger(__name__)

_MIN_IC_SAMPLE = 10
_INTRADAY_HOURS = [10, 12, 14, 16]
_IC_STRONG = 0.05
_IC_MODERATE = 0.02


def _ic_state_from_value(ic_value: Optional[float], sample_count: int) -> str:
    if sample_count < _MIN_IC_SAMPLE or ic_value is None:
        return "INSUFFICIENT"
    if ic_value >= _IC_STRONG:
        return "STRONG"
    if ic_value >= _IC_MODERATE:
        return "MODERATE"
    if ic_value < 0:
        return "DEGRADED"
    return "WEAK"


def compute_intraday_ic(session_date: dt.date, update_hour: int) -> Dict[str, Any]:
    """Compute IC between prior-day DAU scores and current session returns up to update_hour.

    Returns ic_state = "INSUFFICIENT" when sample_count < 10.
    """
    if not db.db_read_enabled():
        return {
            "session_date": session_date.isoformat(),
            "update_hour": update_hour,
            "ic_value": None,
            "sample_count": 0,
            "ic_state": "INSUFFICIENT",
        }

    prior_date = session_date - dt.timedelta(days=1)

    try:
        rows = db.safe_fetchall(
            """
            SELECT a.symbol,
                   (a.payload->>'deployable_alpha_utility')::numeric AS dau,
                   p.return_pct
              FROM axiom_scores_daily a
              JOIN signal_pnl_daily p
                ON p.symbol = a.symbol
               AND p.signal_date = a.as_of_date
               AND p.horizon_days = 1
             WHERE a.as_of_date = %s
               AND p.return_pct IS NOT NULL
            """,
            (prior_date,),
        )
    except Exception as exc:
        logger.warning("intraday_ic.query_failed date=%s hour=%d error=%s", session_date, update_hour, exc)
        return {
            "session_date": session_date.isoformat(),
            "update_hour": update_hour,
            "ic_value": None,
            "sample_count": 0,
            "ic_state": "INSUFFICIENT",
        }

    if not rows or len(rows) < _MIN_IC_SAMPLE:
        return {
            "session_date": session_date.isoformat(),
            "update_hour": update_hour,
            "ic_value": None,
            "sample_count": len(rows) if rows else 0,
            "ic_state": "INSUFFICIENT",
        }

    daus = [float(r[1] or 50.0) for r in rows]
    returns = [float(r[2]) for r in rows]
    sample_count = len(daus)

    # Pearson correlation as IC
    try:
        import statistics
        mean_d = statistics.mean(daus)
        mean_r = statistics.mean(returns)
        num = sum((d - mean_d) * (r - mean_r) for d, r in zip(daus, returns))
        den_d = sum((d - mean_d) ** 2 for d in daus) ** 0.5
        den_r = sum((r - mean_r) ** 2 for r in returns) ** 0.5
        ic_value = round(num / (den_d * den_r + 1e-12), 6) if (den_d * den_r) > 1e-12 else None
    except Exception:
        ic_value = None

    ic_state = _ic_state_from_value(ic_value, sample_count)
    return {
        "session_date": session_date.isoformat(),
        "update_hour": update_hour,
        "ic_value": ic_value,
        "sample_count": sample_count,
        "ic_state": ic_state,
    }


def compute_time_of_day_ic_calendar(lookback_days: int = 63) -> Dict[str, Any]:
    """Compute average IC per hour-of-day over the lookback period.

    Returns a dict with keys 10, 12, 14, 16 (as strings) and best_hour.
    """
    if not db.db_read_enabled():
        return {
            "10": {"avg_ic": None, "sample_count": 0},
            "12": {"avg_ic": None, "sample_count": 0},
            "14": {"avg_ic": None, "sample_count": 0},
            "16": {"avg_ic": None, "sample_count": 0},
            "best_hour": "16",
        }

    since = dt.date.today() - dt.timedelta(days=lookback_days)

    try:
        rows = db.safe_fetchall(
            """
            SELECT update_hour, ic_value
              FROM intraday_ic_daily
             WHERE session_date >= %s
               AND ic_value IS NOT NULL
            """,
            (since,),
        )
    except Exception:
        rows = []

    hour_data: Dict[int, List[float]] = {h: [] for h in _INTRADAY_HOURS}
    for row in (rows or []):
        h = int(row[0])
        v = float(row[1])
        if h in hour_data:
            hour_data[h].append(v)

    result: Dict[str, Any] = {}
    best_hour = "16"
    best_ic = float("-inf")

    for h in _INTRADAY_HOURS:
        vals = hour_data[h]
        if vals:
            avg = sum(vals) / len(vals)
        else:
            avg = None
        result[str(h)] = {"avg_ic": round(avg, 6) if avg is not None else None, "sample_count": len(vals)}
        if avg is not None and avg > best_ic:
            best_ic = avg
            best_hour = str(h)

    result["best_hour"] = best_hour
    return result


def store_intraday_ic(result: Dict[str, Any]) -> None:
    if not db.db_read_enabled():
        return
    try:
        db.safe_execute(
            """
            INSERT INTO intraday_ic_daily (session_date, update_hour, ic_value, sample_count, ic_state)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (session_date, update_hour) DO UPDATE
               SET ic_value = EXCLUDED.ic_value,
                   sample_count = EXCLUDED.sample_count,
                   ic_state = EXCLUDED.ic_state
            """,
            (
                result["session_date"],
                result["update_hour"],
                result["ic_value"],
                result["sample_count"],
                result["ic_state"],
            ),
        )
    except Exception as exc:
        logger.warning("intraday_ic.store_failed error=%s", exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/intraday/calendar")
def get_intraday_ic_calendar(lookback_days: int = Query(default=63)) -> Dict[str, Any]:
    return compute_time_of_day_ic_calendar(lookback_days)


@router.post("/intraday/compute")
def compute_intraday_ic_endpoint(
    session_date: Optional[str] = Query(default=None),
    update_hour: int = Query(default=16),
) -> Dict[str, Any]:
    aod = dt.date.fromisoformat(session_date) if session_date else dt.date.today()
    result = compute_intraday_ic(aod, update_hour)
    store_intraday_ic(result)
    return result
