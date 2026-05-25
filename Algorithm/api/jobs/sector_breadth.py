"""Per-sector breadth computation job.

Aggregates daily signals by sector (via market_symbols.sector join) to
produce per-sector breadth metrics and a rotation heatmap ranking.
Stores in sector_breadth_daily; one row per (as_of_date, sector).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import statistics
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from api import db, security

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_sector_breadth(as_of_date: dt.date, lookback: int = 252) -> List[Dict[str, Any]]:
    """Return per-sector breadth metrics for the given date.

    Joins prosperity_signals_daily (column: as_of) with market_symbols
    (column: sector) to group signals by sector.
    """
    rows = db.safe_fetchall(
        """
        SELECT ms.sector, psd.score, psd.signal
        FROM prosperity_signals_daily psd
        JOIN market_symbols ms ON ms.symbol = psd.symbol
        WHERE psd.as_of = %s
          AND psd.lookback = %s
          AND ms.sector IS NOT NULL
        """,
        (as_of_date, lookback),
    )
    if not rows:
        return []

    sector_data: Dict[str, List] = {}
    for sector, score, signal in rows:
        sector_data.setdefault(sector, []).append(
            (float(score) if score is not None else 0.0, str(signal) if signal else "HOLD")
        )

    results: List[Dict[str, Any]] = []
    for sector, entries in sector_data.items():
        scores = [e[0] for e in entries]
        signals = [e[1] for e in entries]
        n = len(scores)

        buy_count = sum(1 for s in signals if s == "BUY")
        sell_count = sum(1 for s in signals if s == "SELL")
        hold_count = sum(1 for s in signals if s == "HOLD")
        positive_count = sum(1 for s in scores if s > 0)

        avg_score = round(sum(scores) / n, 4)
        breadth_confirmation_score = round(buy_count / n * 100, 2)
        participation_breadth_score = round(positive_count / n * 100, 2)

        dispersion = round(statistics.stdev(scores) * 100, 2) if n > 1 else 0.0

        if breadth_confirmation_score >= 65 and participation_breadth_score >= 60:
            breadth_state = "EXPANDING"
        elif breadth_confirmation_score <= 30 or participation_breadth_score <= 30:
            breadth_state = "CONTRACTING"
        elif dispersion > 60:
            breadth_state = "STRESSED"
        else:
            breadth_state = "NEUTRAL"

        results.append({
            "sector": sector,
            "symbol_count": n,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "hold_count": hold_count,
            "avg_score": avg_score,
            "breadth_confirmation_score": breadth_confirmation_score,
            "participation_breadth_score": participation_breadth_score,
            "breadth_state": breadth_state,
        })

    results.sort(key=lambda r: r["breadth_confirmation_score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rotation_rank"] = i

    return results


# ---------------------------------------------------------------------------
# DB store / load
# ---------------------------------------------------------------------------

def store_sector_breadth(as_of_date: dt.date, sectors: List[Dict[str, Any]]) -> int:
    count = 0
    for r in sectors:
        try:
            db.safe_execute(
                """
                INSERT INTO sector_breadth_daily (
                    as_of_date, sector, symbol_count,
                    buy_count, sell_count, hold_count,
                    avg_score, breadth_confirmation_score,
                    participation_breadth_score, breadth_state,
                    rotation_rank, meta
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (as_of_date, sector) DO UPDATE SET
                    symbol_count                = EXCLUDED.symbol_count,
                    buy_count                   = EXCLUDED.buy_count,
                    sell_count                  = EXCLUDED.sell_count,
                    hold_count                  = EXCLUDED.hold_count,
                    avg_score                   = EXCLUDED.avg_score,
                    breadth_confirmation_score  = EXCLUDED.breadth_confirmation_score,
                    participation_breadth_score = EXCLUDED.participation_breadth_score,
                    breadth_state               = EXCLUDED.breadth_state,
                    rotation_rank               = EXCLUDED.rotation_rank,
                    meta                        = EXCLUDED.meta,
                    updated_at                  = now()
                """,
                (
                    as_of_date,
                    r["sector"],
                    r["symbol_count"],
                    r["buy_count"],
                    r["sell_count"],
                    r["hold_count"],
                    r.get("avg_score"),
                    r["breadth_confirmation_score"],
                    r["participation_breadth_score"],
                    r["breadth_state"],
                    r.get("rotation_rank"),
                    json.dumps({}),
                ),
            )
            count += 1
        except Exception:
            logger.warning(
                "sector_breadth.store_row_failed",
                extra={"sector": r.get("sector"), "as_of_date": str(as_of_date)},
            )
    return count


def load_sector_breadth_latest() -> Optional[Dict[str, Any]]:
    """Return the most recently stored sector breadth snapshot."""
    if not db.db_read_enabled():
        return None
    try:
        date_row = db.safe_fetchone(
            "SELECT MAX(as_of_date) FROM sector_breadth_daily",
            (),
        )
        if not date_row or not date_row[0]:
            return None
        latest_date = date_row[0]

        rows = db.safe_fetchall(
            """
            SELECT sector, symbol_count, buy_count, sell_count, hold_count,
                   avg_score, breadth_confirmation_score, participation_breadth_score,
                   breadth_state, rotation_rank
            FROM sector_breadth_daily
            WHERE as_of_date = %s
            ORDER BY rotation_rank ASC NULLS LAST
            """,
            (latest_date,),
        )
        sectors = []
        for row in rows:
            sectors.append({
                "sector": row[0],
                "symbol_count": row[1],
                "buy_count": row[2],
                "sell_count": row[3],
                "hold_count": row[4],
                "avg_score": float(row[5]) if row[5] is not None else None,
                "breadth_confirmation_score": float(row[6]) if row[6] is not None else None,
                "participation_breadth_score": float(row[7]) if row[7] is not None else None,
                "breadth_state": row[8],
                "rotation_rank": row[9],
            })
        return {"as_of_date": latest_date.isoformat(), "sectors": sectors}
    except Exception:
        logger.warning("sector_breadth.load_latest_failed")
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class SectorBreadthSnapshotRequest(BaseModel):
    as_of_date: Optional[str] = None
    lookback: int = 252


@router.post("/sector-breadth/daily-snapshot")
async def sector_breadth_daily_snapshot(
    req: SectorBreadthSnapshotRequest, request: Request
):
    if not db.db_read_enabled():
        return {"status": "skipped", "reason": "db_read_disabled"}

    as_of = (
        dt.date.fromisoformat(req.as_of_date)
        if req.as_of_date
        else dt.date.today() - dt.timedelta(days=1)
    )
    sectors = compute_sector_breadth(as_of, req.lookback)
    stored = 0
    if sectors and db.db_write_enabled():
        stored = store_sector_breadth(as_of, sectors)

    return {
        "status": "ok" if sectors else "no_data",
        "as_of_date": as_of.isoformat(),
        "lookback": req.lookback,
        "sector_count": len(sectors),
        "stored": stored,
        "sectors": sectors,
    }


@router.get("/sector-breadth/latest")
async def sector_breadth_latest(request: Request):
    data = load_sector_breadth_latest()
    if not data:
        return {"status": "no_data", "sectors": []}
    return {"status": "ok", **data}
