"""Phase 10.1: Intraday score API routes."""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from api import db, security
from api.axiom.intraday.intraday_engine import IntradaySnapshot, run_intraday_update

router = APIRouter(
    prefix="/axiom/intraday",
    tags=["axiom_intraday"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)
logger = logging.getLogger(__name__)


@router.get("/{symbol}")
def get_intraday_snapshot(symbol: str) -> Dict[str, Any]:
    """Return the latest intraday snapshot for a symbol."""
    if not db.db_read_enabled():
        return {"symbol": symbol, "status": "db_disabled"}

    row = db.safe_fetchone(
        """
        SELECT intraday_flow_score, intraday_behavioral_score,
               intraday_composite, vwap_deviation,
               intraday_momentum_score, volume_surge_score,
               session_return_pct, alert_eligible, update_time
          FROM axiom_intraday_updates
         WHERE symbol = %s
           AND session_date = CURRENT_DATE
         ORDER BY update_time DESC
         LIMIT 1
        """,
        (symbol.upper(),),
    )
    if not row:
        return {"symbol": symbol, "status": "no_data"}

    return {
        "symbol": symbol,
        "intraday_flow_score": row[0],
        "intraday_behavioral_score": row[1],
        "intraday_composite": row[2],
        "vwap_deviation": row[3],
        "intraday_momentum_score": row[4],
        "volume_surge_score": row[5],
        "session_return_pct": row[6],
        "alert_eligible": row[7],
        "update_time": row[8].isoformat() if row[8] else None,
    }


@router.get("/universe/latest")
def get_universe_intraday() -> Dict[str, Any]:
    """Return latest intraday snapshots for all symbols today."""
    if not db.db_read_enabled():
        return {"status": "db_disabled", "snapshots": []}

    rows = db.safe_fetchall(
        """
        SELECT DISTINCT ON (symbol)
               symbol, intraday_composite, alert_eligible, update_time
          FROM axiom_intraday_updates
         WHERE session_date = CURRENT_DATE
         ORDER BY symbol, update_time DESC
        """,
        (),
    )
    snapshots = [
        {
            "symbol": row[0],
            "intraday_composite": row[1],
            "alert_eligible": row[2],
            "update_time": row[3].isoformat() if row[3] else None,
        }
        for row in (rows or [])
    ]
    return {"session_date": dt.date.today().isoformat(), "snapshots": snapshots}


class IntradayRunRequest(BaseModel):
    symbols: List[str]
    avg_daily_volume: float = 1_000_000.0


@router.post("/run")
def run_intraday_update_endpoint(req: IntradayRunRequest) -> Dict[str, Any]:
    """Manually trigger intraday score update for a list of symbols."""
    results = []
    for symbol in req.symbols[:50]:  # cap at 50 symbols per call
        # Stub: in production this would pull real intraday bars
        snapshot = run_intraday_update(
            symbol=symbol.upper(),
            intraday_bars=[],
            daily_axiom_dau=50.0,
            avg_daily_volume=req.avg_daily_volume,
        )
        results.append({"symbol": symbol.upper(), "alert_eligible": snapshot.alert_eligible})

    return {"status": "ok", "updated": len(results), "results": results}
