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
from api.universe import AXIOM_UNIVERSE

router = APIRouter(
    prefix="/axiom/intraday",
    tags=["axiom_intraday"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)
logger = logging.getLogger(__name__)


def _store_intraday_snapshot(snap: IntradaySnapshot) -> None:
    """Persist IntradaySnapshot to axiom_intraday_updates table."""
    if not db.db_read_enabled():
        return
    try:
        db.safe_execute(
            """
            INSERT INTO axiom_intraday_updates
                (symbol, update_time, session_date, intraday_flow_score,
                 intraday_behavioral_score, intraday_composite, vwap_deviation,
                 intraday_momentum_score, volume_surge_score, session_return_pct,
                 alert_eligible, meta)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (symbol, update_time) DO UPDATE SET
                intraday_composite        = EXCLUDED.intraday_composite,
                vwap_deviation            = EXCLUDED.vwap_deviation,
                volume_surge_score        = EXCLUDED.volume_surge_score,
                session_return_pct        = EXCLUDED.session_return_pct,
                alert_eligible            = EXCLUDED.alert_eligible,
                meta                      = EXCLUDED.meta
            """,
            (
                snap.symbol,
                snap.timestamp,
                snap.timestamp.date(),
                snap.intraday_flow_score,
                snap.intraday_behavioral_score,
                snap.intraday_composite,
                snap.vwap_deviation,
                snap.intraday_momentum_score,
                snap.volume_surge_score,
                snap.session_return_pct,
                snap.alert_eligible,
                json.dumps({"alert_type": snap.alert_type, "source": snap.source,
                            "daily_axiom_dau": snap.daily_axiom_dau}),
            ),
        )
    except Exception as exc:
        logger.warning("intraday.store_failed symbol=%s err=%s", snap.symbol, exc)


def _load_daily_dau(symbol: str) -> float:
    """Load the most recent DAU for a symbol from axiom_scores_daily."""
    if not db.db_read_enabled():
        return 50.0
    try:
        row = db.safe_fetchone(
            """
            SELECT (payload->>'deployable_alpha_utility')::numeric
              FROM axiom_scores_daily
             WHERE symbol = %s
             ORDER BY as_of_date DESC LIMIT 1
            """,
            (symbol,),
        )
        if row and row[0] is not None:
            return float(row[0])
    except Exception:
        pass
    return 50.0


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
    """Return latest intraday snapshots for all 30 AXIOM universe symbols today."""
    if not db.db_read_enabled():
        return {"status": "db_disabled", "snapshots": [], "as_of": dt.datetime.utcnow().isoformat()}

    try:
        rows = db.safe_fetchall(
            """
            SELECT DISTINCT ON (symbol)
                   symbol, intraday_composite, vwap_deviation,
                   volume_surge_score, session_return_pct, alert_eligible,
                   update_time, meta
              FROM axiom_intraday_updates
             WHERE session_date = CURRENT_DATE
             ORDER BY symbol, update_time DESC
            """,
            (),
        ) or []
    except Exception:
        rows = []
    by_symbol: Dict[str, Any] = {}
    for row in (rows or []):
        meta = row[7] if isinstance(row[7], dict) else {}
        by_symbol[str(row[0])] = {
            "symbol": str(row[0]),
            "intraday_composite": float(row[1]) if row[1] is not None else None,
            "vwap_deviation": float(row[2]) if row[2] is not None else None,
            "volume_surge_score": float(row[3]) if row[3] is not None else None,
            "session_return_pct": float(row[4]) if row[4] is not None else None,
            "alert_eligible": bool(row[5]),
            "update_time": row[6].isoformat() if row[6] else None,
            "source": meta.get("source", "no_data"),
        }

    snapshots = [
        by_symbol.get(sym, {
            "symbol": sym,
            "intraday_composite": None,
            "vwap_deviation": None,
            "volume_surge_score": None,
            "session_return_pct": None,
            "alert_eligible": False,
            "update_time": None,
            "source": "no_data",
        })
        for sym in AXIOM_UNIVERSE
    ]
    return {"as_of": dt.datetime.utcnow().isoformat(), "snapshots": snapshots}


class IntradayRunRequest(BaseModel):
    symbols: List[str]
    avg_daily_volume: float = 1_000_000.0


@router.post("/run")
def run_intraday_update_endpoint(req: IntradayRunRequest) -> Dict[str, Any]:
    """Trigger intraday score update; loads daily DAU from DB, stores results, broadcasts alerts."""
    symbols = req.symbols if req.symbols else list(AXIOM_UNIVERSE)
    results = []
    alert_eligible_snapshots = []

    for symbol in symbols[:50]:
        sym = symbol.upper()
        daily_dau = _load_daily_dau(sym)
        snapshot = run_intraday_update(
            symbol=sym,
            intraday_bars=[],  # live bars fetched in production via Polygon
            daily_axiom_dau=daily_dau,
            avg_daily_volume=req.avg_daily_volume,
        )
        _store_intraday_snapshot(snapshot)
        if snapshot.alert_eligible:
            alert_eligible_snapshots.append(snapshot)
        results.append({
            "symbol": sym,
            "alert_eligible": snapshot.alert_eligible,
            "intraday_composite": snapshot.intraday_composite,
            "alert_type": snapshot.alert_type,
        })

    # Broadcast alert-eligible snapshots via WebSocket
    if alert_eligible_snapshots:
        try:
            import datetime as _dt
            from api.realtime.websocket_manager import ws_manager
            for snap in alert_eligible_snapshots:
                ws_manager.broadcast_from_thread({
                    "type": "intraday_update",
                    "symbol": snap.symbol,
                    "intraday_composite": snap.intraday_composite,
                    "vwap_deviation": snap.vwap_deviation,
                    "volume_surge_score": snap.volume_surge_score,
                    "session_return_pct": snap.session_return_pct,
                    "update_time": snap.timestamp.isoformat(),
                })
                if snap.alert_type:
                    ws_manager.broadcast_from_thread({
                        "type": "signal_alert",
                        "symbol": snap.symbol,
                        "action": "HOLD",
                        "dau": snap.intraday_composite,
                        "alert_type": snap.alert_type,
                        "intraday_composite": snap.intraday_composite,
                        "daily_dau": snap.daily_axiom_dau,
                        "delta_dau": round((snap.intraday_composite or 0) - (snap.daily_axiom_dau or 0), 2),
                        "timestamp": snap.timestamp.isoformat(),
                    })
        except Exception as exc:
            logger.debug("intraday.ws_broadcast_failed err=%s", exc)

    return {"status": "ok", "updated": len(results), "alert_count": len(alert_eligible_snapshots), "results": results}
