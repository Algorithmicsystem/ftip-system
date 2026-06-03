"""Phase 10.2: WebSocket Alert Endpoint.

WS /ws/alerts — real-time alert stream authenticated by api_key query param.
Clients can subscribe to symbol filters and severity thresholds.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from typing import Any, Dict, Optional, Set

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from api import db, security
from api.realtime.websocket_manager import (
    build_bubble_warning,
    build_opportunity_alert,
    build_risk_alert,
    manager,
)

router = APIRouter(tags=["realtime"])
logger = logging.getLogger(__name__)

_BROADCASTER_INTERVAL_S = 60
_SRI_ALERT_THRESHOLD = 70.0
_SRI_ALERT_COOLDOWN_S = 3600  # 60-minute throttle

_last_sri_alert_time: Optional[dt.datetime] = None


async def _alert_broadcaster() -> None:
    """Background task: scan for new alert-eligible events every 60 seconds."""
    while True:
        await asyncio.sleep(_BROADCASTER_INTERVAL_S)
        if manager.connection_count() == 0:
            continue
        try:
            await _scan_and_broadcast()
        except Exception as exc:
            logger.warning("ws.broadcaster_error error=%s", exc)


async def _scan_and_broadcast() -> None:
    """Query recent axiom + intraday updates and broadcast relevant alerts."""
    global _last_sri_alert_time
    if not db.db_read_enabled():
        return

    two_min_ago = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=2)

    # Symbols with new high-DAU scores in last 2 minutes
    rows = db.safe_fetchall(
        """
        SELECT symbol, payload->>'deployable_alpha_utility', payload->>'regime_label'
          FROM axiom_scores_daily
         WHERE updated_at >= %s
           AND (payload->>'deployable_alpha_utility')::numeric >= 70
         LIMIT 20
        """,
        (two_min_ago,),
    ) or []

    for row in rows:
        symbol = str(row[0])
        dau = float(row[1] or 0.0)
        regime = str(row[2] or "unknown")
        alert = build_opportunity_alert(symbol, dau, regime, "axiom_score_update")
        await manager.broadcast_alert(alert)

    # Intraday alert-eligible symbols in last 5 minutes
    five_min_ago = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)
    intraday_rows = db.safe_fetchall(
        """
        SELECT symbol, intraday_composite
          FROM axiom_intraday_updates
         WHERE alert_eligible = TRUE
           AND update_time >= %s
         LIMIT 20
        """,
        (five_min_ago,),
    ) or []

    for row in intraday_rows:
        symbol = str(row[0])
        composite = float(row[1] or 0.0)
        alert = build_opportunity_alert(symbol, composite, "intraday", "intraday_composite_threshold")
        await manager.broadcast_alert(alert)

    # SRI check: query recent market_breadth_daily for systemic risk
    sri_row = db.safe_fetchone(
        """
        SELECT sri
          FROM market_breadth_daily
         WHERE sri IS NOT NULL
         ORDER BY as_of_date DESC
         LIMIT 1
        """,
        (),
    )
    if sri_row and sri_row[0] is not None:
        sri = float(sri_row[0])
        if sri > _SRI_ALERT_THRESHOLD:
            now = dt.datetime.now(dt.timezone.utc)
            elapsed = (now - _last_sri_alert_time).total_seconds() if _last_sri_alert_time else _SRI_ALERT_COOLDOWN_S + 1
            if elapsed >= _SRI_ALERT_COOLDOWN_S:
                risk_alert = build_risk_alert(
                    symbol="SYSTEM",
                    risk_type="systemic_risk",
                    severity="warning",
                    description=f"SRI at {sri:.1f}",
                )
                await manager.broadcast_alert(risk_alert)
                _last_sri_alert_time = now


@router.websocket("/ws/alerts")
async def ws_alerts_endpoint(
    websocket: WebSocket,
    api_key: str = Query(default=""),
) -> None:
    """WebSocket endpoint for real-time alert streaming."""
    # Validate API key (WS uses query param, not header)
    allowed = security.get_allowed_api_keys()
    if allowed and api_key.strip() not in allowed:
        await websocket.close(code=4001)
        return

    await manager.connect(websocket, api_key)
    subscription: Dict[str, Any] = {"symbols": [], "min_severity": "info"}

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                if "symbols" in msg:
                    subscription["symbols"] = [s.upper() for s in msg["symbols"]]
                if "min_severity" in msg:
                    subscription["min_severity"] = msg["min_severity"]
                await websocket.send_json({"status": "subscription_updated", "subscription": subscription})
            except json.JSONDecodeError:
                await websocket.send_json({"status": "error", "detail": "invalid_json"})
    except WebSocketDisconnect:
        await manager.disconnect(websocket, api_key)
    except Exception:
        await manager.disconnect(websocket, api_key)
