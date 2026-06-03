"""Phase 10.2: WebSocket Connection Manager and Alert Payload Builders."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------


class WebSocketManager:
    """Manages WebSocket connections keyed by api_key."""

    def __init__(self) -> None:
        # Maps api_key → list of WebSocket objects
        self.active_connections: Dict[str, List[Any]] = {}

    async def connect(self, websocket: Any, api_key: str) -> None:
        await websocket.accept()
        self.active_connections.setdefault(api_key, []).append(websocket)
        logger.info("ws.connected api_key=%s total=%d", api_key[:8], self.connection_count())

    async def disconnect(self, websocket: Any, api_key: str) -> None:
        conns = self.active_connections.get(api_key, [])
        try:
            conns.remove(websocket)
        except ValueError:
            pass
        if not conns:
            self.active_connections.pop(api_key, None)
        logger.info("ws.disconnected api_key=%s total=%d", api_key[:8], self.connection_count())

    async def send_to_key(self, api_key: str, message: Dict[str, Any]) -> None:
        dead: List[Any] = []
        for ws in list(self.active_connections.get(api_key, [])):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws, api_key)

    async def broadcast_alert(self, alert: Dict[str, Any]) -> None:
        for api_key in list(self.active_connections.keys()):
            await self.send_to_key(api_key, alert)

    def connection_count(self) -> int:
        return sum(len(v) for v in self.active_connections.values())


# Module-level singleton
manager = WebSocketManager()


# ---------------------------------------------------------------------------
# Alert payload builders
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_opportunity_alert(
    symbol: str,
    dau: float,
    regime: str,
    driver: str,
) -> Dict[str, Any]:
    return {
        "alert_type": "opportunity",
        "symbol": symbol,
        "timestamp": _now_iso(),
        "severity": "info" if dau < 75 else "warning",
        "payload": {
            "dau": dau,
            "regime": regime,
            "driver": driver,
        },
    }


def build_risk_alert(
    symbol: str,
    risk_type: str,
    severity: str,
    description: str,
) -> Dict[str, Any]:
    return {
        "alert_type": "risk",
        "symbol": symbol,
        "timestamp": _now_iso(),
        "severity": severity,
        "payload": {
            "risk_type": risk_type,
            "description": description,
        },
    }


def build_regime_change_alert(
    from_regime: str,
    to_regime: str,
    symbol_count: int,
) -> Dict[str, Any]:
    return {
        "alert_type": "regime_change",
        "symbol": None,
        "timestamp": _now_iso(),
        "severity": "warning",
        "payload": {
            "from_regime": from_regime,
            "to_regime": to_regime,
            "symbol_count": symbol_count,
        },
    }


def build_bubble_warning(
    symbol: str,
    scps_score: float,
    bfs_score: float,
) -> Dict[str, Any]:
    severity = "critical" if max(scps_score, bfs_score) > 80 else "warning"
    return {
        "alert_type": "bubble_warning",
        "symbol": symbol,
        "timestamp": _now_iso(),
        "severity": severity,
        "payload": {
            "scps_score": scps_score,
            "bfs_score": bfs_score,
        },
    }


def build_earnings_stress_alert(
    symbol: str,
    pess_score: float,
    days_to_earnings: int,
) -> Dict[str, Any]:
    severity = "critical" if pess_score > 80 else "warning"
    return {
        "alert_type": "earnings_stress",
        "symbol": symbol,
        "timestamp": _now_iso(),
        "severity": severity,
        "payload": {
            "pess_score": pess_score,
            "days_to_earnings": days_to_earnings,
        },
    }
