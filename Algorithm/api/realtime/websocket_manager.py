"""Phase 10.2: WebSocket Connection Manager and Alert Payload Builders."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy connection manager (keyed by api_key) — used by /ws/alerts
# ---------------------------------------------------------------------------


class WebSocketManager:
    """Manages WebSocket connections keyed by api_key."""

    def __init__(self) -> None:
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


# Module-level singleton (legacy /ws/alerts feed)
manager = WebSocketManager()


# ---------------------------------------------------------------------------
# Intelligence feed manager — thread-safe broadcast for /ws/intelligence
# ---------------------------------------------------------------------------


class IntelligenceWebSocketManager:
    """
    Thread-safe WebSocket manager for the /ws/intelligence intelligence feed.

    APScheduler runs jobs in background threads, not the async event loop.
    broadcast_from_thread() uses asyncio.run_coroutine_threadsafe() to safely
    send messages from those threads to all connected WebSocket clients.
    """

    def __init__(self) -> None:
        self._connections: List[Any] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the event loop at startup so background threads can broadcast."""
        self._loop = loop

    async def connect(self, websocket: Any) -> None:
        await websocket.accept()
        self._connections.append(websocket)
        logger.info("ws.intelligence.connect total=%d", len(self._connections))

    def disconnect(self, websocket: Any) -> None:
        self._connections = [c for c in self._connections if c is not websocket]
        logger.info("ws.intelligence.disconnect total=%d", len(self._connections))

    async def broadcast_async(self, message: Dict[str, Any]) -> None:
        """Broadcast to all connected clients; prune dead connections."""
        payload = json.dumps(message, default=str)
        dead = []
        for ws in list(self._connections):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def broadcast_from_thread(self, message: Dict[str, Any]) -> None:
        """Thread-safe broadcast from APScheduler jobs or other background threads."""
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.broadcast_async(message), self._loop)

    def connection_count(self) -> int:
        return len(self._connections)


# Module-level singleton for the intelligence feed
ws_manager = IntelligenceWebSocketManager()


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
