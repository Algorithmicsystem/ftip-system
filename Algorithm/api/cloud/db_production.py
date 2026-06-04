"""Phase 22.2: Production Database Pool Optimization."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

from api import db

logger = logging.getLogger(__name__)

# Module-level slow query counter (incremented by middleware / manual calls)
_slow_query_count: int = 0


def record_slow_query() -> None:
    global _slow_query_count
    _slow_query_count += 1


def reset_slow_query_count() -> None:
    global _slow_query_count
    _slow_query_count = 0


def check_connection_health(timeout_ms: int = 2000) -> bool:
    """Ping DB with SELECT 1, respecting timeout."""
    if not db.db_enabled():
        return False
    start = time.monotonic()
    try:
        row = db.safe_fetchone("SELECT 1 AS ping")
        elapsed_ms = (time.monotonic() - start) * 1000
        if elapsed_ms > timeout_ms:
            logger.warning("db_health_check slow elapsed_ms=%.0f", elapsed_ms)
        return row is not None
    except Exception as exc:
        logger.warning("db_health_check failed err=%s", exc)
        return False


def get_db_pool_stats() -> Dict[str, Any]:
    """Return current pool utilisation statistics."""
    if not db.db_enabled():
        return {
            "pool_size": 0,
            "connections_in_use": 0,
            "connections_available": 0,
            "pool_utilization_pct": 0.0,
            "slow_query_count": _slow_query_count,
        }

    pool_size = 0
    connections_in_use = 0
    try:
        pool = db.get_pool()
        # psycopg ConnectionPool exposes _pool (idle list) and _stats
        stats = pool.get_stats() if hasattr(pool, "get_stats") else {}
        pool_size = stats.get("pool_max", stats.get("pool_size", 0))
        pool_available = stats.get("pool_available", 0)
        connections_in_use = max(0, pool_size - pool_available)
    except Exception:
        pass

    utilization = (connections_in_use / pool_size * 100.0) if pool_size > 0 else 0.0
    if utilization >= 90:
        logger.warning("db_pool_exhaustion pool_utilization=%.1f%%", utilization)

    return {
        "pool_size": pool_size,
        "connections_in_use": connections_in_use,
        "connections_available": max(0, pool_size - connections_in_use),
        "pool_utilization_pct": round(utilization, 1),
        "slow_query_count": _slow_query_count,
    }
