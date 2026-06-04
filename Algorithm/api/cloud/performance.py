"""Phase 22.4: Performance Profiling and Optimization."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    endpoint: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_minute: float
    error_rate_pct: float
    cache_hit_rate: float


def _empty_metrics(endpoint: str) -> PerformanceMetrics:
    return PerformanceMetrics(
        endpoint=endpoint,
        p50_ms=0.0, p95_ms=0.0, p99_ms=0.0,
        requests_per_minute=0.0,
        error_rate_pct=0.0,
        cache_hit_rate=0.0,
    )


def compute_endpoint_performance_metrics(
    endpoint: str,
    lookback_minutes: int = 60,
) -> PerformanceMetrics:
    if not db.db_read_enabled():
        return _empty_metrics(endpoint)

    try:
        row = db.safe_fetchone(
            """
            SELECT
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY response_time_ms) AS p50,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) AS p95,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms) AS p99,
                COUNT(*)::float / GREATEST(%s / 60.0, 1) AS rpm,
                COUNT(*) FILTER (WHERE response_code >= 400)::float /
                    NULLIF(COUNT(*), 0) * 100 AS error_rate_pct
            FROM api_usage_log
            WHERE endpoint LIKE %s
              AND created_at >= NOW() - (%s || ' minutes')::interval
            """,
            (lookback_minutes, f"%{endpoint}%", str(lookback_minutes)),
        )
    except Exception:
        return _empty_metrics(endpoint)

    if not row or row[0] is None:
        return _empty_metrics(endpoint)

    return PerformanceMetrics(
        endpoint=endpoint,
        p50_ms=round(float(row[0] or 0), 1),
        p95_ms=round(float(row[1] or 0), 1),
        p99_ms=round(float(row[2] or 0), 1),
        requests_per_minute=round(float(row[3] or 0), 2),
        error_rate_pct=round(float(row[4] or 0), 2),
        cache_hit_rate=0.0,  # enriched by cache-specific function
    )


def compute_system_performance_report(lookback_minutes: int = 60) -> Dict[str, Any]:
    top_endpoints: List[PerformanceMetrics] = []
    slowest: List[Dict[str, Any]] = []
    highest_error: List[Dict[str, Any]] = []
    overall_p99 = 0.0
    overall_rpm = 0.0
    cache_hit_rate = 0.0

    if db.db_read_enabled():
        try:
            rows = db.safe_fetchall(
                """
                SELECT
                    endpoint,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY response_time_ms) AS p50,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) AS p95,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms) AS p99,
                    COUNT(*)::float / GREATEST(%s / 60.0, 1) AS rpm,
                    COUNT(*) FILTER (WHERE response_code >= 400)::float /
                        NULLIF(COUNT(*), 0) * 100 AS error_rate_pct,
                    COUNT(*) AS total_requests
                FROM api_usage_log
                WHERE created_at >= NOW() - (%s || ' minutes')::interval
                  AND endpoint IS NOT NULL
                GROUP BY endpoint
                ORDER BY total_requests DESC
                LIMIT 10
                """,
                (lookback_minutes, str(lookback_minutes)),
            ) or []

            for r in rows:
                ep = r[0] or "unknown"
                m = PerformanceMetrics(
                    endpoint=ep,
                    p50_ms=round(float(r[1] or 0), 1),
                    p95_ms=round(float(r[2] or 0), 1),
                    p99_ms=round(float(r[3] or 0), 1),
                    requests_per_minute=round(float(r[4] or 0), 2),
                    error_rate_pct=round(float(r[5] or 0), 2),
                    cache_hit_rate=0.0,
                )
                top_endpoints.append(m)

            if top_endpoints:
                overall_p99 = max(m.p99_ms for m in top_endpoints)
                overall_rpm = sum(m.requests_per_minute for m in top_endpoints)

            slowest = sorted(
                [{"endpoint": m.endpoint, "p99_ms": m.p99_ms} for m in top_endpoints if m.p99_ms > 1000],
                key=lambda x: x["p99_ms"], reverse=True,
            )
            highest_error = sorted(
                [{"endpoint": m.endpoint, "error_rate_pct": m.error_rate_pct}
                 for m in top_endpoints if m.error_rate_pct > 0],
                key=lambda x: x["error_rate_pct"], reverse=True,
            )

        except Exception as exc:
            logger.warning("performance_report failed err=%s", exc)

        # Cache hit rate from universal cache table
        try:
            cache_row = db.safe_fetchone(
                """
                SELECT
                    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '1 hour')::float /
                    NULLIF(COUNT(*), 0)
                FROM universal_intelligence_cache
                WHERE expires_at > NOW()
                """
            )
            if cache_row and cache_row[0] is not None:
                cache_hit_rate = round(float(cache_row[0]), 3)
        except Exception:
            pass

    return {
        "top_endpoints": top_endpoints,
        "slowest_endpoints": slowest,
        "highest_error_endpoints": highest_error,
        "overall_p99_ms": overall_p99,
        "overall_requests_per_minute": overall_rpm,
        "cache_hit_rate_pct": cache_hit_rate * 100,
    }


def compute_universal_endpoint_cache_effectiveness() -> Dict[str, Any]:
    if not db.db_read_enabled():
        return {
            "cache_hit_rate": 0.0,
            "avg_cache_response_ms": 0.0,
            "avg_db_response_ms": 0.0,
            "estimated_compute_saved_pct": 0.0,
        }

    try:
        row = db.safe_fetchone(
            """
            SELECT
                COUNT(*) FILTER (WHERE expires_at > NOW())::float /
                    NULLIF(COUNT(*), 0) AS hit_rate,
                AVG(response_time_ms) FILTER (WHERE response_code = 200) AS avg_cache_ms
            FROM api_usage_log
            WHERE endpoint LIKE '%intelligence/universal%'
              AND created_at >= NOW() - INTERVAL '1 hour'
            """
        )
    except Exception:
        row = None

    hit_rate   = float(row[0] or 0) if row and row[0] is not None else 0.0
    cache_ms   = float(row[1] or 0) if row and row[1] is not None else 0.0
    db_ms      = cache_ms / max(hit_rate, 0.01) * (1 - hit_rate) if hit_rate < 1 else cache_ms
    saved_pct  = hit_rate * 100.0

    return {
        "cache_hit_rate": round(hit_rate, 3),
        "avg_cache_response_ms": round(cache_ms, 1),
        "avg_db_response_ms": round(db_ms, 1),
        "estimated_compute_saved_pct": round(saved_pct, 1),
    }
