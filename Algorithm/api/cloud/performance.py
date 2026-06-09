"""Phase 22.4 / Prompt 6: In-memory performance tracker + DB-based reporting."""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory PerformanceTracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """
    Thread-safe in-memory tracker for endpoint response times.
    Uses a circular buffer (deque) per endpoint, keeping the last 1000 requests.
    Allows real p50/p95/p99 computation without DB overhead.
    CPython GIL makes deque.append and integer increments thread-safe.
    """

    MAX_SAMPLES_PER_ENDPOINT = 1000

    def __init__(self) -> None:
        # {endpoint: deque of (timestamp, response_time_ms)}
        self._samples: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.MAX_SAMPLES_PER_ENDPOINT)
        )
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._start_time = time.time()

    def record(self, endpoint: str, response_time_ms: float, is_error: bool = False) -> None:
        # Skip non-user-facing endpoints and outlier pipeline runs from p95
        if endpoint.startswith("/orchestration/") or endpoint.startswith("/jobs/"):
            return
        if response_time_ms > 30000:
            return
        now = time.time()
        self._samples[endpoint].append((now, response_time_ms))
        self._request_counts[endpoint] += 1
        if is_error:
            self._error_counts[endpoint] += 1

    def clear(self) -> None:
        self._samples.clear()
        self._error_counts.clear()
        self._request_counts.clear()
        self._start_time = time.time()

    def get_percentiles(self, endpoint: str, window_seconds: int = 3600) -> Dict[str, Any]:
        now = time.time()
        cutoff = now - window_seconds
        times = [ms for ts, ms in self._samples.get(endpoint, []) if ts >= cutoff]

        if len(times) < 5:
            return {"p50": None, "p95": None, "p99": None, "sample_count": len(times)}

        sorted_times = sorted(times)
        n = len(sorted_times)
        return {
            "p50": round(sorted_times[int(n * 0.50)], 1),
            "p95": round(sorted_times[int(n * 0.95)], 1),
            "p99": round(sorted_times[int(n * 0.99)], 1),
            "sample_count": n,
            "avg": round(sum(sorted_times) / n, 1),
        }

    def get_error_rate(self, endpoint: str) -> float:
        total = self._request_counts.get(endpoint, 0)
        if total == 0:
            return 0.0
        return round(self._error_counts.get(endpoint, 0) / total * 100, 3)

    def get_top_endpoints_by_volume(self, n: int = 10) -> List[Dict[str, Any]]:
        return sorted(
            [{"endpoint": ep, "requests": count}
             for ep, count in self._request_counts.items()],
            key=lambda x: x["requests"],
            reverse=True,
        )[:n]

    def get_slowest_endpoints(self, n: int = 5) -> List[Dict[str, Any]]:
        results = []
        for endpoint in list(self._request_counts):
            perf = self.get_percentiles(endpoint)
            if perf.get("p95") is not None:
                results.append({
                    "endpoint": endpoint,
                    "p95_ms": perf["p95"],
                    "p99_ms": perf["p99"],
                    "samples": perf["sample_count"],
                })
        return sorted(results, key=lambda x: x["p95_ms"], reverse=True)[:n]

    def get_system_p95(self, warmup_seconds: float = 30.0) -> float:
        all_times = []
        now = time.time()
        cutoff = now - 3600
        warmup_cutoff = self._start_time + warmup_seconds
        for samples in self._samples.values():
            all_times.extend([
                ms for ts, ms in samples
                if ts >= cutoff and ts >= warmup_cutoff
            ])
        if not all_times:
            return 0.0
        all_times.sort()
        return round(all_times[int(len(all_times) * 0.95)], 1)

    def get_summary(self) -> Dict[str, Any]:
        uptime = time.time() - self._start_time
        top = self.get_top_endpoints_by_volume(10)
        slowest = self.get_slowest_endpoints(5)
        system_p95 = self.get_system_p95()
        total_requests = sum(self._request_counts.values())
        total_errors = sum(self._error_counts.values())

        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "overall_error_rate_pct": round(total_errors / max(total_requests, 1) * 100, 3),
            "system_p95_ms": round(system_p95, 1),
            "meets_sla": system_p95 < 200,
            "top_endpoints": top,
            "slowest_endpoints": slowest,
            "endpoint_count": len(self._request_counts),
        }


# Module-level singleton
perf_tracker = PerformanceTracker()


# ---------------------------------------------------------------------------
# Legacy DB-based metrics (kept for backwards compatibility)
# ---------------------------------------------------------------------------

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
                COUNT(*) FILTER (WHERE status_code >= 400)::float /
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
        cache_hit_rate=0.0,
    )


def compute_system_performance_report(lookback_minutes: int = 60) -> Dict[str, Any]:
    # Primary: use in-memory tracker (always available, no DB required)
    summary = perf_tracker.get_summary()

    # Enrich with DB data if available
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
                    COUNT(*) FILTER (WHERE status_code >= 400)::float /
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

            db_endpoints = []
            for r in rows:
                db_endpoints.append({
                    "endpoint": r[0] or "unknown",
                    "p50_ms": round(float(r[1] or 0), 1),
                    "p95_ms": round(float(r[2] or 0), 1),
                    "p99_ms": round(float(r[3] or 0), 1),
                    "rpm": round(float(r[4] or 0), 2),
                    "error_rate_pct": round(float(r[5] or 0), 2),
                })

            if db_endpoints:
                summary["db_endpoints"] = db_endpoints
        except Exception as exc:
            logger.debug("db_performance_report failed err=%s", exc)

    return summary


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
                AVG(response_time_ms) FILTER (WHERE status_code = 200) AS avg_cache_ms
            FROM api_usage_log
            WHERE endpoint LIKE '%intelligence/universal%'
              AND created_at >= NOW() - INTERVAL '1 hour'
            """
        )
    except Exception:
        row = None

    hit_rate = float(row[0] or 0) if row and row[0] is not None else 0.0
    cache_ms = float(row[1] or 0) if row and row[1] is not None else 0.0
    db_ms = cache_ms / max(hit_rate, 0.01) * (1 - hit_rate) if hit_rate < 1 else cache_ms
    saved_pct = hit_rate * 100.0

    return {
        "cache_hit_rate": round(hit_rate, 3),
        "avg_cache_response_ms": round(cache_ms, 1),
        "avg_db_response_ms": round(db_ms, 1),
        "estimated_compute_saved_pct": round(saved_pct, 1),
    }
