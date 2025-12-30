import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any, Deque, Dict, List, Optional

from fastapi import APIRouter, Request


class MetricsTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._status_4xx: int = 0
        self._status_5xx: int = 0
        self._rate_limit_hits: int = 0
        self._narrator_calls: int = 0
        self._snapshot_runs: int = 0
        self._strategy_graph_runs: int = 0
        self._last_runs: Deque[Dict[str, Any]] = deque(maxlen=20)

    def record_request(self, path: str, status_code: int) -> None:
        with self._lock:
            self._request_counts[path] += 1
            if 400 <= status_code < 500:
                self._status_4xx += 1
            if status_code >= 500:
                self._status_5xx += 1

    def record_rate_limit_hit(self) -> None:
        with self._lock:
            self._rate_limit_hits += 1

    def record_narrator_call(self) -> None:
        with self._lock:
            self._narrator_calls += 1

    def record_run(
        self,
        kind: str,
        trace_id: Optional[str],
        status: str,
        timings: Optional[Dict[str, Any]] = None,
        rows_written: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "kind": kind,
            "trace_id": trace_id,
            "status": status,
            "timings": timings or {},
            "rows_written": rows_written or {},
            "recorded_at": time.time(),
        }
        with self._lock:
            self._last_runs.appendleft(payload)
            if kind == "snapshot":
                self._snapshot_runs += 1
            if kind == "strategy_graph":
                self._strategy_graph_runs += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "request_counts": dict(self._request_counts),
                "status_4xx": self._status_4xx,
                "status_5xx": self._status_5xx,
                "rate_limit_hits": self._rate_limit_hits,
                "narrator_calls": self._narrator_calls,
                "snapshot_runs": self._snapshot_runs,
                "strategy_graph_runs": self._strategy_graph_runs,
            }

    def last_runs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._last_runs)


metrics_tracker = MetricsTracker()
router = APIRouter(prefix="/ops", tags=["ops"])


@router.get("/metrics")
async def metrics() -> Dict[str, Any]:
    return metrics_tracker.snapshot()


@router.get("/last_runs")
async def last_runs() -> List[Dict[str, Any]]:
    return metrics_tracker.last_runs()


@router.get("/domain")
async def domain_readiness(request: Request) -> Dict[str, Any]:
    # Lazy import to avoid circular dependency when configuring middleware.
    from api.security import get_allowed_origins

    base_url = str(request.base_url).rstrip("/")
    return {"allowed_origins": get_allowed_origins(), "base_url": base_url}
