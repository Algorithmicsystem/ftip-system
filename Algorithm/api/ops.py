import datetime as dt
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any, Deque, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Request


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

# ---------------------------------------------------------------------------
# Intelligence Digest helpers
# ---------------------------------------------------------------------------

def _ic_health(as_of_date: dt.date) -> Dict[str, Any]:
    from api import db
    if not db.db_read_enabled():
        return {"ic_state": "UNKNOWN", "sample_count": 0, "mean_ic": None}
    row = db.safe_fetchone(
        """
        SELECT ic_state, sample_count, mean_ic, as_of_date
        FROM signal_ic_daily
        WHERE score_field = 'composite' AND horizon_label = '21d'
          AND as_of_date <= %s
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (as_of_date,),
    )
    if not row:
        return {"ic_state": "INSUFFICIENT", "sample_count": 0, "mean_ic": None, "as_of_date": None}
    return {
        "ic_state": str(row[0] or "INSUFFICIENT"),
        "sample_count": int(row[1] or 0),
        "mean_ic": float(row[2]) if row[2] is not None else None,
        "as_of_date": row[3].isoformat() if row[3] else None,
    }


def _market_posture(as_of_date: dt.date) -> Dict[str, Any]:
    from api import db
    if not db.db_read_enabled():
        return {"breadth_state": "UNKNOWN", "dominant_regime": None}
    breadth_row = db.safe_fetchone(
        "SELECT breadth_state FROM market_breadth_daily WHERE as_of_date = %s",
        (as_of_date,),
    )
    breadth = str(breadth_row[0]) if breadth_row and breadth_row[0] else "UNKNOWN"

    regime_row = db.safe_fetchone(
        """
        SELECT payload->>'regime_label' AS regime, COUNT(*) AS cnt
        FROM axiom_scores_daily
        WHERE as_of_date = %s
          AND payload->>'regime_label' IS NOT NULL
        GROUP BY 1
        ORDER BY cnt DESC
        LIMIT 1
        """,
        (as_of_date,),
    )
    dominant_regime = str(regime_row[0]) if regime_row and regime_row[0] else None
    return {"breadth_state": breadth, "dominant_regime": dominant_regime}


def _top_opportunities(as_of_date: dt.date, limit: int = 5) -> List[Dict[str, Any]]:
    from api import db
    if not db.db_read_enabled():
        return []
    rows = db.safe_fetchall(
        """
        SELECT
            a.symbol,
            (a.payload->>'deployable_alpha_utility')::numeric  AS dau,
            a.payload->>'regime_label'                          AS regime,
            a.payload->>'deployability_tier'                    AS tier,
            COALESCE(p.signal, 'UNKNOWN')                       AS signal,
            m.sector
        FROM axiom_scores_daily a
        LEFT JOIN prosperity_signals_daily p
            ON p.symbol = a.symbol AND p.as_of = a.as_of_date
        LEFT JOIN market_symbols m
            ON m.symbol = a.symbol
        WHERE a.as_of_date = %s
          AND (a.payload->>'deployable_alpha_utility')::numeric >= 60
          AND COALESCE(p.signal, 'UNKNOWN') IN ('BUY', 'SELL')
        ORDER BY dau DESC
        LIMIT %s
        """,
        (as_of_date, limit),
    )
    return [
        {
            "symbol": r[0],
            "dau": float(r[1] or 0),
            "regime": str(r[2] or ""),
            "tier": str(r[3] or ""),
            "signal": str(r[4] or ""),
            "sector": str(r[5] or ""),
        }
        for r in rows
    ]


def _sector_rotation(as_of_date: dt.date) -> List[Dict[str, Any]]:
    from api import db
    if not db.db_read_enabled():
        return []
    rows = db.safe_fetchall(
        """
        SELECT
            COALESCE(m.sector, 'Unknown')   AS sector,
            COUNT(*) FILTER (WHERE p.signal = 'BUY')  AS buy_count,
            COUNT(*) FILTER (WHERE p.signal = 'SELL') AS sell_count,
            COUNT(*) FILTER (WHERE p.signal = 'HOLD') AS hold_count,
            AVG((a.payload->>'deployable_alpha_utility')::numeric) AS avg_dau
        FROM axiom_scores_daily a
        JOIN prosperity_signals_daily p
            ON p.symbol = a.symbol AND p.as_of = a.as_of_date
        LEFT JOIN market_symbols m
            ON m.symbol = a.symbol
        WHERE a.as_of_date = %s
        GROUP BY 1
        ORDER BY buy_count DESC, avg_dau DESC
        """,
        (as_of_date,),
    )
    return [
        {
            "sector": r[0],
            "buy_count": int(r[1] or 0),
            "sell_count": int(r[2] or 0),
            "hold_count": int(r[3] or 0),
            "avg_dau": round(float(r[4] or 0), 2),
        }
        for r in rows
    ]


def _provider_status(as_of_date: dt.date) -> List[Dict[str, Any]]:
    from api import db
    if not db.db_read_enabled():
        return []
    rows = db.safe_fetchall(
        """
        SELECT provider, status, is_enabled, message
        FROM provider_reliability_daily
        WHERE as_of_date = %s
        ORDER BY provider
        """,
        (as_of_date,),
    )
    return [
        {
            "provider": r[0],
            "status": str(r[1] or "unknown"),
            "is_enabled": bool(r[2]),
            "message": str(r[3] or ""),
        }
        for r in rows
    ]


def _calibration_quality(as_of_date: dt.date) -> Dict[str, Any]:
    from api import db
    if not db.db_read_enabled():
        return {"quality_score": None, "horizon_label": None}
    row = db.safe_fetchone(
        """
        SELECT horizon_label,
               (payload->>'overall_hit_rate')::numeric AS hit_rate,
               (payload->>'sample_count')::numeric      AS n
        FROM axiom_calibration_snapshots
        WHERE as_of_date <= %s
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (as_of_date,),
    )
    if not row:
        return {"quality_score": None, "horizon_label": None, "sample_count": None}
    hit_rate = float(row[1]) if row[1] is not None else None
    return {
        "horizon_label": str(row[0] or ""),
        "hit_rate": hit_rate,
        "quality_score": round(hit_rate * 100, 1) if hit_rate is not None else None,
        "sample_count": int(row[2]) if row[2] is not None else None,
    }


@router.get("/metrics")
async def metrics() -> Dict[str, Any]:
    return metrics_tracker.snapshot()


@router.get("/last_runs")
async def last_runs() -> List[Dict[str, Any]]:
    return metrics_tracker.last_runs()


@router.get("/regime/transitions")
def regime_transitions(
    limit: int = Query(default=20, ge=1, le=100),
    since: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Return the most recent regime transition events."""
    from api.jobs.regime_monitor import load_regime_transitions
    since_date = dt.date.fromisoformat(since) if since else None
    transitions = load_regime_transitions(limit=limit, since=since_date)
    return {
        "count": len(transitions),
        "transitions": transitions,
    }


@router.post("/regime/detect")
def regime_detect(
    as_of_date: Optional[str] = Query(default=None),
    store: bool = Query(default=True),
) -> Dict[str, Any]:
    """Detect a regime transition for the given date and optionally store it."""
    from api.jobs.regime_monitor import detect_regime_transition, store_regime_transition
    date = dt.date.fromisoformat(as_of_date) if as_of_date else dt.date.today()
    transition = detect_regime_transition(date)
    if transition is None:
        return {
            "status": "no_transition",
            "as_of_date": date.isoformat(),
            "message": "Dominant regime unchanged from prior day.",
        }
    stored = store_regime_transition(transition) if store else False
    return {
        "status": "transition_detected",
        "stored": stored,
        "transition": {
            **transition,
            "as_of_date": transition["as_of_date"].isoformat()
            if hasattr(transition["as_of_date"], "isoformat")
            else str(transition["as_of_date"]),
        },
    }


@router.get("/intelligence")
def intelligence_digest(
    as_of_date: Optional[str] = Query(default=None),
    top_n: int = Query(default=5, ge=1, le=20),
) -> Dict[str, Any]:
    """CIO daily intelligence digest."""
    date = dt.date.fromisoformat(as_of_date) if as_of_date else dt.date.today()
    return {
        "as_of_date": date.isoformat(),
        "market_posture": _market_posture(date),
        "ic_health": _ic_health(date),
        "top_opportunities": _top_opportunities(date, limit=top_n),
        "sector_rotation": _sector_rotation(date),
        "provider_status": _provider_status(date),
        "calibration_quality": _calibration_quality(date),
    }


@router.get("/domain")
async def domain_readiness(request: Request) -> Dict[str, Any]:
    # Lazy import to avoid circular dependency when configuring middleware.
    from api.security import get_allowed_origins

    base_url = str(request.base_url).rstrip("/")
    return {"allowed_origins": get_allowed_origins(), "base_url": base_url}
