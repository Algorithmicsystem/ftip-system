"""Phase 19.4: API usage analytics and metrics."""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)

_PERIOD_DAYS = {
    "1d": 1, "7d": 7, "30d": 30, "90d": 90,
}


@dataclass
class UsageMetrics:
    tenant_id: str
    period: str
    total_requests: int = 0
    requests_by_endpoint: Dict[str, int] = field(default_factory=dict)
    requests_by_tier: Dict[str, int] = field(default_factory=dict)
    most_used_symbols: List[str] = field(default_factory=list)
    peak_hour: int = 0
    error_rate: float = 0.0
    data_freshness_requests: int = 0


def _period_since(period: str) -> dt.datetime:
    days = _PERIOD_DAYS.get(period, 7)
    return dt.datetime.utcnow() - dt.timedelta(days=days)


def compute_usage_metrics(tenant_id: str, period: str = "7d") -> UsageMetrics:
    if not db.db_read_enabled():
        return UsageMetrics(tenant_id=tenant_id, period=period)

    since = _period_since(period)

    try:
        total_row = db.safe_fetchone(
            "SELECT COUNT(*) FROM api_usage_log WHERE tenant_id = %s AND created_at >= %s",
            (tenant_id, since),
        )
        total = int(total_row[0]) if total_row and total_row[0] else 0
    except Exception:
        total = 0

    requests_by_endpoint: Dict[str, int] = {}
    try:
        rows = db.safe_fetchall(
            """
            SELECT endpoint, COUNT(*) AS cnt
              FROM api_usage_log
             WHERE tenant_id = %s AND created_at >= %s
             GROUP BY endpoint ORDER BY cnt DESC LIMIT 20
            """,
            (tenant_id, since),
        )
        requests_by_endpoint = {r[0]: int(r[1]) for r in (rows or [])}
    except Exception:
        pass

    most_used_symbols: List[str] = []
    try:
        sym_rows = db.safe_fetchall(
            """
            SELECT symbol, COUNT(*) AS cnt
              FROM api_usage_log
             WHERE tenant_id = %s AND created_at >= %s AND symbol IS NOT NULL
             GROUP BY symbol ORDER BY cnt DESC LIMIT 5
            """,
            (tenant_id, since),
        )
        most_used_symbols = [r[0] for r in (sym_rows or [])]
    except Exception:
        pass

    peak_hour = 0
    try:
        peak_row = db.safe_fetchone(
            """
            SELECT EXTRACT(HOUR FROM created_at)::int AS hr
              FROM api_usage_log
             WHERE tenant_id = %s AND created_at >= %s
             GROUP BY hr ORDER BY COUNT(*) DESC LIMIT 1
            """,
            (tenant_id, since),
        )
        if peak_row and peak_row[0] is not None:
            peak_hour = int(peak_row[0])
    except Exception:
        pass

    error_rate = 0.0
    try:
        err_row = db.safe_fetchone(
            """
            SELECT
                COUNT(*) FILTER (WHERE status_code >= 400) * 100.0 / NULLIF(COUNT(*), 0)
              FROM api_usage_log
             WHERE tenant_id = %s AND created_at >= %s
            """,
            (tenant_id, since),
        )
        if err_row and err_row[0] is not None:
            error_rate = round(float(err_row[0]), 2)
    except Exception:
        pass

    data_freshness_requests = 0
    try:
        fresh_row = db.safe_fetchone(
            """
            SELECT COUNT(*) FROM api_usage_log
             WHERE tenant_id = %s AND created_at >= %s AND as_of_date IS NOT NULL
            """,
            (tenant_id, since),
        )
        if fresh_row and fresh_row[0]:
            data_freshness_requests = int(fresh_row[0])
    except Exception:
        pass

    return UsageMetrics(
        tenant_id=tenant_id,
        period=period,
        total_requests=total,
        requests_by_endpoint=requests_by_endpoint,
        requests_by_tier={},
        most_used_symbols=most_used_symbols,
        peak_hour=peak_hour,
        error_rate=error_rate,
        data_freshness_requests=data_freshness_requests,
    )


def compute_usage_history(tenant_id: str, period: str = "30d") -> List[Dict[str, Any]]:
    """Daily request counts for time-series charting."""
    if not db.db_read_enabled():
        return []
    since = _period_since(period)
    try:
        rows = db.safe_fetchall(
            """
            SELECT DATE(created_at) AS day, COUNT(*) AS cnt
              FROM api_usage_log
             WHERE tenant_id = %s AND created_at >= %s
             GROUP BY day ORDER BY day ASC
            """,
            (tenant_id, since),
        )
        return [{"date": str(r[0]), "requests": int(r[1])} for r in (rows or [])]
    except Exception:
        return []


def compute_platform_analytics(period: str = "7d") -> Dict[str, Any]:
    """Platform-wide analytics (all tenants)."""
    if not db.db_read_enabled():
        return {"period": period, "total_requests": 0, "active_tenants": 0,
                "top_endpoints": [], "top_symbols": [], "error_rate": 0.0}

    since = _period_since(period)
    try:
        total_row = db.safe_fetchone(
            "SELECT COUNT(*) FROM api_usage_log WHERE created_at >= %s", (since,)
        )
        total = int(total_row[0]) if total_row and total_row[0] else 0

        tenant_row = db.safe_fetchone(
            "SELECT COUNT(DISTINCT tenant_id) FROM api_usage_log WHERE created_at >= %s",
            (since,),
        )
        active_tenants = int(tenant_row[0]) if tenant_row and tenant_row[0] else 0

        ep_rows = db.safe_fetchall(
            """
            SELECT endpoint, COUNT(*) AS cnt
              FROM api_usage_log WHERE created_at >= %s
             GROUP BY endpoint ORDER BY cnt DESC LIMIT 10
            """,
            (since,),
        )
        top_endpoints = [{"endpoint": r[0], "count": int(r[1])} for r in (ep_rows or [])]

        sym_rows = db.safe_fetchall(
            """
            SELECT symbol, COUNT(*) AS cnt
              FROM api_usage_log WHERE created_at >= %s AND symbol IS NOT NULL
             GROUP BY symbol ORDER BY cnt DESC LIMIT 10
            """,
            (since,),
        )
        top_symbols = [{"symbol": r[0], "count": int(r[1])} for r in (sym_rows or [])]

        err_row = db.safe_fetchone(
            """
            SELECT COUNT(*) FILTER (WHERE status_code >= 400) * 100.0 / NULLIF(COUNT(*), 0)
              FROM api_usage_log WHERE created_at >= %s
            """,
            (since,),
        )
        error_rate = round(float(err_row[0]), 2) if err_row and err_row[0] else 0.0

        return {
            "period": period,
            "total_requests": total,
            "active_tenants": active_tenants,
            "top_endpoints": top_endpoints,
            "top_symbols": top_symbols,
            "error_rate": error_rate,
        }
    except Exception as exc:
        logger.warning("usage_analytics.platform_failed error=%s", exc)
        return {"period": period, "total_requests": 0, "active_tenants": 0,
                "top_endpoints": [], "top_symbols": [], "error_rate": 0.0}


def detect_churn_risk(tenant_id: str) -> Dict[str, Any]:
    """Compare last 7 days vs prior 7 days to detect churn risk."""
    if not db.db_read_enabled():
        return {"tenant_id": tenant_id, "churn_risk": "unknown", "pct_change": 0.0}

    now = dt.datetime.utcnow()
    week_ago = now - dt.timedelta(days=7)
    two_weeks_ago = now - dt.timedelta(days=14)

    try:
        recent_row = db.safe_fetchone(
            "SELECT COUNT(*) FROM api_usage_log WHERE tenant_id = %s AND created_at >= %s",
            (tenant_id, week_ago),
        )
        prior_row = db.safe_fetchone(
            "SELECT COUNT(*) FROM api_usage_log WHERE tenant_id = %s AND created_at >= %s AND created_at < %s",
            (tenant_id, two_weeks_ago, week_ago),
        )
        recent = int(recent_row[0]) if recent_row and recent_row[0] else 0
        prior = int(prior_row[0]) if prior_row and prior_row[0] else 0

        if prior == 0:
            pct_change = 0.0
            churn_risk = "low" if recent > 0 else "unknown"
        else:
            pct_change = round((recent - prior) / prior * 100, 1)
            drop = (prior - recent) / prior * 100
            if drop > 50:
                churn_risk = "high"
            elif drop > 20:
                churn_risk = "medium"
            else:
                churn_risk = "low"

        return {
            "tenant_id": tenant_id,
            "churn_risk": churn_risk,
            "pct_change": pct_change,
            "recent_requests": recent,
            "prior_requests": prior,
        }
    except Exception as exc:
        logger.warning("usage_analytics.churn_risk_failed error=%s", exc)
        return {"tenant_id": tenant_id, "churn_risk": "unknown", "pct_change": 0.0}
