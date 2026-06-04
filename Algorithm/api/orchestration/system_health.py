"""Phase 17.4: System Health Monitor."""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)

_KEY_TABLES = [
    ("axiom_scores_daily", 26),
    ("signal_pnl_daily", 48),
    ("signal_ic_daily", 26),
    ("market_breadth_daily", 26),
    ("morning_briefings", 26),
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SystemHealth:
    as_of: dt.datetime
    overall_health_score: float
    overall_status: str

    database_health: Dict[str, Any]
    data_freshness: Dict[str, Any]
    provider_health: Dict[str, Any]
    ml_model_health: Dict[str, Any]
    signal_quality: Dict[str, Any]
    pipeline_health: Dict[str, Any]
    sri_level: Dict[str, Any]

    active_alerts: List[str]
    resolved_since_last_check: List[str]


# ---------------------------------------------------------------------------
# Status helper (exposed for unit testing)
# ---------------------------------------------------------------------------

def compute_health_status(score: float) -> str:
    if score >= 70:
        return "healthy"
    if score >= 40:
        return "degraded"
    return "critical"


# ---------------------------------------------------------------------------
# Component checkers
# ---------------------------------------------------------------------------

def _check_db_health() -> Dict[str, Any]:
    if not db.db_enabled():
        return {
            "connectivity": "disabled",
            "migration_version": "none",
            "table_counts": {},
        }
    try:
        row = db.safe_fetchone("SELECT 1")
        connected = row is not None
    except Exception:
        connected = False

    if not connected:
        return {
            "connectivity": "down",
            "migration_version": "unknown",
            "table_counts": {},
        }

    # Migration version
    try:
        ver_row = db.safe_fetchone(
            "SELECT version FROM schema_migrations ORDER BY applied_at DESC LIMIT 1"
        )
        migration_version = str(ver_row[0]) if ver_row else "none"
    except Exception:
        migration_version = "unknown"

    # Table counts
    counts: Dict[str, int] = {}
    for table, _ in _KEY_TABLES:
        try:
            row = db.safe_fetchone(f"SELECT COUNT(*) FROM {table}")
            counts[table] = int(row[0]) if row and row[0] is not None else 0
        except Exception:
            counts[table] = -1

    return {
        "connectivity": "ok",
        "migration_version": migration_version,
        "table_counts": counts,
    }


def _check_data_freshness() -> tuple[Dict[str, Any], int]:
    """Returns (freshness_dict, stale_count)."""
    freshness: Dict[str, Any] = {}

    if not db.db_read_enabled():
        for table, _ in _KEY_TABLES:
            freshness[table] = {"hours_since_last_row": None, "status": "db_disabled"}
        return freshness, 0

    stale = 0
    for table, max_hours in _KEY_TABLES:
        try:
            if table == "morning_briefings":
                row = db.safe_fetchone(
                    "SELECT briefing_date FROM morning_briefings ORDER BY briefing_date DESC LIMIT 1"
                )
                if row and row[0]:
                    delta = (dt.date.today() - row[0]).days * 24.0
                    is_fresh = delta < max_hours
                else:
                    delta = None
                    is_fresh = False
            else:
                row = db.safe_fetchone(
                    f"SELECT MAX(as_of_date) FROM {table}"
                )
                if row and row[0]:
                    delta = (dt.date.today() - row[0]).days * 24.0
                    is_fresh = delta < max_hours
                else:
                    delta = None
                    is_fresh = False

            status = "fresh" if is_fresh else "stale"
            if not is_fresh:
                stale += 1
            freshness[table] = {"hours_since_last_row": delta, "status": status}
        except Exception:
            freshness[table] = {"hours_since_last_row": None, "status": "error"}

    return freshness, stale


def _check_provider_health() -> Dict[str, Any]:
    if not db.db_read_enabled():
        return {"status": "db_disabled", "providers": {}}
    try:
        since = dt.date.today() - dt.timedelta(days=1)
        rows = db.safe_fetchall(
            """
            SELECT provider_name, status
              FROM provider_reliability_daily
             WHERE as_of_date >= %s
             ORDER BY as_of_date DESC
            """,
            (since,),
        ) or []
        providers: Dict[str, str] = {}
        for r in rows:
            name = str(r[0] or "unknown")
            if name not in providers:
                providers[name] = str(r[1] or "unknown")
        failed = [k for k, v in providers.items() if v != "ok"]
        return {
            "status": "ok" if not failed else "degraded",
            "providers": providers,
            "failed_providers": failed,
        }
    except Exception:
        return {"status": "unknown", "providers": {}, "failed_providers": []}


def _check_ml_health() -> Dict[str, Any]:
    if not db.db_read_enabled():
        return {
            "model_version": "no_model_trained",
            "last_training": None,
            "psi_score": 0.0,
            "drift_warning": False,
        }
    try:
        row = db.safe_fetchone(
            """
            SELECT model_version, trained_at,
                   (metadata->>'psi_score')::numeric,
                   (metadata->>'drift_warning')::boolean
              FROM ml_model_registry
             WHERE regime_label IS NULL
             ORDER BY trained_at DESC LIMIT 1
            """
        )
        if not row:
            return {
                "model_version": "no_model_trained",
                "last_training": None,
                "psi_score": 0.0,
                "drift_warning": False,
            }
        psi = float(row[2] or 0.0)
        drift = bool(row[3] or False) or (psi > 0.25)
        return {
            "model_version": str(row[0] or ""),
            "last_training": str(row[1]) if row[1] else None,
            "psi_score": psi,
            "drift_warning": drift,
        }
    except Exception:
        return {"model_version": "unknown", "last_training": None, "psi_score": 0.0, "drift_warning": False}


def _check_signal_quality() -> Dict[str, Any]:
    if not db.db_read_enabled():
        return {"ic_state": "INSUFFICIENT", "calibration_hit_rate": None}
    try:
        ic_row = db.safe_fetchone(
            "SELECT ic_state FROM signal_ic_daily ORDER BY as_of_date DESC LIMIT 1"
        )
        ic_state = str(ic_row[0] or "INSUFFICIENT") if ic_row else "INSUFFICIENT"

        cal_row = db.safe_fetchone(
            """
            SELECT AVG((metadata->>'hit_rate')::numeric)
              FROM axiom_calibration_snapshots
             WHERE created_at >= now() - interval '30 days'
            """
        )
        hit_rate = float(cal_row[0]) if cal_row and cal_row[0] is not None else None
        return {"ic_state": ic_state, "calibration_hit_rate": hit_rate}
    except Exception:
        return {"ic_state": "unknown", "calibration_hit_rate": None}


def _check_pipeline_health() -> Dict[str, Any]:
    if not db.db_read_enabled():
        return {"last_run_status": "no_data", "last_run_date": None}
    try:
        from api.orchestration.pipeline_orchestrator import get_pipeline_status
        status = get_pipeline_status()
        if not status:
            return {"last_run_status": "no_runs", "last_run_date": None}
        return {
            "last_run_status": status.get("overall_status", "unknown"),
            "last_run_date": status.get("as_of_date"),
            "total_errors": status.get("total_errors", 0),
        }
    except Exception:
        return {"last_run_status": "unknown", "last_run_date": None}


def _check_sri() -> Dict[str, Any]:
    if not db.db_read_enabled():
        return {"sri": None, "sri_label": "unknown"}
    try:
        row = db.safe_fetchone(
            "SELECT sri FROM market_breadth_daily WHERE sri IS NOT NULL ORDER BY as_of_date DESC LIMIT 1"
        )
        if not row or row[0] is None:
            return {"sri": None, "sri_label": "unknown"}
        sri = float(row[0])
        if sri >= 85:
            label = "critical"
        elif sri >= 70:
            label = "high_alert"
        elif sri >= 50:
            label = "warning"
        elif sri >= 25:
            label = "elevated"
        else:
            label = "stable"
        return {"sri": sri, "sri_label": label}
    except Exception:
        return {"sri": None, "sri_label": "unknown"}


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------

def compute_system_health(as_of: Optional[dt.datetime] = None) -> SystemHealth:
    as_of = as_of or dt.datetime.utcnow()
    score = 100.0
    alerts: List[str] = []

    db_health = _check_db_health()
    if db_health.get("connectivity") == "down":
        score -= 30.0
        alerts.append("database_connectivity_failure")
    elif db_health.get("connectivity") == "disabled":
        pass  # test mode, no penalty

    freshness, stale_count = _check_data_freshness()
    score -= stale_count * 5.0
    if stale_count > 0:
        alerts.append(f"{stale_count}_stale_data_tables")

    provider = _check_provider_health()
    failed_providers = provider.get("failed_providers", [])
    score -= len(failed_providers) * 5.0
    for p in failed_providers:
        alerts.append(f"provider_down_{p}")

    ml_health = _check_ml_health()
    if ml_health.get("drift_warning"):
        score -= 5.0
        alerts.append("ml_drift_warning")

    signal_quality = _check_signal_quality()
    ic_state = signal_quality.get("ic_state", "INSUFFICIENT")
    if ic_state in ("DEGRADED", "WEAK"):
        score -= 10.0
        alerts.append(f"ic_state_{ic_state.lower()}")

    pipeline_health = _check_pipeline_health()
    if pipeline_health.get("last_run_status") in ("failed", "partial"):
        score -= 5.0
        alerts.append("pipeline_last_run_degraded")

    sri_level = _check_sri()
    sri_val = sri_level.get("sri")
    if sri_val is not None and sri_val >= 70:
        score -= 10.0
        alerts.append(f"high_systemic_risk_sri={sri_val:.0f}")

    score = max(0.0, min(100.0, score))
    status = compute_health_status(score)

    return SystemHealth(
        as_of=as_of,
        overall_health_score=round(score, 1),
        overall_status=status,
        database_health=db_health,
        data_freshness=freshness,
        provider_health=provider,
        ml_model_health=ml_health,
        signal_quality=signal_quality,
        pipeline_health=pipeline_health,
        sri_level=sri_level,
        active_alerts=alerts,
        resolved_since_last_check=[],
    )


def get_health_history(lookback_days: int = 30) -> List[Dict[str, Any]]:
    """Returns daily health snapshots — empty list when DB disabled."""
    if not db.db_read_enabled():
        return []
    try:
        since = dt.date.today() - dt.timedelta(days=lookback_days)
        rows = db.safe_fetchall(
            """
            SELECT as_of_date, sri, breadth_state
              FROM market_breadth_daily
             WHERE as_of_date >= %s
             ORDER BY as_of_date DESC
            """,
            (since,),
        ) or []
        return [
            {
                "as_of_date": str(r[0]),
                "sri": float(r[1]) if r[1] is not None else None,
                "breadth_state": str(r[2]) if r[2] else "UNKNOWN",
            }
            for r in rows
        ]
    except Exception:
        return []
