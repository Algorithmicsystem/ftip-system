"""Phase 22.3: Production Monitoring and Alerting."""
from __future__ import annotations

import datetime as dt
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

PRODUCTION_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "api_error_rate_pct":       {"warning": 1.0,  "critical": 5.0},
    "api_latency_p99_ms":       {"warning": 500,  "critical": 2000},
    "db_pool_utilization_pct":  {"warning": 70,   "critical": 90},
    "pipeline_failure_count":   {"warning": 1,    "critical": 3},
    "ml_psi_score":             {"warning": 0.15, "critical": 0.25},
    "sri_level":                {"warning": 60,   "critical": 80},
    "data_staleness_hours":     {"warning": 25,   "critical": 48},
}

_VALID_SEVERITIES = {"info", "warning", "critical", "page"}
_VALID_CATEGORIES = {"database", "api", "pipeline", "ml", "business"}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProductionAlert:
    alert_id: str
    severity: str
    category: str
    title: str
    description: str
    threshold_value: float
    actual_value: float
    triggered_at: dt.datetime
    resolved_at: Optional[dt.datetime] = None
    auto_resolved: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_alert(
    severity: str,
    category: str,
    title: str,
    description: str,
    threshold_value: float,
    actual_value: float,
) -> ProductionAlert:
    return ProductionAlert(
        alert_id=str(uuid.uuid4()),
        severity=severity,
        category=category,
        title=title,
        description=description,
        threshold_value=threshold_value,
        actual_value=actual_value,
        triggered_at=dt.datetime.utcnow(),
    )


def _severity_for(key: str, value: float) -> Optional[str]:
    thresholds = PRODUCTION_THRESHOLDS.get(key)
    if thresholds is None:
        return None
    if value >= thresholds["critical"]:
        return "critical"
    if value >= thresholds["warning"]:
        return "warning"
    return None


# ---------------------------------------------------------------------------
# Component checks
# ---------------------------------------------------------------------------

def _check_api_metrics() -> List[ProductionAlert]:
    alerts: List[ProductionAlert] = []
    if not db.db_read_enabled():
        return alerts
    try:
        row = db.safe_fetchone(
            """
            SELECT
                COUNT(*) FILTER (WHERE response_code >= 400)::float /
                    NULLIF(COUNT(*), 0) * 100 AS error_rate_pct,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms) AS p99_ms
            FROM api_usage_log
            WHERE created_at >= NOW() - INTERVAL '1 hour'
            """
        )
        if row:
            error_rate = float(row[0] or 0)
            p99 = float(row[1] or 0)

            sev = _severity_for("api_error_rate_pct", error_rate)
            if sev:
                alerts.append(_make_alert(
                    sev, "api", "API Error Rate Elevated",
                    f"Error rate {error_rate:.1f}% exceeds threshold",
                    PRODUCTION_THRESHOLDS["api_error_rate_pct"][sev],
                    error_rate,
                ))

            sev = _severity_for("api_latency_p99_ms", p99)
            if sev:
                alerts.append(_make_alert(
                    sev, "api", "API Latency p99 Elevated",
                    f"p99 latency {p99:.0f}ms exceeds threshold",
                    PRODUCTION_THRESHOLDS["api_latency_p99_ms"][sev],
                    p99,
                ))
    except Exception:
        pass
    return alerts


def _check_pipeline_failures() -> List[ProductionAlert]:
    alerts: List[ProductionAlert] = []
    if not db.db_read_enabled():
        return alerts
    try:
        row = db.safe_fetchone(
            """
            SELECT COUNT(*) FROM pipeline_runs
            WHERE status = 'failed'
              AND started_at >= NOW() - INTERVAL '24 hours'
            """
        )
        count = float(row[0] or 0) if row else 0.0
        sev = _severity_for("pipeline_failure_count", count)
        if sev:
            alerts.append(_make_alert(
                sev, "pipeline", "Pipeline Failures Detected",
                f"{int(count)} pipeline failure(s) in last 24 hours",
                PRODUCTION_THRESHOLDS["pipeline_failure_count"][sev],
                count,
            ))
    except Exception:
        pass
    return alerts


def _check_ml_drift() -> List[ProductionAlert]:
    alerts: List[ProductionAlert] = []
    if not db.db_read_enabled():
        return alerts
    try:
        row = db.safe_fetchone(
            """
            SELECT psi_score FROM ml_model_registry
            WHERE is_active = true
            ORDER BY trained_at DESC LIMIT 1
            """
        )
        if row and row[0] is not None:
            psi = float(row[0])
            sev = _severity_for("ml_psi_score", psi)
            if sev:
                alerts.append(_make_alert(
                    sev, "ml", "ML Model Drift Detected",
                    f"PSI score {psi:.3f} indicates model drift",
                    PRODUCTION_THRESHOLDS["ml_psi_score"][sev],
                    psi,
                ))
    except Exception:
        pass
    return alerts


def _check_sri() -> List[ProductionAlert]:
    alerts: List[ProductionAlert] = []
    if not db.db_read_enabled():
        return alerts
    try:
        row = db.safe_fetchone(
            """
            SELECT sri FROM market_breadth_daily
            ORDER BY as_of_date DESC LIMIT 1
            """
        )
        if row and row[0] is not None:
            sri = float(row[0])
            sev = _severity_for("sri_level", sri)
            if sev:
                alerts.append(_make_alert(
                    sev, "business", "SRI Risk Level Elevated",
                    f"Systemic Risk Index at {sri:.1f}",
                    PRODUCTION_THRESHOLDS["sri_level"][sev],
                    sri,
                ))
    except Exception:
        pass
    return alerts


def _check_data_staleness() -> List[ProductionAlert]:
    alerts: List[ProductionAlert] = []
    if not db.db_read_enabled():
        return alerts
    try:
        row = db.safe_fetchone(
            """
            SELECT EXTRACT(EPOCH FROM (NOW() - MAX(as_of_date::timestamptz))) / 3600
            FROM axiom_scores_daily
            """
        )
        if row and row[0] is not None:
            hours = float(row[0])
            sev = _severity_for("data_staleness_hours", hours)
            if sev:
                alerts.append(_make_alert(
                    sev, "database", "Data Staleness Alert",
                    f"Most recent AXIOM scores are {hours:.1f} hours old",
                    PRODUCTION_THRESHOLDS["data_staleness_hours"][sev],
                    hours,
                ))
    except Exception:
        pass
    return alerts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_production_health() -> Dict[str, Any]:
    alerts: List[ProductionAlert] = []
    alerts.extend(_check_api_metrics())
    alerts.extend(_check_pipeline_failures())
    alerts.extend(_check_ml_drift())
    alerts.extend(_check_sri())
    alerts.extend(_check_data_staleness())

    critical_count = sum(1 for a in alerts if a.severity == "critical")
    warning_count  = sum(1 for a in alerts if a.severity == "warning")

    if critical_count > 0:
        overall_status = "critical"
        recommendation = "Immediate action required — critical thresholds breached"
    elif warning_count > 0:
        overall_status = "degraded"
        recommendation = "System degraded — review warnings before peak trading hours"
    else:
        overall_status = "healthy"
        recommendation = "All production thresholds within acceptable range"

    return {
        "overall_status": overall_status,
        "alerts": alerts,
        "thresholds_checked": len(PRODUCTION_THRESHOLDS),
        "thresholds_breached": len(alerts),
        "recommendation": recommendation,
    }


def send_alert_notification(alert: ProductionAlert) -> bool:
    if alert.severity == "page":
        pagerduty_key = os.getenv("PAGERDUTY_KEY")
        if pagerduty_key:
            logger.critical(
                "PAGERDUTY_PAGE alert_id=%s title=%s actual=%.2f",
                alert.alert_id, alert.title, alert.actual_value,
            )
            return True
        logger.critical(
            "ALERT[page] %s: %s (actual=%.2f, threshold=%.2f)",
            alert.title, alert.description, alert.actual_value, alert.threshold_value,
        )
        return False

    if alert.severity == "critical":
        slack_webhook = os.getenv("SLACK_WEBHOOK")
        if slack_webhook:
            logger.critical(
                "SLACK_CRITICAL alert_id=%s title=%s actual=%.2f",
                alert.alert_id, alert.title, alert.actual_value,
            )
            return True
        logger.critical(
            "ALERT[critical] %s: %s (actual=%.2f, threshold=%.2f)",
            alert.title, alert.description, alert.actual_value, alert.threshold_value,
        )
        return False

    logger.warning(
        "ALERT[%s] %s: %s (actual=%.2f, threshold=%.2f)",
        alert.severity, alert.title, alert.description,
        alert.actual_value, alert.threshold_value,
    )
    return False
