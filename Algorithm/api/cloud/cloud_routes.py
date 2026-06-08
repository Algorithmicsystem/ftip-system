"""Phase 22: Cloud infrastructure endpoints under /cloud/ prefix."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from fastapi import APIRouter

router = APIRouter(prefix="/cloud", tags=["cloud"])


# ---------------------------------------------------------------------------
# 22.1 Config
# ---------------------------------------------------------------------------

@router.get("/config")
def get_cloud_config() -> Dict[str, Any]:
    from api.cloud.config_production import get_production_config, validate_production_secrets
    cfg = get_production_config()
    secrets = validate_production_secrets()
    return {
        "config": cfg,
        "secrets_validation": secrets,
    }


# ---------------------------------------------------------------------------
# 22.2 DB Pool
# ---------------------------------------------------------------------------

@router.get("/db/pool-stats")
def get_db_pool_stats() -> Dict[str, Any]:
    from api.cloud.db_production import get_db_pool_stats
    return get_db_pool_stats()


# ---------------------------------------------------------------------------
# 22.3 Monitoring
# ---------------------------------------------------------------------------

@router.get("/monitoring/health")
def get_monitoring_health() -> Dict[str, Any]:
    from api.cloud.monitoring import check_production_health
    result = check_production_health()
    # Serialize ProductionAlert dataclasses
    result["alerts"] = [asdict(a) for a in result["alerts"]]
    return result


@router.get("/monitoring/alerts")
def get_monitoring_alerts() -> Dict[str, Any]:
    from api.cloud.monitoring import check_production_health
    result = check_production_health()
    return {
        "alerts": [asdict(a) for a in result["alerts"]],
        "overall_status": result["overall_status"],
        "count": len(result["alerts"]),
    }


@router.get("/monitoring/thresholds")
def get_monitoring_thresholds() -> Dict[str, Any]:
    from api.cloud.monitoring import PRODUCTION_THRESHOLDS
    return {"thresholds": PRODUCTION_THRESHOLDS}


# ---------------------------------------------------------------------------
# 22.4 Performance
# ---------------------------------------------------------------------------

@router.get("/performance/report")
def get_performance_report(lookback_minutes: int = 60) -> Dict[str, Any]:
    from api.cloud.performance import perf_tracker, compute_system_performance_report
    summary = perf_tracker.get_summary()
    # Enrich with DB data if available
    db_enriched = compute_system_performance_report(lookback_minutes)
    if "db_endpoints" in db_enriched:
        summary["db_endpoints"] = db_enriched["db_endpoints"]
    return summary


@router.get("/performance/sla")
def get_performance_sla() -> Dict[str, Any]:
    from api.cloud.performance import perf_tracker
    system_p95 = perf_tracker.get_system_p95()
    sla_target_ms = 200.0
    meets_sla = system_p95 < sla_target_ms
    total_requests = sum(perf_tracker._request_counts.values())
    total_errors = sum(perf_tracker._error_counts.values())
    return {
        "sla_target_ms": sla_target_ms,
        "system_p95_ms": round(system_p95, 1),
        "meets_sla": meets_sla,
        "sla_status": "passing" if meets_sla else "breached",
        "total_requests": total_requests,
        "total_errors": total_errors,
        "overall_error_rate_pct": round(total_errors / max(total_requests, 1) * 100, 3),
        "slowest_endpoints": perf_tracker.get_slowest_endpoints(5),
        "checked_at": __import__("datetime").datetime.utcnow().isoformat(),
    }


@router.get("/monitoring/dashboard")
def get_monitoring_dashboard() -> Dict[str, Any]:
    from dataclasses import asdict as _asdict
    from api.cloud.monitoring import check_production_health
    from api.cloud.performance import perf_tracker
    from api.cloud.readiness_check import run_production_readiness_check

    health = check_production_health()
    sla = perf_tracker.get_summary()
    readiness = run_production_readiness_check()

    return {
        "checked_at": health.get("checked_at"),
        "overall_status": health["overall_status"],
        "alerts": [_asdict(a) for a in health["alerts"]],
        "alert_count": len(health["alerts"]),
        "system_p95_ms": sla.get("system_p95_ms", 0.0),
        "meets_sla": sla.get("meets_sla", False),
        "total_requests": sla.get("total_requests", 0),
        "overall_error_rate_pct": sla.get("overall_error_rate_pct", 0.0),
        "data_freshness_hours": health.get("data_freshness_hours"),
        "deployment_confidence": readiness.get("deployment_confidence", "unknown"),
        "readiness_passed": readiness.get("passed", 0),
        "readiness_total": readiness.get("passed", 0) + readiness.get("failed", 0),
        "recommendation": health["recommendation"],
    }


@router.get("/performance/endpoint/{endpoint_path:path}")
def get_endpoint_performance(endpoint_path: str, lookback_minutes: int = 60) -> Dict[str, Any]:
    from api.cloud.performance import compute_endpoint_performance_metrics
    m = compute_endpoint_performance_metrics(endpoint_path, lookback_minutes)
    return asdict(m)


# ---------------------------------------------------------------------------
# 22.6 Readiness
# ---------------------------------------------------------------------------

@router.get("/readiness")
def get_readiness_check() -> Dict[str, Any]:
    from api.cloud.readiness_check import run_production_readiness_check
    return run_production_readiness_check()


@router.get("/readiness/summary")
def get_readiness_summary() -> Dict[str, Any]:
    from api.cloud.readiness_check import run_production_readiness_check
    result = run_production_readiness_check()
    return {
        "ready_for_production": result["ready_for_production"],
        "deployment_confidence": result["deployment_confidence"],
        "passed": result["passed"],
        "failed": result["failed"],
    }
