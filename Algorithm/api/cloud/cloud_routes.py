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
    from api.cloud.performance import compute_system_performance_report
    result = compute_system_performance_report(lookback_minutes)
    # Serialize PerformanceMetrics dataclasses
    result["top_endpoints"] = [asdict(m) for m in result["top_endpoints"]]
    return result


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
