"""Phase 22.6: Production Readiness Checklist."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

from api import db

logger = logging.getLogger(__name__)

_CRITICAL_CHECKS = {
    "database_connected",
    "migrations_current",
    "env_secrets_present",
    "axiom_scores_populated",
    "health_endpoint_responds",
    "audit_trail_active",
}


def _check(passed: bool, message: str, critical: bool = False) -> Dict[str, Any]:
    return {"passed": passed, "message": message, "critical": critical}


def _safe_count(query: str, params: tuple = ()) -> int:
    try:
        row = db.safe_fetchone(query, params or None)
        return int(row[0] or 0) if row else 0
    except Exception:
        return -1  # -1 signals DB error, distinct from 0


def run_production_readiness_check() -> Dict[str, Any]:
    checks: Dict[str, Dict[str, Any]] = {}
    db_ok = db.db_enabled()

    # ── Infrastructure checks ─────────────────────────────────────────────────

    # 1. database_connected
    if db_ok:
        try:
            row = db.safe_fetchone("SELECT 1")
            checks["database_connected"] = _check(row is not None, "Database responded to SELECT 1", critical=True)
        except Exception as exc:
            checks["database_connected"] = _check(False, f"Database connection failed: {exc}", critical=True)
    else:
        checks["database_connected"] = _check(False, "Database disabled (FTIP_DB_ENABLED not set)", critical=True)

    # 2. migrations_current
    if db_ok:
        try:
            row = db.safe_fetchone(
                "SELECT COUNT(*) FROM schema_migrations WHERE applied_at IS NOT NULL"
            )
            count = int(row[0] or 0) if row else 0
            checks["migrations_current"] = _check(
                count >= 90,
                f"Applied migrations: {count} (expected ≥ 90)",
                critical=True,
            )
        except Exception:
            checks["migrations_current"] = _check(False, "Could not query schema_migrations", critical=True)
    else:
        checks["migrations_current"] = _check(False, "DB disabled — cannot verify migrations", critical=True)

    # 3. pool_configured
    from api.cloud.config_production import get_production_config
    cfg = get_production_config()
    pool_size = cfg.get("db_pool_size", 1)
    checks["pool_configured"] = _check(
        pool_size > 1,
        f"DB pool configured at {pool_size} connections",
    )

    # 4. env_secrets_present
    from api.cloud.config_production import validate_production_secrets
    secret_result = validate_production_secrets()
    checks["env_secrets_present"] = _check(
        secret_result["ready_for_production"],
        "All required secrets present" if secret_result["ready_for_production"]
        else f"Missing secrets: {', '.join(secret_result['missing_required'])}",
        critical=True,
    )

    # ── Data checks ───────────────────────────────────────────────────────────

    # 5. axiom_scores_populated
    n = _safe_count("SELECT COUNT(*) FROM axiom_scores_daily")
    checks["axiom_scores_populated"] = _check(
        n > 0,
        f"axiom_scores_daily has {max(n,0)} rows" if n >= 0 else "DB query failed",
        critical=True,
    )

    # 6. universe_configured
    n = _safe_count("SELECT COUNT(DISTINCT symbol) FROM axiom_scores_daily")
    checks["universe_configured"] = _check(
        n >= 5,
        f"Universe has {max(n,0)} symbols" if n >= 0 else "DB query failed",
    )

    # 7. recent_data
    if db_ok:
        try:
            row = db.safe_fetchone(
                """
                SELECT EXTRACT(EPOCH FROM (NOW() - MAX(as_of_date::timestamptz)))/3600
                FROM axiom_scores_daily
                """
            )
            hours = float(row[0] or 999) if row and row[0] is not None else 999
            checks["recent_data"] = _check(
                hours < 48,
                f"Most recent score is {hours:.1f}h old",
            )
        except Exception:
            checks["recent_data"] = _check(False, "Could not query recency")
    else:
        checks["recent_data"] = _check(False, "DB disabled")

    # 8. ic_gate_operational
    n = _safe_count("SELECT COUNT(*) FROM signal_ic_daily")
    checks["ic_gate_operational"] = _check(
        n > 0,
        f"signal_ic_daily has {max(n,0)} rows" if n >= 0 else "DB query failed",
    )

    # ── ML checks ─────────────────────────────────────────────────────────────

    # 9. model_registered
    n = _safe_count("SELECT COUNT(*) FROM ml_model_registry")
    checks["model_registered"] = _check(
        n >= 1,
        f"ml_model_registry has {max(n,0)} model(s)" if n >= 0 else "Table may not exist",
    )

    # 10. drift_acceptable
    if db_ok and n >= 1:
        try:
            row = db.safe_fetchone(
                "SELECT psi_score FROM ml_model_registry WHERE is_active = true ORDER BY trained_at DESC LIMIT 1"
            )
            if row and row[0] is not None:
                psi = float(row[0])
                checks["drift_acceptable"] = _check(psi < 0.25, f"Active model PSI={psi:.3f}")
            else:
                checks["drift_acceptable"] = _check(True, "No active model PSI on record — acceptable")
        except Exception:
            checks["drift_acceptable"] = _check(True, "ML registry not queried")
    else:
        checks["drift_acceptable"] = _check(True, "No models registered — skipping drift check")

    # ── Intelligence checks ───────────────────────────────────────────────────

    # 11. morning_briefing_recent
    if db_ok:
        try:
            row = db.safe_fetchone(
                "SELECT EXTRACT(EPOCH FROM (NOW() - MAX(generated_at)))/3600 FROM morning_briefings"
            )
            hours = float(row[0] or 999) if row and row[0] is not None else 999
            checks["morning_briefing_recent"] = _check(
                hours < 25,
                f"Morning briefing generated {hours:.1f}h ago",
            )
        except Exception:
            checks["morning_briefing_recent"] = _check(False, "Could not query morning_briefings")
    else:
        checks["morning_briefing_recent"] = _check(False, "DB disabled")

    # 12. sri_computed
    n = _safe_count("SELECT COUNT(*) FROM market_breadth_daily WHERE sri IS NOT NULL")
    checks["sri_computed"] = _check(
        n > 0,
        f"market_breadth_daily has {max(n,0)} SRI rows" if n >= 0 else "DB query failed",
    )

    # 13. pipeline_ran_recently
    if db_ok:
        try:
            row = db.safe_fetchone(
                """
                SELECT EXTRACT(EPOCH FROM (NOW() - MAX(started_at)))/3600
                FROM pipeline_runs
                WHERE status = 'success'
                """
            )
            hours = float(row[0] or 999) if row and row[0] is not None else 999
            checks["pipeline_ran_recently"] = _check(
                hours < 26,
                f"Last successful pipeline run was {hours:.1f}h ago",
            )
        except Exception:
            checks["pipeline_ran_recently"] = _check(False, "Could not query pipeline_runs")
    else:
        checks["pipeline_ran_recently"] = _check(False, "DB disabled")

    # ── API checks ────────────────────────────────────────────────────────────

    # 14. health_endpoint_responds
    try:
        from fastapi.testclient import TestClient
        from api.main import app
        with TestClient(app) as client:
            r = client.get("/orchestration/health")
            checks["health_endpoint_responds"] = _check(
                r.status_code == 200,
                f"GET /orchestration/health → {r.status_code}",
                critical=True,
            )
    except Exception as exc:
        checks["health_endpoint_responds"] = _check(False, f"Health check failed: {exc}", critical=True)

    # 15. universal_endpoint_responds
    try:
        from fastapi.testclient import TestClient
        from api.main import app
        with TestClient(app) as client:
            r = client.get("/intelligence/universal/AAPL")
            checks["universal_endpoint_responds"] = _check(
                r.status_code in (200, 404),
                f"GET /intelligence/universal/AAPL → {r.status_code}",
            )
    except Exception as exc:
        checks["universal_endpoint_responds"] = _check(False, f"Universal endpoint failed: {exc}")

    # 16. docs_accessible
    try:
        from fastapi.testclient import TestClient
        from api.main import app
        with TestClient(app) as client:
            r = client.get("/developer/api-docs")
            checks["docs_accessible"] = _check(
                r.status_code == 200,
                f"GET /developer/api-docs → {r.status_code}",
            )
    except Exception as exc:
        checks["docs_accessible"] = _check(False, f"Docs endpoint failed: {exc}")

    # ── Compliance checks ─────────────────────────────────────────────────────

    # 17. audit_trail_active
    n = _safe_count("SELECT COUNT(*) FROM audit_trail")
    checks["audit_trail_active"] = _check(
        n > 0,
        f"audit_trail has {max(n,0)} records" if n >= 0 else "DB query failed",
        critical=True,
    )

    # 18. soc2_readiness_computed
    try:
        from api.compliance.soc2_readiness import assess_soc2_readiness
        result = assess_soc2_readiness()
        score = result.get("readiness_score", 0)
        checks["soc2_readiness_computed"] = _check(
            score > 0,
            f"SOC 2 readiness score: {score:.1f}%",
        )
    except Exception as exc:
        checks["soc2_readiness_computed"] = _check(False, f"SOC 2 check failed: {exc}")

    # ── Performance checks ────────────────────────────────────────────────────

    # 19. cache_populated
    n = _safe_count("SELECT COUNT(*) FROM universal_intelligence_cache WHERE expires_at > NOW()")
    checks["cache_populated"] = _check(
        n > 0,
        f"universal_intelligence_cache has {max(n,0)} live entries" if n >= 0 else "DB query failed",
    )

    # 20. scheduler_configured
    scheduler_set = bool(os.getenv("FTIP_SCHEDULER_ENABLED"))
    checks["scheduler_configured"] = _check(
        scheduler_set,
        "FTIP_SCHEDULER_ENABLED is set" if scheduler_set else "FTIP_SCHEDULER_ENABLED not set",
    )

    # ── Compute summary ───────────────────────────────────────────────────────
    passed  = sum(1 for c in checks.values() if c["passed"])
    failed  = sum(1 for c in checks.values() if not c["passed"])
    warnings = sum(1 for c in checks.values() if not c["passed"] and not c["critical"])
    critical_failures = sum(1 for c in checks.values() if not c["passed"] and c["critical"])

    if passed >= 18 and critical_failures == 0:
        confidence = "high"
    elif passed >= 14:
        confidence = "medium"
    else:
        confidence = "low"

    ready = critical_failures == 0 and passed >= 18

    return {
        "ready_for_production": ready,
        "passed": passed,
        "failed": failed,
        "warnings": warnings,
        "checks": checks,
        "deployment_confidence": confidence,
    }
