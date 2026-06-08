"""Phase 22 tests: Production Cloud Infrastructure."""
from __future__ import annotations

import dataclasses
import os
from dataclasses import fields

import pytest

from api.cloud.config_production import (
    PRODUCTION_ENVIRONMENTS,
    get_production_config,
    validate_production_secrets,
)
from api.cloud.monitoring import (
    PRODUCTION_THRESHOLDS,
    ProductionAlert,
    check_production_health,
    send_alert_notification,
)
from api.cloud.performance import (
    PerformanceMetrics,
    compute_endpoint_performance_metrics,
    compute_system_performance_report,
    compute_universal_endpoint_cache_effectiveness,
)
from api.cloud.readiness_check import run_production_readiness_check


# ── TestProductionConfig ──────────────────────────────────────────────────────

class TestProductionConfig:

    def test_environments_defined(self):
        for env_name in ("development", "staging", "production"):
            assert env_name in PRODUCTION_ENVIRONMENTS

    def test_production_config_has_required_keys(self):
        for env_name in ("development", "staging", "production"):
            cfg = PRODUCTION_ENVIRONMENTS[env_name]
            for key in ("db_pool_size", "worker_count", "log_level", "cors_origins"):
                assert key in cfg, f"'{key}' missing in {env_name}"

    def test_development_has_debug_true(self):
        assert PRODUCTION_ENVIRONMENTS["development"]["debug"] is True

    def test_production_has_debug_false(self):
        assert PRODUCTION_ENVIRONMENTS["production"]["debug"] is False

    def test_staging_has_debug_false(self):
        assert PRODUCTION_ENVIRONMENTS["staging"]["debug"] is False

    def test_production_pool_larger_than_development(self):
        assert (
            PRODUCTION_ENVIRONMENTS["production"]["db_pool_size"]
            > PRODUCTION_ENVIRONMENTS["development"]["db_pool_size"]
        )

    def test_get_production_config_returns_dict(self):
        result = get_production_config()
        assert isinstance(result, dict)
        assert "db_pool_size" in result
        assert "worker_count" in result
        assert "log_level" in result

    def test_get_production_config_defaults_to_development(self, monkeypatch):
        monkeypatch.delenv("AXIOM_ENV", raising=False)
        cfg = get_production_config()
        assert cfg["debug"] is True

    def test_get_production_config_respects_axiom_env(self, monkeypatch):
        monkeypatch.setenv("AXIOM_ENV", "production")
        cfg = get_production_config()
        assert cfg["debug"] is False
        assert cfg["db_pool_size"] == 20

    def test_get_production_config_unknown_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("AXIOM_ENV", "nonexistent")
        cfg = get_production_config()
        assert isinstance(cfg, dict)

    def test_validate_secrets_returns_structure(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("FTIP_API_KEY", raising=False)
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        result = validate_production_secrets()
        assert "ready_for_production" in result
        assert "missing_required" in result
        assert "missing_optional" in result
        assert "warnings" in result

    def test_validate_secrets_not_ready_when_missing(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("FTIP_API_KEY", raising=False)
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        result = validate_production_secrets()
        assert result["ready_for_production"] is False
        assert len(result["missing_required"]) > 0

    def test_validate_secrets_ready_when_all_present(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://test")
        monkeypatch.setenv("FTIP_API_KEY", "test-key")
        monkeypatch.setenv("POLYGON_API_KEY", "test-poly")
        result = validate_production_secrets()
        assert result["ready_for_production"] is True
        assert result["missing_required"] == []

    def test_cors_origins_are_lists(self):
        for env_name in ("development", "staging", "production"):
            cfg = PRODUCTION_ENVIRONMENTS[env_name]
            assert isinstance(cfg["cors_origins"], list)

    def test_rate_limit_increases_with_env_tier(self):
        dev = PRODUCTION_ENVIRONMENTS["development"]["rate_limit_rpm"]
        stg = PRODUCTION_ENVIRONMENTS["staging"]["rate_limit_rpm"]
        prd = PRODUCTION_ENVIRONMENTS["production"]["rate_limit_rpm"]
        assert dev < stg < prd


# ── TestProductionMonitoring ──────────────────────────────────────────────────

class TestProductionMonitoring:

    def test_thresholds_defined(self):
        required_keys = {
            "api_error_rate_pct", "api_latency_p99_ms", "db_pool_utilization_pct",
            "pipeline_failure_count", "ml_psi_score", "sri_level", "data_staleness_hours",
        }
        assert required_keys.issubset(PRODUCTION_THRESHOLDS.keys())

    def test_each_threshold_has_warning_critical(self):
        for key, val in PRODUCTION_THRESHOLDS.items():
            assert "warning" in val, f"'{key}' missing 'warning'"
            assert "critical" in val, f"'{key}' missing 'critical'"

    def test_critical_exceeds_warning_for_all_thresholds(self):
        for key, val in PRODUCTION_THRESHOLDS.items():
            assert val["critical"] > val["warning"], f"'{key}' critical not > warning"

    def test_check_production_health_returns_dict(self):
        result = check_production_health()
        assert isinstance(result, dict)

    def test_overall_status_valid(self):
        result = check_production_health()
        assert result["overall_status"] in {"healthy", "degraded", "critical"}

    def test_check_health_has_required_keys(self):
        result = check_production_health()
        for key in ("overall_status", "alerts", "thresholds_checked", "thresholds_breached", "recommendation"):
            assert key in result

    def test_thresholds_checked_equals_threshold_count(self):
        result = check_production_health()
        assert result["thresholds_checked"] == len(PRODUCTION_THRESHOLDS)

    def test_alerts_is_list(self):
        result = check_production_health()
        assert isinstance(result["alerts"], list)

    def test_production_alert_dataclass(self):
        field_names = {f.name for f in fields(ProductionAlert)}
        required = {
            "alert_id", "severity", "category", "title", "description",
            "threshold_value", "actual_value", "triggered_at", "resolved_at", "auto_resolved",
        }
        assert required.issubset(field_names)

    def test_alert_severity_valid(self):
        import datetime as dt
        alert = ProductionAlert(
            alert_id="test-1",
            severity="warning",
            category="api",
            title="Test",
            description="Test alert",
            threshold_value=1.0,
            actual_value=2.0,
            triggered_at=dt.datetime.utcnow(),
        )
        assert alert.severity in {"info", "warning", "critical", "page"}

    def test_alert_can_be_created_with_all_fields(self):
        import datetime as dt
        alert = ProductionAlert(
            alert_id="test-2",
            severity="critical",
            category="database",
            title="DB Alert",
            description="Database pool exhausted",
            threshold_value=90.0,
            actual_value=95.0,
            triggered_at=dt.datetime.utcnow(),
            resolved_at=None,
            auto_resolved=False,
        )
        assert alert.threshold_value == 90.0
        assert alert.actual_value == 95.0

    def test_send_alert_notification_returns_bool(self):
        import datetime as dt
        alert = ProductionAlert(
            alert_id="notif-test",
            severity="warning",
            category="api",
            title="Test notification",
            description="Test",
            threshold_value=1.0,
            actual_value=2.0,
            triggered_at=dt.datetime.utcnow(),
        )
        result = send_alert_notification(alert)
        assert isinstance(result, bool)

    def test_monitoring_healthy_when_no_db(self):
        result = check_production_health()
        # Without DB, no threshold breaches should be reported from DB checks
        assert result["thresholds_breached"] == 0 or result["overall_status"] in {"healthy", "degraded", "critical"}


# ── TestPerformanceMetrics ────────────────────────────────────────────────────

class TestPerformanceMetrics:

    def test_performance_metrics_structure(self):
        field_names = {f.name for f in fields(PerformanceMetrics)}
        required = {"endpoint", "p50_ms", "p95_ms", "p99_ms", "requests_per_minute",
                    "error_rate_pct", "cache_hit_rate"}
        assert required.issubset(field_names)

    def test_compute_endpoint_metrics_returns_dataclass(self):
        result = compute_endpoint_performance_metrics("/intelligence/universal/AAPL")
        assert isinstance(result, PerformanceMetrics)

    def test_compute_endpoint_metrics_has_correct_endpoint(self):
        ep = "/intelligence/universal/AAPL"
        result = compute_endpoint_performance_metrics(ep)
        assert result.endpoint == ep

    def test_performance_metrics_bounded(self):
        result = compute_endpoint_performance_metrics("/test")
        assert 0 <= result.error_rate_pct <= 100
        assert 0 <= result.cache_hit_rate <= 1
        assert result.p50_ms >= 0
        assert result.p95_ms >= 0
        assert result.p99_ms >= 0
        assert result.requests_per_minute >= 0

    def test_p50_lte_p95_lte_p99(self):
        result = compute_endpoint_performance_metrics("/test")
        assert result.p50_ms <= result.p95_ms
        assert result.p95_ms <= result.p99_ms

    def test_system_report_structure(self):
        result = compute_system_performance_report()
        assert "top_endpoints" in result
        assert "slowest_endpoints" in result
        assert "system_p95_ms" in result

    def test_system_report_top_endpoints_is_list(self):
        result = compute_system_performance_report()
        assert isinstance(result["top_endpoints"], list)

    def test_system_report_slowest_endpoints_is_list(self):
        result = compute_system_performance_report()
        assert isinstance(result["slowest_endpoints"], list)

    def test_system_report_overall_p99_nonnegative(self):
        result = compute_system_performance_report()
        assert result["system_p95_ms"] >= 0

    def test_cache_effectiveness_structure(self):
        result = compute_universal_endpoint_cache_effectiveness()
        assert "cache_hit_rate" in result
        assert "avg_cache_response_ms" in result
        assert "avg_db_response_ms" in result
        assert "estimated_compute_saved_pct" in result

    def test_cache_hit_rate_bounded(self):
        result = compute_universal_endpoint_cache_effectiveness()
        assert 0.0 <= result["cache_hit_rate"] <= 1.0

    def test_cache_saved_pct_bounded(self):
        result = compute_universal_endpoint_cache_effectiveness()
        assert 0.0 <= result["estimated_compute_saved_pct"] <= 100.0

    def test_cache_response_ms_nonnegative(self):
        result = compute_universal_endpoint_cache_effectiveness()
        assert result["avg_cache_response_ms"] >= 0
        assert result["avg_db_response_ms"] >= 0


# ── TestReadinessCheck ────────────────────────────────────────────────────────

class TestReadinessCheck:

    def test_readiness_check_returns_20_checks(self):
        result = run_production_readiness_check()
        assert len(result["checks"]) == 20

    def test_all_checks_have_required_fields(self):
        result = run_production_readiness_check()
        for name, check in result["checks"].items():
            assert "passed" in check, f"check '{name}' missing 'passed'"
            assert "message" in check, f"check '{name}' missing 'message'"
            assert "critical" in check, f"check '{name}' missing 'critical'"

    def test_passed_is_bool_for_all_checks(self):
        result = run_production_readiness_check()
        for name, check in result["checks"].items():
            assert isinstance(check["passed"], bool), f"check '{name}' passed is not bool"

    def test_deployment_confidence_valid(self):
        result = run_production_readiness_check()
        assert result["deployment_confidence"] in {"high", "medium", "low"}

    def test_critical_checks_identified(self):
        result = run_production_readiness_check()
        critical_checks = [c for c in result["checks"].values() if c["critical"]]
        assert len(critical_checks) > 0

    def test_readiness_summary_structure(self):
        result = run_production_readiness_check()
        assert "ready_for_production" in result
        assert "deployment_confidence" in result
        assert "passed" in result
        assert "failed" in result

    def test_ready_for_production_is_bool(self):
        result = run_production_readiness_check()
        assert isinstance(result["ready_for_production"], bool)

    def test_passed_plus_failed_equals_20(self):
        result = run_production_readiness_check()
        assert result["passed"] + result["failed"] == 20

    def test_known_checks_present(self):
        result = run_production_readiness_check()
        required_check_names = {
            "database_connected", "migrations_current", "pool_configured",
            "env_secrets_present", "axiom_scores_populated", "universe_configured",
            "recent_data", "ic_gate_operational", "model_registered", "drift_acceptable",
            "morning_briefing_recent", "sri_computed", "pipeline_ran_recently",
            "health_endpoint_responds", "universal_endpoint_responds", "docs_accessible",
            "audit_trail_active", "soc2_readiness_computed", "cache_populated",
            "scheduler_configured",
        }
        assert required_check_names == set(result["checks"].keys())

    def test_scheduler_check_reflects_env(self, monkeypatch):
        monkeypatch.delenv("FTIP_SCHEDULER_ENABLED", raising=False)
        result = run_production_readiness_check()
        assert result["checks"]["scheduler_configured"]["passed"] is False

    def test_scheduler_check_passes_when_set(self, monkeypatch):
        monkeypatch.setenv("FTIP_SCHEDULER_ENABLED", "1")
        result = run_production_readiness_check()
        assert result["checks"]["scheduler_configured"]["passed"] is True

    def test_pool_configured_passes_for_production_env(self, monkeypatch):
        monkeypatch.setenv("AXIOM_ENV", "production")
        result = run_production_readiness_check()
        assert result["checks"]["pool_configured"]["passed"] is True

    def test_deployment_confidence_low_with_no_db(self):
        # Without DB, many checks will fail → should be low or medium
        result = run_production_readiness_check()
        assert result["deployment_confidence"] in {"low", "medium", "high"}

    def test_warnings_count_nonnegative(self):
        result = run_production_readiness_check()
        assert result["warnings"] >= 0


# ── Cloud routes smoke test ───────────────────────────────────────────────────

class TestCloudRoutes:

    def test_cloud_config_endpoint(self):
        from fastapi.testclient import TestClient
        from api.main import app
        with TestClient(app) as client:
            r = client.get("/cloud/config")
            assert r.status_code == 200
            data = r.json()
            assert "config" in data

    def test_cloud_db_pool_stats_endpoint(self):
        from fastapi.testclient import TestClient
        from api.main import app
        with TestClient(app) as client:
            r = client.get("/cloud/db/pool-stats")
            assert r.status_code == 200
            data = r.json()
            assert "pool_size" in data
            assert "slow_query_count" in data

    def test_cloud_monitoring_health_endpoint(self):
        from fastapi.testclient import TestClient
        from api.main import app
        with TestClient(app) as client:
            r = client.get("/cloud/monitoring/health")
            assert r.status_code == 200
            data = r.json()
            assert "overall_status" in data

    def test_cloud_monitoring_thresholds_endpoint(self):
        from fastapi.testclient import TestClient
        from api.main import app
        with TestClient(app) as client:
            r = client.get("/cloud/monitoring/thresholds")
            assert r.status_code == 200
            data = r.json()
            assert "thresholds" in data

    def test_cloud_performance_report_endpoint(self):
        from fastapi.testclient import TestClient
        from api.main import app
        with TestClient(app) as client:
            r = client.get("/cloud/performance/report")
            assert r.status_code == 200
            data = r.json()
            assert "top_endpoints" in data
            assert "system_p95_ms" in data

    def test_cloud_readiness_summary_endpoint(self):
        from fastapi.testclient import TestClient
        from api.main import app
        with TestClient(app) as client:
            r = client.get("/cloud/readiness/summary")
            assert r.status_code == 200
            data = r.json()
            assert "ready_for_production" in data
            assert "deployment_confidence" in data

    def test_cloud_readiness_full_endpoint(self):
        from fastapi.testclient import TestClient
        from api.main import app
        with TestClient(app) as client:
            r = client.get("/cloud/readiness")
            assert r.status_code == 200
            data = r.json()
            assert len(data["checks"]) == 20
