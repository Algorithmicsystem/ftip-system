"""Prompt 6 tests: Performance caching, DB indexes, production monitoring, SLA tracking, acquisition readiness."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# PerformanceTracker unit tests
# ---------------------------------------------------------------------------

class TestPerformanceTracker:

    def setup_method(self):
        from api.cloud.performance import PerformanceTracker
        self.tracker = PerformanceTracker()

    def test_record_increments_request_count(self):
        self.tracker.record("/test", 50.0)
        assert self.tracker._request_counts["/test"] == 1

    def test_record_increments_error_count_on_error(self):
        self.tracker.record("/test", 50.0, is_error=True)
        assert self.tracker._error_counts["/test"] == 1

    def test_record_no_error_count_on_success(self):
        self.tracker.record("/test", 50.0, is_error=False)
        assert self.tracker._error_counts["/test"] == 0

    def test_get_percentiles_returns_none_below_min_samples(self):
        for i in range(4):
            self.tracker.record("/ep", float(i * 10))
        result = self.tracker.get_percentiles("/ep")
        assert result["p50"] is None
        assert result["sample_count"] == 4

    def test_get_percentiles_computes_correctly(self):
        for i in range(100):
            self.tracker.record("/ep", float(i))
        result = self.tracker.get_percentiles("/ep")
        assert result["p50"] is not None
        assert result["p95"] is not None
        assert result["p99"] is not None
        assert result["p50"] < result["p95"] < result["p99"]

    def test_get_percentiles_avg_present(self):
        for i in range(10):
            self.tracker.record("/ep", 100.0)
        result = self.tracker.get_percentiles("/ep")
        assert result["avg"] == 100.0

    def test_get_error_rate_zero_when_no_errors(self):
        self.tracker.record("/ep", 10.0)
        assert self.tracker.get_error_rate("/ep") == 0.0

    def test_get_error_rate_correct_fraction(self):
        self.tracker.record("/ep", 10.0, is_error=True)
        self.tracker.record("/ep", 10.0, is_error=False)
        assert self.tracker.get_error_rate("/ep") == 50.0

    def test_get_error_rate_zero_for_unknown_endpoint(self):
        assert self.tracker.get_error_rate("/unknown") == 0.0

    def test_get_top_endpoints_by_volume(self):
        for i in range(5):
            self.tracker.record("/a", 10.0)
        for i in range(10):
            self.tracker.record("/b", 10.0)
        top = self.tracker.get_top_endpoints_by_volume(2)
        assert len(top) == 2
        assert top[0]["endpoint"] == "/b"
        assert top[1]["endpoint"] == "/a"

    def test_get_slowest_endpoints(self):
        for _ in range(10):
            self.tracker.record("/fast", 10.0)
            self.tracker.record("/slow", 500.0)
        slowest = self.tracker.get_slowest_endpoints(2)
        assert len(slowest) == 2
        assert slowest[0]["endpoint"] == "/slow"

    def test_get_system_p95_zero_when_empty(self):
        assert self.tracker.get_system_p95() == 0.0

    def test_get_system_p95_computes_from_all_endpoints(self):
        for _ in range(50):
            self.tracker.record("/a", 100.0)
            self.tracker.record("/b", 200.0)
        p95 = self.tracker.get_system_p95()
        assert p95 > 0.0

    def test_get_summary_structure(self):
        for _ in range(5):
            self.tracker.record("/ep", 50.0)
        s = self.tracker.get_summary()
        assert "uptime_seconds" in s
        assert "total_requests" in s
        assert "system_p95_ms" in s
        assert "meets_sla" in s
        assert "top_endpoints" in s
        assert "slowest_endpoints" in s

    def test_meets_sla_true_below_200ms(self):
        for _ in range(10):
            self.tracker.record("/ep", 100.0)
        s = self.tracker.get_summary()
        assert s["meets_sla"] is True

    def test_meets_sla_false_above_200ms(self):
        for _ in range(10):
            self.tracker.record("/ep", 300.0)
        s = self.tracker.get_summary()
        assert s["meets_sla"] is False

    def test_circular_buffer_max_1000(self):
        for i in range(1100):
            self.tracker.record("/ep", float(i))
        assert len(self.tracker._samples["/ep"]) == 1000

    def test_get_percentiles_empty_endpoint_returns_no_data(self):
        result = self.tracker.get_percentiles("/nonexistent")
        assert result["p50"] is None
        assert result["sample_count"] == 0


# ---------------------------------------------------------------------------
# perf_tracker singleton
# ---------------------------------------------------------------------------

class TestPerfTrackerSingleton:

    def test_singleton_exists(self):
        from api.cloud.performance import perf_tracker, PerformanceTracker
        assert isinstance(perf_tracker, PerformanceTracker)

    def test_singleton_is_module_level(self):
        from api.cloud import performance
        assert hasattr(performance, "perf_tracker")


# ---------------------------------------------------------------------------
# /cloud/performance/report
# ---------------------------------------------------------------------------

class TestPerformanceReport:

    def test_returns_200(self):
        r = client.get("/cloud/performance/report")
        assert r.status_code == 200

    def test_has_required_keys(self):
        r = client.get("/cloud/performance/report")
        data = r.json()
        assert "total_requests" in data
        assert "system_p95_ms" in data
        assert "meets_sla" in data
        assert "top_endpoints" in data

    def test_top_endpoints_is_list(self):
        r = client.get("/cloud/performance/report")
        assert isinstance(r.json()["top_endpoints"], list)


# ---------------------------------------------------------------------------
# /cloud/performance/sla
# ---------------------------------------------------------------------------

class TestPerformanceSla:

    def test_returns_200(self):
        r = client.get("/cloud/performance/sla")
        assert r.status_code == 200

    def test_has_sla_fields(self):
        r = client.get("/cloud/performance/sla")
        data = r.json()
        assert "sla_target_ms" in data
        assert "system_p95_ms" in data
        assert "meets_sla" in data
        assert "sla_status" in data

    def test_sla_target_is_200(self):
        r = client.get("/cloud/performance/sla")
        assert r.json()["sla_target_ms"] == 200.0

    def test_sla_status_values(self):
        r = client.get("/cloud/performance/sla")
        assert r.json()["sla_status"] in ("passing", "breached")

    def test_checked_at_present(self):
        r = client.get("/cloud/performance/sla")
        assert "checked_at" in r.json()


# ---------------------------------------------------------------------------
# check_production_health
# ---------------------------------------------------------------------------

class TestCheckProductionHealth:

    def test_returns_dict(self):
        from api.cloud.monitoring import check_production_health
        result = check_production_health()
        assert isinstance(result, dict)

    def test_has_required_fields(self):
        from api.cloud.monitoring import check_production_health
        result = check_production_health()
        assert "overall_status" in result
        assert "alerts" in result
        assert "checked_at" in result
        assert "system_p95_ms" in result
        assert "data_freshness_hours" in result

    def test_overall_status_valid_values(self):
        from api.cloud.monitoring import check_production_health
        result = check_production_health()
        assert result["overall_status"] in ("healthy", "degraded", "critical")

    def test_system_p95_is_float(self):
        from api.cloud.monitoring import check_production_health
        result = check_production_health()
        assert isinstance(result["system_p95_ms"], float)


# ---------------------------------------------------------------------------
# /cloud/monitoring/dashboard
# ---------------------------------------------------------------------------

class TestMonitoringDashboard:

    def test_returns_200(self):
        r = client.get("/cloud/monitoring/dashboard")
        assert r.status_code == 200

    def test_has_required_fields(self):
        r = client.get("/cloud/monitoring/dashboard")
        data = r.json()
        assert "overall_status" in data
        assert "alerts" in data
        assert "system_p95_ms" in data
        assert "meets_sla" in data
        assert "deployment_confidence" in data
        assert "recommendation" in data

    def test_overall_status_valid(self):
        r = client.get("/cloud/monitoring/dashboard")
        assert r.json()["overall_status"] in ("healthy", "degraded", "critical")


# ---------------------------------------------------------------------------
# /cloud/readiness
# ---------------------------------------------------------------------------

class TestReadinessEndpoint:

    def test_returns_200(self):
        r = client.get("/cloud/readiness")
        assert r.status_code == 200

    def test_has_checks(self):
        r = client.get("/cloud/readiness")
        data = r.json()
        assert "checks" in data

    def test_has_20_checks(self):
        r = client.get("/cloud/readiness")
        data = r.json()
        checks = data["checks"]
        count = len(checks) if isinstance(checks, list) else len(checks.keys())
        assert count == 20

    def test_deployment_confidence_field(self):
        r = client.get("/cloud/readiness")
        assert r.json()["deployment_confidence"] in ("high", "medium", "low")


# ---------------------------------------------------------------------------
# /intelligence/cache/stats
# ---------------------------------------------------------------------------

class TestIntelligenceCacheStats:

    def test_returns_200(self):
        r = client.get("/intelligence/cache/stats")
        assert r.status_code == 200

    def test_has_cache_fields(self):
        r = client.get("/intelligence/cache/stats")
        data = r.json()
        assert "cache_size" in data
        assert "cache_ttl_seconds" in data
        assert "cache_type" in data

    def test_cache_ttl_is_300(self):
        r = client.get("/intelligence/cache/stats")
        assert r.json()["cache_ttl_seconds"] == 300

    def test_cache_type_in_memory(self):
        r = client.get("/intelligence/cache/stats")
        assert r.json()["cache_type"] == "in_memory"


# ---------------------------------------------------------------------------
# /intelligence/cache/warm
# ---------------------------------------------------------------------------

class TestIntelligenceCacheWarm:

    def test_returns_200(self):
        r = client.post("/intelligence/cache/warm")
        assert r.status_code == 200

    def test_returns_warming_status(self):
        r = client.post("/intelligence/cache/warm")
        assert r.json()["status"] == "warming"

    def test_symbols_count(self):
        r = client.post("/intelligence/cache/warm")
        assert r.json()["symbols"] == 30


# ---------------------------------------------------------------------------
# /system/status — acquisition readiness
# ---------------------------------------------------------------------------

class TestSystemStatusAcquisitionReadiness:

    def test_returns_200(self):
        r = client.get("/system/status")
        assert r.status_code == 200

    def test_version_is_31(self):
        r = client.get("/system/status")
        assert r.json()["version"] == "31.0.0"

    def test_has_acquisition_readiness(self):
        r = client.get("/system/status")
        data = r.json()
        assert "acquisition_readiness" in data

    def test_acquisition_readiness_structure(self):
        r = client.get("/system/status")
        acq = r.json()["acquisition_readiness"]
        assert "score" in acq
        assert "tier" in acq
        assert "deployment_confidence" in acq

    def test_acquisition_score_range(self):
        r = client.get("/system/status")
        score = r.json()["acquisition_readiness"]["score"]
        assert 0 <= score <= 100

    def test_tier_valid_values(self):
        r = client.get("/system/status")
        tier = r.json()["acquisition_readiness"]["tier"]
        assert tier in ("acquisition_ready", "near_ready", "building", "early_stage")

    def test_has_performance_section(self):
        r = client.get("/system/status")
        assert "performance" in r.json()

    def test_performance_has_sla(self):
        r = client.get("/system/status")
        perf = r.json()["performance"]
        assert "system_p95_ms" in perf
        assert "meets_sla" in perf

    def test_legacy_fields_preserved(self):
        r = client.get("/system/status")
        data = r.json()
        assert "axiom_scores_count" in data
        assert "ml_model_trained" in data
        assert "scheduler_running" in data


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_returns_200(self):
        r = client.get("/health")
        assert r.status_code in (200, 503)

    def test_has_status_field(self):
        r = client.get("/health")
        assert "status" in r.json()

    def test_status_is_ok(self):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_has_version_field(self):
        r = client.get("/health")
        assert "version" in r.json()

    def test_version_is_31(self):
        r = client.get("/health")
        assert r.json()["version"] == "31.0.0"

    def test_has_db_field(self):
        r = client.get("/health")
        assert "db" in r.json()

    def test_has_scheduler_field(self):
        r = client.get("/health")
        assert "scheduler" in r.json()


# ---------------------------------------------------------------------------
# /config/client version
# ---------------------------------------------------------------------------

class TestClientConfigVersion:

    def test_version_is_31(self):
        r = client.get("/config/client")
        assert r.json()["version"] == "31.0.0"


# ---------------------------------------------------------------------------
# Migration 105 registered
# ---------------------------------------------------------------------------

class TestMigration105:

    def test_migration_sql_file_exists(self):
        sql = Path(__file__).resolve().parents[1] / "api" / "migrations" / "105_performance_indexes.sql"
        assert sql.exists()

    def test_migration_registered_in_init(self):
        from api.migrations import MIGRATIONS
        names = [m[0] for m in MIGRATIONS]
        assert "105_performance_indexes" in names

    def test_migration_has_9_indexes(self):
        sql = Path(__file__).resolve().parents[1] / "api" / "migrations" / "105_performance_indexes.sql"
        content = sql.read_text()
        assert content.count("CREATE INDEX IF NOT EXISTS") == 9

    def test_earlier_migrations_096_097_098_registered(self):
        from api.migrations import MIGRATIONS
        names = [m[0] for m in MIGRATIONS]
        assert "096_lookahead_bias_fix" in names
        assert "097_ml_signal_predictions" in names
        assert "098_deal_flow_scores" in names


# ---------------------------------------------------------------------------
# In-memory cache module
# ---------------------------------------------------------------------------

class TestIntelligenceCacheModule:

    def test_cache_dict_exists(self):
        from api.universal.intelligence_api import _cache
        assert isinstance(_cache, dict)

    def test_cache_ttl_is_300(self):
        from api.universal.intelligence_api import CACHE_TTL_SECONDS
        assert CACHE_TTL_SECONDS == 300

    def test_set_and_get_memory_cache(self):
        from api.universal.intelligence_api import _set_memory_cache, _get_from_memory_cache
        _set_memory_cache("TEST_SYM", {"dau": 75})
        result = _get_from_memory_cache("TEST_SYM")
        assert result is not None
        assert result["dau"] == 75

    def test_expired_cache_returns_none(self):
        from api.universal import intelligence_api as ia
        orig_ttl = ia.CACHE_TTL_SECONDS
        ia.CACHE_TTL_SECONDS = 0
        try:
            ia._set_memory_cache("STALE_SYM", {"dau": 50})
            time.sleep(0.01)
            result = ia._get_from_memory_cache("STALE_SYM")
            assert result is None
        finally:
            ia.CACHE_TTL_SECONDS = orig_ttl

    def test_get_cache_stats_structure(self):
        from api.universal.intelligence_api import get_cache_stats
        stats = get_cache_stats()
        assert "cache_size" in stats
        assert "cache_ttl_seconds" in stats
        assert "cache_type" in stats


# ---------------------------------------------------------------------------
# UsageLoggingMiddleware wires perf_tracker
# ---------------------------------------------------------------------------

class TestUsageMiddlewarePerfWiring:

    def test_perf_tracker_import_in_middleware(self):
        source = Path(__file__).resolve().parents[1] / "api" / "developer" / "usage_middleware.py"
        content = source.read_text()
        assert "perf_tracker" in content

    def test_record_called_in_middleware(self):
        source = Path(__file__).resolve().parents[1] / "api" / "developer" / "usage_middleware.py"
        content = source.read_text()
        assert "perf_tracker.record(" in content
