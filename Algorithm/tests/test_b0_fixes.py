"""B0 production fixes: pipeline auth, scheduler watchdog, schema status display, p95 warmup."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

WEBAPP = Path(__file__).resolve().parents[1] / "api" / "webapp"


# ---------------------------------------------------------------------------
# Fix 1 — Pipeline trigger auth: primary key = enterprise tier
# ---------------------------------------------------------------------------

class TestPipelineTriggerAuth:

    def test_primary_key_is_enterprise_tier_bypass_exists(self):
        src = Path(__file__).parents[1].joinpath("api", "jobs", "tenant_auth.py").read_text()
        assert "primary_key" in src
        assert "enterprise" in src
        assert "get_api_key" in src

    def test_run_pipeline_js_uses_api_post(self):
        js = (WEBAPP / "js" / "panels" / "system_status.js").read_text()
        assert "API.post" in js

    def test_require_tier_allows_primary_key_without_db(self):
        from api.jobs.tenant_auth import require_tier
        src = inspect.getsource(require_tier)
        assert "primary_key" in src
        assert "enterprise" in src

    def test_pipeline_endpoint_not_401_when_no_db(self):
        # When DB is disabled (test env), require_tier returns None → no 401
        with TestClient(app) as client:
            r = client.post(
                "/orchestration/pipeline/run",
                json={},
            )
        assert r.status_code != 401, "Pipeline endpoint must not return 401 in DB-disabled mode"

    def test_pipeline_endpoint_allows_primary_key(self):
        key = os.environ.get("FTIP_API_KEYS", os.environ.get("FTIP_API_KEY", ""))
        if not key:
            pytest.skip("No API key in test environment")
        with TestClient(app) as client:
            r = client.post(
                "/orchestration/pipeline/run",
                json={},
                headers={"X-FTIP-API-Key": key.split(",")[0].strip()},
            )
        assert r.status_code != 401, "Primary key must not get 401 on pipeline trigger"


# ---------------------------------------------------------------------------
# Fix 2 — Scheduler watchdog
# ---------------------------------------------------------------------------

class TestSchedulerWatchdog:

    def test_watchdog_function_exists(self):
        from api.jobs import scheduler as sched_mod
        src = inspect.getsource(sched_mod)
        assert "_scheduler_watchdog" in src

    def test_start_scheduler_starts_watchdog(self):
        src = inspect.getsource(__import__("api.jobs.scheduler", fromlist=["start_scheduler"]).start_scheduler)
        assert "watchdog" in src

    def test_scheduler_manager_has_running_property(self):
        from api.jobs.scheduler import scheduler_manager
        assert hasattr(scheduler_manager, "running")

    def test_scheduler_stop_no_op_when_not_started(self):
        from api.jobs.scheduler import SchedulerManager
        mgr = SchedulerManager()
        mgr.stop()  # must not raise and must not log spuriously
        assert not mgr.running

    def test_manual_trigger_comment_in_source(self):
        src = Path(__file__).parents[1].joinpath("api", "jobs", "scheduler.py").read_text()
        assert "_job_full_daily_pipeline" in src
        assert "Railway" in src or "manually" in src or "console" in src

    def test_scheduler_suppressed_in_pytest(self):
        from api.jobs.scheduler import start_scheduler
        src = inspect.getsource(start_scheduler)
        assert "pytest" in src


# ---------------------------------------------------------------------------
# Fix 3 — Schema / migrations display
# ---------------------------------------------------------------------------

class TestMigrationsDisplay:

    def test_system_status_has_migrations_applied_field(self):
        with TestClient(app) as client:
            r = client.get("/system/status")
        assert r.status_code == 200
        assert "migrations_applied" in r.json()

    def test_system_status_migrations_is_int(self):
        with TestClient(app) as client:
            r = client.get("/system/status")
        val = r.json()["migrations_applied"]
        assert isinstance(val, int)

    def test_system_status_js_shows_schema_label(self):
        js = (WEBAPP / "js" / "panels" / "system_status.js").read_text()
        assert "Schema" in js or "schema" in js

    def test_system_status_js_shows_up_to_date(self):
        js = (WEBAPP / "js" / "panels" / "system_status.js").read_text()
        assert "Up to date" in js or "up to date" in js

    def test_system_status_js_shows_applied_count(self):
        js = (WEBAPP / "js" / "panels" / "system_status.js").read_text()
        assert "applied" in js

    def test_status_cache_globals_exist_in_main(self):
        import api.main as m
        src = inspect.getsource(m)
        assert "_STATUS_CACHE" in src
        assert "_STATUS_CACHE_TTL" in src


# ---------------------------------------------------------------------------
# Fix 4 — P95 warmup exclusion
# ---------------------------------------------------------------------------

class TestP95WarmupExclusion:

    def test_performance_tracker_has_warmup_param(self):
        from api.cloud.performance import PerformanceTracker
        sig = inspect.signature(PerformanceTracker.get_system_p95)
        assert "warmup_seconds" in sig.parameters

    def test_warmup_samples_excluded(self):
        from api.cloud.performance import PerformanceTracker
        import time
        tracker = PerformanceTracker()
        # Manually backdate _start_time to simulate post-warmup
        tracker._start_time = time.time() - 60
        tracker.record("/test", 100.0)
        tracker.record("/test", 200.0)
        tracker.record("/test", 300.0)
        tracker.record("/test", 400.0)
        tracker.record("/test", 500.0)
        p95 = tracker.get_system_p95(warmup_seconds=30.0)
        assert p95 > 0, "Should have samples after warmup"

    def test_warmup_samples_excluded_within_window(self):
        from api.cloud.performance import PerformanceTracker
        import time
        tracker = PerformanceTracker()
        # _start_time is fresh (just now) — all samples are in warmup window
        for _ in range(10):
            tracker.record("/test", 999.0)
        p95 = tracker.get_system_p95(warmup_seconds=30.0)
        assert p95 == 0.0, "All samples during warmup should be excluded"

    def test_scheduler_running_check_uses_manager(self):
        import api.main as m
        src = inspect.getsource(m.health)
        assert "scheduler_manager" in src or "_sm" in src
        assert 'getattr(sched_module, "_scheduler"' not in src
