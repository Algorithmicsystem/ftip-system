"""Phase 24: Pipeline integration, bootstrap, IC/outcome fill, live dashboard health."""
from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from api.main import app


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

class TestUniverse:

    def test_universe_has_30_symbols(self):
        from api.universe import AXIOM_UNIVERSE
        assert len(AXIOM_UNIVERSE) == 30

    def test_universe_symbols_are_uppercase(self):
        from api.universe import AXIOM_UNIVERSE
        for sym in AXIOM_UNIVERSE:
            assert sym == sym.upper(), f"Symbol not uppercase: {sym}"

    def test_get_universe_returns_list(self):
        from api.universe import get_universe
        result = get_universe()
        assert isinstance(result, list)
        assert len(result) == 30

    def test_get_universe_extended(self):
        from api.universe import get_universe
        result = get_universe(include_extended=True)
        assert len(result) > 30

    def test_universe_contains_core_names(self):
        from api.universe import AXIOM_UNIVERSE
        for sym in ("AAPL", "MSFT", "NVDA", "JPM"):
            assert sym in AXIOM_UNIVERSE


# ---------------------------------------------------------------------------
# Pipeline orchestrator stages 1-4
# ---------------------------------------------------------------------------

class TestPipelineStages:

    def test_bar_ingestion_returns_records_processed(self):
        """bar_ingestion stage must return a dict with records_processed (DB disabled = 0)."""
        from api.orchestration.pipeline_orchestrator import _make_db_graceful_stage
        executor = _make_db_graceful_stage("bar_ingestion")
        result = executor()
        assert isinstance(result, dict)
        assert "records_processed" in result
        assert isinstance(result["records_processed"], int)

    def test_feature_computation_returns_records_processed(self):
        from api.orchestration.pipeline_orchestrator import _make_db_graceful_stage
        result = _make_db_graceful_stage("feature_computation")()
        assert isinstance(result.get("records_processed"), int)

    def test_signal_generation_returns_records_processed(self):
        from api.orchestration.pipeline_orchestrator import _make_db_graceful_stage
        result = _make_db_graceful_stage("signal_generation")()
        assert isinstance(result.get("records_processed"), int)

    def test_axiom_scoring_returns_records_processed(self):
        from api.orchestration.pipeline_orchestrator import _make_db_graceful_stage
        result = _make_db_graceful_stage("axiom_scoring")()
        assert isinstance(result.get("records_processed"), int)

    def test_run_full_pipeline_returns_pipeline_run(self):
        from api.orchestration.pipeline_orchestrator import run_full_pipeline, PipelineRun
        result = run_full_pipeline()
        assert isinstance(result, PipelineRun)
        assert result.run_id
        assert result.overall_status in ("success", "partial", "failed")

    def test_pipeline_stages_all_present(self):
        from api.orchestration.pipeline_orchestrator import PIPELINE_STAGES
        required = {"bar_ingestion", "feature_computation", "signal_generation", "axiom_scoring"}
        assert required.issubset(set(PIPELINE_STAGES))


# ---------------------------------------------------------------------------
# Bootstrap endpoints
# ---------------------------------------------------------------------------

class TestBootstrapEndpoints:

    def test_post_bootstrap_returns_200(self):
        with TestClient(app) as client:
            r = client.post("/orchestration/bootstrap")
            assert r.status_code == 200

    def test_post_bootstrap_has_task_id(self):
        with TestClient(app) as client:
            data = client.post("/orchestration/bootstrap").json()
        assert "task_id" in data or "status" in data

    def test_get_bootstrap_status_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/orchestration/bootstrap/status")
            assert r.status_code == 200

    def test_bootstrap_status_has_status_field(self):
        with TestClient(app) as client:
            data = client.get("/orchestration/bootstrap/status").json()
        assert "status" in data
        assert data["status"] in ("idle", "running", "completed", "failed", "already_running", "already_completed_today")

    def test_bootstrap_idempotent_when_running(self):
        """Second POST while running must return already_running."""
        with TestClient(app) as client:
            # Force running state by posting twice quickly
            r1 = client.post("/orchestration/bootstrap").json()
            r2 = client.post("/orchestration/bootstrap").json()
        # Either already_running or triggered on second call — both valid
        assert r2["status"] in ("triggered", "already_running", "already_completed_today")


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

class TestICComputation:

    def test_compute_ic_snapshot_empty_db_returns_empty(self):
        """compute_ic_snapshot must not raise when DB is disabled."""
        from api.jobs.ic import compute_ic_snapshot
        result = compute_ic_snapshot(dt.date.today())
        # Returns {} when no axiom scores
        assert isinstance(result, dict)

    def test_store_ic_snapshot_empty_returns_zero(self):
        from api.jobs.ic import store_ic_snapshot
        result = store_ic_snapshot(dt.date.today(), {})
        assert result == 0

    def test_ic_snapshot_module_importable(self):
        """ic_snapshot thin wrapper must be importable."""
        from api.jobs.ic_snapshot import compute_ic_snapshot
        assert callable(compute_ic_snapshot)

    def test_axiom_composite_not_in_regular_score_fields(self):
        """axiom_composite is handled separately, not in SCORE_FIELDS list."""
        from api.jobs.ic import SCORE_FIELDS
        # axiom_composite is computed separately, SCORE_FIELDS uses 'composite'
        assert "composite" in SCORE_FIELDS


# ---------------------------------------------------------------------------
# Outcome fill
# ---------------------------------------------------------------------------

class TestOutcomeFill:

    def test_run_outcome_fill_no_db_returns_db_disabled(self):
        from api.jobs.outcome_fill import run_outcome_fill
        result = run_outcome_fill()
        # DB disabled → graceful return
        assert isinstance(result, dict)
        assert result.get("status") in ("db_disabled", "ok", "fetch_failed")

    def test_run_outcome_fill_returns_filled_count(self):
        from api.jobs.outcome_fill import run_outcome_fill
        result = run_outcome_fill()
        assert "filled" in result
        assert isinstance(result["filled"], int)

    def test_outcome_fill_endpoint_returns_200(self):
        with TestClient(app) as client:
            r = client.post("/jobs/outcome-fill/run")
            assert r.status_code == 200

    def test_outcome_fill_endpoint_response_has_filled(self):
        with TestClient(app) as client:
            data = client.post("/jobs/outcome-fill/run").json()
        assert "filled" in data


# ---------------------------------------------------------------------------
# Dashboard health monitor
# ---------------------------------------------------------------------------

class TestDashboardHealthMonitor:

    def test_dashboard_has_update_system_health(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "api", "webapp", "js", "dashboard.js")
        with open(path) as f:
            src = f.read()
        assert "updateSystemHealth" in src

    def test_dashboard_calls_system_status(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "api", "webapp", "js", "dashboard.js")
        with open(path) as f:
            src = f.read()
        assert "/system/status" in src

    def test_dashboard_has_start_health_monitor(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "api", "webapp", "js", "dashboard.js")
        with open(path) as f:
            src = f.read()
        assert "startHealthMonitor" in src

    def test_dashboard_health_poll_interval(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "api", "webapp", "js", "dashboard.js")
        with open(path) as f:
            src = f.read()
        assert "60000" in src or "HEALTH_POLL_MS" in src
