"""Phase 17 tests: Intelligence Orchestration — universal API, pipeline, self-improvement, health."""
from __future__ import annotations

import datetime as dt

import pytest

from api.universal.intelligence_api import (
    UniversalIntelligenceResponse,
    _analyst_rating,
    assemble_universal_intelligence,
    cache_universal_response,
    get_cached_universal_response,
)
from api.orchestration.pipeline_orchestrator import (
    PIPELINE_STAGES,
    PipelineRun,
    StageResult,
    run_full_pipeline,
)
from api.orchestration.self_improvement import (
    SelfImprovementStatus,
    check_self_improvement_status,
    get_improvement_history,
    trigger_self_improvement,
)
from api.orchestration.system_health import (
    SystemHealth,
    compute_health_status,
    compute_system_health,
    get_health_history,
)


# ===========================================================================
# TestUniversalIntelligence
# ===========================================================================

class TestUniversalIntelligence:
    def test_assemble_db_disabled(self):
        resp = assemble_universal_intelligence("AAPL")
        assert isinstance(resp, UniversalIntelligenceResponse)
        assert resp.symbol == "AAPL"
        # None-valued optional fields — no error
        assert resp.osms_score is None or isinstance(resp.osms_score, float)
        assert resp.var_1d_99 is None or isinstance(resp.var_1d_99, float)

    def test_analyst_rating_strong_buy(self):
        assert _analyst_rating(85.0, "BUY") == "Strong Buy"

    def test_analyst_rating_buy(self):
        assert _analyst_rating(70.0, "BUY") == "Buy"

    def test_analyst_rating_strong_sell(self):
        assert _analyst_rating(25.0, "SELL") == "Strong Sell"

    def test_analyst_rating_sell(self):
        assert _analyst_rating(45.0, "SELL") == "Sell"

    def test_analyst_rating_hold(self):
        assert _analyst_rating(55.0, "HOLD") == "Hold"

    def test_staleness_warning_triggered(self):
        old_date = dt.date.today() - dt.timedelta(days=2)
        resp = assemble_universal_intelligence("AAPL", as_of_date=old_date)
        assert resp.staleness_warning is True
        assert resp.data_freshness_hours >= 24.0

    def test_staleness_warning_false_fresh(self):
        resp = assemble_universal_intelligence("AAPL", as_of_date=dt.date.today())
        assert resp.staleness_warning is False
        assert resp.data_freshness_hours < 24.0

    def test_all_required_fields_present(self):
        resp = assemble_universal_intelligence("MSFT")
        required = [
            "symbol", "as_of_date", "signal_label", "dau", "ml_adjusted_dau",
            "analyst_rating", "conviction", "regime_label", "regime_strength",
            "systemic_risk_index", "ic_state", "intelligence_quality_score",
            "days_of_live_data", "eis_score", "caps_score", "fragility_score",
            "scps_score", "bfs_score", "factor_composite_score",
            "osms_score", "ias_score", "pess_score", "var_1d_99", "sri",
            "primary_driver", "primary_conclusion", "top_supporting_evidence",
            "top_risk", "signal_batting_average", "dossier_event_count",
            "moat_score", "data_freshness_hours", "staleness_warning",
        ]
        for f in required:
            assert hasattr(resp, f), f"Missing field: {f}"

    def test_cache_roundtrip(self):
        resp = assemble_universal_intelligence("AAPL")
        result = cache_universal_response("AAPL", resp)
        assert isinstance(result, bool)
        # cache may return False (DB disabled); response still has correct symbol
        assert resp.symbol == "AAPL"
        cached = get_cached_universal_response("AAPL", resp.as_of_date)
        if cached is not None:
            assert cached.get("symbol") == "AAPL"


# ===========================================================================
# TestPipelineOrchestrator
# ===========================================================================

class TestPipelineOrchestrator:
    def test_pipeline_creates_run_id(self):
        result = run_full_pipeline()
        assert isinstance(result, PipelineRun)
        assert len(result.run_id) > 0

    def test_stages_all_present(self):
        result = run_full_pipeline()
        assert len(result.stages) == len(PIPELINE_STAGES)
        for stage_name in PIPELINE_STAGES:
            assert stage_name in result.stages, f"Missing stage: {stage_name}"

    def test_stage_result_has_required_fields(self):
        result = run_full_pipeline()
        for name, stage in result.stages.items():
            assert isinstance(stage.status, str), f"status missing for {name}"
            assert isinstance(stage.duration_seconds, float), f"duration missing for {name}"
            assert stage.status in ("success", "failed", "skipped")

    def test_blocking_stage_stops_pipeline(self):
        def fail():
            raise RuntimeError("simulated bar_ingestion failure")

        result = run_full_pipeline(_stage_executors={"bar_ingestion": fail})
        assert result.stages["bar_ingestion"].status == "failed"
        # signal_generation depends on feature_computation which depends on bar_ingestion
        assert result.stages["signal_generation"].status == "skipped"

    def test_nonblocking_stage_continues(self):
        def fail():
            raise RuntimeError("simulated pnl failure")

        result = run_full_pipeline(_stage_executors={"pnl_compute": fail})
        assert result.stages["pnl_compute"].status == "failed"
        # ic_computation has no dependency on pnl_compute — must still run
        assert result.stages["ic_computation"].status != "skipped"

    def test_overall_status_partial(self):
        def fail():
            raise RuntimeError("test failure")

        result = run_full_pipeline(_stage_executors={"pnl_compute": fail})
        # pnl_compute fails (non-blocking); other stages succeed → partial
        assert result.overall_status == "partial"

    def test_overall_status_success(self):
        def noop():
            return {"records_processed": 0}

        overrides = {stage: noop for stage in PIPELINE_STAGES}
        result = run_full_pipeline(_stage_executors=overrides)
        assert result.overall_status == "success"


# ===========================================================================
# TestSelfImprovement
# ===========================================================================

class TestSelfImprovement:
    def test_status_returns_dataclass(self):
        status = check_self_improvement_status()
        assert isinstance(status, SelfImprovementStatus)

    def test_status_no_model_default(self):
        # With DB disabled, should default to "no_model_trained"
        status = check_self_improvement_status()
        assert status.last_model_version == "no_model_trained"

    def test_status_has_required_fields(self):
        status = check_self_improvement_status()
        assert isinstance(status.current_psi_score, float)
        assert isinstance(status.drift_warning, bool)
        assert isinstance(status.effective_breadth, int)
        assert isinstance(status.amqs_score, float)
        assert isinstance(status.next_recommended_action, str)

    def test_trigger_insufficient_samples(self):
        # With DB disabled → status=skipped
        result = trigger_self_improvement(min_new_samples=20)
        assert result.get("status") == "skipped"

    def test_improvement_history_list(self):
        history = get_improvement_history()
        assert isinstance(history, list)


# ===========================================================================
# TestSystemHealth
# ===========================================================================

class TestSystemHealth:
    def test_health_returns_dataclass(self):
        health = compute_system_health()
        assert isinstance(health, SystemHealth)

    def test_overall_health_bounded(self):
        health = compute_system_health()
        assert 0.0 <= health.overall_health_score <= 100.0

    def test_overall_status_healthy(self):
        # Test the classification logic directly
        assert compute_health_status(75.0) == "healthy"
        assert compute_health_status(100.0) == "healthy"

    def test_overall_status_degraded(self):
        assert compute_health_status(55.0) == "degraded"

    def test_overall_status_critical(self):
        assert compute_health_status(30.0) == "critical"
        assert compute_health_status(0.0) == "critical"

    def test_overall_status_consistent_with_score(self):
        health = compute_system_health()
        expected = compute_health_status(health.overall_health_score)
        assert health.overall_status == expected

    def test_database_health_present(self):
        health = compute_system_health()
        assert "connectivity" in health.database_health
        assert "migration_version" in health.database_health

    def test_data_freshness_present(self):
        health = compute_system_health()
        assert len(health.data_freshness) >= 3

    def test_active_alerts_list(self):
        health = compute_system_health()
        assert isinstance(health.active_alerts, list)

    def test_health_history_list(self):
        history = get_health_history()
        assert isinstance(history, list)
