"""Regression tests for Phase 16 — IC calibration → Kelly hit-rate chain."""

from __future__ import annotations

import datetime as dt
import math
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 1. snapshot_ic_as_calibration function
# ---------------------------------------------------------------------------

def test_snapshot_ic_as_calibration_importable():
    from api.jobs.ic import snapshot_ic_as_calibration
    assert callable(snapshot_ic_as_calibration)


def test_hit_rate_is_0_5_when_mean_ic_is_zero():
    """mean_ic=0 → hit_rate exactly 0.5 (no edge)."""
    from api.jobs.ic import snapshot_ic_as_calibration

    empty_summary = {
        "sample_count": 0, "mean_ic": 0.0, "std_ic": None,
        "icir": None, "t_stat": None,
        "ic_mean_21d": None, "ic_mean_63d": None,
        "icir_21d": None, "icir_63d": None,
        "ic_state": "INSUFFICIENT",
    }
    with patch("api.jobs.ic.load_ic_history", return_value=[]), \
         patch("api.jobs.ic.compute_ic_decay_summary", return_value=empty_summary), \
         patch("api.axiom.persistence.persist_axiom_calibration_snapshot", return_value="k"):
        result = snapshot_ic_as_calibration(dt.date(2024, 6, 1), write_ok=True)

    assert result["hit_rate"] == 0.5


def test_hit_rate_increases_with_positive_ic():
    """mean_ic > 0 → hit_rate > 0.5."""
    from api.jobs.ic import snapshot_ic_as_calibration

    positive_summary = {
        "sample_count": 100, "mean_ic": 0.10, "std_ic": 0.05,
        "icir": 2.0, "t_stat": 20.0,
        "ic_mean_21d": 0.10, "ic_mean_63d": 0.09,
        "icir_21d": 2.0, "icir_63d": 1.9,
        "ic_state": "MODERATE",
    }
    with patch("api.jobs.ic.load_ic_history", return_value=[{"ic_value": 0.1}] * 10), \
         patch("api.jobs.ic.compute_ic_decay_summary", return_value=positive_summary), \
         patch("api.axiom.persistence.persist_axiom_calibration_snapshot", return_value="k"):
        result = snapshot_ic_as_calibration(dt.date(2024, 6, 1), write_ok=True)

    assert result["hit_rate"] > 0.5


def test_hit_rate_clamped_between_0_01_and_0_99():
    """hit_rate is always within [0.01, 0.99] regardless of IC extremes."""
    from api.jobs.ic import snapshot_ic_as_calibration

    extreme_summary = {
        "sample_count": 10, "mean_ic": 10.0, "std_ic": 0.0,
        "icir": None, "t_stat": None,
        "ic_mean_21d": None, "ic_mean_63d": None,
        "icir_21d": None, "icir_63d": None,
        "ic_state": "STRONG",
    }
    with patch("api.jobs.ic.load_ic_history", return_value=[]), \
         patch("api.jobs.ic.compute_ic_decay_summary", return_value=extreme_summary), \
         patch("api.axiom.persistence.persist_axiom_calibration_snapshot", return_value="k"):
        result = snapshot_ic_as_calibration(dt.date(2024, 6, 1), write_ok=True)

    assert 0.01 <= result["hit_rate"] <= 0.99


def test_snapshot_not_stored_when_write_disabled():
    """write_ok=False → persist not called, snapshot_stored=False."""
    from api.jobs.ic import snapshot_ic_as_calibration

    summary = {
        "sample_count": 5, "mean_ic": 0.05, "std_ic": 0.02,
        "icir": 2.5, "t_stat": 5.0,
        "ic_mean_21d": 0.05, "ic_mean_63d": 0.04,
        "icir_21d": 2.5, "icir_63d": 2.0,
        "ic_state": "MODERATE",
    }
    with patch("api.jobs.ic.load_ic_history", return_value=[]), \
         patch("api.jobs.ic.compute_ic_decay_summary", return_value=summary), \
         patch("api.axiom.persistence.persist_axiom_calibration_snapshot") as mock_persist:
        result = snapshot_ic_as_calibration(dt.date(2024, 6, 1), write_ok=False)

    mock_persist.assert_not_called()
    assert result["snapshot_stored"] is False


def test_snapshot_result_has_required_keys():
    from api.jobs.ic import snapshot_ic_as_calibration

    summary = {
        "sample_count": 50, "mean_ic": 0.08, "std_ic": 0.03,
        "icir": 2.7, "t_stat": 19.0,
        "ic_mean_21d": 0.08, "ic_mean_63d": 0.07,
        "icir_21d": 2.7, "icir_63d": 2.5,
        "ic_state": "MODERATE",
    }
    with patch("api.jobs.ic.load_ic_history", return_value=[]), \
         patch("api.jobs.ic.compute_ic_decay_summary", return_value=summary), \
         patch("api.axiom.persistence.persist_axiom_calibration_snapshot", return_value="k"):
        result = snapshot_ic_as_calibration(dt.date(2024, 6, 1), write_ok=True)

    for key in ("hit_rate", "mean_ic", "ic_state", "sample_count", "snapshot_stored"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 2. Stage 9 in the pipeline
# ---------------------------------------------------------------------------

def test_pipeline_has_nine_stages():
    """run_daily_pipeline now produces 9 stage keys."""
    from api.jobs.orchestrator import run_daily_pipeline

    all_stages = [
        "market_breadth", "sector_breadth", "ic_snapshot",
        "alerts", "screen", "signal_pnl",
        "provider_reliability", "linkage_refresh", "ic_calibration",
    ]

    _mock_health = MagicMock()
    _mock_health.status = "ok"
    _mock_health.providers = []

    with patch("api.jobs.orchestrator.db") as mock_db, \
         patch("api.jobs.breadth.compute_market_breadth", return_value={"breadth_state": "NEUTRAL", "universe_size": 0}), \
         patch("api.jobs.breadth.store_market_breadth", return_value=False), \
         patch("api.jobs.sector_breadth.compute_sector_breadth", return_value=[]), \
         patch("api.jobs.sector_breadth.store_sector_breadth", return_value=0), \
         patch("api.jobs.ic.compute_ic_snapshot", return_value={}), \
         patch("api.jobs.ic.store_ic_snapshot", return_value=0), \
         patch("api.jobs.alerts.run_alert_scan",
               return_value=MagicMock(rules_evaluated=0, fired=0, suppressed=0,
                                      already_fired_today=0, webhook_delivered=0, webhook_failed=0)), \
         patch("api.axiom.screener.screen_universe",
               return_value={"total_screened": 0, "count": 0, "results": [], "ic_state": None, "breadth_state": None}), \
         patch("api.jobs.pnl.compute_signal_pnl", return_value=[]), \
         patch("api.jobs.pnl.store_signal_pnl", return_value=0), \
         patch("api.providers.get_providers_health", return_value=_mock_health), \
         patch("api.providers.reliability.snapshot_provider_reliability", return_value=0), \
         patch("api.signals.linkage.SymbolLinkageGraph.build_from_sector", return_value=0), \
         patch("api.jobs.ic.snapshot_ic_as_calibration",
               return_value={"hit_rate": 0.52, "mean_ic": 0.04, "ic_state": "MODERATE",
                             "sample_count": 30, "snapshot_stored": True}):
        mock_db.db_write_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        result = run_daily_pipeline(dt.date(2024, 6, 1))

    for stage in all_stages:
        assert stage in result["stages"], f"Missing stage: {stage}"


def test_ic_calibration_stage_reports_hit_rate():
    """ic_calibration stage exposes hit_rate in pipeline result."""
    from api.jobs.orchestrator import run_daily_pipeline

    _mock_health = MagicMock()
    _mock_health.status = "ok"
    _mock_health.providers = []

    with patch("api.jobs.orchestrator.db") as mock_db, \
         patch("api.jobs.breadth.compute_market_breadth", return_value={"breadth_state": "NEUTRAL", "universe_size": 0}), \
         patch("api.jobs.breadth.store_market_breadth", return_value=False), \
         patch("api.jobs.sector_breadth.compute_sector_breadth", return_value=[]), \
         patch("api.jobs.sector_breadth.store_sector_breadth", return_value=0), \
         patch("api.jobs.ic.compute_ic_snapshot", return_value={}), \
         patch("api.jobs.ic.store_ic_snapshot", return_value=0), \
         patch("api.jobs.alerts.run_alert_scan",
               return_value=MagicMock(rules_evaluated=0, fired=0, suppressed=0,
                                      already_fired_today=0, webhook_delivered=0, webhook_failed=0)), \
         patch("api.axiom.screener.screen_universe",
               return_value={"total_screened": 0, "count": 0, "results": [], "ic_state": None, "breadth_state": None}), \
         patch("api.jobs.pnl.compute_signal_pnl", return_value=[]), \
         patch("api.jobs.pnl.store_signal_pnl", return_value=0), \
         patch("api.providers.get_providers_health", return_value=_mock_health), \
         patch("api.providers.reliability.snapshot_provider_reliability", return_value=0), \
         patch("api.signals.linkage.SymbolLinkageGraph.build_from_sector", return_value=0), \
         patch("api.jobs.ic.snapshot_ic_as_calibration",
               return_value={"hit_rate": 0.54, "mean_ic": 0.10, "ic_state": "MODERATE",
                             "sample_count": 50, "snapshot_stored": True}):
        mock_db.db_write_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        result = run_daily_pipeline(dt.date(2024, 6, 1))

    cal_stage = result["stages"]["ic_calibration"]
    assert cal_stage["status"] == "ok"
    assert cal_stage["hit_rate"] == 0.54
    assert cal_stage["ic_state"] == "MODERATE"
    assert "Kelly hit-rate 0.54" in result["headline"]


def test_hit_rate_gaussian_cdf_approximation():
    """Verify the Gaussian CDF math: Φ(0.10) ≈ 0.5398."""
    import math
    ic = 0.10
    hit_rate = 0.5 + 0.5 * math.erf(ic / math.sqrt(2.0))
    assert abs(hit_rate - 0.5398) < 0.001
