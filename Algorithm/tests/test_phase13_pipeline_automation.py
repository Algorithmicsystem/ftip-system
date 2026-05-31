"""Regression tests for Phase 13 — daily pipeline end-to-end automation."""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_STAGES = [
    "market_breadth", "sector_breadth", "ic_snapshot",
    "alerts", "screen", "signal_pnl",
    "provider_reliability", "linkage_refresh",
]


def _run_pipeline_with_mocks(as_of=dt.date(2024, 6, 1), skip_stages=None, **overrides):
    """Run run_daily_pipeline with all external calls mocked out."""
    from api.jobs.orchestrator import run_daily_pipeline

    mock_health = MagicMock()
    mock_health.status = "ok"
    mock_health.providers = []

    defaults = dict(
        compute_market_breadth=lambda *_: {"breadth_state": "EXPANDING", "universe_size": 100},
        store_market_breadth=lambda *_: True,
        compute_sector_breadth=lambda *_: [{"breadth_state": "EXPANDING"}],
        store_sector_breadth=lambda *_: 1,
        compute_ic_snapshot=lambda *_: {"composite_21d": 0.05},
        store_ic_snapshot=lambda *_: 3,
        run_alert_scan=lambda *_: MagicMock(
            rules_evaluated=5, fired=1, suppressed=0,
            already_fired_today=0, webhook_delivered=1, webhook_failed=0,
        ),
        screen_universe=lambda *_: {"total_screened": 50, "count": 8,
                                    "ic_state": "MODERATE", "breadth_state": "EXPANDING",
                                    "results": [{"symbol": "AAPL", "dau": 0.72}]},
        compute_signal_pnl=lambda *_: [{"symbol": "AAPL"}],
        store_signal_pnl=lambda *_: 1,
        get_providers_health=lambda: mock_health,
        snapshot_provider_reliability=lambda *_, **__: 3,
        build_from_sector=lambda **__: 12,
    )
    defaults.update(overrides)

    with patch("api.jobs.orchestrator.db") as mock_db, \
         patch("api.jobs.breadth.compute_market_breadth", defaults["compute_market_breadth"]), \
         patch("api.jobs.breadth.store_market_breadth", defaults["store_market_breadth"]), \
         patch("api.jobs.sector_breadth.compute_sector_breadth", defaults["compute_sector_breadth"]), \
         patch("api.jobs.sector_breadth.store_sector_breadth", defaults["store_sector_breadth"]), \
         patch("api.jobs.ic.compute_ic_snapshot", defaults["compute_ic_snapshot"]), \
         patch("api.jobs.ic.store_ic_snapshot", defaults["store_ic_snapshot"]), \
         patch("api.jobs.alerts.run_alert_scan", defaults["run_alert_scan"]), \
         patch("api.axiom.screener.screen_universe", defaults["screen_universe"]), \
         patch("api.jobs.pnl.compute_signal_pnl", defaults["compute_signal_pnl"]), \
         patch("api.jobs.pnl.store_signal_pnl", defaults["store_signal_pnl"]), \
         patch("api.providers.get_providers_health", defaults["get_providers_health"]), \
         patch("api.providers.reliability.snapshot_provider_reliability",
               defaults["snapshot_provider_reliability"]), \
         patch("api.signals.linkage.SymbolLinkageGraph.build_from_sector",
               defaults["build_from_sector"]):
        mock_db.db_write_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        kwargs = {"skip_stages": set(skip_stages)} if skip_stages else {}
        return run_daily_pipeline(as_of, **kwargs)


# ---------------------------------------------------------------------------
# 1. Pipeline now has 8 stages
# ---------------------------------------------------------------------------

def test_pipeline_runs_all_eight_stages():
    """run_daily_pipeline must produce all 8 stage keys in result."""
    from api.jobs.orchestrator import run_daily_pipeline
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
         patch("api.providers.get_providers_health",
               return_value=MagicMock(status="ok", providers=[])), \
         patch("api.providers.reliability.snapshot_provider_reliability", return_value=0), \
         patch("api.signals.linkage.SymbolLinkageGraph.build_from_sector", return_value=0):
        mock_db.db_write_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        result = run_daily_pipeline(dt.date(2024, 6, 1))

    for stage in ALL_STAGES:
        assert stage in result["stages"], f"Missing stage: {stage}"


def test_pipeline_result_has_status_and_headline():
    """Result must have status, headline, as_of_date, stages, top_opportunities."""
    from api.jobs.orchestrator import run_daily_pipeline
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
         patch("api.providers.get_providers_health",
               return_value=MagicMock(status="ok", providers=[])), \
         patch("api.providers.reliability.snapshot_provider_reliability", return_value=0), \
         patch("api.signals.linkage.SymbolLinkageGraph.build_from_sector", return_value=0):
        mock_db.db_write_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        result = run_daily_pipeline(dt.date(2024, 6, 1))

    assert "status" in result
    assert "headline" in result
    assert "as_of_date" in result
    assert "stages" in result
    assert "top_opportunities" in result


# ---------------------------------------------------------------------------
# 2. New stages populate correctly
# ---------------------------------------------------------------------------

def test_signal_pnl_stage_records_stored():
    """signal_pnl stage must report rows_stored from store_signal_pnl."""
    from api.jobs.orchestrator import run_daily_pipeline
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
         patch("api.jobs.pnl.compute_signal_pnl", return_value=[{"x": 1}, {"x": 2}, {"x": 3}]), \
         patch("api.jobs.pnl.store_signal_pnl", return_value=3), \
         patch("api.providers.get_providers_health",
               return_value=MagicMock(status="ok", providers=[])), \
         patch("api.providers.reliability.snapshot_provider_reliability", return_value=0), \
         patch("api.signals.linkage.SymbolLinkageGraph.build_from_sector", return_value=0):
        mock_db.db_write_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        result = run_daily_pipeline(dt.date(2024, 6, 1))

    pnl_stage = result["stages"]["signal_pnl"]
    assert pnl_stage["status"] == "ok"
    assert pnl_stage["rows_computed"] == 3
    assert pnl_stage["rows_stored"] == 3


def test_provider_reliability_stage_reports_degraded():
    """provider_reliability stage must list degraded providers by name."""
    from api.jobs.orchestrator import run_daily_pipeline
    degraded_p = MagicMock()
    degraded_p.name = "finnhub"
    degraded_p.status = "down"
    degraded_p.enabled = True
    ok_p = MagicMock()
    ok_p.name = "fred"
    ok_p.status = "ok"
    ok_p.enabled = True
    mock_health = MagicMock()
    mock_health.status = "down"
    mock_health.providers = [degraded_p, ok_p]

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
         patch("api.providers.get_providers_health", return_value=mock_health), \
         patch("api.providers.reliability.snapshot_provider_reliability", return_value=0), \
         patch("api.signals.linkage.SymbolLinkageGraph.build_from_sector", return_value=0):
        mock_db.db_write_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        result = run_daily_pipeline(dt.date(2024, 6, 1))

    prov_stage = result["stages"]["provider_reliability"]
    assert prov_stage["status"] == "ok"
    assert "finnhub" in prov_stage["degraded"]
    assert "fred" not in prov_stage["degraded"]
    assert "finnhub" in result["headline"]


def test_linkage_refresh_stage_reports_links_written():
    """linkage_refresh stage must report links_written from build_from_sector."""
    from api.jobs.orchestrator import run_daily_pipeline
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
         patch("api.providers.get_providers_health",
               return_value=MagicMock(status="ok", providers=[])), \
         patch("api.providers.reliability.snapshot_provider_reliability", return_value=0), \
         patch("api.signals.linkage.SymbolLinkageGraph.build_from_sector", return_value=42):
        mock_db.db_write_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        result = run_daily_pipeline(dt.date(2024, 6, 1))

    link_stage = result["stages"]["linkage_refresh"]
    assert link_stage["status"] == "ok"
    assert link_stage["links_written"] == 42


# ---------------------------------------------------------------------------
# 3. skip_stages
# ---------------------------------------------------------------------------

def test_skip_stages_marks_stage_as_skipped():
    """Stages in skip_stages must have status='skipped' and not run their fn."""
    from api.jobs.orchestrator import run_daily_pipeline
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
         patch("api.providers.get_providers_health",
               return_value=MagicMock(status="ok", providers=[])), \
         patch("api.providers.reliability.snapshot_provider_reliability", return_value=0), \
         patch("api.signals.linkage.SymbolLinkageGraph.build_from_sector", return_value=0):
        mock_db.db_write_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        result = run_daily_pipeline(
            dt.date(2024, 6, 1),
            skip_stages={"linkage_refresh", "provider_reliability"},
        )

    assert result["stages"]["linkage_refresh"]["status"] == "skipped"
    assert result["stages"]["provider_reliability"]["status"] == "skipped"
    assert result["stages"]["market_breadth"]["status"] == "ok"


# ---------------------------------------------------------------------------
# 4. Stage failure isolation
# ---------------------------------------------------------------------------

def test_one_stage_failure_does_not_abort_rest():
    """A crash in signal_pnl must not prevent provider_reliability or linkage_refresh."""
    from api.jobs.orchestrator import run_daily_pipeline
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
         patch("api.jobs.pnl.compute_signal_pnl", side_effect=RuntimeError("DB timeout")), \
         patch("api.jobs.pnl.store_signal_pnl", return_value=0), \
         patch("api.providers.get_providers_health",
               return_value=MagicMock(status="ok", providers=[])), \
         patch("api.providers.reliability.snapshot_provider_reliability", return_value=0), \
         patch("api.signals.linkage.SymbolLinkageGraph.build_from_sector", return_value=0):
        mock_db.db_write_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        result = run_daily_pipeline(dt.date(2024, 6, 1))

    assert result["stages"]["signal_pnl"]["status"] == "error"
    assert result["stages"]["provider_reliability"]["status"] == "ok"
    assert result["stages"]["linkage_refresh"]["status"] == "ok"
    assert result["status"] in ("partial", "error")


# ---------------------------------------------------------------------------
# 5. DailyRunRequest accepts skip_stages
# ---------------------------------------------------------------------------

def test_daily_run_request_accepts_skip_stages():
    from api.jobs.orchestrator import DailyRunRequest
    req = DailyRunRequest(skip_stages=["linkage_refresh", "signal_pnl"])
    assert "linkage_refresh" in req.skip_stages
    assert "signal_pnl" in req.skip_stages


def test_daily_run_request_skip_stages_defaults_empty():
    from api.jobs.orchestrator import DailyRunRequest
    req = DailyRunRequest()
    assert req.skip_stages == []
