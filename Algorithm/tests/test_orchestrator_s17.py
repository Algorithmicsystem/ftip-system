"""Session 17: Daily pipeline orchestrator tests."""
from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import security


def _db_env(monkeypatch):
    monkeypatch.setenv("FTIP_API_KEY", "secret")
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_READ_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_WRITE_ENABLED", "1")
    security.reset_auth_cache()


AUTH = {"X-FTIP-API-Key": "secret"}
_DATE = dt.date(2025, 1, 2)


# ---------------------------------------------------------------------------
# _run_stage unit tests
# ---------------------------------------------------------------------------

class TestRunStage:
    def _stages(self):
        return {}

    def test_success_records_ok_status(self):
        from api.jobs.orchestrator import _run_stage
        stages = {}
        result = _run_stage("test", lambda: {"foo": "bar"}, stages)
        assert stages["test"]["status"] == "ok"
        assert stages["test"]["foo"] == "bar"
        assert "duration_ms" in stages["test"]
        assert result == {"foo": "bar"}

    def test_failure_records_error_status(self):
        from api.jobs.orchestrator import _run_stage
        stages = {}
        def boom():
            raise ValueError("kaboom")
        result = _run_stage("test", boom, stages)
        assert result is None
        assert stages["test"]["status"] == "error"
        assert "kaboom" in stages["test"]["error"]
        assert "duration_ms" in stages["test"]

    def test_error_does_not_raise(self):
        from api.jobs.orchestrator import _run_stage
        stages = {}
        def boom():
            raise RuntimeError("test error")
        _run_stage("exploding", boom, stages)
        assert stages["exploding"]["status"] == "error"

    def test_multiple_stages_are_isolated(self):
        from api.jobs.orchestrator import _run_stage
        stages = {}
        _run_stage("a", lambda: {"x": 1}, stages)
        _run_stage("b", lambda: (_ for _ in ()).throw(Exception("fail")), stages)
        _run_stage("c", lambda: {"z": 3}, stages)
        assert stages["a"]["status"] == "ok"
        assert stages["b"]["status"] == "error"
        assert stages["c"]["status"] == "ok"

    def test_duration_ms_is_non_negative(self):
        from api.jobs.orchestrator import _run_stage
        stages = {}
        _run_stage("t", lambda: {}, stages)
        assert stages["t"]["duration_ms"] >= 0


# ---------------------------------------------------------------------------
# _build_headline unit tests
# ---------------------------------------------------------------------------

class TestBuildHeadline:
    def test_basic_headline(self):
        from api.jobs.orchestrator import _build_headline
        stages = {
            "market_breadth": {"status": "ok", "breadth_state": "EXPANDING"},
            "screen": {"status": "ok", "count": 5, "ic_state": "MODERATE"},
            "alerts": {"status": "ok", "fired": 2},
            "ic_snapshot": {"status": "ok", "rows_written": 10},
        }
        headline = _build_headline(stages)
        assert "EXPANDING" in headline
        assert "MODERATE" in headline
        assert "2 alerts" in headline
        assert "5 conviction" in headline
        assert "10 rows" in headline

    def test_no_alerts_no_screen(self):
        from api.jobs.orchestrator import _build_headline
        stages = {
            "market_breadth": {"status": "ok", "breadth_state": "CONTRACTING"},
            "screen": {"status": "ok", "count": 0, "ic_state": "WEAK"},
            "alerts": {"status": "ok", "fired": 0},
            "ic_snapshot": {"status": "ok", "rows_written": 0},
        }
        headline = _build_headline(stages)
        assert "CONTRACTING" in headline
        assert "alert" not in headline
        assert "candidate" not in headline

    def test_singular_alert(self):
        from api.jobs.orchestrator import _build_headline
        stages = {
            "market_breadth": {"status": "error"},
            "screen": {"status": "ok", "count": 1, "ic_state": "STRONG"},
            "alerts": {"status": "ok", "fired": 1},
            "ic_snapshot": {"status": "ok"},
        }
        headline = _build_headline(stages)
        assert "1 alert fired" in headline
        assert "1 conviction candidate" in headline

    def test_missing_stages_fall_back_to_unknown(self):
        from api.jobs.orchestrator import _build_headline
        headline = _build_headline({})
        assert "UNKNOWN" in headline


# ---------------------------------------------------------------------------
# run_daily_pipeline unit tests
# ---------------------------------------------------------------------------

def _make_alert_summary():
    m = MagicMock()
    m.rules_evaluated = 5
    m.fired = 1
    m.suppressed = 0
    m.already_fired_today = 0
    m.webhook_delivered = 1
    m.webhook_failed = 0
    return m


class TestRunDailyPipeline:
    def _run(self, **overrides):
        from api.jobs.orchestrator import run_daily_pipeline

        from unittest.mock import MagicMock as _MM
        _mock_health = _MM(); _mock_health.status = "ok"; _mock_health.providers = []
        defaults = {
            "api.jobs.breadth.compute_market_breadth": lambda d, lb: {
                "breadth_state": "EXPANDING",
                "universe_size": 150,
            },
            "api.jobs.breadth.store_market_breadth": lambda d, p: True,
            "api.jobs.sector_breadth.compute_sector_breadth": lambda d, lb: [
                {"sector": "technology", "breadth_state": "EXPANDING"},
                {"sector": "energy", "breadth_state": "CONTRACTING"},
            ],
            "api.jobs.sector_breadth.store_sector_breadth": lambda d, s: 2,
            "api.jobs.ic.compute_ic_snapshot": lambda d: {"ic_state": "MODERATE", "ic": 0.35},
            "api.jobs.ic.store_ic_snapshot": lambda d, s: 3,
            "api.jobs.alerts.run_alert_scan": lambda d: _make_alert_summary(),
            "api.axiom.screener.screen_universe": lambda d, **kw: {
                "status": "ok",
                "total_screened": 10,
                "count": 3,
                "ic_state": "MODERATE",
                "breadth_state": "EXPANDING",
                "results": [
                    {"symbol": "NVDA", "signal_label": "BUY", "dau": 85.0,
                     "conviction_score": 78.0, "suggested_weight_pct": "8.5%",
                     "regime_label": "fundamental_convergence"},
                ],
            },
            "api.jobs.pnl.compute_signal_pnl": lambda d: [],
            "api.jobs.pnl.store_signal_pnl": lambda rows: 0,
            "api.providers.get_providers_health": lambda: _mock_health,
            "api.providers.reliability.snapshot_provider_reliability": lambda h, **kw: 0,
            "api.signals.linkage.SymbolLinkageGraph.build_from_sector": lambda self, **kw: 0,
        }
        defaults.update(overrides)

        patches = {}
        for target, val in defaults.items():
            patches[target] = val

        with (
            patch("api.jobs.orchestrator.db") as mock_db,
            patch("api.jobs.breadth.compute_market_breadth", patches["api.jobs.breadth.compute_market_breadth"]),
            patch("api.jobs.breadth.store_market_breadth", patches["api.jobs.breadth.store_market_breadth"]),
            patch("api.jobs.sector_breadth.compute_sector_breadth", patches["api.jobs.sector_breadth.compute_sector_breadth"]),
            patch("api.jobs.sector_breadth.store_sector_breadth", patches["api.jobs.sector_breadth.store_sector_breadth"]),
            patch("api.jobs.ic.compute_ic_snapshot", patches["api.jobs.ic.compute_ic_snapshot"]),
            patch("api.jobs.ic.store_ic_snapshot", patches["api.jobs.ic.store_ic_snapshot"]),
            patch("api.jobs.alerts.run_alert_scan", patches["api.jobs.alerts.run_alert_scan"]),
            patch("api.axiom.screener.screen_universe", patches["api.axiom.screener.screen_universe"]),
            patch("api.jobs.pnl.compute_signal_pnl", patches["api.jobs.pnl.compute_signal_pnl"]),
            patch("api.jobs.pnl.store_signal_pnl", patches["api.jobs.pnl.store_signal_pnl"]),
            patch("api.providers.get_providers_health", patches["api.providers.get_providers_health"]),
            patch("api.providers.reliability.snapshot_provider_reliability",
                  patches["api.providers.reliability.snapshot_provider_reliability"]),
            patch("api.signals.linkage.SymbolLinkageGraph.build_from_sector",
                  patches["api.signals.linkage.SymbolLinkageGraph.build_from_sector"]),
        ):
            mock_db.db_write_enabled.return_value = True
            return run_daily_pipeline(_DATE)

    def test_returns_dict_with_required_keys(self):
        result = self._run()
        for key in ("status", "as_of_date", "started_at", "finished_at",
                    "duration_ms", "stages", "headline", "top_opportunities"):
            assert key in result, f"missing key: {key}"

    def test_all_stages_ok_yields_ok_status(self):
        result = self._run()
        assert result["status"] == "ok"

    def test_as_of_date_matches_input(self):
        result = self._run()
        assert result["as_of_date"] == _DATE.isoformat()

    def test_eight_stages_present(self):
        result = self._run()
        assert set(result["stages"].keys()) == {
            "market_breadth", "sector_breadth", "ic_snapshot", "alerts", "screen",
            "signal_pnl", "provider_reliability", "linkage_refresh",
        }

    def test_partial_status_when_one_stage_fails(self):
        from unittest.mock import MagicMock as _MM
        from api.jobs.orchestrator import run_daily_pipeline

        def bad_breadth(d, lb):
            raise RuntimeError("no data")

        _h = _MM(); _h.status = "ok"; _h.providers = []
        with (
            patch("api.jobs.orchestrator.db") as mock_db,
            patch("api.jobs.breadth.compute_market_breadth", bad_breadth),
            patch("api.jobs.breadth.store_market_breadth", lambda d, p: False),
            patch("api.jobs.sector_breadth.compute_sector_breadth", lambda d, lb: []),
            patch("api.jobs.sector_breadth.store_sector_breadth", lambda d, s: 0),
            patch("api.jobs.ic.compute_ic_snapshot", lambda d: {}),
            patch("api.jobs.ic.store_ic_snapshot", lambda d, s: 0),
            patch("api.jobs.alerts.run_alert_scan", lambda d: _make_alert_summary()),
            patch("api.axiom.screener.screen_universe", lambda d, **kw: {
                "status": "ok", "total_screened": 0, "count": 0,
                "ic_state": "MODERATE", "breadth_state": "UNKNOWN", "results": [],
            }),
            patch("api.jobs.pnl.compute_signal_pnl", lambda d: []),
            patch("api.jobs.pnl.store_signal_pnl", lambda rows: 0),
            patch("api.providers.get_providers_health", lambda: _h),
            patch("api.providers.reliability.snapshot_provider_reliability", lambda h, **kw: 0),
            patch("api.signals.linkage.SymbolLinkageGraph.build_from_sector", lambda self, **kw: 0),
        ):
            mock_db.db_write_enabled.return_value = True
            result = run_daily_pipeline(_DATE)
        assert result["status"] == "partial"
        assert result["stages"]["market_breadth"]["status"] == "error"

    def test_top_opportunities_populated(self):
        result = self._run()
        assert isinstance(result["top_opportunities"], list)
        assert len(result["top_opportunities"]) >= 1
        assert result["top_opportunities"][0]["symbol"] == "NVDA"

    def test_headline_is_nonempty_string(self):
        result = self._run()
        assert isinstance(result["headline"], str)
        assert len(result["headline"]) > 5

    def test_duration_ms_present(self):
        result = self._run()
        assert result["duration_ms"] >= 0


# ---------------------------------------------------------------------------
# Route contract tests
# ---------------------------------------------------------------------------

class TestOrchestratorRoutes:
    def test_daily_run_requires_auth(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.post("/jobs/daily-run", json={})
        assert resp.status_code == 401

    def test_daily_run_route_in_openapi(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        assert "/jobs/daily-run" in resp.json().get("paths", {})

    def test_daily_run_returns_digest(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app

        def fake_pipeline(as_of_date, **kwargs):
            return {
                "status": "ok",
                "as_of_date": as_of_date.isoformat(),
                "started_at": "2025-01-02T00:00:00Z",
                "finished_at": "2025-01-02T00:00:01Z",
                "duration_ms": 1000,
                "stages": {
                    "market_breadth": {"status": "ok", "duration_ms": 50},
                    "sector_breadth":  {"status": "ok", "duration_ms": 40},
                    "ic_snapshot":     {"status": "ok", "duration_ms": 30},
                    "alerts":          {"status": "ok", "duration_ms": 20},
                    "screen":          {"status": "ok", "duration_ms": 10},
                },
                "headline": "EXPANDING breadth, MODERATE IC.",
                "top_opportunities": [],
            }

        monkeypatch.setattr("api.jobs.orchestrator.run_daily_pipeline", fake_pipeline)
        client = TestClient(app)
        resp = client.post(
            "/jobs/daily-run",
            json={"as_of_date": "2025-01-02"},
            headers=AUTH,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["as_of_date"] == "2025-01-02"
        assert "stages" in data
        assert "headline" in data

    def test_daily_run_default_date_fallback(self, monkeypatch):
        """No as_of_date → falls back to yesterday without crashing."""
        _db_env(monkeypatch)
        from api.main import app
        import datetime as dt

        captured = {}

        def fake_pipeline(as_of_date, **kwargs):
            captured["date"] = as_of_date
            return {
                "status": "ok",
                "as_of_date": as_of_date.isoformat(),
                "started_at": "", "finished_at": "",
                "duration_ms": 0,
                "stages": {},
                "headline": "UNKNOWN breadth, UNKNOWN IC.",
                "top_opportunities": [],
            }

        monkeypatch.setattr("api.jobs.orchestrator.run_daily_pipeline", fake_pipeline)
        client = TestClient(app)
        resp = client.post("/jobs/daily-run", json={}, headers=AUTH)
        assert resp.status_code == 200
        assert captured["date"] == dt.date.today() - dt.timedelta(days=1)
