"""Phase 33: PE and Corporate Intelligence Module tests.

Covers:
  3.1  store_entity_financials (upsert)
  3.2  compute_entity_health (health score + components)
  3.3  compute_exit_timing (readiness + recommendation)
  3.4  get_portfolio_overview (org-level summary)
  3.5  get_portfolio_stress_alerts (distressed entities)
  3.6  API endpoint structure
"""
from __future__ import annotations

import datetime as dt
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_period(
    entity_id="ENT001",
    period_end=dt.date(2024, 3, 31),
    revenue=100.0,
    ebitda=25.0,
    net_income=12.0,
    total_debt=60.0,
    cash=20.0,
    capex=5.0,
    free_cash_flow=15.0,
):
    """Return a tuple matching private_entity_financials SELECT column order."""
    return (period_end, revenue, ebitda, net_income, total_debt, cash, free_cash_flow)


def _meta_row(
    entry_date=dt.date(2022, 1, 1),
    entry_ev=500.0,
    target_exit_date=dt.date(2026, 1, 1),
    target_multiple=3.0,
    name="Acme Co",
    sector="Technology",
):
    return (entry_date, entry_ev, target_exit_date, target_multiple, name, sector)


# ---------------------------------------------------------------------------
# 3.1 — store_entity_financials
# ---------------------------------------------------------------------------

class TestStoreEntityFinancials:
    def test_returns_false_when_write_disabled(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: False)
        result = mod.store_entity_financials("ENT001", dt.date(2024, 3, 31), {"revenue": 100.0})
        assert result is False

    def test_returns_true_on_success(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        executed = []
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_execute", lambda sql, params: executed.append(params))
        result = mod.store_entity_financials("ENT001", dt.date(2024, 3, 31), {"revenue": 100.0, "ebitda": 25.0})
        assert result is True
        assert len(executed) == 1
        assert executed[0][0] == "ENT001"

    def test_returns_false_on_exception(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_execute", lambda *a: (_ for _ in ()).throw(Exception("db err")))
        result = mod.store_entity_financials("ENT001", dt.date(2024, 3, 31), {})
        assert result is False

    def test_passes_period_type(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        executed = []
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_execute", lambda sql, params: executed.append(params))
        mod.store_entity_financials("ENT001", dt.date(2024, 12, 31), {}, period_type="annual")
        assert executed[0][2] == "annual"

    def test_missing_financials_fields_stored_as_none(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        executed = []
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_execute", lambda sql, params: executed.append(params))
        mod.store_entity_financials("ENT001", dt.date(2024, 3, 31), {})
        # revenue at index 3 should be None
        assert executed[0][3] is None


# ---------------------------------------------------------------------------
# 3.2 — compute_entity_health
# ---------------------------------------------------------------------------

class TestComputeEntityHealth:
    def test_no_data_returns_no_data_status(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [])
        result = mod.compute_entity_health("ENT001")
        assert result["status"] == "no_data"
        assert result["health_score"] is None

    def test_healthy_company_scores_above_60(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        # revenue=100, ebitda=30 (30% margin), debt=60 (2x), fcf=22, cash=40
        rows = [
            (dt.date(2024, 3, 31), 100.0, 30.0, 15.0, 60.0, 40.0, 22.0),
            (dt.date(2023, 12, 31), 90.0,  27.0, 13.0, 60.0, 35.0, 20.0),
            (dt.date(2023, 9, 30),  85.0,  25.0, 12.0, 65.0, 30.0, 18.0),
            (dt.date(2023, 6, 30),  82.0,  24.0, 11.0, 65.0, 28.0, 17.0),
        ]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_entity_health("ENT001")
        assert result["status"] == "ok"
        assert result["health_score"] is not None
        assert result["health_score"] > 60.0

    def test_distressed_company_scores_below_40(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        # Negative EBITDA, high debt, poor FCF
        rows = [
            (dt.date(2024, 3, 31), 50.0, -5.0, -8.0, 200.0, 3.0, -8.0),
            (dt.date(2023, 12, 31), 55.0, -2.0, -5.0, 190.0, 5.0, -5.0),
            (dt.date(2023, 9, 30),  58.0,  2.0, -2.0, 185.0, 7.0,  0.0),
            (dt.date(2023, 6, 30),  60.0,  5.0,  1.0, 180.0, 9.0,  2.0),
        ]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_entity_health("ENT001")
        assert result["health_score"] is not None
        assert result["health_score"] < 40.0

    def test_alert_set_when_health_below_threshold(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        rows = [(dt.date(2024, 3, 31), 50.0, -5.0, -8.0, 200.0, 3.0, -8.0)]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_entity_health("ENT001")
        assert result["alert"] is True

    def test_alert_not_set_when_health_above_threshold(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        rows = [
            (dt.date(2024, 3, 31), 100.0, 30.0, 15.0, 60.0, 40.0, 22.0),
            (dt.date(2023, 12, 31), 90.0, 27.0, 13.0, 60.0, 35.0, 20.0),
            (dt.date(2023, 9, 30),  85.0, 25.0, 12.0, 65.0, 30.0, 18.0),
            (dt.date(2023, 6, 30),  82.0, 24.0, 11.0, 65.0, 28.0, 17.0),
        ]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_entity_health("ENT001")
        assert result["alert"] is False

    def test_health_score_bounded_0_100(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        rows = [(dt.date(2024, 3, 31), 100.0, 100.0, 100.0, 0.001, 999.0, 100.0)]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_entity_health("ENT001")
        if result["health_score"] is not None:
            assert 0.0 <= result["health_score"] <= 100.0

    def test_components_present_in_result(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        rows = [(dt.date(2024, 3, 31), 100.0, 25.0, 12.0, 50.0, 20.0, 15.0)]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_entity_health("ENT001")
        assert "ebitda_margin_score" in result["components"]
        assert "leverage_score" in result["components"]
        assert "fcf_conversion_score" in result["components"]
        assert "liquidity_score" in result["components"]

    def test_high_leverage_lowers_score(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        low_lev  = [(dt.date(2024, 3, 31), 100.0, 25.0, 12.0, 25.0,  20.0, 15.0)]   # 1x
        high_lev = [(dt.date(2024, 3, 31), 100.0, 25.0, 12.0, 250.0, 20.0, 15.0)]   # 10x
        calls = iter([low_lev, high_lev])
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: next(calls))
        low_result  = mod.compute_entity_health("ENT001")
        high_result = mod.compute_entity_health("ENT001")
        assert low_result["health_score"] > high_result["health_score"]


# ---------------------------------------------------------------------------
# 3.3 — compute_exit_timing
# ---------------------------------------------------------------------------

class TestComputeExitTiming:
    def test_not_found_when_entity_missing(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: None)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [])
        result = mod.compute_exit_timing("ENT999")
        assert result["status"] == "not_found"

    def test_db_disabled_returns_early(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: False)
        result = mod.compute_exit_timing("ENT001")
        assert result["status"] == "db_disabled"

    def test_well_held_improving_company_is_exit_ready(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        # Entry 3 years ago, improving EBITDA, low leverage
        entry = dt.date.today() - dt.timedelta(days=3*365)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: _meta_row(entry_date=entry))
        rows = [
            (dt.date(2024, 3, 31), 120.0, 35.0, 18.0, 70.0, 30.0, 25.0),  # latest
            (dt.date(2023, 12, 31), 110.0, 28.0, 14.0, 75.0, 25.0, 20.0), # prior
        ]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_exit_timing("ENT001")
        assert result["status"] == "ok"
        assert result["exit_readiness_score"] is not None
        assert result["exit_readiness_score"] >= 60.0
        assert result["recommendation"] in ("exit_ready", "monitor_and_prepare")

    def test_early_stage_company_recommends_hold(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        entry = dt.date.today() - dt.timedelta(days=180)  # 6 months
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: _meta_row(entry_date=entry))
        rows = [
            (dt.date(2024, 3, 31), 50.0, 8.0, 2.0, 40.0, 5.0, 3.0),
            (dt.date(2023, 12, 31), 48.0, 7.5, 1.5, 42.0, 4.0, 2.5),
        ]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_exit_timing("ENT001")
        assert result["recommendation"] in ("hold_and_improve", "monitor_and_prepare")

    def test_ebitda_trend_improving_when_growing(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        entry = dt.date.today() - dt.timedelta(days=2*365)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: _meta_row(entry_date=entry))
        rows = [
            (dt.date(2024, 3, 31), 100.0, 30.0, 15.0, 60.0, 20.0, 22.0),  # latest
            (dt.date(2023, 12, 31), 90.0, 20.0, 10.0, 60.0, 18.0, 15.0),  # 50% growth
        ]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_exit_timing("ENT001")
        assert result["ebitda_trend"] == "improving"

    def test_ebitda_trend_declining_when_shrinking(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        entry = dt.date.today() - dt.timedelta(days=2*365)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: _meta_row(entry_date=entry))
        rows = [
            (dt.date(2024, 3, 31), 100.0, 15.0, 5.0, 60.0, 10.0, 8.0),   # latest
            (dt.date(2023, 12, 31), 100.0, 25.0, 12.0, 60.0, 15.0, 18.0), # 40% decline
        ]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_exit_timing("ENT001")
        assert result["ebitda_trend"] == "declining"

    def test_months_held_computed(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        entry = dt.date.today() - dt.timedelta(days=365)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: _meta_row(entry_date=entry))
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [])
        result = mod.compute_exit_timing("ENT001")
        assert result["months_held"] is not None
        assert abs(result["months_held"] - 12.0) < 1.0

    def test_readiness_score_bounded(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        entry = dt.date.today() - dt.timedelta(days=4*365)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: _meta_row(entry_date=entry))
        rows = [(dt.date(2024, 3, 31), 100.0, 50.0, 30.0, 0.01, 100.0, 45.0)]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_exit_timing("ENT001")
        if result["exit_readiness_score"] is not None:
            assert 0.0 <= result["exit_readiness_score"] <= 100.0


# ---------------------------------------------------------------------------
# 3.4 — get_portfolio_overview
# ---------------------------------------------------------------------------

class TestGetPortfolioOverview:
    def _setup(self, monkeypatch, entity_rows, health_periods=None):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda sql, params: (
            entity_rows if "private_entities" in sql else (health_periods or [])
        ))

    def test_db_disabled_returns_early(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: False)
        result = mod.get_portfolio_overview("ORG001")
        assert result["status"] == "db_disabled"
        assert result["entities"] == []

    def test_empty_portfolio(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [])
        result = mod.get_portfolio_overview("ORG001")
        assert result["entity_count"] == 0
        assert result["avg_health_score"] is None
        assert result["alert_count"] == 0

    def test_returns_entity_count(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        entity_rows = [
            ("ENT001", "Acme", "Tech", dt.date(2022, 1, 1), dt.date(2026, 1, 1), "AAPL"),
            ("ENT002", "Beta", "Fin",  dt.date(2021, 6, 1), None, None),
        ]
        # Returns entity list from private_entities, then empty periods per entity
        call_count = [0]
        def fake_fetchall(sql, params):
            call_count[0] += 1
            if "private_entities" in sql:
                return entity_rows
            return []  # no periods → no_data health
        monkeypatch.setattr(mod.db, "safe_fetchall", fake_fetchall)
        result = mod.get_portfolio_overview("ORG001")
        assert result["entity_count"] == 2
        assert result["status"] == "ok"

    def test_alert_count_correct(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        entity_rows = [("ENT001", "Acme", "Tech", None, None, None)]
        # Distressed health: negative EBITDA
        periods = [(dt.date(2024, 3, 31), 50.0, -5.0, -8.0, 200.0, 3.0, -8.0)]
        def fake_fetchall(sql, params):
            if "private_entities" in sql:
                return entity_rows
            return periods
        monkeypatch.setattr(mod.db, "safe_fetchall", fake_fetchall)
        result = mod.get_portfolio_overview("ORG001")
        assert result["alert_count"] == 1


# ---------------------------------------------------------------------------
# 3.5 — get_portfolio_stress_alerts
# ---------------------------------------------------------------------------

class TestGetPortfolioStressAlerts:
    def test_returns_only_distressed_entities(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        entity_rows = [
            ("ENT001", "Healthy", "Tech",  None, None, None),
            ("ENT002", "Sick",    "Fin",   None, None, None),
        ]
        # ENT001 gets healthy periods, ENT002 gets distressed
        call_count = [0]
        def fake_fetchall(sql, params):
            if "private_entities" in sql:
                return entity_rows
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                # healthy
                return [
                    (dt.date(2024, 3, 31), 100.0, 30.0, 15.0, 60.0, 40.0, 22.0),
                    (dt.date(2023, 12, 31), 90.0, 27.0, 13.0, 60.0, 35.0, 20.0),
                    (dt.date(2023, 9, 30), 85.0, 25.0, 12.0, 65.0, 30.0, 18.0),
                    (dt.date(2023, 6, 30), 82.0, 24.0, 11.0, 65.0, 28.0, 17.0),
                ]
            else:
                return [(dt.date(2024, 3, 31), 50.0, -5.0, -8.0, 200.0, 3.0, -8.0)]
        monkeypatch.setattr(mod.db, "safe_fetchall", fake_fetchall)
        result = mod.get_portfolio_stress_alerts("ORG001")
        assert result["alert_count"] == 1
        assert result["alerts"][0]["entity_id"] == "ENT002"

    def test_no_alerts_when_all_healthy(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        entity_rows = [("ENT001", "Healthy", "Tech", None, None, None)]
        def fake_fetchall(sql, params):
            if "private_entities" in sql:
                return entity_rows
            return [
                (dt.date(2024, 3, 31), 100.0, 30.0, 15.0, 60.0, 40.0, 22.0),
                (dt.date(2023, 12, 31), 90.0, 27.0, 13.0, 60.0, 35.0, 20.0),
                (dt.date(2023, 9, 30), 85.0, 25.0, 12.0, 65.0, 30.0, 18.0),
                (dt.date(2023, 6, 30), 82.0, 24.0, 11.0, 65.0, 28.0, 17.0),
            ]
        monkeypatch.setattr(mod.db, "safe_fetchall", fake_fetchall)
        result = mod.get_portfolio_stress_alerts("ORG001")
        assert result["alert_count"] == 0
        assert result["alerts"] == []


# ---------------------------------------------------------------------------
# 3.6 — API endpoint structure
# ---------------------------------------------------------------------------

class TestPERouterEndpoints:
    def test_pe_router_has_all_endpoints(self):
        from api.jobs.pe_routes import router
        paths = {getattr(r, "path", None) for r in router.routes}
        assert "/pe/entity/financials" in paths
        assert "/pe/entity/{entity_id}/health" in paths
        assert "/pe/entity/{entity_id}/exit-timing" in paths
        assert "/pe/portfolio/{org_id}/overview" in paths
        assert "/pe/portfolio/{org_id}/stress-alerts" in paths

    def test_health_endpoint_returns_no_data_for_unknown_entity(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [])
        from api.jobs.pe_routes import get_entity_health
        result = get_entity_health("UNKNOWN")
        assert result["status"] == "no_data"

    def test_post_financials_endpoint_returns_stored_status(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_execute", lambda *a: None)
        from api.jobs.pe_routes import post_entity_financials, EntityFinancialsIn
        payload = EntityFinancialsIn(
            entity_id="ENT001",
            period_end="2024-03-31",
            revenue=100.0,
            ebitda=25.0,
        )
        result = post_entity_financials(payload)
        assert result["status"] == "stored"
        assert result["entity_id"] == "ENT001"

    def test_post_financials_endpoint_returns_failed_when_disabled(self, monkeypatch):
        import api.jobs.pe_intelligence as mod
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: False)
        from api.jobs.pe_routes import post_entity_financials, EntityFinancialsIn
        payload = EntityFinancialsIn(entity_id="ENT001", period_end="2024-03-31")
        result = post_entity_financials(payload)
        assert result["status"] == "failed"
