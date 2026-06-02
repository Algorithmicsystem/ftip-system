"""Phase 34: SMB Intelligence Module tests.

Covers:
  4.1  store_smb_financials (upsert)
  4.2  forecast_cash_flow (12-month projection)
  4.3  compute_supplier_risks (concentration + AP growth + cash buffer)
  4.4  API endpoint structure
"""
from __future__ import annotations

import datetime as dt
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _months_ago(n: int) -> dt.date:
    d = dt.date.today()
    month = d.month - 1 - n
    year = d.year + month // 12
    month = month % 12 + 1
    import calendar
    return dt.date(year, month, calendar.monthrange(year, month)[1])


def _make_history(n: int = 6, base_revenue=100.0, revenue_trend=2.0,
                  base_cost=70.0, cost_trend=1.0, base_cash=200.0) -> list:
    """Return n monthly rows (DESC) for safe_fetchall mocking."""
    rows = []
    for i in range(n):
        month = _months_ago(i)
        rev = base_revenue + revenue_trend * (n - 1 - i)
        cost_op = base_cost + cost_trend * (n - 1 - i)
        rows.append((
            month,
            rev,            # revenue
            cost_op * 0.5,  # cogs
            cost_op * 0.3,  # operating_expenses
            rev - cost_op,  # net_income
            base_cash,      # cash_balance
            cost_op * 0.2,  # payroll
        ))
    return rows


def _make_supplier_rows(
    concentration=0.30,
    ap=50.0,
    cash=100.0,
    cogs=50.0,
    n=2,
) -> list:
    """Rows for the supplier concentration SELECT (month_end, conc, ap, cash, cogs)."""
    rows = []
    for i in range(n):
        rows.append((_months_ago(i), concentration, ap * (1 - i * 0.05), cash, cogs))
    return rows


# ---------------------------------------------------------------------------
# 4.1 — store_smb_financials
# ---------------------------------------------------------------------------

class TestStoreSMBFinancials:
    def test_returns_false_when_write_disabled(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: False)
        result = mod.store_smb_financials("SMB001", dt.date(2024, 3, 31), {"revenue": 100.0})
        assert result is False

    def test_returns_true_on_success(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        executed = []
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_execute", lambda sql, params: executed.append(params))
        result = mod.store_smb_financials("SMB001", dt.date(2024, 3, 31), {"revenue": 100.0})
        assert result is True
        assert executed[0][0] == "SMB001"
        assert executed[0][1] == dt.date(2024, 3, 31)

    def test_returns_false_on_exception(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_execute", lambda *a: (_ for _ in ()).throw(Exception("fail")))
        result = mod.store_smb_financials("SMB001", dt.date(2024, 3, 31), {})
        assert result is False

    def test_missing_fields_stored_as_none(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        executed = []
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_execute", lambda sql, params: executed.append(params))
        mod.store_smb_financials("SMB001", dt.date(2024, 3, 31), {})
        assert executed[0][2] is None  # revenue


# ---------------------------------------------------------------------------
# 4.2 — forecast_cash_flow
# ---------------------------------------------------------------------------

class TestForecastCashFlow:
    def test_no_data_returns_no_data_status(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [])
        result = mod.forecast_cash_flow("SMB001")
        assert result["status"] == "no_data"
        assert result["forecast"] == []

    def test_forecast_has_correct_number_of_periods(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: _make_history())
        result = mod.forecast_cash_flow("SMB001", horizon_months=12)
        assert result["status"] == "ok"
        assert len(result["forecast"]) == 12

    def test_forecast_horizon_respected(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: _make_history())
        result = mod.forecast_cash_flow("SMB001", horizon_months=6)
        assert len(result["forecast"]) == 6

    def test_growing_revenue_produces_positive_trend(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: _make_history(revenue_trend=5.0, cost_trend=0.0))
        result = mod.forecast_cash_flow("SMB001")
        assert result["revenue_monthly_trend"] > 0.0

    def test_cash_runway_none_when_always_positive(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        # High revenue, low cost, large cash balance
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: _make_history(
            base_revenue=200.0, base_cost=50.0, base_cash=5000.0
        ))
        result = mod.forecast_cash_flow("SMB001")
        assert result["cash_runway_months"] is None
        assert result["runway_status"] == "healthy"

    def test_cash_runway_detected_when_burning_cash(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        # Low revenue, high cost, tiny cash balance
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: _make_history(
            base_revenue=50.0, base_cost=150.0, base_cash=100.0,
            revenue_trend=0.0, cost_trend=0.0
        ))
        result = mod.forecast_cash_flow("SMB001")
        assert result["cash_runway_months"] is not None
        assert result["cash_runway_months"] < 12
        assert result["runway_status"] in ("critical", "warning", "caution")

    def test_forecast_period_has_expected_keys(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: _make_history())
        result = mod.forecast_cash_flow("SMB001", horizon_months=1)
        period = result["forecast"][0]
        assert "month" in period
        assert "projected_revenue" in period
        assert "projected_costs" in period
        assert "net_cash_flow" in period
        assert "cumulative_cash" in period

    def test_cumulative_cash_starts_from_latest_balance(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        history = _make_history(base_cash=500.0, base_revenue=100.0, base_cost=80.0)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: history)
        result = mod.forecast_cash_flow("SMB001", horizon_months=1)
        assert result["latest_cash_balance"] == pytest.approx(500.0)

    def test_critical_runway_status_when_very_low(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        # Burn $95/mo with only $50 cash → runway = 0 months
        history = _make_history(base_revenue=5.0, base_cost=100.0, base_cash=50.0,
                                revenue_trend=0.0, cost_trend=0.0)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: history)
        result = mod.forecast_cash_flow("SMB001")
        assert result["runway_status"] == "critical"


# ---------------------------------------------------------------------------
# 4.3 — compute_supplier_risks
# ---------------------------------------------------------------------------

class TestComputeSupplierRisks:
    def test_db_disabled_returns_early(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: False)
        result = mod.compute_supplier_risks("SMB001")
        assert result["status"] == "db_disabled"

    def test_no_data_status_when_empty(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [])
        result = mod.compute_supplier_risks("SMB001")
        assert result["status"] == "no_data"

    def test_low_risk_when_all_healthy(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        # Low concentration, healthy AP, ample cash
        rows = _make_supplier_rows(concentration=0.20, ap=50.0, cash=400.0, cogs=50.0)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_supplier_risks("SMB001")
        assert result["overall_risk"] == "low"
        assert result["risk_count"] == 0

    def test_high_concentration_triggers_risk(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        rows = _make_supplier_rows(concentration=0.80, cash=400.0, cogs=50.0)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_supplier_risks("SMB001")
        risk_types = [r["risk_type"] for r in result["risks"]]
        assert "supplier_concentration" in risk_types
        assert result["overall_risk"] == "high"

    def test_medium_concentration_is_medium_severity(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        rows = _make_supplier_rows(concentration=0.60, cash=400.0, cogs=50.0)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_supplier_risks("SMB001")
        conc_risk = next(r for r in result["risks"] if r["risk_type"] == "supplier_concentration")
        assert conc_risk["severity"] == "medium"

    def test_low_cash_buffer_triggers_risk(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        # Cash = 30, COGS = 50 → 0.6 months buffer → critical
        rows = _make_supplier_rows(concentration=0.20, ap=50.0, cash=30.0, cogs=50.0)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_supplier_risks("SMB001")
        risk_types = [r["risk_type"] for r in result["risks"]]
        assert "low_cash_buffer" in risk_types

    def test_ap_growth_triggers_risk(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        # AP grew 50%: latest=150, prior=100
        rows = [
            (_months_ago(0), 0.20, 150.0, 200.0, 50.0),  # latest
            (_months_ago(1), 0.20, 100.0, 200.0, 50.0),  # prior
        ]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_supplier_risks("SMB001")
        risk_types = [r["risk_type"] for r in result["risks"]]
        assert "accounts_payable_growth" in risk_types

    def test_result_has_expected_keys(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        rows = _make_supplier_rows()
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.compute_supplier_risks("SMB001")
        assert "overall_risk" in result
        assert "top_supplier_concentration" in result
        assert "cash_buffer_months" in result
        assert "risk_count" in result
        assert "risks" in result


# ---------------------------------------------------------------------------
# 4.4 — API endpoint structure
# ---------------------------------------------------------------------------

class TestSMBRouterEndpoints:
    def test_smb_router_has_all_endpoints(self):
        from api.jobs.smb_routes import router
        paths = {getattr(r, "path", None) for r in router.routes}
        assert "/smb/entity/financials" in paths
        assert "/smb/entity/{entity_id}/cash-flow-forecast" in paths
        assert "/smb/entity/{entity_id}/supplier-risks" in paths

    def test_post_financials_returns_stored(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_execute", lambda *a: None)
        from api.jobs.smb_routes import post_smb_financials, SMBFinancialsIn
        payload = SMBFinancialsIn(entity_id="SMB001", month_end="2024-03-31", revenue=100.0)
        result = post_smb_financials(payload)
        assert result["status"] == "stored"
        assert result["entity_id"] == "SMB001"

    def test_post_financials_returns_failed_when_disabled(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: False)
        from api.jobs.smb_routes import post_smb_financials, SMBFinancialsIn
        payload = SMBFinancialsIn(entity_id="SMB001", month_end="2024-03-31")
        result = post_smb_financials(payload)
        assert result["status"] == "failed"

    def test_cash_flow_forecast_endpoint_no_data(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [])
        from api.jobs.smb_routes import get_cash_flow_forecast
        result = get_cash_flow_forecast("UNKNOWN")
        assert result["status"] == "no_data"

    def test_supplier_risks_endpoint_no_data(self, monkeypatch):
        import api.jobs.smb_intelligence as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [])
        from api.jobs.smb_routes import get_supplier_risks
        result = get_supplier_risks("UNKNOWN")
        assert result["status"] == "no_data"
