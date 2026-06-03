"""Phase 15 tests: Advanced SMB Platform."""
from __future__ import annotations

import pytest

from api.smb.pricing_intelligence import (
    SECTOR_PRICING_POWER_BASE,
    compute_input_cost_pressure,
    compute_pricing_power_score,
    generate_pricing_recommendation,
)
from api.smb.customer_concentration import (
    CustomerConcentrationReport,
    build_customer_concentration_report,
    compute_customer_axiom_monitoring,
    compute_customer_concentration_score,
    generate_concentration_alerts,
)
from api.smb.working_capital import (
    SECTOR_WC_BENCHMARKS,
    WorkingCapitalAnalysis,
    build_working_capital_analysis,
    compute_cash_conversion_cycle,
    compute_optimal_ccc,
    compute_trapped_working_capital,
    generate_wc_recommendations,
)
from api.smb.credit_intelligence import (
    build_credit_intelligence,
    compute_borrowing_capacity_score,
    compute_rate_environment_score,
    generate_credit_recommendation,
)
from api.smb.sector_modules import (
    compute_manufacturing_intelligence,
    compute_profservices_intelligence,
    compute_restaurant_intelligence,
    get_sector_module,
)


# ===========================================================================
# TestPricingIntelligence
# ===========================================================================

def _history_with_expanding_margins(delta: float = 0.05) -> list:
    """Most-recent-first list; most recent gm is delta higher than prior 4Q avg."""
    prior_gm = 0.40
    return [
        {"gross_margin": prior_gm + delta, "revenue_growth_yoy": 0.10, "cogs_growth_yoy": 0.04},
        {"gross_margin": prior_gm, "revenue_growth_yoy": 0.08},
        {"gross_margin": prior_gm, "revenue_growth_yoy": 0.07},
        {"gross_margin": prior_gm, "revenue_growth_yoy": 0.06},
        {"gross_margin": prior_gm, "revenue_growth_yoy": 0.06},
    ]


def _history_with_compressing_margins(delta: float = 0.05) -> list:
    prior_gm = 0.40
    return [
        {"gross_margin": prior_gm - delta, "revenue_growth_yoy": -0.02, "cogs_growth_yoy": 0.05},
        {"gross_margin": prior_gm, "revenue_growth_yoy": 0.01},
        {"gross_margin": prior_gm, "revenue_growth_yoy": 0.01},
        {"gross_margin": prior_gm, "revenue_growth_yoy": 0.01},
        {"gross_margin": prior_gm, "revenue_growth_yoy": 0.01},
    ]


class TestPricingIntelligence:
    def test_pricing_power_high_expanding_margins(self):
        history = _history_with_expanding_margins(0.05)
        score = compute_pricing_power_score(history, "Technology")
        assert score > 65, f"Expected > 65, got {score}"

    def test_pricing_power_low_compressing(self):
        history = _history_with_compressing_margins(0.05)
        score = compute_pricing_power_score(history, "Restaurant")
        assert score < 40, f"Expected < 40, got {score}"

    def test_pricing_power_sector_base_technology_higher_than_restaurant(self):
        tech_score = compute_pricing_power_score([], "Technology")
        rest_score = compute_pricing_power_score([], "Restaurant")
        assert tech_score > rest_score

    def test_input_cost_pressure_high(self):
        history = [{"cogs_growth_yoy": 0.15, "revenue_growth_yoy": 0.05, "cogs_pct_revenue": 0.65}]
        score = compute_input_cost_pressure(history)
        assert score > 65, f"Expected > 65, got {score}"

    def test_input_cost_pressure_low_same_pace(self):
        history = [{"cogs_growth_yoy": 0.07, "revenue_growth_yoy": 0.07, "cogs_pct_revenue": 0.60}]
        score = compute_input_cost_pressure(history)
        assert 40 <= score <= 65, f"Expected ~50, got {score}"

    def test_recommendation_raise_high_power_high_pressure(self):
        result = generate_pricing_recommendation(70.0, 70.0, "compressing")
        assert result["action"] == "raise_prices"

    def test_recommendation_defend_low_power_high_pressure(self):
        result = generate_pricing_recommendation(30.0, 70.0, "compressing")
        assert result["action"] == "defend_volume"

    def test_recommendation_hold_low_power_low_pressure(self):
        result = generate_pricing_recommendation(30.0, 30.0, "stable")
        assert result["action"] == "hold"

    def test_price_increase_potential_bounded(self):
        for power in [0.0, 25.0, 50.0, 75.0, 100.0]:
            for pressure in [0.0, 50.0, 100.0]:
                result = generate_pricing_recommendation(power, pressure, "stable")
                p = result["price_increase_potential_pct"]
                assert 0.0 <= p <= 0.30, f"Out of bounds: power={power}, pressure={pressure}, p={p}"


# ===========================================================================
# TestCustomerConcentration
# ===========================================================================

class TestCustomerConcentration:
    def test_hhi_single_customer(self):
        result = compute_customer_concentration_score({"c1": 100_000})
        assert result["concentration_risk_score"] == 100.0

    def test_hhi_equal_5_customers(self):
        breakdown = {f"c{i}": 20_000 for i in range(5)}
        result = compute_customer_concentration_score(breakdown)
        assert abs(result["concentration_risk_score"] - 20.0) < 0.1

    def test_label_critical_high_concentration(self):
        result = compute_customer_concentration_score({"c1": 80_000, "c2": 20_000})
        assert result["concentration_risk_score"] > 50
        assert result["concentration_label"] == "critical"

    def test_label_safe_low_concentration(self):
        breakdown = {f"c{i}": 5_000 for i in range(20)}
        result = compute_customer_concentration_score(breakdown)
        assert result["concentration_risk_score"] < 10
        assert result["concentration_label"] == "safe"

    def test_revenue_at_risk_computed(self):
        breakdown = {"big": 400_000, "small": 600_000}
        report = build_customer_concentration_report(
            "e1", breakdown, annual_revenue=1_000_000
        )
        # top customer is "small" at 60% → at_risk = 0.60 × 1M = 600K
        assert report.estimated_revenue_at_risk == pytest.approx(600_000, rel=0.01)

    def test_alert_structural_risk_on_high_concentration(self):
        breakdown = {"big": 600_000, "other": 400_000}
        report = build_customer_concentration_report("e1", breakdown, annual_revenue=1_000_000)
        alert_types = [a["alert_type"] for a in report.alerts]
        assert "structural_risk" in alert_types

    def test_axiom_monitoring_empty_no_tickers(self):
        result = compute_customer_axiom_monitoring({})
        assert result == []

    def test_concentration_score_bounded(self):
        for n in [1, 2, 5, 10, 50]:
            breakdown = {f"c{i}": 1_000 for i in range(n)}
            result = compute_customer_concentration_score(breakdown)
            s = result["concentration_risk_score"]
            assert 0.0 <= s <= 100.0, f"Score {s} out of bounds for n={n}"


# ===========================================================================
# TestWorkingCapital
# ===========================================================================

def _financials_wc(ar=90_000, revenue=365_000, cogs=219_000, inventory=30_000, ap=30_000):
    return {
        "accounts_receivable": ar,
        "revenue": revenue,
        "cogs": cogs,
        "inventory": inventory,
        "accounts_payable": ap,
    }


class TestWorkingCapital:
    def test_ccc_computed(self):
        # AR_days=90, Inventory_days≈50, AP_days=50 → CCC=90
        f = _financials_wc(
            ar=90_000, revenue=365_000,
            cogs=219_000, inventory=30_000, ap=30_000
        )
        result = compute_cash_conversion_cycle(f)
        # AR_days = 90000/365000*365 = 90
        # AP_days = 30000/219000*365 ≈ 50
        # Inventory_days = 30000/219000*365 ≈ 50
        # CCC = 90 + 50 - 50 = 90
        assert abs(result["ar_days"] - 90.0) < 0.5
        assert abs(result["ccc"] - 90.0) < 1.0

    def test_optimal_ccc_technology_lower_than_manufacturing(self):
        tech = compute_optimal_ccc("Technology")
        mfg = compute_optimal_ccc("Manufacturing")
        assert tech["ccc"] < mfg["ccc"]

    def test_trapped_wc_zero_when_at_optimal(self):
        trapped = compute_trapped_working_capital(15.0, 15.0, 1_000_000)
        assert trapped == 0.0

    def test_trapped_wc_zero_when_below_optimal(self):
        trapped = compute_trapped_working_capital(10.0, 15.0, 1_000_000)
        assert trapped == 0.0

    def test_trapped_wc_positive_when_above_optimal(self):
        # CCC 60 days, optimal 15 days, 1M revenue
        # trapped = (60-15)/365 * 1M ≈ 123K
        trapped = compute_trapped_working_capital(60.0, 15.0, 1_000_000)
        assert trapped > 0.0

    def test_potential_cash_release_never_negative(self):
        for ccc, opt in [(10, 50), (50, 10), (50, 50)]:
            analysis = build_working_capital_analysis(
                "e1",
                _financials_wc(ar=ccc / 365 * 365_000),
                sector="Technology",
            )
            assert analysis.potential_cash_release_usd >= 0.0

    def test_ar_recommendation_triggered(self):
        # Technology benchmark ar_days=45; set AR so ar_days=70 (>45+10)
        # ar_days = ar / revenue * 365 → ar = 70/365 * revenue
        revenue = 365_000
        ar = int(70 / 365 * revenue)
        f = _financials_wc(ar=ar, revenue=revenue, cogs=219_000, inventory=0, ap=15_000)
        analysis = build_working_capital_analysis("e1", f, sector="Technology")
        actions = [r["action"] for r in analysis.recommendations]
        assert any("collect" in a or "factor" in a for a in actions), f"No AR rec. Actions: {actions}"

    def test_ap_recommendation_triggered(self):
        # Technology benchmark ap_days=30; set AP so ap_days=15 (<30-10)
        # ap_days = ap / cogs * 365 → ap = 15/365 * cogs
        cogs = 219_000
        ap = int(15 / 365 * cogs)
        f = _financials_wc(ar=45_000, revenue=365_000, cogs=cogs, inventory=0, ap=ap)
        analysis = build_working_capital_analysis("e1", f, sector="Technology")
        actions = [r["action"] for r in analysis.recommendations]
        assert "extend_payables" in actions, f"AP rec missing. Actions: {actions}"

    def test_recommendations_have_required_keys(self):
        f = _financials_wc(ar=120_000, revenue=365_000, cogs=219_000, inventory=50_000, ap=5_000)
        analysis = build_working_capital_analysis("e1", f, sector="Technology")
        for rec in analysis.recommendations:
            assert "action" in rec
            assert "estimated_cash_release_usd" in rec


# ===========================================================================
# TestCreditIntelligence
# ===========================================================================

def _strong_financials():
    return {
        "ebitda": 500_000,
        "annual_debt_service": 200_000,  # DSCR = 2.5
        "total_debt": 500_000,           # leverage = 1.0
        "ebitda_margin": 0.25,
        "revenue_growth_yoy": 0.10,
        "revenue": 2_000_000,
    }


def _distressed_financials():
    return {
        "ebitda": 50_000,
        "annual_debt_service": 100_000,  # DSCR = 0.5
        "total_debt": 500_000,           # leverage = 10.0
        "ebitda_margin": 0.05,
        "revenue_growth_yoy": -0.10,
        "revenue": 1_000_000,
    }


class TestCreditIntelligence:
    def test_dscr_strong_high_capacity(self):
        result = compute_borrowing_capacity_score(_strong_financials())
        assert result["borrowing_capacity_score"] > 70

    def test_dscr_distressed_low_capacity(self):
        result = compute_borrowing_capacity_score(_distressed_financials())
        assert result["borrowing_capacity_score"] < 40

    def test_rate_environment_favorable_recovery(self):
        result = compute_rate_environment_score("RECOVERY")
        assert result["score"] > 60
        assert result["label"] == "favorable"

    def test_rate_environment_elevated_high_vol(self):
        result = compute_rate_environment_score("HIGH_VOL")
        assert result["score"] < 40

    def test_recommendation_borrow_now_high_capacity_favorable(self):
        # capacity_score=80 (>70), rate_score=75 (>60 = favorable)
        rec = generate_credit_recommendation(80.0, 75.0, 100_000, 1_000_000)
        assert rec == "borrow_now"

    def test_recommendation_wait_high_capacity_elevated_rates(self):
        # capacity_score=80 (>70), rate_score=35 (<40 = elevated)
        rec = generate_credit_recommendation(80.0, 35.0, 100_000, 1_000_000)
        assert rec == "wait"

    def test_borrowing_capacity_bounded(self):
        for f in [_strong_financials(), _distressed_financials()]:
            result = compute_borrowing_capacity_score(f)
            s = result["borrowing_capacity_score"]
            assert 0.0 <= s <= 100.0


# ===========================================================================
# TestSectorModules
# ===========================================================================

def _restaurant_ops(food_pct: float, labor_pct: float, seats: int = 60):
    return {"food_cost_pct": food_pct, "labor_cost_pct": labor_pct, "seats": seats, "revenue": 600_000}


def _restaurant_financials(revenue: float = 600_000):
    return {"revenue": revenue}


class TestSectorModules:
    def test_restaurant_prime_cost_critical(self):
        metrics = compute_restaurant_intelligence(
            _restaurant_financials(),
            _restaurant_ops(food_pct=0.38, labor_pct=0.35),
        )
        assert metrics.prime_cost_pct > 0.70
        assert metrics.restaurant_intelligence_score < 40

    def test_restaurant_prime_cost_excellent(self):
        metrics = compute_restaurant_intelligence(
            _restaurant_financials(),
            _restaurant_ops(food_pct=0.28, labor_pct=0.28),
        )
        assert metrics.prime_cost_pct < 0.60
        assert metrics.restaurant_intelligence_score > 65

    def test_restaurant_food_cost_optimal(self):
        metrics = compute_restaurant_intelligence(
            _restaurant_financials(),
            _restaurant_ops(food_pct=0.30, labor_pct=0.29),
        )
        # 28-32% food cost is optimal — should not penalize score
        assert metrics.food_cost_pct == pytest.approx(0.30)
        assert metrics.restaurant_intelligence_score > 55

    def test_manufacturing_inventory_turns_healthy(self):
        metrics = compute_manufacturing_intelligence(
            {"revenue": 1_000_000, "cogs": 650_000, "gross_margin": 0.35},
            {"inventory_turns": 6.0, "top_supplier_concentration": 0.30},
        )
        assert metrics.manufacturing_intelligence_score > 55

    def test_manufacturing_inventory_turns_poor(self):
        score_good = compute_manufacturing_intelligence(
            {"revenue": 1_000_000, "cogs": 650_000, "gross_margin": 0.35},
            {"inventory_turns": 6.0, "top_supplier_concentration": 0.30},
        ).manufacturing_intelligence_score

        score_bad = compute_manufacturing_intelligence(
            {"revenue": 1_000_000, "cogs": 650_000, "gross_margin": 0.35},
            {"inventory_turns": 1.0, "top_supplier_concentration": 0.30},
        ).manufacturing_intelligence_score

        assert score_bad < score_good

    def test_profservices_revenue_per_employee_healthy(self):
        # $130K per employee: revenue=1.3M, headcount=10
        metrics = compute_profservices_intelligence(
            {"revenue": 1_300_000},
            {"headcount": 10},
        )
        assert metrics.revenue_per_employee == pytest.approx(130_000)
        assert metrics.profservices_intelligence_score > 60

    def test_profservices_revenue_per_employee_poor(self):
        # $50K per employee: revenue=500K, headcount=10
        metrics = compute_profservices_intelligence(
            {"revenue": 500_000},
            {"headcount": 10},
        )
        assert metrics.revenue_per_employee == pytest.approx(50_000)
        assert metrics.profservices_intelligence_score < 40

    def test_sector_routing_restaurant(self):
        result = get_sector_module(
            "Restaurant",
            {"revenue": 600_000},
            {"food_cost_pct": 0.30, "labor_cost_pct": 0.32, "seats": 60},
        )
        assert result["status"] == "ok"
        assert "restaurant_intelligence_score" in result

    def test_sector_routing_manufacturing(self):
        result = get_sector_module(
            "Manufacturing",
            {"revenue": 1_000_000, "cogs": 650_000, "gross_margin": 0.35},
            {"inventory_turns": 5.0, "top_supplier_concentration": 0.30},
        )
        assert result["status"] == "ok"
        assert "manufacturing_intelligence_score" in result

    def test_sector_routing_unknown_returns_no_module(self):
        result = get_sector_module("Accounting", {}, {})
        assert result["status"] == "no_module"
