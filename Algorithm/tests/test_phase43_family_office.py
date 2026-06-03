"""Phase 14 tests: Family Office and RIA Intelligence Layer."""
from __future__ import annotations

import datetime as dt
import pytest
from typing import List

from api.family_office.multi_asset import (
    ASSET_CLASS_PROFILES,
    PortfolioPosition,
    PortfolioSnapshot,
    build_portfolio_snapshot,
    compute_asset_class_allocation,
    compute_duration_risk_score,
    compute_portfolio_axiom_score,
)
from api.family_office.concentration_risk import (
    ConcentrationRiskReport,
    build_concentration_report,
    compute_concentration_index,
    compute_single_stock_risk,
    compute_tax_aware_diversification,
    generate_diversification_recommendations,
)
from api.family_office.goal_based import (
    InvestmentGoal,
    compute_goal_funding_status,
    compute_liability_matching_score,
    compute_probability_of_success,
    generate_goal_based_recommendations,
)
from api.family_office.estate_intelligence import (
    build_estate_intelligence_report,
    compute_dynasty_trust_analysis,
    compute_estate_tax_exposure,
    compute_gifting_intelligence,
)
from api.family_office.ria_intelligence import (
    RIAClientProfile,
    compute_client_review_trigger,
    compute_ria_book_analytics,
    generate_client_brief,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _pos(
    pid: str,
    asset_class: str,
    ticker: str,
    weight: float,
    value: float = 100_000,
    cost: float = 80_000,
    gain_pct: float = 0.10,
    axiom: float = None,
    metadata: dict = None,
) -> PortfolioPosition:
    return PortfolioPosition(
        position_id=pid,
        asset_class=asset_class,
        ticker_or_id=ticker,
        weight=weight,
        current_value_usd=value,
        cost_basis_usd=cost,
        unrealized_gain_pct=gain_pct,
        axiom_score=axiom,
        metadata=metadata or {},
    )


def _equal_portfolio(n: int = 4) -> List[PortfolioPosition]:
    w = 1.0 / n
    return [
        _pos(f"p{i}", "equity_us", f"TICK{i}", w, w * 1_000_000)
        for i in range(n)
    ]


# ===========================================================================
# TestMultiAsset
# ===========================================================================

class TestMultiAsset:
    def test_eight_asset_class_profiles(self):
        assert len(ASSET_CLASS_PROFILES) == 8

    def test_allocation_sums_to_one(self):
        positions = [
            _pos("a", "equity_us", "AAPL", 0.50, 500_000),
            _pos("b", "fixed_income_investment_grade", "TLT", 0.30, 300_000),
            _pos("c", "cash", "CASH", 0.20, 200_000),
        ]
        alloc = compute_asset_class_allocation(positions)
        assert abs(sum(alloc.values()) - 1.0) < 1e-6

    def test_equity_weight_correct(self):
        positions = [
            _pos("a", "equity_us", "AAPL", 0.40),
            _pos("b", "equity_international", "EFA", 0.20),
            _pos("c", "cash", "CASH", 0.40),
        ]
        snap = build_portfolio_snapshot("p1", "FO", positions)
        assert abs(snap.equity_weight - 0.60) < 1e-6

    def test_duration_risk_high_regime_penalized(self):
        score_normal = compute_duration_risk_score(7.0, "TRENDING", 4.5)
        score_highvol = compute_duration_risk_score(7.0, "HIGH_VOL", 4.5)
        assert score_highvol < score_normal

    def test_duration_risk_bounded(self):
        for dur in [0.0, 5.0, 10.0, 20.0]:
            score = compute_duration_risk_score(dur, "TRENDING", 4.5)
            assert 0.0 <= score <= 100.0

    def test_portfolio_axiom_score_bounded(self):
        positions = [
            _pos("a", "equity_us", "AAPL", 0.60, axiom=75.0),
            _pos("b", "cash", "CASH", 0.40),
        ]
        result = compute_portfolio_axiom_score(positions, dt.date.today())
        assert 0.0 <= result["weighted_axiom_score"] <= 100.0


# ===========================================================================
# TestConcentrationRisk
# ===========================================================================

class TestConcentrationRisk:
    def test_hhi_one_position_equals_one(self):
        positions = [_pos("p1", "equity_us", "AAPL", 1.0, 1_000_000)]
        assert compute_concentration_index(positions) == 1.0

    def test_hhi_equal_weight_is_one_over_n(self):
        n = 5
        positions = _equal_portfolio(n)
        hhi = compute_concentration_index(positions)
        assert abs(hhi - 1.0 / n) < 1e-5

    def test_hhi_bounded_between_zero_and_one(self):
        for n in [1, 2, 5, 10, 20]:
            positions = _equal_portfolio(n)
            hhi = compute_concentration_index(positions)
            assert 0.0 <= hhi <= 1.0

    def test_single_stock_extreme_risk(self):
        # Large weight + high fragility → extreme
        axiom_payload = {
            "engine_scores": {
                "critical_fragility": {
                    "score": 90.0,
                    "components": {"mtrs_component": 90.0, "scps_component": 50.0},
                }
            }
        }
        result = compute_single_stock_risk("AAPL", 0.50, axiom_payload, 1_000_000)
        assert result["risk_label"] == "extreme"

    def test_single_stock_manageable_risk(self):
        # Small weight + low fragility → manageable
        axiom_payload = {
            "engine_scores": {
                "critical_fragility": {
                    "score": 20.0,
                    "components": {"mtrs_component": 20.0, "scps_component": 20.0},
                }
            }
        }
        result = compute_single_stock_risk("MSFT", 0.05, axiom_payload, 1_000_000)
        assert result["risk_label"] == "manageable"

    def test_recommendation_reduce_on_large_fragile(self):
        positions = [
            _pos("p1", "equity_us", "AAPL", 0.40, 400_000, axiom=55.0,
                 metadata={"axiom_payload": {
                     "engine_scores": {"critical_fragility": {"score": 70.0, "components": {}}}
                 }}),
            _pos("p2", "equity_us", "MSFT", 0.30, 300_000),
            _pos("p3", "cash", "CASH", 0.30, 300_000),
        ]
        report = build_concentration_report("port1", positions, 1_000_000)
        actions = [r["action"] for r in report.diversification_recommendations]
        assert "reduce" in actions

    def test_tax_loss_harvest_candidates(self):
        positions = [
            _pos("p1", "equity_us", "AAPL", 0.50, 500_000, gain_pct=-0.10),
            _pos("p2", "equity_us", "MSFT", 0.50, 500_000, gain_pct=0.20),
        ]
        result = compute_tax_aware_diversification(positions)
        assert len(result["loss_harvest_candidates"]) == 1
        assert result["loss_harvest_candidates"][0]["ticker_or_id"] == "AAPL"

    def test_tax_high_cost_sorted_descending(self):
        positions = [
            _pos("p1", "equity_us", "AAPL", 0.40, 400_000, gain_pct=0.50),
            _pos("p2", "equity_us", "MSFT", 0.30, 300_000, gain_pct=0.80),
            _pos("p3", "equity_us", "GOOG", 0.30, 300_000, gain_pct=0.20),
        ]
        result = compute_tax_aware_diversification(positions)
        high_cost = result["high_cost_positions"]
        assert len(high_cost) == 3
        gains = [h["unrealized_gain_pct"] for h in high_cost]
        assert gains == sorted(gains, reverse=True)


# ===========================================================================
# TestGoalBased
# ===========================================================================

def _make_goal(
    goal_id="g1",
    goal_type="retirement",
    label="Retirement",
    target=2_000_000,
    years=10.0,
    req_return=0.07,
    risk_budget=0.60,
) -> InvestmentGoal:
    return InvestmentGoal(
        goal_id=goal_id,
        goal_type=goal_type,
        label=label,
        target_amount_usd=target,
        target_date_years=years,
        current_funding_usd=0.0,
        required_return_annual=req_return,
        risk_budget=risk_budget,
    )


class TestGoalBased:
    def test_goal_on_track(self):
        goal = _make_goal(target=1_000_000, years=10.0, req_return=0.07)
        result = compute_goal_funding_status(goal, 1_000_000, 0.10)
        assert result["on_track"] is True

    def test_goal_off_track(self):
        goal = _make_goal(target=10_000_000, years=5.0, req_return=0.07)
        result = compute_goal_funding_status(goal, 100_000, 0.07)
        assert result["on_track"] is False
        assert result["status"] == "underfunded"

    def test_probability_bounded(self):
        result = compute_probability_of_success(0.07, 0.07, 0.15, 10.0)
        assert 0.0 <= result["probability_of_success"] <= 1.0

    def test_probability_high_when_low_required(self):
        result = compute_probability_of_success(0.01, 0.10, 0.05, 10.0)
        assert result["probability_of_success"] > 0.80

    def test_probability_low_when_very_high_required(self):
        result = compute_probability_of_success(0.50, 0.07, 0.15, 5.0)
        assert result["probability_of_success"] < 0.20

    def test_liability_match_covered(self):
        positions = [
            _pos("p1", "equity_us", "AAPL", 0.70, 700_000),
            _pos("p2", "cash", "CASH", 0.30, 300_000),
        ]
        snap = build_portfolio_snapshot("p1", "FO", positions)
        liabilities = [{"label": "near_term", "amount_usd": 100_000, "years_until_due": 1.0}]
        result = compute_liability_matching_score(snap, liabilities)
        assert result["short_term_covered"] is True

    def test_liability_match_underfunded(self):
        positions = [_pos("p1", "private_equity", "PE1", 1.0, 500_000)]
        snap = build_portfolio_snapshot("p1", "FO", positions)
        liabilities = [{"label": "urgent", "amount_usd": 600_000, "years_until_due": 0.5}]
        result = compute_liability_matching_score(snap, liabilities)
        assert result["short_term_covered"] is False

    def test_recommendations_returned(self):
        positions = _equal_portfolio(4)
        snap = build_portfolio_snapshot("p1", "FO", positions)
        goals = [_make_goal(target=5_000_000, years=5.0, risk_budget=0.30)]
        recs = generate_goal_based_recommendations(goals, snap, "HIGH_VOL")
        assert isinstance(recs, list)
        assert len(recs) > 0


# ===========================================================================
# TestEstateIntelligence
# ===========================================================================

class TestEstateIntelligence:
    def test_tax_below_exemption_is_zero(self):
        result = compute_estate_tax_exposure(5_000_000, estate_exemption_usd=13_610_000)
        assert result["federal_estate_tax_usd"] == 0.0
        assert result["total_estate_tax_usd"] == 0.0

    def test_tax_above_exemption_positive(self):
        result = compute_estate_tax_exposure(30_000_000, estate_exemption_usd=13_610_000)
        assert result["federal_estate_tax_usd"] > 0.0

    def test_effective_rate_bounded(self):
        for val in [5_000_000, 20_000_000, 50_000_000, 100_000_000]:
            result = compute_estate_tax_exposure(val)
            assert 0.0 <= result["effective_estate_tax_rate"] <= 0.40

    def test_gifting_reduces_estate(self):
        result = compute_gifting_intelligence(10_000_000, n_recipients=4, years=10)
        assert result["estate_value_after_gifting_usd"] < 10_000_000
        assert result["estimated_estate_tax_savings_usd"] > 0.0

    def test_dynasty_advantage_positive(self):
        result = compute_dynasty_trust_analysis(10_000_000, 0.07, 100)
        assert result["dynasty_advantage_usd"] > 0.0

    def test_dynasty_grows_with_time(self):
        r10 = compute_dynasty_trust_analysis(10_000_000, 0.07, 10)
        r100 = compute_dynasty_trust_analysis(10_000_000, 0.07, 100)
        assert r100["dynasty_terminal_value_usd"] > r10["dynasty_terminal_value_usd"]


# ===========================================================================
# TestRIAIntelligence
# ===========================================================================

def _make_client(
    client_id="c1",
    name="Alice Family",
    value=5_000_000,
    risk="moderate",
    horizon=20.0,
    income_need=200_000,
    tax_bracket=0.37,
    esg=False,
    axiom=65.0,
    last_review=None,
    sri=50.0,
) -> RIAClientProfile:
    return RIAClientProfile(
        client_id=client_id,
        client_name=name,
        portfolio_value_usd=value,
        risk_tolerance=risk,
        time_horizon_years=horizon,
        income_need_annual=income_need,
        tax_bracket=tax_bracket,
        esg_preference=esg,
        axiom_score=axiom,
        last_review_date=last_review,
        sri_score=sri,
    )


class TestRIAIntelligence:
    def test_review_trigger_high_sri(self):
        client = _make_client(sri=75.0)
        result = compute_client_review_trigger(client, "TRENDING", 75.0)
        assert "systemic_risk_elevated" in result["triggers"]

    def test_no_review_needed_healthy_client(self):
        client = _make_client(
            sri=40.0,
            axiom=70.0,
            last_review=dt.date.today() - dt.timedelta(days=10),
        )
        result = compute_client_review_trigger(client, "TRENDING", 40.0)
        assert result["urgency"] == "routine"

    def test_urgency_urgent_when_sri_above_80(self):
        client = _make_client(sri=85.0)
        result = compute_client_review_trigger(client, "TRENDING", 85.0)
        assert result["urgency"] == "urgent"

    def test_brief_structure(self):
        client = _make_client()
        positions = [
            _pos("a", "equity_us", "AAPL", 0.60, 3_000_000),
            _pos("b", "fixed_income_investment_grade", "TLT", 0.30, 1_500_000),
            _pos("c", "cash", "CASH", 0.10, 500_000),
        ]
        snap = build_portfolio_snapshot("c1", "Alice Family", positions)
        ctx = {"weighted_axiom_score": 65.0, "regime_label": "TRENDING", "sri_score": 50.0}
        brief = generate_client_brief(client, snap, ctx)
        for key in ("client_id", "market_outlook", "review_urgency", "key_messages", "regime"):
            assert key in brief

    def test_brief_key_messages_non_empty(self):
        client = _make_client()
        positions = [_pos("a", "equity_us", "AAPL", 1.0, 5_000_000)]
        snap = build_portfolio_snapshot("c1", "Alice Family", positions)
        ctx = {"weighted_axiom_score": 65.0, "regime_label": "TRENDING", "sri_score": 50.0}
        brief = generate_client_brief(client, snap, ctx)
        assert len(brief["key_messages"]) > 0

    def test_book_analytics_structure(self):
        clients = [
            _make_client("c1", value=5_000_000),
            _make_client("c2", value=3_000_000, risk="conservative"),
        ]
        result = compute_ria_book_analytics(clients)
        for key in ("total_aum", "avg_axiom_score", "clients_needing_review", "risk_distribution"):
            assert key in result

    def test_book_aum_sums_correctly(self):
        clients = [
            _make_client("c1", value=5_000_000),
            _make_client("c2", value=3_000_000),
        ]
        result = compute_ria_book_analytics(clients)
        assert result["total_aum"] == 8_000_000
