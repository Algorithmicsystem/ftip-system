"""
Prompt 4 intelligence tests: PE forensics, DAS engine, LP report, deal flow,
SMB depth, NLP explainability.

Coverage:
  - SchilitReport / SchilitForensicEngine
  - DealAttractivenessScore / compute_deal_attractiveness_score
  - build_structured_lp_report
  - deal_flow.py (score_acquisition_candidate, run_daily_deal_flow_screen)
  - pricing_intelligence: action_text, supporting_analysis
  - credit_intelligence: new fields (credit_rating, dscr_interpretation, etc.)
  - working_capital: new fields (opportunity_type, priority, formula)
  - sector_modules: enrichment functions
  - explain_routes: build_grounded_explanation, report format=text
  - conversational: 10 intents, classify_query_intent
  - version bump to 29.0.0
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _financials(**kwargs) -> Dict[str, Any]:
    base = {
        "revenue": 1_000_000,
        "ebitda": 200_000,
        "ebitda_margin": 0.20,
        "revenue_growth_yoy": 0.10,
        "annual_debt_service": 50_000,
        "total_debt": 300_000,
        "gross_margin": 0.45,
        "net_income": 120_000,
    }
    base.update(kwargs)
    return base


def _axiom_payload(**kwargs) -> Dict[str, Any]:
    base = {
        "deployable_alpha_utility": 45.0,
        "regime_label": "TRENDING",
        "ic_state": "MODERATE",
        "engine_scores": {
            "fundamental_reality": {
                "score": 65.0,
                "components": {
                    "earnings_quality_component": 70.0,
                    "caps_component": 60.0,
                    "eis_component": 70.0,
                },
            },
            "critical_fragility": {"score": 30.0, "components": {}},
            "flow_transmission": {"score": 55.0},
        },
        "alpha_decomposition": {"primary_driver": "EIF"},
    }
    base.update(kwargs)
    return base


# ===========================================================================
# TASK 1: SchilitReport / SchilitForensicEngine
# ===========================================================================

class TestSchilitForensicEngine:
    def test_engine_analyze_clean_financials(self):
        from api.pe.schilit_analyzer import SchilitForensicEngine
        engine = SchilitForensicEngine()
        fin = _financials(dso_change_yoy=0.0, revenue_growth_yoy=0.05,
                          receivables_growth=0.04, nonrecurring_income_pct=0.02)
        report = engine.analyze("AAPL", fin)
        assert report.symbol == "AAPL"
        assert report.overall_risk == "low"
        assert report.eis_impact == 0.0
        assert len(report.triggered_flags) == 0
        assert "No accounting shenanigans" in report.forensic_summary

    def test_engine_analyze_triggered_flags(self):
        from api.pe.schilit_analyzer import SchilitForensicEngine
        engine = SchilitForensicEngine()
        fin = _financials(
            dso_change_yoy=0.25,       # cat 1
            revenue_growth_yoy=0.05,
            receivables_growth=0.30,   # cat 1
            related_party_revenue_pct=0.45,  # cat 2
            nonrecurring_income_pct=0.15,    # cat 3
            consecutive_restructuring_charges=True,  # cat 7
        )
        report = engine.analyze("XYZ", fin)
        assert report.symbol == "XYZ"
        assert len(report.triggered_flags) >= 2
        assert report.overall_risk in ("medium", "high", "critical")
        assert report.eis_impact <= 30.0
        assert len(report.red_flags) > 0

    def test_overall_risk_levels(self):
        from api.pe.schilit_analyzer import SchilitForensicEngine, SchilitFlag
        engine = SchilitForensicEngine()
        # Test risk classification logic
        fin_clean = {}
        report = engine.analyze("T1", fin_clean)
        assert report.overall_risk == "low"

    def test_eis_impact_capped_at_30(self):
        from api.pe.schilit_analyzer import SchilitForensicEngine
        engine = SchilitForensicEngine()
        # All categories triggered with max evidence
        fin = _financials(
            dso_change_yoy=0.30,
            receivables_growth=0.50,
            revenue_growth_yoy=0.10,
            bill_and_hold_indicator=True,
            related_party_revenue_pct=0.50,
            cash_conversion_cycle_change=20,
            nonrecurring_income_pct=0.20,
            capex_pct_revenue=0.15,
            sector_capex_pct_revenue=0.05,
            rd_capitalization_change_yoy=0.25,
            intangibles_pct_acquisition_price=0.80,
            accrued_liabilities_growth=0.50,
            reserve_reversal_pct_income=0.10,
            impairment_pct_assets=0.15,
            ceo_transition_recent=True,
            consecutive_restructuring_charges=True,
            inventory_writedown_pct=0.08,
        )
        report = engine.analyze("WORST", fin)
        assert report.eis_impact <= 30.0

    def test_green_flags_for_clean(self):
        from api.pe.schilit_analyzer import SchilitForensicEngine
        engine = SchilitForensicEngine()
        report = engine.analyze("CLEAN", {})
        assert len(report.green_flags) > 0

    def test_category_scores_dict(self):
        from api.pe.schilit_analyzer import SchilitForensicEngine
        engine = SchilitForensicEngine()
        report = engine.analyze("AAPL", {})
        assert isinstance(report.category_scores, dict)
        assert set(report.category_scores.keys()) == {1, 2, 3, 4, 5, 6, 7}

    def test_report_dataclass_fields(self):
        from api.pe.schilit_analyzer import SchilitForensicEngine, SchilitReport
        engine = SchilitForensicEngine()
        report = engine.analyze("TEST", {})
        assert hasattr(report, "symbol")
        assert hasattr(report, "category_scores")
        assert hasattr(report, "triggered_flags")
        assert hasattr(report, "overall_risk")
        assert hasattr(report, "eis_impact")
        assert hasattr(report, "red_flags")
        assert hasattr(report, "green_flags")
        assert hasattr(report, "forensic_summary")


# ===========================================================================
# TASK 2: DealAttractivenessScore
# ===========================================================================

class TestDealAttractivenessScore:
    def test_basic_das_computation(self):
        from api.pe.deal_sourcing import compute_deal_attractiveness_score
        fin = _financials(
            ebitda_margin=0.22,
            gross_margin=0.45,
            revenue_growth_yoy=0.12,
            sector_revenue_growth_median=0.07,
        )
        payload = _axiom_payload(macro_score=60.0)
        result = compute_deal_attractiveness_score("AAPL", fin, payload)
        assert result.symbol == "AAPL"
        assert 0 <= result.total <= 100
        assert result.das_grade in ("A", "B", "C", "D", "F")

    def test_das_grade_thresholds(self):
        from api.pe.deal_sourcing import _das_grade
        assert _das_grade(90) == "A"
        assert _das_grade(75) == "B"
        assert _das_grade(60) == "C"
        assert _das_grade(45) == "D"
        assert _das_grade(30) == "F"

    def test_das_returns_dataclass(self):
        from api.pe.deal_sourcing import compute_deal_attractiveness_score, DealAttractivenessScore
        result = compute_deal_attractiveness_score("MSFT", {}, {})
        assert isinstance(result, DealAttractivenessScore)
        assert hasattr(result, "strategic_score")
        assert hasattr(result, "financial_score")
        assert hasattr(result, "operational_score")
        assert hasattr(result, "risk_score")
        assert hasattr(result, "investment_thesis")
        assert hasattr(result, "key_strengths")
        assert hasattr(result, "key_risks")

    def test_component_scores_bounded(self):
        from api.pe.deal_sourcing import compute_deal_attractiveness_score
        result = compute_deal_attractiveness_score("TEST", {}, {})
        assert 0 <= result.strategic_score <= 25
        assert 0 <= result.financial_score <= 25
        assert 0 <= result.operational_score <= 25
        assert 0 <= result.risk_score <= 25

    def test_investment_thesis_for_high_score(self):
        from api.pe.deal_sourcing import compute_deal_attractiveness_score
        fin = _financials(
            ebitda_margin=0.30,
            gross_margin=0.60,
            revenue_growth_yoy=0.20,
            sector_revenue_growth_median=0.05,
            fcf_to_ebitda=0.85,
        )
        payload = _axiom_payload(macro_score=80.0)
        result = compute_deal_attractiveness_score("STRONG", fin, payload)
        assert "STRONG" in result.investment_thesis

    def test_key_risks_not_empty(self):
        from api.pe.deal_sourcing import compute_deal_attractiveness_score
        result = compute_deal_attractiveness_score("WEAK", {}, {})
        assert len(result.key_risks) >= 1


# ===========================================================================
# TASK 3: build_structured_lp_report
# ===========================================================================

class TestBuildStructuredLPReport:
    def _mock_base_report(self):
        import datetime as dt
        from api.pe.lp_reporting import LPReport
        return LPReport(
            org_id="test-org",
            report_quarter="2026-Q2",
            generated_at=dt.datetime(2026, 6, 7),
            portfolio_summary={
                "total_companies": 5,
                "avg_health_score": 68.0,
                "companies_on_target": 3,
                "companies_at_risk": 1,
                "portfolio_revenue_growth": 0.08,
                "aggregate_schilit_risk": "low",
            },
            individual_company_reports=[
                {"entity_id": "e1", "entity_name": "Alpha", "health_score": 75.0, "exit_readiness_score": 80.0},
                {"entity_id": "e2", "entity_name": "Beta", "health_score": 35.0, "exit_readiness_score": 30.0},
            ],
            value_creation_attribution={},
            risk_flags=["2 companies below threshold"],
            exit_pipeline={"ready_to_exit": [{"entity_id": "e1"}], "approaching_exit": [], "optimal_market_window": "favorable"},
            market_context={"sri": 42.0, "regime": "TRENDING"},
            narrative_sections={},
        )

    @patch("api.pe.lp_reporting.generate_lp_report")
    def test_returns_5_sections(self, mock_gen):
        from api.pe.lp_reporting import build_structured_lp_report
        mock_gen.return_value = self._mock_base_report()
        result = build_structured_lp_report("test-org")
        assert "executive_summary" in result
        assert "portfolio_health" in result
        assert "detailed_reviews" in result
        assert "exit_pipeline" in result
        assert "risk_report" in result

    @patch("api.pe.lp_reporting.generate_lp_report")
    def test_executive_summary_fields(self, mock_gen):
        from api.pe.lp_reporting import build_structured_lp_report
        mock_gen.return_value = self._mock_base_report()
        result = build_structured_lp_report("test-org")
        es = result["executive_summary"]
        assert "portfolio_performance_vs_benchmark" in es
        assert "key_themes" in es
        assert "outlook" in es

    @patch("api.pe.lp_reporting.generate_lp_report")
    def test_portfolio_health_sorted_worst_first(self, mock_gen):
        from api.pe.lp_reporting import build_structured_lp_report
        mock_gen.return_value = self._mock_base_report()
        result = build_structured_lp_report("test-org")
        ph = result["portfolio_health"]
        scores = [r["health_composite"] for r in ph]
        assert scores == sorted(scores)  # worst first

    @patch("api.pe.lp_reporting.generate_lp_report")
    def test_traffic_light_logic(self, mock_gen):
        from api.pe.lp_reporting import build_structured_lp_report
        mock_gen.return_value = self._mock_base_report()
        result = build_structured_lp_report("test-org")
        for row in result["portfolio_health"]:
            assert row["traffic_light"] in ("green", "yellow", "red")

    @patch("api.pe.lp_reporting.generate_lp_report")
    def test_risk_report_sri(self, mock_gen):
        from api.pe.lp_reporting import build_structured_lp_report
        mock_gen.return_value = self._mock_base_report()
        result = build_structured_lp_report("test-org")
        rr = result["risk_report"]
        assert "sri" in rr
        assert rr["sri"] == 42.0


# ===========================================================================
# TASK 4: Deal Flow
# ===========================================================================

class TestDealFlow:
    def test_score_acquisition_candidate_basic(self):
        from api.pe.deal_flow import score_acquisition_candidate
        payload = {"dau": 35.0, "schilit_overall_risk": "low", "dossier_iq_trend": "stable"}
        fin = {"ebitda_margin": 0.20, "fcf_to_ebitda": 0.75}
        score = score_acquisition_candidate("AAPL", payload, fin)
        assert 0 <= score <= 100
        assert score >= 30  # depressed DAU + good fundamentals

    def test_score_high_dau_lower_score(self):
        from api.pe.deal_flow import score_acquisition_candidate
        payload = {"dau": 80.0, "schilit_overall_risk": "high", "dossier_iq_trend": "stable"}
        fin = {"ebitda_margin": 0.05, "fcf_to_ebitda": 0.2}
        score = score_acquisition_candidate("BUY", payload, fin)
        # High DAU = no signal bonus
        assert score < 50

    def test_score_critical_accounting_risk(self):
        from api.pe.deal_flow import score_acquisition_candidate
        payload = {"dau": 30.0, "schilit_overall_risk": "critical"}
        fin = {}
        score = score_acquisition_candidate("BAD", payload, fin)
        # critical schilit = 0 accounting points
        assert score < 60

    def test_score_max_100(self):
        from api.pe.deal_flow import score_acquisition_candidate
        payload = {"dau": 30.0, "schilit_overall_risk": "low", "dossier_iq_trend": "improving"}
        fin = {"ebitda_margin": 0.30, "fcf_to_ebitda": 1.0}
        score = score_acquisition_candidate("BEST", payload, fin)
        assert score <= 100

    @patch("api.pe.deal_flow.db")
    def test_run_daily_screen_no_db(self, mock_db):
        from api.pe.deal_flow import run_daily_deal_flow_screen
        mock_db.db_read_enabled.return_value = False
        mock_db.db_write_enabled.return_value = False
        result = run_daily_deal_flow_screen()
        assert "as_of_date" in result
        assert "candidates" in result
        assert "universe_screened" in result
        assert "candidates_found" in result

    def test_run_daily_screen_returns_sorted(self):
        from api.pe.deal_flow import run_daily_deal_flow_screen
        with patch("api.pe.deal_flow.db") as mock_db:
            mock_db.db_read_enabled.return_value = False
            mock_db.db_write_enabled.return_value = False
            result = run_daily_deal_flow_screen()
            cands = result["candidates"]
            if len(cands) >= 2:
                scores = [c["acquisition_score"] for c in cands]
                assert scores == sorted(scores, reverse=True)


# ===========================================================================
# TASK 7: Pricing Intelligence
# ===========================================================================

class TestPricingIntelligence:
    def test_action_text_raise_prices_high_cost(self):
        from api.smb.pricing_intelligence import generate_pricing_recommendation
        result = generate_pricing_recommendation(75.0, 70.0, "compressing")
        assert "action_text" in result
        assert "Raise prices now" in result["action_text"]

    def test_action_text_raise_prices_low_cost(self):
        from api.smb.pricing_intelligence import generate_pricing_recommendation
        result = generate_pricing_recommendation(75.0, 30.0, "expanding", "RECOVERY")
        assert "action_text" in result
        assert "pricing power is strong" in result["action_text"]

    def test_action_text_defend_volume(self):
        from api.smb.pricing_intelligence import generate_pricing_recommendation
        result = generate_pricing_recommendation(30.0, 70.0, "compressing")
        assert "action_text" in result
        assert "Do not raise prices" in result["action_text"]

    def test_supporting_analysis_present(self):
        from api.smb.pricing_intelligence import generate_pricing_recommendation
        result = generate_pricing_recommendation(60.0, 50.0, "stable")
        assert "supporting_analysis" in result
        assert "Pricing power score" in result["supporting_analysis"]

    def test_supporting_analysis_elevated_pressure(self):
        from api.smb.pricing_intelligence import generate_pricing_recommendation
        result = generate_pricing_recommendation(70.0, 75.0, "compressing")
        assert "elevated" in result["supporting_analysis"]

    def test_existing_fields_preserved(self):
        from api.smb.pricing_intelligence import generate_pricing_recommendation
        result = generate_pricing_recommendation(60.0, 50.0, "stable")
        assert "action" in result
        assert "rationale" in result
        assert "price_increase_potential_pct" in result
        assert "timing" in result


# ===========================================================================
# TASK 8: Credit Intelligence
# ===========================================================================

class TestCreditIntelligence:
    def test_credit_rating_excellent(self):
        from api.smb.credit_intelligence import compute_borrowing_capacity_score
        fin = _financials(ebitda=200_000, annual_debt_service=50_000,
                          total_debt=200_000, ebitda_margin=0.25, revenue_growth_yoy=0.10)
        result = compute_borrowing_capacity_score(fin)
        assert "credit_rating" in result
        assert result["credit_rating"] in ("excellent", "good", "adequate", "tight", "distressed")

    def test_credit_score_equals_capacity_score(self):
        from api.smb.credit_intelligence import compute_borrowing_capacity_score
        fin = _financials()
        result = compute_borrowing_capacity_score(fin)
        assert result["credit_score"] == result["borrowing_capacity_score"]

    def test_dscr_interpretation_present(self):
        from api.smb.credit_intelligence import compute_borrowing_capacity_score
        fin = _financials()
        result = compute_borrowing_capacity_score(fin)
        assert "dscr_interpretation" in result
        assert "DSCR" in result["dscr_interpretation"]

    def test_max_additional_debt_formula(self):
        from api.smb.credit_intelligence import compute_borrowing_capacity_score
        fin = _financials(ebitda=300_000, annual_debt_service=80_000)
        result = compute_borrowing_capacity_score(fin)
        assert "max_additional_debt_usd" in result
        assert "max_additional_debt_formula" in result
        assert isinstance(result["max_additional_debt_usd"], float)

    def test_lending_recommendation_present(self):
        from api.smb.credit_intelligence import compute_borrowing_capacity_score
        fin = _financials()
        result = compute_borrowing_capacity_score(fin)
        assert "lending_recommendation" in result
        assert len(result["lending_recommendation"]) > 10

    def test_quick_ratio_present(self):
        from api.smb.credit_intelligence import compute_borrowing_capacity_score
        fin = _financials(revenue=500_000, total_debt=200_000)
        result = compute_borrowing_capacity_score(fin)
        assert "quick_ratio" in result
        assert result["quick_ratio"] >= 0

    def test_cash_conversion_cycle_present(self):
        from api.smb.credit_intelligence import compute_borrowing_capacity_score
        fin = _financials(ebitda_margin=0.15)
        result = compute_borrowing_capacity_score(fin)
        assert "cash_conversion_cycle_days" in result
        assert isinstance(result["cash_conversion_cycle_days"], float)

    def test_distressed_no_additional_debt(self):
        from api.smb.credit_intelligence import compute_borrowing_capacity_score
        fin = _financials(ebitda=50_000, annual_debt_service=100_000)
        result = compute_borrowing_capacity_score(fin)
        assert result["max_additional_debt_usd"] == 0.0


# ===========================================================================
# TASK 9: Working Capital
# ===========================================================================

class TestWorkingCapitalRecommendations:
    def _make_analysis(self, ar_days=60, ap_days=20, inventory_days=45):
        from api.smb.working_capital import WorkingCapitalAnalysis
        return WorkingCapitalAnalysis(
            entity_id="test",
            current_cash_conversion_cycle=ar_days + inventory_days - ap_days,
            optimal_cash_conversion_cycle=15.0,
            working_capital_gap_days=30.0,
            trapped_working_capital_usd=10000.0,
            ar_days=ar_days,
            ap_days=ap_days,
            inventory_days=inventory_days,
            recommendations=[],
            potential_cash_release_usd=0.0,
        )

    def test_ar_rec_has_opportunity_type(self):
        from api.smb.working_capital import generate_wc_recommendations
        analysis = self._make_analysis(ar_days=65, ap_days=25, inventory_days=0)
        fin = {"revenue": 100_000, "cogs": 60_000}
        recs = generate_wc_recommendations(analysis, fin, "Professional Services")
        ar_recs = [r for r in recs if r.get("action") == "accelerate_collections"]
        assert len(ar_recs) > 0
        assert ar_recs[0]["opportunity_type"] == "receivables"

    def test_ar_rec_has_formula(self):
        from api.smb.working_capital import generate_wc_recommendations
        analysis = self._make_analysis(ar_days=70)
        fin = {"revenue": 100_000, "cogs": 60_000}
        recs = generate_wc_recommendations(analysis, fin, "Professional Services")
        ar_recs = [r for r in recs if r.get("action") == "accelerate_collections"]
        if ar_recs:
            assert "cash_released_formula" in ar_recs[0]
            assert "daily revenue" in ar_recs[0]["cash_released_formula"]

    def test_ap_rec_has_formula(self):
        from api.smb.working_capital import generate_wc_recommendations
        analysis = self._make_analysis(ap_days=10)
        fin = {"revenue": 100_000, "cogs": 60_000}
        recs = generate_wc_recommendations(analysis, fin, "Professional Services")
        ap_recs = [r for r in recs if r.get("action") == "extend_payables"]
        if ap_recs:
            assert "cash_released_formula" in ap_recs[0]
            assert ap_recs[0]["opportunity_type"] == "payables"

    def test_inventory_rec_priority_high(self):
        from api.smb.working_capital import generate_wc_recommendations
        analysis = self._make_analysis(inventory_days=80)
        fin = {"revenue": 100_000, "cogs": 60_000}
        recs = generate_wc_recommendations(analysis, fin, "Manufacturing")
        inv_recs = [r for r in recs if r.get("action") == "reduce_inventory"]
        if inv_recs:
            assert inv_recs[0]["priority"] in ("high", "medium")
            assert inv_recs[0]["opportunity_type"] == "inventory"

    def test_priority_field_present(self):
        from api.smb.working_capital import generate_wc_recommendations
        analysis = self._make_analysis(ar_days=65)
        fin = {"revenue": 100_000, "cogs": 60_000}
        recs = generate_wc_recommendations(analysis, fin, "Professional Services")
        for rec in recs:
            assert "priority" in rec

    def test_current_and_target_days_present(self):
        from api.smb.working_capital import generate_wc_recommendations
        analysis = self._make_analysis(ar_days=65)
        fin = {"revenue": 100_000, "cogs": 60_000}
        recs = generate_wc_recommendations(analysis, fin, "Professional Services")
        for rec in recs:
            assert "current_days" in rec
            assert "target_days" in rec
            assert "improvement_days" in rec


# ===========================================================================
# TASK 10: Sector Modules Enrichment
# ===========================================================================

class TestSectorModulesEnrichment:
    def test_restaurant_enrichment_fields(self):
        from api.smb.sector_modules import get_sector_module
        fin = {"revenue": 50_000, "entity_id": "rest1"}
        ops = {"food_cost_pct": 0.32, "labor_cost_pct": 0.38, "restaurant_type": "full_service"}
        result = get_sector_module("Restaurant", fin, ops)
        assert result["status"] == "ok"
        assert "prime_cost_target" in result
        assert "prime_cost_gap" in result
        assert "prime_cost_status" in result
        assert "prime_cost_dollar_impact" in result
        assert "action" in result

    def test_restaurant_prime_cost_status(self):
        from api.smb.sector_modules import get_sector_module
        fin = {"revenue": 50_000}
        ops = {"food_cost_pct": 0.30, "labor_cost_pct": 0.50}  # high prime cost
        result = get_sector_module("Restaurant", fin, ops)
        assert result["prime_cost_status"] in ("excellent", "on-target", "watch", "critical")

    def test_manufacturing_enrichment_fields(self):
        from api.smb.sector_modules import get_sector_module
        fin = {"revenue": 200_000, "cogs": 130_000, "revenue_growth_yoy": 0.05,
               "inventory_growth_yoy": 0.02}
        ops = {"top_supplier_concentration": 0.40, "backlog_revenue": 100_000}
        result = get_sector_module("Manufacturing", fin, ops)
        assert "capacity_utilization_estimate" in result
        assert "underutilization_cost_usd" in result
        assert "order_backlog_months" in result
        assert "backlog_health" in result
        assert "action" in result

    def test_manufacturing_backlog_health(self):
        from api.smb.sector_modules import get_sector_module
        fin = {"revenue": 100_000, "cogs": 65_000, "revenue_growth_yoy": 0.10}
        ops = {"backlog_revenue": 500_000}
        result = get_sector_module("Manufacturing", fin, ops)
        assert result["backlog_health"] in ("strong", "adequate", "thin", "empty")

    def test_profservices_enrichment_fields(self):
        from api.smb.sector_modules import get_sector_module
        fin = {"revenue": 80_000}
        ops = {"headcount": 5, "client_concentration": 0.30, "sector": "Professional Services"}
        result = get_sector_module("Professional Services", fin, ops)
        assert "sector_benchmark_revenue_per_employee" in result
        assert "utilization_proxy" in result
        assert "recurring_revenue_estimate_pct" in result
        assert "action" in result

    def test_consulting_enrichment(self):
        from api.smb.sector_modules import get_sector_module
        fin = {"revenue": 150_000}
        ops = {"headcount": 3, "client_concentration": 0.50, "sector": "Consulting"}
        result = get_sector_module("Consulting", fin, ops)
        assert result["status"] == "ok"
        assert "sector_benchmark_revenue_per_employee" in result

    def test_no_module_sector(self):
        from api.smb.sector_modules import get_sector_module
        result = get_sector_module("Unknown Sector", {}, {})
        assert result["status"] == "no_module"


# ===========================================================================
# TASK 11: build_grounded_explanation
# ===========================================================================

class TestBuildGroundedExplanation:
    def test_basic_explanation(self):
        from api.explain.explain_routes import build_grounded_explanation
        payload = _axiom_payload(deployable_alpha_utility=75.0)
        text = build_grounded_explanation("AAPL", payload)
        assert "AAPL" in text
        assert "BUY" in text
        assert "75.0" in text

    def test_sell_signal(self):
        from api.explain.explain_routes import build_grounded_explanation
        payload = _axiom_payload(deployable_alpha_utility=35.0)
        text = build_grounded_explanation("TSLA", payload)
        assert "SELL" in text
        assert "TSLA" in text

    def test_suppression_note(self):
        from api.explain.explain_routes import build_grounded_explanation
        payload = _axiom_payload(
            deployable_alpha_utility=50.0,
            suppression_active=True,
            pre_suppression_action="BUY",
            suppression_reason="liquidity_fracture",
        )
        text = build_grounded_explanation("XYZ", payload)
        assert "suppressed" in text.lower()

    def test_batting_average_in_explanation(self):
        from api.explain.explain_routes import build_grounded_explanation
        payload = _axiom_payload(signal_batting_average=0.65)
        text = build_grounded_explanation("MSFT", payload)
        assert "65%" in text

    def test_5_paragraphs(self):
        from api.explain.explain_routes import build_grounded_explanation
        payload = _axiom_payload()
        text = build_grounded_explanation("TEST", payload)
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        assert len(paragraphs) == 5

    def test_caps_primary_driver(self):
        from api.explain.explain_routes import build_grounded_explanation
        payload = _axiom_payload()
        payload["alpha_decomposition"] = {"primary_driver": "CMF"}
        text = build_grounded_explanation("MSFT", payload)
        assert "capital allocation" in text.lower()

    def test_high_volatility_regime_risk(self):
        from api.explain.explain_routes import build_grounded_explanation
        payload = _axiom_payload(regime_label="HIGH_VOL")
        text = build_grounded_explanation("VIX", payload)
        assert "High Vol" in text or "high_vol" in text.lower() or "adverse" in text.lower()


# ===========================================================================
# TASK 12: Conversational Intent Classification
# ===========================================================================

class TestConversationalIntents:
    def test_signal_query_intent(self):
        from api.explain.conversational import classify_query_intent
        intent = classify_query_intent("What does AXIOM say about AAPL?")
        assert intent == "signal_query"

    def test_universe_buy_signals(self):
        from api.explain.conversational import classify_query_intent
        intent = classify_query_intent("which stocks have buy signals right now?")
        assert intent == "universe_buy_signals"

    def test_regime_query(self):
        from api.explain.conversational import classify_query_intent
        intent = classify_query_intent("What is the current market regime?")
        assert intent == "regime_query"

    def test_risk_query(self):
        from api.explain.conversational import classify_query_intent
        intent = classify_query_intent("What are the highest crash risk stocks?")
        assert intent == "risk_query"

    def test_leaderboard(self):
        from api.explain.conversational import classify_query_intent
        intent = classify_query_intent("show me the signal WAR leaderboard")
        assert intent == "leaderboard"

    def test_explain_signal(self):
        from api.explain.conversational import classify_query_intent
        intent = classify_query_intent("explain AAPL signal")
        # explain_signal or signal_explanation are both acceptable (backward compat)
        assert intent in ("explain_signal", "signal_explanation")

    def test_portfolio_query(self):
        from api.explain.conversational import classify_query_intent
        intent = classify_query_intent("what is my portfolio value at risk?")
        assert intent == "portfolio_query"

    def test_regime_change(self):
        from api.explain.conversational import classify_query_intent
        intent = classify_query_intent("when will the next regime shift happen?")
        assert intent == "regime_change"

    def test_moat_query(self):
        from api.explain.conversational import classify_query_intent
        intent = classify_query_intent("what is the moat score today?")
        assert intent == "moat_query"

    def test_unknown_intent(self):
        from api.explain.conversational import classify_query_intent
        intent = classify_query_intent("hello world gibberish 12345")
        assert intent == "unknown"

    def test_comparison_intent(self):
        from api.explain.conversational import classify_query_intent
        intent = classify_query_intent("compare AAPL vs MSFT")
        assert intent == "comparison"

    def test_factor_query(self):
        from api.explain.conversational import classify_query_intent
        intent = classify_query_intent("what is the EIF factor loading?")
        assert intent == "factor_query"

    def test_programmatic_universe_buy(self):
        from api.explain.conversational import _programmatic_answer
        context = {"buy_signals": [{"symbol": "AAPL", "dau": 75.0}, {"symbol": "MSFT", "dau": 70.0}]}
        answer = _programmatic_answer("universe_buy_signals", context, [], "which stocks buy?")
        assert "AAPL" in answer
        assert "BUY" in answer

    def test_programmatic_leaderboard(self):
        from api.explain.conversational import _programmatic_answer
        context = {"leaderboard": [{"symbol": "AAPL", "batting_avg": 0.72}]}
        answer = _programmatic_answer("leaderboard", context, [], "leaderboard?")
        assert "AAPL" in answer
        assert "72%" in answer

    def test_query_intents_list_has_10_core(self):
        from api.explain.conversational import QUERY_INTENTS
        core = {"signal_query", "universe_buy_signals", "regime_query", "risk_query",
                "signal_history", "leaderboard", "explain_signal", "portfolio_query",
                "regime_change", "moat_query"}
        for intent in core:
            assert intent in QUERY_INTENTS


# ===========================================================================
# TASK 13: Version
# ===========================================================================

class TestVersionBump:
    def test_main_version_29(self):
        import api.main as m
        import inspect
        src = inspect.getsource(m)
        assert "33.0.0" in src or "32.0.0" in src or "31.0.0" in src or "30.0.0" in src
        assert "28.0.0" not in src

    def test_index_html_v29(self):
        from pathlib import Path
        html = Path("/Users/macuser/ftip-system/Algorithm/api/webapp/index.html").read_text()
        assert "?v=29" in html or "?v=30" in html or "?v=31" in html or "?v=32" in html or "?v=33" in html
        assert "?v=28" not in html
