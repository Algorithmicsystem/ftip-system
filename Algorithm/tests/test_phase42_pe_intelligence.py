"""Phase 42: Deep PE Intelligence tests — Schilit, deal sourcing, comps, supply chain, LP reporting."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# TestSchilitAnalyzer
# ---------------------------------------------------------------------------

class TestSchilitAnalyzer:

    def _clean_financials(self) -> Dict:
        """Financials with no shenanigan signals."""
        return {
            "revenue_growth_yoy": 0.08,
            "dso_change_yoy": 0.02,
            "receivables_growth": 0.07,
            "bill_and_hold_indicator": False,
            "sector_revenue_growth": 0.07,
            "cash_conversion_cycle_change": -2.0,
            "nonrecurring_income_pct": 0.04,
            "net_income": 100.0,
            "reported_net_income": 100.0,
            "core_earnings": 96.0,
        }

    def test_category_1_triggers_on_dso_growth(self):
        from api.pe.schilit_analyzer import flag_category_1
        fin = {
            "dso_change_yoy": 0.20,
            "receivables_growth": 0.35,
            "revenue_growth_yoy": 0.10,
        }
        result = flag_category_1(fin)
        assert result.triggered is True
        assert result.category == 1

    def test_category_2_triggers_on_revenue_anomaly(self):
        from api.pe.schilit_analyzer import flag_category_2
        fin = {
            "revenue_growth_yoy": 0.60,
            "sector_revenue_growth": 0.10,  # 6× sector median → > 3×
        }
        result = flag_category_2(fin)
        assert result.triggered is True
        assert result.category == 2

    def test_category_3_triggers_on_nonrecurring(self):
        from api.pe.schilit_analyzer import flag_category_3
        fin = {
            "nonrecurring_income_pct": 0.15,
            "net_income": 100.0,
        }
        result = flag_category_3(fin)
        assert result.triggered is True
        assert result.category == 3

    def test_category_3_triggers_on_core_vs_reported_gap(self):
        from api.pe.schilit_analyzer import flag_category_3
        fin = {
            "core_earnings": 75.0,
            "reported_net_income": 100.0,
        }
        result = flag_category_3(fin)
        assert result.triggered is True
        assert any("0.75" in e for e in result.evidence)

    def test_all_clean_returns_high_score(self):
        from api.pe.schilit_analyzer import run_full_schilit_analysis
        result = run_full_schilit_analysis(self._clean_financials())
        assert result["schilit_score"] > 80
        assert result["recommendation"] == "clean"

    def test_multiple_flags_reduce_score(self):
        from api.pe.schilit_analyzer import run_full_schilit_analysis
        # Triggers cat 1 (3 signals=15), cat 2 (3 signals=18), cat 3 (2 signals=8),
        # cat 4 (2 signals=9), cat 7 (2 signals=5) → total impact=55 → score=45
        fin = {
            # Cat 1: DSO spike + bill-and-hold + channel stuffing
            "dso_change_yoy": 0.25,
            "bill_and_hold_indicator": True,
            "receivables_growth": 0.85,        # 0.85 - 0.60 = 0.25 > 0.20
            "revenue_growth_yoy": 0.60,
            # Cat 2: revenue anomaly + CCC + related party
            "sector_revenue_growth": 0.10,     # 0.60 > 3×0.10
            "cash_conversion_cycle_change": 15.0,
            "related_party_revenue_pct": 0.35,
            # Cat 3: non-recurring + core vs reported gap
            "nonrecurring_income_pct": 0.20,
            "core_earnings": 75.0,
            "net_income": 100.0,
            "reported_net_income": 100.0,
            # Cat 4: R&D cap + depreciation life extension
            "rd_capitalization_change_yoy": 0.15,
            "depreciation_life_change_yoy": 2.0,
            # Cat 7: big bath + consecutive restructuring
            "impairment_pct_assets": 0.15,
            "ceo_transition_recent": True,
            "consecutive_restructuring_charges": True,
        }
        result = run_full_schilit_analysis(fin)
        assert result["schilit_score"] < 50

    def test_audit_risk_flag_on_3_categories(self):
        from api.pe.schilit_analyzer import run_full_schilit_analysis
        # Trigger cat 1, 2, 3 simultaneously
        fin = {
            "dso_change_yoy": 0.30,          # cat 1
            "receivables_growth": 0.50,
            "revenue_growth_yoy": 0.80,       # cat 2
            "sector_revenue_growth": 0.08,
            "nonrecurring_income_pct": 0.25,  # cat 3
            "net_income": 100.0,
        }
        result = run_full_schilit_analysis(fin)
        assert result["audit_risk_flag"] is True
        assert result["triggered_flags"] >= 3

    def test_recommendation_clean(self):
        from api.pe.schilit_analyzer import run_full_schilit_analysis
        result = run_full_schilit_analysis(self._clean_financials())
        assert result["recommendation"] == "clean"

    def test_recommendation_investigate(self):
        from api.pe.schilit_analyzer import run_full_schilit_analysis
        # Cat1(3→15) + Cat2(3→18) + Cat3(2→8) + Cat7(1→2.5) = 43.5 → score=56.5 → investigate
        fin = {
            # Cat 1: DSO + bill-and-hold + channel stuffing (0.55-0.30=0.25>0.20)
            "dso_change_yoy": 0.30,
            "bill_and_hold_indicator": True,
            "receivables_growth": 0.55,
            "revenue_growth_yoy": 0.30,   # lower so channel-stuffing diff is >0.20
            # Cat 2: revenue anomaly + CCC + related party
            "sector_revenue_growth": 0.05,   # 0.30 > 3×0.05=0.15
            "cash_conversion_cycle_change": 30.0,
            "related_party_revenue_pct": 0.40,
            # Cat 3: non-recurring + core/reported gap
            "nonrecurring_income_pct": 0.30,
            "core_earnings": 50.0,
            "reported_net_income": 100.0,
            "net_income": 100.0,
            # Cat 7: big bath (impairment + CEO transition)
            "impairment_pct_assets": 0.20,
            "ceo_transition_recent": True,
        }
        result = run_full_schilit_analysis(fin)
        assert result["recommendation"] in ("investigate", "avoid")

    def test_management_integrity_degrades(self):
        from api.pe.schilit_analyzer import run_full_schilit_analysis
        clean_result = run_full_schilit_analysis(self._clean_financials())
        flagged_fin = {
            "dso_change_yoy": 0.25,
            "receivables_growth": 0.40,
            "revenue_growth_yoy": 0.60,
            "sector_revenue_growth": 0.08,
        }
        flagged_result = run_full_schilit_analysis(flagged_fin)
        assert flagged_result["management_integrity_score"] < clean_result["management_integrity_score"]

    def test_eis_impact_applied(self):
        from api.pe.schilit_analyzer import run_full_schilit_analysis
        fin = {
            "dso_change_yoy": 0.25,
            "receivables_growth": 0.40,
            "revenue_growth_yoy": 0.05,
            "nonrecurring_income_pct": 0.18,
            "net_income": 100.0,
        }
        result = run_full_schilit_analysis(fin)
        assert result["composite_eis_impact"] > 0.0

    def test_schilit_integrated_into_eis(self):
        from api.axiom.engines.fundamental import compute_eis
        # Provide Schilit-specific fields to trigger the integration path
        fin_clean = {
            "cfo": 100.0, "net_income": 80.0,
            "dso_change_yoy": 0.01,
            "revenue_growth_yoy": 0.05,
        }
        fin_shenanigan = {
            "cfo": 100.0, "net_income": 80.0,
            "dso_change_yoy": 0.30,
            "receivables_growth": 0.50,
            "revenue_growth_yoy": 0.05,
            "nonrecurring_income_pct": 0.20,
            "net_income": 80.0,
        }
        eis_clean = compute_eis(fin_clean)
        eis_shenanigan = compute_eis(fin_shenanigan)
        assert eis_clean >= eis_shenanigan


# ---------------------------------------------------------------------------
# TestDealSourcing
# ---------------------------------------------------------------------------

class TestDealSourcing:

    def _attractive_payload(self) -> Dict:
        return {
            "deployable_alpha_utility": 75.0,
            "engine_scores": {
                "fundamental_reality": {
                    "components": {"caps_component": 80.0, "earnings_quality_component": 75.0}
                },
                "critical_fragility": {
                    "components": {"pess_component": 20.0, "scps_component": 15.0}
                },
            },
            "revenue_growth_yoy": 0.18,
            "market_cap": 800_000_000.0,
        }

    def _unattractive_payload(self) -> Dict:
        return {
            "deployable_alpha_utility": 25.0,
            "engine_scores": {
                "fundamental_reality": {
                    "components": {"caps_component": 25.0, "earnings_quality_component": 30.0}
                },
                "critical_fragility": {
                    "components": {"pess_component": 75.0, "scps_component": 60.0}
                },
            },
            "revenue_growth_yoy": -0.10,
            "market_cap": 5_000_000_000.0,
        }

    def test_das_high_for_attractive_target(self):
        from api.pe.deal_sourcing import compute_das
        fin = {
            "fcf_yield": 0.09,
            "sector_median_fcf_yield": 0.04,
            "sector_std_fcf_yield": 0.02,
            "eps_ttm": 4.0,
            "current_price": 35.0,
            "wacc_estimate": 0.09,
            "growth_rate_estimate": 0.05,
            "insider_buys_6m": 5000,
            "insider_sells_6m": 1000,
            "buyback_yield": 0.03,
        }
        result = compute_das("AAPL", self._attractive_payload(), fin)
        assert result["das"] > 65

    def test_das_low_for_unattractive(self):
        from api.pe.deal_sourcing import compute_das
        fin = {
            "fcf_yield": -0.02,
            "sector_median_fcf_yield": 0.04,
            "sector_std_fcf_yield": 0.02,
            "current_price": 100.0,
            "eps_ttm": 1.0,
            "wacc_estimate": 0.09,
            "insider_buys_6m": 0,
            "insider_sells_6m": 5000,
            "buyback_yield": 0.0,
        }
        result = compute_das("XYZ", self._unattractive_payload(), fin)
        assert result["das"] < 40

    def test_das_components_sum_correctly(self):
        from api.pe.deal_sourcing import compute_das
        fin = {
            "fcf_yield": 0.05,
            "sector_median_fcf_yield": 0.04,
            "sector_std_fcf_yield": 0.02,
            "insider_buys_6m": 100,
            "insider_sells_6m": 50,
            "buyback_yield": 0.025,
        }
        result = compute_das("TEST", self._attractive_payload(), fin)
        expected = (
            result["fcf_yield_score"] * 0.30
            + result["caps_score"] * 0.25
            + result["valuation_discount_score"] * 0.25
            + result["management_quality_score"] * 0.20
        )
        assert abs(result["das"] - expected) < 0.1

    def test_strategic_theme_consolidation(self):
        from api.pe.deal_sourcing import classify_strategic_themes
        payload = self._attractive_payload()
        themes = classify_strategic_themes("AAPL", payload, "Technology")
        assert "consolidation_play" in themes

    def test_strategic_theme_turnaround(self):
        from api.pe.deal_sourcing import classify_strategic_themes
        payload = {
            "deployable_alpha_utility": 35.0,
            "dau_trend_3m": 5.0,
            "engine_scores": {
                "fundamental_reality": {
                    "components": {"caps_component": 40.0, "earnings_quality_component": 40.0}
                },
                "critical_fragility": {"components": {"pess_component": 30.0, "scps_component": 20.0}},
            },
            "revenue_growth_yoy": 0.05,
            "market_cap": 500_000_000.0,
        }
        themes = classify_strategic_themes("XYZ", payload, "Industrials")
        assert "turnaround" in themes

    def test_strategic_theme_bolt_on(self):
        from api.pe.deal_sourcing import classify_strategic_themes
        payload = {
            "deployable_alpha_utility": 72.0,
            "dau_trend_3m": 2.0,
            "engine_scores": {
                "fundamental_reality": {
                    "components": {"caps_component": 75.0, "earnings_quality_component": 75.0}
                },
                "critical_fragility": {"components": {"pess_component": 20.0, "scps_component": 15.0}},
            },
            "revenue_growth_yoy": 0.08,
            "market_cap": 900_000_000.0,  # < $2B
        }
        themes = classify_strategic_themes("SMALL", payload, "Technology")
        assert "bolt_on" in themes

    def test_screen_applies_min_das_filter(self):
        from api.pe.deal_sourcing import screen_for_deal_candidates
        with patch("api.pe.deal_sourcing.db.db_read_enabled", return_value=False):
            results = screen_for_deal_candidates(min_das=70.0)
        assert results == []  # DB disabled → empty

    def test_candidates_sorted_by_das(self):
        from api.pe.deal_sourcing import DealCandidate
        candidates = [
            DealCandidate("A", "A", "Tech", 1e9, 72.0, {}, 0.05, 70.0, 10.0, "neutral", "neutral", 60.0, None, {}, []),
            DealCandidate("B", "B", "Tech", 1e9, 88.0, {}, 0.07, 80.0, 20.0, "strong_buyer", "buying", 80.0, None, {}, []),
            DealCandidate("C", "C", "Tech", 1e9, 55.0, {}, 0.02, 50.0, 5.0, "seller", "selling", 40.0, None, {}, []),
        ]
        sorted_c = sorted(candidates, key=lambda x: x.das_score, reverse=True)
        assert sorted_c[0].symbol == "B"
        assert sorted_c[-1].symbol == "C"


# ---------------------------------------------------------------------------
# TestCompsEngine
# ---------------------------------------------------------------------------

class TestCompsEngine:

    def test_quality_premium_for_high_eis(self):
        from api.pe.comps_engine import compute_quality_adjusted_multiple
        base = 12.0
        adjusted = compute_quality_adjusted_multiple(base, eis_score=85.0, caps_score=80.0, schilit_score=100.0)
        assert adjusted > base

    def test_quality_discount_for_shenanigans(self):
        from api.pe.comps_engine import compute_quality_adjusted_multiple
        base = 12.0
        adjusted = compute_quality_adjusted_multiple(base, eis_score=50.0, caps_score=50.0, schilit_score=20.0)
        assert adjusted < base

    def test_implied_valuation_range_structure(self):
        from api.pe.comps_engine import run_comps_analysis
        result = run_comps_analysis(
            "TEST", {"ebitda": 50.0, "market_cap": 400.0},
            "Technology", 70.0, 70.0, 90.0
        )
        assert "low" in result.implied_valuation_range
        assert "median" in result.implied_valuation_range
        assert "high" in result.implied_valuation_range
        assert result.implied_valuation_range["low"] < result.implied_valuation_range["median"]
        assert result.implied_valuation_range["median"] < result.implied_valuation_range["high"]

    def test_pe_premium_applied(self):
        from api.pe.comps_engine import SECTOR_TRADING_MULTIPLES, PE_PREMIUM, compute_quality_adjusted_multiple
        sector_multiple = SECTOR_TRADING_MULTIPLES["Technology"]["ev_ebitda"]
        transaction_multiple = sector_multiple * (1.0 + PE_PREMIUM)
        quality_multiple = compute_quality_adjusted_multiple(transaction_multiple, 50.0, 50.0, 100.0)
        assert quality_multiple > sector_multiple

    def test_confidence_high_clean_company(self):
        from api.pe.comps_engine import run_comps_analysis
        result = run_comps_analysis(
            "TEST", {"ebitda": 50.0}, "Technology", 70.0, 70.0, 90.0
        )
        assert result.confidence == "high"

    def test_confidence_low_schilit(self):
        from api.pe.comps_engine import run_comps_analysis
        result = run_comps_analysis(
            "TEST", {"ebitda": 50.0}, "Technology", 70.0, 70.0, 35.0
        )
        assert result.confidence == "low"

    def test_upside_positive_when_undervalued(self):
        from api.pe.comps_engine import run_comps_analysis
        # ebitda=100, quality multiple ~16x => implied ~1600; market_cap=400 → upside > 0
        result = run_comps_analysis(
            "TEST", {"ebitda": 100.0, "market_cap": 400.0},
            "Technology", 65.0, 65.0, 90.0
        )
        assert result.upside_to_consensus > 0.0


# ---------------------------------------------------------------------------
# TestSupplyChainRisk
# ---------------------------------------------------------------------------

class TestSupplyChainRisk:

    def test_stress_index_neutral_no_data(self):
        from api.pe.supply_chain_risk import compute_supply_chain_stress_index
        with patch("api.pe.supply_chain_risk.db.db_read_enabled", return_value=False):
            result = compute_supply_chain_stress_index("org1", [])
        assert result["portfolio_supply_chain_risk"] == 50.0

    def test_supply_chain_risk_bounded(self):
        from api.pe.supply_chain_risk import compute_supply_chain_stress_index
        with patch("api.pe.supply_chain_risk.db.db_read_enabled", return_value=False):
            result = compute_supply_chain_stress_index("org1", [{"entity_id": "e1"}])
        assert 0.0 <= result["portfolio_supply_chain_risk"] <= 100.0

    def test_alert_generated_high_fragility(self):
        from api.pe.supply_chain_risk import monitor_portfolio_analogs
        mock_entity_rows = [("e1", "Company A", "Technology", "AAPL")]
        mock_axiom_row = [{
            "engine_scores": {
                "critical_fragility": {
                    "score": 75.0,
                    "components": {
                        "mtrs_component": 75.0,
                        "scps_component": 40.0,
                        "pess_component": 40.0,
                    }
                }
            },
            "regime_label": "TRENDING",
        }]
        with patch("api.pe.supply_chain_risk.db.db_read_enabled", return_value=True):
            with patch("api.pe.supply_chain_risk.db.safe_fetchall", return_value=mock_entity_rows):
                with patch("api.pe.supply_chain_risk.db.safe_fetchone", return_value=(mock_axiom_row[0],)):
                    result = monitor_portfolio_analogs("org1")
        assert any(a["alert_type"] == "supply_chain_stress" for a in result["alerts"])

    def test_no_alert_low_fragility(self):
        from api.pe.supply_chain_risk import monitor_portfolio_analogs
        mock_entity_rows = [("e1", "Company A", "Technology", "AAPL")]
        mock_axiom_payload = {
            "engine_scores": {
                "critical_fragility": {
                    "score": 30.0,
                    "components": {
                        "mtrs_component": 30.0,
                        "scps_component": 20.0,
                        "pess_component": 20.0,
                    }
                }
            },
            "regime_label": "TRENDING",
        }
        with patch("api.pe.supply_chain_risk.db.db_read_enabled", return_value=True):
            with patch("api.pe.supply_chain_risk.db.safe_fetchall", return_value=mock_entity_rows):
                with patch("api.pe.supply_chain_risk.db.safe_fetchone", return_value=(mock_axiom_payload,)):
                    result = monitor_portfolio_analogs("org1")
        assert len(result["alerts"]) == 0
        assert "Company A" in result["clean_entities"]

    def test_systemic_risk_label(self):
        from api.pe.supply_chain_risk import _systemic_label
        assert _systemic_label(10.0) == "low"
        assert _systemic_label(40.0) == "moderate"
        assert _systemic_label(65.0) == "high"
        assert _systemic_label(80.0) == "critical"


# ---------------------------------------------------------------------------
# TestLPReporting
# ---------------------------------------------------------------------------

class TestLPReporting:

    def test_portfolio_summary_structure(self):
        from api.pe.lp_reporting import generate_portfolio_summary
        with patch("api.pe.lp_reporting.db.db_read_enabled", return_value=False):
            result = generate_portfolio_summary("org1", dt.date.today())
        required_keys = {
            "total_companies", "avg_health_score", "health_distribution",
            "avg_ebitda_margin", "portfolio_revenue_growth",
            "companies_on_target", "companies_at_risk",
            "aggregate_schilit_risk", "unrealized_value_trend",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_health_distribution_sums(self):
        from api.pe.lp_reporting import _health_bucket
        scores = [80.0, 60.0, 45.0, 30.0, 75.0, 55.0]
        dist = {"strong": 0, "healthy": 0, "watch": 0, "stressed": 0}
        for s in scores:
            dist[_health_bucket(s)] += 1
        total = sum(dist.values())
        assert total == len(scores)

    def test_value_creation_sums(self):
        from api.pe.lp_reporting import generate_value_creation_attribution
        with patch("api.pe.lp_reporting.db.db_read_enabled", return_value=False):
            result = generate_value_creation_attribution("org1")
        total = result["total_value_change"]
        parts = (
            result["revenue_contribution"]
            + result["margin_contribution"]
            + result["leverage_contribution"]
            + result["multiple_contribution"]
        )
        assert abs(total - parts) < 1e-6

    def test_exit_pipeline_structure(self):
        from api.pe.lp_reporting import generate_exit_pipeline
        with patch("api.pe.lp_reporting.db.db_read_enabled", return_value=False):
            result = generate_exit_pipeline("org1")
        assert "ready_to_exit" in result
        assert "approaching_exit" in result
        assert "early_stage" in result
        assert "optimal_market_window" in result

    def test_narrative_generated(self):
        from api.pe.lp_reporting import _build_narratives
        summary = {
            "total_companies": 5, "avg_health_score": 68.0,
            "companies_on_target": 3, "companies_at_risk": 1,
            "portfolio_revenue_growth": 0.12,
        }
        exit_pipe = {"ready_to_exit": [{"entity_name": "PortCo A"}], "optimal_market_window": "favorable"}
        narratives = _build_narratives(summary, ["high leverage at PortCo B"], exit_pipe, {"regime": "TRENDING", "sri": 35.0})
        assert len(narratives["executive_summary"]) > 20
        assert "5 companies" in narratives["executive_summary"]

    def test_lp_report_has_all_sections(self):
        from api.pe.lp_reporting import generate_lp_report
        with patch("api.pe.lp_reporting.db.db_read_enabled", return_value=False):
            with patch("api.pe.lp_reporting.db.db_write_enabled", return_value=False):
                report = generate_lp_report("org1", "2026-Q1")
        assert report.portfolio_summary is not None
        assert isinstance(report.risk_flags, list)
        assert report.exit_pipeline is not None
        assert report.narrative_sections is not None
        assert "executive_summary" in report.narrative_sections
        assert report.report_quarter == "2026-Q1"
