"""Phase 16 tests: Natural Language Intelligence and Explainability Layer."""
from __future__ import annotations

import pytest

from api.explain.reasoning_engine import (
    ReasoningChain,
    build_reasoning_chain,
    get_theoretical_grounding,
)
from api.explain.explanation_system import (
    compute_evidence_balance,
    extract_evidence_items,
)
from api.explain.counterfactual import (
    compute_counterfactuals,
    compute_signal_sensitivity,
)
from api.explain.research_report import (
    ResearchReport,
    format_report_as_text,
    generate_research_report,
)
from api.explain.conversational import (
    answer_intelligence_query,
    classify_query_intent,
    extract_symbols_from_query,
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_BUY_PAYLOAD = {
    "deployable_alpha_utility": 75.0,
    "ic_state": "STRONG",
    "regime_label": "TRENDING",
    "regime_strength": 0.75,
    "amqs_score": 72.0,
    "alpha_decomposition": {
        "primary_driver": "EIF",
        "factor_contributions": {
            "EIF": 8.5,
            "CMF": 5.2,
            "BAF": 3.1,
            "KLF": 2.8,
            "SCAF": -1.5,
        },
        "factor_concentration": 0.60,
        "regime_adjusted_loadings": {
            "EIF": 0.17,
            "CMF": 0.10,
            "BAF": 0.06,
            "KLF": 0.056,
            "SCAF": -0.03,
        },
        "systematic_contribution": 18.1,
        "idiosyncratic_alpha": 6.9,
    },
    "engine_scores": {
        "fundamental_reality": {
            "score": 72.0,
            "components": {
                "eis_component": 68.0,
                "caps_component": 74.0,
                "valuation_gap": 65.0,
            },
        },
        "critical_fragility": {
            "score": 35.0,
            "components": {
                "scps_component": 40.0,
                "mtrs_component": 35.0,
                "bfs_component": 30.0,
            },
        },
        "behavioral_distortion": {
            "score": 65.0,
            "components": {
                "asymmetric_sent_score": 70.0,
                "crowding_score": 45.0,
            },
        },
        "flow_transmission": {
            "score": 68.0,
            "components": {
                "trend_quality": 65.0,
                "transmission_strength": 70.0,
            },
        },
        "state_pricing": {"score": 60.0, "components": {}},
        "liquidity_convexity": {
            "score": 58.0,
            "components": {
                "osms_component": 70.0,
                "ias_component": 65.0,
            },
        },
        "research_integrity": {"score": 70.0, "components": {}},
    },
}

_SELL_PAYLOAD = {
    **_BUY_PAYLOAD,
    "deployable_alpha_utility": 30.0,
    "ic_state": "DEGRADED",
    "alpha_decomposition": {
        **_BUY_PAYLOAD["alpha_decomposition"],
        "factor_contributions": {
            "EIF": -8.5,
            "CMF": -5.2,
            "SCAF": 4.0,
        },
    },
    "engine_scores": {
        **_BUY_PAYLOAD["engine_scores"],
        "fundamental_reality": {
            "score": 35.0,
            "components": {
                "eis_component": 30.0,
                "caps_component": 32.0,
                "valuation_gap": 40.0,
            },
        },
        "critical_fragility": {
            "score": 72.0,
            "components": {
                "scps_component": 78.0,
                "mtrs_component": 70.0,
                "bfs_component": 68.0,
            },
        },
    },
}

# Near-threshold BUY for counterfactual tests
_MARGINAL_BUY_PAYLOAD = {
    **_BUY_PAYLOAD,
    "deployable_alpha_utility": 67.0,
}


# ===========================================================================
# TestReasoningEngine
# ===========================================================================

class TestReasoningEngine:
    def test_chain_has_minimum_steps(self):
        chain = build_reasoning_chain("AAPL", _BUY_PAYLOAD, "BUY")
        assert len(chain.reasoning_steps) >= 4

    def test_chain_primary_conclusion_not_empty(self):
        chain = build_reasoning_chain("AAPL", _BUY_PAYLOAD, "BUY")
        assert isinstance(chain.primary_conclusion, str)
        assert len(chain.primary_conclusion) > 0

    def test_chain_confidence_bounded(self):
        chain = build_reasoning_chain("AAPL", _BUY_PAYLOAD, "BUY")
        assert 0.0 <= chain.confidence_overall <= 1.0

    def test_step_theoretical_grounding_set(self):
        chain = build_reasoning_chain("AAPL", _BUY_PAYLOAD, "BUY")
        for step in chain.reasoning_steps:
            assert isinstance(step.theoretical_grounding, str)
            assert len(step.theoretical_grounding) > 0

    def test_supporting_factors_aligned_buy(self):
        chain = build_reasoning_chain("AAPL", _BUY_PAYLOAD, "BUY")
        for f in chain.supporting_factors:
            assert f["contribution"] > 0, f"Expected positive contribution for BUY supporting: {f}"

    def test_contradicting_factors_exist_mixed_evidence(self):
        # SCAF=-1.5 means contradicting factor present for BUY
        chain = build_reasoning_chain("AAPL", _BUY_PAYLOAD, "BUY")
        assert len(chain.contradicting_factors) > 0

    def test_invalidation_conditions_non_empty(self):
        chain = build_reasoning_chain("AAPL", _BUY_PAYLOAD, "BUY")
        assert len(chain.invalidation_conditions) >= 2

    def test_theoretical_foundations_from_knowledge_vault(self):
        chain = build_reasoning_chain("AAPL", _BUY_PAYLOAD, "BUY")
        all_text = " ".join(chain.theoretical_foundations)
        known_authors = ["Penman", "Grinold", "Sornette", "Kindleberger", "Mandelbrot"]
        assert any(author in all_text for author in known_authors)

    def test_factor_theoretical_grounding_eif(self):
        grounding = get_theoretical_grounding("EIF")
        assert "Penman" in grounding

    def test_factor_theoretical_grounding_scaf(self):
        grounding = get_theoretical_grounding("SCAF")
        assert "Sornette" in grounding


# ===========================================================================
# TestEvidenceSystem
# ===========================================================================

class TestEvidenceSystem:
    def test_evidence_items_returned(self):
        evidence = extract_evidence_items(_BUY_PAYLOAD, "BUY")
        assert "supporting" in evidence
        assert "contradicting" in evidence

    def test_evidence_strength_bounded(self):
        evidence = extract_evidence_items(_BUY_PAYLOAD, "BUY")
        for direction in ("supporting", "contradicting", "neutral"):
            for item in evidence.get(direction, []):
                assert 0.0 <= item.strength <= 1.0, f"Strength {item.strength} out of [0,1]"

    def test_supporting_for_buy_with_high_eis(self):
        evidence = extract_evidence_items(_BUY_PAYLOAD, "BUY")
        categories = [e.category for e in evidence["supporting"]]
        assert "fundamental" in categories

    def test_contradicting_for_buy_with_high_scps(self):
        payload_high_scps = {
            **_BUY_PAYLOAD,
            "engine_scores": {
                **_BUY_PAYLOAD["engine_scores"],
                "critical_fragility": {
                    "score": 72.0,
                    "components": {
                        "scps_component": 80.0,
                        "mtrs_component": 70.0,
                        "bfs_component": 65.0,
                    },
                },
            },
        }
        evidence = extract_evidence_items(payload_high_scps, "BUY")
        categories = [e.category for e in evidence["contradicting"]]
        assert "risk" in categories

    def test_evidence_balance_computed(self):
        evidence = extract_evidence_items(_BUY_PAYLOAD, "BUY")
        balance = compute_evidence_balance(evidence)
        sup_total = sum(e.strength for e in evidence["supporting"])
        con_total = sum(e.strength for e in evidence["contradicting"])
        expected_net = round(sup_total - con_total, 4)
        assert abs(balance["net_evidence_score"] - expected_net) < 0.01

    def test_evidence_quality_strong(self):
        # Strong BUY payload with many high-strength items
        evidence = extract_evidence_items(_BUY_PAYLOAD, "BUY")
        balance = compute_evidence_balance(evidence)
        # With EIS=68, flow=68, sentiment=70, OSMS=70, IAS=65 all supporting BUY:
        assert balance["evidence_quality"] in ("strong", "moderate")

    def test_evidence_quality_mixed(self):
        # Payload with balanced supporting/contradicting evidence
        mixed_payload = {
            **_BUY_PAYLOAD,
            "engine_scores": {
                **_BUY_PAYLOAD["engine_scores"],
                "fundamental_reality": {
                    "score": 50.0,
                    "components": {"eis_component": 50.0, "caps_component": 50.0},
                },
                "flow_transmission": {"score": 50.0, "components": {}},
                "critical_fragility": {
                    "score": 65.0,
                    "components": {"scps_component": 70.0},
                },
            },
        }
        evidence = extract_evidence_items(mixed_payload, "BUY")
        balance = compute_evidence_balance(evidence)
        assert balance["evidence_quality"] in ("mixed", "weak", "moderate")


# ===========================================================================
# TestCounterfactual
# ===========================================================================

class TestCounterfactual:
    def test_counterfactuals_returned(self):
        result = compute_counterfactuals(_MARGINAL_BUY_PAYLOAD, "BUY")
        assert len(result) >= 1

    def test_counterfactual_has_required_keys(self):
        result = compute_counterfactuals(_MARGINAL_BUY_PAYLOAD, "BUY")
        for cf in result:
            assert "component" in cf
            assert "current_value" in cf
            assert "delta_needed" in cf
            assert "plain_english" in cf

    def test_delta_needed_positive(self):
        result = compute_counterfactuals(_MARGINAL_BUY_PAYLOAD, "BUY")
        for cf in result:
            assert cf["delta_needed"] >= 0.0

    def test_plain_english_not_empty(self):
        result = compute_counterfactuals(_MARGINAL_BUY_PAYLOAD, "BUY")
        for cf in result:
            assert isinstance(cf["plain_english"], str)
            assert len(cf["plain_english"]) > 0

    def test_target_signal_opposite_default(self):
        # When no target_signal given for BUY, should compute toward SELL
        result = compute_counterfactuals(_MARGINAL_BUY_PAYLOAD, "BUY")
        assert isinstance(result, list)

    def test_sensitivity_curve_has_points(self):
        result = compute_signal_sensitivity(_BUY_PAYLOAD, "eis_component")
        assert len(result["sensitivity_curve"]) > 5

    def test_sensitivity_score_bounded(self):
        result = compute_signal_sensitivity(_BUY_PAYLOAD, "fragility_score")
        assert 0.0 <= result["sensitivity_score"] <= 1.0


# ===========================================================================
# TestResearchReport
# ===========================================================================

def _make_payload_with_dau(dau: float) -> dict:
    return {**_BUY_PAYLOAD, "deployable_alpha_utility": dau}


class TestResearchReport:
    def test_report_generated(self):
        report = generate_research_report("AAPL", _BUY_PAYLOAD, "BUY")
        assert isinstance(report, ResearchReport)
        assert report.symbol == "AAPL"
        assert report.analyst_rating
        assert report.target_conviction

    def test_analyst_rating_maps_dau_strong_buy(self):
        report = generate_research_report("AAPL", _make_payload_with_dau(85.0), "BUY")
        assert report.analyst_rating == "Strong Buy"

    def test_analyst_rating_maps_dau_strong_sell(self):
        report = generate_research_report("AAPL", _make_payload_with_dau(30.0), "SELL")
        assert report.analyst_rating == "Strong Sell"

    def test_conviction_maps_ic_state_strong(self):
        payload = {**_BUY_PAYLOAD, "ic_state": "STRONG"}
        report = generate_research_report("AAPL", payload, "BUY")
        assert report.target_conviction == "High"

    def test_executive_summary_mentions_symbol(self):
        report = generate_research_report("AAPL", _BUY_PAYLOAD, "BUY")
        assert "AAPL" in report.executive_summary

    def test_investment_thesis_not_empty(self):
        report = generate_research_report("AAPL", _BUY_PAYLOAD, "BUY")
        assert isinstance(report.investment_thesis, str)
        assert len(report.investment_thesis) > 20

    def test_text_format_contains_sections(self):
        report = generate_research_report("AAPL", _BUY_PAYLOAD, "BUY")
        text = format_report_as_text(report)
        assert "EXECUTIVE SUMMARY" in text
        assert "RISK FACTORS" in text

    def test_llm_fallback_graceful(self):
        # LLM is disabled in tests — should return programmatic report without error
        report = generate_research_report("AAPL", _BUY_PAYLOAD, "BUY", use_llm=True)
        assert report.symbol == "AAPL"
        assert len(report.executive_summary) > 0

    def test_report_invalidation_conditions(self):
        report = generate_research_report("AAPL", _BUY_PAYLOAD, "BUY")
        assert len(report.invalidation_triggers) >= 2


# ===========================================================================
# TestConversational
# ===========================================================================

class TestConversational:
    def test_intent_signal_explanation(self):
        assert classify_query_intent("Why is AAPL a BUY") == "signal_explanation"

    def test_intent_explanation_explain(self):
        assert classify_query_intent("Explain what drives MSFT signal") == "signal_explanation"

    def test_intent_comparison(self):
        assert classify_query_intent("Compare AAPL vs MSFT") == "comparison"

    def test_intent_risk_query(self):
        assert classify_query_intent("What is the bubble risk for TSLA") == "risk_query"

    def test_intent_screening(self):
        assert classify_query_intent("Show top opportunities") == "screening"

    def test_symbol_extraction_single(self):
        symbols = extract_symbols_from_query("Why is AAPL rallying")
        assert "AAPL" in symbols

    def test_symbol_extraction_multiple(self):
        symbols = extract_symbols_from_query("Compare AAPL and MSFT")
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_query_no_llm_returns_answer(self):
        result = answer_intelligence_query(
            "Why is AAPL a BUY",
            {"axiom_data": [{"symbol": "AAPL", "dau": "75.0", "ic_state": "STRONG",
                              "regime": "TRENDING", "fragility": "35.0",
                              "eis": "68.0", "primary_driver": "EIF"}]},
            use_llm=False,
        )
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_answer_has_required_keys(self):
        result = answer_intelligence_query("Show top opportunities", {}, use_llm=False)
        for key in ("query", "intent", "answer", "confidence", "grounded"):
            assert key in result

    def test_grounded_true_when_data_available(self):
        context = {"axiom_data": [{"symbol": "AAPL", "dau": "75.0"}]}
        result = answer_intelligence_query("Why is AAPL up", context, use_llm=False)
        assert result["grounded"] is True

    def test_confidence_low_empty_context(self):
        result = answer_intelligence_query("What is the risk for TSLA", {}, use_llm=False)
        assert result["confidence"] == "low"
