from __future__ import annotations

from typing import Any, Dict, Tuple

from api.axiom.contracts import AxiomArtifact
from api.axiom.engine import AXIOM_FRAMEWORK_VERSION, build_axiom_artifact
from api.axiom.engines import (
    score_behavioral_distortion,
    score_flow_transmission,
    score_liquidity_convexity,
    score_research_integrity,
    score_state_pricing,
)
from api.axiom.mappers import build_axiom_engine_input


def _phase2_case(
    *,
    state_case: str = "strong",
    behavior_case: str = "continuation",
    flow_case: str = "strong",
    liquidity_case: str = "strong",
    research_case: str = "strong",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    strong_state = state_case == "strong"
    strong_flow = flow_case == "strong"
    strong_liquidity = liquidity_case == "strong"
    strong_research = research_case == "strong"
    continuation = behavior_case == "continuation"
    euphoria = behavior_case == "euphoria"
    recovery = behavior_case == "recovery"

    normalized_bundle = {
        "symbol_meta": {"symbol": "NVDA", "sector": "Technology"},
        "raw_supporting_fields": {
            "signal": {
                "symbol": "NVDA",
                "action": "BUY" if strong_state and strong_flow else "HOLD",
                "score": 0.86 if strong_state and strong_flow else 0.12,
                "confidence": 0.72 if strong_research else 0.38,
                "horizon_days": 21,
            },
            "key_features": {
                "ret_21d": 0.11 if strong_flow else -0.05,
                "mom_vol_adj_21d": 0.72 if strong_flow else -0.18,
                "regime_label": "trend" if strong_flow else "transition",
            },
            "quality": {
                "quality_score": 86.0 if strong_research else 42.0,
                "warnings": [] if strong_research else ["coverage remains thin"],
            },
        },
        "canonical_alpha_core": {
            "lineage": {
                "snapshot_id": "snap-nvda-2024-01-02",
                "snapshot_version": "phase8_canonical_snapshot_v1",
                "feature_version": "phase8_canonical_features_v1",
                "signal_version": "phase9_canonical_signal_v1",
            },
            "signal_payload": {
                "suppression_flags": []
                if strong_flow and not euphoria
                else ["weak_breadth", "market_stress"],
                "adjusted_confidence_notes": []
                if strong_research
                else ["Confidence is reduced because validation remains thin."],
            },
        },
        "market_price_volume": {
            "meta": {"coverage_score": 0.95},
            "latest_close": 118.0,
            "ret_21d": 0.11 if strong_flow else -0.05,
            "realized_vol_21d": 0.17 if strong_liquidity else 0.48,
            "realized_vol_63d": 0.19 if strong_liquidity else 0.44,
            "vol_of_vol_proxy": 0.18 if strong_liquidity else 0.83,
            "gap_pct": 0.012 if strong_liquidity else 0.07,
            "gap_instability_10d": 0.22 if strong_liquidity else 1.64,
            "abs_gap_mean_10d": 0.012 if strong_liquidity else 0.041,
            "return_dispersion_21d": 0.011 if strong_flow else 0.038,
            "return_dispersion_63d": 0.013 if strong_flow else 0.042,
            "downside_asymmetry_21d": 0.92 if strong_liquidity else 2.06,
            "downside_asymmetry_63d": 0.95 if strong_liquidity else 1.91,
            "maxdd_21d": -0.06 if not recovery else -0.16,
            "maxdd_63d": -0.11 if not recovery else -0.26,
            "maxdd_126d": -0.15 if not recovery else -0.39,
            "atr_pct": 0.036 if strong_liquidity else 0.082,
        },
        "fundamental_filing": {
            "meta": {"coverage_score": 0.9 if strong_state else 0.46},
            "provenance": {"confidence": 82.0 if strong_state else 44.0},
            "normalized_metrics": {
                "revenue_growth_yoy": 0.18 if strong_state else -0.04,
                "gross_margin": 0.63 if strong_state else 0.21,
                "operating_margin": 0.28 if strong_state else 0.01,
                "net_margin": 0.21 if strong_state else 0.02,
                "return_on_assets": 0.12 if strong_state else 0.01,
                "return_on_equity": 0.24 if strong_state else 0.03,
                "positive_fcf_ratio": 0.92 if strong_state else 0.34,
                "free_cash_flow": 1_240_000_000.0 if strong_state else -45_000_000.0,
                "free_cash_flow_margin": 0.18 if strong_state else -0.02,
                "current_ratio": 1.75 if strong_state else 0.88,
                "cash_ratio": 0.86 if strong_state else 0.12,
                "debt_to_equity": 0.44 if strong_state else 1.88,
                "liabilities_to_assets": 0.45 if strong_state else 0.86,
            },
            "quality_proxies": {
                "profitability_strength": 82.0 if strong_state else 36.0,
                "balance_sheet_resilience": 76.0 if strong_state else 34.0,
                "cash_flow_durability": 79.0 if strong_state else 28.0,
                "reporting_completeness_score": 84.0 if strong_state else 42.0,
                "reporting_quality_proxy": 78.0 if strong_state else 39.0,
            },
            "durability_proxies": {
                "balance_sheet_resilience": 74.0 if strong_state else 32.0,
                "cash_flow_durability": 77.0 if strong_state else 26.0,
            },
            "filing_recency_days": 32.0 if strong_state else 221.0,
            "coverage_flags": {
                "income_statement": True,
                "balance_sheet": strong_state,
                "cash_flow_statement": strong_state,
            },
            "strength_summary": ["margins are expanding", "cash conversion is strong"]
            if strong_state
            else [],
            "weakness_summary": []
            if strong_state
            else ["cash conversion is weak", "leverage is elevated"],
            "provider_snapshot": {
                "alphavantage_overview": {
                    "analyst_target_price": 158.0 if strong_state else 92.0,
                    "pe_ratio": 19.0 if strong_state else 42.0,
                    "peg_ratio": 1.05 if strong_state else 3.1,
                }
            },
            "company_profile": {"market_cap": 820_000_000_000.0},
            "latest_quarter": {
                "gross_margin": 0.63 if strong_state else 0.21,
                "op_margin": 0.28 if strong_state else 0.01,
            },
        },
        "event_catalyst_risk": {
            "meta": {"coverage_score": 0.86},
            "event_overhang_score": 22.0 if not euphoria else 76.0,
            "event_uncertainty_score": 24.0 if not euphoria else 73.0,
            "event_risk_classification": "post_event_repricing_state"
            if recovery
            else "event_distorted"
            if euphoria
            else "low_event_risk",
        },
        "liquidity_execution_fragility": {
            "meta": {"coverage_score": 0.89},
            "implementation_fragility_score": 24.0 if strong_liquidity else 79.0,
            "liquidity_quality_score": 84.0 if strong_liquidity else 34.0,
            "tradability_caution_score": 22.0 if strong_liquidity else 76.0,
            "overnight_gap_risk_score": 24.0 if strong_liquidity else 72.0,
            "friction_proxy_score": 18.0 if strong_liquidity else 68.0,
            "execution_cleanliness_score": 78.0 if strong_liquidity else 29.0,
        },
        "market_breadth_internals": {
            "meta": {"coverage_score": 0.82},
            "breadth_confirmation_score": 74.0 if strong_flow else 28.0,
        },
        "cross_asset_confirmation": {
            "meta": {"coverage_score": 0.82},
            "cross_asset_conflict_score": 22.0 if strong_state else 71.0,
            "benchmark_confirmation_score": 70.0 if strong_flow else 34.0,
            "sector_confirmation_score": 72.0 if strong_flow else 29.0,
        },
        "stress_spillover_conditions": {
            "meta": {"coverage_score": 0.8},
            "market_stress_score": 18.0 if strong_research and strong_liquidity else 77.0,
        },
        "quality_provenance": {
            "meta": {"coverage_score": 0.9, "confidence": 82.0 if strong_research else 41.0}
        },
        "sentiment_narrative_flow": {"meta": {"coverage_score": 0.8}},
        "macro_cross_asset": {"meta": {"coverage_score": 0.84}},
        "domain_availability": {
            "market_price_volume": 0.95,
            "fundamental_filing": 0.9 if strong_state else 0.46,
            "macro_cross_asset": 0.84,
            "sentiment_narrative_flow": 0.8,
        },
    }

    feature_factor_bundle = {
        "composite_intelligence": {
            "Opportunity Quality Score": 82.0 if strong_state and strong_flow else 33.0,
            "Cross-Domain Conviction Score": 78.0 if strong_state and strong_flow else 36.0,
            "Market Structure Integrity Score": 74.0 if strong_flow else 28.0,
            "Macro Alignment Score": 69.0 if strong_state else 27.0,
            "Regime Stability Score": 72.0 if strong_flow else 24.0,
            "Fundamental Durability Score": 82.0 if strong_state else 28.0,
            "Narrative Crowding Index": 24.0 if continuation else 78.0 if euphoria else 34.0,
            "Signal Fragility Index": 26.0 if strong_liquidity else 82.0,
        },
        "market_structure": {
            "trend_quality_score": 78.0 if strong_flow else 26.0,
            "momentum_consistency_score": 75.0 if strong_flow else 31.0,
            "breakout_follow_through_score": 71.0 if strong_flow else 25.0,
            "price_volume_alignment_score": 72.0 if strong_flow else 34.0,
            "directional_persistence_score": 76.0 if strong_flow else 30.0,
            "reversal_pressure_score": 24.0 if strong_flow else 72.0,
            "trend_exhaustion_score": 22.0 if continuation else 77.0 if euphoria else 42.0,
        },
        "sentiment_narrative_intelligence": {
            "sentiment_direction_score": 68.0 if continuation else 34.0 if euphoria else 58.0,
            "sentiment_level_score": 62.0 if continuation else 84.0 if euphoria else 28.0,
            "sentiment_trend_score": 64.0 if continuation else 74.0 if euphoria else 38.0,
            "attention_intensity_score": 54.0 if continuation else 88.0 if euphoria else 44.0,
            "novelty_score": 58.0 if continuation else 78.0 if euphoria else 46.0,
            "repetition_score": 38.0 if continuation else 82.0 if euphoria else 26.0,
            "narrative_concentration_score": 42.0 if continuation else 79.0 if euphoria else 36.0,
            "contradiction_score": 24.0 if continuation else 66.0 if euphoria else 31.0,
            "hype_to_price_divergence_score": 18.0 if continuation else 76.0 if euphoria else 32.0,
            "positive_news_weak_price_divergence": 12.0 if continuation else 72.0 if euphoria else 24.0,
            "negative_news_resilient_price_divergence": 28.0 if continuation else 16.0 if euphoria else 68.0,
            "event_pressure_score": 24.0 if continuation else 74.0 if euphoria else 36.0,
            "crowding_proxy_score": 28.0 if continuation else 81.0 if euphoria else 34.0,
        },
        "macro_alignment": {
            "macro_alignment_score": 71.0 if strong_state else 28.0,
            "growth_alignment_score": 67.0 if strong_state else 31.0,
            "risk_on_risk_off_alignment": 65.0 if strong_state else 29.0,
            "macro_regime_consistency": 69.0 if strong_state else 24.0,
            "macro_conflict_score": 24.0 if strong_state else 72.0,
            "macro_fragility_score": 28.0 if strong_state else 76.0,
            "rates_sensitivity_proxy": 44.0 if strong_state else 72.0,
            "inflation_stress_proxy": 31.0 if strong_state else 73.0,
        },
        "cross_asset_relative_context": {
            "benchmark_relative_strength": 68.0 if strong_flow else 31.0,
            "sector_relative_strength": 71.0 if strong_flow else 28.0,
            "sector_confirmation_score": 70.0 if strong_flow else 28.0,
            "relative_context_quality": 74.0 if strong_flow else 35.0,
            "idiosyncratic_strength_vs_market": 66.0 if strong_flow else 27.0,
            "idiosyncratic_weakness_vs_market": 18.0 if strong_flow else 68.0,
        },
        "domain_agreement": {
            "domain_agreement_score": 76.0 if strong_state and strong_flow else 32.0,
            "domain_conflict_score": 24.0 if strong_state and strong_flow else 71.0,
        },
        "fragility_intelligence": {
            "instability_score": 24.0 if strong_liquidity else 82.0,
            "volatility_stress_score": 22.0 if strong_liquidity else 85.0,
            "drawdown_sensitivity_score": 29.0 if strong_liquidity else 86.0,
            "anomaly_pressure_score": 21.0 if strong_liquidity else 73.0,
            "clean_setup_score": 76.0 if strong_liquidity else 24.0,
            "noisy_setup_score": 26.0 if strong_liquidity else 82.0,
        },
        "regime_intelligence": {
            "regime_label": "trend" if strong_flow else "transition",
            "transition_risk": 28.0 if strong_flow else 82.0,
            "regime_instability": 24.0 if strong_flow else 78.0,
        },
        "proprietary_scores": {
            "Narrative Crowding Index": {
                "score": 24.0 if continuation else 78.0 if euphoria else 34.0
            },
            "Signal Fragility Index": {
                "score": 26.0 if strong_liquidity else 82.0
            },
        },
    }

    strategy_bundle = {
        "strategy_posture": "actionable_long" if strong_state and strong_flow else "wait",
        "conviction_tier": "high" if strong_state and strong_flow else "low",
        "confidence_score": 78.0 if strong_research else 36.0,
        "actionability_score": 72.0 if strong_state and strong_flow else 24.0,
        "execution_posture": {
            "preferred_posture": "confirmation_add"
            if strong_flow
            else "wait_for_confirmation",
            "signal_cleanliness": "clean" if strong_liquidity else "noisy",
            "urgency_level": "measured",
            "patience_level": "medium" if strong_flow else "high",
        },
        "confirmation_triggers": ["price confirms with volume"],
        "deterioration_triggers": ["relative strength rolls over"],
        "fragility_vetoes": [{"name": "crowding"}] if euphoria else [],
        "invalidators": {"top_invalidators": ["macro alignment deteriorates"]},
    }

    report_context = {
        "deployment_permission": "limited_live_eligible"
        if strong_research
        else "paper_shadow_only",
        "trust_tier": "limited_live" if strong_research else "paper_only",
        "live_readiness_score": 79.0 if strong_research else 36.0,
        "model_readiness_status": "ready" if strong_research else "constrained",
        "evaluation": {
            "calibration_summary": {
                "confidence_reliability_score": 76.0 if strong_research else 39.0,
                "confidence_monotonicity": "higher_confidence_buckets_outperform"
                if strong_research
                else "broken",
            },
            "signal_scorecard": {
                "final_signal_overall": {
                    "matured_count": 18 if strong_research else 2,
                    "hit_rate": 0.69 if strong_research else 0.35,
                }
            },
            "strategy_scorecard": {
                "actionable_vs_watchlist_return_spread": 0.05
                if strong_research
                else -0.015,
            },
            "ranking_scorecard": {
                "confidence_monotonicity": "higher_confidence_buckets_outperform"
                if strong_research
                else "broken",
            },
        },
        "deployment_readiness": {
            "model_readiness": {
                "live_readiness_score": 79.0 if strong_research else 36.0,
                "model_readiness_status": "ready" if strong_research else "constrained",
                "evaluation_consistency": {
                    "consistency_score": 74.0 if strong_research else 32.0,
                    "matured_prediction_count": 18 if strong_research else 2,
                    "hit_rate": 0.69 if strong_research else 0.35,
                },
                "live_readiness_blockers": []
                if strong_research
                else ["calibration remains weak"],
            },
            "deployment_permission": {
                "deployment_permission": "limited_live_eligible"
                if strong_research
                else "paper_shadow_only",
                "trust_tier": "limited_live" if strong_research else "paper_only",
                "deployment_blockers": []
                if strong_research
                else ["research integrity remains weak"],
            },
        },
        "canonical_validation": {
            "net_return_summary": {
                "average_edge_return": 0.028 if strong_research else -0.008
            },
            "walkforward_summary": {"window_count": 3 if strong_research else 1},
            "readiness_scorecard": {
                "paper_vs_live_candidate_quality_summary": 0.05
                if strong_research
                else -0.01
            },
            "suppression_effect_summary": {
                "suppression_effect_edge_spread": 0.03
                if strong_research
                else -0.02
            },
        },
        "operational_guardrails": {
            "drift_monitoring": {"model_drift_score": 18.0 if strong_research else 72.0},
            "system_health": {
                "system_health_status": "healthy" if strong_research else "degraded"
            },
            "control_state": {
                "current_operating_mode": "paper_shadow"
                if strong_research
                else "paused",
                "pause_required": False if strong_research else True,
            },
            "operational_alerts": []
            if strong_research
            else [{"recommended_action": "pause higher-trust modes"}],
        },
        "source_governance": {
            "commercialization_readiness": {
                "buyer_demo_suitability": "cleaner_candidate"
                if strong_research
                else "conditional_review_required",
                "commercialization_risk_score": 24.0 if strong_research else 71.0,
            }
        },
        "portfolio_construction": {
            "current_candidate": {
                "portfolio_candidate_score": 76.0 if strong_research else 31.0,
                "portfolio_fit_quality": 70.0 if strong_research else 28.0,
                "execution_quality_score": 77.0 if strong_liquidity else 32.0,
                "size_band": "exploratory allocation band"
                if strong_research
                else "paper / shadow band",
                "weight_band": "0.50x pilot band"
                if strong_research
                else "0.00x live weight",
                "risk_budget_band": "pilot_risk_band"
                if strong_research
                else "shadow_risk_band",
            }
        },
    }
    return normalized_bundle, feature_factor_bundle, strategy_bundle, report_context


def _build_input(**kwargs: Any):
    normalized_bundle, feature_factor_bundle, strategy_bundle, report_context = _phase2_case(
        **kwargs
    )
    return build_axiom_engine_input(
        normalized_bundle,
        job_context={"symbol": "NVDA", "as_of_date": "2024-01-02"},
        feature_factor_bundle=feature_factor_bundle,
        strategy_bundle=strategy_bundle,
        report_context=report_context,
    )


def test_state_pricing_scores_stronger_state_setup_higher() -> None:
    strong_input = _build_input(state_case="strong")
    weak_input = _build_input(state_case="weak")

    strong_score = score_state_pricing(strong_input)
    weak_score = score_state_pricing(weak_input)

    assert strong_score.score is not None
    assert weak_score.score is not None
    assert strong_score.score > weak_score.score
    assert strong_score.status in {"available", "partial"}


def test_behavioral_engine_distinguishes_continuation_euphoria_and_recovery() -> None:
    continuation_input = _build_input(behavior_case="continuation")
    euphoria_input = _build_input(behavior_case="euphoria")
    recovery_input = _build_input(behavior_case="recovery", flow_case="weak")

    continuation_score = score_behavioral_distortion(continuation_input)
    euphoria_score = score_behavioral_distortion(euphoria_input)
    recovery_score = score_behavioral_distortion(recovery_input)

    assert continuation_score.score is not None
    assert euphoria_score.score is not None
    assert recovery_score.score is not None
    assert continuation_score.score > euphoria_score.score
    assert "crowded_narrative" in euphoria_score.flags
    assert "washed_out_reversal" in recovery_score.flags


def test_flow_transmission_scores_strong_flow_above_conflicted_flow() -> None:
    strong_input = _build_input(flow_case="strong")
    weak_input = _build_input(flow_case="weak")

    strong_score = score_flow_transmission(strong_input)
    weak_score = score_flow_transmission(weak_input)

    assert strong_score.score is not None
    assert weak_score.score is not None
    assert strong_score.score > weak_score.score


def test_liquidity_convexity_scores_strong_liquidity_above_weak() -> None:
    strong_input = _build_input(liquidity_case="strong")
    weak_input = _build_input(liquidity_case="weak")

    strong_score = score_liquidity_convexity(strong_input)
    weak_score = score_liquidity_convexity(weak_input)

    assert strong_score.score is not None
    assert weak_score.score is not None
    assert strong_score.score > weak_score.score
    assert "no_option_surface_data" in strong_score.flags


def test_research_integrity_scores_strong_validation_above_drifted() -> None:
    strong_input = _build_input(research_case="strong")
    weak_input = _build_input(research_case="weak")

    strong_score = score_research_integrity(strong_input)
    weak_score = score_research_integrity(weak_input)

    assert strong_score.score is not None
    assert weak_score.score is not None
    assert strong_score.score > weak_score.score
    assert "active_model_drift" in weak_score.flags


def test_phase2_axiom_artifact_populates_all_engines_and_explanation() -> None:
    normalized_bundle, feature_factor_bundle, strategy_bundle, report_context = _phase2_case()
    artifact = build_axiom_artifact(
        normalized_bundle=normalized_bundle,
        job_context={"symbol": "NVDA", "as_of_date": "2024-01-02"},
        feature_factor_bundle=feature_factor_bundle,
        strategy_bundle=strategy_bundle,
        report_context=report_context,
    )
    validated = AxiomArtifact.model_validate(artifact)

    assert validated.framework_version == AXIOM_FRAMEWORK_VERSION
    assert all(
        validated.engine_scores[name].status in {"available", "partial"}
        for name in (
            "fundamental_reality",
            "state_pricing",
            "behavioral_distortion",
            "flow_transmission",
            "liquidity_convexity",
            "critical_fragility",
            "research_integrity",
        )
    )
    assert validated.explanation["strongest_engine"]
    assert validated.explanation["weakest_engine"]
    assert validated.explanation["monitoring_triggers"] is not None


def test_phase2_regime_and_trade_family_shift_with_behavior_and_liquidity() -> None:
    continuation_artifact = build_axiom_artifact(
        normalized_bundle=_phase2_case(behavior_case="continuation")[0],
        job_context={"symbol": "NVDA", "as_of_date": "2024-01-02"},
        feature_factor_bundle=_phase2_case(behavior_case="continuation")[1],
        strategy_bundle=_phase2_case(behavior_case="continuation")[2],
        report_context=_phase2_case(behavior_case="continuation")[3],
    )
    euphoria_artifact = build_axiom_artifact(
        normalized_bundle=_phase2_case(behavior_case="euphoria", liquidity_case="weak")[0],
        job_context={"symbol": "NVDA", "as_of_date": "2024-01-02"},
        feature_factor_bundle=_phase2_case(behavior_case="euphoria", liquidity_case="weak")[1],
        strategy_bundle=_phase2_case(behavior_case="euphoria", liquidity_case="weak")[2],
        report_context=_phase2_case(behavior_case="euphoria", liquidity_case="weak")[3],
    )

    assert continuation_artifact["regime_label"] in {
        "behavioral_continuation",
        "fundamental_convergence",
        "compensation_capture",
    }
    assert euphoria_artifact["regime_label"] in {"euphoria_critical", "liquidity_fracture"}


def test_phase2_deployability_sets_size_band_and_monitoring_triggers() -> None:
    normalized_bundle, feature_factor_bundle, strategy_bundle, report_context = _phase2_case()
    artifact = build_axiom_artifact(
        normalized_bundle=normalized_bundle,
        job_context={"symbol": "NVDA", "as_of_date": "2024-01-02"},
        feature_factor_bundle=feature_factor_bundle,
        strategy_bundle=strategy_bundle,
        report_context=report_context,
    )

    decision = artifact["deployability_decision"]
    assert decision["deployability_tier"] in {
        "live_candidate",
        "paper_trade_only",
        "monitor_only",
        "not_actionable",
    }
    assert decision["size_band_recommendation"] in {"large", "medium", "small", "none"}
    assert decision["monitoring_triggers"]
