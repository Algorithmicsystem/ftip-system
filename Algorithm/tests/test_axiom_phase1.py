from __future__ import annotations

from typing import Any, Dict, Tuple

from api.axiom.contracts import AxiomArtifact, ENGINE_KEYS
from api.axiom.deployability import classify_axiom_deployability
from api.axiom.engine import AXIOM_FRAMEWORK_VERSION, build_axiom_artifact
from api.axiom.engines import (
    score_behavioral_distortion,
    score_critical_fragility,
    score_flow_transmission,
    score_fundamental_reality,
    score_liquidity_convexity,
    score_research_integrity,
    score_state_pricing,
)
from api.axiom.mappers import build_axiom_engine_input
from api.axiom.regime import classify_axiom_regime
from api.axiom.scorecard import build_axiom_scorecard


def _build_axiom_inputs(
    *,
    fundamental_case: str = "strong",
    fragility_case: str = "calm",
    partial: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    strong_fundamentals = fundamental_case == "strong"
    calm_setup = fragility_case == "calm"

    fundamentals_meta = {
        "meta": {"coverage_score": 0.88 if strong_fundamentals and not partial else 0.36},
        "provenance": {"confidence": 81.0 if strong_fundamentals and not partial else 38.0},
        "quality_proxies": {
            "profitability_strength": 82.0 if strong_fundamentals else 28.0,
            "balance_sheet_resilience": 76.0 if strong_fundamentals else 32.0,
            "cash_flow_durability": 79.0 if strong_fundamentals else 24.0,
            "reporting_completeness_score": 84.0 if strong_fundamentals and not partial else 32.0,
            "reporting_quality_proxy": 78.0 if strong_fundamentals and not partial else 34.0,
        },
        "durability_proxies": {
            "balance_sheet_resilience": 74.0 if strong_fundamentals else 30.0,
            "cash_flow_durability": 77.0 if strong_fundamentals else 22.0,
        },
        "normalized_metrics": {
            "revenue_growth_yoy": 0.18 if strong_fundamentals else -0.06,
            "gross_margin": 0.64 if strong_fundamentals else 0.19,
            "operating_margin": 0.29 if strong_fundamentals else -0.01,
            "net_margin": 0.22 if strong_fundamentals else -0.03,
            "return_on_assets": 0.12 if strong_fundamentals else -0.01,
            "return_on_equity": 0.25 if strong_fundamentals else -0.04,
            "positive_fcf_ratio": 0.92 if strong_fundamentals else 0.21,
            "free_cash_flow": 1_250_000_000.0 if strong_fundamentals else -140_000_000.0,
            "free_cash_flow_margin": 0.19 if strong_fundamentals else -0.08,
            "current_ratio": 1.75 if strong_fundamentals else 0.81,
            "cash_ratio": 0.84 if strong_fundamentals else 0.08,
            "debt_to_equity": 0.42 if strong_fundamentals else 2.25,
            "liabilities_to_assets": 0.44 if strong_fundamentals else 0.92,
        },
        "filing_recency_days": 38.0 if strong_fundamentals else 242.0,
        "coverage_flags": {
            "income_statement": strong_fundamentals and not partial,
            "balance_sheet": strong_fundamentals and not partial,
            "cash_flow_statement": strong_fundamentals and not partial,
        },
        "strength_summary": ["expanding margins", "positive cash conversion"]
        if strong_fundamentals
        else [],
        "weakness_summary": []
        if strong_fundamentals
        else ["negative free cash flow", "elevated leverage"],
        "coverage_caveats": ["fundamental coverage is incomplete"] if partial else [],
        "provider_snapshot": {
            "alphavantage_overview": {
                "analyst_target_price": 158.0 if strong_fundamentals else 82.0,
                "pe_ratio": 19.0 if strong_fundamentals else 44.0,
                "peg_ratio": 1.05 if strong_fundamentals else 3.3,
            }
        },
        "company_profile": {"market_cap": 820_000_000_000.0},
        "latest_quarter": {
            "gross_margin": 0.64 if strong_fundamentals else 0.19,
            "op_margin": 0.29 if strong_fundamentals else -0.01,
        },
    }
    if partial:
        fundamentals_meta["provider_snapshot"] = {}
        fundamentals_meta["latest_quarter"] = {}
        fundamentals_meta["strength_summary"] = []

    market_domain = {
        "meta": {"coverage_score": 0.92},
        "latest_close": 118.0 if strong_fundamentals else 96.0,
        "ret_21d": 0.12 if calm_setup else -0.08,
        "realized_vol_21d": 0.18 if calm_setup else 0.49,
        "realized_vol_63d": 0.2 if calm_setup else 0.45,
        "vol_of_vol_proxy": 0.18 if calm_setup else 0.86,
        "gap_pct": 0.011 if calm_setup else 0.078,
        "gap_instability_10d": 0.22 if calm_setup else 1.78,
        "abs_gap_mean_10d": 0.012 if calm_setup else 0.046,
        "return_dispersion_21d": 0.011 if calm_setup else 0.039,
        "return_dispersion_63d": 0.013 if calm_setup else 0.043,
        "downside_asymmetry_21d": 0.92 if calm_setup else 2.08,
        "downside_asymmetry_63d": 0.94 if calm_setup else 1.95,
        "maxdd_21d": -0.06 if calm_setup else -0.24,
        "maxdd_63d": -0.11 if calm_setup else -0.39,
        "maxdd_126d": -0.16 if calm_setup else -0.51,
    }
    event_domain = {
        "meta": {"coverage_score": 0.84 if not partial else 0.25},
        "event_overhang_score": 18.0 if calm_setup else 79.0,
        "event_uncertainty_score": 24.0 if calm_setup else 74.0,
        "event_risk_classification": "low_event_risk" if calm_setup else "event_distorted",
    }
    liquidity_domain = {
        "meta": {"coverage_score": 0.87 if not partial else 0.42},
        "implementation_fragility_score": 22.0 if calm_setup else 78.0,
        "liquidity_quality_score": 82.0 if calm_setup else 36.0,
        "tradability_caution_score": 24.0 if calm_setup else 76.0,
        "overnight_gap_risk_score": 21.0 if calm_setup else 72.0,
        "friction_proxy_score": 18.0 if calm_setup else 69.0,
        "execution_cleanliness_score": 77.0 if calm_setup else 31.0,
    }
    breadth_domain = {
        "meta": {"coverage_score": 0.78},
        "breadth_confirmation_score": 72.0 if calm_setup else 24.0,
    }
    cross_asset_domain = {
        "meta": {"coverage_score": 0.8},
        "cross_asset_conflict_score": 22.0 if calm_setup else 71.0,
    }
    stress_domain = {
        "meta": {"coverage_score": 0.79},
        "market_stress_score": 18.0 if calm_setup else 76.0,
    }
    quality_domain = {
        "meta": {"coverage_score": 0.9, "confidence": 83.0 if not partial else 46.0},
    }
    sentiment_domain = {
        "meta": {"coverage_score": 0.66 if not partial else 0.34},
    }
    macro_domain = {
        "meta": {"coverage_score": 0.71 if not partial else 0.48},
    }
    signal_payload = {
        "suppression_flags": [] if calm_setup else ["weak_breadth", "market_stress"],
        "adjusted_confidence_notes": []
        if calm_setup
        else ["Confidence reduced because the environment is unstable."],
    }
    feature_factor_bundle = {
        "composite_intelligence": {
            "Opportunity Quality Score": 82.0 if strong_fundamentals else 34.0,
            "Cross-Domain Conviction Score": 78.0 if calm_setup else 38.0,
            "Market Structure Integrity Score": 74.0 if calm_setup else 31.0,
            "Macro Alignment Score": 68.0 if calm_setup else 29.0,
            "Regime Stability Score": 72.0 if calm_setup else 26.0,
            "Fundamental Durability Score": 82.0 if strong_fundamentals else 25.0,
            "Narrative Crowding Index": 23.0 if calm_setup else 72.0,
            "Signal Fragility Index": 28.0 if calm_setup else 81.0,
        },
        "market_structure": {
            "trend_quality_score": 76.0 if calm_setup else 28.0,
            "momentum_consistency_score": 73.0 if calm_setup else 31.0,
            "breakout_follow_through_score": 69.0 if calm_setup else 26.0,
            "price_volume_alignment_score": 71.0 if calm_setup else 34.0,
            "directional_persistence_score": 74.0 if calm_setup else 29.0,
            "reversal_pressure_score": 24.0 if calm_setup else 72.0,
            "trend_exhaustion_score": 22.0 if calm_setup else 68.0,
        },
        "sentiment_narrative_intelligence": {
            "sentiment_direction_score": 67.0 if calm_setup else 28.0,
            "sentiment_level_score": 61.0 if calm_setup else 74.0,
            "sentiment_trend_score": 63.0 if calm_setup else 34.0,
            "attention_intensity_score": 52.0 if calm_setup else 84.0,
            "novelty_score": 58.0 if calm_setup else 72.0,
            "repetition_score": 39.0 if calm_setup else 79.0,
            "narrative_concentration_score": 42.0 if calm_setup else 77.0,
            "contradiction_score": 22.0 if calm_setup else 63.0,
            "hype_to_price_divergence_score": 18.0 if calm_setup else 71.0,
            "positive_news_weak_price_divergence": 14.0 if calm_setup else 68.0,
            "negative_news_resilient_price_divergence": 42.0 if calm_setup else 18.0,
            "event_pressure_score": 26.0 if calm_setup else 72.0,
            "crowding_proxy_score": 29.0 if calm_setup else 78.0,
        },
        "macro_alignment": {
            "macro_alignment_score": 71.0 if calm_setup else 27.0,
            "growth_alignment_score": 66.0 if calm_setup else 31.0,
            "risk_on_risk_off_alignment": 64.0 if calm_setup else 28.0,
            "macro_regime_consistency": 69.0 if calm_setup else 24.0,
            "macro_conflict_score": 24.0 if calm_setup else 72.0,
            "macro_fragility_score": 28.0 if calm_setup else 76.0,
            "rates_sensitivity_proxy": 44.0 if calm_setup else 71.0,
            "inflation_stress_proxy": 31.0 if calm_setup else 73.0,
        },
        "cross_asset_relative_context": {
            "benchmark_relative_strength": 68.0 if calm_setup else 32.0,
            "sector_relative_strength": 71.0 if calm_setup else 29.0,
            "sector_confirmation_score": 69.0 if calm_setup else 26.0,
            "relative_context_quality": 73.0 if calm_setup else 34.0,
            "idiosyncratic_strength_vs_market": 66.0 if calm_setup else 27.0,
            "idiosyncratic_weakness_vs_market": 18.0 if calm_setup else 69.0,
        },
        "domain_agreement": {
            "domain_agreement_score": 76.0 if calm_setup else 33.0,
            "domain_conflict_score": 24.0 if calm_setup else 71.0,
        },
        "fragility_intelligence": {
            "instability_score": 24.0 if calm_setup else 81.0,
            "volatility_stress_score": 22.0 if calm_setup else 84.0,
            "drawdown_sensitivity_score": 29.0 if calm_setup else 86.0,
            "anomaly_pressure_score": 21.0 if calm_setup else 73.0,
            "clean_setup_score": 74.0 if calm_setup else 28.0,
            "noisy_setup_score": 27.0 if calm_setup else 79.0,
        },
        "regime_intelligence": {
            "regime_label": "trend" if calm_setup else "transition",
            "transition_risk": 28.0 if calm_setup else 82.0,
            "regime_instability": 26.0 if calm_setup else 79.0,
        },
        "fundamental_durability": {
            "profitability_quality_score": 75.0 if strong_fundamentals else 28.0,
        },
        "proprietary_scores": {
            "Narrative Crowding Index": {"score": 23.0 if calm_setup else 72.0},
            "Signal Fragility Index": {"score": 28.0 if calm_setup else 81.0},
        },
    }
    normalized_bundle = {
        "symbol_meta": {"symbol": "NVDA", "sector": "Technology"},
        "domain_availability": {
            "market_price_volume": 0.92,
            "fundamental_filing": 0.88 if strong_fundamentals and not partial else 0.36,
            "macro_cross_asset": 0.71 if not partial else 0.48,
        },
        "raw_supporting_fields": {
            "signal": {
                "symbol": "NVDA",
                "action": "BUY" if calm_setup and strong_fundamentals else "HOLD",
                "score": 0.9 if calm_setup and strong_fundamentals else 0.12,
                "confidence": 0.79 if not partial and calm_setup else 0.41,
            },
            "key_features": {
                "ret_21d": market_domain["ret_21d"],
                "mom_vol_adj_21d": 0.67 if calm_setup else -0.22,
                "regime_label": "trend" if calm_setup else "transition",
            },
            "quality": {
                "quality_score": 84.0 if not partial else 46.0,
                "warnings": [] if not partial else ["coverage remains partial"],
            },
        },
        "canonical_alpha_core": {
            "lineage": {
                "snapshot_id": "snap-nvda-2024-01-02",
                "snapshot_version": "phase8_canonical_snapshot_v1",
                "feature_version": "phase8_canonical_features_v1",
                "signal_version": "phase9_canonical_signal_v1",
            },
            "signal_payload": signal_payload,
        },
        "market_price_volume": market_domain,
        "fundamental_filing": fundamentals_meta,
        "liquidity_execution_fragility": liquidity_domain,
        "event_catalyst_risk": event_domain,
        "market_breadth_internals": breadth_domain,
        "cross_asset_confirmation": cross_asset_domain,
        "stress_spillover_conditions": stress_domain,
        "quality_provenance": quality_domain,
        "sentiment_narrative_flow": sentiment_domain,
        "macro_cross_asset": macro_domain,
    }
    return normalized_bundle, feature_factor_bundle


def _build_engine_scores(normalized_bundle: Dict[str, Any], feature_factor_bundle: Dict[str, Any]):
    strong_case = (
        (feature_factor_bundle.get("composite_intelligence") or {}).get(
            "Opportunity Quality Score",
            0.0,
        )
        >= 60.0
    )
    engine_input = build_axiom_engine_input(
        normalized_bundle,
        job_context={"symbol": "NVDA", "as_of_date": "2024-01-02"},
        feature_factor_bundle=feature_factor_bundle,
        strategy_bundle={
            "strategy_posture": "actionable_long" if strong_case else "wait",
            "conviction_tier": "high" if strong_case else "low",
            "confidence_score": 76.0 if strong_case else 38.0,
            "actionability_score": 72.0 if strong_case else 24.0,
            "execution_posture": {
                "preferred_posture": "confirmation_add" if strong_case else "wait_for_confirmation",
                "signal_cleanliness": "clean" if strong_case else "noisy",
                "urgency_level": "measured",
                "patience_level": "medium" if strong_case else "high",
            },
            "confirmation_triggers": ["price confirms with volume"],
            "deterioration_triggers": ["relative strength rolls over"],
            "fragility_vetoes": [{"name": "crowding"}] if not strong_case else [],
            "invalidators": {"top_invalidators": ["macro alignment deteriorates"]},
        },
        report_context={
            "deployment_permission": "limited_live_eligible" if strong_case else "paper_shadow_only",
            "trust_tier": "limited_live" if strong_case else "paper_only",
            "live_readiness_score": 77.0 if strong_case else 38.0,
            "model_readiness_status": "ready" if strong_case else "constrained",
            "evaluation": {
                "calibration_summary": {
                    "confidence_reliability_score": 74.0 if strong_case else 41.0,
                    "confidence_monotonicity": "higher_confidence_buckets_outperform"
                    if strong_case
                    else "broken",
                },
                "signal_scorecard": {
                    "final_signal_overall": {
                        "matured_count": 18 if strong_case else 3,
                        "hit_rate": 0.68 if strong_case else 0.36,
                    }
                },
                "strategy_scorecard": {
                    "actionable_vs_watchlist_return_spread": 0.04
                    if strong_case
                    else -0.01,
                },
                "ranking_scorecard": {
                    "confidence_monotonicity": "higher_confidence_buckets_outperform"
                    if strong_case
                    else "broken",
                },
            },
            "deployment_readiness": {
                "model_readiness": {
                    "live_readiness_score": 77.0 if strong_case else 38.0,
                    "model_readiness_status": "ready" if strong_case else "constrained",
                    "evaluation_consistency": {
                        "consistency_score": 73.0 if strong_case else 34.0,
                        "matured_prediction_count": 18 if strong_case else 3,
                        "hit_rate": 0.68 if strong_case else 0.36,
                    },
                    "live_readiness_blockers": []
                    if strong_case
                    else ["calibration remains weak"],
                },
                "deployment_permission": {
                    "deployment_permission": "limited_live_eligible"
                    if strong_case
                    else "paper_shadow_only",
                    "trust_tier": "limited_live" if strong_case else "paper_only",
                    "deployment_blockers": []
                    if strong_case
                    else ["fragility remains elevated"],
                },
                "risk_budgeting": {
                    "risk_budget_tier": "pilot" if strong_case else "shadow_only",
                },
            },
            "canonical_validation": {
                "net_return_summary": {
                    "average_edge_return": 0.025 if strong_case else -0.006
                },
                "walkforward_summary": {"window_count": 3 if strong_case else 1},
                "readiness_scorecard": {
                    "paper_vs_live_candidate_quality_summary": 0.05
                    if strong_case
                    else -0.01
                },
                "suppression_effect_summary": {
                    "suppression_effect_edge_spread": 0.03
                    if strong_case
                    else -0.01
                },
            },
            "operational_guardrails": {
                "drift_monitoring": {
                    "model_drift_score": 18.0 if strong_case else 68.0
                },
                "system_health": {
                    "system_health_status": "healthy" if strong_case else "degraded"
                },
                "control_state": {
                    "current_operating_mode": "paper_shadow"
                    if strong_case
                    else "paused",
                    "pause_required": False if strong_case else True,
                },
                "operational_alerts": []
                if strong_case
                else [{"recommended_action": "pause higher-trust modes"}],
            },
            "source_governance": {
                "commercialization_readiness": {
                    "buyer_demo_suitability": "cleaner_candidate"
                    if strong_case
                    else "conditional_review_required",
                    "commercialization_risk_score": 22.0 if strong_case else 68.0,
                }
            },
            "portfolio_construction": {
                "current_candidate": {
                    "portfolio_candidate_score": 74.0 if strong_case else 32.0,
                    "portfolio_fit_quality": 68.0 if strong_case else 28.0,
                    "execution_quality_score": 76.0 if strong_case else 34.0,
                    "size_band": "exploratory allocation band"
                    if strong_case
                    else "paper / shadow band",
                    "weight_band": "0.50x pilot band"
                    if strong_case
                    else "0.00x live weight",
                    "risk_budget_band": "pilot_risk_band"
                    if strong_case
                    else "shadow_risk_band",
                }
            },
        },
    )
    engine_scores = {
        "fundamental_reality": score_fundamental_reality(engine_input),
        "state_pricing": score_state_pricing(engine_input),
        "behavioral_distortion": score_behavioral_distortion(engine_input),
        "flow_transmission": score_flow_transmission(engine_input),
        "liquidity_convexity": score_liquidity_convexity(engine_input),
        "critical_fragility": score_critical_fragility(engine_input),
        "research_integrity": score_research_integrity(engine_input),
    }
    return engine_input, engine_scores


def test_axiom_contract_schema_validates_artifact():
    normalized_bundle, feature_factor_bundle = _build_axiom_inputs()
    artifact = build_axiom_artifact(
        normalized_bundle=normalized_bundle,
        job_context={"symbol": "NVDA", "as_of_date": "2024-01-02"},
        feature_factor_bundle=feature_factor_bundle,
    )
    validated = AxiomArtifact.model_validate(artifact)

    assert validated.framework_version == AXIOM_FRAMEWORK_VERSION
    assert validated.symbol == "NVDA"
    assert set(validated.engine_scores.keys()) == set(ENGINE_KEYS)
    assert validated.engine_scores["fundamental_reality"].status in {"available", "partial"}
    assert validated.engine_scores["critical_fragility"].status in {"available", "partial"}


def test_axiom_mapper_degrades_honestly_with_partial_input():
    normalized_bundle, feature_factor_bundle = _build_axiom_inputs(partial=True)

    engine_input = build_axiom_engine_input(
        normalized_bundle,
        job_context={"symbol": "NVDA", "as_of_date": "2024-01-02"},
        feature_factor_bundle=feature_factor_bundle,
    )

    assert engine_input.framework_version == AXIOM_FRAMEWORK_VERSION
    assert engine_input.fundamental.coverage_score < 50.0
    assert engine_input.fundamental.provider_confidence < 50.0
    assert "fundamental coverage is incomplete" in engine_input.warnings
    assert engine_input.partial_engine_hints["research_integrity"] > 0.0


def test_fundamental_reality_scores_stronger_setup_higher_than_weaker_setup():
    strong_bundle, strong_factors = _build_axiom_inputs(fundamental_case="strong")
    weak_bundle, weak_factors = _build_axiom_inputs(
        fundamental_case="weak",
        partial=True,
    )

    strong_input = build_axiom_engine_input(
        strong_bundle,
        job_context={"symbol": "NVDA", "as_of_date": "2024-01-02"},
        feature_factor_bundle=strong_factors,
    )
    weak_input = build_axiom_engine_input(
        weak_bundle,
        job_context={"symbol": "NVDA", "as_of_date": "2024-01-02"},
        feature_factor_bundle=weak_factors,
    )
    strong_score = score_fundamental_reality(strong_input)
    weak_score = score_fundamental_reality(weak_input)

    assert strong_score.score is not None
    assert weak_score.score is not None
    assert strong_score.score > weak_score.score
    assert strong_score.coverage > weak_score.coverage
    assert "thin_fundamental_coverage" in weak_score.flags


def test_critical_fragility_scores_unstable_setup_higher_than_calm_setup():
    calm_bundle, calm_factors = _build_axiom_inputs(fragility_case="calm")
    unstable_bundle, unstable_factors = _build_axiom_inputs(fragility_case="unstable")

    calm_input = build_axiom_engine_input(
        calm_bundle,
        job_context={"symbol": "NVDA", "as_of_date": "2024-01-02"},
        feature_factor_bundle=calm_factors,
    )
    unstable_input = build_axiom_engine_input(
        unstable_bundle,
        job_context={"symbol": "NVDA", "as_of_date": "2024-01-02"},
        feature_factor_bundle=unstable_factors,
    )
    calm_score = score_critical_fragility(calm_input)
    unstable_score = score_critical_fragility(unstable_input)

    assert calm_score.score is not None
    assert unstable_score.score is not None
    assert unstable_score.score > calm_score.score
    assert "market_stress" in unstable_score.flags
    assert "event_distortion" in unstable_score.flags


def test_axiom_scorecard_penalizes_high_fragility_and_low_coverage():
    strong_bundle, strong_factors = _build_axiom_inputs(
        fundamental_case="strong",
        fragility_case="calm",
    )
    fragile_bundle, fragile_factors = _build_axiom_inputs(
        fundamental_case="strong",
        fragility_case="unstable",
        partial=True,
    )

    strong_input, strong_scores = _build_engine_scores(strong_bundle, strong_factors)
    fragile_input, fragile_scores = _build_engine_scores(fragile_bundle, fragile_factors)

    strong_scorecard = build_axiom_scorecard(strong_input, strong_scores)
    fragile_scorecard = build_axiom_scorecard(fragile_input, fragile_scores)

    assert strong_scorecard.deployable_alpha_utility > fragile_scorecard.deployable_alpha_utility
    assert strong_scorecard.friction_burden < fragile_scorecard.friction_burden
    assert strong_scorecard.validated_edge > fragile_scorecard.validated_edge
    assert strong_scorecard.cross_engine_alignment > fragile_scorecard.cross_engine_alignment
    assert strong_scorecard.path_survivability > fragile_scorecard.path_survivability
    assert strong_scorecard.false_positive_penalty < fragile_scorecard.false_positive_penalty
    assert strong_scorecard.exceptional_opportunity > fragile_scorecard.exceptional_opportunity


def test_axiom_regime_classifies_fundamental_convergence_for_clean_quality_setup():
    normalized_bundle, feature_factor_bundle = _build_axiom_inputs(
        fundamental_case="strong",
        fragility_case="calm",
    )
    engine_input, engine_scores = _build_engine_scores(normalized_bundle, feature_factor_bundle)
    scorecard = build_axiom_scorecard(engine_input, engine_scores)
    regime = classify_axiom_regime(engine_input, engine_scores, scorecard)

    assert regime.regime_label == "fundamental_convergence"
    assert regime.trade_family == "convergence"
    assert "quality_supported" in regime.flags


def test_axiom_deployability_distinguishes_live_candidate_from_not_actionable():
    strong_bundle, strong_factors = _build_axiom_inputs(
        fundamental_case="strong",
        fragility_case="calm",
    )
    weak_bundle, weak_factors = _build_axiom_inputs(
        fundamental_case="weak",
        fragility_case="unstable",
        partial=True,
    )

    strong_input, strong_scores = _build_engine_scores(strong_bundle, strong_factors)
    strong_scorecard = build_axiom_scorecard(strong_input, strong_scores)
    strong_regime = classify_axiom_regime(strong_input, strong_scores, strong_scorecard)
    strong_decision = classify_axiom_deployability(
        strong_input,
        strong_scores,
        strong_scorecard,
        strong_regime,
    )

    weak_input, weak_scores = _build_engine_scores(weak_bundle, weak_factors)
    weak_scorecard = build_axiom_scorecard(weak_input, weak_scores)
    weak_regime = classify_axiom_regime(weak_input, weak_scores, weak_scorecard)
    weak_decision = classify_axiom_deployability(
        weak_input,
        weak_scores,
        weak_scorecard,
        weak_regime,
    )

    assert strong_decision.deployability_tier == "live_candidate"
    assert weak_decision.deployability_tier == "not_actionable"
    assert strong_decision.review_required is False
    assert "insufficient_axiom_coverage" in weak_decision.invalidation_flags or "stale_fundamental_backbone" in weak_decision.invalidation_flags
    assert "false_positive_pressure" in weak_decision.invalidation_flags


def test_axiom_scorecard_uses_regime_weighting_profiles_and_exceptional_setup_separation():
    strong_bundle, strong_factors = _build_axiom_inputs(
        fundamental_case="strong",
        fragility_case="calm",
    )
    transition_bundle, transition_factors = _build_axiom_inputs(
        fundamental_case="strong",
        fragility_case="unstable",
    )

    strong_input, strong_scores = _build_engine_scores(strong_bundle, strong_factors)
    transition_input, transition_scores = _build_engine_scores(
        transition_bundle, transition_factors
    )

    strong_scorecard = build_axiom_scorecard(strong_input, strong_scores)
    transition_scorecard = build_axiom_scorecard(transition_input, transition_scores)

    assert strong_scorecard.regime_weighting_profile == "trend_confirmation"
    assert transition_scorecard.regime_weighting_profile == "transition_defensive"
    assert strong_scorecard.exceptional_opportunity > transition_scorecard.exceptional_opportunity
    assert strong_scorecard.timing_support > transition_scorecard.timing_support
    assert strong_scorecard.mispricing_readiness > transition_scorecard.mispricing_readiness


def test_axiom_remaining_engines_now_expose_real_scores_or_partial_coverage():
    normalized_bundle, feature_factor_bundle = _build_axiom_inputs(partial=True)
    artifact = build_axiom_artifact(
        normalized_bundle=normalized_bundle,
        job_context={"symbol": "NVDA", "as_of_date": "2024-01-02"},
        feature_factor_bundle=feature_factor_bundle,
    )

    for engine_name in (
        "state_pricing",
        "behavioral_distortion",
        "flow_transmission",
        "liquidity_convexity",
        "research_integrity",
    ):
        payload = artifact["engine_scores"][engine_name]
        assert payload["status"] in {"available", "partial", "unavailable"}
        assert "phase1_not_implemented" not in payload["flags"]
