from __future__ import annotations

from typing import Any, Dict, Tuple

from api.axiom import (
    build_axiom_artifact,
    build_axiom_institutional_report_pack,
    build_axiom_lineage,
    build_axiom_workspace_profile,
)
from api.axiom.mappers import build_axiom_engine_input


def _phase4_fixture(
    *,
    audience_type: str = "hedge_fund",
    report_profile: str = "ic_memo",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    normalized_bundle = {
        "symbol_meta": {"symbol": "NVDA", "sector": "Technology"},
        "raw_supporting_fields": {
            "signal": {
                "symbol": "NVDA",
                "action": "BUY",
                "score": 0.81,
                "confidence": 0.67,
                "horizon_days": 21,
            },
            "key_features": {
                "ret_21d": 0.12,
                "mom_vol_adj_21d": 0.78,
                "regime_label": "trend",
            },
            "quality": {"quality_score": 89.0, "warnings": []},
        },
        "canonical_alpha_core": {
            "lineage": {
                "snapshot_id": "snap-nvda-2024-01-02",
                "snapshot_version": "phase8_canonical_snapshot_v1",
                "feature_version": "phase8_canonical_features_v1",
                "signal_version": "phase9_canonical_signal_v1",
            },
            "signal_payload": {
                "suppression_flags": [],
                "adjusted_confidence_notes": [],
            },
        },
        "market_price_volume": {
            "meta": {"coverage_score": 0.95},
            "latest_close": 118.0,
            "ret_21d": 0.12,
            "realized_vol_21d": 0.18,
            "realized_vol_63d": 0.2,
            "vol_of_vol_proxy": 0.2,
            "gap_pct": 0.01,
            "gap_instability_10d": 0.24,
            "abs_gap_mean_10d": 0.012,
            "return_dispersion_21d": 0.012,
            "return_dispersion_63d": 0.014,
            "downside_asymmetry_21d": 0.95,
            "downside_asymmetry_63d": 0.98,
            "maxdd_21d": -0.05,
            "maxdd_63d": -0.11,
            "maxdd_126d": -0.16,
        },
        "fundamental_filing": {
            "meta": {"coverage_score": 0.9},
            "provenance": {"confidence": 84.0},
            "normalized_metrics": {
                "revenue_growth_yoy": 0.19,
                "gross_margin": 0.64,
                "operating_margin": 0.29,
                "net_margin": 0.22,
                "return_on_assets": 0.13,
                "return_on_equity": 0.25,
                "positive_fcf_ratio": 0.94,
                "free_cash_flow": 1_340_000_000.0,
                "free_cash_flow_margin": 0.19,
                "current_ratio": 1.8,
                "cash_ratio": 0.88,
                "debt_to_equity": 0.41,
                "liabilities_to_assets": 0.44,
            },
            "quality_proxies": {
                "profitability_strength": 84.0,
                "balance_sheet_resilience": 77.0,
                "cash_flow_durability": 79.0,
                "reporting_completeness_score": 85.0,
                "reporting_quality_proxy": 81.0,
            },
            "durability_proxies": {
                "balance_sheet_resilience": 76.0,
                "cash_flow_durability": 80.0,
            },
            "filing_recency_days": 36.0,
            "coverage_flags": {
                "income_statement": True,
                "balance_sheet": True,
                "cash_flow_statement": True,
            },
            "strength_summary": ["margins are expanding", "cash conversion remains strong"],
            "weakness_summary": ["capex intensity remains elevated"],
            "provider_snapshot": {
                "alphavantage_overview": {
                    "analyst_target_price": 158.0,
                    "pe_ratio": 20.0,
                    "peg_ratio": 1.05,
                }
            },
            "company_profile": {"market_cap": 820_000_000_000.0},
            "latest_quarter": {"gross_margin": 0.64, "op_margin": 0.29},
        },
        "event_catalyst_risk": {
            "meta": {"coverage_score": 0.84},
            "event_overhang_score": 21.0,
            "event_uncertainty_score": 24.0,
            "event_risk_classification": "low_event_risk",
        },
        "liquidity_execution_fragility": {
            "meta": {"coverage_score": 0.88},
            "implementation_fragility_score": 26.0,
            "liquidity_quality_score": 84.0,
            "tradability_caution_score": 22.0,
            "overnight_gap_risk_score": 23.0,
            "friction_proxy_score": 18.0,
            "execution_cleanliness_score": 77.0,
        },
        "market_breadth_internals": {
            "meta": {"coverage_score": 0.82},
            "breadth_confirmation_score": 73.0,
        },
        "cross_asset_confirmation": {
            "meta": {"coverage_score": 0.81},
            "cross_asset_conflict_score": 24.0,
            "benchmark_confirmation_score": 69.0,
            "sector_confirmation_score": 71.0,
        },
        "stress_spillover_conditions": {
            "meta": {"coverage_score": 0.8},
            "market_stress_score": 18.0,
        },
        "quality_provenance": {
            "meta": {"coverage_score": 0.92, "confidence": 84.0}
        },
        "sentiment_narrative_flow": {"meta": {"coverage_score": 0.8}},
        "macro_cross_asset": {"meta": {"coverage_score": 0.84}},
        "domain_availability": {
            "market_price_volume": 0.95,
            "fundamental_filing": 0.9,
            "macro_cross_asset": 0.84,
            "sentiment_narrative_flow": 0.8,
            "liquidity_execution_fragility": 0.88,
            "quality_provenance": 0.92,
        },
    }
    feature_factor_bundle = {
        "composite_intelligence": {
            "Opportunity Quality Score": 83.0,
            "Cross-Domain Conviction Score": 77.0,
            "Market Structure Integrity Score": 75.0,
            "Macro Alignment Score": 70.0,
            "Regime Stability Score": 73.0,
            "Fundamental Durability Score": 82.0,
            "Narrative Crowding Index": 26.0,
            "Signal Fragility Index": 28.0,
        },
        "market_structure": {
            "trend_quality_score": 78.0,
            "momentum_consistency_score": 74.0,
            "breakout_follow_through_score": 72.0,
            "price_volume_alignment_score": 73.0,
            "directional_persistence_score": 75.0,
            "reversal_pressure_score": 22.0,
            "trend_exhaustion_score": 24.0,
        },
        "sentiment_narrative_intelligence": {
            "sentiment_direction_score": 66.0,
            "sentiment_level_score": 61.0,
            "sentiment_trend_score": 65.0,
            "attention_intensity_score": 52.0,
            "novelty_score": 57.0,
            "repetition_score": 36.0,
            "narrative_concentration_score": 41.0,
            "contradiction_score": 22.0,
            "hype_to_price_divergence_score": 20.0,
            "positive_news_weak_price_divergence": 18.0,
            "negative_news_resilient_price_divergence": 32.0,
            "event_pressure_score": 23.0,
        },
        "macro_alignment": {
            "macro_alignment_score": 72.0,
            "growth_alignment_score": 67.0,
            "risk_on_risk_off_alignment": 66.0,
            "macro_regime_consistency": 70.0,
            "macro_conflict_score": 23.0,
            "macro_fragility_score": 26.0,
            "rates_sensitivity_proxy": 43.0,
            "inflation_stress_proxy": 29.0,
        },
        "cross_asset_relative_context": {
            "benchmark_relative_strength": 68.0,
            "sector_relative_strength": 72.0,
            "sector_confirmation_score": 71.0,
            "relative_context_quality": 74.0,
            "idiosyncratic_strength_vs_market": 67.0,
            "idiosyncratic_weakness_vs_market": 17.0,
        },
        "domain_agreement": {
            "domain_agreement_score": 76.0,
            "domain_conflict_score": 24.0,
        },
        "fragility_intelligence": {
            "instability_score": 25.0,
            "volatility_stress_score": 24.0,
            "drawdown_sensitivity_score": 29.0,
            "anomaly_pressure_score": 22.0,
            "clean_setup_score": 76.0,
            "noisy_setup_score": 24.0,
        },
        "regime_intelligence": {
            "regime_label": "trend",
            "transition_risk": 28.0,
            "regime_instability": 25.0,
        },
        "proprietary_scores": {
            "Narrative Crowding Index": {"score": 26.0, "coverage_status": "available"},
            "Signal Fragility Index": {"score": 28.0, "coverage_status": "available"},
        },
    }
    strategy_bundle = {
        "final_signal": "BUY",
        "strategy_posture": "actionable_long",
        "confidence_score": 68.0,
        "confidence": 0.68,
        "conviction_tier": "high",
        "actionability_score": 71.0,
        "execution_posture": {
            "preferred_posture": "enter_on_confirmation",
            "urgency_level": "measured",
            "patience_level": "medium",
            "signal_cleanliness": "clean",
        },
        "invalidators": {"top_invalidators": ["macro alignment deteriorates materially"]},
        "confirmation_triggers": ["price confirms with stronger volume"],
        "deterioration_triggers": ["relative strength rolls over"],
    }
    report_context = {
        "symbol": "NVDA",
        "as_of_date": "2024-01-02",
        "signal_summary": "AXIOM sees a constructive long setup with above-average deployability.",
        "overall_analysis": "The current setup is supported by durable fundamentals, coherent macro pricing, and clean market structure.",
        "fundamental_analysis": "Fundamentals remain strong with durable profitability and clean balance-sheet support.",
        "risk_quality_analysis": "Fragility remains contained, though execution discipline still matters.",
        "deployment_permission_analysis": "Live escalation is supported only while liquidity integrity and research integrity remain intact.",
        "execution_quality_analysis": "Execution quality is solid and slippage risk remains manageable.",
        "evaluation_research_analysis": "Historical validation is constructive but still sample-aware.",
        "why_this_signal": {
            "top_positive_drivers": [{"label": "fundamental_reality", "detail": "cash generation and margins remain strong"}],
            "top_negative_drivers": [{"label": "execution", "detail": "confirmation discipline still matters"}],
        },
        "evaluation": {
            "calibration_summary": {
                "confidence_reliability_score": 71.0,
                "calibration_health_status": "healthy",
            },
            "signal_scorecard": {"final_signal_overall": {"hit_rate": 0.62}},
            "ranking_scorecard": {"deployable_alpha_utility": {"monotonicity": "higher_confidence_buckets_outperform"}},
        },
        "deployment_readiness": {
            "model_readiness": {
                "model_readiness_status": "ready",
                "live_readiness_score": 74.0,
                "evaluation_consistency": {"consistency_score": 72.0},
            },
            "deployment_permission": {
                "deployment_permission": "limited_live_eligible",
                "trust_tier": "controlled_live",
            },
            "risk_budgeting": {"risk_budget_tier": "controlled"},
        },
        "canonical_validation": {
            "walkforward_summary": {"window_count": 4},
            "net_return_summary": {"average_edge_return": 0.018},
            "readiness_scorecard": {"paper_vs_live_candidate_quality_summary": 0.03},
            "suppression_effect_summary": {"suppression_effect_edge_spread": 0.015},
        },
        "operational_guardrails": {
            "drift_monitoring": {"model_drift_score": 18.0},
            "system_health": {"system_health_status": "healthy"},
            "control_state": {"current_operating_mode": "limited_live", "pause_required": False},
        },
        "source_governance": {
            "commercialization_readiness": {
                "buyer_demo_suitability": "cleaner_candidate",
                "commercialization_risk_score": 24.0,
            }
        },
        "portfolio_construction": {
            "current_candidate": {
                "portfolio_candidate_score": 69.0,
                "portfolio_fit_quality": 63.0,
                "execution_quality_score": 74.0,
                "size_band": "medium",
            }
        },
    }
    axiom = build_axiom_artifact(
        normalized_bundle=normalized_bundle,
        job_context={
            "symbol": "NVDA",
            "as_of_date": "2024-01-02",
            "audience_type": audience_type,
            "report_profile": report_profile,
        },
        feature_factor_bundle=feature_factor_bundle,
        strategy_bundle=strategy_bundle,
        report_context=report_context,
    )
    axiom = {
        **axiom,
        "historical_evidence": {
            "status": "available",
            "history_horizon_label": "21d",
        },
        "calibration_summary": {
            "status": "available",
            "horizon_label": "21d",
            "matured_count": 18,
            "dau_spread": 0.051,
            "dau_bucket_summary": {
                "buckets": [
                    {"bucket_label": "80-100", "average_net_edge_return": 0.034},
                    {"bucket_label": "0-20", "average_net_edge_return": -0.017},
                ]
            },
            "regime_outcome_summary": [{"regime_label": "fundamental_convergence", "average_net_edge_return": 0.028}],
            "deployability_tier_outcome_summary": [{"deployability_tier": "live_candidate", "average_net_edge_return": 0.03}],
        },
        "portfolio_governance": {
            "symbol": "NVDA",
            "portfolio_rank_score": 67.0,
            "portfolio_fit_label": "core_candidate",
            "final_size_band": "medium",
            "overlap_penalty": 18.0,
            "fragility_penalty": 15.0,
            "liquidity_penalty": 8.0,
            "research_penalty": 7.0,
        },
        "evidence_backed_deployability": {
            "deployability_tier": "live_candidate",
            "size_band": "medium",
            "evidence_summary": "Historical evidence, liquidity integrity, and research integrity support controlled live deployment.",
        },
    }
    workspace = build_axiom_workspace_profile(
        {
            "audience_type": audience_type,
            "report_profile": report_profile,
            "workspace_name": "Institutional Demo Desk",
        }
    )
    engine_input = build_axiom_engine_input(
        normalized_bundle,
        job_context={"symbol": "NVDA", "as_of_date": "2024-01-02"},
        feature_factor_bundle=feature_factor_bundle,
        strategy_bundle=strategy_bundle,
        report_context=report_context,
    ).model_dump(mode="python")
    lineage = build_axiom_lineage(
        engine_input=engine_input,
        axiom_artifact=axiom,
        workspace_profile=workspace,
    )
    pack = build_axiom_institutional_report_pack(
        axiom_artifact=axiom,
        report_context={
            **report_context,
            "axiom_history_record": {
                "forward_outcomes": {
                    "21d": {
                        "matured": True,
                        "net_edge_return": 0.034,
                        "mae": 0.021,
                        "mfe": 0.087,
                    }
                }
            },
            "axiom_calibration_status": "available",
            "axiom_portfolio_governance_summary": "Portfolio fit remains core-candidate quality with moderate size guidance.",
            "axiom_final_size_band": "medium",
        },
        workspace_profile=workspace,
        lineage=lineage,
    )
    return axiom, report_context, workspace, lineage, pack


def test_axiom_phase4_builds_institutional_report_pack() -> None:
    axiom, _report_context, workspace, lineage, pack = _phase4_fixture()

    assert pack["reporting_version"] == "axiom50_phase4_reporting_v1"
    assert pack["workspace_profile"]["audience_type"] == workspace["audience_type"]
    assert pack["summary_card"]["symbol"] == "NVDA"
    assert pack["summary_card"]["deployable_alpha_utility"] == axiom["deployable_alpha_utility"]
    assert pack["institutional_one_pager"]["executive_summary"]
    assert pack["ic_memo"]["recommended_action"]["tier"] == "live_candidate"
    assert pack["risk_deployability_memo"]["fragility_engine"]["score"] is not None
    assert pack["historical_evidence_summary"]["matured_count"] == 18
    assert pack["lineage_summary"]["lineage_summary"] == lineage["lineage_summary"]


def test_axiom_phase4_lineage_blocks_capture_provenance_types() -> None:
    _axiom, _report_context, _workspace, lineage, _pack = _phase4_fixture(
        audience_type="family_office",
        report_profile="portfolio_review",
    )

    critical_fragility = lineage["engine_lineage"]["critical_fragility"]
    gap_block = next(
        block
        for block in critical_fragility["blocks"]
        if block["component"] == "gap_jump_risk_component"
    )
    research_block = next(
        block
        for block in lineage["engine_lineage"]["research_integrity"]["blocks"]
        if block["component"] == "out_of_sample_reliability_component"
    )

    assert gap_block["coverage_status"] in {"available", "partial"}
    assert "market_price_volume.gap_instability_10d" in gap_block["derived_from"]
    assert gap_block["evidence_type"] in {"direct_source", "partial_proxy"}
    assert research_block["evidence_type"] == "historical_replay_estimate"
    assert lineage["weakest_evidence_areas"] is not None
    assert lineage["lineage_summary"]


def test_axiom_phase4_workspace_profile_adapts_by_audience() -> None:
    hedge_fund_profile = build_axiom_workspace_profile({"audience_type": "hedge_fund"})
    family_office_profile = build_axiom_workspace_profile({"audience_type": "family_office"})

    assert hedge_fund_profile["report_profile"] == "trading_focused"
    assert "deployability" in hedge_fund_profile["emphasis_domains"]
    assert family_office_profile["workflow_profile"] == "capital_preservation_committee"
    assert "capital_preservation" in family_office_profile["emphasis_domains"]
