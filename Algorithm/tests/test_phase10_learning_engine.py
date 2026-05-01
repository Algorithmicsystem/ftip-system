from api.assistant import reports
from api.assistant.phase10 import (
    CONTINUOUS_LEARNING_ARTIFACT_KIND,
    CONTINUOUS_LEARNING_VERSION,
    build_continuous_learning_artifact,
)
from api.assistant.storage import AssistantStorage


def _build_report(
    symbol: str,
    *,
    regime_label: str = "trend",
    final_signal: str = "HOLD",
    strategy_posture: str = "watchlist_positive",
    conviction_tier: str = "moderate",
    confidence_score: float = 56.0,
    actionability_score: float = 45.0,
    deployment_permission: str = "paper_shadow_only",
    trust_tier: str = "paper_only",
    live_readiness_score: float = 57.0,
    portfolio_candidate_score: float = 60.0,
    portfolio_fit_quality: float = 49.0,
    opportunity_quality: float = 63.0,
    cross_domain_conviction: float = 65.0,
    signal_fragility: float = 35.0,
    regime_stability: float = 58.0,
    narrative_crowding: float = 62.0,
    macro_alignment: float = 56.0,
    fundamental_durability: float = 61.0,
    evaluation_reliability: float = 55.0,
    evaluation_hit_rate: float = 0.55,
) -> dict:
    proprietary_scores = {
        "Market Structure Integrity Score": {"score": max(0.0, opportunity_quality - 3.0)},
        "Regime Stability Score": {"score": regime_stability},
        "Signal Fragility Index": {"score": signal_fragility},
        "Narrative Crowding Index": {"score": narrative_crowding},
        "Fundamental Durability Score": {"score": fundamental_durability},
        "Macro Alignment Score": {"score": macro_alignment},
        "Cross-Domain Conviction Score": {"score": cross_domain_conviction},
        "Opportunity Quality Score": {"score": opportunity_quality},
    }
    report = reports.build_analysis_report(
        symbol=symbol,
        as_of_date="2024-01-02",
        horizon="swing",
        risk_mode="balanced",
        signal={
            "action": final_signal,
            "score": 0.64,
            "confidence": 0.58,
            "horizon_days": 21,
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend remains constructive."},
        },
        key_features={
            "ret_21d": 0.08,
            "vol_21d": 0.24,
            "atr_pct": 0.035,
            "regime_label": regime_label,
        },
        quality={
            "bars_ok": True,
            "fundamentals_ok": True,
            "sentiment_ok": True,
            "news_ok": True,
            "warnings": [],
        },
        evidence={
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend remains constructive."},
            "sources": ["market_bars_daily", "news_raw"],
        },
        data_bundle={
            "market_price_volume": {
                "ret_21d": 0.08,
                "atr_pct": 0.035,
                "realized_vol_21d": 0.24,
            },
            "sentiment_narrative_flow": {
                "top_narratives": [{"topic": "ai_infrastructure"}],
            },
            "relative_context": {
                "sector": "Technology",
                "benchmark_proxy": "QQQ",
            },
            "macro_cross_asset": {
                "benchmark_proxy": "QQQ",
            },
        },
        feature_factor_bundle={
            "proprietary_scores": proprietary_scores,
            "composite_intelligence": {
                name: payload["score"] for name, payload in proprietary_scores.items()
            },
            "regime_intelligence": {"regime_label": regime_label},
        },
        strategy={
            "strategy_version": "phase4_institutional_v1",
            "final_signal": final_signal,
            "strategy_posture": strategy_posture,
            "confidence": confidence_score / 100.0,
            "confidence_score": confidence_score,
            "conviction_tier": conviction_tier,
            "actionability_score": actionability_score,
            "participant_fit": ["swing trader"],
            "primary_participant_fit": "swing trader",
            "scenario_matrix": {"base": {"summary": "Constructive base case."}},
            "execution_posture": {
                "preferred_posture": "wait_for_confirmation",
                "urgency_level": "measured",
                "patience_level": "high",
                "signal_cleanliness": "mixed_clean",
                "entry_quality_proxy": 55.0,
                "risk_context_summary": "Execution remains measured.",
            },
        },
    )
    report = reports.attach_deployment_context(
        report,
        {
            "deployment_readiness_version": "phase8_capital_readiness_v1",
            "deployment_mode": {
                "active_mode": "paper_shadow",
                "rollout_stage": "forward_shadow_validation",
            },
            "model_readiness": {
                "model_readiness_status": "constrained",
                "live_readiness_score": live_readiness_score,
                "live_readiness_blockers": [
                    "confidence calibration quality is not strong enough for live escalation"
                ],
                "recent_degradation_flags": ["confidence reliability is still building"],
            },
            "signal_admission_control": {
                "admitted_for_strategy": True,
                "admitted_for_paper": True,
                "admitted_for_live": False,
            },
            "deployment_permission": {
                "deployment_permission": deployment_permission,
                "deployment_blockers": ["fragility remains too high for live admission"],
                "deployment_rationale": "The setup remains paper-only while reliability improves.",
                "trust_tier": trust_tier,
                "minimum_required_review": "analyst_review",
                "human_review_required": True,
            },
            "risk_budgeting": {
                "risk_budget_tier": "shadow_only",
                "exposure_caution_level": "high",
                "fragility_adjusted_size_band": "0.10x-0.25x pilot unit",
                "confidence_adjusted_size_band": "0.10x-0.25x pilot unit",
                "maximum_risk_mode_allowed": "paper_shadow",
            },
            "rollout_workflow": {
                "rollout_stage": "forward_shadow_validation",
                "readiness_checkpoint": "watch",
                "promotion_criteria": ["confidence reliability remains above the stage threshold"],
                "demotion_criteria": ["fragility rises into the blocked zone"],
                "stage_transition_notes": ["Continue paper/shadow evidence collection before live escalation."],
            },
            "drift_monitor": {
                "pause_recommended": False,
                "degrade_to_paper_recommended": False,
                "drift_alerts": ["confidence reliability is below the live-support comfort zone"],
                "deployment_risk_alerts": [],
            },
            "audit_snapshot": {
                "rationale_summary": "Paper-shadow only until calibration improves.",
            },
        },
        readiness_artifact_id=f"readiness-{symbol}",
        deployment_audit_artifact_id=f"audit-{symbol}",
    )
    report = reports.attach_portfolio_context(
        report,
        {
            "portfolio_construction_version": "phase9_portfolio_construction_v1",
            "current_candidate": {
                "symbol": symbol,
                "candidate_classification": "watchlist_candidate",
                "ranked_opportunity_score": opportunity_quality,
                "portfolio_candidate_score": portfolio_candidate_score,
                "watchlist_priority_score": portfolio_candidate_score + 3.0,
                "deployability_rank": max(0.0, live_readiness_score - 5.0),
                "portfolio_rank": 2,
                "portfolio_fit_quality": portfolio_fit_quality,
                "overlap_score": 72.0,
                "redundancy_score": 77.0,
                "diversification_contribution_score": 33.0,
                "most_redundant_symbol": "AAPL",
                "size_band": "paper / shadow band",
                "weight_band": "0.00x live weight",
                "risk_budget_band": "shadow_risk_band",
                "execution_quality_score": 54.0,
                "friction_penalty": 36.0,
                "turnover_penalty": 43.0,
                "wait_for_better_entry_flag": True,
                "confirmation_preferred_flag": True,
                "candidate_blockers": ["the idea is redundant with existing tracked exposures"],
            },
            "workflow": {
                "prioritized_watchlist": ["AAPL", symbol],
                "active_portfolio_candidates": ["AAPL"],
                "priority_shift_flag": True,
                "rotation_pressure_score": 68.0,
            },
            "portfolio_context_summary": "The setup remains a watchlist candidate because redundancy and readiness still cap portfolio use.",
            "portfolio_fit_analysis": "Overlap is elevated versus the leading peer, so diversification contribution is modest.",
            "execution_quality_analysis": "Execution remains measured and confirmation is still preferred.",
            "portfolio_workflow_summary": "The current idea remains on the watchlist while cleaner peers stay ahead.",
        },
        portfolio_construction_artifact_id=f"portfolio-{symbol}",
    )
    report["evaluation"] = {
        "status": "available",
        "evaluation_version": "phase6_eval_v1",
        "calibration_summary": {
            "confidence_reliability_score": evaluation_reliability,
            "confidence_monotonicity": "higher_confidence_buckets_outperform",
            "bucketed_confidence_stats": [{"matured_count": 5}, {"matured_count": 4}],
            "calibration_drift_notes": ["trend cohort reliability remains below the cross-sectional median"],
        },
        "signal_scorecard": {
            "final_signal_overall": {"hit_rate": evaluation_hit_rate},
        },
        "factor_attribution_summary": {
            "proprietary_score_attribution": [
                {
                    "score_name": "Narrative Crowding Index",
                    "favorable_vs_unfavorable_return_spread": 0.052,
                    "monotonicity": "higher_buckets_outperform",
                    "bucket_results": [{"matured_count": 5}, {"matured_count": 4}],
                },
                {
                    "score_name": "Opportunity Quality Score",
                    "favorable_vs_unfavorable_return_spread": 0.061,
                    "monotonicity": "higher_buckets_outperform",
                    "bucket_results": [{"matured_count": 6}, {"matured_count": 4}],
                },
            ],
            "strategy_component_attribution": [
                {
                    "score_name": "trend_following",
                    "favorable_vs_unfavorable_return_spread": 0.043,
                    "bucket_results": [{"matured_count": 5}, {"matured_count": 4}],
                }
            ],
        },
        "weakest_conditions": [{"dimension": "regime_label", "label": regime_label}],
    }
    report["evaluation_summary"] = "Evaluation context is available."
    report["confidence_reliability_summary"] = "Confidence reliability remains constrained in the active regime."
    report["regime_usefulness_summary"] = "Trend reliability trails the best historical regimes in this cohort."
    return report


def test_learning_engine_builds_research_compounding_artifact() -> None:
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    current_report = _build_report("NVDA", regime_label="trend")
    current_report_id = store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, current_report)
    store.save_artifact(
        session_id,
        reports.ANALYSIS_REPORT_KIND,
        _build_report(
            "AAPL",
            regime_label="trend",
            opportunity_quality=71.0,
            cross_domain_conviction=70.0,
            narrative_crowding=45.0,
            evaluation_reliability=68.0,
            evaluation_hit_rate=0.61,
        ),
    )
    store.save_artifact(
        session_id,
        reports.ANALYSIS_REPORT_KIND,
        _build_report(
            "XOM",
            regime_label="choppy",
            opportunity_quality=59.0,
            cross_domain_conviction=57.0,
            narrative_crowding=31.0,
            evaluation_reliability=73.0,
            evaluation_hit_rate=0.64,
        ),
    )
    store.save_artifact(
        session_id,
        CONTINUOUS_LEARNING_ARTIFACT_KIND,
        {
            "cohort_summary": {"horizon": "swing", "risk_mode": "balanced"},
            "experiment_registry": {
                "open_experiments": [
                    {"title": "Reweight Narrative Crowding Index", "validation_status": "proposed"}
                ],
                "approved_improvements": [],
                "rejected_improvements": [],
            },
        },
    )

    artifact = build_continuous_learning_artifact(
        current_report=current_report,
        current_report_id=current_report_id,
        session_id=session_id,
        store=store,
    )

    assert artifact["continuous_learning_version"] == CONTINUOUS_LEARNING_VERSION
    assert artifact["cohort_summary"]["tracked_reports"] >= 3
    assert artifact["active_setup_archetype"]["archetype_name"]
    assert artifact["signal_family_library"]["archetype_cohorts"]
    assert artifact["regime_conditioned_learnings"]
    assert artifact["feature_interaction_candidates"]
    assert artifact["reweighting_candidates"]
    assert artifact["research_hypotheses"]
    assert artifact["drift_alerts"]
    assert artifact["experiment_registry"]["open_experiments"]
    assert artifact["experiment_registry"]["open_experiments"][0]["validation_status"] in {
        "proposed",
        "under_review",
    }
    assert artifact["motif_discovery"]["active_motifs"] or artifact["motif_discovery"]["motif_library"]
    assert artifact["improvement_queue"]
    assert artifact["learning_summary"]
    assert artifact["regime_learning_summary"]
    assert artifact["adaptation_queue_summary"]
    assert artifact["experiment_registry_summary"]
    assert artifact["archetype_motif_summary"]
    assert any(
        item["title"] == "Reweight Narrative Crowding Index"
        and item["validation_status"] == "under_review"
        for item in artifact["experiment_registry"]["open_experiments"]
    )
