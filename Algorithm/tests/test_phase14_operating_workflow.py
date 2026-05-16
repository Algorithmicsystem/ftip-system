from __future__ import annotations

import datetime as dt

from api.assistant import reports
from api.assistant.phase5.context import build_narrator_context
from api.assistant.phase5.routing import route_question
from api.assistant.phase6 import PREDICTION_RECORD_KIND
from api.assistant.phase12 import OPERATIONAL_INCIDENT_ARTIFACT_KIND, SHADOW_DECISION_RECORD_KIND
from api.assistant.phase14 import (
    OPERATING_WORKFLOW_VERSION,
    build_operating_workflow_artifact,
)
from api.assistant.phase14.postmortem import classify_failure_mode
from api.assistant.storage import AssistantStorage


def _prediction(
    symbol: str,
    *,
    generated_at: str,
    as_of_date: str,
    edge_return: float,
    permission: str,
    candidate_classification: str,
) -> dict:
    return {
        "generated_at": generated_at,
        "symbol": symbol,
        "as_of_date": as_of_date,
        "horizon": "swing",
        "risk_mode": "balanced",
        "horizon_days": 21,
        "final_signal": "BUY",
        "signal_action": "BUY",
        "confidence_score": 64.0 if edge_return >= 0 else 42.0,
        "actionability_score": 66.0 if edge_return >= 0 else 31.0,
        "deployment_permission": permission,
        "candidate_classification": candidate_classification,
        "suppression_flags": [] if edge_return >= 0 else ["market_stress"],
        "proprietary_scores": {
            "Opportunity Quality Score": 71.0 if edge_return >= 0 else 43.0,
            "Cross-Domain Conviction Score": 68.0 if edge_return >= 0 else 39.0,
            "Signal Fragility Index": 32.0 if edge_return >= 0 else 69.0,
        },
        "feature_vector": {
            "event_risk_classification": "low_event_risk" if edge_return >= 0 else "event_distorted",
            "tradability_state": "clean_liquid_setup" if edge_return >= 0 else "tradable_with_caution",
            "breadth_state": "broad_healthy_participation" if edge_return >= 0 else "narrow_leadership",
            "cross_asset_conflict_score": 24.0 if edge_return >= 0 else 68.0,
            "market_stress_score": 28.0 if edge_return >= 0 else 65.0,
            "implementation_fragility_score": 31.0 if edge_return >= 0 else 67.0,
            "signal_regime_label": "trend" if edge_return >= 0 else "transition",
        },
        "slices": {
            "regime_label": "trend" if edge_return >= 0 else "transition",
            "event_risk_state": "low_event_risk" if edge_return >= 0 else "event_distorted",
            "liquidity_state": "clean_liquid_setup" if edge_return >= 0 else "tradable_with_caution",
            "breadth_state": "broad_healthy_participation" if edge_return >= 0 else "narrow_leadership",
            "cross_asset_state": "supportive" if edge_return >= 0 else "conflicted",
            "stress_state": "stable" if edge_return >= 0 else "unstable",
            "fragility_tier": "contained" if edge_return >= 0 else "fragile",
        },
        "outcome": {
            "outcome_status": "matured",
            "matured": True,
            "gross_edge_return": edge_return,
            "net_edge_return": edge_return - 0.003,
            "gross_trade_return": edge_return,
            "net_trade_return": edge_return - 0.003,
            "estimated_cost_bps": 30.0,
            "mae": 0.018 if edge_return >= 0 else 0.074,
            "mfe": 0.071 if edge_return >= 0 else 0.012,
            "invalidation_triggered": edge_return < 0,
            "signal_half_life_days": 10 if edge_return >= 0 else 4,
            "continuation_decay_score": 22.0 if edge_return >= 0 else 64.0,
            "friction_cost_summary": {"cost_rate": 0.003, "total_bps": 30.0},
        },
    }


def _base_report() -> dict:
    report = reports.build_analysis_report(
        symbol="NVDA",
        as_of_date="2024-01-19",
        horizon="swing",
        risk_mode="balanced",
        signal={
            "action": "BUY",
            "score": 0.64,
            "confidence": 0.58,
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend remains constructive."},
        },
        key_features={"ret_21d": 0.07, "vol_21d": 0.24, "regime_label": "trend"},
        quality={"bars_ok": True, "news_ok": True, "sentiment_ok": True, "warnings": []},
        evidence={
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend remains constructive."},
            "sources": ["market_bars_daily", "news_raw", "sentiment_daily"],
        },
        data_bundle={
            "event_catalyst_risk": {"event_risk_classification": "event_distorted"},
            "liquidity_execution_fragility": {"tradability_state": "tradable_with_caution"},
        },
        strategy={
            "strategy_version": "phase4_institutional_v1",
            "final_signal": "HOLD",
            "strategy_posture": "watchlist_positive",
            "confidence": 0.56,
            "confidence_score": 56.0,
            "conviction_tier": "moderate",
            "actionability_score": 44.0,
            "primary_participant_fit": "swing trader",
        },
    )
    report = reports.attach_canonical_validation_context(
        report,
        {
            "validation_version": "phase10_research_truth_v1",
            "status": "available",
            "prediction_linkage_summary": {"total_predictions": 18, "matured_count": 14},
            "walkforward_summary": {"window_count": 2},
            "net_return_summary": {"average_edge_return": 0.012, "hit_rate": 0.59},
            "friction_cost_summary": {"average_cost_drag": 0.003},
            "readiness_scorecard": {"paper_vs_live_candidate_quality_summary": 0.03},
            "suppression_effect_summary": {"suppression_effect_edge_spread": 0.02},
            "failure_modes": ["event_distorted setups remain weakest"],
            "validation_summary": "Canonical validation remains available.",
            "walkforward_validation_summary": "Walk-forward validation remains constructive.",
            "net_of_friction_summary": "Net-of-friction edge remains positive.",
            "suppression_readiness_validation_summary": "Suppression and readiness remain directionally helpful.",
            "drawdown_invalidation_summary": "Drawdown and invalidation behavior remain watchable.",
        },
        canonical_validation_artifact_id="validation-14",
    )
    report = reports.attach_deployment_context(
        report,
        {
            "deployment_readiness_version": "phase8_capital_readiness_v1",
            "deployment_mode": {"active_mode": "paper_shadow", "rollout_stage": "forward_shadow_validation"},
            "model_readiness": {
                "model_readiness_status": "constrained",
                "live_readiness_score": 54.0,
                "live_readiness_blockers": ["confidence calibration quality is not strong enough"],
            },
            "signal_admission_control": {
                "admitted_for_strategy": True,
                "admitted_for_paper": True,
                "admitted_for_live": False,
            },
            "deployment_permission": {
                "deployment_permission": "paper_shadow_only",
                "deployment_blockers": ["event distortion and calibration still block live escalation"],
                "deployment_rationale": "Keep the setup in paper-shadow while confirmation and calibration improve.",
                "trust_tier": "paper_only",
                "minimum_required_review": "senior_analyst_review",
                "human_review_required": True,
            },
            "risk_budgeting": {"risk_budget_tier": "shadow_only"},
            "rollout_workflow": {
                "rollout_stage": "forward_shadow_validation",
                "readiness_checkpoint": "watch",
                "promotion_criteria": ["shadow reliability remains above threshold"],
                "demotion_criteria": ["drift or event fragility rises materially"],
            },
            "drift_monitor": {
                "pause_recommended": False,
                "degrade_to_paper_recommended": True,
                "drift_alerts": ["confidence reliability remains below the live comfort zone"],
                "deployment_risk_alerts": ["event distortion is still suppressing cleaner actionability"],
            },
            "audit_snapshot": {"rationale_summary": "Paper-shadow only while event and calibration risk remain elevated."},
        },
        readiness_artifact_id="readiness-14",
        deployment_audit_artifact_id="audit-14",
    )
    report["candidate_classification"] = "watchlist_candidate"
    report["ranked_opportunity_score"] = 62.0
    report["portfolio_candidate_score"] = 60.0
    report["marginal_portfolio_utility"] = 58.0
    report["portfolio_fit_quality"] = 47.0
    report["size_band"] = "paper / shadow band"
    report["candidate_blockers"] = ["portfolio overlap remains elevated"]
    report["portfolio_workflow_summary"] = "The name remains a watchlist-positive candidate because overlap and event risk still cap priority."
    report["learning_priority"] = "confidence_and_event_discipline"
    report["improvement_queue"] = [
        {"title": "Tighten event-distortion penalty", "priority": "high"},
        {"title": "Recalibrate confidence buckets", "priority": "high"},
    ]
    report["setup_archetype"] = {"archetype_name": "Watchlist Only Thesis"}
    report = reports.attach_operational_context(
        report,
        {
            "operational_guardrails_version": "phase12_operational_guardrails_v1",
            "system_health": {
                "system_health_status": "degraded",
                "provider_health_status": "watch",
                "data_pipeline_health": "watch",
                "artifact_pipeline_health": "healthy",
                "provider_degradation_notes": ["news freshness is lagging"],
                "degraded_domain_list": ["news", "event"],
                "data_reliability_score": 49.0,
            },
            "shadow_mode": {
                "shadow_mode_status": "active_shadow",
                "shadow_vs_realized_summary": "Shadow tracking is active with mixed but improving results.",
                "shadow_reliability_summary": "Shadow reliability reads 58 / 100.",
                "shadow_promotion_candidate": False,
                "shadow_demotion_reason": "The setup remains paper-shadow only until event and calibration noise fade.",
                "shadow_cohort": {"tracked_shadow_decisions": 9},
            },
            "drift_monitoring": {
                "model_drift_score": 43.0,
                "environment_shift_score": 52.0,
                "calibration_health_status": "watch",
                "confidence_reliability_alert": "confidence reliability is below the promotion threshold",
                "readiness_gate_reliability_alert": "readiness gate reliability still needs more sample depth",
                "monotonicity_break_alert": None,
            },
            "control_state": {
                "current_operating_mode": "shadow_only",
                "pause_required": False,
                "downgrade_to_shadow_recommended": True,
                "downgrade_reason": "Event distortion and calibration noise remain too high for stronger trust.",
                "recovery_criteria": ["event distortion fades", "confidence reliability recovers"],
                "operator_attention_required": True,
            },
            "operational_alerts": [
                {"alert_summary": "Event distortion is suppressing cleaner deployment support."}
            ],
            "incident_history": [
                {"summary": "A recent event-driven false positive required shadow-only handling."}
            ],
            "system_health_summary": "System health is degraded and should stay in shadow-only interpretation.",
            "shadow_mode_summary": "Shadow mode remains active with mixed but improving reliability.",
            "drift_control_summary": "Drift remains manageable but confidence calibration is still a watch item.",
            "incident_history_summary": "Recent incident history is centered on event-driven false positives.",
            "health_snapshot": {},
            "shadow_decision_record": {
                "recorded_at": "2024-01-19T12:00:00Z",
                "symbol": "NVDA",
                "signal": "HOLD",
                "deployment_permission": "paper_shadow_only",
                "trust_tier": "paper_only",
                "candidate_classification": "watchlist_candidate",
                "blockers": ["event distortion and calibration still block live escalation"],
            },
        },
        operational_guardrails_artifact_id="ops-14",
        health_snapshot_artifact_id="health-14",
        shadow_decision_artifact_id="shadow-14",
        operational_incident_artifact_ids=["incident-14"],
    )
    return reports.attach_source_governance_context(
        report,
        {
            "source_governance_version": "phase13_source_governance_v1",
            "source_profile": "buyer_demo",
            "commercialization_readiness": {
                "buyer_safe_profile_status": "conditional_review_required",
                "buyer_demo_suitability": "conditional_review_required",
                "commercialization_risk_score": 58.0,
                "licensing_risk_tier": "elevated",
                "commercial_blockers": ["review-required news dependencies remain active"],
                "commercial_cleanup_queue": [
                    {"source_name": "google_news_rss", "priority": "high", "cleanup_reason": "internal-only news fallback remains in the stack"}
                ],
                "disallowed_sources": ["google_news_rss"],
                "gated_domains": [{"domain": "sentiment_narrative_flow"}],
                "degraded_due_to_profile": ["sentiment_narrative_flow"],
            },
            "commercialization_readiness_summary": "Buyer-demo profile remains conditionally usable but still has review-required news dependencies.",
            "source_governance_summary": "Source governance is clean enough for demo use, but not yet fully clean-room ready.",
            "buyer_diligence_summary": "Replace internal-only news fallbacks before a stricter external deployment profile is used.",
        },
        source_governance_artifact_id="source-14",
    )


def test_phase14_operating_workflow_builds_review_loops_and_journal() -> None:
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    now = dt.datetime.now(dt.timezone.utc)
    report = _base_report()

    prior_report = {**report, "symbol": "NVDA", "strategy": {**(report.get("strategy") or {}), "final_signal": "BUY", "strategy_posture": "trend_continuation_candidate"}, "deployment_permission": "limited_live_eligible", "trust_tier": "elevated", "current_operating_mode": "normal", "candidate_classification": "secondary_candidate"}
    peer_report = {**report, "symbol": "AAPL", "deployment_permission": "limited_live_eligible", "trust_tier": "elevated", "candidate_classification": "top_priority_candidate", "marginal_portfolio_utility": 76.0, "portfolio_candidate_score": 79.0}
    store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, prior_report)
    store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, peer_report)

    for idx, edge in enumerate([0.026, 0.018, -0.022, 0.011, -0.014, 0.021], start=1):
        store.save_artifact(
            session_id,
            PREDICTION_RECORD_KIND,
            _prediction(
                "NVDA" if idx % 2 else "AAPL",
                generated_at=(now - dt.timedelta(days=min(idx, 6))).isoformat(),
                as_of_date=f"2024-01-{10 + idx:02d}",
                edge_return=edge,
                permission="limited_live_eligible" if edge > 0 else "paper_shadow_only",
                candidate_classification="top_priority_candidate" if edge > 0 else "watchlist_candidate",
            ),
        )
    store.save_artifact(
        session_id,
        SHADOW_DECISION_RECORD_KIND,
        {
            "recorded_at": (now - dt.timedelta(days=2)).isoformat(),
            "symbol": "NVDA",
            "deployment_permission": "paper_shadow_only",
            "trust_tier": "paper_only",
            "candidate_classification": "watchlist_candidate",
            "signal": "HOLD",
            "blockers": ["event distortion remains elevated"],
        },
    )
    store.save_artifact(
        session_id,
        OPERATIONAL_INCIDENT_ARTIFACT_KIND,
        {
            "recorded_at": (now - dt.timedelta(days=1)).isoformat(),
            "summary": "A recent event-driven false positive remained trapped in shadow mode.",
            "alert_summary": "Event distortion remains the top operational incident.",
        },
    )

    artifact = build_operating_workflow_artifact(
        current_report=report,
        current_report_id="report-14",
        session_id=session_id,
        store=store,
    )

    assert artifact["operating_workflow_version"] == OPERATING_WORKFLOW_VERSION
    assert artifact["todays_candidate_triage"]
    assert artifact["changed_signals"]
    assert artifact["weekly_operating_review"]["review_window_days"] == 7
    assert artifact["monthly_refinement_review"]["review_window_days"] == 30
    assert artifact["shadow_decision_journal"]["shadow_decision_count"] >= 1
    assert artifact["failure_mode_classification"] in {
        "event_distortion",
        "liquidity_gap_fragility",
        "hostile_macro_or_stress_context",
        "portfolio_overlap_misfit",
        "confidence_or_readiness_overstatement",
        "watchlist_only_thesis",
    }
    assert artifact["trust_recovery_checklist"]
    assert artifact["operator_runbook"]["daily_workflow"]
    assert artifact["operator_attention_items"]


def test_phase14_attaches_to_report_and_routes_operator_questions() -> None:
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    report = _base_report()
    store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, {**report, "strategy": {**(report.get("strategy") or {}), "final_signal": "BUY"}})

    artifact = build_operating_workflow_artifact(
        current_report=report,
        current_report_id="report-14",
        session_id=session_id,
        store=store,
    )
    enriched = reports.attach_operating_workflow_context(
        report,
        artifact,
        operating_workflow_artifact_id="ops-workflow-14",
    )

    assert enriched["operating_workflow_artifact_id"] == "ops-workflow-14"
    assert enriched["daily_operating_summary"]
    assert enriched["weekly_operating_summary"]
    assert enriched["monthly_operating_summary"]
    assert enriched["shadow_journal_summary"]
    assert enriched["postmortem_summary"]
    assert enriched["trust_maintenance_summary"]
    assert enriched["operator_runbook_summary"]

    route = route_question("What changed today and what should I review first this week?")
    assert route["intent"] == "operator_workflow"
    assert route["answer_mode"] == "operator"

    narrator_context = build_narrator_context(
        enriched,
        active_analysis=reports.build_active_analysis_reference(
            enriched,
            session_id=session_id,
            report_id="report-14",
        ),
        route=route,
        user_message="What changed today and what should I review first this week?",
    )
    assert narrator_context["operating_workflow_snapshot"]["daily_operating_summary"]
    assert narrator_context["section_summaries"]["operator_runbook_summary"]


def test_phase14_postmortem_classifier_flags_event_distortion() -> None:
    assert classify_failure_mode(_base_report()) == "event_distortion"
