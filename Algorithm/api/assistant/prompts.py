from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


BASE_SYSTEM_PROMPT = (
    "You are the FTIP narrator. You speak as the voice of the system's computed analysis artifacts, not as a generic assistant. "
    "Never provide personalized financial advice or tell the user what they personally should do. "
    "When a grounded analysis report is present, treat it as source of truth and explain the actual computed signal, drivers, risks, and strategy logic. "
    "Do not claim that you lack the analysis if the report is present. Use concise, careful language, cite the actual report sections when helpful, and distinguish system output from personal advice. "
    "When page, workspace, dossier, workflow, export, recommendation, or committee context is present, answer as a platform copilot anchored to that exact visible state. "
    "Lead with what matters now, what is differentiated about the AXIOM view, what evidence is strongest or weakest, and what operational or workflow constraint is currently binding. "
    "When provider-health, proof-cycle, recommendation-state, or committee context is present, treat those as live platform constraints rather than background color. "
    "When bounded workspace continuity is present, use it as recent operating context across pages without implying long-term memory beyond what the platform is actually providing."
)


def build_chat_messages(
    history: List[Dict[str, str]], user_message: str, context: Optional[Dict[str, Any]]
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
    messages.extend(history[-12:])

    if context:
        context_line = "Context:\n" + json.dumps(
            context,
            indent=2,
            default=str,
            sort_keys=True,
            ensure_ascii=False,
        )
        user_payload = f"{user_message}\n\n{context_line}"
    else:
        user_payload = user_message

    messages.append({"role": "user", "content": user_payload})
    return messages


def summarize_analysis_report(report: Dict[str, Any]) -> str:
    strategy = report.get("strategy") or {}
    event_overlay = ((report.get("data_bundle") or {}).get("event_catalyst_risk") or {})
    quality_provenance = ((report.get("data_bundle") or {}).get("quality_provenance") or {})
    premium_connector_summary = quality_provenance.get("premium_connector_summary") or {}
    return " ".join(
        [
            f"Analysis report for {report.get('symbol', '?')} as of {report.get('as_of_date', '?')}.",
            f"AXIOM: {report.get('axiom_summary', 'n/a')}.",
            f"Signal: {(report.get('signal') or {}).get('action', 'n/a')} -> {strategy.get('final_signal', 'n/a')}.",
            f"Score: {(report.get('signal') or {}).get('score', 'n/a')}.",
            f"Confidence: {strategy.get('confidence', (report.get('signal') or {}).get('confidence', 'n/a'))}.",
            f"Deployment permission: {report.get('deployment_permission', 'n/a')} under {report.get('deployment_mode', 'research_only')}.",
            f"Trust tier: {report.get('trust_tier', 'unknown')} with live readiness {report.get('live_readiness_score', 'n/a')}.",
            f"Portfolio classification: {report.get('candidate_classification', 'n/a')} with portfolio score {report.get('portfolio_candidate_score', 'n/a')}.",
            f"Portfolio risk model: {report.get('portfolio_risk_model_summary', 'n/a')}.",
            f"Learning archetype: {(report.get('setup_archetype') or {}).get('archetype_name', 'n/a')} with research priority {report.get('learning_priority', 'n/a')}.",
            f"Canonical validation: {report.get('canonical_validation_summary', 'n/a')}.",
            f"Operational guardrails: {report.get('system_health_summary', 'n/a')}.",
            f"Commercial readiness: {report.get('commercialization_readiness_summary', 'n/a')}.",
            f"Data-provider stack: {report.get('data_provider_quality_summary', 'n/a')}.",
            f"AXIOM historical evidence: {report.get('axiom_evidence_summary', 'n/a')}.",
            f"AXIOM calibration: {report.get('axiom_calibration_summary_text', 'n/a')}.",
            f"AXIOM portfolio governance: {report.get('axiom_portfolio_governance_summary', 'n/a')}.",
            f"AXIOM institutional memo: {report.get('axiom_ic_memo_summary', 'n/a')}.",
            f"AXIOM lineage: {report.get('axiom_lineage_summary', 'n/a')}.",
            f"AXIOM why now: {report.get('axiom_why_now_summary', 'n/a')}.",
            f"AXIOM unique mispricing: {report.get('axiom_unique_mispricing_summary', 'n/a')}.",
            f"AXIOM timing support / evidence readiness / path survivability: {report.get('axiom_timing_support', 'n/a')} / {report.get('axiom_evidence_readiness', 'n/a')} / {report.get('axiom_path_survivability', 'n/a')}.",
            f"AXIOM false-positive penalty / exceptionality: {report.get('axiom_false_positive_penalty', 'n/a')} / {report.get('axiom_exceptional_opportunity', 'n/a')}.",
            f"AXIOM event overhang / catalyst quality / estimate revision support: {report.get('axiom_event_overhang_support', 'n/a')} / {report.get('axiom_catalyst_quality', 'n/a')} / {report.get('axiom_estimate_revision_support', 'n/a')}.",
            f"AXIOM source-strength support / penalty / premium bonus / evidence recency: {report.get('axiom_source_strength_support', 'n/a')} / {report.get('axiom_source_strength_penalty', 'n/a')} / {report.get('axiom_premium_evidence_bonus', 'n/a')} / {report.get('axiom_evidence_recency_quality', 'n/a')}.",
            f"Event intelligence: risk {event_overlay.get('event_risk_classification', 'n/a')} with catalyst quality {event_overlay.get('catalyst_quality', 'n/a')}, filings change {event_overlay.get('filings_change_signal', 'n/a')}, estimate revision support {event_overlay.get('estimate_revision_support', 'n/a')}, and premium evidence bonus {event_overlay.get('premium_evidence_bonus', 'n/a')}.",
            f"Platform workflow: {report.get('platform_overview_summary', 'n/a')}.",
            f"Platform dossier: {report.get('platform_dossier_summary', 'n/a')}.",
            f"Platform controls: {report.get('platform_workflow_actions_summary', 'n/a')}.",
            f"Platform exports: {report.get('platform_export_summary', 'n/a')}.",
            f"Platform access: {report.get('platform_access_control_summary', 'n/a')}.",
            f"Platform dashboard: {report.get('platform_dashboard_summary', 'n/a')}.",
            f"Platform analytics: {report.get('platform_analytics_summary', 'n/a')}.",
            f"Platform demo readiness: {report.get('platform_demo_readiness_summary', 'n/a')}.",
            f"Platform bootstrap: {report.get('platform_bootstrap_summary_text', 'n/a')}.",
            f"Platform readiness report: {report.get('platform_readiness_report_summary', 'n/a')}.",
            f"Platform pilot package: {report.get('platform_pilot_package_summary', 'n/a')}.",
            f"Platform proof cycle: {report.get('platform_proof_cycle_summary', 'n/a')}.",
            f"Platform tracking: {report.get('platform_tracking_summary', 'n/a')}.",
            f"Platform outcomes: {report.get('platform_outcome_summary', 'n/a')}.",
            f"Platform calibration hardening: {report.get('platform_calibration_hardening_summary', 'n/a')}.",
            f"Platform drift: {report.get('platform_drift_summary_text', 'n/a')}.",
            f"Platform benchmarks: {report.get('platform_benchmark_summary', 'n/a')}.",
            f"Platform model credibility: {report.get('platform_model_credibility_summary', 'n/a')}.",
            f"Premium connector readiness: {premium_connector_summary.get('summary', 'n/a')}.",
            f"Operating workflow: {report.get('daily_operating_summary', 'n/a')} {report.get('weekly_operating_summary', '')} {report.get('monthly_operating_summary', '')}".strip(),
            f"Overall view: {report.get('overall_analysis', '')}",
        ]
    )


def _grounding_block(report: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
    data_bundle = report.get("data_bundle") or {}
    quality_provenance = data_bundle.get("quality_provenance") or {}
    machine_context = {
        "symbol": report.get("symbol"),
        "as_of_date": report.get("as_of_date"),
        "horizon": report.get("horizon"),
        "risk_mode": report.get("risk_mode"),
        "scenario": report.get("scenario"),
        "analysis_depth": report.get("analysis_depth"),
        "refresh_mode": report.get("refresh_mode"),
        "market_regime": report.get("market_regime"),
        "freshness_summary": report.get("freshness_summary"),
        "signal": report.get("signal"),
        "strategy": report.get("strategy"),
        "why_this_signal": report.get("why_this_signal"),
        "key_features": report.get("key_features"),
        "quality": report.get("quality"),
        "evidence": report.get("evidence"),
        "evidence_map": report.get("evidence_map"),
        "deployment_readiness": report.get("deployment_readiness"),
        "portfolio_construction": report.get("portfolio_construction"),
        "portfolio_risk_model": report.get("portfolio_risk_model"),
        "continuous_learning": report.get("continuous_learning"),
        "operational_guardrails": report.get("operational_guardrails"),
        "source_governance": report.get("source_governance"),
        "event_intelligence": data_bundle.get("event_catalyst_risk"),
        "provider_quality_provenance": quality_provenance,
        "premium_connector_summary": quality_provenance.get(
            "premium_connector_summary"
        ),
        "axiom": report.get("axiom"),
        "axiom_proprietary_synthesis": report.get("axiom_proprietary_synthesis"),
        "axiom_support_vs_drag_summary": report.get("axiom_support_vs_drag_summary"),
        "axiom_why_now_summary": report.get("axiom_why_now_summary"),
        "axiom_unique_mispricing_summary": report.get(
            "axiom_unique_mispricing_summary"
        ),
        "axiom_event_overhang_support": report.get("axiom_event_overhang_support"),
        "axiom_filings_change_signal": report.get("axiom_filings_change_signal"),
        "axiom_catalyst_quality": report.get("axiom_catalyst_quality"),
        "axiom_estimate_revision_support": report.get(
            "axiom_estimate_revision_support"
        ),
        "axiom_source_strength_support": report.get(
            "axiom_source_strength_support"
        ),
        "axiom_source_strength_penalty": report.get(
            "axiom_source_strength_penalty"
        ),
        "axiom_premium_evidence_bonus": report.get("axiom_premium_evidence_bonus"),
        "axiom_evidence_recency_quality": report.get(
            "axiom_evidence_recency_quality"
        ),
        "axiom_setup_character_summary": report.get("axiom_setup_character_summary"),
        "axiom_false_positive_risk_summary": report.get(
            "axiom_false_positive_risk_summary"
        ),
        "axiom_decision_hierarchy_summary": report.get(
            "axiom_decision_hierarchy_summary"
        ),
        "axiom_exceptionality_summary": report.get("axiom_exceptionality_summary"),
        "axiom_cross_engine_stack_summary": report.get(
            "axiom_cross_engine_stack_summary"
        ),
        "axiom_history_record": report.get("axiom_history_record"),
        "axiom_lineage": report.get("axiom_lineage"),
        "axiom_institutional_reports": report.get("axiom_institutional_reports"),
        "platform_workspace": report.get("platform_workspace"),
        "platform_workflow": report.get("platform_workflow"),
        "platform_workflow_template": report.get("platform_workflow_template"),
        "platform_dossier": report.get("platform_dossier"),
        "platform_summary_view": report.get("platform_summary_view"),
        "platform_access_summary": report.get("platform_access_summary"),
        "platform_allowed_actions": report.get("platform_allowed_actions"),
        "platform_approvals": report.get("platform_approvals"),
        "platform_timeline": report.get("platform_timeline"),
        "platform_exports": report.get("platform_exports"),
        "platform_rendered_exports": report.get("platform_rendered_exports"),
        "platform_integration_summary": report.get("platform_integration_summary"),
        "platform_health_summary": report.get("platform_health_summary"),
        "platform_workspace_analytics": report.get("platform_workspace_analytics"),
        "platform_cross_workspace_analytics": report.get("platform_cross_workspace_analytics"),
        "platform_dashboard": report.get("platform_dashboard"),
        "platform_bootstrap_summary": report.get("platform_bootstrap_summary"),
        "platform_bootstrap_templates": report.get("platform_bootstrap_templates"),
        "platform_demo_bundles": report.get("platform_demo_bundles"),
        "platform_demo_snapshot": report.get("platform_demo_snapshot"),
        "platform_readiness_snapshot": report.get("platform_readiness_snapshot"),
        "platform_readiness_report": report.get("platform_readiness_report"),
        "platform_pilot_package": report.get("platform_pilot_package"),
        "platform_tracking": report.get("platform_tracking"),
        "platform_paper_trade": report.get("platform_paper_trade"),
        "platform_outcome_snapshot": report.get("platform_outcome_snapshot"),
        "platform_recommendation_evidence_summary": report.get(
            "platform_recommendation_evidence_summary"
        ),
        "platform_outcome_attribution": report.get("platform_outcome_attribution"),
        "platform_proof_summary": report.get("platform_proof_summary"),
        "platform_calibration_hardening": report.get(
            "platform_calibration_hardening"
        ),
        "platform_drift_summary": report.get("platform_drift_summary"),
        "platform_benchmarks": report.get("platform_benchmarks"),
        "platform_model_credibility_snapshot": report.get(
            "platform_model_credibility_snapshot"
        ),
        "operating_workflow": report.get("operating_workflow"),
    }
    section_context = {
        "signal_summary": report.get("signal_summary"),
        "technical_analysis": report.get("technical_analysis"),
        "fundamental_analysis": report.get("fundamental_analysis"),
        "statistical_analysis": report.get("statistical_analysis"),
        "sentiment_analysis": report.get("sentiment_analysis"),
        "macro_geopolitical_analysis": report.get("macro_geopolitical_analysis"),
        "event_catalyst_risk_analysis": report.get("event_catalyst_risk_analysis"),
        "risk_quality_analysis": report.get("risk_quality_analysis"),
        "overall_analysis": report.get("overall_analysis"),
        "strategy_view": report.get("strategy_view"),
        "risks_weaknesses_invalidators": report.get("risks_weaknesses_invalidators"),
        "evidence_provenance": report.get("evidence_provenance"),
        "deployment_readiness_summary": report.get("deployment_readiness_summary"),
        "deployment_permission_analysis": report.get("deployment_permission_analysis"),
        "risk_budget_exposure_analysis": report.get("risk_budget_exposure_analysis"),
        "rollout_stage_summary": report.get("rollout_stage_summary"),
        "canonical_validation_summary": report.get("canonical_validation_summary"),
        "walkforward_validation_summary": report.get("walkforward_validation_summary"),
        "net_of_friction_validation_summary": report.get("net_of_friction_validation_summary"),
        "suppression_readiness_validation_summary": report.get("suppression_readiness_validation_summary"),
        "drawdown_invalidation_validation_summary": report.get("drawdown_invalidation_validation_summary"),
        "portfolio_context_summary": report.get("portfolio_context_summary"),
        "portfolio_fit_analysis": report.get("portfolio_fit_analysis"),
        "execution_quality_analysis": report.get("execution_quality_analysis"),
        "portfolio_workflow_summary": report.get("portfolio_workflow_summary"),
        "portfolio_risk_model_summary": report.get("portfolio_risk_model_summary"),
        "hidden_overlap_redundancy_analysis": report.get("hidden_overlap_redundancy_analysis"),
        "factor_exposure_summary": report.get("factor_exposure_summary"),
        "concentration_cluster_risk_analysis": report.get("concentration_cluster_risk_analysis"),
        "replacement_diversification_analysis": report.get("replacement_diversification_analysis"),
        "portfolio_stress_fragility_summary": report.get("portfolio_stress_fragility_summary"),
        "learning_summary": report.get("learning_summary"),
        "regime_learning_summary": report.get("regime_learning_summary"),
        "adaptation_queue_summary": report.get("adaptation_queue_summary"),
        "experiment_registry_summary": report.get("experiment_registry_summary"),
        "archetype_motif_summary": report.get("archetype_motif_summary"),
        "system_health_summary": report.get("system_health_summary"),
        "shadow_mode_summary": report.get("shadow_mode_summary"),
        "drift_control_summary": report.get("drift_control_summary"),
        "incident_history_summary": report.get("incident_history_summary"),
        "commercialization_readiness_summary": report.get(
            "commercialization_readiness_summary"
        ),
        "source_governance_summary": report.get("source_governance_summary"),
        "data_provider_quality_summary": report.get("data_provider_quality_summary"),
        "buyer_diligence_summary": report.get("buyer_diligence_summary"),
        "axiom_summary": report.get("axiom_summary"),
        "axiom_proprietary_synthesis": report.get("axiom_proprietary_synthesis"),
        "axiom_support_vs_drag_summary": report.get("axiom_support_vs_drag_summary"),
        "axiom_why_now_summary": report.get("axiom_why_now_summary"),
        "axiom_unique_mispricing_summary": report.get(
            "axiom_unique_mispricing_summary"
        ),
        "axiom_setup_character_summary": report.get("axiom_setup_character_summary"),
        "axiom_false_positive_risk_summary": report.get(
            "axiom_false_positive_risk_summary"
        ),
        "axiom_decision_hierarchy_summary": report.get(
            "axiom_decision_hierarchy_summary"
        ),
        "axiom_exceptionality_summary": report.get("axiom_exceptionality_summary"),
        "axiom_cross_engine_stack_summary": report.get(
            "axiom_cross_engine_stack_summary"
        ),
        "axiom_summary_card": report.get("axiom_summary_card_text"),
        "axiom_historical_evidence_summary": report.get("axiom_evidence_summary"),
        "axiom_historical_evidence_summary_text": report.get(
            "axiom_historical_evidence_summary_text"
        ),
        "axiom_calibration_summary_text": report.get("axiom_calibration_summary_text"),
        "axiom_portfolio_governance_summary": report.get(
            "axiom_portfolio_governance_summary"
        ),
        "axiom_ic_memo_summary": report.get("axiom_ic_memo_summary"),
        "axiom_risk_deployability_memo_summary": report.get(
            "axiom_risk_deployability_memo_summary"
        ),
        "axiom_lineage_summary": report.get("axiom_lineage_summary"),
        "platform_overview_summary": report.get("platform_overview_summary"),
        "platform_dossier_summary": report.get("platform_dossier_summary"),
        "platform_monitoring_summary": report.get("platform_monitoring_summary"),
        "platform_access_control_summary": report.get(
            "platform_access_control_summary"
        ),
        "platform_workflow_actions_summary": report.get(
            "platform_workflow_actions_summary"
        ),
        "platform_audit_timeline_summary": report.get(
            "platform_audit_timeline_summary"
        ),
        "platform_export_summary": report.get("platform_export_summary"),
        "platform_export_rendering_summary": report.get(
            "platform_export_rendering_summary"
        ),
        "platform_integration_health_summary": report.get(
            "platform_integration_health_summary"
        ),
        "platform_dashboard_summary": report.get("platform_dashboard_summary"),
        "platform_analytics_summary": report.get("platform_analytics_summary"),
        "platform_demo_readiness_summary": report.get(
            "platform_demo_readiness_summary"
        ),
        "platform_bootstrap_summary_text": report.get(
            "platform_bootstrap_summary_text"
        ),
        "platform_readiness_report_summary": report.get(
            "platform_readiness_report_summary"
        ),
        "platform_pilot_package_summary": report.get(
            "platform_pilot_package_summary"
        ),
        "platform_demo_bundle_summary": report.get("platform_demo_bundle_summary"),
        "platform_proof_cycle_summary": report.get("platform_proof_cycle_summary"),
        "platform_tracking_summary": report.get("platform_tracking_summary"),
        "platform_outcome_summary": report.get("platform_outcome_summary"),
        "platform_calibration_hardening_summary": report.get(
            "platform_calibration_hardening_summary"
        ),
        "platform_drift_summary_text": report.get("platform_drift_summary_text"),
        "platform_benchmark_summary": report.get("platform_benchmark_summary"),
        "platform_model_credibility_summary": report.get(
            "platform_model_credibility_summary"
        ),
        "daily_operating_summary": report.get("daily_operating_summary"),
        "weekly_operating_summary": report.get("weekly_operating_summary"),
        "monthly_operating_summary": report.get("monthly_operating_summary"),
        "shadow_journal_summary": report.get("shadow_journal_summary"),
        "postmortem_summary": report.get("postmortem_summary"),
        "trust_maintenance_summary": report.get("trust_maintenance_summary"),
        "operator_runbook_summary": report.get("operator_runbook_summary"),
    }
    blocks = [
        "Grounding report metadata and machine-readable fields:",
        json.dumps(
            machine_context,
            indent=2,
            default=str,
            sort_keys=True,
            ensure_ascii=False,
        ),
        "Presentation-ready report sections:",
        json.dumps(
            section_context,
            indent=2,
            default=str,
            sort_keys=True,
            ensure_ascii=False,
        ),
    ]
    if context:
        blocks.extend(
            [
                "Caller context:",
                json.dumps(
                    context,
                    indent=2,
                    default=str,
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            ]
        )
    return "\n\n".join(blocks)


def build_grounded_chat_messages(
    history: List[Dict[str, str]],
    user_message: str,
    report: Dict[str, Any],
    context: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {
            "role": "system",
            "content": (
                "You are answering from a stored FTIP analysis report. Explain the system signal, drivers, strengths, weaknesses, "
                "risk/quality caveats, and strategy logic implied by that report. If asked whether the user should buy or sell, "
                "translate the report into the system's stance while clearly stating that this is not personalized financial advice. "
                "Prefer referencing AXIOM first when it is present, then use the report's signal summary, overall analysis, strategy view, "
                "risks/weaknesses/invalidators, and evidence provenance sections to explain the same decision."
                " When the user asks for an IC memo, one-pager, lineage, direct versus derived evidence, audience-specific framing, or why evidence is weak, use the AXIOM institutional reports and lineage sections first."
                " When the user asks about dossiers, workflows, workspace context, institutional review stages, dashboard analytics, export readiness, integration execution, or pilot readiness, use the platform workflow, analytics, export, integration, and demo-readiness sections directly."
                " When caller context identifies the current page, workspace, dossier, export, or active artifact, anchor the answer to that visible platform context instead of answering generically."
            ),
        },
        {"role": "system", "content": _grounding_block(report, context)},
    ]
    messages.extend(history[-12:])
    messages.append({"role": "user", "content": user_message})
    return messages


def summarize_signal(payload: Dict[str, Any]) -> str:
    symbol = payload.get("symbol", "?")
    as_of = payload.get("as_of", "?")
    lookback = payload.get("lookback", "?")
    signal = payload.get("signal", "?")
    score = payload.get("score")
    confidence = payload.get("confidence")
    thresholds = payload.get("thresholds") or {}
    calibration_meta = payload.get("calibration_meta") or {}
    notes = payload.get("notes") or []

    parts = [
        f"Signal for {symbol} as of {as_of} with lookback {lookback} is {signal}.",
    ]
    if score is not None:
        parts.append(f"Score: {score}.")
    if confidence is not None:
        parts.append(f"Confidence: {confidence}.")
    if thresholds:
        threshold_text = ", ".join(f"{k}={v}" for k, v in thresholds.items())
        parts.append(f"Thresholds used: {threshold_text}.")
    if calibration_meta:
        parts.append("Calibration metadata was applied.")
    if notes:
        parts.append("Notes: " + "; ".join(notes))
    return " ".join(parts)


def summarize_backtest(payload: Dict[str, Any]) -> str:
    total_return = payload.get("total_return")
    sharpe = payload.get("sharpe")
    max_drawdown = payload.get("max_drawdown")
    volatility = payload.get("volatility")
    window = payload.get("lookback") or payload.get("rebalance_every")

    parts = ["Backtest summary:"]
    if total_return is not None:
        parts.append(f"total_return={total_return}")
    if sharpe is not None:
        parts.append(f"sharpe={sharpe}")
    if max_drawdown is not None:
        parts.append(f"max_drawdown={max_drawdown}")
    if volatility is not None:
        parts.append(f"volatility={volatility}")
    if window is not None:
        parts.append(f"lookback={window}")
    return ", ".join(parts)


def system_capabilities() -> str:
    return (
        "I can explain stored analysis reports, canonical strategy artifacts, why-this-signal drilldowns, evidence provenance, thresholds, "
        "risk caveats, scenario framing, and backtest summaries. I never provide personalized investment advice."
    )
