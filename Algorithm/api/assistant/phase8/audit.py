from __future__ import annotations

from typing import Any, Dict

from .common import DEPLOYMENT_READINESS_VERSION, now_utc, safe_float, score_value


def build_deployment_audit_record(
    report: Dict[str, Any],
    *,
    deployment_mode_state: Dict[str, Any],
    model_readiness: Dict[str, Any],
    signal_admission: Dict[str, Any],
    deployment_permission: Dict[str, Any],
    risk_budgeting: Dict[str, Any],
    rollout_workflow: Dict[str, Any],
    drift_monitor: Dict[str, Any],
) -> Dict[str, Any]:
    strategy = report.get("strategy") or {}
    agreement = report.get("domain_agreement") or {}
    evaluation = report.get("evaluation") or {}
    signal = report.get("signal") or {}
    return {
        "audit_version": DEPLOYMENT_READINESS_VERSION,
        "captured_at": now_utc(),
        "symbol": report.get("symbol"),
        "as_of_date": report.get("as_of_date"),
        "horizon": report.get("horizon"),
        "risk_mode": report.get("risk_mode"),
        "deployment_mode": deployment_mode_state.get("active_mode"),
        "rollout_stage": rollout_workflow.get("rollout_stage"),
        "signal": signal.get("action"),
        "final_signal": strategy.get("final_signal"),
        "strategy_posture": strategy.get("strategy_posture"),
        "confidence_score": safe_float(strategy.get("confidence_score")),
        "conviction_tier": strategy.get("conviction_tier"),
        "actionability_score": safe_float(strategy.get("actionability_score")),
        "fragility_score": score_value((report.get("proprietary_scores") or {}).get("Signal Fragility Index")),
        "fragility_tier": strategy.get("fragility_tier"),
        "domain_agreement_score": safe_float(agreement.get("domain_agreement_score")),
        "domain_conflict_score": safe_float(agreement.get("domain_conflict_score")),
        "model_readiness_status": model_readiness.get("model_readiness_status"),
        "live_readiness_score": safe_float(model_readiness.get("live_readiness_score")),
        "deployment_permission": deployment_permission.get("deployment_permission"),
        "deployment_blockers": deployment_permission.get("deployment_blockers") or [],
        "deployment_rationale": deployment_permission.get("deployment_rationale"),
        "trust_tier": deployment_permission.get("trust_tier"),
        "minimum_required_review": deployment_permission.get("minimum_required_review"),
        "human_review_required": deployment_permission.get("human_review_required"),
        "risk_budget_tier": risk_budgeting.get("risk_budget_tier"),
        "exposure_caution_level": risk_budgeting.get("exposure_caution_level"),
        "fragility_adjusted_size_band": risk_budgeting.get("fragility_adjusted_size_band"),
        "confidence_adjusted_size_band": risk_budgeting.get("confidence_adjusted_size_band"),
        "maximum_risk_mode_allowed": risk_budgeting.get("maximum_risk_mode_allowed"),
        "admission_decision": signal_admission.get("primary_decision"),
        "report_version": report.get("report_version"),
        "strategy_version": strategy.get("strategy_version") or report.get("strategy_version"),
        "evaluation_context_snapshot": {
            "status": evaluation.get("status"),
            "matured_count": ((evaluation.get("signal_scorecard") or {}).get("final_signal_overall") or {}).get("matured_count"),
            "confidence_reliability_score": ((evaluation.get("calibration_summary") or {}).get("confidence_reliability_score")),
            "strongest_conditions": evaluation.get("strongest_conditions") or [],
            "weakest_conditions": evaluation.get("weakest_conditions") or [],
        },
        "pause_recommended": drift_monitor.get("pause_recommended"),
        "degrade_to_paper_recommended": drift_monitor.get("degrade_to_paper_recommended"),
        "drift_alerts": drift_monitor.get("drift_alerts") or [],
        "deployment_risk_alerts": drift_monitor.get("deployment_risk_alerts") or [],
        "rationale_summary": (
            f"Mode {deployment_mode_state.get('active_mode')} / {deployment_permission.get('deployment_permission')} "
            f"with readiness {safe_float(model_readiness.get('live_readiness_score')) or 0.0:.1f} and trust tier "
            f"{deployment_permission.get('trust_tier')}."
        ),
    }
