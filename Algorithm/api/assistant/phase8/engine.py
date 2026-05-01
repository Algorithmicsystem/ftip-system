from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.assistant.storage import AssistantStorage

from .admission import build_signal_admission_control
from .audit import build_deployment_audit_record
from .common import (
    DEPLOYMENT_AUDIT_RECORD_KIND,
    DEPLOYMENT_READINESS_ARTIFACT_KIND,
    DEPLOYMENT_READINESS_VERSION,
    compact_list,
    now_utc,
)
from .drift import build_drift_monitor
from .modes import build_deployment_mode_state
from .readiness import build_model_readiness
from .risk import build_risk_budget_framework
from .rollout import build_rollout_workflow
from .trust import build_deployment_permission


def _prior_audit_summary(
    store: AssistantStorage,
    *,
    symbol: Optional[str],
    horizon: Optional[str],
    risk_mode: Optional[str],
) -> Dict[str, Any]:
    audits = store.list_artifacts(kind=DEPLOYMENT_AUDIT_RECORD_KIND, limit=50)
    filtered: List[Dict[str, Any]] = []
    for artifact in audits:
        payload = artifact.get("payload") or {}
        if symbol and str(payload.get("symbol")) != str(symbol):
            continue
        if horizon and str(payload.get("horizon")) != str(horizon):
            continue
        if risk_mode and str(payload.get("risk_mode")) != str(risk_mode):
            continue
        filtered.append(payload)
    return {
        "recent_audit_count": len(filtered),
        "recent_blocked_count": sum(
            1
            for item in filtered
            if str(item.get("deployment_permission") or "").startswith("blocked")
        ),
        "recent_pause_recommendation_count": sum(
            1 for item in filtered if bool(item.get("pause_recommended"))
        ),
        "recent_live_eligible_count": sum(
            1
            for item in filtered
            if str(item.get("deployment_permission") or "").endswith("eligible")
        ),
    }


def _summary_text(
    *,
    deployment_mode_state: Dict[str, Any],
    model_readiness: Dict[str, Any],
    deployment_permission: Dict[str, Any],
    risk_budgeting: Dict[str, Any],
    rollout_workflow: Dict[str, Any],
    drift_monitor: Dict[str, Any],
) -> Dict[str, str]:
    readiness_summary = (
        f"Deployment mode is {deployment_mode_state.get('active_mode')} and rollout stage is {rollout_workflow.get('rollout_stage')}. "
        f"Model readiness is {model_readiness.get('model_readiness_status')} at {model_readiness.get('live_readiness_score')} / 100. "
        f"Key blockers are {', '.join(model_readiness.get('live_readiness_blockers') or ['none'])}."
    )
    permission_summary = (
        f"Deployment permission is {deployment_permission.get('deployment_permission')} with trust tier {deployment_permission.get('trust_tier')}. "
        f"Minimum review is {deployment_permission.get('minimum_required_review')} and human review required is {deployment_permission.get('human_review_required')}."
    )
    risk_budget_summary = (
        f"Risk budget tier is {risk_budgeting.get('risk_budget_tier')} with {risk_budgeting.get('exposure_caution_level')} exposure caution. "
        f"Fragility-adjusted size band is {risk_budgeting.get('fragility_adjusted_size_band')} and confidence-adjusted size band is {risk_budgeting.get('confidence_adjusted_size_band')}."
    )
    rollout_summary = (
        f"Readiness checkpoint is {rollout_workflow.get('readiness_checkpoint')}; next eligible stage is {rollout_workflow.get('next_eligible_stage') or 'none'}. "
        f"Pause recommended is {drift_monitor.get('pause_recommended')} and degrade-to-paper recommended is {drift_monitor.get('degrade_to_paper_recommended')}."
    )
    return {
        "readiness_summary": readiness_summary,
        "permission_summary": permission_summary,
        "risk_budget_summary": risk_budget_summary,
        "rollout_summary": rollout_summary,
    }


def build_deployment_readiness_artifact(
    *,
    current_report: Dict[str, Any],
    store: AssistantStorage,
) -> Dict[str, Any]:
    deployment_mode_state = build_deployment_mode_state()
    prior_audit_summary = _prior_audit_summary(
        store,
        symbol=current_report.get("symbol"),
        horizon=current_report.get("horizon"),
        risk_mode=current_report.get("risk_mode"),
    )
    model_readiness = build_model_readiness(
        current_report,
        deployment_mode_state=deployment_mode_state,
        prior_audit_summary=prior_audit_summary,
    )
    signal_admission = build_signal_admission_control(
        current_report,
        model_readiness=model_readiness,
        deployment_mode_state=deployment_mode_state,
    )
    drift_monitor = build_drift_monitor(
        current_report,
        model_readiness=model_readiness,
        prior_audit_summary=prior_audit_summary,
        deployment_mode_state=deployment_mode_state,
    )
    deployment_permission = build_deployment_permission(
        current_report,
        deployment_mode_state=deployment_mode_state,
        model_readiness=model_readiness,
        signal_admission=signal_admission,
        drift_monitor=drift_monitor,
    )
    risk_budgeting = build_risk_budget_framework(
        current_report,
        model_readiness=model_readiness,
        deployment_permission=deployment_permission,
        deployment_mode_state=deployment_mode_state,
    )
    rollout_workflow = build_rollout_workflow(
        deployment_mode_state=deployment_mode_state,
        model_readiness=model_readiness,
        deployment_permission=deployment_permission,
        drift_monitor=drift_monitor,
        prior_audit_summary=prior_audit_summary,
    )
    audit_snapshot = build_deployment_audit_record(
        current_report,
        deployment_mode_state=deployment_mode_state,
        model_readiness=model_readiness,
        signal_admission=signal_admission,
        deployment_permission=deployment_permission,
        risk_budgeting=risk_budgeting,
        rollout_workflow=rollout_workflow,
        drift_monitor=drift_monitor,
    )
    summaries = _summary_text(
        deployment_mode_state=deployment_mode_state,
        model_readiness=model_readiness,
        deployment_permission=deployment_permission,
        risk_budgeting=risk_budgeting,
        rollout_workflow=rollout_workflow,
        drift_monitor=drift_monitor,
    )
    return {
        "deployment_readiness_kind": DEPLOYMENT_READINESS_ARTIFACT_KIND,
        "deployment_readiness_version": DEPLOYMENT_READINESS_VERSION,
        "generated_at": now_utc(),
        "deployment_mode": deployment_mode_state,
        "model_readiness": model_readiness,
        "signal_admission_control": signal_admission,
        "deployment_permission": deployment_permission,
        "risk_budgeting": risk_budgeting,
        "rollout_workflow": rollout_workflow,
        "drift_monitor": drift_monitor,
        "prior_audit_summary": prior_audit_summary,
        "audit_snapshot": audit_snapshot,
        "readiness_summary": summaries["readiness_summary"],
        "permission_summary": summaries["permission_summary"],
        "risk_budget_summary": summaries["risk_budget_summary"],
        "rollout_summary": summaries["rollout_summary"],
    }
