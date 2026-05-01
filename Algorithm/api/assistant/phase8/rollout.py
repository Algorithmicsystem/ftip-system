from __future__ import annotations

from typing import Any, Dict, List

from .common import compact_list, safe_float


_STAGE_ORDER = [
    "historical_validation",
    "forward_shadow_validation",
    "low_risk_live_pilot",
    "limited_live_monitoring",
    "scaled_live_governed",
]


def build_rollout_workflow(
    *,
    deployment_mode_state: Dict[str, Any],
    model_readiness: Dict[str, Any],
    deployment_permission: Dict[str, Any],
    drift_monitor: Dict[str, Any],
    prior_audit_summary: Dict[str, Any],
) -> Dict[str, Any]:
    current_stage = deployment_mode_state.get("rollout_stage") or "historical_validation"
    readiness_score = safe_float(model_readiness.get("live_readiness_score")) or 0.0
    blocked = str(deployment_permission.get("deployment_permission") or "").startswith("blocked")

    if drift_monitor.get("pause_recommended"):
        readiness_checkpoint = "fail"
    elif blocked or drift_monitor.get("degrade_to_paper_recommended"):
        readiness_checkpoint = "watch"
    elif readiness_score >= 80.0:
        readiness_checkpoint = "pass"
    else:
        readiness_checkpoint = "watch"

    next_stage = None
    if current_stage in _STAGE_ORDER:
        current_index = _STAGE_ORDER.index(current_stage)
        if current_index + 1 < len(_STAGE_ORDER):
            next_stage = _STAGE_ORDER[current_index + 1]

    promotion_criteria: List[str] = [
        "confidence reliability remains above the stage threshold",
        "freshness and coverage stay within acceptable live-use bounds",
        "fragility does not breach the mode-specific ceiling",
        "no active pause recommendation is present",
    ]
    if next_stage:
        promotion_criteria.append(f"current stage evidence is strong enough to graduate toward {next_stage}")

    demotion_criteria = [
        "freshness or coverage degrades materially",
        "fragility rises into the blocked zone",
        "confidence reliability falls below the live-use floor",
        "drift monitoring recommends paper or paused mode",
    ]

    rollback_reason = None
    if drift_monitor.get("pause_recommended"):
        rollback_reason = "pause conditions are active, so the system should step back from live-use support"
    elif drift_monitor.get("degrade_to_paper_recommended"):
        rollback_reason = "degradation flags point back toward paper / shadow discipline"

    stage_transition_notes = [
        deployment_mode_state.get("mode_description"),
        f"Recent deployment audits: {prior_audit_summary.get('recent_audit_count', 0)} tracked, {prior_audit_summary.get('recent_blocked_count', 0)} blocked, {prior_audit_summary.get('recent_pause_recommendation_count', 0)} pause recommendations.",
    ]
    if next_stage:
        stage_transition_notes.append(f"Next eligible stage is {next_stage} if promotion criteria are satisfied.")
    if rollback_reason:
        stage_transition_notes.append(rollback_reason)

    return {
        "rollout_stage": current_stage,
        "promotion_criteria": compact_list(promotion_criteria, limit=8),
        "demotion_criteria": compact_list(demotion_criteria, limit=8),
        "rollback_reason": rollback_reason,
        "readiness_checkpoint": readiness_checkpoint,
        "stage_transition_notes": compact_list(stage_transition_notes, limit=8),
        "next_eligible_stage": next_stage,
    }
