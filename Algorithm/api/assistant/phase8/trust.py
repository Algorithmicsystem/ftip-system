from __future__ import annotations

from typing import Any, Dict, List

from .common import compact_list, safe_float


def _trust_tier(
    *,
    live_readiness_score: float,
    admitted_for_live: bool,
    admitted_for_paper: bool,
    deployment_mode_state: Dict[str, Any],
    drift_monitor: Dict[str, Any],
) -> str:
    if deployment_mode_state.get("active_mode") == "paused" or drift_monitor.get("pause_recommended"):
        return "paused"
    if not admitted_for_paper:
        return "blocked"
    if not admitted_for_live:
        return "paper_only"
    if live_readiness_score >= 86.0 and deployment_mode_state.get("active_mode") == "scaled_live":
        return "validated_live"
    if live_readiness_score >= 79.0 and deployment_mode_state.get("active_mode") in {"limited_live", "scaled_live"}:
        return "limited_live"
    return "conditional_live"


def build_deployment_permission(
    report: Dict[str, Any],
    *,
    deployment_mode_state: Dict[str, Any],
    model_readiness: Dict[str, Any],
    signal_admission: Dict[str, Any],
    drift_monitor: Dict[str, Any],
) -> Dict[str, Any]:
    strategy = report.get("strategy") or {}
    readiness_score = safe_float(model_readiness.get("live_readiness_score")) or 0.0
    blockers: List[str] = list(model_readiness.get("live_readiness_blockers") or [])
    blockers.extend(signal_admission.get("rejection_reasons") or [])
    if drift_monitor.get("pause_recommended"):
        blockers.append("drift and degradation checks recommend pausing live-use support")
    elif drift_monitor.get("degrade_to_paper_recommended"):
        blockers.append("drift and degradation checks recommend falling back to paper/shadow mode")

    admitted_for_strategy = bool(signal_admission.get("admitted_for_strategy"))
    admitted_for_paper = bool(signal_admission.get("admitted_for_paper"))
    admitted_for_live = bool(signal_admission.get("admitted_for_live"))
    mode = deployment_mode_state.get("active_mode")

    if mode == "paused":
        permission = "blocked_paused"
    elif not admitted_for_strategy:
        permission = "blocked_weak_evidence"
    elif mode == "research_only":
        permission = "analysis_only"
    elif not admitted_for_paper:
        permission = "watchlist_only"
    elif mode == "paper_shadow" or drift_monitor.get("degrade_to_paper_recommended"):
        permission = "paper_shadow_only"
    elif not admitted_for_live:
        permission = "paper_shadow_only"
    elif mode == "low_risk_live":
        permission = "low_risk_live_eligible"
    elif mode == "limited_live":
        permission = "limited_live_eligible"
    elif mode == "scaled_live" and readiness_score >= 86.0:
        permission = "scaled_live_eligible"
    else:
        permission = "limited_live_eligible"

    trust_tier = _trust_tier(
        live_readiness_score=readiness_score,
        admitted_for_live=admitted_for_live,
        admitted_for_paper=admitted_for_paper,
        deployment_mode_state=deployment_mode_state,
        drift_monitor=drift_monitor,
    )

    review_map = {
        "analysis_only": "analyst_review",
        "watchlist_only": "analyst_review",
        "paper_shadow_only": "analyst_review",
        "low_risk_live_eligible": "senior_analyst_and_risk_review",
        "limited_live_eligible": "portfolio_manager_and_risk_review",
        "scaled_live_eligible": "investment_committee_review",
        "blocked_paused": "risk_committee_review",
        "blocked_weak_evidence": "analyst_review",
    }
    minimum_required_review = review_map.get(permission, deployment_mode_state.get("review_floor"))
    human_review_required = permission not in {"analysis_only"}

    rationale = (
        f"Deployment mode is {mode}, trust tier is {trust_tier}, and the current permission is {permission}. "
        f"Readiness is {model_readiness.get('model_readiness_status')} at {readiness_score:.1f} / 100, "
        f"with strategy posture {strategy.get('strategy_posture') or strategy.get('final_signal') or 'n/a'}."
    )

    return {
        "deployment_permission": permission,
        "deployment_blockers": compact_list(blockers, limit=8),
        "deployment_rationale": rationale,
        "trust_tier": trust_tier,
        "minimum_required_review": minimum_required_review,
        "human_review_required": human_review_required,
        "paper_vs_live_classification": {
            "analysis_worthy": admitted_for_strategy,
            "paper_worthy": admitted_for_paper,
            "live_worthy": admitted_for_live and permission.endswith("eligible"),
        },
    }
