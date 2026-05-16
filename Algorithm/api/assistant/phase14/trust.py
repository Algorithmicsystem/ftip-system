from __future__ import annotations

from typing import Any, Dict, List

from .common import compact_list, safe_float


def build_trust_maintenance(current_report: Dict[str, Any]) -> Dict[str, Any]:
    validation = current_report.get("canonical_validation") or {}
    linkage = validation.get("prediction_linkage_summary") or {}
    walkforward = validation.get("walkforward_summary") or {}
    promotion_reasons: List[str] = []
    demotion_reasons: List[str] = []

    if current_report.get("shadow_promotion_candidate"):
        promotion_reasons.append(
            "Shadow evidence is improving enough to consider a staged trust promotion review."
        )
    if (linkage.get("matured_count") or 0) >= 12:
        promotion_reasons.append("A larger matured outcome base is now available for review.")
    if (walkforward.get("window_count") or 0) >= 2:
        promotion_reasons.append("Walk-forward coverage is deep enough to support a stricter promotion discussion.")

    if current_report.get("pause_required"):
        demotion_reasons.append("Pause conditions are currently active.")
    if current_report.get("downgrade_to_shadow_recommended"):
        demotion_reasons.append("Operational controls are recommending a downgrade to shadow mode.")
    if current_report.get("shadow_demotion_reason"):
        demotion_reasons.append(str(current_report.get("shadow_demotion_reason")))
    if (safe_float(current_report.get("live_readiness_score")) or 0.0) < 55.0:
        demotion_reasons.append("Live readiness remains below the controlled-use comfort zone.")

    required_evidence = compact_list(
        [
            "Sustain positive walk-forward behavior across additional windows.",
            "Keep confidence and readiness calibration reliable out of sample.",
            "Reduce open drift and operational-health warnings.",
            "Demonstrate cleaner shadow-to-realized follow-through in the active regime.",
        ],
        limit=6,
    )
    recovery_checklist = compact_list(
        list(current_report.get("recovery_criteria") or [])
        + [
            "Restore degraded providers and freshness-sensitive domains.",
            "Resolve the highest-severity operational alert before trust is raised again.",
            "Reconfirm the current deployment permission after the next weekly review.",
        ],
        limit=6,
    )
    rollback = (
        current_report.get("downgrade_reason")
        or "Rollback to shadow-only use if calibration, drift, or provider-health evidence weakens further."
    )
    promotion_candidates = (
        [
            {
                "symbol": current_report.get("symbol"),
                "reason": promotion_reasons[0],
                "required_evidence": required_evidence[:3],
            }
        ]
        if promotion_reasons
        else []
    )
    demotion_candidates = (
        [
            {
                "symbol": current_report.get("symbol"),
                "reason": demotion_reasons[0],
                "rollback_recommendation": rollback,
            }
        ]
        if demotion_reasons
        else []
    )
    summary = (
        f"Trust maintenance currently has {len(promotion_candidates)} promotion candidate(s) and "
        f"{len(demotion_candidates)} demotion candidate(s). "
        f"Primary trust focus: {(demotion_reasons or promotion_reasons or ['maintain current staged trust'])[0]}"
    )
    return {
        "trust_maintenance_summary": summary,
        "promotion_candidate": promotion_candidates[0] if promotion_candidates else None,
        "demotion_candidate": demotion_candidates[0] if demotion_candidates else None,
        "rollback_recommendation": rollback,
        "trust_recovery_checklist": recovery_checklist,
        "required_evidence_for_promotion": required_evidence,
        "trust_promotion_candidates": promotion_candidates,
        "trust_demotion_candidates": demotion_candidates,
    }
