from __future__ import annotations

from typing import Any, Dict, Sequence

from .common import compact_list, safe_float


def build_weekly_review(
    current_report: Dict[str, Any],
    *,
    weekly_validation: Dict[str, Any],
    recent_shadow_records: Sequence[Dict[str, Any]],
    recent_incidents: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    linkage = weekly_validation.get("prediction_linkage_summary") or {}
    net_summary = weekly_validation.get("net_return_summary") or {}
    walkforward = weekly_validation.get("walkforward_summary") or {}
    readiness = weekly_validation.get("readiness_scorecard") or {}
    suppression = weekly_validation.get("suppression_effect_summary") or {}
    strongest = compact_list(weekly_validation.get("strongest_conditions") or [], limit=4)
    weakest = compact_list(weekly_validation.get("weakest_conditions") or [], limit=4)
    failures = compact_list(weekly_validation.get("failure_modes") or [], limit=4)
    weekly_attention = compact_list(
        [
            current_report.get("confidence_reliability_alert"),
            current_report.get("readiness_gate_reliability_alert"),
            current_report.get("monotonicity_break_alert"),
            *[
                item.get("summary") or item.get("alert_summary")
                for item in recent_incidents[:4]
                if isinstance(item, dict)
            ],
            *(weekly_validation.get("weakest_conditions") or []),
        ],
        limit=6,
    )
    weekly_summary = (
        f"This week the platform tracked {linkage.get('total_predictions') or 0} cohort decisions with "
        f"{linkage.get('matured_count') or 0} matured outcomes, net edge {net_summary.get('average_edge_return')}, "
        f"hit rate {net_summary.get('hit_rate')}, and {walkforward.get('window_count') or 0} walk-forward windows."
    )
    return {
        "weekly_operating_review": {
            "review_window_days": 7,
            "validation_status": weekly_validation.get("status"),
            "prediction_linkage_summary": linkage,
            "walkforward_summary": walkforward,
            "shadow_records_reviewed": len(recent_shadow_records),
            "recent_incidents": len(recent_incidents),
        },
        "weekly_quality_summary": {
            "average_net_edge_return": net_summary.get("average_edge_return"),
            "hit_rate": net_summary.get("hit_rate"),
            "confidence_reliability": (
                (weekly_validation.get("calibration_summary") or {}).get(
                    "confidence_reliability_score"
                )
            ),
            "readiness_spread": readiness.get("paper_vs_live_candidate_quality_summary"),
            "suppression_spread": suppression.get("suppression_effect_edge_spread"),
        },
        "weekly_signal_review": {
            "strongest_conditions": strongest,
            "weakest_conditions": weakest,
            "matured_count": linkage.get("matured_count"),
        },
        "weekly_risk_review": {
            "failure_modes": failures,
            "shadow_reliability_summary": current_report.get("shadow_reliability_summary"),
            "system_health_status": current_report.get("system_health_status"),
            "provider_health_status": current_report.get("provider_health_status"),
            "model_drift_score": current_report.get("model_drift_score"),
        },
        "weekly_refinement_notes": compact_list(
            [
                current_report.get("suppression_readiness_validation_summary"),
                current_report.get("drift_control_summary"),
                current_report.get("portfolio_workflow_summary"),
                current_report.get("source_governance_summary"),
            ],
            limit=5,
        ),
        "weekly_operator_attention_items": weekly_attention,
        "weekly_operating_summary": weekly_summary,
    }
