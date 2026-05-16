from __future__ import annotations

from typing import Any, Dict, Sequence

from .common import compact_list, safe_float


def classify_failure_mode(report: Dict[str, Any]) -> str:
    event_state = str(
        ((report.get("data_bundle") or {}).get("event_catalyst_risk") or {}).get(
            "event_risk_classification"
        )
        or ""
    ).lower()
    tradability_state = str(
        ((report.get("data_bundle") or {}).get("liquidity_execution_fragility") or {}).get(
            "tradability_state"
        )
        or ""
    ).lower()
    if "event" in event_state or "repricing" in event_state:
        return "event_distortion"
    if "caution" in tradability_state or "unstable" in tradability_state:
        return "liquidity_gap_fragility"
    if (safe_float(report.get("cross_asset_conflict_score")) or 0.0) >= 60.0 or (
        safe_float(report.get("market_stress_score")) or 0.0
    ) >= 60.0:
        return "hostile_macro_or_stress_context"
    if (safe_float(report.get("hidden_overlap_score")) or 0.0) >= 70.0:
        return "portfolio_overlap_misfit"
    if (safe_float(report.get("live_readiness_score")) or 0.0) < 55.0:
        return "confidence_or_readiness_overstatement"
    return "watchlist_only_thesis"


def build_postmortem_report(
    current_report: Dict[str, Any],
    *,
    weekly_validation: Dict[str, Any],
    monthly_validation: Dict[str, Any],
    recent_incidents: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    failure_mode = classify_failure_mode(current_report)
    failure_modes = compact_list(
        [
            *(weekly_validation.get("failure_modes") or []),
            *(monthly_validation.get("failure_modes") or []),
            *[
                item.get("summary") or item.get("alert_summary")
                for item in recent_incidents[:4]
                if isinstance(item, dict)
            ],
        ],
        limit=6,
    )
    lesson = {
        "event_distortion": "Event windows deserve stricter watchlist treatment and tighter invalidation sensitivity.",
        "liquidity_gap_fragility": "Implementation fragility should stay elevated and size-band logic should remain constrained.",
        "hostile_macro_or_stress_context": "Cross-asset conflict and stress context should stay central to suppression and trust gating.",
        "portfolio_overlap_misfit": "Standalone strength is not enough when the portfolio is already carrying similar latent risk.",
        "confidence_or_readiness_overstatement": "Confidence and readiness thresholds need more evidence before the trust state can rise.",
    }.get(
        failure_mode,
        "Keep the thesis in structured review until confirmation and realized follow-through improve.",
    )
    postmortem_queue = compact_list(
        [
            current_report.get("drawdown_invalidation_summary"),
            current_report.get("suppression_readiness_validation_summary"),
            current_report.get("portfolio_stress_fragility_summary"),
            current_report.get("drift_control_summary"),
            *failure_modes,
        ],
        limit=6,
    )
    return {
        "postmortem_report": {
            "failure_mode_classification": failure_mode,
            "supporting_evidence": compact_list(
                [
                    current_report.get("deployment_rationale"),
                    current_report.get("risk_quality_analysis"),
                    current_report.get("event_catalyst_risk_analysis"),
                    current_report.get("liquidity_execution_fragility_analysis"),
                ],
                limit=4,
            ),
            "recent_failure_modes": failure_modes,
        },
        "failure_mode_classification": failure_mode,
        "misclassification_summary": (
            f"Active post-mortem lens is {failure_mode}, using the current blockers, suppression state, and recent failure cohorts as the primary evidence base."
        ),
        "confidence_error_analysis": current_report.get(
            "suppression_readiness_validation_summary"
        )
        or "Confidence error analysis is still building.",
        "fragility_miss_summary": current_report.get(
            "drawdown_invalidation_summary"
        )
        or "Fragility miss analysis is still building.",
        "lesson_extracted": lesson,
        "postmortem_queue": postmortem_queue,
        "postmortem_summary": (
            f"Current post-mortem focus is {failure_mode}. Key lesson: {lesson}"
        ),
    }
