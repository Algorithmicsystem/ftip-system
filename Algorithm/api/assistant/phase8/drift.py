from __future__ import annotations

from typing import Any, Dict, List

from .common import compact_list, safe_float, score_value


def build_drift_monitor(
    report: Dict[str, Any],
    *,
    model_readiness: Dict[str, Any],
    prior_audit_summary: Dict[str, Any],
    deployment_mode_state: Dict[str, Any],
) -> Dict[str, Any]:
    evaluation = report.get("evaluation") or {}
    evaluation_consistency = model_readiness.get("evaluation_consistency") or {}
    calibration_score = safe_float(evaluation_consistency.get("confidence_reliability_score")) or 0.0
    fragility_score = score_value((report.get("proprietary_scores") or {}).get("Signal Fragility Index")) or 0.0
    readiness_score = safe_float(model_readiness.get("live_readiness_score")) or 0.0
    fallback_ratio = safe_float((model_readiness.get("fallback_summary") or {}).get("fallback_domain_ratio")) or 0.0
    weakest = evaluation.get("weakest_conditions") or []

    degraded_reliability_regimes = compact_list(
        f"{item.get('dimension')}={item.get('label')}"
        for item in weakest
    )

    drift_alerts: List[str] = []
    risk_alerts: List[str] = []

    if calibration_score < 55.0:
        drift_alerts.append("confidence reliability is below the live-support comfort zone")
    if evaluation_consistency.get("confidence_monotonicity") not in (None, "higher_confidence_buckets_outperform"):
        drift_alerts.append("confidence bucket monotonicity is not clean")
    if fallback_ratio >= 0.25:
        drift_alerts.append("fallback usage is elevated across core domains")
    if readiness_score < 60.0:
        risk_alerts.append("live readiness has slipped below the controlled-live comfort zone")
    if fragility_score >= 58.0:
        risk_alerts.append("fragility is clustering in a range that historically requires caution")
    if degraded_reliability_regimes:
        risk_alerts.append("current conditions overlap with historically weaker evaluation slices")
    if prior_audit_summary.get("recent_pause_recommendation_count", 0) >= 2:
        drift_alerts.append("recent deployment audits have repeatedly recommended pause conditions")
    if prior_audit_summary.get("recent_blocked_count", 0) >= 3:
        risk_alerts.append("recent deployment audits have repeatedly blocked setups")

    pause_recommended = (
        deployment_mode_state.get("active_mode") == "paused"
        or readiness_score < 40.0
        or calibration_score < 45.0
        or fallback_ratio >= 0.5
        or (len(drift_alerts) + len(risk_alerts) >= 5 and readiness_score < 55.0)
    )
    degrade_to_paper_recommended = (
        not pause_recommended
        and deployment_mode_state.get("live_allowed")
        and (
            readiness_score < 65.0
            or calibration_score < 58.0
            or fragility_score >= 52.0
            or bool(degraded_reliability_regimes)
        )
    )
    increased_review_required = (
        pause_recommended
        or degrade_to_paper_recommended
        or len(drift_alerts) >= 2
        or len(risk_alerts) >= 2
    )

    return {
        "pause_recommended": pause_recommended,
        "degrade_to_paper_recommended": degrade_to_paper_recommended,
        "increased_review_required": increased_review_required,
        "degraded_reliability_regimes": degraded_reliability_regimes,
        "drift_alerts": compact_list(drift_alerts, limit=8),
        "deployment_risk_alerts": compact_list(risk_alerts, limit=8),
    }
