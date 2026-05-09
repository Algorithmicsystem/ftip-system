from __future__ import annotations

from typing import Any, Dict, List

from .common import (
    SEVERITY_CAUTION,
    SEVERITY_ELEVATED,
    SEVERITY_INFO,
    SEVERITY_SERIOUS,
    STATUS_CRITICAL,
    STATUS_DEGRADED,
    STATUS_HEALTHY,
    STATUS_WATCH,
    clamp,
    compact_list,
    safe_float,
)


def _learning_drift_severity(item: Dict[str, Any]) -> str:
    normalized = str(item.get("severity") or "").strip().lower()
    if normalized in {"critical", "high"}:
        return SEVERITY_SERIOUS
    if normalized in {"moderate", "medium"}:
        return SEVERITY_ELEVATED
    if normalized in {"low", "watch"}:
        return SEVERITY_CAUTION
    return SEVERITY_INFO


def build_operational_drift_monitor(
    current_report: Dict[str, Any],
    *,
    health_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    evaluation = current_report.get("evaluation") or {}
    validation = current_report.get("canonical_validation") or {}
    learning_alerts = current_report.get("learning_drift_alerts") or []
    phase8_alerts = current_report.get("drift_alerts") or []
    phase8_risk_alerts = current_report.get("deployment_risk_alerts") or []
    calibration = (evaluation.get("calibration_summary") or {})
    readiness_scorecard = validation.get("readiness_scorecard") or {}
    suppression_effect = validation.get("suppression_effect_summary") or {}
    net_returns = validation.get("net_return_summary") or {}
    ranking_scorecard = validation.get("ranking_scorecard") or {}
    current_feature_vector = ((current_report.get("data_bundle") or {}).get("canonical_alpha_core") or {}).get("feature_vector") or {}
    breadth_confirmation = safe_float(current_feature_vector.get("breadth_confirmation_score"))
    market_stress = safe_float(current_feature_vector.get("market_stress_score")) or safe_float(current_report.get("market_stress_score")) or 0.0
    cross_asset_conflict = safe_float(current_feature_vector.get("cross_asset_conflict_score")) or safe_float(current_report.get("cross_asset_conflict_score")) or 0.0
    regime_instability = safe_float((current_report.get("regime_intelligence") or {}).get("regime_instability")) or 0.0
    confidence_reliability = safe_float(calibration.get("confidence_reliability_score"))
    readiness_quality = safe_float(
        readiness_scorecard.get("paper_vs_live_candidate_quality_summary")
    )
    suppression_edge = safe_float(
        suppression_effect.get("suppression_effect_edge_spread")
    )
    net_edge = safe_float(net_returns.get("average_edge_return"))
    monotonicity = (
        calibration.get("confidence_monotonicity")
        or ranking_scorecard.get("confidence_monotonicity")
    )

    drift_alerts: List[Dict[str, Any]] = []
    for message in phase8_alerts:
        drift_alerts.append(
            {
                "drift_alert": str(message),
                "drift_type": "deployment_drift",
                "affected_component": "deployment_readiness",
                "drift_severity": SEVERITY_ELEVATED,
                "drift_window": "recent_cohort",
                "drift_supporting_evidence": str(message),
                "drift_recommended_action": "Increase review requirements and keep shadow discipline firm.",
            }
        )
    for message in phase8_risk_alerts:
        drift_alerts.append(
            {
                "drift_alert": str(message),
                "drift_type": "risk_behavior_drift",
                "affected_component": "deployment_gating",
                "drift_severity": SEVERITY_ELEVATED,
                "drift_window": "recent_cohort",
                "drift_supporting_evidence": str(message),
                "drift_recommended_action": "Lower trust and review live-like escalation decisions carefully.",
            }
        )
    for item in learning_alerts:
        drift_alerts.append(
            {
                "drift_alert": item.get("evidence")
                or item.get("affected_component")
                or "learning drift alert",
                "drift_type": "learning_drift",
                "affected_component": item.get("affected_component") or "unknown",
                "drift_severity": _learning_drift_severity(item),
                "drift_window": item.get("window") or "active_learning_cycle",
                "drift_supporting_evidence": item.get("evidence") or "No evidence note supplied.",
                "drift_recommended_action": item.get("recommended_action")
                or "Move the issue into governed review rather than silently adapting.",
            }
        )

    confidence_reliability_alert = bool(
        confidence_reliability is not None and confidence_reliability < 58.0
    )
    readiness_gate_reliability_alert = bool(
        readiness_quality is not None and readiness_quality <= 0.0
    )
    monotonicity_break_alert = monotonicity not in (
        None,
        "higher_confidence_buckets_outperform",
    )
    confidence_overstatement_flag = bool(
        confidence_reliability is not None
        and confidence_reliability < 50.0
        and (net_edge is not None and net_edge <= 0.0)
    )
    confidence_understatement_flag = bool(
        confidence_reliability is not None
        and confidence_reliability >= 70.0
        and (net_edge is not None and net_edge > 0.0)
        and str(current_report.get("deployment_permission") or "") == "paper_shadow_only"
    )
    if confidence_reliability_alert:
        drift_alerts.append(
            {
                "drift_alert": "Confidence calibration has weakened materially.",
                "drift_type": "calibration_drift",
                "affected_component": "confidence_calibration",
                "drift_severity": SEVERITY_SERIOUS
                if confidence_overstatement_flag
                else SEVERITY_ELEVATED,
                "drift_window": "active_validation_window",
                "drift_supporting_evidence": f"Confidence reliability score is {confidence_reliability}.",
                "drift_recommended_action": "Downgrade deployment trust until calibration reliability recovers.",
            }
        )
    if readiness_gate_reliability_alert:
        drift_alerts.append(
            {
                "drift_alert": "Readiness buckets are not separating quality cleanly enough.",
                "drift_type": "gating_drift",
                "affected_component": "readiness_gates",
                "drift_severity": SEVERITY_ELEVATED,
                "drift_window": "active_validation_window",
                "drift_supporting_evidence": f"Paper-vs-live quality spread is {readiness_quality}.",
                "drift_recommended_action": "Keep stronger deployment modes gated until readiness buckets recover.",
            }
        )
    if monotonicity_break_alert:
        drift_alerts.append(
            {
                "drift_alert": "Confidence bucket monotonicity is no longer clean.",
                "drift_type": "ranking_drift",
                "affected_component": "confidence_buckets",
                "drift_severity": SEVERITY_ELEVATED,
                "drift_window": "active_validation_window",
                "drift_supporting_evidence": str(monotonicity),
                "drift_recommended_action": "Treat confidence more cautiously and re-check calibration buckets.",
            }
        )
    if suppression_edge is not None and suppression_edge <= 0.0:
        drift_alerts.append(
            {
                "drift_alert": "Suppression logic is not improving net outcomes right now.",
                "drift_type": "suppression_drift",
                "affected_component": "suppression_logic",
                "drift_severity": SEVERITY_ELEVATED,
                "drift_window": "active_validation_window",
                "drift_supporting_evidence": f"Suppression edge spread is {suppression_edge}.",
                "drift_recommended_action": "Review suppression thresholds before trusting them more aggressively.",
            }
        )

    model_drift_score = clamp(
        len(drift_alerts) * 12.0
        + max(0.0, 60.0 - (confidence_reliability or 60.0)) * 0.75
        + (12.0 if readiness_gate_reliability_alert else 0.0)
        + (10.0 if monotonicity_break_alert else 0.0)
        + (18.0 if confidence_overstatement_flag else 0.0)
        + (6.0 if health_snapshot.get("fallback_overuse_alert") else 0.0),
        0.0,
        100.0,
    )
    environment_shift_score = clamp(
        (
            market_stress
            + cross_asset_conflict
            + regime_instability
            + max(0.0, 100.0 - (breadth_confirmation or 55.0))
        )
        / 4.0,
        0.0,
        100.0,
    )
    calibration_health_status = (
        STATUS_HEALTHY
        if confidence_reliability is not None and confidence_reliability >= 72.0 and not monotonicity_break_alert
        else STATUS_WATCH
        if confidence_reliability is not None and confidence_reliability >= 58.0
        else STATUS_DEGRADED
        if confidence_reliability is not None and confidence_reliability >= 42.0
        else STATUS_CRITICAL
    )

    return {
        "model_drift_score": round(model_drift_score, 2),
        "environment_shift_score": round(environment_shift_score, 2),
        "drift_alerts": drift_alerts,
        "calibration_health_status": calibration_health_status,
        "confidence_reliability_alert": confidence_reliability_alert,
        "readiness_gate_reliability_alert": readiness_gate_reliability_alert,
        "monotonicity_break_alert": monotonicity_break_alert,
        "calibration_drift_summary": compact_list(
            [
                f"confidence reliability {confidence_reliability}",
                f"monotonicity {monotonicity}" if monotonicity else None,
                f"readiness spread {readiness_quality}" if readiness_quality is not None else None,
                f"suppression edge {suppression_edge}" if suppression_edge is not None else None,
            ],
            limit=6,
        ),
        "confidence_overstatement_flag": confidence_overstatement_flag,
        "confidence_understatement_flag": confidence_understatement_flag,
    }
