from __future__ import annotations

from typing import Any, Dict, List

from .common import (
    STATUS_CRITICAL,
    STATUS_DEGRADED,
    clamp,
    compact_list,
    health_rank,
)


def build_control_state(
    current_report: Dict[str, Any],
    *,
    health_snapshot: Dict[str, Any],
    shadow_mode: Dict[str, Any],
    drift_monitor: Dict[str, Any],
) -> Dict[str, Any]:
    system_health_status = str(health_snapshot.get("system_health_status") or "watch")
    provider_health_status = str(health_snapshot.get("provider_health_status") or "watch")
    current_deployment_mode = str(current_report.get("deployment_mode") or "research_only")
    live_readiness = float(current_report.get("live_readiness_score") or 0.0)
    drift_score = float(drift_monitor.get("model_drift_score") or 0.0)
    environment_shift = float(drift_monitor.get("environment_shift_score") or 0.0)
    data_reliability = float(health_snapshot.get("data_reliability_score") or 0.0)
    calibration_health = str(drift_monitor.get("calibration_health_status") or "watch")
    pause_recommended = bool(current_report.get("pause_recommended")) or bool(
        health_rank(system_health_status) >= health_rank(STATUS_DEGRADED)
        and (drift_score >= 60.0 or data_reliability < 40.0)
    )
    pause_required = bool(
        current_deployment_mode == "paused"
        or health_rank(system_health_status) >= health_rank(STATUS_CRITICAL)
        or health_snapshot.get("critical_domain_missing_flag")
        or calibration_health == STATUS_CRITICAL
        or data_reliability < 28.0
        or drift_score >= 82.0
    )
    downgrade_to_shadow_recommended = bool(
        not pause_required
        and (
            bool(current_report.get("degrade_to_paper_recommended"))
            or current_deployment_mode in {"low_risk_live", "limited_live", "scaled_live"}
            and (
                health_rank(provider_health_status) >= health_rank(STATUS_DEGRADED)
                or health_rank(calibration_health) >= health_rank(STATUS_DEGRADED)
                or drift_score >= 58.0
                or environment_shift >= 68.0
                or live_readiness < 65.0
            )
        )
    )

    subsystem_block_flags: List[str] = []
    degraded_domains = set(health_snapshot.get("degraded_domain_list") or [])
    if "cross_asset" in degraded_domains:
        subsystem_block_flags.append("cross_asset_confirmation")
    if "event" in degraded_domains:
        subsystem_block_flags.append("event_risk_overlay")
    if "macro" in degraded_domains:
        subsystem_block_flags.append("macro_context_escalation")
    if health_snapshot.get("critical_domain_missing_flag"):
        subsystem_block_flags.append("deployment_recommendations")
    if health_rank(calibration_health) >= health_rank(STATUS_DEGRADED):
        subsystem_block_flags.append("high_trust_deployment_modes")

    if pause_required:
        current_operating_mode = "paused"
    elif downgrade_to_shadow_recommended or shadow_mode.get("shadow_mode_status") == "active_shadow":
        current_operating_mode = "shadow_only"
    elif pause_recommended or drift_score >= 45.0:
        current_operating_mode = "increased_review"
    else:
        current_operating_mode = "normal"

    if pause_required:
        downgrade_reason = (
            "Critical data-health, calibration, or drift conditions have breached the pause threshold."
        )
    elif downgrade_to_shadow_recommended:
        downgrade_reason = (
            "Provider, calibration, or drift conditions have weakened enough that live-like trust should fall back to shadow discipline."
        )
    elif pause_recommended:
        downgrade_reason = "Review burden is elevated because operational conditions have weakened."
    else:
        downgrade_reason = "No forced downgrade is active."

    recovery_criteria = compact_list(
        [
            "Critical domains return to fresh or fully covered status."
            if health_snapshot.get("critical_domain_missing_flag")
            else None,
            "Fallback-only operation falls back below the caution threshold."
            if health_snapshot.get("fallback_overuse_alert")
            else None,
            "Confidence calibration recovers into the watch or healthy range."
            if health_rank(calibration_health) >= health_rank(STATUS_DEGRADED)
            else None,
            "Model drift score cools below 45 and environment shift stabilizes."
            if drift_score >= 45.0 or environment_shift >= 60.0
            else None,
            "Operator review confirms that weakened subsystems are safe to re-enable."
            if subsystem_block_flags
            else None,
        ],
        limit=6,
    )
    operator_attention_required = bool(
        pause_required
        or downgrade_to_shadow_recommended
        or pause_recommended
        or subsystem_block_flags
    )
    return {
        "current_operating_mode": current_operating_mode,
        "pause_recommended": pause_recommended,
        "pause_required": pause_required,
        "downgrade_to_shadow_recommended": downgrade_to_shadow_recommended,
        "downgrade_reason": downgrade_reason,
        "subsystem_block_flags": subsystem_block_flags,
        "recovery_criteria": recovery_criteria,
        "operator_attention_required": operator_attention_required,
        "shadow_promotion_candidate": bool(shadow_mode.get("shadow_promotion_candidate")),
        "portfolio_risk_model_reliability_flag": bool(
            float(current_report.get("correlation_breakdown_risk") or 0.0) < 70.0
        ),
        "operational_risk_score": round(
            clamp(
                (
                    drift_score
                    + environment_shift
                    + (100.0 - data_reliability)
                    + (100.0 - min(live_readiness, 100.0))
                )
                / 4.0,
                0.0,
                100.0,
            ),
            2,
        ),
    }
