from __future__ import annotations

from typing import Any, Dict, List

from .common import (
    SEVERITY_CAUTION,
    SEVERITY_CRITICAL,
    SEVERITY_ELEVATED,
    SEVERITY_INFO,
    SEVERITY_SERIOUS,
    compact_list,
    severity_rank,
)


def _alert(
    alert_id: str,
    severity: str,
    domain: str,
    summary: str,
    recommended_action: str,
    *,
    operator_review_required: bool = False,
    escalation_path: str = "operations_review",
) -> Dict[str, Any]:
    return {
        "alert_id": alert_id,
        "alert_severity": severity,
        "alert_domain": domain,
        "alert_summary": summary,
        "recommended_action": recommended_action,
        "operator_review_required": operator_review_required,
        "escalation_path": escalation_path,
    }


def build_operational_alerts(
    current_report: Dict[str, Any],
    *,
    health_snapshot: Dict[str, Any],
    shadow_mode: Dict[str, Any],
    drift_monitor: Dict[str, Any],
    control_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []

    if health_snapshot.get("freshness_alert"):
        alerts.append(
            _alert(
                "freshness-alert",
                SEVERITY_ELEVATED
                if health_snapshot.get("critical_domain_missing_flag")
                else SEVERITY_CAUTION,
                "data_freshness",
                "One or more critical data domains are stale or only stale-but-usable.",
                "Treat new signals more cautiously until freshness normalizes.",
                operator_review_required=bool(
                    health_snapshot.get("critical_domain_missing_flag")
                ),
            )
        )
    if health_snapshot.get("fallback_overuse_alert"):
        alerts.append(
            _alert(
                "fallback-overuse",
                SEVERITY_ELEVATED,
                "fallbacks",
                "Fallback-only or fallback-heavy operation is elevated across the active domain set.",
                "Downgrade trust and review which domains are leaning on degraded source paths.",
                operator_review_required=True,
            )
        )
    if health_snapshot.get("provider_degradation_notes"):
        alerts.append(
            _alert(
                "provider-degradation",
                SEVERITY_CAUTION
                if str(health_snapshot.get("provider_health_status")) == "watch"
                else SEVERITY_SERIOUS
                if str(health_snapshot.get("provider_health_status")) == "degraded"
                else SEVERITY_CRITICAL,
                "provider_health",
                "Provider or domain-health notes indicate degraded external support.",
                "Inspect degraded providers and confirm that canonical fallbacks are still acceptable.",
                operator_review_required=True,
                escalation_path="data_operations_review",
            )
        )

    for index, item in enumerate(drift_monitor.get("drift_alerts") or []):
        alerts.append(
            _alert(
                f"drift-{index}",
                str(item.get("drift_severity") or SEVERITY_INFO),
                str(item.get("affected_component") or "model_drift"),
                str(item.get("drift_alert") or item.get("drift_supporting_evidence") or "Operational drift alert."),
                str(item.get("drift_recommended_action") or "Review the affected component."),
                operator_review_required=severity_rank(
                    item.get("drift_severity")
                )
                >= severity_rank(SEVERITY_ELEVATED),
                escalation_path="research_and_risk_review",
            )
        )

    if control_state.get("downgrade_to_shadow_recommended"):
        alerts.append(
            _alert(
                "downgrade-shadow",
                SEVERITY_SERIOUS,
                "deployment_controls",
                "Operational controls recommend downgrading to shadow or paper discipline.",
                str(control_state.get("downgrade_reason") or "Reduce deployment trust."),
                operator_review_required=True,
                escalation_path="risk_committee_review",
            )
        )
    if control_state.get("pause_required"):
        alerts.append(
            _alert(
                "critical-pause",
                SEVERITY_CRITICAL,
                "kill_switch",
                "The operational kill-switch threshold has been breached.",
                "Pause higher-trust deployment support until recovery criteria are satisfied.",
                operator_review_required=True,
                escalation_path="critical_pause",
            )
        )
    elif control_state.get("pause_recommended"):
        alerts.append(
            _alert(
                "pause-recommended",
                SEVERITY_ELEVATED,
                "kill_switch",
                "Operational controls are recommending a pause review.",
                "Escalate review and be prepared to pause higher-trust modes if conditions worsen.",
                operator_review_required=True,
                escalation_path="risk_committee_review",
            )
        )

    if shadow_mode.get("shadow_mode_status") == "active_shadow":
        alerts.append(
            _alert(
                "shadow-active",
                SEVERITY_INFO,
                "shadow_mode",
                "The platform is currently collecting forward evidence in shadow mode.",
                "Keep tracking shadow-vs-realized behavior before raising trust.",
            )
        )
    elif shadow_mode.get("shadow_demotion_reason"):
        alerts.append(
            _alert(
                "shadow-demotion",
                SEVERITY_CAUTION,
                "shadow_mode",
                str(shadow_mode.get("shadow_demotion_reason")),
                "Keep the platform in shadow discipline until the demotion reason clears.",
                operator_review_required=True,
            )
        )

    alerts.sort(key=lambda item: severity_rank(item.get("alert_severity")), reverse=True)
    strongest = alerts[0] if alerts else None
    return [
        *alerts,
        _alert(
            "operational-summary",
            SEVERITY_INFO,
            "summary",
            "Operational alert summary: " + ", ".join(
                compact_list(
                    [
                        strongest.get("alert_summary") if strongest else None,
                        control_state.get("current_operating_mode"),
                        shadow_mode.get("shadow_mode_status"),
                    ],
                    limit=3,
                )
            ),
            "Use the highest-severity active alert as the current operator focus.",
            operator_review_required=False,
            escalation_path="summary_only",
        ),
    ]
