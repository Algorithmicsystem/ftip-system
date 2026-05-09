from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .common import (
    HEALTH_SNAPSHOT_ARTIFACT_KIND,
    OPERATIONAL_INCIDENT_ARTIFACT_KIND,
    SHADOW_DECISION_RECORD_KIND,
    compact_list,
)


def build_health_snapshot_record(
    current_report: Dict[str, Any],
    *,
    health_snapshot: Dict[str, Any],
    control_state: Dict[str, Any],
    shadow_mode: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "kind": HEALTH_SNAPSHOT_ARTIFACT_KIND,
        "captured_at": health_snapshot.get("captured_at"),
        "symbol": current_report.get("symbol"),
        "as_of_date": current_report.get("as_of_date"),
        "horizon": current_report.get("horizon"),
        "risk_mode": current_report.get("risk_mode"),
        "system_health_status": health_snapshot.get("system_health_status"),
        "provider_health_status": health_snapshot.get("provider_health_status"),
        "data_pipeline_health": health_snapshot.get("data_pipeline_health"),
        "artifact_pipeline_health": health_snapshot.get("artifact_pipeline_health"),
        "data_reliability_score": health_snapshot.get("data_reliability_score"),
        "current_operating_mode": control_state.get("current_operating_mode"),
        "shadow_mode_status": shadow_mode.get("shadow_mode_status"),
    }


def build_operational_incidents(
    current_report: Dict[str, Any],
    *,
    alerts: Sequence[Dict[str, Any]],
    control_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    incidents: List[Dict[str, Any]] = []
    for item in alerts:
        if not item.get("operator_review_required"):
            continue
        incidents.append(
            {
                "kind": OPERATIONAL_INCIDENT_ARTIFACT_KIND,
                "recorded_at": current_report.get("generated_at"),
                "symbol": current_report.get("symbol"),
                "as_of_date": current_report.get("as_of_date"),
                "event_type": (
                    "pause_event"
                    if item.get("alert_domain") == "kill_switch"
                    and item.get("alert_severity") == "critical_pause"
                    else "degradation_event"
                ),
                "alert_id": item.get("alert_id"),
                "alert_domain": item.get("alert_domain"),
                "alert_severity": item.get("alert_severity"),
                "summary": item.get("alert_summary"),
                "recommended_action": item.get("recommended_action"),
                "current_operating_mode": control_state.get("current_operating_mode"),
            }
        )
    return incidents[:8]


def summarize_incident_history(
    *,
    prior_incidents: Sequence[Dict[str, Any]],
    current_incidents: Sequence[Dict[str, Any]],
    limit: int = 8,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for item in [*current_incidents, *prior_incidents]:
        summary = item.get("summary") or item.get("alert_summary")
        if not summary:
            continue
        output.append(
            {
                "recorded_at": item.get("recorded_at") or item.get("created_at"),
                "alert_domain": item.get("alert_domain") or item.get("event_type"),
                "alert_severity": item.get("alert_severity") or "info",
                "summary": summary,
                "recommended_action": item.get("recommended_action"),
            }
        )
        if len(output) >= limit:
            break
    return output
