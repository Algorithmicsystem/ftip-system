from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.assistant.storage import AssistantStorage

from .alerts import build_operational_alerts
from .audit import (
    build_health_snapshot_record,
    build_operational_incidents,
    summarize_incident_history,
)
from .common import (
    HEALTH_SNAPSHOT_ARTIFACT_KIND,
    OPERATIONAL_GUARDRAILS_ARTIFACT_KIND,
    OPERATIONAL_GUARDRAILS_VERSION,
    OPERATIONAL_INCIDENT_ARTIFACT_KIND,
    SHADOW_DECISION_RECORD_KIND,
    compact_list,
    dedupe_dicts_by_key,
    now_utc,
)
from .controls import build_control_state
from .drift import build_operational_drift_monitor
from .health import build_health_snapshot
from .shadow import build_shadow_decision_record, build_shadow_mode_summary


def _load_prior_shadow_records(
    store: AssistantStorage,
    *,
    symbol: Optional[str],
    horizon: Optional[str],
    risk_mode: Optional[str],
    limit: int = 60,
) -> List[Dict[str, Any]]:
    artifacts = store.list_artifacts(kind=SHADOW_DECISION_RECORD_KIND, limit=limit)
    output: List[Dict[str, Any]] = []
    for artifact in artifacts:
        payload = artifact.get("payload") or {}
        if symbol and str(payload.get("symbol")) != str(symbol):
            continue
        if horizon and str(payload.get("horizon")) != str(horizon):
            continue
        if risk_mode and str(payload.get("risk_mode")) != str(risk_mode):
            continue
        output.append(payload)
    return output


def _load_prior_incidents(
    store: AssistantStorage,
    *,
    limit: int = 40,
) -> List[Dict[str, Any]]:
    artifacts = store.list_artifacts(kind=OPERATIONAL_INCIDENT_ARTIFACT_KIND, limit=limit)
    return [artifact.get("payload") or {} for artifact in artifacts]


def _load_prior_health_snapshots(
    store: AssistantStorage,
    *,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    artifacts = store.list_artifacts(kind=HEALTH_SNAPSHOT_ARTIFACT_KIND, limit=limit)
    return [artifact.get("payload") or {} for artifact in artifacts]


def _system_health_summary(
    health_snapshot: Dict[str, Any],
    control_state: Dict[str, Any],
) -> str:
    return (
        f"System health is {health_snapshot.get('system_health_status')} with data reliability {health_snapshot.get('data_reliability_score')} / 100, "
        f"provider health {health_snapshot.get('provider_health_status')}, data pipeline {health_snapshot.get('data_pipeline_health')}, and artifact pipeline {health_snapshot.get('artifact_pipeline_health')}. "
        f"Current operating mode is {control_state.get('current_operating_mode')}."
    )


def _shadow_mode_summary(shadow_mode: Dict[str, Any]) -> str:
    cohort = shadow_mode.get("shadow_cohort") or {}
    return (
        f"Shadow mode is {shadow_mode.get('shadow_mode_status')} with {cohort.get('tracked_shadow_decisions') or 0} tracked shadow decisions, "
        f"{cohort.get('paper_only_count') or 0} paper-only cases, {cohort.get('live_like_count') or 0} live-like candidates, and reliability {shadow_mode.get('shadow_reliability_score')} / 100."
    )


def _drift_control_summary(
    drift_monitor: Dict[str, Any],
    control_state: Dict[str, Any],
) -> str:
    return (
        f"Model drift score is {drift_monitor.get('model_drift_score')} / 100 and environment shift is {drift_monitor.get('environment_shift_score')} / 100. "
        f"Calibration health is {drift_monitor.get('calibration_health_status')}. "
        f"Pause required is {control_state.get('pause_required')} and downgrade-to-shadow recommended is {control_state.get('downgrade_to_shadow_recommended')}."
    )


def _incident_history_summary(
    incident_history: List[Dict[str, Any]],
    alerts: List[Dict[str, Any]],
) -> str:
    top_alert = alerts[0] if alerts else {}
    return (
        f"There are {len(incident_history)} recent operational incidents or active operator-review alerts. "
        f"Highest-severity focus is {top_alert.get('alert_domain') or 'none'}: {top_alert.get('alert_summary') or 'no acute alert'}."
    )


def build_operational_guardrails_artifact(
    *,
    current_report: Dict[str, Any],
    current_report_id: Optional[str],
    session_id: Optional[str],
    store: AssistantStorage,
) -> Dict[str, Any]:
    prior_shadow_records = _load_prior_shadow_records(
        store,
        symbol=current_report.get("symbol"),
        horizon=current_report.get("horizon"),
        risk_mode=current_report.get("risk_mode"),
    )
    prior_incidents = _load_prior_incidents(store)
    prior_health_snapshots = _load_prior_health_snapshots(store)
    health_snapshot = build_health_snapshot(current_report)
    shadow_decision_record = build_shadow_decision_record(
        current_report,
        report_id=current_report_id,
        session_id=session_id,
    )
    shadow_mode = build_shadow_mode_summary(
        current_report,
        shadow_decision_record=shadow_decision_record,
        prior_shadow_records=prior_shadow_records,
    )
    drift_monitor = build_operational_drift_monitor(
        current_report,
        health_snapshot=health_snapshot,
    )
    control_state = build_control_state(
        current_report,
        health_snapshot=health_snapshot,
        shadow_mode=shadow_mode,
        drift_monitor=drift_monitor,
    )
    alerts = build_operational_alerts(
        current_report,
        health_snapshot=health_snapshot,
        shadow_mode=shadow_mode,
        drift_monitor=drift_monitor,
        control_state=control_state,
    )
    current_incidents = build_operational_incidents(
        current_report,
        alerts=alerts,
        control_state=control_state,
    )
    incident_history = summarize_incident_history(
        prior_incidents=prior_incidents,
        current_incidents=current_incidents,
    )
    health_snapshot_record = build_health_snapshot_record(
        current_report,
        health_snapshot=health_snapshot,
        control_state=control_state,
        shadow_mode=shadow_mode,
    )
    strongest_alert = alerts[0] if alerts else {}
    provider_degradation_notes = compact_list(
        health_snapshot.get("provider_degradation_notes") or [],
        limit=8,
    )
    degraded_domains = compact_list(
        health_snapshot.get("degraded_domain_list") or [],
        limit=8,
    )
    return {
        "operational_guardrails_kind": OPERATIONAL_GUARDRAILS_ARTIFACT_KIND,
        "operational_guardrails_version": OPERATIONAL_GUARDRAILS_VERSION,
        "generated_at": now_utc(),
        "system_health": health_snapshot,
        "shadow_mode": shadow_mode,
        "drift_monitoring": drift_monitor,
        "control_state": control_state,
        "operational_alerts": alerts,
        "incident_history": incident_history,
        "health_snapshot": health_snapshot_record,
        "shadow_decision_record": shadow_decision_record,
        "current_incidents": current_incidents,
        "prior_operational_context": {
            "prior_shadow_records": len(prior_shadow_records),
            "prior_incidents": len(prior_incidents),
            "prior_health_snapshots": len(prior_health_snapshots),
        },
        "system_health_summary": _system_health_summary(
            health_snapshot,
            control_state,
        ),
        "shadow_mode_summary": _shadow_mode_summary(shadow_mode),
        "drift_control_summary": _drift_control_summary(
            drift_monitor,
            control_state,
        ),
        "incident_history_summary": _incident_history_summary(
            incident_history,
            alerts,
        ),
        "provider_degradation_notes": provider_degradation_notes,
        "degraded_domain_list": degraded_domains,
        "active_alert": strongest_alert,
        "current_operating_mode": control_state.get("current_operating_mode"),
        "pause_required": control_state.get("pause_required"),
        "downgrade_to_shadow_recommended": control_state.get(
            "downgrade_to_shadow_recommended"
        ),
        "operator_attention_required": control_state.get(
            "operator_attention_required"
        ),
    }
