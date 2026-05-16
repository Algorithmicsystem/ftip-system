from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Sequence

from api.assistant import reports
from api.assistant.phase6 import PREDICTION_RECORD_KIND
from api.assistant.phase12 import OPERATIONAL_INCIDENT_ARTIFACT_KIND, SHADOW_DECISION_RECORD_KIND
from api.assistant.storage import AssistantStorage
from api.research.backtest import build_validation_artifact

from .common import OPERATING_WORKFLOW_ARTIFACT_KIND, OPERATING_WORKFLOW_VERSION, as_datetime, compact_list, now_utc
from .daily import build_daily_workflow
from .journal import build_shadow_decision_journal
from .monthly import build_monthly_refinement
from .postmortem import build_postmortem_report
from .runbook import build_operator_runbook
from .trust import build_trust_maintenance
from .weekly import build_weekly_review


def _load_reports(
    store: AssistantStorage,
    *,
    session_id: Optional[str],
    limit: int = 80,
) -> List[Dict[str, Any]]:
    artifacts = store.list_artifacts(kind=reports.ANALYSIS_REPORT_KIND, session_id=session_id, limit=limit)
    output: List[Dict[str, Any]] = []
    for artifact in artifacts:
        payload = dict(artifact.get("payload") or {})
        payload["report_id"] = artifact.get("id")
        payload["session_id"] = artifact.get("session_id")
        payload["_artifact_created_at"] = artifact.get("created_at")
        output.append(payload)
    return output


def _load_payloads(
    store: AssistantStorage,
    *,
    kind: str,
    session_id: Optional[str],
    limit: int = 200,
) -> List[Dict[str, Any]]:
    artifacts = store.list_artifacts(kind=kind, session_id=session_id, limit=limit)
    output: List[Dict[str, Any]] = []
    for artifact in artifacts:
        payload = dict(artifact.get("payload") or {})
        payload["_artifact_created_at"] = artifact.get("created_at")
        output.append(payload)
    return output


def _within_days(rows: Sequence[Dict[str, Any]], *, days: int) -> List[Dict[str, Any]]:
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    output: List[Dict[str, Any]] = []
    for row in rows:
        timestamp = (
            as_datetime(row.get("recorded_at"))
            or as_datetime(row.get("generated_at"))
            or as_datetime(row.get("created_at"))
            or as_datetime(row.get("_artifact_created_at"))
        )
        if timestamp is None:
            continue
        if timestamp >= cutoff:
            output.append(row)
    return output


def _build_window_validation(
    records: Sequence[Dict[str, Any]],
    *,
    horizon: Optional[str],
    risk_mode: Optional[str],
    min_sample_size: int,
) -> Dict[str, Any]:
    if not records:
        return {
            "status": "limited",
            "prediction_linkage_summary": {"total_predictions": 0, "matured_count": 0},
            "net_return_summary": {},
            "walkforward_summary": {"window_count": 0},
            "readiness_scorecard": {},
            "suppression_effect_summary": {},
            "failure_modes": [],
            "strongest_conditions": [],
            "weakest_conditions": [],
        }
    return build_validation_artifact(
        records=list(records),
        cohort_horizon=horizon,
        cohort_risk_mode=risk_mode,
        min_sample_size=min_sample_size,
    )


def _operating_summary(
    daily: Dict[str, Any],
    weekly: Dict[str, Any],
    monthly: Dict[str, Any],
    trust: Dict[str, Any],
    postmortem: Dict[str, Any],
) -> str:
    return (
        f"{daily.get('daily_operating_summary')} "
        f"{weekly.get('weekly_operating_summary')} "
        f"{monthly.get('monthly_operating_summary')} "
        f"{trust.get('trust_maintenance_summary')} "
        f"{postmortem.get('postmortem_summary')}"
    )


def build_operating_workflow_artifact(
    *,
    current_report: Dict[str, Any],
    current_report_id: Optional[str],
    session_id: Optional[str],
    store: AssistantStorage,
) -> Dict[str, Any]:
    recent_reports = _load_reports(store, session_id=session_id)
    prediction_records = _load_payloads(
        store,
        kind=PREDICTION_RECORD_KIND,
        session_id=session_id,
        limit=300,
    )
    shadow_records = _load_payloads(
        store,
        kind=SHADOW_DECISION_RECORD_KIND,
        session_id=session_id,
        limit=160,
    )
    incidents = _load_payloads(
        store,
        kind=OPERATIONAL_INCIDENT_ARTIFACT_KIND,
        session_id=session_id,
        limit=120,
    )

    weekly_predictions = _within_days(prediction_records, days=7)
    monthly_predictions = _within_days(prediction_records, days=30)
    weekly_shadows = _within_days(shadow_records, days=7)
    monthly_shadows = _within_days(shadow_records, days=30)
    weekly_incidents = _within_days(incidents, days=7)
    monthly_incidents = _within_days(incidents, days=30)

    weekly_validation = _build_window_validation(
        weekly_predictions,
        horizon=current_report.get("horizon"),
        risk_mode=current_report.get("risk_mode"),
        min_sample_size=4,
    )
    monthly_validation = _build_window_validation(
        monthly_predictions,
        horizon=current_report.get("horizon"),
        risk_mode=current_report.get("risk_mode"),
        min_sample_size=6,
    )

    current_shadow_record = (
        (current_report.get("operational_guardrails") or {}).get("shadow_decision_record")
        or {}
    )
    daily = build_daily_workflow(
        current_report,
        current_report_id=str(current_report_id or ""),
        recent_reports=recent_reports,
        recent_shadow_records=weekly_shadows,
        recent_incidents=weekly_incidents,
    )
    weekly = build_weekly_review(
        current_report,
        weekly_validation=weekly_validation,
        recent_shadow_records=weekly_shadows,
        recent_incidents=weekly_incidents,
    )
    monthly = build_monthly_refinement(
        current_report,
        monthly_validation=monthly_validation,
        recent_incidents=monthly_incidents,
    )
    journal = build_shadow_decision_journal(
        current_report,
        current_shadow_record=current_shadow_record,
        recent_shadow_records=monthly_shadows,
    )
    postmortem = build_postmortem_report(
        current_report,
        weekly_validation=weekly_validation,
        monthly_validation=monthly_validation,
        recent_incidents=monthly_incidents,
    )
    trust = build_trust_maintenance(current_report)
    monthly["trust_promotion_candidates"] = trust.get("trust_promotion_candidates") or []
    monthly["trust_demotion_candidates"] = trust.get("trust_demotion_candidates") or []
    runbook = build_operator_runbook(current_report)

    operator_attention_items = compact_list(
        [
            *(daily.get("daily_operator_attention_items") or []),
            *(weekly.get("weekly_operator_attention_items") or []),
            *(postmortem.get("postmortem_queue") or []),
            *(trust.get("trust_recovery_checklist") or []),
        ],
        limit=10,
    )

    return {
        "operating_workflow_kind": OPERATING_WORKFLOW_ARTIFACT_KIND,
        "operating_workflow_version": OPERATING_WORKFLOW_VERSION,
        "generated_at": now_utc(),
        "current_report_id": current_report_id,
        "daily_workflow": daily,
        "weekly_validation": weekly_validation,
        "monthly_validation": monthly_validation,
        **daily,
        **weekly,
        **monthly,
        **journal,
        **postmortem,
        **trust,
        **runbook,
        "operator_attention_items": operator_attention_items,
        "operating_workflow_summary": _operating_summary(
            daily,
            weekly,
            monthly,
            trust,
            postmortem,
        ),
    }
