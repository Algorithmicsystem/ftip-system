from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, Iterable, List, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import AnalysisLink, DossierRecord, DossierSection, WorkflowStageState


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _text(value: Any, default: str = "") -> str:
    if value in (None, "", [], {}):
        return default
    return str(value)


def build_analysis_link(
    *,
    report: Dict[str, Any],
    report_id: Optional[str],
    session_id: Optional[str],
    axiom_artifact_id: Optional[str],
    axiom_report_pack_artifact_id: Optional[str],
    axiom_lineage_artifact_id: Optional[str],
    axiom_history_artifact_id: Optional[str] = None,
    axiom_calibration_artifact_id: Optional[str] = None,
) -> AnalysisLink:
    return AnalysisLink(
        link_id=str(uuid.uuid4()),
        report_id=report_id,
        session_id=session_id,
        axiom_artifact_id=axiom_artifact_id,
        axiom_report_pack_artifact_id=axiom_report_pack_artifact_id,
        axiom_lineage_artifact_id=axiom_lineage_artifact_id,
        axiom_history_artifact_id=axiom_history_artifact_id,
        axiom_calibration_artifact_id=axiom_calibration_artifact_id,
        linked_at=now_utc(),
        source_summary={
            "symbol": report.get("symbol"),
            "as_of_date": report.get("as_of_date"),
            "axiom_framework_version": report.get("axiom_framework_version"),
            "axiom_deployability_tier": report.get("axiom_deployability_tier"),
            "axiom_regime_label": report.get("axiom_regime_label"),
        },
    )


def _section(
    key: str,
    title: str,
    summary: Any,
    *,
    payload: Optional[Dict[str, Any]] = None,
    status: str = "available",
) -> DossierSection:
    text = _text(summary, "No current summary available.")
    return DossierSection(
        section_key=key,
        title=title,
        summary=text,
        status=status if text else "partial",
        payload=sanitize_payload(payload or {}),
    )


def build_dossier_sections(report: Dict[str, Any]) -> List[DossierSection]:
    axiom = report.get("axiom") or {}
    summary_card = report.get("axiom_summary_card") or {}
    historical = report.get("axiom_historical_evidence_report") or report.get("axiom_historical_evidence") or {}
    monitoring_state = {
        "monitoring_triggers": report.get("monitoring_triggers") or [],
        "invalidation_flags": axiom.get("invalidation_flags") or [],
        "downgrade_triggers": (report.get("axiom_risk_deployability_memo") or {}).get("downgrade_triggers") or [],
    }
    sections = [
        _section(
            "executive_summary",
            "Executive Summary",
            report.get("overall_analysis") or report.get("axiom_summary"),
            payload={"summary_card": summary_card},
        ),
        _section(
            "axiom_scorecard",
            "AXIOM Scorecard",
            report.get("axiom_summary_card_text") or report.get("axiom_summary"),
            payload=summary_card,
        ),
        _section(
            "regime_trade_setup",
            "Regime & Trade Setup",
            report.get("axiom_regime_trade_family_summary") or report.get("strategy_view"),
            payload={
                "regime_label": report.get("axiom_regime_label"),
                "trade_family": report.get("axiom_trade_family"),
                "deployability_tier": report.get("axiom_deployability_tier"),
            },
        ),
        _section(
            "fragility_risk",
            "Fragility & Risk",
            report.get("axiom_risk_deployability_memo_summary") or report.get("risk_quality_analysis"),
            payload=(report.get("axiom_risk_deployability_memo") or {}).get("fragility_engine") or {},
        ),
        _section(
            "liquidity_execution",
            "Liquidity & Execution",
            report.get("execution_quality_analysis"),
            payload=(report.get("axiom_risk_deployability_memo") or {}).get("liquidity_convexity_engine") or {},
        ),
        _section(
            "historical_evidence",
            "Historical Evidence",
            report.get("axiom_historical_evidence_summary_text") or report.get("axiom_calibration_summary_text"),
            payload=historical,
            status="available" if historical else "partial",
        ),
        _section(
            "portfolio_fit",
            "Portfolio Fit",
            report.get("axiom_portfolio_governance_summary") or report.get("portfolio_fit_analysis"),
            payload=report.get("axiom_portfolio_governance") or {},
        ),
        _section(
            "decision_status",
            "Decision Status",
            report.get("deployment_permission_analysis") or report.get("deployment_permission"),
            payload={
                "deployability_tier": report.get("axiom_evidence_backed_deployability_tier")
                or report.get("axiom_deployability_tier"),
                "size_band": report.get("axiom_final_size_band")
                or report.get("axiom_size_band_recommendation"),
                "deployment_permission": report.get("deployment_permission"),
            },
        ),
        _section(
            "monitoring_triggers",
            "Monitoring Triggers",
            ", ".join(monitoring_state["monitoring_triggers"]) or "No monitoring triggers are currently attached.",
            payload=monitoring_state,
        ),
        _section(
            "lineage_summary",
            "Lineage Summary",
            report.get("axiom_lineage_summary") or report.get("evidence_provenance"),
            payload=report.get("axiom_lineage") or {},
        ),
    ]
    return sections


def build_dossier_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    summary_card = report.get("axiom_summary_card") or {}
    return sanitize_payload(
        {
            "symbol": report.get("symbol"),
            "as_of_date": report.get("as_of_date"),
            "signal_summary": report.get("signal_summary"),
            "overall_analysis": report.get("overall_analysis"),
            "axiom_summary_card": summary_card,
            "deployable_alpha_utility": report.get("axiom_deployable_alpha_utility")
            or summary_card.get("deployable_alpha_utility"),
            "validated_edge": report.get("axiom_validated_edge")
            or summary_card.get("validated_edge"),
            "deployability_tier": report.get("axiom_evidence_backed_deployability_tier")
            or report.get("axiom_deployability_tier"),
            "regime_label": report.get("axiom_regime_label"),
            "trade_family": report.get("axiom_trade_family"),
            "size_band": report.get("axiom_final_size_band")
            or report.get("axiom_size_band_recommendation"),
            "evidence_status": report.get("axiom_calibration_status")
            or summary_card.get("evidence_status")
            or "partial",
            "portfolio_fit_label": report.get("axiom_portfolio_fit_label"),
            "monitoring_triggers": report.get("monitoring_triggers") or [],
        }
    )


def build_monitoring_state(report: Dict[str, Any]) -> Dict[str, Any]:
    risk_memo = report.get("axiom_risk_deployability_memo") or {}
    return sanitize_payload(
        {
            "monitoring_triggers": report.get("monitoring_triggers") or [],
            "downgrade_triggers": risk_memo.get("downgrade_triggers") or [],
            "invalidation_flags": (report.get("axiom") or {}).get("invalidation_flags") or [],
            "current_operating_mode": report.get("current_operating_mode"),
        }
    )


def build_dossier_record(
    *,
    dossier_id: str,
    workflow_id: str,
    entity_id: str,
    dossier_type: str,
    title: str,
    report: Dict[str, Any],
    analysis_link: AnalysisLink,
    workflow_stage_state: Optional[WorkflowStageState],
    metadata: Optional[Dict[str, Any]] = None,
) -> DossierRecord:
    summary = build_dossier_summary(report)
    sections = build_dossier_sections(report)
    return DossierRecord(
        dossier_id=dossier_id,
        workflow_id=workflow_id,
        entity_id=entity_id,
        dossier_type=dossier_type,
        title=title,
        current_summary=summary,
        latest_analysis_link=analysis_link,
        latest_axiom_analysis_id=analysis_link.axiom_artifact_id,
        latest_deployability_tier=_text(summary.get("deployability_tier")) or None,
        latest_regime_label=_text(summary.get("regime_label")) or None,
        latest_trade_family=_text(summary.get("trade_family")) or None,
        latest_size_band=_text(summary.get("size_band")) or None,
        evidence_status=_text(summary.get("evidence_status"), "partial"),
        workflow_stage_state=workflow_stage_state,
        sections=sections,
        monitoring_state=build_monitoring_state(report),
        historical_evidence_summary=sanitize_payload(report.get("axiom_historical_evidence_report") or report.get("axiom_historical_evidence") or {}),
        lineage_summary=sanitize_payload(report.get("axiom_lineage") or {}),
        metadata=sanitize_payload(metadata or {}),
        created_at=now_utc(),
        updated_at=now_utc(),
    )


def refresh_dossier_record(
    dossier: Dict[str, Any],
    *,
    report: Dict[str, Any],
    analysis_link: AnalysisLink,
    workflow_stage_state: Optional[WorkflowStageState] = None,
) -> Dict[str, Any]:
    existing = DossierRecord.model_validate(dossier)
    updated = build_dossier_record(
        dossier_id=existing.dossier_id,
        workflow_id=existing.workflow_id,
        entity_id=existing.entity_id,
        dossier_type=existing.dossier_type,
        title=existing.title,
        report=report,
        analysis_link=analysis_link,
        workflow_stage_state=workflow_stage_state or existing.workflow_stage_state,
        metadata={
            **dict(existing.metadata or {}),
            "analysis_link_count": int((existing.metadata or {}).get("analysis_link_count") or 0) + 1,
        },
    )
    payload = updated.model_dump(mode="python")
    payload["created_at"] = existing.created_at
    payload["updated_at"] = now_utc()
    return sanitize_payload(payload)


def dossier_preview(dossier: Dict[str, Any]) -> Dict[str, Any]:
    record = DossierRecord.model_validate(dossier)
    recommendation_state = (
        ((record.metadata or {}).get("recommendation_state") or {}).get("state")
        or record.current_summary.get("recommendation_state")
    )
    latest_committee_decision = (record.metadata or {}).get("latest_committee_decision") or {}
    return {
        "dossier_id": record.dossier_id,
        "title": record.title,
        "dossier_type": record.dossier_type,
        "latest_deployability_tier": record.latest_deployability_tier,
        "latest_regime_label": record.latest_regime_label,
        "latest_trade_family": record.latest_trade_family,
        "latest_size_band": record.latest_size_band,
        "evidence_status": record.evidence_status,
        "stage": (record.workflow_stage_state or WorkflowStageState(stage="unknown")).stage,
        "status": (record.workflow_stage_state or WorkflowStageState(stage="unknown")).status,
        "symbol": record.current_summary.get("symbol"),
        "as_of_date": record.current_summary.get("as_of_date"),
        "recommendation_state": recommendation_state,
        "recommendation_locked": bool(
            ((record.metadata or {}).get("recommendation_state") or {}).get("locked")
        ),
        "unresolved_concern_count": int(
            (record.current_summary or {}).get("unresolved_concern_count") or 0
        ),
        "latest_committee_decision_status": latest_committee_decision.get("decision_status"),
    }
