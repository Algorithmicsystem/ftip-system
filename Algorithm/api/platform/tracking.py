from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import (
    PaperTradeRecord,
    RecommendationTrack,
    TrackingStatus,
)


TRACKING_HORIZON_DAYS: Mapping[str, int] = {
    "5d": 5,
    "21d": 21,
    "63d": 63,
}


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def normalize_horizons(horizons: Iterable[str]) -> List[str]:
    output: List[str] = []
    for raw in horizons:
        label = str(raw or "").strip().lower()
        if not label:
            continue
        if label not in TRACKING_HORIZON_DAYS:
            continue
        if label not in output:
            output.append(label)
    return output or ["5d", "21d", "63d"]


def build_tracking_status(
    *,
    status: str = "active",
    evidence_mode: str = "paper_tracked",
    matured_horizons: Optional[Iterable[str]] = None,
    pending_horizons: Optional[Iterable[str]] = None,
    evidence_status: str = "pending",
    warnings: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    last_refreshed_at: Optional[str] = None,
) -> TrackingStatus:
    return TrackingStatus(
        status=status,
        evidence_mode=evidence_mode,
        matured_horizons=normalize_horizons(matured_horizons or []),
        pending_horizons=normalize_horizons(pending_horizons or []),
        evidence_status=evidence_status,
        last_refreshed_at=last_refreshed_at or now_utc(),
        warnings=[str(item) for item in warnings or [] if str(item or "").strip()],
        metadata=sanitize_payload(metadata or {}),
    )


def build_recommendation_track(
    *,
    organization_id: Optional[str],
    workspace_id: Optional[str],
    workflow_id: str,
    dossier_id: str,
    entity_id: Optional[str],
    symbol: str,
    axiom_artifact_id: Optional[str],
    axiom_history_artifact_id: Optional[str],
    report_id: Optional[str],
    session_id: Optional[str],
    recommendation_state_at_start: str,
    deployability_tier_at_start: Optional[str],
    size_band_at_start: Optional[str],
    regime_label: Optional[str],
    trade_family: Optional[str],
    strongest_engine_at_start: Optional[str],
    weakest_engine_at_start: Optional[str],
    signal_action_at_start: Optional[str],
    evidence_status_at_start: Optional[str],
    start_deployable_alpha_utility: Optional[float],
    start_validated_edge: Optional[float],
    start_overall_coverage: Optional[float],
    start_overall_confidence: Optional[float],
    start_engine_scores: Optional[Dict[str, Any]],
    start_source_context: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    tracking_start_at: Optional[str] = None,
) -> RecommendationTrack:
    created_at = now_utc()
    return RecommendationTrack(
        track_id=str(uuid.uuid4()),
        organization_id=organization_id,
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        dossier_id=dossier_id,
        entity_id=entity_id,
        symbol=str(symbol or "").upper(),
        axiom_artifact_id=axiom_artifact_id,
        axiom_history_artifact_id=axiom_history_artifact_id,
        report_id=report_id,
        session_id=session_id,
        recommendation_state_at_start=str(recommendation_state_at_start or "draft"),
        deployability_tier_at_start=deployability_tier_at_start,
        size_band_at_start=size_band_at_start,
        regime_label=regime_label,
        trade_family=trade_family,
        strongest_engine_at_start=strongest_engine_at_start,
        weakest_engine_at_start=weakest_engine_at_start,
        signal_action_at_start=signal_action_at_start,
        evidence_status_at_start=evidence_status_at_start,
        start_deployable_alpha_utility=start_deployable_alpha_utility,
        start_validated_edge=start_validated_edge,
        start_overall_coverage=start_overall_coverage,
        start_overall_confidence=start_overall_confidence,
        start_engine_scores=sanitize_payload(start_engine_scores or {}),
        start_source_context=sanitize_payload(start_source_context or {}),
        created_at=created_at,
        tracking_start_at=tracking_start_at or created_at,
        tracking_status=build_tracking_status(
            status="active",
            evidence_status="pending",
            pending_horizons=TRACKING_HORIZON_DAYS.keys(),
            metadata={"created_from": "platform_phase9a"},
        ),
        metadata=sanitize_payload(metadata or {}),
    )


def build_paper_trade_record(
    *,
    track: Dict[str, Any],
    entry_reference_date: str,
    entry_price: Optional[float],
    tracked_horizons: Iterable[str],
    thesis_state_at_entry: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PaperTradeRecord:
    created_at = now_utc()
    return PaperTradeRecord(
        paper_trade_id=str(uuid.uuid4()),
        track_id=str(track.get("track_id") or ""),
        organization_id=track.get("organization_id"),
        workspace_id=track.get("workspace_id"),
        workflow_id=str(track.get("workflow_id") or ""),
        dossier_id=str(track.get("dossier_id") or ""),
        symbol=str(track.get("symbol") or "").upper(),
        entry_reference_date=entry_reference_date,
        entry_price=entry_price,
        thesis_state_at_entry=sanitize_payload(thesis_state_at_entry or {}),
        tracked_horizons=normalize_horizons(tracked_horizons),
        current_status="active",
        outcome_summary={},
        metadata=sanitize_payload(metadata or {}),
        created_at=created_at,
        updated_at=created_at,
    )

