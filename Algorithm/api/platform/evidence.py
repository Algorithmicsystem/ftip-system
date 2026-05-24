from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from api.axiom.analytics import (
    build_bucket_summary,
    build_group_summary,
    matured_records,
    outcome_metrics,
    safe_float,
)
from api.assistant.reports import sanitize_payload
from api.platform.benchmarks import (
    build_benchmark_comparison_summary,
    history_like_records,
    latest_snapshots_by_track,
)
from api.platform.contracts import (
    CalibrationHardeningSummary,
    DriftEvidenceSummary,
    ModelCredibilitySnapshot,
    RecommendationEvidenceSummary,
    StabilityAssessment,
    WorkspaceProofSummary,
)


def _evidence_maturity_level(matured_count: int) -> str:
    if matured_count >= 24:
        return "developing"
    if matured_count >= 8:
        return "emerging"
    if matured_count >= 1:
        return "limited"
    return "insufficient"


def _paper_status_counts(snapshots: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"supportive": 0, "mixed": 0, "weak": 0, "insufficient_sample": 0}
    for snapshot in latest_snapshots_by_track(snapshots):
        status = str(((snapshot.get("evidence_status") or {}).get("status")) or "insufficient_sample")
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_recommendation_evidence_summary(
    *,
    dossier_id: str,
    track: Dict[str, Any],
    paper_trade: Dict[str, Any],
    snapshot: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not snapshot:
        return RecommendationEvidenceSummary(
            dossier_id=dossier_id,
            track_id=str(track.get("track_id") or ""),
            paper_trade_id=str(paper_trade.get("paper_trade_id") or ""),
            symbol=str(track.get("symbol") or "").upper(),
            evidence_status="insufficient_sample",
            tracking_status=str((track.get("tracking_status") or {}).get("status") or "active"),
            summary="Paper tracking has started, but no realized outcome snapshot is available yet.",
            warnings=["no realized paper-trade snapshot is attached yet"],
        ).model_dump(mode="python")
    assessment = dict(snapshot.get("assessment") or {})
    evidence = dict(snapshot.get("evidence_status") or {})
    return RecommendationEvidenceSummary(
        dossier_id=dossier_id,
        track_id=str(track.get("track_id") or ""),
        paper_trade_id=str(paper_trade.get("paper_trade_id") or ""),
        symbol=str(track.get("symbol") or "").upper(),
        evidence_status=str(evidence.get("status") or "partial"),
        tracking_status=str((snapshot.get("tracking_status") or {}).get("status") or "active"),
        supportive_horizons=list(assessment.get("supportive_horizons") or []),
        contradicted_horizons=list(assessment.get("contradicted_horizons") or []),
        summary=str(assessment.get("summary") or evidence.get("rationale") or "Paper evidence is not yet mature."),
        warnings=list(evidence.get("warnings") or []),
    ).model_dump(mode="python")


def build_drift_evidence_summary(
    *,
    workspace_id: Optional[str],
    organization_id: Optional[str],
    records: Sequence[Dict[str, Any]],
    horizon_label: str = "21d",
) -> Dict[str, Any]:
    matured = matured_records(records, horizon_label=horizon_label)
    if len(matured) < 8:
        return DriftEvidenceSummary(
            workspace_id=workspace_id,
            organization_id=organization_id,
            status="insufficient_sample",
            sample_count=len(records),
            matured_count=len(matured),
            summary="Drift evidence is sample-constrained and cannot yet compare recent vs older behavior reliably.",
            warnings=["insufficient matured tracked recommendations for drift analysis"],
        ).model_dump(mode="python")

    rows = sorted(matured, key=lambda item: str(item.get("as_of_date") or ""))
    midpoint = max(1, len(rows) // 2)
    older = rows[:midpoint]
    recent = rows[midpoint:]
    older_metrics = outcome_metrics(older, horizon_label=horizon_label)
    recent_metrics = outcome_metrics(recent, horizon_label=horizon_label)
    older_return = safe_float(older_metrics.get("average_net_edge_return"))
    recent_return = safe_float(recent_metrics.get("average_net_edge_return"))
    delta = None if older_return is None or recent_return is None else round(recent_return - older_return, 6)
    status = "stable"
    warnings: List[str] = []
    if delta is not None and delta <= -0.03:
        status = "degrading"
        warnings.append("recent tracked paper evidence is materially weaker than older evidence")
    elif delta is not None and delta >= 0.03:
        status = "improving"
    stability = StabilityAssessment(
        dimension="overall",
        label=horizon_label,
        status=status,
        older_sample_count=int(older_metrics.get("matured_count") or 0),
        recent_sample_count=int(recent_metrics.get("matured_count") or 0),
        older_average_net_edge_return=older_return,
        recent_average_net_edge_return=recent_return,
        delta=delta,
        summary=(
            f"Older average net edge is {older_return}, recent average net edge is {recent_return}."
        ),
        warnings=warnings,
    )
    return DriftEvidenceSummary(
        workspace_id=workspace_id,
        organization_id=organization_id,
        status=status,
        sample_count=len(records),
        matured_count=len(matured),
        stability_assessments=[stability],
        warnings=warnings,
        summary=(
            "Recent paper-tracked evidence is degrading versus older cohorts."
            if status == "degrading"
            else "Recent paper-tracked evidence is improving versus older cohorts."
            if status == "improving"
            else "Recent and older paper-tracked evidence are broadly stable."
        ),
    ).model_dump(mode="python")


def build_calibration_hardening_summary(
    *,
    platform_version: str,
    workspace_id: Optional[str],
    organization_id: Optional[str],
    tracks: Sequence[Dict[str, Any]],
    paper_trades: Sequence[Dict[str, Any]],
    snapshots: Sequence[Dict[str, Any]],
    horizon_label: str = "21d",
) -> Dict[str, Any]:
    records = history_like_records(tracks=tracks, snapshots=snapshots)
    matured = matured_records(records, horizon_label=horizon_label)
    dau_buckets = build_bucket_summary(
        records,
        value_getter=lambda record: safe_float(record.get("deployable_alpha_utility")),
        horizon_label=horizon_label,
        favorable_high=True,
        bucket_prefix="tracked_dau",
    )
    validated_buckets = build_bucket_summary(
        records,
        value_getter=lambda record: safe_float(record.get("validated_edge")),
        horizon_label=horizon_label,
        favorable_high=True,
        bucket_prefix="tracked_validated_edge",
    )
    deployability_rows = build_group_summary(
        matured,
        key_getter=lambda record: str(record.get("deployability_tier") or "unknown"),
        horizon_label=horizon_label,
    )
    regime_rows = build_group_summary(
        matured,
        key_getter=lambda record: str(record.get("regime_label") or "unknown"),
        horizon_label=horizon_label,
    )
    trade_family_rows = build_group_summary(
        matured,
        key_getter=lambda record: str(record.get("trade_family") or "none"),
        horizon_label=horizon_label,
    )
    recommendation_rows = build_group_summary(
        matured,
        key_getter=lambda record: str(record.get("recommendation_state") or "draft"),
        horizon_label=horizon_label,
    )
    drift = build_drift_evidence_summary(
        workspace_id=workspace_id,
        organization_id=organization_id,
        records=records,
        horizon_label=horizon_label,
    )
    top_bucket = ((dau_buckets.get("buckets") or [None])[0]) or {}
    paper_hit_rate = safe_float(top_bucket.get("hit_rate"))
    downside_rate = None
    if matured:
        triggered = 0
        for record in matured:
            outcome = (record.get("forward_outcomes") or {}).get(horizon_label) or {}
            if outcome.get("invalidation_triggered") is True:
                triggered += 1
        downside_rate = round(triggered / len(matured), 4)
    status = "available" if len(matured) >= 6 else "insufficient_sample"
    evidence_status = (
        "supportive"
        if len(matured) >= 6 and (safe_float(top_bucket.get("average_net_edge_return")) or 0.0) > 0
        else "mixed"
        if len(matured) >= 3
        else "insufficient_sample"
    )
    warnings: List[str] = []
    if len(matured) < 6:
        warnings.append("paper-tracked calibration remains sample-constrained")
    if str(drift.get("status") or "") == "degrading":
        warnings.append("recent paper-tracked bucket behavior is degrading")
    return CalibrationHardeningSummary(
        platform_version=platform_version,
        workspace_id=workspace_id,
        organization_id=organization_id,
        horizon_label=horizon_label,
        status=status,
        evidence_status=evidence_status,
        tracked_recommendation_count=len(tracks),
        paper_trade_count=len(paper_trades),
        sample_count=len(records),
        matured_count=len(matured),
        dau_bucket_realized_outcomes=dau_buckets,
        validated_edge_bucket_realized_outcomes=validated_buckets,
        deployability_tier_realized_outcomes=deployability_rows,
        regime_realized_outcomes=regime_rows,
        trade_family_realized_outcomes=trade_family_rows,
        recommendation_state_realized_outcomes=recommendation_rows,
        paper_trade_hit_rate=paper_hit_rate,
        downside_rate=downside_rate,
        drift_summary=DriftEvidenceSummary.model_validate(drift),
        warnings=warnings,
        summary=(
            f"Paper-tracked calibration covers {len(matured)} matured recommendation(s) at {horizon_label}."
            if matured
            else "Paper-tracked calibration does not yet have matured recommendations."
        ),
    ).model_dump(mode="python")


def build_workspace_proof_summary(
    *,
    platform_version: str,
    workspace_id: Optional[str],
    organization_id: Optional[str],
    tracks: Sequence[Dict[str, Any]],
    snapshots: Sequence[Dict[str, Any]],
    calibration_hardening: Optional[Dict[str, Any]] = None,
    benchmark_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    latest_snapshots = latest_snapshots_by_track(snapshots)
    counts = _paper_status_counts(snapshots)
    matured_count = sum(
        1
        for snapshot in latest_snapshots
        if ((snapshot.get("evidence_status") or {}).get("sample_size") or 0) > 0
    )
    benchmark_summary = benchmark_summary or build_benchmark_comparison_summary(
        platform_version=platform_version,
        workspace_id=workspace_id,
        organization_id=organization_id,
        tracks=tracks,
        snapshots=snapshots,
    )
    calibration_hardening = calibration_hardening or build_calibration_hardening_summary(
        platform_version=platform_version,
        workspace_id=workspace_id,
        organization_id=organization_id,
        tracks=tracks,
        paper_trades=[],
        snapshots=snapshots,
    )
    evidence_maturity = _evidence_maturity_level(matured_count)
    warnings = list(calibration_hardening.get("warnings") or [])
    replay_consistency_label = (
        "supportive"
        if str(calibration_hardening.get("evidence_status") or "") == "supportive"
        and counts.get("weak", 0) == 0
        else "mixed"
        if matured_count
        else "partial"
    )
    return WorkspaceProofSummary(
        platform_version=platform_version,
        workspace_id=workspace_id,
        organization_id=organization_id,
        tracked_recommendation_count=len(tracks),
        active_tracking_count=sum(
            1
            for track in tracks
            if str(((track.get("tracking_status") or {}).get("status")) or "active")
            == "active"
        ),
        matured_tracking_count=matured_count,
        supportive_count=counts.get("supportive", 0),
        mixed_count=counts.get("mixed", 0),
        weak_count=counts.get("weak", 0),
        insufficient_count=counts.get("insufficient_sample", 0),
        evidence_maturity_level=evidence_maturity,
        replay_consistency_label=replay_consistency_label,
        top_regime_rows=list(calibration_hardening.get("regime_realized_outcomes") or [])[:5],
        top_tier_rows=list(calibration_hardening.get("deployability_tier_realized_outcomes") or [])[:5],
        top_trade_family_rows=list(calibration_hardening.get("trade_family_realized_outcomes") or [])[:5],
        warnings=warnings[:8],
        summary=(
            f"Proof summary covers {len(tracks)} tracked recommendation(s) with "
            f"{counts.get('supportive', 0)} supportive, {counts.get('mixed', 0)} mixed, and "
            f"{counts.get('weak', 0)} weak paper-tracked outcome set(s)."
        ),
    ).model_dump(mode="python")


def build_model_credibility_snapshot(
    *,
    platform_version: str,
    workspace_id: Optional[str],
    organization_id: Optional[str],
    proof_summary: Dict[str, Any],
    calibration_hardening: Dict[str, Any],
    drift_summary: Dict[str, Any],
) -> Dict[str, Any]:
    status = (
        "supportive"
        if str(calibration_hardening.get("evidence_status") or "") == "supportive"
        and str(drift_summary.get("status") or "") not in {"degrading"}
        else "partial"
        if int(proof_summary.get("matured_tracking_count") or 0) > 0
        else "limited"
    )
    warnings = [
        *list(proof_summary.get("warnings") or []),
        *list(calibration_hardening.get("warnings") or []),
        *list(drift_summary.get("warnings") or []),
    ]
    return ModelCredibilitySnapshot(
        platform_version=platform_version,
        workspace_id=workspace_id,
        organization_id=organization_id,
        status=status,
        tracked_recommendation_count=int(proof_summary.get("tracked_recommendation_count") or 0),
        replay_evidence_status=str(proof_summary.get("replay_consistency_label") or "partial"),
        paper_evidence_status=str(calibration_hardening.get("evidence_status") or "partial"),
        calibration_status=str(calibration_hardening.get("status") or "partial"),
        drift_status=str(drift_summary.get("status") or "partial"),
        buyer_summary=(
            "Paper-tracked proof is becoming institutionally useful but still requires more matured samples."
            if status != "supportive"
            else "Paper-tracked proof and calibration remain broadly supportive for pilot credibility discussions."
        ),
        warnings=warnings[:10],
    ).model_dump(mode="python")
