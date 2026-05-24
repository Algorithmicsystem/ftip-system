from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from api.assistant.reports import sanitize_payload
from api.axiom.analytics import (
    build_bucket_summary,
    matured_records,
    outcome_metrics,
    safe_float,
    select_outcome,
)
from api.platform.contracts import BenchmarkComparisonSummary, CohortBenchmarkRow


def latest_snapshots_by_track(
    snapshots: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ordered = sorted(
        snapshots,
        key=lambda item: (
            str(item.get("snapshot_date") or ""),
            str(item.get("updated_at") or item.get("created_at") or ""),
        ),
        reverse=True,
    )
    latest: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for snapshot in ordered:
        track_id = str(snapshot.get("track_id") or "")
        if not track_id or track_id in seen:
            continue
        seen.add(track_id)
        latest.append(snapshot)
    return latest


def _history_like_records(
    *,
    tracks: Sequence[Dict[str, Any]],
    snapshots: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    snapshot_by_track = {
        str(item.get("track_id") or ""): item
        for item in latest_snapshots_by_track(snapshots)
    }
    records: List[Dict[str, Any]] = []
    for track in tracks:
        snapshot = snapshot_by_track.get(str(track.get("track_id") or ""))
        if not snapshot:
            continue
        windows = dict(snapshot.get("windows") or {})
        records.append(
            sanitize_payload(
                {
                    "track_id": track.get("track_id"),
                    "symbol": track.get("symbol"),
                    "as_of_date": (track.get("tracking_start_at") or track.get("created_at") or "")[:10],
                    "deployable_alpha_utility": track.get("start_deployable_alpha_utility"),
                    "validated_edge": track.get("start_validated_edge"),
                    "deployability_tier": track.get("deployability_tier_at_start"),
                    "regime_label": track.get("regime_label"),
                    "trade_family": track.get("trade_family"),
                    "size_band_recommendation": track.get("size_band_at_start"),
                    "recommendation_state": track.get("recommendation_state_at_start"),
                    "evidence_status": ((snapshot.get("evidence_status") or {}).get("status")),
                    "forward_outcomes": windows,
                    "engine_scores": track.get("start_engine_scores") or {},
                }
            )
        )
    return records


def history_like_records(
    *,
    tracks: Sequence[Dict[str, Any]],
    snapshots: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return _history_like_records(tracks=tracks, snapshots=snapshots)


def _cohort_rows(
    records: Sequence[Dict[str, Any]],
    *,
    horizon_label: str,
    dimension: str,
    key_getter: Callable[[Dict[str, Any]], Optional[str]],
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    matured = matured_records(records, horizon_label=horizon_label)
    for record in matured:
        label = str(key_getter(record) or "unknown")
        groups[label].append(record)

    rows: List[Dict[str, Any]] = []
    for label, group in groups.items():
        metrics = outcome_metrics(group, horizon_label=horizon_label)
        avg_forward = None
        values = [
            safe_float(select_outcome(item, horizon_label=horizon_label).get("forward_return"))
            for item in group
        ]
        clean_values = [value for value in values if value is not None]
        if clean_values:
            avg_forward = round(sum(clean_values) / len(clean_values), 6)
        avg_net_edge = safe_float(metrics.get("average_net_edge_return"))
        rows.append(
            CohortBenchmarkRow(
                dimension=dimension,
                label=label,
                matured_count=int(metrics.get("matured_count") or 0),
                average_net_edge_return=avg_net_edge,
                average_forward_return=avg_forward,
                hit_rate=safe_float(metrics.get("hit_rate")),
                average_mae=safe_float(metrics.get("average_mae")),
                average_mfe=safe_float(metrics.get("average_mfe")),
                excess_vs_hold=(
                    None
                    if avg_net_edge is None or avg_forward is None
                    else round(avg_net_edge - avg_forward, 6)
                ),
                evidence_status=(
                    "available"
                    if int(metrics.get("matured_count") or 0) >= 3
                    else "insufficient_sample"
                ),
                metadata={"sample_count": int(metrics.get("sample_count") or 0)},
            ).model_dump(mode="python")
        )
    rows.sort(
        key=lambda item: (
            int(item.get("matured_count") or 0),
            safe_float(item.get("average_net_edge_return")) or -999.0,
        ),
        reverse=True,
    )
    return rows


def build_benchmark_comparison_summary(
    *,
    platform_version: str,
    workspace_id: Optional[str],
    organization_id: Optional[str],
    tracks: Sequence[Dict[str, Any]],
    snapshots: Sequence[Dict[str, Any]],
    horizon_label: str = "21d",
) -> Dict[str, Any]:
    records = _history_like_records(tracks=tracks, snapshots=snapshots)
    matured = matured_records(records, horizon_label=horizon_label)
    cohorts = {
        "deployability_tier": _cohort_rows(
            records,
            horizon_label=horizon_label,
            dimension="deployability_tier",
            key_getter=lambda record: str(record.get("deployability_tier") or "unknown"),
        ),
        "trade_family": _cohort_rows(
            records,
            horizon_label=horizon_label,
            dimension="trade_family",
            key_getter=lambda record: str(record.get("trade_family") or "none"),
        ),
        "regime": _cohort_rows(
            records,
            horizon_label=horizon_label,
            dimension="regime",
            key_getter=lambda record: str(record.get("regime_label") or "unknown"),
        ),
        "size_band": _cohort_rows(
            records,
            horizon_label=horizon_label,
            dimension="size_band",
            key_getter=lambda record: str(record.get("size_band_recommendation") or "none"),
        ),
        "recommendation_state": _cohort_rows(
            records,
            horizon_label=horizon_label,
            dimension="recommendation_state",
            key_getter=lambda record: str(record.get("recommendation_state") or "draft"),
        ),
        "evidence_status": _cohort_rows(
            records,
            horizon_label=horizon_label,
            dimension="evidence_status",
            key_getter=lambda record: str(record.get("evidence_status") or "unknown"),
        ),
    }
    dau_quantile = build_bucket_summary(
        records,
        value_getter=lambda record: safe_float(record.get("deployable_alpha_utility")),
        horizon_label=horizon_label,
        favorable_high=True,
        bucket_prefix="paper_dau",
    )
    ranked_rows = [
        row
        for rows in cohorts.values()
        for row in rows
        if row.get("evidence_status") != "insufficient_sample"
    ]
    ranked_rows.sort(
        key=lambda item: safe_float(item.get("average_net_edge_return")) or -999.0,
        reverse=True,
    )
    status = "available" if len(matured) >= 4 else "insufficient_sample"
    summary = (
        f"Benchmark comparison covers {len(matured)} matured tracked recommendation(s) at {horizon_label}."
        if matured
        else "Benchmark comparison does not yet have matured tracked recommendations."
    )
    payload = BenchmarkComparisonSummary(
        platform_version=platform_version,
        workspace_id=workspace_id,
        organization_id=organization_id,
        horizon_label=horizon_label,
        status=status,
        cohorts=cohorts,
        dau_quantile_comparison=dau_quantile,
        strongest_cohorts=ranked_rows[:5],
        weakest_cohorts=list(reversed(ranked_rows[-5:])),
        warnings=(
            []
            if len(matured) >= 4
            else ["benchmark/cohort evidence is sample-constrained"]
        ),
        summary=summary,
    )
    return payload.model_dump(mode="python")
