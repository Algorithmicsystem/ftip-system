from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Sequence

from api.axiom.analytics import (
    as_date,
    build_bucket_summary,
    build_group_summary,
    matured_records,
    outcome_metrics,
    safe_float,
)
from api.axiom.common import rounded
from api.axiom.contracts import AxiomCalibrationArtifact
from api.axiom.engine import AXIOM_FRAMEWORK_VERSION
from api.axiom.history import AXIOM_PHASE3_VERSION


AXIOM_CALIBRATION_VERSION = "axiom50_phase3_calibration_v1"


def _engine_conditioned_summary(
    records: Sequence[Dict[str, Any]],
    *,
    horizon_label: str,
) -> List[Dict[str, Any]]:
    specs = [
        ("fundamental_reality", True),
        ("state_pricing", True),
        ("behavioral_distortion", True),
        ("flow_transmission", True),
        ("liquidity_convexity", True),
        ("critical_fragility", False),
        ("research_integrity", True),
    ]
    rows: List[Dict[str, Any]] = []
    for engine_name, favorable_high in specs:
        summary = build_bucket_summary(
            records,
            value_getter=lambda record, key=engine_name: safe_float(
                ((record.get("engine_scores") or {}).get(key) or {}).get("score")
            ),
            horizon_label=horizon_label,
            favorable_high=favorable_high,
            bucket_prefix=engine_name,
        )
        rows.append(
            {
                "engine": engine_name,
                **summary,
            }
        )
    return rows


def build_axiom_calibration_artifact(
    records: Sequence[Dict[str, Any]],
    *,
    as_of_date: Optional[dt.date] = None,
    horizon_label: str = "21d",
    min_sample_size: int = 6,
) -> Dict[str, Any]:
    cutoff = as_date(as_of_date) if as_of_date is not None else None
    if cutoff is None:
        cutoff = max(
            (as_date(record.get("as_of_date")) for record in records if record.get("as_of_date")),
            default=None,
        )
    matured = matured_records(records, horizon_label=horizon_label, as_of_date=cutoff)
    dau_summary = build_bucket_summary(
        records,
        value_getter=lambda record: safe_float(record.get("deployable_alpha_utility")),
        horizon_label=horizon_label,
        favorable_high=True,
        bucket_prefix="dau",
    )
    validated_edge_curve = build_bucket_summary(
        records,
        value_getter=lambda record: safe_float(record.get("validated_edge")),
        horizon_label=horizon_label,
        favorable_high=True,
        bucket_prefix="validated_edge",
    )
    regime_summary = build_group_summary(
        matured,
        key_getter=lambda record: str(record.get("regime_label") or "unknown"),
        horizon_label=horizon_label,
    )
    trade_family_summary = build_group_summary(
        matured,
        key_getter=lambda record: str(record.get("trade_family") or "none"),
        horizon_label=horizon_label,
    )
    deployability_summary = build_group_summary(
        matured,
        key_getter=lambda record: str(
            ((record.get("evidence_backed_deployability") or {}).get("deployability_tier"))
            or record.get("deployability_tier")
            or "unknown"
        ),
        horizon_label=horizon_label,
    )
    engine_summary = _engine_conditioned_summary(matured, horizon_label=horizon_label)

    top_bucket = ((dau_summary.get("buckets") or [None])[0]) or {}
    top_return = safe_float(top_bucket.get("average_net_edge_return")) or 0.0
    top_hit_rate = safe_float(top_bucket.get("hit_rate")) or 0.0
    top_mae = safe_float(top_bucket.get("average_mae")) or 0.0
    supportive_for_live = (
        len(matured) >= max(12, min_sample_size)
        and top_return > 0.0
        and top_hit_rate >= 0.5
        and top_mae <= 0.08
    )
    supportive_for_paper = (
        len(matured) >= min_sample_size
        and top_return >= -0.002
        and top_hit_rate >= 0.45
    )
    status = "available" if len(matured) >= min_sample_size else "limited"
    if len(matured) == 0:
        status = "insufficient_sample"

    summary = (
        f"AXIOM calibration at {horizon_label} currently tracks {len(matured)} matured records. "
        f"Top-bucket DAU edge is {rounded(top_return, digits=6)} with hit rate {rounded(top_hit_rate, digits=4)}."
    )

    artifact = AxiomCalibrationArtifact(
        calibration_version=AXIOM_CALIBRATION_VERSION,
        framework_version=AXIOM_FRAMEWORK_VERSION,
        as_of_date=cutoff.isoformat() if cutoff else None,
        horizon_label=horizon_label,
        status=status,
        sample_count=len(records),
        matured_count=len(matured),
        dau_bucket_summary=dau_summary,
        validated_edge_curve=validated_edge_curve,
        regime_outcome_summary=regime_summary,
        trade_family_outcome_summary=trade_family_summary,
        deployability_tier_outcome_summary=deployability_summary,
        engine_conditioned_outcome_summary=engine_summary,
        evidence_supportive_for_live=supportive_for_live,
        evidence_supportive_for_paper=supportive_for_paper,
        summary=summary,
        diagnostics={
            "phase3_version": AXIOM_PHASE3_VERSION,
            "overall_outcome_metrics": outcome_metrics(matured, horizon_label=horizon_label),
            "minimum_sample_size": min_sample_size,
        },
    )
    return artifact.model_dump(mode="python")
