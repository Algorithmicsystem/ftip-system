from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .common import confidence_bucket, mean, safe_float


def _matured(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [record for record in records if (record.get("outcome") or {}).get("matured") is True]


def _value_stat(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    return mean((record.get("outcome") or {}).get(key) for record in records)


def _stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    matured = _matured(records)
    positive = [
        record
        for record in matured
        if ((record.get("outcome") or {}).get("final_signal_correct")) is True
    ]
    total_defined = [
        record
        for record in matured
        if (record.get("outcome") or {}).get("final_signal_correct") in {True, False}
    ]
    return {
        "sample_count": len(records),
        "matured_count": len(matured),
        "hit_rate": round(len(positive) / len(total_defined), 4) if total_defined else None,
        "average_forward_return": round(_value_stat(matured, "forward_return") or 0.0, 6)
        if matured
        else None,
        "average_edge_return": round(_value_stat(matured, "net_edge_return") or 0.0, 6)
        if matured
        else None,
        "average_gross_edge_return": round(_value_stat(matured, "gross_edge_return") or 0.0, 6)
        if matured
        else None,
        "average_mae": round(_value_stat(matured, "mae") or 0.0, 6) if matured else None,
        "average_mfe": round(_value_stat(matured, "mfe") or 0.0, 6) if matured else None,
        "invalidation_rate": round(
            sum(1 for record in matured if (record.get("outcome") or {}).get("invalidation_triggered"))
            / len(matured),
            4,
        )
        if matured
        else None,
    }


def default_dimension_specs() -> List[Tuple[str, Callable[[Dict[str, Any]], Optional[str]]]]:
    return [
        ("regime_label", lambda record: record.get("regime_label") or (record.get("slices") or {}).get("regime_label")),
        ("fragility_tier", lambda record: record.get("fragility_tier") or (record.get("slices") or {}).get("fragility_tier")),
        ("event_risk_state", lambda record: (record.get("slices") or {}).get("event_risk_state")),
        ("liquidity_state", lambda record: (record.get("slices") or {}).get("liquidity_state")),
        ("breadth_state", lambda record: (record.get("slices") or {}).get("breadth_state")),
        ("cross_asset_state", lambda record: (record.get("slices") or {}).get("cross_asset_state")),
        ("stress_state", lambda record: (record.get("slices") or {}).get("stress_state")),
        ("confidence_bucket", lambda record: confidence_bucket(record.get("confidence_score"))),
        ("conviction_tier", lambda record: record.get("conviction_tier")),
        ("deployment_permission", lambda record: record.get("deployment_permission")),
        ("candidate_classification", lambda record: record.get("candidate_classification")),
        (
            "suppression_state",
            lambda record: "suppressed"
            if (record.get("signal_payload") or {}).get("suppression_flags") or record.get("suppression_flags")
            else "unsuppressed",
        ),
    ]


def build_cohort_breakdown(
    records: List[Dict[str, Any]],
    *,
    min_matured_count: int = 2,
) -> Dict[str, Any]:
    breakdown: Dict[str, Any] = {}
    strongest: List[Dict[str, Any]] = []
    weakest: List[Dict[str, Any]] = []

    for dimension, getter in default_dimension_specs():
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            label = getter(record)
            if label in (None, "", "unknown"):
                continue
            buckets[str(label)].append(record)
        rows: List[Dict[str, Any]] = []
        for label, bucket in buckets.items():
            stats = _stats(bucket)
            row = {"dimension": dimension, "label": label, **stats}
            rows.append(row)
            if (stats.get("matured_count") or 0) >= min_matured_count and stats.get("average_edge_return") is not None:
                strongest.append(row)
                weakest.append(row)
        rows.sort(
            key=lambda item: (
                -(item.get("matured_count") or 0),
                -(safe_float(item.get("average_edge_return")) or -999.0),
            )
        )
        breakdown[dimension] = rows

    strongest.sort(
        key=lambda item: (
            -(safe_float(item.get("average_edge_return")) or -999.0),
            -(item.get("matured_count") or 0),
        )
    )
    weakest.sort(
        key=lambda item: (
            safe_float(item.get("average_edge_return")) or 999.0,
            -(item.get("matured_count") or 0),
        )
    )
    return {
        "cohort_breakdown": breakdown,
        "strongest_conditions": strongest[:6],
        "weakest_conditions": weakest[:6],
    }

