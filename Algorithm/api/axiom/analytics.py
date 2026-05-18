from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from api.axiom.common import rounded


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def as_date(value: Any) -> Optional[dt.date]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError:
        return None


def select_outcome(
    record: Dict[str, Any],
    *,
    horizon_label: str,
) -> Dict[str, Any]:
    return dict((record.get("forward_outcomes") or {}).get(horizon_label) or {})


def matured_records(
    records: Sequence[Dict[str, Any]],
    *,
    horizon_label: str,
    as_of_date: Optional[dt.date] = None,
) -> List[Dict[str, Any]]:
    matured: List[Dict[str, Any]] = []
    for record in records:
        outcome = select_outcome(record, horizon_label=horizon_label)
        if not outcome or outcome.get("matured") is not True:
            continue
        exit_date = as_date(outcome.get("exit_date"))
        if as_of_date is not None and exit_date is not None and exit_date > as_of_date:
            continue
        matured.append(record)
    return matured


def outcome_metrics(
    records: Sequence[Dict[str, Any]],
    *,
    horizon_label: str,
) -> Dict[str, Any]:
    matured = matured_records(records, horizon_label=horizon_label)
    net_returns = [
        safe_float(select_outcome(record, horizon_label=horizon_label).get("net_edge_return"))
        for record in matured
    ]
    mae_values = [
        safe_float(select_outcome(record, horizon_label=horizon_label).get("mae"))
        for record in matured
    ]
    mfe_values = [
        safe_float(select_outcome(record, horizon_label=horizon_label).get("mfe"))
        for record in matured
    ]
    hit_defined = [
        record
        for record in matured
        if select_outcome(record, horizon_label=horizon_label).get("final_signal_correct") in {True, False}
    ]
    hit_rate = None
    if hit_defined:
        hit_rate = sum(
            1
            for record in hit_defined
            if select_outcome(record, horizon_label=horizon_label).get("final_signal_correct") is True
        ) / len(hit_defined)
    return {
        "sample_count": len(records),
        "matured_count": len(matured),
        "average_net_edge_return": rounded(mean(net_returns), digits=6),
        "average_mae": rounded(mean(mae_values), digits=6),
        "average_mfe": rounded(mean(mfe_values), digits=6),
        "hit_rate": rounded(hit_rate, digits=4),
    }


def _bucket_count(sample_count: int) -> int:
    if sample_count >= 50:
        return 10
    if sample_count >= 20:
        return 5
    if sample_count >= 9:
        return 3
    if sample_count >= 4:
        return 2
    return 0


def build_bucket_summary(
    records: Sequence[Dict[str, Any]],
    *,
    value_getter: Callable[[Dict[str, Any]], Optional[float]],
    horizon_label: str,
    favorable_high: bool = True,
    bucket_prefix: str = "bucket",
) -> Dict[str, Any]:
    matured = matured_records(records, horizon_label=horizon_label)
    valued: List[Tuple[Dict[str, Any], float]] = []
    for record in matured:
        value = value_getter(record)
        if value is not None:
            valued.append((record, float(value)))
    bucket_count = _bucket_count(len(valued))
    if bucket_count == 0:
        return {
            "status": "insufficient_sample",
            "bucket_count": 0,
            "buckets": [],
            "favorable_vs_unfavorable_return_spread": None,
            "monotonicity": "insufficient",
        }
    valued.sort(key=lambda item: item[1], reverse=favorable_high)
    size = int(math.ceil(len(valued) / bucket_count))
    buckets: List[Dict[str, Any]] = []
    bucket_returns: List[Optional[float]] = []
    for index in range(bucket_count):
        rows = valued[index * size : (index + 1) * size]
        if not rows:
            continue
        bucket_records = [record for record, _value in rows]
        values = [value for _record, value in rows]
        metrics = outcome_metrics(bucket_records, horizon_label=horizon_label)
        bucket_returns.append(metrics.get("average_net_edge_return"))
        buckets.append(
            {
                "label": f"{bucket_prefix}_{index + 1}",
                "bucket": index + 1,
                "score_min": rounded(min(values), digits=4),
                "score_max": rounded(max(values), digits=4),
                **metrics,
            }
        )
    spread = None
    if len(buckets) >= 2:
        spread = (
            (safe_float(buckets[0].get("average_net_edge_return")) or 0.0)
            - (safe_float(buckets[-1].get("average_net_edge_return")) or 0.0)
        )
    monotonicity = "mixed"
    clean_returns = [value for value in bucket_returns if value is not None]
    if len(clean_returns) < 2:
        monotonicity = "insufficient"
    elif all(left >= right for left, right in zip(clean_returns, clean_returns[1:])):
        monotonicity = "favorable_buckets_outperform"
    elif all(left <= right for left, right in zip(clean_returns, clean_returns[1:])):
        monotonicity = "unfavorable_buckets_outperform"
    return {
        "status": "available",
        "bucket_count": len(buckets),
        "buckets": buckets,
        "favorable_vs_unfavorable_return_spread": rounded(spread, digits=6),
        "monotonicity": monotonicity,
    }


def build_group_summary(
    records: Sequence[Dict[str, Any]],
    *,
    key_getter: Callable[[Dict[str, Any]], Optional[str]],
    horizon_label: str,
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = str(key_getter(record) or "unknown")
        groups[key].append(record)
    rows: List[Dict[str, Any]] = []
    for label, group in groups.items():
        rows.append({"label": label, **outcome_metrics(group, horizon_label=horizon_label)})
    rows.sort(
        key=lambda item: (
            item.get("matured_count") or 0,
            safe_float(item.get("average_net_edge_return")) or -999.0,
        ),
        reverse=True,
    )
    return rows
