from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .common import compact_list, hold_band, horizon_days, mean, median, monotonic_label, percentile_bucket, safe_float


def _matured(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        record
        for record in records
        if ((record.get("outcome") or {}).get("matured") is True)
    ]


def _stats(records: List[Dict[str, Any]], *, correct_key: str = "final_signal_correct") -> Dict[str, Any]:
    matured = _matured(records)
    returns = [safe_float((record.get("outcome") or {}).get("forward_return")) for record in matured]
    edge_key = "final_signal_edge_return" if correct_key == "final_signal_correct" else "raw_signal_edge_return"
    edge_returns = [safe_float((record.get("outcome") or {}).get(edge_key)) for record in matured]
    favorable = [safe_float((record.get("outcome") or {}).get("favorable_excursion")) for record in matured]
    adverse = [safe_float((record.get("outcome") or {}).get("adverse_excursion")) for record in matured]
    abs_returns = [safe_float((record.get("outcome") or {}).get("absolute_forward_return")) for record in matured]
    correctness = [
        1.0
        for record in matured
        if (record.get("outcome") or {}).get(correct_key) is True
    ]
    total_defined = [
        record
        for record in matured
        if (record.get("outcome") or {}).get(correct_key) in {True, False}
    ]
    hit_rate = len(correctness) / len(total_defined) if total_defined else None
    return {
        "sample_count": len(records),
        "matured_count": len(matured),
        "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        "average_forward_return": round(mean(returns) or 0.0, 6) if matured else None,
        "median_forward_return": round(median(returns) or 0.0, 6) if matured else None,
        "average_edge_return": round(mean(edge_returns) or 0.0, 6) if matured else None,
        "median_edge_return": round(median(edge_returns) or 0.0, 6) if matured else None,
        "average_absolute_return": round(mean(abs_returns) or 0.0, 6) if matured else None,
        "average_favorable_excursion": round(mean(favorable) or 0.0, 6) if matured else None,
        "average_adverse_excursion": round(mean(adverse) or 0.0, 6) if matured else None,
        "downside_tail_behavior": round(min((value for value in returns if value is not None), default=0.0), 6)
        if matured
        else None,
        "positive_outcome_rate": round(
            sum(1 for value in returns if value is not None and value > 0) / len(matured),
            4,
        )
        if matured
        else None,
    }


def build_signal_scorecard(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_raw: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_final: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_raw[str(record.get("signal_action") or "HOLD").upper()].append(record)
        by_final[str(record.get("final_signal") or "HOLD").upper()].append(record)

    hold_records = by_final.get("HOLD", [])
    hold_usefulness = None
    hold_matured = _matured(hold_records)
    if hold_matured:
        helpful = 0
        for record in hold_matured:
            threshold = hold_band(horizon_days(record))
            if abs(safe_float((record.get("outcome") or {}).get("forward_return")) or 0.0) <= threshold:
                helpful += 1
        hold_usefulness = round(helpful / len(hold_matured), 4)

    return {
        "raw_signal_overall": _stats(records, correct_key="raw_signal_correct"),
        "final_signal_overall": _stats(records, correct_key="final_signal_correct"),
        "raw_signal_breakdown": {
            signal: _stats(group, correct_key="raw_signal_correct") for signal, group in by_raw.items()
        },
        "final_signal_breakdown": {
            signal: _stats(group, correct_key="final_signal_correct")
            for signal, group in by_final.items()
        },
        "hold_precision": hold_usefulness,
    }


def _is_actionable(record: Dict[str, Any]) -> bool:
    posture = str(record.get("strategy_posture") or "").lower()
    return posture in {
        "actionable_long",
        "actionable_short",
        "trend_continuation_candidate",
        "opportunistic_reversal",
    } or (safe_float(record.get("actionability_score")) or 0.0) >= 60.0


def build_strategy_scorecard(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    actionable = [record for record in records if _is_actionable(record)]
    watchlist = [record for record in records if not _is_actionable(record)]
    high_conviction = [
        record
        for record in records
        if str(record.get("conviction_tier") or "") in {"high", "very_high"}
    ]
    low_conviction = [
        record
        for record in records
        if str(record.get("conviction_tier") or "") in {"very_low", "low"}
    ]
    fragile = [
        record
        for record in records
        if str(record.get("fragility_tier") or "") not in {"", "contained", "stable"}
        or (safe_float((record.get("proprietary_scores") or {}).get("Signal Fragility Index")) or 0.0) >= 60.0
    ]
    clean = [record for record in records if record not in fragile]
    wait_helpfulness = None
    watchlist_matured = _matured(watchlist)
    if watchlist_matured:
        helpful = 0
        for record in watchlist_matured:
            threshold = hold_band(horizon_days(record))
            if abs(safe_float((record.get("outcome") or {}).get("forward_return")) or 0.0) <= threshold:
                helpful += 1
        wait_helpfulness = round(helpful / len(watchlist_matured), 4)

    actionable_stats = _stats(actionable)
    watchlist_stats = _stats(watchlist)
    return {
        "overall": _stats(records),
        "actionable_setups": actionable_stats,
        "watchlist_or_wait_setups": watchlist_stats,
        "high_conviction": _stats(high_conviction),
        "low_conviction": _stats(low_conviction),
        "fragile_setups": _stats(fragile),
        "cleaner_setups": _stats(clean),
        "wait_classification_usefulness": wait_helpfulness,
        "actionable_vs_watchlist_return_spread": round(
            (safe_float(actionable_stats.get("average_edge_return")) or 0.0)
            - (safe_float(watchlist_stats.get("average_edge_return")) or 0.0),
            6,
        )
        if actionable_stats.get("matured_count") and watchlist_stats.get("matured_count")
        else None,
    }


def _bucket_count(sample_count: int) -> int:
    if sample_count >= 20:
        return 5
    if sample_count >= 9:
        return 3
    if sample_count >= 4:
        return 2
    return 0


def _bucketize(
    records: List[Dict[str, Any]],
    *,
    value_getter: Callable[[Dict[str, Any]], Optional[float]],
    favorable_high: bool,
) -> List[List[Dict[str, Any]]]:
    valued = [(record, value_getter(record)) for record in records]
    valued = [(record, value) for record, value in valued if value is not None]
    count = _bucket_count(len(valued))
    if count == 0:
        return []
    valued.sort(key=lambda item: item[1], reverse=favorable_high)
    bucket_size = int(math.ceil(len(valued) / count))
    buckets: List[List[Dict[str, Any]]] = []
    for index in range(count):
        subset = [record for record, _value in valued[index * bucket_size : (index + 1) * bucket_size]]
        if subset:
            buckets.append(subset)
    return buckets


def _bucket_summary(
    records: List[Dict[str, Any]],
    *,
    label: str,
    value_getter: Callable[[Dict[str, Any]], Optional[float]],
    favorable_high: bool,
) -> Dict[str, Any]:
    buckets = _bucketize(records, value_getter=value_getter, favorable_high=favorable_high)
    if not buckets:
        return {
            "score_name": label,
            "status": "insufficient_sample",
            "bucket_results": [],
        }
    bucket_stats = []
    avg_returns: List[Optional[float]] = []
    for index, bucket in enumerate(buckets, start=1):
        values = [value_getter(record) for record in bucket]
        stats = _stats(bucket)
        avg_returns.append(safe_float(stats.get("average_edge_return")))
        bucket_stats.append(
            {
                "bucket": index,
                "bucket_label": "favorable" if index == 1 else "unfavorable" if index == len(buckets) else "middle",
                "score_min": min(value for value in values if value is not None),
                "score_max": max(value for value in values if value is not None),
                "sample_count": stats.get("sample_count"),
                "matured_count": stats.get("matured_count"),
                "average_forward_return": stats.get("average_forward_return"),
                "average_edge_return": stats.get("average_edge_return"),
                "hit_rate": stats.get("hit_rate"),
            }
        )
    top = bucket_stats[0]
    bottom = bucket_stats[-1]
    return {
        "score_name": label,
        "status": "available",
        "bucket_order": "favorable_to_unfavorable",
        "bucket_results": bucket_stats,
        "favorable_vs_unfavorable_return_spread": round(
            (safe_float(top.get("average_edge_return")) or 0.0)
            - (safe_float(bottom.get("average_edge_return")) or 0.0),
            6,
        ),
        "favorable_vs_unfavorable_hit_rate_spread": round(
            (safe_float(top.get("hit_rate")) or 0.0)
            - (safe_float(bottom.get("hit_rate")) or 0.0),
            6,
        ),
        "monotonicity": monotonic_label(avg_returns),
    }


def build_ranking_scorecard(
    records: List[Dict[str, Any]],
    *,
    score_specs: List[Tuple[str, Callable[[Dict[str, Any]], Optional[float]], bool]],
) -> Dict[str, Any]:
    results = [
        _bucket_summary(records, label=name, value_getter=getter, favorable_high=favorable_high)
        for name, getter, favorable_high in score_specs
    ]
    available = [result for result in results if result.get("status") == "available"]
    return {
        "bucket_results": results,
        "available_bucket_count": len(available),
    }


def build_calibration_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _confidence_label(series: List[Optional[float]]) -> str:
        clean = [value for value in series if value is not None]
        if len(clean) < 2:
            return "insufficient"
        if all(left <= right for left, right in zip(clean, clean[1:])):
            return "higher_confidence_buckets_outperform"
        if all(left >= right for left, right in zip(clean, clean[1:])):
            return "lower_confidence_buckets_outperform"
        return "mixed"

    matured = _matured(records)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    conviction_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in matured:
        confidence_score = safe_float(record.get("confidence_score"))
        buckets[percentile_bucket(confidence_score)].append(record)
        conviction_groups[str(record.get("conviction_tier") or "unknown")].append(record)

    bucket_stats = []
    ordered_labels = ["0-20", "20-40", "40-60", "60-80", "80-100", "unknown"]
    hit_rates: List[Optional[float]] = []
    avg_returns: List[Optional[float]] = []
    for label in ordered_labels:
        group = buckets.get(label, [])
        if not group:
            continue
        stats = _stats(group)
        hit_rates.append(safe_float(stats.get("hit_rate")))
        avg_returns.append(safe_float(stats.get("average_forward_return")))
        avg_returns[-1] = safe_float(stats.get("average_edge_return"))
        bucket_stats.append(
            {
                "bucket": label,
                **stats,
            }
        )

    reliability_score = None
    drift_notes: List[str] = []
    if bucket_stats:
        high = bucket_stats[-1]
        low = bucket_stats[0]
        hit_gap = (safe_float(high.get("hit_rate")) or 0.0) - (safe_float(low.get("hit_rate")) or 0.0)
        monotonic_pairs = 0.0
        if len(hit_rates) >= 2:
            monotonic_pairs = sum(
                1 for left, right in zip(hit_rates, hit_rates[1:]) if (left or 0.0) <= (right or 0.0)
            ) / (len(hit_rates) - 1)
        reliability_score = round(
            100.0
            * (
                0.45 * max(0.0, hit_gap + 0.5)
                + 0.35 * monotonic_pairs
                + 0.20 * (1.0 if (safe_float(high.get("average_edge_return")) or 0.0) >= (safe_float(low.get("average_edge_return")) or 0.0) else 0.0)
            ),
            2,
        )
        if (safe_float(high.get("hit_rate")) or 0.0) < (safe_float(low.get("hit_rate")) or 0.0):
            drift_notes.append("Higher-confidence buckets are not reliably outperforming lower-confidence buckets.")
        if (safe_float(high.get("average_edge_return")) or 0.0) <= 0:
            drift_notes.append("The highest-confidence bucket is not yet translating into positive average forward returns.")

    return {
        "bucketed_confidence_stats": bucket_stats,
        "conviction_tier_stats": {
            label: _stats(group) for label, group in conviction_groups.items()
        },
        "confidence_reliability_score": reliability_score,
        "confidence_monotonicity": _confidence_label(hit_rates),
        "calibration_drift_notes": drift_notes,
    }


def build_regime_breakdown(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    matured = _matured(records)
    dimensions = {
        "regime_label": lambda record: record.get("regime_label"),
        "fragility_tier": lambda record: record.get("fragility_tier"),
        "conviction_tier": lambda record: record.get("conviction_tier"),
        "crowding_state": lambda record: (record.get("slices") or {}).get("crowding_state"),
        "macro_alignment_state": lambda record: (record.get("slices") or {}).get("macro_alignment_state"),
        "agreement_state": lambda record: (record.get("slices") or {}).get("agreement_state"),
        "freshness_quality": lambda record: (record.get("slices") or {}).get("freshness_quality"),
        "participant_fit_primary": lambda record: record.get("participant_fit_primary"),
    }
    breakdown: Dict[str, Any] = {}
    condition_rows: List[Dict[str, Any]] = []
    for name, getter in dimensions.items():
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in matured:
            grouped[str(getter(record) or "unknown")].append(record)
        rows = []
        for label, group in grouped.items():
            stats = _stats(group)
            row = {"label": label, **stats}
            rows.append(row)
            if stats.get("matured_count"):
                condition_rows.append(
                    {
                        "dimension": name,
                        "label": label,
                        "average_forward_return": stats.get("average_forward_return"),
                        "average_edge_return": stats.get("average_edge_return"),
                        "hit_rate": stats.get("hit_rate"),
                        "sample_count": stats.get("matured_count"),
                    }
                )
        rows.sort(key=lambda item: (item.get("matured_count") or 0, safe_float(item.get("average_forward_return")) or -999), reverse=True)
        breakdown[name] = rows

    qualified = [row for row in condition_rows if (row.get("sample_count") or 0) >= 2]
    strongest = sorted(
        qualified,
        key=lambda item: (
            safe_float(item.get("average_edge_return")) or -999.0,
            safe_float(item.get("hit_rate")) or -999.0,
        ),
        reverse=True,
    )[:5]
    weakest = sorted(
        qualified,
        key=lambda item: (
            safe_float(item.get("average_edge_return")) or 999.0,
            safe_float(item.get("hit_rate")) or 999.0,
        ),
    )[:5]
    return {
        "regime_breakdown": breakdown,
        "strongest_conditions": strongest,
        "weakest_conditions": weakest,
    }
