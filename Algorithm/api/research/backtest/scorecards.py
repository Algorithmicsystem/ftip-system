from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from api.assistant.phase6.scorecards import (
    build_calibration_summary,
    build_signal_scorecard,
    build_strategy_scorecard,
)

from .common import mean, median, monotonic_label, safe_float


def _matured(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [record for record in records if (record.get("outcome") or {}).get("matured") is True]


def build_return_summaries(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    matured = _matured(records)
    gross = [safe_float((record.get("outcome") or {}).get("gross_edge_return")) for record in matured]
    net = [safe_float((record.get("outcome") or {}).get("net_edge_return")) for record in matured]
    trade_gross = [safe_float((record.get("outcome") or {}).get("gross_trade_return")) for record in matured]
    trade_net = [safe_float((record.get("outcome") or {}).get("net_trade_return")) for record in matured]
    gross_costs = [safe_float((record.get("outcome") or {}).get("estimated_cost_bps")) for record in matured]
    hit_defined = [
        record for record in matured if (record.get("outcome") or {}).get("final_signal_correct") in {True, False}
    ]
    hit_rate = (
        sum(1 for record in hit_defined if (record.get("outcome") or {}).get("final_signal_correct") is True) / len(hit_defined)
        if hit_defined
        else None
    )
    return {
        "gross_return_summary": {
            "matured_count": len(matured),
            "average_edge_return": round(mean(gross) or 0.0, 6) if matured else None,
            "median_edge_return": round(median(gross) or 0.0, 6) if matured else None,
            "average_trade_return": round(mean(trade_gross) or 0.0, 6) if matured else None,
            "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        },
        "net_return_summary": {
            "matured_count": len(matured),
            "average_edge_return": round(mean(net) or 0.0, 6) if matured else None,
            "median_edge_return": round(median(net) or 0.0, 6) if matured else None,
            "average_trade_return": round(mean(trade_net) or 0.0, 6) if matured else None,
            "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        },
        "friction_cost_summary": {
            "average_cost_bps": round(mean(gross_costs) or 0.0, 4) if matured else None,
            "median_cost_bps": round(median(gross_costs) or 0.0, 4) if matured else None,
            "average_cost_drag": round((mean(gross) or 0.0) - (mean(net) or 0.0), 6)
            if matured
            else None,
        },
    }


def build_liquidity_bucket_cost_summary(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in _matured(records):
        friction = (record.get("outcome") or {}).get("friction_cost_summary") or {}
        label = friction.get("liquidity_bucket") or "unknown"
        buckets[str(label)].append(record)
    rows: List[Dict[str, Any]] = []
    for label, bucket in buckets.items():
        costs = [safe_float((item.get("outcome") or {}).get("estimated_cost_bps")) for item in bucket]
        net = [safe_float((item.get("outcome") or {}).get("net_edge_return")) for item in bucket]
        rows.append(
            {
                "label": label,
                "sample_count": len(bucket),
                "average_cost_bps": round(mean(costs) or 0.0, 4),
                "average_net_edge_return": round(mean(net) or 0.0, 6),
            }
        )
    rows.sort(key=lambda item: item["sample_count"], reverse=True)
    return rows


def build_gap_loss_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    matured = _matured(records)
    adverse = [abs(safe_float((record.get("outcome") or {}).get("adverse_excursion")) or 0.0) for record in matured]
    gap_risk = [
        safe_float(((record.get("outcome") or {}).get("friction_cost_summary") or {}).get("gap_risk_bps"))
        for record in matured
    ]
    return {
        "average_adverse_excursion": round(mean(adverse) or 0.0, 6) if matured else None,
        "median_adverse_excursion": round(median(adverse) or 0.0, 6) if matured else None,
        "average_gap_risk_bps": round(mean(gap_risk) or 0.0, 4) if matured else None,
    }


def _bucket_count(sample_count: int) -> int:
    if sample_count >= 20:
        return 5
    if sample_count >= 9:
        return 3
    if sample_count >= 4:
        return 2
    return 0


def _stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    matured = _matured(records)
    edge = [safe_float((record.get("outcome") or {}).get("net_edge_return")) for record in matured]
    hit_defined = [
        record for record in matured if (record.get("outcome") or {}).get("final_signal_correct") in {True, False}
    ]
    return {
        "sample_count": len(records),
        "matured_count": len(matured),
        "average_edge_return": round(mean(edge) or 0.0, 6) if matured else None,
        "hit_rate": round(
            sum(1 for record in hit_defined if (record.get("outcome") or {}).get("final_signal_correct") is True) / len(hit_defined),
            4,
        )
        if hit_defined
        else None,
    }


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
    return [
        [record for record, _value in valued[index * bucket_size : (index + 1) * bucket_size]]
        for index in range(count)
        if valued[index * bucket_size : (index + 1) * bucket_size]
    ]


def _score_specs(records: List[Dict[str, Any]]) -> List[Tuple[str, Callable[[Dict[str, Any]], Optional[float]], bool]]:
    specs: List[Tuple[str, Callable[[Dict[str, Any]], Optional[float]], bool]] = [
        ("canonical_signal_score", lambda record: safe_float(record.get("score")), True),
        ("confidence_score", lambda record: safe_float(record.get("confidence_score")), True),
        ("actionability_score", lambda record: safe_float(record.get("actionability_score")), True),
        (
            "implementation_fragility_score",
            lambda record: safe_float((record.get("feature_vector") or {}).get("implementation_fragility_score")),
            False,
        ),
        (
            "market_stress_score",
            lambda record: safe_float((record.get("feature_vector") or {}).get("market_stress_score")),
            False,
        ),
    ]
    score_names = [
        "Opportunity Quality Score",
        "Cross-Domain Conviction Score",
        "Signal Fragility Index",
        "Market Structure Integrity Score",
        "Fundamental Durability Score",
        "Macro Alignment Score",
        "portfolio_candidate_score",
        "watchlist_priority_score",
        "ranked_opportunity_score",
    ]
    for score_name in score_names:
        if any(
            safe_float((record.get("proprietary_scores") or {}).get(score_name)) is not None
            or safe_float(record.get(score_name)) is not None
            for record in records
        ):
            favorable_high = score_name not in {"Signal Fragility Index"}
            specs.append(
                (
                    score_name,
                    lambda record, key=score_name: safe_float(
                        (record.get("proprietary_scores") or {}).get(key) if "Score" in key or "Index" in key else record.get(key)
                    ),
                    favorable_high,
                )
            )
    return specs


def build_ranking_validation(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for name, getter, favorable_high in _score_specs(records):
        buckets = _bucketize(records, value_getter=getter, favorable_high=favorable_high)
        if not buckets:
            results.append({"score_name": name, "status": "insufficient_sample", "bucket_results": []})
            continue
        bucket_rows: List[Dict[str, Any]] = []
        averages: List[Optional[float]] = []
        for index, bucket in enumerate(buckets, start=1):
            values = [getter(record) for record in bucket if getter(record) is not None]
            stats = _stats(bucket)
            averages.append(safe_float(stats.get("average_edge_return")))
            bucket_rows.append(
                {
                    "bucket": index,
                    "bucket_label": "favorable" if index == 1 else "unfavorable" if index == len(buckets) else "middle",
                    "score_min": min(values) if values else None,
                    "score_max": max(values) if values else None,
                    **stats,
                }
            )
        top = bucket_rows[0]
        bottom = bucket_rows[-1]
        results.append(
            {
                "score_name": name,
                "status": "available",
                "bucket_results": bucket_rows,
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
                "monotonicity": monotonic_label(averages),
            }
        )
    return {"bucket_results": results}


def build_suppression_effect_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    suppressed = [
        record
        for record in records
        if (record.get("signal_payload") or {}).get("suppression_flags") or record.get("suppression_flags")
    ]
    unsuppressed = [record for record in records if record not in suppressed]
    suppressed_stats = _stats(suppressed)
    unsuppressed_stats = _stats(unsuppressed)
    return {
        "suppressed_setups": suppressed_stats,
        "unsuppressed_setups": unsuppressed_stats,
        "suppression_effect_edge_spread": round(
            (safe_float(unsuppressed_stats.get("average_edge_return")) or 0.0)
            - (safe_float(suppressed_stats.get("average_edge_return")) or 0.0),
            6,
        )
        if suppressed_stats.get("matured_count") and unsuppressed_stats.get("matured_count")
        else None,
    }


def build_readiness_scorecard(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        label = (
            record.get("deployment_permission")
            or ("paper_shadow_only" if safe_float(record.get("confidence_score")) and safe_float(record.get("confidence_score")) < 50 else "analysis_only")
        )
        buckets[str(label)].append(record)
    results = {label: _stats(bucket) for label, bucket in buckets.items()}
    live_like = results.get("live_eligible") or results.get("limited_live_eligible") or {}
    paper_like = results.get("paper_shadow_only") or results.get("analysis_only") or {}
    return {
        "permission_buckets": results,
        "paper_vs_live_candidate_quality_summary": round(
            (safe_float(live_like.get("average_edge_return")) or 0.0)
            - (safe_float(paper_like.get("average_edge_return")) or 0.0),
            6,
        )
        if live_like.get("matured_count") and paper_like.get("matured_count")
        else None,
    }


def build_mae_mfe_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    matured = _matured(records)
    mae = [safe_float((record.get("outcome") or {}).get("mae")) for record in matured]
    mfe = [safe_float((record.get("outcome") or {}).get("mfe")) for record in matured]
    half_life = [safe_float((record.get("outcome") or {}).get("signal_half_life_days")) for record in matured]
    decay = [safe_float((record.get("outcome") or {}).get("continuation_decay_score")) for record in matured]
    invalidations = [
        record for record in matured if (record.get("outcome") or {}).get("invalidation_triggered") is True
    ]
    return {
        "mae_distribution": {
            "average": round(mean(mae) or 0.0, 6) if matured else None,
            "median": round(median(mae) or 0.0, 6) if matured else None,
        },
        "mfe_distribution": {
            "average": round(mean(mfe) or 0.0, 6) if matured else None,
            "median": round(median(mfe) or 0.0, 6) if matured else None,
        },
        "invalidation_frequency": round(len(invalidations) / len(matured), 4) if matured else None,
        "continuation_decay_summary": {
            "average_half_life_days": round(mean(half_life) or 0.0, 2) if matured else None,
            "average_decay_score": round(mean(decay) or 0.0, 4) if matured else None,
        },
        "fragility_vs_drawdown_summary": round(
            mean(
                (
                    (safe_float((record.get("feature_vector") or {}).get("implementation_fragility_score")) or safe_float((record.get("proprietary_scores") or {}).get("Signal Fragility Index")) or 0.0)
                    * (safe_float((record.get("outcome") or {}).get("mae")) or 0.0)
                )
                for record in matured
            )
            or 0.0,
            6,
        )
        if matured
        else None,
    }


def build_failure_modes(
    *,
    weakest_conditions: List[Dict[str, Any]],
    suppression_effect_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = list(weakest_conditions[:5])
    suppressed = suppression_effect_summary.get("suppressed_setups") or {}
    if suppressed.get("matured_count"):
        rows.append(
            {
                "dimension": "suppression_state",
                "label": "suppressed",
                "average_forward_return": suppressed.get("average_edge_return"),
                "hit_rate": suppressed.get("hit_rate"),
                "matured_count": suppressed.get("matured_count"),
            }
        )
    return rows[:6]


def build_validation_scorecards(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "signal_scorecard": build_signal_scorecard(records),
        "strategy_scorecard": build_strategy_scorecard(records),
        "calibration_summary": build_calibration_summary(records),
        "ranking_scorecard": build_ranking_validation(records),
        "readiness_scorecard": build_readiness_scorecard(records),
        "suppression_effect_summary": build_suppression_effect_summary(records),
        "mae_mfe_summary": build_mae_mfe_summary(records),
        **build_return_summaries(records),
        "liquidity_bucket_cost_summary": build_liquidity_bucket_cost_summary(records),
        "gap_loss_summary": build_gap_loss_summary(records),
    }
