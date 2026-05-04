from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .common import clamp, compact_list, safe_float, score_bucket


def pairwise_overlap(
    left: Dict[str, Any],
    right: Dict[str, Any],
    relationship: Dict[str, Any],
    exposure_similarity: Dict[str, Any],
) -> Dict[str, Any]:
    realized_corr = abs(safe_float(relationship.get("pairwise_correlation")) or 0.0) * 100.0
    stress_adjust = safe_float(relationship.get("stress_correlation_adjustment")) or 0.0
    stability = safe_float(relationship.get("correlation_stability_score")) or 40.0
    covariance_cluster = safe_float(relationship.get("covariance_cluster_score")) or 30.0
    factor_similarity = safe_float(exposure_similarity.get("factor_similarity_score")) or 40.0
    macro_similarity = safe_float(exposure_similarity.get("macro_exposure_similarity")) or 40.0

    left_fragility = safe_float(left.get("signal_fragility_index")) or 50.0
    right_fragility = safe_float(right.get("signal_fragility_index")) or 50.0
    fragility_similarity = clamp(100.0 - abs(left_fragility - right_fragility), 0.0, 100.0)

    left_event = safe_float(left.get("event_overhang_score")) or 0.0
    right_event = safe_float(right.get("event_overhang_score")) or 0.0
    event_cluster_overlap = clamp((left_event * 0.5) + (right_event * 0.5), 0.0, 100.0)

    theme_overlap = 82.0 if left.get("theme_tag") == right.get("theme_tag") else 34.0
    regime_overlap = 84.0 if left.get("regime_label") == right.get("regime_label") else 38.0
    benchmark_overlap = 86.0 if left.get("benchmark_proxy") == right.get("benchmark_proxy") else 42.0

    hidden_overlap_score = round(
        clamp(
            (realized_corr * 0.34)
            + (max(0.0, stability - 45.0) * 0.10)
            + (covariance_cluster * 0.14)
            + (factor_similarity * 0.18)
            + (macro_similarity * 0.08)
            + (fragility_similarity * 0.07)
            + (event_cluster_overlap * 0.05)
            + (theme_overlap * 0.04),
            0.0,
            100.0,
        ),
        2,
    )
    overlap_score = round(
        clamp(
            (hidden_overlap_score * 0.72)
            + (benchmark_overlap * 0.12)
            + (regime_overlap * 0.08)
            + (max(0.0, stress_adjust) * 0.08),
            0.0,
            100.0,
        ),
        2,
    )
    redundancy_score = round(
        clamp(
            (overlap_score * 0.52)
            + ((100.0 - abs((safe_float(left.get("opportunity_quality_score")) or 0.0) - (safe_float(right.get("opportunity_quality_score")) or 0.0))) * 0.14)
            + ((100.0 - abs((safe_float(left.get("confidence_score")) or 0.0) - (safe_float(right.get("confidence_score")) or 0.0))) * 0.10)
            + (factor_similarity * 0.14)
            + (fragility_similarity * 0.10),
            0.0,
            100.0,
        ),
        2,
    )
    complementarity_score = round(
        clamp(
            100.0
            - (realized_corr * 0.45)
            - (factor_similarity * 0.22)
            - (max(0.0, benchmark_overlap - 70.0) * 0.08)
            + ((100.0 - fragility_similarity) * 0.12)
            + (20.0 if left.get("theme_tag") != right.get("theme_tag") else 0.0),
            0.0,
            100.0,
        ),
        2,
    )
    diversification_contribution_score = round(
        clamp(
            (complementarity_score * 0.62)
            + ((100.0 - redundancy_score) * 0.18)
            + ((100.0 - event_cluster_overlap) * 0.10)
            + ((100.0 - max(realized_corr, 0.0)) * 0.10),
            0.0,
            100.0,
        ),
        2,
    )
    overlap_confidence = round(
        clamp(
            min(
                float(relationship.get("relationship_confidence") or 20.0),
                float(exposure_similarity.get("exposure_similarity_confidence") or 20.0),
            ),
            18.0,
            92.0,
        ),
        2,
    )
    overlap_drivers = compact_list(
        [
            "realized return correlation is elevated" if realized_corr >= 55.0 else None,
            "factor exposure vectors are highly similar" if factor_similarity >= 68.0 else None,
            "both candidates cluster around the same benchmark / macro profile" if benchmark_overlap >= 80.0 or macro_similarity >= 70.0 else None,
            "fragility loads are similar" if fragility_similarity >= 72.0 else None,
            "event windows overlap" if event_cluster_overlap >= 64.0 else None,
            "theme and narrative clustering are elevated" if theme_overlap >= 80.0 else None,
            "stress correlation rises when the tape is unstable" if stress_adjust >= 12.0 else None,
        ],
        limit=5,
    )
    overlap_rationale = (
        "These candidates behave like a concentrated cluster under realized and factor-adjusted views."
        if hidden_overlap_score >= 72.0
        else "These candidates share some exposure, but diversification remains meaningful."
        if hidden_overlap_score >= 48.0
        else "The candidates appear materially differentiated across realized behavior and factor structure."
    )

    return {
        "peer_symbol": right.get("symbol"),
        "pairwise_correlation": relationship.get("pairwise_correlation"),
        "pairwise_covariance": relationship.get("pairwise_covariance"),
        "correlation_stability_score": relationship.get("correlation_stability_score"),
        "covariance_cluster_score": relationship.get("covariance_cluster_score"),
        "relationship_confidence": relationship.get("relationship_confidence"),
        "overlap_score": overlap_score,
        "redundancy_score": redundancy_score,
        "hidden_overlap_score": hidden_overlap_score,
        "diversification_contribution_score": diversification_contribution_score,
        "complementarity_score": complementarity_score,
        "overlap_rationale": overlap_rationale,
        "overlap_drivers": overlap_drivers,
        "overlap_confidence": overlap_confidence,
        "factor_similarity_score": factor_similarity,
        "event_cluster_overlap": round(event_cluster_overlap, 2),
    }


def summarize_overlap(
    current: Dict[str, Any],
    pairwise_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    relevant = [row for row in pairwise_rows if row.get("symbol") == current.get("symbol")]
    relevant.sort(key=lambda item: float(item.get("hidden_overlap_score") or 0.0), reverse=True)
    top = relevant[:5]
    if not top:
        return (
            {
                "overlap_score": 24.0,
                "redundancy_score": 22.0,
                "hidden_overlap_score": 20.0,
                "diversification_contribution_score": 74.0,
                "complementarity_score": 68.0,
                "overlap_rationale": "No meaningful peer overlap was detected from the tracked cohort.",
                "overlap_drivers": [],
                "overlap_confidence": 24.0,
                "most_redundant_symbol": None,
            },
            [],
        )
    avg_overlap = round(sum(float(item.get("overlap_score") or 0.0) for item in top) / len(top), 2)
    avg_hidden = round(sum(float(item.get("hidden_overlap_score") or 0.0) for item in top) / len(top), 2)
    avg_div = round(sum(float(item.get("diversification_contribution_score") or 0.0) for item in top) / len(top), 2)
    avg_comp = round(sum(float(item.get("complementarity_score") or 0.0) for item in top) / len(top), 2)
    overlap_bucket = score_bucket(avg_hidden)
    rationale = (
        "Realized covariance and exposure structure both indicate substantial hidden overlap."
        if overlap_bucket == "high"
        else "The cohort shows mixed overlap: some shared risk exists, but diversification is still available."
        if overlap_bucket == "moderate"
        else "The cohort appears meaningfully diversified on the current realized and factor-adjusted view."
    )
    return (
        {
            "overlap_score": avg_overlap,
            "redundancy_score": round(max(float(item.get("redundancy_score") or 0.0) for item in top), 2),
            "hidden_overlap_score": avg_hidden,
            "diversification_contribution_score": avg_div,
            "complementarity_score": avg_comp,
            "overlap_rationale": rationale,
            "overlap_drivers": compact_list(
                [driver for item in top for driver in (item.get("overlap_drivers") or [])],
                limit=6,
            ),
            "overlap_confidence": round(
                sum(float(item.get("overlap_confidence") or 0.0) for item in top) / len(top),
                2,
            ),
            "most_redundant_symbol": top[0].get("peer_symbol"),
        },
        top,
    )
