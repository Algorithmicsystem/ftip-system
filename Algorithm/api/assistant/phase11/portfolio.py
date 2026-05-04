from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .common import clamp, compact_list, mean, safe_float, score_bucket


def _active_candidates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    active = [
        row
        for row in rows
        if str(row.get("candidate_classification") or "") in {
            "top_priority_candidate",
            "secondary_candidate",
            "hedge/context_candidate",
        }
    ]
    return active or rows[: min(5, len(rows))]


def build_portfolio_risk_overlay(
    rows: List[Dict[str, Any]],
    pairwise_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    active = _active_candidates(rows)
    cohort_size = max(len(active), 1)
    sector_counts = Counter(str(row.get("sector") or "unknown") for row in active)
    benchmark_counts = Counter(str(row.get("benchmark_proxy") or "unknown") for row in active)
    exposure_counts = Counter(str(row.get("exposure_cluster") or "unknown") for row in active)
    macro_counts = Counter(str(row.get("macro_state") or "mixed") for row in active)
    theme_counts = Counter(str(row.get("theme_tag") or "broad_theme") for row in active)
    event_counts = Counter(score_bucket(row.get("event_overhang_score")) for row in active)
    fragility_counts = Counter(score_bucket(row.get("implementation_fragility_score") or row.get("signal_fragility_index")) for row in active)

    active_symbols = {str(row.get("symbol") or "") for row in active}
    active_pairs = [
        row
        for row in pairwise_rows
        if str(row.get("symbol") or "") in active_symbols
        and str(row.get("peer_symbol") or "") in active_symbols
    ]
    average_hidden_overlap = mean(row.get("hidden_overlap_score") for row in active_pairs) or 24.0
    average_pairwise_correlation = mean(
        (abs(safe_float(row.get("pairwise_correlation")) or 0.0) * 100.0)
        for row in active_pairs
    ) or 22.0
    average_fragility = mean(
        row.get("implementation_fragility_score") or row.get("signal_fragility_index")
        for row in active
    ) or 36.0
    average_stress = mean(row.get("market_stress_score") for row in active) or 34.0
    average_gap_risk = mean(row.get("gap_instability_10d") for row in active) or 0.12

    max_sector_share = max(sector_counts.values()) / cohort_size
    max_exposure_share = max(exposure_counts.values()) / cohort_size
    max_macro_share = max(macro_counts.values()) / cohort_size
    max_theme_share = max(theme_counts.values()) / cohort_size
    high_event_share = event_counts.get("high", 0) / cohort_size
    high_fragility_share = fragility_counts.get("high", 0) / cohort_size

    hidden_concentration_score = round(
        clamp(
            (average_hidden_overlap * 0.34)
            + (average_pairwise_correlation * 0.18)
            + (max_sector_share * 100.0 * 0.10)
            + (max_exposure_share * 100.0 * 0.16)
            + (max_macro_share * 100.0 * 0.08)
            + (max_theme_share * 100.0 * 0.08)
            + (high_event_share * 100.0 * 0.03)
            + (high_fragility_share * 100.0 * 0.03),
            0.0,
            100.0,
        ),
        2,
    )
    cluster_risk_score = round(
        clamp(
            (hidden_concentration_score * 0.46)
            + (max_exposure_share * 100.0 * 0.22)
            + (max_sector_share * 100.0 * 0.12)
            + (high_event_share * 100.0 * 0.10)
            + (high_fragility_share * 100.0 * 0.10),
            0.0,
            100.0,
        ),
        2,
    )
    fragility_cluster_risk = round(
        clamp((average_fragility * 0.68) + (high_fragility_share * 100.0 * 0.32), 0.0, 100.0),
        2,
    )
    event_cluster_risk = round(
        clamp((high_event_share * 100.0 * 0.62) + (mean(row.get("event_overhang_score") for row in active) or 0.0) * 0.38, 0.0, 100.0),
        2,
    )
    macro_cluster_risk = round(
        clamp((max_macro_share * 100.0 * 0.58) + (mean(row.get("cross_asset_conflict_score") for row in active) or 0.0) * 0.42, 0.0, 100.0),
        2,
    )
    narrative_cluster_risk = round(
        clamp((max_theme_share * 100.0 * 0.56) + (mean(row.get("narrative_crowding_index") for row in active) or 0.0) * 0.44, 0.0, 100.0),
        2,
    )
    diversification_health_score = round(
        clamp(
            100.0
            - (hidden_concentration_score * 0.38)
            - (cluster_risk_score * 0.22)
            - (fragility_cluster_risk * 0.18)
            + max(0.0, 18.0 - average_pairwise_correlation) * 0.9,
            0.0,
            100.0,
        ),
        2,
    )
    portfolio_stress_score = round(
        clamp(
            (average_stress * 0.42)
            + (macro_cluster_risk * 0.20)
            + (average_hidden_overlap * 0.12)
            + (event_cluster_risk * 0.10)
            + (fragility_cluster_risk * 0.16),
            0.0,
            100.0,
        ),
        2,
    )
    portfolio_fragility_score = round(
        clamp(
            (average_fragility * 0.36)
            + (fragility_cluster_risk * 0.26)
            + (event_cluster_risk * 0.12)
            + (portfolio_stress_score * 0.14)
            + (min(average_gap_risk, 1.0) * 100.0 * 0.12),
            0.0,
            100.0,
        ),
        2,
    )
    correlation_breakdown_risk = round(
        clamp(
            (mean(row.get("correlation_breakdown_risk") for row in active_pairs) or 0.0) * 0.58
            + (portfolio_stress_score * 0.24)
            + (cluster_risk_score * 0.18),
            0.0,
            100.0,
        ),
        2,
    )

    hidden_warning = (
        f"Hidden overlap is elevated: realized and factor-adjusted overlap is clustering around {round(average_hidden_overlap, 1)} / 100."
        if hidden_concentration_score >= 62.0
        else None
    )
    exposure_warning = (
        f"Exposure clustering is concentrated in {max(exposure_counts, key=exposure_counts.get)}."
        if max_exposure_share >= 0.5
        else None
    )
    sector_warning = (
        f"Sector concentration is leaning heavily into {max(sector_counts, key=sector_counts.get)}."
        if max_sector_share >= 0.5
        else None
    )
    event_warning = (
        "Event windows are clustered across the active candidate set."
        if high_event_share >= 0.34
        else None
    )
    fragility_warning = (
        "Fragility clustering is elevated across the active candidate set."
        if high_fragility_share >= 0.34
        else None
    )
    breakdown_warning = (
        "Correlation breakdown risk is elevated if the market slips into a stress transition."
        if correlation_breakdown_risk >= 58.0
        else None
    )
    warnings = compact_list(
        [
            hidden_warning,
            exposure_warning,
            sector_warning,
            event_warning,
            fragility_warning,
            breakdown_warning,
        ],
        limit=6,
    )

    return {
        "concentration_profile": {
            "sector": dict(sector_counts),
            "benchmark": dict(benchmark_counts),
            "exposure_cluster": dict(exposure_counts),
            "macro_state": dict(macro_counts),
            "theme": dict(theme_counts),
            "event_bucket": dict(event_counts),
            "fragility_bucket": dict(fragility_counts),
        },
        "hidden_concentration_score": hidden_concentration_score,
        "cluster_risk_score": cluster_risk_score,
        "fragility_cluster_risk": fragility_cluster_risk,
        "event_cluster_risk": event_cluster_risk,
        "macro_cluster_risk": macro_cluster_risk,
        "narrative_cluster_risk": narrative_cluster_risk,
        "diversification_health_score": diversification_health_score,
        "portfolio_stress_score": portfolio_stress_score,
        "portfolio_fragility_score": portfolio_fragility_score,
        "correlation_breakdown_risk": correlation_breakdown_risk,
        "stress_concentration_warning": hidden_warning or breakdown_warning,
        "clustered_gap_risk_warning": fragility_warning or event_warning,
        "unstable_portfolio_state_flag": (
            portfolio_stress_score >= 62.0
            or portfolio_fragility_score >= 62.0
            or correlation_breakdown_risk >= 62.0
        ),
        "portfolio_risk_warnings": warnings,
    }
