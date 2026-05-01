from __future__ import annotations

from typing import Any, Dict, List

from .common import safe_float, sample_confidence


def _templates() -> List[Dict[str, Any]]:
    return [
        {
            "interaction_candidate": "trend_plus_low_fragility_plus_macro_alignment",
            "description": "clean structure + low fragility + supportive macro alignment",
            "predicate": lambda row: (
                (safe_float(row.get("market_structure_integrity_score")) or 0.0) >= 62.0
                and (safe_float(row.get("signal_fragility_index")) or 100.0) <= 40.0
                and (safe_float(row.get("macro_alignment_score")) or 0.0) >= 60.0
            ),
        },
        {
            "interaction_candidate": "high_conviction_plus_portfolio_fit",
            "description": "high cross-domain conviction + healthy portfolio fit + live-eligible gate",
            "predicate": lambda row: (
                (safe_float(row.get("cross_domain_conviction_score")) or 0.0) >= 65.0
                and (safe_float(row.get("portfolio_fit_quality")) or 0.0) >= 58.0
                and str(row.get("deployment_permission") or "").endswith("eligible")
            ),
        },
        {
            "interaction_candidate": "crowding_plus_fragility_failure_cluster",
            "description": "high crowding + elevated fragility + weaker opportunity quality",
            "predicate": lambda row: (
                (safe_float(row.get("narrative_crowding_index")) or 0.0) >= 60.0
                and (safe_float(row.get("signal_fragility_index")) or 0.0) >= 55.0
                and (safe_float(row.get("opportunity_quality_score")) or 100.0) <= 65.0
            ),
        },
        {
            "interaction_candidate": "watchlist_overlap_drag",
            "description": "watchlist or redundant candidate with elevated overlap and modest portfolio fit",
            "predicate": lambda row: (
                str(row.get("candidate_classification") or "")
                in {"watchlist_candidate", "redundant_candidate", "no_trade_candidate"}
                and (safe_float(row.get("overlap_score")) or 0.0) >= 68.0
                and (safe_float(row.get("portfolio_fit_quality")) or 100.0) <= 55.0
            ),
        },
    ]


def build_feature_interaction_candidates(cohort: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    interactions: List[Dict[str, Any]] = []
    for template in _templates():
        matches = [row for row in cohort if template["predicate"](row)]
        sample_size = len(matches)
        if sample_size == 0:
            continue
        avg_hit_rate = (
            sum(
                safe_float(row.get("evaluation_hit_rate")) or 0.0
                for row in matches
            )
            / sample_size
        )
        avg_portfolio = (
            sum(
                safe_float(row.get("portfolio_candidate_score")) or 0.0
                for row in matches
            )
            / sample_size
        )
        motif_strength = round(
            max(
                0.0,
                min(
                    100.0,
                    (avg_portfolio * 0.55)
                    + (avg_hit_rate * 100.0 * 0.25)
                    + (sample_size * 4.0),
                ),
            ),
            2,
        )
        interactions.append(
            {
                "interaction_candidate": template["interaction_candidate"],
                "description": template["description"],
                "sample_size": sample_size,
                "sample_confidence": sample_confidence(sample_size),
                "conditional_usefulness": (
                    "constructive"
                    if avg_hit_rate >= 0.58 and avg_portfolio >= 62.0
                    else "cautionary"
                    if avg_hit_rate <= 0.5
                    else "mixed"
                ),
                "motif_strength": motif_strength,
                "risk_of_false_pattern": "high" if sample_size < 4 else "moderate" if sample_size < 8 else "low",
                "validation_status": (
                    "validated"
                    if sample_size >= 8 and motif_strength >= 62.0
                    else "under_review"
                    if sample_size >= 4
                    else "exploratory"
                ),
            }
        )
    interactions.sort(
        key=lambda item: (float(item.get("motif_strength") or 0.0), item.get("sample_size") or 0),
        reverse=True,
    )
    return interactions[:8]
