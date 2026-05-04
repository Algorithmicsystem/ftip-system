from __future__ import annotations

from typing import Any, Dict, List

from .common import clamp, compact_list, safe_float


def enrich_with_marginal_utility(
    rows: List[Dict[str, Any]],
    *,
    portfolio_overlay: Dict[str, Any],
) -> List[Dict[str, Any]]:
    hidden_concentration = safe_float(portfolio_overlay.get("hidden_concentration_score")) or 30.0
    cluster_risk = safe_float(portfolio_overlay.get("cluster_risk_score")) or 30.0
    portfolio_stress = safe_float(portfolio_overlay.get("portfolio_stress_score")) or 30.0

    enriched: List[Dict[str, Any]] = []
    for row in rows:
        base_score = safe_float(row.get("portfolio_candidate_score")) or safe_float(
            row.get("ranked_opportunity_score")
        ) or 0.0
        diversification_bonus = (safe_float(row.get("diversification_contribution_score")) or 0.0) * 0.22
        complementarity_bonus = (safe_float(row.get("complementarity_score")) or 0.0) * 0.14
        fragility_penalty = (
            (safe_float(row.get("signal_fragility_index")) or 0.0) * 0.12
            + (safe_float(row.get("implementation_fragility_score")) or 0.0) * 0.08
        )
        overlap_penalty = (
            (safe_float(row.get("hidden_overlap_score")) or 0.0) * 0.18
            + (safe_float(row.get("redundancy_score")) or 0.0) * 0.08
        )
        concentration_penalty = hidden_concentration * 0.06 + cluster_risk * 0.04
        readiness_bonus = (safe_float(row.get("live_readiness_score")) or 0.0) * 0.08
        execution_penalty = (
            (safe_float(row.get("friction_penalty")) or 0.0) * 0.10
            + (safe_float(row.get("turnover_penalty")) or 0.0) * 0.08
        )
        stress_penalty = portfolio_stress * 0.05 + (safe_float(row.get("market_stress_score")) or 0.0) * 0.07

        marginal_utility = round(
            clamp(
                base_score
                + diversification_bonus
                + complementarity_bonus
                + readiness_bonus
                - fragility_penalty
                - overlap_penalty
                - concentration_penalty
                - execution_penalty
                - stress_penalty,
                0.0,
                100.0,
            ),
            2,
        )
        enriched.append(
            {
                **row,
                "marginal_portfolio_utility": marginal_utility,
                "portfolio_contribution_score": round(
                    clamp(
                        (marginal_utility * 0.64)
                        + ((safe_float(row.get("portfolio_fit_quality")) or 0.0) * 0.18)
                        + ((safe_float(row.get("deployability_rank")) or 0.0) * 0.08)
                        + ((safe_float(row.get("execution_quality_score")) or 0.0) * 0.10),
                        0.0,
                        100.0,
                    ),
                    2,
                ),
                "marginal_fragility_penalty": round(fragility_penalty, 2),
                "marginal_diversification_bonus": round(
                    diversification_bonus + complementarity_bonus,
                    2,
                ),
                "replacement_value_score": round(
                    clamp(
                        (safe_float(row.get("portfolio_fit_quality")) or 0.0) * 0.32
                        + (safe_float(row.get("diversification_contribution_score")) or 0.0) * 0.26
                        + (safe_float(row.get("quality_vs_fragility_ratio")) or 0.0) * 0.22
                        - (safe_float(row.get("hidden_overlap_score")) or 0.0) * 0.20,
                        0.0,
                        100.0,
                    ),
                    2,
                ),
            }
        )

    enriched.sort(
        key=lambda item: (
            float(item.get("marginal_portfolio_utility") or 0.0),
            float(item.get("portfolio_contribution_score") or 0.0),
        ),
        reverse=True,
    )
    for index, row in enumerate(enriched, start=1):
        row["portfolio_fit_rank"] = index

    for row in enriched:
        better_alternative = next(
            (
                peer
                for peer in enriched
                if peer.get("symbol") != row.get("symbol")
                and float(peer.get("marginal_portfolio_utility") or 0.0)
                > float(row.get("marginal_portfolio_utility") or 0.0) + 6.0
                and float(peer.get("hidden_overlap_score") or 0.0)
                <= float(row.get("hidden_overlap_score") or 0.0) + 4.0
            ),
            None,
        )
        row["replacement_candidate"] = better_alternative.get("symbol") if better_alternative else None
        row["substitution_score"] = round(
            clamp(
                (
                    (float(better_alternative.get("marginal_portfolio_utility") or 0.0) - float(row.get("marginal_portfolio_utility") or 0.0))
                    if better_alternative
                    else 0.0
                )
                + (float(row.get("hidden_overlap_score") or 0.0) * 0.30),
                0.0,
                100.0,
            ),
            2,
        )
        row["better_alternative_flag"] = better_alternative is not None
        row["diversification_upgrade_flag"] = bool(
            better_alternative
            and float(better_alternative.get("diversification_contribution_score") or 0.0)
            > float(row.get("diversification_contribution_score") or 0.0) + 8.0
        )
        row["overlap_reduction_flag"] = bool(
            better_alternative
            and float(better_alternative.get("hidden_overlap_score") or 0.0)
            < float(row.get("hidden_overlap_score") or 0.0) - 8.0
        )
        row["portfolio_quality_upgrade_reason"] = (
            f"{better_alternative.get('symbol')} offers a higher marginal utility with cleaner diversification."
            if better_alternative
            else None
        )

        blockers = list(row.get("candidate_blockers") or [])
        if float(row.get("hidden_overlap_score") or 0.0) >= 74.0 and float(
            row.get("diversification_contribution_score") or 0.0
        ) <= 36.0:
            blockers.append("hidden overlap is too elevated for incremental capital")
        if row.get("better_alternative_flag"):
            blockers.append("a cleaner portfolio substitute is currently available")
        row["candidate_blockers"] = compact_list(blockers, limit=8)
        if (
            row.get("candidate_classification") in {"top_priority_candidate", "secondary_candidate"}
            and float(row.get("hidden_overlap_score") or 0.0) >= 78.0
            and row.get("better_alternative_flag")
        ):
            row["candidate_classification"] = "redundant_candidate"
        elif (
            row.get("candidate_classification") == "watchlist_candidate"
            and float(row.get("marginal_portfolio_utility") or 0.0) >= 70.0
            and float(row.get("portfolio_fit_quality") or 0.0) >= 62.0
        ):
            row["candidate_classification"] = "secondary_candidate"
    return enriched
