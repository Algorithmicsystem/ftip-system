from __future__ import annotations

from typing import Any, Dict, List

from .common import clamp, compact_list, permission_rank, trust_rank


_CONVICTION_RANK = {
    "very_high": 92.0,
    "high": 80.0,
    "moderate": 62.0,
    "low": 44.0,
    "very_low": 26.0,
}

_POSTURE_ADJUSTMENT = {
    "actionable_long": 9.0,
    "trend_continuation_candidate": 7.0,
    "opportunistic_reversal": 4.0,
    "watchlist_positive": 2.0,
    "watchlist_negative": -3.0,
    "fragile_hold": -5.0,
    "wait": -7.0,
    "no_trade": -12.0,
    "actionable_short": 5.0,
}


def build_candidate_ranking(
    snapshot: Dict[str, Any],
    execution_quality: Dict[str, Any],
) -> Dict[str, Any]:
    conviction_rank = _CONVICTION_RANK.get(
        str(snapshot.get("conviction_tier") or "low"),
        44.0,
    )
    permission_score = permission_rank(snapshot.get("deployment_permission"))
    trust_score = trust_rank(snapshot.get("trust_tier"))
    opportunity_quality = float(snapshot.get("opportunity_quality_score") or 0.0)
    cross_domain_conviction = float(snapshot.get("cross_domain_conviction_score") or 0.0)
    actionability = float(snapshot.get("actionability_score") or 0.0)
    confidence = float(snapshot.get("confidence_score") or 0.0)
    fundamental_durability = float(snapshot.get("fundamental_durability_score") or 0.0)
    macro_alignment = float(snapshot.get("macro_alignment_score") or 0.0)
    regime_stability = float(snapshot.get("regime_stability_score") or 0.0)
    live_readiness = float(snapshot.get("live_readiness_score") or 0.0)
    evaluation_reliability = float(snapshot.get("evaluation_reliability_score") or 50.0)
    execution_score = float(execution_quality.get("execution_quality_score") or 0.0)
    fragility = float(snapshot.get("signal_fragility_index") or 100.0)
    crowding = float(snapshot.get("narrative_crowding_index") or 0.0)
    axiom_present = bool(snapshot.get("axiom_framework_version"))
    axiom_gross = float(snapshot.get("axiom_gross_opportunity") or 0.0)
    axiom_edge = float(snapshot.get("axiom_validated_edge") or 0.0)
    axiom_utility = float(snapshot.get("axiom_deployable_alpha_utility") or 0.0)
    if not axiom_present:
        axiom_gross = (
            (opportunity_quality * 0.42)
            + (cross_domain_conviction * 0.22)
            + (fundamental_durability * 0.16)
            + (macro_alignment * 0.10)
            + (regime_stability * 0.10)
        )
        axiom_edge = (
            (axiom_gross * 0.34)
            + (confidence * 0.18)
            + (actionability * 0.16)
            + (live_readiness * 0.16)
            + (evaluation_reliability * 0.16)
        )
        axiom_utility = (
            (axiom_edge * 0.38)
            + (execution_score * 0.14)
            + (permission_score * 0.10)
            + (trust_score * 0.08)
            + (max(0.0, 100.0 - fragility) * 0.18)
            + (max(0.0, 100.0 - crowding) * 0.12)
        )
    axiom_friction = float(snapshot.get("axiom_friction_burden") or 0.0)
    if not axiom_present:
        axiom_friction = (
            (fragility * 0.46)
            + (crowding * 0.18)
            + (float(execution_quality.get("friction_penalty") or 0.0) * 0.18)
            + (float(execution_quality.get("turnover_penalty") or 0.0) * 0.08)
            + (max(0.0, 60.0 - evaluation_reliability) * 0.10)
        )
    axiom_fragility = float(snapshot.get("axiom_critical_fragility_score") or fragility)
    axiom_liquidity = float(snapshot.get("axiom_liquidity_convexity_score") or 50.0)
    axiom_research = float(snapshot.get("axiom_research_integrity_score") or 50.0)
    if not axiom_present:
        axiom_liquidity = (execution_score * 0.62) + (max(0.0, 100.0 - fragility) * 0.20) + (
            permission_score * 0.18
        )
        axiom_research = (
            (evaluation_reliability * 0.52)
            + (live_readiness * 0.22)
            + (trust_score * 0.12)
            + (max(0.0, 100.0 - crowding) * 0.14)
        )
    posture_adjust = _POSTURE_ADJUSTMENT.get(
        str(snapshot.get("strategy_posture") or "wait"),
        0.0,
    )

    raw_score = (
        (axiom_utility * 0.28)
        + (axiom_edge * 0.12)
        + (axiom_gross * 0.08)
        + (axiom_liquidity * 0.08)
        + (axiom_research * 0.1)
        + (opportunity_quality * 0.11)
        + (cross_domain_conviction * 0.08)
        + (actionability * 0.06)
        + (confidence * 0.05)
        + (conviction_rank * 0.05)
        + (fundamental_durability * 0.04)
        + (macro_alignment * 0.04)
        + (regime_stability * 0.03)
        + (live_readiness * 0.05)
        + (evaluation_reliability * 0.05)
        + (execution_score * 0.05)
        + (permission_score * 0.06)
        + (trust_score * 0.04)
        - (axiom_fragility * 0.1)
        - (axiom_friction * 0.06)
        - (fragility * 0.05)
        - (crowding * 0.04)
        - (max(0.0, 55.0 - axiom_research) * 0.18)
        - (max(0.0, 52.0 - axiom_liquidity) * 0.14)
        - (float(execution_quality.get("friction_penalty") or 0.0) * 0.03)
        - (float(execution_quality.get("turnover_penalty") or 0.0) * 0.02)
        + posture_adjust
    )

    ranked_opportunity_score = round(clamp(raw_score, 0.0, 100.0), 2)
    quality_vs_fragility_ratio = round(
        clamp(
            50.0
            + (
                max(float(snapshot.get("opportunity_quality_score") or 0.0), axiom_gross)
                - max(float(snapshot.get("signal_fragility_index") or 0.0), axiom_fragility)
            )
            * 0.9,
            0.0,
            100.0,
        ),
        2,
    )
    confidence_adjusted_rank = round(
        clamp(
            ranked_opportunity_score * (0.55 + (float(snapshot.get("confidence_score") or 0.0) / 200.0)),
            0.0,
            100.0,
        ),
        2,
    )
    conviction_adjusted_rank = round(
        clamp(
            ranked_opportunity_score * (0.55 + (conviction_rank / 200.0)),
            0.0,
            100.0,
        ),
        2,
    )
    deployability_rank = round(
        clamp(
            ((axiom_utility or ranked_opportunity_score) * 0.42)
            + (ranked_opportunity_score * 0.16)
            + (permission_score * 0.22)
            + (trust_score * 0.10)
            + ((100.0 - axiom_fragility) * 0.05)
            + (axiom_liquidity * 0.03)
            + (axiom_research * 0.02),
            0.0,
            100.0,
        ),
        2,
    )
    watchlist_priority_score = round(
        clamp(
            (ranked_opportunity_score * 0.52)
            + (quality_vs_fragility_ratio * 0.18)
            + (execution_score * 0.12)
            + (max(0.0, 100.0 - permission_score) * 0.08)
            + (opportunity_quality * 0.10),
            0.0,
            100.0,
        ),
        2,
    )

    positive_contributors: List[str] = []
    if axiom_utility >= 65.0:
        positive_contributors.append("AXIOM deployable alpha utility is strong")
    if axiom_research >= 62.0:
        positive_contributors.append("research integrity is supportive")
    if axiom_liquidity >= 58.0:
        positive_contributors.append("liquidity integrity is supportive")
    if opportunity_quality >= 65.0:
        positive_contributors.append("opportunity quality is strong")
    if cross_domain_conviction >= 65.0:
        positive_contributors.append("cross-domain conviction is supportive")
    if live_readiness >= 70.0:
        positive_contributors.append("deployment readiness is constructive")
    if execution_score >= 62.0:
        positive_contributors.append("execution quality is usable")

    penalties: List[str] = []
    if axiom_fragility >= 55.0:
        penalties.append("AXIOM fragility is elevated")
    if axiom_research < 50.0:
        penalties.append("research integrity is weak")
    if axiom_liquidity < 48.0:
        penalties.append("liquidity integrity is weak")
    if fragility >= 55.0:
        penalties.append("fragility is elevated")
    if crowding >= 60.0:
        penalties.append("narrative crowding is elevated")
    if permission_score < 60.0:
        penalties.append("deployment permission caps deployability")
    if float(execution_quality.get("friction_penalty") or 0.0) >= 48.0:
        penalties.append("execution friction is non-trivial")
    if float(execution_quality.get("turnover_penalty") or 0.0) >= 46.0:
        penalties.append("turnover burden is elevated")

    return {
        "ranked_opportunity_score": ranked_opportunity_score,
        "portfolio_candidate_score": ranked_opportunity_score,
        "watchlist_priority_score": watchlist_priority_score,
        "deployability_rank": deployability_rank,
        "quality_vs_fragility_ratio": quality_vs_fragility_ratio,
        "confidence_adjusted_rank": confidence_adjusted_rank,
        "conviction_adjusted_rank": conviction_adjusted_rank,
        "positive_contributors": compact_list(positive_contributors, limit=5),
        "penalties": compact_list(penalties, limit=5),
    }
