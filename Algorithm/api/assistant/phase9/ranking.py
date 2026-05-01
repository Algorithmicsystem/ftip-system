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
    fragility = float(snapshot.get("signal_fragility_index") or 100.0)
    crowding = float(snapshot.get("narrative_crowding_index") or 0.0)
    posture_adjust = _POSTURE_ADJUSTMENT.get(
        str(snapshot.get("strategy_posture") or "wait"),
        0.0,
    )

    raw_score = (
        (float(snapshot.get("opportunity_quality_score") or 0.0) * 0.21)
        + (float(snapshot.get("cross_domain_conviction_score") or 0.0) * 0.13)
        + (float(snapshot.get("actionability_score") or 0.0) * 0.14)
        + (float(snapshot.get("confidence_score") or 0.0) * 0.11)
        + (conviction_rank * 0.08)
        + (float(snapshot.get("fundamental_durability_score") or 0.0) * 0.07)
        + (float(snapshot.get("macro_alignment_score") or 0.0) * 0.06)
        + (float(snapshot.get("regime_stability_score") or 0.0) * 0.06)
        + (float(snapshot.get("live_readiness_score") or 0.0) * 0.08)
        + (float(snapshot.get("evaluation_reliability_score") or 50.0) * 0.06)
        + (float(execution_quality.get("execution_quality_score") or 0.0) * 0.08)
        + (permission_score * 0.06)
        + (trust_score * 0.04)
        - (fragility * 0.11)
        - (crowding * 0.05)
        - (float(execution_quality.get("friction_penalty") or 0.0) * 0.03)
        - (float(execution_quality.get("turnover_penalty") or 0.0) * 0.02)
        + posture_adjust
    )

    ranked_opportunity_score = round(clamp(raw_score, 0.0, 100.0), 2)
    quality_vs_fragility_ratio = round(
        clamp(
            50.0
            + (
                float(snapshot.get("opportunity_quality_score") or 0.0)
                - float(snapshot.get("signal_fragility_index") or 0.0)
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
            (ranked_opportunity_score * 0.58)
            + (permission_score * 0.22)
            + (trust_score * 0.10)
            + ((100.0 - fragility) * 0.10),
            0.0,
            100.0,
        ),
        2,
    )
    watchlist_priority_score = round(
        clamp(
            (ranked_opportunity_score * 0.52)
            + (quality_vs_fragility_ratio * 0.18)
            + (float(execution_quality.get("execution_quality_score") or 0.0) * 0.12)
            + (max(0.0, 100.0 - permission_score) * 0.08)
            + (float(snapshot.get("opportunity_quality_score") or 0.0) * 0.10),
            0.0,
            100.0,
        ),
        2,
    )

    positive_contributors: List[str] = []
    if float(snapshot.get("opportunity_quality_score") or 0.0) >= 65.0:
        positive_contributors.append("opportunity quality is strong")
    if float(snapshot.get("cross_domain_conviction_score") or 0.0) >= 65.0:
        positive_contributors.append("cross-domain conviction is supportive")
    if float(snapshot.get("live_readiness_score") or 0.0) >= 70.0:
        positive_contributors.append("deployment readiness is constructive")
    if float(execution_quality.get("execution_quality_score") or 0.0) >= 62.0:
        positive_contributors.append("execution quality is usable")

    penalties: List[str] = []
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
