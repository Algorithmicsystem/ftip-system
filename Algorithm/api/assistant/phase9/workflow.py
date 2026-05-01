from __future__ import annotations

from typing import Any, Dict, List

from .common import compact_list


def classify_candidate(
    snapshot: Dict[str, Any],
    ranking: Dict[str, Any],
    overlap_summary: Dict[str, Any],
    size_support: Dict[str, Any],
) -> Dict[str, Any]:
    permission = str(snapshot.get("deployment_permission") or "analysis_only")
    posture = str(snapshot.get("strategy_posture") or "wait")
    fragility = float(snapshot.get("signal_fragility_index") or 100.0)
    ranked_score = float(ranking.get("portfolio_candidate_score") or 0.0)
    deployability = float(ranking.get("deployability_rank") or 0.0)
    overlap_score = float(overlap_summary.get("overlap_score") or 0.0)
    diversification = float(overlap_summary.get("diversification_contribution_score") or 0.0)

    blockers: List[str] = []
    if permission in {"analysis_only", "blocked_weak_evidence", "blocked_paused"}:
        blockers.append("deployment gating blocks portfolio use")
    if fragility >= 60.0:
        blockers.append("fragility is too elevated")
    if overlap_score >= 78.0 and diversification <= 34.0:
        blockers.append("the idea is redundant with existing tracked exposures")
    if size_support.get("size_band", "").startswith("blocked"):
        blockers.append("size support is blocked by the portfolio-control layer")

    if permission in {"blocked_weak_evidence", "blocked_paused"}:
        classification = "blocked_candidate"
    elif overlap_score >= 78.0 and diversification <= 34.0:
        classification = "redundant_candidate"
    elif fragility >= 60.0:
        classification = "too_fragile_candidate"
    elif posture in {"actionable_short"} or str(snapshot.get("final_signal") or "").upper() == "SELL":
        classification = "hedge/context_candidate"
    elif ranked_score >= 76.0 and deployability >= 72.0 and permission.endswith("eligible"):
        classification = "top_priority_candidate"
    elif ranked_score >= 62.0 and permission not in {"analysis_only", "blocked_weak_evidence", "blocked_paused"}:
        classification = "secondary_candidate"
    elif posture in {"wait", "no_trade", "fragile_hold"}:
        classification = "no_trade_candidate"
    else:
        classification = "watchlist_candidate"

    return {
        "candidate_classification": classification,
        "candidate_blockers": compact_list(blockers, limit=6),
        "portfolio_usable": classification in {
            "top_priority_candidate",
            "secondary_candidate",
            "hedge/context_candidate",
        },
    }


def build_workflow_state(
    current: Dict[str, Any],
    ranking: Dict[str, Any],
    classification: Dict[str, Any],
    overlap_summary: Dict[str, Any],
    top_candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    higher_ranked_peer = next(
        (
            item
            for item in top_candidates
            if item.get("symbol") != current.get("symbol")
            and float(item.get("portfolio_candidate_score") or 0.0)
            > float(ranking.get("portfolio_candidate_score") or 0.0)
        ),
        None,
    )
    priority_shift_flag = bool(
        higher_ranked_peer
        and float(overlap_summary.get("overlap_score") or 0.0) >= 68.0
        and (
            float(higher_ranked_peer.get("portfolio_candidate_score") or 0.0)
            - float(ranking.get("portfolio_candidate_score") or 0.0)
            >= 6.0
        )
    )
    rebalance_attention_flag = priority_shift_flag or classification["candidate_classification"] in {
        "redundant_candidate",
        "too_fragile_candidate",
        "blocked_candidate",
    }
    candidate_upgrade_reason = (
        "ranked opportunity, deployability, and portfolio fit are strong enough to justify elevated review priority"
        if classification["candidate_classification"] == "top_priority_candidate"
        else None
    )
    candidate_downgrade_reason = (
        "portfolio redundancy or deployment gating is capping priority"
        if classification["candidate_classification"] in {"redundant_candidate", "blocked_candidate"}
        else "fragility and execution caution are capping priority"
        if classification["candidate_classification"] == "too_fragile_candidate"
        else None
    )
    replacement_candidate_notes = (
        f"{higher_ranked_peer.get('symbol')} currently offers a cleaner portfolio-adjusted candidate score."
        if higher_ranked_peer
        else None
    )
    rotation_pressure_score = round(
        max(
            0.0,
            min(
                100.0,
                (
                    (float(higher_ranked_peer.get("portfolio_candidate_score") or 0.0)
                    - float(ranking.get("portfolio_candidate_score") or 0.0))
                    if higher_ranked_peer
                    else 0.0
                )
                + (float(overlap_summary.get("overlap_score") or 0.0) * 0.35),
            ),
        ),
        2,
    )

    return {
        "candidate_watchlist": [
            item.get("symbol")
            for item in top_candidates
            if item.get("candidate_classification") in {"watchlist_candidate", "secondary_candidate"}
        ][:8],
        "prioritized_watchlist": [
            item.get("symbol")
            for item in top_candidates
            if item.get("candidate_classification") in {"top_priority_candidate", "secondary_candidate"}
        ][:8],
        "active_portfolio_candidates": [
            item.get("symbol")
            for item in top_candidates
            if item.get("candidate_classification") in {
                "top_priority_candidate",
                "secondary_candidate",
                "hedge/context_candidate",
            }
        ][:8],
        "blocked_candidates": [
            {
                "symbol": item.get("symbol"),
                "classification": item.get("candidate_classification"),
                "reasons": item.get("candidate_blockers") or [],
            }
            for item in top_candidates
            if item.get("candidate_classification")
            in {"blocked_candidate", "redundant_candidate", "too_fragile_candidate", "no_trade_candidate"}
        ][:8],
        "stale_review_needed": [
            item.get("symbol")
            for item in top_candidates
            if str(item.get("freshness_status") or "unknown") in {"mixed_stale", "stale", "stale_but_usable"}
        ][:8],
        "priority_shift_flag": priority_shift_flag,
        "rebalance_attention_flag": rebalance_attention_flag,
        "candidate_upgrade_reason": candidate_upgrade_reason,
        "candidate_downgrade_reason": candidate_downgrade_reason,
        "replacement_candidate_notes": replacement_candidate_notes,
        "rotation_pressure_score": rotation_pressure_score,
    }
