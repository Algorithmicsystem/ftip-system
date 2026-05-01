from __future__ import annotations

from typing import Any, Dict

from .common import clamp, permission_rank


def build_size_band_support(
    snapshot: Dict[str, Any],
    ranking: Dict[str, Any],
    execution_quality: Dict[str, Any],
    overlap_summary: Dict[str, Any],
    exposure_framework: Dict[str, Any],
) -> Dict[str, Any]:
    permission = str(snapshot.get("deployment_permission") or "analysis_only")
    base_band_map = {
        "scaled_live_eligible": ("standard allocation band", "1.00x governed band", "standard_risk_band"),
        "limited_live_eligible": ("standard allocation band", "0.75x disciplined band", "disciplined_risk_band"),
        "low_risk_live_eligible": ("exploratory allocation band", "0.50x pilot band", "pilot_risk_band"),
        "paper_shadow_only": ("paper / shadow band", "0.00x live weight", "shadow_risk_band"),
        "watchlist_only": ("watchlist only", "0.00x live weight", "watchlist_risk_band"),
        "analysis_only": ("blocked for size", "0.00x live weight", "research_only_band"),
        "blocked_weak_evidence": ("blocked for size", "0.00x live weight", "blocked_risk_band"),
        "blocked_paused": ("blocked for size", "0.00x live weight", "paused_risk_band"),
    }
    size_band, weight_band, risk_budget_band = base_band_map.get(
        permission,
        ("watchlist only", "0.00x live weight", "watchlist_risk_band"),
    )

    fragility = float(snapshot.get("signal_fragility_index") or 100.0)
    overlap_score = float(overlap_summary.get("overlap_score") or 0.0)
    diversification = float(overlap_summary.get("diversification_contribution_score") or 50.0)
    execution_score = float(execution_quality.get("execution_quality_score") or 0.0)
    readiness = float(snapshot.get("live_readiness_score") or 0.0)
    base_rank = float(ranking.get("portfolio_candidate_score") or 0.0)

    fragility_adjustment = "neutral"
    confidence_adjustment = "neutral"
    concentration_adjustment = "neutral"
    overlap_adjustment = "neutral"
    deployment_mode_adjustment = "neutral"
    caution_level = "measured"

    if fragility >= 58.0:
        fragility_adjustment = "reduce_due_to_fragility"
        caution_level = "high"
    elif fragility >= 46.0:
        fragility_adjustment = "moderate_fragility_haircut"
        caution_level = "elevated"

    if float(snapshot.get("confidence_score") or 0.0) < 55.0:
        confidence_adjustment = "reduce_due_to_confidence"
        caution_level = "high"

    if overlap_score >= 76.0 or diversification <= 34.0:
        overlap_adjustment = "reduce_due_to_overlap"
        concentration_adjustment = "reduce_due_to_redundancy"
        caution_level = "high"
    elif overlap_score >= 62.0:
        overlap_adjustment = "moderate_overlap_haircut"
        concentration_adjustment = "moderate_cluster_haircut"
        caution_level = "elevated"

    if exposure_framework.get("active_warnings"):
        concentration_adjustment = "reduce_due_to_cluster_concentration"
        if caution_level == "measured":
            caution_level = "elevated"

    if permission_rank(permission) < 60.0 or readiness < 70.0:
        deployment_mode_adjustment = "reduce_due_to_deployment_gate"
        caution_level = "high" if caution_level != "extreme" else caution_level

    if execution_score < 52.0:
        caution_level = "high"
    if permission in {"analysis_only", "blocked_weak_evidence", "blocked_paused"}:
        caution_level = "extreme"

    if permission in {"analysis_only", "blocked_weak_evidence", "blocked_paused"}:
        max_priority_allowed = "watchlist_only"
    elif overlap_score >= 76.0 and base_rank < 78.0:
        max_priority_allowed = "secondary_only"
    elif readiness < 72.0:
        max_priority_allowed = "paper_only"
    else:
        max_priority_allowed = "top_priority"

    if fragility >= 58.0 or overlap_score >= 82.0:
        size_band = "reduced allocation band"
        weight_band = "0.25x-0.50x guarded weight"
    if permission in {"analysis_only", "blocked_weak_evidence", "blocked_paused"}:
        size_band = "blocked for size due to weak live-readiness"
    elif permission == "paper_shadow_only":
        size_band = "paper / shadow band"
    elif execution_score < 50.0:
        size_band = "reduced allocation band"
        weight_band = "0.25x-0.40x guarded weight"

    return {
        "size_band": size_band,
        "weight_band": weight_band,
        "risk_budget_band": risk_budget_band,
        "fragility_adjustment": fragility_adjustment,
        "confidence_adjustment": confidence_adjustment,
        "concentration_adjustment": concentration_adjustment,
        "overlap_adjustment": overlap_adjustment,
        "deployment_mode_adjustment": deployment_mode_adjustment,
        "max_priority_allowed": max_priority_allowed,
        "caution_level": caution_level,
    }
