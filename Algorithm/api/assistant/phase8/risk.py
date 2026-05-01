from __future__ import annotations

from typing import Any, Dict, List

from .common import safe_float, score_value


def _size_band_from_score(score: float) -> str:
    if score < 35.0:
        return "none"
    if score < 50.0:
        return "0.10x-0.25x pilot unit"
    if score < 62.0:
        return "0.25x-0.40x pilot unit"
    if score < 76.0:
        return "0.40x-0.60x disciplined unit"
    return "0.60x-0.85x governed unit"


def _risk_budget_tier(permission: str, readiness_score: float) -> str:
    if permission in {"blocked_paused", "blocked_weak_evidence", "analysis_only"}:
        return "none"
    if permission in {"watchlist_only", "paper_shadow_only"}:
        return "shadow_only"
    if permission == "low_risk_live_eligible":
        return "pilot_probe"
    if permission == "limited_live_eligible":
        return "disciplined_small"
    if permission == "scaled_live_eligible" and readiness_score >= 86.0:
        return "governed_scale_candidate"
    return "measured"


def build_risk_budget_framework(
    report: Dict[str, Any],
    *,
    model_readiness: Dict[str, Any],
    deployment_permission: Dict[str, Any],
    deployment_mode_state: Dict[str, Any],
) -> Dict[str, Any]:
    strategy = report.get("strategy") or {}
    macro = report.get("macro_intelligence") or {}
    relative = (report.get("data_bundle") or {}).get("relative_context") or {}
    proprietary_scores = report.get("proprietary_scores") or {}

    readiness_score = safe_float(model_readiness.get("live_readiness_score")) or 0.0
    confidence_score = safe_float(strategy.get("confidence_score")) or 0.0
    fragility_score = score_value(proprietary_scores.get("Signal Fragility Index")) or 60.0
    macro_alignment = score_value(proprietary_scores.get("Macro Alignment Score")) or 50.0
    crowding = score_value(proprietary_scores.get("Narrative Crowding Index")) or 50.0
    permission = str(deployment_permission.get("deployment_permission") or "analysis_only")

    fragility_adjusted_score = max(0.0, min(readiness_score, 100.0 - fragility_score))
    confidence_adjusted_score = min(readiness_score, confidence_score)
    risk_budget_tier = _risk_budget_tier(permission, readiness_score)

    exposure_caution_level = (
        "extreme"
        if permission in {"blocked_paused", "blocked_weak_evidence", "analysis_only"}
        else "high"
        if fragility_score >= 58.0 or readiness_score < 60.0
        else "elevated"
        if crowding >= 60.0 or macro_alignment < 45.0
        else "moderate"
        if permission == "paper_shadow_only"
        else "measured"
    )

    max_risk_mode_allowed = deployment_mode_state.get("max_risk_mode_allowed")
    if readiness_score >= 86.0 and fragility_score <= 35.0 and macro_alignment >= 62.0:
        max_risk_mode_allowed = "scaled_live"
    elif readiness_score >= 79.0 and fragility_score <= 42.0:
        max_risk_mode_allowed = "limited_live"
    elif readiness_score >= 72.0 and fragility_score <= 50.0:
        max_risk_mode_allowed = "low_risk_live"
    elif readiness_score >= 45.0:
        max_risk_mode_allowed = "paper_shadow"
    else:
        max_risk_mode_allowed = "research_only"

    concentration_warning = (
        "Narrative crowding and fragility are both elevated; concentration should stay explicitly constrained."
        if crowding >= 60.0 and fragility_score >= 52.0
        else "No concentration red flag dominates, but size should still respect deployment mode caps."
    )
    diversification_warning = (
        "Macro alignment is conflicted or mixed, so diversification assumptions should not rely on one clean beta regime."
        if macro_alignment < 50.0
        else "Diversification assumptions are usable, but still need cross-position review."
    )
    correlation_warning = (
        relative.get("relative_move_summary", {}).get("market_relative_note")
        or relative.get("relative_move_summary", {}).get("sector_relative_note")
        or macro.get("macro_conflict_score")
        or "Benchmark / correlation context is not strong enough to ignore spillover risk."
    )

    return {
        "risk_budget_tier": risk_budget_tier,
        "exposure_caution_level": exposure_caution_level,
        "concentration_warning": concentration_warning,
        "fragility_adjusted_size_band": _size_band_from_score(fragility_adjusted_score),
        "confidence_adjusted_size_band": _size_band_from_score(confidence_adjusted_score),
        "maximum_risk_mode_allowed": max_risk_mode_allowed,
        "diversification_warning": diversification_warning,
        "correlation_context_warning": str(correlation_warning),
    }
