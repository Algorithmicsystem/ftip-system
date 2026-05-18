from __future__ import annotations

from typing import Any, Dict, Optional

from api.axiom.analytics import safe_float
from api.axiom.common import clamp, inverse_score, rounded
from api.axiom.contracts import AxiomPortfolioGovernanceArtifact
from api.axiom.history import AXIOM_PHASE3_VERSION


AXIOM_PORTFOLIO_GOVERNANCE_VERSION = "axiom50_phase3_portfolio_v1"

_DEPLOYABILITY_ORDER = {
    "live_candidate": 4,
    "paper_trade_only": 3,
    "monitor_only": 2,
    "not_actionable": 1,
}


def build_evidence_backed_deployability(
    *,
    current_axiom: Dict[str, Any],
    calibration_summary: Dict[str, Any],
) -> Dict[str, Any]:
    base_tier = str(current_axiom.get("deployability_tier") or "monitor_only")
    engine_scores = current_axiom.get("engine_scores") or {}
    fragility = safe_float(((engine_scores.get("critical_fragility") or {}).get("score"))) or 100.0
    liquidity = safe_float(((engine_scores.get("liquidity_convexity") or {}).get("score"))) or 50.0
    research = safe_float(((engine_scores.get("research_integrity") or {}).get("score"))) or 50.0
    utility = safe_float(current_axiom.get("deployable_alpha_utility")) or 0.0
    coverage = safe_float((current_axiom.get("coverage_summary") or {}).get("overall_coverage")) or 0.0

    matured_count = int(calibration_summary.get("matured_count") or 0)
    supportive_for_live = bool(calibration_summary.get("evidence_supportive_for_live"))
    supportive_for_paper = bool(calibration_summary.get("evidence_supportive_for_paper"))
    calibration_status = str(calibration_summary.get("status") or "insufficient_sample")

    tier = base_tier
    if base_tier == "live_candidate" and (not supportive_for_live or matured_count < 12):
        tier = "paper_trade_only"
    if tier == "paper_trade_only" and not supportive_for_paper and matured_count >= 6:
        tier = "monitor_only"
    if coverage < 45.0 or research < 35.0 or liquidity < 35.0 or fragility >= 78.0:
        tier = "not_actionable"

    size_band = "none"
    if tier == "live_candidate":
        size_band = "large" if fragility <= 28.0 and liquidity >= 70.0 and research >= 72.0 else "medium"
    elif tier == "paper_trade_only":
        size_band = "small"
    elif tier == "monitor_only":
        size_band = "none"

    evidence_summary = (
        f"Evidence-backed deployability is {tier} with calibration status {calibration_status}, "
        f"{matured_count} matured records, research integrity {rounded(research)}, and liquidity integrity {rounded(liquidity)}."
    )
    monitoring_triggers = list(
        (((current_axiom.get("deployability_decision") or {}).get("monitoring_triggers")) or [])
    )
    downgrade_triggers = list(current_axiom.get("invalidation_flags") or [])
    if fragility >= 55.0:
        downgrade_triggers.append("critical_fragility_rising")
    if research < 50.0:
        downgrade_triggers.append("research_integrity_weakening")
    if liquidity < 48.0:
        downgrade_triggers.append("liquidity_integrity_deteriorating")

    return {
        "deployability_tier": tier,
        "size_band": size_band,
        "status": calibration_status,
        "evidence_supportive_for_live": supportive_for_live,
        "evidence_supportive_for_paper": supportive_for_paper,
        "matured_count": matured_count,
        "evidence_summary": evidence_summary,
        "monitoring_triggers": sorted(set(monitoring_triggers)),
        "downgrade_triggers": sorted(set(downgrade_triggers)),
    }


def build_axiom_portfolio_governance(
    *,
    current_report: Dict[str, Any],
    axiom_artifact: Dict[str, Any],
    calibration_summary: Dict[str, Any],
) -> Dict[str, Any]:
    evidence_backed = build_evidence_backed_deployability(
        current_axiom=axiom_artifact,
        calibration_summary=calibration_summary,
    )
    current_candidate = (current_report.get("portfolio_construction") or {}).get("current_candidate") or {}
    hidden_overlap = safe_float(current_report.get("hidden_overlap_score"))
    if hidden_overlap is None:
        hidden_overlap = safe_float(current_candidate.get("overlap_score"))
    hidden_overlap = hidden_overlap or 0.0
    fragility = safe_float(
        (((axiom_artifact.get("engine_scores") or {}).get("critical_fragility") or {}).get("score"))
    ) or 100.0
    liquidity = safe_float(
        (((axiom_artifact.get("engine_scores") or {}).get("liquidity_convexity") or {}).get("score"))
    ) or 50.0
    research = safe_float(
        (((axiom_artifact.get("engine_scores") or {}).get("research_integrity") or {}).get("score"))
    ) or 50.0
    utility = safe_float(axiom_artifact.get("deployable_alpha_utility")) or 0.0
    base_tier = str(axiom_artifact.get("deployability_tier") or "monitor_only")
    portfolio_fit_quality = safe_float(current_report.get("portfolio_fit_quality")) or safe_float(
        current_candidate.get("portfolio_fit_quality")
    ) or 50.0

    overlap_penalty = hidden_overlap * 0.18
    fragility_penalty = max(0.0, fragility - 35.0) * 0.52
    liquidity_penalty = max(0.0, 58.0 - liquidity) * 0.56
    research_penalty = max(0.0, 62.0 - research) * 0.56
    governance_bonus = 6.0 if evidence_backed.get("evidence_supportive_for_live") else 2.5 if evidence_backed.get("evidence_supportive_for_paper") else 0.0

    portfolio_rank_score = clamp(
        utility
        + (portfolio_fit_quality * 0.16)
        + governance_bonus
        - overlap_penalty
        - fragility_penalty
        - liquidity_penalty
        - research_penalty,
        0.0,
        100.0,
    )
    evidence_tier = str(evidence_backed.get("deployability_tier") or base_tier)
    fit_label = "watchlist_only"
    if evidence_tier == "not_actionable" or portfolio_rank_score < 28.0:
        fit_label = "avoid"
    elif str(axiom_artifact.get("trade_family") or "none") == "convexity":
        fit_label = "convexity_overlay"
    elif evidence_tier == "live_candidate" and portfolio_rank_score >= 72.0:
        fit_label = "core_candidate"
    elif portfolio_rank_score >= 50.0:
        fit_label = "tactical_candidate"

    rationale = (
        f"Portfolio governance starts from AXIOM utility {rounded(utility)} and fit quality {rounded(portfolio_fit_quality)}, "
        f"then penalizes overlap {rounded(overlap_penalty)}, fragility {rounded(fragility_penalty)}, liquidity {rounded(liquidity_penalty)}, "
        f"and research {rounded(research_penalty)}."
    )

    artifact = AxiomPortfolioGovernanceArtifact(
        governance_version=AXIOM_PORTFOLIO_GOVERNANCE_VERSION,
        symbol=str(current_report.get("symbol") or current_candidate.get("symbol") or ""),
        as_of_date=str(current_report.get("as_of_date") or ""),
        status=str(evidence_backed.get("status") or "insufficient_sample"),
        base_deployability_tier=base_tier,
        evidence_backed_deployability_tier=evidence_tier,
        portfolio_rank_score=rounded(portfolio_rank_score) or 0.0,
        overlap_penalty=rounded(overlap_penalty) or 0.0,
        fragility_penalty=rounded(fragility_penalty) or 0.0,
        liquidity_penalty=rounded(liquidity_penalty) or 0.0,
        research_penalty=rounded(research_penalty) or 0.0,
        final_size_band=str(evidence_backed.get("size_band") or "none"),
        portfolio_fit_label=fit_label,
        monitoring_triggers=list(evidence_backed.get("monitoring_triggers") or []),
        downgrade_triggers=list(evidence_backed.get("downgrade_triggers") or []),
        evidence_summary=str(evidence_backed.get("evidence_summary") or ""),
        rationale=rationale,
    )
    payload = artifact.model_dump(mode="python")
    payload["inverse_fragility_support"] = rounded(inverse_score(fragility))
    payload["phase3_version"] = AXIOM_PHASE3_VERSION
    return payload
