from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from api.axiom.common import rounded
from api.axiom.contracts import (
    AxiomArtifact,
    AxiomInstitutionalReportPack,
    AxiomWorkspaceProfile,
)


AXIOM_REPORTING_VERSION = "axiom50_phase4_reporting_v1"
AXIOM_REPORT_PACK_ARTIFACT_KIND = "assistant_axiom_report_pack_artifact"


def _text_list(values: Sequence[Any], *, limit: int = 4) -> List[str]:
    items: List[str] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        items.append(str(value))
        if len(items) >= limit:
            break
    return items


def _engine_label(payload: Dict[str, Any]) -> str:
    name = str(payload.get("engine") or "unknown")
    score = payload.get("score")
    return f"{name.replace('_', ' ')} ({rounded(score) if score is not None else 'n/a'})"


def _historical_evidence_payload(
    axiom: AxiomArtifact,
    report_context: Dict[str, Any],
) -> Dict[str, Any]:
    calibration = dict(axiom.calibration_summary or {})
    historical = dict(axiom.historical_evidence or {})
    history_record = report_context.get("axiom_history_record") or {}
    deployability_rows = list(calibration.get("deployability_tier_outcome_summary") or [])
    regime_rows = list(calibration.get("regime_outcome_summary") or [])
    dau_buckets = ((calibration.get("dau_bucket_summary") or {}).get("buckets") or [])
    top_bucket = dau_buckets[0] if dau_buckets else {}
    bottom_bucket = dau_buckets[-1] if dau_buckets else {}
    recent_symbol_evidence: List[str] = []
    for horizon, payload in (history_record.get("forward_outcomes") or {}).items():
        if not isinstance(payload, dict):
            continue
        if payload.get("matured"):
            recent_symbol_evidence.append(
                f"{horizon}: net edge {rounded(payload.get('net_edge_return'), digits=4)} with MAE {rounded(payload.get('mae'), digits=4)} and MFE {rounded(payload.get('mfe'), digits=4)}."
            )
    evidence_notes = [
        f"Calibration status is {str(calibration.get('status') or historical.get('status') or 'partial').replace('_', ' ')}.",
        f"Matured record count is {calibration.get('matured_count', 0)} on the {calibration.get('horizon_label', historical.get('history_horizon_label', '21d'))} horizon.",
    ]
    if top_bucket or bottom_bucket:
        evidence_notes.append(
            f"Top-vs-bottom DAU bucket spread is {rounded(calibration.get('dau_spread'), digits=4)} with strongest bucket average net edge {rounded(top_bucket.get('average_net_edge_return'), digits=4)} and weakest bucket average net edge {rounded(bottom_bucket.get('average_net_edge_return'), digits=4)}."
        )
    if not recent_symbol_evidence:
        recent_symbol_evidence.append("Recent symbol-specific matured evidence is limited, so historical support remains sample-constrained.")
    return {
        "status": calibration.get("status") or historical.get("status") or "partial",
        "horizon_label": calibration.get("horizon_label") or historical.get("history_horizon_label") or "21d",
        "matured_count": calibration.get("matured_count", 0),
        "dau_bucket_stats": {
            "spread": calibration.get("dau_spread"),
            "strongest_bucket": top_bucket,
            "weakest_bucket": bottom_bucket,
        },
        "regime_outcome_stats": regime_rows[:5],
        "deployability_tier_outcome_stats": deployability_rows[:5],
        "recent_symbol_evidence": recent_symbol_evidence[:4],
        "evidence_notes": evidence_notes,
        "weak_evidence_note": (
            "Evidence is partial because matured replay coverage or historical overlays are still thin."
            if str(calibration.get("status") or historical.get("status") or "").lower()
            not in {"available", "supportive", "strong"}
            else None
        ),
    }


def _summary_card(
    axiom: AxiomArtifact,
    report_context: Dict[str, Any],
    historical_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    explanation = dict(axiom.explanation or {})
    strongest = dict(explanation.get("strongest_engine") or {})
    weakest = dict(explanation.get("weakest_engine") or {})
    evidence_backed = dict(axiom.evidence_backed_deployability or {})
    return {
        "symbol": axiom.symbol,
        "as_of": axiom.as_of,
        "regime_label": axiom.regime_label,
        "trade_family": axiom.trade_family,
        "deployability_tier": evidence_backed.get("deployability_tier")
        or axiom.deployability_tier,
        "size_band": evidence_backed.get("size_band")
        or report_context.get("axiom_final_size_band")
        or report_context.get("axiom_size_band_recommendation")
        or "none",
        "deployable_alpha_utility": axiom.deployable_alpha_utility,
        "validated_edge": axiom.validated_edge,
        "strongest_engine": strongest,
        "weakest_engine": weakest,
        "top_positive_drivers": _text_list(
            [item.get("detail") or item.get("label") for item in explanation.get("top_positive_drivers") or []]
        ),
        "top_negative_drivers": _text_list(
            [item.get("detail") or item.get("label") for item in explanation.get("top_negative_drivers") or []]
        ),
        "monitoring_triggers": _text_list(
            explanation.get("monitoring_triggers") or report_context.get("monitoring_triggers") or []
        ),
        "evidence_status": historical_evidence.get("status") or report_context.get("axiom_calibration_status") or "partial",
        "summary": evidence_backed.get("evidence_summary")
        or explanation.get("summary")
        or report_context.get("axiom_summary"),
    }


def _one_pager(
    axiom: AxiomArtifact,
    report_context: Dict[str, Any],
    workspace_profile: AxiomWorkspaceProfile,
    historical_evidence: Dict[str, Any],
    lineage: Dict[str, Any],
) -> Dict[str, Any]:
    explanation = dict(axiom.explanation or {})
    evidence_backed = dict(axiom.evidence_backed_deployability or {})
    return {
        "executive_summary": report_context.get("overall_analysis") or explanation.get("summary"),
        "axiom_scorecard": {
            "gross_opportunity": axiom.gross_opportunity,
            "friction_burden": axiom.friction_burden,
            "validated_edge": axiom.validated_edge,
            "deployable_alpha_utility": axiom.deployable_alpha_utility,
        },
        "regime_trade_family": {
            "regime_label": axiom.regime_label,
            "trade_family": axiom.trade_family,
            "rationale": explanation.get("regime_rationale"),
        },
        "why_this_opportunity_exists": _text_list(
            [
                explanation.get("gross_opportunity_reason"),
                report_context.get("fundamental_analysis"),
                report_context.get("strategy_view"),
            ]
        ),
        "main_risks_fragility": _text_list(
            [
                explanation.get("fragility_reason"),
                report_context.get("risk_quality_analysis"),
                report_context.get("risks_weaknesses_invalidators"),
            ]
        ),
        "liquidity_execution_reality": _text_list(
            [
                (axiom.engine_scores.get("liquidity_convexity") or {}).summary
                if axiom.engine_scores.get("liquidity_convexity")
                else None,
                report_context.get("execution_quality_analysis"),
                report_context.get("deployment_permission_analysis"),
            ]
        ),
        "historical_evidence_summary": historical_evidence,
        "deployability_size_guidance": {
            "deployability_tier": evidence_backed.get("deployability_tier")
            or report_context.get("axiom_evidence_backed_deployability_tier")
            or axiom.deployability_tier,
            "size_band": evidence_backed.get("size_band")
            or report_context.get("axiom_final_size_band")
            or report_context.get("axiom_size_band_recommendation")
            or "none",
            "rationale": evidence_backed.get("evidence_summary")
            or explanation.get("deployability_rationale"),
        },
        "monitoring_invalidation_triggers": {
            "monitoring_triggers": _text_list(explanation.get("monitoring_triggers") or []),
            "invalidation_flags": _text_list(axiom.invalidation_flags),
            "weakest_evidence_areas": lineage.get("weakest_evidence_areas") or [],
        },
        "audience_emphasis": workspace_profile.emphasis_domains,
    }


def _ic_memo(
    axiom: AxiomArtifact,
    report_context: Dict[str, Any],
    workspace_profile: AxiomWorkspaceProfile,
    historical_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    explanation = dict(axiom.explanation or {})
    evidence_backed = dict(axiom.evidence_backed_deployability or {})
    state_pricing = axiom.engine_scores.get("state_pricing")
    fragility = axiom.engine_scores.get("critical_fragility")
    liquidity = axiom.engine_scores.get("liquidity_convexity")
    research = axiom.engine_scores.get("research_integrity")
    return {
        "thesis": report_context.get("overall_analysis") or explanation.get("summary"),
        "market_pricing_view": state_pricing.summary if state_pricing else "State-pricing context is unavailable.",
        "axiom_mispricing_or_compensation_view": _text_list(
            [
                explanation.get("gross_opportunity_reason"),
                report_context.get("signal_summary"),
                report_context.get("fundamental_analysis"),
            ]
        ),
        "evidence_quality_and_calibration": {
            "research_integrity": research.summary if research else None,
            "calibration_status": report_context.get("axiom_calibration_status"),
            "historical_evidence": historical_evidence.get("evidence_notes") or [],
        },
        "fragility_path_risk_analysis": fragility.summary if fragility else None,
        "liquidity_implementation_notes": liquidity.summary if liquidity else None,
        "portfolio_fit": {
            "portfolio_fit_label": report_context.get("axiom_portfolio_fit_label"),
            "portfolio_rank_score": report_context.get("axiom_portfolio_rank_score"),
            "portfolio_summary": report_context.get("axiom_portfolio_governance_summary"),
        },
        "recommended_action": {
            "tier": evidence_backed.get("deployability_tier")
            or report_context.get("axiom_evidence_backed_deployability_tier")
            or axiom.deployability_tier,
            "size_band": evidence_backed.get("size_band")
            or report_context.get("axiom_final_size_band")
            or report_context.get("axiom_size_band_recommendation"),
            "rationale": evidence_backed.get("evidence_summary")
            or explanation.get("deployability_rationale"),
        },
        "escalation_or_downgrade_conditions": _text_list(
            [
                *(report_context.get("monitoring_triggers") or []),
                *(report_context.get("deployment_blockers") or []),
                *(report_context.get("deterioration_triggers") or []),
            ],
            limit=6,
        ),
        "audience_type": workspace_profile.audience_type,
    }


def _risk_deployability_memo(
    axiom: AxiomArtifact,
    report_context: Dict[str, Any],
    historical_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    fragility = axiom.engine_scores.get("critical_fragility")
    liquidity = axiom.engine_scores.get("liquidity_convexity")
    research = axiom.engine_scores.get("research_integrity")
    evidence_backed = dict(axiom.evidence_backed_deployability or {})
    return {
        "fragility_engine": fragility.model_dump(mode="python") if fragility else {},
        "liquidity_convexity_engine": liquidity.model_dump(mode="python") if liquidity else {},
        "research_integrity_engine": research.model_dump(mode="python") if research else {},
        "evidence_backed_deployability": evidence_backed,
        "size_band": evidence_backed.get("size_band")
        or report_context.get("axiom_final_size_band")
        or report_context.get("axiom_size_band_recommendation"),
        "downgrade_triggers": _text_list(
            [
                *(report_context.get("deployment_blockers") or []),
                *(report_context.get("deterioration_triggers") or []),
                *(axiom.explanation.get("monitoring_triggers") or []),
            ],
            limit=6,
        ),
        "scenario_sensitivities": _text_list(
            [
                report_context.get("risk_quality_analysis"),
                report_context.get("deployment_permission_analysis"),
                report_context.get("execution_quality_analysis"),
                *(historical_evidence.get("recent_symbol_evidence") or []),
            ],
            limit=5,
        ),
        "memo_summary": evidence_backed.get("evidence_summary")
        or axiom.explanation.get("deployability_rationale"),
    }


def build_axiom_institutional_report_pack(
    *,
    axiom_artifact: Dict[str, Any] | AxiomArtifact,
    report_context: Dict[str, Any],
    workspace_profile: Dict[str, Any] | AxiomWorkspaceProfile,
    lineage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    axiom = (
        axiom_artifact
        if isinstance(axiom_artifact, AxiomArtifact)
        else AxiomArtifact.model_validate(axiom_artifact)
    )
    profile = (
        workspace_profile
        if isinstance(workspace_profile, AxiomWorkspaceProfile)
        else AxiomWorkspaceProfile.model_validate(workspace_profile)
    )
    lineage = lineage or {}
    historical_evidence = _historical_evidence_payload(axiom, report_context)
    summary_card = _summary_card(axiom, report_context, historical_evidence)
    one_pager = _one_pager(axiom, report_context, profile, historical_evidence, lineage)
    ic_memo = _ic_memo(axiom, report_context, profile, historical_evidence)
    risk_memo = _risk_deployability_memo(axiom, report_context, historical_evidence)
    report_pack = AxiomInstitutionalReportPack(
        reporting_version=AXIOM_REPORTING_VERSION,
        framework_version=axiom.framework_version,
        symbol=axiom.symbol,
        as_of=axiom.as_of,
        workspace_profile=profile,
        summary_card=summary_card,
        institutional_one_pager=one_pager,
        ic_memo=ic_memo,
        risk_deployability_memo=risk_memo,
        historical_evidence_summary=historical_evidence,
        lineage_summary={
            "lineage_summary": lineage.get("lineage_summary"),
            "weakest_evidence_areas": lineage.get("weakest_evidence_areas") or [],
        },
    )
    payload = report_pack.model_dump(mode="python")
    payload["memo_serialization"] = {
        "summary_card_title": f"{axiom.symbol} AXIOM Summary Card",
        "one_pager_title": f"{axiom.symbol} Institutional One-Pager",
        "ic_memo_title": f"{axiom.symbol} IC Memo",
        "risk_memo_title": f"{axiom.symbol} Risk & Deployability Memo",
    }
    return payload
