from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.axiom.contracts import AxiomWorkspaceProfile


AXIOM_WORKSPACE_PROFILE_VERSION = "axiom50_phase4_workspace_v1"

_DEFAULT_REPORT_PROFILE_BY_AUDIENCE = {
    "hedge_fund": "trading_focused",
    "private_equity": "diligence_focused",
    "investment_bank": "ic_memo",
    "family_office": "portfolio_review",
    "research_team": "portfolio_review",
    "general": "trading_focused",
}

_WORKFLOW_PROFILE_BY_AUDIENCE = {
    "hedge_fund": "active_trading_desk",
    "private_equity": "investment_committee",
    "investment_bank": "capital_markets_committee",
    "family_office": "capital_preservation_committee",
    "research_team": "research_review",
    "general": "default",
}

_EMPHASIS_BY_AUDIENCE = {
    "hedge_fund": [
        "deployability",
        "timing",
        "fragility",
        "liquidity",
        "convexity",
        "historical_evidence",
    ],
    "private_equity": [
        "fundamental_reality",
        "durability",
        "balance_sheet",
        "downside_protection",
        "scenario_path",
        "research_integrity",
    ],
    "investment_bank": [
        "valuation_narrative",
        "market_timing",
        "positioning_context",
        "risk_committee_language",
        "source_lineage",
    ],
    "family_office": [
        "capital_preservation",
        "durability",
        "deployability",
        "drawdown_control",
        "evidence_quality",
    ],
    "research_team": [
        "engine_transparency",
        "historical_evidence",
        "calibration",
        "lineage",
        "monitoring_triggers",
    ],
    "general": [
        "executive_summary",
        "evidence_quality",
        "risks",
        "deployability",
    ],
}

_SECTION_ORDER_BY_PROFILE = {
    "trading_focused": [
        "summary_card",
        "institutional_one_pager",
        "risk_deployability_memo",
        "historical_evidence_summary",
    ],
    "ic_memo": [
        "ic_memo",
        "institutional_one_pager",
        "historical_evidence_summary",
        "risk_deployability_memo",
    ],
    "diligence_focused": [
        "institutional_one_pager",
        "ic_memo",
        "risk_deployability_memo",
        "historical_evidence_summary",
    ],
    "portfolio_review": [
        "summary_card",
        "risk_deployability_memo",
        "historical_evidence_summary",
        "institutional_one_pager",
    ],
    "risk_committee": [
        "risk_deployability_memo",
        "historical_evidence_summary",
        "summary_card",
        "ic_memo",
    ],
}


def build_axiom_workspace_profile(
    request_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    request_context = request_context or {}
    audience_type = str(request_context.get("audience_type") or "general").strip().lower()
    if audience_type not in _DEFAULT_REPORT_PROFILE_BY_AUDIENCE:
        audience_type = "general"
    report_profile = str(
        request_context.get("report_profile")
        or _DEFAULT_REPORT_PROFILE_BY_AUDIENCE[audience_type]
    ).strip().lower()
    if report_profile not in _SECTION_ORDER_BY_PROFILE:
        report_profile = _DEFAULT_REPORT_PROFILE_BY_AUDIENCE[audience_type]

    emphasis = list(_EMPHASIS_BY_AUDIENCE.get(audience_type, _EMPHASIS_BY_AUDIENCE["general"]))
    sections = list(_SECTION_ORDER_BY_PROFILE.get(report_profile, _SECTION_ORDER_BY_PROFILE["trading_focused"]))
    notes: List[str] = [
        f"Profile is adapted for {audience_type.replace('_', ' ')} users.",
        f"Report profile emphasizes {report_profile.replace('_', ' ')} communication.",
    ]
    if audience_type == "hedge_fund":
        notes.append("AXIOM explanations prioritize timing, deployability, and path-risk evidence.")
    elif audience_type == "private_equity":
        notes.append("AXIOM explanations prioritize durability, downside protection, and balance-sheet quality.")
    elif audience_type == "investment_bank":
        notes.append("AXIOM explanations prioritize market-pricing narrative, positioning, and committee framing.")
    elif audience_type == "family_office":
        notes.append("AXIOM explanations prioritize capital preservation, drawdown control, and evidence discipline.")
    elif audience_type == "research_team":
        notes.append("AXIOM explanations prioritize calibration, lineage, and engine transparency.")

    profile = AxiomWorkspaceProfile(
        workspace_id=str(request_context.get("workspace_id") or "default"),
        workspace_name=str(request_context.get("workspace_name") or "FTIP Default Workspace"),
        workspace_status="profiled",
        audience_type=audience_type,
        report_profile=report_profile,
        workflow_profile=_WORKFLOW_PROFILE_BY_AUDIENCE.get(audience_type, "default"),
        emphasis_domains=emphasis,
        preferred_sections=sections,
        profile_notes=notes,
    )
    payload = profile.model_dump(mode="python")
    payload["workspace_profile_version"] = AXIOM_WORKSPACE_PROFILE_VERSION
    return payload
