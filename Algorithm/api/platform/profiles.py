from __future__ import annotations

from typing import Dict, List

from api.platform.contracts import PlatformProfile


_PROFILES: Dict[str, PlatformProfile] = {
    "hf_core": PlatformProfile(
        profile_id="hf_core",
        audience_type="hedge_fund",
        default_workflow_template="hedge_fund_research",
        default_report_profile="trading_focused",
        default_memo_emphasis=["deployability", "fragility", "liquidity", "timing"],
        preferred_axiom_sections=[
            "axiom_summary_card",
            "axiom_historical_evidence_summary_text",
            "axiom_risk_deployability_memo_summary",
            "axiom_lineage_summary",
        ],
        preferred_dossier_sections=[
            "executive_summary",
            "portfolio_fit",
            "decision_status",
            "monitoring_triggers",
        ],
    ),
    "pe_core": PlatformProfile(
        profile_id="pe_core",
        audience_type="private_equity",
        default_workflow_template="private_equity_diligence",
        default_report_profile="diligence_focused",
        default_memo_emphasis=["durability", "valuation", "downside", "evidence"],
        preferred_axiom_sections=[
            "axiom_summary_card",
            "axiom_ic_memo_summary",
            "axiom_lineage_summary",
        ],
        preferred_dossier_sections=[
            "executive_summary",
            "axiom_scorecard",
            "historical_evidence",
            "lineage_summary",
        ],
    ),
    "ib_core": PlatformProfile(
        profile_id="ib_core",
        audience_type="investment_bank",
        default_workflow_template="investment_bank_advisory",
        default_report_profile="ic_memo",
        default_memo_emphasis=["market_context", "valuation", "risk_framing"],
        preferred_axiom_sections=[
            "axiom_summary_card",
            "axiom_ic_memo_summary",
            "axiom_historical_evidence_summary_text",
        ],
        preferred_dossier_sections=[
            "executive_summary",
            "regime_trade_setup",
            "decision_status",
            "lineage_summary",
        ],
    ),
    "fo_core": PlatformProfile(
        profile_id="fo_core",
        audience_type="family_office",
        default_workflow_template="family_office_review",
        default_report_profile="risk_committee",
        default_memo_emphasis=["durability", "capital_preservation", "monitoring"],
        preferred_axiom_sections=[
            "axiom_summary_card",
            "axiom_risk_deployability_memo_summary",
            "axiom_historical_evidence_summary_text",
        ],
        preferred_dossier_sections=[
            "executive_summary",
            "fragility_risk",
            "decision_status",
            "monitoring_triggers",
        ],
    ),
    "research_core": PlatformProfile(
        profile_id="research_core",
        audience_type="research_team",
        default_workflow_template="research_watchlist",
        default_report_profile="trading_focused",
        default_memo_emphasis=["research", "evidence", "watchlist"],
        preferred_axiom_sections=[
            "axiom_summary_card",
            "axiom_lineage_summary",
            "axiom_historical_evidence_summary_text",
        ],
        preferred_dossier_sections=[
            "executive_summary",
            "axiom_scorecard",
            "historical_evidence",
            "monitoring_triggers",
        ],
    ),
}

_AUDIENCE_TO_PROFILE = {
    "hedge_fund": "hf_core",
    "private_equity": "pe_core",
    "investment_bank": "ib_core",
    "family_office": "fo_core",
    "research_team": "research_core",
    "general": "research_core",
}


def list_platform_profiles() -> List[PlatformProfile]:
    return [profile.model_copy(deep=True) for profile in _PROFILES.values()]


def get_platform_profile(profile_id: str | None) -> PlatformProfile:
    if profile_id:
        profile = _PROFILES.get(str(profile_id))
        if profile is not None:
            return profile.model_copy(deep=True)
    return _PROFILES["research_core"].model_copy(deep=True)


def get_platform_profile_for_audience(audience_type: str | None) -> PlatformProfile:
    profile_id = _AUDIENCE_TO_PROFILE.get(str(audience_type or "general"), "research_core")
    return get_platform_profile(profile_id)

