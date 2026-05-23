from __future__ import annotations

from typing import Dict, List

from api.platform.contracts import PlatformProfile


_PROFILES: Dict[str, PlatformProfile] = {
    "hf_core": PlatformProfile(
        profile_id="hf_core",
        audience_type="hedge_fund",
        default_workflow_template="hedge_fund_research",
        default_report_profile="trading_focused",
        default_workspace_name_pattern="HF {organization_name} Workspace",
        default_memo_emphasis=["deployability", "fragility", "liquidity", "timing"],
        default_dashboard_emphasis=["high_dau", "pending_approvals", "fragility", "exports"],
        default_export_pack_emphasis=[
            "axiom_summary_pack",
            "ic_memo_pack",
            "risk_deployability_pack",
            "dossier_pack",
        ],
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
        pilot_bootstrap_defaults={
            "demo_bundle_id": "hedge_fund_demo_bundle",
            "seed_demo_bundle": True,
            "include_exports": True,
            "include_integrations": True,
        },
    ),
    "pe_core": PlatformProfile(
        profile_id="pe_core",
        audience_type="private_equity",
        default_workflow_template="private_equity_diligence",
        default_report_profile="diligence_focused",
        default_workspace_name_pattern="PE {organization_name} Diligence Workspace",
        default_memo_emphasis=["durability", "valuation", "downside", "evidence"],
        default_dashboard_emphasis=["dossier_quality", "committee_state", "exports"],
        default_export_pack_emphasis=[
            "institutional_one_pager_pack",
            "ic_memo_pack",
            "dossier_pack",
        ],
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
        pilot_bootstrap_defaults={
            "demo_bundle_id": "private_equity_demo_bundle",
            "seed_demo_bundle": True,
            "include_exports": True,
            "include_integrations": False,
        },
    ),
    "ib_core": PlatformProfile(
        profile_id="ib_core",
        audience_type="investment_bank",
        default_workflow_template="investment_bank_advisory",
        default_report_profile="ic_memo",
        default_workspace_name_pattern="IB {organization_name} Advisory Workspace",
        default_memo_emphasis=["market_context", "valuation", "risk_framing"],
        default_dashboard_emphasis=["client_memos", "exports", "monitoring"],
        default_export_pack_emphasis=[
            "institutional_one_pager_pack",
            "ic_memo_pack",
            "axiom_summary_pack",
        ],
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
        pilot_bootstrap_defaults={
            "demo_bundle_id": "investment_bank_demo_bundle",
            "seed_demo_bundle": True,
            "include_exports": True,
            "include_integrations": False,
        },
    ),
    "fo_core": PlatformProfile(
        profile_id="fo_core",
        audience_type="family_office",
        default_workflow_template="family_office_review",
        default_report_profile="risk_committee",
        default_workspace_name_pattern="FO {organization_name} Review Workspace",
        default_memo_emphasis=["durability", "capital_preservation", "monitoring"],
        default_dashboard_emphasis=["capital_preservation", "committee_state", "warnings"],
        default_export_pack_emphasis=[
            "axiom_summary_pack",
            "risk_deployability_pack",
            "dossier_pack",
        ],
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
        pilot_bootstrap_defaults={
            "demo_bundle_id": "family_office_demo_bundle",
            "seed_demo_bundle": True,
            "include_exports": True,
            "include_integrations": False,
        },
    ),
    "research_core": PlatformProfile(
        profile_id="research_core",
        audience_type="research_team",
        default_workflow_template="research_watchlist",
        default_report_profile="trading_focused",
        default_workspace_name_pattern="Research {organization_name} Workspace",
        default_memo_emphasis=["research", "evidence", "watchlist"],
        default_dashboard_emphasis=["watchlist", "evidence", "workflow"],
        default_export_pack_emphasis=[
            "axiom_summary_pack",
            "institutional_one_pager_pack",
            "dossier_pack",
        ],
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
        pilot_bootstrap_defaults={
            "demo_bundle_id": "research_team_demo_bundle",
            "seed_demo_bundle": True,
            "include_exports": True,
            "include_integrations": False,
        },
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
