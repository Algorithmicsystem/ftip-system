from __future__ import annotations

from typing import Dict, List

from api.platform.contracts import WorkflowTemplate


_TEMPLATES: Dict[str, WorkflowTemplate] = {
    "hedge_fund_research": WorkflowTemplate(
        template_id="hedge_fund_research",
        audience_type="hedge_fund",
        title="Hedge Fund Research",
        description="Idea triage, AXIOM evidence review, portfolio-fit screening, and monitored decisioning.",
        default_sections=[
            "executive_summary",
            "axiom_scorecard",
            "regime_trade_setup",
            "fragility_risk",
            "liquidity_execution",
            "historical_evidence",
            "portfolio_fit",
            "decision_status",
            "monitoring_triggers",
            "lineage_summary",
        ],
        stage_sequence=[
            "intake",
            "analysis",
            "evidence_review",
            "portfolio_fit",
            "decision",
            "monitoring",
        ],
        preferred_report_profile="trading_focused",
        preferred_report_pack_emphasis=["deployability", "fragility", "timing", "evidence"],
        expected_axiom_emphasis=[
            "deployable_alpha_utility",
            "research_integrity",
            "critical_fragility",
            "liquidity_convexity",
        ],
        orientation="trading_research",
    ),
    "hedge_fund_portfolio_review": WorkflowTemplate(
        template_id="hedge_fund_portfolio_review",
        audience_type="hedge_fund",
        title="Hedge Fund Portfolio Review",
        description="Allocator-style review of overlapping ideas, sizing, and workflow-ranked exposures.",
        default_sections=[
            "executive_summary",
            "portfolio_fit",
            "fragility_risk",
            "historical_evidence",
            "decision_status",
            "monitoring_triggers",
        ],
        stage_sequence=[
            "intake",
            "portfolio_fit",
            "risk_committee",
            "decision",
            "monitoring",
        ],
        preferred_report_profile="portfolio_review",
        preferred_report_pack_emphasis=["portfolio_fit", "overlap_control", "size_guidance", "evidence"],
        expected_axiom_emphasis=[
            "deployable_alpha_utility",
            "critical_fragility",
            "research_integrity",
            "flow_transmission",
        ],
        orientation="portfolio_review",
    ),
    "private_equity_diligence": WorkflowTemplate(
        template_id="private_equity_diligence",
        audience_type="private_equity",
        title="Private Equity Diligence",
        description="Durability-led fundamental review with downside, valuation, and committee-quality memoing.",
        default_sections=[
            "executive_summary",
            "axiom_scorecard",
            "regime_trade_setup",
            "historical_evidence",
            "decision_status",
            "lineage_summary",
        ],
        stage_sequence=[
            "intake",
            "business_quality",
            "valuation",
            "downside_risk",
            "committee_view",
            "watch_or_pursue",
        ],
        preferred_report_profile="diligence_focused",
        preferred_report_pack_emphasis=["fundamental_reality", "durability", "downside_protection", "evidence"],
        expected_axiom_emphasis=[
            "fundamental_reality",
            "research_integrity",
            "state_pricing",
            "critical_fragility",
        ],
        orientation="diligence",
    ),
    "investment_bank_advisory": WorkflowTemplate(
        template_id="investment_bank_advisory",
        audience_type="investment_bank",
        title="Investment Bank Advisory",
        description="Valuation, market-context, and advisory memo workflow for client-facing framing.",
        default_sections=[
            "executive_summary",
            "regime_trade_setup",
            "historical_evidence",
            "decision_status",
            "lineage_summary",
        ],
        stage_sequence=[
            "intake",
            "market_context",
            "valuation_positioning",
            "risk_factors",
            "client_memo",
            "monitor",
        ],
        preferred_report_profile="ic_memo",
        preferred_report_pack_emphasis=["valuation_narrative", "market_context", "risk_framing", "evidence"],
        expected_axiom_emphasis=[
            "state_pricing",
            "flow_transmission",
            "behavioral_distortion",
            "research_integrity",
        ],
        orientation="advisory",
    ),
    "family_office_review": WorkflowTemplate(
        template_id="family_office_review",
        audience_type="family_office",
        title="Family Office Review",
        description="Capital-preservation and durability workflow with measured opportunity review and monitoring.",
        default_sections=[
            "executive_summary",
            "axiom_scorecard",
            "fragility_risk",
            "historical_evidence",
            "decision_status",
            "monitoring_triggers",
        ],
        stage_sequence=[
            "intake",
            "quality_and_durability",
            "capital_preservation",
            "opportunity_review",
            "decision",
            "monitoring",
        ],
        preferred_report_profile="risk_committee",
        preferred_report_pack_emphasis=["durability", "capital_preservation", "evidence", "monitoring"],
        expected_axiom_emphasis=[
            "fundamental_reality",
            "critical_fragility",
            "research_integrity",
            "liquidity_convexity",
        ],
        orientation="capital_preservation",
    ),
    "research_watchlist": WorkflowTemplate(
        template_id="research_watchlist",
        audience_type="research_team",
        title="Research Watchlist",
        description="General-purpose watchlist workflow for teams building reusable AXIOM-linked coverage dossiers.",
        default_sections=[
            "executive_summary",
            "axiom_scorecard",
            "historical_evidence",
            "portfolio_fit",
            "monitoring_triggers",
            "lineage_summary",
        ],
        stage_sequence=[
            "intake",
            "analysis",
            "evidence_review",
            "watchlist",
            "monitoring",
        ],
        preferred_report_profile="trading_focused",
        preferred_report_pack_emphasis=["summary", "evidence", "monitoring"],
        expected_axiom_emphasis=[
            "deployable_alpha_utility",
            "research_integrity",
            "critical_fragility",
        ],
        orientation="watchlist",
    ),
}


def list_workflow_templates() -> List[WorkflowTemplate]:
    return [template.model_copy(deep=True) for template in _TEMPLATES.values()]


def get_workflow_template(template_id: str) -> WorkflowTemplate:
    template = _TEMPLATES.get(str(template_id or ""))
    if template is None:
        return _TEMPLATES["research_watchlist"].model_copy(deep=True)
    return template.model_copy(deep=True)

