from __future__ import annotations

from typing import Any, Dict, List

from api.assistant.reports import sanitize_payload
from api.platform.contracts import DemoSeedBundle


def _seed_blueprint(
    *,
    symbol: str,
    display_name: str,
    workflow_title: str,
    dossier_title: str,
    stage: str,
    decision_status: str,
    recommendation_state: str,
    regime_label: str,
    trade_family: str,
    deployability_tier: str,
    size_band: str,
    evidence_status: str,
    dau: float,
    validated_edge: float,
    review_comments: List[Dict[str, Any]],
    assignments: List[Dict[str, Any]],
    pack_types: List[str],
) -> Dict[str, Any]:
    return sanitize_payload(
        {
            "symbol": symbol,
            "display_name": display_name,
            "workflow_title": workflow_title,
            "dossier_title": dossier_title,
            "stage": stage,
            "decision_status": decision_status,
            "recommendation_state": recommendation_state,
            "regime_label": regime_label,
            "trade_family": trade_family,
            "deployability_tier": deployability_tier,
            "size_band": size_band,
            "evidence_status": evidence_status,
            "deployable_alpha_utility": dau,
            "validated_edge": validated_edge,
            "review_comments": review_comments,
            "assignments": assignments,
            "pack_types": pack_types,
        }
    )


_BUNDLES: Dict[str, DemoSeedBundle] = {
    "hedge_fund_demo_bundle": DemoSeedBundle(
        bundle_id="hedge_fund_demo_bundle",
        platform_profile="hf_core",
        audience_type="hedge_fund",
        workflow_template_id="hedge_fund_research",
        title="Hedge Fund Demo Bundle",
        description="Seeds a trading-research pilot with committee visibility, paper-only recommendation state, and export history.",
        seeded_entities=[
            _seed_blueprint(
                symbol="NVDA",
                display_name="NVIDIA",
                workflow_title="NVDA Hedge Fund Research",
                dossier_title="NVDA Institutional Dossier",
                stage="decision",
                decision_status="approved_with_conditions",
                recommendation_state="approved_paper",
                regime_label="fundamental_convergence",
                trade_family="convergence",
                deployability_tier="paper_trade_only",
                size_band="small",
                evidence_status="limited",
                dau=62.75,
                validated_edge=49.64,
                review_comments=[
                    {
                        "comment_type": "liquidity_concern",
                        "severity": "material",
                        "body": "Execution depth remains thin for live sizing.",
                    },
                    {
                        "comment_type": "monitoring_note",
                        "severity": "watch",
                        "body": "Watch post-event fragility behavior before escalation.",
                    },
                ],
                assignments=[
                    {"slot_type": "owner", "assignee_placeholder": "pilot-owner"},
                    {
                        "slot_type": "primary_reviewer",
                        "assignee_placeholder": "pilot-reviewer",
                    },
                    {
                        "slot_type": "committee_reviewer",
                        "assignee_placeholder": "pilot-committee",
                    },
                ],
                pack_types=[
                    "axiom_summary_pack",
                    "ic_memo_pack",
                    "risk_deployability_pack",
                    "dossier_pack",
                ],
            )
        ],
        default_export_packs=[
            "axiom_summary_pack",
            "ic_memo_pack",
            "risk_deployability_pack",
            "dossier_pack",
        ],
        default_dashboard_emphasis=[
            "pending_approvals",
            "high_dau",
            "fragility",
            "stored_exports",
        ],
        default_dossier_sections=[
            "executive_summary",
            "portfolio_fit",
            "decision_status",
            "monitoring_triggers",
        ],
        walkthrough_hints=[
            "Open the committee decision card to review the paper-only approval conditions.",
            "Inspect the stored IC memo and risk/deployability pack for a durable export trail.",
            "Use the review panel to see why the recommendation is capped at paper-only.",
        ],
        readiness_narrative="Trading research workflow is demo-ready, but live escalation remains intentionally blocked by liquidity concerns.",
        includes_comments=True,
        includes_committee_state=True,
        includes_exports=True,
        includes_integrations=True,
        metadata={"demo_seeded": True},
    ),
    "private_equity_demo_bundle": DemoSeedBundle(
        bundle_id="private_equity_demo_bundle",
        platform_profile="pe_core",
        audience_type="private_equity",
        workflow_template_id="private_equity_diligence",
        title="Private Equity Demo Bundle",
        description="Seeds a diligence workflow with durability framing, evidence gaps, and institutional memoing.",
        seeded_entities=[
            _seed_blueprint(
                symbol="MSFT",
                display_name="Microsoft",
                workflow_title="MSFT Diligence Workflow",
                dossier_title="MSFT Diligence Dossier",
                stage="committee_view",
                decision_status="watch",
                recommendation_state="watch_only",
                regime_label="compensation_capture",
                trade_family="compensation",
                deployability_tier="monitor_only",
                size_band="small",
                evidence_status="supportive",
                dau=71.4,
                validated_edge=57.8,
                review_comments=[
                    {
                        "comment_type": "valuation_concern",
                        "severity": "watch",
                        "body": "Valuation still needs a clearer downside buffer for pursue status.",
                    }
                ],
                assignments=[
                    {"slot_type": "owner", "assignee_placeholder": "deal-lead"},
                    {
                        "slot_type": "risk_reviewer",
                        "assignee_placeholder": "downside-review",
                    },
                ],
                pack_types=[
                    "institutional_one_pager_pack",
                    "ic_memo_pack",
                    "dossier_pack",
                ],
            )
        ],
        default_export_packs=[
            "institutional_one_pager_pack",
            "ic_memo_pack",
            "dossier_pack",
        ],
        default_dashboard_emphasis=["dossier_quality", "evidence", "committee_state"],
        default_dossier_sections=[
            "executive_summary",
            "axiom_scorecard",
            "historical_evidence",
            "lineage_summary",
        ],
        walkthrough_hints=[
            "Use the one-pager to review durability, evidence quality, and downside framing.",
            "The dossier remains watch-only until valuation and downside conditions improve.",
        ],
        readiness_narrative="Diligence workflow is provisioned for review, with explicit valuation concerns still open.",
        includes_comments=True,
        includes_committee_state=True,
        includes_exports=True,
        metadata={"demo_seeded": True},
    ),
    "investment_bank_demo_bundle": DemoSeedBundle(
        bundle_id="investment_bank_demo_bundle",
        platform_profile="ib_core",
        audience_type="investment_bank",
        workflow_template_id="investment_bank_advisory",
        title="Investment Bank Demo Bundle",
        description="Seeds an advisory memo workflow with market-context framing and client-facing exports.",
        seeded_entities=[
            _seed_blueprint(
                symbol="AAPL",
                display_name="Apple",
                workflow_title="AAPL Advisory Workflow",
                dossier_title="AAPL Advisory Dossier",
                stage="client_memo",
                decision_status="approved",
                recommendation_state="approved_paper",
                regime_label="behavioral_continuation",
                trade_family="transmission",
                deployability_tier="paper_trade_only",
                size_band="small",
                evidence_status="supportive",
                dau=66.8,
                validated_edge=52.3,
                review_comments=[
                    {
                        "comment_type": "committee_note",
                        "severity": "info",
                        "body": "Client memo framing is ready, but timing remains best treated as monitored.",
                    }
                ],
                assignments=[
                    {"slot_type": "owner", "assignee_placeholder": "coverage-banker"},
                    {
                        "slot_type": "primary_reviewer",
                        "assignee_placeholder": "memo-reviewer",
                    },
                ],
                pack_types=[
                    "institutional_one_pager_pack",
                    "ic_memo_pack",
                    "axiom_summary_pack",
                ],
            )
        ],
        default_export_packs=[
            "institutional_one_pager_pack",
            "ic_memo_pack",
            "axiom_summary_pack",
        ],
        default_dashboard_emphasis=["client_memos", "exports", "monitoring"],
        default_dossier_sections=[
            "executive_summary",
            "regime_trade_setup",
            "decision_status",
            "lineage_summary",
        ],
        walkthrough_hints=[
            "Start with the institutional one-pager to review the valuation and market-context framing.",
            "Use the recommendation-state panel to explain why the memo is approved for monitored use, not live deployment.",
        ],
        readiness_narrative="Advisory memo workflow is export-ready and demo-ready, with deliberate timing caution preserved.",
        includes_comments=True,
        includes_committee_state=True,
        includes_exports=True,
        metadata={"demo_seeded": True},
    ),
    "family_office_demo_bundle": DemoSeedBundle(
        bundle_id="family_office_demo_bundle",
        platform_profile="fo_core",
        audience_type="family_office",
        workflow_template_id="family_office_review",
        title="Family Office Demo Bundle",
        description="Seeds a capital-preservation workflow with simplified committee state and monitoring visibility.",
        seeded_entities=[
            _seed_blueprint(
                symbol="COST",
                display_name="Costco",
                workflow_title="COST Family Office Review",
                dossier_title="COST Capital Preservation Dossier",
                stage="decision",
                decision_status="approved_with_conditions",
                recommendation_state="watch_only",
                regime_label="recovery_reset",
                trade_family="recovery",
                deployability_tier="monitor_only",
                size_band="small",
                evidence_status="supportive",
                dau=58.6,
                validated_edge=47.9,
                review_comments=[
                    {
                        "comment_type": "risk_concern",
                        "severity": "watch",
                        "body": "Capital preservation framing remains favorable, but entry patience is still required.",
                    }
                ],
                assignments=[
                    {"slot_type": "owner", "assignee_placeholder": "fo-owner"},
                    {
                        "slot_type": "committee_reviewer",
                        "assignee_placeholder": "fo-committee",
                    },
                ],
                pack_types=[
                    "axiom_summary_pack",
                    "risk_deployability_pack",
                    "dossier_pack",
                ],
            )
        ],
        default_export_packs=[
            "axiom_summary_pack",
            "risk_deployability_pack",
            "dossier_pack",
        ],
        default_dashboard_emphasis=["capital_preservation", "warnings", "monitoring"],
        default_dossier_sections=[
            "executive_summary",
            "fragility_risk",
            "decision_status",
            "monitoring_triggers",
        ],
        walkthrough_hints=[
            "Review the monitoring and decision panels to show capital-preservation safeguards.",
            "The risk/deployability memo captures why the recommendation remains watch-only.",
        ],
        readiness_narrative="Family-office workflow is easy to walk through, with a conservative recommendation state and explicit monitoring posture.",
        includes_comments=True,
        includes_committee_state=True,
        includes_exports=True,
        metadata={"demo_seeded": True},
    ),
    "research_team_demo_bundle": DemoSeedBundle(
        bundle_id="research_team_demo_bundle",
        platform_profile="research_core",
        audience_type="research_team",
        workflow_template_id="research_watchlist",
        title="Research Team Demo Bundle",
        description="Seeds an analysis-first watchlist workflow with evidence review, stored exports, and watch-only recommendations.",
        seeded_entities=[
            _seed_blueprint(
                symbol="AMD",
                display_name="AMD",
                workflow_title="AMD Watchlist Workflow",
                dossier_title="AMD Research Watchlist Dossier",
                stage="watchlist",
                decision_status="deferred",
                recommendation_state="under_review",
                regime_label="indeterminate",
                trade_family="none",
                deployability_tier="monitor_only",
                size_band="none",
                evidence_status="limited",
                dau=43.1,
                validated_edge=35.4,
                review_comments=[
                    {
                        "comment_type": "evidence_gap",
                        "severity": "material",
                        "body": "Calibration and export history are seeded, but live evidence remains intentionally weak.",
                    }
                ],
                assignments=[
                    {"slot_type": "owner", "assignee_placeholder": "research-owner"},
                    {
                        "slot_type": "observer",
                        "assignee_placeholder": "research-observer",
                    },
                ],
                pack_types=[
                    "axiom_summary_pack",
                    "institutional_one_pager_pack",
                    "dossier_pack",
                ],
            )
        ],
        default_export_packs=[
            "axiom_summary_pack",
            "institutional_one_pager_pack",
            "dossier_pack",
        ],
        default_dashboard_emphasis=["watchlist", "evidence", "workflow"],
        default_dossier_sections=[
            "executive_summary",
            "axiom_scorecard",
            "historical_evidence",
            "monitoring_triggers",
        ],
        walkthrough_hints=[
            "Use the watchlist dossier to show how AXIOM analysis becomes a reusable research object.",
            "The unresolved evidence-gap comment explains why the recommendation remains under review.",
        ],
        readiness_narrative="Research workflow is ready for demo use, with an intentionally non-promotional evidence posture.",
        includes_comments=True,
        includes_committee_state=True,
        includes_exports=True,
        metadata={"demo_seeded": True},
    ),
}


def list_demo_seed_bundles() -> List[DemoSeedBundle]:
    return [bundle.model_copy(deep=True) for bundle in _BUNDLES.values()]


def get_demo_seed_bundle(bundle_id: str | None) -> DemoSeedBundle:
    bundle = _BUNDLES.get(str(bundle_id or ""))
    if bundle is None:
        return _BUNDLES["research_team_demo_bundle"].model_copy(deep=True)
    return bundle.model_copy(deep=True)


def build_demo_seed_report(
    *,
    symbol: str,
    bundle: DemoSeedBundle,
    blueprint: Dict[str, Any],
) -> Dict[str, Any]:
    deployability_tier = str(
        blueprint.get("deployability_tier")
        or "paper_trade_only"
    )
    regime_label = str(blueprint.get("regime_label") or "indeterminate")
    trade_family = str(blueprint.get("trade_family") or "none")
    size_band = str(blueprint.get("size_band") or "small")
    evidence_status = str(blueprint.get("evidence_status") or "limited")
    deployable_alpha_utility = float(blueprint.get("deployable_alpha_utility") or 55.0)
    validated_edge = float(blueprint.get("validated_edge") or 44.0)
    workflow_title = str(blueprint.get("workflow_title") or f"{symbol} Demo Workflow")
    return sanitize_payload(
        {
            "symbol": symbol,
            "as_of_date": "2026-05-23",
            "overall_analysis": (
                f"{symbol} is a deterministic demo-seeded AXIOM report for the {bundle.title.lower()} "
                f"and should be interpreted as pilot packaging rather than live market advice."
            ),
            "signal_summary": (
                f"{symbol} is staged for a {deployability_tier.replace('_', ' ')} recommendation "
                f"inside the {workflow_title} workflow."
            ),
            "strategy_view": (
                f"Workflow currently emphasizes {regime_label.replace('_', ' ')} reasoning, "
                f"{trade_family.replace('_', ' ')} framing, and controlled institutional packaging."
            ),
            "risk_quality_analysis": (
                "Risk framing is intentionally conservative because this report is demo seeded "
                "and should showcase workflow governance rather than live market certainty."
            ),
            "execution_quality_analysis": (
                "Execution and export quality are provisioned for pilot use, with stored pack history "
                "and explicit readiness checks attached."
            ),
            "portfolio_fit_analysis": (
                "Portfolio fit is summarized for pilot walkthrough purposes and should not be treated "
                "as a live allocator directive."
            ),
            "deployment_permission_analysis": (
                f"Current institutional recommendation state is {deployability_tier.replace('_', ' ')} "
                f"with size band {size_band.replace('_', ' ')}."
            ),
            "deployment_permission": "paper_shadow_only",
            "monitoring_triggers": [
                "Readiness warnings increase materially.",
                "Committee conditions remain unresolved.",
                "Export integrity drifts from the stored version history.",
            ],
            "data_bundle": {"symbol_meta": {"sector": "Technology"}},
            "axiom_summary": (
                f"AXIOM demo packaging sees {symbol} as a {regime_label.replace('_', ' ')} setup "
                f"with {deployability_tier.replace('_', ' ')} deployability."
            ),
            "axiom_summary_card_text": (
                f"Demo-seeded {symbol} dossier with {deployability_tier.replace('_', ' ')} "
                f"deployability, {size_band.replace('_', ' ')} size, and explicit committee context."
            ),
            "axiom_summary_card": {
                "symbol": symbol,
                "regime_label": regime_label,
                "trade_family": trade_family,
                "deployability_tier": deployability_tier,
                "size_band": size_band,
                "evidence_status": evidence_status,
                "deployable_alpha_utility": deployable_alpha_utility,
                "validated_edge": validated_edge,
            },
            "axiom_deployable_alpha_utility": deployable_alpha_utility,
            "axiom_validated_edge": validated_edge,
            "axiom_regime_label": regime_label,
            "axiom_trade_family": trade_family,
            "axiom_deployability_tier": deployability_tier,
            "axiom_evidence_backed_deployability_tier": deployability_tier,
            "axiom_size_band_recommendation": size_band,
            "axiom_final_size_band": size_band,
            "axiom_portfolio_fit_label": "watchlist_only"
            if deployability_tier != "live_candidate"
            else "core_candidate",
            "axiom_historical_evidence_report": {
                "status": "available",
                "history_horizon_label": "21d",
                "recent_symbol_evidence": [
                    f"{symbol} demo evidence is provisioned for reproducible pilot walkthroughs."
                ],
                "demo_seeded": True,
            },
            "axiom_historical_evidence_summary_text": (
                "Historical evidence is provisioned for pilot use and explicitly marked as demo seeded."
            ),
            "axiom_calibration_summary_text": (
                "Calibration context is attached for pilot packaging, but this demo object should not be mistaken for live production evidence."
            ),
            "axiom_portfolio_governance_summary": (
                f"Portfolio governance keeps the dossier in {deployability_tier.replace('_', ' ')} status "
                f"with a {size_band.replace('_', ' ')} size band."
            ),
            "axiom_portfolio_governance": {
                "symbol": symbol,
                "portfolio_rank_score": round(max(deployable_alpha_utility - 6.5, 0.0), 2),
                "portfolio_fit_label": "watchlist_only"
                if deployability_tier != "live_candidate"
                else "core_candidate",
                "final_size_band": size_band,
                "rationale": (
                    "Demo-seeded portfolio guidance reflects workflow packaging rather than live OMS sizing."
                ),
            },
            "axiom_risk_deployability_memo": {
                "fragility_engine": {"score": 35.0 if evidence_status == "limited" else 26.0},
                "liquidity_convexity_engine": {"score": 63.0 if size_band == "small" else 74.0},
                "downgrade_triggers": [
                    "Export readiness falls below partial.",
                    "Committee conditions remain unresolved.",
                ],
                "demo_seeded": True,
            },
            "axiom_lineage_summary": (
                "Lineage is traceable to demo-seeded workflow objects, stored exports, and explicit platform audit context."
            ),
            "axiom_lineage": {
                "lineage_version": "axiom50_phase4_lineage_v1",
                "demo_seeded": True,
                "engine_lineage": {
                    "critical_fragility": {
                        "engine": "critical_fragility",
                        "confidence": 58.0,
                        "blocks": [
                            {
                                "component": "platform_demo_seed_component",
                                "derived_from": [
                                    "platform.demo_seed_bundle",
                                    "platform.review_comments",
                                    "platform.committee_decision",
                                ],
                                "evidence_type": "historical_replay_estimate",
                                "coverage_status": "partial",
                            }
                        ],
                    }
                },
            },
            "axiom_artifact_id": f"demo-axiom-{symbol.lower()}",
            "axiom_history_artifact_id": f"demo-axiom-history-{symbol.lower()}",
            "axiom_calibration_artifact_id": f"demo-axiom-calibration-{symbol.lower()}",
            "axiom_report_pack_artifact_id": f"demo-axiom-pack-{symbol.lower()}",
            "axiom_lineage_artifact_id": f"demo-axiom-lineage-{symbol.lower()}",
            "platform_demo_seeded": True,
            "platform_demo_bundle_id": bundle.bundle_id,
            "platform_demo_bundle_title": bundle.title,
        }
    )

