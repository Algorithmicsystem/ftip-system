from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import ExportSection


PACK_TYPE_TITLES = {
    "axiom_summary_pack": "AXIOM Summary Pack",
    "institutional_one_pager_pack": "Institutional One-Pager Pack",
    "ic_memo_pack": "IC Memo Pack",
    "risk_deployability_pack": "Risk & Deployability Pack",
    "historical_evidence_pack": "Historical Evidence Pack",
    "dossier_pack": "Full Dossier Pack",
}


def supported_pack_types() -> List[str]:
    return list(PACK_TYPE_TITLES.keys())


def _section(section_key: str, title: str, content: Any, payload: Optional[Dict[str, Any]] = None) -> ExportSection:
    return ExportSection(
        section_key=section_key,
        title=title,
        content=str(content or "No current content is available."),
        payload=sanitize_payload(payload or {}),
        status="available" if content not in (None, "", [], {}) else "partial",
    )


def build_export_sections(
    *,
    pack_type: str,
    report: Dict[str, Any],
    dossier: Dict[str, Any],
) -> List[ExportSection]:
    dossier_sections = list(dossier.get("sections") or [])
    historical = report.get("axiom_historical_evidence_report") or {}
    risk_memo = report.get("axiom_risk_deployability_memo") or {}
    lineage = report.get("axiom_lineage") or {}
    if pack_type == "axiom_summary_pack":
        return [
            _section("summary_card", "AXIOM Summary Card", report.get("axiom_summary_card_text"), report.get("axiom_summary_card") or {}),
            _section("drivers", "Drivers", report.get("axiom_summary"), report.get("axiom_explanation") or {}),
            _section("monitoring", "Monitoring", report.get("axiom_risk_deployability_memo_summary") or report.get("platform_monitoring_summary"), risk_memo),
        ]
    if pack_type == "institutional_one_pager_pack":
        return [
            _section("executive_summary", "Executive Summary", report.get("overall_analysis")),
            _section("scorecard", "AXIOM Scorecard", report.get("axiom_summary_card_text"), report.get("axiom_summary_card") or {}),
            _section("regime_trade_family", "Regime and Trade Family", report.get("axiom_regime_trade_family_summary") or report.get("strategy_view")),
            _section("opportunity", "Why This Opportunity Exists", report.get("axiom_summary")),
            _section("fragility", "Main Risks / Fragility", report.get("risk_quality_analysis"), risk_memo.get("fragility_engine") or {}),
            _section("liquidity", "Liquidity / Execution Reality", report.get("execution_quality_analysis"), risk_memo.get("liquidity_convexity_engine") or {}),
            _section("historical", "Historical Evidence Summary", report.get("axiom_historical_evidence_summary_text"), historical),
            _section("deployability", "Deployability and Size Guidance", report.get("axiom_risk_deployability_memo_summary"), risk_memo),
            _section("monitoring", "Monitoring / Invalidation Triggers", report.get("platform_monitoring_summary"), dossier.get("monitoring_state") or {}),
        ]
    if pack_type == "ic_memo_pack":
        return [
            _section("thesis", "Thesis", (report.get("axiom_ic_memo") or {}).get("thesis") or report.get("overall_analysis")),
            _section("market_pricing", "What the Market Is Pricing", (report.get("axiom_ic_memo") or {}).get("market_pricing_view") or report.get("strategy_view")),
            _section("mispricing", "What AXIOM Believes Is Mispriced or Compensated", report.get("axiom_summary")),
            _section("evidence", "Evidence Quality and Calibration Summary", report.get("axiom_calibration_summary_text"), report.get("axiom_calibration_summary") or {}),
            _section("fragility", "Fragility / Path-Risk Analysis", report.get("risk_quality_analysis"), risk_memo.get("fragility_engine") or {}),
            _section("liquidity", "Liquidity / Implementation Notes", report.get("execution_quality_analysis"), risk_memo.get("liquidity_convexity_engine") or {}),
            _section("portfolio", "Portfolio Fit", report.get("axiom_portfolio_governance_summary"), report.get("axiom_portfolio_governance") or {}),
            _section("action", "Recommended Action", report.get("deployment_permission_analysis"), {"tier": report.get("axiom_evidence_backed_deployability_tier")}),
            _section("escalation", "Conditions Required for Escalation or Downgrade", report.get("platform_monitoring_summary"), dossier.get("monitoring_state") or {}),
        ]
    if pack_type == "risk_deployability_pack":
        return [
            _section("fragility_engine", "Critical Fragility", report.get("risk_quality_analysis"), risk_memo.get("fragility_engine") or {}),
            _section("liquidity_engine", "Liquidity and Convexity", report.get("execution_quality_analysis"), risk_memo.get("liquidity_convexity_engine") or {}),
            _section("research_integrity", "Research Integrity", report.get("axiom_summary"), (report.get("axiom") or {}).get("engine_scores", {}).get("research_integrity") or {}),
            _section("deployability", "Evidence-Backed Deployability", report.get("axiom_risk_deployability_memo_summary"), risk_memo),
            _section("monitoring", "Downgrade / Scenario Triggers", report.get("platform_monitoring_summary"), dossier.get("monitoring_state") or {}),
        ]
    if pack_type == "historical_evidence_pack":
        return [
            _section("historical_summary", "Historical Evidence Summary", report.get("axiom_historical_evidence_summary_text"), historical),
            _section("calibration", "Calibration Summary", report.get("axiom_calibration_summary_text"), report.get("axiom_calibration_summary") or {}),
            _section("symbol_evidence", "Recent Symbol-Specific Evidence", report.get("axiom_historical_evidence_summary_text"), historical.get("recent_symbol_evidence") or {}),
        ]
    sections: List[ExportSection] = [
        _section("executive_summary", "Executive Summary", report.get("overall_analysis"), dossier.get("current_summary") or {}),
    ]
    for item in dossier_sections:
        sections.append(
            _section(
                str(item.get("section_key") or "section"),
                str(item.get("title") or "Section"),
                item.get("summary"),
                item.get("payload") or {},
            )
        )
    sections.append(_section("lineage", "Lineage Summary", report.get("axiom_lineage_summary"), lineage))
    return sections
