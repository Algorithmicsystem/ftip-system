"""Phase 16.4: Grounded Research Report Generator."""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api import config, db
from api.assistant.phase3.common import clamp
from api.explain.counterfactual import compute_counterfactuals
from api.explain.explanation_system import EvidenceItem, compute_evidence_balance, extract_evidence_items
from api.explain.reasoning_engine import ReasoningChain, build_reasoning_chain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ResearchReport:
    symbol: str
    report_date: dt.date
    analyst_rating: str
    target_conviction: str
    executive_summary: str
    investment_thesis: str
    risk_summary: str
    fundamental_section: Dict[str, Any]
    technical_section: Dict[str, Any]
    risk_section: Dict[str, Any]
    alternative_data_section: Dict[str, Any]
    factor_section: Dict[str, Any]
    invalidation_triggers: List[str]
    reasoning_chain: ReasoningChain
    evidence: Dict[str, Any]
    counterfactuals: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Rating helpers
# ---------------------------------------------------------------------------

def _dau_to_analyst_rating(dau: float, signal_label: str) -> str:
    if dau > 80:
        return "Strong Buy"
    if dau > 65:
        return "Buy"
    if dau > 50:
        return "Hold"
    if dau > 35:
        return "Sell"
    return "Strong Sell"


def _ic_to_conviction(ic_state: str) -> str:
    if ic_state in ("STRONG", "MODERATE"):
        return "High"
    if ic_state == "WEAK":
        return "Moderate"
    return "Low"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_fundamental_section(engine_scores: Dict[str, Any]) -> Dict[str, Any]:
    fundamental = engine_scores.get("fundamental_reality") or {}
    comps = fundamental.get("components") or {}
    return {
        "engine_score": fundamental.get("score"),
        "eis": comps.get("eis_component"),
        "caps": comps.get("caps_component"),
        "valuation_gap": comps.get("valuation_gap"),
        "interpretation": (
            "Earnings quality and shareholder value creation — "
            "grounded in Penman, Schilit, and Rappaport frameworks"
        ),
    }


def _build_technical_section(engine_scores: Dict[str, Any], regime: str) -> Dict[str, Any]:
    flow = engine_scores.get("flow_transmission") or {}
    state_pricing = engine_scores.get("state_pricing") or {}
    return {
        "flow_score": flow.get("score"),
        "state_pricing_score": state_pricing.get("score"),
        "regime": regime,
        "flow_components": flow.get("components"),
        "interpretation": "Price transmission and momentum — Kyle (1985) flow dynamics",
    }


def _build_risk_section(engine_scores: Dict[str, Any]) -> Dict[str, Any]:
    fragility = engine_scores.get("critical_fragility") or {}
    comps = fragility.get("components") or {}
    return {
        "fragility_score": fragility.get("score"),
        "scps": comps.get("scps_component"),
        "mtrs": comps.get("mtrs_component"),
        "bfs": comps.get("bfs_component"),
        "interpretation": (
            "Sornette critical point score + Mandelbrot tail risk + "
            "systematic crash detection"
        ),
    }


def _build_alternative_section(engine_scores: Dict[str, Any]) -> Dict[str, Any]:
    liquidity = engine_scores.get("liquidity_convexity") or {}
    comps = liquidity.get("components") or {}
    research = engine_scores.get("research_integrity") or {}
    return {
        "osms": comps.get("osms_component"),
        "ias": comps.get("ias_component"),
        "research_integrity_score": research.get("score"),
        "interpretation": "Options smart money + institutional accumulation signals",
    }


def _build_factor_section(axiom_payload: Dict[str, Any]) -> Dict[str, Any]:
    alpha_decomp = axiom_payload.get("alpha_decomposition") or {}
    return {
        "primary_driver": alpha_decomp.get("primary_driver"),
        "factor_contributions": alpha_decomp.get("factor_contributions"),
        "systematic_contribution": alpha_decomp.get("systematic_contribution"),
        "idiosyncratic_alpha": alpha_decomp.get("idiosyncratic_alpha"),
        "factor_concentration": alpha_decomp.get("factor_concentration"),
    }


# ---------------------------------------------------------------------------
# Narrative builders (programmatic, no LLM)
# ---------------------------------------------------------------------------

def _build_executive_summary(
    symbol: str,
    analyst_rating: str,
    target_conviction: str,
    dau: float,
    primary_driver: str,
    top_risk: str,
) -> str:
    return (
        f"{symbol} receives a {analyst_rating} rating with {target_conviction} conviction. "
        f"The primary driver is the {primary_driver} factor with DAU of {dau:.1f}. "
        f"Key risk is {top_risk}, which could invalidate the thesis if conditions deteriorate."
    )


def _build_investment_thesis(
    symbol: str,
    signal_label: str,
    regime: str,
    supporting_evidence: List[EvidenceItem],
    eis: Optional[float],
    caps: Optional[float],
) -> str:
    top_support = (
        " and ".join(e.description for e in supporting_evidence[:2])
        if supporting_evidence else "systematic factor alignment"
    )
    regime_stance = (
        "supports" if regime.upper() in ("TRENDING", "RECOVERY") and signal_label == "BUY"
        else "adds caution to"
    )
    eis_stmt = (
        f"Earnings quality (EIS={eis:.0f}) indicates {'high' if eis > 65 else 'moderate'} integrity. "
        if eis is not None else ""
    )
    caps_stmt = (
        f"CAPS score of {caps:.0f} signals {'strong' if caps > 65 else 'adequate'} shareholder value creation. "
        if caps is not None else ""
    )
    return (
        f"The {signal_label} thesis for {symbol} is supported by {top_support}. "
        f"The {regime} regime context {regime_stance} this position. "
        f"{eis_stmt}{caps_stmt}"
    ).strip()


def _build_risk_summary(
    fragility: Optional[float],
    scps: Optional[float],
    signal_label: str,
) -> str:
    frags = f"Fragility score {fragility:.0f} indicates " if fragility is not None else ""
    risk_level = (
        "elevated tail-risk environment"
        if fragility is not None and fragility > 60
        else "moderate risk environment"
    )
    scps_warn = (
        f" Sornette SCPS of {scps:.0f} warns of potential critical-point dynamics."
        if scps is not None and scps > 65 else ""
    )
    return (
        f"{frags}{risk_level}.{scps_warn} "
        f"{'BUY conviction should be reduced if fragility rises above 70.' if signal_label == 'BUY' else ''}"
    ).strip()


# ---------------------------------------------------------------------------
# LLM enhancement (optional)
# ---------------------------------------------------------------------------

def _try_llm_enhance(
    symbol: str,
    executive_summary: str,
    investment_thesis: str,
    context_str: str,
) -> tuple:
    try:
        from api.llm.client import LLMClient
        client = LLMClient()
        system_msg = (
            "You are an institutional equity analyst. Polish the following research note "
            "narratives using only the provided AXIOM data. Do not add facts not present in the context. "
            "Return exactly two paragraphs separated by '|||': executive_summary then investment_thesis."
        )
        user_msg = (
            f"Symbol: {symbol}\n"
            f"Context: {context_str}\n\n"
            f"Executive Summary (to polish):\n{executive_summary}\n\n"
            f"Investment Thesis (to polish):\n{investment_thesis}"
        )
        reply, _, _ = client.complete_chat(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            max_tokens=400,
        )
        parts = reply.split("|||")
        polished_exec = parts[0].strip() + " [AI-Enhanced]" if parts else executive_summary
        polished_thesis = parts[1].strip() + " [AI-Enhanced]" if len(parts) > 1 else investment_thesis
        return polished_exec, polished_thesis
    except Exception as exc:
        logger.debug("explain.llm_enhance_failed symbol=%s err=%s", symbol, exc)
        return executive_summary, investment_thesis


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_research_report(
    symbol: str,
    axiom_payload: Dict[str, Any],
    signal_label: str,
    use_llm: bool = False,
) -> ResearchReport:
    dau = float(axiom_payload.get("deployable_alpha_utility") or 50.0)
    ic_state = str(axiom_payload.get("ic_state") or "INSUFFICIENT")
    regime = str(axiom_payload.get("regime_label") or "UNKNOWN")
    engine_scores = axiom_payload.get("engine_scores") or {}
    alpha_decomp = axiom_payload.get("alpha_decomposition") or {}
    primary_driver = str(alpha_decomp.get("primary_driver") or "systematic factors")

    analyst_rating = _dau_to_analyst_rating(dau, signal_label)
    target_conviction = _ic_to_conviction(ic_state)

    # Build sections
    fundamental_section = _build_fundamental_section(engine_scores)
    technical_section = _build_technical_section(engine_scores, regime)
    risk_section_data = _build_risk_section(engine_scores)
    alt_section = _build_alternative_section(engine_scores)
    factor_section = _build_factor_section(axiom_payload)

    # Build reasoning chain and evidence
    chain = build_reasoning_chain(symbol, axiom_payload, signal_label)
    evidence = extract_evidence_items(axiom_payload, signal_label)
    evidence_balance = compute_evidence_balance(evidence)
    counterfactuals = compute_counterfactuals(axiom_payload, signal_label)

    # Gather inputs for narratives
    fragility = risk_section_data.get("fragility_score")
    scps = risk_section_data.get("scps")
    eis = fundamental_section.get("eis")
    caps = fundamental_section.get("caps")
    supporting_evidence = evidence.get("supporting") or []

    top_risk = (
        "Sornette critical point dynamics (SCPS elevated)" if scps and float(scps) > 65
        else "Fragility spike above 70" if fragility and float(fragility) > 55
        else "Regime deterioration to HIGH_VOL"
    )

    executive_summary = _build_executive_summary(
        symbol, analyst_rating, target_conviction, dau, primary_driver, top_risk
    )
    investment_thesis = _build_investment_thesis(
        symbol, signal_label, regime, supporting_evidence,
        float(eis) if eis else None,
        float(caps) if caps else None,
    )
    risk_summary = _build_risk_summary(
        float(fragility) if fragility else None,
        float(scps) if scps else None,
        signal_label,
    )

    # Optional LLM polish
    if use_llm and config.llm_enabled():
        ctx = json.dumps({
            "dau": dau, "regime": regime, "eis": eis, "caps": caps,
            "fragility": fragility, "scps": scps,
        })
        executive_summary, investment_thesis = _try_llm_enhance(
            symbol, executive_summary, investment_thesis, ctx
        )

    invalidation_triggers = chain.invalidation_conditions

    report = ResearchReport(
        symbol=symbol,
        report_date=dt.date.today(),
        analyst_rating=analyst_rating,
        target_conviction=target_conviction,
        executive_summary=executive_summary,
        investment_thesis=investment_thesis,
        risk_summary=risk_summary,
        fundamental_section=fundamental_section,
        technical_section=technical_section,
        risk_section=risk_section_data,
        alternative_data_section=alt_section,
        factor_section=factor_section,
        invalidation_triggers=invalidation_triggers,
        reasoning_chain=chain,
        evidence=evidence_balance,
        counterfactuals=counterfactuals,
    )

    _save_report(report, axiom_payload, use_llm)
    return report


def _save_report(report: ResearchReport, payload: Dict, use_llm: bool) -> None:
    if not db.db_write_enabled():
        return
    try:
        db.safe_execute(
            """
            INSERT INTO research_reports
                (report_id, symbol, report_date, analyst_rating, dau, conviction,
                 executive_summary, report_json, use_llm, created_at)
            VALUES (gen_random_uuid()::text, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, now())
            """,
            (
                report.symbol,
                report.report_date,
                report.analyst_rating,
                payload.get("deployable_alpha_utility"),
                report.target_conviction,
                report.executive_summary,
                json.dumps({"rating": report.analyst_rating, "conviction": report.target_conviction}),
                use_llm,
            ),
        )
    except Exception as exc:
        logger.debug("explain.save_report_failed symbol=%s err=%s", report.symbol, exc)


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------

def format_report_as_text(report: ResearchReport) -> str:
    supporting = report.evidence.get("supporting_count", 0)
    lines = [
        f"=== AXIOM RESEARCH — {report.symbol} ===",
        f"Rating: {report.analyst_rating} | Conviction: {report.target_conviction}",
        f"DAU: {report.reasoning_chain.dau:.1f} | Regime: {report.technical_section.get('regime', 'N/A')} | Date: {report.report_date}",
        "",
        "EXECUTIVE SUMMARY",
        report.executive_summary,
        "",
        "INVESTMENT THESIS",
        report.investment_thesis,
        "",
        f"SUPPORTING EVIDENCE ({supporting} signals)",
    ]

    for i, cf in enumerate(report.counterfactuals[:3], 1):
        lines.append(f"{i}. {cf.get('plain_english', '')}")

    lines += [
        "",
        "RISK FACTORS",
        report.risk_summary,
        "",
        "INVALIDATION CONDITIONS",
    ]
    for cond in report.invalidation_triggers:
        lines.append(f"- {cond}")

    lines += [
        "",
        "THEORETICAL FOUNDATIONS",
    ]
    for foundation in report.reasoning_chain.theoretical_foundations:
        lines.append(f"- {foundation}")

    return "\n".join(lines)
