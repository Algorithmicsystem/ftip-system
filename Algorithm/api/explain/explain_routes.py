"""Phase 16.6: Explanation API endpoints."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from api.jobs.tenant_auth import require_tier

router = APIRouter(
    prefix="/explain",
    tags=["explain"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_axiom_payload(symbol: str) -> Dict[str, Any]:
    """Load latest AXIOM payload from DB, or return minimal stub."""
    import json
    from api import db
    if not db.db_enabled():
        return {}
    try:
        row = db.safe_fetchone(
            """
            SELECT payload,
                   payload->>'deployable_alpha_utility' AS dau,
                   payload->>'ic_state' AS ic_state,
                   payload->>'regime_label' AS regime
              FROM axiom_scores_daily
             WHERE symbol = %s
             ORDER BY as_of_date DESC
             LIMIT 1
            """,
            (symbol,),
        )
        if row and row[0]:
            raw = row[0]
            if isinstance(raw, dict):
                payload = raw
            elif isinstance(raw, str):
                try:
                    payload = json.loads(raw)
                except Exception:
                    payload = {}
            else:
                payload = {}
            # Derive primary_driver from highest engine score if missing
            if not payload.get("primary_driver"):
                engines = (payload.get("engine_scores") or {})
                if engines:
                    best = max(engines.items(), key=lambda kv: (kv[1] or {}).get("score", 0) if isinstance(kv[1], dict) else 0, default=(None, None))
                    if best[0]:
                        alpha = payload.setdefault("alpha_decomposition", {})
                        alpha.setdefault("primary_driver", best[0])
            return payload
    except Exception:
        pass
    return {}


def _signal_from_dau(dau: float) -> str:
    if dau >= 65:
        return "BUY"
    if dau <= 40:
        return "SELL"
    return "HOLD"


def build_grounded_explanation(symbol: str, payload: Dict[str, Any]) -> str:
    """
    Build a grounded signal explanation referencing actual computed values.
    Every sentence traces to a specific number from the AXIOM payload.
    """
    dau = float(payload.get("deployable_alpha_utility") or 50.0)
    signal = _signal_from_dau(dau)
    engine_scores = payload.get("engine_scores") or {}
    fund_comps = (engine_scores.get("fundamental_reality") or {}).get("components") or {}
    eis = float(fund_comps.get("eis_component") or fund_comps.get("earnings_quality_component") or 50.0)
    caps = float(fund_comps.get("caps_component") or 50.0)
    frag = (engine_scores.get("critical_fragility") or {}).get("score")
    factor_composite = float((engine_scores.get("flow_transmission") or {}).get("score") or 50.0)
    alpha_decomp = payload.get("alpha_decomposition") or {}
    primary = str(alpha_decomp.get("primary_driver") or "EIF")
    regime = str(payload.get("regime_label") or "unknown")
    ic_state = str(payload.get("ic_state") or "INSUFFICIENT")
    suppression = bool(payload.get("suppression_active") or False)
    pre_suppression = str(payload.get("pre_suppression_action") or signal)
    suppression_reason = str(payload.get("suppression_reason") or "risk_filter")

    action_phrase = {
        "BUY": "a BUY signal",
        "SELL": "a SELL signal",
        "HOLD": "a HOLD signal",
    }.get(signal, "a HOLD signal")

    line1 = f"AXIOM assigns {symbol} {action_phrase} with a Deployable Alpha Utility (DAU) score of {dau:.1f}/100."

    driver_map = {
        "EIF": (f"The primary driver is earnings quality (EIS: {eis:.0f}/100). " +
                ("Earnings integrity is strong with no major shenanigan flags." if eis > 65 else
                 "Earnings quality shows elevated accounting risk." if eis < 40 else
                 "Earnings quality is moderate.")),
        "CMF": (f"The primary driver is capital allocation quality (CAPS: {caps:.0f}/100). " +
                ("CAPS reflects strong competitive advantage and shareholder value creation." if caps > 65 else
                 "CAPS indicates constrained capital allocation quality." if caps < 40 else
                 "CAPS is adequate.")),
        "BAF": (f"The primary driver is behavioral/flow dynamics (Factor Composite: {factor_composite:.0f}/100). " +
                ("Factor tilts are strongly aligned with signal direction." if factor_composite > 65 else
                 "Factor environment is working against this signal." if factor_composite < 40 else
                 "Factor environment is neutral.")),
        "SCAF": ("The primary driver is crash risk assessment (SCAF). " +
                 ("Sornette crash probability is elevated — this SELL signal reflects genuine tail risk." if signal in ("SELL",) else
                  "Crash risk is being monitored but is not imminent.")),
    }
    line2 = driver_map.get(primary, f"The primary driver is the {primary} engine contributing most to the signal.")

    regime_display = regime.replace("_", " ").title()
    if suppression:
        line3 = (f"Note: AXIOM's underlying assessment was {pre_suppression} but the signal is suppressed "
                 f"to HOLD due to {suppression_reason.replace('_', ' ')}. "
                 f"The {regime_display} regime has activated risk filters.")
    else:
        conviction = {"STRONG": "high", "MODERATE": "moderate", "WEAK": "low"}.get(ic_state, "low")
        line3 = f"In the current {regime_display} regime, this signal carries {conviction} confidence (IC state: {ic_state})."

    batting = payload.get("signal_batting_average") or payload.get("signal_war")
    if batting is not None:
        line4 = (f"Historical track record: AXIOM has been correct on {symbol} signals "
                 f"{float(batting)*100:.0f}% of the time at the 21-day horizon.")
    else:
        line4 = "Signal track record is building — insufficient history for batting average."

    frag_val = float(frag) if frag is not None else None
    if frag_val and frag_val > 65:
        top_risk = f"elevated fragility score ({frag_val:.0f}/100)"
    elif regime.upper() in ("HIGH_VOL", "LIQUIDITY_FRACTURE"):
        top_risk = f"adverse market regime ({regime_display})"
    else:
        top_risk = "regime deterioration"
    line5 = f"Primary risk to this signal: {top_risk}."

    return f"{line1}\n\n{line2}\n\n{line3}\n\n{line4}\n\n{line5}"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryIn(BaseModel):
    query: str
    use_llm: bool = True


class CompareIn(BaseModel):
    symbol_a: str
    symbol_b: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/signal/{symbol}")
def get_signal_reasoning(symbol: str) -> Dict[str, Any]:
    """Full reasoning chain for the latest signal for a symbol."""
    import os
    from api.explain.reasoning_engine import build_reasoning_chain
    payload = _load_axiom_payload(symbol)
    dau = float(payload.get("deployable_alpha_utility") or 50.0)
    signal_label = _signal_from_dau(dau)
    chain = build_reasoning_chain(symbol, payload, signal_label)
    engine_scores = payload.get("engine_scores") or {}
    fund_comps = (engine_scores.get("fundamental_reality") or {}).get("components") or {}
    alpha_decomp = payload.get("alpha_decomposition") or {}
    frag = (engine_scores.get("critical_fragility") or {}).get("score")

    response: Dict[str, Any] = {
        "symbol": chain.symbol,
        "as_of_date": chain.as_of_date.isoformat(),
        "signal_label": chain.signal_label,
        "dau": chain.dau,
        "primary_conclusion": chain.primary_conclusion,
        "reasoning_steps": [
            {
                "step_id": s.step_id,
                "claim": s.claim,
                "evidence": s.evidence,
                "confidence": s.confidence,
                "source": s.source,
                "theoretical_grounding": s.theoretical_grounding,
            }
            for s in chain.reasoning_steps
        ],
        "supporting_factors": chain.supporting_factors,
        "contradicting_factors": chain.contradicting_factors,
        "confidence_overall": chain.confidence_overall,
        "invalidation_conditions": chain.invalidation_conditions,
        "theoretical_foundations": chain.theoretical_foundations,
        "explanation_text": build_grounded_explanation(symbol, payload),
        "llm_enhanced": False,
    }

    # OpenAI enhancement — additive, never blocks
    from api import config as _cfg
    if _cfg.openai_api_key():
        try:
            from api.llm.openai_client import synthesize_signal_explanation
            frag_val = float(frag) if frag is not None else 50.0
            top_risk = (
                f"elevated fragility ({frag_val:.0f}/100)" if frag_val > 65
                else f"regime ({payload.get('regime_label', 'unknown')})"
            )
            ai_text = synthesize_signal_explanation({
                "symbol": symbol,
                "signal_label": signal_label,
                "dau": dau,
                "eis_score": float(fund_comps.get("eis_component") or 50.0),
                "caps_score": float(fund_comps.get("caps_component") or 50.0),
                "factor_composite": float((engine_scores.get("flow_transmission") or {}).get("score") or 50.0),
                "primary_driver": str(alpha_decomp.get("primary_driver") or "EIF"),
                "regime_label": str(payload.get("regime_label") or "Neutral"),
                "top_risk": top_risk,
            })
            if ai_text:
                response["ai_synthesis"] = ai_text
                response["llm_enhanced"] = True
        except Exception:
            pass

    return response


@router.get("/evidence/{symbol}")
def get_evidence(symbol: str) -> Dict[str, Any]:
    """Supporting vs contradicting evidence breakdown."""
    from api.explain.explanation_system import compute_evidence_balance, extract_evidence_items
    payload = _load_axiom_payload(symbol)
    dau = float(payload.get("deployable_alpha_utility") or 50.0)
    signal_label = _signal_from_dau(dau)
    evidence = extract_evidence_items(payload, signal_label)
    balance = compute_evidence_balance(evidence)
    return {
        "symbol": symbol,
        "signal_label": signal_label,
        "supporting": [
            {"direction": e.direction, "category": e.category, "description": e.description,
             "strength": e.strength, "data_point": e.data_point, "source": e.source}
            for e in evidence["supporting"]
        ],
        "contradicting": [
            {"direction": e.direction, "category": e.category, "description": e.description,
             "strength": e.strength, "data_point": e.data_point, "source": e.source}
            for e in evidence["contradicting"]
        ],
        "balance": balance,
    }


@router.get("/counterfactual/{symbol}")
def get_counterfactuals(
    symbol: str,
    target_signal: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Counterfactual analysis — what would need to change to flip the signal."""
    from api.explain.counterfactual import compute_counterfactuals
    payload = _load_axiom_payload(symbol)
    dau = float(payload.get("deployable_alpha_utility") or 50.0)
    signal_label = _signal_from_dau(dau)
    counterfactuals = compute_counterfactuals(payload, signal_label, target_signal)
    return {
        "symbol": symbol,
        "current_signal": signal_label,
        "current_dau": dau,
        "target_signal": target_signal,
        "counterfactuals": counterfactuals,
    }


@router.get("/sensitivity/{symbol}")
def get_sensitivity(
    symbol: str,
    component: str = Query(default="eis_component"),
) -> Dict[str, Any]:
    """DAU sensitivity to a specific component."""
    from api.explain.counterfactual import compute_signal_sensitivity
    payload = _load_axiom_payload(symbol)
    return compute_signal_sensitivity(payload, component)


@router.get("/report/{symbol}")
def get_research_report(
    symbol: str,
    use_llm: bool = Query(default=False),
    format: str = Query(default="json"),
) -> Any:
    """Generate research report. ?format=text returns plain text."""
    from api.explain.research_report import generate_research_report, format_report_as_text
    payload = _load_axiom_payload(symbol)
    dau = float(payload.get("deployable_alpha_utility") or 50.0)
    signal_label = _signal_from_dau(dau)
    report = generate_research_report(symbol, payload, signal_label, use_llm=use_llm)
    if format == "text":
        return {"symbol": symbol, "report_text": format_report_as_text(report), "format": "text"}
    return {
        "symbol": report.symbol,
        "report_date": report.report_date.isoformat(),
        "analyst_rating": report.analyst_rating,
        "target_conviction": report.target_conviction,
        "executive_summary": report.executive_summary,
        "investment_thesis": report.investment_thesis,
        "risk_summary": report.risk_summary,
        "fundamental_section": report.fundamental_section,
        "technical_section": report.technical_section,
        "risk_section": report.risk_section,
        "factor_section": report.factor_section,
        "invalidation_triggers": report.invalidation_triggers,
        "evidence": report.evidence,
        "counterfactuals": report.counterfactuals[:3],
        "explanation_text": build_grounded_explanation(symbol, payload),
        "llm_synthesis": report.llm_synthesis,
        "llm_enhanced": report.llm_enhanced,
    }


@router.get("/report/{symbol}/text")
def get_research_report_text(
    symbol: str,
    use_llm: bool = Query(default=False),
) -> Dict[str, Any]:
    """Text-formatted research report."""
    from api.explain.research_report import format_report_as_text, generate_research_report
    payload = _load_axiom_payload(symbol)
    dau = float(payload.get("deployable_alpha_utility") or 50.0)
    signal_label = _signal_from_dau(dau)
    report = generate_research_report(symbol, payload, signal_label, use_llm=use_llm)
    return {"symbol": symbol, "report_text": format_report_as_text(report)}


@router.post("/query")
def post_intelligence_query(body: QueryIn) -> Dict[str, Any]:
    """Natural language intelligence query."""
    from api.explain.conversational import (
        answer_intelligence_query,
        classify_query_intent,
        extract_symbols_from_query,
        fetch_grounded_context,
    )
    intent = classify_query_intent(body.query)
    symbols = extract_symbols_from_query(body.query)
    context = fetch_grounded_context(body.query, intent, symbols)
    return answer_intelligence_query(body.query, context, use_llm=body.use_llm)


@router.post("/compare")
def post_compare(body: CompareIn) -> Dict[str, Any]:
    """Side-by-side AXIOM comparison of two symbols."""
    from api.explain.conversational import compare_symbols
    return compare_symbols(body.symbol_a, body.symbol_b)
