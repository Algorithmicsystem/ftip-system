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
    dependencies=[Depends(require_tier("enterprise"))],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_axiom_payload(symbol: str) -> Dict[str, Any]:
    """Load latest AXIOM payload from DB, or return minimal stub."""
    from api import db
    if not db.db_read_enabled():
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
            return dict(row[0]) if isinstance(row[0], dict) else {}
    except Exception:
        pass
    return {}


def _signal_from_dau(dau: float) -> str:
    if dau >= 65:
        return "BUY"
    if dau <= 40:
        return "SELL"
    return "HOLD"


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
    from api.explain.reasoning_engine import build_reasoning_chain
    payload = _load_axiom_payload(symbol)
    dau = float(payload.get("deployable_alpha_utility") or 50.0)
    signal_label = _signal_from_dau(dau)
    chain = build_reasoning_chain(symbol, payload, signal_label)
    return {
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
    }


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
) -> Dict[str, Any]:
    """Generate or retrieve research report for a symbol."""
    from api.explain.research_report import generate_research_report
    payload = _load_axiom_payload(symbol)
    dau = float(payload.get("deployable_alpha_utility") or 50.0)
    signal_label = _signal_from_dau(dau)
    report = generate_research_report(symbol, payload, signal_label, use_llm=use_llm)
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
