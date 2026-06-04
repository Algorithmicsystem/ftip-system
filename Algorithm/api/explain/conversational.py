"""Phase 16.5: Conversational Intelligence Interface."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from api import config, db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

QUERY_INTENTS = [
    "signal_explanation",
    "comparison",
    "risk_query",
    "regime_query",
    "screening",
    "factor_query",
    "historical",
    "unknown",
]

_INTENT_KEYWORDS = {
    "signal_explanation": {"why", "explain", "what drives", "what's driving", "reason", "because"},
    "comparison": {"compare", "vs", "versus", "better", "which"},
    "risk_query": {"risk", "fragile", "bubble", "crash", "danger", "vulnerable"},
    "regime_query": {"regime", "environment", "market", "macro", "conditions"},
    "screening": {"top", "best", "screen", "opportunities", "find", "list"},
    "factor_query": {"factor", "eif", "scaf", "loading", "decomposition", "caps", "eis"},
    "historical": {"history", "historical", "similar", "analog", "past", "before"},
}


def classify_query_intent(query: str) -> str:
    q = query.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                return intent
    return "unknown"


# ---------------------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------------------

_COMMON_WORDS = {
    "I", "A", "IS", "THE", "AND", "OR", "FOR", "IN", "OF", "TO", "BE",
    "NOT", "VS", "WHY", "WHAT", "HOW", "SHOW", "TOP", "ME", "IT", "MY",
    "AT", "BY", "AN", "ON", "IF", "AS", "DO", "NO", "SO", "UP", "AM",
    "BUY", "SELL", "HOLD", "ETF", "IPO", "CEO", "CFO", "EPS", "GDP",
    "FED", "SEC", "IMF", "ECB",
}


def extract_symbols_from_query(query: str) -> List[str]:
    """Extract likely ticker symbols (1-5 uppercase chars) from a query."""
    candidates = re.findall(r'\b[A-Z]{1,5}\b', query)
    return [s for s in candidates if s not in _COMMON_WORDS and len(s) >= 2]


# ---------------------------------------------------------------------------
# Grounded context fetching
# ---------------------------------------------------------------------------

def fetch_grounded_context(
    query: str,
    intent: str,
    symbols: List[str],
) -> Dict[str, Any]:
    """Fetch relevant AXIOM data from DB to answer the query."""
    if not db.db_read_enabled():
        return {}

    context: Dict[str, Any] = {"intent": intent, "symbols": symbols}

    try:
        if intent in ("signal_explanation", "risk_query", "factor_query", "comparison") and symbols:
            rows = db.safe_fetchall(
                """
                SELECT symbol, as_of_date,
                       payload->>'deployable_alpha_utility' AS dau,
                       payload->>'ic_state' AS ic_state,
                       payload->>'regime_label' AS regime,
                       payload->'engine_scores'->'critical_fragility'->>'score' AS fragility,
                       payload->'engine_scores'->'fundamental_reality'->'components'->>'eis_component' AS eis,
                       payload->'alpha_decomposition'->>'primary_driver' AS primary_driver
                  FROM axiom_scores_daily
                 WHERE symbol = ANY(%s)
                 ORDER BY as_of_date DESC
                 LIMIT %s
                """,
                (symbols, len(symbols) * 2),
            )
            if rows:
                context["axiom_data"] = [
                    {
                        "symbol": r[0],
                        "as_of_date": str(r[1]),
                        "dau": r[2],
                        "ic_state": r[3],
                        "regime": r[4],
                        "fragility": r[5],
                        "eis": r[6],
                        "primary_driver": r[7],
                    }
                    for r in rows
                ]

        if intent == "regime_query":
            row = db.safe_fetchone(
                "SELECT regime_label, regime_strength FROM market_breadth_daily ORDER BY as_of_date DESC LIMIT 1"
            )
            if row:
                context["current_regime"] = {"label": row[0], "strength": row[1]}

        if intent == "screening":
            rows = db.safe_fetchall(
                """
                SELECT symbol,
                       payload->>'deployable_alpha_utility' AS dau,
                       payload->>'regime_label' AS regime
                  FROM axiom_scores_daily
                 ORDER BY (payload->>'deployable_alpha_utility')::float DESC
                 LIMIT 10
                """,
            )
            if rows:
                context["top_opportunities"] = [
                    {"symbol": r[0], "dau": r[1], "regime": r[2]} for r in rows
                ]

    except Exception as exc:
        logger.debug("explain.fetch_context_failed intent=%s err=%s", intent, exc)

    return context


# ---------------------------------------------------------------------------
# Programmatic answers (no LLM)
# ---------------------------------------------------------------------------

def _programmatic_answer(
    intent: str,
    context: Dict[str, Any],
    symbols: List[str],
    query: str,
) -> str:
    if intent == "signal_explanation":
        axiom_data = context.get("axiom_data")
        if axiom_data:
            d = axiom_data[0]
            parts = [f"{d.get('symbol', '?')} has DAU={d.get('dau', '?')}"]
            if d.get("ic_state"):
                parts.append(f"({d['ic_state']} IC)")
            if d.get("primary_driver"):
                parts.append(f"Primary driver: {d['primary_driver']}")
            if d.get("regime"):
                parts.append(f"Regime: {d['regime']}")
            if d.get("fragility"):
                parts.append(f"Fragility: {d['fragility']}")
            if d.get("eis"):
                parts.append(f"EIS: {d['eis']}")
            return ". ".join(parts) + "."
        return (
            f"Signal explanation for {symbols[0] if symbols else 'symbol'} — "
            "data not available. Submit financials to generate AXIOM scores."
        )

    if intent == "comparison":
        axiom_data = context.get("axiom_data")
        if axiom_data and len(axiom_data) >= 2:
            a, b = axiom_data[0], axiom_data[1]
            return (
                f"{a['symbol']}: DAU={a['dau']} | {b['symbol']}: DAU={b['dau']}. "
                f"{'Higher DAU → stronger signal at ' + a['symbol'] if float(a['dau'] or 0) > float(b['dau'] or 0) else 'Higher DAU → stronger signal at ' + b['symbol']}."
            )
        return "Comparison requires both symbols to have AXIOM scores."

    if intent == "risk_query":
        axiom_data = context.get("axiom_data")
        if axiom_data:
            d = axiom_data[0]
            return (
                f"{d['symbol']} fragility={d['fragility']}, EIS={d['eis']}. "
                f"IC={d['ic_state']}, regime={d['regime']}."
            )
        return "Risk data not available for the specified symbol."

    if intent == "regime_query":
        regime_data = context.get("current_regime")
        if regime_data:
            return f"Current regime: {regime_data['label']} (strength={regime_data['strength']})."
        return "Regime data unavailable — check market breadth feed."

    if intent == "screening":
        opps = context.get("top_opportunities")
        if opps:
            lines = [f"{o['symbol']}: DAU={o['dau']}" for o in opps[:5]]
            return "Top opportunities by DAU:\n" + "\n".join(lines)
        return "Screening data not available — no AXIOM scores found."

    return f"Query received (intent={intent}). Provide symbol(s) for detailed AXIOM analysis."


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

def answer_intelligence_query(
    query: str,
    context: Dict[str, Any],
    use_llm: bool = True,
) -> Dict[str, Any]:
    intent = classify_query_intent(query)
    symbols = extract_symbols_from_query(query)
    grounded = bool(context and (
        context.get("axiom_data") or
        context.get("current_regime") or
        context.get("top_opportunities")
    ))
    confidence = "high" if grounded else "low"

    if use_llm and config.llm_enabled():
        try:
            from api.llm.client import LLMClient
            import json as _json
            client = LLMClient()
            system_msg = (
                "You are an institutional financial analyst using the AXIOM intelligence system. "
                "Answer ONLY based on the provided AXIOM data context. "
                "Every number you cite must come from the provided context. "
                "If you don't have sufficient data to answer, say so explicitly."
            )
            context_str = _json.dumps(context, default=str)[:3000]
            user_msg = f"Question: {query}\n\nAXIOM Context:\n{context_str}"
            reply, _, _ = client.complete_chat(
                [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                max_tokens=500,
            )
            return {
                "query": query,
                "intent": intent,
                "answer": reply,
                "data_used": list(context.keys()),
                "confidence": confidence,
                "grounded": grounded,
            }
        except Exception as exc:
            logger.debug("explain.llm_query_failed err=%s", exc)

    # Fallback to programmatic
    answer = _programmatic_answer(intent, context, symbols, query)
    return {
        "query": query,
        "intent": intent,
        "answer": answer,
        "confidence": confidence,
        "grounded": grounded,
    }


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_symbols(
    symbol_a: str,
    symbol_b: str,
) -> Dict[str, Any]:
    """Side-by-side AXIOM comparison of two symbols."""
    context = fetch_grounded_context("", "comparison", [symbol_a, symbol_b])
    axiom_data = context.get("axiom_data") or []

    a_data = next((d for d in axiom_data if d["symbol"] == symbol_a), {})
    b_data = next((d for d in axiom_data if d["symbol"] == symbol_b), {})

    def _safe_float(v: Any) -> Optional[float]:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    a_dau = _safe_float(a_data.get("dau"))
    b_dau = _safe_float(b_data.get("dau"))

    winner = None
    if a_dau is not None and b_dau is not None:
        winner = symbol_a if a_dau > b_dau else symbol_b if b_dau > a_dau else "tie"

    return {
        symbol_a: a_data,
        symbol_b: b_data,
        "preferred_symbol": winner,
        "basis": "DAU (Deployable Alpha Utility) — higher is stronger signal",
    }
