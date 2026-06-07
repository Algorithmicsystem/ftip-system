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
    "signal_query",
    "universe_buy_signals",
    "regime_query",
    "risk_query",
    "signal_history",
    "leaderboard",
    "explain_signal",
    "portfolio_query",
    "regime_change",
    "moat_query",
    # Legacy intents preserved
    "comparison",
    "factor_query",
    "screening",
    "historical",
    # Backward-compat alias
    "signal_explanation",
    "unknown",
]

import re as _re

_INTENT_PATTERNS = {
    # signal_explanation must come first to preserve backward compat for "why...buy" and "explain"
    "signal_explanation": [r"\bexplain\b", r"\bwhy\b", r"what.*driving", r"\bbecause\b", r"what drives"],
    "universe_buy_signals": [
        r"which.*buy.*signal",
        r"best.*opportunit",
        r"top.*signal",
        r"strongest.*signal",
        r"screen.*universe",
    ],
    "regime_query": [
        r"what.*regime",
        r"market.*regime",
        r"current.*market.*condition",
        r"macro.*environment",
    ],
    # portfolio_query before risk_query to catch "value at risk" first
    "portfolio_query": [
        r"portfolio",
        r"\bvar\b",
        r"value.*at.*risk",
        r"stress.*test",
    ],
    "risk_query": [
        r"risk.*symbol",
        r"highest.*risk",
        r"crash.*risk",
        r"\bsri\b",
        r"fragil",
        r"danger",
        r"bubble.*risk",
    ],
    "signal_query": [
        r"what.*axiom.*say.*about\s+(\w+)",
        r"what.*signal.*for\s+(\w+)",
        r"should\s+i\s+buy",
    ],
    "signal_history": [
        r"\w+.*yesterday",
        r"\w+.*last.*week",
        r"what happened.*\w+",
    ],
    "leaderboard": [
        r"best.*track.*record",
        r"signal.*war",
        r"batting.*average",
        r"most.*accurate",
        r"best.*performing.*signal",
    ],
    "explain_signal": [
        r"why.*\w+.*sell",
    ],
    "regime_change": [
        r"when.*regime.*change",
        r"transition.*probabilit",
        r"next.*regime",
        r"regime.*shift",
    ],
    "moat_query": [
        r"moat.*score",
        r"how.*long.*running",
        r"data.*depth",
        r"intelligence.*quality",
        r"dossier.*iq",
    ],
    "comparison": [r"compar", r"\bvs\b", r"versus", r"\bbetter\b", r"\bwhich\b"],
    "factor_query": [r"\bfactor\b", r"\beif\b", r"\bscaf\b", r"\bcaps\b", r"\beis\b"],
    "screening": [r"\btop\b", r"\bbest\b", r"\bscreen\b", r"\blist\b.*symbol"],
    "historical": [r"histor", r"analog", r"\bpast\b"],
}


def classify_query_intent(query: str) -> str:
    q = query.lower()
    for intent, patterns in _INTENT_PATTERNS.items():
        for pat in patterns:
            if _re.search(pat, q):
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
        if intent in ("signal_explanation", "signal_query", "explain_signal", "risk_query", "factor_query", "comparison", "signal_history") and symbols:
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
                "SELECT payload->>'regime_label' AS regime_label, payload->>'regime_strength' AS regime_strength FROM axiom_scores_daily ORDER BY as_of_date DESC LIMIT 1"
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

        if intent == "universe_buy_signals":
            try:
                rows = db.safe_fetchall(
                    """
                    SELECT DISTINCT ON (symbol) symbol,
                           (payload->>'deployable_alpha_utility')::numeric AS dau,
                           payload->>'regime_label' AS regime
                      FROM axiom_scores_daily
                     WHERE (payload->>'deployable_alpha_utility')::numeric >= 65
                     ORDER BY symbol, as_of_date DESC
                    """,
                ) or []
                context["buy_signals"] = [
                    {"symbol": r[0], "dau": float(r[1] or 65), "regime": r[2]} for r in rows
                ]
            except Exception:
                pass

        if intent == "leaderboard":
            try:
                rows = db.safe_fetchall(
                    """
                    SELECT symbol, batting_average_21d
                      FROM signal_war
                     ORDER BY batting_average_21d DESC NULLS LAST
                     LIMIT 10
                    """,
                ) or []
                context["leaderboard"] = [{"symbol": r[0], "batting_avg": float(r[1] or 0)} for r in rows]
            except Exception:
                pass

        if intent == "moat_query":
            try:
                row = db.safe_fetchone(
                    "SELECT moat_score, moat_tier, data_depth_score FROM company_moat_scores ORDER BY updated_at DESC LIMIT 1"
                )
                if row:
                    context["moat"] = {"moat_score": row[0], "tier": row[1], "data_depth": row[2]}
            except Exception:
                pass

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
    # signal_query and signal_explanation both resolve via axiom_data
    if intent in ("signal_explanation", "signal_query", "signal_history"):
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

    if intent == "explain_signal":
        axiom_data = context.get("axiom_data")
        if axiom_data:
            d = axiom_data[0]
            sym = d.get("symbol", symbols[0] if symbols else "?")
            dau_val = d.get("dau", "?")
            primary = d.get("primary_driver", "unknown")
            regime = d.get("regime", "unknown")
            eis = d.get("eis", "?")
            return (
                f"{sym} DAU={dau_val}. Primary driver: {primary}. "
                f"EIS={eis}, regime={regime}. "
                f"For a full grounded explanation, use /explain/signal/{sym}."
            )
        return f"No AXIOM data found for {symbols[0] if symbols else 'symbol'}. Ensure scores are loaded."

    if intent == "universe_buy_signals":
        buy_signals = context.get("buy_signals")
        if buy_signals:
            lines = [f"{b['symbol']}: DAU={b['dau']:.0f}" for b in buy_signals[:5]]
            return f"Current BUY signals (DAU >= 65):\n" + "\n".join(lines)
        return "No BUY signals found in the current universe."

    if intent == "leaderboard":
        leaderboard = context.get("leaderboard")
        if leaderboard:
            lines = [f"{l['symbol']}: {l['batting_avg']*100:.0f}% accuracy" for l in leaderboard[:5]]
            return "Top signal accuracy leaders (21-day horizon):\n" + "\n".join(lines)
        return "Signal WAR leaderboard data not available."

    if intent == "moat_query":
        moat = context.get("moat")
        if moat:
            return (
                f"Moat score: {moat.get('moat_score', 'N/A')}, "
                f"tier: {moat.get('tier', 'N/A')}, "
                f"data depth: {moat.get('data_depth', 'N/A')}."
            )
        return "Moat/intelligence quality data not available."

    if intent == "portfolio_query":
        axiom_data = context.get("axiom_data")
        if axiom_data:
            d = axiom_data[0]
            return (
                f"Portfolio risk snapshot — {d['symbol']}: "
                f"fragility={d.get('fragility', 'N/A')}, regime={d.get('regime', 'N/A')}. "
                "For portfolio VaR/stress test, use /backtest endpoints."
            )
        return "Portfolio risk query: provide specific symbols or use /pe/portfolio endpoints."

    if intent == "regime_change":
        regime_data = context.get("current_regime")
        if regime_data:
            return (
                f"Current regime: {regime_data.get('label', 'unknown')} "
                f"(strength={regime_data.get('strength', 'N/A')}). "
                "Regime transition probabilities are computed daily from breadth signals."
            )
        return "Regime transition data unavailable — check market breadth feed."

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
        context.get("top_opportunities") or
        context.get("buy_signals") or
        context.get("leaderboard") or
        context.get("moat")
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
    result = {
        "query": query,
        "intent": intent,
        "answer": answer,
        "confidence": confidence,
        "grounded": grounded,
    }

    try:
        from api.compliance.audit_trail import write_audit_record
        write_audit_record(
            "analysis.query_answered",
            "query", query[:120],
            {"intent": intent, "grounded": grounded, "confidence": confidence},
            symbol=symbols[0] if symbols else None,
            output_summary=f"Query answered: intent={intent}, grounded={grounded}",
        )
    except Exception:
        pass

    return result


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
