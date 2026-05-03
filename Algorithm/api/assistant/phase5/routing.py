from __future__ import annotations

from typing import Any, Dict, List, Tuple


_INTENT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "signal_summary": {
        "keywords": (
            "signal",
            "summary",
            "view",
            "stance",
        ),
        "mode": "executive_summary",
        "sections": (
            "signal_summary",
            "overall_analysis",
            "strategy_view",
        ),
        "followups": (
            "Why is this the current signal?",
            "What is the strongest evidence behind it?",
            "What would need to improve for conviction to rise?",
        ),
    },
    "technical": {
        "keywords": (
            "technical",
            "trend",
            "momentum",
            "price",
            "volume",
            "breakout",
            "support",
            "resistance",
            "regime",
            "structure",
        ),
        "mode": "analyst",
        "sections": (
            "technical_analysis",
            "statistical_analysis",
            "signal_summary",
        ),
        "followups": (
            "What is the strongest technical evidence?",
            "What makes the setup fragile from a market-structure perspective?",
            "How stable is the current regime?",
        ),
    },
    "fundamental": {
        "keywords": (
            "fundamental",
            "filing",
            "balance sheet",
            "cash flow",
            "profitability",
            "durability",
            "leverage",
            "revenue",
            "margin",
        ),
        "mode": "analyst",
        "sections": (
            "fundamental_analysis",
            "overall_analysis",
            "evidence_provenance",
        ),
        "followups": (
            "What is weakest in the fundamentals?",
            "How strong is the durability profile?",
            "Is missing coverage reducing confidence here?",
        ),
    },
    "sentiment_narrative": {
        "keywords": (
            "sentiment",
            "narrative",
            "news",
            "crowding",
            "hype",
            "attention",
            "flow",
            "crowded",
            "disagreement",
        ),
        "mode": "analyst",
        "sections": (
            "sentiment_analysis",
            "risk_quality_analysis",
            "overall_analysis",
        ),
        "followups": (
            "How serious is the narrative crowding?",
            "Is sentiment confirming or diverging from price?",
            "What event pressure matters most right now?",
        ),
    },
    "macro_geopolitical": {
        "keywords": (
            "macro",
            "rates",
            "inflation",
            "growth",
            "cross asset",
            "benchmark",
            "sector",
            "geopolitical",
            "policy",
            "market",
        ),
        "mode": "analyst",
        "sections": (
            "macro_geopolitical_analysis",
            "strategy_view",
            "overall_analysis",
        ),
        "followups": (
            "How does the macro backdrop affect this setup?",
            "Is the stock moving with beta or on idiosyncratic strength?",
            "What macro change would alter the posture?",
        ),
    },
    "market_depth": {
        "keywords": (
            "event",
            "earnings",
            "catalyst",
            "liquidity",
            "gap",
            "slippage",
            "implementation",
            "breadth",
            "leadership",
            "internals",
            "dispersion",
            "cross asset",
            "sector support",
            "benchmark support",
            "stress",
            "spillover",
            "contagion",
            "unstable environment",
        ),
        "mode": "risk",
        "sections": (
            "event_catalyst_risk_analysis",
            "liquidity_execution_fragility_analysis",
            "market_breadth_internal_state_analysis",
            "cross_asset_confirmation_analysis",
            "stress_spillover_analysis",
            "risk_quality_analysis",
        ),
        "followups": (
            "Is this setup distorted by an event or catalyst window?",
            "Is liquidity or implementation fragility suppressing confidence?",
            "Do breadth and cross-asset context actually support the move?",
        ),
    },
    "strategy": {
        "keywords": (
            "strategy",
            "hold",
            "buy",
            "sell",
            "posture",
            "participant",
            "actionability",
            "bull case",
            "bear case",
            "base case",
            "stress case",
            "wait",
            "act",
        ),
        "mode": "strategist",
        "sections": (
            "strategy_view",
            "overall_analysis",
            "risks_weaknesses_invalidators",
        ),
        "followups": (
            "Why is this HOLD instead of BUY?",
            "What is the bear case?",
            "What would change the posture materially?",
        ),
    },
    "risk_invalidation": {
        "keywords": (
            "risk",
            "invalid",
            "invalidate",
            "fragile",
            "fragility",
            "weak",
            "weakness",
            "deterioration",
            "veto",
            "no trade",
        ),
        "mode": "risk",
        "sections": (
            "risks_weaknesses_invalidators",
            "risk_quality_analysis",
            "strategy_view",
        ),
        "followups": (
            "What are the top invalidators?",
            "What makes this setup fragile?",
            "What deterioration would push the posture lower?",
        ),
    },
    "confidence_conviction": {
        "keywords": (
            "confidence",
            "conviction",
            "certainty",
            "uncertainty",
            "calibration",
            "reliability",
            "trust",
        ),
        "mode": "strategist",
        "sections": (
            "strategy_view",
            "risk_quality_analysis",
            "evidence_provenance",
        ),
        "followups": (
            "Why is conviction at this level?",
            "What is degrading confidence most?",
            "What would improve confidence quality?",
        ),
    },
    "evidence_provenance": {
        "keywords": (
            "evidence",
            "provenance",
            "source",
            "coverage",
            "freshness",
            "missing",
            "data",
            "support",
            "prove",
        ),
        "mode": "evidence",
        "sections": (
            "evidence_provenance",
            "signal_summary",
            "overall_analysis",
        ),
        "followups": (
            "What evidence is strongest?",
            "What evidence is missing or stale?",
            "Which domains agree and which conflict?",
        ),
    },
    "evaluation_performance": {
        "keywords": (
            "historical",
            "history",
            "evaluation",
            "performed",
            "performance",
            "reliability",
            "calibration",
            "worked",
            "fails",
            "failure mode",
            "edge",
        ),
        "mode": "evidence",
        "sections": (
            "evaluation_research_analysis",
            "strategy_view",
            "overall_analysis",
        ),
        "followups": (
            "How reliable are the confidence buckets?",
            "Which regimes have been strongest historically?",
            "What failure modes stand out in the scorecards?",
        ),
    },
    "deployment_readiness": {
        "keywords": (
            "live capital",
            "real money",
            "deployment",
            "deploy",
            "deployable",
            "readiness",
            "ready for live",
            "paper",
            "shadow",
            "paused",
            "pause",
            "blocked",
            "eligible",
            "trust tier",
            "human review",
            "review",
            "capital",
        ),
        "mode": "deployment",
        "sections": (
            "deployment_readiness_summary",
            "deployment_permission_analysis",
            "risk_budget_exposure_analysis",
            "rollout_stage_summary",
            "strategy_view",
            "evaluation_research_analysis",
        ),
        "followups": (
            "Is this setup paper-only or live-eligible?",
            "What is blocking this from live deployment right now?",
            "What would need to improve before trust rises?",
        ),
    },
    "portfolio_construction": {
        "keywords": (
            "portfolio",
            "watchlist",
            "candidate",
            "ranking",
            "rank",
            "priority",
            "redundant",
            "overlap",
            "diversification",
            "fit",
            "size band",
            "weight band",
            "execution quality",
            "friction",
            "turnover",
            "rebalance",
            "rotation",
            "better between",
            "portfolio better",
        ),
        "mode": "portfolio",
        "sections": (
            "portfolio_context_summary",
            "portfolio_fit_analysis",
            "execution_quality_analysis",
            "portfolio_workflow_summary",
            "strategy_view",
        ),
        "followups": (
            "Which idea fits the portfolio better right now?",
            "Why is this a watchlist candidate instead of a deployable one?",
            "What is creating redundancy or concentration here?",
        ),
    },
    "learning_research": {
        "keywords": (
            "learning",
            "learn",
            "drift",
            "experiment",
            "experiments",
            "archetype",
            "motif",
            "improve",
            "improvement",
            "improving",
            "adapt",
            "adaptation",
            "research queue",
            "failure mode",
            "what is the platform learning",
            "strongest setup",
            "strongest setups",
            "useful factor",
            "more useful",
        ),
        "mode": "learning",
        "sections": (
            "learning_summary",
            "regime_learning_summary",
            "adaptation_queue_summary",
            "experiment_registry_summary",
            "archetype_motif_summary",
            "evaluation_research_analysis",
        ),
        "followups": (
            "What is the platform learning lately?",
            "Where is the model drifting right now?",
            "Which experiments deserve review next?",
        ),
    },
    "compare_clarify": {
        "keywords": (
            "difference",
            "compare",
            "versus",
            "vs",
            "instead of",
            "clarify",
            "explain why",
        ),
        "mode": "analyst",
        "sections": (
            "overall_analysis",
            "strategy_view",
            "risks_weaknesses_invalidators",
        ),
        "followups": (
            "What separates the base and bear case?",
            "Which part of the setup is strongest versus weakest?",
            "What change would most improve the thesis?",
        ),
    },
}


def _keyword_score(message: str, keywords: Tuple[str, ...]) -> int:
    score = 0
    for keyword in keywords:
        if keyword in message:
            score += 2 if " " in keyword else 1
    return score


def route_question(message: str) -> Dict[str, Any]:
    text = (message or "").strip().lower()
    if not text:
        return {
            "intent": "signal_summary",
            "answer_mode": "executive_summary",
            "relevant_sections": ["signal_summary", "overall_analysis", "strategy_view"],
            "followup_questions": list(
                _INTENT_DEFINITIONS["signal_summary"]["followups"]
            ),
            "routing_confidence": 0.0,
            "matched_keywords": [],
        }

    scores: List[Tuple[int, str, List[str]]] = []
    for intent, spec in _INTENT_DEFINITIONS.items():
        matches = [keyword for keyword in spec["keywords"] if keyword in text]
        score = _keyword_score(text, spec["keywords"])
        if "why" in text and intent in {"strategy", "compare_clarify"}:
            score += 1
        if "bear case" in text or "bull case" in text or "stress case" in text:
            if intent == "strategy":
                score += 2
        if "invalid" in text or "fragile" in text:
            if intent == "risk_invalidation":
                score += 2
        if any(
            phrase in text
            for phrase in (
                "earnings",
                "catalyst",
                "liquidity",
                "gap risk",
                "implementation",
                "breadth",
                "internals",
                "cross asset",
                "sector support",
                "benchmark support",
                "stress",
                "spillover",
                "contagion",
                "unstable environment",
            )
        ):
            if intent == "market_depth":
                score += 3
        if any(
            phrase in text
            for phrase in (
                "live capital",
                "real money",
                "paper only",
                "paper",
                "deploy",
                "deployment",
                "readiness",
                "trust tier",
                "human review",
                "blocked",
                "paused",
            )
        ):
            if intent == "deployment_readiness":
                score += 3
        if any(
            phrase in text
            for phrase in (
                "portfolio",
                "watchlist",
                "candidate",
                "ranking",
                "rank",
                "redundant",
                "overlap",
                "diversification",
                "size band",
                "weight band",
                "execution quality",
                "friction",
                "turnover",
                "rebalance",
                "rotation",
                "better between",
                "fits better",
            )
        ):
            if intent == "portfolio_construction":
                score += 3
        if any(
            phrase in text
            for phrase in (
                "learning",
                "learn",
                "drift",
                "experiment",
                "archetype",
                "motif",
                "improve",
                "improvement",
                "what is the platform learning",
                "strongest setups",
                "failure mode",
            )
        ):
            if intent == "learning_research":
                score += 3
        scores.append((score, intent, matches))

    scores.sort(key=lambda item: item[0], reverse=True)
    top_score, top_intent, matches = scores[0]
    if top_score <= 0:
        top_intent = "signal_summary" if "signal" in text else "strategy"
        matches = []
        top_score = 1

    spec = _INTENT_DEFINITIONS[top_intent]
    routing_confidence = min(1.0, 0.35 + top_score * 0.12)
    return {
        "intent": top_intent,
        "answer_mode": spec["mode"],
        "relevant_sections": list(spec["sections"]),
        "followup_questions": list(spec["followups"]),
        "routing_confidence": round(routing_confidence, 2),
        "matched_keywords": matches,
    }
