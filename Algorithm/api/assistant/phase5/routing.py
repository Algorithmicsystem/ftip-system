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
    "axiom_primary": {
        "keywords": (
            "axiom",
            "deployability tier",
            "trade family",
            "historical evidence",
            "calibration",
            "replay",
            "ic memo",
            "one pager",
            "decision memo",
            "lineage",
            "direct source",
            "derived source",
            "derived signal",
            "weakest evidence",
            "hedge fund",
            "family office",
            "private equity",
            "investment bank",
            "report profile",
            "portfolio fit",
            "size band",
            "rank score",
            "evidence backed",
            "behavioral continuation",
            "fundamental convergence",
            "compensation capture",
            "convexity opportunity",
            "weakest engine",
            "strongest engine",
            "liquidity integrity",
            "research integrity",
            "behavioral distortion",
            "state pricing",
            "flow transmission",
            "critical fragility",
            "why is deployability",
            "why only paper",
        ),
        "mode": "strategist",
        "sections": (
            "axiom_summary",
            "axiom_summary_card",
            "axiom_proprietary_synthesis",
            "axiom_support_vs_drag_summary",
            "axiom_why_now_summary",
            "axiom_unique_mispricing_summary",
            "axiom_setup_character_summary",
            "axiom_false_positive_risk_summary",
            "axiom_decision_hierarchy_summary",
            "axiom_historical_evidence_summary",
            "axiom_historical_evidence_summary_text",
            "axiom_calibration_summary_text",
            "axiom_portfolio_governance_summary",
            "axiom_ic_memo_summary",
            "axiom_risk_deployability_memo_summary",
            "axiom_lineage_summary",
            "signal_summary",
            "strategy_view",
            "fundamental_analysis",
            "risk_quality_analysis",
            "deployment_permission_analysis",
        ),
        "followups": (
            "Which AXIOM engine is strongest right now?",
            "Which AXIOM engine is weakest and why?",
            "What is stopping this from being a live candidate?",
            "What historical evidence supports the current AXIOM tier?",
            "What direct sources versus derived signals are driving the current AXIOM view?",
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
    "source_governance": {
        "keywords": (
            "commercial",
            "commercialization",
            "commercially safe",
            "buyer demo",
            "buyer-demo",
            "buyer-safe",
            "buyer safe",
            "mixed-risk",
            "mixed risk",
            "source governance",
            "source profile",
            "source stack",
            "clean stack",
            "commercial stack",
            "licensing",
            "license",
            "data rights",
            "data source",
            "data sources",
            "which feeds",
            "production profile",
            "restricted cleanroom",
        ),
        "mode": "commercialization",
        "sections": (
            "commercialization_readiness_summary",
            "source_governance_summary",
            "data_provider_quality_summary",
            "buyer_diligence_summary",
            "evidence_provenance",
        ),
        "followups": (
            "Which sources are safe only for internal research versus buyer-facing use?",
            "What breaks if the cleaner commercial profile is enforced?",
            "What is the current clean-stack path for commercialization?",
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
            "walk forward",
            "walkforward",
            "net of friction",
            "friction",
            "gross vs net",
            "mae",
            "mfe",
            "drawdown",
            "invalidation",
        ),
        "mode": "evidence",
        "sections": (
            "evaluation_research_analysis",
            "canonical_validation_summary",
            "walkforward_validation_summary",
            "net_of_friction_validation_summary",
            "suppression_readiness_validation_summary",
            "drawdown_invalidation_validation_summary",
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
            "hidden overlap",
            "hidden risk",
            "factor exposure",
            "factor",
            "covariance",
            "correlation",
            "cluster",
            "cluster risk",
            "diversification",
            "replacement",
            "replace",
            "substitute",
            "alternative",
            "stacking",
            "stacked",
            "stress overlap",
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
            "portfolio_risk_model_summary",
            "hidden_overlap_redundancy_analysis",
            "factor_exposure_summary",
            "concentration_cluster_risk_analysis",
            "replacement_diversification_analysis",
            "portfolio_stress_fragility_summary",
            "strategy_view",
        ),
        "followups": (
            "Which idea fits the portfolio better right now?",
            "Why is this a watchlist candidate instead of a deployable one?",
            "What is creating redundancy or concentration here?",
            "Which name is strongest but most redundant?",
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
    "operational_health": {
        "keywords": (
            "operational",
            "operations",
            "system health",
            "healthy right now",
            "provider",
            "providers",
            "provider stale",
            "stale provider",
            "fallback",
            "freshness alert",
            "shadow mode",
            "shadow",
            "downgraded",
            "downgrade",
            "kill switch",
            "kill-switch",
            "pause",
            "paused",
            "incident",
            "incident history",
            "degradation",
            "drifting",
            "drift",
            "reliability concern",
            "recover",
            "recovery criteria",
        ),
        "mode": "operations",
        "sections": (
            "system_health_summary",
            "shadow_mode_summary",
            "drift_control_summary",
            "incident_history_summary",
            "deployment_readiness_summary",
            "rollout_stage_summary",
            "evidence_provenance",
        ),
        "followups": (
            "Is the system healthy enough to trust right now?",
            "Why is the platform in shadow mode or a downgraded state?",
            "What would need to recover before trust rises again?",
        ),
    },
    "operator_workflow": {
        "keywords": (
            "what changed today",
            "changed today",
            "what changed",
            "review first",
            "review now",
            "daily review",
            "weekly review",
            "monthly review",
            "monthly refinement",
            "operator workflow",
            "operator runbook",
            "runbook",
            "shadow journal",
            "shadow decision",
            "postmortem",
            "post-mortem",
            "what failed recently",
            "upgraded",
            "downgraded",
            "promotion candidate",
            "demotion candidate",
            "trust maintenance",
            "improvement work",
            "attention items",
            "priority candidates today",
            "what should i review first",
            "dossier",
            "workflow stage",
            "workflow template",
            "workspace",
            "platform summary",
            "case file",
            "approval",
            "approve",
            "reject",
            "request changes",
            "workflow actions",
            "lock recommendation",
            "unlock recommendation",
            "export pack",
            "audit trail",
            "timeline",
            "access role",
            "permissions",
            "integrations",
            "connector health",
            "dashboard",
            "analytics",
            "demo snapshot",
            "pilot readiness",
            "export preview",
            "rendered export",
            "export format",
            "webhook",
            "local archive",
        ),
        "mode": "operator",
        "sections": (
            "platform_overview_summary",
            "platform_dossier_summary",
            "platform_monitoring_summary",
            "platform_access_control_summary",
            "platform_workflow_actions_summary",
            "platform_audit_timeline_summary",
            "platform_export_summary",
            "platform_export_rendering_summary",
            "platform_integration_health_summary",
            "platform_dashboard_summary",
            "platform_analytics_summary",
            "platform_demo_readiness_summary",
            "daily_operating_summary",
            "weekly_operating_summary",
            "monthly_operating_summary",
            "shadow_journal_summary",
            "postmortem_summary",
            "trust_maintenance_summary",
            "operator_runbook_summary",
            "system_health_summary",
            "portfolio_workflow_summary",
        ),
        "followups": (
            "What changed today that deserves review first?",
            "What belongs in this week’s operating review?",
            "What should be prioritized for the next monthly refinement cycle?",
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
                "buyer demo",
                "buyer safe",
                "source profile",
                "source governance",
                "commercial stack",
                "clean stack",
                "licensing",
                "data rights",
            )
        ):
            if intent == "source_governance":
                score += 3
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
                "axiom",
                "deployability tier",
                "trade family",
                "weakest engine",
                "strongest engine",
                "paper trade only",
                "behavioral continuation",
                "fundamental convergence",
                "compensation capture",
                "convexity opportunity",
                "state pricing",
                "behavioral distortion",
                "flow transmission",
                "liquidity integrity",
                "research integrity",
                "critical fragility",
                "why is deployability",
                "why only paper",
            )
        ):
            if intent == "axiom_primary":
                score += 4
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
                "system health",
                "shadow mode",
                "provider stale",
                "downgraded",
                "downgrade",
                "kill switch",
                "kill-switch",
                "incident history",
                "reliability concern",
            )
        ):
            if intent == "operational_health":
                score += 3
        if any(
            phrase in text
            for phrase in (
                "what changed today",
                "what changed",
                "review first",
                "daily review",
                "weekly review",
                "monthly review",
                "monthly refinement",
                "operator runbook",
                "runbook",
                "shadow journal",
                "postmortem",
                "post-mortem",
                "what failed recently",
                "trust maintenance",
                "promotion candidate",
                "demotion candidate",
                "improvement work",
                "attention items",
            )
        ):
            if intent == "operator_workflow":
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
