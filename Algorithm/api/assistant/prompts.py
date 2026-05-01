from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


BASE_SYSTEM_PROMPT = (
    "You are the FTIP narrator. You speak as the voice of the system's computed analysis artifacts, not as a generic assistant. "
    "Never provide personalized financial advice or tell the user what they personally should do. "
    "When a grounded analysis report is present, treat it as source of truth and explain the actual computed signal, drivers, risks, and strategy logic. "
    "Do not claim that you lack the analysis if the report is present. Use concise, careful language, cite the actual report sections when helpful, and distinguish system output from personal advice."
)


def build_chat_messages(
    history: List[Dict[str, str]], user_message: str, context: Optional[Dict[str, Any]]
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
    messages.extend(history[-12:])

    if context:
        context_line = "Context:\n" + json.dumps(context, indent=2, default=str, sort_keys=True)
        user_payload = f"{user_message}\n\n{context_line}"
    else:
        user_payload = user_message

    messages.append({"role": "user", "content": user_payload})
    return messages


def summarize_analysis_report(report: Dict[str, Any]) -> str:
    strategy = report.get("strategy") or {}
    return " ".join(
        [
            f"Analysis report for {report.get('symbol', '?')} as of {report.get('as_of_date', '?')}.",
            f"Signal: {(report.get('signal') or {}).get('action', 'n/a')} -> {strategy.get('final_signal', 'n/a')}.",
            f"Score: {(report.get('signal') or {}).get('score', 'n/a')}.",
            f"Confidence: {strategy.get('confidence', (report.get('signal') or {}).get('confidence', 'n/a'))}.",
            f"Deployment permission: {report.get('deployment_permission', 'n/a')} under {report.get('deployment_mode', 'research_only')}.",
            f"Trust tier: {report.get('trust_tier', 'unknown')} with live readiness {report.get('live_readiness_score', 'n/a')}.",
            f"Portfolio classification: {report.get('candidate_classification', 'n/a')} with portfolio score {report.get('portfolio_candidate_score', 'n/a')}.",
            f"Learning archetype: {(report.get('setup_archetype') or {}).get('archetype_name', 'n/a')} with research priority {report.get('learning_priority', 'n/a')}.",
            f"Overall view: {report.get('overall_analysis', '')}",
        ]
    )


def _grounding_block(report: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
    machine_context = {
        "symbol": report.get("symbol"),
        "as_of_date": report.get("as_of_date"),
        "horizon": report.get("horizon"),
        "risk_mode": report.get("risk_mode"),
        "scenario": report.get("scenario"),
        "analysis_depth": report.get("analysis_depth"),
        "refresh_mode": report.get("refresh_mode"),
        "market_regime": report.get("market_regime"),
        "freshness_summary": report.get("freshness_summary"),
        "signal": report.get("signal"),
        "strategy": report.get("strategy"),
        "why_this_signal": report.get("why_this_signal"),
        "key_features": report.get("key_features"),
        "quality": report.get("quality"),
        "evidence": report.get("evidence"),
        "evidence_map": report.get("evidence_map"),
        "deployment_readiness": report.get("deployment_readiness"),
        "portfolio_construction": report.get("portfolio_construction"),
        "continuous_learning": report.get("continuous_learning"),
    }
    section_context = {
        "signal_summary": report.get("signal_summary"),
        "technical_analysis": report.get("technical_analysis"),
        "fundamental_analysis": report.get("fundamental_analysis"),
        "statistical_analysis": report.get("statistical_analysis"),
        "sentiment_analysis": report.get("sentiment_analysis"),
        "macro_geopolitical_analysis": report.get("macro_geopolitical_analysis"),
        "risk_quality_analysis": report.get("risk_quality_analysis"),
        "overall_analysis": report.get("overall_analysis"),
        "strategy_view": report.get("strategy_view"),
        "risks_weaknesses_invalidators": report.get("risks_weaknesses_invalidators"),
        "evidence_provenance": report.get("evidence_provenance"),
        "deployment_readiness_summary": report.get("deployment_readiness_summary"),
        "deployment_permission_analysis": report.get("deployment_permission_analysis"),
        "risk_budget_exposure_analysis": report.get("risk_budget_exposure_analysis"),
        "rollout_stage_summary": report.get("rollout_stage_summary"),
        "portfolio_context_summary": report.get("portfolio_context_summary"),
        "portfolio_fit_analysis": report.get("portfolio_fit_analysis"),
        "execution_quality_analysis": report.get("execution_quality_analysis"),
        "portfolio_workflow_summary": report.get("portfolio_workflow_summary"),
        "learning_summary": report.get("learning_summary"),
        "regime_learning_summary": report.get("regime_learning_summary"),
        "adaptation_queue_summary": report.get("adaptation_queue_summary"),
        "experiment_registry_summary": report.get("experiment_registry_summary"),
        "archetype_motif_summary": report.get("archetype_motif_summary"),
    }
    blocks = [
        "Grounding report metadata and machine-readable fields:",
        json.dumps(machine_context, indent=2, default=str, sort_keys=True),
        "Presentation-ready report sections:",
        json.dumps(section_context, indent=2, default=str, sort_keys=True),
    ]
    if context:
        blocks.extend(
            [
                "Caller context:",
                json.dumps(context, indent=2, default=str, sort_keys=True),
            ]
        )
    return "\n\n".join(blocks)


def build_grounded_chat_messages(
    history: List[Dict[str, str]],
    user_message: str,
    report: Dict[str, Any],
    context: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {
            "role": "system",
            "content": (
                "You are answering from a stored FTIP analysis report. Explain the system signal, drivers, strengths, weaknesses, "
                "risk/quality caveats, and strategy logic implied by that report. If asked whether the user should buy or sell, "
                "translate the report into the system's stance while clearly stating that this is not personalized financial advice. "
                "Prefer referencing the report's signal summary, overall analysis, strategy view, risks/weaknesses/invalidators, and evidence provenance sections."
            ),
        },
        {"role": "system", "content": _grounding_block(report, context)},
    ]
    messages.extend(history[-12:])
    messages.append({"role": "user", "content": user_message})
    return messages


def summarize_signal(payload: Dict[str, Any]) -> str:
    symbol = payload.get("symbol", "?")
    as_of = payload.get("as_of", "?")
    lookback = payload.get("lookback", "?")
    signal = payload.get("signal", "?")
    score = payload.get("score")
    confidence = payload.get("confidence")
    thresholds = payload.get("thresholds") or {}
    calibration_meta = payload.get("calibration_meta") or {}
    notes = payload.get("notes") or []

    parts = [
        f"Signal for {symbol} as of {as_of} with lookback {lookback} is {signal}.",
    ]
    if score is not None:
        parts.append(f"Score: {score}.")
    if confidence is not None:
        parts.append(f"Confidence: {confidence}.")
    if thresholds:
        threshold_text = ", ".join(f"{k}={v}" for k, v in thresholds.items())
        parts.append(f"Thresholds used: {threshold_text}.")
    if calibration_meta:
        parts.append("Calibration metadata was applied.")
    if notes:
        parts.append("Notes: " + "; ".join(notes))
    return " ".join(parts)


def summarize_backtest(payload: Dict[str, Any]) -> str:
    total_return = payload.get("total_return")
    sharpe = payload.get("sharpe")
    max_drawdown = payload.get("max_drawdown")
    volatility = payload.get("volatility")
    window = payload.get("lookback") or payload.get("rebalance_every")

    parts = ["Backtest summary:"]
    if total_return is not None:
        parts.append(f"total_return={total_return}")
    if sharpe is not None:
        parts.append(f"sharpe={sharpe}")
    if max_drawdown is not None:
        parts.append(f"max_drawdown={max_drawdown}")
    if volatility is not None:
        parts.append(f"volatility={volatility}")
    if window is not None:
        parts.append(f"lookback={window}")
    return ", ".join(parts)


def system_capabilities() -> str:
    return (
        "I can explain stored analysis reports, canonical strategy artifacts, why-this-signal drilldowns, evidence provenance, thresholds, "
        "risk caveats, scenario framing, and backtest summaries. I never provide personalized investment advice."
    )
