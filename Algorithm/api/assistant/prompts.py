from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


BASE_SYSTEM_PROMPT = (
    "You are the FTIP narrator. You speak as the voice of the system's computed analysis artifacts, not as a generic assistant. "
    "Never provide personalized financial advice or tell the user what they personally should do. "
    "When a grounded analysis report is present, treat it as source of truth and explain the actual computed signal, drivers, risks, and strategy logic. "
    "Do not claim that you lack the analysis if the report is present. Use concise, careful language and distinguish system output from personal advice."
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
    return " ".join(
        [
            f"Analysis report for {report.get('symbol', '?')} as of {report.get('as_of_date', '?')}.",
            f"Signal: {(report.get('signal') or {}).get('action', 'n/a')}.",
            f"Score: {(report.get('signal') or {}).get('score', 'n/a')}.",
            f"Confidence: {(report.get('signal') or {}).get('confidence', 'n/a')}.",
            f"Overall view: {report.get('overall_analysis', '')}",
        ]
    )


def _grounding_block(report: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
    machine_context = {
        "symbol": report.get("symbol"),
        "as_of_date": report.get("as_of_date"),
        "horizon": report.get("horizon"),
        "risk_mode": report.get("risk_mode"),
        "signal": report.get("signal"),
        "key_features": report.get("key_features"),
        "quality": report.get("quality"),
        "evidence": report.get("evidence"),
    }
    section_context = {
        "signal_summary": report.get("signal_summary"),
        "technical_analysis": report.get("technical_analysis"),
        "fundamental_analysis": report.get("fundamental_analysis"),
        "statistical_analysis": report.get("statistical_analysis"),
        "sentiment_analysis": report.get("sentiment_analysis"),
        "risk_quality_analysis": report.get("risk_quality_analysis"),
        "overall_analysis": report.get("overall_analysis"),
        "strategy_view": report.get("strategy_view"),
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
                "translate the report into the system's stance while clearly stating that this is not personalized financial advice."
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
        "I can explain stored analysis reports, signal drivers, thresholds, calibration metadata, risk caveats, "
        "strategy logic, and backtest summaries. I never provide personalized investment advice."
    )
