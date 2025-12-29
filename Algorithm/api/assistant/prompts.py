from __future__ import annotations

from typing import Any, Dict, List, Optional


BASE_SYSTEM_PROMPT = (
    "You are the FTIP narrator. You summarize how the system computes signals and backtests. "
    "Never provide financial advice or trading recommendations. Use concise, careful language. "
    "If you do not have enough context, ask clarifying questions instead of guessing. "
    "Always explain which inputs, thresholds, and calibration metadata influenced a signal."
)


def build_chat_messages(history: List[Dict[str, str]], user_message: str, context: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
    messages.extend(history)

    if context:
        ctx_bits = [f"{k}={v}" for k, v in context.items()]
        context_line = "Context: " + ", ".join(ctx_bits)
        user_payload = f"{user_message}\n\n{context_line}"
    else:
        user_payload = user_message

    messages.append({"role": "user", "content": user_payload})
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
        "I can explain how signals were computed, what thresholds and calibration metadata were used, "
        "and summarize backtest results. I never provide investment advice."
    )
