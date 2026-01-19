from __future__ import annotations

from typing import Dict, List, Optional


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_daily_signal(
    features: Dict[str, Optional[float]],
    quality_score: int,
    latest_close: Optional[float],
) -> Dict[str, object]:
    trend_slope = features.get("trend_slope_63d") or 0.0
    mom = features.get("mom_vol_adj_21d") or 0.0
    sentiment = features.get("sentiment_score") or 0.0
    maxdd = features.get("maxdd_63d") or 0.0
    atr_pct = features.get("atr_pct") or 0.0
    vol_63d = features.get("vol_63d") or 0.0

    score = 0.5 * trend_slope + 0.3 * mom + 0.2 * sentiment
    score -= 0.2 * abs(maxdd)
    score = _clamp(score, -1.0, 1.0)

    action = "HOLD"
    if score >= 0.3:
        action = "BUY"
    elif score <= -0.3:
        action = "SELL"

    base_conf = min(1.0, abs(score))
    quality_factor = _clamp(quality_score / 100.0, 0.1, 1.0)
    risk_factor = 1.0
    if vol_63d and vol_63d > 0.6:
        risk_factor = 0.7
    confidence = _clamp(base_conf * quality_factor * risk_factor, 0.0, 1.0)

    reason_codes: List[str] = []
    if trend_slope > 0:
        reason_codes.append("TREND_UP")
    if trend_slope < 0:
        reason_codes.append("TREND_DOWN")
    if mom > 0.2:
        reason_codes.append("MOMENTUM_STRONG")
    if mom < -0.2:
        reason_codes.append("MOMENTUM_WEAK")
    if sentiment > 0.1:
        reason_codes.append("SENTIMENT_POSITIVE")
    if sentiment < -0.1:
        reason_codes.append("SENTIMENT_NEGATIVE")
    if not reason_codes:
        reason_codes.append("NEUTRAL_SCORE")

    entry_low = None
    entry_high = None
    stop_loss = None
    take_profit_1 = None
    take_profit_2 = None
    if latest_close:
        entry_low = latest_close * (1 - 0.5 * atr_pct)
        entry_high = latest_close * (1 + 0.5 * atr_pct)
        if action == "BUY":
            stop_loss = latest_close * (1 - 1.5 * atr_pct)
            take_profit_1 = latest_close * (1 + 2 * atr_pct)
            take_profit_2 = latest_close * (1 + 3 * atr_pct)
        elif action == "SELL":
            stop_loss = latest_close * (1 + 1.5 * atr_pct)
            take_profit_1 = latest_close * (1 - 2 * atr_pct)
            take_profit_2 = latest_close * (1 - 3 * atr_pct)

    return {
        "action": action,
        "score": float(score),
        "confidence": float(confidence),
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
        "reason_codes": reason_codes,
    }
