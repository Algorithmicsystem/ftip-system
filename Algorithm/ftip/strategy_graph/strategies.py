from __future__ import annotations

import math
from typing import Dict, List, Tuple

StrategyOutput = Dict[str, object]


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, v)))


def _score_to_signal(score: float, thresholds: Tuple[float, float]) -> str:
    buy_thr, sell_thr = thresholds
    if score >= buy_thr:
        return "BUY"
    if score <= sell_thr:
        return "SELL"
    return "HOLD"


def _normalize(score: float, scale: float = 1.0) -> float:
    return _clamp(score / max(scale, 1e-6))


def _confidence(score: float) -> float:
    return float(min(1.0, abs(score)))


def _rationale(*items: str) -> List[str]:
    return [i for i in items if i]


def trend_momentum(features: Dict[str, float]) -> StrategyOutput:
    mom21 = float(features.get("mom_21", 0.0))
    mom63 = float(features.get("mom_63", 0.0))
    trend = float(features.get("trend_sma20_50", 0.0))
    slope = 0.6 * mom21 + 0.4 * mom63
    raw = 0.5 * slope + 0.5 * trend
    normalized = _normalize(raw, 0.25)
    signal = _score_to_signal(normalized, (0.1, -0.1))
    return {
        "strategy_id": "trend_momentum",
        "version": "v1",
        "raw_score": float(raw),
        "normalized_score": float(normalized),
        "signal": signal,
        "confidence": _confidence(normalized),
        "rationale": _rationale(
            "Momentum favors continuation" if normalized > 0 else "Momentum weak/negative",
            "Trend slope positive" if trend > 0 else "Trend slope negative" if trend < 0 else "Trend flat",
        ),
        "feature_contributions": {
            "mom_21": mom21,
            "mom_63": mom63,
            "trend_sma20_50": trend,
        },
    }


def mean_reversion(features: Dict[str, float]) -> StrategyOutput:
    rsi = float(features.get("rsi14", 50.0))
    mom5 = float(features.get("mom_5", 0.0))
    z = _normalize((50.0 - rsi) / 50.0, 1.0)
    raw = -0.6 * mom5 + 0.4 * z
    normalized = _normalize(raw, 0.2)
    signal = _score_to_signal(normalized, (0.15, -0.15))
    return {
        "strategy_id": "mean_reversion",
        "version": "v1",
        "raw_score": float(raw),
        "normalized_score": float(normalized),
        "signal": signal,
        "confidence": _confidence(normalized),
        "rationale": _rationale(
            "RSI stretched to downside" if rsi < 40 else "RSI stretched to upside" if rsi > 60 else "RSI neutral",
            "Short-term momentum down" if mom5 < 0 else "Short-term momentum up" if mom5 > 0 else "Momentum flat",
        ),
        "feature_contributions": {
            "rsi14": rsi,
            "mom_5": mom5,
        },
    }


def volatility_breakout(features: Dict[str, float]) -> StrategyOutput:
    vola = float(features.get("volatility_ann", 0.0))
    range_proxy = abs(float(features.get("mom_5", 0.0)))
    raw = range_proxy + max(0.0, vola - 0.2)
    normalized = _normalize(raw, 0.4)
    signal = _score_to_signal(normalized, (0.2, -0.2))
    return {
        "strategy_id": "volatility_breakout",
        "version": "v1",
        "raw_score": float(raw),
        "normalized_score": float(normalized),
        "signal": signal,
        "confidence": _confidence(normalized) * 0.9,
        "rationale": _rationale(
            "Volatility expanding" if vola > 0.25 else "Volatility muted",
            "Price range expanding" if range_proxy > 0.05 else "Range stable",
        ),
        "feature_contributions": {
            "volatility_ann": vola,
            "mom_5": range_proxy,
        },
    }


def defensive_risk_off(features: Dict[str, float]) -> StrategyOutput:
    vola = float(features.get("volatility_ann", 0.0))
    trend = float(features.get("trend_sma20_50", 0.0))
    drawdown_proxy = max(0.0, -trend)
    raw = -(0.7 * vola + 0.3 * drawdown_proxy)
    normalized = _clamp(raw)
    signal = _score_to_signal(normalized, (0.1, -0.05))
    return {
        "strategy_id": "defensive_risk_off",
        "version": "v1",
        "raw_score": float(raw),
        "normalized_score": float(normalized),
        "signal": signal,
        "confidence": _confidence(normalized),
        "rationale": _rationale(
            "Volatility high" if vola > 0.3 else "Volatility calm",
            "Trend deteriorating" if trend < 0 else "Trend stable/improving",
        ),
        "feature_contributions": {
            "volatility_ann": vola,
            "trend_sma20_50": trend,
        },
    }


def macro_proxy_sentiment(features: Dict[str, float]) -> StrategyOutput:
    sentiment = float(features.get("sentiment", 0.0)) if isinstance(features.get("sentiment"), (int, float)) else 0.0
    mom21 = float(features.get("mom_21", 0.0))
    raw = 0.4 * sentiment + 0.6 * mom21
    normalized = _normalize(raw, 0.25)
    signal = _score_to_signal(normalized, (0.1, -0.1))
    return {
        "strategy_id": "macro_proxy_sentiment",
        "version": "v1",
        "raw_score": float(raw),
        "normalized_score": float(normalized),
        "signal": signal,
        "confidence": _confidence(normalized) * 0.8,
        "rationale": _rationale(
            "Sentiment supportive" if sentiment > 0 else "Sentiment cautious" if sentiment < 0 else "Sentiment neutral",
            "Momentum assists macro view" if mom21 > 0 else "Momentum disagrees with macro" if mom21 < 0 else "Momentum flat",
        ),
        "feature_contributions": {
            "sentiment": sentiment,
            "mom_21": mom21,
        },
    }


def run_strategies(features: Dict[str, float]) -> List[StrategyOutput]:
    funcs = [
        trend_momentum,
        mean_reversion,
        volatility_breakout,
        defensive_risk_off,
        macro_proxy_sentiment,
    ]
    outputs: List[StrategyOutput] = []
    for fn in funcs:
        outputs.append(fn(features))
    return outputs


__all__ = ["run_strategies"]
