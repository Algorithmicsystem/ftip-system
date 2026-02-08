from __future__ import annotations

from typing import Dict, List

StrategyOutput = Dict[str, object]


def _weights_by_regime(regime: str) -> Dict[str, float]:
    base = {
        "trend_momentum": 0.28,
        "mean_reversion": 0.18,
        "volatility_breakout": 0.16,
        "defensive_risk_off": 0.20,
        "macro_proxy_sentiment": 0.18,
    }
    if regime == "TRENDING":
        base["trend_momentum"] += 0.07
        base["macro_proxy_sentiment"] += 0.02
    elif regime == "CHOPPY":
        base["mean_reversion"] += 0.07
    elif regime == "HIGH_VOL":
        base["defensive_risk_off"] += 0.08
    elif regime == "RISK_OFF":
        base["defensive_risk_off"] += 0.12
    total = sum(base.values()) or 1.0
    return {k: float(v / total) for k, v in base.items()}


def _thresholds_for_regime(regime: str) -> Dict[str, float]:
    defaults = {
        "TRENDING": {"buy": 0.18, "sell": -0.18},
        "CHOPPY": {"buy": 0.25, "sell": -0.25},
        "HIGH_VOL": {"buy": 0.32, "sell": -0.32},
        "RISK_OFF": {"buy": 0.35, "sell": -0.22},
    }
    return defaults.get(regime, {"buy": 0.2, "sell": -0.2})


def _disagreement_penalty(signals: List[str]) -> float:
    if not signals:
        return 1.0
    buys = sum(1 for s in signals if s == "BUY")
    sells = sum(1 for s in signals if s == "SELL")
    if buys and sells:
        return 0.6
    if buys + sells < len(signals):
        return 0.85
    return 1.0


def combine(regime: str, strategies: List[StrategyOutput]) -> Dict[str, object]:
    weights = _weights_by_regime(regime)
    weighted_score = 0.0
    total_w = 0.0
    signals: List[str] = []

    for strat in strategies:
        sid = str(strat.get("strategy_id"))
        w = float(weights.get(sid, 0.0))
        score = float(strat.get("normalized_score", 0.0))
        weighted_score += w * score
        total_w += w
        signals.append(str(strat.get("signal")))

    total_w = total_w or 1.0
    final_score = weighted_score / total_w
    thresholds = _thresholds_for_regime(regime)
    final_signal = "HOLD"
    if final_score >= thresholds["buy"]:
        final_signal = "BUY"
    elif final_score <= thresholds["sell"]:
        final_signal = "SELL"

    base_conf = min(1.0, abs(final_score))
    conf = base_conf * _disagreement_penalty(signals)

    risk_overlay_applied = False
    if regime in {"RISK_OFF", "HIGH_VOL"}:
        risk_overlay_applied = True
        if final_signal == "BUY" and conf > 0.65:
            conf = 0.65
        conf *= 0.9
        if final_signal == "BUY" and final_score < thresholds["buy"] * 1.1:
            final_signal = "HOLD"

    return {
        "ensemble_method": "WEIGHTED_VOTE+RISK_OVERLAY+SHRINK",  # deterministic descriptor
        "final_signal": final_signal,
        "final_score": float(final_score),
        "final_confidence": float(conf),
        "thresholds": thresholds,
        "risk_overlay_applied": risk_overlay_applied,
        "strategies_used": [
            {"strategy_id": k, "weight": v} for k, v in weights.items()
        ],
    }


__all__ = ["combine"]
