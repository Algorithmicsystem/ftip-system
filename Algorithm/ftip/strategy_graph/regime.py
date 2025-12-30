from __future__ import annotations

import datetime as dt
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from api.main import Candle


RegimeDecision = Tuple[str, Dict[str, float]]


def _pct_change(values: List[float]) -> List[float]:
    if len(values) < 2:
        return [0.0 for _ in values]
    out = [0.0]
    for i in range(1, len(values)):
        prev = values[i - 1]
        cur = values[i]
        out.append(0.0 if prev == 0 else (cur / prev - 1.0))
    return out


def classify_regime(candles: List["Candle"], features: Dict[str, float]) -> RegimeDecision:
    closes = [float(c.close) for c in candles]
    returns = _pct_change(closes)
    vol_ann = float(features.get("volatility_ann", 0.0))
    trend = float(features.get("trend_sma20_50", 0.0))

    drawdown = 0.0
    if closes:
        peak = closes[0]
        max_dd = 0.0
        for c in closes:
            peak = max(peak, c)
            dd = (c / peak - 1.0) if peak else 0.0
            max_dd = min(max_dd, dd)
        drawdown = abs(max_dd)

    vol_fallback = 0.0
    if len(returns) > 2:
        mu = sum(returns[1:]) / max(len(returns) - 1, 1)
        var = sum((r - mu) ** 2 for r in returns[1:]) / max(len(returns) - 2, 1)
        vol_fallback = (var ** 0.5) * (252.0 ** 0.5)

    vol_metric = vol_ann if vol_ann > 0 else vol_fallback
    slope = 0.0
    if len(closes) >= 5:
        window = closes[-5:]
        slope = (window[-1] - window[0]) / max(window[0], 1e-6)

    if drawdown >= 0.18 or vol_metric >= 0.65:
        regime = "RISK_OFF"
    elif vol_metric >= 0.45:
        regime = "HIGH_VOL"
    elif abs(trend) >= 0.05 or abs(slope) >= 0.04:
        regime = "TRENDING"
    else:
        regime = "CHOPPY"

    details = {
        "volatility_ann": vol_ann,
        "volatility_fallback": vol_fallback,
        "drawdown": drawdown,
        "trend_sma20_50": trend,
        "slope_5": slope,
    }
    return regime, details


__all__ = ["classify_regime"]
