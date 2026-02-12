from __future__ import annotations

from typing import Dict, Mapping, Optional


def _normalize_long_only(weights: Mapping[str, float]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    total = 0.0
    for symbol, weight in sorted(weights.items()):
        w = float(weight)
        if w <= 0.0:
            continue
        cleaned[symbol] = w
        total += w
    if total <= 0.0:
        return {}
    return {symbol: (weight / total) for symbol, weight in cleaned.items()}


def apply_exposure_caps(
    weights: Mapping[str, float], max_weight: float
) -> Dict[str, float]:
    capped: Dict[str, float] = {}
    cap = max(0.0, float(max_weight))
    for symbol, weight in sorted(weights.items()):
        capped[symbol] = min(max(float(weight), 0.0), cap)
    normalized = _normalize_long_only(capped)
    if not normalized:
        return {}

    # Re-cap after normalization and redistribute until stable.
    result = dict(normalized)
    for _ in range(8):
        over = {k: v for k, v in result.items() if v > cap}
        if not over:
            break
        fixed_total = sum(min(v, cap) for v in result.values())
        free_symbols = [k for k in result.keys() if k not in over]
        free_total = sum(result[k] for k in free_symbols)
        next_result: Dict[str, float] = {}
        for symbol in sorted(result.keys()):
            if symbol in over:
                next_result[symbol] = cap
            elif free_total > 0.0:
                remaining = max(0.0, 1.0 - fixed_total)
                next_result[symbol] = (result[symbol] / free_total) * remaining
            else:
                next_result[symbol] = 0.0
        result = _normalize_long_only(next_result)
        if not result:
            return {}

    return {symbol: min(weight, cap) for symbol, weight in sorted(result.items())}


def volatility_targeting(
    weights: Mapping[str, float],
    vol_estimates: Mapping[str, float],
    target_vol: float,
    max_weight: float,
) -> Dict[str, float]:
    target = max(float(target_vol), 1e-8)
    scaled: Dict[str, float] = {}
    for symbol, base_weight in sorted(weights.items()):
        raw = max(float(base_weight), 0.0)
        vol = float(vol_estimates.get(symbol, 0.0))
        effective_vol = max(vol, 1e-8)
        scale = min(1.0, target / effective_vol)
        scaled[symbol] = raw * scale
    return apply_exposure_caps(scaled, max_weight=max_weight)


def correlation_guard_stub(
    weights: Mapping[str, float],
    correlation_matrix: Optional[Mapping[str, Mapping[str, float]]] = None,
    threshold: float = 0.9,
) -> Dict[str, float]:
    _ = correlation_matrix
    _ = threshold
    return {symbol: float(weight) for symbol, weight in sorted(weights.items())}
