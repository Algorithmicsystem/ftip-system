from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence


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


def compute_return_correlation_matrix(
    returns_by_symbol: Mapping[str, Sequence[float]],
) -> Dict[str, Dict[str, float]]:
    """Pairwise Pearson correlations from per-symbol daily return streams.

    Uses the last min(len_a, len_b) observations so mismatched-length series
    are handled without error.  Returns a symmetric matrix; diagonal = 1.0.
    Missing or flat series get correlation 0.0.
    """
    symbols = sorted(returns_by_symbol.keys())
    matrix: Dict[str, Dict[str, float]] = {s: {} for s in symbols}

    for i, a in enumerate(symbols):
        matrix[a][a] = 1.0
        ra = list(returns_by_symbol[a])
        for j in range(i + 1, len(symbols)):
            b = symbols[j]
            rb = list(returns_by_symbol[b])
            n = min(len(ra), len(rb))
            if n < 2:
                corr = 0.0
            else:
                x = ra[-n:]
                y = rb[-n:]
                mean_x = sum(x) / n
                mean_y = sum(y) / n
                cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
                std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
                std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)
                if std_x < 1e-10 or std_y < 1e-10:
                    corr = 0.0
                else:
                    corr = max(-1.0, min(1.0, cov / (std_x * std_y)))
            matrix[a][b] = corr
            matrix[b][a] = corr

    return matrix


def correlation_guard(
    weights: Mapping[str, float],
    correlation_matrix: Optional[Mapping[str, Mapping[str, float]]] = None,
    threshold: float = 0.85,
    max_iterations: int = 20,
) -> Dict[str, float]:
    """Reduce weights of highly-correlated position pairs.

    For each iteration, finds the pair with the highest absolute correlation
    above `threshold` and applies a proportional haircut to the smaller-weight
    position.  Continues until no pair exceeds the threshold or `max_iterations`
    is reached, then re-normalises.

    When `correlation_matrix` is None the function is a pass-through so callers
    that haven't yet wired up correlation data are unaffected.
    """
    result = {s: float(w) for s, w in weights.items() if float(w) > 0}
    if not result or correlation_matrix is None:
        return {s: float(w) for s, w in sorted(weights.items())}

    for _ in range(max_iterations):
        syms = sorted(result.keys())
        worst_pair: Optional[tuple] = None
        worst_corr = float(threshold)

        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                a, b = syms[i], syms[j]
                row = correlation_matrix.get(a) or {}
                corr = abs(float(row.get(b, 0.0)))
                if corr > worst_corr:
                    worst_corr = corr
                    worst_pair = (a, b)

        if worst_pair is None:
            break

        a, b = worst_pair
        # Proportional haircut: how far above threshold the correlation sits.
        haircut = (worst_corr - threshold) / max(1.0 - threshold, 1e-8)
        haircut = min(max(haircut, 0.0), 1.0)

        # Apply haircut to the lower-weight leg.
        if result[a] <= result[b]:
            result[a] *= 1.0 - haircut
        else:
            result[b] *= 1.0 - haircut

        result = {s: w for s, w in result.items() if w > 1e-10}
        if not result:
            return {}

    total = sum(result.values())
    if total <= 0.0:
        return {}
    return {s: w / total for s, w in sorted(result.items())}


# Backward-compatible alias — existing callers that use the stub name still work.
# With correlation_matrix=None the behaviour is identical to the old stub.
def correlation_guard_stub(
    weights: Mapping[str, float],
    correlation_matrix: Optional[Mapping[str, Mapping[str, float]]] = None,
    threshold: float = 0.85,
) -> Dict[str, float]:
    return correlation_guard(weights, correlation_matrix=correlation_matrix, threshold=threshold)
