"""Phase 11.3: Dynamic Correlation Monitor.

Computes rolling correlation matrices, detects correlation regime shifts,
and identifies crisis-level correlation spikes.
"""
from __future__ import annotations

import math
from typing import Dict, List

from api.assistant.phase3.common import clamp


def _pearson_corr(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a, b = a[:n], b[:n]
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    std_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    std_b = math.sqrt(sum((x - mean_b) ** 2 for x in b))
    if std_a < 1e-12 or std_b < 1e-12:
        return 0.0
    return cov / (std_a * std_b)


def compute_rolling_correlation_matrix(
    symbol_returns: Dict[str, List[float]],
    window: int = 21,
) -> Dict:
    """Compute pairwise Pearson correlation matrix over a rolling window.

    Correlation regime:
        avg < 0.30 → "normal"
        avg 0.30-0.60 → "elevated"
        avg > 0.60 → "crisis"
    """
    symbols = list(symbol_returns.keys())
    matrix: Dict[str, Dict[str, float]] = {s: {} for s in symbols}
    pairwise: List[float] = []

    for i, s1 in enumerate(symbols):
        r1 = symbol_returns[s1][-window:] if len(symbol_returns[s1]) >= window else symbol_returns[s1]
        for j, s2 in enumerate(symbols):
            if i == j:
                matrix[s1][s2] = 1.0
                continue
            r2 = symbol_returns[s2][-window:] if len(symbol_returns[s2]) >= window else symbol_returns[s2]
            corr = _pearson_corr(r1, r2)
            matrix[s1][s2] = round(corr, 6)
            if i < j:
                pairwise.append(corr)

    avg_pairwise = sum(pairwise) / len(pairwise) if pairwise else 0.0
    max_pairwise = max(pairwise) if pairwise else 0.0

    if avg_pairwise > 0.60:
        regime = "crisis"
    elif avg_pairwise > 0.30:
        regime = "elevated"
    else:
        regime = "normal"

    return {
        "correlation_matrix": matrix,
        "avg_pairwise_correlation": round(avg_pairwise, 4),
        "max_pairwise_correlation": round(max_pairwise, 4),
        "correlation_regime": regime,
    }


def compute_correlation_regime_score(
    avg_correlation: float,
    trend_21d: float,
) -> float:
    """Correlation Regime Score (CRS): 0–100, higher = more crisis-like."""
    crs = avg_correlation * 100.0 * (1.0 + trend_21d * 2.0)
    return round(clamp(crs, 0.0, 100.0), 2)


def detect_correlation_spike(
    current_avg: float,
    historical_avg: float,
    threshold_multiplier: float = 1.5,
) -> Dict:
    """Detect a correlation spike when current exceeds historical × threshold."""
    spike_detected = current_avg > historical_avg * threshold_multiplier

    if historical_avg > 1e-10:
        ratio = current_avg / historical_avg
    else:
        ratio = 0.0

    if ratio >= 3.0:
        severity = "severe"
    elif ratio >= 2.0:
        severity = "moderate"
    else:
        severity = "minor"

    return {
        "spike_detected": spike_detected,
        "current_avg": round(current_avg, 4),
        "historical_avg": round(historical_avg, 4),
        "severity": severity,
    }
