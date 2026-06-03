"""FACTOR_REGIME_MATRIX — empirically and theoretically grounded factor weights per regime."""
from __future__ import annotations
from typing import Dict, List
from api.assistant.phase3.common import clamp
from api.axiom.factors.factor_model import FactorLoading

_ALL_FACTORS = ["EIF", "CMF", "BAF", "KLF", "SCAF", "ICF", "GBF", "MTRF", "MQF", "VIF", "RTF", "NTFF"]
_EQUAL_WEIGHT = 1.0 / 12.0


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights to sum to exactly 1.0."""
    total = sum(weights.values())
    if total == 0:
        return {k: _EQUAL_WEIGHT for k in weights}
    return {k: v / total for k, v in weights.items()}


# Raw weights (before normalization) — based on theoretical and empirical grounding
_FACTOR_REGIME_MATRIX_RAW: Dict[str, Dict[str, float]] = {
    "TRENDING": {
        "MQF": 0.20, "RTF": 0.15, "EIF": 0.12, "BAF": 0.12,
        "ICF": 0.10, "CMF": 0.08, "KLF": 0.08, "SCAF": 0.08,
        "GBF": 0.07, "NTFF": 0.05, "VIF": 0.03, "MTRF": 0.02,
    },
    "CHOPPY": {
        "VIF": 0.20, "CMF": 0.18, "EIF": 0.15, "KLF": 0.12,
        "NTFF": 0.08, "BAF": 0.08, "RTF": 0.07, "ICF": 0.05,
        "SCAF": 0.02, "MQF": 0.02, "GBF": 0.02, "MTRF": 0.01,
    },
    "HIGH_VOL": {
        "KLF": 0.25, "MTRF": 0.20, "SCAF": 0.15, "BAF": 0.12,
        "EIF": 0.10, "RTF": 0.08, "CMF": 0.05, "NTFF": 0.02,
        "ICF": 0.01, "VIF": 0.01, "MQF": 0.01, "GBF": 0.00,
    },
    "RECOVERY": {
        "VIF": 0.25, "EIF": 0.20, "CMF": 0.15, "MQF": 0.12,
        "BAF": 0.08, "RTF": 0.08, "ICF": 0.05, "KLF": 0.04,
        "NTFF": 0.02, "SCAF": 0.01, "GBF": 0.00, "MTRF": 0.00,
    },
}

# Normalized matrix — each regime sums to exactly 1.0
FACTOR_REGIME_MATRIX: Dict[str, Dict[str, float]] = {
    regime: _normalize(weights)
    for regime, weights in _FACTOR_REGIME_MATRIX_RAW.items()
}


def get_regime_factor_weights(regime_label: str) -> Dict[str, float]:
    """Return weight dict for a regime. Falls back to equal weights if unknown."""
    regime_upper = str(regime_label or "").upper()
    if regime_upper in FACTOR_REGIME_MATRIX:
        return FACTOR_REGIME_MATRIX[regime_upper]
    # Unknown regime: equal weights
    return {f: _EQUAL_WEIGHT for f in _ALL_FACTORS}


def compute_factor_composite_score(
    factor_loadings: List[FactorLoading],
    regime_label: str,
) -> float:
    """Factor Composite Score (FCS) 0-100: regime-weighted factor model aggregate."""
    weights = get_regime_factor_weights(regime_label)
    if not factor_loadings:
        return 50.0

    weighted_sum = sum(
        fl.loading * weights.get(fl.factor_name, _EQUAL_WEIGHT)
        for fl in factor_loadings
    )
    # center at 50, scale: max weighted_sum ≈ 1.0 → score ≈ 100
    fcs = clamp(weighted_sum * 50.0 + 50.0, 0.0, 100.0)
    return round(fcs, 2)
