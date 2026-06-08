"""AXIOM Formula Registry — canonical parameter catalogue for IP audit and due diligence."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

FORMULA_REGISTRY: Dict[str, Dict[str, Any]] = {
    "signal_war": {
        "description": "Win-Adjusted Return: composite signal return normalised by batting average and max drawdown.",
        "parameters": {
            "return_weight": 0.50,
            "batting_average_weight": 0.30,
            "max_drawdown_penalty": 0.20,
        },
        "formula": "WAR = (avg_return * return_weight) + (batting_avg * batting_average_weight) - (max_dd * max_drawdown_penalty)",
        "version": "3.1",
        "category": "signal_scoring",
    },
    "kelly_sizing": {
        "description": "Fractional Kelly position sizing with AXIOM-specific risk dampener.",
        "parameters": {
            "kelly_fraction": 0.25,
            "max_position_pct": 0.08,
            "min_position_pct": 0.01,
            "dau_scaling_floor": 50.0,
            "dau_scaling_ceil": 90.0,
        },
        "formula": "size = kelly_fraction * (p - q/b) * dau_scalar, clamped to [min_position_pct, max_position_pct]",
        "version": "2.4",
        "category": "position_sizing",
    },
    "axiom_composite_dau": {
        "description": "Deployable Alpha Utility: fuses engine scores into a single 0-100 signal quality score.",
        "parameters": {
            "fundamental_reality_weight": 0.30,
            "critical_fragility_weight": 0.25,
            "prosperity_composite_weight": 0.20,
            "behavioral_finance_weight": 0.15,
            "alternative_signals_weight": 0.10,
        },
        "formula": "DAU = sum(engine_score_i * weight_i) for i in engines, normalised 0-100",
        "version": "5.2",
        "category": "composite_scoring",
    },
    "intraday_composite": {
        "description": "Intraday signal overlay weighting momentum, liquidity, and order-flow imbalance.",
        "parameters": {
            "momentum_weight": 0.40,
            "liquidity_weight": 0.35,
            "order_flow_weight": 0.25,
            "decay_halflife_minutes": 45,
        },
        "formula": "intraday_score = exp(-t/halflife) * (momentum*w1 + liquidity*w2 + order_flow*w3)",
        "version": "1.8",
        "category": "intraday",
    },
    "dossier_iq": {
        "description": "Intelligence quality score for company event dossiers — rates information density and recency.",
        "parameters": {
            "recency_decay_days": 30,
            "impact_score_floor": 0.1,
            "source_diversity_weight": 0.20,
            "event_volume_weight": 0.30,
            "impact_magnitude_weight": 0.50,
        },
        "formula": "IQ = impact_magnitude*w3 + event_volume_normalized*w2 + source_diversity*w1, decayed by recency",
        "version": "2.0",
        "category": "intelligence",
    },
    "ensemble_blending": {
        "description": "ML ensemble blending weights for combining gradient boosting, neural net, and linear models.",
        "parameters": {
            "gradient_boost_weight": 0.50,
            "neural_net_weight": 0.30,
            "linear_model_weight": 0.20,
            "out_of_sample_penalty": 0.05,
            "regime_shift_dampener": 0.15,
        },
        "formula": "ensemble = sum(model_pred_i * weight_i) - out_of_sample_penalty during regime shifts",
        "version": "4.1",
        "category": "ml_ensemble",
    },
}


def get_formula_hash() -> str:
    """SHA-256 fingerprint of the full registry for tamper-evident audits."""
    canonical = json.dumps(FORMULA_REGISTRY, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
