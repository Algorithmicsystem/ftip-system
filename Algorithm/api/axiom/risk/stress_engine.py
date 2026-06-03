"""Phase 11.2: Stress Testing Engine.

Tests portfolio positions against historical crisis scenarios and
a Sornette-derived bubble crash scenario.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api.assistant.phase3.common import clamp

_SCENARIO_KEYS = ["liquidity_fracture", "euphoria_critical", "compensation_capture", "recovery_reset"]

SCENARIO_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "liquidity_fracture": {
        "historical_drawdown_pct": -0.35,
        "correlation_spike": 0.90,
        "vol_multiplier": 2.5,
        "recovery_days": 180,
        "sector_impact": {
            "Technology": 1.4,
            "Finance": 1.8,
            "Utilities": 0.6,
            "Healthcare": 0.7,
        },
    },
    "euphoria_critical": {
        "historical_drawdown_pct": -0.50,
        "correlation_spike": 0.70,
        "vol_multiplier": 3.0,
        "recovery_days": 730,
        "sector_impact": {
            "Technology": 2.0,
            "Finance": 1.5,
            "Utilities": 0.5,
            "Healthcare": 0.6,
        },
    },
    "compensation_capture": {
        "historical_drawdown_pct": -0.15,
        "correlation_spike": 0.60,
        "vol_multiplier": 1.5,
        "recovery_days": 90,
        "sector_impact": {
            "Technology": 1.2,
            "Finance": 1.0,
            "Utilities": 0.8,
            "Healthcare": 0.7,
        },
    },
    "recovery_reset": {
        "historical_drawdown_pct": +0.30,
        "correlation_spike": 0.30,
        "vol_multiplier": 0.8,
        "recovery_days": 0,
        "sector_impact": {
            "Technology": 1.3,
            "Finance": 1.5,
            "Utilities": 0.7,
            "Healthcare": 0.9,
        },
    },
}


@dataclass
class StressScenario:
    scenario_id: str
    scenario_name: str
    reference_date: dt.date
    regime_label: str
    vix_at_entry: float
    cape_at_entry: float
    description: str
    historical_drawdown_pct: float
    recovery_days: int
    sector_impact: Dict[str, float] = field(default_factory=dict)


def run_stress_test(
    positions: Dict[str, float],
    axiom_scores: Dict[str, Dict],
    scenarios: Optional[List[str]] = None,
) -> Dict:
    """Run historical stress scenarios against a portfolio.

    Returns scenario-level losses, worst/best scenario, and stress VaR.
    """
    if scenarios is None:
        scenarios = _SCENARIO_KEYS

    results: Dict[str, Dict] = {}

    for scenario_label in scenarios:
        params = SCENARIO_PARAMETERS.get(scenario_label)
        if params is None:
            continue

        position_losses: Dict[str, float] = {}

        for symbol, weight in positions.items():
            scores = axiom_scores.get(symbol, {})
            sector = scores.get("sector", "Unknown")
            sector_multiplier = params["sector_impact"].get(sector, 1.0)

            fragility_score = float(scores.get("fragility_score", 50))
            fragility_multiplier = 1.0 + (fragility_score - 50) / 100.0  # 0.5–1.5

            scps_score = float(scores.get("scps_score", 50))
            scps_multiplier = 1.0 + scps_score / 200.0  # 1.0–1.5

            position_loss = (
                params["historical_drawdown_pct"]
                * sector_multiplier
                * fragility_multiplier
                * scps_multiplier
                * weight
            )
            position_losses[symbol] = position_loss

        portfolio_loss_pct = sum(position_losses.values())
        portfolio_loss_absolute = portfolio_loss_pct * 1_000_000

        # Worst 3 positions in this scenario (largest absolute loss)
        sorted_losses = sorted(
            [(sym, loss) for sym, loss in position_losses.items()],
            key=lambda x: x[1],  # most negative first for loss scenarios
        )
        worst_positions = [
            {"symbol": sym, "position_loss_pct": round(loss, 4)}
            for sym, loss in sorted_losses[:3]
        ]

        results[scenario_label] = {
            "portfolio_loss_pct": round(portfolio_loss_pct, 4),
            "portfolio_loss_absolute": round(portfolio_loss_absolute, 2),
            "worst_positions": worst_positions,
            "recovery_days_estimate": params["recovery_days"],
            "stress_var": round(abs(portfolio_loss_pct), 4),
        }

    if not results:
        return {
            "scenarios": {},
            "worst_scenario": None,
            "best_scenario": None,
            "average_stress_loss": 0.0,
            "stress_var_99": 0.0,
        }

    # Worst/best scenario by portfolio loss
    worst_scenario = min(results, key=lambda s: results[s]["portfolio_loss_pct"])
    best_scenario = max(results, key=lambda s: results[s]["portfolio_loss_pct"])

    losses = [results[s]["portfolio_loss_pct"] for s in results]
    average_stress_loss = round(sum(losses) / len(losses), 4)
    stress_var_99 = round(abs(min(losses)), 4)  # 99th pct of 4 scenarios ≈ worst

    return {
        "scenarios": results,
        "worst_scenario": worst_scenario,
        "best_scenario": best_scenario,
        "average_stress_loss": average_stress_loss,
        "stress_var_99": stress_var_99,
    }


def run_sornette_scenario(
    positions: Dict[str, float],
    axiom_scores: Dict[str, Dict],
) -> Dict:
    """Estimate crash loss if all high-SCPS positions collapse simultaneously."""
    high_risk: List[str] = []
    total_exposure = 0.0
    estimated_loss = 0.0

    for symbol, weight in positions.items():
        scores = axiom_scores.get(symbol, {})
        scps = float(scores.get("scps_score", 0))
        if scps > 70.0:
            high_risk.append(symbol)
            crash_severity = clamp((scps - 70.0) / 30.0 * 0.60, 0.0, 0.60)
            estimated_loss += crash_severity * weight
            total_exposure += weight

    if total_exposure > 0.30:
        recommendation = "reduce_exposure"
    elif total_exposure > 0.10:
        recommendation = "monitor"
    else:
        recommendation = "acceptable"

    return {
        "high_risk_symbols": high_risk,
        "total_sornette_exposure": round(total_exposure, 4),
        "estimated_crash_loss": round(estimated_loss, 4),
        "recommendation": recommendation,
    }
