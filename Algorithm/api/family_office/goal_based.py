"""Phase 14.3: Goal-Based Intelligence."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api.assistant.phase3.common import clamp
from api.family_office.multi_asset import PortfolioSnapshot


@dataclass
class InvestmentGoal:
    goal_id: str
    goal_type: str          # retirement, education, liquidity, legacy, purchase
    label: str
    target_amount_usd: float
    target_date_years: float        # years from now
    current_funding_usd: float
    required_return_annual: float   # pre-computed required rate
    risk_budget: float              # max acceptable volatility


# ---------------------------------------------------------------------------
# Goal funding status
# ---------------------------------------------------------------------------

def compute_goal_funding_status(
    goal: InvestmentGoal,
    current_portfolio_value: float,
    expected_annual_return: float,
) -> Dict[str, Any]:
    """Project portfolio value to goal date and measure gap."""
    years = max(goal.target_date_years, 0.0)
    projected_value = current_portfolio_value * ((1.0 + expected_annual_return) ** years)
    funding_gap = goal.target_amount_usd - projected_value
    funding_ratio = projected_value / goal.target_amount_usd if goal.target_amount_usd > 0 else 0.0
    on_track = funding_ratio >= 1.0

    if funding_ratio >= 1.20:
        status = "well_funded"
    elif funding_ratio >= 1.0:
        status = "on_track"
    elif funding_ratio >= 0.80:
        status = "at_risk"
    else:
        status = "underfunded"

    additional_annual_savings_needed = 0.0
    if funding_gap > 0 and years > 0:
        r = expected_annual_return
        if abs(r) < 1e-9:
            additional_annual_savings_needed = funding_gap / years
        else:
            additional_annual_savings_needed = funding_gap * r / ((1 + r) ** years - 1)

    return {
        "goal_id": goal.goal_id,
        "goal_type": goal.goal_type,
        "label": goal.label,
        "target_amount_usd": round(goal.target_amount_usd, 2),
        "projected_value_usd": round(projected_value, 2),
        "funding_gap_usd": round(funding_gap, 2),
        "funding_ratio": round(funding_ratio, 4),
        "on_track": on_track,
        "status": status,
        "additional_annual_savings_needed": round(max(0.0, additional_annual_savings_needed), 2),
    }


# ---------------------------------------------------------------------------
# Monte Carlo probability of success
# ---------------------------------------------------------------------------

def compute_probability_of_success(
    required_return: float,
    expected_return: float,
    volatility_annual: float,
    years: float,
    n_simulations: int = 1000,
) -> Dict[str, Any]:
    """Monte Carlo simulation using stdlib random (seed=42, no scipy)."""
    rng = random.Random(42)
    successes = 0
    terminal_values: List[float] = []

    dt = 1.0 / 12.0  # monthly steps
    n_steps = max(1, int(years / dt))
    mu_step = expected_return * dt
    sigma_step = volatility_annual * math.sqrt(dt)

    for _ in range(n_simulations):
        cumulative_return = 0.0
        for _ in range(n_steps):
            shock = rng.gauss(0.0, 1.0)
            step_return = mu_step + sigma_step * shock
            cumulative_return += step_return
        terminal_values.append(cumulative_return)
        if cumulative_return >= required_return * years:
            successes += 1

    probability = successes / n_simulations
    sorted_tv = sorted(terminal_values)
    p5 = sorted_tv[int(0.05 * n_simulations)]
    p50 = sorted_tv[int(0.50 * n_simulations)]
    p95 = sorted_tv[int(0.95 * n_simulations)]

    return {
        "probability_of_success": round(probability, 4),
        "required_return": round(required_return, 4),
        "expected_return": round(expected_return, 4),
        "volatility_annual": round(volatility_annual, 4),
        "years": round(years, 2),
        "n_simulations": n_simulations,
        "terminal_return_p5": round(p5, 4),
        "terminal_return_p50": round(p50, 4),
        "terminal_return_p95": round(p95, 4),
    }


# ---------------------------------------------------------------------------
# Liability matching
# ---------------------------------------------------------------------------

def compute_liability_matching_score(
    portfolio: PortfolioSnapshot,
    future_liabilities: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Match portfolio liquidity buckets to liability time horizons."""
    short_liabilities = sum(
        l["amount_usd"] for l in future_liabilities if l.get("years_until_due", 0) <= 2
    )
    medium_liabilities = sum(
        l["amount_usd"] for l in future_liabilities if 2 < l.get("years_until_due", 0) <= 7
    )
    long_liabilities = sum(
        l["amount_usd"] for l in future_liabilities if l.get("years_until_due", 0) > 7
    )

    total_value = portfolio.total_value_usd or 1.0

    # Liquid assets: cash + daily-liquidity equity/fixed income
    liquid_value = (portfolio.cash_weight + portfolio.equity_weight) * total_value
    # Medium-term: fixed income bucket approximation
    medium_value = portfolio.fixed_income_weight * total_value
    # Long-term: alternatives
    long_value = portfolio.alternatives_weight * total_value

    short_covered = liquid_value >= short_liabilities if short_liabilities > 0 else True
    medium_covered = (liquid_value + medium_value) >= (short_liabilities + medium_liabilities)
    long_covered = total_value >= (short_liabilities + medium_liabilities + long_liabilities)

    coverage_ratios = {
        "short_term": round(liquid_value / short_liabilities, 4) if short_liabilities > 0 else None,
        "medium_term": round((liquid_value + medium_value) / (short_liabilities + medium_liabilities), 4)
        if (short_liabilities + medium_liabilities) > 0 else None,
        "long_term": round(total_value / (short_liabilities + medium_liabilities + long_liabilities), 4)
        if (short_liabilities + medium_liabilities + long_liabilities) > 0 else None,
    }

    score = (
        (1.0 if short_covered else 0.5) * 0.50
        + (1.0 if medium_covered else 0.5) * 0.30
        + (1.0 if long_covered else 0.5) * 0.20
    ) * 100.0

    return {
        "liability_matching_score": round(score, 2),
        "short_term_covered": short_covered,
        "medium_term_covered": medium_covered,
        "long_term_covered": long_covered,
        "coverage_ratios": coverage_ratios,
        "total_liabilities_usd": round(short_liabilities + medium_liabilities + long_liabilities, 2),
        "liquid_assets_usd": round(liquid_value, 2),
    }


# ---------------------------------------------------------------------------
# Goal-based recommendations
# ---------------------------------------------------------------------------

def generate_goal_based_recommendations(
    goals: List[InvestmentGoal],
    portfolio: PortfolioSnapshot,
    axiom_regime: str,
) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    total_value = portfolio.total_value_usd

    for goal in goals:
        funding = compute_goal_funding_status(
            goal, total_value, 0.07  # 7% baseline expected return
        )
        status = funding["status"]

        if status == "underfunded" and goal.target_date_years <= 5:
            recs.append({
                "goal_id": goal.goal_id,
                "action": "increase_savings",
                "priority": "high",
                "rationale": (
                    f"Goal '{goal.label}' is underfunded ({funding['funding_ratio']*100:.1f}% funded) "
                    f"with only {goal.target_date_years:.1f} years remaining — "
                    f"increase annual savings by ${funding['additional_annual_savings_needed']:,.0f}"
                ),
            })
        elif status == "at_risk":
            recs.append({
                "goal_id": goal.goal_id,
                "action": "review_return_assumption",
                "priority": "medium",
                "rationale": (
                    f"Goal '{goal.label}' funding ratio {funding['funding_ratio']*100:.1f}% — "
                    "consider higher-return allocation or extended timeline"
                ),
            })

        if goal.risk_budget < portfolio.equity_weight and axiom_regime in ("HIGH_VOL", "compensation_capture"):
            recs.append({
                "goal_id": goal.goal_id,
                "action": "reduce_equity_exposure",
                "priority": "high",
                "rationale": (
                    f"Portfolio equity weight {portfolio.equity_weight*100:.1f}% exceeds risk budget "
                    f"{goal.risk_budget*100:.1f}% during {axiom_regime} regime — reduce equity exposure"
                ),
            })

    if not recs:
        recs.append({
            "goal_id": "portfolio",
            "action": "maintain",
            "priority": "low",
            "rationale": "All goals on track — maintain current allocation and monitor quarterly",
        })

    return recs
