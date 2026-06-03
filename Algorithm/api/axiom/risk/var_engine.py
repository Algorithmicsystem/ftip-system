"""Phase 11.1: Historical simulation VaR and CVaR engine.

Implements historical simulation, parametric VaR, and portfolio VaR
with Mandelbrot fat-tail correction via MTRS scores.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from api.assistant.phase3.common import clamp

_Z_SCORES = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326, 0.999: 3.090}


def compute_historical_var(
    returns: List[float],
    confidence: float = 0.99,
    horizon_days: int = 1,
    mtrs_score: float = 50.0,
) -> Dict:
    """Historical simulation VaR using the empirical return distribution.

    Returns None values if len(returns) < 20.
    VaR is returned as a positive number representing a loss.
    """
    if len(returns) < 20:
        return {
            "var_1d": None,
            "var_horizon": None,
            "cvar_1d": None,
            "cvar_horizon": None,
            "confidence": confidence,
            "horizon_days": horizon_days,
            "sample_count": 0,
            "mtrs_adjustment": None,
            "methodology": "historical_simulation_mtrs_adjusted",
        }

    sorted_returns = sorted(returns)
    n = len(sorted_returns)
    idx = max(0, min(int(math.floor((1.0 - confidence) * n)), n - 1))

    # VaR threshold (negative number — the worst return at this percentile)
    var_value = sorted_returns[idx]
    raw_var_1d = -var_value  # positive

    # CVaR: mean of all returns at or below the VaR threshold
    tail = [r for r in sorted_returns if r <= var_value]
    raw_cvar_1d = -(sum(tail) / len(tail)) if tail else raw_var_1d

    # Mandelbrot fat-tail correction
    mtrs_adjustment = 1.0 + (mtrs_score / 100.0) * 0.30
    adj_var = raw_var_1d * mtrs_adjustment
    adj_cvar = raw_cvar_1d * mtrs_adjustment

    scale = math.sqrt(horizon_days)

    return {
        "var_1d": round(adj_var, 6),
        "var_horizon": round(adj_var * scale, 6),
        "cvar_1d": round(adj_cvar, 6),
        "cvar_horizon": round(adj_cvar * scale, 6),
        "confidence": confidence,
        "horizon_days": horizon_days,
        "sample_count": n,
        "mtrs_adjustment": round(mtrs_adjustment, 4),
        "methodology": "historical_simulation_mtrs_adjusted",
    }


def compute_parametric_var(
    volatility_annual: float,
    confidence: float = 0.99,
    horizon_days: int = 1,
    mtrs_score: float = 50.0,
) -> Dict:
    """Parametric VaR using normal distribution assumption.

    Fat-tail adjustment corrects for normal distribution underestimating tail risk.
    Does not import scipy — uses hardcoded z-scores for common confidence levels.
    """
    daily_vol = volatility_annual / math.sqrt(252)
    z = _Z_SCORES.get(confidence, _Z_SCORES[0.99])
    raw_var_1d = abs(z) * daily_vol

    # Fat-tail adjustment
    fat_tail = 1.0 + (mtrs_score / 100.0) * 0.40
    var_1d = raw_var_1d * fat_tail
    scale = math.sqrt(horizon_days)

    # Parametric CVaR: approximately 1.25× VaR (conservative normal approximation)
    cvar_1d = var_1d * 1.25

    return {
        "var_1d": round(var_1d, 6),
        "var_horizon": round(var_1d * scale, 6),
        "cvar_1d": round(cvar_1d, 6),
        "cvar_horizon": round(cvar_1d * scale, 6),
        "confidence": confidence,
        "horizon_days": horizon_days,
        "sample_count": None,
        "mtrs_adjustment": round(fat_tail, 4),
        "methodology": "parametric_normal_mtrs_adjusted",
    }


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


def compute_portfolio_var(
    position_weights: Dict[str, float],
    symbol_returns: Dict[str, List[float]],
    confidence: float = 0.99,
    horizon_days: int = 1,
) -> Dict:
    """Portfolio VaR using the full portfolio return series.

    Computes marginal VaR per position and diversification benefit.
    """
    symbols = [s for s in position_weights if s in symbol_returns and symbol_returns[s]]
    if not symbols:
        return {
            "portfolio_var_1d": None,
            "portfolio_cvar_1d": None,
            "portfolio_var_horizon": None,
            "marginal_var": {},
            "largest_var_contributor": None,
            "diversification_benefit": None,
            "concentration_risk": False,
            "confidence": confidence,
            "horizon_days": horizon_days,
            "methodology": "portfolio_historical_simulation",
        }

    # Step 1: Portfolio daily return series
    min_len = min(len(symbol_returns[s]) for s in symbols)
    port_returns: List[float] = []
    for i in range(min_len):
        day_ret = sum(position_weights[s] * symbol_returns[s][i] for s in symbols)
        port_returns.append(day_ret)

    # Step 2: Portfolio VaR
    port_var = compute_historical_var(port_returns, confidence=confidence, horizon_days=horizon_days)

    if port_var["var_1d"] is None:
        return {
            "portfolio_var_1d": None,
            "portfolio_cvar_1d": None,
            "portfolio_var_horizon": None,
            "marginal_var": {},
            "largest_var_contributor": None,
            "diversification_benefit": None,
            "concentration_risk": False,
            "confidence": confidence,
            "horizon_days": horizon_days,
            "methodology": "portfolio_historical_simulation",
        }

    pvar = port_var["var_1d"]

    # Step 3: Marginal VaR per position
    marginal_var: Dict[str, float] = {}
    for sym in symbols:
        corr = _pearson_corr(symbol_returns[sym], port_returns)
        marginal_var[sym] = round(pvar * corr * position_weights[sym], 6)

    # Largest contributor
    if marginal_var:
        largest = max(marginal_var, key=lambda s: abs(marginal_var[s]))
    else:
        largest = None

    # Step 4: Diversification benefit
    individual_vars = {}
    for sym in symbols:
        iv = compute_historical_var(symbol_returns[sym], confidence=confidence, horizon_days=1)
        individual_vars[sym] = iv["var_1d"] or 0.0
    undiversified = sum(position_weights[s] * individual_vars[s] for s in symbols)
    diversification_benefit = round(undiversified - pvar, 6)

    # Concentration risk: any single position > 40% of total marginal VaR
    total_marginal_abs = sum(abs(v) for v in marginal_var.values())
    concentration_risk = (
        any(abs(v) / total_marginal_abs > 0.40 for v in marginal_var.values())
        if total_marginal_abs > 1e-12
        else False
    )

    return {
        "portfolio_var_1d": round(pvar, 6),
        "portfolio_cvar_1d": port_var["cvar_1d"],
        "portfolio_var_horizon": port_var["var_horizon"],
        "marginal_var": marginal_var,
        "largest_var_contributor": largest,
        "diversification_benefit": diversification_benefit,
        "concentration_risk": concentration_risk,
        "confidence": confidence,
        "horizon_days": horizon_days,
        "methodology": "portfolio_historical_simulation",
    }
