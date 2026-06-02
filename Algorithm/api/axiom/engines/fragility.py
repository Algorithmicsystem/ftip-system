from __future__ import annotations

import math
from typing import Dict, List, Optional

from api.assistant.phase3.common import bounded_score, clamp, mean
from api.axiom.contracts import AxiomEngineInput, EngineScore


# ---------------------------------------------------------------------------
# 1.4 Sornette Critical Point Score
# ---------------------------------------------------------------------------

def _ols_slope(values: list) -> float:
    """Return OLS slope of values over equally spaced index."""
    n = len(values)
    if n < 2:
        return 0.0
    t = list(range(n))
    t_mean = sum(t) / n
    v_mean = sum(values) / n
    cov = sum((t[i] - t_mean) * (values[i] - v_mean) for i in range(n))
    var_t = sum((ti - t_mean) ** 2 for ti in t)
    return cov / var_t if var_t > 0 else 0.0


def compute_scps(price_series: list, window: int = 120) -> float:
    """Sornette Critical Point Score (0–100, higher = more bubble-like pressure).

    Grounded in Why Stock Markets Crash (Sornette).

    Simplified LPPLS detection via:
    - Price acceleration: slope of second half minus first half in log-space
    - Log-periodic oscillation: dispersion of log-returns around trend
    """
    valid = [float(p) for p in (price_series or []) if p is not None and float(p) > 0]
    series = valid[-window:]
    if len(series) < 10:
        return 50.0

    log_prices = [math.log(p) for p in series]
    n = len(log_prices)
    mid = n // 2

    # Acceleration: rising slope in log-space = bubble fingerprint
    slope_first = _ols_slope(log_prices[:mid])
    slope_second = _ols_slope(log_prices[mid:])
    acceleration = slope_second - slope_first  # positive = accelerating

    # Scale: 0.005/bar acceleration (daily return increasing by 0.5 ppt/step) → +20 pts
    accel_score = clamp(50.0 + acceleration * 4000.0, 0.0, 100.0)

    # Log-periodic oscillation: std dev of log-returns relative to typical noise
    log_returns = [log_prices[i] - log_prices[i - 1] for i in range(1, n)]
    if log_returns:
        lr_mean = sum(log_returns) / len(log_returns)
        lr_std = (sum((r - lr_mean) ** 2 for r in log_returns) / len(log_returns)) ** 0.5
        oscillation_score = clamp((lr_std / 0.03) * 50.0, 0.0, 100.0)
    else:
        oscillation_score = 50.0

    scps = accel_score * 0.90 + oscillation_score * 0.10
    return round(clamp(scps, 0.0, 100.0), 2)


# ---------------------------------------------------------------------------
# 1.5 Mandelbrot Tail Risk Score
# ---------------------------------------------------------------------------

def compute_mtrs(return_series: list) -> float:
    """Mandelbrot Tail Risk Score (0–100, higher = fatter tails).

    Grounded in Mandelbrot (1963) and Fractals and Scaling in Finance.

    Uses Hill estimator for power-law tail exponent + excess kurtosis.
    """
    returns = [float(r) for r in (return_series or []) if r is not None]
    n = len(returns)
    if n < 10:
        return 50.0

    mean_r = sum(returns) / n
    variance = sum((r - mean_r) ** 2 for r in returns) / n
    if variance <= 0:
        return 50.0

    kurt_num = sum((r - mean_r) ** 4 for r in returns) / n
    excess_kurtosis = kurt_num / (variance ** 2) - 3.0

    # Hill estimator on top-10% absolute returns
    abs_ret = sorted([abs(r) for r in returns], reverse=True)
    k = max(int(n * 0.10), 5)
    if k >= n or abs_ret[k] <= 0:
        hill_alpha = 3.0
    else:
        try:
            log_sum = sum(math.log(abs_ret[i] / abs_ret[k]) for i in range(k))
            hill_alpha = k / log_sum if log_sum > 0 else 3.0
        except (ValueError, ZeroDivisionError):
            hill_alpha = 3.0

    # Lower alpha = fatter tails: alpha<2 → extreme; alpha>5 → Gaussian-like
    tail_score = clamp((5.0 - min(hill_alpha, 5.0)) / 3.0 * 100.0, 0.0, 100.0)
    kurt_score = clamp((excess_kurtosis / 6.0) * 100.0, 0.0, 100.0)

    mtrs = tail_score * 0.60 + kurt_score * 0.40
    return round(clamp(mtrs, 0.0, 100.0), 2)


# ---------------------------------------------------------------------------
# 1.8 Shiller-Kindleberger Bubble Fragility Score
# ---------------------------------------------------------------------------

_KINDLEBERGER_STAGE_SCORE: Dict[str, float] = {
    "displacement":  20.0,
    "boom":          40.0,
    "euphoria":      80.0,
    "profit_taking": 60.0,
    "panic":         90.0,
    "recovery":      10.0,
    "normal":        30.0,
}


def compute_bfs(market_context: dict) -> float:
    """Shiller-Kindleberger Bubble Fragility Score (0–100, higher = more bubble risk).

    Grounded in Irrational Exuberance (Shiller) and Manias Panics and Crashes
    (Kindleberger).

    Formula: CAPE_z_score × 0.35 + kindleberger_stage_score × 0.40 + narrative_intensity × 0.25
    """
    cape_z = market_context.get("cape_z_score")
    stage = str(market_context.get("kindleberger_stage") or "normal").lower()
    narrative_intensity = float(market_context.get("narrative_intensity") or 50.0)

    # CAPE z-score: high z = overvalued; bounded [-2, +4]
    if cape_z is not None:
        cape_score = clamp((float(cape_z) + 2.0) / 6.0 * 100.0, 0.0, 100.0)
    else:
        cape_score = 50.0

    stage_score = _KINDLEBERGER_STAGE_SCORE.get(stage, 30.0)
    ni_score = clamp(narrative_intensity, 0.0, 100.0)

    bfs = cape_score * 0.35 + stage_score * 0.40 + ni_score * 0.25
    return round(clamp(bfs, 0.0, 100.0), 2)


def _rounded(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 2)


def _weighted_average(items: List[tuple[Optional[float], float]]) -> Optional[float]:
    usable = [(value, weight) for value, weight in items if value is not None and weight > 0]
    if not usable:
        return None
    total_weight = sum(weight for _value, weight in usable)
    if total_weight <= 0:
        return None
    return sum(float(value) * float(weight) for value, weight in usable) / total_weight


def score_critical_fragility(engine_input: AxiomEngineInput) -> EngineScore:
    candidate = engine_input.fragility

    # MTRS: compute from raw return series if available; else 50 (neutral)
    _return_series = engine_input.source_context.get("return_series", [])
    mtrs_score = compute_mtrs(_return_series) if _return_series else 50.0

    # SCPS: compute from raw price series if available; else 50
    _price_series = engine_input.source_context.get("price_series", [])
    scps_component = compute_scps(_price_series) if _price_series else 50.0

    # BFS: compute from market context if available
    _market_ctx = engine_input.source_context.get("market_bubble_context", {})
    bfs_component = compute_bfs(_market_ctx) if _market_ctx else 50.0

    # MTRS enhances vol_21d signal: blended fat-tail + raw vol
    _raw_vol_21d = bounded_score(candidate.realized_vol_21d, low=0.12, high=0.6)
    _vol_mtrs_blend = (
        _raw_vol_21d * 0.60 + mtrs_score * 0.40
        if _raw_vol_21d is not None
        else mtrs_score
    )

    volatility_instability_component = _weighted_average(
        [
            (candidate.volatility_stress_score, 0.28),
            (_vol_mtrs_blend, 0.24),                    # MTRS-enhanced vol_21d
            (bounded_score(candidate.realized_vol_63d, low=0.12, high=0.5), 0.16),
            (bounded_score(candidate.vol_of_vol_proxy, low=0.0, high=1.0), 0.16),
            (candidate.instability_score, 0.16),
        ]
    )
    gap_jump_risk_component = _weighted_average(
        [
            (candidate.gap_instability_10d and bounded_score(candidate.gap_instability_10d, low=0.0, high=2.0), 0.36),
            (candidate.overnight_gap_risk_score, 0.32),
            (bounded_score(candidate.abs_gap_mean_10d, low=0.0, high=0.05), 0.18),
            (bounded_score(abs(candidate.gap_pct or 0.0), low=0.0, high=0.08), 0.14),
        ]
    )
    drawdown_fragility_component = _weighted_average(
        [
            (candidate.drawdown_sensitivity_score, 0.28),
            (bounded_score(abs(candidate.maxdd_21d or 0.0), low=0.0, high=0.18), 0.22),
            (bounded_score(abs(candidate.maxdd_63d or 0.0), low=0.0, high=0.3), 0.22),
            (bounded_score(abs(candidate.maxdd_126d or 0.0), low=0.0, high=0.45), 0.14),
            (bounded_score(candidate.downside_asymmetry_21d, low=0.8, high=2.2), 0.14),
        ]
    )
    crowding_fragility_component = _weighted_average(
        [
            (candidate.narrative_crowding_score, 0.45),
            (candidate.event_overhang_score, 0.35),
            (candidate.event_uncertainty_score, 0.2),
        ]
    )
    liquidity_fragility_component = _weighted_average(
        [
            (candidate.implementation_fragility_score, 0.3),
            (candidate.tradability_caution_score, 0.18),
            (candidate.friction_proxy_score, 0.18),
            (candidate.anomaly_pressure_score, 0.14),
            (candidate.overnight_gap_risk_score, 0.12),
            (bounded_score(candidate.liquidity_quality_score, low=0.0, high=100.0, invert=True), 0.08),
        ]
    )
    regime_transition_risk_component = _weighted_average(
        [
            (candidate.regime_transition_score, 0.3),
            (candidate.regime_instability_score, 0.18),
            (candidate.market_stress_score, 0.22),
            (candidate.cross_asset_conflict_score, 0.16),
            (bounded_score(candidate.breadth_confirmation_score, low=0.0, high=100.0, invert=True), 0.14),
        ]
    )
    score = _weighted_average(
        [
            (volatility_instability_component, 0.18),   # reduced: MTRS absorbed some weight
            (gap_jump_risk_component, 0.16),
            (drawdown_fragility_component, 0.20),
            (crowding_fragility_component, 0.12),
            (liquidity_fragility_component, 0.17),
            (regime_transition_risk_component, 0.10),   # reduced: SCPS captures regime-break
            (scps_component, 0.15),                     # Sornette critical point
            (bfs_component, 0.10),                      # Shiller-Kindleberger bubble
        ]
    )
    component_values = {
        "volatility_instability_component": _rounded(volatility_instability_component),
        "gap_jump_risk_component": _rounded(gap_jump_risk_component),
        "drawdown_fragility_component": _rounded(drawdown_fragility_component),
        "crowding_fragility_component": _rounded(crowding_fragility_component),
        "liquidity_fragility_component": _rounded(liquidity_fragility_component),
        "regime_transition_risk_component": _rounded(regime_transition_risk_component),
        "scps_component": _rounded(scps_component),
        "bfs_component": _rounded(bfs_component),
    }
    available_count = sum(1 for value in component_values.values() if value is not None)
    coverage = clamp(
        mean(
            [
                candidate.coverage_score,
                (available_count / max(len(component_values), 1)) * 100.0,
                100.0 if candidate.signal_fragility_score is not None else 55.0,
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    confidence = clamp(
        mean(
            [
                coverage,
                candidate.signal_fragility_score,
                candidate.clean_setup_score and (100.0 - candidate.clean_setup_score),
                candidate.instability_score,
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    flags: List[str] = list(candidate.suppression_flags)
    if candidate.event_risk_classification in {"high_event_risk", "event_distorted"}:
        flags.append("event_distortion")
    if (candidate.market_stress_score or 0.0) >= 65.0:
        flags.append("market_stress")
    if (candidate.cross_asset_conflict_score or 0.0) >= 60.0:
        flags.append("cross_asset_conflict")
    if (candidate.implementation_fragility_score or 0.0) >= 65.0:
        flags.append("implementation_fragility")
    if (candidate.regime_transition_score or 0.0) >= 65.0:
        flags.append("regime_transition")
    flags = sorted(set(flag for flag in flags if flag))

    if score is None:
        return EngineScore(
            score=None,
            confidence=round(confidence, 2),
            coverage=round(coverage, 2),
            status="unavailable" if coverage <= 0 else "partial",
            components={},
            flags=flags or ["fragility_unavailable"],
            summary="Critical Fragility cannot score the setup because the current bundle does not contain enough realized instability or implementation-risk evidence.",
        )

    status = "available" if coverage >= 65 and confidence >= 55 else "partial"
    summary_parts = [
        f"Critical Fragility reads {_rounded(score)} / 100, where higher means more path-risk and deployability drag.",
        f"Volatility instability is {_rounded(volatility_instability_component)}, drawdown fragility is {_rounded(drawdown_fragility_component)}, and liquidity fragility is {_rounded(liquidity_fragility_component)}.",
        f"Coverage is {round(coverage, 1)} / 100 and confidence is {round(confidence, 1)} / 100.",
    ]
    if flags:
        summary_parts.append(f"Primary fragility flags: {', '.join(flags[:4])}.")
    return EngineScore(
        score=round(score, 2),
        confidence=round(confidence, 2),
        coverage=round(coverage, 2),
        status=status,
        components={key: value for key, value in component_values.items() if value is not None},
        flags=flags,
        summary=" ".join(summary_parts),
    )
