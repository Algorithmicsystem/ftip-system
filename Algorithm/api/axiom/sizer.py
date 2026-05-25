"""Session 12: Fractional-Kelly / risk-budget position sizer.

Given AXIOM scores + calibration quality + IC state + portfolio context,
this module computes a suggested position weight that is:
  1. Anchored to the Deployable Alpha Utility (DAU) score
  2. Scaled by a fractional-Kelly multiplier adjusted for IC state
  3. Penalised by critical fragility
  4. Constrained by risk-budget cap and available portfolio heat
  5. Gated by the deployability tier

No FastAPI / DB dependencies here — pure deterministic computation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Kelly multipliers by IC state (fraction of theoretical Kelly to use)
_IC_KELLY_MULTIPLIER = {
    "STRONG":       0.50,
    "MODERATE":     0.35,
    "WEAK":         0.20,
    "DEGRADED":     0.00,
    "INSUFFICIENT": 0.25,
}

# Deployability tier weight caps
_TIER_CAP = {
    "live_candidate":  None,   # no extra cap beyond risk budget
    "paper_trade_only": 0.02,
    "monitor_only":    0.00,
    "not_actionable":  0.00,
}

_FRAGILITY_SOFT_THRESHOLD = 50.0   # above this, gentle linear penalty starts
_FRAGILITY_HARD_VETO      = 82.0   # above this, weight → 0 regardless of anything

_MIN_WEIGHT_OUTPUT = 0.0001        # weights below this are rounded to 0


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class KellySizeResult:
    symbol: str
    as_of_date: str
    suggested_weight: float           # final recommended weight (fraction of portfolio)
    kelly_gross_weight: float         # weight before fractional / IC scaling
    fractional_kelly_applied: float   # the Kelly multiplier that was used
    ic_kelly_multiplier: float        # multiplier from IC state
    fragility_penalty_applied: float  # 0–1 penalty applied due to fragility
    active_constraint: str            # which constraint was binding
    size_band: str                    # "large" / "medium" / "small" / "none"
    deployability_tier: str
    ic_state: str
    downside_flags: List[str]
    rationale: str
    inputs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_kelly_size(
    *,
    symbol: str,
    as_of_date: str,
    dau: float,                              # 0–100 DAU score
    fragility_score: float,                  # 0–100 critical fragility (higher = worse)
    liquidity_score: float = 60.0,           # 0–100 liquidity convexity
    research_score: float = 60.0,            # 0–100 research integrity
    overall_confidence: float = 60.0,        # 0–100 overall confidence
    deployability_tier: str = "live_candidate",
    hit_rate: Optional[float] = None,        # calibration hit rate 0–1 (or None)
    ic_state: str = "INSUFFICIENT",
    fractional_kelly: float = 0.5,           # base Kelly fraction before IC adjustment
    max_weight: float = 0.10,                # risk budget per position (10% default)
    portfolio_heat: float = 0.0,             # current sum of open position weights
) -> KellySizeResult:
    dau_n      = _clamp(dau / 100.0, 0.0, 1.0)
    conf_n     = _clamp(overall_confidence / 100.0, 0.0, 1.0)
    frag       = _clamp(fragility_score, 0.0, 100.0)
    liquidity  = _clamp(liquidity_score, 0.0, 100.0)
    research   = _clamp(research_score, 0.0, 100.0)

    downside_flags: list[str] = []

    # ------------------------------------------------------------------
    # 1. Base weight = max_weight × DAU fraction × confidence scalar
    # ------------------------------------------------------------------
    confidence_scalar = 0.6 + 0.4 * conf_n   # [0.6, 1.0]
    kelly_gross = max_weight * dau_n * confidence_scalar

    # ------------------------------------------------------------------
    # 2. Calibration boost: if hit_rate > 0.5 we know the edge is real
    # ------------------------------------------------------------------
    calibration_edge = 0.0
    if hit_rate is not None and math.isfinite(hit_rate):
        # edge above coin-flip, scaled so 70% hit rate adds 20% more
        calibration_edge = _clamp(hit_rate - 0.5, 0.0, 0.30) * 2.0  # 0 → 0.6
    kelly_gross = kelly_gross * (1.0 + calibration_edge * 0.3)

    # ------------------------------------------------------------------
    # 3. Fractional Kelly × IC-state multiplier
    # ------------------------------------------------------------------
    ic_mult = _IC_KELLY_MULTIPLIER.get(ic_state, 0.25)
    effective_kelly = fractional_kelly * ic_mult    # e.g. 0.5 × 0.35 = 0.175
    kelly_weight = kelly_gross * effective_kelly

    if ic_state == "DEGRADED":
        downside_flags.append("ic_state_degraded")

    # ------------------------------------------------------------------
    # 4. Fragility penalty
    # ------------------------------------------------------------------
    fragility_penalty = 0.0
    if frag >= _FRAGILITY_HARD_VETO:
        fragility_penalty = 1.0
        downside_flags.append("critical_fragility_veto")
    elif frag > _FRAGILITY_SOFT_THRESHOLD:
        fragility_penalty = _clamp(
            (frag - _FRAGILITY_SOFT_THRESHOLD) / (_FRAGILITY_HARD_VETO - _FRAGILITY_SOFT_THRESHOLD),
            0.0, 0.75
        )
        if fragility_penalty > 0.3:
            downside_flags.append("critical_fragility_elevated")

    kelly_weight *= (1.0 - fragility_penalty)

    # ------------------------------------------------------------------
    # 5. Secondary quality penalties (liquidity / research)
    # ------------------------------------------------------------------
    if liquidity < 40.0:
        kelly_weight *= 0.6
        downside_flags.append("liquidity_integrity_weak")
    if research < 40.0:
        kelly_weight *= 0.6
        downside_flags.append("research_integrity_weak")

    # ------------------------------------------------------------------
    # 6. Identify binding constraint and apply caps
    # ------------------------------------------------------------------
    tier_cap = _TIER_CAP.get(deployability_tier, 0.0)
    available_heat = _clamp(max_weight - portfolio_heat, 0.0, max_weight)
    active_constraint = "kelly"

    final_weight = kelly_weight

    # Risk budget cap
    if final_weight > max_weight:
        final_weight = max_weight
        active_constraint = "risk_budget"

    # Portfolio heat cap
    if final_weight > available_heat:
        final_weight = available_heat
        active_constraint = "portfolio_heat"

    # Deployability tier cap
    if tier_cap is not None:
        if final_weight > tier_cap:
            final_weight = tier_cap
            active_constraint = "deployability"
        if tier_cap == 0.0:
            downside_flags.append(f"deployability_tier_{deployability_tier}")

    # IC degraded overrides everything
    if ic_state == "DEGRADED":
        final_weight = 0.0
        active_constraint = "ic_degraded"

    # Round tiny values to zero
    if final_weight < _MIN_WEIGHT_OUTPUT:
        final_weight = 0.0

    # ------------------------------------------------------------------
    # 7. Size band
    # ------------------------------------------------------------------
    size_band = _size_band(final_weight, max_weight, deployability_tier)

    # ------------------------------------------------------------------
    # 8. Rationale
    # ------------------------------------------------------------------
    rationale = _build_rationale(
        dau=dau,
        kelly_gross=kelly_gross,
        effective_kelly=effective_kelly,
        fragility_penalty=fragility_penalty,
        fragility_score=fragility_score,
        final_weight=final_weight,
        ic_state=ic_state,
        active_constraint=active_constraint,
        deployability_tier=deployability_tier,
        hit_rate=hit_rate,
    )

    return KellySizeResult(
        symbol=symbol,
        as_of_date=as_of_date,
        suggested_weight=round(final_weight, 6),
        kelly_gross_weight=round(kelly_gross, 6),
        fractional_kelly_applied=round(effective_kelly, 4),
        ic_kelly_multiplier=round(ic_mult, 4),
        fragility_penalty_applied=round(fragility_penalty, 4),
        active_constraint=active_constraint,
        size_band=size_band,
        deployability_tier=deployability_tier,
        ic_state=ic_state,
        downside_flags=sorted(set(downside_flags)),
        rationale=rationale,
        inputs={
            "dau": round(dau, 2),
            "fragility_score": round(fragility_score, 2),
            "liquidity_score": round(liquidity, 2),
            "research_score": round(research, 2),
            "overall_confidence": round(overall_confidence, 2),
            "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
            "ic_state": ic_state,
            "fractional_kelly": fractional_kelly,
            "max_weight": max_weight,
            "portfolio_heat": portfolio_heat,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _size_band(weight: float, max_weight: float, tier: str) -> str:
    if weight <= 0.0 or tier in ("not_actionable", "monitor_only"):
        return "none"
    if tier == "paper_trade_only":
        return "paper"
    ratio = weight / max_weight if max_weight > 0 else 0.0
    if ratio >= 0.70:
        return "large"
    if ratio >= 0.35:
        return "medium"
    if ratio > 0.0:
        return "small"
    return "none"


def _build_rationale(
    *,
    dau: float,
    kelly_gross: float,
    effective_kelly: float,
    fragility_penalty: float,
    fragility_score: float,
    final_weight: float,
    ic_state: str,
    active_constraint: str,
    deployability_tier: str,
    hit_rate: Optional[float],
) -> str:
    pct = f"{final_weight * 100:.2f}%"
    parts = [
        f"DAU {dau:.1f}/100 drives a gross Kelly weight of {kelly_gross * 100:.2f}%.",
        f"Fractional Kelly {effective_kelly:.3f} (IC state: {ic_state}) reduces this.",
    ]
    if fragility_penalty > 0.05:
        parts.append(
            f"Critical fragility {fragility_score:.1f} applies a {fragility_penalty * 100:.0f}% drag."
        )
    if hit_rate is not None:
        parts.append(f"Calibration hit rate {hit_rate * 100:.1f}% used as edge signal.")
    parts.append(
        f"Binding constraint: {active_constraint.replace('_', ' ')}. "
        f"Final suggested weight: {pct} ({deployability_tier.replace('_', ' ')})."
    )
    return " ".join(parts)
