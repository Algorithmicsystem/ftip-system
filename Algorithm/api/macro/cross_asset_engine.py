"""Phase 18.3: Cross-Asset Intelligence Engine."""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from api.assistant.phase3.common import clamp

logger = logging.getLogger(__name__)

CROSS_ASSET_SIGNALS = [
    "yield_curve_slope", "credit_spread", "term_premium",
    "dxy_trend", "em_fx_stress", "safe_haven_demand",
    "copper_trend", "oil_regime", "gold_demand",
    "vix_regime", "vix_term_structure", "cross_vol_dispersion",
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CrossAssetSnapshot:
    as_of_date: dt.date
    equity_regime_confirmed: bool
    cross_asset_confirmation_score: float
    regime_consistency: str

    fixed_income_signal: str
    currency_signal: str
    commodity_signal: str
    volatility_signal: str

    carry_environment: str
    value_environment: str
    momentum_environment: str
    defensive_environment: str

    equity_signal_amplifier: float
    macro_headwind_score: float
    macro_tailwind_score: float

    macro_narrative: str


# ---------------------------------------------------------------------------
# Signal classifiers (pure functions)
# ---------------------------------------------------------------------------

def _classify_fixed_income(yield_curve_slope: Optional[float]) -> str:
    if yield_curve_slope is None:
        return "neutral"
    if yield_curve_slope > 1.0:
        return "risk_on"
    if yield_curve_slope >= 0.0:
        return "neutral"
    return "risk_off"


def _classify_volatility(vix_level: Optional[float]) -> str:
    if vix_level is None:
        return "neutral"
    if vix_level > 35:
        return "extreme_risk_off"
    if vix_level > 25:
        return "risk_off"
    if vix_level >= 15:
        return "neutral"
    return "risk_on"


def _classify_commodity(copper_return_90d: Optional[float]) -> str:
    if copper_return_90d is None:
        return "neutral"
    if copper_return_90d > 0.05:
        return "risk_on"
    if copper_return_90d < -0.05:
        return "risk_off"
    return "neutral"


def _classify_currency(dxy_return_30d: Optional[float]) -> str:
    if dxy_return_30d is None:
        return "neutral"
    if dxy_return_30d > 0.02:
        return "risk_off"
    if dxy_return_30d < -0.02:
        return "risk_on"
    return "neutral"


def _signal_confirmation(signal: str, equity_regime: str) -> float:
    """Returns 0.0–1.0 how much signal confirms equity regime."""
    if equity_regime in ("TRENDING", "RECOVERY"):
        return {"risk_on": 1.0, "neutral": 0.5, "risk_off": 0.0, "extreme_risk_off": 0.0}.get(signal, 0.5)
    if equity_regime in ("HIGH_VOL", "COMPENSATION_CAPTURE", "LIQUIDITY_FRACTURE"):
        return {"risk_off": 1.0, "extreme_risk_off": 1.0, "neutral": 0.5, "risk_on": 0.0}.get(signal, 0.5)
    return 0.5  # CHOPPY or unknown: neutral confirmation


def _cardi_environment(carry_score: float, value_score: float,
                        momentum_score: float, defensive_score: float) -> tuple:
    def _label(score: float) -> str:
        if score > 60:
            return "favorable"
        if score < 40:
            return "unfavorable"
        return "neutral"
    return _label(carry_score), _label(value_score), _label(momentum_score), _label(defensive_score)


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------

def compute_cross_asset_snapshot(
    macro_inputs: Dict[str, Any],
    equity_regime_label: str,
    vix_level: Optional[float] = None,
    yield_curve_slope: Optional[float] = None,
    credit_spread: Optional[float] = None,
    copper_return_90d: Optional[float] = None,
    dxy_return_30d: Optional[float] = None,
    as_of_date: Optional[dt.date] = None,
) -> CrossAssetSnapshot:
    as_of_date = as_of_date or dt.date.today()

    fi_signal = _classify_fixed_income(yield_curve_slope)
    vol_signal = _classify_volatility(vix_level)
    comm_signal = _classify_commodity(copper_return_90d)
    fx_signal = _classify_currency(dxy_return_30d)

    signals = [fi_signal, vol_signal, comm_signal, fx_signal]
    confirmations = [_signal_confirmation(s, equity_regime_label) for s in signals]
    conf_score = round(sum(confirmations) / len(confirmations) * 100.0, 2)

    equity_confirmed = conf_score > 50.0
    if conf_score >= 60:
        consistency = "consistent"
    elif conf_score <= 40:
        consistency = "divergent"
    else:
        consistency = "mixed"

    if conf_score > 70:
        amplifier = 0.15
    elif conf_score < 30:
        amplifier = -0.30
    else:
        amplifier = 0.0

    # CARDI components
    credit_carry = 50.0 if credit_spread is None else clamp(100.0 - credit_spread / 3.0, 0.0, 100.0)
    value_env = 50.0
    if yield_curve_slope is not None:
        value_env = clamp(50.0 + yield_curve_slope * 10, 0.0, 100.0)
    momentum_env = 50.0
    if copper_return_90d is not None:
        momentum_env = clamp(50.0 + copper_return_90d * 200, 0.0, 100.0)
    defensive_env = 50.0
    if vix_level is not None:
        defensive_env = clamp(100.0 - vix_level * 1.5, 0.0, 100.0)

    carry_lbl, value_lbl, momentum_lbl, defensive_lbl = _cardi_environment(
        credit_carry, value_env, momentum_env, defensive_env
    )

    headwind = clamp(100.0 - conf_score, 0.0, 100.0)
    tailwind = conf_score

    # Narrative
    curve_desc = "normal" if (yield_curve_slope or 0) >= 0 else "inverted"
    vix_desc = f"VIX at {vix_level:.0f}" if vix_level is not None else "VIX data unavailable"
    copper_desc = (
        f"Copper {'+' if (copper_return_90d or 0) >= 0 else ''}{(copper_return_90d or 0):.1%} (90d)"
        if copper_return_90d is not None else "Copper data unavailable"
    )
    regime_env = "risk-on" if equity_regime_label in ("TRENDING", "RECOVERY") else "risk-off"
    narrative = (
        f"Yield curve {curve_desc} with {vix_desc} signals {fi_signal.replace('_', ' ')} environment. "
        f"{copper_desc} {'confirms' if comm_signal == fi_signal else 'contradicts'} equity regime "
        f"with cross-asset confirmation of {conf_score:.0f}%."
    )

    return CrossAssetSnapshot(
        as_of_date=as_of_date,
        equity_regime_confirmed=equity_confirmed,
        cross_asset_confirmation_score=conf_score,
        regime_consistency=consistency,
        fixed_income_signal=fi_signal,
        currency_signal=fx_signal,
        commodity_signal=comm_signal,
        volatility_signal=vol_signal,
        carry_environment=carry_lbl,
        value_environment=value_lbl,
        momentum_environment=momentum_lbl,
        defensive_environment=defensive_lbl,
        equity_signal_amplifier=amplifier,
        macro_headwind_score=round(headwind, 2),
        macro_tailwind_score=round(tailwind, 2),
        macro_narrative=narrative,
    )


def apply_cross_asset_overlay(
    dau: float,
    cross_asset_snapshot: CrossAssetSnapshot,
    regime_label: str,
) -> float:
    adjusted = dau * (1.0 + cross_asset_snapshot.equity_signal_amplifier)
    return round(clamp(adjusted, 0.0, 100.0), 2)


def compute_cross_asset_for_equity(
    symbol: str,
    axiom_payload: Dict[str, Any],
    macro_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    macro_context = macro_context or {}
    regime = str(axiom_payload.get("regime_label") or "UNKNOWN")
    dau = float(axiom_payload.get("deployable_alpha_utility") or 50.0)

    snapshot = compute_cross_asset_snapshot(
        macro_inputs=macro_context,
        equity_regime_label=regime,
        vix_level=macro_context.get("vix_level"),
        yield_curve_slope=macro_context.get("yield_curve_slope"),
        credit_spread=macro_context.get("credit_spread"),
        copper_return_90d=macro_context.get("copper_return_90d"),
        dxy_return_30d=macro_context.get("dxy_return_30d"),
    )

    adjusted_dau = apply_cross_asset_overlay(dau, snapshot, regime)

    return {
        "symbol": symbol,
        "dau": dau,
        "cross_asset_adjusted_dau": adjusted_dau,
        "amplifier": snapshot.equity_signal_amplifier,
        "cross_asset_confirmation": snapshot.cross_asset_confirmation_score,
        "macro_narrative": snapshot.macro_narrative,
        "regime_consistency": snapshot.regime_consistency,
    }
