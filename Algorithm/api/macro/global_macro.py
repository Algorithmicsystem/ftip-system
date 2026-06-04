"""Phase 18.4: Global Macro Intelligence Engine."""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api.assistant.phase3.common import clamp

logger = logging.getLogger(__name__)

# Macro regime labels
MACRO_REGIMES = {
    "gdp": ["expansion", "moderate", "stagnation", "contraction"],
    "inflation": ["deflation", "low_inflation", "moderate_inflation", "high_inflation", "hyperinflation"],
    "monetary": ["emergency_easing", "easing", "neutral", "tightening", "emergency_tightening"],
    "credit": ["expansion", "late_cycle", "contraction", "recovery"],
}

# Factor implications per macro environment
_FAVORED: Dict[str, List[str]] = {
    "expansion_low_inflation": ["MQF", "CMF", "VIF"],
    "expansion_moderate_inflation": ["CMF", "EIF", "MQF"],
    "contraction": ["KLF", "MTRF", "SCAF"],
    "high_inflation": ["ICF", "MTRF"],
    "hyperinflation": ["ICF", "SCAF"],
    "tightening": ["EIF", "CMF"],
    "emergency_tightening": ["EIF", "SCAF"],
    "default": ["GBF"],
}

_UNFAVORED: Dict[str, List[str]] = {
    "expansion_low_inflation": ["SCAF", "MTRF"],
    "expansion_moderate_inflation": ["SCAF", "KLF"],
    "contraction": ["MQF", "CMF", "VIF"],
    "high_inflation": ["CMF", "MQF"],
    "hyperinflation": ["MQF", "CMF", "VIF"],
    "tightening": ["BAF", "NTFF"],
    "emergency_tightening": ["BAF", "MQF", "CMF"],
    "default": ["NTFF"],
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class MacroIntelligenceSnapshot:
    as_of_date: dt.date
    gdp_regime: str
    inflation_regime: str
    monetary_regime: str
    credit_regime: str

    growth_outlook_score: float
    inflation_risk_score: float
    policy_support_score: float
    credit_availability_score: float

    equity_macro_score: float
    fixed_income_macro_score: float
    commodity_macro_score: float

    favored_axiom_factors: List[str]
    unfavored_axiom_factors: List[str]

    macro_regime_label: str
    macro_environment_score: float
    investment_implications: str


# ---------------------------------------------------------------------------
# Regime classifiers
# ---------------------------------------------------------------------------

def _classify_gdp_regime(gdp_growth: float) -> str:
    if gdp_growth > 2.5:
        return "expansion"
    if gdp_growth > 1.0:
        return "moderate"
    if gdp_growth >= 0.0:
        return "stagnation"
    return "contraction"


def _classify_inflation_regime(cpi_yoy: float) -> str:
    if cpi_yoy < 0.0:
        return "deflation"
    if cpi_yoy < 2.0:
        return "low_inflation"
    if cpi_yoy < 4.0:
        return "moderate_inflation"
    if cpi_yoy < 7.0:
        return "high_inflation"
    return "hyperinflation"


def _classify_monetary_regime(fed_funds_rate: float, fed_funds_rate_1y_ago: float) -> str:
    change_bps = (fed_funds_rate - fed_funds_rate_1y_ago) * 100.0
    if change_bps > 200:
        return "emergency_tightening"
    if change_bps > 50:
        return "tightening"
    if change_bps < -50:
        return "easing"
    if fed_funds_rate <= 0.25:
        return "emergency_easing"
    return "neutral"


def _classify_credit_regime(ig_credit_spread: float) -> str:
    if ig_credit_spread < 100:
        return "expansion"
    if ig_credit_spread < 150:
        return "late_cycle"
    if ig_credit_spread < 250:
        return "recovery"
    return "contraction"


# ---------------------------------------------------------------------------
# Score maps
# ---------------------------------------------------------------------------

_GROWTH_SCORES = {
    "expansion": 80.0, "moderate": 60.0, "stagnation": 40.0, "contraction": 20.0,
}
_INFLATION_RISK_SCORES = {
    "deflation": 30.0, "low_inflation": 20.0, "moderate_inflation": 40.0,
    "high_inflation": 70.0, "hyperinflation": 90.0,
}
_POLICY_SCORES = {
    "emergency_easing": 90.0, "easing": 75.0, "neutral": 55.0,
    "tightening": 35.0, "emergency_tightening": 15.0,
}
_CREDIT_SCORES = {
    "expansion": 80.0, "late_cycle": 55.0, "contraction": 25.0, "recovery": 65.0,
}


def _get_favored_factors(gdp_regime: str, inflation_regime: str, monetary_regime: str) -> List[str]:
    key1 = f"{gdp_regime}_{inflation_regime}"
    key2 = inflation_regime
    key3 = monetary_regime
    for key in [key1, key2, key3]:
        if key in _FAVORED:
            return _FAVORED[key]
    if gdp_regime == "contraction":
        return _FAVORED["contraction"]
    return _FAVORED["default"]


def _get_unfavored_factors(gdp_regime: str, inflation_regime: str, monetary_regime: str,
                            favored: List[str]) -> List[str]:
    key1 = f"{gdp_regime}_{inflation_regime}"
    key2 = inflation_regime
    key3 = monetary_regime
    unfavored: List[str] = []
    for key in [key1, key2, key3]:
        if key in _UNFAVORED:
            unfavored = _UNFAVORED[key]
            break
    if not unfavored:
        if gdp_regime == "contraction":
            unfavored = _UNFAVORED["contraction"]
        else:
            unfavored = _UNFAVORED["default"]
    # Ensure no overlap with favored
    return [f for f in unfavored if f not in favored]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def classify_macro_regime(
    gdp_growth: float,
    cpi_yoy: float,
    fed_funds_rate: float,
    fed_funds_rate_1y_ago: float,
    ig_credit_spread: float = 120.0,
    as_of_date: Optional[dt.date] = None,
) -> MacroIntelligenceSnapshot:
    as_of_date = as_of_date or dt.date.today()

    gdp_regime = _classify_gdp_regime(gdp_growth)
    inflation_regime = _classify_inflation_regime(cpi_yoy)
    monetary_regime = _classify_monetary_regime(fed_funds_rate, fed_funds_rate_1y_ago)
    credit_regime = _classify_credit_regime(ig_credit_spread)

    growth_score = _GROWTH_SCORES.get(gdp_regime, 50.0)
    inflation_risk = _INFLATION_RISK_SCORES.get(inflation_regime, 40.0)
    policy_support = _POLICY_SCORES.get(monetary_regime, 55.0)
    credit_avail = _CREDIT_SCORES.get(credit_regime, 55.0)

    equity_macro = clamp(
        growth_score * 0.40
        + (100.0 - inflation_risk) * 0.20
        + policy_support * 0.25
        + credit_avail * 0.15,
        0.0, 100.0,
    )

    fi_macro = clamp(
        (100.0 - growth_score) * 0.40
        + (100.0 - inflation_risk) * 0.30
        + policy_support * 0.30,
        0.0, 100.0,
    )

    commodity_macro = clamp(
        growth_score * 0.50
        + inflation_risk * 0.30
        + (100.0 - policy_support) * 0.20,
        0.0, 100.0,
    )

    macro_env = clamp(
        equity_macro * 0.50 + fi_macro * 0.30 + commodity_macro * 0.20,
        0.0, 100.0,
    )

    favored = _get_favored_factors(gdp_regime, inflation_regime, monetary_regime)
    unfavored = _get_unfavored_factors(gdp_regime, inflation_regime, monetary_regime, favored)

    macro_label = f"{gdp_regime}_{inflation_regime}_{monetary_regime}"

    implications = (
        f"Current macro environment: {gdp_regime} GDP growth with {inflation_regime.replace('_', ' ')} "
        f"under a {monetary_regime.replace('_', ' ')} monetary regime. "
        f"Equity macro support score is {equity_macro:.0f}/100; "
        f"favored factors include {', '.join(favored[:3])}. "
        f"Credit conditions are in {credit_regime.replace('_', ' ')} phase."
    )

    return MacroIntelligenceSnapshot(
        as_of_date=as_of_date,
        gdp_regime=gdp_regime,
        inflation_regime=inflation_regime,
        monetary_regime=monetary_regime,
        credit_regime=credit_regime,
        growth_outlook_score=round(growth_score, 2),
        inflation_risk_score=round(inflation_risk, 2),
        policy_support_score=round(policy_support, 2),
        credit_availability_score=round(credit_avail, 2),
        equity_macro_score=round(equity_macro, 2),
        fixed_income_macro_score=round(fi_macro, 2),
        commodity_macro_score=round(commodity_macro, 2),
        favored_axiom_factors=favored,
        unfavored_axiom_factors=unfavored,
        macro_regime_label=macro_label,
        macro_environment_score=round(macro_env, 2),
        investment_implications=implications,
    )


def compute_macro_factor_overlay(
    macro_snapshot: MacroIntelligenceSnapshot,
    current_factor_loadings: Dict[str, float],
) -> Dict[str, float]:
    adjusted: Dict[str, float] = {}
    favored_set = set(macro_snapshot.favored_axiom_factors)
    unfavored_set = set(macro_snapshot.unfavored_axiom_factors)

    macro_strength = macro_snapshot.macro_environment_score / 100.0

    for factor, loading in current_factor_loadings.items():
        if factor in favored_set:
            # Boost favored factors proportional to macro strength
            adjustment = loading * (1.0 + macro_strength * 0.30)
        elif factor in unfavored_set:
            # Dampen unfavored factors
            adjustment = loading * (1.0 - macro_strength * 0.20)
        else:
            adjustment = loading
        adjusted[factor] = round(clamp(adjustment, -2.0, 2.0), 4)

    return adjusted
