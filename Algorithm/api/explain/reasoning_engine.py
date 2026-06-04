"""Phase 16.1: Structured Reasoning Engine — grounded, rule-based reasoning chains."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api.assistant.phase3.common import clamp


# ---------------------------------------------------------------------------
# Knowledge vault — theoretical grounding by factor
# ---------------------------------------------------------------------------

_FACTOR_GROUNDING: Dict[str, str] = {
    "EIF": "Financial Statement Analysis (Penman) + Financial Shenanigans (Schilit)",
    "CMF": "Creating Shareholder Value (Rappaport)",
    "BAF": "Prospect Theory (Kahneman-Tversky 1979)",
    "KLF": "Continuous Auctions and Insider Trading (Kyle 1985)",
    "SCAF": "Why Stock Markets Crash (Sornette)",
    "ICF": "Expected Returns (Ilmanen)",
    "GBF": "Active Portfolio Management (Grinold-Kahn)",
    "MTRF": "The Variation of Certain Speculative Prices (Mandelbrot)",
    "MQF": "Returns to Buying Winners and Selling Losers (Jegadeesh-Titman)",
    "VIF": "Security Analysis (Graham-Dodd) + Penman + Rappaport triple-screen",
    "RTF": "Manias Panics and Crashes (Kindleberger) + The Most Important Thing (Marks)",
    "NTFF": "Noise Trader Risk (DeLong-Shleifer-Summers-Waldmann)",
}


def get_theoretical_grounding(factor_name: str) -> str:
    return _FACTOR_GROUNDING.get(
        factor_name,
        "Active Portfolio Management (Grinold-Kahn): systematic factor investing framework",
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ReasoningStep:
    step_id: int
    claim: str
    evidence: List[str]
    confidence: float
    source: str
    theoretical_grounding: str


@dataclass
class ReasoningChain:
    symbol: str
    as_of_date: dt.date
    signal_label: str
    dau: float
    primary_conclusion: str
    reasoning_steps: List[ReasoningStep]
    supporting_factors: List[Dict[str, Any]]
    contradicting_factors: List[Dict[str, Any]]
    confidence_overall: float
    invalidation_conditions: List[str]
    theoretical_foundations: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deployability_tier(dau: float) -> str:
    if dau >= 75:
        return "high-conviction deployable"
    if dau >= 60:
        return "standard deployable"
    if dau >= 45:
        return "weak deployable"
    return "below deployment threshold"


def _regime_alignment(signal_label: str, regime: str) -> str:
    regime = regime.upper()
    if signal_label == "BUY":
        if regime in ("TRENDING", "RECOVERY"):
            return "supports"
        if regime in ("HIGH_VOL", "LIQUIDITY_FRACTURE"):
            return "opposes"
        return "is neutral toward"
    if signal_label == "SELL":
        if regime in ("HIGH_VOL", "COMPENSATION_CAPTURE"):
            return "supports"
        if regime in ("RECOVERY", "TRENDING"):
            return "opposes"
        return "is neutral toward"
    return "is neutral toward"


# ---------------------------------------------------------------------------
# Step builders
# ---------------------------------------------------------------------------

def _step_signal_direction(dau: float, signal_label: str) -> ReasoningStep:
    tier = _deployability_tier(dau)
    return ReasoningStep(
        step_id=1,
        claim=f"Signal is {signal_label} with DAU of {dau:.1f}",
        evidence=[
            f"AXIOM deployable alpha utility: {dau:.1f}",
            f"Deployability tier: {tier}",
        ],
        confidence=1.0,
        source="axiom_scorecard",
        theoretical_grounding=(
            "Active Portfolio Management (Grinold-Kahn): "
            "DAU reflects IC × breadth × quality"
        ),
    )


def _step_primary_factor(alpha_decomp: Dict[str, Any]) -> Optional[ReasoningStep]:
    if not alpha_decomp:
        return None
    primary_driver = alpha_decomp.get("primary_driver", "")
    if not primary_driver:
        return None
    factor_contributions = alpha_decomp.get("factor_contributions", {})
    regime_adjusted = alpha_decomp.get("regime_adjusted_loadings", {})
    contribution = factor_contributions.get(primary_driver, 0.0)
    loading = regime_adjusted.get(primary_driver, 0.0)
    grounding = get_theoretical_grounding(primary_driver)
    factor_conc = alpha_decomp.get("factor_concentration", 0.5)
    confidence = clamp(0.5 + factor_conc * 0.5, 0.0, 1.0)
    return ReasoningStep(
        step_id=2,
        claim=(
            f"Primary driver is {primary_driver} factor "
            f"contributing {contribution:.1f} DAU points"
        ),
        evidence=[
            f"Factor loading: {loading:.3f}",
            f"Factor contribution: {contribution:.1f} points",
        ],
        confidence=round(confidence, 3),
        source="alpha_decomposition",
        theoretical_grounding=grounding,
    )


def _step_fundamental_quality(engine_scores: Dict[str, Any]) -> Optional[ReasoningStep]:
    fundamental = engine_scores.get("fundamental_reality", {})
    comps = fundamental.get("components", {})
    eis = comps.get("eis_component")
    caps = comps.get("caps_component")
    if eis is None and caps is None:
        return None
    eis_val = float(eis) if eis is not None else 50.0
    quality_label = (
        "high integrity" if eis_val > 65 else
        "moderate quality" if eis_val > 45 else
        "quality concerns"
    )
    confidence = clamp(abs(eis_val - 50.0) / 50.0 + 0.4, 0.0, 1.0)
    evidence = [
        f"EIS score: {eis_val:.1f} — {quality_label}",
        "Penman-Schilit accruals analysis",
    ]
    if caps is not None:
        evidence.append(f"CAPS score: {float(caps):.1f}")
    return ReasoningStep(
        step_id=3,
        claim=f"Earnings quality (EIS) of {eis_val:.1f} indicates {quality_label}",
        evidence=evidence,
        confidence=round(confidence, 3),
        source="fundamental_reality",
        theoretical_grounding=(
            "Financial Statement Analysis (Penman) + Financial Shenanigans (Schilit)"
        ),
    )


def _step_regime_context(
    regime: str, regime_strength: float, signal_label: str, breadth_state: str
) -> ReasoningStep:
    alignment = _regime_alignment(signal_label, regime)
    confidence = clamp(0.4 + regime_strength * 0.5, 0.0, 1.0)
    return ReasoningStep(
        step_id=4,
        claim=f"Current {regime} regime {alignment} this signal",
        evidence=[
            f"Regime strength: {regime_strength:.2f}",
            f"Regime breadth state: {breadth_state}",
        ],
        confidence=round(confidence, 3),
        source="regime_context",
        theoretical_grounding=(
            "Manias Panics and Crashes (Kindleberger): "
            "Regime context determines signal reliability"
        ),
    )


def _step_risk_assessment(
    engine_scores: Dict[str, Any], steps: List[ReasoningStep]
) -> Optional[ReasoningStep]:
    fragility_engine = engine_scores.get("critical_fragility", {})
    fragility = fragility_engine.get("score")
    if fragility is None:
        return None
    fragility = float(fragility)
    comps = fragility_engine.get("components", {})
    scps = comps.get("scps_component")
    bfs = comps.get("bfs_component")

    risk_label = (
        "elevated" if fragility > 60 else
        "moderate" if fragility > 40 else
        "low"
    )
    confidence = clamp(abs(fragility - 50.0) / 50.0 + 0.4, 0.0, 1.0)
    evidence = [f"Critical fragility score: {fragility:.1f}"]
    if scps is not None:
        evidence.append(f"Sornette SCPS component: {float(scps):.1f}")
    if bfs is not None:
        evidence.append(f"Mandelbrot BFS component: {float(bfs):.1f}")

    main_step = ReasoningStep(
        step_id=5,
        claim=f"Fragility score of {fragility:.1f} indicates {risk_label} risk environment",
        evidence=evidence,
        confidence=round(confidence, 3),
        source="critical_fragility",
        theoretical_grounding=(
            "Why Stock Markets Crash (Sornette) + "
            "Fractals and Scaling in Finance (Mandelbrot)"
        ),
    )
    steps.append(main_step)

    if scps is not None and float(scps) > 70:
        steps.append(ReasoningStep(
            step_id=51,
            claim=f"Sornette SCPS of {float(scps):.1f} warns of critical point / bubble conditions",
            evidence=[f"SCPS: {float(scps):.1f} — above critical 70 threshold"],
            confidence=0.70,
            source="critical_fragility",
            theoretical_grounding="Why Stock Markets Crash (Sornette)",
        ))
    return None  # steps already appended directly


def _step_ic_calibration(ic_state: str, amqs_score: Optional[float]) -> ReasoningStep:
    will_full_kelly = ic_state in ("STRONG", "MODERATE")
    amqs_str = f"AMQS score: {amqs_score:.1f}" if amqs_score is not None else "AMQS: computing"
    confidence = {"STRONG": 0.95, "MODERATE": 0.75, "WEAK": 0.55, "DEGRADED": 0.35}.get(ic_state, 0.40)
    return ReasoningStep(
        step_id=6,
        claim=(
            f"IC gate state is {ic_state} — signal "
            f"{'will' if will_full_kelly else 'will not'} be sized at full Kelly"
        ),
        evidence=[f"IC state: {ic_state}", amqs_str],
        confidence=confidence,
        source="ic_gate",
        theoretical_grounding=(
            "Active Portfolio Management (Grinold-Kahn): "
            "IR = IC × sqrt(breadth) / TE"
        ),
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_reasoning_chain(
    symbol: str,
    axiom_payload: Dict[str, Any],
    signal_label: str,
    as_of_date: Optional[dt.date] = None,
) -> ReasoningChain:
    as_of_date = as_of_date or dt.date.today()
    dau = float(axiom_payload.get("deployable_alpha_utility") or 50.0)
    ic_state = str(axiom_payload.get("ic_state") or "INSUFFICIENT")
    regime = str(axiom_payload.get("regime_label") or "UNKNOWN")
    regime_strength = float(axiom_payload.get("regime_strength") or 0.5)
    amqs_score = axiom_payload.get("amqs_score")
    engine_scores = axiom_payload.get("engine_scores") or {}
    alpha_decomp = axiom_payload.get("alpha_decomposition") or {}
    factor_contributions = alpha_decomp.get("factor_contributions") or {}
    regime_adjusted = alpha_decomp.get("regime_adjusted_loadings") or {}
    breadth_state = str(axiom_payload.get("breadth_state") or "neutral")

    steps: List[ReasoningStep] = []

    # Step 1 — Signal direction
    steps.append(_step_signal_direction(dau, signal_label))

    # Step 2 — Primary factor
    s2 = _step_primary_factor(alpha_decomp)
    if s2:
        steps.append(s2)

    # Step 3 — Fundamental quality
    s3 = _step_fundamental_quality(engine_scores)
    if s3:
        steps.append(s3)

    # Step 4 — Regime context
    steps.append(_step_regime_context(regime, regime_strength, signal_label, breadth_state))

    # Step 5 — Risk (appends directly, may append 5b too)
    _step_risk_assessment(engine_scores, steps)
    if not any(s.step_id == 5 for s in steps):
        # Fallback if no fragility data
        steps.append(ReasoningStep(
            step_id=5,
            claim="Risk environment assessed as moderate — limited fragility data",
            evidence=["Fragility components not available"],
            confidence=0.4,
            source="critical_fragility",
            theoretical_grounding="Why Stock Markets Crash (Sornette) + Fractals and Scaling in Finance (Mandelbrot)",
        ))

    # Step 6 — IC calibration
    steps.append(_step_ic_calibration(ic_state, amqs_score))

    # Supporting and contradicting factors
    if signal_label == "BUY":
        supporting_factors = [
            {"factor": k, "contribution": v, "loading": regime_adjusted.get(k, 0.0)}
            for k, v in factor_contributions.items() if v > 0
        ]
        contradicting_factors = [
            {"factor": k, "contribution": v, "loading": regime_adjusted.get(k, 0.0)}
            for k, v in factor_contributions.items() if v < 0
        ]
    elif signal_label == "SELL":
        supporting_factors = [
            {"factor": k, "contribution": v, "loading": regime_adjusted.get(k, 0.0)}
            for k, v in factor_contributions.items() if v < 0
        ]
        contradicting_factors = [
            {"factor": k, "contribution": v, "loading": regime_adjusted.get(k, 0.0)}
            for k, v in factor_contributions.items() if v > 0
        ]
    else:
        supporting_factors = []
        contradicting_factors = []

    # Confidence
    step_confs = [s.confidence for s in steps]
    raw_conf = sum(step_confs) / len(step_confs) if step_confs else 0.5
    if ic_state == "INSUFFICIENT":
        raw_conf -= 0.10
    fragility_engine = engine_scores.get("critical_fragility", {})
    fragility_val = float(fragility_engine.get("score") or 50.0)
    if fragility_val > 65:
        raw_conf -= 0.05
    confidence_overall = round(clamp(raw_conf, 0.0, 1.0), 3)

    # Invalidation conditions
    invalidation = [
        "EIS drops below 40 (earnings quality deteriorates)",
        f"Regime transitions from {regime} to HIGH_VOL or liquidity_fracture",
    ]
    if signal_label == "BUY":
        invalidation.append("Fragility score rises above 70 (market stress spike)")
    fragility_comps = fragility_engine.get("components", {})
    scps_val = float(fragility_comps.get("scps_component") or 0.0)
    if scps_val > 60:
        invalidation.append("Sornette crash signal completes — critical point reached")

    # Primary conclusion
    primary_conclusion = (
        f"{symbol} receives a {signal_label} signal with DAU {dau:.1f}; "
        f"primary driver is {alpha_decomp.get('primary_driver') or 'systematic factors'} "
        f"in a {regime} regime."
    )

    # Theoretical foundations — deduplicated
    foundations: List[str] = []
    seen: set = set()
    for s in steps:
        if s.theoretical_grounding and s.theoretical_grounding not in seen:
            foundations.append(s.theoretical_grounding)
            seen.add(s.theoretical_grounding)

    return ReasoningChain(
        symbol=symbol,
        as_of_date=as_of_date,
        signal_label=signal_label,
        dau=dau,
        primary_conclusion=primary_conclusion,
        reasoning_steps=steps,
        supporting_factors=supporting_factors,
        contradicting_factors=contradicting_factors,
        confidence_overall=confidence_overall,
        invalidation_conditions=invalidation,
        theoretical_foundations=foundations,
    )
