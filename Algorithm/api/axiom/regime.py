from __future__ import annotations

from typing import Dict

from api.assistant.phase3.common import clamp, mean
from api.axiom.common import inverse_score
from api.axiom.contracts import (
    AxiomEngineInput,
    AxiomRegimeDecision,
    AxiomScorecard,
    EngineScore,
)


def classify_axiom_regime(
    engine_input: AxiomEngineInput,
    engine_scores: Dict[str, EngineScore],
    scorecard: AxiomScorecard,
) -> AxiomRegimeDecision:
    fundamental = engine_scores["fundamental_reality"]
    state_pricing = engine_scores["state_pricing"]
    behavioral = engine_scores["behavioral_distortion"]
    flow = engine_scores["flow_transmission"]
    liquidity = engine_scores["liquidity_convexity"]
    fragility = engine_scores["critical_fragility"]
    research = engine_scores["research_integrity"]
    support = engine_input.support
    fragility_input = engine_input.fragility
    flags: list[str] = []
    label = "indeterminate"
    trade_family = "none"
    rationale = "AXIOM remains conservative because the seven-engine stack is still conflicted."

    if (
        (fragility.score or 0.0) >= 82.0
        and (behavioral.score or 100.0) <= 42.0
        and (
            (support.narrative_crowding_index or 0.0) >= 68.0
            or (support.hype_to_price_divergence_score or 0.0) >= 60.0
        )
    ):
        label = "euphoria_critical"
        rationale = "Narrative and price behavior look overheated while fragility is already elevated, so AXIOM treats the regime as euphoric and unstable."
        flags.extend(["behavioral_overshoot", "fragility_dominant"])
    elif (
        (fragility_input.implementation_fragility_score or 0.0) >= 75.0
        and (liquidity.score or 100.0) <= 42.0
    ):
        label = "liquidity_fracture"
        rationale = "Execution survivability is too weak relative to opportunity quality, so liquidity fracture dominates the regime."
        flags.extend(["liquidity_fracture", "execution_breakdown"])
    elif (
        (fundamental.score or 0.0) >= 68.0
        and (state_pricing.score or 0.0) >= 62.0
        and (fragility.score or 100.0) <= 45.0
        and (research.score or 0.0) >= 55.0
    ):
        label = "fundamental_convergence"
        trade_family = "convergence"
        rationale = "Fundamentals, state pricing, and research integrity are aligned while fragility is contained, so AXIOM sees a convergence regime."
        flags.append("quality_supported")
    elif (
        (state_pricing.score or 0.0) >= 66.0
        and (fundamental.score or 0.0) >= 55.0
        and (flow.score or 0.0) >= 52.0
        and 42.0 <= (fragility.score or 0.0) <= 66.0
    ):
        label = "compensation_capture"
        trade_family = "compensation"
        rationale = "The setup offers coherent compensation for residual macro or fragility risk, rather than a clean convergence profile."
        flags.append("risk_premium_supported")
    elif (
        (behavioral.score or 0.0) >= 64.0
        and (flow.score or 0.0) >= 62.0
        and (support.narrative_crowding_index or 100.0) <= 58.0
        and (fragility_input.cross_asset_conflict_score or 100.0) < 45.0
    ):
        label = "behavioral_continuation"
        trade_family = "transmission"
        rationale = "Narrative behavior still looks usable rather than crowded, and clean transmission support points to behavioral continuation."
        flags.append("trend_supported")
    elif (
        (liquidity.score or 0.0) >= 58.0
        and (support.negative_news_resilient_price_divergence or 0.0) >= 55.0
        and (fragility.score or 0.0) <= 58.0
        and (support.live_readiness_score or 0.0) >= 45.0
    ):
        label = "convexity_opportunity"
        trade_family = "convexity"
        rationale = "Execution survivability is adequate and the setup retains asymmetric payoff potential, so AXIOM classifies it as a convexity opportunity."
        flags.append("asymmetry_supported")
    elif (
        (fragility_input.maxdd_63d or 0.0) <= -0.15
        and (fundamental.score or 0.0) >= 55.0
        and (behavioral.score or 0.0) >= 54.0
    ):
        label = "recovery_reset"
        trade_family = "recovery"
        rationale = "The setup is emerging from a material drawdown with enough fundamental and behavioral support to justify a recovery reset classification."
        flags.append("drawdown_recovery")

    confidence = clamp(
        mean(
            [
                scorecard.overall_coverage,
                scorecard.overall_confidence,
                fundamental.confidence,
                state_pricing.confidence,
                behavioral.confidence,
                flow.confidence,
                liquidity.confidence,
                research.confidence,
                inverse_score(fragility.confidence) if fragility.confidence else 50.0,
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    return AxiomRegimeDecision(
        regime_label=label,
        trade_family=trade_family,
        confidence=round(confidence, 2),
        rationale=rationale,
        flags=sorted(
            set(
                flags
                + list(fragility.flags)
                + list(state_pricing.flags)
                + list(behavioral.flags)
                + list(flow.flags)
                + list(liquidity.flags)
                + list(research.flags)
            )
        ),
    )
