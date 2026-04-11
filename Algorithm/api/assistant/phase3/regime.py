from __future__ import annotations

from typing import Any, Dict, List, Optional

from .common import bounded_score, clamp, component, mean


def classify_regime(
    *,
    trend_quality_score: Optional[float],
    momentum_consistency_score: Optional[float],
    directional_persistence_score: Optional[float],
    breakout_readiness_score: Optional[float],
    squeeze_score: Optional[float],
    chop_score: Optional[float],
    volatility_stress_score: Optional[float],
    instability_score: Optional[float],
    transition_risk_score: Optional[float],
) -> Dict[str, Any]:
    state_scores: Dict[str, Optional[float]] = {
        "trending": mean(
            [
                trend_quality_score,
                momentum_consistency_score,
                directional_persistence_score,
                breakout_readiness_score,
                100.0 - (instability_score or 50.0),
            ]
        ),
        "choppy": mean(
            [
                chop_score,
                100.0 - (directional_persistence_score or 50.0),
                100.0 - (trend_quality_score or 50.0),
                100.0 - (breakout_readiness_score or 50.0),
            ]
        ),
        "high_vol": mean(
            [
                volatility_stress_score,
                instability_score,
                transition_risk_score,
            ]
        ),
        "squeeze": mean(
            [
                squeeze_score,
                breakout_readiness_score,
                100.0 - (volatility_stress_score or 50.0),
                100.0 - (instability_score or 50.0),
            ]
        ),
        "transition": mean(
            [
                transition_risk_score,
                instability_score,
                bounded_score(abs((trend_quality_score or 50.0) - (momentum_consistency_score or 50.0)), low=0.0, high=60.0),
                bounded_score(abs((directional_persistence_score or 50.0) - (breakout_readiness_score or 50.0)), low=0.0, high=60.0),
            ]
        ),
    }

    ranked_states = sorted(
        [(state, score) for state, score in state_scores.items() if score is not None],
        key=lambda item: item[1],
        reverse=True,
    )
    regime_label = ranked_states[0][0] if ranked_states else "unknown"
    top_score = ranked_states[0][1] if ranked_states else None
    second_score = ranked_states[1][1] if len(ranked_states) > 1 else None

    regime_confidence = None
    if top_score is not None:
        confidence_edge = top_score - (second_score or 0.0)
        regime_confidence = clamp(
            0.55 * top_score + 0.45 * bounded_score(confidence_edge, low=0.0, high=35.0),
            0.0,
            100.0,
        )

    regime_instability = mean(
        [
            instability_score,
            transition_risk_score,
            volatility_stress_score,
            bounded_score(abs((top_score or 0.0) - (second_score or top_score or 0.0)), low=0.0, high=20.0, invert=True),
        ]
    )

    transition_probability = mean(
        [
            transition_risk_score,
            regime_instability,
            bounded_score(abs((momentum_consistency_score or 50.0) - (directional_persistence_score or 50.0)), low=0.0, high=50.0),
        ]
    )

    regime_support = 0.0
    if regime_label == "trending":
        regime_support = ((regime_confidence or 50.0) - (regime_instability or 50.0)) / 50.0
    elif regime_label == "squeeze":
        regime_support = ((breakout_readiness_score or 50.0) - (regime_instability or 50.0)) / 60.0
    elif regime_label == "choppy":
        regime_support = -0.25
    elif regime_label in {"high_vol", "transition"}:
        regime_support = -((regime_instability or 50.0) / 100.0)
    regime_support = clamp(regime_support, -1.0, 1.0)

    components: List[Dict[str, Any]] = [
        component("trend_quality", trend_quality_score, 0.22, "Slope quality and trend cleanliness."),
        component("momentum_consistency", momentum_consistency_score, 0.18, "Cross-horizon return alignment."),
        component("directional_persistence", directional_persistence_score, 0.18, "Persistence of daily direction."),
        component("breakout_readiness", breakout_readiness_score, 0.14, "Readiness for range escape."),
        component("squeeze_score", squeeze_score, 0.12, "Compression and release setup quality."),
        component("chop_score", chop_score, 0.16, "Reversal-heavy or indecisive behavior."),
    ]

    return {
        "regime_label": regime_label,
        "regime_confidence": round(regime_confidence, 2) if regime_confidence is not None else None,
        "regime_instability": round(regime_instability, 2) if regime_instability is not None else None,
        "transition_risk": round(transition_probability, 2) if transition_probability is not None else None,
        "regime_transition_probability": round(transition_probability, 2) if transition_probability is not None else None,
        "trend_quality": round(trend_quality_score, 2) if trend_quality_score is not None else None,
        "chop_intensity": round(chop_score, 2) if chop_score is not None else None,
        "breakout_readiness": round(breakout_readiness_score, 2) if breakout_readiness_score is not None else None,
        "directional_persistence": round(directional_persistence_score, 2) if directional_persistence_score is not None else None,
        "squeeze_intensity": round(squeeze_score, 2) if squeeze_score is not None else None,
        "regime_support": round(regime_support, 4),
        "state_scores": {key: round(value, 2) if value is not None else None for key, value in state_scores.items()},
        "components": components,
    }

