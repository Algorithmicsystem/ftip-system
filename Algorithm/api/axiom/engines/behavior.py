from __future__ import annotations

from typing import List

from api.assistant.phase3.common import bounded_score, clamp, mean
from api.axiom.common import inverse_score, midrange_score, rounded, weighted_average
from api.axiom.contracts import AxiomEngineInput, EngineScore

# Kahneman-Tversky (1979) loss aversion coefficient
_LOSS_AVERSION_COEFFICIENT = 2.25


# ---------------------------------------------------------------------------
# 1.3 Kahneman-Tversky Asymmetric Sentiment Score
# ---------------------------------------------------------------------------

def compute_asymmetric_sentiment(positive: float, negative: float) -> float:
    """Kahneman-Tversky Asymmetric Sentiment Score (0–100).

    Grounded in Prospect Theory (Kahneman-Tversky 1979) and Thinking Fast
    and Slow.  Loss aversion coefficient = 2.25 (exact paper value).

    Equal positive and negative sentiment → net score < 50 because losses
    are weighted 2.25× gains.  Returns higher score when positive sentiment
    dominates after asymmetric weighting.
    """
    p = clamp(float(positive or 0.0), 0.0, 100.0) / 100.0
    n = clamp(float(negative or 0.0), 0.0, 100.0) / 100.0

    # Prospect-theory net: gain utility minus loss aversion × loss disutility
    net = p - _LOSS_AVERSION_COEFFICIENT * n

    # Piecewise mapping centred at net=0 → 50 (no sentiment = neutral):
    #   net = +1   → 100 (pure positive)
    #   net =  0   →  50 (neutral)
    #   net = -2.25→   0 (pure negative, full loss aversion applied)
    if net >= 0:
        score = 50.0 + net * 50.0
    else:
        score = 50.0 + net * (50.0 / _LOSS_AVERSION_COEFFICIENT)
    return round(clamp(score, 0.0, 100.0), 2)


def score_behavioral_distortion(engine_input: AxiomEngineInput) -> EngineScore:
    support = engine_input.support
    fragility = engine_input.fragility
    fundamental = engine_input.fundamental

    raw_narrative_intensity = weighted_average(
        [
            (support.attention_intensity_score, 0.24),
            (support.novelty_score, 0.16),
            (support.repetition_score, 0.16),
            (support.narrative_concentration_score, 0.18),
            (support.event_pressure_score, 0.12),
            (support.sentiment_level_score, 0.14),
        ]
    )
    narrative_intensity_component = midrange_score(
        raw_narrative_intensity,
        center=56.0,
        tolerance=44.0,
    )
    crowding_component = weighted_average(
        [
            (inverse_score(support.narrative_crowding_index), 0.35),
            (inverse_score(fragility.narrative_crowding_score), 0.2),
            (inverse_score(support.narrative_concentration_score), 0.18),
            (inverse_score(support.repetition_score), 0.14),
            (inverse_score(support.attention_intensity_score), 0.13),
        ]
    )
    extrapolation_stretch_component = weighted_average(
        [
            (inverse_score(support.trend_exhaustion_score), 0.28),
            (inverse_score(support.hype_to_price_divergence_score), 0.24),
            (inverse_score(support.positive_news_weak_price_divergence), 0.18),
            (inverse_score(fragility.event_overhang_score), 0.15),
            (inverse_score(support.event_pressure_score), 0.15),
        ]
    )
    # Asymmetric sentiment: positive direction vs negative crowding pressure
    _pos_sent = support.sentiment_direction_score or 50.0
    _neg_sent = support.narrative_crowding_index or 50.0
    asymmetric_sent_score = compute_asymmetric_sentiment(_pos_sent, _neg_sent)

    underreaction_continuation_component = weighted_average(
        [
            (asymmetric_sent_score, 0.18),      # replaces raw sentiment_direction
            (support.sentiment_trend_score, 0.12),
            (support.trend_quality_score, 0.16),
            (support.directional_persistence_score, 0.16),
            (support.price_volume_alignment_score, 0.14),
            (support.domain_agreement_score, 0.12),
            (inverse_score(support.narrative_crowding_index), 0.12),
        ]
    )
    washed_out_reversal_proxy = weighted_average(
        [
            (support.negative_news_resilient_price_divergence, 0.32),
            (bounded_score(abs(fragility.maxdd_63d or 0.0), low=0.0, high=0.35), 0.18),
            (bounded_score(abs(fragility.maxdd_126d or 0.0), low=0.0, high=0.5), 0.16),
            (inverse_score(fragility.event_overhang_score), 0.12),
            (fundamental.fundamental_durability_score, 0.12),
            (inverse_score(fragility.implementation_fragility_score), 0.1),
        ]
    )
    reversal_setup_component = midrange_score(
        washed_out_reversal_proxy,
        center=52.0,
        tolerance=52.0,
    )
    contradiction_penalty_component = weighted_average(
        [
            (inverse_score(support.contradiction_score), 0.34),
            (inverse_score(support.hype_to_price_divergence_score), 0.24),
            (inverse_score(support.domain_conflict_score), 0.2),
            (inverse_score(fragility.cross_asset_conflict_score), 0.12),
            (inverse_score(fragility.event_uncertainty_score), 0.1),
        ]
    )
    score = weighted_average(
        [
            (narrative_intensity_component, 0.12),
            (crowding_component, 0.18),
            (extrapolation_stretch_component, 0.16),
            (underreaction_continuation_component, 0.28),
            (reversal_setup_component, 0.12),
            (contradiction_penalty_component, 0.14),
        ]
    )

    component_values = {
        "narrative_intensity_component": rounded(narrative_intensity_component),
        "crowding_component": rounded(crowding_component),
        "extrapolation_stretch_component": rounded(extrapolation_stretch_component),
        "underreaction_continuation_component": rounded(underreaction_continuation_component),
        "reversal_setup_component": rounded(reversal_setup_component),
        "contradiction_penalty_component": rounded(contradiction_penalty_component),
    }
    available_count = sum(1 for value in component_values.values() if value is not None)
    coverage = clamp(
        mean(
            [
                engine_input.partial_engine_hints.get("behavioral_distortion", 0.0),
                engine_input.domain_coverage.get("sentiment", 0.0),
                engine_input.domain_coverage.get("event", 0.0),
                (available_count / max(len(component_values), 1)) * 100.0,
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
                support.domain_agreement_score,
                inverse_score(support.domain_conflict_score),
                support.sentiment_direction_score,
                support.attention_intensity_score,
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    flags: List[str] = []
    if (support.narrative_crowding_index or 0.0) >= 68.0:
        flags.append("crowded_narrative")
    if (support.hype_to_price_divergence_score or 0.0) >= 60.0:
        flags.append("behavioral_overshoot")
    if (support.negative_news_resilient_price_divergence or 0.0) >= 60.0:
        flags.append("washed_out_reversal")
    if (support.sentiment_direction_score or 0.0) >= 60.0 and (support.trend_quality_score or 0.0) >= 60.0:
        flags.append("underreaction_continuation")
    if (support.contradiction_score or 0.0) >= 60.0:
        flags.append("narrative_contradiction")

    if score is None:
        return EngineScore(
            score=None,
            confidence=round(confidence, 2),
            coverage=round(coverage, 2),
            status="unavailable" if coverage <= 0.0 else "partial",
            components={},
            flags=flags or ["behavioral_context_unavailable"],
            summary="Behavioral Distortion cannot score the setup because narrative and event context are too thin.",
        )

    status = "available" if coverage >= 62.0 and confidence >= 52.0 else "partial"
    summary = (
        "Behavioral Distortion is scored as behavioral opportunity quality, where higher means the narrative tape is more usable than dangerous. "
        f"The engine reads {rounded(score)} / 100 with underreaction/continuation quality at {rounded(underreaction_continuation_component)} "
        f"and crowding quality at {rounded(crowding_component)}."
    )
    return EngineScore(
        score=round(score, 2),
        confidence=round(confidence, 2),
        coverage=round(coverage, 2),
        status=status,
        components={key: value for key, value in component_values.items() if value is not None},
        flags=sorted(set(flags)),
        summary=summary,
    )
