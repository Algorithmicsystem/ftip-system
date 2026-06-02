from __future__ import annotations

from typing import List, Optional

from api.assistant.phase3.common import bounded_score, clamp, mean
from api.axiom.common import inverse_score, rounded, weighted_average
from api.axiom.contracts import AxiomEngineInput, EngineScore


# ---------------------------------------------------------------------------
# 1.7 Ilmanen Cross-Asset Return Driver Index
# ---------------------------------------------------------------------------

def compute_cardi(macro_data: dict) -> float:
    """Ilmanen Cross-Asset Return Driver Index (0–100).

    Grounded in Expected Returns (Ilmanen).

    Formula: carry_score × 0.25 + value_score × 0.25 + momentum_score × 0.30
             + defensive_score × 0.20

    Positive term spread + strong 12-1 cross-asset momentum → CARDI > 65.
    """
    carry_score = macro_data.get("carry_score")       # term spread / credit carry
    value_score = macro_data.get("value_score")       # CAPE inverse / E/P spread
    momentum_score = macro_data.get("momentum_score") # 12-1 cross-asset momentum
    defensive_score = macro_data.get("defensive_score")  # low-vol / quality factor

    components = [
        (carry_score,    0.25),
        (value_score,    0.25),
        (momentum_score, 0.30),
        (defensive_score, 0.20),
    ]
    usable = [
        (clamp(float(v), 0.0, 100.0), w)
        for v, w in components
        if v is not None
    ]
    if not usable:
        return 50.0
    total_weight = sum(w for _, w in usable)
    if total_weight <= 0:
        return 50.0
    cardi = sum(v * w for v, w in usable) / total_weight
    return round(clamp(cardi, 0.0, 100.0), 2)


def score_state_pricing(engine_input: AxiomEngineInput) -> EngineScore:
    support = engine_input.support
    fragility = engine_input.fragility
    fundamental = engine_input.fundamental

    # CARDI: compute from macro_cardi_inputs if present, else 50 (neutral)
    _macro_cardi = engine_input.source_context.get("macro_cardi_inputs", {})
    cardi_score = compute_cardi(_macro_cardi) if _macro_cardi else None

    macro_alignment_component = weighted_average(
        [
            (cardi_score if cardi_score is not None else support.macro_alignment_score, 0.25),
            (support.macro_growth_alignment_score, 0.16),
            (support.risk_on_alignment_score, 0.15),
            (support.macro_regime_consistency_score, 0.16),
            (inverse_score(support.macro_conflict_score), 0.16),
            (inverse_score(fragility.market_stress_score), 0.12),
        ]
    )
    factor_compensation_component = weighted_average(
        [
            (support.opportunity_quality_score, 0.2),
            (support.cross_domain_conviction_score, 0.16),
            (support.fundamental_durability_score, 0.16),
            (support.domain_agreement_score, 0.14),
            (support.relative_context_quality_score, 0.14),
            (support.idiosyncratic_strength_score, 0.1),
            (bounded_score(support.ret_21d, low=-0.15, high=0.25), 0.05),
            (bounded_score(support.mom_vol_adj_21d, low=-0.8, high=1.2), 0.05),
        ]
    )
    discount_rate_regime_component = weighted_average(
        [
            (support.regime_stability_score, 0.25),
            (support.macro_regime_consistency_score, 0.22),
            (inverse_score(support.inflation_stress_proxy), 0.18),
            (inverse_score(support.macro_fragility_score), 0.18),
            (inverse_score(fragility.regime_transition_score), 0.17),
        ]
    )
    cross_asset_confirmation_component = weighted_average(
        [
            (inverse_score(fragility.cross_asset_conflict_score), 0.3),
            (support.sector_confirmation_score, 0.2),
            (support.relative_context_quality_score, 0.16),
            (support.benchmark_relative_strength_score, 0.12),
            (support.sector_relative_strength_score, 0.12),
            (fragility.breadth_confirmation_score, 0.1),
        ]
    )
    bad_state_exposure_component = weighted_average(
        [
            (inverse_score(support.macro_fragility_score), 0.28),
            (inverse_score(fragility.market_stress_score), 0.22),
            (inverse_score(fragility.signal_fragility_score), 0.14),
            (inverse_score(fragility.event_uncertainty_score), 0.14),
            (inverse_score(fragility.regime_instability_score), 0.12),
            (inverse_score(support.idiosyncratic_weakness_score), 0.1),
        ]
    )
    state_pricing_conflict_component = weighted_average(
        [
            (inverse_score(support.domain_conflict_score), 0.28),
            (inverse_score(support.macro_conflict_score), 0.28),
            (inverse_score(fragility.cross_asset_conflict_score), 0.24),
            (inverse_score(support.contradiction_score), 0.2),
        ]
    )
    score = weighted_average(
        [
            (macro_alignment_component, 0.22),
            (factor_compensation_component, 0.22),
            (discount_rate_regime_component, 0.14),
            (cross_asset_confirmation_component, 0.16),
            (bad_state_exposure_component, 0.12),
            (state_pricing_conflict_component, 0.14),
        ]
    )

    component_values = {
        "macro_alignment_component": rounded(macro_alignment_component),
        "factor_compensation_component": rounded(factor_compensation_component),
        "discount_rate_regime_component": rounded(discount_rate_regime_component),
        "cross_asset_confirmation_component": rounded(cross_asset_confirmation_component),
        "bad_state_exposure_component": rounded(bad_state_exposure_component),
        "state_pricing_conflict_component": rounded(state_pricing_conflict_component),
    }
    available_count = sum(1 for value in component_values.values() if value is not None)
    coverage = clamp(
        mean(
            [
                engine_input.partial_engine_hints.get("state_pricing", 0.0),
                engine_input.domain_coverage.get("macro", 0.0),
                engine_input.domain_coverage.get("cross_asset", 0.0),
                engine_input.domain_coverage.get("market", 0.0),
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
                support.macro_alignment_score,
                support.cross_domain_conviction_score,
                fundamental.provider_confidence,
                support.regime_stability_score,
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    flags: List[str] = []
    if (support.macro_conflict_score or 0.0) >= 60.0:
        flags.append("macro_conflict")
    if (fragility.cross_asset_conflict_score or 0.0) >= 60.0:
        flags.append("cross_asset_conflict")
    if (support.domain_conflict_score or 0.0) >= 55.0:
        flags.append("domain_conflict")
    if (fragility.market_stress_score or 0.0) >= 65.0:
        flags.append("stress_state")
    if fundamental.coverage_score < 45.0:
        flags.append("thin_fundamental_anchor")

    if score is None:
        return EngineScore(
            score=None,
            confidence=round(confidence, 2),
            coverage=round(coverage, 2),
            status="unavailable" if coverage <= 0.0 else "partial",
            components={},
            flags=flags or ["state_pricing_unavailable"],
            summary="State Pricing cannot score the setup because macro, relative, or compensation evidence is too thin.",
        )

    status = "available" if coverage >= 65.0 and confidence >= 55.0 else "partial"
    summary = (
        f"State Pricing reads {rounded(score)} / 100. Macro alignment is {rounded(macro_alignment_component)}, "
        f"cross-asset confirmation is {rounded(cross_asset_confirmation_component)}, and conflict quality is "
        f"{rounded(state_pricing_conflict_component)}. Coverage is {round(coverage, 1)} / 100."
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
